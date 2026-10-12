#!/usr/bin/env python
"""离线量化合并后的模型 -> AWQ W4A16。

校准集必须用真实截图
--------------------
量化的校准数据是拿来决定每一层激活缩放因子的。
用合成图（比如随机噪声或渲染文字）校准，得到的缩放因子与真实分布不匹配，
量化误差会被放大。这里直接从训练集中采样真实 GUI 截图——
既省事，又保证校准分布与部署分布一致。

视觉塔量不量化，是本项目的关键消融
----------------------------------
默认 ``--quantize-vision`` 关闭，即跳过 ``visual.*``，只压语言部分。
理由：定位精度几乎完全由视觉编码器决定，压它风险最大。

但这不能只凭直觉下结论。把两个版本都跑一遍，对比
「显存省了多少」vs「偏移 p50 涨了多少」，才是选型依据。
用 ``--quantize-vision`` 开启对照实验，跑完交给
``python -m vfs.report`` 出对比结论。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIB_DEFAULT = ROOT / "data" / "calib_gui.jsonl"
TRAIN_DEFAULT = ROOT / "data" / "train" / "gui_grounding_train.json"


def write_calibration_file(out: Path, n: int, train_file: Path) -> Path:
    """从训练集中采样真实截图构造校准集。"""
    if not train_file.exists():
        raise SystemExit(
            f"未找到训练集 {train_file}，先跑 scripts/01_prepare_data.py\n"
            "（校准集必须用真实截图，否则缩放因子与部署分布不匹配）"
        )
    rows = []
    with train_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    if not rows:
        raise SystemExit(f"训练集为空: {train_file}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            imgs = r.get("images") or []
            if not imgs:
                continue
            # 取该样本里 human 轮的文本作为配套 prompt
            text = ""
            for c in r.get("conversations", []):
                if c.get("from") == "human":
                    text = c.get("value", "").replace("<image>", "").strip()
                    break
            f.write(json.dumps({"image_path": imgs[0], "text": text or "Describe this screenshot."},
                               ensure_ascii=False) + "\n")
    print(f"[量化] 校准集（真实截图）: {out}，{len(rows)} 条")
    return out


def quantize_llmcompressor(model: str, out: str, scheme: str, calib: Path,
                           n_calib: int, train_file: Path,
                           quantize_vision: bool) -> None:
    calib = write_calibration_file(calib, n_calib, train_file)

    ignore = ["lm_head"] if quantize_vision else ["lm_head", "visual.*"]
    print(f"[量化] ignore 列表: {ignore}"
          f"（{'含视觉塔' if quantize_vision else '跳过视觉塔'}）")

    script = f'''
import json, torch
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

MODEL = {model!r}
OUT = {out!r}
SCHEME = {scheme!r}
IGNORE = {ignore!r}

rows = [json.loads(l) for l in open({str(calib)!r}, encoding="utf-8")]
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

def _to_ds(r):
    img = Image.open(r["image_path"]).convert("RGB")
    msgs = [{{"role": "user", "content": [
        {{"type": "image"}}, {{"type": "text", "text": r["text"]}}]}}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return {{"images": img, "text": text}}

ds = Dataset.from_list([_to_ds(r) for r in rows])
recipe = AWQModifier(ignore=IGNORE, scheme=SCHEME, targets=["Linear"])

oneshot(
    model=MODEL, dataset=ds, recipe=recipe,
    output_dir=OUT, max_seq_length=2048,
    num_calibration_samples=len(rows),
    trust_remote_code_model=True,
)
print("QUANT_DONE", OUT)
'''
    subprocess.run([sys.executable, "-c", script], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="模型离线量化（AWQ）")
    ap.add_argument("--model", default="outputs/qwen25vl-3b-gui-sft-merged")
    ap.add_argument("--out", default="outputs/qwen25vl-3b-gui-sft-merged-awq")
    ap.add_argument("--scheme", default="W4A16", help="W4A16(AWQ) / W8A8(INT8)")
    ap.add_argument("--num-calib", type=int, default=256)
    ap.add_argument("--train-file", default=str(TRAIN_DEFAULT))
    ap.add_argument("--quantize-vision", action="store_true",
                    help="连视觉塔一起量化（对照实验用，默认跳过）")
    args = ap.parse_args()

    quantize_llmcompressor(
        args.model, args.out, args.scheme,
        Path(CALIB_DEFAULT), args.num_calib,
        Path(args.train_file), args.quantize_vision,
    )
    tag = "含视觉塔" if args.quantize_vision else "跳过视觉塔"
    print(f"\n[完成] {args.out}（{tag}）")
    print("下一步：把 configs/serve/awq_int4.yaml 的 model 指向该目录后跑扫描")


if __name__ == "__main__":
    main()
