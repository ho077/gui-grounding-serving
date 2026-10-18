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
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIB_DEFAULT = ROOT / "data" / "calib_gui.jsonl"
TRAIN_DEFAULT = ROOT / "data" / "train" / "gui_grounding_train.json"


def write_calibration_file(out: Path, n: int, train_file: Path,
                           max_side: int = 0) -> Path:
    """从训练集中采样真实截图构造校准集。

    max_side>0 时把长边压到该值再落盘（缓存到 `data/calib_img/`）。
    原因：ShowUI-desktop 的截图是 3360x2100 这类，按默认 max_pixels 不缩放，
    单图就是 9000 个图像 token —— 64 条校准要跑 36 分钟，而量化只用来看
    每一层输入通道的相对大小，分辨率本身不改变 UI 内容的分布。
    实测压到 1280 长边后单图约 2000 token，耗时降到 1/5。
    """
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
    cache_dir = out.parent / "calib_img"
    if max_side > 0:
        cache_dir.mkdir(parents=True, exist_ok=True)

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
            img_path = imgs[0]
            if max_side > 0:
                from PIL import Image
                src = Path(img_path)
                dst = cache_dir / f"{src.stem}_{max_side}.png"
                if not dst.exists():
                    with Image.open(src) as im:
                        im = im.convert("RGB")
                        im.thumbnail((max_side, max_side))
                        im.save(dst)
                img_path = str(dst)
            f.write(json.dumps({"image_path": img_path,
                                "text": text or "Describe this screenshot."},
                               ensure_ascii=False) + "\n")
    print(f"[量化] 校准集（真实截图）: {out}，{len(rows)} 条")
    return out


def quantize_llmcompressor(model: str, out: str, scheme: str, calib: Path,
                           n_calib: int, train_file: Path,
                           quantize_vision: bool,
                           calib_max_side: int = 1280) -> None:
    calib = write_calibration_file(calib, n_calib, train_file, calib_max_side)

    # 序列长度按校准图算，而不是写死：max_side² / 28²（每 28x28 一个图像 token）
    # + 文本余量。写死会在换数据集/换分辨率时突然报
    # "Mismatch in image token count ... Likely due to truncation".
    import math
    tokens = (calib_max_side ** 2) / 784 if calib_max_side > 0 else 9000
    max_seq = max(2048, int(math.ceil((tokens + 512) / 1024)) * 1024)
    print(f"[量化] 校准图长边 {calib_max_side or '原图'}，"
          f"估计 {int(tokens)} 图像 token → max_seq_length={max_seq}")

    # ignore 名单在**加载模型之后**动态生成，见下方脚本。
    # 血泪教训：compressed-tensors 的 ignore 只对叶子模块（Linear）逐名匹配，
    # 写 "visual" / "visual.*" / "re:.*visual.*" 都不会把整棵子树排除，
    # 视觉塔照样被量化，于是留下 out_features=3420 的层
    # （Qwen2-VL 视觉 MLP 中间维度），在两个地方炸：
    #   量化时 "org_w_shape (3420) must be a multiple of group_size (128)"
    #   服务时 Marlin 内核要求 out_features 能被 64 整除
    ignore = [] if quantize_vision else ["lm_head"]
    print(f"[量化] ignore 列表: {ignore}"
          f"（{'含视觉塔' if quantize_vision else '跳过视觉塔'}）")

    # per-channel 方案：Ampere 上唯一能被 vLLM 内核吃下的 4bit 权重方案
    if scheme == "W4A16_channel":
        scheme = {
            "W4A16_channel": {
                "weights": {
                    "num_bits": 4, "type": "int", "symmetric": True,
                    "strategy": "channel",
                },
            },
        }
        print("[量化] 使用 per-channel W4A16（group_size=None，适配 SM86）")

    script = f'''
import json, os, torch
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

MODEL = {model!r}
OUT = {out!r}
SCHEME = {scheme!r}
IGNORE = {ignore!r}
IGNORE_VISION = {not quantize_vision!r}
MAX_SEQ = {max_seq}

rows = [json.loads(l) for l in open({str(calib)!r}, encoding="utf-8")]
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

def _to_ds(r):
    img = Image.open(r["image_path"]).convert("RGB")
    msgs = [{{"role": "user", "content": [
        {{"type": "image"}}, {{"type": "text", "text": r["text"]}}]}}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return {{"images": img, "text": text}}

ds = Dataset.from_list([_to_ds(r) for r in rows])


def _collate(features):
    """按模态字段正确拼接，交给 llmcompressor 的默认 collator 会把
    image_grid_thw 压成 1 维，模型里 `for t, h, w in grid_thw` 直接报
    "not enough values to unpack (expected 3, got 1)"。
    """
    import torch
    # 只保留模型真正吃的键；其余（如原始 text）丢掉，
    # 否则下游对 list 调 .dim() 会报 "'list' object has no attribute 'dim'"。
    keep = ("input_ids", "attention_mask", "pixel_values",
            "image_grid_thw", "labels", "token_type_ids")
    out = {{}}
    for k in keep:
        vals = [f[k] for f in features if k in f]
        if not vals:
            continue
        if not torch.is_tensor(vals[0]):
            vals = [torch.tensor(v) for v in vals]
        if k == "image_grid_thw":
            out[k] = torch.cat([v.reshape(-1, 3) for v in vals], dim=0)
        elif k == "pixel_values":
            out[k] = torch.cat([v.reshape(-1, v.shape[-1]) for v in vals], dim=0)
        else:
            # processor 返回的已经是 [1, L]，直接 stack 会变成 [1, 1, L]，
            # 模型里 attention_mask [1,L] 与 labels [1,1] 对不上，报
            # "IndexError: The shape of the mask [1, 3072] at index 1
            #  does not match the shape of the indexed tensor [1, 1]"。
            v2 = [v.squeeze(0) if (v.dim() > 1 and v.shape[0] == 1) else v
                  for v in vals]
            out[k] = (torch.stack(v2, dim=0)
                      if all(v.shape == v2[0].shape for v in v2)
                      else torch.cat(v2, dim=0))
    return out

# 必须自己先用多模态类加载再传实例。
# 只传路径的话，llmcompressor 内部走 AutoModelForCausalLM，
# 对 Qwen2_5_VLConfig 直接报 "Unrecognized configuration class"。
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)

# 视觉塔要"整棵子树"排除，只能把其中每个 Linear 逐名列出：
# ignore 只匹配叶子模块名，写容器名（"visual"）不会向下传播。
if IGNORE_VISION:
    base = [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and "visual" in n.split(".")]
    # 同时给出带与不带 "model." 前缀的两种写法：
    # named_modules() 给的是 model.visual.blocks.0.mlp.gate_proj，
    # 而 safetensors 里的键是 visual.blocks.0.mlp.gate_proj，
    # compressed-tensors 匹配的是后者。
    vision_linears = base + [n[len("model."):] for n in base
                             if n.startswith("model.")]
    IGNORE = IGNORE + vision_linears
    print("[量化] 排除视觉塔 Linear 数 =", len(vision_linears))

# recipe 必须在 IGNORE 定稿之后创建，否则拿到的是旧列表
# SCHEME 可以是预设名（"W4A16"，group_size=128），也可以传 dict。
# Ampere(3090, SM86) 上必须走 per-channel：
#   - Machete 内核要 SM90；
#   - AllSpark 在 Ampere 只支持 group_size=-1（per-channel）；
#   - Marlin 要求 out_features 能被 64 整除，而视觉 MLP 的 3420 不满足；
#   - BitBLAS 要额外安装。
# 用 group_size=128 的 W4A16 会得到"能量化成功、但 vLLM 加载不了"的产物。
recipe = AWQModifier(ignore=IGNORE, scheme=SCHEME, targets=["Linear"])

oneshot(
    model=model, dataset=ds, recipe=recipe,
    # 校准图是真实 GUI 截图（ShowUI-desktop 多为 3360x2100 / 2880x1800），
    # 按默认 max_pixels 不缩放，实测单图就有 **9000 个图像 token**。
    # 给 2048 或 8192 都会被截断，报
    # "Mismatch in image token count ... Likely due to truncation='max_length'"。
    output_dir=OUT, max_seq_length=MAX_SEQ,
    num_calibration_samples=len(rows),
    data_collator=_collate,
)
print("QUANT_DONE", OUT)

# ---------------------------------------------------------------------------
# 必须做的收尾：修 ignore 命名，否则 vLLM 加载不了这个产物。
#
# llmcompressor 把 ignore 写成**模块路径** `model.visual.blocks.0.mlp.gate_proj`，
# 而 safetensors 里的键是 `visual.blocks.0.mlp.gate_proj`（没有 `model.` 前缀）。
# vLLM 按权重键匹配 ignore，于是一条都没命中 → 它以为视觉塔也被量化了 →
# 去找 WNA16 内核 → 视觉 MLP 是 3420 维，Marlin 要求能被 64 整除 → 直接报错：
#   "Weight output_size_per_partition = 3420 is not divisible by min_thread_n = 64"
#
# 这一步原本是手工改 config.json 的，漏掉它量化产物就起不来，
# 所以固化进脚本。详见 docs/06-error-log.md 第 25 条。
# ---------------------------------------------------------------------------
import shutil

cfg_path = os.path.join(OUT, "config.json")
if os.path.exists(cfg_path):
    shutil.copy(cfg_path, cfg_path + ".bak")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config") or {{}}
    old = qc.get("ignore") or []
    fixed = set(old)
    for n in old:
        if n.startswith("model."):
            fixed.add(n[len("model."):])
    if fixed != set(old):
        qc["ignore"] = sorted(fixed)
        cfg["quantization_config"] = qc
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[修复] ignore 命名：{{len(old)}} -> {{len(fixed)}} 条"
              f"（同时保留带/不带 model. 前缀两种写法）")
    else:
        print("[修复] ignore 无需改动")
else:
    print("!! 没找到 config.json，跳过 ignore 修复")
'''
    subprocess.run([sys.executable, "-c", script], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="模型离线量化（AWQ）")
    # 默认量化**基座**而不是 SFT 产物：报告里的 AWQ 一档量化的是基座，
    # 这样"量化损失"才是相对同一起点度量的（见 docs/07-results.md 第 3 节）。
    ap.add_argument("--model",
                    default=os.environ.get("MODEL_DIR", "models")
                    + "/Qwen2.5-VL-3B-Instruct",
                    help="要量化的模型目录（默认取 $MODEL_DIR/Qwen2.5-VL-3B-Instruct）")
    ap.add_argument("--out", default="outputs/qwen25vl-3b-base-awq")
    ap.add_argument("--scheme", default="W4A16", help="W4A16(AWQ) / W8A8(INT8)")
    ap.add_argument("--num-calib", type=int, default=32,
                    help="校准样本数（256 条会让校准慢一个数量级，32 条足够）")
    ap.add_argument("--train-file", default=str(TRAIN_DEFAULT))
    ap.add_argument("--calib-max-side", type=int, default=768,
                    help="校准图长边（0=用原图，3360x2100 会让校准慢 5 倍）")
    ap.add_argument("--quantize-vision", action="store_true",
                    help="连视觉塔一起量化（对照实验用，默认跳过）")
    args = ap.parse_args()

    quantize_llmcompressor(
        args.model, args.out, args.scheme,
        Path(CALIB_DEFAULT), args.num_calib,
        Path(args.train_file), args.quantize_vision,
        args.calib_max_side,
    )
    tag = "含视觉塔" if args.quantize_vision else "跳过视觉塔"
    print(f"\n[完成] {args.out}（{tag}）")
    print("下一步：把 configs/serve/awq_int4.yaml 的 model 指向该目录后跑扫描")


if __name__ == "__main__":
    main()
