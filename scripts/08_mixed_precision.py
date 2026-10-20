#!/usr/bin/env python
"""混合位宽消融：在 LLM 主干内部按模块类别 / 层深决定「谁保持 FP16」。

必须在装了 torch + llmcompressor 的解释器里运行（本机是 train venv）：
    /root/work/venvs/train/bin/python scripts/08_mixed_precision.py --help

为什么要有这个脚本
------------------
``scripts/04_quantize.py`` 只有一个开关：视觉塔压不压。但实测已经把
归因做掉了一半 —— 现有 AWQ 产物的 ``config.json`` 里 ignore 有 162 条拢共
全是 ``visual.*``（视觉塔 32 层里的每个 Linear），也就是：

    **同口径下 4bit 那 −16.5 点的损失，100% 发生在 LLM 主干内部。**
    （base_side1600 80.5% → awq_side1600 64.0%）

于是问题收敛成：LLM 主干里，损失具体压在哪些 Linear 上？能不能用
最小的额外位宽预算把它买回来？

为什么不顺便把视觉塔也压了
--------------------------
视觉塔 MLP 中间维是 **3420**，而 vLLM 的 Marlin WNA16 内核要求
``out_features`` 能被 64 整除（3420 / 64 = 53.4）。所以在 W4A16 + Marlin
这条路上视觉塔**不具备可量化性**（详见 docs/06-error-log.md 第 25 条）。
这不是工程取舍，是内核的硬约束 —— 反过来讲，它恰好把「保住视觉塔」
变成一条不可辩驳的默认设置。

各档位的位宽账本（Qwen2.5-VL-3B：36 层 / hidden 2048 / mlp 11008 / kv-heads 2）
--------------------------------------------------------------------------------
每层 attn ≈ 9.4M 参（q 4.19 + k 0.52 + v 0.52 + o 4.19）
每层 mlp  ≈ 67.6M 参（gate/up/down 各 22.5）
⇒ attn 只占每层参数的 **12.2%**。

所以「保 attention」是全部选项里最便宜的一档，账如下（权重部分，
视觉塔恒 1.0GB FP16、embed 恒 0.6GB FP16 未计入 LLM 部分）：

| 档位            | LLM 4bit 部分 | LLM FP16 部分 | LLM 权重合计 |
|-----------------|---------------|---------------|--------------|
| 全 4bit（现有） | 2774M         | 0             | 1.39 GB      |
| 保 attn         | 2434M         | 340M          | 1.90 GB      |
| 保 mlp          | 340M          | 2434M         | 5.04 GB      |
| 全 FP16         | 0             | 2774M         | 5.55 GB      |

也就是说：保 attn 只多花约 **0.5GB**。如果这一档能把 −16.5 点救回大半，
就得到一个很硬的结论 —— VLM 的 4bit 量化瓶颈不在参数量最大的 FFN，
而在那 12% 的 attention 投影上。反过来若保 mlp 才是关键，
那这条路的代价就接近「干脆别量化」，结论同样清楚。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TRAIN_DEFAULT = ROOT / "data" / "train" / "gui_grounding_train.json"
CALIB_DEFAULT = ROOT / "data" / "calib_gui.jsonl"


def _load_write_calibration():
    """复用 04_quantize.py 里已验证过的校准集构造逻辑。

    不走 importlib.import_module("04_quantize")：模块名以数字开头，
    虽然通常也能 import，但用 spec_from_file_location 不依赖这一点。
    """
    p = SCRIPTS / "04_quantize.py"
    spec = importlib.util.spec_from_file_location("quant04", p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"无法加载 {p}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.write_calibration_file


def strip_prefix(name: str) -> str:
    """去掉 `model.` 前缀。

    named_modules() 给的是 `model.language.layers.0.mlp.gate_proj`，
    而 safetensors 的键有时不带 `model.`。两边都要写（见 04_quantize.py）。
    """
    return name[len("model."):] if name.startswith("model.") else name


def classify(name: str) -> str:
    """把一个 Linear 的模块路径归到细粒度类别。

    除了粗粒度（attn / mlp / vision / head），MLP 再拆成
    gate_up / down —— vLLM 侧 gate_proj+up_proj 融合成 gate_up_proj，
    两者的参数量差 3 倍（1824M vs 611M），量化敏感度未必相同，
    部署时"只保哪一个"是不同价位。
    """
    p = strip_prefix(name)
    parts = p.split(".")
    if parts[0] == "lm_head":
        return "head"
    if parts[0] == "visual":
        return "vision"
    if "self_attn" in parts:
        return "attn"
    if "mlp" in parts:
        leaf = parts[-1]
        if leaf.startswith(("gate", "up")):
            return "mlp_gateup"
        return "mlp_down"
    return "other"


def layer_index(name: str) -> int | None:
    """从 `...layers.<i>.mlp.gate_proj` 里取出层号 i。"""
    parts = strip_prefix(name).split(".")
    try:
        i = parts.index("layers")
    except ValueError:
        return None
    if i + 1 < len(parts) and parts[i + 1].isdigit():
        return int(parts[i + 1])
    return None


def parse_ranges(spec: str, n_max: int = 1000) -> set[int]:
    """解析 "0-11,24-35,7" 这种层号写法。"""
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(chunk))
    return {i for i in out if 0 <= i < n_max}


def decide(name: str, keep_groups: set[str], keep_layers: set[int] | None
           ) -> str:
    """返回 'keep'（保 FP16）或 'quant'。"""
    g = classify(name)
    # 视觉塔与 lm_head 恒保 FP16：前者受 Marlin 的 64 整除约束，
    # 后者直接决定输出的坐标词元，压它的收益远小于风险。
    if g in ("vision", "head"):
        return "keep"
    # `mlp` 是 `mlp_gateup` + `mlp_down` 的并集：写 mlp 等价于两个细类都保
    if g in keep_groups or (
            g.startswith("mlp_") and "mlp" in keep_groups):
        return "keep"
    if keep_layers is not None:
        i = layer_index(name)
        if i is not None and i in keep_layers:
            return "keep"
    return "quant"


def sizeof_module(m) -> int:
    """Linear 的参数个数（含 bias，bias 一般没有）。"""
    n = 0
    for p in m.parameters(recurse=False):
        n += p.numel()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM 主干混合位宽消融")
    ap.add_argument("--model",
                    default=os.environ.get("MODEL_DIR", "models")
                    + "/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", required=True, help="产物目录")
    ap.add_argument("--tag", default="", help="profile 文件名后缀")
    ap.add_argument("--keep", default="",
                    help="保持 FP16 的模块类别，逗号分隔：attn,mlp,vision,head")
    ap.add_argument("--keep-layers", default="",
                    help="这些层号的**全部** Linear 保持 FP16，如 0-11,24-35")
    ap.add_argument("--num-calib", type=int, default=32)
    ap.add_argument("--train-file", default=str(TRAIN_DEFAULT))
    ap.add_argument("--calib-max-side", type=int, default=768)
    ap.add_argument("--scheme", default="W4A16",
                    help="量化 preset 名（llmcompressor 0.5.2 只接受 preset 名）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只统计 lut：谁被保留、参数分布、位宽账本，不真量化")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText

    keep_groups = {s.strip() for s in args.keep.split(",") if s.strip()}
    keep_layers = (parse_ranges(args.keep_layers)
                   if args.keep_layers.strip() else None)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True)

    linears = [(n, m) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear)]
    if not linears:
        raise SystemExit("没找到任何 Linear，结构可能变了，请检查 classify()")

    ignore: list[str] = []
    n_keep = n_quant = p_keep = p_quant = p_keep_llm = 0
    groups: dict[str, list[int]] = {}
    for n, m in linears:
        g = classify(n)
        verdict = decide(n, keep_groups, keep_layers)
        p = sizeof_module(m)
        groups.setdefault(g, []).append(p)
        if verdict == "keep":
            n_keep += 1
            p_keep += p
            ignore.append(n)
            # 只有本来会被量化的（LLM 主干内的）模块才谈得上"多花钱"：
            # 视觉塔 + lm_head 无论哪档都是 FP16，把它们算进增量为虚增（实测
            # keep_attn 磁盘增量 3.7-3.2=0.5GB，而按 p_keep 全会算成 1.84GB）
            if g in ("attn", "mlp"):
                p_keep_llm += p
        else:
            n_quant += 1
            p_quant += p

    # ignore 对叶子名逐条匹配，且要同时给带 / 不带 model. 前缀两种写法
    ignore = ignore + [strip_prefix(n) for n in ignore]

    gb = lambda nparams, nbits: nparams * nbits / 8 / 1024 ** 3
    llm_p = sum(groups.get("attn", [])) + sum(groups.get("mlp", []))
    all_p = sum(sum(v) for v in groups.values())
    print(f"[结构] Linear 共 {len(linears)} 个 / {all_p/1e6:.0f}M 参；"
          f"其中 LLM 主干 {llm_p/1e6:.0f}M"
          f"（attn {sum(groups.get('attn',[]))/1e6:.0f}M, "
          f"mlp {sum(groups.get('mlp',[]))/1e6:.0f}M），"
          f"视觉塔+head {sum(groups.get('vision',[])+groups.get('head',[]))/1e6:.0f}M")
    print(f"[档位] 保留 {n_keep} 个 Linear / {p_keep/1e6:.0f}M 参；"
          f"量化 {n_quant} 个 / {p_quant/1e6:.0f}M 参")
    print(f"[位宽] LLM 权重 ≈ {gb(p_quant,4)+gb(p_keep,16):.2f} GB"
          f"（全 FP16 需 {gb(p_quant+p_keep,16):.2f} GB）"
          f" → 相对全量化多花 {gb(p_keep_llm,16)-gb(p_keep_llm,4):.2f} GB"
          f"（主干内保留 {p_keep_llm/1e6:.0f}M 参）")

    profile = {
        "tag": args.tag or Path(args.out).name,
        "out": args.out,
        "keep_groups": sorted(keep_groups),
        "keep_layers": sorted(keep_layers) if keep_layers else None,
        "n_linear_total": len(linears),
        "n_keep": n_keep, "n_quant": n_quant,
        "params_keep": p_keep, "params_quant": p_quant,
        "params_by_group": {k: sum(v) for k, v in groups.items()},
        "llm_weight_gb": round(gb(p_quant, 4) + gb(p_keep, 16), 3),
        "llm_weight_gb_fp16": round(gb(p_quant + p_keep, 16), 3),
    }
    prof_dir = ROOT / "results" / "mp"
    prof_dir.mkdir(parents=True, exist_ok=True)
    (prof_dir / f"profile_{profile['tag']}.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[profile] {prof_dir}/profile_{profile['tag']}.json")

    if args.dry_run:
        print("[dry-run] 跳过量化")
        return

    write_calibration = _load_write_calibration()
    calib = write_calibration(
        Path(CALIB_DEFAULT), args.num_calib, Path(args.train_file),
        args.calib_max_side)

    tokens = (args.calib_max_side ** 2) / 784 if args.calib_max_side > 0 else 9000
    max_seq = max(2048, int(math.ceil((tokens + 512) / 1024)) * 1024)
    print(f"[量化] max_seq_length={max_seq}, ignore={len(ignore)} 条")

    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    rows = [json.loads(l) for l in calib.open(encoding="utf-8")]

    def _to_ds(r):
        from PIL import Image
        img = Image.open(r["image_path"]).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": r["text"]}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=False)
        return {"images": img, "text": text}

    ds = Dataset.from_list([_to_ds(r) for r in rows])

    def _collate(features):
        """必须自己拼：默认 collator 会把 image_grid_thw 压成 1 维，
        模型里 `for t, h, w in grid_thw` 直接 unpack 失败。
        """
        keep = ("input_ids", "attention_mask", "pixel_values",
                "image_grid_thw", "labels", "token_type_ids")
        out = {}
        for k in keep:
            vals = [f[k] for f in features if k in f]
            if not vals:
                continue
            if not torch.is_tensor(vals[0]):
                vals = [torch.tensor(v) for v in vals]
            if k == "image_grid_thw":
                out[k] = torch.cat([v.reshape(-1, 3) for v in vals], dim=0)
            elif k == "pixel_values":
                out[k] = torch.cat([v.reshape(-1, vals[0].shape[-1])
                                    for v in vals], dim=0)
            else:
                v2 = [v.squeeze(0) if (v.dim() > 1 and v.shape[0] == 1) else v
                      for v in vals]
                out[k] = (torch.stack(v2, dim=0)
                          if all(v.shape == v2[0].shape for v in v2)
                          else torch.cat(v2, dim=0))
        return out

    # device_map="cpu" 加载的，量化前挪到 GPU
    model.to("cuda:0")
    # llmcompressor 0.5.2 起 AWQModifier 只接受 **preset 名**（字符串），
    # 手写 {"W4A16_channel": {...}} 会直接被 pydantic 拒掉：
    #   `scheme` must either be a preset scheme name or a dictionary of
    #   preset scheme names
    # 这里用 "W4A16"（group_size=128 / symmetric / int4），与已有全量化产物
    # qwen25vl-3b-base-awq 的 config.json 完全一致 —— 混合位宽必须和 V0 基线
    # 用同一套量化口径，否则比出来的不是"保了哪些层"，而是"换了套量化方案"。
    scheme = args.scheme
    recipe = AWQModifier(ignore=ignore, scheme=scheme, targets=["Linear"])
    oneshot(model=model, dataset=ds, recipe=recipe, output_dir=args.out,
            max_seq_length=max_seq, num_calibration_samples=len(rows),
            data_collator=_collate)
    print("QUANT_DONE", args.out)

    # ignore 命名修复：llmcompressor 写的是模块路径（带 model.），
    # 而 vLLM 按 safetensors 权重键匹配，必须对两种写法都给全。
    import subprocess
    subprocess.run([sys.executable, str(SCRIPTS / "fix_awq_ignore.py"),
                    args.out], check=False)

    # 口径自检：混合位宽的意义在于"只改动了哪些层保精度"，所以量化超参必须
    # 和既有 V0 产物逐字段一致。不一致时下面的对比就变成两件事混在一起。
    import json as _json
    ref_env = os.environ.get("AWQ_BASE")
    ref = str(ref_env) if ref_env else ""
    try:
        ours = _json.loads((Path(args.out) / "config.json").read_text())[
            "quantization_config"]
        g = list(ours.get("config_groups", {}).values())[0]
        w = g.get("weights", {})
        sig = dict(bits=w.get("num_bits"), sym=w.get("symmetric"),
                   group=w.get("group_size"), strat=w.get("strategy"),
                   act=list(g.get("input_activations", {}) or {}.keys()))
        print(f"[口径] 本产物量化超参 = {sig}")
        if ref and Path(ref).exists():
            r = _json.loads((Path(ref) / "config.json").read_text())[
                "quantization_config"]
            rg = list(r.get("config_groups", {}).values())[0]
            rw = rg.get("weights", {})
            rsig = dict(bits=rw.get("num_bits"), sym=rw.get("symmetric"),
                        group=rw.get("group_size"), strat=rw.get("strategy"),
                        act=list(rg.get("input_activations", {}) or {}.keys()))
            ok = sig == rsig
            print(f"[口径] V0 基线超参     = {rsig}  →  {'一致' if ok else '**不一致**'}")
            if not ok:
                print("[警告] 与 V0 基线量化方案不同，本档结果不能直接与之对减！")
    except Exception as e:  # 自检失败不该中断流程，但要看得见
        print(f"[口径] 自检跳过：{type(e).__name__} {e}")


if __name__ == "__main__":
    main()
