"""把 compressed-tensors W4A16 产物**解量化回 BF16**，另存为一个普通模型目录。

用途：隔离"权重"与"推理内核"。
  - 原产物走 vLLM Marlin g64 内核 -> 63.9%
  - 同一份权重量化解量化后走普通 BF16 路径 -> ？
若后者显著高于前者，说明 g64 的**权重本身没问题**，掉点是 vLLM 0.8.5 的
Marlin group_size=64 路径造成的；若两者接近，则是权重侧的真实差异。

用法：python dequant_export.py <产物目录> <输出目录>
"""
import json
import os
import shutil
import sys

import torch
from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
    unpack_from_int32)
from safetensors.torch import load_file, save_file

DROP_SUFFIX = (".weight_scale", ".weight_shape", ".weight_zero_point",
               ".weight_g_idx", ".g_idx_sort_indices")


def dequant_one(packed: torch.Tensor, scale: torch.Tensor,
                shape, group_size: int) -> torch.Tensor:
    w = unpack_from_int32(packed, 4, torch.Size(list(shape))).float()
    rows, cols = w.shape
    ng = cols // group_size
    s = scale.float().reshape(rows, ng)
    s_full = s.repeat_interleave(group_size, dim=1)[:, :cols]
    return (w * s_full).to(torch.bfloat16)


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    qc = cfg.get("quantization_config") or {}
    gs = 128
    try:
        g = list(qc.get("config_groups", {}).values())[0]
        gs = int(g.get("weights", {}).get("group_size") or 128)
    except Exception:  # noqa: BLE001
        pass
    print(f"[信息] group_size={gs}，源目录 {src}")

    shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    out = {}
    n_q = n_fp = 0
    for sh in shards:
        tensors = load_file(os.path.join(src, sh))
        for k, v in tensors.items():
            if k.endswith(DROP_SUFFIX):
                continue
            if k.endswith(".weight_packed"):
                base = k[: -len(".weight_packed")]
                scale = tensors.get(base + ".weight_scale")
                shape = tensors.get(base + ".weight_shape")
                out[base + ".weight"] = dequant_one(v, scale, shape, gs)
                n_q += 1
            else:
                out[k] = v
                n_fp += 1
        print(f"  {sh}: 处理完，累计 {n_q} 个量化张量 / {n_fp} 个原样张量")

    os.makedirs(dst, exist_ok=True)
    save_file(out, os.path.join(dst, "model.safetensors"),
              metadata={"format": "pt"})
    for f in os.listdir(src):
        if f.endswith(".safetensors") or f == "recipe.yaml":
            continue
        s = os.path.join(src, f)
        d = os.path.join(dst, f)
        if os.path.isfile(s):
            shutil.copy(s, d)
    # 变成普通 BF16 模型：去掉量化配置，显式声明 dtype
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = "bfloat16"
    cfg["dtype"] = "bfloat16"
    with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[完成] {dst}：{n_q} 个解量化 + {n_fp} 个原样，"
          f"quantization_config 已移除")


if __name__ == "__main__":
    main()
