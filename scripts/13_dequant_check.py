"""解量化 compressed-tensors pack-quantized 权重，与 BF16 原权重比误差。

用途：判断"g64 反而比 g128 差 3.8 点"是真实现象还是推理侧问题。
若 g64 的解量化权重更接近 BF16（相对误差更小）而评测更差 => 问题在推理内核；
若 g64 的解量化权重本身就更远 => 是量化侧的真实差异。
"""
import glob

import torch
from safetensors import safe_open

BF16_DIR = "/root/models/Qwen2.5-VL-3B-Instruct"
V0 = "/hy-tmp/models/qwen25vl-3b-base-awq/model.safetensors"
G64 = "/hy-tmp/models/qwen25vl-3b-base-awq-g64/model.safetensors"

PROBES = [
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.17.mlp.gate_proj",
    "model.layers.35.self_attn.q_proj",
    "model.layers.35.mlp.down_proj",
]

_bf16_cache = {}


def bf16_weight(base: str):
    if base in _bf16_cache:
        return _bf16_cache[base]
    for shard in sorted(glob.glob(BF16_DIR + "/*.safetensors")):
        with safe_open(shard, "pt") as f:
            if base + ".weight" in f.keys():
                w = f.get_tensor(base + ".weight").float()
                _bf16_cache[base] = w
                return w
    _bf16_cache[base] = None
    return None


def unpack_int4(packed: torch.Tensor, order: str) -> torch.Tensor:
    """每个 int32 装 8 个 int4，按列交错展开；两种 nibble 序都试。"""
    p = packed.to(torch.int32)
    out = torch.zeros(p.shape[0], p.shape[1] * 8, dtype=torch.int8)
    bits = range(8) if order == "low" else range(7, -1, -1)
    for pos, bit in zip(range(8), bits):
        nib = ((p >> (4 * bit)) & 0xF).to(torch.int8)
        nib = torch.where(nib >= 8, nib - 16, nib)      # 4bit 有符号
        out[:, pos::8] = nib
    return out


def rel_error(path: str, base: str, group_size: int):
    with safe_open(path, "pt") as f:
        if base + ".weight_packed" not in f.keys():
            return None, "缺失"
        packed = f.get_tensor(base + ".weight_packed")
        scale = f.get_tensor(base + ".weight_scale").float()
        shape = f.get_tensor(base + ".weight_shape").tolist()
    ref = bf16_weight(base)
    if ref is None:
        return None, "无 BF16 参照"
    rows, cols = packed.shape[0], packed.shape[1] * 8
    ng = cols // group_size
    s = scale.reshape(rows, ng)
    s_full = s.repeat_interleave(group_size, dim=1)[:, :cols]
    best = None
    for order in ("low", "high"):
        deq = unpack_int4(packed, order).float() * s_full
        e = ((deq - ref).norm() / ref.norm()).item()
        if best is None or e < best:
            best = e
    return best, f"shape={tuple(shape)}"


def main() -> None:
    print("%-36s %10s %10s   %s" % ("模块", "V0 g128", "g64", "判定"))
    for p in PROBES:
        e0, m0 = rel_error(V0, p, 128)
        e1, m1 = rel_error(G64, p, 64)
        if e0 is None or e1 is None:
            print("%-36s %10s %10s   %s" % (p, e0, e1, m0 or m1))
            continue
        verdict = "g64 更接近 BF16" if e1 < e0 else "g64 更远（更差）"
        print("%-36s %10.5f %10.5f   %s" % (p, e0, e1, verdict))


if __name__ == "__main__":
    main()
