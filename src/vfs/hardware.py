"""硬件能力探测 —— 决定哪些精度档位在当前机器上**能跑**。

动机
----
大多数 VLM 部署教程直接写「BF16 / FP8 / INT4 三选一」，
但精度能不能用取决于 GPU 的微架构而非用户意愿，典型例子：

    RTX 3090 (Ampere, SM 8.6)  **不支持** FP8
    RTX 4090 (Ada,    SM 8.9)  支持 FP8
    RTX 50xx (Blackwell, SM 12.0) 支持 FP8 / FP4

在 3090 上照抄 Ada 的 FP8 配置，vLLM 会在加载阶段直接抛错，
浪费一次动辄十几分钟的模型加载。本模块在跑 benchmark 之前就把
不可用的档位剔除掉，让「一条命令跑出全对比表」这件事真正成立。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional

# 计算能力 -> Marketing 代号。键是 (major, minor)
_ARCH_NAME = {
    (7, 0): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere (A100/HGX)",
    (8, 6): "Ampere (RTX 30xx)",
    (8, 9): "Ada Lovelace (RTX 40xx / L40)",
    (9, 0): "Hopper (H100 / H800)",
    (10, 0): "Blackwell (GB100)",
    (12, 0): "Blackwell (RTX 50xx)",
}

# vLLM 各 quantization 方式的前置条件
# require: fp8 -> SM >= 8.9 ; fp4 -> SM >= 12.0 ; awq/gptq/压缩炸弹模型 -> 需要预先量化好的权重
_QUANT_REQUIREMENT = {
    "bf16": "无特殊要求",
    "fp16": "无特殊要求",
    "awq": "需要预先量化好的 AWQ 权重",
    "gptq": "需要预先量化好的 GPTQ 权重",
    "gptq_marlin": "需要预先量化好的 GPTQ 权重 + Marlin kernel",
    "compressed-tensors": "需要 llm-compressor 产出的权重",
    "fp8": "需要 SM >= 8.9（Ada Lovelace / Hopper 及以上）",
    "nvfp4": "需要 SM >= 12.0（Blackwell）",
    "bitsandbytes": "无需预量化，但推理速度通常慢于 BF16",
}


@dataclass
class GPUInfo:
    """单张 GPU 的关键属性。"""

    index: int
    name: str
    sm_version: tuple[int, int]
    arch_name: str
    total_mem_gib: float
    driver_version: str = ""
    cuda_version: str = ""

    @property
    def supports_fp8(self) -> bool:
        return self.sm_version >= (8, 9)

    @property
    def supports_nvfp4(self) -> bool:
        return self.sm_version >= (12, 0)

    @property
    def supports_bf16(self) -> bool:
        # BF16 自 Ampere 起普及
        return self.sm_version >= (8, 0)

    @property
    def supports_flash_attn(self) -> bool:
        return self.sm_version >= (8, 0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sm_version"] = f"{self.sm_version[0]}.{self.sm_version[1]}"
        d["supports_fp8"] = self.supports_fp8
        d["supports_nvfp4"] = self.supports_nvfp4
        d["supports_bf16"] = self.supports_bf16
        return d


def _run_smi(args: list[str]) -> Optional[str]:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout if out.returncode == 0 else None
    except (subprocess.SubprocessError, OSError):
        return None


def detect_gpus() -> list[GPUInfo]:
    """通过 nvidia-smi 读取 GPU 列表。取不到信息时返回空列表而不是报错，
    因为在纯 CPU 的开发机上仍需要能 import 本包。"""
    smi = _run_smi(
        [
            "--query-gpu=index,name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not smi:
        return []

    gpus: list[GPUInfo] = []
    for line in smi.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx = int(parts[0])
        name = parts[1]
        try:
            major, minor = (int(x) for x in parts[2].split("."))
        except ValueError:
            major, minor = 0, 0
        try:
            mem_mib = float(parts[3])
        except ValueError:
            mem_mib = 0.0
        gpus.append(
            GPUInfo(
                index=idx,
                name=name,
                sm_version=(major, minor),
                arch_name=_ARCH_NAME.get((major, minor), f"Unknown SM{major}{minor}"),
                total_mem_gib=round(mem_mib / 1024, 2),
                driver_version=parts[4] if len(parts) > 4 else "",
            )
        )
    return gpus


def cuda_version() -> str:
    smi = _run_smi(["--query-gpu=cuda_version", "--format=csv,noheader"])
    return smi.strip().splitlines()[0].strip() if smi else "unknown"


def quant_method_available(method: str, gpu: GPUInfo, has_pretrained_weights: bool = True) -> tuple[bool, str]:
    """判断某个 vLLM quantization 方法能否在该 GPU 上使用。

    返回 ``(可用与否, 原因)``。
    """
    m = method.lower()
    if m in ("bf16", "fp16", "auto", "none", ""):
        if m == "bf16" and not gpu.supports_bf16:
            return False, f"{gpu.name} 不支持 BF16，请改用 fp16"
        return True, "OK（由 vLLM 在加载时转精度）"

    if m in ("fp8", "fp8_dynamic", "fp8-tensorwise"):
        return (
            (True, "OK") if gpu.supports_fp8
            else (False, f"{gpu.arch_name} (SM{gpu.sm_version[0]}{gpu.sm_version[1]}) 不支持 FP8；3090 请改用 AWQ/GPTQ-INT4")
        )

    if m in ("nvfp4", "fp4"):
        return (
            (True, "OK") if gpu.supports_nvfp4
            else (False, "NVFP4 需要 Blackwell (SM 12.0+)")
        )

    if m in ("awq", "awq_marlin", "gptq", "gptq_marlin", "compressed-tensors", "gguf"):
        return (True, "OK") if has_pretrained_weights else (
            False, "该档位需要先离线量化出权重，请先跑 scripts/04_quantize.py"
        )

    if m == "bitsandbytes":
        return True, "OK（在线量化，适合无预量化权重时兜底）"

    return False, f"未知的 quantization 方法：{method}"


def summarize() -> dict:
    """汇总本机 GPU 与可用精度档位，供 CLI 打印。"""
    gpus = detect_gpus()
    if not gpus:
        return {"gpu_count": 0, "gpus": [], "available_quant": [], "cuda_version": "unknown"}

    primary = gpus[0]
    available = []
    for method in ("bf16", "fp16", "fp8", "nvfp4", "awq", "gptq", "bitsandbytes"):
        ok, reason = quant_method_available(method, primary)
        available.append({
            "method": method,
            "available": ok,
            "reason": reason,
            "requirement": _QUANT_REQUIREMENT.get(method, "-"),
        })

    return {
        "gpu_count": len(gpus),
        "gpus": [g.to_dict() for g in gpus],
        "available_quant": available,
        "cuda_version": cuda_version(),
    }


def main() -> None:
    print(json.dumps(summarize(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
