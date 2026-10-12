"""按 YAML 配置拉起 vLLM 服务，并在**拉起之前**做硬件能力校验。

为什么必须前置校验
------------------
一次 vLLM 加载 Qwen3-VL-4B 要几十秒到几分钟。若在 3090 上误配 FP8，
要等到权重加载阶段才报 dtype 不支持 —— 每个档位浪费几分钟，
跑一轮 BF16/FP8/INT4 的扫描就浪费一轮 coffee break。
这里直接在启动前用 ``hardware`` 模块判掉，不支持就明确跳过并给出替代建议。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from . import hardware


def load_config(path: Path) -> dict:
    if yaml is None:
        raise SystemExit("需要 PyYAML：pip install pyyaml")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_cmd(cfg: dict, extra: list[str] | None = None) -> list[str]:
    """把 YAML 翻译成 vllm serve 的参数向量。"""
    cmd = ["vllm", "serve", str(cfg["model"])]

    mapping = {
        "served_model_name": "--served-model-name",
        "dtype": "--dtype",
        "quantization": "--quantization",
        "max_model_len": "--max-model-len",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "limit_mm_per_prompt": "--limit-mm-per-prompt",
        "tensor_parallel_size": "--tensor-parallel-size",
        "port": "--port",
        "host": "--host",
        "max_num_seqs": "--max-num-seqs",
    }
    for key, flag in mapping.items():
        v = cfg.get(key)
        if v not in (None, "", False):
            cmd += [flag, str(v)]

    # prefix caching：默认 False（benchmark 要求），显式判断是否要开
    if not cfg.get("enable_prefix_caching", False):
        cmd.append("--no-enable-prefix-caching")
    if cfg.get("enforce_eager", False):
        cmd.append("--enforce-eager")

    cmd += extra or []
    return cmd


def check_compatibility(cfg: dict) -> tuple[bool, str]:
    gpus = hardware.detect_gpus()
    if not gpus:
        return False, "未检测到 GPU"
    gpu = gpus[0]
    method = cfg.get("quantization") or "auto"
    ok, reason = hardware.quant_method_available(
        method, gpu,
        has_pretrained_weights=bool(cfg.get("pretrained_quant", True)),
    )
    return ok, f"{gpu.name}（{gpu.arch_name}）: {reason}"


def main() -> None:
    ap = argparse.ArgumentParser(description="按配置拉起 vLLM 服务")
    ap.add_argument("--config", required=True, help="configs/serve/*.yaml")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令并检查兼容性")
    ap.add_argument("--skip-check", action="store_true", help="跳过硬件能力校验")
    args, extra = ap.parse_known_args()

    cfg = load_config(Path(args.config))

    if not args.skip_check:
        ok, reason = check_compatibility(cfg)
        print(f"[兼容] {reason}")
        if not ok:
            print(f"[跳过] 当前硬件无法运行 `{'quantization=' + str(cfg.get('quantization'))}`，"
                  f"请改用 AWQ/GPTQ-INT4 档位。")
            return

    cmd = build_cmd(cfg, extra)
    print("[命令]", " ".join(cmd))
    if args.dry_run:
        return

    print("[启动] 服务开始加载，日志直接透传 …\n")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
