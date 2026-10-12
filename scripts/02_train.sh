#!/usr/bin/env bash
# GUI grounding LoRA 微调
#
# 用法：
#   bash scripts/02_train.sh
#   bash scripts/02_train.sh configs/sft/qwen25vl_3b_gui_lora_3090.yaml
set -euo pipefail

CFG="${1:-configs/sft/qwen25vl_3b_gui_lora_3090.yaml}"
LF_DIR="${LF_DIR:-/root/LLaMA-Factory}"

echo "==> 训练配置: ${CFG}"

# 把本项目产出的数据集注册给 LLaMA-Factory。
# 直接覆盖它的 dataset_info.json 太粗暴，这里做合并：以它原有条目为底，再叠加我们的。
python - <<'PY'
import json, os
from pathlib import Path

lf = Path(os.environ.get("LF_DIR", "/root/LLaMA-Factory"))
ours_p = Path("data/dataset_info.json")
if not ours_p.exists():
    raise SystemExit("未找到 data/dataset_info.json，先跑 scripts/01_prepare_data.py")

target = lf / "data" / "dataset_info.json"
base = {}
if target.exists():
    base = json.loads(target.read_text(encoding="utf-8"))
base.update(json.loads(ours_p.read_text(encoding="utf-8")))
target.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[OK] 已注册数据集 -> {target}")
PY

cd "${LF_DIR}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_DISABLED=true

llamafactory-cli train "$(realpath "${OLDPWD}/${CFG}")"

echo
echo "==> 训练完成。LoRA 权重在 saves/ 下，下一步："
echo "    bash scripts/03_merge_lora.sh"
