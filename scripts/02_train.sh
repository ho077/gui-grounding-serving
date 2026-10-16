#!/usr/bin/env bash
# GUI grounding LoRA 微调
#
# 用法：
#   MODEL_DIR=/root/models LF_DIR=/root/work/LLaMA-Factory bash scripts/02_train.sh
#   bash scripts/02_train.sh configs/sft/qwen25vl_3b_gui_lora_3090.yaml
#
# 两个环境变量（与 configs/serve/*.yaml 同一套 ${VAR} 约定）：
#   MODEL_DIR  基座权重目录，配置里写 ${MODEL_DIR}/Qwen2.5-VL-3B-Instruct
#   LF_DIR     LLaMA-Factory 根目录
#   PYTHON     做配置展开用的解释器（需要 PyYAML），默认 python
set -euo pipefail

CFG="${1:-configs/sft/qwen25vl_3b_gui_lora_3090.yaml}"
LF_DIR="${LF_DIR:-/root/work/LLaMA-Factory}"
PYTHON="${PYTHON:-python}"

echo "==> 训练配置: ${CFG}"

# 把本项目产出的数据集注册给 LLaMA-Factory。
# 直接覆盖它的 dataset_info.json 太粗暴，这里做合并：以它原有条目为底，再叠加我们的。
"${PYTHON}" - <<'PY'
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

# 展开配置里的 ${VAR}。
# LLaMA-Factory 直接读 YAML，不认 ${...}；不展开就得为每台机器维护一份副本
# （本机在 D 盘、云端在 /root），必然出现"改了一处漏一处"。
# 这里复用 vfs.serve._expand_env，保证 serve / train 两侧展开行为一致。
EXPANDED="$("${PYTHON}" - "${CFG}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "src")
import yaml
from vfs.serve import _expand_env

src = Path(sys.argv[1])
cfg = _expand_env(yaml.safe_load(src.read_text(encoding="utf-8")))
out = Path(".cache") / f"{src.stem}.expanded.yaml"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
               encoding="utf-8")
print(out)
PY
)"
CFG_PATH="$(realpath "${EXPANDED}")"
echo "==> 展开后配置: ${CFG_PATH}"

cd "${LF_DIR}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_DISABLED=true

llamafactory-cli train "${CFG_PATH}"

echo
echo "==> 训练完成。LoRA 权重在 saves/ 下，下一步："
echo "    bash scripts/03_merge_lora.sh"
