#!/usr/bin/env bash
# 第三假设：校准集太小导致 α 搜索的 x_mean 有噪声。
#   g64        32 张校准 + 20 点网格 = 63.9%
#   g64_fine   32 张校准 + 41 点网格 = 63.3%   （网格不是杠杆）
#   g64_calib  101 张校准（训练集全部去重图片）+ 41 点网格 = ?
#   （本档：数据量是不是杠杆。注意训练数据文件里的图片路径是旧的
#     /hy-tmp/train_images_repro/，实际在 /hy-tmp/train_images/，
#     已用 sed 修正成 gui_grounding_train_fixed.json）
# 其余（ignore / 量化超参 / 模型）全部一致，仍是单一变量。
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

SERVE_VENV="${SERVE_VENV:-/root/work/venvs/serve/bin/python}"
TRAIN_VENV="${TRAIN_VENV:-/root/work/venvs/train/bin/python}"
MODEL="${MODEL:-/root/models/Qwen2.5-VL-3B-Instruct}"
AWQ_BASE="${AWQ_BASE:-/hy-tmp/models/qwen25vl-3b-base-awq}"
export MODEL_DIR="${MODEL_DIR:-$(dirname "$MODEL")}"
export AWQ_BASE
OUTROOT="${OUTROOT:-/hy-tmp/models}"
EVAL_SIDE=1264

kill_server() {
  pkill -f "vfs.grounding" 2>/dev/null || true
  for p in $(ps -eo pid,args | grep -E '[s]pawn_main|[v]llm.entrypoints' | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
  sleep 8
}

wait_ready() {
  for i in $(seq 1 60); do
    sleep 5
    if curl -s -m 5 http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
      echo "[就绪] $((i * 5))s"; return 0
    fi
  done
  echo "[超时] 服务没起来"; return 1
}

name="g64_calib128"
dir="${OUTROOT}/qwen25vl-3b-base-awq-${name}"
tag="awq_${name}_full${EVAL_SIDE}"

if [ -f "results/grounding_${tag}.json" ]; then
  echo "[跳过] ${tag} 已有结果"
  exit 0
fi

echo "========== [${name}] $(date +%H:%M:%S) =========="
kill_server
"$TRAIN_VENV" scripts/08_mixed_precision.py \
  --model "$MODEL" --out "$dir" --tag "$name" \
  --group-size 64 --alpha-grid 40 --num-calib 101 \
  --train-file data/train/gui_grounding_train_fixed.json \
  2>&1 | grep -E "^\[(α|结构|档位|位宽|profile|量化|口径)\]|Traceback|Error" || true

kill_server
AWQ_DIR="$dir" setsid nohup "$SERVE_VENV" -m vfs.serve \
  --config configs/serve/awq_base.yaml \
  --mm-processor-kwargs '{"max_pixels":1605632}' \
  > "logs/serve_${name}.log" 2>&1 < /dev/null &
wait_ready || { echo "[失败] $name 服务未就绪"; exit 1; }
mid=$(curl -s -m 5 http://127.0.0.1:8000/v1/models \
      | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
[ -n "$mid" ] || mid="$(basename "$dir")"
echo "[服务] $mid"
"$SERVE_VENV" -m vfs.grounding --base-url http://127.0.0.1:8000 \
  --model "$mid" --data-dir data/eval/ScreenSpot \
  --coord-space pixel --tag "$tag" \
  --max-side "$EVAL_SIDE" --concurrency 8 --timeout 180 --out-dir results
echo "----- [${name}] done $(date +%H:%M:%S) -----"
