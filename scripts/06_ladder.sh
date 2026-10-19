#!/usr/bin/env bash
# 分辨率阶梯 + 同口径 4bit 对比。
#
# 为什么要单独跑这一组（而不是沿用 05_scan.sh 的结果）
# -----------------------------------------------------
# 早先"压分辨率 -59.7 点"（80.0% → 20.3%）是测量口径污染，不是模型能力：
#   ① 图在**服务端**被降到 1.6M px，而 GT 框仍是原图坐标（两者不同源，系统性偏移）
#   ② 当时的 --coord-space auto 把绝对像素误判成 [0,1000] 归一化（再乘一次分辨率）
# 判据：被降采样的 mobile/web 崩到个位数，几乎没被降采样的 desktop 只掉十几点
# （desktоp 只有 35% 的图 > 1.6M px，mobile/web 是 100%）。详见 docs/07-results.md §11。
#
# 本脚本的做法：三件事必须同源
#   1. 缩放只在**客户端**做（--max-side），服务端 max_pixels 足够大 => resize 为空操作
#   2. GT 框跟着缩放后同比缩（src/vfs/grounding.py 里已修复，并有边界校验）
#   3. 坐标口径显式 --coord-space pixel
#
# 用法：MODEL_DIR=/path/to/models VENV=/path/to/venv/bin/python bash scripts/06_ladder.sh
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
V="${VENV:-python}"
MODEL="${MODEL:-qwen25vl-3b-base}"
AWQ_MODEL="${AWQ_MODEL:-qwen25vl-3b-base-awq}"
AWQ_DIR="${AWQ_DIR:-${MODEL_DIR}/qwen25vl-3b-base-awq}"
# 阶梯统一用同一份 400 条子集，保证前后可比；全量档另跑。
SUBSET="${SUBSET:-400}"

kill_server() {
  pkill -f "vfs.grounding" 2>/dev/null || true
  pkill -f "vfs.jedi_eval" 2>/dev/null || true
  for p in $(ps -eo pid,args | grep -E '[r]esource_tracker|[s]pawn_main|[v]llm.entrypoints' | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
  sleep 8
  echo "[清理] 显存 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}
wait_ready() {
  for i in $(seq 1 60); do
    sleep 5
    if curl -s -m 5 http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
      echo "[就绪] ${i}x5s"; return 0
    fi
  done
  echo "[超时] 服务没起来"; return 1
}
eval_tag() { # $1=tag  $2=model  其余透传
  local tag="$1" model="$2"; shift 2
  "$V" -m vfs.grounding --base-url http://127.0.0.1:8000 \
    --model "$model" --data-dir data/eval/ScreenSpot \
    --coord-space pixel --tag "$tag" "$@" \
    --concurrency 8 --timeout 120 --out-dir results
}

########## A. BF16 基座：分辨率阶梯（同一份子集） ##########
kill_server
MODEL_DIR="$MODEL_DIR" setsid nohup "$V" -m vfs.serve \
  --config configs/serve/bf16_base.yaml > logs/serve_base_ladder.log 2>&1 < /dev/null &
sleep 3; wait_ready || { echo "A serve failed"; exit 1; }

for SIDE in 1024 768 512; do
  echo "----- base @ max-side $SIDE -----"
  eval_tag "base_side$SIDE" "$MODEL" --max-side "$SIDE" --limit "$SUBSET"
done
# 最顶端：原分辨率（同子集），用来读"拐点在哪"
eval_tag base_native_sub400 "$MODEL" --limit "$SUBSET"

########## B. AWQ int4：与 A 完全同口径（含全量档） ##########
kill_server
AWQ_DIR="$AWQ_DIR" setsid nohup "$V" -m vfs.serve \
  --config configs/serve/awq_base.yaml \
  --mm-processor-kwargs '{"max_pixels":1605632}' > logs/serve_awq_ladder.log 2>&1 < /dev/null &
sleep 3; wait_ready || { echo "B serve failed"; exit 1; }

eval_tag awq_side1512 "$AWQ_MODEL" --max-side 1512
eval_tag awq_side1024 "$AWQ_MODEL" --max-side 1024 --limit "$SUBSET"

echo "===== all done ====="
