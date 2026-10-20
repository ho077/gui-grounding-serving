#!/usr/bin/env bash
# 混合位宽 Pareto 的运行时一侧：每档起服务 -> 压测 -> 采样显存 -> 关服务。
#
# 精度那一半见 scripts/09_mp_sweep.sh；这里补的是"部署时真正花的显存和吞吐"。
# 权重体积只是下界，运行时还要算 KV cache、激活、以及 vLLM 的显存池预留
# （gpu_memory_utilization），所以两者都列出来才有意义。
#
# 口径与 09_mp_sweep.sh 对齐：同一份 BF16 校准集、同一套 W4A16 量化超参。
# 断点续跑：某档 bench JSON 已存在就跳过。
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

SERVE_VENV="${SERVE_VENV:-python}"
if [ "$SERVE_VENV" = "python" ] && [ -x /root/work/venvs/serve/bin/python ]; then
  SERVE_VENV=/root/work/venvs/serve/bin/python
fi
if ! "$SERVE_VENV" -c "import vllm" > /dev/null 2>&1; then
  echo "[致命] SERVE_VENV=$SERVE_VENV 里没有 vllm"
  exit 1
fi

MODEL="${MODEL:-/root/models/Qwen2.5-VL-3B-Instruct}"
AWQ_BASE="${AWQ_BASE:-/hy-tmp/models/qwen25vl-3b-base-awq}"
export MODEL_DIR="${MODEL_DIR:-$(dirname "$MODEL")}"
OUTROOT="${OUTROOT:-/hy-tmp/models}"
EVAL_SIDE=1264          # 与评测口径一致：客户端长边 1264
BENCH_SIDE="${BENCH_SIDE:-$EVAL_SIDE}"

kill_server() {
  pkill -f "vfs.grounding" 2>/dev/null || true
  for p in $(ps -eo pid,args | grep -E '[s]pawn_main|[v]llm.entrypoints' | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
  sleep 8
  echo "[清理] 显存 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
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

# $1=name  $2=model_dir  $3=yaml(bf16|awq)
bench_case() {
  local name="$1" dir="$2" yaml="$3"
  if [ -f "results/bench_mp_${name}_c4.json" ]; then
    echo "========== [$name] 已有压测结果，跳过 =========="
    return 0
  fi
  echo "========== [$name] $(date +%H:%M:%S) =========="
  kill_server
  if [ "$yaml" = "bf16" ]; then
    setsid nohup "$SERVE_VENV" -m vfs.serve \
      --config configs/serve/bf16_bench.yaml \
      > "logs/serve_bench_${name}.log" 2>&1 < /dev/null &
  else
    AWQ_DIR="$dir" setsid nohup "$SERVE_VENV" -m vfs.serve \
      --config configs/serve/awq_base.yaml \
      --mm-processor-kwargs '{"max_pixels":1605632}' \
      > "logs/serve_bench_${name}.log" 2>&1 < /dev/null &
  fi
  wait_ready || { echo "[失败] $name 服务未就绪"; return 1; }
  local mid
  mid=$(curl -s -m 5 http://127.0.0.1:8000/v1/models \
        | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" \
        2>/dev/null || true)
  [ -n "$mid" ] || mid="$(basename "$dir")"
  echo "[压测] $mid：${BENCH_SIDE}px 图 / max_tokens 32 / 并发 4 / 64 请求"
  "$SERVE_VENV" -m vfs.bench --base-url http://127.0.0.1:8000 \
    --model "$mid" --tag "mp_${name}" \
    --image-size "$BENCH_SIDE" --max-tokens 32 \
    --concurrency 4 --num-requests 64 --num-warmup 4 \
    --wait-health --sample-vram --out-dir results \
    || echo "[警告] $name 压测失败"
  kill_server
}

mkdir -p logs

bench_case bf16            "$MODEL"                                     bf16
bench_case v0              "$AWQ_BASE"                                  awq
bench_case keep_attn       "${OUTROOT}/qwen25vl-3b-base-awq-keep-attn"        awq
bench_case keep_mlp_gateup "${OUTROOT}/qwen25vl-3b-base-awq-keep-mlp-gateup"  awq
bench_case keep_mlp        "${OUTROOT}/qwen25vl-3b-base-awq-keep-mlp"         awq

# 粒度三档也压一遍：group_size 越细，每步要读的 scale 越多（128->32 是 4 倍），
# 看 TPOT 会不会被 scale 访存拖慢（W4A16 本来就是访存瓶颈）
bench_case g64            "${OUTROOT}/qwen25vl-3b-base-awq-g64"              awq
bench_case g32_fine       "${OUTROOT}/qwen25vl-3b-base-awq-g32-fine"         awq
# 两个负收益档也压上，运行时那一列才完整
bench_case keep_mlp_down  "${OUTROOT}/qwen25vl-3b-base-awq-keep-mlp-down"    awq
bench_case keep_deep      "${OUTROOT}/qwen25vl-3b-base-awq-keep-deep"        awq

echo "===== BENCH DONE $(date +%H:%M:%S) ====="
