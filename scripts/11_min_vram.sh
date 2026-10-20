#!/usr/bin/env bash
# 每档混合位宽产物，最低能以多少 gpu_memory_utilization 起服务。
#
# 为什么不是 bench 里的 peak_vram：那个数在默认 0.9 下被设置项本身顶住
# （所有档都是 21.6 GiB），区分度为零。部署时真正的问题是"这块卡最少要多大
# 才能塞下它"，所以这里对每个档位从低到高试，直到能返回一次推理。
#
# 已知参照（vLLM 0.8.5 / 3090 24GB）：全 int4 AWQ 在 0.45（≈10.8GB）能起，
# BF16 在同预算下直接 No available memory for the cache blocks。
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

SERVE_VENV="${SERVE_VENV:-python}"
if [ "$SERVE_VENV" = "python" ] && [ -x /root/work/venvs/serve/bin/python ]; then
  SERVE_VENV=/root/work/venvs/serve/bin/python
fi

MODEL="${MODEL:-/root/models/Qwen2.5-VL-3B-Instruct}"
AWQ_BASE="${AWQ_BASE:-/hy-tmp/models/qwen25vl-3b-base-awq}"
export MODEL_DIR="${MODEL_DIR:-$(dirname "$MODEL")}"
OUTROOT="${OUTROOT:-/hy-tmp/models}"
# 从 0.25 试到 0.90，步长 0.05；第一个能服务的就是最低预算
STEPS="${STEPS:-0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.70 0.80 0.90}"

kill_server() {
  pkill -f "vfs.grounding" 2>/dev/null || true
  for p in $(ps -eo pid,args | grep -E '[s]pawn_main|[v]llm.entrypoints' | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
  sleep 6
}

# $1=name  $2=model_dir  $3=yaml(bf16|awq)
probe_min() {
  local name="$1" dir="$2" yaml="$3"
  echo "========== [$name] 最低显存预算 =========="
  local best=""
  for u in $STEPS; do
    kill_server
    if [ "$yaml" = "bf16" ]; then
      VLLM_UTIL="$u" setsid nohup "$SERVE_VENV" -m vfs.serve \
        --config configs/serve/bf16_bench.yaml \
        --gpu-memory-utilization "$u" \
        > "logs/serve_minvram_${name}_${u}.log" 2>&1 < /dev/null &
    else
      AWQ_DIR="$dir" VLLM_UTIL="$u" setsid nohup "$SERVE_VENV" -m vfs.serve \
        --config configs/serve/awq_base.yaml \
        --gpu-memory-utilization "$u" \
        --mm-processor-kwargs '{"max_pixels":1605632}' \
        > "logs/serve_minvram_${name}_${u}.log" 2>&1 < /dev/null &
    fi
    local ok=""
    for i in $(seq 1 24); do
      sleep 5
      if curl -s -m 5 http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
        ok=1; break
      fi
      # 已经明确失败（OOM / 内核不支持）就别等了
      if grep -qa -E "No available memory|CUDA out of memory|Engine core initialization failed" \
         "logs/serve_minvram_${name}_${u}.log" 2>/dev/null; then
        break
      fi
    done
    if [ -n "$ok" ]; then
      best="$u"
      echo "  ✅ util=$u（$(python3 -c "print('%.1f'%($u*24))") GiB 预算）可以服务"
      break
    else
      echo "  ❌ util=$u 起不来"
    fi
  done
  if [ -z "$best" ]; then
    echo "  [$name] 在 $STEPS 范围内都起不来"
  fi
  echo "[$name] min_util=$best"
  kill_server
}

mkdir -p logs
probe_min bf16            "$MODEL"                                     bf16
probe_min v0              "$AWQ_BASE"                                  awq
probe_min keep_attn       "${OUTROOT}/qwen25vl-3b-base-awq-keep-attn"        awq
probe_min keep_mlp_gateup "${OUTROOT}/qwen25vl-3b-base-awq-keep-mlp-gateup"  awq
probe_min keep_mlp        "${OUTROOT}/qwen25vl-3b-base-awq-keep-mlp"         awq
echo "===== MINVRAM DONE $(date +%H:%M:%S) ====="
