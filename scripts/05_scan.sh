#!/usr/bin/env bash
# 全档位扫描：对每个精度档位依次「起服务 → 测定位精度 → 测速度 → 汇总」
#
# 这是整个项目的主流程，产出 results/ 下的 grounding_*.json 与 bench_*.json。
#
# 用法：
#   bash scripts/05_scan.sh
#   VARIANTS="bf16 awq_int4" bash scripts/05_scan.sh     # 只扫部分档位
#   EVAL_LIMIT=200 bash scripts/05_scan.sh               # 小规模试跑
set -uo pipefail

PYTHON="${PYTHON:-python}"       # 云端传 venv 里的绝对路径
VARIANTS="${VARIANTS:-bf16 awq_int4 fp8}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
NUM_REQUESTS="${NUM_REQUESTS:-64}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_TOKENS="${MAX_TOKENS:-32}"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}"

mkdir -p results logs

# ---- 先做一次硬件体检，把不支持的档位剔掉 -----------------------------
# 3090 上 FP8 会被判 SKIP。与其等 vLLM 加载几分钟后在权重阶段报错，
# 不如启动前就判掉。
echo "==> 硬件能力体检"
"${PYTHON}" - <<'PY' > results/_variant_status.tsv
import sys
from pathlib import Path
sys.path.insert(0, "src")
from vfs import serve
for v in ["bf16", "awq_int4", "fp8"]:
    p = Path(f"configs/serve/{v}.yaml")
    if not p.exists():
        print(f"{v}\tMISSING\t配置文件不存在")
        continue
    ok, reason = serve.check_compatibility(serve.load_config(p))
    print(f"{v}\t{'OK' if ok else 'SKIP'}\t{reason}")
PY
cat results/_variant_status.tsv

wait_health() {
  local url="$1" max="${2:-300}" i=0
  while [ $i -lt $max ]; do
    if curl -sf "${url}/health" > /dev/null 2>&1; then
      return 0
    fi
    sleep 2
    i=$((i + 2))
  done
  return 1
}

for variant in $VARIANTS; do
  status=$(awk -F'\t' -v v="$variant" '$1==v {print $2}' results/_variant_status.tsv)
  reason=$(awk -F'\t' -v v="$variant" '$1==v {print $3}' results/_variant_status.tsv)

  if [ "$status" != "OK" ]; then
    echo
    echo "==> [${variant}] 跳过 —— ${reason}"
    continue
  fi

  MODEL=$("${PYTHON}" -c "import yaml;print(yaml.safe_load(open('configs/serve/${variant}.yaml'))['served_model_name'])")
  echo
  echo "==> [${variant}] 启动服务 (${MODEL})"

  "${PYTHON}" -m vfs.serve --config "configs/serve/${variant}.yaml" \
      --port "${PORT}" > "logs/serve_${variant}.log" 2>&1 &
  SRV_PID=$!

  if ! wait_health "${BASE_URL}" 600; then
    echo "    !! 服务启动超时，查看 logs/serve_${variant}.log"
    tail -20 "logs/serve_${variant}.log"
    kill ${SRV_PID} 2>/dev/null || true
    continue
  fi
  echo "    服务就绪"

  echo "    -> 定位精度评测"
  "${PYTHON}" -m vfs.grounding \
      --base-url "${BASE_URL}" \
      --model "${MODEL}" \
      --data-dir data/eval/ScreenSpot \
      --limit "${EVAL_LIMIT}" \
      --tag "${variant}" \
      --coord-space auto \
      --out-dir results || echo "    !! 评测失败（见上）"

  echo "    -> 速度压测（并发 ${CONCURRENCY}）"
  "${PYTHON}" -m vfs.bench \
      --base-url "${BASE_URL}" \
      --model "${MODEL}" \
      --num-requests "${NUM_REQUESTS}" \
      --concurrency "${CONCURRENCY}" \
      --max-tokens "${MAX_TOKENS}" \
      --tag "${variant}" \
      --out-dir results || echo "    !! 压测失败（见上）"

  echo "    停止服务"
  kill ${SRV_PID} 2>/dev/null || true
  wait ${SRV_PID} 2>/dev/null || true
  # vLLM 是多进程的，只杀主进程会留下 worker 继续占显存
  # （实测残留 21GB，下一档位直接 OOM）。按进程命令行补一刀。
  pkill -9 -f 'vllm[ ]serve' 2>/dev/null || true
  pkill -9 -f 'python -c from vllm' 2>/dev/null || true
  sleep 8
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "    显存残留: ${USED} MiB"
done

rm -f results/_variant_status.tsv

echo
echo "==> 汇总"
"${PYTHON}" -m vfs.report --results-dir results
