#!/usr/bin/env bash
# 混合位宽 sweep：每档「量化 -> 起服务 -> 全量评测」，全部走干净口径。
#
# 为什么评测口径钉死在 --max-side 1264 而不是 1280
# ---------------------------------------------------------------------
# 两件事同时在约束这个数字：
#
# (1) 1280 是趣旨模型真正的精度拐点。全量 1272 条实测：
#       原分辨率 80.0% / 1280 80.6%（p90 偏移 338→225）/ 1024 75.5%
#     即 1280 与原图持平而 1024 掉 5.1 点，所以评测必须落在 1280 附近，
#     否则测的是"压缩过头"而不是"量化本身的代价"。
#     （400 子集上曾得出"1024 与原图持平"的错误结论，是抽样偏差，
#       见 docs/06-error-log.md 第 35 条。）
#
# (2) AWQ 服务受 vLLM 0.8.5 的 dummy profiling 限制，
#     max_pixels 只能设在 1,605,632。客户端送进去的图一旦超过它，
#     服务端会**再缩一次**，而 GT 框是按客户端缩放后的尺度存的 ——
#     于是又变成不同源（第 34 条 error-log 那个坑的第三次变体）。
#       1280² = 1,638,400  > 1,605,632   ← 越线
#       1264² = 1,597,696  < 1,605,632   ← 安全，且只差 1.25% 像素
#
# 早先用不同的 --max-side 测同一份 AWQ，掉点被这个缩放二次污染得面目全非：
#
#   口径           送进服务端峰值 px  是否被二次缩放  相对同档 BF16 的掉点
#   max-side 1024  1.05M             否             -11.2（desktop -15.6）
#   max-side 1512  2.29M             部分           -12.1（desktop -23.8）
#   max-side 1600  2.56M             是             -16.4（desktop -27.0）
#
# desktop 恶化最明显，因为它宽高比最接近 1（1280x1024 这类），最早越线。
#
# 用法：TRAIN_VENV=/root/work/venvs/train/bin/python SERVE_VENV=... \
#       bash scripts/09_mp_sweep.sh
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

TRAIN_VENV="${TRAIN_VENV:-python}"
SERVE_VENV="${SERVE_VENV:-python}"

# 本机这套是在 /root/work/venvs/{serve,train} 里跑的，直接 bash 执行时
# 上面两个变量不会自动带进来，会回落成系统 python —— 而系统 python 没有
# vllm / llmcompressor，症状是 serve.py 打一行"No module named 'vllm'"就退出，
# 日志里几乎什么都没有。这里按存在性兜一层。
# 注意这里必须显式判断 `= "python"`：上面已经把 SERVE_VENV 赋成非空的
# "python" 了，此时再写 ${SERVE_VENV:-/path} 是不会生效的（变量非空）。
if [ "$SERVE_VENV" = "python" ] && [ -x /root/work/venvs/serve/bin/python ]; then
  SERVE_VENV=/root/work/venvs/serve/bin/python
fi
if [ "$TRAIN_VENV" = "python" ] && [ -x /root/work/venvs/train/bin/python ]; then
  TRAIN_VENV=/root/work/venvs/train/bin/python
fi
if ! "$SERVE_VENV" -c "import vllm" > /dev/null 2>&1; then
  echo "[致命] SERVE_VENV=$SERVE_VENV 里没有 vllm。"
  echo "       请用 SERVE_VENV=/path/to/venv/bin/python bash scripts/09_mp_sweep.sh"
  exit 1
fi
echo "[环境] SERVE_VENV=$SERVE_VENV"
echo "[环境] TRAIN_VENV=$TRAIN_VENV"

MODEL="${MODEL:-/root/models/Qwen2.5-VL-3B-Instruct}"
AWQ_BASE="${AWQ_BASE:-/hy-tmp/models/qwen25vl-3b-base-awq}"
# configs/serve/*.yaml 里写的是 ${MODEL_DIR}/Qwen2.5-VL-3B-Instruct 这种占位，
# serve.py 用 os.path.expandvars 展开。不 export 的话它会展开成
# "/Qwen2.5-VL-3B-Instruct"（空变量），vLLM 直接退出且日志几乎为空。
export MODEL_DIR="${MODEL_DIR:-$(dirname "$MODEL")}"
OUTROOT="${OUTROOT:-/hy-tmp/models}"
EVAL_SIDE=1264

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
      echo "[就绪] $((i * 5))s"; return 0
    fi
  done
  echo "[超时] 服务没起来"; return 1
}

# $1=name  $2=model_dir  $3=quant args（空则跳过量化）
run_case() {
  local name="$1" dir="$2"; local -a qargs=()
  if [ -n "${3:-}" ]; then
    IFS=' ' read -r -a qargs <<< "$3"
  fi
  echo "========== [$name] $(date +%H:%M:%S) =========="

  if [ ${#qargs[@]} -gt 0 ]; then
    echo "[1/3] 量化 -> $dir"
    "$TRAIN_VENV" scripts/08_mixed_precision.py \
      --model "$MODEL" --out "$dir" --tag "$name" "${qargs[@]}" \
      2>&1 | grep -E "^\[(结构|档位|位宽|profile|量化)\]|Traceback|Error" || true
  else
    echo "[1/3] 跳过量化（复用现有产物 $dir）"
  fi

  kill_server
  echo "[2/3] 起服务"
  AWQ_DIR="$dir" setsid nohup "$SERVE_VENV" -m vfs.serve \
    --config configs/serve/awq_base.yaml \
    --mm-processor-kwargs '{"max_pixels":1605632}' \
    > "logs/serve_${name}.log" 2>&1 < /dev/null &
  wait_ready || { echo "[失败] $name 服务未就绪"; return 1; }

  # served-model-name 由 vLLM 按权重路径生成，不同产物不一样。
  # 与其猜，不如直接问服务 —— 猜错会得到 404，还会导致 landed requests 被静默丢弃。
  local mid
  mid=$(curl -s -m 5 http://127.0.0.1:8000/v1/models \
        | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" \
        2>/dev/null || true)
  [ -n "$mid" ] || mid="$(basename "$dir")"
  echo "[服务] served-model-name = $mid"

  echo "[3/3] 评测 @ max-side $EVAL_SIDE（全量）"
  "$SERVE_VENV" -m vfs.grounding --base-url http://127.0.0.1:8000 \
    --model "$mid" --data-dir data/eval/ScreenSpot \
    --coord-space pixel --tag "awq_${name}_full${EVAL_SIDE}" \
    --max-side "$EVAL_SIDE" --concurrency 8 --timeout 180 --out-dir results
  echo "----- [$name] done $(date +%H:%M:%S) -----"
}

mkdir -p logs

# BF16 基座同口径对照：没有它就算不出"4bit 真实损失"（每档都要有各自的同档基线）
echo "========== [base_full_${EVAL_SIDE}] $(date +%H:%M:%S) =========="
kill_server
setsid nohup "$SERVE_VENV" -m vfs.serve \
  --config configs/serve/bf16_base.yaml > "logs/serve_base_${EVAL_SIDE}.log" 2>&1 < /dev/null &
wait_ready || { echo "[失败] base 服务未就绪"; exit 1; }
base_mid=$(curl -s -m 5 http://127.0.0.1:8000/v1/models \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
[ -n "$base_mid" ] || base_mid="$(basename "$MODEL")"
"$SERVE_VENV" -m vfs.grounding --base-url http://127.0.0.1:8000 \
  --model "$base_mid" --data-dir data/eval/ScreenSpot \
  --coord-space pixel --tag "base_full_${EVAL_SIDE}" \
  --max-side "$EVAL_SIDE" --concurrency 8 --timeout 180 --out-dir results

# V0：干净的 4bit 基线（现有全量化产物，不再量化，只用来对齐口径）
run_case v0_base "$AWQ_BASE" ""

# V1：attention 保 FP16 —— 只多约 0.5GB，是本实验信息量最大的一档
run_case keep_attn "${OUTROOT}/qwen25vl-3b-base-awq-keep-attn" "--keep attn"

kill_server
echo "===== SWEEP DONE $(date +%H:%M:%S) ====="
