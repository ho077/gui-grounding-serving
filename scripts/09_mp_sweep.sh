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
# 08_mixed_precision.py 用它做"量化超参是否与 V0 一致"的自检
export AWQ_BASE
# configs/serve/*.yaml 里写的是 ${MODEL_DIR}/Qwen2.5-VL-3B-Instruct 这种占位，
# serve.py 用 os.path.expandvars 展开。不 export 的话它会展开成
# "/Qwen2.5-VL-3B-Instruct"（空变量），vLLM 直接退出且日志几乎为空。
export MODEL_DIR="${MODEL_DIR:-$(dirname "$MODEL")}"
OUTROOT="${OUTROOT:-/hy-tmp/models}"
EVAL_SIDE=1264

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 用 KiB 读显存占用，避免 MiB/GiB 单位在不同驱动上解析不一致
gpu_used_kib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '
}

# 显存可能因惰性回收而滞后报告，所以要"杀了之后的若干秒内仍 <=2GiB"才算干净
wait_gpu_free() {
  kill_server
  local i=""
  for i in $(seq 1 30); do
    used=$(gpu_used_kib)
    if [ "${used:-9999999}" -le 2097152 ]; then
      echo "[显存] 已释放（${used} KiB，等待 $((i * 3))s）"
      return 0
    fi
    sleep 3
  done
  echo "[警告] 显存仍有 $(gpu_used_kib) KiB 未释放，继续尝试"
  return 1
}

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

  # 断点续跑：同一档位的 JSON 已存在就跳过（sweep 中途失败时不用从头再来，
  # 每档大约 25 分钟）
  if [ -f "results/grounding_awq_${name}_full${EVAL_SIDE}.json" ]; then
    echo "========== [$name] 已有结果，跳过 =========="
    return 0
  fi

  echo "========== [$name] $(date +%H:%M:%S) =========="

  # 必须在量化**之前**清干净：llmcompressor 要把整份 BF16 载到 GPU 上跑校准，
  # 而上一档的 vLLM 服务此时还占着 20+ GiB —— 顺序写反就会 CUDA OOM
  # （实测：keep_attn 量化时上一档服务仍在，报 out of memory 而不是量化失败）。
  if [ ${#qargs[@]} -gt 0 ]; then
    wait_gpu_free
  else
    kill_server
  fi

  if [ ${#qargs[@]} -gt 0 ] && [ ! -f "$dir/config.json" ]; then
    echo "[1/3] 量化 -> $dir"
    "$TRAIN_VENV" scripts/08_mixed_precision.py \
      --model "$MODEL" --out "$dir" --tag "$name" "${qargs[@]}" \
      2>&1 | grep -E "^\[(结构|档位|位宽|profile|量化|口径)\]|Traceback|Error" || true
  else
    echo "[1/3] 跳过量化（复用已有产物 $dir）"
    # 已量化的老产物也要过一遍 ignore 命名修复（混合位宽档必须：
    # 否则 vLLM 会把保 FP16 的层建成 packed 参数，加载 KeyError）
    if [ ${#qargs[@]} -gt 0 ]; then
      "$TRAIN_VENV" scripts/fix_awq_ignore.py "$dir" 2>&1 | tail -4 || true
    fi
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
if [ -f "results/grounding_base_full_${EVAL_SIDE}.json" ]; then
  echo "========== [base_full_${EVAL_SIDE}] 已有结果，跳过 =========="
else
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
  echo "----- [base_full_${EVAL_SIDE}] done $(date +%H:%M:%S) -----"
fi

# V0：干净的 4bit 基线（现有全量化产物，不再量化，只用来对齐口径）
run_case v0_base "$AWQ_BASE" ""

# V1：attention 保 FP16 —— 只多约 0.5GB，是本实验信息量最大的一档
run_case keep_attn "${OUTROOT}/qwen25vl-3b-base-awq-keep-attn" "--keep attn"

# V2：深层的全部 Linear 保 FP16（24-35 层）—— Pareto 曲线的第三个点。
#     若 coord 回归主要依赖深层，这一档会比 keep_attn 挽回更多，代价约 +1.4GB。
run_case keep_deep "${OUTROOT}/qwen25vl-3b-base-awq-keep-deep" "--keep-layers 24-35"

# V3：MLP 保 FP16、attention 仍压 4bit —— 归因矩阵的第四个格子。
#   2x2 矩阵（attn x mlp 各自 int4/FP16）：
#       mlp=int4, attn=int4  -> V0        67.7%（-11.9）
#       mlp=int4, attn=FP16  -> keep_attn 70.0%（-9.7）  => attention 只值 +2.3
#       mlp=FP16, attn=int4  -> keep_mlp  ?             <- 本档：若 ≈ V0 则损失全在 MLP
#       mlp=FP16, attn=FP16  -> BF16      79.7%
#   代价约 +3.4GB（≈回到 BF16 体积），纯为归因而设，不作为部署候选。
run_case keep_mlp "${OUTROOT}/qwen25vl-3b-base-awq-keep-mlp" "--keep mlp"

# V4/V5：MLP 内部再拆 —— 2x2 矩阵已经说明"12 点损失里 11 点在 MLP"，
#   剩下的问题是这 11 点在 gate_up（1824M 参，压缩的是宽度维）还是
#   down_proj（611M 参）。两者的 FP16 代价差 3 倍：
#     保 down   → +0.85GB（3.2 -> 4.05GB）
#     保 gate_up→ +2.55GB（3.2 -> 5.75GB）
#   若"只保 down"就能拿回大头，部署 Pareto 点会好看很多。
run_case keep_mlp_down "${OUTROOT}/qwen25vl-3b-base-awq-keep-mlp-down" "--keep mlp_down"
run_case keep_mlp_gateup "${OUTROOT}/qwen25vl-3b-base-awq-keep-mlp-gateup" "--keep mlp_gateup"

# V6：只把量化粒度改细 —— group_size 128 -> 64，**没有任何一层保 FP16**。
#   这是比"保 FP16"便宜得多的补救：权重仍是 4bit，只有 scale 存储翻倍
#   （2775M 参 / 64 = 43M 个 scale，约 +0.05 GB）。
#   12.4 节已证明损失集中在 gate_up 的**量化误差**上，若细化粒度能拿回大头，
#   Pareto 上会多一个"3.3 GB / 接近 76%"的点，比 keep_mlp_gateup 实用得多。
#   风险：Marlin 内核对 group_size 的支持（32/64/128）需实测确认。
run_case g64 "${OUTROOT}/qwen25vl-3b-base-awq-g64" "--group-size 64"

# V7/V8：把 α 搜索补细后再试细粒度 —— 12.7 节的负结果怀疑是搜索收敛错了。
#   llmcompressor 的 AWQModifier._compute_best_scale 用 `n_grid=20` +
#   `for ratio in range(n_grid)`，于是 ratio 只覆盖 {0, 0.05, ..., 0.95}，
#   **ratio=1.0（纯激活驱动缩放）永远不被试**，duo_scaling 下 0.05 步长也偏粗。
#   --alpha-grid 40 把网格改成 0~1.0 共 41 个点（步长 0.025），
#   其余（校准集、ignore、量化超参）与 g64 完全一致 —— 单一变量。
#   注意搜索本身是量化网格感知的（_pseudo_quantize_tensor 带 group_size），
#   所以这一档是在 g64 网格上重搜 α，正是"联合搜索"缺的那一步。
run_case g64_fine "${OUTROOT}/qwen25vl-3b-base-awq-g64-fine" "--group-size 64 --alpha-grid 40"
# V8：同样补细的网格，试更细的粒度 32（Marlin 支持），看曲线往哪边走
run_case g32_fine "${OUTROOT}/qwen25vl-3b-base-awq-g32-fine" "--group-size 32 --alpha-grid 40"

kill_server
echo "===== SWEEP DONE $(date +%H:%M:%S) ====="
