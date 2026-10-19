#!/usr/bin/env bash
# Jedi-3B（NeurIPS'25）全量 ScreenSpot 评测，走官方 ComputerUse 协议。
#
# 为什么主评测脚本跑不了这个模型
# ------------------------------
# 判定"能不能直接接进来"看的不是 architecture 字段，而是它的推理协议：
#   - 主评测 src/vfs/grounding.py 用 {"point":[x,y]} 问答式 prompt
#   - Jedi 用的是 **工具调用**：system 里给 ComputerUse 函数签名 + 屏幕分辨率，
#     模型输出 <tool_call>{"name":"computer_use","arguments":{...,"coordinate":[x,y]}}</tool_call>
# 喂错协议等于把模型推到分布外，同批样本上会明显掉点（对比数见 docs/07-results.md §11.4）。
# 复刻官方协议的代码在 src/vfs/jedi_eval.py：工具签名模板 + <tool_call> 解析
# + 与 processor 同规则的 smart_resize（边长对齐 28、像素 <= 1.8M）。
#
# 为什么要用 1.8M px 预算
# -----------------------
# vLLM 0.8.5 在这个 checkpoint 上对 > 约 1.9M px 的请求**永不调度**：无报错、无日志、
# GPU/CPU 空转（docs/06-error-log.md 第 33 条）。客户端先按同规则缩好，
# 服务端 resize 就变成空操作，绕开这个坑。代价是输入上限低于论文的 1080p+ 口径。
#
# 用法：VENV=/path/to/venv/bin/python bash scripts/07_jedi_eval.sh [tag]
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
V="${VENV:-python}"
TAG="${1:-jedi_official}"
MAX_PX="${MAX_PX:-1800000}"

# 服务没起就先起来（--max_pixels 2116800 对齐 Jedi 自己的 preprocessor_config）
if ! curl -s -m 5 http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
  echo "[起服务] Jedi-3B-1080p"
  setsid nohup "$V" -m vfs.serve \
    --config configs/serve/jedi.yaml \
    --mm-processor-kwargs "{\"max_pixels\": 2116800}" \
    > logs/serve_jedi.log 2>&1 < /dev/null &
  for i in $(seq 1 60); do
    sleep 5
    curl -s -m 5 http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && break
  done
fi

"$V" -m vfs.jedi_eval --model "${MODEL_NAME:-jedi-3b}" --tag "$TAG" \
  --max-pixels "$MAX_PX" --concurrency 8 --timeout 120 --out-dir results

echo "===== full done ====="
curl -s -m 8 http://127.0.0.1:8000/v1/models > /dev/null && echo "server: healthy" || echo "server: DEAD"
