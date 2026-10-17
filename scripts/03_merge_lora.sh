#!/usr/bin/env bash
# 把 LoRA 权重合并回基座 -> 一份标准的、可被 vLLM / AWQ 直接加载的模型目录
#
# 为什么必须有这一步：QLoRA 出来的 adapter 依赖 4-bit 基座，
# 直接拿去做 AWQ 量化会报 dtype 不匹配（VLM 微调部署最常见的卡点）。
#
# 用法：
#   MODEL_DIR=/root/models LF_DIR=/root/work/LLaMA-Factory \
#   LF_CLI=/root/work/venvs/train/bin/llamafactory-cli \
#   OUT=/hy-tmp/models/merged bash scripts/03_merge_lora.sh
set -euo pipefail

LF_DIR="${LF_DIR:-/root/work/LLaMA-Factory}"
LF_CLI="${LF_CLI:-llamafactory-cli}"     # 绝对路径，避免 ssh 里 PATH 被外层展开
BASE_MODEL="${BASE_MODEL:-${MODEL_DIR:?请设置 MODEL_DIR}/Qwen2.5-VL-3B-Instruct}"
ADAPTER="${ADAPTER:-saves/qwen25vl-3b-gui/lora/sft}"
# 合并后约 7GB。根分区通常吃紧，默认写数据盘；需要长期保存再拷回 /root。
OUT="${OUT:-/hy-tmp/models/qwen25vl-3b-gui-sft-merged}"
TEMPLATE="${TEMPLATE:-qwen2_vl}"

ADAPTER_ABS="$(realpath "${ADAPTER}")"
mkdir -p "$(dirname "${OUT}")"
OUT_ABS="$(realpath -m "${OUT}")"

echo "==> 合并 LoRA"
echo "    base   : ${BASE_MODEL}"
echo "    adapter: ${ADAPTER_ABS}"
echo "    out    : ${OUT_ABS}"

cd "${LF_DIR}"
"${LF_CLI}" export \
    --model_name_or_path "${BASE_MODEL}" \
    --adapter_name_or_path "${ADAPTER_ABS}" \
    --template "${TEMPLATE}" \
    --trust_remote_code true \
    --export_dir "${OUT_ABS}" \
    --export_size 5 \
    --export_legacy_format false

echo
echo "==> 合并完成：${OUT_ABS}"
echo "    该目录可直接作为 vLLM 的 --model，也可作为 AWQ 量化的输入"
