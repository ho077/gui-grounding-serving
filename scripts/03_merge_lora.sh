#!/usr/bin/env bash
# 把 LoRA 权重合并回基座 -> 一份标准的、可被 vLLM / AWQ 直接加载的模型目录
#
# 为什么必须有这一步：QLoRA 出来的 adapter 依赖 4-bit 基座，
# 直接拿去做 AWQ 量化会报 dtype 不匹配（VLM 微调部署最常见的卡点）。
set -euo pipefail

LF_DIR="${LF_DIR:-/root/LLaMA-Factory}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
ADAPTER="${ADAPTER:-saves/qwen25vl-3b-gui/lora/sft}"
OUT="${OUT:-outputs/qwen25vl-3b-gui-sft-merged}"
TEMPLATE="${TEMPLATE:-qwen2_vl}"

echo "==> 合并 LoRA"
echo "    base   : ${BASE_MODEL}"
echo "    adapter: ${ADAPTER}"
echo "    out    : ${OUT}"

cd "${LF_DIR}"
llamafactory-cli export \
    --model_name_or_path "${BASE_MODEL}" \
    --adapter_name_or_path "$(realpath "${OLDPWD}/${ADAPTER}")" \
    --template "${TEMPLATE}" \
    --trust_remote_code true \
    --export_dir "$(realpath "${OLDPWD}/${OUT}")" \
    --export_size 5 \
    --export_legacy_format false

echo
echo "==> 合并完成：${OUT}"
echo "    该目录可直接作为 vLLM 的 --model，也可作为 AWQ 量化的输入"
