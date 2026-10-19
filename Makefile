.PHONY: help install hardware data data-web train merge quantize scan serve-bf16 serve-awq \
        bench eval report ladder jedi-eval test clean

PYTHON ?= python
# 基座权重所在目录（与 configs/serve/*.yaml、configs/sft/*.yaml 里的 ${MODEL_DIR} 同一套约定）
MODEL_DIR ?= models
# AWQ 量化产物目录，供 make serve-awq 使用
AWQ_DIR ?= outputs/qwen25vl-3b-base-awq
# make eval / make bench 用的服务名（与 configs/serve/*.yaml 的 served_model_name 对应）
BENCH_MODEL ?= qwen25vl-3b-base
export PYTHONPATH := src:$(PYTHONPATH)

help:
	@echo "gui-grounding-serving —— GUI 元素定位的微调 / 量化 / 部署 / 联合评测"
	@echo ""
	@echo "    make install     安装评测与压测依赖（不含训练）"
	@echo "    make hardware    打印 GPU 与可用精度档位"
	@echo "    make data        下载并转换 GUI grounding 数据集"
	@echo "    make train       LoRA 微调"
	@echo "    make merge       合并 LoRA 回基座"
	@echo "    make quantize    AWQ 量化（默认跳过视觉塔）"
	@echo "    make scan        全档位扫描：起服务 → 测精度 → 测速度"
	@echo "    make report      汇总 results/ 并回填 README 表格"
	@echo "    make ladder      分辨率阶梯 + 同口径 4bit 对比（口径订正后的主实验）"
	@echo "    make jedi-eval   Jedi-3B 官方协议评测"
	@echo "    make test        冒烟测试：校验评测与压测的计时/判定逻辑"
	@echo "    make pipeline    data → train → merge → quantize → scan → report"
	@echo ""

install:
	$(PYTHON) -m pip install -r requirements.txt

hardware:
	$(PYTHON) -m vfs.hardware

# 数据集：复现报告里的 SFT v3（62.7%，四轮微调里最好的一版）。
# 这组参数是调试出来的，**不能省**——用脚本默认值会得到 303 条 + [0,1000] 目标的
# v1 数据，训出来只有 9.8%（见 docs/07-results.md 2.2 / 2.3）。
#   --parquet          用完整数据（7496 个元素），而非只覆盖 101 张图的元数据 JSON
#   --max-per-image    每张图取几个元素。默认 3 → 101 张只出 303 条，这是数据量上不去的真因
#   --target-space     pixel = 基座原生口径；强加 [0,1000] 会把定位精度扰糙（2.5 节）
# 产出 1600 条（101 张 × 16 元素），训练约 100 步 / 50 分钟。
data:
	$(PYTHON) scripts/01_prepare_data.py --skip-download --parquet \
		--parquet-limit 2000 --max-per-image 16 --target-space pixel

# 可选：再并入网页数据（复现 SFT v4 = 59.4%，比 v3 更差，作为多样性消融）。
# 会 Range 下载 ShowUI-web 图像包前 1.5GB 并解压，需 curl/tar，仅 Linux/macOS。
data-web:
	$(PYTHON) scripts/01_prepare_data.py --skip-download --parquet \
		--parquet-limit 1200 --max-per-image 6 --target-space pixel \
		--web-fetch-mb 1536 --web-dir data/ShowUI-web --web-limit 1200

train:
	bash scripts/02_train.sh

merge:
	bash scripts/03_merge_lora.sh

# 量化基座（不是 SFT 产物），校准集与参数对齐报告第 3 节。
# 脚本会自动修 ignore 命名，否则产物在 vLLM 里加载不了（docs/06-error-log.md 第 25 条）。
quantize:
	$(PYTHON) scripts/04_quantize.py \
		--model $(MODEL_DIR)/Qwen2.5-VL-3B-Instruct \
		--out outputs/qwen25vl-3b-base-awq \
		--num-calib 32 --calib-max-side 768

scan:
	bash scripts/05_scan.sh

# 起服务。AWQ 的两个额外参数是**必须的**，少了任一都起不来：
#   --mm-processor-kwargs 把 max_pixels 钉在 1.6M，让 dummy profiling 的
#     图像 token 数（2048）与文本截断长度对齐，否则报
#     "Mismatch in image token count between text and input_ids"
#   --max-num-batched-tokens 同上，配合 profiling
serve-bf16:
	MODEL_DIR=$(MODEL_DIR) $(PYTHON) -m vfs.serve --config configs/serve/bf16_base.yaml

serve-awq:
	AWQ_DIR=$(AWQ_DIR) MODEL_DIR=$(MODEL_DIR) $(PYTHON) -m vfs.serve \
		--config configs/serve/awq_base.yaml \
		--max-num-batched-tokens 16384 \
		--mm-processor-kwargs '{"max_pixels": 1605632}'

# 压测口径与报告一致：1408px 图、32 token、并发 8、64 请求、4 预热、采样显存峰值
bench:
	$(PYTHON) -m vfs.bench --base-url http://127.0.0.1:8000 \
		--model $(BENCH_MODEL) --num-requests 64 --concurrency 8 \
		--max-tokens 32 --image-size 1408 --num-warmup 4 \
		--tag bench --sample-vram

eval:
	$(PYTHON) -m vfs.grounding --base-url http://127.0.0.1:8000 \
		--model $(BENCH_MODEL) --data-dir data/eval/ScreenSpot \
		--coord-space pixel --tag eval

ladder:
	@# 分辨率阶梯 + 同口径 4bit 对比。所有涉及"压分辨率/压比特"的结论都出自这里，
	@# 因为它保证三件事同源：客户端缩放、GT 同比缩放、坐标按 pixel 解释。
	@# 缺一项就退化成 2026-10-19 那次口径污染（docs/07-results.md §11）。
	VENV=$(PYTHON) MODEL_DIR=$(MODEL_DIR) AWQ_DIR=$(AWQ_DIR) bash scripts/06_ladder.sh

jedi-eval:
	@# Jedi-3B 全量评测，走官方 ComputerUse 协议（src/vfs/jedi_eval.py）。
	VENV=$(PYTHON) bash scripts/07_jedi_eval.sh

report:
	$(PYTHON) -m vfs.report --results-dir results

test:
	$(PYTHON) tests/test_grounding.py

pipeline: data train merge quantize scan report

clean:
	rm -f results/*.json results/REPORT.md
	rm -rf logs/*
