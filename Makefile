.PHONY: help install hardware data train merge quantize scan eval report test clean

PYTHON ?= python
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
	@echo "    make test        冒烟测试：校验评测与压测的计时/判定逻辑"
	@echo "    make pipeline    data → train → merge → quantize → scan → report"
	@echo ""

install:
	$(PYTHON) -m pip install -r requirements.txt

hardware:
	$(PYTHON) -m vfs.hardware

data:
	$(PYTHON) scripts/01_prepare_data.py

train:
	bash scripts/02_train.sh

merge:
	bash scripts/03_merge_lora.sh

quantize:
	$(PYTHON) scripts/04_quantize.py

scan:
	bash scripts/05_scan.sh

eval:
	$(PYTHON) -m vfs.grounding --data-dir data/eval/ScreenSpot --tag eval

report:
	$(PYTHON) -m vfs.report --results-dir results

test:
	$(PYTHON) tests/test_grounding.py

pipeline: data train merge quantize scan report

clean:
	rm -f results/*.json results/REPORT.md
	rm -rf logs/*
