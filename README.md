# gui-grounding-serving

**Qwen2.5-VL-3B 的 GUI 元素定位：微调 → 量化 → vLLM 部署 → 精度与速度联合评测。**

```
  截图 + 指令  ->  定位元素坐标  ->  LoRA 微调  ->  AWQ 量化  ->  vLLM 部署  ->  分层评测
                                                                                 ├─ 准确率
                                                                                 ├─ 坐标偏移 px
                                                                                 └─ 按面积/平台分层
```

---

## Motivation：只报准确率是不够的

GUI agent 能不能用，第一步取决于 **grounding**——把「点击设置图标」这句话落到屏幕上正确的位置。
定位不准，后面再多步规划也全崩。

端侧部署必须量化。但量化到底把定位精度压坏了多少？现有工作几乎只报**准确率**，
而准确率是**二值**的：预测点落在 GT 框内就是 1，否则 0。

问题在于量化损伤是**渐进**的：权重压低比特后，坐标往往先开始「飘」，
但还落在框内——此时准确率纹丝不动，看起来「INT4 无损」。
等指标突然掉下来时，已经越界了。

**所以这里额外统计预测点到 GT 框中心的像素偏移。**
准确率不变而偏移增大 = 已经在退化，只是还没越界。这是判断「能不能上 INT4」的主要依据。

---

## Method

| 环节 | 做法 |
|---|---|
| 微调 | LoRA（rank 32）SFT，输出归一化到 `[0,1000]` 的坐标 JSON |
| 合并 | LoRA 合并回基座，得到标准模型目录（QLoRA adapter 不能直接量化，会报 dtype 不匹配） |
| 量化 | AWQ W4A16（`llmcompressor`），校准集用**真实截图** |
| 部署 | vLLM，OpenAI 兼容接口 |
| 评测 | 准确率 + 偏移 p50/p90 + 按**元素面积 / 平台 / 指令类型**分层 |
| 速度 | 自建流式压测，测 TTFT / TPOT / 吞吐 / 显存峰值 |

分层是核心：小图标和大按钮对量化的敏感度差一个量级，
平均下来「只掉 3 个点」可能掩盖掉「小图标掉 11 个点」。

---

## Results

> 数字来自 `results/`，由 `make scan` + `make report` 自动生成，**非手写**。
> 当前尚未跑通全链路，表格为空是**预期状态**。

<!--RESULTS_TABLE_START-->

| 档位 | 输入上限 | 准确率 | 保守准确率 | 偏移 p50 (px) | TTFT p50 (ms) | 输出吞吐 (tok/s) | 显存占用 |
|---|---|---:|---:|---:|---:|---:|---:|
| BF16 基座（原分辨率） | ≤12.8M px | **80.0%** | 79.0% | 24.4 | 1115 | 44.8 | 21.6 GiB |
| BF16 基座（压分辨率） | 1.6M px | 20.3% | 19.9% | 357.9 | — | — | — |
| AWQ W4A16 | 1.6M px | 16.8% | 16.4% | 381.7 | 1129 | 42.5 | 21.6 / **8.7** GiB |
| SFT v3（像素目标，1600 条桌面） | ≤12.8M px | 62.7% | 62.7% | 61.7 | — | — | — |
| SFT v4（+1200 条网页） | ≤12.8M px | 59.4% | 59.4% | 74.7 | — | — | — |

**压测口径**：1408px 图、max_tokens 32、并发 8、64 请求、4 预热，两档均 `enforce_eager`
（AWQ 必须，BF16 对齐同一把尺）。AWQ 的"8.7 GiB"是 `gpu_memory_utilization=0.45`
下的实测占用 —— 同一预算下 BF16 直接起不来（`No available memory for the cache blocks`）。

**三个反直觉的实测结论**：

1. **压分辨率比压比特贵得多** —— 同一份 BF16 权重，输入从原分辨率压到 160 万像素
   就掉 59.7 个点；而在同一分辨率下，4bit AWQ 只再掉 3.5 个点。
   端侧落地的主要矛盾是分辨率，不是比特数。
2. **4bit 换不来速度** —— TPOT 反而涨 7.8%（W4A16 每步要把 4bit 权重解包回 16bit，
   且该档位必须 `enforce_eager`）。它真正买到的是显存：把"塞不进 10.8GB"变成
   "只占 8.7 GiB"。
3. **微调是负结果** —— 四轮实验（数据量、lr/rank、训练目标口径、补网页数据）
   最好的一版 62.7%，仍比零样本低 17.3 个点。

> 表中 AWQ 与 BF16 的对比必须**同分辨率**看：1.6M px 口径下 20.3% → 16.8%，
> 这才是量化损失。AWQ 档因 vLLM 0.8.5 的 dummy profiling 限制无法在原分辨率服务，
> 这是已知局限，见 [docs/07-results.md](docs/07-results.md) 第 3.3 节。

**已测部分的口径**：ScreenSpot 1272 条（mobile 502 / web 436 / desktop 334）；
坐标口径 `--coord-space pixel`（基座原生输出绝对像素，实测确认）；
"保守准确率"把解析失败与模型拒答一并算未命中。

<!--RESULTS_TABLE_END-->
<!--RESULTS_TABLE_END-->

**测试条件**：Qwen2.5-VL-3B · RTX 3090 24GB · ScreenSpot · max_tokens=32
· 关闭 prefix caching · 4 请求预热

---

## 快速开始

无 GPU 也能跑的部分（环境自检、评测逻辑校验）：

```bash
make install
make hardware     # 看看这台机器支持哪些精度档位
make test         # 校验评测与压测的判定逻辑
```

云端（RTX 3090 24GB）完整流程：

```bash
pip install -r requirements-train.txt
make pipeline     # data → train → merge → quantize → scan → report
```

单步执行见 `make help`。

---

## 目录结构

```
configs/
  sft/         微调配置（3090）
  serve/       部署配置（BF16 / FP8 / AWQ-INT4）
scripts/
  01_prepare_data.py   数据集下载与格式转换
  02_train.sh          LoRA 微调
  03_merge_lora.sh     LoRA 合并回基座
  04_quantize.py       AWQ 量化（可消融视觉塔）
  05_scan.sh           全档位扫描
src/vfs/
  grounding.py  定位精度评测（核心）
  hardware.py   GPU 能力探测与精度兼容性判定
  vram.py       显存峰值采样
  bench.py      自建流式压测
  serve.py      配置化拉起 vLLM
  report.py     汇总并自动写出选型结论
docs/
  04-gui-grounding-research.md   方向调研与参考仓库取舍
```

---

## 关键决策

### 1. 先查硬件能力，再决定跑哪些档位

精度能不能用取决于 GPU 微架构，不是取决于你想不想用：

| GPU | 架构 | FP8 | NVFP4 |
|---|---|---|---|
| RTX 3090 | Ampere SM 8.6 | ❌ | ❌ |
| RTX 4090 / L40 | Ada SM 8.9 | ✅ | ❌ |
| RTX 50xx | Blackwell SM 12.0 | ✅ | ✅ |

3090 上照抄 FP8 配置，vLLM 要等到权重加载阶段才报错。
`src/vfs/hardware.py` 在**启动前**就剔除不可用的档位。

### 2. 视觉塔量不量化，做消融而不是拍脑袋

默认跳过 `visual.*`，只压语言部分——定位精度主要由视觉编码器决定，压它风险最大。
但这不能只凭直觉。用 `--quantize-vision` 跑对照实验，
比较「显存省了多少」vs「偏移涨了多少」，结论才站得住。

### 3. 校准集必须用真实截图

校准数据决定每层的激活缩放因子。用合成图校准，缩放因子与真实分布不匹配，
量化误差会被放大。这里直接从训练集采样真实截图。

### 4. Benchmark 必须关闭 prefix caching

vLLM 官方 recipe 明文要求，多数教程不提。
不关的话重复截图第二次几乎零 prefill，TTFT 好看到失真，档位间不可比。

### 5. 压测客户端的计时要能自证

自建压测不能自证清白。`tests/` 里用「行为已知」的 mock 服务反推计时，
再用官方 `vllm bench serve` 在真实服务上交叉验证。

---

## 上游依赖与致谢

本项目是**编排层与评测层**，重活交给现成引擎，未改动其源码：

| 项目 | 角色 | 怎么用 |
|---|---|---|
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 微调引擎 | 调用 `llamafactory-cli train` / `export` |
| [vLLM](https://github.com/vllm-project/vllm) | 推理与服务 | 调用 `vllm serve`，并用其官方压测工具交叉验证 |
| [llmcompressor](https://github.com/vllm-project/llm-compressor) | 离线量化 | AWQ W4A16，输出 vLLM 原生兼容格式 |
| [ShowUI](https://github.com/showlab/ShowUI) | 训练范式参考 | 数据格式与 grounding 训练流程 |
| [GUI-Actor](https://github.com/microsoft/GUI-Actor) | 评测与锚点 | 评测脚本思路；其公开数字用作自检坐标系 |

**未照搬 GUI-Actor 的 action head**：那是新增模型结构，会破坏标准量化部署链路，
且会引入第二个变量，让「精度损失来自量化还是结构」说不清。本项目只保留**量化**一个变量。

对照锚点（GUI-Actor 论文，Qwen2.5-VL-3B）：零样本 ScreenSpot-v2 **80.9** / Pro **25.9**。
我们报的数如果离它太远，说明评测脚本或数据处理有问题——这是纠错机制，不是追赶目标。

---

## 已知限制

- Qwen2.5-VL-3B 版 `template` 为 `qwen2_vl`；换模型需同步改配置与坐标解析
- `freeze_vision_tower=true` 默认开启，可作为消融项关闭重跑
- 3090 无 FP8，`configs/serve/fp8.yaml` 会被自动跳过
- ScreenSpot-Pro 需 `--with-pro` 额外下载，分辨率更高、显存压力更大

---

## License

Apache-2.0
