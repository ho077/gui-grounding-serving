# 实测错误清单

> 从"云端 3090 从零部署"到"评测跑出可信数字"全过程踩过的坑。
> 分三类：**环境类**（下次换机器还会遇到）、**数据类**（换数据集还会遇到）、
> **评测逻辑类**（最危险 —— 不报错，只是悄悄给错数字）。
>
> 第三类是本文的重点：前两类一出错就崩，第三类会给出一个**看起来完全合理**的错数。

---

## 0. 先确认环境约束（决定了后面所有选型）

| 项 | 实测值 | 推论 |
|---|---|---|
| GPU | RTX 3090 24GB，SM 8.6 (Ampere) | **不支持 FP8**（需 8.9+），FP8 档位必须启动前判掉 |
| Driver | 550.163 | 最高支持 **CUDA 12.4** → torch / vLLM 只能选 cu124 构建 |
| 根分区 `/` | 30G（overlay） | 装不下全部；`/root/.cache` 实测占 8.8G，先删它 |
| 数据盘 | `/hy-tmp` 50G（约 72h 保留） | 大模型产物（合并/量化后）放这里 |
| 保护期 | `/root` 关机后 10 天 | 代码与数据放 `/root`，别放 `/tmp` |

---

## 1. 环境类

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 1 | `python3 -m venv` 报 `ensurepip returned non-zero exit status 1` | Debian/Ubuntu 把 venv 拆成独立包 `python3-venv`，镜像没装 | 装 `virtualenv` 后改用 `virtualenv --system-site-packages`（顺带继承系统 torch 2.5.1+cu124） |
| 2 | vLLM 安装 `ResolutionImpossible` | `PIP_CONSTRAINT` 里的 `torch==2.5.1` 与 vllm 0.8.5.post1 要求的 `torch==2.6.0` 冲突。**"vllm 0.8.5 用 torch 2.5.1"这个想当然的判断是错的** | 去掉 torch 约束。装完是 `torch 2.6.0+cu124`，仍是 cu124 构建，driver 550 照样兼容 —— 原先担心的"升级 torch 会突破 CUDA 12.4"没发生 |
| 3 | 换源后仍 `ResolutionImpossible` | 卡点是 `xformers==0.0.29.post3`：阿里云源只到 post2，清华源才有 | 换清华源 |
| 4 | `git clone` GitHub 130s 超时，但 `curl` 同一域名返回 200 | 该环境的 GitHub 直连不通（与本机同样的坑） | 本地 `tar czf` 打包（排除 `.git`、`__pycache__`）后 `scp` 上传 |
| 5 | `pkill -f 01_prepare_data` 把自己的 shell 也杀了，ssh 返回空输出 | 远程命令行字符串里含同样文本，`-f` 匹配到自身 | 括号技巧：`pkill -9 -f "01_prepare[_]data"` |
| 6 | vLLM 启动报 `Found conflicts between 'rope_type=default' (modern field) and 'type=mrope' (legacy field)` | vllm 0.8.5 把 `transformers` 拉到了 **5.20.0**，与 Qwen2.5-VL 的 `config.json` 不兼容 | serve venv 降级 `transformers==4.57.6`（与 train venv 对齐） |
| 7 | 根分区写满 | `/root/.cache` 占 8.8G（pip 缓存） | 删缓存 + 全部迁到 `/root` + pip 统一加 `--no-cache-dir` 防复发 |
| 12 | 训练启动报 `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4` | train venv 里 `pyarrow 26` + `datasets 4` 是 numpy 2 时代的版本，numpy 却停在 1.26；而 `llmcompressor 0.4.1` 又要求 `numpy<2` | **不要升 numpy**（会破 llmcompressor），改降 `pyarrow==18.1.0`，两边都满足 |
| 13 | ssh 里 `export PATH=...:$PATH` 把远端命令搞成语法错误 | 本机 shell 提前展开 `$PATH`，把 Windows PATH（含 `(x86)` 括号）注入远端 | 不碰 PATH：改用绝对路径（新增 `LF_CLI` 变量）；或把启动器写成远端脚本文件再执行 |
| 14 | 停掉 vLLM 后显存仍占 21GB，端口却已拒连 | vLLM 的多进程 worker（`python -c from vllm...`）没被主进程带走 | 用 `nvidia-smi --query-compute-apps` 找占用者，按 PID 单独 kill |

---

## 2. 数据类

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 8 | ShowUI-web 下到 43.7GB 把 50G 数据盘撑爆（HF 报磁盘空间警告） | 默认把三个训练集一起下了 | 把大数据集改 **opt-in**（`--with-web` / `--with-amex`），默认只下 `ShowUI-desktop-8K`(0.5G) + `ScreenSpot`(1.1G) |
| 9 | `01_prepare_data.py` 解析出训练集 303 条、**评测集 0 条** | 脚本按 ShowUI 的嵌套 `element` 结构写，而 ScreenSpot 是**扁平**结构，一行一个元素 | 重写 `load_screenspot()`：识别扁平格式，并**优先用按平台拆分的三个文件**（否则 1272 条会被算成 2544 条） |
| 10 | 数据准备脚本日志永远报 `评测集 ScreenSpot: 约 0 个标注元素` | `summarize_eval` 数的是 `row["element"]`，同样是嵌套假设 | 复用评测侧的 `_screenspot_files()`，保证"准备阶段说有多少条"和"评测实际跑多少条"是同一口径 |
| 11 | 诊断脚本报 `'Sample' object has no attribute 'image_path'` | 样本在解析时就把图转成 base64 了，字段是 `img_b64` / `img_w` / `img_h` | —（小事，但每次写新脚本都会再踩一次，记下来） |

**ScreenSpot 的数据格式**（极易踩，单独记）：

```
bbox     = [x, y, w, h]  **像素**（不是 xyxy、也不是归一化）
平台     = mobile 502 + web 436 + desktop 334 = 1272 条
指令字段 = per-platform 文件用 "instruction"；hf_test_full.json 用 "task"
```

判错 bbox 表示，偏移量会离谱到一眼假，所以 `_xywh_to_xyxy()` 显式区分三种表示。

---

## 3. 评测逻辑类（最危险：不报错，只给错数）

### 3.1 坐标约定断裂 —— 零样本 acc 从 80.9% 掉到 0.5%

**现象**：零样本基线 acc **0.5%**、offset p50 **1638px**，而 GUI-Actor 给的锚点是 80.9/25.9。

**断链三环**：

1. Qwen2.5-VL 基座**原生输出绝对像素**坐标（如 `{"point": [2413, 159]}`）
2. prompt 只写了 `Format: {"point": [x, y]}`，**没声明坐标空间** → 模型按预训练默认行为作答
3. 评测侧硬编码 `px = pt[0] / 1000 * img_w` 按归一化还原 → 2560 宽图上 `2413` 被算成 **6164px**，偏移 3700px

**反直觉的验证**：给 prompt 补上明确的 [0,1000] 声明后重跑，模型**照旧输出像素**
（同一条样本：旧 prompt `(2408,159)` → 新 prompt `(2413,159)`）
→ 这是预训练烧进去的行为，**改 prompt 让它听话是行不通的**。

**修复**：

| 位置 | 改动 |
|---|---|
| `PROMPT_TEMPLATE` | 显式声明坐标空间；训练与评测共用同一模板 |
| `point_to_pixels()` | 新增，支持 `auto` / `pixel` / `norm`；`auto` 下任一坐标 >1000 即判像素 |
| `--coord-space` | CLI 显式指定。零样本用 `pixel`，SFT 后用 `auto` |

**已知局限**：两轴都 ≤1000px 的小图（如 960×540 的 desktop 截图）上，
像素坐标与归一化坐标数值上不可区分（`(923,87)` 两种解释都合法），只能显式指定。

**结果**：修复后 **81.0%**，与锚点 80.9 吻合 —— 锚点自检机制如设计般生效。

### 3.2 分层抽样退化 —— `--limit 200` 只看了一个平台

三个平台文件按 mobile/web/desktop 顺序读，截断式 limit 会把 200 条全落在 mobile
（实测：按平台统计只剩 mobile 198）。分层统计退化成单平台数字，**看起来跑通了，其实只看了三分之一**。

修复：`_stratified_limit()` 按平台比例分摊。

### 3.3 失败样本被悄悄排除 —— 16.7% 的样本从分母里消失

**现象**：加载 1272 条，统计只用 1060 条，212 条（16.7%）"解析失败"被排除。

dump 原始输出后发现**两类完全不同的东西被混为一谈**：

| 类型 | 占比 | 原始输出 | 性质 |
|---|---|---|---|
| 用了 Qwen 原生 `point_2d` 键 | 约 6 成 | `{"point_2d": [1345, 116], "label": "search bar"}` | **我的解析器太脆**，能救 |
| 模型答 "There are none." | 约 4 成 | `There are none.` | **模型答错了**，应计为未命中 |

第二类是真正的问题：模型明确说"找不到元素"是答错了，
却和格式失败一起被排除在分母外 —— **等于把答错的样本藏起来系统性抬高准确率**
（实测影响约 3 个百分点）。

**修复**：

| 改动 | 说明 |
|---|---|
| `_JSON_POINT` 兼容 `point_2d` | 救回 Qwen 原生格式 |
| `is_refusal()` 识别拒答 | 拒答单独计数，与格式失败区分 |
| `accuracy_strict` | 保守口径：解析失败与拒答一律算未命中。**两个口径必须同时报** |
| `failure_details` → `failures_{tag}.jsonl` | 每条失败的原始输出落盘，可回溯 |

**为什么必须报两个口径**：量化后模型格式更容易走形，
"失败即排除"的口径会让"量化无损"的假象被放大 —— 正好是本项目要防的那种误判。

---

## 4. 训练配置类（版本相关，最容易照抄旧教程踩到）

用 LLaMA-Factory **0.9.6** 时，两份流传很广的旧配置直接跑不通：

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 15 | `ValueError: Some keys are not used by the HfArgumentParser: ['train_mm_proj']` | 0.9.x 已移除 `train_mm_proj`，语义挪到了 `freeze_multi_modal_projector`（默认 `true`） | 写 `freeze_multi_modal_projector: false` —— 即"投影层不冻结、跟着学" |
| 16 | `RuntimeError: shape mismatch: value tensor of shape [3, 2067] cannot be broadcast to indexing result of shape [3, 2048]` | `cutoff_len: 2048` **恰好等于图像 token 数**：`image_max_pixels 1605632 / 28² = 2048`。文本还没进来就被截满，position_ids 与截断后的序列长度对不上 | `cutoff_len: 4096`（图像 2048 + 指令约 60，留余量） |

第 16 条与服务侧的 `max_model_len` 是同一个坑的两个面：

> **凡是"长度上限"参数，都要先算一遍图像 token 数（像素面积 / 28²），再留文本余量。**

| 场景 | 参数 | 算出来的图像 token | 踩坑值 | 正确值 |
|---|---|---|---|---|
| vLLM 部署（全分辨率 2560×1440） | `max_model_len` | ≈ 4600 | 4096（截掉长边） | 8192 |
| LLaMA-Factory 训练（`image_max_pixels` 1605632） | `cutoff_len` | 2048（正好） | 2048（直接崩） | 4096 |

---

## 5. 量化与服务类

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 17 | `ImportError: cannot import name 'oneshot' from 'llmcompressor'` | 装的 0.4.1 没有顶层 `oneshot` | — |
| 18 | 装上 0.5.2 后又`ModuleNotFoundError: llmcompressor.modifiers.awq` | **0.4.1 根本没有 AWQ 模块**（只有 quantization / smoothquant / obcq） | 升到 **0.5.2**（只连带动 `compressed-tensors`，torch / transformers / numpy 都不动） |
| 19 | 想直接升最新版 → `Would install torch-2.14.0 ... cuda-toolkit-13.0.3.0` | 新版 llmcompressor 绑 CUDA 13 工具链，而**驱动 550 最高只到 CUDA 12.4** | 不要盲升；按 CUDA 12.4 能用的版本挑（0.5.2） |
| 20 | `ValueError: Unrecognized configuration class Qwen2_5_VLConfig for this kind of AutoModel: AutoModelForCausalLM` | `oneshot(model="<路径>")` 内部用 `AutoModelForCausalLM` 加载，VLM 的 config 类不匹配 | 自己用 `AutoModelForImageTextToText.from_pretrained(...)` 加载后**把模型实例**传给 `oneshot` |
| 21 | `ValueError: Mismatch in image token count ... Got ids=[8177] and text=[9000]` | 校准图 3360×2100 按默认 `max_pixels` 不缩放，单图就 **9000 个图像 token**，`max_seq_length` 给 2048/8192 都会被截断 | `max_seq_length=12288` |
| 22 | `serve.py` 报 `FileNotFoundError: 'vllm'` | 构造命令时用的是裸 `vllm`，要求 venv/bin 在 PATH 上；ssh 非交互执行时没有 | 改成 `sys.executable -m vllm.entrypoints.openai.api_server` |
| 23 | 改完又报 `api_server.py: error: unrecognized arguments: <模型路径>` | `vllm serve` 收位置参数，`api_server` 要 `--model` | 用 `--model <路径>` |
| 24 | 停掉 vLLM 后显存仍占 20GB，`ps` 里却找不到 | 残留的是 `python -c from multiprocessing.spawn import spawn_main`——命令行里既没有 `vllm` 也没有 `api_server`，所有常见 pkill 模式都打不中 | `ps -eo pid,rss,cmd --sort=-rss \| head` 按内存排序找它，按 PID kill。**别用 `pkill -f xxx` 且命令串里含 xxx，会杀掉自己**（已在第 5 条踩过） |
| 25 | 视觉塔明明没被量化，vLLM 却报 `Marlin: output_size_per_partition = 3420 not divisible by min_thread_n = 64` | **llmcompressor 把 ignore 写成模块路径 `model.visual.*`，而 safetensors 的键是 `visual.*`（无 `model.` 前缀）**，vLLM 按后者匹配 → 一条没命中 → 误判视觉塔也被量化 | 改 AWQ 产物 `config.json` 的 `quantization_config.ignore`，给每条 `model.visual.X` 补一份 `visual.X`（两种都留）。**无需重新量化** |
| 26 | 装 bitblas 后启动崩溃 `CUDA error: illegal memory access`，栈在 `triton_heuristics.benchmark_all_configs` | vLLM 优先选 BitBLAS 内核对量化层自动调优，本机（driver 550 + 该版 triton）下调优内核非法访存 | 卸载 bitblas 回落到 **Marlin**（前提是第 25 条已修）。3090 上别指望 BitBLAS |
| 27 | Marlin 已选中，仍在 inductor/triton 自动调优处 `illegal memory access` | torch.compile 在量化模型上触发新编译内核，本机 triton 自动调优不稳 | 该档位设 `enforce_eager: true`（放弃 CUDA Graph）。**做速度对比时 BF16 档也要同样设置**，否则不是同一把尺 |
| 28 | 压测 **39/64 请求失败**，但服务端零报错 | **两个 vLLM 服务同时监听 8000**（SO_REUSEPORT），请求被分流，一半打到旧服务上 → 间歇 `404 The model ... does not exist` | 杀干净再测：`ps aux \| grep api_server \| awk '{print $2}' \| xargs kill -9`。**别用 RSS 阈值猜**（我因此漏杀过一次，白跑一轮压测） |
| 29 | `ValueError: No available memory for the cache blocks` | BF16 在 `gpu_memory_utilization=0.45`（10.8GB）下，7.1GB 权重 + 激活已把池占满，排不出 KV cache | AWQ（3.2GB 权重）在同预算下能正常起服务，实测占 8.7 GiB —— 这正是量化的真实收益 |
| 30 | vLLM 杀掉后显存仍被占 10.9GB，`ps` 里却找不到进程 | 共享卡容器里的僵尸进程：PID 在 nvidia-smi 里存在但 `ps` 看不到，父进程已退出又没人 reap | 重启容器才能彻底回收。写测试脚本时先 `nvidia-smi --query-compute-apps=pid` 确认，别只看 `ps` |

### Jedi-3B 集成踩的三个坑（第 31–33 条）

接一个"别人训好的权重"听起来是零成本的事，实测踩了三个坑，且**全部不报错**：

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 31 | 用 `--max-side` 缩图后 Jedi 档只有 **11.0%**，其中 mobile 1.0% / web 1.6%（desktop 反而 38.3%） | **图缩了，GT 标注没缩**：`_xywh_to_xyxy(bbox, nw, nh)` 拿**缩放后**的尺寸去判定 xywh，`x+w` 超出新边界后被误判成 xyxy，整份标注作废（失败样本里能看到 `img_w=1120` 但 `bbox[0]=2291`） | 判定必须用**原始**分辨率，判完再乘缩放系数。加了自动校验：1272 条全部框在图像界内、缩放前后相对坐标最大偏差 0.0000 |
| 32 | 换了模型后输出在"像素"和"归一化"之间摇摆，同一批样本 acc 大幅跳动 | **协议不匹配**：Jedi 不是用 `{"point": [x, y]}` 回答的，它用 **ComputerUse 工具调用**——system 里给函数签名并声明屏幕分辨率，输出 `<tool_call>{"name":"computer_use","arguments":{"action":…,"coordinate":[x,y]}}</tool_call>` | 单独写 `src/vfs/jedi_eval.py` 复刻官方协议（工具签名模板 + `<tool_call>` 解析 + `smart_resize` 前置），不把它硬塞进主评测那套 `{"point":[x,y]}` prompt |
| 33 | 输入 ≥ 约 1.9M px 时请求**永远不被调度**：无报错、无日志、GPU/CPU 全空转，最后整个服务进程消失 | vLLM 0.8.5 在该 checkpoint 上做 >2.1M px 的二次 resize 时会挂死（缩小显存占用到 0.62 也一样，排除显存因素） | 客户端先做与 processor **同规则**的 `smart_resize`（边长对齐 28、像素 ≤1.8M），服务端 resize 变成空操作。副作用是输入分辨率受限于 1.8M px，评测口径要写清楚 |

第 32 条最容易被忽略：同一个 Qwen2.5-VL 架构的两个 checkpoint，**输出协议可以完全不同**。
判断"能不能直接插进来"的标准不是 `architectures` 字段，而是它的推理代码里**怎么提问、怎么解析答案**。

### 测量口径污染：推翻自己两个旗舰结论的那次排查（第 34 条）

前三条都是"跑不起来/结果离谱"，第 34 条最难：**结果看起来很合理，而且内部自洽，只是错在链路同源性上。**

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 34 | BF16 / AWQ / SFT v2 三个**不同的**模型档位压到 1.6M px 后，都出现同一个形态：**desktop 44~59%、mobile 9.7~10.6%、web 2.3~3.3%**，p50 偏移 357~406 px。据此得出了"压分辨率 −59.7 点、压比特只要 −3.5 点" | 三件事不同源，**测到的是缩放链路的误差而不是模型能力**：① 图在**服务端**被降采样，GT 框仍是原图坐标 → 系统性偏移；② 该批评测用的是 `--coord-space auto`，把绝对像素误判成归一化后又乘了一次分辨率；③ 恰好 desktop 只有 35% 的图 >1.6M px（mobile/web 是 **100%**），于是 desktop 幸存、mobile/web 崩 —— **"谁崩谁不崩"的分布本身就是指纹** | 三者必须同源：**客户端**缩放 + GT 同比缩放 + 显式 `--coord-space pixel`（固化在 `scripts/06_ladder.sh`）。**判据**：只要看到"desktop 明显好于 mobile/web"，先怀疑坐标链路，别急着给模型下结论 |
| 35 | 同源口径下的 400 条子集得出"**长边 1024 与原分辨率持平**（81.3% vs 81.4%）"，全量 1272 条复核却是 **1024 掉 4.5 点**（75.5% vs 80.0%），真正的无损档是 **1280**（80.6%） | **31% 的子集不足以判定 5 个点级别的差异**。当时把"±2 点"当作噪声上界，但那是由两条**全量**结果（side1512 / side1600）估出来的；子集额外引入**抽样偏差**，1σ 远大于 2 点。更隐蔽的是：分层结果确实 consistent（m 桶 63.5%→84.5%），所以**方向对了、幅度错了** —— 只看分层会以为整体验证通过了 | 凡是要写进结论的分辨率/量化差异，**必须在全量上复核一遍**。子集只用于快速筛掉明显不可能的选项（如 512 的 −35 点），用于"无损/持平"这种断言则必然翻车。订正记录见 `docs/07-results.md` §11.7 |
| 36 | AWQ 服务只能把 `max_pixels` 钉在 1,605,632（vLLM 0.8.5 的 dummy profiling 限制），于是客户端 `--max-side 1600` 送进去的近正方形图（峰值 2.56M px）会**被服务端再缩一次**，而 GT 是按客户端缩放后的尺度存的 —— 第 34 条那个坑的第三次变体 | 服务端降采样对客户端**不可见**。客户端以为自己控制了一切，实际超过 `max_pixels` 的那部分图会被悄悄再 resize 一次，GT 不跟着走 | 客户端的长边必须让**峰值像素**落在服务端限制以内：1280² = 1,638,400 越线，**1264² = 1,597,696 安全**。<br>**判据**：同一份 AWQ，换个 `--max-side` 就变脸（1024 −11.2 / 1512 −12.1 / 1600 −16.4，desktop 尤其离谱 −15.6 / −23.8 / −27.0）⇒ 一定有人在偷偷缩放 |
| 37 | sweep 脚本里把「量化」写在了「杀掉上一档服务」**之前**，于是 llmcompressor 把整份 BF16 载上 GPU 时，上一档 vLLM 还占着 21.28 GiB，直接 `torch.OutOfMemoryError`；而清理函数自己打印的 `[清理] 显存 2 MiB` 让人误以为卡是空的 | **读一次 nvidia-smi 不代表显存释放完成**：`kill -9` 之后驱动回收有延迟，且 `spawn_main` worker 可能还没退出。真正的因果关系是"谁在这一刻占着卡"，不是"清理时打印了多少" | `run_case` 里先 `wait_gpu_free`（杀进程 → 轮询直到 ≤2 GiB 且保持若干秒）**再**量化；同时导出 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。给 sweep 加**断点续跑**（结果 JSON 存在就跳过该档），否则中途失败要重付 25 分钟/档 |
| 38 | `AWQModifier(scheme={"W4A16_channel": {...}})` 在 llmcompressor 0.5.2 上被 pydantic 直接拒掉：`scheme` must either be a preset scheme name or a dictionary of preset scheme names。**已有的 AWQ 产物是用更早的版本跑出来的**，脚本没动但依赖升过级 | **量化超参会随 llmcompressor 版本变口径**。更危险的是产物配置里写的是 `group_size=128 / strategy=group`，而脚本注释说自己用的是 per-channel —— 两者早就对不上，只是没人发现 | 改用 preset 名 `"W4A16"`，并在量化后**读产物 config.json 与 V0 基线逐字段比对**（bits/symmetric/group_size/strategy/activations），不一致就打警告。混合位宽实验的前提是"只改了哪些层保精度"，量化方案变了的话对比就失去意义 |
| 40 | 首轮混合位宽压测得出"BF16 TPOT 46.9 ms vs AWQ 99.3 ms"，看着坐实了"4bit 更慢"；换成同一把尺后变成 **85.0 vs 100.5 ms** | **两档的服务配置本来就不同**：`bf16_base.yaml` 是 `max_model_len 8192` 且**没有** `enforce_eager`，而 `awq_base.yaml` 是 `16384 + enforce_eager`（量化模型在这台机器上不开 eager 会 `illegal memory access`）。BF16 因此白拿了 CUDA Graph，测到的是"有没有图"而不是"量化快不快"；`max_model_len` 不同还让峰值显存不可比（19.8 vs 21.6 GiB） | 新增 `configs/serve/bf16_bench.yaml`（BF16 也 `enforce_eager` + 同 `max_model_len`），压测脚本统一走它。**判据**：只要两个档位的 serve 配置不是同一份 yaml 派生的，速度/显存对比就无效 —— 先 diff 配置再跑压测。另外峰值显存在 `gpu_memory_utilization=0.90` 下会被设置项本身顶住（各档都 ~21.6 GiB），要用"最低能起来的 util"才有区分度 |
| 39 | 混合位宽档（keep-attn）量化成功、权重也对（attention 是 FP16 的 `q_proj.weight`，MLP 是 `weight_packed`），但 vLLM 一加载就 `KeyError: 'layers.0.self_attn.qkv_proj.weight'` | llmcompressor 按 **transformers** 的模块路径写 `ignore`（`model.language_model.layers.0.self_attn.q_proj`），而 vLLM 侧同一模块叫 **`language_model.model.layers.0.self_attn.qkv_proj`**（融合 q/k/v）。vLLM 的 `should_ignore_layer` 只做**精确字符串匹配**（除非条目以 `re:` 开头），名字对不上 ⇒ 它认为该层要量化，就建了 packed 参数，加载 FP16 权重时找不到 `.weight` 键。全量化产物碰不到这个坑，因为它的 ignore 里只有视觉塔、从来没有 FP16 语言层 | `scripts/fix_awq_ignore.py` 把每个条目归一成裸路径后补齐三种写法：`language_model.model.*`（vLLM 侧）、`model.*`（权重键）、裸名。修完的参数名从 1365 掉到 1005，attention 恢复成普通 `qkv_proj.weight`。**判据**：产物权重明明是 FP16 而 vLLM 建了 packed 参数 ⇒ 一定是 `ignore` 命名不匹配，先 `python scripts/fix_awq_ignore.py <目录>` |

复算（同源口径，长边 1600 ≈1.78M px，全量 1272 条）：BF16 **80.5%** vs AWQ **64.0%** ——
结论从"压分辨率 −59.7 / 压比特 −3.5"翻转为
"**压到长边 1024 几乎无损，4bit 才是那 12~16 点**"。完整订正见 `docs/07-results.md` §11。

这条之所以值得单列：**它错得不吵不闹。**三个档位互相印证、绝对值也可信，唯独缺少一个外部锚点。
当时的验收方式是"内部自洽"，而现在多了一条硬规矩 ——
压分辨率的实验必须先看**平台分布是否被"谁被缩放了"解释得掉**。

---

## 6. 一条通用教训

> **数字差一个数量级时，先怀疑自己的测量，再怀疑模型。**
>
> 0.5% vs 80.9% 这种量级的偏差，几乎不可能是"模型真的不行"，
> 而是链路里某一环的约定对不上。本次三处逻辑错误（3.1 / 3.2 / 3.3）
> 全是同一个模式：**数字看起来合理，所以没人怀疑它。**
>
> 能抓到它们，靠的是两个外部锚点：
> 1. 论文公开的 Qwen2.5-VL-3B 数字（80.9）—— 偏离太远就说明脚本有问题
> 2. dump 原始输出逐条比对 —— 6 条样本就足以证伪"模型能力 0.5%"
