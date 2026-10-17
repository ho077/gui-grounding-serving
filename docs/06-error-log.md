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
