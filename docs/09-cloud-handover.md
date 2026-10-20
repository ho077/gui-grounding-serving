# 09 · 云端环境交接与停机/复机手册

> 写给"隔一段时间才回来"的自己。目标：**关掉 GPU 不丢任何证据**，复机 10 分钟内能接着跑。
> 最近一次更新：2026-10-21（停机前核查，HEAD `6b6eea0` → 本次提交）

---

## 1. 环境速查

| 项 | 值 |
|---|---|
| 登录 | `ssh -p 44466 root@i-2.gpushare.com` |
| 代码（云端，**不是 git 仓库**） | `/root/work/gui-grounding-serving` |
| **权威副本（本地，有 git）** | `D:\WorkSpace\codexWorkplace\portfolio\gui-grounding-serving` |
| 服务 venv（含 vLLM 0.8.5） | `/root/work/venvs/serve/bin/python` |
| 训练/量化 venv（llmcompressor 0.5.2） | `/root/work/venvs/train/bin/python` |
| 跑模块必须带 | `PYTHONPATH=/root/work/gui-grounding-serving/src` |
| GPU | 单卡 24 GiB（实测 24576 MiB） |
| 磁盘 | `/root` overlay 30G 用 23G（76%）；`/hy-tmp` 100G 用 86G（86%，几乎全是模型） |

**纪律**：云端只当算力，代码与结果一律回本地 git 提交。云端改完忘了 scp 回来 = 丢。
（`data/train/`、`results/_variant_status.tsv`、`logs/` 被 gitignore，不入库。）

### 权重与产物（`/hy-tmp/models`，共 83G）

| 目录 | 说明 |
|---|---|
| `/root/models/Qwen2.5-VL-3B-Instruct` | 基座 BF16 |
| `/hy-tmp/models/Jedi-3B-1080p` | Jedi-3B（ComputerUse 协议，82.0%） |
| `qwen25vl-3b-base-awq` | V0 基线：全 int4 g128 |
| `...-awq-g64` / `-g64-fine` / `-g64_calib128` | group_size=64 三档消融 |
| `...-awq-g32-fine` | **g32，同体积下精度最好的工作点** |
| `...-awq-keep-attn` / `-keep-mlp` / `-keep-mlp-gateup` / `-keep-mlp-down` / `-keep-deep` | 混合位宽五档 |
| `qwen25vl-3b-awq-g64-dequant` | g64 解量化回 BF16（用于隔离内核） |
| `qwen25vl-3b-gui-sft-{merged,v2,v3,v4}-merged` | 四轮 SFT（**全部负结果**） |

---

## 2. 停机前核查（2026-10-21 已执行，全部通过）

| 检查项 | 命令 | 结果 |
|---|---|---|
| GPU 是否还在烧 | `nvidia-smi --query-gpu=memory.used,utilization.gpu` | **2 MiB / 0%** —— 无残留 |
| 残留进程 | `pgrep -af "vllm\|vfs\|python"` | 只剩 `supervisord` + `jupyter-lab`（平台自带，不占 GPU） |
| 云端结果 vs 本地 | `ls results/` 双向 diff | 14 项缺失 → **已全部拉回**（见 §3） |
| 云端代码 vs 本地 | `find` + `comm` | 缺 `data/train/gui_grounding_train_fixed.json`（gitignore，已单独留存）、`requirements*.txt` 已拉回 |
| 本地 git | `git status` | 有未提交产物 → **已补提交** |

**结论：现在可以停机，不会丢东西。**

---

## 3. 本次停机前补回的缺失产物

- `results/grounding_base_full_1024.json` / `_1280.json`（+ 对应 failures）
  → 分辨率阶梯的**全量**证据，README §"1024 掉 4.5 点" 的唯一出处。
- `results/minvram_g32_fine.txt` / `minvram_g64.txt`
  → 最低显存预算表缺的两档（都是 6.0 GiB）。
- `results/grounding_awq_keep_deep_full1264.json` + failures（67.4% 那一档）
- `results/archive_remote/`：旧一轮 9 档压测（`bench_mixed/`）+ 13 份量化 profile（`mp/`）
  → 单独归档，**不要**和最新一轮 9 档数据混用（跨轮次压测数字不可比，见 error-log 噪声纪律）。
- `scripts/11_min_vram.sh`：补 g32_fine / g64 两档探测。
- `scripts/test_coord_convention.py`：坐标口径回归测试（从云端找回）。

---

## 4. 复机 SOP（10 分钟回到可跑状态）

```bash
# 0) 连上先确认卡还在、环境没被重置
ssh -p 44466 root@i-2.gpushare.com
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
ls /root/work/venvs/serve/bin/python /root/work/venvs/train/bin/python   # 缺了就是被重置过
ls -d /hy-tmp/models/qwen25vl-3b-base-awq*                                # 量化产物还在不在

# 1) 环境被重置过 → 从本地把代码推回去（云端没有 .git）
cd D:/WorkSpace/codexWorkplace/portfolio/gui-grounding-serving
tar czf /tmp/repo.tgz --exclude=.git --exclude=__pycache__ .
scp -P 44466 /tmp/repo.tgz root@i-2.gpushare.com:/tmp/
ssh -p 44466 root@i-2.gpushare.com 'mkdir -p /root/work/gui-grounding-serving && \
  tar xzf /tmp/repo.tgz -C /root/work/gui-grounding-serving'

# 2) 起服务（后台，前台会被 SIGTERM）
ssh -p 44466 root@i-2.gpushare.com 'cd /root/work/gui-grounding-serving && \
  setsid nohup bash scripts/10_mp_bench.sh > /root/bench.out 2>&1 &'
# 轮询：ssh ... 'tail -5 /root/bench.out'

# 3) 结果回本地 → 提交
scp -P 44466 root@i-2.gpushare.com:/root/work/gui-grounding-serving/results/xxx.json results/
git add -A results && git commit -m "..."
git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 push origin HEAD
```

### 复机必看的三个坑（踩过的）

1. **给远端跑脚本用 `setsid nohup ... &`**，前台执行会被 SIGTERM 打断。
2. **python 在 Windows 下写文本文件会把 `\n` 变 `\r\n`**，远端 bash 会报
   `set: pipefail: invalid option name`。写 .sh 用
   `open(p, 'w', encoding='utf-8', newline='\n')`（别用 `'wb'` 先截断，写失败就丢内容）。
3. **压测/评测数字跨轮次不可比**：TTFT ±40%、TPOT ±5%。要对比必须同轮连续跑完。

---

## 5. 停机方式差异（重要，决定数据还在不在）

| 方式 | GPU 计费 | 数据盘 | 适用 |
|---|---|---|---|
| **关机 / 停止（保留实例）** | 停 | **保留**（通常只收存储费） | ✅ 推荐：还会回来继续跑 |
| **释放 / 销毁实例** | 停 | **销毁** | ⚠️ 只有确认不再用才选 |

本机**疑似在 10-18 17:29 被重置过一次**（文件恢复了，git 元数据没恢复），
所以不要假设"停机=数据一定在"。**每次停机前按 §2 过一遍，把 results 拉回本地 git。**

---

## 6. 回来之后做什么（按性价比排序）

| # | 方向 | 为什么值 | 预估算力 |
|---|---|---|---|
| 1 | **升级 vLLM**（脱离 0.8.5 的 ~1.9M px 调度上限） | 最高杠杆：Jedi 现在只能跑到 82.0 而非论文 88.6 | 环境重装，半天 |
| 2 | **量化 Jedi-3B** 拿 4bit 精度保持率 | 文献里没有这个数，是新的可引用结果 | ~2h |
| 3 | **AWQ 摘掉 `enforce_eager`** | 或能改写"4bit 不提速"这条结论 | ~1h |
| 4 | 补 ScreenSpot-Pro / v2 | ScreenSpot 已近饱和，含金量下滑 | ~3h |
| 5 | 端到端 agent demo（OSWorld） | 故事完整性 | 1~2 天 |
| — | 自适应视觉 token 剪枝 | **不推荐**：五组预实验已全负 | — |

### 简历 / 讲述口径

见 `docs/08-resume-bullets.md`（4 条主 bullet + 证据包 + 面试问答 + 3 条主动暴露的局限）。
**不可写**：首次研究 VLM 量化敏感度、g32 显著优于 g128（p=0.434 未达显著）、量化能加速。
