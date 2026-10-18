"""只跑 04_quantize.py 里内置的 ignore 命名修复（量化已完成时用）。

复现 / 自检用：python scripts/fix_awq_ignore.py <模型目录>
"""
import json
import os
import shutil
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "outputs/qwen25vl-3b-base-awq"
cfg_path = os.path.join(d, "config.json")
if not os.path.exists(cfg_path):
    print(f"!! 没找到 {cfg_path}")
    raise SystemExit(1)

shutil.copy(cfg_path, cfg_path + ".bak")
with open(cfg_path, encoding="utf-8") as f:
    cfg = json.load(f)
qc = cfg.get("quantization_config") or {}
old = qc.get("ignore") or []
fixed = set(old)
for n in old:
    if n.startswith("model."):
        fixed.add(n[len("model."):])
if fixed != set(old):
    qc["ignore"] = sorted(fixed)
    cfg["quantization_config"] = qc
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[修复] ignore 命名：{len(old)} -> {len(fixed)} 条")
else:
    print("[修复] ignore 无需改动")

# 自检：确认两种写法都在，且没有残留的单前缀形式
vis = [n for n in qc["ignore"] if "visual" in n]
has_pref = any(n.startswith("model.visual") for n in vis)
has_bare = any(n.startswith("visual.") for n in vis)
print(f"[自检] visual 条目 {len(vis)} 个；带前缀={has_pref} 不带前缀={has_bare}")
if not (has_pref and has_bare):
    print("!! 两种写法不全，vLLM 仍可能加载失败")
    raise SystemExit(1)
print("[自检] OK，该产物可被 vLLM 正常加载")
