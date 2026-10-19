"""修 AWQ / compressed-tensors 产物的 ignore 命名，让 vLLM 能正确识别
"哪些层保 FP16"。

只修配置、不碰权重：python scripts/fix_awq_ignore.py <模型目录>

为什么要修
----------
llmcompressor 按 transformers 的模块路径写 ignore（`model.language_model.layers.0...`），
而 vLLM 侧该模型的语言主干挂在 `language_model.model.layers.*` 下，且
`should_ignore_layer` 只做**精确字符串匹配**（除非条目以 `re:` 开头是正则）。
命名对不上时，vLLM 会把本该保 FP16 的层建成 packed 参数，加载 FP16 权重时报
`KeyError: 'layers.0.self_attn.qkv_proj.weight'` —— 全量化产物（V0）碰不到这个问题，
因为它的 ignore 只含视觉塔，从来没有任何 FP16 语言层。

做法
----
把每个条目归一成"裸模块路径"，再补齐所有可能被匹配到的写法：
  language_model.model.<path>   <- vLLM 侧前缀（必需）
  model.<path>                  <- 权重键 / 部分 vLLM 代码路径
  <path>                        <- 裸名
"""
import json
import os
import shutil
import sys


def variants(name: str) -> set:
    """给一个模块路径，返回所有可能被精确匹配到的写法。"""
    p = name
    if p.startswith("model.language_model."):
        p = p[len("model.language_model."):]
    elif p.startswith("language_model."):
        p = p[len("language_model."):]
    elif p.startswith("model."):
        p = p[len("model."):]
    return {"language_model.model." + p, "model." + p, p}


def main() -> None:
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
        fixed |= variants(n)
    if fixed != set(old):
        qc["ignore"] = sorted(fixed)
        cfg["quantization_config"] = qc
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[修复] ignore 命名：{len(old)} -> {len(fixed)} 条")
    else:
        print("[修复] ignore 无需改动")

    ign = qc["ignore"]

    # 自检 1：视觉塔两种写法都要在（V0 时代留下的坑）
    vis = [n for n in ign if "visual" in n]
    has_pref = any(n.startswith("model.visual") for n in vis)
    has_bare = any(n.startswith("visual.") for n in vis)
    print(f"[自检] visual 条目 {len(vis)} 个；带前缀={has_pref} 不带前缀={has_bare}")
    if not (has_pref and has_bare):
        print("!! 两种写法不全，vLLM 仍可能加载失败")
        raise SystemExit(1)

    # 自检 2：混合位宽档的语言主干条目，必须带 vLLM 侧前缀
    lang = [n for n in ign if ".layers." in n and "self_attn" in n]
    if lang:
        vllm_style = [n for n in lang if n.startswith("language_model.model.")]
        flat = [n for n in lang if n.startswith("model.layers.")]
        print(f"[自检] 语言主干条目 {len(lang)} 个；vLLM 侧前缀 {len(vllm_style)} 个；"
              f"扁平命名 {len(flat)} 个")
        if not (vllm_style and flat):
            print("!! 缺少 vLLM 侧前缀（language_model.model.*）或扁平命名，"
                  "混合位宽档会 KeyError")
            raise SystemExit(1)
    print("[自检] OK，该产物可被 vLLM 正常加载")


if __name__ == "__main__":
    main()
