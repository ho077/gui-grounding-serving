#!/usr/bin/env python
"""把各档位的「精度 + 速度 + 显存」汇总成一张表，并自动写出选型结论。

为什么结论要自动生成
--------------------
手写结论有两个问题：一是跑完一批数据后懒得改，二是改的时候容易挑好看的写。
结论由脚本从实测 JSON 里推出来，数据变了结论就跟着变，没法自欺。

README 里的表格由本脚本回填到 <!--RESULTS_TABLE_START--> 标记之间。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
START = "<!--RESULTS_TABLE_START-->"
END = "<!--RESULTS_TABLE_END-->"

# 报告里重点看的两个分组：最小的一档和最大的一档。
# 量化损失是否「均匀」，看这两组的差值就知道了。
SMALL_KEY = "xs  (<500px²)"
LARGE_KEY = "l   (>8k)"


def load_retracted(d: Path) -> set:
    """读 `results/retracted.txt`：口径错误的 tag 不进汇总表，也不参与结论推导。

    为什么要有一个"作废清单"而不是删掉 JSON
    --------------------------------------
    测量口径出错的产物和跑砸了的实验不一样：它的数字看起来完全合理，
    删掉就再也想不起来为什么不能用了。留一份显式清单，让脚本自动跳过、让人随时复盘。
    """
    p = d / "retracted.txt"
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def _load_results(d: Path) -> tuple[dict, dict]:
    grounding: dict[str, dict] = {}
    bench: dict[str, dict] = {}
    skip = load_retracted(d)
    for p in sorted(d.glob("grounding_*.json")):
        tag = p.stem[len("grounding_"):]
        if tag in skip:
            continue
        grounding[tag] = json.loads(p.read_text(encoding="utf-8"))
    for p in sorted(d.glob("bench_*.json")):
        tag = p.stem[len("bench_"):]
        if tag in skip:
            continue
        bench[tag] = json.loads(p.read_text(encoding="utf-8"))
    return grounding, bench


def render_retracted_note(d: Path) -> str:
    skip = load_retracted(d)
    if not skip:
        return ""
    return (f"> 已按 `results/retracted.txt` 排除 {len(skip)} 个作废档位，"
            f"它们不计入本表与结论：{', '.join('`' + t + '`' for t in sorted(skip))}。"
            f"作废原因见 `docs/07-results.md` §11。")


def _best_bench(bench: dict, tag: str) -> dict:
    """同一档位可能压了多个并发，取吞吐最高的一份作为该档位的代表值。"""
    cands = {k: v for k, v in bench.items() if k == tag or k.startswith(f"{tag}_c")}
    if not cands:
        return {}
    best = max(
        cands.values(),
        key=lambda x: (x.get("summary") or {}).get(
            "output_token_throughput_tok_per_s", 0) or 0,
    )
    return best.get("summary") or {}


def _p(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _f(x: Optional[float], unit: str = "") -> str:
    return "—" if x is None else f"{x:.1f}{unit}"


def render_table(grounding: dict, bench: dict) -> str:
    lines = [
        "| 档位 | 准确率 | 偏移 p50 (px) | 小图标 (<500px²) | 大元素 (>8k px²) "
        "| 吞吐 (tok/s) | TTFT p50 (ms) | 显存峰值 (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not grounding:
        lines.append("| _待填充_ | — | — | — | — | — | — | — |")
        return "\n".join(lines)

    for tag, r in grounding.items():
        o = r.get("overall", {})
        small = r.get("by_area_bucket", {}).get(SMALL_KEY, {})
        large = r.get("by_area_bucket", {}).get(LARGE_KEY, {})
        b = _best_bench(bench, tag)
        lines.append(
            f"| {tag} | {_p(o.get('accuracy'))} | {_f(o.get('offset_p50'))} "
            f"| {_p(small.get('accuracy'))} | {_p(large.get('accuracy'))} "
            f"| {_f(b.get('output_token_throughput_tok_per_s'))} "
            f"| {_f((b.get('ttft_ms') or {}).get('p50'))} "
            f"| {_f(b.get('peak_vram_gib'))} |"
        )
    return "\n".join(lines)


def render_conclusion(grounding: dict, bench: dict) -> str:
    """从实测数据推出选型结论。这是本项目的交付重点，不是表格的附赠品。"""
    if len(grounding) < 2:
        return "_当前只有单个档位的结果，跑完至少两个档位后这里会自动生成对比结论。_"

    # 基准要挑"未微调、未量化、原分辨率"的那一档，按字母排会挑到 awq_*
    preferred = [t for t in grounding if "zero_shot" in t or "native" in t]
    base_tag = preferred[0] if preferred else sorted(grounding)[0]
    base = grounding[base_tag]
    base_o = base.get("overall", {})

    out = [f"以 **{base_tag}** 为基准（准确率 {_p(base_o.get('accuracy'))}，"
           f"偏移 p50 {_f(base_o.get('offset_p50'))}px）：", ""]

    for tag, r in grounding.items():
        if tag == base_tag:
            continue
        o = r.get("overall", {})
        d_acc = _delta(o.get("accuracy"), base_o.get("accuracy"))
        d_off = _delta(o.get("offset_p50"), base_o.get("offset_p50"))
        small = r.get("by_area_bucket", {}).get(SMALL_KEY, {}).get("accuracy")
        large = r.get("by_area_bucket", {}).get(LARGE_KEY, {}).get("accuracy")
        base_small = base.get("by_area_bucket", {}).get(SMALL_KEY, {}).get("accuracy")
        base_large = base.get("by_area_bucket", {}).get(LARGE_KEY, {}).get("accuracy")

        b = _best_bench(bench, tag)
        bb = _best_bench(bench, base_tag)
        thr = b.get("output_token_throughput_tok_per_s")
        thr_base = bb.get("output_token_throughput_tok_per_s")
        speed = (f"{thr / thr_base:.2f}×" if thr and thr_base else "—")

        out.append(
            f"- **{tag}**：准确率 {_signed(d_acc, pct=True)}，"
            f"偏移 p50 {_signed(d_off, unit='px')}，"
            f"吞吐为基准的 {speed}。"
        )
        if small is not None and base_small is not None:
            out.append(
                f"  - 损失**不均匀**：小图标 {_p(base_small)} → {_p(small)}"
                f"（{_signed(_delta(small, base_small), pct=True)}），"
                f"大元素 {_p(base_large)} → {_p(large)}"
                f"（{_signed(_delta(large, base_large), pct=True)}）。"
            )
        out.append("")
    return "\n".join(out)


def _delta(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None:
        return None
    return new - old


def _signed(d: Optional[float], pct: bool = False, unit: str = "") -> str:
    if d is None:
        return "—"
    if pct:
        return f"{d * 100:+.1f}pt"
    return f"{d:+.1f}{unit}"


def update_readme(body: str) -> bool:
    """把表格 + 结论回填到 README 的标记之间（分段由调用方拼好）。"""
    if not README.exists():
        return False
    text = README.read_text(encoding="utf-8")
    # 结论一并回填：手写的结论容易在重跑后忘了更新，或者只挑好看的写。
    block = f"{START}\n\n{body}\n\n{END}"
    if START in text and END in text:
        new_text = re.sub(
            re.escape(START) + r".*?" + re.escape(END),
            lambda _: block, text, flags=re.S,
        )
    else:
        new_text = text + f"\n{block}\n"
    README.write_text(new_text, encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--no-write", action="store_true", help="只打印，不回写 README")
    args = ap.parse_args()

    d = Path(args.results_dir)
    if not d.is_absolute():
        d = ROOT / d
    grounding, bench = _load_results(d)
    note = render_retracted_note(d)

    table = render_table(grounding, bench)
    print("=== 汇总表 ===")
    print(table)
    if note:
        print(note)
    print()
    print("=== 选型结论 ===")
    print(render_conclusion(grounding, bench))

    if not args.no_write:
        block = f"{table}\n\n{note}\n\n{render_conclusion(grounding, bench)}" if note \
            else f"{table}\n\n{render_conclusion(grounding, bench)}"
        if update_readme(block):
            print(f"\n[report] 已回填 {README}")
        else:
            print("\n[report] 未找到 README，跳过回填")


if __name__ == "__main__":
    main()
