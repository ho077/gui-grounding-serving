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


def _load_results(d: Path) -> tuple[dict, dict]:
    grounding: dict[str, dict] = {}
    bench: dict[str, dict] = {}
    for p in sorted(d.glob("grounding_*.json")):
        tag = p.stem[len("grounding_"):]
        grounding[tag] = json.loads(p.read_text(encoding="utf-8"))
    for p in sorted(d.glob("bench_*.json")):
        tag = p.stem[len("bench_"):]
        bench[tag] = json.loads(p.read_text(encoding="utf-8"))
    return grounding, bench


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

    base_tag = "bf16" if "bf16" in grounding else sorted(grounding)[0]
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


def update_readme(table: str, conclusion: str) -> bool:
    if not README.exists():
        return False
    text = README.read_text(encoding="utf-8")
    # 结论一并回填：手写的结论容易在重跑后忘了更新，或者只挑好看的写。
    block = f"{START}\n\n{table}\n\n{conclusion}\n\n{END}"
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

    table = render_table(grounding, bench)
    print("=== 汇总表 ===")
    print(table)
    print()
    print("=== 选型结论 ===")
    print(render_conclusion(grounding, bench))

    if not args.no_write:
        if update_readme(table, render_conclusion(grounding, bench)):
            print(f"\n[report] 已回填 {README}")
        else:
            print("\n[report] 未找到 README，跳过回填")


if __name__ == "__main__":
    main()
