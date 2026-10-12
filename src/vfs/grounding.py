#!/usr/bin/env python
"""GUI grounding 评测：不止报准确率，还要报「定位精度退化了多少」。

为什么不用准确率一个数字收工
----------------------------
准确率是二值的：预测点落在 GT 框内就是 1，否则 0。
量化带来的损伤是**渐进**的——权重被压低比特后，坐标往往先「飘」，
但还落在框内，此时准确率纹丝不动，看起来「INT4 无损」。
等你把量化压得更狠，它才会突然掉一大截。

只报准确率会错过这个中间过程，得出的结论是假的安全。
所以这里额外统计**预测点到 GT 框中心的像素偏移**：
准确率不变而偏移增大 = 已经在退化，只是还没越界。
这是本项目判断「能不能上 INT4」的主要依据，而不是那一个准确率数字。

坐标约定
--------
模型输出归一化到 [0, 1000] 的整数坐标（Qwen 系列惯例），
评测时按各图真实分辨率还原为像素。这样跨分辨率可比。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp

# 训练与评测必须使用同一个 prompt，否则测的是分布漂移而不是模型能力
PROMPT_TEMPLATE = (
    "Locate the UI element on the screenshot that matches the instruction. "
    "Output ONLY the center point in JSON.\n"
    "Instruction: {instruction}\n"
    'Format: {{"point": [x, y]}}'
)

# 面积分桶（像素²）。小图标与大按钮对量化的敏感度差一个量级，
# 这是分层分析里信息量最大的一维。
AREA_BUCKETS = [
    (0, 500, "xs  (<500px²)"),
    (500, 2000, "s   (0.5k-2k)"),
    (2000, 8000, "m   (2k-8k)"),
    (8000, 10**9, "l   (>8k)"),
]


@dataclass
class Sample:
    """一条 grounding 样本。bbox / point 一律存像素坐标。"""

    img_b64: str
    img_w: int
    img_h: int
    instruction: str
    bbox: list[float]          # [x1, y1, x2, y2] 像素
    platform: str = "unknown"  # mobile / web / desktop
    inst_type: str = "unknown"  # text / icon
    source: str = ""

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Record:
    hit: bool
    offset_px: float
    area: float
    platform: str
    inst_type: str
    error: str = ""


@dataclass
class EvalResult:
    tag: str
    records: list[Record] = field(default_factory=list)
    failures: int = 0

    @property
    def n(self) -> int:
        return len(self.records)

    def _agg(self, recs: list[Record]) -> dict:
        if not recs:
            return {"n": 0, "accuracy": None, "offset_p50": None, "offset_p90": None}
        hits = [r for r in recs if r.hit]
        offs = sorted(r.offset_px for r in recs)
        return {
            "n": len(recs),
            "accuracy": round(len(hits) / len(recs), 4),
            "offset_p50": round(_pct(offs, 50), 2),
            "offset_p90": round(_pct(offs, 90), 2),
        }

    def summary(self) -> dict:
        out = {
            "tag": self.tag,
            "overall": self._agg(self.records),
            "by_platform": {},
            "by_instruction_type": {},
            "by_area_bucket": {},
            "parse_failures": self.failures,
        }
        for key, fieldname in (("by_platform", "platform"),
                               ("by_instruction_type", "inst_type")):
            buckets: dict[str, list[Record]] = {}
            for r in self.records:
                buckets.setdefault(getattr(r, fieldname), []).append(r)
            out[key] = {k: self._agg(v) for k, v in sorted(buckets.items())}

        for lo, hi, label in AREA_BUCKETS:
            subset = [r for r in self.records if lo <= r.area < hi]
            out["by_area_bucket"][label] = self._agg(subset)
        return out


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round((q / 100) * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


# ---------------------------------------------------------------- 坐标解析

_JSON_POINT = re.compile(r'"point"\s*:\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]')
_LOOSE_PAIR = re.compile(r"([0-9]{1,4})\s*[,，\s]\s*([0-9]{1,4})")


def parse_point(text: str) -> Optional[tuple[float, float]]:
    """从模型输出解析归一化到 [0,1000] 的坐标。

    训练时只教一种格式，但评测时必须宽容：量化后模型更容易吐出
    带解释文字或格式走形的输出。**把格式问题算成定位失败会污染结论**，
    所以能救回来的都救回来，救不回的单独计入 parse_failures。
    """
    if not text:
        return None
    m = _JSON_POINT.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = _LOOSE_PAIR.search(text.replace("\n", " "))
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        if 0 <= x <= 1000 and 0 <= y <= 1000:
            return x, y
    return None


def _point_in_bbox(px: float, py: float, bbox: list[float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


# ---------------------------------------------------------------- 载入数据

def _b64_of(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def load_screenspot(data_dir: Path, limit: int = 0) -> list[Sample]:
    """加载 ScreenSpot 评测集。

    ScreenSpot 的元数据里 bbox 是**归一化** [0,1]，需要乘图像尺寸还原。
    这个细节搞错会让所有偏移量的量级完全不对——是本项目最容易踩的数据坑。
    """
    samples: list[Sample] = []
    for meta in sorted(data_dir.rglob("*.json")):
        if meta.name in ("hf_train.json",):
            continue
        try:
            rows = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        img_root = meta.parent.parent / "images"
        for row in rows:
            img_url = row.get("img_url") or row.get("image")
            if not img_url:
                continue
            img_path = img_root / img_url
            if not img_path.exists():
                continue
            iw, ih = row.get("img_size", [0, 0])[:2]
            if not iw or not ih:
                continue
            for el in row.get("element", []):
                bbox = el.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = _to_pixels(bbox, iw, ih)
                samples.append(Sample(
                    img_b64=_b64_of(img_path),
                    img_w=iw, img_h=ih,
                    instruction=el.get("instruction", ""),
                    bbox=[x1, y1, x2, y2],
                    platform=row.get("platform", meta.parent.name),
                    inst_type=el.get("data_type", "unknown"),
                    source=meta.parent.name,
                ))
            if limit and len(samples) >= limit:
                return samples
    return samples


def _to_pixels(bbox: list[float], w: int, h: int) -> list[float]:
    """归一化 bbox → 像素 bbox；若已是像素则原样返回。"""
    if max(abs(v) for v in bbox) <= 1.5:
        x1, y1, x2, y2 = bbox
        return [x1 * w, y1 * h, x2 * w, y2 * h]
    return list(map(float, bbox))


# ---------------------------------------------------------------- 调用模型

async def _one(session: aiohttp.ClientSession, url: str, model: str,
               s: Sample, timeout: float) -> Record:
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{s.img_b64}"}},
                {"type": "text",
                 "text": PROMPT_TEMPLATE.format(instruction=s.instruction)},
            ],
        }],
        "max_tokens": 64,
        "temperature": 0,
    }
    try:
        async with session.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                return Record(False, 0.0, s.area, s.platform, s.inst_type,
                              f"HTTP {resp.status}")
            data = await resp.json()
            text = data["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        return Record(False, 0.0, s.area, s.platform, s.inst_type,
                      f"{type(e).__name__}: {e}")

    pt = parse_point(text or "")
    if pt is None:
        return Record(False, 0.0, s.area, s.platform, s.inst_type, "parse")

    px = pt[0] / 1000.0 * s.img_w
    py = pt[1] / 1000.0 * s.img_h
    cx, cy = s.center
    return Record(
        hit=_point_in_bbox(px, py, s.bbox),
        offset_px=math.hypot(px - cx, py - cy),
        area=s.area,
        platform=s.platform,
        inst_type=s.inst_type,
    )


async def evaluate(base_url: str, model: str, samples: list[Sample],
                   concurrency: int = 8, timeout: float = 60.0,
                   tag: str = "run") -> EvalResult:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    res = EvalResult(tag=tag)
    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        async def wrapped(s: Sample) -> Record:
            async with sem:
                return await _one(session, url, model, s, timeout)

        tasks = [asyncio.create_task(wrapped(s)) for s in samples]
        for i, t in enumerate(asyncio.as_completed(tasks)):
            r = await t
            if r.error:
                res.failures += 1
            else:
                res.records.append(r)
            if (i + 1) % 100 == 0:
                print(f"  ... {i + 1}/{len(samples)}", flush=True)
    return res


# ---------------------------------------------------------------- 出表

def render_markdown(results: list[dict]) -> str:
    """把多个档位的评测结果渲染成一张 Markdown 表。"""
    lines = ["### 整体", "",
             "| 档位 | 样本 | 准确率 | 偏移 p50 (px) | 偏移 p90 (px) |",
             "|---|---:|---:|---:|---:|"]
    for r in results:
        o = r["overall"]
        lines.append(
            f"| {r['tag']} | {o['n']} | {_p(o['accuracy'])} | "
            f"{_f(o['offset_p50'])} | {_f(o['offset_p90'])} |")

    for key, title in (("by_platform", "按平台"),
                       ("by_instruction_type", "按指令类型"),
                       ("by_area_bucket", "按目标面积")):
        lines += ["", f"### {title}", "", "| 档位 | 分组 | 样本 | 准确率 | 偏移 p50 |",
                  "|---|---|---:|---:|---:|"]
        for r in results:
            for k, v in r.get(key, {}).items():
                if not v.get("n"):
                    continue
                lines.append(f"| {r['tag']} | {k} | {v['n']} | {_p(v['accuracy'])} | "
                             f"{_f(v['offset_p50'])} |")
    return "\n".join(lines)


def _p(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _f(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.1f}"


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="GUI grounding 评测")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen2.5-vl-3b-gui")
    ap.add_argument("--data-dir", default="data/eval/ScreenSpot")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    d = Path(args.data_dir)
    if not d.exists():
        raise SystemExit(f"评测集不存在: {d}\n先跑 scripts/01_prepare_data.py")

    samples = load_screenspot(d, args.limit)
    if not samples:
        raise SystemExit(f"未加载到样本，检查目录结构: {d}")
    print(f"[评测] {d} -> {len(samples)} 条")

    res = asyncio.run(evaluate(args.base_url, args.model, samples,
                               args.concurrency, args.timeout, args.tag))
    summary = res.summary()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"grounding_{args.tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    o = summary["overall"]
    print(f"[评测] {args.tag}: acc={_p(o['accuracy'])} "
          f"offset p50={_f(o['offset_p50'])}px p90={_f(o['offset_p90'])}px "
          f"| 解析失败 {res.failures}")
    print()
    print(render_markdown([summary]))


if __name__ == "__main__":
    main()
