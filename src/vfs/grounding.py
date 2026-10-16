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

# 训练与评测必须使用同一个 prompt，否则测的是分布漂移而不是模型能力。
# 坐标空间必须显式声明：Qwen2.5-VL 基座的原生 grounding 行为是输出绝对像素，
# 而 ShowUI/GUI-Actor 一系的训练数据用 [0,1000] 归一化。早期版本没写清楚，
# 零样本档位按像素输出、被 eval 的归一化缩放毁掉（acc 0.5%、offset p50 1638px，
# 而同样样本按像素解释 6/6 命中）——约定不匹配被算成了定位失败。
PROMPT_TEMPLATE = (
    "Locate the UI element on the screenshot that matches the instruction. "
    "Output ONLY the center point in JSON.\n"
    "Instruction: {instruction}\n"
    "Coordinates are normalized to [0, 1000]: (0, 0) is the top-left corner "
    "and (1000, 1000) is the bottom-right corner.\n"
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
    failures: int = 0      # 格式问题，救不回来（不计入准确率分母）
    refusals: int = 0      # 模型回答"找不到"（明确答错，计入保守口径分母）
    # 每条失败的详情，供事后审计：只给个数字等于把证据扔了，
    # 事后想查"到底哪类样本答不出来"就无从下手。
    failure_details: list[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.records)

    @property
    def failure_breakdown(self) -> dict[str, int]:
        """按错误类型统计：parse / refusal / HTTP / 异常类名。"""
        out: dict[str, int] = {}
        for d in self.failure_details:
            k = str(d.get("error", "unknown"))
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    @property
    def n_total(self) -> int:
        """参与评测的样本总数（= 成功定位 + 格式失败 + 明确答错）。"""
        return len(self.records) + self.failures + self.refusals

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
        overall = self._agg(self.records)
        # 保守口径：解析失败与"找不到"一律算未命中。
        # 只看 accuracy 会虚高——把答不出来和答错的样本悄悄排除在分母外，
        # 正好是量化后模型更容易触发的情况（格式更易走形），
        # 于是"量化无损"的假象会被这个口径放大。
        if self.n_total:
            overall["accuracy_strict"] = round(
                sum(1 for r in self.records if r.hit) / self.n_total, 4)
        out = {
            "tag": self.tag,
            "overall": overall,
            "by_platform": {},
            "by_instruction_type": {},
            "by_area_bucket": {},
            "parse_failures": self.failures,
            "refusals": self.refusals,
            "n_total": self.n_total,
            "failure_breakdown": self.failure_breakdown,
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

# Qwen2.5-VL 原生 grounding 输出用的是 "point_2d" 键（还常带一个 "label"），
# 只认 "point" 会把这一类全判成解析失败（实测占失败样本的 6 成）。
_JSON_POINT = re.compile(
    r'"point(?:_2d)?"\s*:\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]')
_LOOSE_PAIR = re.compile(r"([0-9]{1,4})\s*[,，\s]\s*([0-9]{1,4})")

# 模型明确表示"找不到这个元素"。这是**答错了**，不是格式问题
# ——混进解析失败里排除掉，等于把答错的样本从分母中藏起来。
_REFUSAL = re.compile(
    r"there (?:are|is) (?:none|no|not)|not found|can(?:not|'t) find"
    r"|no such|does ?n[o']t exist|unable to (?:find|locate)|no element",
    re.IGNORECASE)


def is_refusal(text: str) -> bool:
    """模型是否明确回答"没有这个元素"。"""
    return bool(text) and bool(_REFUSAL.search(text))


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


def point_to_pixels(pt: tuple[float, float], w: int, h: int,
                    space: str = "auto") -> tuple[float, float]:
    """把模型输出的坐标转成像素，再与像素 bbox 比较。

    space:
      auto  —— 任一坐标 >1000 时判为绝对像素（归一化坐标不可能越界），
               否则按 [0,1000] 解释。默认值：训练/评测 prompt 已声明
               [0,1000]，SFT 后的模型必守约；但零样本基座与量化后格式
               走形的输出仍可能吐像素，auto 防止约定不匹配被算成定位失败
               （这正是零样本基线 acc 0.5% 事故的修复）。
               已知局限：两轴都 ≤1000px 的小图（如 960x540 的 desktop
               截图）上像素与归一化数值上不可区分，此类样本按 [0,1000]
               解释，会系统性低估零样本 desktop 档。
      pixel —— 一律按绝对像素解释。用于已实测确认输出约定的模型
               （零样本 Qwen2.5-VL 基座即属此类）。
      norm  —— 一律按 [0,1000] 解释。
    """
    if space == "pixel":
        return float(pt[0]), float(pt[1])
    if space == "auto" and (pt[0] > 1000 or pt[1] > 1000):
        return float(pt[0]), float(pt[1])
    return pt[0] / 1000.0 * w, pt[1] / 1000.0 * h


# ---------------------------------------------------------------- 载入数据

def _b64_of(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _image_size(p: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(p) as im:
        return im.size


def _xywh_to_xyxy(bbox: list[float], iw: int, ih: int) -> list[float]:
    """把 bbox 统一成 [x1, y1, x2, y2] 像素。

    ScreenSpot 实测用的是 **[x, y, w, h] 像素**（不是 xyxy、也不是归一化）。
    三种表示都见过，判错一种偏移量就会离谱到一眼假，所以显式区分：
        四个数都 ≤1.5          -> 归一化，乘分辨率
        x+w、y+h 都在图像内     -> xywh 像素
        否则                    -> 按 xyxy 像素处理
    """
    x, y, w, h = (float(v) for v in bbox)
    if max(x, y, w, h) <= 1.5:
        x, y, w, h = x * iw, y * ih, w * iw, h * ih
        return [x, y, x + w, y + h]
    if x + w <= iw * 1.02 and y + h <= ih * 1.02:
        return [x, y, x + w, y + h]
    return [x, y, w, h]


def _screenspot_files(data_dir: Path) -> list[tuple[Path, str]]:
    """定位 ScreenSpot 的元数据文件，返回 [(文件, 默认平台)]。

    同一个 repo 里元数据有两套视图，结构不一样：
      - `screenspot_{mobile,web,desktop}.json`：扁平，一行一个元素，
        字段是 img_filename / instruction
      - `hf_test_full.json`：也是扁平的，但字段叫 img_url / task，且带 img_size
    两套都会认；**优先用按平台拆分的那套**（指令文案更完整），
    否则 1272 条会被算成 2544 条。
    """
    md = data_dir / "metadata"
    if not md.is_dir():
        md = data_dir

    per_platform = [md / f"screenspot_{p}.json"
                    for p in ("mobile", "web", "desktop")]
    per_platform = [p for p in per_platform if p.exists()]
    if per_platform:
        return [(p, p.stem.rsplit("_", 1)[-1]) for p in per_platform]
    full = md / "hf_test_full.json"
    if full.exists():
        return [(full, "")]
    return []


def load_screenspot(data_dir: Path, limit: int = 0) -> list[Sample]:
    """加载 ScreenSpot 评测集（1272 条 = mobile 502 + web 436 + desktop 334）。"""
    files = _screenspot_files(data_dir)
    buckets: list[list[Sample]] = []
    for meta, default_platform in files:
        samples: list[Sample] = []          # 每个平台文件一个桶，便于比例抽样
        try:
            rows = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        img_root = data_dir / "images"
        for row in rows:
            img_name = (row.get("img_filename") or row.get("img_url")
                        or row.get("image"))
            if not img_name:
                continue
            img_path = img_root / img_name
            if not img_path.exists():
                continue
            bbox = row.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            try:
                iw, ih = _image_size(img_path)
            except Exception:
                continue
            samples.append(Sample(
                img_b64=_b64_of(img_path),
                img_w=iw, img_h=ih,
                instruction=row.get("instruction") or row.get("task") or "",
                bbox=_xywh_to_xyxy(bbox, iw, ih),
                platform=row.get("split") or default_platform or "unknown",
                inst_type=row.get("data_type", "unknown"),
                source=meta.stem,
            ))
        buckets.append(samples)
    return _stratified_limit(buckets, limit)


def _stratified_limit(buckets: list[list[Sample]], limit: int) -> list[Sample]:
    """按平台比例分摊 --limit。

    ScreenSpot 的三个平台文件按 mobile/web/desktop 顺序读，直接截断会让
    `--limit 200` 全落在 mobile（实测如此），分层统计退化成单平台数字，
    看起来像跑通了、其实只看了三分之一。抽样必须按比例分摊到各平台。
    """
    total = sum(len(b) for b in buckets)
    if limit <= 0 or limit >= total:
        return [s for b in buckets for s in b]
    picked: list[list[Sample]] = []
    for b in buckets:
        k = round(limit * len(b) / total)
        picked.append(b[:k])
    short = limit - sum(len(p) for p in picked)
    for b, p in zip(buckets, picked):        # 余数补给还有富余的桶
        if short <= 0:
            break
        need = min(short, len(b) - len(p))
        if need > 0:
            p.extend(b[len(p):len(p) + need])
            short -= need
    return [s for p in picked for s in p]


def _to_pixels(bbox: list[float], w: int, h: int) -> list[float]:
    """归一化 bbox → 像素 bbox；若已是像素则原样返回。"""
    if max(abs(v) for v in bbox) <= 1.5:
        x1, y1, x2, y2 = bbox
        return [x1 * w, y1 * h, x2 * w, y2 * h]
    return list(map(float, bbox))


# ---------------------------------------------------------------- 调用模型

async def _one(session: aiohttp.ClientSession, url: str, model: str,
               s: Sample, timeout: float,
               space: str = "auto") -> tuple[Record, Optional[dict]]:
    """跑一条样本，返回 (记录, 失败详情或 None)。

    失败详情连同模型原始输出一起返回，由调用方落盘 —— 只记个数不记内容，
    事后就没法判断"是模型答不出来，还是我的解析太脆"。
    """

    def _fail(error: str, raw: str = "") -> tuple[Record, dict]:
        return (Record(False, 0.0, s.area, s.platform, s.inst_type, error),
                {"error": error, "raw": raw, "platform": s.platform,
                 "inst_type": s.inst_type, "area": round(s.area, 1),
                 "img_w": s.img_w, "img_h": s.img_h,
                 "instruction": s.instruction, "bbox": list(s.bbox),
                 "source": s.source})

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
                return _fail(f"HTTP {resp.status}")
            data = await resp.json()
            text = data["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        return _fail(f"{type(e).__name__}: {e}")

    pt = parse_point(text or "")
    if pt is None:
        return _fail("refusal" if is_refusal(text or "") else "parse",
                     text or "")

    px, py = point_to_pixels(pt, s.img_w, s.img_h, space)
    cx, cy = s.center
    return (Record(
        hit=_point_in_bbox(px, py, s.bbox),
        offset_px=math.hypot(px - cx, py - cy),
        area=s.area,
        platform=s.platform,
        inst_type=s.inst_type,
    ), None)


async def evaluate(base_url: str, model: str, samples: list[Sample],
                   concurrency: int = 8, timeout: float = 60.0,
                   tag: str = "run", coord_space: str = "auto") -> EvalResult:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    res = EvalResult(tag=tag)
    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        async def wrapped(s: Sample) -> Record:
            async with sem:
                return await _one(session, url, model, s, timeout, coord_space)

        tasks = [asyncio.create_task(wrapped(s)) for s in samples]
        for i, t in enumerate(asyncio.as_completed(tasks)):
            r, detail = await t
            if r.error == "refusal":
                res.refusals += 1
                res.failure_details.append(detail or {"error": "refusal"})
            elif r.error:
                res.failures += 1
                res.failure_details.append(detail or {"error": r.error})
            else:
                res.records.append(r)
            if (i + 1) % 100 == 0:
                print(f"  ... {i + 1}/{len(samples)}", flush=True)
    return res


# ---------------------------------------------------------------- 出表

def render_markdown(results: list[dict]) -> str:
    """把多个档位的评测结果渲染成一张 Markdown 表。"""
    lines = ["### 整体", "",
             "| 档位 | 样本 | 准确率 | 准确率(保守) | 失败/拒答 | "
             "偏移 p50 (px) | 偏移 p90 (px) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        o = r["overall"]
        # 保守口径 = 把解析失败与"找不到"都算未命中；两者之差就是被藏起来的样本
        lines.append(
            f"| {r['tag']} | {o['n']} | {_p(o['accuracy'])} | "
            f"{_p(o.get('accuracy_strict'))} | "
            f"{r.get('parse_failures', 0)}/{r.get('refusals', 0)} | "
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
    ap.add_argument("--coord-space", choices=("auto", "pixel", "norm"),
                    default="auto",
                    help="模型输出坐标的解释口径：auto=越界判像素/否则按 [0,1000]；"
                         "pixel=一律按绝对像素（零样本基座实测用）；"
                         "norm=一律按 [0,1000]")
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
                               args.concurrency, args.timeout, args.tag,
                               args.coord_space))
    summary = res.summary()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"grounding_{args.tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 失败明细单独落盘：结论表里的每个"失败 N 条"都要能回溯到具体样本
    if res.failure_details:
        fp = out_dir / f"failures_{args.tag}.jsonl"
        with fp.open("w", encoding="utf-8") as f:
            for d in res.failure_details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[评测] 失败明细 -> {fp}（{len(res.failure_details)} 条）")

    o = summary["overall"]
    print(f"[评测] {args.tag}: acc={_p(o['accuracy'])} "
          f"acc(保守)={_p(o.get('accuracy_strict'))} "
          f"offset p50={_f(o['offset_p50'])}px p90={_f(o['offset_p90'])}px "
          f"| 解析失败 {res.failures} 拒答 {res.refusals}"
          + (f" 明细 {res.failure_breakdown}" if res.failure_breakdown else ""))
    print()
    print(render_markdown([summary]))


if __name__ == "__main__":
    main()
