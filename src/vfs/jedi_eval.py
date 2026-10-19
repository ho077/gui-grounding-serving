#!/usr/bin/env python
"""用 Jedi-3B 的**官方协议**跑 ScreenSpot 评测。

为什么要单独一个文件
--------------------
Jedi-3B（xlangai，NeurIPS'25 spotlight）不是用 {"point": [x, y]} 回答 GUI 定位的，
它用的是 **ComputerUse 工具调用**协议 + `<tool_call>` 包裹的输出：

    <tool_call>
    {"name": "computer_use",
     "arguments": {"action": "mouse_move", "coordinate": [x, y]}}
    </tool_call>

并且要在 system 里用 tools XML 告诉它「屏幕分辨率是多少」，坐标是**该分辨率下的绝对像素**。
拿我们自己那套「归一化到 [0,1000] 的 {"point":[x,y]}」prompt 喂它，属于分布外：
模型会在两种口径之间摇摆（同一张图，一版输出像素、一版输出归一化），
拿这种输出算准确率，测的是 prompt 不匹配，不是模型能力。

所以这里复刻官方的三件事：
    1. 客户端用 smart_resize 缩放（把 await-the-processor 的降采样前置）
    2. system 里给 ComputerUse 函数签名 + 真实屏幕分辨率
    3. 解析 <tool_call> 里的 coordinate

顺带解决了另一个问题：直接把 >2.1M 像素的大图丢给 vLLM，它会把图在内部
再降采样一次 —— 我们在同一台机器上实测，这一步会让引擎静默挂死
（无报错、无日志，请求永远不被调度）。客户端先缩到合法尺寸后，
processor 的 resize 变成空操作，请求稳定返回。
详见 docs/06-error-log.md。
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
from pathlib import Path
from typing import Optional

import aiohttp
from PIL import Image

try:                                    # 复用主评测的结果结构，保证两张表同格式
    from vfs.grounding import (EvalResult, Record, Sample, load_screenspot,
                               parse_point, render_markdown)
except ImportError:                     # 直接 python src/vfs/jedi_eval.py 时
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vfs.grounding import (EvalResult, Record, Sample, load_screenspot,
                               parse_point, render_markdown)


# ------------------------------------------------------------ 官方 prompt 素材

_TOOL_DESC = """{"type": "function", "function": {"name": "computer_use", \
"description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\\n* The screen's resolution is __SCREEN_W__x__SCREEN_H__.\\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", \
"parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\\n* `type`: Type a string of text on the keyboard.\\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `left_click`: Click the left mouse button.\\n* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `right_click`: Click the right mouse button.\\n* `middle_click`: Click the middle mouse button.\\n* `double_click`: Double-click the left mouse button.\\n* `scroll`: Performs a scroll of the mouse scroll wheel.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}, "keys": {"description": "Required only by `action=key`.", "type": "array"}, "text": {"description": "Required only by `action=type`.", "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.", "type": "array"}, "pixels": {"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}, "time": {"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}"""

FN_CALL_TEMPLATE = (
    "You are a helpful assistant.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    "{tool_desc}\n"
    "</tools>\n\n"
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
    "</tool_call>"
)


def build_system_prompt(w: int, h: int) -> str:
    """把（缩放后的）屏幕分辨率写进工具描述——模型输出的坐标系由这一行决定。"""
    # JSON 里全是花括号，不能用 str.format 插值（会被当成字段），改占位符替换
    desc = (_TOOL_DESC.replace("__SCREEN_W__", str(w))
                      .replace("__SCREEN_H__", str(h)))
    return FN_CALL_TEMPLATE.format(tool_desc=desc)


# ------------------------------------------------------------ 图像预处理

# Jedi preprocessor_config：patch_size 14 × merge_size 2 = 每 token 覆盖 28×28 像素
FACTOR = 28
MIN_PIXELS = 12544
MAX_PIXELS = 2116800


def smart_resize(height: int, width: int, factor: int = FACTOR,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """复刻 transformers 的 smart_resize：把图缩到合法像素区间 + 边长对齐 factor。

    不用 PIL 直接 resize 的原因：Qwen2.5-VL 的 processor 内部还会做一次同规则的
    resize。如果我们送进去的尺寸不是它认可的（未对齐 28、或超了 max_pixels），
    它会在 embedding 时再算一遍 —— 既让送进模型的分辨率不可控，
    又在本机触发过引擎挂死。这里算出**同一个**目标尺寸再喂进去，
    processor 的 resize 就成了空操作。
    """
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def build_samples(data_dir: Path, limit: int = 0, max_pixels: int = MAX_PIXELS,
                  min_pixels: int = MIN_PIXELS) -> list[Sample]:
    """读 ScreenSpot 原图 -> smart_resize -> 同步缩放 GT bbox。"""
    srcs = load_screenspot(data_dir, limit, 0)
    out: list[Sample] = []
    for s in srcs:
        im = Image.open(io.BytesIO(base64.b64decode(s.img_b64))).convert("RGB")
        ow, oh = im.size
        nh, nw = smart_resize(oh, ow, FACTOR, min_pixels, max_pixels)
        if (nw, nh) != (ow, oh):
            im = im.resize((nw, nh))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        fx, fy = nw / ow, nh / oh
        out.append(Sample(
            img_b64=base64.b64encode(buf.getvalue()).decode(),
            img_w=nw, img_h=nh,
            instruction=s.instruction,
            bbox=[s.bbox[0] * fx, s.bbox[1] * fy, s.bbox[2] * fx, s.bbox[3] * fy],
            platform=s.platform, inst_type=s.inst_type, source=s.source,
        ))
    return out


# ------------------------------------------------------------ 解析与调用

_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_jedi_point(text: str) -> Optional[tuple[float, float]]:
    """优先按官方 <tool_call> 协议解析；退化格式走主评测的通用解析。"""
    m = _TOOL_CALL.search(text or "")
    if m:
        try:
            obj = json.loads(m.group(1).strip())
        except Exception:            # noqa: BLE001 - JSON 走形，交给兜底
            obj = None
        if isinstance(obj, dict):
            args = obj.get("arguments") or {}
            c = args.get("coordinate") if isinstance(args, dict) else None
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                try:
                    return float(c[0]), float(c[1])
                except (TypeError, ValueError):
                    return None
    return parse_point(text or "")


async def _one(session: aiohttp.ClientSession, url: str, model: str, s: Sample,
               timeout: float, max_tokens: int) -> tuple[Record, Optional[dict]]:
    def _fail(error: str, raw: str = "") -> tuple[Record, dict]:
        return (Record(False, 0.0, s.area, s.platform, s.inst_type, error),
                {"error": error, "raw": raw[:400], "platform": s.platform,
                 "inst_type": s.inst_type, "area": round(s.area, 1),
                 "img_w": s.img_w, "img_h": s.img_h,
                 "instruction": s.instruction, "bbox": list(s.bbox),
                 "source": s.source})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_system_prompt(s.img_w, s.img_h)},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{s.img_b64}"}},
                {"type": "text", "text": s.instruction},
            ]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    try:
        async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                return _fail(f"HTTP {resp.status}")
            data = await resp.json()
            text = data["choices"][0]["message"]["content"]
    except Exception as e:                       # noqa: BLE001
        return _fail(f"{type(e).__name__}: {e}")

    pt = parse_jedi_point(text or "")
    if pt is None:
        from vfs.grounding import is_refusal
        return _fail("refusal" if is_refusal(text or "") else "parse",
                     text or "")

    px, py = pt                                  # Jedi 输出的就是当前缩放空间的像素
    cx, cy = s.center
    return (Record(
        hit=s.bbox[0] <= px <= s.bbox[2] and s.bbox[1] <= py <= s.bbox[3],
        offset_px=math.hypot(px - cx, py - cy),
        area=s.area, platform=s.platform, inst_type=s.inst_type,
    ), None)


async def evaluate(base_url: str, model: str, samples: list[Sample],
                   concurrency: int, timeout: float, tag: str,
                   max_tokens: int) -> EvalResult:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    res = EvalResult(tag=tag)
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        async def wrapped(s: Sample) -> tuple[Record, Optional[dict]]:
            async with sem:
                return await _one(session, url, model, s, timeout, max_tokens)

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
            if (i + 1) % 200 == 0:
                print(f"  ... {i + 1}/{len(samples)}", flush=True)
    return res


# ------------------------------------------------------------ CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Jedi-3B 官方协议评测")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="jedi-3b")
    ap.add_argument("--data-dir", default="data/eval/ScreenSpot")
    ap.add_argument("--tag", default="jedi")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    ap.add_argument("--min-pixels", type=int, default=MIN_PIXELS)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    d = Path(args.data_dir)
    if not d.exists():
        raise SystemExit(f"评测集不存在: {d}")
    samples = build_samples(d, args.limit, args.max_pixels, args.min_pixels)
    if not samples:
        raise SystemExit("未加载到样本")
    print(f"[Jedi 评测] {len(samples)} 条，"
          f"尺寸样例 {samples[0].img_w}x{samples[0].img_h}", flush=True)

    res = asyncio.run(evaluate(args.base_url, args.model, samples,
                               args.concurrency, args.timeout, args.tag,
                               args.max_tokens))
    summary = res.summary()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"grounding_{args.tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if res.failure_details:
        fp = out_dir / f"failures_{args.tag}.jsonl"
        with fp.open("w", encoding="utf-8") as f:
            for det in res.failure_details:
                f.write(json.dumps(det, ensure_ascii=False) + "\n")
        print(f"[Jedi 评测] 失败明细 -> {fp}（{len(res.failure_details)} 条）")

    o = summary["overall"]
    print(f"[Jedi 评测] {args.tag}: acc={o['accuracy'] * 100:.1f}% "
          f"offset p50={o['offset_p50']}px | "
          f"解析失败 {res.failures} 拒答 {res.refusals}")
    print()
    print(render_markdown([summary]))


if __name__ == "__main__":
    main()
