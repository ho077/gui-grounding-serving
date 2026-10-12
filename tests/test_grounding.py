#!/usr/bin/env python
"""校验 grounding 评测的核心逻辑。

评测算错了，后面所有档位对比和选型结论全是错的，而且错得很隐蔽
（数字看起来都合理）。所以这里把判定链条逐段钉死：

1. 坐标解析能不能扛住格式走形的输出
2. 归一化 bbox 转像素有没有搞反（搞反会让偏移量级完全不对）
3. 命中判定与偏移计算
4. 端到端：mock 服务返回已知坐标，反推 hit / offset
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfs.grounding import (  # noqa: E402
    Sample, Record, EvalResult, evaluate, parse_point, _to_pixels,
)

# mock 服务固定返回归一化坐标 (500, 500)，即图像正中
MOCK_POINT = (500, 500)
PORT = 8231

results: list[bool] = []
out: list[str] = []


def log(s: str = "") -> None:
    out.append(s)
    print(s, flush=True)


def check(name: str, got, want, tol: float = 1e-6) -> None:
    ok = (abs(got - want) <= tol) if isinstance(got, (int, float)) \
        and isinstance(want, (int, float)) else (got == want)
    results.append(ok)
    log(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")


# ---------------------------------------------------------------- mock 服务

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        ln = int(self.headers.get("Content-Length", 0))
        self.rfile.read(ln)
        body = json.dumps({
            "choices": [{
                "message": {"content": json.dumps({"point": list(MOCK_POINT)})},
                "finish_reason": "stop",
            }]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def serve(port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ---------------------------------------------------------------- 用例

def test_parse_point() -> None:
    log("--- 坐标解析 ---")
    check("标准 JSON", parse_point('{"point": [123, 456]}'), (123.0, 456.0))
    check("带解释文字", parse_point('我先看看\n{"point": [10, 20]}'), (10.0, 20.0))
    check("退化 click()", parse_point("click(300,400)"), (300.0, 400.0))
    check("空格分隔", parse_point("point: 111 222"), (111.0, 222.0))
    check("无法解析", parse_point("我找不到这个元素"), None)
    check("空输出", parse_point(""), None)
    # 越界的整数对不应被误当成坐标
    check("越界值拒绝", parse_point("12345 67890"), None)
    log()


def test_to_pixels() -> None:
    log("--- 归一化 bbox → 像素 ---")
    got = _to_pixels([0.1, 0.2, 0.3, 0.4], 1000, 500)
    check("归一化输入", got, [100.0, 100.0, 300.0, 200.0])
    got2 = _to_pixels([10, 20, 30, 40], 1000, 500)
    check("已是像素则原样", got2, [10.0, 20.0, 30.0, 40.0])
    log()


def test_sample_geom() -> None:
    log("--- 几何量 ---")
    s = Sample(img_b64="", img_w=1000, img_h=500, instruction="x",
               bbox=[100, 100, 300, 200])
    check("中心", s.center, (200.0, 150.0))
    check("面积", s.area, 20000.0)
    log()


def test_aggregation() -> None:
    log("--- 分层聚合 ---")
    r = EvalResult(tag="t")
    # 两条命中、一条未命中
    r.records = [
        Record(hit=True, offset_px=2.0, area=100.0, platform="mobile", inst_type="text"),
        Record(hit=True, offset_px=4.0, area=90000.0, platform="web", inst_type="icon"),
        Record(hit=False, offset_px=50.0, area=300.0, platform="mobile", inst_type="icon"),
    ]
    s = r.summary()
    check("整体准确率", s["overall"]["accuracy"], 2 / 3, tol=1e-3)
    check("偏移 p50", s["overall"]["offset_p50"], 4.0)
    check("mobile 组样本数", s["by_platform"]["mobile"]["n"], 2)
    check("xs 桶样本数", s["by_area_bucket"]["xs  (<500px²)"]["n"], 2)
    check("l 桶样本数", s["by_area_bucket"]["l   (>8k)"]["n"], 1)
    log()


def test_end_to_end() -> None:
    log("--- 端到端（mock 服务返回图像中心）---")
    srv = serve(PORT)
    try:
        # 1000x500 的图，mock 返回 (500,500) 归一化 -> 像素 (500,250) 即正中心
        center_hit = Sample(img_b64=_dummy(), img_w=1000, img_h=500,
                            instruction="a", bbox=[400, 200, 600, 300],
                            platform="web", inst_type="text")
        corner_miss = Sample(img_b64=_dummy(), img_w=1000, img_h=500,
                             instruction="b", bbox=[0, 0, 100, 50],
                             platform="desktop", inst_type="icon")
        res = asyncio.run(evaluate(
            f"http://127.0.0.1:{PORT}", "mock",
            [center_hit, corner_miss], concurrency=2, timeout=20, tag="mock"))

        check("样本数", res.n, 2)
        check("解析失败数", res.failures, 0)
        hits = {r.hit for r in res.records}
        check("含命中与未命中各一", sorted(hits), [False, True])
        offs = sorted(r.offset_px for r in res.records)
        check("中心样本偏移 ≈ 0", offs[0], 0.0, tol=1.0)
        # 角落样本：预测(500,250) 到 bbox 中心(50,25) 的距离
        check("角落样本偏移 ≈ 503", offs[1], 503.0, tol=2.0)
    finally:
        srv.shutdown()
    log()


def _dummy() -> str:
    """1x1 透明 PNG 的 base64。mock 服务不读内容，只为让 payload 合法。"""
    return ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def main() -> None:
    log("=" * 60)
    log("grounding 评测自检")
    log("=" * 60)
    test_parse_point()
    test_to_pixels()
    test_sample_geom()
    test_aggregation()
    test_end_to_end()
    log("=" * 60)
    log("总结: " + ("全部通过" if all(results) else f"{results.count(False)} 项失败"))
    Path("_test_out.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
