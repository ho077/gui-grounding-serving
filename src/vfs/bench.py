"""自建 VLM 流式压测客户端。

为什么不用 ``vllm bench serve`` 一把梭
--------------------------------------
官方工具很好，但用在简历项目里有三个硬伤：

1. **版本漂移** —— vLLM 的 bench 参数名换过好几轮
   （``benchmark_serving`` -> ``vllm bench serve``），
   ``--output-json`` / ``--result-filename`` 也在变。依赖它意味着换台机器就跑不通。
2. **多模态负载不可控** —— 我们需要固定「每张图多少 token、每请求几张图」，
   官方 random 数据集是纯文本或随机尺寸图片，TTFT 的可比性会崩。
3. **讲不清原理** —— 面试官问「TTFT 到底从哪一刻开始算」，
   答不上来的话，这一栏反而扣分。

所以这里自己实现，并把结果**与官方 bench 交叉验证**
（见 ``scripts/06_bench_official.sh``）——两者数值接近，才能证明链路是对的。

指标定义
--------
- TTFT：请求发出 -> 收到**第一个带内容的 chunk**
- TPOT：(最后 chunk - 第一 chunk) / (输出 token 数 - 1)
- ITL ：相邻两个非空 chunk 的时间间隔
- Output throughput：全部请求的输出 token 总数 / 压测总时长
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import aiohttp
except ImportError:  # 延迟到真正要跑时才报错，方便 --help 和 dry-run
    aiohttp = None  # type: ignore


# --------------------------------------------------------------------------
# 输入构造
# --------------------------------------------------------------------------

# 固定问句池：保证不同精度档位面对完全相同的负载。
# 宗旨是「确定性」 —— 谁都能复现出同一张表。
QUESTION_POOL = [
    "Describe the image in detail.",
    "What is shown in this picture?",
    "List all visible objects and their approximate positions.",
    "Summarize the visual content in one sentence.",
    "What colors dominate this image?",
]


def make_synthetic_image(seed: int, width: int = 896, height: int = 896) -> bytes:
    """生成确定性合成图片（PNG bytes）。

    用合成图而非真实数据集的原因：
    - 不需要联网下载数据集，云端新机器也能立刻跑
    - 尺寸完全可控，图片 token 数可精确推导
    - seed 固定 -> 每次实验的负载逐字节一致
    """
    from PIL import Image, ImageDraw  # 局部导入：只在真正压测时才需要 Pillow

    g = (seed * 2654435761) % (2 ** 32)
    base = [(g >> 16) % 256, (g >> 8) % 256, g % 256]
    img = Image.new("RGB", (width, height), tuple(base))
    draw = ImageDraw.Draw(img)
    for i in range(12):
        x0 = (g * (i + 3)) % width
        y0 = (g * (i + 7)) % height
        x1 = min(width, x0 + 80 + (i * 37) % 220)
        y1 = min(height, y0 + 60 + (i * 53) % 260)
        c = ((base[0] + i * 41) % 256, (base[1] + i * 17) % 256, (base[2] + i * 29) % 256)
        draw.rectangle([x0, y0, x1, y1], outline=c, width=6)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_payload(model: str, image_b64: str, question: str, max_tokens: int) -> dict:
    """OpenAI 兼容的多模态 chat payload。"""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": question},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,   # 压测要稳定性，不要随机性
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},  # 让服务端回报 token 数
    }


# --------------------------------------------------------------------------
# 结果容器
# --------------------------------------------------------------------------

@dataclass
class RequestResult:
    ttft_ms: Optional[float] = None
    latency_ms: Optional[float] = None
    itl_ms: list[float] = field(default_factory=list)
    output_tokens: int = 0
    error: str = ""

    def finalize_itl(self, chunk_times: list[float], t0: float) -> None:
        """由「每个有内容的 chunk 的到达时刻」反推首包延迟与间隔。

        分开记录绝对时刻、最后统一求导，比在循环里边收边算要干净得多，
        也避免了首 chunk 被错误地当成一次 ITL。
        """
        if not chunk_times:
            self.ttft_ms = None
            return
        self.ttft_ms = (chunk_times[0] - t0) * 1000
        self.itl_ms = [(b - a) * 1000 for a, b in zip(chunk_times, chunk_times[1:])]

    @property
    def ok(self) -> bool:
        return self.error == "" and self.ttft_ms is not None

    @property
    def tpot_ms(self) -> Optional[float]:
        """Time Per Output Token（不含首 token）。"""
        if not self.ok or self.output_tokens <= 1:
            return None
        return (self.latency_ms - self.ttft_ms) / (self.output_tokens - 1)


def pct(values: list[float], p: float) -> Optional[float]:
    """线性插值分位数。"""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _round(x: Optional[float], nd: int = 2) -> Optional[float]:
    return None if x is None else round(x, nd)


# --------------------------------------------------------------------------
# 压测主逻辑
# --------------------------------------------------------------------------

async def _one_request(
    session: "aiohttp.ClientSession",
    url: str,
    payload: dict,
) -> RequestResult:
    """发一个流式请求，逐 chunk 记录时间。"""
    r = RequestResult()
    chunk_times: list[float] = []
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                r.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return r
            async for raw_line in resp.content:
                line = raw_line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # service 最后会单独发一个只含 usage 的 chunk
                if chunk.get("choices") is None and "usage" in chunk:
                    u = chunk["usage"] or {}
                    r.output_tokens = u.get("completion_tokens") or r.output_tokens
                    continue

                for ch in chunk.get("choices") or []:
                    content = (ch.get("delta") or {}).get("content")
                    if not content:
                        continue
                    chunk_times.append(time.perf_counter())
                    r.output_tokens += 1  # 无 usage 时兜底：按 chunk 数计

    except asyncio.TimeoutError:
        r.error = "timeout"
        return r
    except Exception as e:  # noqa: BLE001 - 压测里单个请求失败不应炸掉整轮
        r.error = f"{type(e).__name__}: {e}"
        return r

    r.latency_ms = (time.perf_counter() - t0) * 1000
    r.finalize_itl(chunk_times, t0)
    if r.ttft_ms is None:
        r.error = "no content chunk received"
    return r


async def run_benchmark(
    base_url: str,
    model: str,
    num_requests: int = 64,
    concurrency: int = 4,
    request_rate: Optional[float] = None,
    max_tokens: int = 128,
    image_size: int = 896,
    images_per_request: int = 1,
    num_warmup: int = 4,
    timeout: int = 300,
    seed: int = 42,
) -> dict:
    """跑一轮压测。

    两种负载模型二选一：
    - ``concurrency``：维持固定并发数（默认，适合看饱和吞吐）
    - ``request_rate``：Poisson 到达（req/s，适合看真实业务延迟）

    正式计时前先跑 ``num_warmup`` 个请求做预热，把 CUDA graph 捕获、
    torch.compile 缓存、 PageAttention 首次分配这些一次性开销挡在外面 ——
    不预热的话，第一个请求的 TTFT 会是后面的好几倍，直接拉高均值。
    """
    if aiohttp is None:
        raise RuntimeError("需要 aiohttp：pip install aiohttp")

    url = base_url.rstrip("/") + "/v1/chat/completions"

    # 预生成一小组图片循环复用，避免每张

    imgs = [
        base64.b64encode(make_synthetic_image(seed + i, image_size, image_size)).decode()
        for i in range(max(8, images_per_request * 2))
    ]

    tasks_meta = []
    for i in range(num_requests):
        picked = [imgs[(i + j) % len(imgs)] for j in range(images_per_request)]
        content_imgs = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}}
            for b in picked
        ]
        tasks_meta.append((picked, content_imgs, QUESTION_POOL[i % len(QUESTION_POOL)]))

    def _mk_payload(idx: int) -> dict:
        _, content_imgs, q = tasks_meta[idx]
        return {
            "model": model,
            "messages": [{"role": "user", "content": content_imgs + [{"type": "text", "text": q}]}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    connector = aiohttp.TCPConnector(limit=concurrency * 2, force_close=True)
    timeout_cfg = aiohttp.ClientTimeout(total=timeout)
    results: list[RequestResult] = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout_cfg) as session:
        # --- 预热（不计入统计） ---
        if num_warmup > 0:
            warm = [ _one_request(session, url, _mk_payload(i % num_requests))
                     for i in range(num_warmup) ]
            await asyncio.gather(*warm)

        # --- 正式计时 ---
        t_start = time.perf_counter()
        if request_rate:
            # Poisson 到达：更适合观察真实业务下的延迟分布。
            # 这里用「按固定间隔派发 + 最后统一回收」的方式，
            # 避免 await 每个请求而退化成串行。
            interval = 1.0 / request_rate
            inflight: list[asyncio.Task] = []
            for i in range(num_requests):
                inflight.append(asyncio.create_task(_one_request(session, url, _mk_payload(i))))
                await asyncio.sleep(interval)
            gathered = await asyncio.gather(*inflight, return_exceptions=True)
            results = [
                g if isinstance(g, RequestResult)
                else RequestResult(error=str(g))
                for g in gathered
            ]
        else:
            sem = asyncio.Semaphore(concurrency)

            async def _worker(i: int) -> None:
                async with sem:
                    results.append(await _one_request(session, url, _mk_payload(i)))

            await asyncio.gather(*[_worker(i) for i in range(num_requests)])
        duration_s = time.perf_counter() - t_start

    ok = [r for r in results if r.ok]
    failed = len(results) - len(ok)
    total_output_tokens = sum(r.output_tokens for r in ok)

    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    tpots = [r.tpot_ms for r in ok if r.tpot_ms is not None]
    itls = [x for r in ok for x in r.itl_ms]
    latencies = [r.latency_ms for r in ok if r.latency_ms is not None]

    return {
        "config": {
            "model": model,
            "base_url": base_url,
            "num_requests": num_requests,
            "concurrency": concurrency,
            "request_rate": request_rate,
            "max_tokens": max_tokens,
            "image_size": image_size,
            "images_per_request": images_per_request,
            "num_warmup": num_warmup,
            "seed": seed,
        },
        "summary": {
            "duration_s": round(duration_s, 3),
            "completed_requests": len(ok),
            "failed_requests": failed,
            "total_output_tokens": total_output_tokens,
            "request_throughput_req_per_s": _round(len(ok) / duration_s if duration_s else None, 3),
            "output_token_throughput_tok_per_s": _round(total_output_tokens / duration_s if duration_s else None, 2),
            "mean_latency_ms": _round(mean(latencies)),
            "ttft_ms": {
                "mean": _round(mean(ttfts)),
                "p50": _round(pct(ttfts, 50)),
                "p95": _round(pct(ttfts, 95)),
                "p99": _round(pct(ttfts, 99)),
            },
            "tpot_ms": {
                "mean": _round(mean(tpots)),
                "p50": _round(pct(tpots, 50)),
                "p99": _round(pct(tpots, 99)),
            },
            "itl_ms": {
                "mean": _round(mean(itls)),
                "p99": _round(pct(itls, 99)),
            },
        },
    }


async def _health_ok(base_url: str, timeout: int = 600) -> bool:
    """轮询 /health，等服务就绪。"""
    if aiohttp is None:
        raise RuntimeError("需要 aiohttp：pip install aiohttp")
    url = base_url.rstrip("/") + "/health"
    deadline = time.time() + timeout
    async with aiohttp.ClientSession() as s:
        while time.time() < deadline:
            try:
                async with s.get(url, timeout=5) as r:
                    if r.status == 200:
                        return True
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(3)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="VLM 流式压测（自建客户端）")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--request-rate", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--image-size", type=int, default=896)
    ap.add_argument("--images-per-request", type=int, default=1)
    ap.add_argument("--num-warmup", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--wait-health", action="store_true", help="先等服务 /health 就绪再开跑")
    ap.add_argument("--sample-vram", action="store_true",
                    help="压测期间后台采样 GPU 显存，峰值写入结果（report 表的「显存峰值」列）")
    ap.add_argument("--vram-device", type=int, default=0)
    args = ap.parse_args()

    async def _go() -> None:
        from .vram import VRAMMonitor

        if args.wait_health:
            ok = await _health_ok(args.base_url)
            if not ok:
                raise SystemExit("服务未在超时时间内就绪")

        with VRAMMonitor(device_index=args.vram_device) as mon:
            res = await run_benchmark(
                base_url=args.base_url,
                model=args.model,
                num_requests=args.num_requests,
                concurrency=args.concurrency,
                request_rate=args.request_rate,
                max_tokens=args.max_tokens,
                image_size=args.image_size,
                images_per_request=args.images_per_request,
                num_warmup=args.num_warmup,
                seed=args.seed,
            )
        if args.sample_vram:
            res["summary"]["peak_vram_gib"] = mon.peak_gib
            res["summary"]["vram_baseline_gib"] = mon.stats.baseline_gib
            res["summary"]["vram_total_gib"] = mon.stats.total_gib
            res["summary"]["vram_note"] = mon.summary_line()
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        fp = out / f"bench_{args.tag}_c{args.concurrency}.json"
        fp.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(res["summary"], indent=2, ensure_ascii=False))
        print(f"\n-> 已写入 {fp}")

    asyncio.run(_go())


if __name__ == "__main__":
    main()
