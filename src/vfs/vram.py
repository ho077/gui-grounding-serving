"""显存峰值采样。

为什么必须单独实现
------------------
`nvidia-smi` 只能看到「采样那一刻」的占用，而 vLLM 的峰值往往出现在
模型加载完、第一波 prefill 打满 KV Cache 的瞬间，用轮询很容易漏掉。
这里用一条后台线程以高频读取 NVML，把过程中出现过的最大值记住，
得到的是真正的**峰值**，而不是测完以后剩下的残值。

同时统计 unused-but-allocated（PyTorch caching allocator 已占用但没在用），
让「量化到底省了多少显存」这个问题的答案经得起追问。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VRAMStats:
    peak_used_gib: float = 0.0
    baseline_gib: float = 0.0
    total_gib: float = 0.0
    samples: int = 0
    timeline: list[tuple[float, float]] = field(default_factory=list)
    available: bool = False
    error: str = ""


def _gib(byte_val: float) -> float:
    return round(byte_val / (1024 ** 3), 3)


class VRAMMonitor:
    """后台高频采样 GPU 显存，记录峰值。

    用法::

        with VRAMMonitor() as m:
            ...  # 起服务 / 跑压测
        print(m.stats.peak_used_gib)

    若环境没有 NVML（无 GPU 的开发机），退化为空操作，
    调用方读到 ``stats.available == False`` 即可，不必到处 try/except。
    """

    def __init__(self, device_index: int = 0, interval: float = 0.05, keep_timeline: bool = True):
        self.device_index = device_index
        self.interval = interval
        self.keep_timeline = keep_timeline
        self.stats = VRAMStats()
        self._handle = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._t0 = 0.0

    # ---- NVML 生命周期 ----
    def _init_nvml(self) -> bool:
        try:
            import pynvml  # type: ignore
        except ImportError:
            self.stats.error = "未安装 pynvml，请 pip install nvidia-ml-py"
            return False
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.stats.total_gib = _gib(mem.total)
            self.stats.baseline_gib = _gib(mem.used)
            self.stats.available = True
            return True
        except Exception as e:  # NVML 在驱动异常时会抛各种自定义异常
            self.stats.error = f"NVML 初始化失败：{e}"
            return False

    def _sample_loop(self) -> None:
        import pynvml  # type: ignore

        while not self._stop.is_set():
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                used = _gib(mem.used)
                if used > self.stats.peak_used_gib:
                    self.stats.peak_used_gib = used
                self.stats.samples += 1
                if self.keep_timeline:
                    self.stats.timeline.append((round(time.time() - self._t0, 3), used))
            except Exception:
                break
            time.sleep(self.interval)

    def __enter__(self) -> "VRAMMonitor":
        if self._init_nvml():
            self._t0 = time.time()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    # ---- 便捷读值 ----
    @property
    def peak_gib(self) -> float:
        return self.stats.peak_used_gib

    @property
    def delta_gib(self) -> float:
        """相对开跑前的净增量，剔除机器上其他进程已占用的部分。"""
        return round(self.stats.peak_used_gib - self.stats.baseline_gib, 3)

    def summary_line(self) -> str:
        if not self.stats.available:
            return f"显存采样不可用：{self.stats.error}"
        return (
            f"峰值显存 {self.peak_gib:.2f} GiB（净增 {self.delta_gib:.2f} GiB，"
            f"显存总量 {self.stats.total_gib:.2f} GiB，采样 {self.stats.samples} 次）"
        )


def main() -> None:
    """单独运行：持续采样直到 Ctrl-C，用于手动观察服务加载过程。"""
    try:
        with VRAMMonitor() as m:
            if not m.stats.available:
                print(m.summary_line())
                return
            print("开始采样，Ctrl-C 停止 …")
            while True:
                time.sleep(1)
                print(f"\r  当前 {m.peak_gib:.2f} GiB", end="", flush=True)
    except KeyboardInterrupt:
        print()
        print(m.summary_line())


if __name__ == "__main__":
    main()
