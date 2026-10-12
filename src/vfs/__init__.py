"""gui-grounding-serving 工具包。

按职责分两层：
    通用工程层  hardware / vram / bench / serve   —— 与任务无关，任何 VLM 部署都用得上
    任务层      grounding / report                —— 专门针对 GUI 元素定位

分层的意义：换一个任务（比如换成文档抽取）时，通用层一行不用改。

子模块走**延迟导入**：`python -m vfs.report` 这类入口如果在这里就被
提前 import 进 sys.modules，会触发 RuntimeWarning 并带来不可预期的行为。
"""

import importlib

_MODULES = {"bench", "grounding", "hardware", "report", "serve", "vram"}
__all__ = sorted(_MODULES)
__version__ = "0.2.0"


def __getattr__(name: str):
    if name in _MODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
