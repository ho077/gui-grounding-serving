"""坐标约定修复的单元验证：parse_point + point_to_pixels。

对应事故：零样本 Qwen2.5-VL 输出绝对像素，eval 按 [0,1000] 缩放，
导致 acc 0.5% / offset p50 1638px（同样样本按像素解释 6/6 命中）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vfs.grounding import parse_point, point_to_pixels  # noqa: E402

cases = [
    # (模型原始输出, 图宽, 图高, space, 期望像素点)
    ('{"point": [2408, 159]}', 2560, 1440, "auto", (2408.0, 159.0)),   # 像素越界→判像素
    ('{"point": [1098, 203]}', 2360, 1640, "auto", (1098.0, 203.0)),
    ('{"point": [500, 500]}', 2560, 1440, "auto", (1280.0, 720.0)),    # [0,1000] 中点
    ('{"point": [500, 500]}', 2560, 1440, "norm", (1280.0, 720.0)),
    ('{"point": [2408, 159]}', 2560, 1440, "pixel", (2408.0, 159.0)),  # 强制像素
    ('{"point": [2408, 159]}', 2560, 1440, "norm", (6164.48, 228.96)),  # 强制归一化(旧行为)
    ('{"point": [0, 0]}', 960, 540, "auto", (0.0, 0.0)),
    ('{"point": [1000, 1000]}', 960, 540, "auto", (960.0, 540.0)),
]

fails = 0
for text, w, h, space, expect in cases:
    pt = parse_point(text)
    got = point_to_pixels(pt, w, h, space)
    ok = abs(got[0] - expect[0]) < 0.01 and abs(got[1] - expect[1]) < 0.01
    fails += 0 if ok else 1
    print(f"{'OK ' if ok else 'FAIL'} {text} @{w}x{h} [{space}] -> {got}")

# 退化输出：格式走形时应能救回或明确失败
assert parse_point('{"point": [123, 456]}') == (123.0, 456.0)
assert parse_point("the point is (123, 456)") == (123.0, 456.0)
assert parse_point("no coordinates here") is None
assert parse_point("") is None
print("parse_point 边界用例通过")

sys.exit(1 if fails else 0)
