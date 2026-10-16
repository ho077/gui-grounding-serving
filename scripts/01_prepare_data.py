#!/usr/bin/env python
"""下载 GUI grounding 数据集并转成训练 / 评测两种格式。

数据源
------
训练（默认）：
    showlab/ShowUI-desktop-8K   桌面截图（8K 量级）       ~0.5 GB
训练（可选，需显式加 flag）：
    showlab/ShowUI-web          网页截图，images.tar.gz   ~40 GB
    Yuxiang007/AMEX             移动端，分包压缩          ~87 GB
评测：
    KevinQHLin/ScreenSpot       原始 ScreenSpot           ~1.1 GB
    likaixin/ScreenSpot-Pro     高分辨率专业软件（可选）  ~1.6 GB

体积是硬约束：租来的云盘常常只有 50 GB，默认三个一起下会直接撑爆。
因此默认只下 desktop-8K + ScreenSpot，大集用 --with-web / --with-amex 显式开启。

网络
----
`huggingface.co` 在国内直连常常超时。脚本默认把 `HF_ENDPOINT` 指向
`hf-mirror.com`，实测可用；若你网络能直连，加 `--endpoint https://huggingface.co`。

坐标约定
--------
ShowUI 系列元数据里的 point / bbox 是**归一化** [0,1]。
训练目标统一乘 1000 变成 [0,1000] 整数（Qwen 系列惯例），
与 `src/vfs/grounding.py` 的解析严格对齐——两边必须一致，否则训练出的模型
在评测时会被判成全错。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN_REPOS = {
    "ShowUI-desktop": "showlab/ShowUI-desktop-8K",   # ~0.5 GB，默认下
}
# 大数据集：默认不下，避免撑爆小数据盘
BIG_TRAIN_REPOS = {
    "ShowUI-web": "showlab/ShowUI-web",              # ~40 GB
    "AMEX": "Yuxiang007/AMEX",                       # ~87 GB
}
EVAL_REPOS = {
    "ScreenSpot": "KevinQHLin/ScreenSpot",           # ~1.1 GB
}


def log(msg: str) -> None:
    print(f"[data] {msg}", flush=True)


def _hub():
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download
    except ImportError:
        raise SystemExit(
            "缺少 huggingface_hub。先装依赖：\n"
            "    pip install -U huggingface_hub\n"
            "（若在国内，可加 -i https://pypi.tuna.tsinghua.edu.cn/simple）"
        )


def download(repo_id: str, local_dir: Path, endpoint: str) -> bool:
    os.environ["HF_ENDPOINT"] = endpoint
    snapshot_download = _hub()
    try:
        snapshot_download(
            repo_id=repo_id, repo_type="dataset",
            local_dir=str(local_dir), max_workers=4,
        )
        return True
    except Exception as e:  # noqa: BLE001
        log(f"  !! 下载失败 {repo_id}: {type(e).__name__}: {e}")
        return False


def find_images(d: Path) -> Path:
    """在数据集目录里定位图片根目录。"""
    for cand in ("images", "img", "image", "screenshots"):
        p = d / cand
        if p.is_dir():
            return p
    if any(d.glob("*.png")) or any(d.glob("*.jpg")):
        return d
    # 再下探一层（某些 repo 按平台分子目录）
    for sub in sorted(d.iterdir()):
        if sub.is_dir():
            for cand in ("images", "img", "image"):
                if (sub / cand).is_dir():
                    return sub / cand
    return d


def iter_metadata(d: Path):
    """遍历数据集目录下所有可能的标注 json。"""
    seen = set()
    for p in sorted(d.rglob("*.json")):
        if p.stat().st_size > 200 * 1024 * 1024:
            continue
        if p.resolve() in seen:
            continue
        seen.add(p.resolve())
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(rows, list) and rows and isinstance(rows[0], dict) \
                and ("element" in rows[0] or "bbox" in rows[0]):
            yield p, rows


def to_1000(pt, iw: int, ih: int) -> tuple[int, int]:
    """归一化 [0,1] → [0,1000] 整数。已经是像素的按分辨率折算回去。"""
    x, y = float(pt[0]), float(pt[1])
    if 0 <= x <= 1.5 and 0 <= y <= 1.5:
        return int(round(x * 1000)), int(round(y * 1000))
    return int(round(x / max(iw, 1) * 1000)), int(round(y / max(ih, 1) * 1000))


def build_train(roots: list[Path], out: Path, max_per_image: int,
                limit: int) -> int:
    """把 ShowUI 系数据转成 LLaMA-Factory sharegpt 格式。"""
    from vfs.grounding import PROMPT_TEMPLATE  # 与评测共用同一 prompt

    rows: list[dict] = []
    for root in roots:
        if not root.exists():
            continue
        img_root = find_images(root)
        n_before = len(rows)
        for meta, items in iter_metadata(root):
            for row in items:
                img = row.get("img_url") or row.get("image") or ""
                ip = img_root / img
                if not ip.exists():
                    continue
                size = row.get("img_size") or [0, 0]
                iw, ih = int(size[0]), int(size[1])
                for el in row.get("element", [])[:max_per_image]:
                    pt = el.get("point")
                    if not pt:
                        continue
                    gx, gy = to_1000(pt, iw, ih)
                    rows.append({
                        "id": f"{root.name}-{len(rows)}",
                        "images": [str(ip.resolve())],
                        "conversations": [
                            {"from": "human",
                             "value": "<image>\n" + PROMPT_TEMPLATE.format(
                                 instruction=el.get("instruction", ""))},
                            {"from": "gpt",
                             "value": json.dumps({"point": [gx, gy]})},
                        ],
                    })
                    if limit and len(rows) >= limit:
                        break
                if limit and len(rows) >= limit:
                    break
            if limit and len(rows) >= limit:
                break
        log(f"  {root.name}: +{len(rows) - n_before} 条")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"训练集 -> {out}（{len(rows)} 条）")
    return len(rows)


def write_dataset_info(train_file: Path) -> None:
    info = {
        "gui_grounding_train": {
            "file_name": str(train_file.resolve()),
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {"role_tag": "from", "content_tag": "value",
                     "user_tag": "human", "assistant_tag": "gpt"},
        }
    }
    p = ROOT / "data" / "dataset_info.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"dataset_info -> {p}")


def summarize_eval(roots: list[Path]) -> None:
    """统计评测集条目数。

    早期版本按 ShowUI 的嵌套 `element` 结构数，而 ScreenSpot 是扁平结构，
    于是永远报 0 —— 正是这个 0 让人以为数据没下全、白查一轮磁盘。
    这里复用评测侧的同一套文件选择逻辑（vfs.grounding._screenspot_files），
    保证"数据准备说有多少条"和"评测实际跑多少条"是同一个口径。
    """
    from vfs.grounding import _screenspot_files  # 单一口径来源

    for root in roots:
        if not root.exists():
            log(f"  !! 评测集缺失: {root}")
            continue
        files = _screenspot_files(root)
        if not files:
            log(f"  评测集 {root.name}: 未识别到元数据（目录结构可能变了）")
            continue
        n = 0
        for meta, _ in files:
            try:
                rows = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(rows, list):
                n += len(rows)
        log(f"  评测集 {root.name}: {n} 个标注元素"
            f"（{len(files)} 个元数据文件）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="https://hf-mirror.com",
                    help="HF 端点；国内默认走镜像")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--limit", type=int, default=0, help="限制训练样本数（调试用）")
    ap.add_argument("--max-per-image", type=int, default=3,
                    help="每张图最多取几个元素，避免单图刷屏")
    ap.add_argument("--with-web", action="store_true",
                    help="额外下 ShowUI-web（约 40 GB，需确认磁盘够）")
    ap.add_argument("--with-amex", action="store_true",
                    help="额外下 AMEX（约 87 GB，需确认磁盘够）")
    ap.add_argument("--with-pro", action="store_true",
                    help="额外下 ScreenSpot-Pro（约 1.6 GB，高分辨率更难）")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    base = Path(args.data_dir)
    if not base.is_absolute():
        base = ROOT / base
    raw = base / "raw"
    (base / "train").mkdir(parents=True, exist_ok=True)
    (base / "eval").mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        log(f"HF_ENDPOINT = {args.endpoint}")
        for name, repo in TRAIN_REPOS.items():
            log(f"下载训练数据 {repo} ...")
            download(repo, raw / name, args.endpoint)
        for name, repo in EVAL_REPOS.items():
            log(f"下载评测集 {repo} ...")
            download(repo, base / "eval" / name, args.endpoint)
        if args.with_web:
            download(BIG_TRAIN_REPOS["ShowUI-web"], raw / "ShowUI-web",
                     args.endpoint)
        if args.with_amex:
            download(BIG_TRAIN_REPOS["AMEX"], raw / "AMEX", args.endpoint)
        if args.with_pro:
            download("likaixin/ScreenSpot-Pro", base / "eval" / "ScreenSpot-Pro",
                     args.endpoint)

    train_roots = [raw / n for n in TRAIN_REPOS]
    if args.with_web:
        train_roots.append(raw / "ShowUI-web")
    if args.with_amex:
        train_roots.append(raw / "AMEX")
    train_file = base / "train" / "gui_grounding_train.json"
    n = build_train(train_roots, train_file, args.max_per_image, args.limit)
    if n == 0:
        log("!! 训练集为空。常见原因：")
        log("   1) 下载失败 —— 换 --endpoint 重试")
        log("   2) 图片路径与元数据对不上 —— 检查 find_images 的探测结果")
        raise SystemExit(1)
    write_dataset_info(train_file)
    summarize_eval([base / "eval" / n for n in EVAL_REPOS])

    log("完成。下一步：bash scripts/02_train.sh")


if __name__ == "__main__":
    main()
