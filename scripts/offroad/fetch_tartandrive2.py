"""TartanDrive 2.0 の信号 (KITTI 形式) と rosbag のメタデータを取る。

設定: configs/offroad/tartandrive2.yaml。画像・点群・rosbag 本体は取らない。
すでにあるファイルは取り直さない。取った一覧を <data_dir>/fetch_manifest.csv に書く。

    uv run python scripts/offroad/fetch_tartandrive2.py            # 全部
    uv run python scripts/offroad/fetch_tartandrive2.py --dry-run  # 取るファイルの数と大きさ (HEAD) だけ
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def walk_files(node) -> list[str]:
    """files.yaml の入れ子 ({key: {files: [...], <sub>: {...}}}) から、ファイルの key を全部集める。"""
    out: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "files" and isinstance(v, list):
                out.extend(v)
            else:
                out.extend(walk_files(v))
    return out


def select_keys(index: dict, fc: dict) -> list[str]:
    keys = []
    for k in walk_files(index.get("kitti", {})):
        parts = k.split("/")
        # kitti/all_topics/<run>/<dir>/<file> か kitti/all_topics/<run>/<file>
        if len(parts) == 5 and parts[3] in fc["kitti_signal_dirs"]:
            keys.append(k)
        elif len(parts) == 4 and parts[3] in fc["kitti_extra_files"]:
            keys.append(k)
    for k in walk_files(index.get("bags", {})):
        if any(k.endswith(s) for s in fc["bag_meta_suffixes"]):
            keys.append(k)
    return sorted(set(keys))


def url_of(src: dict, key: str) -> str:
    return f"{src['endpoint']}/{src['bucket']}/{urllib.parse.quote(key)}"


def head_size(url: str, timeout: float) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        n = r.headers.get("Content-Length")
        return int(n) if n is not None else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offroad/tartandrive2.yaml")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    src, fc = cfg["source"], cfg["fetch"]
    data = Path(cfg["data_dir"])
    index = load_cfg(src["file_index"])
    keys = select_keys(index, fc)
    print(f"対象 {len(keys)} ファイル")
    rows, total = [], 0
    for i, key in enumerate(keys):
        dst = data / key
        url = url_of(src, key)
        try:
            size = head_size(url, fc["timeout_s"])
        except Exception as e:  # 404 などは記録して続ける
            rows.append(dict(key=key, status=f"head_error:{e}", bytes=""))
            continue
        if size is not None and size > fc["max_file_mb"] * 1e6:
            rows.append(dict(key=key, status="skipped_too_large", bytes=size))
            continue
        total += size or 0
        if a.dry_run:
            rows.append(dict(key=key, status="dry_run", bytes=size))
            continue
        if dst.exists() and size is not None and dst.stat().st_size == size:
            rows.append(dict(key=key, status="exists", bytes=size))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=fc["timeout_s"]) as r, open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                tmp.rename(dst)
                rows.append(dict(key=key, status="ok", bytes=dst.stat().st_size))
                break
            except Exception as e:
                if attempt == 2:
                    rows.append(dict(key=key, status=f"error:{e}", bytes=""))
                time.sleep(2)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(keys)}  {total / 1e6:.0f} MB", flush=True)
    data.mkdir(parents=True, exist_ok=True)
    man = data / ("fetch_manifest_dry.csv" if a.dry_run else "fetch_manifest.csv")
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "status", "bytes"])
        w.writeheader()
        w.writerows(rows)
    st: dict[str, int] = {}
    for r in rows:
        st[r["status"].split(":")[0]] = st.get(r["status"].split(":")[0], 0) + 1
    print(f"合計 {total / 1e6:.0f} MB、状態 {st} -> {man}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
