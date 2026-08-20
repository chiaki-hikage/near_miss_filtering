#!/usr/bin/env python3
"""comma2k19 の demo split (Hugging Face) を取得して、通常のディレクトリ構成に展開する。

Hugging Face の `commaai/comma2k19` には 2 通りの置き方がある。

  data/demo-*.parquet   64 セグメント / 約 228 MB。映像なし。CAN は raw_can まで入る
  raw_data/Chunk_N.zip  1 チャンク 8.7〜9.9 GB。映像込みの完全版

このスクリプトは前者を落とし、後者と同じディレクトリ構成に並べ替える。
展開後は Chunk_N.zip を展開した場合とまったく同じ扱いで
run_detection.py / inspect_segment.py / validate_signals.py にかけられる。

使い方:
  python scripts/fetch_demo_dataset.py --out data/comma2k19_demo

映像 (video.hevc) と raw_log.bz2 は demo split に含まれない。
CAN のみを使う現在の範囲では困らない。目視確認は preview.png で行う。
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = "commaai/comma2k19"
SHARDS = (
    "data/demo-00000-of-00003.parquet",
    "data/demo-00001-of-00003.parquet",
    "data/demo-00002-of-00003.parquet",
)
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/"

# parquet の列名 → 展開先のディレクトリ。CAN のみを対象にする。
CAN_LAYOUT = {
    "speed": ("processed_log__CAN__speed__t", "processed_log__CAN__speed__value"),
    "steering_angle": ("processed_log__CAN__steering_angle__t", "processed_log__CAN__steering_angle__value"),
    "wheel_speed": ("processed_log__CAN__wheel_speed__t", "processed_log__CAN__wheel_speed__value"),
    "radar": ("processed_log__CAN__radar__t", "processed_log__CAN__radar__value"),
}
RAW_CAN_FIELDS = {
    "t": "processed_log__CAN__raw_can__t",
    "address": "processed_log__CAN__raw_can__address",
    "src": "processed_log__CAN__raw_can__src",
    "data": "processed_log__CAN__raw_can__data",
}


def download(url: str, dest: Path) -> Path:
    """既にあれば再取得しない。"""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  既取得: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  取得中: {dest.name}", end="", flush=True)

    def hook(block: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100.0, 100.0 * block * block_size / total)
            print(f"\r  取得中: {dest.name}  {pct:5.1f}%  ({total / 1e6:.0f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.rename(dest)
    print()
    return dest


def _save(path: Path, arr: np.ndarray) -> None:
    """comma2k19 は拡張子なしのファイルに npy を書いている。同じ形にする。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        np.save(fh, arr, allow_pickle=False)


def _as_2d(values: list) -> np.ndarray:
    """list<list<double>> を (N, K) にする。"""
    if not values:
        return np.empty((0, 0))
    return np.asarray([list(v) for v in values], dtype=np.float64)


def write_segment(row: dict, out_root: Path) -> Path:
    """1 行 (= 1 セグメント) を <drive_id>/<segment>/ 以下に展開する。"""
    drive_id, seg_no = row["segment_id"].rsplit("/", 1)
    seg_dir = out_root / drive_id / seg_no
    can_dir = seg_dir / "processed_log" / "CAN"
    log = row["log"]

    for name, (t_key, v_key) in CAN_LAYOUT.items():
        t = np.asarray(log[t_key], dtype=np.float64)
        raw = log[v_key]
        v = np.asarray(raw, dtype=np.float64) if raw and not isinstance(raw[0], list) else _as_2d(raw)
        _save(can_dir / name / "t", t)
        _save(can_dir / name / "value", v)

    raw_dir = can_dir / "raw_can"
    _save(raw_dir / "t", np.asarray(log[RAW_CAN_FIELDS["t"]], dtype=np.float64))
    _save(raw_dir / "address", np.asarray(log[RAW_CAN_FIELDS["address"]], dtype=np.int64))
    _save(raw_dir / "src", np.asarray(log[RAW_CAN_FIELDS["src"]], dtype=np.int64))
    # 8 バイトのペイロードを |S8 で保持する。Chunk_N.zip 側と同じ持ち方。
    _save(raw_dir / "data", np.asarray([bytes(b) for b in log[RAW_CAN_FIELDS["data"]]], dtype="S8"))

    preview = row.get("preview")
    if preview and preview.get("bytes"):
        (seg_dir / "preview.png").write_bytes(preview["bytes"])

    return seg_dir


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/comma2k19_demo"), help="展開先")
    p.add_argument("--cache", type=Path, default=Path("data/_hf_cache"), help="parquet の保存先")
    p.add_argument("--shards", type=int, nargs="*", default=None, help="取得するシャード番号 (既定: すべて)")
    p.add_argument("--parquet", type=Path, nargs="*", default=None, help="取得済みの parquet を直接指定する")
    args = p.parse_args()

    if args.parquet:
        paths = list(args.parquet)
    else:
        targets = SHARDS if args.shards is None else [SHARDS[i] for i in args.shards]
        print(f"Hugging Face から取得します: {REPO}")
        paths = [download(BASE_URL + s, args.cache / Path(s).name) for s in targets]

    print(f"\n展開先: {args.out}")
    total = 0
    drives: set[str] = set()
    for path in paths:
        table = pq.read_table(path)
        for row in table.to_pylist():
            seg_dir = write_segment(row, args.out)
            drives.add(seg_dir.parent.name)
            total += 1
            print(f"\r  展開 {total} セグメント", end="", flush=True)
    print()

    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"\nセグメント {total} 件 / ドライブ {len(drives)} 本 / {size / 1e6:.0f} MB")
    print(f"\n次の手順:")
    print(f"  python scripts/validate_signals.py {args.out} --max-segments 3")
    print(f"  python scripts/run_detection.py {args.out} --out out/demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
