#!/usr/bin/env python3
"""comma1M の各セグメントの位置だけを安価に取得する (段階 A)。

localizer.safetensors は 1 件 2.5 MB、映像は 1 件 75 MB/カメラある。
位置で絞り込む段階では全体を落とさず、safetensors のヘッダを読んでから
states の必要な行だけを HTTP Range で取る。1 件あたり約 2.3 KB。

使い方:
  python scripts/fetch_comma1m_positions.py --dry-run
  python scripts/fetch_comma1m_positions.py --out out/comma1m/positions.csv

出力列:
  segment_id, n_rows, duration_s, lat/lon/alt/speed の開始・中央・終了,
  start-end 直線距離, 移動方位
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import requests
from pymap3d import ecef2geodetic

from near_miss.io import comma1m

ROWS = (0.0, 0.5, 1.0)
LABELS = {0.0: "start", 0.5: "mid", 1.0: "end"}
BYTES_PER_SEGMENT = 2048 + len(ROWS) * 10 * 8   # ヘッダ + 3 行 x 10 列
RATE_HZ = 100.0                                  # states の実測レート

_local = threading.local()


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def segment_list(cache: Path, limit: int | None) -> list[str]:
    """セグメント ID 一覧。取得済みならキャッシュを使う。"""
    path = cache / "segments.json"
    if path.exists():
        ids = json.loads(path.read_text())
    else:
        ids = comma1m.list_segments(requests.Session())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ids))
    return ids[:limit] if limit else ids


def fetch_one(segment_id: str) -> dict | None:
    try:
        r = comma1m.fetch_state_rows(session(), segment_id, rows=ROWS)
    except Exception as exc:
        return {"segment_id": segment_id, "error": str(exc)[:120]}

    out: dict = {
        "segment_id": segment_id,
        "n_rows": r["n_rows"],
        "duration_s": round(r["n_rows"] / RATE_HZ, 2),
        "error": "",
    }
    ecef = {}
    for frac, v in r["rows"].items():
        lab = LABELS[frac]
        lat, lon, alt = ecef2geodetic(*v[0:3])
        ecef[lab] = np.asarray(v[0:3], dtype=float)
        out[f"lat_{lab}"] = float(lat)
        out[f"lon_{lab}"] = float(lon)
        out[f"alt_{lab}"] = float(alt)
        out[f"speed_{lab}"] = float(np.linalg.norm(v[7:10]))
    out["chord_m"] = float(np.linalg.norm(ecef["end"] - ecef["start"]))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("out/comma1m/positions.csv"))
    p.add_argument("--cache", type=Path, default=comma1m.DEFAULT_CACHE)
    p.add_argument("--limit", type=int, default=None, help="先頭 N セグメントだけ")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--dry-run", action="store_true", help="転送量の見積りだけ出して終了")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ids = segment_list(args.cache, args.limit)
    total_mb = len(ids) * BYTES_PER_SEGMENT / 1e6
    print(f"対象セグメント : {len(ids)}")
    print(f"取得方法       : HTTP Range (ヘッダ + states {len(ROWS)} 行)")
    print(f"想定転送量     : 約 {total_mb:.1f} MB  ({BYTES_PER_SEGMENT} B/セグメント)")
    print(f"比較           : localizer 全件なら {len(ids) * 2.54 / 1000:.1f} GB, "
          f"映像全件なら {len(ids) * 150 / 1000:.1f} GB")
    if args.dry_run:
        return 0

    rows: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(fetch_one, ids), 1):
            if res:
                rows.append(res)
            if i % 200 == 0:
                print(f"  {i}/{len(ids)}  {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_err = int((df["error"] != "").sum())
    print(f"\n{args.out}  {len(df)} 行  (失敗 {n_err})  {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
