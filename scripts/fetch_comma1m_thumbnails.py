#!/usr/bin/env python3
"""comma1M の thumbnail.jpg を取得する (段階 C: 天候の確認)。

localizer には録画日時が入っていないので、地理だけでは
「積雪しうる地域」までしか言えない。実際に雪・雨・濡れた路面かどうかは
映像を見るしかないが、1 分の映像は 1 カメラ 75 MB ある。
thumbnail.jpg は 1 件 13 KB (482x302) で、路面と空の状態を見るには足りる。

使い方:
  python scripts/fetch_comma1m_thumbnails.py --dry-run
  python scripts/fetch_comma1m_thumbnails.py --regions out/comma1m/regions.csv
  python scripts/fetch_comma1m_thumbnails.py --cold-tier heavy moderate
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd
import requests

from near_miss.io import comma1m

THUMB_BYTES = 13_500
_local = threading.local()


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def fetch_one(args: tuple[str, Path]) -> str:
    sid, cache = args
    try:
        comma1m.ensure_file(session(), sid, cache, name="thumbnail.jpg")
        return ""
    except Exception as exc:
        return f"{sid}: {str(exc)[:100]}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regions", type=Path, default=Path("out/comma1m/regions.csv"))
    p.add_argument("--cache", type=Path, default=comma1m.DEFAULT_CACHE / "thumbnails")
    p.add_argument("--cold-tier", nargs="*", default=None, help="絞り込む寒冷度 (既定は全件)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.regions)
    if args.cold_tier:
        df = df[df["cold_tier"].isin(args.cold_tier)]
    ids = list(df["segment_id"])[: args.limit]

    have = {p.stem for p in args.cache.glob("*.thumbnail.jpg")} if args.cache.exists() else set()
    have = {h.replace(".thumbnail", "") for h in have}
    todo = [i for i in ids if i not in have]

    print(f"対象   : {len(ids)} 件 (取得済み {len(ids) - len(todo)})")
    print(f"転送量 : 約 {len(todo) * THUMB_BYTES / 1e6:.1f} MB")
    if args.dry_run:
        return 0

    args.cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    errs = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, e in enumerate(ex.map(fetch_one, [(s, args.cache) for s in todo]), 1):
            if e:
                errs.append(e)
            if i % 400 == 0:
                print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    print(f"完了 {len(todo) - len(errs)} 件 / 失敗 {len(errs)} 件  {time.time() - t0:.0f}s")
    for e in errs[:5]:
        print("  ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
