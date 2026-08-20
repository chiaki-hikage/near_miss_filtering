#!/usr/bin/env python3
"""選んだセグメントの localizer.safetensors を取得する (段階 D)。

1 件 2.54 MB。全件 (4,216) だと 10.7 GB になるので、
位置と thumbnail で絞ったものだけを落とす。

グループ:
  snow    寒冷地かつ雪らしい見た目のもの   snow_score の降順
  wet     雨・濡れ路面・曇天らしいもの     wet_score の降順
  control 好天の対照群                     clear_score の降順
いずれも夜間 (val_mean < 0.25) は除く。夜間は雪と暗さの区別がつかない。

使い方:
  python scripts/fetch_comma1m_localizers.py --group snow wet control --top 300 --dry-run
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

MB_PER_FILE = 2.54
GROUPS = {
    "snow": dict(sort="snow_score", cold=("heavy", "moderate")),
    "wet": dict(sort="wet_score", cold=None),
    "control": dict(sort="clear_score", cold=None),
}
_local = threading.local()


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def select(df: pd.DataFrame, group: str, top: int) -> pd.DataFrame:
    spec = GROUPS[group]
    d = df[~df["is_dark"].astype(bool)]
    if spec["cold"]:
        d = d[d["cold_tier"].isin(spec["cold"])]
    d = d.sort_values(spec["sort"], ascending=False).head(top).copy()
    d["group"] = group
    return d


def fetch_one(args: tuple[str, Path]) -> str:
    sid, cache = args
    try:
        comma1m.ensure_file(session(), sid, cache)
        return ""
    except Exception as exc:
        return f"{sid}: {str(exc)[:100]}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weather", type=Path, default=Path("out/comma1m/weather.csv"))
    p.add_argument("--group", nargs="+", default=["snow", "wet", "control"], choices=list(GROUPS))
    p.add_argument("--top", type=int, default=300, help="グループごとの件数")
    p.add_argument("--cache", type=Path, default=comma1m.DEFAULT_CACHE)
    p.add_argument("--out", type=Path, default=Path("out/comma1m/selection.csv"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.weather)
    sel = pd.concat([select(df, g, args.top) for g in args.group], ignore_index=True)
    # 同じセグメントが複数グループに入りうる。先に来たグループを優先
    sel = sel.drop_duplicates("segment_id", keep="first")

    have = {p.stem for p in args.cache.glob("*.safetensors")}
    todo = [s for s in sel["segment_id"] if s not in have]
    print(f"選択  : {len(sel)} 件  " + "  ".join(f"{g}={int((sel['group'] == g).sum())}" for g in args.group))
    print(f"未取得: {len(todo)} 件")
    print(f"転送量: 約 {len(todo) * MB_PER_FILE / 1000:.2f} GB  ({MB_PER_FILE} MB/件)")
    if args.dry_run:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(args.out, index=False)
    args.cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    errs = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, e in enumerate(ex.map(fetch_one, [(s, args.cache) for s in todo]), 1):
            if e:
                errs.append(e)
            if i % 100 == 0:
                mb = i * MB_PER_FILE
                print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s  {mb / (time.time() - t0):.1f} MB/s",
                      file=sys.stderr, flush=True)
    print(f"完了 {len(todo) - len(errs)} 件 / 失敗 {len(errs)} 件  {time.time() - t0:.0f}s")
    for e in errs[:5]:
        print("  ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
