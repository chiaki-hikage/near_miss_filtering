#!/usr/bin/env python3
"""commaCarSegments からセグメントを取得する。

  # 車種の一覧とセグメント数を見る (database.json を落とすだけ。約 9 MB)
  python scripts/fetch_car_segments.py --list

  # 取得量を見積もるだけ。ダウンロードはしない
  python scripts/fetch_car_segments.py TOYOTA_RAV4_TSS2 --routes 3 --per-route 5 --dry-run

  # 実際に取得する
  python scripts/fetch_car_segments.py TOYOTA_RAV4_TSS2 --routes 3 --per-route 5

--routes / --per-route を指定すると、同じルートの連続したセグメントを選ぶ。
60 秒境界を跨ぐイベントを連結して扱えるかを確認するにはこちらが要る。
--limit だけを指定した場合は、ルートをまたいで先頭から詰めて選ぶ。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from near_miss.io.comma_car_segments import (
    DEFAULT_CACHE,
    SegmentName,
    ensure_segments,
    fetch_database,
    load_database,
    local_path,
    select_segments,
)

# ヘッダ実測の平均 (12 件サンプル, TOYOTA_RAV4_TSS2)。見積り表示に使う。
TYPICAL_SEGMENT_MB = 1.38


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("platform", nargs="?", help="車種キー (例 TOYOTA_RAV4_TSS2)")
    p.add_argument("--list", action="store_true", help="車種の一覧を出して終わる")
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="保存先")
    p.add_argument("--limit", type=int, default=None, help="取得するセグメント数の上限")
    p.add_argument("--routes", type=int, default=None, help="使うルート数")
    p.add_argument("--per-route", type=int, default=None, help="1 ルートあたりの連続セグメント数")
    p.add_argument("--workers", type=int, default=4, help="並列ダウンロード数")
    p.add_argument("--dry-run", action="store_true", help="選ばれるセグメントと取得量だけ出す")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    fetch_database(args.cache)

    if args.list or not args.platform:
        db = load_database(args.cache)
        rows = sorted(db.items(), key=lambda kv: -len(kv[1]))
        print(f"車種 {len(rows)} 種類 / セグメント合計 {sum(len(v) for v in db.values()):,}")
        print()
        print(f"{'車種キー':<34} {'セグメント数':>10} {'概算サイズ':>10}")
        for k, v in rows[:40]:
            print(f"{k:<34} {len(v):>10,} {len(v) * TYPICAL_SEGMENT_MB / 1024:>8.1f} GB")
        if len(rows) > 40:
            print(f"... 他 {len(rows) - 40} 種類")
        return 0

    names = select_segments(
        args.platform,
        limit=args.limit,
        routes=args.routes,
        per_route=args.per_route,
        cache_dir=args.cache,
    )
    have = [n for n in names if local_path(n, args.cache).is_file()]
    todo = [n for n in names if n not in set(have)]

    print(f"車種       : {args.platform}")
    print(f"選択       : {len(names)} セグメント ({len(set(SegmentName.parse(n).drive_id for n in names))} ルート)")
    print(f"取得済み   : {len(have)}")
    print(f"取得予定   : {len(todo)}  概算 {len(todo) * TYPICAL_SEGMENT_MB:.0f} MB")
    print(f"保存先     : {args.cache / 'segments'}")
    print()
    for n in names:
        mark = "済" if n in set(have) else "→"
        print(f"  {mark} {n}")

    if args.dry_run:
        print("\n--dry-run のため取得しません。")
        return 0
    if not todo:
        print("\nすべて取得済みです。")
        return 0

    print()
    done = {"n": 0}

    def progress(name, result):
        done["n"] += 1
        state = "失敗" if isinstance(result, Exception) else "完了"
        print(f"  [{done['n']}/{len(todo)}] {state} {name}")

    results = ensure_segments(todo, args.cache, workers=args.workers, on_done=progress)
    failed = [n for n, r in results.items() if isinstance(r, Exception)]
    total = sum(local_path(n, args.cache).stat().st_size for n in names if local_path(n, args.cache).is_file())
    print()
    print(f"取得完了 {len(todo) - len(failed)} / {len(todo)}  (失敗 {len(failed)})")
    print(f"ローカル合計 {total / 1e6:.1f} MB")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
