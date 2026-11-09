#!/usr/bin/env python3
"""comma2k19 の Chunk_N.zip から、指定したセグメントの CAN・IMU・global_pose だけを取る。

映像 (video.hevc) と raw_log.bz2 は取らない。zip 全体 (1 チャンク 9 GB 前後) も落とさない。
HTTP の範囲指定で目次と要る部分だけを読む (near_miss.io.comma2k19_remote)。

  # 取得量を見積もるだけ (目次だけ読む)
  uv run python scripts/fetch/fetch_comma2k19_remote.py --segments data/comma2k19_civic/selection.csv --dry-run

  # 取得する。途中で止めても、取得済みのファイルは飛ばして続きから取れる
  uv run python scripts/fetch/fetch_comma2k19_remote.py --segments data/comma2k19_civic/selection.csv

--segments の CSV には chunk, drive, seg の列が要る (他の列は無視する)。
書き出す構成は Chunk_N.zip を展開したときと同じ
(<out>/<dongle>|<route>/<seg>/processed_log/...)。
<out>/fetch_manifest.csv に、セグメントごとのファイル数・読んだバイト数・結果を書く。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import _bootstrap  # noqa: F401

from near_miss.io.comma2k19_remote import CHUNK_URL, DEFAULT_INCLUDE, fetch_segment, open_chunk, segment_members


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segments", type=Path, required=True, help="chunk, drive, seg の列を持つ CSV")
    p.add_argument("--out", type=Path, default=None, help="保存先。既定は --segments と同じディレクトリ")
    p.add_argument("--include", nargs="+", default=list(DEFAULT_INCLUDE),
                   help="取るファイル (セグメントからの相対パスの glob)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out or args.segments.parent
    with args.segments.open(encoding="utf-8") as fh:
        rows = [(int(r["chunk"]), r["drive"], str(int(r["seg"]))) for r in csv.DictReader(fh)]
    by_chunk: dict[int, list[tuple[str, str]]] = {}
    for c, d, s in rows:
        by_chunk.setdefault(c, []).append((d, s))
    print(f"セグメント {len(rows)} 件 / チャンク {sorted(by_chunk)}")
    print(f"取るファイル: {', '.join(args.include)}")
    print(f"保存先: {out}")

    manifest_path = out / "fetch_manifest.csv"
    fields = ["chunk", "drive", "seg", "status", "n_files", "n_fetched", "bytes_read", "mode", "error"]
    done: dict[tuple[str, str], dict] = {}
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as fh:
            done = {(r["drive"], r["seg"]): r for r in csv.DictReader(fh)}

    total_cmp, total_raw, results = 0, 0, []
    t0 = time.time()
    for c in sorted(by_chunk):
        f, z = open_chunk(c)
        segs = by_chunk[c]
        plan = {(d, s): segment_members(z, c, d, s, tuple(args.include)) for d, s in segs}
        missing = [k for k, v in plan.items() if not v]
        cmp_ = sum(m.info.compress_size for v in plan.values() for m in v)
        raw = sum(m.info.file_size for v in plan.values() for m in v)
        total_cmp += cmp_
        total_raw += raw
        print(f"\nChunk_{c}: {len(segs)} セグメント  圧縮 {cmp_ / 1e6:.0f} MB / 展開後 {raw / 1e6:.0f} MB"
              f"  (目次 {f.nbytes / 1e6:.1f} MB)" + (f"  ファイルの無いセグメント {len(missing)}" if missing else ""))
        if args.dry_run:
            continue

        def work(key):
            d, s = key
            try:
                st = fetch_segment(f, plan[key], out / d / s)
                return key, dict(st, status="ok" if plan[key] else "no_files", error="")
            except Exception as e:  # 1 件の失敗で全体を止めない。manifest に残して後で取り直す
                return key, {"status": "error", "error": f"{type(e).__name__}: {e}", "n_files": len(plan[key]),
                             "n_fetched": "", "bytes_read": "", "mode": ""}

        n = 0
        with ThreadPoolExecutor(args.workers) as ex:
            for fut in as_completed([ex.submit(work, k) for k in segs]):
                (d, s), st = fut.result()
                done[(d, s)] = {"chunk": c, "drive": d, "seg": s, **{k: st.get(k, "") for k in fields[3:]}}
                n += 1
                if n % 20 == 0 or n == len(segs):
                    print(f"  {n}/{len(segs)}  経過 {time.time() - t0:.0f} 秒", flush=True)
        results += [done[k] for k in segs]
        out.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(done.values())

    print(f"\n合計: 圧縮 {total_cmp / 1e6:.0f} MB / 展開後 {total_raw / 1e6:.0f} MB")
    if args.dry_run:
        return 0
    bad = [r for r in results if r["status"] != "ok"]
    meta = {"source": CHUNK_URL, "include": args.include, "segments_csv": str(args.segments),
            "n_segments": len(results), "n_error": len(bad), "fetched_at_unix": int(time.time())}
    (out / "fetch_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"失敗 {len(bad)} 件" + (f" (例 {bad[0]['drive']}/{bad[0]['seg']}: {bad[0]['error']})" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
