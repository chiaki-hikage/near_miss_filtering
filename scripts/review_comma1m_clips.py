#!/usr/bin/env python3
"""候補の前後だけ映像を切り出して確認する (段階 E)。

fcamera.hevc は 1 セグメント 75 MB あるので全部は落とさない。
frame_info.safetensors (50 KB) の索引から、直前の鍵フレームから必要な範囲までの
バイト位置を求め、HTTP Range でそこだけ取る。6 秒で 8〜9 MB 程度。

入力 CSV に必要な列:
  segment_id, t_center   (t_center は localizer と同じ boot time)
任意: label, note        出力ファイル名と一覧に載せる

出力 (--out 配下):
  <label>_<segment_id 先頭 8 桁>.mp4   切り出した映像
  <label>_<segment_id 先頭 8 桁>.png   等間隔のコマ並べ
  clips.csv                            取得結果と転送量

使い方:
  python scripts/review_comma1m_clips.py --targets out/comma1m/review_targets.csv --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd
import requests

from near_miss.io import comma1m

# 実測の平均。1928x1208 の HEVC で 1 フレーム約 62 KB
BYTES_PER_FRAME = 62_000


def build_grid(hevc: Path, png: Path, cols: int, tile_w: int) -> bool:
    """等間隔に抜いたコマを 1 枚に並べる。"""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(hevc)],
        capture_output=True, text=True,
    )
    try:
        n = int(probe.stdout.strip())
    except ValueError:
        return False
    step = max(1, n // (cols * 2))
    vf = f"select='not(mod(n\\,{step}))',scale={tile_w}:-1,tile={cols}x2"
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(hevc), "-vf", vf,
                        "-frames:v", "1", "-y", str(png)], capture_output=True)
    return r.returncode == 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", type=Path, default=Path("out/comma1m/review_targets.csv"))
    p.add_argument("--out", type=Path, default=Path("out/comma1m/clips"))
    p.add_argument("--cache", type=Path, default=comma1m.DEFAULT_CACHE)
    p.add_argument("--pre", type=float, default=3.0, help="中心より前に何秒取るか")
    p.add_argument("--post", type=float, default=3.0)
    p.add_argument("--camera", default="fcamera", choices=("fcamera", "ecamera"))
    p.add_argument("--cols", type=int, default=6, help="コマ並べの列数 (2 段)")
    p.add_argument("--tile-width", type=int, default=320)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tg = pd.read_csv(args.targets)
    if args.limit:
        tg = tg.head(args.limit)
    span = args.pre + args.post
    # 直前の鍵フレームまで戻るぶん (最大 1.5 秒) を上乗せして見積もる
    est_mb = len(tg) * (span + 1.5) * 20.0 * BYTES_PER_FRAME / 1e6
    print(f"対象   : {len(tg)} 件  前後 {args.pre}/{args.post} 秒  カメラ {args.camera}")
    print(f"転送量 : 約 {est_mb:.0f} MB  (全編なら {len(tg) * 75} MB)")
    if args.dry_run:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    rows = []
    # 同じセグメントに複数の候補が乗ることがあるので、行番号を名前に入れて上書きを防ぐ
    for i, (_, r) in enumerate(tg.iterrows()):
        sid = str(r["segment_id"])
        label = str(r.get("label", "clip"))
        stem = f"{i:02d}_{label}_{sid[:8]}"
        hevc = args.out / f"{stem}.hevc"
        try:
            m = comma1m.fetch_clip(s, sid, float(r["t_center"]) - args.pre,
                                   float(r["t_center"]) + args.post, hevc,
                                   camera=args.camera, cache=args.cache)
        except Exception as exc:
            print(f"  失敗 {sid}: {exc}")
            rows.append({"segment_id": sid, "label": label, "error": str(exc)[:120]})
            continue
        mp4 = args.out / f"{stem}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(hevc), "-c", "copy", "-y", str(mp4)],
                       capture_output=True)
        png = args.out / f"{stem}.png"
        ok = build_grid(hevc, png, args.cols, args.tile_width)
        hevc.unlink()
        m.update(label=label, note=r.get("note", ""), grid=str(png) if ok else "",
                 mb=round(m["bytes"] / 1e6, 2), error="")
        rows.append(m)
        print(f"  {stem}  {m['n_frames']} frame  {m['mb']} MB")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "clips.csv", index=False)
    total = df["bytes"].sum() / 1e6 if "bytes" in df else 0
    print(f"\n合計 {total:.0f} MB  ->  {args.out / 'clips.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
