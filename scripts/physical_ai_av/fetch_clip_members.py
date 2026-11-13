#!/usr/bin/env python3
"""NVIDIA PhysicalAI-AV のチャンク zip から、指定したクリップのファイルだけを HTTP の範囲指定で取り出す。

チャンク zip (前方カメラなら 1 個 約 2 GB) を丸ごとは落とさない。zip の目次と、要るファイルの範囲だけを読む。

  uv run python scripts/physical_ai_av/fetch_clip_members.py --feature camera_front_wide_120fov --chunk 0 \
      --clips db2c5fbe 5e9a00f1 --max-mb 200 --dry-run

認証: Hugging Face に保存した名前付きトークン (既定 physical_ai_av) を使う。トークンの文字列は表示・保存しない。
出力: data/physical_ai_av/<directory>/<feature>/chunk_<NNNN>/<zip の中のファイル名> と、取得の記録 fetch_log.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import zipfile
from pathlib import Path

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
REVISION = "2ac2e9c056e0c8426face5bfad64b0961a295b28"     # 2026-11-06 版。docs/physical_ai_av.md
ROOT = Path("data/physical_ai_av")


def token(name: str) -> str:
    from huggingface_hub.utils import get_stored_tokens
    t = get_stored_tokens().get(name)
    if not t:
        sys.exit(f"保存されたトークン '{name}' が無い (huggingface-cli auth login で名前を付けて保存する)")
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature", required=True, help="features.csv の feature (例 camera_front_wide_120fov)")
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--clips", nargs="+", required=True, help="clip_id か、その先頭の文字列")
    ap.add_argument("--max-mb", type=float, default=200.0, help="この量を超えるなら取得しない")
    ap.add_argument("--token-name", default="physical_ai_av")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfFileSystem
    feats = {r["feature"]: r for r in csv.DictReader((ROOT / "features.csv").open(encoding="utf-8"))}
    f = feats[args.feature]
    rel = f["chunk_path"].format(chunk_id=args.chunk)
    fs = HfFileSystem(token=token(args.token_name))
    path = f"datasets/{REPO_ID}@{REVISION}/{rel}"
    out_dir = ROOT / f["directory"] / args.feature / f"chunk_{args.chunk:04d}"
    with fs.open(path, "rb", block_size=1 << 20) as fh:
        z = zipfile.ZipFile(fh)
        members = [i for i in z.infolist() if any(i.filename.startswith(c) for c in args.clips)]
        total = sum(i.compress_size for i in members) / 1e6
        print(f"{rel}: {len(members)} ファイル、{total:.1f} MB (上限 {args.max_mb} MB)")
        for i in members:
            print(f"  {i.filename}  {i.file_size / 1e6:.2f} MB")
        if total > args.max_mb:
            print("上限を超えるので取得しない"); return 1
        if args.dry_run:
            return 0
        out_dir.mkdir(parents=True, exist_ok=True)
        log = out_dir / "fetch_log.csv"
        new = not log.exists()
        with log.open("a", encoding="utf-8", newline="") as lf:
            w = csv.writer(lf)
            if new:
                w.writerow(["fetched_at", "revision", "zip", "member", "bytes", "crc32"])
            for i in members:
                dst = out_dir / i.filename
                if dst.exists() and dst.stat().st_size == i.file_size:
                    print("  既にある:", dst.name); continue
                data = z.read(i)                      # zipfile が CRC を照合する
                dst.write_bytes(data)
                w.writerow([dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), REVISION, rel, i.filename, i.file_size, f"{i.CRC:08x}"])
    print("->", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
