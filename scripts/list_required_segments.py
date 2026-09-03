#!/usr/bin/env python3
"""VLM 後段の入力生成に要る comma2k19 セグメントを列挙する。

Chunk_1 は 188 セグメント 9.0 GB あるが、人手確認済み 32 件の評価に要るのは
**37 本だけ**。しかも 1 セグメント 49 MB のうち必要なのは

    video.hevc      36 MB   フレームキャッシュ
    processed_log/   6.3 MB CAN の 20 Hz グリッド

の 2 つで、raw_log.bz2 / global_pose / preview.png は要らない。
全部運ぶと 9.0 GB、必要なぶんだけなら 1.6 GB。

映像が手元に無くても動く (labels.csv だけで所要が決まる)。EC2 で
「何を持ってくればよいか」を先に出すために使える。

出力 (--out 配下):
  required_segments.csv   セグメント一覧。手元にあるかも記録
  required_files.txt      相対パスの一覧。rsync --files-from にそのまま渡せる

使い方:
  uv run python scripts/list_required_segments.py
  uv run python scripts/list_required_segments.py --data-root raw_data/Chunk_1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from near_miss.config import load_yaml
from near_miss.vlm import frames as fr
from near_miss.vlm.windows import episodes_from_labels

# 1 セグメントのうち転送するもの。raw_log.bz2 / global_pose / preview.png は
# 読まないので運ばない。
NEEDED = ("video.hevc", "processed_log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, default=Path("out/chunk1/labels.csv"))
    p.add_argument("--data-root", type=Path, default=Path("raw_data/Chunk_1"))
    p.add_argument("--out", type=Path, default=Path("out/chunk1/vlm"))
    p.add_argument("--config", type=Path, default=Path("configs/vlm.yaml"))
    return p.parse_args()


def diagnose(root: Path) -> None:
    """データ配置を点検する。

    「映像が無いので飛ばします」だけでは、置き場が違うのか・映像だけ無いのか・
    ドライブ名が化けているのかを切り分けられない。ここで実際に見えているものを出す。

    ドライブ名には '|' が入る (b0c9d2329ad1606b|2018-07-30--13-03-07)。
    転送やアーカイブの経路によっては、この文字が落ちたり置き換わったりする。
    """
    print(f"データ置き場: {root}")
    if not root.is_dir():
        print("  ** ディレクトリがありません。--data-root を確認してください")
        return
    drives = sorted(d for d in root.iterdir() if d.is_dir())
    segs = [s for d in drives for s in d.iterdir() if s.is_dir() and s.name.isdigit()]
    vid = [s for s in segs if (s / "video.hevc").is_file()]
    log = [s for s in segs if (s / "processed_log").is_dir()]
    print(f"  ドライブ {len(drives)} / セグメント {len(segs)} "
          f"/ video.hevc {len(vid)} / processed_log {len(log)}")
    if drives:
        print(f"  ドライブ名の例: {drives[0].name}")
        if "|" not in drives[0].name:
            print("  ** ドライブ名に '|' がありません。転送の途中で文字が失われた"
                  "可能性があります (labels.csv 側は '|' を含みます)")
    if segs and not vid:
        print("  ** セグメントはあるが video.hevc が 1 つもありません。"
              "映像だけ転送されていない可能性があります")
    if not segs and drives:
        print("  ** ドライブ直下に数字のセグメントディレクトリがありません。"
              "階層が 1 段ずれていないか確認してください "
              "(raw_data/Chunk_1/<drive_id>/<番号>/video.hevc)")
    print()


def _size_mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1e6
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
    return 0.0


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    if not args.labels.is_file():
        raise SystemExit(f"ラベルがありません: {args.labels}")

    diagnose(args.data_root)
    eps = episodes_from_labels(pd.read_csv(args.labels), cfg)
    # 手元にあるかに依らず、**評価に要る全セグメント**を出す。
    # 映像窓が 4 秒遡るぶん、評価区間の先頭よりさらに前が要ることがある。
    want: dict[tuple[str, int], set[str]] = {}
    for ep in eps:
        for seg in fr.needed_segments(ep, cfg, ep.timeline(cfg)):
            want.setdefault((ep.drive_id, seg), set()).add(ep.event_id)

    rows = []
    for (drive_id, seg), evs in sorted(want.items()):
        d = args.data_root / drive_id / str(seg)
        have = {n: (d / n).exists() for n in NEEDED}
        rows.append({
            "drive_id": drive_id, "segment": seg,
            "events": "|".join(sorted(evs)),
            "video_hevc": have["video.hevc"],
            "processed_log": have["processed_log"],
            "mb": round(sum(_size_mb(d / n) for n in NEEDED), 1),
        })
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "required_segments.csv", index=False)

    # rsync --files-from / tar -T に渡せる相対パス一覧
    lines = []
    for _, r in df.iterrows():
        base = f"{r.drive_id}/{int(r.segment)}"
        lines.append(f"{base}/video.hevc")
        lines.append(f"{base}/processed_log")
    (args.out / "required_files.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    both = df[df.video_hevc & df.processed_log]
    lack = df[~(df.video_hevc & df.processed_log)]
    total = df.mb.sum()
    print(f"要るセグメント: {len(df)} 本 / エピソード {len(eps)} 件")
    print(f"  手元に揃っている : {len(both)} 本  ({both.mb.sum() / 1000:.2f} GB)")
    if len(lack):
        print(f"  欠けている       : {len(lack)} 本")
        for _, r in lack.iterrows():
            miss = [n for n, k in (("video.hevc", r.video_hevc),
                                   ("processed_log", r.processed_log)) if not k]
            print(f"    {r.drive_id.split('|')[-1]} seg{int(r.segment)}"
                  f"  欠: {','.join(miss)}  (要 {r.events})")
    if total:
        print(f"\n  転送量の見込み: {total / 1000:.2f} GB"
              "  ※ raw_log.bz2 / global_pose / preview.png は含めない")

    print(f"\n出力: {args.out / 'required_segments.csv'}")
    print(f"      {args.out / 'required_files.txt'}")
    lack_all = df[~(df.video_hevc | df.processed_log)]
    if len(lack_all) == len(df):
        print(f"""
EC2 側での用意:

  # comma2k19 は S3 から取り込む (チャンク丸ごと 約 9.7 GB)
  uv run python scripts/fetch_from_s3.py comma2k19 --chunk Chunk_1 --dry-run
  uv run python scripts/fetch_from_s3.py comma2k19 --chunk Chunk_1

  # 人手ラベルは git で共有する (再生成できないため)
  #   out/chunk1/labels.csv
  #   out/chunk1/vlm/labels_onset.csv
  # フレームキャッシュとリクエストは EC2 側で作り直す:
  uv run python scripts/build_vlm_inputs.py --hwaccel
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
