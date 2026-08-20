#!/usr/bin/env python3
"""候補区間を動画確認できる形に並べ直す。

candidates.csv の時刻はブロック先頭からの相対値なので、そのままでは
映像の頭出しができない。映像はセグメント単位のファイル (20 Hz) なので、
セグメント内の相対秒とフレーム番号に直したうえで、ファイルの場所を付ける。

使い方:
  python scripts/make_review_list.py out/chunk1 raw_data/Chunk_1 [--top 30]

出力:
  <結果ディレクトリ>/review.csv  全候補。映像とプレビューの場所つき
  <結果ディレクトリ>/review.md   上位 N 件の確認シート。判定を書き込んで戻す
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

VIDEO_FPS = 20.0

# 確認シートに出す文脈量。(列名, 表示名, 小数桁)
SHEET_COLUMNS = (
    ("v_mps_mean", "平均車速 km/h", 0),
    ("ttc_s_min", "最小TTC s", 2),
    ("thw_s_min", "最小車間時間 s", 2),
    ("ax_mps2_min", "最小縦加速度 m/s2", 2),
    ("ay_kin_mps2_absmax", "最大横加速度 m/s2", 2),
    ("lat_jerk_mps3_absmax", "最大横躍度 m/s3", 2),
    ("lead_distance_m_min", "最小車間距離 m", 1),
)

LABELS = {
    "hard_brake": "急ブレーキ",
    "hard_accel": "急加速",
    "brake_jerk": "制動ジャーク",
    "hard_steer": "急操舵",
    "lateral_accel": "横加速度過大",
    "weaving": "蛇行",
    "short_thw": "車間時間過小",
    "low_ttc": "衝突余裕時間過小",
    "abs_active": "ABS作動",
}


def build_review(candidates: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    df = candidates.copy()
    df.insert(0, "rank", range(1, len(df) + 1))

    seg_dir = df.apply(lambda r: data_root / r["drive_id"] / str(int(r["segment"])), axis=1)
    df["segment_dir"] = [str(p) for p in seg_dir]
    df["video"] = [str(p / "video.hevc") if (p / "video.hevc").exists() else "" for p in seg_dir]
    df["preview"] = [str(p / "preview.png") if (p / "preview.png").exists() else "" for p in seg_dir]

    df["t_in_segment_end_s"] = (df["t_in_segment_s"] + df["duration_s"]).round(2)
    # 候補がセグメント境界を跨いでいると、終端は次のセグメントに入る
    df["crosses_segment"] = df["t_in_segment_end_s"] > 60.0
    df["v_kmh_mean"] = (df["v_mps_mean"] * 3.6).round(0)
    df["event_labels"] = df["event_types"].fillna("").apply(
        lambda s: "、".join(LABELS.get(x, x) for x in s.split("|") if x)
    )
    df["判定"] = ""      # ヒヤリハット / 通常運転 / 判断不能 を記入する
    df["備考"] = ""
    return df


def write_sheet(df: pd.DataFrame, path: Path, top: int) -> None:
    lines = [
        "# ヒヤリハット候補 確認シート",
        "",
        f"上位 {min(top, len(df))} 件 / 全 {len(df)} 件。severity の降順。",
        "",
        "`判定` に **ヒヤリハット / 通常運転 / 判断不能** を記入してください。",
        "この判定を閾値の較正に使います。severity は順位付けの目安であって",
        "危険度の絶対尺度ではないため、スコアが低い側にも確認対象は残っています。",
        "",
        "映像は 20 Hz。`ffplay -ss <開始秒> <video>` などで頭出しできます。",
        "",
    ]
    for _, r in df.head(top).iterrows():
        lines.append(f"## {int(r['rank'])}. {r['event_labels']}　(severity {r['severity']:.2f})")
        lines.append("")
        lines.append(f"- ドライブ: `{r['drive_id']}`  セグメント **{int(r['segment'])}**")
        span = f"{r['t_in_segment_s']:.2f} 〜 {r['t_in_segment_end_s']:.2f} 秒"
        if r["crosses_segment"]:
            span += "（**次のセグメントに跨ります**）"
        lines.append(f"- 位置: セグメント内 {span}　フレーム {int(r['video_frame'])} 付近")
        if r["video"]:
            lines.append(f"- 映像: `{r['video']}`")
        detail = []
        for col, name, digits in SHEET_COLUMNS:
            if col not in r or pd.isna(r[col]):
                continue
            val = r["v_kmh_mean"] if col == "v_mps_mean" else r[col]
            detail.append(f"{name} {val:.{digits}f}")
        lines.append(f"- 計測値: {' / '.join(detail)}")
        if r["op_tx_mean"] == 1.0:
            lines.append("- openpilot が制御フレームを送信中（作動しているかは判別できていません）")
        lines.append("")
        lines.append("| 判定 | 備考 |")
        lines.append("|---|---|")
        lines.append("|  |  |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path, help="run_detection.py の出力ディレクトリ")
    p.add_argument("data_root", type=Path, help="セグメントを含むディレクトリ (例 raw_data/Chunk_1)")
    p.add_argument("--top", type=int, default=30, help="確認シートに載せる件数")
    p.add_argument("--name", type=str, default="review",
                   help="出力ファイル名の先頭。既に判定を書き込んだシートを上書きしないために使う")
    p.add_argument("--exclude-labeled", type=Path, default=None,
                   help="このラベル CSV に載っている候補を除く")
    args = p.parse_args()

    cand_path = args.results / "candidates.csv"
    if not cand_path.exists():
        print(f"候補ファイルがありません: {cand_path}")
        return 1
    candidates = pd.read_csv(cand_path)
    if candidates.empty:
        print("候補が 0 件です。")
        return 0

    review = build_review(candidates, args.data_root)
    if args.exclude_labeled and args.exclude_labeled.exists():
        labeled = pd.read_csv(args.exclude_labeled)
        keep = []
        for _, r in review.iterrows():
            hit = (
                (labeled["drive_id"] == r["drive_id"])
                & (labeled["segment"] == r["segment"])
                & ~((labeled["t_end"] < r["t_start"]) | (labeled["t_start"] > r["t_end"]))
            )
            keep.append(not hit.any())
        review = review[keep].reset_index(drop=True)
        print(f"  判定済みを除外: 残り {len(review)} 件")

    out_csv = args.results / f"{args.name}.csv"
    out_md = args.results / f"{args.name}.md"
    for path in (out_csv, out_md):
        if path.exists() and args.name == "review":
            print(f"  {path} は既にあります。--name で別名を指定してください")
            return 1
    review.to_csv(out_csv, index=False)
    write_sheet(review, out_md, args.top)

    missing = int((review["video"] == "").sum())
    print(f"  {out_csv}  {len(review)} 行")
    print(f"  {out_md}   上位 {min(args.top, len(review))} 件")
    if missing:
        print(f"  注意: 映像が見つからない候補が {missing} 件あります")
    crossing = int(review["crosses_segment"].sum())
    if crossing:
        print(f"  注意: セグメント境界を跨ぐ候補が {crossing} 件あります")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
