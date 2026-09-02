#!/usr/bin/env python3
"""オンライン評価用の onset 時刻を人手で付けるための様式を作る (Phase 1 段階 0)。

Phase 1 は人手確認済みの 32 件だけを使う。うち risky=True の 8 件について、
「いつ危険が立ち上がったか」を人が付ける。オンライン判定の主指標

    delta_onset = t_alarm - t_onset_human

の基準になる。**CAN 由来の t_start は基準にしない。** t_start は検出器が
閾値を超えた時刻であって、人が危険を認識できる時刻ではないため。

出力 (--out 配下):
  labels_onset.csv              記入用。t_onset_seg_s / t_apparent_seg_s が空欄
  labels_onset.md               確認シート。動画の場所と該当秒が載る
  convert_segments.sh           該当セグメントを mp4 に変換する

**変換した mp4 の再生時刻 = セグメント内の秒**になるようにしてある。
注釈者は再生時刻をそのまま書き写せばよく、オフセットの計算が要らない。
生の video.hevc はタイムスタンプを持たないので、必ず PTS を打ち直すこと
(raw_data/hevc2mpeg.sh と同じ理由)。

使い方:
  uv run python scripts/make_onset_template.py
  uv run python scripts/make_onset_template.py --labels out/chunk1/labels.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from near_miss.config import load_yaml
from near_miss.vlm.windows import (
    available_segments,
    episodes_from_labels,
    timeline_available,
)

DEFAULT_LABELS = Path("out/chunk1/labels.csv")
DEFAULT_DATA_ROOT = Path("raw_data/Chunk_1")
DEFAULT_OUT = Path("out/chunk1/vlm")
DEFAULT_CONFIG = Path("configs/vlm.yaml")

# 同一ドライブでこの秒数以内に隣接する候補は、同じ交通状況の可能性が高い。
# 実測: 2018-07-30--13-44-30 の seg10 と seg11 が 4.7 秒差で並んでいる。
# 独立標本として数えると positive 8 件が水増しになるため、人手で確認する。
ADJACENT_S = 15.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return p.parse_args()


def find_adjacent(eps) -> dict[str, str]:
    """同一ドライブで近接する positive の組を拾う。"""
    pairs: dict[str, str] = {}
    for a in eps:
        for b in eps:
            if a.event_id >= b.event_id or a.drive_id != b.drive_id:
                continue
            gap = max(a.t_start, b.t_start) - min(a.t_end, b.t_end)
            if gap <= ADJACENT_S:
                pairs[a.event_id] = b.event_id
                pairs[b.event_id] = a.event_id
    return pairs


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    if not args.labels.is_file():
        raise SystemExit(f"ラベルがありません: {args.labels}")

    labels = pd.read_csv(args.labels)
    eps = [e for e in episodes_from_labels(labels, cfg) if e.risky]
    if not eps:
        raise SystemExit("risky=True の行がありません")

    adjacent = find_adjacent(eps)
    fps = float(cfg["video"]["fps"])
    rows = []
    needed: list[tuple[str, int]] = []
    avail_cache: dict[str, set[int]] = {}

    for ep in eps:
        avail = avail_cache.setdefault(
            ep.drive_id, available_segments(args.data_root, ep.drive_id))
        full = ep.timeline(cfg)
        tl, tr = timeline_available(ep, cfg, avail)
        if not tl:
            print(f"  {ep.event_id}: 候補の開始が手元にありません。飛ばします")
            continue

        p_lo = ep.to_segment(tl[0], cfg)
        p_hi = ep.to_segment(tl[-1], cfg)
        p_cand = ep.to_segment(ep.t_start, cfg)
        segs = sorted({p_lo.segment, p_cand.segment, p_hi.segment})
        for s in segs:
            needed.append((ep.drive_id, s))
        rows.append({
            "event_id": ep.event_id,
            "drive_id": ep.drive_id,
            "segment": ep.segment,
            "verdict": ep.verdict,
            "event_types": ep.event_types,
            "note": ep.note,
            # --- 参考 (CAN 由来。基準にはしない) ---
            "t_start": round(ep.t_start, 2),
            "t_end": round(ep.t_end, 2),
            "duration_s": round(ep.duration_s, 2),
            "t_in_segment_s": round(ep.t_in_segment_s, 2),
            "cand_start_seg": p_cand.segment,
            "cand_start_seg_s": round(p_cand.t_seg, 2),
            "cand_start_frame": p_cand.frame,
            # --- 評価区間 ---
            "eval_start_seg": p_lo.segment,
            "eval_start_seg_s": round(p_lo.t_seg, 2),
            "eval_end_seg": p_hi.segment,
            "eval_end_seg_s": round(p_hi.t_seg, 2),
            "n_eval_points": len(tl),
            "n_eval_points_full": len(full),
            # 手元にセグメントが無くて切り詰めた分。黙って縮めない。
            "pre_lost_s": tr.pre_lost_s,
            "post_lost_s": tr.post_lost_s,
            "missing_segments": "|".join(str(x) for x in tr.missing_segments),
            "segments_needed": "|".join(str(s) for s in segs),
            # --- ここから人手記入 ---
            "onset_segment": ep.segment,        # 既定値。違えば直す
            "t_onset_seg_s": "",                # 予兆が見て取れる最初の瞬間
            "t_apparent_seg_s": "",             # 明らかに危険と分かる瞬間
            "onset_cue": "",                    # cut_in / lead_brake / signal / ...
            "same_episode_as": adjacent.get(ep.event_id, ""),
            "onset_note": "",
            "annotator": "",
        })

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = args.out / "labels_onset.csv"
    if csv_path.exists():
        raise SystemExit(f"既にあります (記入済みを上書きしません): {csv_path}")
    df.to_csv(csv_path, index=False)

    _write_sheet(args.out / "labels_onset.md", df, cfg, args.data_root)
    have = [(d, sg) for d, sg in sorted(set(needed))
            if (args.data_root / d / str(sg) / "video.hevc").is_file()]
    _write_convert(args.out / "convert_segments.sh", have,
                   args.data_root, args.out / "clips", fps)

    print(f"positive {len(df)} 件 / 評価時刻 {int(df.n_eval_points.sum())} 点"
          f" (欠落なしなら {int(df.n_eval_points_full.sum())} 点)")
    cut = df[df.missing_segments != ""]
    if not cut.empty:
        print(f"  手元にないセグメントのため評価区間を切り詰めた: {len(cut)} 件")
        for _, r in cut.iterrows():
            print(f"    {r.event_id} seg{r.missing_segments} が無い"
                  f" -> 前 {r.pre_lost_s:.1f} 秒 / 後 {r.post_lost_s:.1f} 秒を失う")
        worst = cut[cut.pre_lost_s > 3.0]
        for _, r in worst.iterrows():
            print(f"    ** {r.event_id} は候補開始前が {r.pre_lost_s:.1f} 秒失われており、"
                  "オンセット前の誤警報と検出遅れの測定に使えません")
    pairs = df[df.same_episode_as != ""]
    if not pairs.empty:
        print(f"  同一事象の可能性がある組: {'/'.join(sorted(set(pairs.event_id)))}"
              " -> 人手で確認して same_episode_as を確定してください")
    print(f"\n出力: {csv_path}")
    print(f"      {args.out / 'labels_onset.md'}")
    print(f"      {args.out / 'convert_segments.sh'}  (先にこれを流して mp4 を作る)")
    return 0


def _write_sheet(path: Path, df: pd.DataFrame, cfg: dict, data_root: Path) -> None:
    tl = cfg["timeline"]
    lines = [
        "# オンライン評価用 onset 時刻の付与シート (Phase 1)",
        "",
        f"対象: 人手確認済み risky=True の {len(df)} 件。",
        "",
        "## 記入するもの",
        "",
        "`out/chunk1/vlm/labels_onset.csv` の次の列を埋めてください。",
        "",
        "| 列 | 内容 |",
        "|---|---|",
        "| `t_onset_seg_s` | **予兆が見て取れる最初の瞬間**。まだ危険と断定できなくてよい |",
        "| `t_apparent_seg_s` | **明らかに危険と分かる瞬間** |",
        "| `onset_cue` | 根拠 (`cut_in` / `lead_brake` / `signal` / `pedestrian` / `other`) |",
        "| `onset_segment` | 上の 2 つがどのセグメント内の秒か。既定は候補と同じセグメント |",
        "| `same_episode_as` | 別の候補と同一事象なら相手の `event_id`。無ければ空 |",
        "| `annotator` | 記入者 |",
        "",
        "## 手順",
        "",
        "```bash",
        "sh out/chunk1/vlm/convert_segments.sh      # 該当セグメントを mp4 に変換",
        "```",
        "",
        "**変換後の mp4 は再生時刻がセグメント内の秒と一致します。**",
        "プレイヤーの表示時刻をそのまま書き写してください。オフセットの計算は要りません。",
        "",
        "生の `video.hevc` はタイムスタンプを持たないため、直接シークすると位置がずれます。",
        "必ず変換したものを見てください。",
        "",
        f"評価区間は候補の {tl['pre_s']} 秒前から {tl['post_s']} 秒後までです。",
        "この範囲の外に onset がある場合は `onset_note` に書いてください",
        "(評価区間の設定を見直す材料にします)。",
        "",
        "---",
        "",
    ]
    for _, r in df.iterrows():
        lines += [
            f"## {r.event_id}　{r.verdict}",
            "",
            f"- ドライブ: `{r.drive_id}`　セグメント **{r.segment}**",
            f"- 人手の所見: {r.note if str(r.note) not in ('', 'nan', '—') else '(記載なし)'}",
            f"- フィルタの検出: `{r.event_types}`",
            f"- 動画: `clips/{_clip_name(r.drive_id, int(r.segment))}`",
            f"- **候補の開始**: セグメント {r.cand_start_seg} の "
            f"**{r.cand_start_seg_s:.2f} 秒**（フレーム {r.cand_start_frame}）",
            f"- 候補の長さ: {r.duration_s:.2f} 秒",
            f"- 評価区間: seg{r.eval_start_seg} {r.eval_start_seg_s:.2f} 秒 "
            f"〜 seg{r.eval_end_seg} {r.eval_end_seg_s:.2f} 秒"
            f"（{r.n_eval_points} 時刻）",
        ]
        if str(r.missing_segments):
            lines.append(f"- **注意: セグメント {r.missing_segments} が手元に無いため、"
                         f"評価区間を前 {r.pre_lost_s:.1f} 秒 / 後 {r.post_lost_s:.1f} 秒 切り詰めています**")
        if str(r.same_episode_as):
            lines.append(f"- **{r.same_episode_as} と同一事象の可能性があります。**"
                         " 確認して `same_episode_as` を確定してください")
        if len(str(r.segments_needed).split("|")) > 1:
            lines.append(f"- 評価区間が複数セグメントに跨ります: {r.segments_needed}")
        lines += ["", "| t_onset_seg_s | t_apparent_seg_s | onset_cue |", "|---|---|---|",
                  "|  |  |  |", ""]
    lines += ["---", "",
              f"生成: {datetime.now(timezone.utc).isoformat()}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def _clip_name(drive_id: str, segment: int) -> str:
    return f"{drive_id.split('|')[-1]}_seg{segment}.mp4"


def _write_convert(path: Path, needed, data_root: Path, out_dir: Path, fps: float) -> None:
    lines = [
        "#!/bin/sh",
        "# onset 付与に使うセグメントを mp4 に変換する。",
        "#",
        "# video.hevc は生の Annex-B ストリームでタイムスタンプを持たない",
        "# (ffprobe が幅も高さも返さない)。setpts で 20 Hz を打ち直さないと",
        "# 再生時刻がセグメント内の秒と一致せず、注釈した時刻がずれる。",
        "set -e",
        f'OUT="{out_dir}"',
        'mkdir -p "$OUT"',
        "",
    ]
    for drive_id, seg in needed:
        src = data_root / drive_id / str(seg) / "video.hevc"
        dst = f"$OUT/{_clip_name(drive_id, seg)}"
        lines += [
            f'if [ -f "{src}" ]; then',
            f'  echo "-> {dst}"',
            f'  ffmpeg -v error -y -framerate {fps:g} -i "{src}" \\',
            f'    -vf "setpts=N/({fps:g}*TB)" -c:v libx264 -preset fast -crf 18 \\',
            f'    -movflags +faststart "{dst}"',
            "else",
            f'  echo "見つかりません: {src}" >&2',
            "fi",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


if __name__ == "__main__":
    raise SystemExit(main())
