#!/usr/bin/env python3
"""モード B の判定を元映像に字幕として重ねたレビュー用 MP4 を作る。

**推論にも採点にも影響しない。** 既存の出力 (results_*_mode_b.jsonl と
フレームキャッシュ) をそのまま読んで、目で追える形にするだけ。

    元映像 (framecache, 20 Hz)  +  判定 (0.5 秒ごと)  ->  1 本の MP4

判定が「いつ」変わったかを見るのが目的なので、**因果性はここでも守る**。
時刻 t のフレームに重ねるのは `t_eval <= t` の判定だけで、評価前のフレームには
何も出さない。未来の判定を先に見せると、この動画で確かめたいことが崩れる。

出力は 1 モデル × 1 エピソード につき 1 本。エピソードが comma2k19 の
セグメント境界を跨ぐ場合も 1 本にまとめる (事象が切れるのを避けるため)。
ファイル名にセグメント番号を入れて対応が分かるようにしてある。

使い方:
  uv run python scripts/make_review_video.py \
      --results out/chunk1/vlm/results_qwen2_5_vl_7b_mode_b.jsonl
  uv run python scripts/make_review_video.py --results ... --events P05,P08
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from near_miss.config import load_yaml
from near_miss.vlm import frames as fr
from near_miss.vlm import overlay as ov
from near_miss.vlm.windows import (
    Episode,
    available_segments,
    episodes_from_labels,
    timeline_available,
)

# 字幕に出す項目。(応答のキー, 見出し)
BODY_FIELDS = (
    ("scene", "情景"),
    ("ego_behavior", "自車"),
    ("change_from_previous", "変化"),
    ("expected_next", "予期"),
    ("evidence_detail", "根拠"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True,
                   help="results_<model>_mode_b.jsonl")
    p.add_argument("--labels", type=Path, default=Path("out/chunk1/labels.csv"))
    p.add_argument("--onset", type=Path, default=Path("out/chunk1/vlm/labels_onset.csv"))
    p.add_argument("--cache", type=Path, default=Path("out/chunk1/vlm/framecache"))
    p.add_argument("--data-root", type=Path, default=Path("raw_data/Chunk_1"))
    p.add_argument("--out", type=Path, default=Path("out/chunk1/vlm/review_video"))
    p.add_argument("--config", type=Path, default=Path("configs/vlm.yaml"))
    p.add_argument("--events", default=None, help="対象を絞る (例 P05,P08)")
    p.add_argument("--width", type=int, default=960,
                   help="出力幅。キャッシュは 640 px なので拡大して字幕を読みやすくする")
    p.add_argument("--crf", type=int, default=20)
    p.add_argument("--fields", default=None,
                   help="字幕に出す項目を絞る (既定は "
                        + ",".join(k for k, _ in BODY_FIELDS) + ")")
    return p.parse_args()


def load_judgments(path: Path, fields=BODY_FIELDS
                   ) -> tuple[str, dict[str, list[ov.Judgment]]]:
    """結果 JSONL を、エピソードごとの時系列に直す。"""
    by_event: dict[str, list[ov.Judgment]] = {}
    model = path.stem
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("mode") != "online":
            continue
        model = r.get("model", model)
        resp = r.get("response") or {}
        lines = [(label, str(resp[k])) for k, label in fields
                 if isinstance(resp.get(k), str) and resp[k].strip()]
        by_event.setdefault(r["event_id"], []).append(ov.Judgment(
            t_eval=float(r["t_eval"]),
            state=str(resp.get("state", "unknown")),
            hazard_type=str(resp.get("hazard_type", "-")),
            risk_level=int(resp.get("risk_level", 0) or 0),
            confidence=float(resp.get("confidence", 0.0) or 0.0),
            lines=lines,
        ))
    for v in by_event.values():
        v.sort(key=lambda j: j.t_eval)
    return model, by_event


def header(ep: Episode, j: ov.Judgment | None, t: float, t0: float,
           marks: dict[str, float]) -> str:
    """1 行目。判定と、人手が付けた基準時刻の通過を出す。"""
    s = f"{ep.event_id}  t={t - t0:+.1f}s"
    if j is None:
        s += "  [評価前]"
    else:
        s += (f"  {j.state}/{j.hazard_type} lv{j.risk_level}"
              f" conf{j.confidence:.2f}  (判定 t={j.t_eval - t0:+.1f}s)")
    tags = [name for name, tm in marks.items() if np.isfinite(tm) and t >= tm]
    if ep.t_start <= t <= ep.t_end:
        tags.append("候補区間")
    return s + ("   " + " ".join(f"[{x}]" for x in tags) if tags else "")


def render_one(ep: Episode, js: list[ov.Judgment], times: list[float],
               model: str, cfg: dict, args, onset: pd.DataFrame) -> Path | None:
    """1 エピソード分の MP4 を書く。"""
    from PIL import Image

    fps = float(cfg["video"]["fps"])
    step = 1.0 / fps
    t0, t1 = times[0], times[-1]
    # 判定は 0.5 秒ごとだが、映像は 20 Hz のまま流して動きを見えるようにする。
    grid = [t0 + i * step for i in range(int(round((t1 - t0) / step)) + 1)]

    marks: dict[str, float] = {}
    if ep.event_id in onset.index:
        r = onset.loc[ep.event_id]
        marks["onset"] = float(r.get("t_onset_human", np.nan))
        marks["明確"] = float(r.get("t_apparent_human", np.nan))

    font, ok = ov.load_font(max(13, args.width // 52))
    small, _ = ov.load_font(max(12, args.width // 60))
    if not ok:
        print(f"  {ov.FONT_HINT}")

    out_dir = args.out / model
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{ep.event_id}_{ep.drive_id.split('|')[-1]}_seg{ep.segment}.mp4"
    dst = out_dir / name

    proc = None
    written = 0
    for t in grid:
        p = ep.to_segment(t, cfg)
        src = fr.frame_path(args.cache, ep.drive_id, p.segment, p.frame)
        if not src.is_file():
            continue
        img = Image.open(src).convert("RGB")
        if img.width != args.width:
            h = int(round(img.height * args.width / img.width))
            img = img.resize((args.width, h - h % 2), Image.LANCZOS)
        j = ov.judgment_at(js, t)
        img = ov.draw_caption(img, j, header(ep, j, t, t0, marks), font, small)

        if proc is None:
            proc = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-y",
                 "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{img.width}x{img.height}", "-r", f"{fps:g}", "-i", "-",
                 "-c:v", "libx264", "-preset", "fast", "-crf", str(args.crf),
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
                stdin=subprocess.PIPE)
        proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
        written += 1

    if proc is None:
        print(f"  {ep.event_id}: フレームが 1 枚も無いので飛ばします")
        return None
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg が失敗しました: {dst}")
    print(f"  {dst.name}  {written} フレーム / {written / fps:.1f} 秒 / "
          f"判定 {len(js)} 件")
    return dst


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    if not args.results.is_file():
        raise SystemExit(f"結果がありません: {args.results}")

    fields = BODY_FIELDS
    if args.fields:
        want_f = [x.strip() for x in args.fields.split(",")]
        fields = tuple(f for f in BODY_FIELDS if f[0] in want_f)
        if not fields:
            raise SystemExit(f"未知の項目: {args.fields}  "
                             f"選べるのは {[k for k, _ in BODY_FIELDS]}")
    model, by_event = load_judgments(args.results, fields)
    if not by_event:
        raise SystemExit("モード B の結果が入っていません (--results を確認してください)")

    eps = episodes_from_labels(pd.read_csv(args.labels), cfg)
    onset = (pd.read_csv(args.onset).set_index("event_id")
             if args.onset.is_file() else pd.DataFrame().set_index(pd.Index([])))
    want = {x.strip() for x in args.events.split(",")} if args.events else None

    print(f"モデル: {model} / 判定のあるエピソード {len(by_event)} 件")
    made = 0
    avail_cache: dict[str, set[int]] = {}
    for ep in eps:
        if ep.event_id not in by_event:
            continue
        if want and ep.event_id not in want:
            continue
        avail = avail_cache.setdefault(
            ep.drive_id, available_segments(args.data_root, ep.drive_id))
        times, _ = timeline_available(ep, cfg, avail)
        if not times:
            # 映像が手元に無い場合は、判定のある範囲だけで組む
            js = by_event[ep.event_id]
            times = [js[0].t_eval, js[-1].t_eval]
        if render_one(ep, by_event[ep.event_id], times, model, cfg, args, onset):
            made += 1
    print(f"\n出力 {made} 本 -> {args.out / model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
