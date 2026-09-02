#!/usr/bin/env python3
"""記入済みの onset シート (Markdown) を取り込んで検証する (Phase 1 段階 0)。

labels_onset.md の各イベントの表に書かれた

    | t_onset_seg_s | t_apparent_seg_s | onset_cue |

を読み、labels_onset.csv へ書き戻す。あわせて boot time に変換し、
**評価区間の中に収まっているか**を検査する。

収まっていない場合は黙って切り詰めない。オンセットが評価区間の外にあるなら、
それは「検出できなかった」のではなく「測れない」であって、両者を混同すると
検出遅れの解釈を誤る。

使い方:
  uv run python scripts/import_onset_sheet.py
  uv run python scripts/import_onset_sheet.py --sheet out/chunk1/vlm/labels_onset.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from near_miss.config import load_yaml
from near_miss.vlm.schema import HAZARD_TYPES
from near_miss.vlm.windows import (
    Episode,
    available_segments,
    from_segment,
    timeline_available,
)

DEFAULT_DIR = Path("out/chunk1/vlm")
DEFAULT_CONFIG = Path("configs/vlm.yaml")
DEFAULT_DATA_ROOT = Path("raw_data/Chunk_1")

HEAD_RE = re.compile(r"^##\s+(P\d+)")
ROW_RE = re.compile(r"^\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sheet", type=Path, default=DEFAULT_DIR / "labels_onset.md")
    p.add_argument("--csv", type=Path, default=DEFAULT_DIR / "labels_onset.csv")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--dry-run", action="store_true", help="検証だけして書き戻さない")
    return p.parse_args()


def parse_sheet(path: Path) -> dict[str, dict[str, str]]:
    """見出しごとに、記入行を 1 つだけ拾う。

    ヘッダ行と区切り行は読み飛ばす。空欄のままの表は結果に含めない。
    """
    out: dict[str, dict[str, str]] = {}
    event = None
    in_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEAD_RE.match(line)
        if m:
            event, in_table = m.group(1), False
            continue
        if line.startswith("| t_onset_seg_s"):
            in_table = True
            continue
        if not in_table or event is None:
            continue
        if line.startswith("|---"):
            continue
        m = ROW_RE.match(line)
        if not m:
            in_table = False
            continue
        onset, apparent, cue = (x.strip() for x in m.groups())
        if onset or apparent or cue:
            out.setdefault(event, {"t_onset_seg_s": onset,
                                   "t_apparent_seg_s": apparent,
                                   "onset_cue": cue})
        in_table = False
    return out


def _num(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    if not args.sheet.is_file():
        raise SystemExit(f"シートがありません: {args.sheet}")
    if not args.csv.is_file():
        raise SystemExit(f"CSV がありません: {args.csv}")

    filled = parse_sheet(args.sheet)
    df = pd.read_csv(args.csv)
    print(f"シートの記入: {len(filled)} 件 / CSV の行: {len(df)} 件\n")

    problems: list[str] = []
    rows = []
    avail_cache: dict[str, set[int]] = {}

    for _, r in df.iterrows():
        eid = str(r["event_id"])
        rec = dict(r)
        f = filled.get(eid)
        if not f:
            problems.append(f"{eid}: シートが空欄")
            rows.append(rec)
            continue

        ep = Episode(
            event_id=eid, drive_id=str(r["drive_id"]), segment=int(r["segment"]),
            t_start=float(r["t_start"]), t_end=float(r["t_end"]),
            t_in_segment_s=float(r["t_in_segment_s"]), risky=True,
        )
        seg = int(r["onset_segment"])
        onset = _num(f["t_onset_seg_s"])
        apparent = _num(f["t_apparent_seg_s"])
        cue = f["onset_cue"]

        rec["t_onset_seg_s"] = onset
        rec["t_apparent_seg_s"] = apparent
        rec["onset_cue"] = cue

        if onset is None or apparent is None:
            problems.append(f"{eid}: 数値として読めない ({f})")
            rows.append(rec)
            continue
        if cue not in HAZARD_TYPES:
            problems.append(f"{eid}: onset_cue が語彙にない: {cue!r}")
        if onset > apparent:
            problems.append(f"{eid}: onset {onset} が apparent {apparent} より後")

        # boot time へ。判定側はすべて boot time で行う。
        t_onset = from_segment(seg, onset, ep, cfg)
        t_apparent = from_segment(seg, apparent, ep, cfg)
        rec["t_onset_human"] = round(t_onset, 3)
        rec["t_apparent_human"] = round(t_apparent, 3)
        # 参考: CAN の検出が人手のオンセットからどれだけ遅れたか。
        # 正なら人間のほうが先に気づいている。
        rec["filter_lag_s"] = round(ep.t_start - t_onset, 2)

        avail = avail_cache.setdefault(
            ep.drive_id, available_segments(args.data_root, ep.drive_id))
        tl, tr = timeline_available(ep, cfg, avail)
        lo, hi = (tl[0], tl[-1]) if tl else (float("nan"), float("nan"))
        rec["onset_in_window"] = bool(tl) and lo <= t_onset <= hi
        rec["apparent_in_window"] = bool(tl) and lo <= t_apparent <= hi
        # オンセットが評価区間の手前にある場合、検出遅れには下限が生じる。
        # 「測れない」ではなく「この値より小さくは出ない」なので、床として残す。
        # 採点側はこの床を引いて解釈する。
        floor = max(0.0, lo - t_onset) if tl else float("nan")
        rec["latency_floor_s"] = round(floor, 3)
        stride = float(cfg["timeline"]["stride_s"])
        if not rec["onset_in_window"]:
            if floor <= stride:
                problems.append(
                    f"{eid}: オンセットが評価区間の {floor:.2f} 秒手前。"
                    f"検出遅れは +{floor:.2f} 秒より小さく出ません (評価ストライド"
                    f" {stride} 秒未満なので実用上は測れます)")
            else:
                problems.append(
                    f"{eid}: オンセットが評価区間の {floor:.2f} 秒手前。"
                    "**検出遅れを測れません**")
        rows.append(rec)

    out = pd.DataFrame(rows)
    _report(out)

    if problems:
        print("\n要確認:")
        for p in problems:
            print(f"  - {p}")

    if args.dry_run:
        print("\n--dry-run のため書き戻していません。")
        return 0
    out.to_csv(args.csv, index=False)
    print(f"\n書き戻し: {args.csv}")
    return 0


def _report(df: pd.DataFrame) -> None:
    cols = ["event_id", "segment", "cand_start_seg_s", "t_onset_seg_s",
            "t_apparent_seg_s", "onset_cue", "filter_lag_s",
            "onset_in_window", "latency_floor_s"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))

    if "filter_lag_s" in df:
        lead = df[df.filter_lag_s > 0]
        print(f"\n人手のオンセットが CAN の検出より先だったもの: {len(lead)}/{len(df)} 件")
        for _, r in lead.sort_values("filter_lag_s", ascending=False).iterrows():
            print(f"  {r.event_id}  {r.filter_lag_s:+.2f} 秒先行  ({r.onset_cue})")
    if "onset_cue" in df:
        print("\nonset_cue の内訳:")
        print(df.onset_cue.value_counts().to_string())


if __name__ == "__main__":
    raise SystemExit(main())
