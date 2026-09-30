#!/usr/bin/env python3
"""人手ラベルの区間ごとに、候補と同じ集計値を作り直す。

ラベルを付けたときと設定が変わっており、`candidates.csv` と時刻で突き合わせても
一部しか一致しない (実測 17/32)。**ラベルの区間そのものを実データから引き直す。**

既存の to_grid / compute_features / _CONTEXT をそのまま使うので、
候補側の列名と定義が一致する。

  uv run python scripts/detect/label_span_features.py \
      --labels out/chunk1/labels.csv --data-root raw_data/Chunk_1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from near_miss.config import (
    DEFAULT_DETECTION,
    DEFAULT_VEHICLE_DIR,
    find_vehicle_config,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.detectors import detect_all
from near_miss.features import compute_features
from near_miss.io import comma2k19
from near_miss.io.canonical import concat_segments
from near_miss.pipeline import split_contiguous
from near_miss.scoring import _context_for, event_severity
from near_miss.signals import to_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, default=Path("out/chunk1/labels.csv"))
    p.add_argument("--data-root", type=Path, default=Path("raw_data/Chunk_1"))
    p.add_argument("--detection", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=Path("out/chunk1/label_features.csv"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.detection)
    vehicles = load_vehicle_configs(args.vehicles)
    lab = pd.read_csv(args.labels)
    refs = comma2k19.find_segments(args.data_root)
    by_drive: dict[str, list] = {}
    for r in refs:
        by_drive.setdefault(r.drive_id, []).append(r)

    rows = []
    for drive_id, g in lab.groupby("drive_id"):
        drefs = by_drive.get(drive_id)
        if not drefs:
            print(f"  セグメントが無い: {drive_id}")
            continue
        vehicle = find_vehicle_config(drefs[0].dongle_id, vehicles)
        for block in split_contiguous(sorted(drefs, key=lambda r: r.index)):
            segs = []
            for ref in block:
                try:
                    segs.append(comma2k19.load_segment(ref, vehicle, with_raw_can=True))
                except Exception as exc:
                    print(f"  読み出し失敗 {ref.segment_id}: {exc}")
            if not segs:
                continue
            merged = concat_segments(segs)
            gs = to_grid(merged, cfg)
            if gs.df.empty:
                continue
            gs = compute_features(gs, cfg, radar=merged.radar, vehicle=vehicle)
            events = detect_all(gs, cfg, max_stage=2)
            t0, t1 = float(gs.t[0]), float(gs.t[-1])
            for _, L in g.iterrows():
                # **重なりで判定し、はみ出しはグリッドに詰める。**
                # edge_trim のぶん区間の端がグリッドの外に出ることがある
                # (実測 0.08 秒)。開始時刻が中に入っていることを求めると落ちる。
                if not (L.t_end >= t0 and L.t_start <= t1):
                    continue
                a, b = max(float(L.t_start), t0), min(float(L.t_end), t1)
                ctx = _context_for(gs, a, b)
                # 区間に重なるイベントを数える。候補側の n_events / event_types と同じ定義。
                ov = [e for e in events
                      if e.t_end >= L.t_start and e.t_start <= L.t_end]
                types = sorted({e.event_type for e in ov})
                # severity は build_candidates と同じ式で出す。
                #   種別ごとに最も強い 1 件だけを足し、共起ボーナスを加える。
                max_sev = float(cfg["scoring"]["max_severity_per_event"])
                by_type: dict[str, float] = {}
                for e in ov:
                    by_type[e.event_type] = max(by_type.get(e.event_type, 0.0),
                                                event_severity(e, max_sev))
                sev = sum(by_type.values()) + float(
                    cfg["scoring"]["cooccurrence_bonus"]) * max(len(by_type) - 1, 0)
                rows.append({
                    "drive_id": drive_id, "segment": L.segment,
                    "t_start": L.t_start, "t_end": L.t_end,
                    "duration_s": float(L.t_end - L.t_start),
                    "risky": bool(L.risky), "verdict": L.verdict, "note": L.note,
                    "n_events": len(ov), "n_types": len(types),
                    "event_types": "|".join(types),
                    "severity": float(sev),
                    **ctx,
                })
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)}/{len(lab)} 件  risky {int(df.risky.sum())}")
    miss = len(lab) - len(df)
    if miss:
        print(f"  ** {miss} 件は区間が見つかりませんでした")
    print(f"出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
