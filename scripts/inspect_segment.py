#!/usr/bin/env python3
"""1 セグメントの中間結果を点検する。

閾値を決めるための道具。各イベント定義について「その特徴量が実際にどこまで
振れたか」と「閾値までどれだけ余裕があったか」を出す。候補が 0 件のときに、
閾値が厳しすぎるのか、そもそもその挙動が無かったのかを切り分けられる。

使い方:
  python scripts/inspect_segment.py <セグメントまたはその親ディレクトリ>
      [--index N] [--dump-csv out/timeseries.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from near_miss.config import DEFAULT_DETECTION, DEFAULT_VEHICLE_DIR, find_vehicle_config, load_vehicle_configs, load_yaml
from near_miss.detectors import detect_all
from near_miss.features import compute_features, feature_summary
from near_miss.io.comma2k19 import find_segments, load_segment
from near_miss.scoring import build_candidates
from near_miss.signals import to_grid


def threshold_margins(gs, cfg) -> pd.DataFrame:
    """各イベント定義について、特徴量の到達値と閾値までの余裕を出す。"""
    rows = []
    for name, spec in cfg["events"].items():
        feature, op, thr = spec["feature"], spec["op"], float(spec["threshold"])
        if feature not in gs.df.columns:
            rows.append({"event": name, "feature": feature, "op": op, "threshold": thr,
                         "reached": np.nan, "margin": np.nan, "coverage": 0.0, "note": "列なし"})
            continue
        x = gs.df[feature].to_numpy()
        finite = np.isfinite(x)
        if not finite.any():
            rows.append({"event": name, "feature": feature, "op": op, "threshold": thr,
                         "reached": np.nan, "margin": np.nan, "coverage": 0.0, "note": "全欠測"})
            continue
        if op == "lt":
            reached, margin = float(np.nanmin(x)), float(np.nanmin(x) - thr)
        elif op == "gt":
            reached, margin = float(np.nanmax(x)), float(thr - np.nanmax(x))
        elif op == "abs_gt":
            reached, margin = float(np.nanmax(np.abs(x))), float(thr - np.nanmax(np.abs(x)))
        else:  # rising
            reached, margin = float(np.nanmax(x)), float(thr - np.nanmax(x))
        rows.append({"event": name, "feature": feature, "op": op, "threshold": thr,
                     "reached": reached, "margin": margin,
                     "coverage": float(finite.mean()),
                     "note": "発火" if margin < 0 else ""})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path)
    p.add_argument("--index", type=int, default=None, help="セグメント番号を指定する")
    p.add_argument("--config", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--dump-csv", type=Path, default=None, help="グリッド化した時系列を書き出す")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    refs = find_segments(args.path)
    if args.index is not None:
        refs = [r for r in refs if r.index == args.index]
    if not refs:
        print(f"セグメントが見つかりません: {args.path}")
        return 1
    ref = refs[0]

    vehicle = find_vehicle_config(ref.dongle_id, load_vehicle_configs(args.vehicles))
    seg = load_segment(ref, vehicle, with_raw_can=vehicle is not None)
    gs = compute_features(to_grid(seg, cfg), cfg, radar=seg.radar, vehicle=vehicle)

    print(f"セグメント : {ref.segment_id}")
    print(f"車種       : {gs.vehicle}   raw_can: {'読み込み済' if gs.raw_can_loaded else '未使用'}")
    print(f"時間長     : {gs.t[-1] - gs.t[0]:.1f} s   グリッド: {gs.rate_hz} Hz   サンプル数: {len(gs.df)}")
    if "op_tx" in gs.df:
        print(f"openpilot 送信区間の割合: {float(np.nanmean(gs.df['op_tx'])):.1%}")
    print()

    print("[特徴量]")
    s = feature_summary(gs)
    s[["coverage", "min", "max"]] = s[["coverage", "min", "max"]].round(3)
    print(s.to_string(index=False))
    print()

    print("[閾値までの余裕]  margin < 0 なら閾値を超えている")
    m = threshold_margins(gs, cfg)
    m[["threshold", "reached", "margin", "coverage"]] = m[["threshold", "reached", "margin", "coverage"]].round(3)
    print(m.to_string(index=False))
    print()

    events = detect_all(gs, cfg, max_stage=2 if gs.raw_can_loaded else 1)
    print(f"[検出] イベント {len(events)} 件")
    for e in events:
        print(f"  {e.event_type:14s} t_rel={e.t_start - gs.t[0]:6.2f}s  長さ={e.duration_s:5.2f}s  "
              f"peak={e.peak_value:8.2f}  ({e.feature} {e.op} {e.threshold})")

    cands = build_candidates(gs, events, cfg)
    print(f"\n[候補区間] {len(cands)} 件")
    for c in cands:
        print(f"  t_rel={c.t_start - gs.t[0]:6.2f}〜{c.t_end - gs.t[0]:6.2f}s  "
              f"severity={c.severity:5.2f}  種類={'|'.join(c.event_types)}")

    if gs.meta.get("skipped_events"):
        print(f"\n[適用しなかった定義] {gs.meta['skipped_events']}")

    if args.dump_csv:
        args.dump_csv.parent.mkdir(parents=True, exist_ok=True)
        gs.df.to_csv(args.dump_csv, index=False)
        print(f"\n時系列を書き出しました: {args.dump_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
