#!/usr/bin/env python3
"""comma1M を天候グループ別にスクリーニングする。

localizer の自己運動だけで作れる特徴量 (速度・縦加速度・躍度・進路変化率・
横加速度) の分布と、既存の検出器が拾うイベントを、天候グループごとに比べる。

やらないこと: スリップの断定。localizer は車両運動拘束と視覚特徴を融合した
推定なので、滑っている最中の値をそのまま滑りの証拠として扱わない。
ここで見るのは「進路と速度がどう変化したか」だけ。

使い方:
  python scripts/screen_comma1m.py --selection out/comma1m/selection.csv --out out/comma1m/screen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from near_miss.config import DEFAULT_VEHICLE_DIR, config_hash, load_vehicle_configs, load_yaml
from near_miss.io import comma1m
from near_miss.pipeline import _annotate_segment, process_block
from near_miss.scoring import candidates_to_frame, events_to_frame
from near_miss.sources import comma1m_source

# 分布を見る特徴量。localizer から作れるものだけ。
FEATURES = [
    "v_mps", "ax_mps2", "jerk_mps3", "yaw_rate_dps",
    "ay_kin_mps2", "lat_jerk_mps3", "net_heading_win_deg",
]
QUANTILES = [0.001, 0.01, 0.5, 0.99, 0.999, 0.9999]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, default=comma1m.DEFAULT_CACHE)
    p.add_argument("--selection", type=Path, default=Path("out/comma1m/selection.csv"))
    p.add_argument("--weather", type=Path, default=Path("out/comma1m/weather.csv"))
    p.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    p.add_argument("--dataset-config", type=Path, default=Path("configs/datasets/comma1m.yaml"))
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=Path("out/comma1m/screen"))
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    ds_cfg = load_yaml(args.dataset_config)
    vehicles = load_vehicle_configs(args.vehicles)
    cfg_hash = config_hash(cfg, [v.raw for v in vehicles])

    sel = pd.read_csv(args.selection)
    group_of = dict(zip(sel["segment_id"], sel["group"]))
    source = comma1m_source(args.cache, vehicles, names=list(group_of), localizer_cfg=ds_cfg["localizer"])
    refs = source.refs[: args.limit]
    print(f"セグメント {len(refs)} 件  設定 {cfg_hash}")

    vehicle = source.vehicle_for(refs[0])
    seg_rows, ev_frames, cand_frames = [], [], []
    samples: dict[str, list[np.ndarray]] = {g: [] for g in set(group_of.values())}
    sample_cols: list[str] = []

    for i, ref in enumerate(refs, 1):
        res = process_block([ref], vehicle, cfg, with_raw_can=False, max_stage=2, source=source)
        if res is None:
            seg_rows.append({"segment_id": ref.drive_id, "status": "skipped"})
            continue
        g = group_of.get(ref.drive_id, "?")
        df = res.gs.df
        if not sample_cols:
            sample_cols = [c for c in FEATURES if c in df.columns]
        samples[g].append(df[sample_cols].to_numpy())

        ev = _annotate_segment(events_to_frame(res.gs, res.events, cfg_hash), res.segment_spans, None)
        cd = _annotate_segment(candidates_to_frame(res.candidates, res.gs, cfg_hash), res.segment_spans, None)
        for f in (ev, cd):
            if not f.empty:
                f["group"] = g
        ev_frames.append(ev)
        cand_frames.append(cd)
        seg_rows.append({
            "segment_id": ref.drive_id, "group": g,
            "duration_s": float(res.gs.t[-1] - res.gs.t[0]),
            "n_events": len(res.events), "n_candidates": len(res.candidates),
            "v_max": float(np.nanmax(df["v_mps"])) if "v_mps" in df else np.nan,
            "status": "ok",
        })
        if i % 100 == 0:
            print(f"  {i}/{len(refs)}")

    segs = pd.DataFrame(seg_rows)
    events = pd.concat([f for f in ev_frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in ev_frames) else pd.DataFrame()
    cands = pd.concat([f for f in cand_frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in cand_frames) else pd.DataFrame()

    args.out.mkdir(parents=True, exist_ok=True)
    if not cands.empty:
        cands = cands.sort_values("severity", ascending=False).reset_index(drop=True)
        w = pd.read_csv(args.weather)
        keep = [c for c in ("segment_id", "country", "admin1", "place", "cold_tier",
                            "snow_score", "wet_score", "clear_score", "lat_mid", "lon_mid") if c in w.columns]
        cands = cands.merge(w[keep], left_on="drive_id", right_on="segment_id",
                            how="left", suffixes=("", "_w"))
    for name, df in (("segments", segs), ("events", events), ("candidates", cands)):
        df.to_csv(args.out / f"{name}.csv", index=False)
        print(f"  {args.out / (name + '.csv')}  {len(df)} 行")

    # --- 集計 -------------------------------------------------------------
    print("\n=== グループ別 ===")
    hours = segs.groupby("group")["duration_s"].sum() / 3600
    print(pd.DataFrame({"セグメント": segs.groupby("group").size(), "時間[h]": hours.round(2)}).to_string())

    if not events.empty:
        piv = events.pivot_table(index="event_type", columns="group", values="t_start", aggfunc="count").fillna(0)
        print("\n--- イベント件数 ---")
        print(piv.astype(int).to_string())
        print("\n--- 1 時間あたり ---")
        print((piv / hours).round(2).to_string())

    print("\n=== 特徴量の分位点 (グループ別) ===")
    rows = []
    for g, arrs in samples.items():
        if not arrs:
            continue
        a = np.concatenate(arrs, axis=0)
        for j, col in enumerate(sample_cols):
            x = a[:, j]
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            r = {"group": g, "feature": col, "n": x.size}
            for q in QUANTILES:
                r[f"p{q * 100:g}"] = float(np.quantile(x, q))
            r["min"] = float(x.min())
            r["max"] = float(x.max())
            rows.append(r)
    q = pd.DataFrame(rows)
    q.to_csv(args.out / "quantiles.csv", index=False)
    for col in sample_cols:
        sub = q[q.feature == col].set_index("group")
        print(f"\n{col}")
        print(sub.drop(columns=["feature"]).round(3).to_string())

    (args.out / "meta.json").write_text(json.dumps(
        {"config_hash": cfg_hash, "n_segments": len(refs)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
