#!/usr/bin/env python3
"""セグメントごとの特徴量の要約統計を出す (L2 の中間集計)。

**閾値を勘ではなく分布から決めるための表。** 判定は一切しない。

  生チャネル (グリッドに載せる前) の min/max   -> L1 のレンジ上界の設計に使う
  特徴量 (グリッド後) の分位点                 -> L3 の閾値の設計に使う

セグメント 1 本 x 列 1 つで 1 行の縦持ち。2,000 セグメントでも数 MB。

  uv run python scripts/summarize_features.py --platform TOYOTA_RAV4_TSS2 -w 8
  uv run python scripts/summarize_features.py --comma2k19 raw_data/Chunk_1 -w 8

**フラグ列は分位点ではなく「立った割合」と「立った回数」を出す。**
分位点を出しても 0 か 1 にしかならず、稀なフラグほど情報が消えるため。
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
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
from near_miss.features import compute_features
from near_miss.io.canonical import concat_segments
from near_miss.pipeline import split_contiguous
from near_miss.signals import to_grid

# 分位点。裾を見たいので上下とも細かく取る。
Q = (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)

# 0/1 しか取らない列。分位点ではなく発生率で要約する。
FLAG_HINT = ("_active", "_flag", "abs_fault", "brake_pressed", "decel_flag",
             "op_tx", "slip_warn", "cruise_active", "acc_braking", "gas_released",
             "tc_disabled", "brake_hold_active", "vsc_active", "abs_active",
             "precollision_active", "turn_signal", "hazard_light")

_S: dict = {}


def is_flag(name: str) -> bool:
    """**名前だけで決める。** 値から推定すると、たまたま 0/1 しか出なかった
    セグメントで連続量がフラグ扱いになり、列の意味が行ごとに変わってしまう
    (実測で brake_pressure / steer_torque_driver が該当した)。"""
    return any(h in name for h in FLAG_HINT)


def stats(name: str, x: np.ndarray, kind: str) -> dict:
    """1 列ぶんの要約。**欠測は除いて数え、その割合を必ず残す。**"""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    n, n_ok = int(x.size), int(ok.sum())
    row = {"column": name, "kind": kind, "n": n, "n_valid": n_ok,
           "valid_ratio": round(n_ok / n, 4) if n else 0.0}
    if n_ok == 0:
        return row
    v = x[ok]
    if is_flag(name):
        # 立ち上がりの回数。区間の長さに依らず「何回起きたか」を数える。
        b = v > 0.5
        row |= {"is_flag": True, "on_ratio": round(float(b.mean()), 5),
                "n_rising": int(np.count_nonzero(b[1:] & ~b[:-1]) + (1 if b.size and b[0] else 0))}
        return row
    qs = np.quantile(v, Q)
    row |= {"is_flag": False, "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()), "std": float(v.std()),
            "absmax": float(np.abs(v).max()), "range": float(v.max() - v.min())}
    row |= {f"p{q:g}".replace("0.", "p"): float(z) for q, z in zip(Q, qs)}
    return row


def init(dataset: str, det: Path, vdir: Path, platform: str) -> None:
    _S["cfg"] = load_yaml(det)
    _S["vehicles"] = load_vehicle_configs(vdir)
    _S["dataset"] = dataset
    _S["platform"] = platform
    if dataset == "ccs":
        from near_miss.io import comma_car_segments as m
    else:
        from near_miss.io import comma2k19 as m
    _S["mod"] = m


def run_drive(args) -> list[dict]:
    """1 ドライブぶん。連番のかたまり単位で連結してから統計を取る。

    **統計はセグメント単位に切り直す。** ブロック単位で出すと、
    連結の仕方が変わったときに数字が動いて比較できなくなる。
    """
    drive_id, refs = args
    cfg, mod = _S["cfg"], _S["mod"]
    if _S["dataset"] == "ccs":
        vehicle = next((v for v in _S["vehicles"] if _S["platform"] in v.platforms), None)
    else:
        vehicle = find_vehicle_config(refs[0].dongle_id, _S["vehicles"])
    if vehicle is None:
        return [{"drive_id": drive_id, "status": "no_vehicle_config"}]

    out: list[dict] = []
    for block in split_contiguous(refs):
        segs, spans = [], []
        for ref in block:
            try:
                sd = mod.load_segment(ref, vehicle, with_raw_can=True)
            except Exception as exc:
                out.append({"drive_id": drive_id, "segment": ref.index,
                            "status": f"load_failed:{type(exc).__name__}"})
                continue
            a, b = sd.t_span
            if not (np.isfinite(a) and np.isfinite(b)):
                out.append({"drive_id": drive_id, "segment": ref.index,
                            "status": "no_time_span"})
                continue
            # --- 生チャネル (グリッド前)。L1 のレンジ上界はこれで設計する ---
            for name, ch in sd.channels.items():
                sel = (ch.t >= a) & (ch.t <= b)
                r = stats(name, ch.v[sel], "raw")
                r |= {"drive_id": drive_id, "segment": ref.index, "status": "ok",
                      "t_span_s": round(float(b - a), 2),
                      "notes": ";".join(sd.notes)[:200]}
                out.append(r)
            spans.append((ref.index, float(a), float(b)))
            segs.append(sd)
        if not segs:
            continue

        merged = concat_segments(segs)
        gs = to_grid(merged, cfg)
        if gs.df.empty:
            continue
        gs = compute_features(gs, cfg, radar=merged.radar, vehicle=vehicle)
        df, t = gs.df, gs.t
        for index, t0, t1 in spans:
            sel = (t >= t0) & (t <= t1)
            if not sel.any():
                continue
            sub = df.loc[sel]
            for col in sub.columns:
                if col == "t":
                    continue
                r = stats(col, sub[col].to_numpy(), "feature")
                r |= {"drive_id": drive_id, "segment": index, "status": "ok"}
                out.append(r)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--platform", help="commaCarSegments の車種キー")
    g.add_argument("--comma2k19", type=Path, help="comma2k19 のチャンク")
    p.add_argument("--limit", type=int, default=None, help="セグメント数の上限")
    p.add_argument("-w", "--workers", type=int, default=1)
    p.add_argument("--detection", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.platform:
        from near_miss.io import comma_car_segments as m
        refs = m.find_segments(platform=args.platform)
        dataset, label = "ccs", f"commaCarSegments {args.platform}"
        out = args.out or Path(f"out/feature_stats_ccs_{args.platform}.csv")
    else:
        from near_miss.io import comma2k19 as m
        refs = m.find_segments(args.comma2k19)
        dataset, label = "c2k19", f"comma2k19 {args.comma2k19}"
        out = args.out or Path("out/feature_stats_c2k19.csv")

    refs = sorted(refs, key=lambda r: (r.drive_id, r.index))
    if args.limit:
        refs = refs[: args.limit]
    by_drive: dict[str, list] = defaultdict(list)
    for r in refs:
        by_drive[r.drive_id].append(r)
    tasks = sorted(by_drive.items())
    print(f"対象: {label}  {len(refs)} セグメント / {len(tasks)} ドライブ  "
          f"worker {args.workers}")

    t0 = time.perf_counter()
    rows: list[dict] = []
    ini = (dataset, args.detection, args.vehicles, args.platform or "")
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers, initializer=init, initargs=ini) as ex:
            for i, part in enumerate(ex.map(run_drive, tasks), 1):
                rows.extend(part)
                if i % 50 == 0:
                    print(f"  {i}/{len(tasks)} ドライブ ...", flush=True)
    else:
        init(*ini)
        for i, task in enumerate(tasks, 1):
            rows.extend(run_drive(task))
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} ドライブ ...", flush=True)

    df = pd.DataFrame(rows)
    df.insert(0, "dataset", dataset)
    lead = ["dataset", "drive_id", "segment", "status", "column", "kind",
            "n", "n_valid", "valid_ratio", "is_flag"]
    df = df[[c for c in lead if c in df.columns]
            + [c for c in df.columns if c not in lead]]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    ok = df[df.status == "ok"]
    n_seg = ok[["drive_id", "segment"]].drop_duplicates().shape[0]
    print(f"\n{time.perf_counter() - t0:.0f} 秒  {len(df):,} 行 / {n_seg} セグメント")
    bad = df[df.status != "ok"]
    if not bad.empty:
        print(f"  ** 異常 {len(bad)} 件: {bad.status.value_counts().to_dict()}")
    print(f"出力: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
