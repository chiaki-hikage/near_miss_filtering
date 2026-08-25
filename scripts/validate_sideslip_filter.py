#!/usr/bin/env python3
"""横滑りフィルタが本物の横滑りを拾えるかを KIT MSDM で確かめる。

commaCarSegments の走行では β が小さく、フィルタが 0 件を返しても
「その挙動が無い」のか「フィルタが動いていない」のか区別できない。
KIT MSDM は β が光学式センサで実測されている唯一のデータなので、
これを正のサンプルとして通し、再現率を測る。

抽出側と同じ経路を通す:
    KIT の走行 -> SegmentData -> 20 Hz グリッド -> features -> 1 次 -> 2 次

判定に使うのは車速・ヨーレート・横加速度・舵角だけで、実測 β は
答え合わせにしか使わない。実測 β が |β| >= 閾値 になっている時間を
どれだけ覆えたかを、走行ごとに出す。

  uv run python scripts/validate_sideslip_filter.py
  uv run python scripts/validate_sideslip_filter.py --truth-deg 5 --out out/kit_msdm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from near_miss.config import (
    DEFAULT_DETECTION,
    DEFAULT_VEHICLE_DIR,
    find_vehicle_config_by_name,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.features import compute_features
from near_miss.io import kit_msdm as kit
from near_miss.signals import to_grid
from near_miss.detectors import _merge_runs, _runs
from near_miss.sideslip import find_sideslip_candidates, stage1_mask, stage2_runs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=kit.DEFAULT_ROOT)
    p.add_argument("--kind", default=None,
                   help="走行の種類で絞る (dynamic / slow / parking / standstill)")
    p.add_argument("--truth-deg", type=float, default=5.0,
                   help="実測 β がこれ以上の時間を「本物の横滑り」とみなす")
    p.add_argument("--at", default="cog", choices=("cog", "ra", "cor"),
                   help="実測 β を取る位置")
    p.add_argument("--min-speed", type=float, default=None,
                   help="適用範囲の下限を設定から変えて試す")
    p.add_argument("--config", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=None, help="結果 CSV の出力先")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.min_speed is not None:
        cfg["sideslip"]["min_speed_mps"] = float(args.min_speed)
    vehicles = load_vehicle_configs(args.vehicles)
    vehicle = find_vehicle_config_by_name("kit_msdm", vehicles)
    if vehicle is None:
        raise SystemExit("configs/vehicles/kit_msdm.yaml がありません")
    param_file = args.root / "parameter.m"
    if not param_file.is_file():
        raise SystemExit(
            f"KIT MSDM が置かれていません: {args.root}\n"
            "  DOI 10.35097/44a91t97pmnha1k9 から取得し、\n"
            "  raw_data/kit_msdm/10.35097-44a91t97pmnha1k9/data/dataset/ に\n"
            "  *.mat と parameter.m が並ぶ形に展開してください。\n"
            "  詳細は docs/environment.md の「データの配置」"
        )
    params = kit.load_parameters(param_file)
    paths = kit.find_runs(args.root, kind=args.kind)
    if not paths:
        raise SystemExit(f"走行がありません: {args.root}")

    s = cfg["sideslip"]
    print(f"物差し   : KIT MSDM {len(paths)} 走行  ({args.root})")
    print(f"正解     : 実測 β ({args.at}) が |β| >= {args.truth_deg} deg の時間")
    print(f"適用範囲 : v >= {s['min_speed_mps']} m/s "
          f"(KIT の限界走行は低速なので、ここで落ちる時間が多い)")
    print()

    rows = []
    for path in paths:
        run = kit.read_run(path)
        sd = kit.segment_data(run, params)
        gs = to_grid(sd, cfg)
        if gs.df.empty:
            continue
        gs = compute_features(gs, cfg, vehicle=vehicle)
        t = gs.df["t"].to_numpy()
        dt = 1.0 / gs.rate_hz

        truth_beta = kit.measured_sideslip_on_grid(run, params, t, at=args.at)
        truth = np.isfinite(truth_beta) & (np.abs(truth_beta) >= args.truth_deg)

        mask1, _ = stage1_mask(gs.df, cfg)
        # 2 次は細切れをつないだ区間を単位に判定する。1 次の秒数も
        # 同じ単位で数えないと「2 次のほうが長い」表になってしまう。
        for a, b in _merge_runs(_runs(mask1), t, float(s["stage1"]["merge_gap_s"])):
            mask1[a : b + 1] = True
        runs1 = mask1.copy()
        passed, _ = stage2_runs(t, gs.df, mask1, cfg, gs.rate_hz)
        mask2 = np.zeros(len(t), dtype=bool)
        for a, b, _c in passed:
            mask2[a : b + 1] = True
        cands, counts = find_sideslip_candidates(gs, cfg)

        # 適用範囲の外にある正解時間は、フィルタが見られない時間として別に出す。
        v = gs.df["v_mps"].to_numpy()
        in_range = np.isfinite(gs.df["beta_model_deg"].to_numpy()) & (
            v >= float(s["min_speed_mps"])
        )
        rows.append({
            "走行": run.name,
            "路面": run.surface,
            "秒": round(float(t[-1] - t[0]), 1),
            "実測|β|最大": round(float(np.nanmax(np.abs(truth_beta))), 1),
            "正解 秒": round(truth.sum() * dt, 1),
            "うち適用範囲内 秒": round((truth & in_range).sum() * dt, 1),
            "1次 秒": round(runs1.sum() * dt, 1),
            "2次 秒": round(mask2.sum() * dt, 1),
            "再現率(適用範囲内)": (
                round(float((mask2 & truth & in_range).sum() / max((truth & in_range).sum(), 1)), 3)
                if (truth & in_range).any() else np.nan
            ),
            "候補": counts.n_candidates,
            "候補|β|推定最大": (
                round(max(abs(c.beta_peak_deg) for c in cands), 1) if cands else np.nan
            ),
            "等級": "|".join(sorted({c.grade for c in cands})) if cands else "",
        })
        print(f"  {run.name:<32} 実測|β|max {rows[-1]['実測|β|最大']:5.1f} deg  "
              f"候補 {counts.n_candidates}", flush=True)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 110)
    print(df.to_string(index=False))
    print("=" * 110)

    has_truth = df[df["正解 秒"] > 0]
    in_scope = df[df["うち適用範囲内 秒"] > 0]
    print(f"\n実測 |β| >= {args.truth_deg} deg を含む走行 : {len(has_truth)} / {len(df)}")
    print(f"  うち適用範囲 (v >= {s['min_speed_mps']}) に入るもの : {len(in_scope)}")
    if not in_scope.empty:
        hit = int((in_scope["候補"] > 0).sum())
        print(f"  候補を 1 件以上出した走行             : {hit} / {len(in_scope)}")
        print(f"  適用範囲内の正解時間を覆った割合      : "
              f"{in_scope['再現率(適用範囲内)'].mean():.1%} (走行ごとの平均)")
    out_of_scope = has_truth[has_truth["うち適用範囲内 秒"] == 0]
    if not out_of_scope.empty:
        print(f"\n適用範囲の外で起きた横滑り ({len(out_of_scope)} 走行):")
        print(out_of_scope[["走行", "路面", "実測|β|最大", "正解 秒"]].to_string(index=False))
        print("  -> 低速 (v < %.0f m/s) の横滑りはこのフィルタでは扱えない。" % s["min_speed_mps"])

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out / "sideslip_filter_recall.csv", index=False)
        print(f"\n出力: {args.out / 'sideslip_filter_recall.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
