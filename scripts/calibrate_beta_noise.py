#!/usr/bin/env python3
"""横滑り角 beta に載るセンサ雑音を実測から求める。

beta = l_r * r / v - k * a_y なので、r と a_y の雑音が独立なら

    sigma_beta(v) = sqrt( (l_r * sigma_r / v)^2 + (k * sigma_ay)^2 )

直進に近い区間 (舵角レートと横加速度が小さい区間) だけを取り出し、
速度で分けた beta の散らばりにこの式を当てはめて sigma_r と sigma_ay を出す。
散らばりは中央絶対偏差から出す。外れ値 (拾い残した旋回) の影響を避けるため。

得られた 2 つの値を車種設定 (configs/vehicles/*.yaml の geometry) に
  yaw_rate_noise_dps / accel_y_noise_mps2
として書き写す。これを入れると features.py が beta_sigma を作り、
横滑りフィルタの 1 次判定が使えるようになる。

  # commaCarSegments (キャッシュ済みのものだけを使う。取得はしない)
  uv run python scripts/calibrate_beta_noise.py --platform TOYOTA_RAV4_TSS2 --limit 300

  # comma2k19
  uv run python scripts/calibrate_beta_noise.py --comma2k19 raw_data/Chunk_1 --limit 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import _bootstrap  # noqa: F401

from near_miss.config import (
    DEFAULT_DETECTION,
    DEFAULT_VEHICLE_DIR,
    find_vehicle_config,
    find_vehicle_config_for_platform,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.features import compute_features
from near_miss.io import comma2k19, comma_car_segments as ccs
from near_miss.io.canonical import concat_segments
from near_miss.pipeline import split_contiguous
from near_miss.signals import to_grid

# 速度の区切り。低速側は 1/v が効くので細かく取る。
SPEED_BINS = (3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 26, 30, 35, 45)
MIN_BIN_SAMPLES = 200


def robust_sigma(x: np.ndarray) -> float:
    """中央絶対偏差から出す標準偏差。旋回の拾い残しに引きずられない。"""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def collect(refs, vehicle, cfg, load, limit: int) -> pd.DataFrame:
    parts, n = [], 0
    by_drive: dict[str, list] = {}
    for r in refs:
        by_drive.setdefault(r.drive_id, []).append(r)
    for drefs in by_drive.values():
        if n >= limit:
            break
        for block in split_contiguous(drefs):
            segs = []
            for ref in block:
                try:
                    segs.append(load(ref, vehicle))
                except Exception as exc:
                    print(f"  読み出し失敗 {ref.segment_id}: {exc}")
            if not segs:
                continue
            merged = concat_segments(segs)
            gs = to_grid(merged, cfg)
            if gs.df.empty:
                continue
            gs = compute_features(gs, cfg, radar=merged.radar, vehicle=vehicle)
            keep = [c for c in ("v_mps", "beta_model_deg", "steer_rate_dps", "ay_kin_mps2")
                    if c in gs.df.columns]
            if len(keep) < 4:
                continue
            parts.append(gs.df[keep])
            n += len(block)
            if n % 50 < len(block):
                print(f"  {n} セグメント", flush=True)
    if not parts:
        raise SystemExit("特徴量を作れるセグメントがありませんでした")
    print(f"  合計 {n} セグメント")
    return pd.concat(parts, ignore_index=True)


def fit(df: pd.DataFrame, l_r: float, k: float, max_steer_rate: float, max_ay: float):
    quiet = df[
        (df.steer_rate_dps.abs() < max_steer_rate)
        & (df.ay_kin_mps2.abs() < max_ay)
        & df.beta_model_deg.notna()
    ]
    vs, ss, ns = [], [], []
    for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
        sub = quiet[(quiet.v_mps >= lo) & (quiet.v_mps < hi)]
        if len(sub) < MIN_BIN_SAMPLES:
            continue
        vs.append(float(sub.v_mps.mean()))
        ss.append(robust_sigma(sub.beta_model_deg.to_numpy()))
        ns.append(len(sub))
    if len(vs) < 3:
        raise SystemExit("当てはめに足る速度域がありません")
    vs, ss = np.array(vs), np.array(ss)

    def model(v, sigma_r, sigma_ay):
        return np.degrees(np.hypot(l_r * np.deg2rad(sigma_r) / v, k * sigma_ay))

    p, _ = curve_fit(model, vs, ss, p0=[0.5, 0.2], bounds=(0, np.inf), maxfev=20000)
    return p, vs, ss, ns, model, len(quiet)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--platform", help="commaCarSegments の車種キー")
    g.add_argument("--comma2k19", type=Path, help="comma2k19 のチャンク")
    p.add_argument("--limit", type=int, default=300, help="使うセグメント数")
    p.add_argument("--cache", type=Path, default=ccs.DEFAULT_CACHE)
    p.add_argument("--config", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--max-steer-rate-dps", type=float, default=3.0,
                   help="直進とみなす舵角レートの上限")
    p.add_argument("--max-ay-mps2", type=float, default=0.3,
                   help="直進とみなす横加速度の上限")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    vehicles = load_vehicle_configs(args.vehicles)

    if args.platform:
        vehicle = find_vehicle_config_for_platform(args.platform, vehicles)
        if vehicle is None:
            raise SystemExit(f"車種設定がありません: {args.platform}")
        refs = ccs.find_segments(args.cache, args.platform)
        load = lambda ref, veh: ccs.load_segment(ref, veh, with_raw_can=True)  # noqa: E731
        label = f"commaCarSegments / {args.platform}"
    else:
        refs = comma2k19.find_segments(args.comma2k19)
        if not refs:
            raise SystemExit(f"セグメントがありません: {args.comma2k19}")
        vehicle = find_vehicle_config(refs[0].dongle_id, vehicles)
        if vehicle is None:
            raise SystemExit(f"車種設定がありません: dongle={refs[0].dongle_id}")
        load = lambda ref, veh: comma2k19.load_segment(ref, veh, with_raw_can=True)  # noqa: E731
        label = f"comma2k19 / {args.comma2k19}"

    l_r = vehicle.center_to_rear_m()
    k = vehicle.sideslip_ay_coeff()
    if l_r is None or k is None:
        raise SystemExit(
            "この車種は l_r / k を出せません。geometry の center_to_front_m と "
            "understeer_gradient を先に決めてください"
        )
    print(f"対象     : {label}")
    print(f"車種設定 : {vehicle.name}  l_r={l_r:.4f} m  k={k:.6f} s^2/m")
    print(f"直進条件 : |舵角レート| < {args.max_steer_rate_dps} deg/s かつ "
          f"|ay_kin| < {args.max_ay_mps2} m/s^2")
    print()
    df = collect(refs, vehicle, cfg, load, args.limit)
    p, vs, ss, ns, model, n_quiet = fit(df, l_r, k, args.max_steer_rate_dps, args.max_ay_mps2)

    print(f"\n直進とみなしたサンプル {n_quiet} / 全 {len(df)} ({n_quiet/len(df):.1%})")
    print(f"\n{'車速 [m/s]':>10}{'n':>9}{'実測 σβ':>10}{'当てはめ':>10}{'比':>7}")
    for v_, s_, n_ in zip(vs, ss, ns):
        m = model(v_, *p)
        print(f"{v_:10.1f}{n_:9d}{s_:10.4f}{m:10.4f}{s_/m:7.2f}")
    resid = ss - model(vs, *p)
    print(f"\n当てはめ残差 rms = {np.sqrt(np.mean(resid**2)):.4f} deg")
    print("\n" + "=" * 60)
    print("configs/vehicles/*.yaml の geometry に書き写す値:")
    print(f"  yaw_rate_noise_dps: {p[0]:.4f}")
    print(f"  accel_y_noise_mps2: {p[1]:.4f}")
    print("=" * 60)
    print(f"\nこのとき sigma_beta は "
          + ", ".join(f"v={v}: {model(float(v), *p):.3f} deg" for v in (10, 20, 30)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
