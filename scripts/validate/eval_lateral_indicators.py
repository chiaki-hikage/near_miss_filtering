#!/usr/bin/env python3
"""横方向の候補に対して、ヨー残差と beta が**補助指標として使えるか**を測る。

ヨー残差 = 実測ヨーレート - 舵角から期待されるヨーレート。滑っていれば開く。
ただし期待値を出すには steer_ratio が要り、Corolla / Prius / Highlander は
未当てはめ。そこで**その走行自身から** steer_ratio を当てはめる。

  1. 事象の前後 (±除外幅) を除いた同じ走行のサンプルで
     yaw = g * (v * delta / L) を原点を通る回帰で解き、g = 1/steer_ratio を出す
  2. その当てはめの残差の標準偏差 sigma を、同じ区間から出す
  3. 事象の窓の中の残差を sigma で割って、外れ具合を見る

**当てはめが悪い走行では指標を使わない。** r^2 と sigma を一緒に出して、
どの事象で信用できるかが分かるようにする。beta は諸元 (横力係数) が要るので、
それがある車種だけで出す。

  uv run python scripts/validate/eval_lateral_indicators.py \\
      --events out/review_lateral/lateral_events.csv --out out/review_lateral

出力
  indicators.csv  1 行 1 事象。当てはめの質と、ヨー残差・beta の外れ具合
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
    find_vehicle_config_for_platform,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.features import compute_features
from near_miss.io import comma_car_segments as ccs
from near_miss.io.canonical import concat_segments
from near_miss.signals import to_grid

FIT_MIN_SPEED = 8.0     # 当てはめに使う下限車速 [m/s]
FIT_MIN_STEER = 5.0     # 同、下限舵角 [deg]。直進だけでは傾きが決まらない
FIT_MIN_YAW = 3.0       # 同、下限ヨーレート [deg/s]。0 付近で割ると発散する
FIT_MIN_SAMPLES = 150   # 当てはめに要る最低サンプル数
EXCLUDE_S = 8.0         # 事象の前後、当てはめから外す幅 [s]
CONTEXT_SEGMENTS = 8    # 当てはめのために前後に足すセグメント数


def fit_bicycle(v: np.ndarray, steer: np.ndarray, yaw: np.ndarray,
                wheelbase: float) -> dict:
    """定常円旋回の式 yaw = v*delta / (a + b*v^2) を当てはめる。

    a = steer_ratio * L、b = steer_ratio * K (K はアンダーステア勾配)。
    1 パラメータ (b = 0) で解くと、b*v^2 のぶんを steer_ratio が吸収してしまい、
    高速の走行で steer_ratio が 40 以上という非物理な値になる。

    v*delta/yaw = a + b*v^2 は (1, v^2) について線形なので、そのまま最小二乗で解く。
    ヨーが 0 付近のサンプルは割り算が発散するので除く。
    """
    d = np.deg2rad(steer)
    y = np.deg2rad(yaw)
    ok = (np.isfinite(v) & np.isfinite(d) & np.isfinite(y)
          & (np.abs(yaw) >= FIT_MIN_YAW) & (np.abs(steer) >= FIT_MIN_STEER))
    v, d, y = v[ok], d[ok], y[ok]
    if v.size < FIT_MIN_SAMPLES:
        return {"n": int(v.size), "a": np.nan, "b": np.nan, "steer_ratio": np.nan,
                "understeer_K": np.nan, "r2": np.nan, "sigma_dps": np.nan}
    z = v * d / y
    # 外れ値 (舵とヨーの符号が逆の瞬間など) を落としてから解く
    lo, hi = np.percentile(z, [2, 98])
    keep = (z >= lo) & (z <= hi) & (z > 0)
    if keep.sum() < FIT_MIN_SAMPLES:
        return {"n": int(keep.sum()), "a": np.nan, "b": np.nan, "steer_ratio": np.nan,
                "understeer_K": np.nan, "r2": np.nan, "sigma_dps": np.nan}
    vv, dd, yy, zz = v[keep], d[keep], y[keep], z[keep]
    X = np.column_stack([np.ones(vv.size), vv ** 2])
    a, b = np.linalg.lstsq(X, zz, rcond=None)[0]
    if a <= 0:
        return {"n": int(vv.size), "a": float(a), "b": float(b), "steer_ratio": np.nan,
                "understeer_K": np.nan, "r2": np.nan, "sigma_dps": np.nan}
    if b < 0:                      # アンダーステア勾配が負になったら 1 パラメータに戻す
        a, b = float(zz.mean()), 0.0
    exp_y = vv * dd / (a + b * vv ** 2)
    res = yy - exp_y
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    return {"n": int(vv.size), "a": float(a), "b": float(b),
            "steer_ratio": float(a / wheelbase),
            "understeer_K": float(b * wheelbase / a) if a > 0 else np.nan,
            "r2": float(1.0 - float(res @ res) / ss_tot) if ss_tot > 0 else np.nan,
            "sigma_dps": float(np.rad2deg(res.std()))}


def evaluate(ev: dict, vehicles, cfg: dict, cache: Path) -> dict:
    out = dict(ev)
    veh = find_vehicle_config_for_platform(ev["platform"], vehicles)
    index = {(r.drive_id, r.index): r for r in ccs.find_segments(cache, ev["platform"])}
    i0 = int(ev["segment"])
    refs = [index[(ev["drive_id"], i)] for i in range(i0 - CONTEXT_SEGMENTS, i0 + CONTEXT_SEGMENTS + 1)
            if (ev["drive_id"], i) in index]
    if not refs:
        out["error"] = "セグメントが手元にない"
        return out
    try:
        segs = [ccs.load_segment(r, veh, with_raw_can=True, with_radar=False) for r in refs]
        merged = concat_segments(segs)
        gs = to_grid(merged, cfg)
        gs = compute_features(gs, cfg, radar=None, vehicle=veh)
    except Exception as exc:                                  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    df = gs.df
    need = ("t", "v_mps", "steer_deg_s", "yaw_rate_dps")
    if df.empty or any(c not in df for c in need):
        out["error"] = "必要なチャネルがない"
        return out

    seg0 = [s for s in segs if s.ref.index == i0][0]
    t0 = float(seg0.t_span[0]) + float(ev["t_in_segment_s"])
    t1 = t0 + float(ev["duration_s"])
    t = df["t"].to_numpy()
    out["n_context_segments"] = len(refs)

    wb = veh.geometry_value("wheelbase_m")
    if wb is None:
        out["error"] = "wheelbase がない"
        return out

    # 当てはめに使うのは、事象から離れた、速度と舵角が足りるサンプル
    far = (t < t0 - EXCLUDE_S) | (t > t1 + EXCLUDE_S)
    v, steer, yaw = (df[c].to_numpy() for c in ("v_mps", "steer_deg_s", "yaw_rate_dps"))
    use = far & (v >= FIT_MIN_SPEED)
    fit = fit_bicycle(v[use], steer[use], yaw[use], wb)
    out.update({f"fit_{k}": val for k, val in fit.items()})
    if not np.isfinite(fit["a"]) or not np.isfinite(fit["steer_ratio"]):
        out["error"] = f"当てはめに足るサンプルがない (n={fit['n']})"
        return out

    # 事象の窓の中の残差
    win = (t >= t0 - 1.0) & (t <= t1 + 1.0)
    exp_dps = np.rad2deg(v * np.deg2rad(steer) / (fit["a"] + fit["b"] * v ** 2))
    resid = yaw - exp_dps
    rw = resid[win]
    ok = np.isfinite(rw)
    if not ok.any():
        out["error"] = "窓が範囲外"
        return out
    k = int(np.argmax(np.abs(rw[ok])))
    peak = float(rw[ok][k])
    out["yaw_expected_at_peak_dps"] = float(exp_dps[win][ok][k])
    out["yaw_meas_at_peak_dps"] = float(yaw[win][ok][k])
    out["yaw_resid_peak_dps"] = peak
    out["yaw_resid_peak_sigma"] = peak / fit["sigma_dps"] if fit["sigma_dps"] > 0 else np.nan
    # 実測が期待より小さければアンダーステア、大きければオーバーステア
    meas, exp_ = out["yaw_meas_at_peak_dps"], out["yaw_expected_at_peak_dps"]
    out["yaw_direction"] = ("アンダー" if abs(meas) < abs(exp_) else "オーバー") if np.isfinite(exp_) else ""
    # 比べるための、同じ走行の事象外での残差の外れ具合
    rf = resid[far & (v >= FIT_MIN_SPEED)]
    rf = rf[np.isfinite(rf)]
    if rf.size:
        out["resid_far_p99_sigma"] = float(np.percentile(np.abs(rf), 99) / fit["sigma_dps"])
        out["resid_far_max_sigma"] = float(np.abs(rf).max() / fit["sigma_dps"])

    # beta は諸元 (横力係数と重心-後軸距離) がある車種でしか出せない
    if "beta_model_deg" in df.columns:
        b = df["beta_model_deg"].to_numpy()[win]
        b = b[np.isfinite(b)]
        out["beta_peak_deg"] = float(b[np.argmax(np.abs(b))]) if b.size else np.nan
        out["beta_n_valid"] = int(b.size)
    else:
        out["beta_peak_deg"], out["beta_n_valid"] = np.nan, 0
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", type=Path, required=True,
                   help="評価する事象の CSV (platform / drive_id / segment / "
                        "t_in_segment_s / duration_s の列が要る)")
    p.add_argument("--cache", type=Path, default=ccs.DEFAULT_CACHE)
    p.add_argument("--config", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=Path("out/review_lateral"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    vehicles = load_vehicle_configs(args.vehicles)
    events = pd.read_csv(args.events).to_dict("records")
    print(f"{len(events)} 件を評価します")

    rows = [evaluate(ev, vehicles, cfg, args.cache) for ev in events]
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "indicators.csv", index=False)

    pd.set_option("display.width", 250)
    cols = [c for c in ("platform", "kind", "lateral_reason", "drive_id", "segment",
                        "fit_n", "fit_steer_ratio", "fit_understeer_K", "fit_r2", "fit_sigma_dps",
                        "yaw_meas_at_peak_dps", "yaw_expected_at_peak_dps",
                        "yaw_resid_peak_dps", "yaw_resid_peak_sigma", "yaw_direction",
                        "resid_far_p99_sigma", "beta_peak_deg", "error") if c in df.columns]
    print(df[cols].round(2).to_string(index=False))
    print(f"\n出力: {args.out / 'indicators.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
