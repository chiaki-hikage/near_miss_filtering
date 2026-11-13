#!/usr/bin/env python3
"""横風の突風を受けたときのヨー残差を 2 輪モデルで計算し、ZOD mini の実測の雑音と比べる (検出限界の感度分析)。

  uv run python scripts/zod/crosswind_sensitivity.py

車両・空力の値は仮定 (configs/zod/crosswind_sensitivity.yaml)。結果は「この仮定ならこの大きさ」という目安で、
ZOD の車両の実際の応答ではない。

ヨー残差は、実際の解析 (scripts/zod/signal_survey.py の fit_yaw_window) と同じく、舵角から定常の式で
予測したヨーレートとの差: Δr = r − v·δsw / (a + b·v²)、a = SR·L、b = SR·Kus。

出力 (out/zod/crosswind_sensitivity/):
  cases.csv          条件ごとの Δr の最大、横加速度の最大、補正の舵角の最大
  detectable.csv     条件ごとに、Δr の最大が雑音の閾値 (p99 と 3σ) を超える最小の突風の大きさ
  plots/             例の時系列、最小の検出可能な突風の図
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def simulate(v, W, T, cy, xcp, K, cfg):
    ve, ae, dr, sm = cfg["vehicle"], cfg["aero"], cfg["driver"], cfg["sim"]
    m, Iz, lf, lr, Cf, Cr, SR = (ve[k] for k in ("mass_kg", "yaw_inertia_kgm2", "lf_m", "lr_m", "cf_n_per_rad", "cr_n_per_rad", "steering_ratio"))
    L = lf + lr
    Kus = m / L * (lr / Cf - lf / Cr)
    a, b = SR * L, SR * Kus
    dt = sm["dt_s"]
    n = int(sm["t_total_s"] / dt)
    t = np.arange(n) * dt
    tg = t - sm["t_gust_start_s"]
    w = np.where((tg >= 0) & (tg <= T), W * (1 - np.cos(2 * np.pi * np.clip(tg, 0, T) / T)) / 2, 0.0)   # 左から吹くと正
    beta = np.zeros(n); r = np.zeros(n); dsw = np.zeros(n)
    lag = int(dr["delay_s"] / dt)
    for i in range(n - 1):
        dsw[i] = -K * r[i - lag] if i >= lag else 0.0                 # rad/rad (deg/(deg/s) の K を rad で同じ数値として使う)
        df = dsw[i] / SR
        af = beta[i] + lf * r[i] / v - df                             # 前輪の横すべり角
        ar = beta[i] - lr * r[i] / v
        Fyf, Fyr = -Cf * af, -Cr * ar
        bw = np.arctan2(w[i], v)
        Fw = -0.5 * ae["rho"] * (v ** 2 + w[i] ** 2) * ae["area_m2"] * cy * bw   # 左からの風は車を右 (負) へ押す
        Nw = Fw * xcp
        beta[i + 1] = beta[i] + dt * ((Fyf + Fyr + Fw) / (m * v) - r[i])
        r[i + 1] = r[i] + dt * ((lf * Fyf - lr * Fyr + Nw) / Iz)
    dsw[-1] = dsw[-2]
    r_pred = v * dsw / (a + b * v ** 2)
    ay = v * (np.gradient(beta, dt) + r)
    return t, w, r, r_pred, dsw, ay, (a, b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/zod/crosswind_sensitivity.yaml"))
    ap.add_argument("--out", type=Path, default=Path("out/zod/crosswind_sensitivity"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    (args.out / "plots").mkdir(parents=True, exist_ok=True)
    g, ae, dr = cfg["gust"], cfg["aero"], cfg["driver"]
    rows = []
    for kmh, W, T, cy, xcp, K in itertools.product(g["speeds_kmh"], g["amplitudes_ms"], g["durations_s"], ae["cy_per_rad"], ae["x_cp_m"], dr["gains"]):
        v = kmh / 3.6
        t, w, r, rp, dsw, ay, (a, b) = simulate(v, W, T, cy, xcp, K, cfg)
        d = np.degrees(r - rp)
        rows.append(dict(speed_kmh=kmh, gust_ms=W, duration_s=T, cy_per_rad=cy, x_cp_m=xcp, driver_gain=K,
                         peak_abs_yaw_resid_deg_s=round(float(np.abs(d).max()), 4),
                         peak_abs_yaw_rate_deg_s=round(float(np.degrees(np.abs(r).max())), 4),
                         peak_abs_ay=round(float(np.abs(ay).max()), 4),
                         peak_abs_steer_correction_deg=round(float(np.degrees(np.abs(dsw).max())), 3),
                         model_a=round(a, 2), model_b=round(b, 4)))
    cases = pd.DataFrame(rows)
    cases.to_csv(args.out / "cases.csv", index=False)

    nz = cfg["noise"]
    det = []
    for keys, x in cases.groupby(["speed_kmh", "duration_s", "cy_per_rad", "x_cp_m", "driver_gain"]):
        x = x.sort_values("gust_ms")
        rec = dict(zip(["speed_kmh", "duration_s", "cy_per_rad", "x_cp_m", "driver_gain"], keys))
        for veh, s in nz.items():
            if not isinstance(s, dict):
                continue
            for name, thr in (("p99", s["p99"]), (f"{nz['k_sigma']}sigma", nz["k_sigma"] * s["std"])):
                ok = x[x.peak_abs_yaw_resid_deg_s > thr]
                rec[f"min_gust_{veh}_{name}"] = float(ok.gust_ms.iloc[0]) if len(ok) else np.nan
        det.append(rec)
    det = pd.DataFrame(det)
    det.to_csv(args.out / "detectable.csv", index=False)

    # 図: 中央の仮定 (C_Y' 2.5、x_cp 0.7 m、補正 K = 1) で、速さ × 突風の長さごとの最小の検出可能な突風
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    vehs = [k for k, v in nz.items() if isinstance(v, dict)]
    for a_, veh in zip(ax, vehs):
        for (cy, xcp), x in det[(det.driver_gain == 1.0)].groupby(["cy_per_rad", "x_cp_m"]):
            if (cy, xcp) not in [(1.5, -0.2), (1.5, 0.3), (2.5, 0.7), (3.5, 1.1)]:
                continue
            for T, xx in x.groupby("duration_s"):
                if T not in (0.5, 2.0):
                    continue
                a_.plot(xx.speed_kmh, xx[f"min_gust_{veh}_p99"], "o-", label=f"C_Y'={cy}, x_cp={xcp} m, T={T} s")
        a_.set_title(f"雑音: {veh} (p99 = {nz[veh]['p99']} deg/s)"); a_.set_xlabel("車速 [km/h]")
        a_.legend(fontsize=7)
    ax[0].set_ylabel("ヨー残差が p99 を超える最小の突風 W [m/s]\n(運転者の補正 K = 1、空欄は 20 m/s でも超えない)")
    fig.suptitle("横風の突風の検出限界 (仮定の車両・空力。目安)"); fig.tight_layout()
    fig.savefig(args.out / "plots" / "min_detectable_gust.png", dpi=100); plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for K in dr["gains"]:
        t, w, r, rp, dsw, ay, _ = simulate(100 / 3.6, 8, 1.0, 2.5, 0.7, K, cfg)
        ax[0].plot(t, w, color="gray") if K == 0 else None
        ax[1].plot(t, np.degrees(r - rp), label=f"K = {K}")
        ax[2].plot(t, np.degrees(dsw), label=f"K = {K}")
    ax[0].set_ylabel("横風 w [m/s]"); ax[1].set_ylabel("ヨー残差 [deg/s]"); ax[2].set_ylabel("補正の舵角 [deg]")
    for veh in vehs:
        s = nz[veh]
        ax[1].axhline(s["p99"], ls="--", lw=0.7); ax[1].axhline(-s["p99"], ls="--", lw=0.7)
    ax[1].legend(fontsize=8); ax[2].set_xlabel("時刻 [s]")
    fig.suptitle("例: 100 km/h、突風 8 m/s・1 s、C_Y' = 2.5、x_cp = 0.7 m (破線: 実測の雑音の p99)")
    fig.tight_layout(); fig.savefig(args.out / "plots" / "example_100kmh_8ms.png", dpi=100); plt.close(fig)

    mid = det[(det.cy_per_rad == 2.5) & (det.x_cp_m == 0.7) & (det.driver_gain == 1.0)]
    print(mid.to_string(index=False))
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
