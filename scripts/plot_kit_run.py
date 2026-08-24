#!/usr/bin/env python3
"""KIT Multi-Surface Driving Maneuvers の 1 走行を時系列で並べる。

横滑り角 beta を主指標に、同じ時間軸で車速・ヨーレート・横加速度・操舵角を重ねる。
beta は光学式センサの実測なので、comma 系のデータでは見られなかった
「本物の横滑り」がどう見えるかの基準になる。

使い方:
  python scripts/plot_kit_run.py dynamic_driving_cobble_1
  python scripts/plot_kit_run.py dynamic_driving_asphalt_a_2 --out out/kit_msdm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from near_miss.io import kit_msdm as kit

plt.rcParams["font.family"] = ["Hiragino Sans", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BETA_MARKS = (5.0, 10.0)          # 参考線 [deg]
HILIGHT = 10.0                    # これを超える区間に色を敷く


def shade_high_beta(ax, t: np.ndarray, beta: np.ndarray, level: float) -> None:
    """|beta| が level を超える区間に薄く色を敷く。"""
    m = np.abs(np.nan_to_num(beta)) > level
    if not m.any():
        return
    d = np.diff(np.r_[0, m.astype(int), 0])
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
        ax.axvspan(t[s], t[e], color="#d94801", alpha=0.07, lw=0)


def plot_run(run: kit.Run, params: kit.VehicleParams, out_path: Path, stride: int = 5) -> None:
    t = run.t[::stride]
    beta = kit.sideslip_deg(run, params, at="cog", min_speed_mps=3.0)[::stride]
    beta_cor = kit.sideslip_deg(run, params, at="cor", min_speed_mps=3.0)[::stride]
    v_kmh = run["v_x_cor_mps"][::stride] * 3.6
    yaw_dps = np.degrees(run["w_z_cor_radps"])[::stride]
    a_y = run["a_y_ra_mps2"][::stride]
    delta_deg = np.degrees(run["delta_stm_rad"])[::stride]
    v = np.hypot(run["v_x_cor_mps"], run["v_y_cor_mps"])[::stride]
    ay_kin = v * np.deg2rad(yaw_dps)

    # CAN で取れる量だけから出した推定 (車速・ヨーレート・横加速度)
    l_r = params.wheelbase_m - params.l_f
    k = params.mass_kg * params.l_f / (params.c_r * params.wheelbase_m)
    beta_model = np.degrees(l_r * np.deg2rad(yaw_dps) / np.where(v > 3, v, np.nan) - k * a_y)

    has_force = "F_trl_N" in run
    n_panels = 6 if has_force else 5
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 2.05 * n_panels), sharex=True)

    # 1. 横滑り角
    ax = axes[0]
    shade_high_beta(ax, t, beta, HILIGHT)
    ax.plot(t, beta, color="#b30000", lw=1.4, label="実測 β (重心, 光学式)")
    ax.plot(t, beta_model, color="#2166ac", lw=0.9, ls="--", alpha=0.85,
            label="推定 β (車速・ヨーレート・横加速度のみ)")
    ax.plot(t, beta_cor, color="#888888", lw=0.7, alpha=0.6, label="実測 β (センサ位置)")
    for lv in BETA_MARKS:
        for s in (+1, -1):
            ax.axhline(s * lv, color="#666666", lw=0.7,
                       ls=":" if lv == BETA_MARKS[0] else "-.", alpha=0.8)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    # 最大点に印を付ける
    i = int(np.nanargmax(np.abs(beta)))
    ax.plot(t[i], beta[i], "o", ms=5, mfc="none", mec="#b30000", mew=1.4)
    ax.annotate(f"|β| 最大 {abs(beta[i]):.1f}°  ({t[i]:.1f} s, {v_kmh[i]:.0f} km/h)",
                xy=(t[i], beta[i]), xytext=(6, 10 if beta[i] > 0 else -18),
                textcoords="offset points", fontsize=8, color="#b30000")
    ax.set_ylabel("横滑り角 β\n[deg]")
    ax.legend(loc="lower right", fontsize=8, ncol=3, framealpha=0.9)

    # 2. 車速
    ax = axes[1]
    shade_high_beta(ax, t, beta, HILIGHT)
    ax.plot(t, v_kmh, color="#08519c", lw=1.3)
    ax.set_ylabel("車速\n[km/h]")
    ax.set_ylim(bottom=0)

    # 3. ヨーレート
    ax = axes[2]
    shade_high_beta(ax, t, beta, HILIGHT)
    ax.plot(t, yaw_dps, color="#238b45", lw=1.3)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_ylabel("ヨーレート\n[deg/s]")

    # 4. 横加速度
    ax = axes[3]
    shade_high_beta(ax, t, beta, HILIGHT)
    ax.plot(t, a_y, color="#6a51a3", lw=1.3, label="実測 a_y (後軸)")
    ax.plot(t, ay_kin, color="#cc7a00", lw=0.9, ls="--", alpha=0.85, label="v × ヨーレート")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    mu = params.mu(run.surface.split("+")[0].replace("cobble", "cobblestone"))
    if mu:
        for s in (+1, -1):
            ax.axhline(s * mu * 9.81, color="#999999", lw=0.8, ls="-.", alpha=0.8)
        ax.text(0.005, 0.06, f"点線は μ·g (μ={mu})", transform=ax.transAxes, fontsize=7.5,
                color="#555555")
    ax.set_ylabel("横加速度\n[m/s²]")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # 5. 実効操舵角
    ax = axes[4]
    shade_high_beta(ax, t, beta, HILIGHT)
    ax.plot(t, delta_deg, color="#a63603", lw=1.3)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_ylabel("実効操舵角 δ\n[deg]")

    # 6. タイロッド力 (前輪の横力の目安。このデータセット固有)
    if has_force:
        ax = axes[5]
        shade_high_beta(ax, t, beta, HILIGHT)
        ax.plot(t, run["F_trl_N"][::stride], color="#1f78b4", lw=1.0, label="左")
        ax.plot(t, run["F_trr_N"][::stride], color="#e31a1c", lw=1.0, label="右")
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.set_ylabel("タイロッド力\n[N]")
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
        ax.margins(x=0.005)
    axes[-1].set_xlabel("時刻 [s] (走行の先頭から)")

    absmax = float(np.nanmax(np.abs(beta)))
    over = float(np.sum(np.abs(np.nan_to_num(beta)) > HILIGHT)) * stride / kit.RATE_HZ
    fig.suptitle(
        f"KIT Multi-Surface Driving Maneuvers  —  {run.name}"
        f"   路面 {run.surface}   {run.duration_s:.1f} s   1000 Hz\n"
        f"|β| 最大 {absmax:.1f}°   |β|>{HILIGHT:.0f}° が {over:.1f} s"
        f"   (色を敷いた区間)   車両 Hyundai IONIQ 5 (後輪駆動)",
        fontsize=10.5, y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"{out_path}  ({absmax:.1f}° 最大, |β|>{HILIGHT:.0f}° が {over:.1f} s)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="走行名 (拡張子なし)。例 dynamic_driving_cobble_1")
    p.add_argument("--root", type=Path, default=kit.DEFAULT_ROOT)
    p.add_argument("--out", type=Path, default=Path("out/kit_msdm"))
    p.add_argument("--stride", type=int, default=5, help="間引き。1000 Hz を 1/stride にする")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    params = kit.load_parameters(args.root / "parameter.m")
    run = kit.read_run(args.root / f"{args.run}.mat")
    plot_run(run, params, args.out / f"{args.run}.png", stride=args.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
