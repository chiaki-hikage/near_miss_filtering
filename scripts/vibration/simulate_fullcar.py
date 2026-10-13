"""Full-car モデルで 6 種類の路面入力を順方向に計算し、図と表を出す。

    uv run python scripts/vibration/simulate_fullcar.py

出力 (out/vibration/fullcar/)
  modes.csv                  固有振動数と減衰比
  concept_road_cases.png     6 ケースの路面形状の概念図
  case_<K>.png               ケースごとの予測時系列
  case_summary.csv           ケースごとのピーク値
  sensitivity_road.csv/png   路面の高さ・長さ・車速を振った感度
  sensitivity_params.csv/png 車両パラメータを範囲の両端に振った感度 (ケース C)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, sosfiltfilt

import fullcar as FC
from near_miss.plotting import configure_japanese_font

REPO = _bootstrap.SRC.parent
DT = 0.001


def measured_like(r: dict, fs_imu: float, bands: dict) -> dict:
    """IMU と同じレートに間引き、実測と同じ帯域通過をかける。"""
    t = np.arange(r["t"][0], r["t"][-1], 1 / fs_imu)
    out = {"t": t}
    for k, band in (("imu_heave", bands["heave"]), ("roll_rate", bands["roll"]), ("pitch_rate", bands["pitch"])):
        x = FC.downsample(r["t"], r[k], t)
        sos = butter(2, band, btype="band", fs=fs_imu, output="sos")
        out[k] = sosfiltfilt(sos, x)
    return out


def run_case(p: FC.Params, key: str, v: float, height: float, length: float, lead_s: float = 1.0,
             tail_s: float = 3.0, contact: str = "unilateral") -> dict:
    s0 = v * lead_s
    case = FC.road_cases(height=height, length=length, s0=s0)[key]
    dur = lead_s + (p.wheelbase + length + p.track[0]) / v + tail_s
    t = np.arange(0, dur, DT)
    r = FC.simulate(p, t, FC.wheel_inputs(case, p, v, t), contact=contact)
    r["case"] = case
    r["t_front"] = lead_s
    return r


def peaks(r: dict, p: FC.Params, m: dict) -> dict:
    lift = r["tire_load"] <= 0
    return dict(
        heave_peak_mps2=np.max(np.abs(r["imu_heave"])),
        heave_bp_peak_mps2=np.max(np.abs(m["imu_heave"])),
        roll_rate_peak_dps=np.max(np.abs(r["roll_rate"])),
        roll_rate_bp_peak_dps=np.max(np.abs(m["roll_rate"])),
        pitch_rate_peak_dps=np.max(np.abs(r["pitch_rate"])),
        pitch_rate_bp_peak_dps=np.max(np.abs(m["pitch_rate"])),
        roll_peak_deg=np.max(np.abs(r["roll"])), pitch_peak_deg=np.max(np.abs(r["pitch"])),
        roll_final_deg=r["roll"][-1],
        susp_defl_peak_mm=1000 * np.max(np.abs(r["susp_defl"])),
        susp_force_peak_n=np.max(np.abs(r["susp_force"])),
        tire_load_min_ratio=np.min(r["tire_load"] / p.static_load),
        tire_load_max_ratio=np.max(r["tire_load"] / p.static_load),
        lift_off=bool(lift.any()),
        lifted_linear=bool(r["lifted_linear"]),
        lift_off_ms=float(lift.any(axis=1).sum() * (r["t"][1] - r["t"][0]) * 1000),
    )


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------

def plot_concepts(p: FC.Params, out: Path, height: float, length: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cases = FC.road_cases(height=height, length=length, s0=3.0)
    s = np.arange(0, 9, 0.005)
    fig, axs = plt.subplots(4, 3, figsize=(15, 11), gridspec_kw=dict(height_ratios=[2, 1, 2, 1]))
    for n, (key, c) in enumerate(cases.items()):
        row, col = (n // 3) * 2, n % 3
        ax, el = axs[row, col], axs[row + 1, col]
        tf = p.track[0]
        tan = np.tan(np.deg2rad(c.joint_angle_deg))
        offset = lambda y: (y + tf / 2) * tan           # 斜め継ぎ目は左輪側から当たる
        hl = c.left(s - offset(-tf / 2))
        hr = c.right(s - offset(tf / 2))
        ys = np.linspace(-tf / 2 - 0.4, tf / 2 + 0.4, 60)
        grid = np.array([(c.left if y < 0 else c.right)(s - offset(y)) for y in ys])
        vmax = max(np.max(np.abs(grid)), 1e-6)
        ax.imshow(grid * 100, extent=[s[0], s[-1], ys[-1], ys[0]], aspect="auto", cmap="RdBu", vmin=-vmax * 100,
                  vmax=vmax * 100)
        # 車両 (前輪が s=2.0 にいる瞬間)。y は右正なので上が右になるよう反転して描く
        xf = 2.0
        ax.add_patch(Rectangle((xf - p.wheelbase - 0.8, -tf / 2 - 0.15), p.wheelbase + 1.6, tf + 0.3,
                               fill=False, ec="k", lw=1.2))
        for xw in (xf, xf - p.wheelbase):
            for yw in (-tf / 2, tf / 2):
                ax.add_patch(Rectangle((xw - 0.33, yw - 0.11), 0.66, 0.22, color="0.2"))
        ax.annotate("", xy=(xf + 1.3, -tf / 2 - 0.3), xytext=(xf - 0.5, -tf / 2 - 0.3), arrowprops=dict(arrowstyle="->"))
        ax.text(xf - 0.5, -tf / 2 - 0.35, "進行方向", fontsize=8, va="bottom")
        rear_note = "" if c.rear_follows else "\n(右後輪は凹凸の外側を通る)"
        ax.set_title(f"{key}: {c.title}{rear_note}", fontsize=10)
        ax.set_ylabel("左右 [m] (上=左 / 下=右)", fontsize=8)
        ax.set_xlim(-1.5, 9)
        el.plot(s, hl * 100, label="左輪の通り道", color="tab:blue")
        el.plot(s, hr * 100, label="右輪の通り道", color="tab:red", ls="--")
        el.set_ylabel("高さ [cm]", fontsize=8)
        el.set_xlabel("距離 [m]", fontsize=8)
        el.set_xlim(-1.5, 9)
        el.legend(fontsize=7, loc="lower left")
        el.grid(alpha=0.3)
    fig.suptitle(f"路面入力 6 ケース (基準: 高さ {height * 100:.0f} cm・長さ {length:.1f} m、色 = 高さ 赤:凹み 青:突起)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_case(r: dict, m: dict, p: FC.Params, out: Path, v: float) -> None:
    import matplotlib.pyplot as plt

    t = r["t"] - r["t_front"]
    tm = m["t"] - r["t_front"]
    c = r["case"]
    fig, axs = plt.subplots(7, 1, figsize=(11, 15), sharex=True)
    for i, (name, col) in enumerate(zip(FC.CORNERS, ("tab:blue", "tab:red", "tab:cyan", "tab:orange"))):
        axs[0].plot(t, r["zr"][:, i] * 100, color=col, label=name)
    axs[0].set_ylabel("路面入力 [cm]\n(包み込み後)", fontsize=9)
    axs[0].legend(ncol=4, fontsize=8, loc="lower right")
    axs[1].plot(t, r["imu_heave"], color="0.6", lw=0.8, label="モデル (1 kHz)")
    axs[1].plot(tm, m["imu_heave"], color="tab:blue", lw=1.0, label="実測と同じ処理 (104 Hz, 0.7〜20 Hz)")
    axs[1].set_ylabel("IMU 上下加速度\n[m/s²] (予測)", fontsize=9)
    axs[1].legend(fontsize=8, loc="upper right")
    ax = axs[2]
    ax.plot(t, r["roll_rate"], color="0.6", lw=0.8)
    ax.plot(tm, m["roll_rate"], color="tab:green", lw=1.0, label="ロール角速度 (実測と同じ処理)")
    ax.set_ylabel("ロール角速度\n[deg/s] (予測)", fontsize=9)
    tw = ax.twinx()
    tw.plot(t, r["roll"], color="darkgreen", lw=1.0, ls="--", label="ロール角 (モデル推定)")
    tw.set_ylabel("ロール角 [deg]", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    tw.legend(fontsize=8, loc="upper right")
    ax = axs[3]
    ax.plot(t, r["pitch_rate"], color="0.6", lw=0.8)
    ax.plot(tm, m["pitch_rate"], color="tab:purple", lw=1.0, label="ピッチ角速度 (実測と同じ処理)")
    ax.set_ylabel("ピッチ角速度\n[deg/s] (予測)", fontsize=9)
    tw = ax.twinx()
    tw.plot(t, r["pitch"], color="indigo", lw=1.0, ls="--", label="ピッチ角 (モデル推定)")
    tw.set_ylabel("ピッチ角 [deg]", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    tw.legend(fontsize=8, loc="upper right")
    cols = ("tab:blue", "tab:red", "tab:cyan", "tab:orange")
    for i, (name, col) in enumerate(zip(FC.CORNERS, cols)):
        axs[4].plot(t, r["susp_defl"][:, i] * 1000, color=col, label=name)
        axs[5].plot(t, (r["susp_force"][:, i] + r["arb_force"][:, i]) / 1000, color=col, label=name)
        axs[6].plot(t, r["tire_load"][:, i] / 1000, color=col, label=name)
    axs[4].set_ylabel("サスペンション\nストローク [mm]\n(伸び正、モデル推定)", fontsize=9)
    axs[5].set_ylabel("サスペンション力 [kN]\n(つり合いからの変化、\nばね+ダンパー+スタビ、推定)", fontsize=9)
    axs[6].set_ylabel("動的接地荷重 [kN]\n(静的荷重込み、推定)", fontsize=9)
    axs[6].axhline(0, color="k", lw=0.8)
    for a in axs[4:]:
        a.legend(ncol=4, fontsize=8, loc="lower right")
    for a in axs:
        a.grid(alpha=0.3)
        a.axvline(0, color="red", lw=0.6)
        a.axvline(p.wheelbase / v, color="red", lw=0.6, ls=":")
    axs[-1].set_xlabel("前輪が路面形状に差しかかってからの時刻 [s]   (赤点線 = 後輪が差しかかる時刻)")
    axs[-1].set_xlim(-0.3, t[-1])
    fig.suptitle(f"ケース {c.key}: {c.title}  車速 {v * 3.6:.0f} km/h  (灰 = モデルの値そのもの)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_sensitivity(df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [("heave_bp_peak_mps2", "上下加速度ピーク [m/s²]"), ("roll_rate_bp_peak_dps", "ロール角速度ピーク [deg/s]"),
               ("pitch_rate_bp_peak_dps", "ピッチ角速度ピーク [deg/s]"), ("tire_load_min_ratio", "接地荷重の最小 / 静的")]
    fig, axs = plt.subplots(3, 4, figsize=(17, 11))
    for r, (var, lab) in enumerate((("height_cm", "高さ [cm]"), ("length_m", "長さ [m]"), ("speed_kmh", "車速 [km/h]"))):
        d = df[df.vary == var]
        for c, (k, kl) in enumerate(metrics):
            ax = axs[r, c]
            for key, g in d.groupby("case"):
                ax.plot(g[var], g[k], "o-", ms=3, label=key)
            ax.set_xlabel(lab, fontsize=9)
            ax.set_ylabel(kl, fontsize=9)
            ax.grid(alpha=0.3)
            if k == "tire_load_min_ratio":
                ax.axhline(0, color="k", lw=0.8)
    axs[0, 0].legend(title="ケース", fontsize=8)
    fig.suptitle("路面形状と車速の感度 (他は基準値: 高さ 3 cm・長さ 0.5 m・80 km/h。ピークは実測と同じ処理後)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_param_sensitivity(df: pd.DataFrame, base: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [("heave_bp_peak_mps2", "上下加速度"), ("roll_rate_bp_peak_dps", "ロール角速度"),
               ("pitch_rate_bp_peak_dps", "ピッチ角速度")]
    fig, axs = plt.subplots(1, 3, figsize=(17, 7), sharey=True)
    params = df.param.unique()
    for ax, (k, lab) in zip(axs, metrics):
        lo = df[df.end == "low"].set_index("param")[k].reindex(params) / base[k] - 1
        hi = df[df.end == "high"].set_index("param")[k].reindex(params) / base[k] - 1
        y = np.arange(len(params))
        ax.barh(y - 0.2, lo * 100, height=0.4, label="範囲の下端", color="tab:blue")
        ax.barh(y + 0.2, hi * 100, height=0.4, label="範囲の上端", color="tab:red")
        ax.set_yticks(y)
        ax.set_yticklabels(params, fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{lab}ピークの変化 [%]", fontsize=10)
        ax.grid(alpha=0.3, axis="x")
    axs[0].legend(fontsize=8)
    fig.suptitle("車両パラメータの不確かさの影響 (ケース C、3 cm・0.5 m・80 km/h、1 つずつ範囲の両端へ)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def lift_off_thresholds(p: FC.Params, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """線形タイヤで接地荷重が 0 になり始める路面の高さ [cm]。

    線形モデルの応答は高さに比例するので、1 cm の単位応答で接地荷重が静的から
    どれだけ減るかを見れば、0 に達する高さが分かる。これより高い (深い) 入力では
    片側拘束の計算が必要になる。
    """
    import matplotlib.pyplot as plt

    rows = []
    for key in ("A", "B", "C", "D", "F"):
        for v_kmh in (30, 50, 65, 80, 100, 120):
            for length in (0.2, 0.3, 0.5, 0.8, 1.2, 2.0):
                r = run_case(p, key, v_kmh / 3.6, 0.03 if key in ("D", "F") else 0.01, length, contact="linear")
                unit = 0.02 if key in ("D", "F") else 0.01        # D・F は高さ 3 cm の 2/3 = 2 cm
                drop = np.max(1 - r["tire_load"] / p.static_load)   # 静的に対する最大の減少率
                rows.append(dict(case=key, speed_kmh=v_kmh, length_m=length,
                                 threshold_cm=round(100 * unit / drop, 2) if drop > 0 else np.inf))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for ax, key in zip(axs, ("A", "B", "C")):
        g = df[df.case == key].pivot(index="length_m", columns="speed_kmh", values="threshold_cm")
        im = ax.imshow(g.to_numpy(), cmap="viridis_r", aspect="auto", origin="lower", vmin=0, vmax=8)
        for (i, j), val in np.ndenumerate(g.to_numpy()):
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="w" if val < 4 else "k")
        ax.set_xticks(range(len(g.columns)))
        ax.set_xticklabels(g.columns)
        ax.set_yticks(range(len(g.index)))
        ax.set_yticklabels(g.index)
        ax.set_xlabel("車速 [km/h]")
        ax.set_ylabel("凹凸の長さ [m]")
        ax.set_title(f"ケース {key}: タイヤが浮き始める高さ [cm]", fontsize=10)
    fig.colorbar(im, ax=axs, shrink=0.8)
    fig.suptitle("線形タイヤで接地荷重が 0 に達する路面の高さ (これを超えると片側拘束の計算が必要)", fontsize=11)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------

PARAM_KEYS = [
    ("vehicle", "mass_total_kg"), ("vehicle", "cg_to_front_m"), ("vehicle", "cg_height_m"),
    ("vehicle", "roll_axis_height_m"), ("vehicle", "inertia_roll_kgm2"), ("vehicle", "inertia_pitch_kgm2"),
    ("unsprung", "mass_front_kg"), ("unsprung", "mass_rear_kg"),
    ("suspension", "ride_freq_front_hz"), ("suspension", "ride_freq_rear_hz"), ("suspension", "damping_ratio"),
    ("suspension", "roll_gradient_deg_per_g"), ("suspension", "roll_stiffness_front_share"),
    ("tire", "stiffness_n_per_m"), ("tire", "envelope_length_m"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs/vibration_fullcar.yaml")
    ap.add_argument("--vib-config", type=Path, default=REPO / "configs/vibration.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "out/vibration/fullcar")
    ap.add_argument("--speed-kmh", type=float, default=80.0)
    ap.add_argument("--height-cm", type=float, default=3.0)
    ap.add_argument("--length-m", type=float, default=0.5)
    args = ap.parse_args()

    matplotlib.use("Agg")
    configure_japanese_font()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = FC.load_config(args.config)
    vib = yaml.safe_load(args.vib_config.read_text())
    bands = vib["band_hz"]
    fs_imu = cfg["measurement"]["imu_rate_hz"]
    p = FC.build_params(cfg)
    v = args.speed_kmh / 3.6
    h = args.height_cm / 100

    pd.DataFrame(FC.modes(p)).round(3).to_csv(args.out / "modes.csv", index=False)
    derived = dict(m_s=p.m_s, k_spring=p.k.tolist(), c_damper=p.c.tolist(), k_arb_equiv=p.k_arb.tolist(),
                   static_load=p.static_load.tolist())
    (args.out / "derived_params.yaml").write_text(yaml.safe_dump(
        {k: (np.round(v_, 1).tolist() if isinstance(v_, list) else round(float(v_), 1)) for k, v_ in derived.items()},
        allow_unicode=True))

    plot_concepts(p, args.out / "concept_road_cases.png", h, args.length_m)
    lift_off_thresholds(FC.build_params(cfg), args.out / "lift_off_threshold.csv", args.out / "lift_off_threshold.png")

    rows = []
    for key in FC.road_cases():
        r = run_case(p, key, v, h, args.length_m)
        m = measured_like(r, fs_imu, bands)
        plot_case(r, m, p, args.out / f"case_{key}.png", v)
        rows.append(dict(case=key, title=r["case"].title, **peaks(r, p, m)))
    pd.DataFrame(rows).round(3).to_csv(args.out / "case_summary.csv", index=False)

    sens = []
    grid = {"height_cm": [1, 2, 3, 4, 5, 7], "length_m": [0.2, 0.3, 0.5, 0.8, 1.2, 2.0],
            "speed_kmh": [30, 50, 65, 80, 100, 120]}
    for var, vals in grid.items():
        for val in vals:
            hh = val / 100 if var == "height_cm" else h
            ll = val if var == "length_m" else args.length_m
            vv = val / 3.6 if var == "speed_kmh" else v
            for key in FC.road_cases():
                r = run_case(p, key, vv, hh, ll)
                m = measured_like(r, fs_imu, bands)
                sens.append(dict(vary=var, height_cm=hh * 100, length_m=ll, speed_kmh=vv * 3.6, case=key,
                                 **peaks(r, p, m)))
    sd = pd.DataFrame(sens).round(3)
    sd.to_csv(args.out / "sensitivity_road.csv", index=False)
    plot_sensitivity(sd, args.out / "sensitivity_road.png")

    r0 = run_case(p, "C", v, h, args.length_m)
    base = peaks(r0, p, measured_like(r0, fs_imu, bands))
    prow = []
    for sec, key in PARAM_KEYS:
        lo, hi = cfg[sec][key]["range"]
        for end, val in (("low", lo), ("high", hi)):
            pp = FC.build_params(cfg, {key: val})
            r = run_case(pp, "C", v, h, args.length_m)
            prow.append(dict(param=f"{key} ({cfg[sec][key]['source']})", end=end, value=val,
                             **peaks(r, pp, measured_like(r, fs_imu, bands))))
    for end, dx in (("low", -0.3), ("high", 0.3)):
        c2 = yaml.safe_load(yaml.safe_dump(cfg))
        c2["imu"]["pos_from_cg_m"]["value"][0] += dx
        pp = FC.build_params(c2)
        r = run_case(pp, "C", v, h, args.length_m)
        prow.append(dict(param="IMU 前後位置 ±0.3 m (assumed)", end=end, value=c2["imu"]["pos_from_cg_m"]["value"][0],
                         **peaks(r, pp, measured_like(r, fs_imu, bands))))
    for end, dy in (("low", -0.15), ("high", 0.15)):
        c2 = yaml.safe_load(yaml.safe_dump(cfg))
        c2["imu"]["pos_from_cg_m"]["value"][1] += dy
        pp = FC.build_params(c2)
        r = run_case(pp, "C", v, h, args.length_m)
        prow.append(dict(param="IMU 左右位置 ±0.15 m (assumed)", end=end, value=c2["imu"]["pos_from_cg_m"]["value"][1],
                         **peaks(r, pp, measured_like(r, fs_imu, bands))))
    pdf = pd.DataFrame(prow).round(3)
    pdf.to_csv(args.out / "sensitivity_params.csv", index=False)
    plot_param_sensitivity(pdf, base, args.out / "sensitivity_params.png")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
