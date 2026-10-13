"""地点 19 の各通過について、IMU と輪速の周波数特性を比べる。

- 4 帯域 (0.5〜3、3〜8、8〜15、15〜30 Hz) の RMS (上下・ロール・ピッチ)
- 8〜15 Hz 帯の主周波数が車速に比例するか (路面の空間周期なら f = v / λ)、一定か (固有振動)
- 振動の包絡線を交差点の入口からの距離でそろえたときと、時間でそろえたときの一致度

ジャイロが実質 10 Hz の走行は、ロール・ピッチを 4 Hz 以上で評価しない。

    uv run python scripts/vibration/site19_spectra.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
import numpy as np
import pandas as pd
import yaml

import find_roll_bumps as F
from near_miss.io.comma2k19 import find_segments
from near_miss.plotting import configure_japanese_font

REPO = _bootstrap.SRC.parent


def spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n = 1 << int(np.ceil(np.log2(len(x) * 4)))
    w = np.hanning(len(x))
    P = np.abs(np.fft.rfft((x - x.mean()) * w, n)) ** 2 / (fs * np.sum(w ** 2)) * 2
    return np.fft.rfftfreq(n, 1 / fs), P


def peak_hz(x: np.ndarray, fs: float, lo: float = 6.0, hi: float = 20.0) -> float:
    f, P = spectrum(x, fs)
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(P[m])])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs/vibration_pavement.yaml")
    ap.add_argument("--vib-config", type=Path, default=REPO / "configs/vibration.yaml")
    ap.add_argument("--data-root", type=Path, default=REPO / "raw_data/Chunk_1")
    ap.add_argument("--out", type=Path, default=REPO / "out/vibration/pavement")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    vib = dict(yaml.safe_load(args.vib_config.read_text()))
    vib["grid_hz"] = 100.0
    fs = vib["grid_hz"]
    bands = [tuple(b) for b in cfg["analysis"]["bands_hz"]]
    passes = pd.read_csv(args.out / "site19_passes.csv")
    refs = find_segments(args.data_root)
    matplotlib.use("Agg")
    configure_japanese_font()
    import matplotlib.pyplot as plt

    rows, curves = [], []
    for _, P in passes.iterrows():
        drive_refs = [r for r in refs if r.drive_id.endswith(P.drive)]
        rot = F.drive_rotation([r.path for r in drive_refs])
        ref = next(r for r in drive_refs if r.index == P.segment)
        s = F.load_seg(ref, rot, vib)
        exit_ = P.exit_s if np.isfinite(P.exit_s) else s.t[-1]
        w = (s.t >= P.entry_s) & (s.t <= exit_)
        gyro_ok = P.gyro_hz >= 50
        dist = np.concatenate([[0.0], np.cumsum((s.v[1:] + s.v[:-1]) / 2 * np.diff(s.t))])
        d_entry = dist - np.interp(P.entry_s, s.t, dist)
        row = dict(drive=P.drive, segment=P.segment, speed_kmh=P.speed_kmh, gps_east_rel_m=P.gps_east_rel_m,
                   gyro_hz=P.gyro_hz, window_s=round(float(exit_ - P.entry_s), 2), truncated=not np.isfinite(P.exit_s))
        for k in ("heave", "roll", "pitch"):
            for lo, hi in bands:
                ok = k == "heave" or gyro_ok or hi <= 4.0
                # 帯域通過は窓の外も含む全体にかけ、窓 (入口〜出口) で RMS を取る (端の影響を避ける)
                row[f"{k}_{lo:g}-{hi:g}"] = round(float(np.sqrt(np.mean(F._bandpass(s.raw[k], lo, hi, fs)[w] ** 2))), 3) if ok else np.nan
        row["heave_peak_hz"] = round(peak_hz(F._bandpass(s.raw["heave"], 0.7, 45, fs)[w], fs), 2)
        for wh in ("FL", "FR", "RL", "RR"):
            row[f"wheel_{wh}_peak_hz"] = round(peak_hz(F._bandpass(s.raw[wh], 2, 45, fs)[w], fs), 2)
            row[f"wheel_{wh}_8-15_rms"] = round(float(np.sqrt(np.mean(F._bandpass(s.raw[wh], 8, 15, fs)[w] ** 2))), 4)
        env = np.convolve(np.abs(F._bandpass(s.raw["heave"], 8, 15, fs)), np.ones(15) / 15, "same")
        wide = (s.t >= P.entry_s - 2) & (s.t <= exit_ + 2)
        idx = np.flatnonzero(wide & (env > 0.3 * env[wide].max()))
        row["vib_start_from_entry_m"] = round(float(d_entry[idx[0]]), 1)
        row["vib_end_from_entry_m"] = round(float(d_entry[idx[-1]]), 1)
        row["vib_start_from_entry_s"] = round(float(s.t[idx[0]] - P.entry_s), 2)
        row["vib_end_from_entry_s"] = round(float(s.t[idx[-1]] - P.entry_s), 2)
        rows.append(row)
        f, Ph = spectrum(F._bandpass(s.raw["heave"], 0.3, 45, fs)[w], fs)
        curves.append(dict(label=f"{P.drive[5:10]} seg{P.segment} {P.speed_kmh:.0f} km/h", f=f, Ph=Ph,
                           t=s.t[wide] - P.entry_s, d=d_entry[wide], env=env[wide], v=P.speed_kmh,
                           lane="縁石側" if P.gps_east_rel_m > 1.5 else "内側", gyro_ok=gyro_ok,
                           roll=s.raw["roll"][w], pitch=s.raw["pitch"][w],
                           wheels={k: F._bandpass(s.raw[k], 2, 45, fs)[w] for k in ("FL", "FR", "RL", "RR")}))
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "site19_spectra.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print(df.to_string(index=False))

    # 8〜15 Hz の主周波数と車速: 一定 (固有振動) か比例 (空間周期) か
    v = df.speed_kmh.to_numpy() / 3.6
    fpk = df.heave_peak_hz.to_numpy()
    ok = np.isfinite(fpk) & (fpk > 8) & (fpk < 15)
    slope, icpt = np.polyfit(v[ok], fpk[ok], 1)
    lam = v[ok] / fpk[ok]
    stats = dict(n=int(ok.sum()), peak_hz_mean=float(fpk[ok].mean()), peak_hz_std=float(fpk[ok].std()),
                 cv_peak=float(fpk[ok].std() / fpk[ok].mean()),
                 implied_wavelength_m_mean=float(lam.mean()), cv_wavelength=float(lam.std() / lam.mean()),
                 slope_hz_per_mps=float(slope), speed_range_kmh=[float(df.speed_kmh.min()), float(df.speed_kmh.max())])
    # 他の地点の振動イベント (広い車速範囲) も参考に
    ev = pd.read_csv(REPO / "out/vibration/longtail/events_all.csv")
    ev = ev[(ev.heave_dom_hz > 8) & (ev.heave_dom_hz < 15)]
    s2, i2 = np.polyfit(ev.speed_kmh / 3.6, ev.heave_dom_hz, 1)
    stats.update(other_events_n=int(len(ev)), other_events_slope_hz_per_mps=float(s2),
                 other_events_peak_mean=float(ev.heave_dom_hz.mean()), other_events_peak_std=float(ev.heave_dom_hz.std()),
                 other_events_speed_range_kmh=[float(ev.speed_kmh.min()), float(ev.speed_kmh.max())],
                 other_events_cv_peak=float(ev.heave_dom_hz.std() / ev.heave_dom_hz.mean()),
                 other_events_cv_wavelength=float(((ev.speed_kmh / 3.6) / ev.heave_dom_hz).std()
                                                  / ((ev.speed_kmh / 3.6) / ev.heave_dom_hz).mean()))
    (args.out / "site19_frequency_vs_speed.yaml").write_text(yaml.safe_dump(
        {k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in stats.items()}, allow_unicode=True))
    print(stats)

    fig, axs = plt.subplots(3, 2, figsize=(15, 14))
    ax = axs[0, 0]
    for c in curves:
        ax.semilogy(c["f"], c["Ph"], label=c["label"] + f" ({c['lane']})", lw=1.0, ls="-" if c["lane"] == "内側" else "--")
    for lo, hi in bands:
        ax.axvline(lo, color="0.7", lw=0.6)
    ax.set_xlim(0, 30)
    ax.set_xlabel("周波数 [Hz]")
    ax.set_ylabel("上下加速度 PSD [(m/s²)²/Hz]\n(交差点の入口〜出口)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    ax = axs[0, 1]
    for c in curves:
        if not c["gyro_ok"]:
            continue
        f, Pr = spectrum(F._bandpass(c["roll"], 0.3, 45, 100.0), 100.0)
        ax.semilogy(f, Pr, label=c["label"], ls="-" if c["lane"] == "内側" else "--")
    ax.set_xlim(0, 30)
    ax.set_xlabel("周波数 [Hz]")
    ax.set_ylabel("ロール角速度 PSD [(deg/s)²/Hz]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    ax = axs[1, 0]
    for c in curves:
        for k, ls in (("FR", "-"), ("FL", ":")):
            f, Pw = spectrum(c["wheels"][k], 100.0)
            ax.semilogy(f, Pw, ls=ls, lw=0.9, label=f"{c['label']} {k}" if k == "FR" else None)
    ax.set_xlim(0, 30)
    ax.set_xlabel("周波数 [Hz]")
    ax.set_ylabel("輪速 PSD [(m/s)²/Hz]\n(実線 FR・点線 FL)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    ax = axs[1, 1]
    ax.scatter(ev.speed_kmh, ev.heave_dom_hz, s=8, color="0.7", label=f"他の振動イベント (n={len(ev)})")
    ax.scatter(df.speed_kmh, df.heave_peak_hz, s=40, color="tab:red", label="地点 19 の通過", zorder=3)
    vv = np.linspace(30, 125, 10)
    lam0 = stats["implied_wavelength_m_mean"]
    ax.plot(vv, vv / 3.6 / lam0, "b--", label=f"空間周期 λ={lam0:.2f} m なら (f ∝ v)")
    ax.axhline(stats["peak_hz_mean"], color="tab:red", ls=":", label="一定の固有振動数なら")
    ax.set_xlabel("車速 [km/h]")
    ax.set_ylabel("上下加速度の主周波数 (8〜15 Hz 帯) [Hz]")
    ax.set_ylim(6, 18)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for ax, key, lab in ((axs[2, 0], "d", "交差点の入口 (前輪) からの距離 [m]"), (axs[2, 1], "t", "交差点の入口 (前輪) からの時刻 [s]")):
        for c in curves:
            ax.plot(c[key], c["env"], label=c["label"], ls="-" if c["lane"] == "内側" else "--")
        ax.axvline(0, color="lime")
        if key == "d":
            ax.axvline(float(passes.crossing_length_m.median()), color="red", label="出口 (中央値)")
        ax.set_xlabel(lab)
        ax.set_ylabel("上下加速度 8〜15 Hz の包絡線 [m/s²]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("地点 19 (一般道の交差点) の各通過: 実測 IMU・輪速の周波数特性", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out / "site19_spectra.png", dpi=95)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
