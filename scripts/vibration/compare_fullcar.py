"""実測の振動イベントと Full-car モデルの予測を比べる。

これは逆推定ではない。路面の形の仮説ごとに、合わせるのは上下加速度の大きさだけ
(単発の凹凸は振幅と時刻、荒れた区間は荒さの強さ) とする。
ロールとピッチは当てはめに使わず、予測として実測と比べる。

仮説
  単発・右 / 単発・左 / 単発・両   1−cos、長さ 0.5 m の凹凸を、その側の前輪 → 後輪が通る
  荒れ・右 / 荒れ・左右独立        ISO 8608 型の荒れた区間 (長さは実測の 8〜15 Hz 振動の継続から求める)

height_cm は正が凹み、負が突起 (絶対値が深さ・高さ)。

評価は 3 帯域 (0.5〜3、3〜8、8〜15 Hz) ごとの振幅比・時間ずれ・波形相関。
荒れた区間は位相が乱数なので、波形相関ではなく帯域 RMS と振動の継続長で比べる。

    uv run python scripts/vibration/compare_fullcar.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar

import find_roll_bumps as F
import fullcar as FC
from near_miss.io.comma2k19 import find_segments
from near_miss.plotting import configure_japanese_font

REPO = _bootstrap.SRC.parent
DT = 0.001
LENGTH_M = 0.5
BANDS = {"0.5-3": (0.5, 3.0), "3-8": (3.0, 8.0), "8-15": (8.0, 15.0)}
SEEDS = range(20)
LEAD_S = 1.5

EVENTS = [
    # (ドライブ, セグメント, 振動開始 [s], 振動終了 [s], 比較の観点, 地点)
    ("2018-08-17--14-55-39", 9, 54.04, 54.59, "地点19 (右輪が強い走行)", 19),
    ("2018-08-17--12-07-08", 41, 34.70, 35.67, "地点19 (別走行)", 19),
    ("2018-08-02--08-34-47", 40, 38.03, 39.01, "地点19 (別走行・最初に確認した事象)", 19),
    ("2018-08-14--10-32-01", 38, 32.92, 33.71, "地点19 (別走行)", 19),
    ("2018-08-15--09-01-03", 21, 29.15, 29.57, "右側の強い衝撃・117 km/h", 0),
    ("2018-08-14--10-32-01", 27, 46.81, 47.57, "低周波型・地点4", 4),
    ("2018-08-17--14-55-39", 6, 46.53, 47.32, "低周波型・地点13", 13),
]

SINGLE = {"単発・右": ("R",), "単発・左": ("L",), "単発・両": ("L", "R")}
ROUGH = {"荒れ・右": ("R",), "荒れ・左右独立": ("L", "R")}


def bp(x: np.ndarray, lo: float, hi: float, fs: float) -> np.ndarray:
    return F._bandpass(x, lo, hi, fs)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def envelope_span(x: np.ndarray, fs: float, frac: float = 0.3) -> tuple[float, float]:
    """8〜15 Hz 成分の包絡線が最大の frac を超えている範囲 [s] (窓先頭から)。"""
    e = np.convolve(np.abs(bp(x, 8, 15, fs)), np.ones(10) / 10, "same")
    idx = np.flatnonzero(e > frac * e.max())
    return idx[0] / fs, idx[-1] / fs


def band_metrics(meas: dict, pred: dict, fs: float) -> dict:
    out = {}
    for k in ("heave", "roll", "pitch"):
        for name, (lo, hi) in BANDS.items():
            m, p = bp(meas[k], lo, hi, fs), bp(pred[k], lo, hi, fs)
            out[f"{k}_{name}_ratio"] = round(rms(m) / rms(p), 2) if rms(p) > 1e-6 else np.nan
            if rms(p) > 1e-6:
                L = int(0.1 * fs)
                cc = [np.corrcoef(m[L + l:len(m) - L + l], p[L:len(p) - L])[0, 1] for l in range(-L, L + 1)]
                j = int(np.argmax(cc))
                out[f"{k}_{name}_corr"] = round(float(cc[j]), 2)
                out[f"{k}_{name}_lag_ms"] = round((j - L) / fs * 1000)
            else:
                out[f"{k}_{name}_corr"] = np.nan
                out[f"{k}_{name}_lag_ms"] = np.nan
    return out


class Event:
    def __init__(self, seg, ev, cfg_vib):
        self.ev = ev
        self.fs = cfg_vib["grid_hz"]
        self.bands = cfg_vib["band_hz"]
        t_start, t_end = ev[2], ev[3]
        w = (seg.t >= t_start - 1.0) & (seg.t <= t_end + 2.0)
        self.t = seg.t[w]
        self.meas = {k: seg.bp[k][w] for k in ("heave", "roll", "pitch")}
        self.v = float(np.mean(seg.v[w]))
        a, b = envelope_span(self.meas["heave"], self.fs)
        self.patch_t0 = self.t[0] + a
        self.patch_len_m = (b - a) * self.v


def processed(r: dict, t_grid: np.ndarray, bands: dict, fs: float) -> dict:
    """モデル出力を IMU 相当に間引き、実測と同じ帯域通過をかける。"""
    out = {}
    for k, src, band in (("heave", "imu_heave", bands["heave"]), ("roll", "roll_rate", bands["roll"]),
                         ("pitch", "pitch_rate", bands["pitch"])):
        out[k] = bp(FC.downsample(r["t"], r[src], t_grid), band[0], band[1], fs)
    return out


def run_road(p, v, left, right, dur, contact):
    t = np.arange(0, dur, DT)
    case = FC.RoadCase("X", "", left, right)
    return FC.simulate(p, t, FC.wheel_inputs(case, p, v, t), contact=contact)


def fit_single(E: Event, p, sides, contact: str) -> dict:
    """単発の凹凸: 線形の単位応答で振幅と時刻を求め、片側拘束なら振幅だけ非線形で詰め直す。"""
    v, fs = E.v, E.fs
    dur = LEAD_S + 4.5
    grid = np.arange(0, dur, 1 / fs)
    s0 = v * LEAD_S

    def road(h):
        bump = lambda s: FC.cosine_bump(s, s0, LENGTH_M, -h)
        zero = lambda s: np.zeros_like(s)
        return (bump if "L" in sides else zero), (bump if "R" in sides else zero)

    unit = processed(run_road(p, v, *road(0.01), dur, "linear"), grid, E.bands, fs)
    best = None
    for tau in np.arange(-0.5, 0.5001, 0.01):
        t_hit = E.ev[2] + tau
        u = np.interp(E.t - t_hit + LEAD_S, grid, unit["heave"], left=0, right=0)
        a = float(u @ E.meas["heave"] / (u @ u))
        e = r2(E.meas["heave"], a * u)
        if best is None or e > best[0]:
            best = (e, tau, a)
    _, tau, a = best
    h = 0.01 * a
    t_hit = E.ev[2] + tau

    def pred_for(hh, cont):
        r = run_road(p, v, *road(hh), dur, cont)
        pr = processed(r, grid, E.bands, fs)
        return {k: np.interp(E.t - t_hit + LEAD_S, grid, x, left=0, right=0) for k, x in pr.items()}, r

    pred, r = pred_for(h, contact)
    if contact == "unilateral" and bool(r["lifted_linear"]):
        res = minimize_scalar(lambda hh: -r2(E.meas["heave"], pred_for(hh, contact)[0]["heave"]),
                              bounds=sorted([0.3 * h, 3.0 * h]), method="bounded", options=dict(xatol=1e-4))
        h = float(res.x)
        pred, r = pred_for(h, contact)
    return dict(height_cm=round(100 * h, 2), t_hit_s=round(t_hit, 2), pred=pred,
                lift_off_ms=float(r["lift_off"].any(axis=1).sum() * DT * 1000))


def fit_rough(E: Event, p, sides, contact: str, seed: int) -> dict:
    """荒れた区間: ISO クラス B の荒さを基準に、上下加速度の RMS が合う倍率を求める。"""
    v, fs = E.v, E.fs
    L = max(E.patch_len_m, 1.0)
    dur = LEAD_S + (L + p.wheelbase) / v + 2.5
    grid = np.arange(0, dur, 1 / fs)
    s0 = v * LEAD_S
    zero = lambda s: np.zeros_like(s)

    def road(scale):
        gd = FC.ISO8608_GD0["B"] * scale ** 2
        left = (lambda s: FC.rough_patch(s, s0, L, gd, seed)) if "L" in sides else zero
        right = (lambda s: FC.rough_patch(s, s0, L, gd, seed + 1000)) if "R" in sides else zero
        return left, right

    def pred_for(scale, cont):
        r = run_road(p, v, *road(scale), dur, cont)
        pr = processed(r, grid, E.bands, fs)
        return {k: np.interp(E.t - E.patch_t0 + LEAD_S, grid, x, left=0, right=0) for k, x in pr.items()}, r

    unit, _ = pred_for(1.0, "linear")
    scale = rms(E.meas["heave"]) / max(rms(unit["heave"]), 1e-12)
    pred, r = pred_for(scale, contact)
    if contact == "unilateral" and bool(r["lifted_linear"]):
        res = minimize_scalar(lambda sc: (rms(pred_for(sc, contact)[0]["heave"]) - rms(E.meas["heave"])) ** 2,
                              bounds=(0.3 * scale, 4.0 * scale), method="bounded", options=dict(xatol=1e-3))
        scale = float(res.x)
        pred, r = pred_for(scale, contact)
    gd0 = FC.ISO8608_GD0["B"] * scale ** 2
    return dict(gd0=gd0, iso_class=iso_class(gd0), pred=pred, patch_len_m=round(L, 1),
                lift_off_ms=float(r["lift_off"].any(axis=1).sum() * DT * 1000))


def iso_class(gd0: float) -> str:
    """ISO 8608 のクラス境界は隣接クラスの幾何平均の中間 (×2)。"""
    for k, g in FC.ISO8608_GD0.items():
        if gd0 < 2 * g:
            return k
    return "F 以上"


def plot_event(E: Event, singles: dict, rough_rep: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    fs = E.fs
    t = E.t - E.ev[2]
    fig, axs = plt.subplots(5, 1, figsize=(12, 15))
    colors = {"単発・右": "tab:red", "単発・左": "tab:blue", "単発・両": "0.5", "荒れ・右": "tab:orange",
              "荒れ・左右独立": "tab:green"}
    for ax, k, lab in zip(axs[:3], ("heave", "roll", "pitch"),
                          ("上下加速度 [m/s²]\n(大きさを合わせた量)", "ロール角速度 [deg/s]\n(予測)", "ピッチ角速度 [deg/s]\n(予測)")):
        ax.plot(t, E.meas[k], color="k", lw=1.3, label="実測 (IMU)")
        for name, d in list(singles.items()) + list(rough_rep.items()):
            ax.plot(t, d["pred"][k], color=colors[name], lw=0.9, alpha=0.8, label=name)
        ax.axvspan(0, E.ev[3] - E.ev[2], color="orange", alpha=0.15)
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(t[0], t[-1])
    axs[0].legend(fontsize=8, ncol=3, loc="upper right")
    axs[2].set_xlabel("振動開始からの時刻 [s]  (荒れた区間は乱数の 1 例。波形の位相は比べられない)")
    # 帯域 RMS
    ax = axs[3]
    names = ["実測"] + list(singles) + list(rough_rep)
    xs = np.arange(len(BANDS) * 3)
    width = 0.8 / len(names)
    for j, name in enumerate(names):
        vals = []
        for k in ("heave", "roll", "pitch"):
            src = E.meas[k] if name == "実測" else (singles.get(name) or rough_rep.get(name))["pred"][k]
            vals += [rms(bp(src, lo, hi, fs)) for lo, hi in BANDS.values()]
        ax.bar(xs + j * width, vals, width, color="k" if name == "実測" else colors[name], label=name)
    ax.set_xticks(xs + 0.4)
    ax.set_xticklabels([f"{k}\n{b}Hz" for k in ("上下", "ロール", "ピッチ") for b in BANDS], fontsize=8)
    ax.set_ylabel("帯域 RMS", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-2)   # 左右同時仮説のロール (理論上 0) で軸が潰れないように
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    # スペクトル
    ax = axs[4]
    for name, x, c in [("実測", E.meas["heave"], "k")] + [(n, d["pred"]["heave"], colors[n]) for n, d in
                                                        list(singles.items())[:1] + list(rough_rep.items())]:
        n = 1 << int(np.ceil(np.log2(len(x) * 2)))
        f = np.fft.rfftfreq(n, 1 / fs)
        P = np.abs(np.fft.rfft(x * np.hanning(len(x)), n))
        ax.semilogy(f, P, color=c, label=name)
    ax.set_xlim(0, 25)
    ax.set_xlabel("周波数 [Hz]")
    ax.set_ylabel("上下加速度の\n振幅スペクトル", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"{E.ev[0]} seg{E.ev[1]} {E.ev[2]:.2f}–{E.ev[3]:.2f} s  {E.v * 3.6:.0f} km/h  {E.ev[4]}\n"
                 f"片側拘束タイヤ。荒れた区間の長さ {E.patch_len_m:.1f} m (8〜15 Hz 振動の継続 × 車速)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=105)
    plt.close(fig)


def reproducibility(events: list[Event], p, rows_rough: pd.DataFrame, contact: str) -> pd.DataFrame:
    """地点 19 の各走行で荒さを合わせ、別の走行の上下加速度 RMS を予測する。"""
    site = [E for E in events if E.ev[5] == 19]
    out = []
    for src in site:
        g = rows_rough[(rows_rough.segment == src.ev[1]) & (rows_rough.drive == src.ev[0])
                       & (rows_rough.hypothesis == "荒れ・左右独立")]
        gd0 = float(np.median(g.gd0))
        for dst in site:
            preds = []
            for seed in list(SEEDS)[:5]:
                sc = np.sqrt(gd0 / FC.ISO8608_GD0["B"])
                d = fit_rough_fixed(dst, p, ("L", "R"), contact, seed, sc)
                preds.append(rms(d["heave"]))
            out.append(dict(fit_on=f"{src.ev[0][5:10]} seg{src.ev[1]}", predict=f"{dst.ev[0][5:10]} seg{dst.ev[1]}",
                            speed_kmh=round(dst.v * 3.6), gd0=gd0, meas_heave_rms=round(rms(dst.meas["heave"]), 3),
                            pred_heave_rms=round(float(np.median(preds)), 3),
                            ratio=round(rms(dst.meas["heave"]) / float(np.median(preds)), 2),
                            meas_roll_rms=round(rms(dst.meas["roll"]), 2)))
    return pd.DataFrame(out)


def fit_rough_fixed(E: Event, p, sides, contact, seed, scale) -> dict:
    v, fs = E.v, E.fs
    L = max(E.patch_len_m, 1.0)
    dur = LEAD_S + (L + p.wheelbase) / v + 2.5
    grid = np.arange(0, dur, 1 / fs)
    s0 = v * LEAD_S
    gd = FC.ISO8608_GD0["B"] * scale ** 2
    r = run_road(p, v, lambda s: FC.rough_patch(s, s0, L, gd, seed), lambda s: FC.rough_patch(s, s0, L, gd, seed + 1000),
                 dur, contact)
    pr = processed(r, grid, E.bands, fs)
    return {k: np.interp(E.t - E.patch_t0 + LEAD_S, grid, x, left=0, right=0) for k, x in pr.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=REPO / "raw_data/Chunk_1")
    ap.add_argument("--config", type=Path, default=REPO / "configs/vibration_fullcar.yaml")
    ap.add_argument("--vib-config", type=Path, default=REPO / "configs/vibration.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "out/vibration/fullcar")
    ap.add_argument("--contact", default="unilateral", choices=("unilateral", "linear"))
    args = ap.parse_args()

    matplotlib.use("Agg")
    configure_japanese_font()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg_vib = yaml.safe_load(args.vib_config.read_text())
    p = FC.build_params(FC.load_config(args.config))
    refs = find_segments(args.data_root)

    events, rows_s, rows_r = [], [], []
    for ev in EVENTS:
        drive_refs = [r for r in refs if r.drive_id.endswith(ev[0])]
        rot = F.drive_rotation([r.path for r in drive_refs])
        seg = F.load_seg(next(r for r in drive_refs if r.index == ev[1]), rot, cfg_vib)
        E = Event(seg, ev, cfg_vib)
        events.append(E)
        base = dict(drive=ev[0], segment=ev[1], note=ev[4], site=ev[5], speed_kmh=round(E.v * 3.6, 1))
        singles = {}
        for name, sides in SINGLE.items():
            d = fit_single(E, p, sides, args.contact)
            singles[name] = d
            rows_s.append(dict(base, hypothesis=name, height_cm=d["height_cm"],
                               shape="凹み" if d["height_cm"] > 0 else "突起", t_hit_s=d["t_hit_s"],
                               lift_off_ms=d["lift_off_ms"],
                               heave_r2=round(r2(E.meas["heave"], d["pred"]["heave"]), 3),
                               roll_r2=round(r2(E.meas["roll"], d["pred"]["roll"]), 3),
                               pitch_r2=round(r2(E.meas["pitch"], d["pred"]["pitch"]), 3),
                               **band_metrics(E.meas, d["pred"], E.fs)))
        rough_rep = {}
        for name, sides in ROUGH.items():
            fits = []
            for seed in SEEDS:
                d = fit_rough(E, p, sides, args.contact, seed)
                bm = band_metrics(E.meas, d["pred"], E.fs)
                a, b = envelope_span(d["pred"]["heave"], E.fs)
                rows_r.append(dict(base, hypothesis=name, seed=seed, gd0=d["gd0"], iso_class=d["iso_class"],
                                   patch_len_m=d["patch_len_m"], lift_off_ms=d["lift_off_ms"],
                                   env_dur_pred_s=round(b - a, 2),
                                   env_dur_meas_s=round(E.patch_len_m / E.v, 2),
                                   **{k: v for k, v in bm.items() if k.endswith("ratio")}))
                fits.append((abs(np.log(bm["roll_0.5-3_ratio"])) if np.isfinite(bm["roll_0.5-3_ratio"]) else 9, d))
            rough_rep[name] = sorted(fits, key=lambda x: x[0])[len(fits) // 2][1]
        tag = f"{ev[0][5:10]}--{ev[0][12:17]}_seg{ev[1]}"
        plot_event(E, singles, rough_rep, args.out / f"compare_{tag}.png")

    ds = pd.DataFrame(rows_s)
    dr = pd.DataFrame(rows_r)
    ds.to_csv(args.out / "compare_summary.csv", index=False)
    dr.to_csv(args.out / "compare_rough.csv", index=False)
    rep = reproducibility(events, p, dr, args.contact)
    rep.to_csv(args.out / "reproducibility_site19.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(ds[["segment", "hypothesis", "height_cm", "lift_off_ms", "heave_r2", "roll_r2",
              "heave_0.5-3_ratio", "heave_3-8_ratio", "heave_8-15_ratio", "roll_0.5-3_ratio", "roll_8-15_ratio"]].to_string(index=False))
    agg = dr.groupby(["segment", "hypothesis"]).agg(
        gd0=("gd0", "median"), iso=("iso_class", lambda x: x.mode()[0]), L=("patch_len_m", "first"),
        lift_ms=("lift_off_ms", "median"), dur_pred=("env_dur_pred_s", "median"), dur_meas=("env_dur_meas_s", "first"),
        **{c: (c, "median") for c in dr.columns if c.endswith("ratio")})
    print(agg.round(2).to_string())
    print(rep.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
