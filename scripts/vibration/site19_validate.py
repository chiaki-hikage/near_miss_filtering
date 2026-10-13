"""地点 19 で路面入力モデルを校正し、別の走行で検証する。

モデル (車両パラメータは configs/vibration_fullcar.yaml で固定)
  旧・単発       1−cos・長さ 0.5 m の凹凸 1 つ。位置と高さを校正 (内側車線は左右両輪、縁石側は右輪のみ)
  旧・荒れ       fullcar.rough_patch (n^-2、0.05〜10 c/m、左右独立)。区間と荒さを校正
  新 A           舗装境界の区間でだけ短波長 (0.2〜10 c/m) の粗さが大きい 2 次元路面
  新 B           A + 左右で完全にそろう成分 (割合 β)
  新 C           B + 交差点全体にわたるなだらかな高さ変化 (高さ H)

校正と検証
  校正は 2 走行 (内側車線 1・縁石側 1) だけで行い、路面のパラメータを決める。
  残りの走行は、校正した路面を変えずに予測する。乱数の組 (seeds) は全走行で同じ。
  走行の違いは、車速の推移 (CAN)、交差点の入口の時刻 (映像)、車線 (映像で判別) だけ。

    uv run python scripts/vibration/site19_validate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
import numpy as np
import pandas as pd
import yaml

import find_roll_bumps as F
import fullcar as FC
import pavement as PV
from near_miss.io.comma2k19 import find_segments
from near_miss.plotting import configure_japanese_font

REPO = _bootstrap.SRC.parent
DT = 0.001
FS = 100.0
LANE_WIDTH_M = 3.4          # 仮定。車線を映像で判別し、縁石側は内側の 1 車線右とする
LANE_SPLIT_M = 1.7          # 内側と縁石側の境界 (車線境界線)
ENV_FRAC = 0.3
HEAVE_BAND = (0.7, 20.0)
RP_BAND = (0.7, 15.0)
BANDS = [(0.5, 3.0), (3.0, 8.0), (8.0, 15.0), (15.0, 30.0)]
MODELS = ["旧・単発", "旧・荒れ", "新A", "新B", "新C"]


# ---------------------------------------------------------------------------
# データ
# ---------------------------------------------------------------------------

class Pass:
    def __init__(self, row, refs, vib):
        self.drive, self.segment = row.drive, int(row.segment)
        self.tag = f"{row.drive[5:10]} seg{self.segment}"
        drive_refs = [r for r in refs if r.drive_id.endswith(row.drive)]
        rot = F.drive_rotation([r.path for r in drive_refs])
        ref = next(r for r in drive_refs if r.index == self.segment)
        self.seg = F.load_seg(ref, rot, vib)
        self.gyro_ok = row.gyro_hz >= 50
        self.entry = float(row.entry_s)
        self.truncated = not np.isfinite(row.exit_s)
        self.exit = float(row.exit_s) if not self.truncated else float(self.seg.t[-1] - 0.3)
        self.lane = "縁石側" if row.gps_east_rel_m > 1.5 else "内側"
        self.y_lane = LANE_WIDTH_M if self.lane == "縁石側" else 0.0
        s = self.seg
        dist = np.concatenate([[0.0], np.cumsum((s.v[1:] + s.v[:-1]) / 2 * np.diff(s.t))])
        self.d = dist - np.interp(self.entry, s.t, dist)        # 前輪の位置 (入口 = 0)
        lo = max(self.entry - 2.0, s.t[0] + 1.6)
        hi = min(self.exit + 2.0, s.t[-1] - 0.2)
        self.win = (s.t >= lo) & (s.t <= hi)                      # 包絡線を比べる範囲
        self.core = (s.t >= self.entry) & (s.t <= self.exit)      # 交差点の中 (RMS を比べる範囲)
        self.meas = {"heave": s.raw["heave"], "roll": s.raw["roll"], "pitch": s.raw["pitch"]}
        self.speed_kmh = float(np.mean(s.v[self.core]) * 3.6)


def env_of(x_heave: np.ndarray) -> np.ndarray:
    return np.convolve(np.abs(F._bandpass(x_heave, 8, 15, FS)), np.ones(15) / 15, "same")


def zones_from_envelope(P: Pass, wheelbase: float) -> list[tuple[float, float]]:
    """校正走行の 8〜15 Hz 包絡線から、振動を起こす区間 (前輪の位置) を取り出す。
    後輪が同じ区間を通るぶん (wheelbase) と、余韻のぶんは区間から除く。"""
    env = env_of(P.meas["heave"])
    w = P.win
    th = ENV_FRAC * env[w].max()
    above = (env > th) & w
    idx = np.flatnonzero(above)
    out = []
    start = idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 5:
            out.append((start, a))
            start = b
    out.append((start, idx[-1]))
    zones = []
    for a, b in out:
        sa, sb = P.d[a], P.d[b]
        zones.append((float(sa), float(max(sa + 1.0, sb - wheelbase))))
    return zones


# ---------------------------------------------------------------------------
# 計算
# ---------------------------------------------------------------------------

def simulate(P: Pass, p: FC.Params, model) -> dict:
    """路面モデル (2 次元、または y を無視する旧モデル) で、この走行を順方向に計算する。"""
    s = P.seg
    t0, t1 = s.t[P.win][0] - 1.5, s.t[P.win][-1]
    t = np.arange(t0, t1, DT)
    s_front = np.interp(t, s.t, P.d)
    zr = PV.track_inputs(model, s_front, P.y_lane, p.wheelbase, p.track[0], p.envelope_m)
    r = FC.simulate(p, t - t0, zr)
    grid = s.t[(s.t >= t0 + 0.5) & (s.t <= t1)]
    out = {}
    for k, src in (("heave", "imu_heave"), ("roll", "roll_rate"), ("pitch", "pitch_rate")):
        x = FC.downsample(r["t"] + t0, r[src], grid)
        full = np.zeros(len(s.t))
        m = (s.t >= grid[0]) & (s.t <= grid[-1])
        full[m] = np.interp(s.t[m], grid, x)
        out[k] = full
    out["lift_off_ms"] = float(r["lift_off"].any(axis=1).sum() * DT * 1000)
    out["zr_max_mm"] = float(np.abs(zr).max() * 1000)
    return out


def metrics(P: Pass, pred: dict) -> dict:
    m = {}
    for k in ("heave", "roll", "pitch"):
        for lo, hi in BANDS:
            ok = k == "heave" or P.gyro_ok or hi <= 4.0
            if not ok:
                continue
            a = np.sqrt(np.mean(F._bandpass(P.meas[k], lo, hi, FS)[P.core] ** 2))
            b = np.sqrt(np.mean(F._bandpass(pred[k], lo, hi, FS)[P.core] ** 2))
            m[f"{k}_{lo:g}-{hi:g}_logerr"] = float(np.log10(b / a)) if b > 1e-9 else np.nan
    for k, band in (("heave", HEAVE_BAND), ("roll", RP_BAND), ("pitch", RP_BAND)):
        if k != "heave" and not P.gyro_ok:
            continue
        y = F._bandpass(P.meas[k], *band, FS)[P.win]
        yh = F._bandpass(pred[k], *band, FS)[P.win]
        m[f"{k}_r2"] = float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
        m[f"{k}_rms_ratio"] = float(np.sqrt(np.mean(yh ** 2)) / np.sqrt(np.mean(y ** 2)))
    e, eh = env_of(P.meas["heave"])[P.win], env_of(pred["heave"])[P.win]
    m["env_r2"] = float(1 - np.sum((e - eh) ** 2) / np.sum((e - e.mean()) ** 2))
    m["env_corr"] = float(np.corrcoef(e, eh)[0, 1]) if eh.std() > 0 else np.nan
    d = P.d[P.win]
    on = lambda env: d[np.flatnonzero(env > ENV_FRAC * env.max())]
    a, b = on(e), on(eh)
    m["onset_err_m"] = float(b[0] - a[0]) if len(b) else np.nan
    m["end_err_m"] = float(b[-1] - a[-1]) if len(b) else np.nan
    m["lift_off_ms"] = pred["lift_off_ms"]
    return m


def median_metric(P, p, make_model, seeds, key) -> float:
    vals = []
    for sd in seeds:
        pr = simulate(P, p, make_model(sd))
        lo, hi = key
        vals.append(np.sqrt(np.mean(F._bandpass(pr["heave"], lo, hi, FS)[P.core] ** 2)))
    return float(np.median(vals))


def meas_band(P, k, lo, hi) -> float:
    return float(np.sqrt(np.mean(F._bandpass(P.meas[k], lo, hi, FS)[P.core] ** 2)))


# ---------------------------------------------------------------------------
# 模型の組み立て
# ---------------------------------------------------------------------------

def lane_y(lane: str) -> tuple[float, float]:
    return (-6.0, LANE_SPLIT_M) if lane == "内側" else (LANE_SPLIT_M, 8.0)


def new_model(kind, zones_by_lane, gd0_by_lane, seed, cfg, beta=0.0, crown=None):
    rc = cfg["road"]
    zl = []
    for lane, zs in zones_by_lane.items():
        for sa, sb in zs:
            zl.append(PV.Zone((sa, sb), lane_y(lane), gd0_by_lane[lane], rc["taper_m"]))
    return PV.PavementModel(kind, zl, PV.Roughness(tuple(rc["band_cpm"]), seed), rc["gd0_outside"],
                            common_fraction=beta, crown=crown)


def old_rough_model(patch_by_lane, gd0_by_lane, seed):
    def h(s, y):
        lane = "縁石側" if y > LANE_SPLIT_M else "内側"
        a, b = patch_by_lane[lane]
        side_seed = seed if y < (LANE_WIDTH_M if lane == "縁石側" else 0.0) else seed + 1000
        return FC.rough_patch(s, a, b - a, gd0_by_lane[lane], side_seed)
    return h


def old_bump_model(bump_by_lane):
    def h(s, y):
        lane = "縁石側" if y > LANE_SPLIT_M else "内側"
        sb, height = bump_by_lane[lane]
        center = LANE_WIDTH_M if lane == "縁石側" else 0.0
        if lane == "縁石側" and y < center:     # 縁石側は右輪だけが通る凹凸
            return np.zeros_like(s)
        return FC.cosine_bump(s, sb, 0.5, height)
    return h


# ---------------------------------------------------------------------------
# 校正
# ---------------------------------------------------------------------------

def calibrate(cal: dict[str, Pass], p: FC.Params, cfg: dict, seeds: list[int]) -> dict:
    out = {}
    zones = {lane: zones_from_envelope(P, p.wheelbase) for lane, P in cal.items()}
    out["zones"] = zones
    tgt = {lane: meas_band(P, "heave", 8, 15) for lane, P in cal.items()}

    def fit_gd0(kind, beta=0.0, crown=None, init=None):
        g = dict(init or {lane: 1e-4 for lane in cal})
        for _ in range(3):
            for lane, P in cal.items():
                pred = median_metric(P, p, lambda sd: new_model(kind, zones, g, sd, cfg, beta, crown), seeds, (8, 15))
                g[lane] = float(np.clip(g[lane] * (tgt[lane] / pred) ** 2, 1e-8, 1.0))
        return g

    out["新A"] = dict(gd0=fit_gd0("A"))
    best = None
    for beta in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        g = fit_gd0("B", beta, init=out["新A"]["gd0"])
        err = 0.0
        for lane, P in cal.items():
            vals = [np.sqrt(np.mean(F._bandpass(simulate(P, p, new_model("B", zones, g, sd, cfg, beta))["roll"], *RP_BAND, FS)[P.core] ** 2))
                    for sd in seeds]
            err += np.log10(np.median(vals) / np.sqrt(np.mean(F._bandpass(P.meas["roll"], *RP_BAND, FS)[P.core] ** 2))) ** 2
        if best is None or err < best[0]:
            best = (err, beta, g)
    out["新B"] = dict(gd0=best[2], beta=best[1], roll_err=float(best[0]))
    beta = best[1]
    L_x = float(np.median([np.interp(P.exit, P.seg.t, P.d) for P in cal.values()]))   # 交差点の出口までの距離
    best = None
    for H in (0.0, 0.01, 0.02, 0.03, 0.045, 0.06):
        crown = (0.0, L_x, H)
        g = fit_gd0("C", beta, crown, init=out["新B"]["gd0"])
        err = 0.0
        for lane, P in cal.items():
            pred = median_metric(P, p, lambda sd: new_model("C", zones, g, sd, cfg, beta, crown), seeds, (0.5, 3))
            err += np.log10(pred / meas_band(P, "heave", 0.5, 3)) ** 2
        if best is None or err < best[0]:
            best = (err, H, g)
    out["新C"] = dict(gd0=best[2], beta=beta, crown=(0.0, L_x, best[1]), low_err=float(best[0]))

    # 旧・荒れ: 区間は全振動区間を 1 つにまとめ、荒さは上下加速度 (0.7〜20 Hz) の RMS で合わせる
    patch = {lane: (zs[0][0], zs[-1][1] + p.wheelbase) for lane, zs in zones.items()}
    g = {lane: 64e-6 for lane in cal}
    for _ in range(3):
        for lane, P in cal.items():
            pred = median_metric(P, p, lambda sd: old_rough_model(patch, g, sd), seeds, HEAVE_BAND)
            g[lane] = float(np.clip(g[lane] * (meas_band(P, "heave", *HEAVE_BAND) / pred) ** 2, 1e-8, 1.0))
    out["旧・荒れ"] = dict(patch=patch, gd0=g)

    # 旧・単発: 位置を 0.25 m 刻みで探し、高さは上下加速度の波形の最小二乗 (単位応答は線形)
    bump = {}
    for lane, P in cal.items():
        best = None
        y = F._bandpass(P.meas["heave"], *HEAVE_BAND, FS)[P.win]
        for sb in np.arange(-2.0, np.interp(P.exit, P.seg.t, P.d) + 2.0, 0.25):
            unit = simulate(P, p, old_bump_model({lane: (sb, 0.01), ("内側" if lane == "縁石側" else "縁石側"): (sb, 0.0)}))
            u = F._bandpass(unit["heave"], *HEAVE_BAND, FS)[P.win]
            a = float(u @ y / (u @ u))
            r2 = 1 - np.sum((y - a * u) ** 2) / np.sum((y - y.mean()) ** 2)
            if best is None or r2 > best[0]:
                best = (r2, sb, 0.01 * a)
        bump[lane] = (float(best[1]), float(best[2]))
    out["旧・単発"] = dict(bump=bump)
    return out


def build(name: str, cal: dict, cfg: dict, seed: int):
    if name == "旧・単発":
        return old_bump_model(cal["旧・単発"]["bump"])
    if name == "旧・荒れ":
        return old_rough_model(cal["旧・荒れ"]["patch"], cal["旧・荒れ"]["gd0"], seed)
    c = cal[name]
    kind = name[-1]
    return new_model(kind, cal["zones"], c["gd0"], seed, cfg, c.get("beta", 0.0), c.get("crown"))


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------

def plot_road(cal, cfg, p, out: Path):
    import matplotlib.pyplot as plt

    s = np.arange(-5, 35, 0.01)
    fig, axs = plt.subplots(3, 1, figsize=(13, 11))
    for ax, lane, y in ((axs[0], "内側", 0.0), (axs[1], "縁石側", LANE_WIDTH_M)):
        for name, c in (("旧・単発", "0.5"), ("旧・荒れ", "tab:orange"), ("新A", "tab:blue"), ("新C", "tab:green")):
            for side, ls in (("右輪", "-"), ("左輪", ":")):
                yy = y + (p.track[0] / 2 if side == "右輪" else -p.track[0] / 2)
                m = build(name, cal, cfg, 0)
                h = m(s, yy) if callable(m) else m.height(s, yy)
                ax.plot(s, h * 1000, color=c, ls=ls, lw=0.9, label=f"{name} {side}")
        for sa, sb in cal["zones"].get(lane, []):
            ax.axvspan(sa, sb, color="tab:red", alpha=0.08)
        ax.set_ylabel("路面の高さ [mm]")
        ax.set_title(f"{lane}車線の左右の輪の軌跡 (乱数の 1 例。赤 = 校正で決めた振動区間)", fontsize=10, loc="left")
        ax.legend(fontsize=7, ncol=4)
        ax.grid(alpha=0.3)
    axs[1].set_xlabel("交差点の入口 (前輪) からの距離 [m]")
    ax = axs[2]
    n = np.logspace(np.log10(0.03), np.log10(12), 300)
    for name, band, gd0, c in (("旧・荒れ (内側)", (0.05, 10), cal["旧・荒れ"]["gd0"]["内側"], "tab:orange"),
                               ("新A (内側の区間)", tuple(cfg["road"]["band_cpm"]), cal["新A"]["gd0"]["内側"], "tab:blue"),
                               ("新C (内側の区間)", tuple(cfg["road"]["band_cpm"]), cal["新C"]["gd0"]["内側"], "tab:green")):
        m = (n >= band[0]) & (n <= band[1])
        ax.loglog(n[m], PV.gd(n[m], gd0), color=c,
                  label=f"{name}: Gd0={gd0:.2e} m³ (ISO {PV.iso_class(gd0)})、帯域 RMS {PV.band_rms(gd0, *band) * 1000:.1f} mm")
    for k, g in PV.ISO8608_GD0.items():
        ax.loglog(n, PV.gd(n, g), color="0.8", lw=0.6)
        ax.text(12, PV.gd(12, g), f" {k}", fontsize=8, color="0.5", va="center")
    ax.set_xlabel("空間周波数 n [cycles/m]")
    ax.set_ylabel("変位 PSD Gd(n) [m³]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("ISO 8608 の変位 PSD (灰 = クラス A〜E の幾何平均)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def plot_overlay(P: Pass, preds: dict, out: Path, role: str):
    import matplotlib.pyplot as plt

    colors = {"旧・単発": "0.45", "旧・荒れ": "tab:orange", "新A": "tab:blue", "新B": "tab:purple", "新C": "tab:green"}
    d = P.d[P.win]
    fig, axs = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
    for ax, k, band, lab in ((axs[0], "heave", HEAVE_BAND, "上下加速度 [m/s²]"), (axs[1], "roll", RP_BAND, "ロール角速度 [deg/s]"),
                             (axs[2], "pitch", RP_BAND, "ピッチ角速度 [deg/s]")):
        ax.plot(d, F._bandpass(P.meas[k], *band, FS)[P.win], color="k", lw=1.2, label="実測")
        for name in MODELS:
            ax.plot(d, F._bandpass(preds[name][0][k], *band, FS)[P.win], color=colors[name], lw=0.8, alpha=0.75, label=name)
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8, ncol=6, loc="upper right")
    ax = axs[3]
    ax.plot(d, env_of(P.meas["heave"])[P.win], color="k", lw=1.6, label="実測")
    for name in MODELS:
        envs = np.array([env_of(pr["heave"])[P.win] for pr in preds[name]])
        ax.plot(d, np.median(envs, axis=0), color=colors[name], lw=1.0, label=f"{name} (乱数 {len(envs)} 組の中央値)")
        if len(envs) > 1:
            ax.fill_between(d, np.percentile(envs, 10, axis=0), np.percentile(envs, 90, axis=0), color=colors[name], alpha=0.12)
    ax.set_ylabel("上下 8〜15 Hz の包絡線 [m/s²]", fontsize=9)
    ax.set_xlabel("交差点の入口 (前輪) からの距離 [m]")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    for a in axs:
        a.axvline(0, color="lime", lw=1)
        a.axvline(np.interp(P.exit, P.seg.t, P.d), color="red", lw=1)
    fig.suptitle(f"{P.drive} seg{P.segment} ({role})  {P.lane}車線  {P.speed_kmh:.0f} km/h"
                 f"{'  ジャイロ 10 Hz のためロール・ピッチは参考' if not P.gyro_ok else ''}"
                 f"{'  セグメント末尾で途切れ' if P.truncated else ''}\n波形の線は乱数の 1 例 (位相は調整していない)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=95)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs/vibration_pavement.yaml")
    ap.add_argument("--fullcar", type=Path, default=REPO / "configs/vibration_fullcar.yaml")
    ap.add_argument("--vib-config", type=Path, default=REPO / "configs/vibration.yaml")
    ap.add_argument("--data-root", type=Path, default=REPO / "raw_data/Chunk_1")
    ap.add_argument("--out", type=Path, default=REPO / "out/vibration/pavement")
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    vib = dict(yaml.safe_load(args.vib_config.read_text()))
    vib["grid_hz"] = FS
    p = FC.build_params(FC.load_config(args.fullcar))
    matplotlib.use("Agg")
    configure_japanese_font()
    n_seeds = args.seeds or cfg["road"]["seeds"]
    seeds = list(range(n_seeds))
    cal_seeds = seeds[: max(4, n_seeds // 2)]

    refs = find_segments(args.data_root)
    passes = [Pass(r, refs, vib) for _, r in pd.read_csv(args.out / "site19_passes.csv").iterrows()]
    cal_inner = next(P for P in passes if [P.drive, P.segment] == cfg["analysis"]["calibration_drive"])
    cal_curb = next(P for P in passes if P.drive.endswith("2018-08-17--14-55-39") and P.segment == 9)
    cal = {"内側": cal_inner, "縁石側": cal_curb}
    calib = calibrate(cal, p, cfg, cal_seeds)

    # 物理的な妥当性: 区間内の路面高さ RMS (帯域つき)、ISO クラス、タイヤの浮き
    plaus = {}
    for name in ("新A", "新B", "新C"):
        c = calib[name]
        plaus[name] = {lane: dict(gd0=g, iso=PV.iso_class(g), band_rms_mm=round(PV.band_rms(g, *cfg["road"]["band_cpm"]) * 1000, 2))
                       for lane, g in c["gd0"].items()}
    plaus["旧・荒れ"] = {lane: dict(gd0=g, iso=PV.iso_class(g), band_rms_mm=round(PV.band_rms(g, 0.05, 10.0) * 1000, 2))
                       for lane, g in calib["旧・荒れ"]["gd0"].items()}
    plaus["旧・単発"] = {lane: dict(position_m=b[0], height_mm=round(b[1] * 1000, 1)) for lane, b in calib["旧・単発"]["bump"].items()}
    (args.out / "calibration.json").write_text(json.dumps(dict(calibration=calib, plausibility=plaus,
                                                                calibration_drives={k: P.tag for k, P in cal.items()},
                                                                seeds_calibration=cal_seeds, seeds_validation=seeds),
                                                           ensure_ascii=False, indent=2, default=float))
    print(json.dumps(plaus, ensure_ascii=False, indent=1, default=float))

    plot_road(calib, cfg, p, args.out / "road_models.png")

    rows = []
    for P in passes:
        role = "校正" if P in cal.values() else "検証"
        preds = {}
        for name in MODELS:
            sds = [0] if name == "旧・単発" else seeds
            preds[name] = [simulate(P, p, build(name, calib, cfg, sd)) for sd in sds]
            for sd, pr in zip(sds, preds[name]):
                rows.append(dict(drive=P.drive, segment=P.segment, tag=P.tag, role=role, lane=P.lane,
                                 speed_kmh=round(P.speed_kmh, 1), gyro_ok=P.gyro_ok, truncated=P.truncated,
                                 model=name, seed=sd, **metrics(P, pr)))
        plot_overlay(P, preds, args.out / f"overlay_{P.tag.replace(' ', '_')}.png", role)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "validation_runs.csv", index=False)

    # 走行ごと (乱数の中央値) → モデルごとの検証走行の要約
    per = df.groupby(["model", "tag", "role", "lane"]).median(numeric_only=True).reset_index()
    per.to_csv(args.out / "validation_per_pass.csv", index=False)
    val = per[per.role == "検証"]
    err_cols = [c for c in per.columns if c.endswith("_logerr")]
    summ = val.groupby("model").agg(
        **{f"|{c}|": (c, lambda x: float(np.nanmedian(np.abs(x)))) for c in err_cols},
        env_r2=("env_r2", "median"), env_corr=("env_corr", "median"),
        heave_r2=("heave_r2", "median"), roll_rms_ratio=("roll_rms_ratio", "median"),
        onset_err_m=("onset_err_m", lambda x: float(np.nanmedian(np.abs(x)))),
        end_err_m=("end_err_m", lambda x: float(np.nanmedian(np.abs(x)))),
        lift_off_ms=("lift_off_ms", "max")).reindex(MODELS)
    # 路面の高さを妥当な上限 (帯域 RMS max_road_rms_mm) に抑えた場合、8〜15 Hz の上下振動は実測の何倍になるか。
    # 応答は路面の振幅にほぼ比例する (タイヤが浮かない範囲) ので、校正値に対する縮小率を掛けて見積もる
    cap = cfg["analysis"]["max_road_rms_mm"] / 1000
    capped = []
    for name, band in (("旧・荒れ", (0.05, 10.0)), ("新A", tuple(cfg["road"]["band_cpm"])),
                       ("新B", tuple(cfg["road"]["band_cpm"])), ("新C", tuple(cfg["road"]["band_cpm"]))):
        for lane, g in calib[name]["gd0"].items():
            f = min(1.0, cap / PV.band_rms(g, *band))
            v = per[(per.model == name) & (per.lane == lane)]
            capped.append(dict(model=name, lane=lane, band_rms_mm=round(PV.band_rms(g, *band) * 1000, 1),
                               scale_to_cap=round(f, 3),
                               heave_8_15_ratio_calibrated=round(float(np.nanmedian(10 ** v["heave_8-15_logerr"])), 3),
                               heave_8_15_ratio_at_cap=round(float(np.nanmedian(10 ** v["heave_8-15_logerr"])) * f, 3)))
    for lane, (sb, h) in calib["旧・単発"]["bump"].items():
        f = min(1.0, 0.01 / abs(h))             # 単発は高さ 1 cm を上限の目安にする
        v = per[(per.model == "旧・単発") & (per.lane == lane)]
        capped.append(dict(model="旧・単発", lane=lane, band_rms_mm=np.nan, scale_to_cap=round(f, 3),
                           heave_8_15_ratio_calibrated=round(float(np.nanmedian(10 ** v["heave_8-15_logerr"])), 3),
                           heave_8_15_ratio_at_cap=round(float(np.nanmedian(10 ** v["heave_8-15_logerr"])) * f, 3)))
    pd.DataFrame(capped).to_csv(args.out / "amplitude_at_plausible_height.csv", index=False)
    print(pd.DataFrame(capped).to_string(index=False))
    summ.to_csv(args.out / "validation_summary.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(summ.round(3).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
