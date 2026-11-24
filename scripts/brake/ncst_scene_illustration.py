"""NCST の路上の高速道路の走行 (DATA43 + HEM 1C161734) で、ブレーキ温度が上がった場面の運転をイラストにする。

映像も、車体の傾き (IMU) も無い。HEM (エンジン・アクセル・車速) の時計は、ブレーキ中にアクセルも踏んでいる時間が一番少なくなるずれで合わせる
(高速道路では 16 s で 0%。車速と圧力の対応による推定 16 s とも一致)。GPS の高度は使わない (128.5〜131.6 分に 2.5 m/s で上がっていて、
エンジンの最大出力で 29.5 t を持ち上げられる 1.7 m/s を超える。ありえない値)。
運転の局面は、エンジンの信号から判定する (判定であって、勾配の実測ではない):
  上り坂か加速    アクセル > 50% かつ エンジントルク > 50% が、前後 60 s の 70% 以上
  下り坂 (推定)   アクセル 0 かつ エンジントルク 0 かつ 回転数 > 1000 rpm (エンジンブレーキ) が、前後 120 s の 90% 以上
                  29.5 t の車がアクセルを離して長く高速を保つには、重力の押す力が要るため
  不明            それ以外
イラストの道の傾きは局面ごとの模式 (上り +4 deg、下り -4 deg)。実際の勾配ではない。
後輪のブレーキの色は実測のブレーキ温度 (熱電対の読み)、ブレーキの表示は圧力 > 5 psi。

    uv run python scripts/brake/ncst_scene_illustration.py --run-id <新しい ID> [--start-min 125 --end-min 165 --step-s 5]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors, patches, transforms
import numpy as np
import pandas as pd

from near_miss.plotting import configure_japanese_font

import ncst_thermal as NT

OUT = Path("out/brake_fade")
PHASE_COL = {"上り坂か加速": "#f5cba7", "下り坂 (推定)": "#aed6f1", "不明": "#eeeeee"}
TILT = {"上り坂か加速": 4.0, "下り坂 (推定)": -4.0, "不明": 0.0}
TNORM = colors.Normalize(40, 170)
TCMAP = matplotlib.colormaps["inferno"]


def pedal_lag(B: pd.DataFrame, H: pd.DataFrame, search: int = 90) -> tuple[int, float]:
    """HEM の時計のずれ (s)。ブレーキ (圧力 > 5 psi) 中にアクセル (> 5%) も踏んでいる時間が一番少ないずれ。
    アクセルとブレーキを同時に踏むことはまず無い、という前提。"""
    on = B.psi.to_numpy() > 5
    res = {}
    for lag in range(-search, search + 1):
        a = np.interp(B.t.to_numpy(), H.t.to_numpy() + lag, H.acc.to_numpy(), left=np.nan, right=np.nan)
        m = np.isfinite(a) & on
        if m.sum() < 50:
            continue
        res[lag] = float(np.mean(a[m] > 5))
    fmin = min(res.values())
    tied = sorted(k for k, f in res.items() if f == fmin)
    return tied[len(tied) // 2], fmin          # 同点なら、同点のずれの中央


def load(logger="DATA43.CSV", hem="1C161734.csv") -> tuple[pd.DataFrame, dict]:
    B = NT.read_logger(logger, "road")
    t0 = B.t.iat[0]
    L = open(NT.DATA / hem, encoding="latin1").read().splitlines()
    cols = L[2].split(",")
    rows = [r.split(",") for r in L[3:]]

    def col(c):
        i = cols.index(c)
        o = []
        for r in rows:
            try:
                o.append(float(r[i]))
            except (ValueError, IndexError):
                o.append(np.nan)
        return np.array(o)

    t = np.array([dt.datetime.strptime(r[0].strip(), "%m/%d/%y %H:%M:%S").timestamp() for r in rows])
    H = pd.DataFrame(dict(t=t, rpm=col("Engine Speed (rpm)"), acc=col("Accelerator Pedal Position 1 (%)"), tq=col("Actual Engine - Percent Torque (%)"),
                          v=col("Wheel-Based Vehicle Speed (kph)"))).groupby("t").first().reset_index()
    H.loc[H.v > 250, "v"] = np.nan
    lag, overlap = pedal_lag(B, H)
    H["t"] = H.t + lag
    g = pd.DataFrame(dict(t=np.arange(np.ceil(max(B.t.min(), H.t.min())), min(B.t.max(), H.t.max()), 1.0)))
    for k in ("rpm", "acc", "tq"):
        g[k] = np.interp(g.t, H.t, H[k])
    hv = H.dropna(subset=["v"])
    idx = np.searchsorted(hv.t.to_numpy(), g.t.to_numpy())
    idx = np.clip(idx, 1, len(hv) - 1)
    dist = np.minimum(np.abs(hv.t.to_numpy()[idx] - g.t), np.abs(hv.t.to_numpy()[idx - 1] - g.t))
    g["v"] = np.where(dist <= 3, np.interp(g.t, hv.t, hv.v), np.nan)      # 3 s より離れた車速は「記録なし」
    g["psi"] = np.interp(g.t, B.t, np.clip(B.psi - 0.012, 0, None))
    g["T"] = np.interp(g.t, B.t[B["T"].notna()], B["T"].dropna())
    g["min"] = (g.t - t0) / 60
    climb = ((g.acc > 50) & (g.tq > 50)).astype(float).rolling(61, center=True, min_periods=30).mean() >= 0.7
    desc = ((g.acc == 0) & (g.tq == 0) & (g.rpm > 1000)).astype(float).rolling(121, center=True, min_periods=60).mean() >= 0.9
    g["phase"] = np.where(desc, "下り坂 (推定)", np.where(climb, "上り坂か加速", "不明"))
    g["engine_brake"] = (g.acc == 0) & (g.tq == 0) & (g.rpm > 1000)
    return g, dict(t0=t0, hem_lag_s=lag, acc_during_brake_at_lag=overlap)


def draw_truck(ax, tilt_deg, T, braking, engine_brake):
    tr = transforms.Affine2D().rotate_deg_around(0, 0, tilt_deg) + ax.transData
    ax.plot([-8, 8], [-1.05, -1.05], color="#555", lw=3, transform=tr)          # 道
    ax.add_patch(patches.Rectangle((-5.2, 0), 7.4, 2.4, fc="#d5dbdb", ec="#333", lw=1.2, transform=tr))       # トレーラー
    ax.add_patch(patches.Rectangle((2.5, 0), 2.6, 2.0, fc="#5d6d7e", ec="#333", lw=1.2, transform=tr))         # トラクター
    ax.add_patch(patches.Rectangle((4.2, 1.0), 0.8, 0.7, fc="#d6eaf8", ec="#333", lw=0.8, transform=tr))       # 窓
    for x, rear in [(-4.3, False), (-3.3, False), (0.9, True), (1.9, True), (4.3, False)]:
        fc = TCMAP(TNORM(T)) if rear else "#222"
        ax.add_patch(patches.Circle((x, -0.45), 0.55, fc=fc, ec="#111", lw=1.2, transform=tr))
    if braking:
        ax.add_patch(patches.Rectangle((-5.45, 0.3), 0.25, 0.6, fc="#e74c3c", ec="none", transform=tr))
    ax.text(1.4, -1.9, f"後輪のブレーキ {T:.0f} °C", ha="center", fontsize=9, color=TCMAP(TNORM(min(T, 150))), transform=tr, fontweight="bold")
    lab = []
    if braking:
        lab.append("ブレーキを踏んでいる")
    if engine_brake:
        lab.append("アクセル 0・エンジンブレーキ")
    ax.text(0, 3.6, "  /  ".join(lab) if lab else "", ha="center", fontsize=10, color="#c0392b" if braking else "#1f618d")


def frame(fig, g, k, start, end, title):
    fig.clf()
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 0.05, 1.0], width_ratios=[1.35, 1.0], hspace=0.35, wspace=0.18)
    r = g.iloc[k]
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-8, 8)
    ax.set_ylim(-3, 4.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(PHASE_COL[r.phase])
    ax.add_patch(patches.Rectangle((-8, -3), 16, 7.3, fc=PHASE_COL[r.phase], ec="none", zorder=-5))
    draw_truck(ax, TILT[r.phase], r["T"], r.psi > 5, bool(r.engine_brake))
    ax.set_title(f"{title}   {r['min']:.1f} 分   局面: {r.phase}", fontsize=11)
    ax.text(-7.8, -2.85, "道の傾きは局面ごとの模式 (実測の勾配ではない)", fontsize=7.5, color="#555")
    # 計器
    axg = fig.add_subplot(gs[0, 1])
    items = [("車速", r.v, 120, "km/h"), ("エンジン回転数", r.rpm, 2200, "rpm"), ("アクセル", r.acc, 100, "%"),
             ("エンジントルク", r.tq, 100, "%"), ("ブレーキ圧", r.psi, 30, "psi"), ("ブレーキ温度", r["T"], 170, "°C")]
    for i, (name, val, vmax, unit) in enumerate(items[::-1]):
        y = i
        axg.barh(y, 1.0, color="#f0f0f0", height=0.6)
        if np.isfinite(val):
            c = TCMAP(TNORM(val)) if name == "ブレーキ温度" else ("#e74c3c" if name == "ブレーキ圧" else "#2e86c1")
            axg.barh(y, min(val / vmax, 1.0), color=c, height=0.6)
            txt = f"{val:.0f} {unit}"
        else:
            txt = "記録なし"
        axg.text(-0.02, y, name, ha="right", va="center", fontsize=9)
        axg.text(1.02, y, txt, ha="left", va="center", fontsize=9)
    axg.set_xlim(0, 1.35)
    axg.set_ylim(-0.6, len(items) - 0.4)
    axg.axis("off")
    # 時系列
    axt = fig.add_subplot(gs[2, :])
    w = (g["min"] >= start) & (g["min"] <= end)
    m = g["min"][w].to_numpy()
    ph = g.phase[w].to_numpy()
    s = 0
    for i in range(1, len(ph) + 1):
        if i == len(ph) or ph[i] != ph[s]:
            axt.axvspan(m[s], m[i - 1], color=PHASE_COL[ph[s]], lw=0)
            s = i
    axt.plot(m, g["T"][w], color="#922b21", lw=1.6, label="ブレーキ温度 [°C] (左軸)")
    axt.set_ylim(40, 175)
    axt.set_ylabel("°C")
    ax2 = axt.twinx()
    ax2.fill_between(m, 0, g.psi[w], color="#e74c3c", alpha=0.5, lw=0, label="ブレーキ圧 [psi] (右軸)")
    ax2.plot(m, g.acc[w] / 100 * 30, color="#2e86c1", lw=0.5, alpha=0.6, label="アクセル (右軸、100% = 30)")
    ax2.set_ylim(0, 60)
    ax2.set_ylabel("psi")
    axt.axvline(r["min"], color="k", lw=1.2)
    axt.set_xlim(start, end)
    axt.set_xlabel("走り始めからの時間 [分]")
    h1, l1 = axt.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    hp = [patches.Patch(color=c, label=k) for k, c in PHASE_COL.items()]
    axt.legend(h1 + h2 + hp, l1 + l2 + list(PHASE_COL), fontsize=7.5, loc="upper left", ncol=3)
    axt.grid(alpha=0.25)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--start-min", type=float, default=125)
    ap.add_argument("--end-min", type=float, default=165)
    ap.add_argument("--step-s", type=int, default=5)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--storyboard-min", type=float, nargs="*", default=[130, 141.5, 147, 156])
    a = ap.parse_args()
    out = OUT / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    configure_japanese_font()
    g, info = load()
    w = (g["min"] >= a.start_min) & (g["min"] <= a.end_min)
    # 局面ごとの集計
    S = dict(hem_clock_lag_s=info["hem_lag_s"], acc_during_brake_frac_at_lag=round(info["acc_during_brake_at_lag"], 4))
    for ph, d in g[w].groupby("phase"):
        S[ph] = dict(minutes=round(len(d) / 60, 1), brake_on_frac=round(float((d.psi > 5).mean()), 3), acc_mean=round(float(d.acc.mean()), 1),
                     rpm_median=round(float(d.rpm.median())), v_median_when_recorded=round(float(d.v.median()), 1) if d.v.notna().any() else None,
                     T_change=round(float(d["T"].iloc[-1] - d["T"].iloc[0]), 1))
    desc = g[w & (g.phase == "下り坂 (推定)")]
    if len(desc):
        S["descent_window_min"] = [round(float(desc["min"].min()), 1), round(float(desc["min"].max()), 1)]
        S["descent_brake_applications"] = int(((desc.psi > 5) & ~(desc.psi.shift() > 5)).sum())
        S["descent_T_start_end_max"] = [round(float(desc["T"].iloc[0]), 1), round(float(desc["T"].iloc[-1]), 1), round(float(desc["T"].max()), 1)]
    S["T_max_in_window"] = [round(float(g["T"][w].max()), 1), round(float(g["min"][w][g["T"][w].idxmax()]), 1)]
    # 絵コンテ
    fig = plt.figure(figsize=(12, 6.2))
    for i, mm in enumerate(a.storyboard_min):
        k = int(np.argmin(np.abs(g["min"] - mm)))
        frame(fig, g, k, a.start_min, a.end_min, f"場面 {i + 1}")
        fig.savefig(out / f"storyboard_{i + 1}_{mm:g}min.png", dpi=110)
    # 動画
    fdir = out / "_frames"
    fdir.mkdir(exist_ok=True)
    ks = np.where(w)[0][:: a.step_s]
    for n, k in enumerate(ks):
        frame(fig, g, k, a.start_min, a.end_min, f"{a.step_s * a.fps}倍速")
        fig.savefig(fdir / f"f{n:05d}.png", dpi=90)
    plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(fdir / "f%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "22", str(out / "scene.mp4")], check=True)
    for p in fdir.glob("*.png"):
        p.unlink()
    fdir.rmdir()
    g[w].to_csv(out / "timeline.csv", index=False)
    (out / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    (out / "meta.json").write_text(json.dumps(dict(vars(a), tilt_deg_illustrative=TILT), ensure_ascii=False, indent=1, default=str))
    print(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
