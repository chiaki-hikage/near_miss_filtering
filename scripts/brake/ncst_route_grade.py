"""NCST の高速道路の走行で、GPS が切れた後 (136.7 分〜) の勾配を、公開地図の経路の標高から推定し、運転の局面と並べる。

入力: ncst_route_fetch.py が取得した経路 (OpenStreetMap、ODbL) と標高 (USGS 3DEP、公有)、
      ncst_scene_illustration.load() の 1 s ごとの表 (ペダル・エンジン・ブレーキ圧・温度・局面)。
推定の前提 (確かめられない):
  - GPS が切れた後も I-15 を南へ走った (直前 8 分の方位が一定で、出発点の Riverside へ戻る道)
  - 距離と時刻の対応: 下りの区間で車速の記録が約 5% しか無いので、目印で区切って区間ごとに一定の速さとみなす
    (GPS が切れた点 = 距離 0、運転の局面の「下り坂 (推定)」2 本の始まり・終わり = 地図の下りの区間 2 本の始まり・終わり)
    目印の間の速さと、まばらに残る車速の記録を比べて確かめる (summary.json の speed_check)
出力: 勾配の分布 (route_profile.csv)、時刻ごとの推定位置と勾配 (timeline_grade.csv)、図、動画、summary.json

    uv run python scripts/brake/ncst_route_grade.py --run-id <新しい ID> [--video]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
import yaml

from near_miss.plotting import configure_japanese_font

import ncst_scene_illustration as SI

OUT = Path("out/brake_fade")
CFG = Path("configs/brake/ncst_route.yaml")


def movavg(x, n):
    return pd.Series(x).rolling(max(1, int(n)), center=True, min_periods=1).mean().to_numpy()


def heading(lat, lon):
    p1, p2 = np.radians(lat[:-1]), np.radians(lat[1:])
    dl = np.radians(lon[1:] - lon[:-1])
    b = np.degrees(np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)))
    return np.r_[b, b[-1]]


def profile(cfg) -> pd.DataFrame:
    d = Path(cfg["data_dir"])
    R = pd.read_csv(d / "route_points.csv")
    E = pd.read_csv(d / "usgs_epqs_elevation.csv")
    R["key"] = list(zip(R.lat.round(6), R.lon.round(6)))
    E["key"] = list(zip(E.lat.round(6), E.lon.round(6)))
    R = R.merge(E[["key", "elev_m", "resolution_m"]], on="key", how="left").drop(columns="key")
    step = float(R.s_m.diff().median())
    P = cfg["profile"]
    R["elev_s"] = movavg(R.elev_m.interpolate(), P["smooth_m"] / step)
    k = int(round(P["grade_ds_m"] / 2 / step))
    e = R.elev_s.to_numpy()
    g = np.full(len(e), np.nan)
    g[k:-k] = (e[2 * k:] - e[:-2 * k]) / (R.s_m.to_numpy()[2 * k:] - R.s_m.to_numpy()[:-2 * k])
    R["grade"] = g
    hd = np.unwrap(np.radians(heading(R.lat.to_numpy(), R.lon.to_numpy())))
    curv = np.gradient(movavg(hd, 200 / step), R.s_m)       # rad/m (左曲がりが負)
    R["radius_m"] = 1 / np.maximum(np.abs(curv), 1e-6)
    return R


def landmarks(R, cfg):
    """峠 (前半の最も高い点) と、峠より後の下りの区間 [(始まり, 終わり), ...] の添字。"""
    A = cfg["align"]
    step = float(R.s_m.diff().median())
    i_top = int(R.elev_s.iloc[: int(len(R) * 0.6)].idxmax())
    g1k = movavg(R.grade.to_numpy(), 1000 / step)
    steep = (g1k < A["descent_grade_pct"] / 100) & (np.arange(len(R)) >= i_top)
    runs, k = [], 0
    while k < len(steep):
        if steep[k]:
            j = k
            while j + 1 < len(steep) and steep[j + 1]:
                j += 1
            if (j - k) * step >= A["descent_min_km"] * 1000:
                runs.append((k, j))
            k = j + 1
        else:
            k += 1
    runs = runs[: len(A["descent_phases_min"])]
    runs[0] = (i_top, runs[0][1])  # 最初の下りは峠から
    return i_top, runs


def align(g, R, fix_min, runs, A):
    tk = [fix_min]
    sk = [0.0]
    for (t0, t1), (i0, i1) in zip(A["descent_phases_min"], runs):
        tk += [t0, t1]
        sk += [float(R.s_m.iat[i0]), float(R.s_m.iat[i1])]
    tk, sk = np.array(tk), np.array(sk)
    m = g["min"].to_numpy()
    s = np.where((m >= tk[0]) & (m <= tk[-1]), np.interp(m, tk, sk), np.nan)
    G = g.copy()
    G["s_est_m"] = s
    for c in ("elev_s", "grade", "radius_m", "lat", "lon"):
        G[c] = np.where(np.isfinite(s), np.interp(np.nan_to_num(s), R.s_m, R[c]), np.nan)
    seg_v = np.diff(sk) / (np.diff(tk) * 60) * 3.6
    checks = []
    for j in range(len(tk) - 1):
        w = (m >= tk[j]) & (m < tk[j + 1])
        n = int(g.v[w].notna().sum())
        checks.append(dict(min=[float(tk[j]), float(tk[j + 1])], km=[round(sk[j] / 1000, 2), round(sk[j + 1] / 1000, 2)],
                           implied_kmh=round(float(seg_v[j]), 1),
                           recorded_kmh_median=round(float(np.nanmedian(g.v[w])), 1) if n else None, recorded_n=n,
                           recorded_kmh_p10_p90=[round(float(np.nanpercentile(g.v[w], q)), 1) for q in (10, 90)] if n > 5 else None))
    return G, checks, (tk, sk)


def profile_plot(R, G, i_top, runs, knots, path):
    configure_japanese_font()
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 0.8, 1, 1]))
    km = R.s_m / 1000
    ax[0].plot(km, R.elev_m, color="#bbb", lw=0.6, label="標高 (USGS 3DEP、50 m ごと)")
    ax[0].plot(km, R.elev_s, color="#2e86c1", label="移動平均 300 m")
    for (i0, i1) in runs:
        for x in ax:
            x.axvspan(km.iat[i0], km.iat[i1], color="#aed6f1", alpha=0.35, lw=0)
    for t, s_ in zip(*knots):
        ax[0].axvline(s_ / 1000, color="#555", ls="--", lw=0.7)
        ax[0].text(s_ / 1000, R.elev_s.max(), f" {t:.1f} 分", fontsize=7.5, va="top", rotation=90)
    ax[0].text(km.iat[i_top], R.elev_s.iat[i_top], f" 峠 {R.elev_s.iat[i_top]:.0f} m", fontsize=8, va="bottom")
    ax[0].set_ylabel("標高 [m]")
    ax[0].legend(fontsize=8, loc="lower left")
    ax[1].plot(km, R.grade * 100, color="#c0392b")
    ax[1].axhline(0, color="#888", lw=0.6)
    ax[1].set_ylabel("勾配 [%]\n上りが正")
    W = G[np.isfinite(G.s_est_m)]
    wk = W.s_est_m / 1000
    ax[2].plot(wk, W["T"], color="#922b21", lw=1.5, label="ブレーキ温度 [°C] (実測)")
    ax[2].set_ylabel("°C")
    a2 = ax[2].twinx()
    a2.fill_between(wk, 0, W.psi, color="#e74c3c", alpha=0.45, lw=0)
    a2.set_ylabel("ブレーキ圧 [psi]", color="#e74c3c")
    ax[2].legend(fontsize=8, loc="upper left")
    for ph, c in SI.PHASE_COL.items():
        mm = W.phase == ph
        ax[3].scatter(wk[mm], W.acc[mm], s=4, color=c if ph != "不明" else "#999", label=f"アクセル [%] ({ph})")
    ax[3].plot(wk, W.rpm / 22, color="#555", lw=0.6, label="エンジン回転数 / 22")
    ax[3].set_ylabel("%")
    ax[3].legend(fontsize=7, loc="upper right", ncol=2)
    ax[3].set_xlabel("GPS が切れた点からの I-15 南行きの距離 [km] (経路は推定)")
    for x in ax:
        x.grid(alpha=0.3)
    sec = ax[0].secondary_xaxis("top", functions=(lambda k: np.interp(k, wk, W["min"]), lambda mm: np.interp(mm, W["min"], wk)))
    sec.set_xlabel("推定の時刻 [分] (区間ごとに一定の速さとみなした対応)")
    sec.set_xticks(np.arange(np.ceil(W["min"].min() / 2.5) * 2.5, W["min"].max() + 0.01, 2.5))
    fig.suptitle("NCST 高速道路の走行: GPS が切れた後の経路を I-15 南行き (Cajon Pass) と推定し、地図の標高から勾配を出した", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


RANK_COL = {"High": "#C0392B", "Medium": "#E08A2E", "Low": "#8A9BB0"}


def frame(fig, G, R, k, start, end, i_view_end, title):
    fig.clf()
    has_risk = "risk_score" in G
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 0.75, 1.0], width_ratios=[1.35, 1.0], hspace=0.38, wspace=0.18)
    r = G.iloc[k]
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-8, 8)
    ax.set_ylim(-3, 4.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(patches.Rectangle((-8, -3), 16, 7.3, fc=SI.PHASE_COL[r.phase], ec="none", zorder=-5))
    tilt = np.degrees(np.arctan(r.grade)) if np.isfinite(r.grade) else 0.0
    SI.draw_truck(ax, tilt, r["T"], r.psi > 5, bool(r.engine_brake))
    ax.set_title(f"{title}   {r['min']:.1f} 分   局面: {r.phase}", fontsize=11)
    gtxt = f"勾配 {r.grade * 100:+.1f}% ({tilt:+.1f}°、地図から推定、傾きは実際の角度)" if np.isfinite(r.grade) else "勾配: 経路の推定の範囲外 (模式なし)"
    ax.text(-7.8, -2.85, gtxt, fontsize=8, color="#333")
    if has_risk:  # フェードリスク (fade_risk_index.py): 札と、順位の色の枠
        rk = r.risk_rank if isinstance(r.risk_rank, str) else None
        col = RANK_COL.get(rk, "#BBBBBB")
        txt = f"フェードリスク {r.risk_score:.0f}  {rk}" if rk else "フェードリスク: 算出なし (勾配の推定なし)"
        ax.add_patch(patches.Rectangle((-8, -3), 16, 7.3, fill=False, lw=5, ec=col, zorder=5))
        risk_badge = (txt, col, bool(rk))
    axg = fig.add_subplot(gs[0, 1])
    items = [("車速", r.v, 120, "km/h"), ("エンジン回転数", r.rpm, 2200, "rpm"), ("アクセル", r.acc, 100, "%"),
             ("エンジントルク", r.tq, 100, "%"), ("ブレーキ圧", r.psi, 30, "psi"), ("ブレーキ温度", r["T"], 170, "°C")]
    for i, (name, val, vmax, unit) in enumerate(items[::-1]):
        axg.barh(i, 1.0, color="#f0f0f0", height=0.6)
        if np.isfinite(val):
            c = SI.TCMAP(SI.TNORM(val)) if name == "ブレーキ温度" else ("#e74c3c" if name == "ブレーキ圧" else "#2e86c1")
            axg.barh(i, min(val / vmax, 1.0), color=c, height=0.6)
            txt = f"{val:.0f} {unit}"
        else:
            txt = "記録なし"
        axg.text(-0.02, i, name, ha="right", va="center", fontsize=9)
        axg.text(1.02, i, txt, ha="left", va="center", fontsize=9)
    axg.set_xlim(0, 1.35)
    axg.set_ylim(-0.6, len(items) - 0.4)
    axg.axis("off")
    if has_risk:  # 札は計器の上に置く (トラックの絵の文字と重ねない)
        txt, col, big = risk_badge
        axg.text(0.5, 1.06, txt, transform=axg.transAxes, ha="center", va="bottom", fontsize=13 if big else 9, color="white", fontweight="bold",
                 bbox=dict(facecolor=col, alpha=0.92, lw=0, boxstyle="round,pad=0.35"))
    # 標高の断面 (縦を強調) と現在地
    axp = fig.add_subplot(gs[1, :])
    km = R.s_m / 1000
    axp.fill_between(km, R.elev_s.min() - 30, R.elev_s, color="#d5dbdb")
    axp.plot(km, R.elev_s, color="#5d6d7e", lw=1)
    if np.isfinite(r.s_est_m):
        axp.plot([r.s_est_m / 1000], [r.elev_s], marker="o", ms=9, color=SI.TCMAP(SI.TNORM(r["T"])), mec="#111")
    axp.set_xlim(0, km.iat[min(len(km) - 1, i_view_end)])
    axp.set_ylabel("標高 [m]", fontsize=8)
    axp.set_xlabel("I-15 南行きの距離 [km] (推定の経路。縦を強調)", fontsize=8)
    axp.tick_params(labelsize=7)
    axp.grid(alpha=0.3)
    axt = fig.add_subplot(gs[2, :])
    w = (G["min"] >= start) & (G["min"] <= end)
    m = G["min"][w].to_numpy()
    axt.plot(m, G["T"][w], color="#922b21", lw=1.6, label="ブレーキ温度 [°C]")
    axt.set_ylim(40, 175)
    a2 = axt.twinx()
    a2.fill_between(m, 0, G.psi[w], color="#e74c3c", alpha=0.5, lw=0, label="ブレーキ圧 [psi]")
    a2.plot(m, G.grade[w] * 100 * 3 + 30, color="#2e86c1", lw=1.2, label="勾配 (推定、0% = 30、1% = 3)")
    a2.axhline(30, color="#2e86c1", lw=0.5, ls=":")
    a2.set_ylim(0, 60)
    if has_risk:  # 時系列の上の帯に、フェードリスクの順位の色を塗る
        rr = G.risk_rank[w].to_numpy()
        for rk, col in RANK_COL.items():
            axt.fill_between(m, 166, 175, where=(rr == rk), color=col, lw=0, step="mid")
        axt.text(start, 175.5, " 上の帯 = フェードリスクの順位 (赤 High・橙 Medium・灰 Low・白 算出なし)", fontsize=7, va="bottom", color="#444")
    axt.axvline(r["min"], color="k", lw=1.2)
    axt.set_xlim(start, end)
    axt.set_xlabel("走り始めからの時間 [分]")
    h1, l1 = axt.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    axt.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left", ncol=3, bbox_to_anchor=(0, 0.93) if has_risk else None)
    axt.grid(alpha=0.25)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--start-min", type=float, default=130)
    ap.add_argument("--end-min", type=float, default=162)
    ap.add_argument("--step-s", type=int, default=5)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--storyboard-min", type=float, nargs="*", default=[138, 142.5, 149, 156])
    ap.add_argument("--risk-run", default=None, help="fade_risk_index.py の run (例 20261119_r31_fade_risk_index)。動画と絵コンテにフェードリスクを重ねる")
    a = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text())
    out = OUT / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    fix = json.loads((Path(cfg["data_dir"]) / "last_fix.json").read_text())
    R = profile(cfg)
    i_top, runs = landmarks(R, cfg)
    g, info = SI.load(cfg["logger_file"], cfg["hem_file"])
    G, checks, knots = align(g, R, fix["min"], runs, cfg["align"])
    if a.risk_run:
        Rk = pd.read_csv(OUT / a.risk_run / "ncst_fwy_score.csv")[["minute", "score", "rank"]]
        Rk["key"] = Rk.minute.round(4)
        G["key"] = G["min"].round(4)
        G = G.merge(Rk[["key", "score", "rank"]].rename(columns=dict(score="risk_score", rank="risk_rank")), on="key", how="left").drop(columns="key")
        G.loc[G.risk_score.isna(), "risk_rank"] = np.nan
    i_end = runs[-1][1]
    i_view = min(len(R) - 1, i_end + int(4000 / float(R.s_m.diff().median())))
    R.to_csv(out / "route_profile.csv", index=False)
    G[(G["min"] >= a.start_min) & (G["min"] <= a.end_min)].to_csv(out / "timeline_grade.csv", index=False)
    seg = R.iloc[i_top:i_end + 1]
    steep = seg.grade < -0.04
    in_desc = np.zeros(len(R), bool)
    for i0, i1 in runs:
        in_desc[i0:i1 + 1] = True
    W = G[np.isfinite(G.s_est_m)]
    by_phase = {}
    for ph, d in W.groupby("phase"):
        by_phase[ph] = dict(minutes=round(len(d) / 60, 1), grade_pct_median=round(float(d.grade.median() * 100), 2),
                            grade_pct_p10_p90=[round(float(np.nanpercentile(d.grade, q) * 100), 2) for q in (10, 90)])
    S = dict(assumption="GPS が切れた後も I-15 を南へ (Cajon Pass) 走った。距離と時刻は目印で区切った区間ごとの一定の速さで対応", last_fix=fix,
             hem_lag_s=info["hem_lag_s"], route_km=round(float(R.s_m.iat[-1] / 1000), 2), elevation_resolution_m=sorted(R.resolution_m.dropna().unique().tolist()),
             summit=dict(km=round(float(R.s_m.iat[i_top] / 1000), 2), elev_m=round(float(R.elev_s.iat[i_top]), 1),
                         lat=round(float(R.lat.iat[i_top]), 5), lon=round(float(R.lon.iat[i_top]), 5)),
             descents=[dict(km=[round(float(R.s_m.iat[i0] / 1000), 2), round(float(R.s_m.iat[i1] / 1000), 2)],
                            elev_m=[round(float(R.elev_s.iat[i0]), 1), round(float(R.elev_s.iat[i1]), 1)],
                            drop_m=round(float(R.elev_s.iat[i0] - R.elev_s.iat[i1]), 1),
                            mean_grade_pct=round(float((R.elev_s.iat[i1] - R.elev_s.iat[i0]) / (R.s_m.iat[i1] - R.s_m.iat[i0]) * 100), 2),
                            min_grade_pct=round(float(R.grade.iloc[i0:i1 + 1].min() * 100), 2),
                            end_latlon=[round(float(R.lat.iat[i1]), 5), round(float(R.lon.iat[i1]), 5)], phase_min=p)
                       for (i0, i1), p in zip(runs, cfg["align"]["descent_phases_min"])],
             summit_to_last_descent_end=dict(drop_m=round(float(R.elev_s.iat[i_top] - R.elev_s.iat[i_end]), 1),
                                             length_km=round(float((R.s_m.iat[i_end] - R.s_m.iat[i_top]) / 1000), 2),
                                             km_steeper_than_4pct=round(float(steep.sum() * R.s_m.diff().median() / 1000), 2),
                                             km_in_descents=round(float(in_desc.sum() * R.s_m.diff().median() / 1000), 2)),
             speed_check=checks, grade_by_phase=by_phase,
             T_at=dict(**{f"{t:g}min": round(float(G.loc[(G["min"] - t).abs().idxmin(), "T"]), 1) for t in knots[0]},
                       max=round(float(W["T"].max()), 1), max_min=round(float(W["min"][W["T"].idxmax()]), 1)))
    profile_plot(R, G, i_top, runs, knots, out / "route_profile.png")
    configure_japanese_font()
    fig = plt.figure(figsize=(12, 7.4))
    for i, mm in enumerate(a.storyboard_min):
        k = int(np.argmin(np.abs(G["min"] - mm)))
        frame(fig, G, R, k, a.start_min, a.end_min, i_view, f"場面 {i + 1}")
        fig.savefig(out / f"storyboard_{i + 1}_{mm:g}min.png", dpi=110)
    if a.video:
        fdir = out / "_frames"
        fdir.mkdir(exist_ok=True)
        ks = np.where((G["min"] >= a.start_min) & (G["min"] <= a.end_min))[0][:: a.step_s]
        for n, k in enumerate(ks):
            frame(fig, G, R, k, a.start_min, a.end_min, i_view, f"{a.step_s * a.fps}倍速")
            fig.savefig(fdir / f"f{n:05d}.png", dpi=90)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(fdir / "f%05d.png"), "-vf",
                        "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "22", str(out / "scene.mp4")], check=True)
        for p in fdir.glob("*.png"):
            p.unlink()
        fdir.rmdir()
    plt.close(fig)
    (out / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    (out / "meta.json").write_text(json.dumps(dict(vars(a), config=CFG.as_posix(), cfg=cfg,
                                                   sources=dict(osm="OpenStreetMap contributors (ODbL), Overpass API",
                                                                elevation="USGS 3DEP via Elevation Point Query Service (public domain)")),
                                              ensure_ascii=False, indent=1, default=str))
    print(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
