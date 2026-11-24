#!/usr/bin/env python3
"""ブレーキフェードの検討の週次報告 (2026-W47、4 枚) の図を作る。ネットワークは使わない。

  uv run --offline python scripts/report/brake_fade_slides_figs.py --out reports/weekly/2026-W47/brake_fade_fig

作る図 (どれも 9.0 × 3.9 in。build_weekly_pptx.py の「図を全幅、注釈を下」の枠に合わせた):
  fig1_datasets.png    使うデータセット 3 つの比較 (○△×)。熱負荷の値は結果ファイルから読む。説明の文は docs/ の調査記録から手で書く
  fig2_temp_eff.png    左: 効きの定義の式 / 右: NCST 台上試験のブレーキ温度と効き (20261118_r9_force_vs_temp)
  fig3_ncst_road.png   NCST 高速道路の走行の 130〜162 分: ブレーキ温度・圧力・地図から推定した勾配 (20261119_r22_ncst_route_scene)
  fig4_udacity.png     Udacity HMB_2: 前方映像の 1 こま、車速・ブレーキトルク・熱の指標、下りのエネルギーの行き先 (20261119_r18_udacity_hmb2_scene)
数値はすべて結果ファイルから読む (図に手で数値を書かない)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams["font.family"] = ["YuGothic", "Hiragino Sans", "DejaVu Sans"]
NAVY, BLUE, SKY, PALE, STEEL, MUTED, TEXT = "#1F2A44", "#2F6FB0", "#3E9BD6", "#EAF2FB", "#8A9BB0", "#6B7280", "#222222"
RED, ORANGE = "#C0392B", "#E08A2E"
FIGSIZE = (9.0, 3.9)
DPI = 250

FORCE = Path("out/brake_fade/20261118_r9_force_vs_temp")
ROUTE = Path("out/brake_fade/20261119_r22_ncst_route_scene")
UDA = Path("out/brake_fade/20261119_r18_udacity_hmb2_scene")
UDA_ALL = Path("out/brake_fade")      # 20261119_r16_udacity_HMB_*_heat
RISK = Path("out/brake_fade/20261119_r31_fade_risk_index")   # フェードリスク指標 (fade_risk_index.py)
RISK_HIGH, RISK_MED = 60, 30


def risk_panel(ax, x, score):
    """フェードリスク (0〜100) の線と、High・Medium の帯。"""
    ax.axhspan(RISK_HIGH, 100, color="#FADBD8", lw=0)
    ax.axhspan(RISK_MED, RISK_HIGH, color="#FDEBD0", lw=0)
    ax.plot(x, score, color=NAVY, lw=1.3)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 30, 60, 100])
    ax.text(1.005, (RISK_HIGH + 100) / 200, "High", transform=ax.transAxes, fontsize=6.5, color=RED, va="center")
    ax.text(1.005, (RISK_MED + RISK_HIGH) / 200, "Med", transform=ax.transAxes, fontsize=6.5, color=ORANGE, va="center")
C2K = Path("out/brake_performance/20261118_r8_rav4")


def save(fig, path):
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("->", path)


def fig1(out: Path):
    ud = json.loads((UDA / "summary.json").read_text())
    c2 = json.loads((C2K / "summary.json").read_text())["q5_heat"]
    ref = ud["heat"]["ref_stop_Jkg"]
    ud_max = max(json.loads(p.read_text())["heat"]["E300_max_Jkg"] for p in UDA_ALL.glob("20261119_r16_udacity_HMB_*_heat/summary.json"))
    route = json.loads((ROUTE / "summary.json").read_text())
    cols = ["NCST (UC Davis ほか)", "Udacity CH2_002", "comma2k19"]
    rows = [
        ("車両・走行", ["大型トラック 29.5 t\n台上試験 (CBD・UDDS) + 路上", "乗用車 (Lincoln MKZ)\n峠越えを含む一般道 約 30 分", "乗用車 (RAV4 2017)\n通勤の高速道路・一般道"]),
        ("ブレーキの信号", [("○", "エア圧 (psi)"), ("△", "ペダル・ブレーキトルク (Nm)\nトルクの意味は要確認"), ("△", "0x224 の連続値 (単位不明)\nペダルは on/off")]),
        ("ブレーキ温度", [("○", f"熱電対 (路上の最高 {route['T_at']['max']:.0f} ℃)"), ("×", "なし"), ("×", "なし")]),
        ("勾配・姿勢", [("△", "IMU なし、GPS 高度は無効\n地図の標高で推定"), ("○", "IMU (Xsens)・GPS・気圧"), ("○", "IMU・GNSS の位置と速度")]),
        ("前方映像", [("×", "なし"), ("○", "前方カメラ 20 Hz"), ("○", "前方カメラ 20 Hz")]),
        ("熱負荷 (直前 300 s)", [("○", "高 (温度で実測)"), ("△", f"低: 最大 {ud_max / ref:.1f} 回分"), ("△", f"低: 最大 {c2['E300_max_in_ref_stops']:.1f} 回分")]),
        ("利用条件", ["CC0", "MIT", "MIT"]),
    ]
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 9.0)
    ax.set_ylim(0, 3.9)
    ax.axis("off")
    x0, wl, wc = 0.05, 1.55, 2.45
    top, hh = 3.83, 0.36
    rh = (top - hh - 0.08) / len(rows)
    sel = {0: "温度と効きの評価", 1: "映像付きの走行場面", 2: "映像付きの走行場面"}
    for j, c in enumerate(cols):
        x = x0 + wl + j * wc
        ax.add_patch(FancyBboxPatch((x + 0.03, top - hh), wc - 0.06, hh, boxstyle="round,pad=0,rounding_size=0.05", fc=NAVY if j == 0 else BLUE, ec="none"))
        ax.text(x + wc / 2, top - hh / 2 + 0.06, c, ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
        ax.text(x + wc / 2, top - hh / 2 - 0.1, f"用途: {sel[j]}", ha="center", va="center", color="white", fontsize=7.5)
    for i, (name, vals) in enumerate(rows):
        y = top - hh - 0.04 - (i + 0.5) * rh
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((x0, y - rh / 2), wl + 3 * wc, rh, fc=PALE, ec="none", zorder=-1))
        ax.text(x0 + 0.08, y, name, ha="left", va="center", fontsize=9, color=NAVY, fontweight="bold")
        for j, v in enumerate(vals):
            x = x0 + wl + j * wc
            if isinstance(v, tuple):
                mark, txt = v
                col = {"○": BLUE, "△": ORANGE, "×": RED}[mark]
                ax.text(x + 0.22, y, mark, ha="center", va="center", fontsize=13, color=col, fontweight="bold")
                ax.text(x + 0.42, y, txt, ha="left", va="center", fontsize=7.6, color=TEXT, linespacing=1.15)
            else:
                ax.text(x + 0.12, y, v, ha="left", va="center", fontsize=7.8, color=TEXT, linespacing=1.15)
    save(fig, out / "fig1_datasets.png")


def fig2(out: Path):
    E = pd.read_csv(FORCE / "brake_events_force.csv")
    S = json.loads((FORCE / "summary.json").read_text())["all"]
    fig = plt.figure(figsize=FIGSIZE)
    # 左: 定義の式
    a0 = fig.add_axes([0.0, 0.0, 0.40, 1.0])
    a0.axis("off")
    a0.add_patch(FancyBboxPatch((0.04, 0.06), 0.92, 0.88, boxstyle="round,pad=0,rounding_size=0.03", fc=PALE, ec="none", transform=a0.transAxes))
    a0.text(0.09, 0.86, "ブレーキの効きの定義 (台上試験)", fontsize=10.5, color=NAVY, fontweight="bold", transform=a0.transAxes)
    a0.text(0.09, 0.71, r"$F_b = F_{tire} - F_{0}$", fontsize=13, color=TEXT, transform=a0.transAxes)
    a0.text(0.09, 0.62, "動力計で測ったタイヤの力から、惰行の力を引く", fontsize=7.8, color=MUTED, transform=a0.transAxes)
    a0.text(0.09, 0.47, r"$\eta = \dfrac{F_b}{p - p_0}\ \propto\ \mu(T)$", fontsize=13, color=TEXT, transform=a0.transAxes)
    a0.text(0.09, 0.36, "p: ブレーキのエア圧、p₀: 押し出し圧 (1.2 psi)\n同じ圧力で出る力 = 摩擦係数 μ に比例", fontsize=7.8, color=MUTED, transform=a0.transAxes, linespacing=1.3)
    a0.text(0.09, 0.20, "縦軸は 50 ℃ 未満の制動の中央値で割った比\nη < 1 なら、同じ踏み方で効きが弱い", fontsize=7.8, color=TEXT, transform=a0.transAxes, linespacing=1.3)
    # 右: 散布図
    ax = fig.add_axes([0.48, 0.15, 0.50, 0.78])
    grp = {"CBD0": ("CBD の暖機", STEEL), "CBD1": ("CBD", SKY), "CBDx3": ("CBD × 3", BLUE), "UDDS": ("UDDS × 3 (速い)", ORANGE)}
    for k, (lab, c) in grp.items():
        g = E[E.traj == k]
        ax.scatter(g["T"], g.rel, s=16 if k != "UDDS" else 22, color=c, alpha=0.85, edgecolor="white", linewidth=0.4, label=f"{lab} {len(g)} 回", zorder=3)
    B = pd.DataFrame(S["bins"])
    ax.errorbar(B.T_mid, B["median"], yerr=[B["median"] - B.ci5, B.ci95 - B["median"]], fmt="s-", color=NAVY, ms=5, capsize=3, lw=1.6,
                label="温度区間の中央値 (90% 区間)", zorder=4)
    ax.axhline(1, color=STEEL, lw=0.8)
    ax.axvspan(130, 170, color="#FBE5D6", alpha=0.6, lw=0, zorder=0)
    hot = B[B.lo >= 130]
    ax.text(143, 1.52, f"130 ℃ 以上: 中央値 {S['median_rel_130_plus']:.2f}\n(130 ℃ 未満 {S['median_rel_below_130']:.2f})", ha="center", fontsize=8, color=RED)
    ax.set_xlim(10, 170)
    ax.set_ylim(0, 1.75)
    ax.set_xlabel("実測のブレーキ温度 [℃] (ライニングの熱電対)", fontsize=8.5)
    ax.set_ylabel("効き η (50 ℃ 未満 = 1)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left", ncol=2, frameon=True)
    save(fig, out / "fig2_temp_eff.png")


def fig3(out: Path):
    G = pd.read_csv(ROUTE / "timeline_grade.csv")
    S = json.loads((ROUTE / "summary.json").read_text())
    Rk = pd.read_csv(RISK / "ncst_fwy_score.csv")
    fig, ax = plt.subplots(4, 1, figsize=FIGSIZE, sharex=True, gridspec_kw=dict(height_ratios=[1.2, 0.8, 0.8, 0.9], hspace=0.14, left=0.085, right=0.86, top=0.97, bottom=0.12))
    for d in S["descents"]:
        for a in ax:
            a.axvspan(*d["phase_min"], color=PALE, lw=0, zorder=0)
    ax[0].plot(G["min"], G["T"], color=RED, lw=1.6)
    imax = G["T"].idxmax()
    ax[0].annotate(f"最高 {G['T'][imax]:.0f} ℃", (G["min"][imax], G["T"][imax]), xytext=(-75, -22), textcoords="offset points", fontsize=8.5, color=RED,
                   arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax[0].set_ylabel("ブレーキ温度\n[℃]", fontsize=8)
    ax[0].set_ylim(70, 185)
    ax[1].fill_between(G["min"], 0, G.psi, color=BLUE, lw=0)
    ax[1].set_ylabel("ブレーキ圧\n[psi]", fontsize=8)
    a1 = ax[1].twinx()
    a1.plot(G["min"], G.v, ".", color=ORANGE, ms=2)
    a1.set_ylabel("車速 [km/h]\n(記録がある所)", fontsize=7.5, color=ORANGE)
    a1.set_ylim(0, 120)
    a1.tick_params(labelsize=7)
    ax[2].plot(G["min"], G.grade * 100, color=NAVY, lw=1.3)
    ax[2].axhline(0, color=STEEL, lw=0.6)
    ax[2].set_ylabel("勾配 [%]\n(地図推定)", fontsize=7.5)
    risk_panel(ax[3], Rk.minute, Rk.score)
    ax[3].set_ylabel("フェード\nリスク", fontsize=7.5)
    km = Rk.score.idxmax()
    ax[3].annotate(f"最大 {Rk.score[km]:.0f} (High)", (Rk.minute[km], Rk.score[km]), xytext=(8, -14), textcoords="offset points", fontsize=7.5, color=RED)
    ax[3].set_xlabel("走り始めからの時間 [分]", fontsize=8.5)
    for a in ax:
        a.tick_params(labelsize=7.5)
        a.grid(alpha=0.3)
    for k, d in enumerate(S["descents"]):
        ax[0].text(np.mean(d["phase_min"]), 165, f"下り {k + 1}: {d['drop_m']:.0f} m・平均 {d['mean_grade_pct']:.1f}%", ha="center", fontsize=7.8, color=NAVY)
    ax[2].text(S["last_fix"]["min"] - 0.2, -3.0, "GPS が切れる →\n(以降は推定)", fontsize=6.5, color=MUTED, ha="right", va="center")
    ax[2].axvline(S["last_fix"]["min"], color=MUTED, lw=0.7, ls="--")
    ax[0].set_xlim(130, 162)
    save(fig, out / "fig3_ncst_road.png")


def fig4(out: Path):
    from PIL import Image

    S = json.loads((UDA / "summary.json").read_text())
    D = pd.read_csv(UDA / "timeseries_5hz.csv.gz")
    w0, w1 = S["video_window_s"]
    W = D[(D.t >= w0) & (D.t <= w1)]
    t_rep = S["representative_moment"]["t"]
    fig = plt.figure(figsize=FIGSIZE)
    ai = fig.add_axes([0.0, 0.08, 0.39, 0.84])
    ai.imshow(Image.open(UDA / "frame_representative.jpg"))
    ai.axis("off")
    r = S["representative_moment"]
    ai.set_title(f"{t_rep:.1f} s: {r['v'] * 3.6:.0f} km/h・勾配 {np.degrees(r['theta']):.1f}°・トルク {r['Tb']:.0f} Nm", fontsize=8, color=NAVY)
    Rk = pd.read_csv(RISK / "udacity_hmb2_score.csv")
    Rk = Rk[(Rk.t >= w0) & (Rk.t <= w1)]
    panels = [("車速\n[km/h]", W.v * 3.6, NAVY), ("ブレーキ\nトルク [Nm]", W.Tb, BLUE), ("熱の指標 E300\n[J/kg]", W.E300, RED), ("フェード\nリスク", None, None)]
    for i, (lab, y, c) in enumerate(panels):
        a = fig.add_axes([0.47, 0.765 - i * 0.215, 0.30, 0.17])
        if y is None:
            risk_panel(a, Rk.t, Rk.score)
            a.axvline(t_rep, color=ORANGE, lw=1)
            a.set_ylabel(lab, fontsize=7)
            a.tick_params(labelsize=6.5)
            a.grid(alpha=0.3)
            a.set_xlabel("時刻 [s]", fontsize=7)
            kr = Rk.score.idxmax()
            a.annotate(f"最大 {Rk.score[kr]:.0f} (Medium)", (Rk.t[kr], Rk.score[kr]), xytext=(6, 12), textcoords="offset points", fontsize=6.5, color=ORANGE, ha="right")
            continue
        a.plot(W.t, y, color=c, lw=1)
        a.axvline(t_rep, color=ORANGE, lw=1)
        a.set_ylabel(lab, fontsize=7)
        a.tick_params(labelsize=6.5)
        a.grid(alpha=0.3)
        if i == 2:
            a.axhline(S["heat"]["ref_stop_Jkg"], color=STEEL, ls="--", lw=0.8)
            a.set_ylim(0, S["heat"]["ref_stop_Jkg"] * 1.35)
            a.text(w0, S["heat"]["ref_stop_Jkg"], " 100 km/h 停止 1 回分", fontsize=6.5, va="bottom", color=MUTED)
        a.set_xticklabels([])
    # 下りのエネルギーの行き先
    e = S["descent_energy"]
    parts = [("ブレーキ", e["brake_Jkg"], RED), ("駆動側の負のトルク\n(エンジンブレーキ等)", e["powertrain_negative_Jkg"], BLUE),
             ("空気・転がりの抵抗", e["resistance_Jkg"], SKY)]
    tot = e["potential_Jkg"]
    ab = fig.add_axes([0.84, 0.12, 0.05, 0.76])
    bottom = 0
    for name, val, c in parts:
        ab.bar(0, val / tot * 100, bottom=bottom, color=c, width=0.8, edgecolor="white")
        ab.text(0.55, bottom + val / tot * 50, f"{name}\n{val / tot * 100:.0f}%", va="center", fontsize=7, color=TEXT, linespacing=1.1)
        bottom += val / tot * 100
    ab.set_xlim(-0.5, 0.5)
    ab.set_ylim(0, 100)
    ab.set_xticks([])
    ab.tick_params(labelsize=6.5)
    ab.set_title(f"下り {e['drop_m']:.0f} m の\n位置エネルギーの行き先", fontsize=7.5, color=NAVY)
    save(fig, out / "fig4_udacity.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig1(a.out)
    fig2(a.out)
    fig3(a.out)
    fig4(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
