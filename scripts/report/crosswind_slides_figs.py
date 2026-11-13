#!/usr/bin/env python3
"""横風外乱の探索の進捗報告 (2026-11-13、4 枚) の図を作る。ネットワークは使わない。

  uv run --offline --extra viz python scripts/report/crosswind_slides_figs.py --out reports/weekly/2026-W46/crosswind_fig

作る図 (どれも 8 × 4.65 in。build_weekly_pptx.py の「図 + 右に注釈」の枠に合わせた):
  fig1_datasets.png   データセットの比較 (○△×)。値は docs/ の調査記録と、nuScenes は公開仕様 (論文・devkit)
  fig2_approach.png   横風を受ける車とヨー残差の定義 / 観測所の風から横風成分を推定する流れ / 全体の流れと件数
  fig3_case.png       E0096・E0097 (000765) の前方カメラ、値、ヨーとヨー残差の時系列
数値は Phase 2 r7・Phase 3 r4 の結果ファイルから読む (図に手で数値を書かない。説明の文だけ手で書く)。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

plt.rcParams["font.family"] = ["YuGothic", "Hiragino Sans", "DejaVu Sans"]   # スライドの Yu Gothic UI に近いもの
NAVY, ACCENT, MUTED, TEXT = "#1F2A44", "#2F6FB0", "#6B7280", "#222222"
C_ACT, C_PRED, C_WIND, C_FORCE = "#1F2A44", "#2F6FB0", "#3E9BD6", "#E08A2E"   # 紺・青・水色が基調。強調だけ赤・オレンジ
PALE, STEEL, EMPH = "#EAF2FB", "#8A9BB0", "#C0392B"
SCR = Path("out/zod/full_screening/20261113_r7")
WTH = Path("out/zod/full_weather/20261113_r4")
RAW = Path("raw_data/zod")
FIGSIZE = (9.0, 3.9)    # 4:3 のスライドで、図を全幅に置き、注釈を下に置く枠
DPI = 250


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def arrow(ax, p0, p1, color, lw=2.5, ms=14, ls="-", z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, ls=ls, zorder=z))


# ------------------------------------------------------------------ 図 1
def fig1(out: Path):
    cols = ["comma2k19", "nuScenes", "NVIDIA\nPhysicalAI-AV", "ZOD\n(Zenseact)"]
    rows = [
        ("走行環境\n規模", [("", "米国 I-280 の約 20 km\n高速道路・約 33 h"), ("", "ボストン・シンガポール\n市街地・15 h (平均 16 km/h)"),
                          ("", "25 か国\n約 1,700 h"), ("", "欧州 6 か国\n走行全体 約 170 h")]),
        ("車両", [("", "RAV4・Civic"), ("", "Renault Zoe 2 台"), ("", "Hyperion 8 / 8.1 搭載車"), ("", "4 台 (識別名のみ)")]),
        ("操舵角\n操舵トルク", [("○", "舵角 (CAN)"), ("○", "舵角 100 Hz\n(CAN 拡張)"), ("×", "無し"), ("◎", "舵角・トルク 100 Hz")]),
        ("ヨーレート\nIMU", [("○", "CAN・ジャイロ"), ("○", "IMU 100 Hz"), ("△", "推定の自車運動\nから計算"), ("◎", "OXTS (RTK) 100 Hz\n+ 車両の IMU")]),
        ("位置・時刻", [("○", "GNSS・日時"), ("△", "地図上の相対位置"), ("×", "国・月・時間帯のみ"), ("○", "緯度経度・UTC")]),
        ("風の観測との\n照合", [("△", "空港 1 地点 (SFO)"), ("–", "未確認"), ("×", "できない"), ("◎", "DWD・SMHI 10 分値\n半径 50 km に複数")]),
        ("横風解析への\n適合", [("△", "解析済み: 横風と\n言える事象なし"), ("×", "低速の市街地中心\n非商用ライセンス"),
                           ("×", "舵角・位置が無い"), ("◎", "採用")]),
    ]
    mark_col = {"◎": "#1F2A44", "○": "#2F6FB0", "△": "#6FA8DC", "×": "#9AA5B4", "–": "#C3CBD6"}
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    x0, lw_ = 1, 14
    cw = (99 - x0 - lw_) / 4
    top, hh = 99, 9.5
    rh = (top - hh - 7) / len(rows)
    zx = x0 + lw_ + 3 * cw
    ax.add_patch(FancyBboxPatch((zx + 0.3, 6.5), cw - 0.6, top - 6.5 - 0.3, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=PALE, ec=ACCENT, lw=2, zorder=0))
    for j, c in enumerate(cols):
        cx = x0 + lw_ + (j + 0.5) * cw
        ax.text(cx, top - hh / 2, c, ha="center", va="center", fontsize=9.5, fontweight="bold", color=NAVY if j < 3 else ACCENT)
    ax.plot([x0, 99], [top - hh, top - hh], color=NAVY, lw=1.2)
    for i, (lab, cells) in enumerate(rows):
        yc = top - hh - (i + 0.5) * rh
        if i < len(rows) - 1:
            ax.plot([x0, 99], [top - hh - (i + 1) * rh] * 2, color="#D1D5DB", lw=0.6)
        ax.text(x0 + 0.5, yc, lab, ha="left", va="center", fontsize=8, color=MUTED, linespacing=1.15)
        for j, (m, t) in enumerate(cells):
            cx = x0 + lw_ + j * cw
            last = i == len(rows) - 1
            if m:
                ax.text(cx + 1.8, yc, m, ha="left", va="center", fontsize=13 if not last else 14, color=mark_col[m], fontweight="bold")
                ax.text(cx + 6.2, yc, t, ha="left", va="center", fontsize=7.6, color=TEXT, linespacing=1.15,
                        fontweight="bold" if (last and j == 3) else "normal")
            else:
                ax.text(cx + cw / 2, yc, t, ha="center", va="center", fontsize=7.6, color=TEXT, linespacing=1.15)
    ax.text(x0, 2.5, "◎ 適する  ○ 使える  △ 制約あり  × 使えない  – 未確認。"
            "comma2k19・NVIDIA・ZOD はデータで確認、nuScenes は公開仕様 (論文・devkit) のみ",
            fontsize=6.8, color=MUTED, va="center")
    fig.savefig(out / "fig1_datasets.png", dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------ 図 2
def car(ax, cx, cy, w, h, angle=0):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="#DCE6F2", ec=NAVY, lw=1.5, zorder=4))
    ax.add_patch(Rectangle((cx - w * 0.35, cy + h * 0.12), w * 0.7, h * 0.16, fc="#AEB8C4", ec="none", zorder=5))


def inside_check(fig, ax, items, box, name):
    """items (文字・図形) が枠 box = (x0, y0, x1, y1) (データ座標) の中に収まっているかを確かめる。はみ出しは名前を出す。"""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for it in items:
        bb = it.get_window_extent(fig.canvas.get_renderer())
        (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        if x0 < box[0] - 0.3 or y0 < box[1] - 0.3 or x1 > box[2] + 0.3 or y1 > box[3] + 0.3:
            bad.append(f"{getattr(it, 'get_text', lambda: type(it).__name__)()!r} ({x0:.1f},{y0:.1f})-({x1:.1f},{y1:.1f})")
    if bad:
        raise RuntimeError(f"{name}: 枠からはみ出す: " + "; ".join(bad))


def fig2(out: Path, summ):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    X, Y = 180, 180 * FIGSIZE[1] / FIGSIZE[0]
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, X); ax.set_ylim(0, Y); ax.axis("off")
    yb, yt = 19, Y - 1                      # 2 つの枠の下端・上端
    L_, R_ = (1, yb, 88, yt), (92, yb, X - 1, yt)
    left, right = [], []
    T = lambda grp, *a, **k: grp.append(ax.text(*a, **k))  # noqa: E731
    P = lambda grp, patch: grp.append(ax.add_patch(patch))  # noqa: E731
    for b in (L_, R_):
        ax.add_patch(FancyBboxPatch((b[0], b[1]), b[2] - b[0], b[3] - b[1], boxstyle="round,pad=0,rounding_size=2", fc="#F4F8FC", ec="#C9D6E6", lw=1))
    # 左: 車両運動
    T(left, 3, yt - 4, "① 車両運動: 舵角から予測したヨーと実測のずれ", fontsize=8.8, fontweight="bold", color=NAVY, va="center")
    cx, cy = 27, 40
    for dx in (-8, 8):
        ax.plot([cx + dx, cx + dx], [yb + 3, yt - 12], color="#9CA3AF", lw=1, ls=(0, (6, 4)))
    car(ax, cx, cy, 7.5, 13)
    for yy in (34, 40.5, 47):
        arrow(ax, (4, yy + 1.5), (cx - 5.5, yy), C_WIND, lw=2.2, ms=11)
    T(left, 4, 52, "横風", color=C_WIND, fontsize=8, fontweight="bold")
    arrow(ax, (cx + 4.2, cy), (cx + 12, cy), C_FORCE, lw=3, ms=13)
    T(left, cx + 9, cy - 4.5, "右へ押す力", color=C_FORCE, fontsize=7.2)
    yy = np.linspace(0, 1, 30)
    ax.plot(cx - 10 * yy ** 2, cy + 7 + 15 * yy, color=C_PRED, lw=2.2, ls="--")
    ax.plot(cx + 0 * yy, cy + 7 + 15 * yy, color=C_ACT, lw=2.2)
    T(left, cx - 9, cy + 24.5, "舵角からの予測", color=C_PRED, fontsize=6.6, ha="right", va="center")
    T(left, cx + 1.5, cy + 24.5, "実際", color=C_ACT, fontsize=6.6, ha="left", va="center")
    hx, hy, R = cx, yb + 7, 3.8
    ax.add_patch(Circle((hx, hy), R, fc="none", ec=NAVY, lw=2))
    th = np.radians(25)
    for a in (np.pi / 2 + th, np.pi + th, -th):
        ax.plot([hx, hx + R * np.cos(a)], [hy, hy + R * np.sin(a)], color=NAVY, lw=1.7)
    ax.add_patch(Circle((hx + R * np.cos(np.pi / 2 + th), hy + R * np.sin(np.pi / 2 + th)), 0.75, color=C_FORCE, zorder=6))
    T(left, hx + 9, hy, "左へ当て舵", fontsize=7.2, color=C_PRED, va="center")
    ex = 47
    T(left, ex, yt - 11, "予測ヨーレート (単軌道モデル)", fontsize=7.2, color=MUTED, va="center")
    T(left, ex, yt - 17.5, r"$r_{pred}=\dfrac{v\,(\delta-c)}{a+b\,v^{2}}$", fontsize=10.5, color=C_PRED, va="center")
    T(left, ex, yt - 25.5, "ヨー残差 (外乱の指標)", fontsize=7.2, color=MUTED, va="center")
    T(left, ex, yt - 30.5, r"$\Delta r = r - r_{pred}$", fontsize=10.5, color=NAVY, va="center")
    T(left, ex, yt - 35, "δ: 舵角  v: 速さ  r: 実測 (OXTS)\nc: 舵角のゼロ点 (走行ごとに補正)\na, b: 車両ごとに別の走行で推定", fontsize=6.2, color=MUTED, va="top", linespacing=1.25)
    T(left, ex, yb + 2, "Δr < 0 が続く\n→ 右へ押す外力\n→ 左からの横風を期待", fontsize=7.2, color=NAVY, va="bottom", linespacing=1.3, fontweight="bold")
    # 右: 気象
    T(right, 94, yt - 4, "② 気象: 周りの観測所の風から横風成分を推定", fontsize=8.8, fontweight="bold", color=NAVY, va="center")
    mx, my, rc = 116, 41, 18
    P(right, Circle((mx, my), rc, fc="none", ec="#B8C7DA", lw=0.8, ls=(0, (4, 3))))
    T(right, mx, my - rc - 1.2, "半径 50 km・最大 4 か所", fontsize=6.2, color=MUTED, ha="center", va="top")
    ax.plot([mx - 2.5, mx + 2.5], [my - 16, my + 16], color="#9CA3AF", lw=5, solid_capstyle="round", alpha=0.5)
    car(ax, mx, my, 3, 5.5)
    arrow(ax, (mx, my + 2.8), (mx + 0.7, my + 8), NAVY, lw=1.5, ms=8)
    T(right, mx + 1.6, my + 8, "方位 H", fontsize=6.2, color=NAVY, va="center")
    for dx, dy, dkm in [(-12, 9, 11), (10, 10, 21), (-9, -10, 25), (12, -8, 32)]:
        sx, sy = mx + dx, my + dy
        ax.plot([mx, sx], [my, sy], color="#C7CDD4", lw=0.8, zorder=1)
        ax.add_patch(Rectangle((sx - 1.2, sy - 1.2), 2.4, 2.4, fc=NAVY, ec="none", zorder=3))
        arrow(ax, (sx - 4.2, sy + 0.5), (sx - 1.4, sy + 0.1), C_WIND, lw=1.5, ms=7)
        T(right, sx + 1.6, sy - 2.8, f"{dkm} km", fontsize=5.8, color=MUTED)
    T(right, 94.5, yt - 10.5, "■ 観測所 (DWD・SMHI)\n   10 分平均の風速・風向", fontsize=6.2, color=NAVY, va="top", linespacing=1.2)
    fx, fw = 139, X - 1 - 139 - 2
    steps = [("距離の重みで補間 (IDW)", r"$\vec W=\sum w_i \vec W_i,\ w_i \propto 1/d_i^{2}$"),
             ("車の方位 H で横風成分に", r"$W_{left}=-W\sin(D-H)$"),
             ("不確かさ", "観測所間のばらつき + 1 つ抜きの誤差")]
    bh, top0 = 13, yt - 9
    for k, (h, f) in enumerate(steps):
        y1 = top0 - k * (bh + 3.5)
        P(right, FancyBboxPatch((fx, y1 - bh), fw, bh, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec="#C9D6E6", lw=0.8))
        T(right, fx + 1.5, y1 - 3.3, h, fontsize=6.2, color=MUTED, va="center")
        T(right, fx + 1.5, y1 - 9, f, fontsize=7.4 if "$" in f else 5.9, color=C_WIND if k < 2 else TEXT, va="center")
        if k < 2:
            arrow(ax, (fx + fw / 2, y1 - bh - 0.3), (fx + fw / 2, y1 - bh - 3.2), MUTED, lw=1.2, ms=7)
    # 下: 全体の流れ
    flow = [("ZOD Sequences", "1,473 本・177 走行"), ("80 km/h 超・直進", f"OXTS で約 {summ['screen_h']:.1f} 時間"),
            ("ヨー残差の閾値超過", "外乱の候補を抽出"), ("横風と照合", f"候補 {summ['n']} 件"),
            ("判定", f"支持 {summ['支持']}・反証 {summ['反証']}・不明 {summ['不明']}")]
    gap = 3.4
    bw = (X - 2 - 4 * gap) / 5
    for k, (h, s) in enumerate(flow):
        x = 1 + k * (bw + gap)
        last = k == len(flow) - 1
        ax.add_patch(FancyBboxPatch((x, 1), bw, 13, boxstyle="round,pad=0,rounding_size=1.5", fc=NAVY if last else PALE, ec=NAVY if last else ACCENT, lw=1.2))
        ax.text(x + bw / 2, 10, h, ha="center", va="center", fontsize=7.3, fontweight="bold", color="white" if last else NAVY)
        ax.text(x + bw / 2, 4.8, s, ha="center", va="center", fontsize=7, color="white" if last else TEXT)
        if not last:
            arrow(ax, (x + bw + 0.2, 7.5), (x + bw + gap - 0.2, 7.5), NAVY, lw=1.5, ms=9)
    ax.text(X / 2, 16.5, "①の外乱の向きと ②の横風の向き・大きさを、候補ごとに照合する", ha="center", va="center", fontsize=7.2, color=NAVY, fontweight="bold")
    inside_check(fig, ax, left, L_, "図 2 ①")
    inside_check(fig, ax, right, R_, "図 2 ②")
    fig.savefig(out / "fig2_approach.png", dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------ 図 3
def fig3(out: Path):
    w = pd.read_csv(WTH / "candidates_weather.csv").set_index("event_id")
    u = pd.read_csv(SCR / "candidates_unique.csv").set_index("event_id")
    cal = pd.read_csv(SCR / "offset_calib_by_drive.csv").set_index("drive_id")
    did = "sierra_20220409T112552"
    z = np.load(SCR / "residuals" / f"{did}.npz")
    c = float(cal.loc[did, "c_car_deg"] + cal.loc[did, "delta_c_deg"])
    t = z["t"]
    vals = {}
    for e in ("E0096", "E0097"):
        a0, a1 = ts(u.loc[e, "t_exceed_start_utc"]), ts(u.loc[e, "t_exceed_end_utc"])
        m = (t >= a0) & (t <= max(a1, a0 + 2))
        vals[e] = dict(a0=a0, a1=a1, steer=float(np.mean(z["steer_deg"][m]) - c), r=float(np.mean(z["r_deg_s"][m])),
                       rp=float(np.mean(z["r_pred_cor"][m])), dr=float(np.mean(z["dr_cor"][m])), tq=w.loc[e, "torque_anom_nm"])
    thr = float(u.loc["E0096", "threshold_deg_s"])
    wl, sg = float(w.loc["E0096", "idw_w_left"]), float(w.loc["E0096", "w_left_sigma"])
    wsp = float(w.loc["E0096", "idw_speed"])
    info = json.loads((RAW / "sequences" / "000765" / "info.json").read_text())
    cf = info["camera_frames"]["front_blur"]
    ct = np.array([ts(f["time"]) for f in cf])
    c0, c1 = ct[0], ct[-1]
    tf = 0.5 * (vals["E0097"]["a0"] + vals["E0097"]["a1"])
    k = int(np.argmin(np.abs(ct - tf)))
    from PIL import Image
    im = Image.open(RAW / cf[k]["filepath"]); im.draft("RGB", (im.width // 2, im.height // 2)); im = im.convert("RGB")
    W_, H_ = im.size
    im = im.crop((int(W_ * 0.06), int(H_ * 0.10), int(W_ * 0.94), int(H_ * 0.76)))

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    axc = fig.add_axes([0.005, 0.455, 0.50, 0.54])
    axc.imshow(im); axc.axis("off"); axc.set_anchor("N")
    axc.text(0.015, 0.97, f"E0097 の閾値超過区間の中 ({datetime.fromtimestamp(ct[k], timezone.utc):%H:%M:%S} UTC)\n2022-04-09 ドイツ北部・北向き・約 120 km/h",
             transform=axc.transAxes, fontsize=6.6, color="white", va="top", bbox=dict(fc="black", alpha=0.55, lw=0, pad=2.5))
    axc.annotate("", xy=(0.30, 0.42), xytext=(0.04, 0.42), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color=C_WIND, lw=3, mutation_scale=18))
    axc.text(0.04, 0.47, f"推定横風 左から {wl:.1f} ± {sg:.1f} m/s", transform=axc.transAxes, color="white", fontsize=7, fontweight="bold",
             bbox=dict(fc=C_WIND, alpha=0.9, lw=0, pad=2))
    axc.text(0.985, 0.03, "風上 (西) 側は開けた畑。並走する大型車・橋は無い", transform=axc.transAxes, fontsize=6.2, color="white",
             ha="right", va="bottom", bbox=dict(fc="black", alpha=0.55, lw=0, pad=2))
    # 値の表
    axt = fig.add_axes([0.53, 0.455, 0.465, 0.54]); axt.axis("off"); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    axt.text(0, 0.96, "閾値超過区間の平均 (左を正)", fontsize=8, fontweight="bold", color=NAVY, va="top")
    hdr = ["", "E0096", "E0097", ""]
    rows = [("推定横風 (10 m)", f"+{wl:.1f}", f"+{float(w.loc['E0097', 'idw_w_left']):.1f}", "m/s", C_WIND),
            ("舵角 (ゼロ点から)", f"{vals['E0096']['steer']:+.2f}", f"{vals['E0097']['steer']:+.2f}", "deg", C_PRED),
            ("予測ヨー r_pred", f"{vals['E0096']['rp']:+.2f}", f"{vals['E0097']['rp']:+.2f}", "deg/s", C_PRED),
            ("実測ヨー r", f"{vals['E0096']['r']:+.2f}", f"{vals['E0097']['r']:+.2f}", "deg/s", C_ACT),
            ("ヨー残差 Δr", f"{vals['E0096']['dr']:+.2f}", f"{vals['E0097']['dr']:+.2f}", "deg/s", EMPH),
            ("操舵トルク (中央値から)", f"{vals['E0096']['tq']:+.1f}", f"{vals['E0097']['tq']:+.1f}", "N·m", MUTED)]
    xs = [0, 0.56, 0.75, 0.92]
    y = 0.84
    for j, h in enumerate(hdr):
        axt.text(xs[j], y, h, fontsize=7.4, fontweight="bold", color=NAVY, ha="left" if j == 0 else "center", va="center")
    for lab, a, b, unit, col in rows:
        y -= 0.108
        axt.plot([0, 1], [y + 0.057, y + 0.057], color="#E5E7EB", lw=0.6)
        axt.text(xs[0], y, lab, fontsize=7.2, color=TEXT, va="center")
        axt.text(xs[1], y, a, fontsize=8.2, color=col, ha="center", va="center", fontweight="bold")
        axt.text(xs[2], y, b, fontsize=8.2, color=col, ha="center", va="center", fontweight="bold")
        axt.text(xs[3], y, unit, fontsize=6.4, color=MUTED, ha="center", va="center")
    axt.text(0, -0.02, "左へ余分に切っているのに、車はわずかに右へ回る\n→ 右へ押す外力を打ち消す当て舵と整合", fontsize=7.2, color=NAVY,
             fontweight="bold", va="bottom", linespacing=1.35)
    fig.text(0.012, 0.415, "E0096 と E0097 は、約 30 s 続いた 1 つの負のずれが閾値を 2 回越えたもの (独立した 2 事象ではない)",
             fontsize=7.2, color=NAVY, fontweight="bold", va="center")
    # 時系列
    t0w = ts("2022-04-09T13:51:32.700+00:00")
    m = (t >= t0w) & (t <= t0w + 40) & (z["blk"] == z["blk"][np.argmin(np.abs(t - vals["E0096"]["a0"]))])
    tt = t[m] - c0
    slow = pd.Series(z["dr_cor"][m].astype(float)).rolling(100, center=True, min_periods=1).mean().to_numpy()
    ax2 = fig.add_axes([0.065, 0.11, 0.925, 0.27])
    ax2.axvspan(0, c1 - c0, color="#F1F4F8", lw=0, zorder=0)
    ax2.text(0.3, 0.75, "前方カメラの 20 s", fontsize=6.2, color=MUTED, transform=ax2.get_xaxis_transform())
    for e in ("E0096", "E0097"):
        ax2.axvspan(vals[e]["a0"] - c0, vals[e]["a1"] - c0, color=C_WIND, alpha=0.22, lw=0)
        ax2.text(vals[e]["a0"] - c0 + 0.2, 0.06, e, fontsize=6.6, color=NAVY, transform=ax2.get_xaxis_transform(), fontweight="bold")
    ax2.plot(tt, z["dr_cor"][m], color="#A9C7E8", lw=0.8, label="ヨー残差 Δr")
    ax2.plot(tt, slow, color=NAVY, lw=1.8, label="持続成分 (5 s 平均)")
    ax2.axhline(-thr, color=EMPH, lw=1.0, ls="--", label=f"閾値 −{thr:.2f}")
    ax2.axhline(0, color="#9CA3AF", lw=0.6)
    ax2.set_ylabel("deg/s", fontsize=7)
    ax2.set_xlabel("カメラの開始からの秒 (OXTS は 50 s のみ記録)", fontsize=6.8)
    ax2.tick_params(labelsize=6.5)
    ax2.set_xlim(tt[0], tt[-1])
    ax2.legend(fontsize=6.3, loc="upper right", ncol=3, frameon=False)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    fig.savefig(out / "fig3_case.png", dpi=DPI)
    plt.close(fig)
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = json.loads((SCR / "summary.json").read_text())
    ws = json.loads((WTH / "summary.json").read_text())
    vd = {}
    for r in ws["verdicts"]:
        vd[r["verdict"]] = vd.get(r["verdict"], 0) + r["n"]
    summ = dict(screen_h=sum(x["screened_min_80"] for x in sc["screened"]) / 60, n=ws["n_candidates"], **vd)
    fig1(out)
    fig2(out, summ)
    vals = fig3(out)
    print(summ)
    print(json.dumps(vals, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
