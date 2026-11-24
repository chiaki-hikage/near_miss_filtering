#!/usr/bin/env python3
"""TartanDrive 2.0 (オフロードの異常振動) の週次報告 (2026-W48、2 枚) の図を作る。新しい取得はしない (画像は bag_cache・tar のキャッシュから)。

  uv run python scripts/report/td2_slides_figs.py --out reports/weekly/2026-W48/td2_fig

作る図 (9.0 × 3.9 in。build_weekly_pptx.py の「図を全幅、注釈を下」の枠に合わせた):
  fig1_overview.png  左: データセットの概要のカード (目的・出典・場所・時期・時間・信号と映像の有無)
                     右: 異常振動の候補 124 件の段 (A/B/C) の内訳と、無作為の時刻との比較
  fig2_v18.png       代表例 V18 (候補 A017): 前方の画像 3 枚と、揺れ・上下加速度・速度と前後の加速度・ペダルの時系列
数値は結果ファイルから読む。カードの説明の文 (目的・出典・場所・信号の状態) は docs/offroad_*.md の調査記録から手で書く。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offroad"))
import _bootstrap  # noqa: E402,F401
import rosbag_range as RB  # noqa: E402
import td2_event_media as EM  # noqa: E402
import td2_review_media as RM  # noqa: E402
import td2_vibration_poc as P  # noqa: E402

plt.rcParams["font.family"] = ["YuGothic", "Hiragino Sans", "DejaVu Sans"]
NAVY, BLUE, SKY, PALE, STEEL, MUTED, TEXT = "#1F2A44", "#2F6FB0", "#3E9BD6", "#EAF2FB", "#8A9BB0", "#6B7280", "#222222"
RED, ORANGE = "#C0392B", "#E08A2E"
FIGSIZE = (9.0, 3.9)
DPI = 250

TD2 = Path("out/offroad/tartandrive2")
INV = TD2 / "20261117_r1"        # 走行ごとの長さ (inventory.csv、summary.json)
DET = TD2 / "20261117_r16"       # 候補 124 件の検出 (summary.json)
PROF = TD2 / "20261118_r6"       # 候補の特徴づけ (ブレーキとアクセルの列を直した現行)
REVIEW = TD2 / "20261118_r7"     # レビューの番号 (V18 = A017)
PROF_OVERLAY = TD2 / "20261117_r20"   # 20261118_r13 の動画が使った特徴づけ (mtvv_total・contrast)
SYNC = TD2 / "20261117_r15"      # racepak の時刻の補正
OVERLAY_LOG = TD2 / "20261118_r13" / "overlay_log.json"


def save(fig, path):
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("->", path)


def fig1(out: Path):
    I = pd.read_csv(INV / "inventory.csv")
    K = I[I.kitti == True].copy()  # noqa: E712
    K["dt"] = K.run.str.extract(r"(\d{4}-\d{2}-\d{2}-\d{2})")[0]
    K["hour"] = K.dt.str[11:13].astype(int)
    K["site"] = K.run.str.replace(r"_?\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_?", "", regex=True)
    inv = json.load(open(INV / "summary.json"))
    det = json.load(open(DET / "summary.json"))
    prof = json.load(open(PROF / "summary.json"))
    d0, d1, nd = K.dt.str[:10].min(), K.dt.str[:10].max(), K.dt.str[:10].nunique()
    n_site = K.site.replace("", np.nan).nunique()

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    # ---- 左: 概要のカード
    ax = fig.add_axes([0.0, 0.0, 0.56, 1.0])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.015, 0.02), 0.97, 0.96, boxstyle="round,pad=0.0,rounding_size=0.02", fc=PALE, ec=STEEL, lw=0.8))
    ax.text(0.04, 0.93, "TartanDrive 2.0 の概要", fontsize=10.5, color=NAVY, fontweight="bold", va="center")
    rows = [
        ("目的", "comma2k19 で少なかった強い振動を補う。ストロークで独立に裏付け"),
        ("出典", "CMU AirLab (CC BY 4.0)。車両は改造した ATV (Yamaha Viking)"),
        ("場所", f"米国ペンシルベニア州西部の私有地。細い道・岩・未舗装・急坂 ({n_site} 地点)"),
        ("時期", f"{d0}〜{d1} の {nd} 日。日中 (走行名の時刻 {K.hour.min()}〜{K.hour.max()} 時台)"),
        ("時間", f"{len(K)} 走行・{inv['kitti_h']:.2f} h (信号の切り出し。bag は {inv['runs_bag']} 走行・{inv['bag_h']:.1f} h)"),
    ]
    y = 0.83
    for k, v in rows:
        ax.text(0.04, y, k, fontsize=7.6, color=BLUE, fontweight="bold", va="center")
        ax.text(0.13, y, v, fontsize=7.3, color=TEXT, va="center")
        y -= 0.075
    y -= 0.02
    ax.text(0.04, y, "信号と映像", fontsize=7.6, color=BLUE, fontweight="bold", va="center")
    y -= 0.075
    sig = [
        ("○", "IMU (100 Hz・398 Hz)", BLUE),
        ("○", "後輪 2 輪のストローク (約 62 Hz)", BLUE),
        ("○", "車輪の回転・GPS の速度", BLUE),
        ("○", "前方カメラ (10 Hz)", BLUE),
        ("△", "ペダル (列の名前が逆。直して使用)", ORANGE),
        ("△", "舵角 (指令値とみられ使わない)", ORANGE),
        ("×", "前輪のストローク (全走行で 0)", RED),
    ]
    for n, (mk, lab, col) in enumerate(sig):
        cx = 0.06 + (n % 2) * 0.46
        cy = y - (n // 2) * 0.072
        ax.text(cx, cy, mk, fontsize=9, color=col, fontweight="bold", va="center", ha="center")
        ax.text(cx + 0.03, cy, lab, fontsize=6.9, color=TEXT, va="center")

    # ---- 右: 候補の段の内訳 (候補 124 件 と 無作為の時刻)
    ar = fig.add_axes([0.64, 0.25, 0.33, 0.58])
    tiers = list(prof["tier_candidates"].keys())
    short = ["A 振動だけ", "B 変化が 1 つ", "C 挙動の変化 + α"]
    cols = [STEEL, SKY, ORANGE]
    nc = prof["candidates"]
    cand = [prof["tier_candidates"][t] / nc for t in tiers]
    rnd = [prof["tier_random"][t] for t in tiers]
    for j, (vals, lab) in enumerate([(cand, f"候補\n{nc} 件"), (rnd, f"無作為の時刻\n{prof['random']} 点")]):
        left = 0.0
        for i, v in enumerate(vals):
            ar.barh(j, v, left=left, color=cols[i], height=0.55, edgecolor="white", lw=1.5)
            txt = f"{v * 100:.0f}%" if j else f"{prof['tier_candidates'][tiers[i]]} 件\n{v * 100:.0f}%"
            if v > 0.06:
                ar.text(left + v / 2, j, txt, ha="center", va="center", fontsize=7 if j == 0 else 7.5,
                        color="white" if i != 0 else TEXT, fontweight="bold" if i == 2 else None)
            left += v
    ar.set_yticks([0, 1])
    ar.set_yticklabels([f"候補\n{nc} 件", f"無作為の\n時刻 {prof['random']:,} 点"], fontsize=7.5)
    ar.invert_yaxis()
    ar.set_xlim(0, 1)
    ar.set_xticks([0, 0.5, 1])
    ar.set_xticklabels(["0", "50%", "100%"], fontsize=7)
    for s in ("top", "right"):
        ar.spines[s].set_visible(False)
    ar.set_title(f"異常振動の候補 {nc} 件 (1 時間あたり {det['per_h']:.0f} 件)", fontsize=8.5, color=NAVY, pad=18)
    for i, s in enumerate(short):
        ar.text(i / 3, 1.04, "■", color=cols[i], fontsize=8, transform=ar.transAxes, va="bottom")
        ar.text(i / 3 + 0.05, 1.04, s, color=TEXT, fontsize=6.6, transform=ar.transAxes, va="bottom")
    fig.text(0.585, 0.04, "挙動の変化 = ヨーの急な変化・ロール角の振れ・後輪の回転の左右差の急変\n"
             "C = 挙動の変化 + (操作の変化か別の挙動の変化)\n自動の判定で、危険だったかは評価していない",
             fontsize=6.2, color=MUTED, va="bottom")
    save(fig, out / "fig1_overview.png")


def fig2(out: Path):
    cfg = yaml.safe_load(open("configs/offroad/td2_vibration_poc.yaml"))
    src = yaml.safe_load(open("configs/offroad/tartandrive2.yaml"))
    root = Path(src["data_dir"])
    RB.CACHE_DIR = root / "bag_cache"
    oc = cfg["overlay"]
    L = pd.read_csv(REVIEW / "review_list.csv").merge(pd.read_csv(REVIEW / "key.csv")[["review_id", "cand_id"]], on="review_id")
    F = pd.read_csv(PROF_OVERLAY / "candidates.csv")
    c = L.merge(F[["cand_id", "mtvv_total", "contrast"]], on="cand_id").set_index("review_id").loc["V18"]
    log = {r["review_id"]: r for r in json.load(open(OVERLAY_LOG))["log"]}["V18"]
    sit = {s["name"]: s for s in log["situations"]}
    ST = pd.read_csv(SYNC / "sync_states.csv")
    sig = RM.signals(c.run, cfg, ST)
    R, S, C = sig["R"], sig["S"], sig["C"]
    g, fs, te = R["g"], R["fs"], float(c.t_abs)
    az = R["X"]["az"] - np.median(R["X"]["az"])
    vlp = P.lp(R["speed"], oc["events"]["speed_lp_hz"], fs)
    acc = np.gradient(vlp, 1 / fs)
    tc, cc = C["controls"]
    bv, tv = P.pedals(cc, cfg)
    ps = oc["pedal_scale"]
    bpct = np.clip((bv - ps["brake_zero"]) / (ps["brake_full"] - ps["brake_zero"]) * 100, 0, 100)
    tpct = np.clip((tv - ps["throttle_zero"]) / (ps["throttle_full"] - ps["throttle_zero"]) * 100, 0, 100)
    span = (-6.0, 3.0)
    m = (g >= te + span[0]) & (g <= te + span[1])
    mc = (tc >= te + span[0]) & (tc <= te + span[1])
    x = g[m] - te
    t_imp = sit["② 大きな衝撃"]["start_s"] + 0.1          # overlay の表示は衝撃の 0.1 s 前から
    k_imp = int(np.argmin(np.abs(g - (te + t_imp))))
    near = (g >= te + t_imp - 0.3) & (g <= te + t_imp + 0.3)
    k_imp = int(np.flatnonzero(near)[np.argmax(np.abs(az[near]))])
    t_imp = g[k_imp] - te

    fr = EM.Frames(src, root, c.run, cfg["media"]["tar"])
    picks = [(t_imp - 3.0, "衝撃の 3 s 前"), (t_imp - 0.3, "衝撃の直前"), (t_imp + 1.0, "衝撃の 1 s 後")]
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    for n, (tt, lab) in enumerate(picks):
        k = fr.k_at(te + tt)
        im = fr.get(k)
        a = fig.add_axes([0.005 + n * 0.17, 0.56, 0.165, 0.38])
        a.imshow(im)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(RED if n == 1 else STEEL); s.set_linewidth(1.6 if n == 1 else 0.6)
        i = int(np.clip(np.searchsorted(g, fr.t_of(k)), 0, len(g) - 1))
        a.set_title(f"{lab} ({fr.t_of(k) - te - t_imp:+.1f} s)  {vlp[i] * 3.6:.0f} km/h", fontsize=6.6, color=RED if n == 1 else NAVY, pad=2)
    fr.close() if hasattr(fr, "close") else None
    # 状況のまとめ (overlay の自動検出の結果)
    ab = fig.add_axes([0.005, 0.02, 0.505, 0.46])
    ab.axis("off"); ab.set_xlim(0, 1); ab.set_ylim(0, 1)
    ab.text(0.01, 0.93, f"V18 (候補 {log['cand_id']})  {c.run}  地点 gupta", fontsize=7, color=NAVY, fontweight="bold", va="center")
    items = [("① 衝撃の前の減速", "① 衝撃の前の減速", "#B7791F"), ("② 大きな衝撃", "② 大きな衝撃", RED), ("③ 衝撃のあとの急減速", "③ 衝撃のあとの急減速", ORANGE)]
    y = 0.76
    for key, lab, col in items:
        s = sit[key]
        ab.add_patch(FancyBboxPatch((0.01, y - 0.075), 0.98, 0.15, boxstyle="round,pad=0,rounding_size=0.02", fc="white", ec=col, lw=1.2))
        ab.text(0.03, y, lab, fontsize=7.2, color=col, fontweight="bold", va="center")
        ab.text(0.40, y, s["desc"], fontsize=7.0, color=TEXT, va="center")
        y -= 0.2
    ab.text(0.01, y + 0.02, "ブレーキ: 使っていない。アクセルを衝撃の前に戻し、約 0.5 s 後に踏み直す", fontsize=6.8, color=TEXT, va="center")
    ab.text(0.01, y - 0.1, "状況は信号からの自動の検出 (人の確認は未了)。画像は日差しのフレアを含む", fontsize=6.0, color=MUTED, va="center")

    # ---- 右: 時系列 4 段
    gs = fig.add_gridspec(4, 1, left=0.585, right=0.93, top=0.97, bottom=0.11, hspace=0.12)
    axs = [fig.add_subplot(gs[i]) for i in range(4)]
    for a in axs[1:]:
        a.sharex(axs[0])
    axs[0].plot(x - t_imp, S["tot"][m], color=NAVY, lw=1)
    axs[0].axhline(c.mtvv_total / c.contrast, color=MUTED, ls="--", lw=0.8)
    axs[0].set_ylabel("揺れ\n[m/s²]", fontsize=6.5)
    axs[0].text(0.01, 0.9, "ISO 2631-1 の 1 秒移動 RMS (点線: 前後 ±30 s の中央値)", transform=axs[0].transAxes, fontsize=5.6, va="top", color=MUTED)
    axs[1].plot(x - t_imp, az[m], color="#374151", lw=0.6)
    axs[1].axhline(oc["events"]["impact_ms2"], color=RED, lw=0.6, ls=":")
    axs[1].axhline(-oc["events"]["impact_ms2"], color=RED, lw=0.6, ls=":")
    axs[1].set_ylabel("上下加速度\n[m/s²]", fontsize=6.5)
    axs[1].annotate(f"{az[k_imp]:+.0f} m/s² ({abs(az[k_imp]) / 9.81:.1f} G)", (0, az[k_imp]), xytext=(8, -2), textcoords="offset points",
                    fontsize=6.5, color=RED, fontweight="bold", va="top")
    axs[2].plot(x - t_imp, vlp[m] * 3.6, color=NAVY, lw=1.2)
    axs[2].set_ylabel("速度\n[km/h]", fontsize=6.5)
    a2 = axs[2].twinx()
    a2.plot(x - t_imp, acc[m], color=ORANGE, lw=0.9)
    a2.axhline(oc["events"]["decel_ms2"], color=ORANGE, lw=0.6, ls=":")
    a2.set_ylabel("前後の加速度\n[m/s²]", fontsize=6.5, color=ORANGE)
    a2.tick_params(labelsize=5.8, colors=ORANGE)
    axs[3].plot(tc[mc] - te - t_imp, tpct[mc], color=SKY, lw=1, label="アクセル")
    axs[3].plot(tc[mc] - te - t_imp, bpct[mc], color=RED, lw=1, label="ブレーキ")
    axs[3].set_ylabel("ペダル\n[%]", fontsize=6.5)
    axs[3].set_ylim(-5, 105)
    axs[3].legend(fontsize=5.8, loc="center left", ncol=1, frameon=False, bbox_to_anchor=(1.0, 0.5), handlelength=1.2)
    axs[3].set_xlabel("衝撃からの秒", fontsize=6.5)
    # 帯: ① は overlay の記録の開始から衝撃まで。③ は衝撃の前後で前後の加速度が急減速の閾値以下の区間 (信号から求める)
    ev = oc["events"]
    s1 = sit["① 衝撃の前の減速"]["start_s"] - t_imp
    post = (g >= te + t_imp - 0.3) & (g <= te + t_imp + ev["post_s"]) & (acc <= ev["decel_ms2"])
    k3 = np.flatnonzero(post)
    spans = [(s1, sit["② 大きな衝撃"]["start_s"] - t_imp, "#B7791F")]
    if len(k3):
        spans.append((g[k3[0]] - te - t_imp, g[k3[-1]] - te - t_imp, ORANGE))
    for a0, a1, col in spans:
        for a in axs:
            a.axvspan(a0, a1, color=col, alpha=0.12, lw=0)
    for a in axs:
        a.axvline(0, color=RED, lw=1)
        a.grid(alpha=0.3)
        a.tick_params(labelsize=5.8)
        a.set_xlim(span[0] - t_imp, span[1] - t_imp)
    for a in axs[:-1]:
        plt.setp(a.get_xticklabels(), visible=False)
    save(fig, out / "fig2_v18.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", choices=["fig1", "fig2"], default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "fig1"):
        fig1(a.out)
    if a.only in (None, "fig2"):
        fig2(a.out)


if __name__ == "__main__":
    main()
