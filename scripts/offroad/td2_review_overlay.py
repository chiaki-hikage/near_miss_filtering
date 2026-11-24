"""レビューの候補の前方映像に、車体のイラスト (後ろ・横・上から) と運転操作 (ハンドル・ブレーキ・アクセル) を重ねた動画を作る。

comma2k19 の振動の動画 (scripts/vibration/vibration_clip_overlay.py) の形を TartanDrive 2.0 に合わせたもの。違い:
  - ロール角・ピッチ角は gps_odom の姿勢 (測った角度) を、前後 ±10 s の中央値からの差で描く (比例の模式ではない。誇張の倍率は画面に書く)
  - タイヤは車体と一体で動く (comma の 2026-11-16 版と同じ)。タイヤの色 = いまの不快の度合い (ISO 2631-1 の 1 秒移動 RMS)
  - 車体の上下のずれは、上下加速度 (0.7〜20 Hz) に比例させた模式 (comma と同じ考え方。変位ではない)
  - 速度計 (km/h、GPS) と前後の加速度 (速度の 1 Hz 以下を微分)
  - ブレーキ・アクセルとみられる信号 (racepak、補正した時刻、単位未確認): 全走行で共通の目盛り (設定 overlay.pedal_scale) で 0〜100% の棒。飛び値は「欠測」
  - 向きの急な変化: 1 s でのヨー角速度 (0.5 Hz 以下) の変化が全走行の p99 以上。舵角の信号が使えないので、操舵か地形による向きの乱れかは区別できない
  - ISO 2631-1 の座面の合成の 1 秒移動 RMS のメーター
  - 状況の欄: 大きな衝撃 (|上下加速度| が閾値以上)、急減速 (前後の加速度が閾値以下)、ブレーキの立ち上がりを、起きた時刻とともに点灯させる。
    下の時系列にも同じ区間を色の帯で描く
  - 舵角 (/ros_talon/steering_angle) は描かない: 値の 30〜73% がちょうど 0 で ±415 で頭打ちになり、旋回 (ヨー角速度 / 速度) との相関が
    走行で −0.95〜−0.3 とばらつくので、実際のハンドルの角度とは言えない (操舵の指令とみられる。2026-11-18 に確かめた)
画像は KITTI 形式の tar から (td2_event_media.Frames、キャッシュがあれば取り直さない)。

    uv run python scripts/offroad/td2_review_overlay.py --review out/offroad/tartandrive2/20261117_r22 \
        --profile out/offroad/tartandrive2/20261117_r20 --sync out/offroad/tartandrive2/20261117_r15 --ids V18 --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge
from scipy.signal import find_peaks
from PIL import Image

import rosbag_range as RB
import td2_event_media as EM
import td2_review_media as RM
import td2_vibration_poc as P

NAVY, SKY, RED, ORANGE, MUTED, TEAL = "#1F2A44", "#3E9BD6", "#C0392B", "#E08A2E", "#6B7280", "#0F766E"
COMFORT_TICKS = [(0.8, "不快"), (2.0, "極めて不快")]   # 0〜8 m/s² の目盛りでは、下の段は詰まるので 2 つだけ書く


def rot(pts, c, deg):
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (np.asarray(pts, float) - c) @ R.T + c


def comfort_color(rms: float, gains: dict):
    """ISO 2631-1 の 1 秒移動 RMS を、濃い灰 → 橙 → 赤 にする (comma の wheel_color と同じ色の道すじ)。"""
    q = float(np.clip((rms - gains["color_lo"]) / (gains["color_hi"] - gains["color_lo"]), 0, 1))
    c0, c1, c2 = np.array([43, 47, 54]), np.array([224, 138, 46]), np.array([192, 57, 43])
    c = c0 + (c1 - c0) * min(q * 2, 1) if q < 0.5 else c1 + (c2 - c1) * (q - 0.5) * 2
    return tuple(c / 255)


def rear_view(ax, z, roll_deg, col, gains):
    """後ろから見た車 (後輪 RL・RR)。タイヤも車体と一体で上下・ロールする。roll_deg: 右下がりが正 (画面で時計回り)。"""
    ax.plot([-130, 130], [0, 0], color="#9AA5B4", lw=2)
    c = np.array([0, 62.0])
    tf = lambda P: rot(P, c, -roll_deg * gains["angle_gain"]) + [0, z]  # noqa: E731
    for xw, name in ((-62, "RL"), (62, "RR")):
        ctr = tf([[xw, 17]])[0]
        ax.add_patch(Circle(ctr, 17, fc=col, ec="none", zorder=3))
        ax.text(ctr[0], ctr[1] - 30, name, fontsize=7, color=MUTED, ha="center")
    ax.add_patch(Polygon(tf([[-80, 32], [80, 32], [80, 72], [58, 74], [44, 112], [-44, 112], [-58, 74], [-80, 72]]),
                         closed=True, fc="#DCE6F2", ec=NAVY, lw=2, zorder=4))
    ax.add_patch(Polygon(tf([[-38, 78], [38, 78], [30, 106], [-30, 106]]), closed=True, fc="#9DB8D6", ec="none", zorder=5))
    ax.text(-125, 132, "左", fontsize=8, color=MUTED)
    ax.text(115, 132, "右", fontsize=8, color=MUTED)
    ax.set_xlim(-130, 130); ax.set_ylim(-30, 145); ax.set_aspect("equal"); ax.axis("off")


def side_view(ax, z, pitch_deg, col, gains):
    """横から見た車 (右が前。見えているのは右側の FR・RR)。タイヤも車体と一体。pitch_deg: 機首上げが正。"""
    ax.plot([-160, 160], [0, 0], color="#9AA5B4", lw=2)
    c = np.array([0, 60.0])
    tf = lambda P: rot(P, c, pitch_deg * gains["angle_gain"]) + [0, z]  # noqa: E731
    for xw, name in ((-82, "RR"), (86, "FR")):
        ctr = tf([[xw, 17]])[0]
        ax.add_patch(Circle(ctr, 17, fc=col, ec="none", zorder=5))
        ax.text(ctr[0], ctr[1] - 30, name, fontsize=7, color=MUTED, ha="center")
    ax.add_patch(Polygon(tf([[-135, 30], [135, 30], [138, 62], [70, 70], [38, 106], [-70, 106], [-112, 72], [-135, 70]]),
                         closed=True, fc="#DCE6F2", ec=NAVY, lw=2, zorder=4))
    ax.annotate("", xy=(158, 128), xytext=(110, 128), arrowprops=dict(arrowstyle="-|>", color=MUTED))
    ax.text(108, 134, "進行方向", fontsize=7, color=MUTED)
    ax.set_xlim(-160, 160); ax.set_ylim(-30, 145); ax.set_aspect("equal"); ax.axis("off")


def speedometer(ax, kmh, acc, vmax, decel_thr):
    """速度計 (半円、0〜vmax km/h) と、前後の加速度の数字。急減速なら赤。"""
    ax.add_patch(Wedge((0, 0), 1.0, 0, 180, width=0.16, fc="#EEF2F7", ec="#9AA5B4", lw=1))
    frac = float(np.clip(kmh / vmax, 0, 1))
    ax.add_patch(Wedge((0, 0), 1.0, 180 - 180 * frac, 180, width=0.16, fc=SKY, ec="none"))
    for v in range(0, int(vmax) + 1, 10):
        a = math.radians(180 - 180 * v / vmax)
        ax.plot([0.80 * math.cos(a), 0.86 * math.cos(a)], [0.80 * math.sin(a), 0.86 * math.sin(a)], color=MUTED, lw=1)
        ax.text(0.68 * math.cos(a), 0.68 * math.sin(a), f"{v}", fontsize=6.5, color=MUTED, ha="center", va="center")
    a = math.radians(180 - 180 * frac)
    ax.plot([0, 0.92 * math.cos(a)], [0, 0.92 * math.sin(a)], color=RED, lw=2.5)
    ax.add_patch(Circle((0, 0), 0.06, fc=NAVY))
    ax.text(0, -0.22, f"{kmh:.0f} km/h", fontsize=13, color=NAVY, ha="center", fontweight="bold")
    hard = acc <= decel_thr
    ax.text(0, -0.48, f"前後 {acc:+.1f} m/s² ({acc / 9.81:+.2f} G)", fontsize=8, color=RED if hard else MUTED, ha="center",
            fontweight="bold" if hard else "normal")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.6, 1.08); ax.set_aspect("equal"); ax.axis("off")


def status_box(ax, items, now):
    """状況の欄。items: [(名前, 始まりの時刻, 終わりの時刻, 説明, 色)]。いまが区間の中なら点灯、過ぎたら薄く残す。"""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0, 0.97, "状況 (自動で検出)", fontsize=7.5, color=NAVY, va="top", fontweight="bold")
    if not items:
        ax.text(0, 0.80, "検出なし", fontsize=7, color=MUTED, va="top")
    for n, (name, t0, t1, desc, col) in enumerate(items[:4]):
        y = 0.86 - n * 0.215
        on = t0 <= now <= t1
        past = now > t1
        fc = col if on else ("#F3F4F6" if not past else "#FFFFFF")
        ax.add_patch(Rectangle((0, y - 0.195), 1, 0.195, fc=fc, ec=col, lw=1.2, alpha=0.9 if on else 1))
        tc_ = "white" if on else (col if past else MUTED)
        ax.text(0.03, y - 0.055, name, fontsize=7, color=tc_, va="center", fontweight="bold")
        ax.text(0.06, y - 0.14, desc, fontsize=6.2, color=tc_, va="center")


def segments(mask: np.ndarray, t: np.ndarray, gap: float, min_len: float) -> list[tuple[float, float]]:
    """真の区間を (始まり, 終わり) の時刻で返す。gap より短い切れ目はつなぎ、min_len より短い区間は捨てる。"""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, s0, prev = [], t[idx[0]], t[idx[0]]
    for k in idx[1:]:
        if t[k] - prev > gap:
            out.append((s0, prev))
            s0 = t[k]
        prev = t[k]
    out.append((s0, prev))
    return [(a, b) for a, b in out if b - a >= min_len]


def pedal(ax, frac, label, color):
    ax.add_patch(Rectangle((0, 0), 1, 1, fc="#EEF2F7", ec="#9AA5B4"))
    if frac is None:
        ax.text(0.5, 0.5, "欠測", fontsize=8, color=RED, ha="center", va="center", rotation=90)
    else:
        ax.add_patch(Rectangle((0, 0), 1, float(np.clip(frac, 0, 1)), fc=color, ec="none"))
    ax.text(0.5, -0.12, label, fontsize=7.5, color=NAVY, ha="center", va="top")
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.35, 1.05); ax.axis("off")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offroad/td2_vibration_poc.yaml")
    ap.add_argument("--source", default="configs/offroad/tartandrive2.yaml")
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--sync", type=Path, required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    oc = cfg["overlay"]
    gains = oc["gains"]
    src = yaml.safe_load(open(a.source))
    root = Path(src["data_dir"])
    RB.CACHE_DIR = root / "bag_cache"
    out = Path("out/offroad/tartandrive2") / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
    L = pd.read_csv(a.review / "review_list.csv")
    key = pd.read_csv(a.review / "key.csv")
    F = pd.read_csv(a.profile / "candidates.csv")
    L = L.merge(key[["review_id", "cand_id"]], on="review_id").merge(F[["cand_id", "mtvv_total", "contrast"]], on="cand_id").set_index("review_id")
    ST = pd.read_csv(a.sync / "sync_states.csv")
    log = []
    for rid in a.ids.split(","):
        c = L.loc[rid]
        sig = RM.signals(c.run, cfg, ST)
        R, S, C = sig["R"], sig["S"], sig["C"]
        fr = EM.Frames(src, root, c.run, cfg["media"]["tar"])
        g, fs, te = R["g"], R["fs"], float(c.t_abs)
        # ---- 信号 (IMU の時刻)
        heave = P.bp(R["X"]["az"] - R["X"]["az"].mean(), *oc["heave_band_hz"], fs)
        az = R["X"]["az"] - np.median(R["X"]["az"])
        ctx = (g >= te - 10) & (g <= te + 10)
        roll = np.degrees(R["roll_g"] - np.median(R["roll_g"][ctx]))
        pitch = np.degrees(R["pitch_g"] - np.median(R["pitch_g"][ctx]))
        vlp = P.lp(R["speed"], oc["events"]["speed_lp_hz"], fs)
        acc = np.gradient(vlp, 1 / fs)                               # 前後の加速度 (GPS の速度から)
        ts, sa = C["shock"]
        base = {s: float(np.median(sa[(ts >= te - 10) & (ts <= te + 10), j])) for s, j in (("RL", 2), ("RR", 3))}
        tc, cc = C["controls"]
        bv, tv = P.pedals(cc, cfg)                       # (ブレーキ, アクセル)。列の割り当ては設定 (reaction.controls_cols)
        # 0〜100% の目盛りは全走行で共通 (走行ごとの p1〜p99 だと、ブレーキを使わない走行で小さな揺れが大きく見えた。2026-11-18)
        ps = oc["pedal_scale"]
        b_lo, b_hi, t_lo, t_hi = ps["brake_zero"], ps["brake_full"], ps["throttle_zero"], ps["throttle_full"]
        bpct = np.clip((bv - b_lo) / (b_hi - b_lo) * 100, 0, 100)
        tpct = np.clip((tv - t_lo) / (t_hi - t_lo) * 100, 0, 100)
        wz_slow = np.degrees(P.lp(R["X"]["wz"], oc["events"]["yaw_lp_hz"], fs))
        nyc = int(oc["events"]["yaw_change_window_s"] * fs)
        dyaw = np.full(len(g), np.nan)
        dyaw[nyc:] = wz_slow[nyc:] - wz_slow[:-nyc]          # 1 s でのヨー角速度 (0.5 Hz 以下) の変化
        span = oc["span_s"]
        lo, hi = te + span[0], te + span[1]
        m = (g >= lo) & (g <= hi)
        x = g[m] - te
        # ---- 状況の検出: te の近くの一番大きい衝撃を軸に、「衝撃の前の減速 → 衝撃 → 衝撃のあとの急減速 → ブレーキの立ち上がり」に並べる
        ev = oc["events"]
        mcw = (tc >= lo) & (tc <= hi)
        near = (g >= te + ev["impact_search_s"][0]) & (g <= te + ev["impact_search_s"][1])
        k_imp = int(np.flatnonzero(near)[np.argmax(np.abs(az[near]))])
        t_imp, a_imp = g[k_imp], float(az[k_imp])
        has_imp = abs(a_imp) >= ev["impact_ms2"]
        hard = m & (acc <= ev["decel_ms2"])
        segs = segments(hard, g, ev["merge_gap_s"], ev["min_len_s"])
        kmh = lambda t: float(np.interp(t, g, vlp)) * 3.6  # noqa: E731
        # ブレーキの立ち上がり: 前 1 s の最小からの上がりが brake_rise_pct 以上になった最初の時刻 (区間ごと)
        bp = np.where(np.isnan(bpct), np.nan, bpct)
        rise_t = []
        for k in np.flatnonzero(mcw):
            w = (tc >= tc[k] - 1.0) & (tc < tc[k])
            if np.isfinite(bp[k]) and w.any() and np.isfinite(bp[w]).any() and bp[k] - np.nanmin(bp[w]) >= ev["brake_rise_pct"]:
                if not rise_t or tc[k] - rise_t[-1][0] > 1.5:
                    rise_t.append((tc[k], float(bp[k] - np.nanmin(bp[w]))))
        items = []
        pre = [(a0, a1_) for a0, a1_ in segs if a1_ < t_imp - 0.1]
        pre_b = [r for r in rise_t if r[0] < t_imp - 0.2]
        if pre or pre_b:
            t0 = min([a0 for a0, _ in pre] + [r[0] for r in pre_b])
            amin = min([float(acc[(g >= a0) & (g <= a1_)].min()) for a0, a1_ in pre], default=0.0)
            desc = (f"{kmh(t0):.0f}→{kmh(t_imp):.0f} km/h、最大 {amin:.1f} m/s²" if pre else "") + ("・ブレーキあり" if pre_b else "")
            items.append(("衝撃の前の減速", t0 - te, t_imp - te - 0.1, desc, "#B7791F"))
        if has_imp:
            items.append(("大きな衝撃", t_imp - te - 0.1, t_imp - te + ev["show_s"], f"上下 {a_imp:+.0f} m/s² ({abs(a_imp) / 9.81:.1f} G)", RED))
        post = [(a0, a1_) for a0, a1_ in segs if t_imp - 0.3 <= a0 <= t_imp + ev["post_s"]]
        if post:
            a0, a1_ = post[0][0], post[-1][1]
            mm = (g >= a0) & (g <= a1_)
            amin = float(acc[mm].min())
            items.append(("衝撃のあとの急減速", a0 - te, a1_ - te + ev["show_s"],
                          f"最大 {amin:.1f} m/s² ({amin / 9.81:.2f} G)・{kmh(a0):.0f}→{kmh(a1_):.0f} km/h", ORANGE))
        post_b = [r for r in rise_t if t_imp - 0.2 <= r[0] <= t_imp + ev["post_s"]]
        if post_b:
            tb, rb = post_b[0]
            items.append(("衝撃のあとのブレーキ", tb - te, tb - te + ev["show_s"], f"+{rb:.0f}%・衝撃の {tb - t_imp:+.1f} s 後", "#7c3aed"))
        # 向きの急な変化 (操舵か地形かは区別できない): 衝撃の前後 yaw_change_span_s で |1 s の変化| が閾値以上の最初の所
        ys = (g >= t_imp + ev["yaw_change_span_s"][0]) & (g <= t_imp + ev["yaw_change_span_s"][1]) & m & (np.abs(np.nan_to_num(dyaw)) >= ev["yaw_change_dps"])
        for a0, a1_ in segments(ys, g, ev["merge_gap_s"], 0.0)[:2]:
            mm = (g >= a0) & (g <= a1_)
            kk = np.flatnonzero(mm)[np.argmax(np.abs(dyaw[mm]))]
            side = "左" if dyaw[kk] > 0 else "右"
            items.append(("向きの急な変化", a0 - te - nyc / fs, a1_ - te + ev["show_s"],
                          f"1 s で {dyaw[kk]:+.0f} deg/s ({side}へ)・衝撃の {g[kk] - t_imp:+.1f} s", TEAL))
        items.sort(key=lambda r: r[1])
        items = items[:4]
        nums = "①②③④"
        items = [(f"{nums[n]} {r[0]}", r[1], r[2], r[3], r[4]) for n, r in enumerate(items)]
        # ---- 時系列の図 (下の帯)。フレームごとに縦線だけ動かす
        figT, axT = plt.subplots(5, 1, figsize=(16.64, 4.2), dpi=100, sharex=True)
        axT[0].plot(x, S["tot"][m], color=NAVY, label="ISO 2631-1 の 1 秒移動 RMS [m/s²]")
        axT[0].axhline(c.mtvv_total / c.contrast, color=MUTED, ls="--", lw=1, label="前後 ±30 s の中央値")
        axT[1].plot(x, az[m], color="#374151", lw=0.7, label="上下加速度 [m/s²]")
        axT[1].axhline(ev["impact_ms2"], color=RED, lw=0.8, ls=":")
        axT[1].axhline(-ev["impact_ms2"], color=RED, lw=0.8, ls=":")
        axT[2].plot(x, roll[m], color="#7c3aed", label="ロール角 [deg]")
        axT[2].plot(x, pitch[m], color="#16a34a", label="ピッチ角 [deg]")
        axT[2].plot(x, wz_slow[m], color=TEAL, lw=1.4, label="ヨー角速度 0.5 Hz 以下 [deg/s] (+ 左)")
        ms = (ts >= lo) & (ts <= hi)
        a2b = axT[2].twinx()
        a2b.plot(ts[ms] - te, sa[ms, 2] - base["RL"], color=RED, lw=0.8, label="後輪ストローク RL [in] (+ 縮み)")
        a2b.plot(ts[ms] - te, sa[ms, 3] - base["RR"], color=TEAL, lw=0.8, label="RR")
        a2b.legend(fontsize=6, loc="upper right", ncol=2); a2b.tick_params(labelsize=6)
        axT[3].plot(x, R["speed"][m] * 3.6, color="#111827", label="速度 [km/h]")
        a3b = axT[3].twinx()
        a3b.plot(x, acc[m], color=ORANGE, lw=1, label="前後の加速度 [m/s²]")
        a3b.axhline(ev["decel_ms2"], color=ORANGE, lw=0.8, ls=":")
        a3b.legend(fontsize=6, loc="upper right"); a3b.tick_params(labelsize=6)
        axT[4].plot(tc[mcw] - te, bpct[mcw], color="#7c3aed", lw=1, label="ブレーキ [%] (補正した時刻)")
        axT[4].plot(tc[mcw] - te, tpct[mcw], color="#16a34a", lw=1, label="アクセル [%] (補正した時刻)")
        for name, a0, a1_, desc, col in items:
            for ax_ in axT:
                ax_.axvspan(a0, a1_ - ev["show_s"] if a1_ - ev["show_s"] > a0 + 0.05 else a0 + 0.1, color=col, alpha=0.15, lw=0)
            axT[0].text(a0, np.nanmax(S["tot"][m]) * 0.92, name[0], fontsize=9, color=col, fontweight="bold")
        for ax_ in axT:
            ax_.legend(fontsize=6, loc="upper left", ncol=3)
            ax_.grid(alpha=0.3)
            ax_.tick_params(labelsize=6)
        axT[-1].set_xlabel("te からの秒 (te = 1 秒移動 RMS の最大。衝撃は 0〜1 s 前)", fontsize=7)
        axT[0].set_xlim(span[0], span[1])
        figT.tight_layout(pad=0.4)
        curT = [ax_.axvline(0, color="#f59e0b", lw=2) for ax_ in axT]

        ks = list(range(fr.k_at(lo), fr.k_at(hi) + 1, oc["frame_step"]))
        fdir = out / f"{rid}_frames"
        fdir.mkdir(exist_ok=True)
        before = fr.used
        from PIL import ImageDraw, ImageFont
        font = ImageFont.truetype(oc["font_path"], 13)
        for n, k in enumerate(ks):
            im = fr.get(k)
            if im is None:
                continue
            tk = fr.t_of(k)
            i = int(np.clip(np.searchsorted(g, tk), 0, len(g) - 1))
            jc = int(np.clip(np.searchsorted(tc, tk), 0, len(tc) - 1))
            cur = float(S["tot"][i]) if np.isfinite(S["tot"][i]) else 0.0
            col = comfort_color(cur, gains)
            # ---- 右のパネル
            figP = plt.figure(figsize=(6.4, 5.44), dpi=100)
            figP.text(0.02, 0.975, f"{rid}  {c.run}  t={c.t:.1f} s", fontsize=8.5, color=NAVY, va="top", fontweight="bold")
            figP.text(0.98, 0.975, f"te {tk - te:+.1f} s", fontsize=11, color=RED, va="top", ha="right", fontweight="bold")
            a1 = figP.add_axes([0.0, 0.53, 0.46, 0.38]); rear_view(a1, heave[i] * gains["unit_per_ms2"], roll[i], col, gains)
            a1.set_title(f"後ろから: ロール {roll[i]:+.1f}°・上下 {heave[i]:+.1f} m/s²", fontsize=7.5, color=NAVY)
            a2 = figP.add_axes([0.48, 0.53, 0.52, 0.38]); side_view(a2, heave[i] * gains["unit_per_ms2"], pitch[i], col, gains)
            a2.set_title(f"横から: ピッチ {pitch[i]:+.1f}° (機首上げ +)", fontsize=7.5, color=NAVY)
            figP.text(0.02, 0.505, f"タイヤの色 = いまの不快の度合い (ISO 1 秒 RMS {gains['color_lo']:.1f} 以下 灰 → 橙 → {gains['color_hi']:.0f} 以上 赤)",
                      fontsize=6.3, color=MUTED)
            a4 = figP.add_axes([0.01, 0.17, 0.38, 0.30]); speedometer(a4, R["speed"][i] * 3.6, acc[i], oc["speed_max_kmh"], ev["decel_ms2"])
            near_c = abs(tc[jc] - tk) < 0.3
            bfrac = None if (not near_c or not np.isfinite(bpct[jc])) else bpct[jc] / 100
            tfrac = None if (not near_c or not np.isfinite(tpct[jc])) else tpct[jc] / 100
            a6 = figP.add_axes([0.41, 0.22, 0.07, 0.22]); pedal(a6, bfrac, "ブレーキ", "#7c3aed")
            a7 = figP.add_axes([0.50, 0.22, 0.07, 0.22]); pedal(a7, tfrac, "アクセル", "#16a34a")
            figP.text(0.40, 0.465, "ペダル (全走行の目盛り)", fontsize=6, color=NAVY)
            a8 = figP.add_axes([0.60, 0.14, 0.39, 0.36]); status_box(a8, items, tk - te)
            am = figP.add_axes([0.06, 0.035, 0.90, 0.045])
            am.barh([0], [oc["iso_max"]], color="#EEF2F7", height=0.8)
            am.barh([0], [min(cur, oc["iso_max"])], color=col, height=0.8)
            for th, nm in COMFORT_TICKS + [(c.mtvv_total, "")]:
                am.axvline(th, color=RED if nm == "" else "#9AA5B4", lw=1.2 if nm == "" else 0.7, ls="--" if nm == "" else "-")
                if nm:
                    am.text(th, 0.7, nm, fontsize=5.5, color=MUTED, ha="center", va="bottom")
            am.set_xlim(0, oc["iso_max"]); am.set_ylim(-0.6, 1.5); am.set_yticks([]); am.tick_params(labelsize=6)
            for s_ in ("top", "right", "left"):
                am.spines[s_].set_visible(False)
            figP.text(0.06, 0.088, f"ISO 2631-1 座面の合成の 1 秒移動 RMS: いま {cur:.2f} m/s² (赤の破線 = この候補の最大)", fontsize=6.8, color=NAVY)
            buf = io.BytesIO(); figP.savefig(buf, format="png"); plt.close(figP)
            pnl = Image.open(buf).convert("RGB")
            for l_ in curT:
                l_.set_xdata([tk - te, tk - te])
            buf2 = io.BytesIO(); figT.savefig(buf2, format="png")
            tsimg = Image.open(buf2).convert("RGB")
            frame = Image.new("RGB", (1664, 544 + tsimg.height + 54), "white")
            frame.paste(im.resize((1024, 544)), (0, 0))
            frame.paste(pnl, (1024, 0))
            frame.paste(tsimg, (0, 544))
            import textwrap
            ImageDraw.Draw(frame).multiline_text((10, 544 + tsimg.height + 6), "\n".join(textwrap.wrap(oc["footnote"], 118)),
                                                 fill=(107, 114, 128), font=font, spacing=4)
            frame.save(fdir / f"{n:03d}.png")
        plt.close(figT)
        fps = 10 / oc["frame_step"]
        for suffix, rate in (("", fps), ("_slow3x", fps / 3)):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{rate}", "-i", str(fdir / "%03d.png"),
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
                            "-movflags", "+faststart", str(out / f"{rid}_overlay{suffix}.mp4")], check=True)
        # 衝撃の近くの 1 枚を図として残す
        k_imp = min(range(len(ks)), key=lambda n: abs(fr.t_of(ks[n]) - (te - 0.5)))
        shutil.copy(fdir / f"{k_imp:03d}.png", out / f"{rid}_overlay_te-0.5s.png")
        shutil.rmtree(fdir)
        log.append(dict(review_id=rid, cand_id=c.cand_id, run=c.run, frames=len(ks), bytes=fr.used - before,
                        brake_p1_p99=[float(b_lo), float(b_hi)], throttle_p1_p99=[float(t_lo), float(t_hi)],
                        stroke_base_in=base, situations=[dict(name=n_, start_s=round(a0, 2), desc=d_) for n_, a0, _a1, d_, _c in items]))
        print(log[-1], flush=True)
    (out / "overlay_log.json").write_text(json.dumps(dict(log=log, config=oc), ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
