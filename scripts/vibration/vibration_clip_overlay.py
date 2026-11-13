"""振動イベントの前方映像に、上下・ロール・ピッチの動きのイラストと ISO 2631-1 の値の推移を重ねた動画を作る。

  uv run --extra viz python scripts/vibration/vibration_clip_overlay.py --run-id <id>

設定: configs/vibration_clip_overlay.yaml。出力: out/vibration/near_miss/iso_overlay/<run_id>/ (既存のものは上書きしない)。
イベントの検出・候補の判定・ISO 2631-1 の値は、既存の結果 (near_miss_vibration_events.csv・iso2631_events.csv) を読むだけ。

1 フレームの構成 (前方映像は 20 Hz、前輪の当たり時刻の前後 clip_s 秒):
  左上  前方カメラ。左上にイベント・時刻、右上に「前輪の当たりから何秒か」
  右上  車の動きのイラスト
          後ろから見た車: 車体の上下とロール (右下がりが正)、タイヤと路面の入力 (左右)
          横から見た車 (右が前): 車体の上下とピッチ (機首上げが正)、路面の入力が前輪 → 後輪の順に通る様子
          ISO 2631-1 の座面の合成 (5 軸) の 1 秒移動 RMS のメーターと、快適性の目安
  下    時系列 (上下加速度・ロール角速度・ピッチ角速度・ISO の 1 秒移動 RMS) と、今の時刻の縦線

イラストは模式図で、車体の上下のずれを上下加速度に、傾きをロール・ピッチの角速度に比例させて描く (帯域通過 0.7 Hz〜、
振動の解析と同じ信号)。**変位・角度そのものではない。** 加速度を積分した変位では、ゆっくりした車体の揺れ (2〜3 Hz) が大きく出て、
乗員が感じる強さ (ISO 2631-1 も加速度で評価する) と逆の印象になったため (2026-11-13 に確かめた)。
比例の係数は全イベントで共通にして、画面に書く。
路面の入力とタイヤの上下も模式。タイヤの上下・路面の高さは測っていない (サスペンションのストロークのセンサーが無い)。
データから決めるのは、前輪・後輪が入力を通る時刻 (イベントの時刻と、ホイールベース / 車速)、左右 (側の指標)、
凸か凹か (上下加速度の最初の向き) だけで、形と高さは模式 (画面に書く)。
ISO の値は scripts/vibration/iso2631_events.py と同じ計算 (同じ前後 4 秒の切り出し) で時系列を作り、
最大値 (MTVV) が iso2631_events.csv の値と合うことを確かめてから描く。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yaml

import iso2631 as ISO
import iso2631_events as IE
import final_qualitative as FQ
import validate_sites_with_fullcar as V

REPO = _bootstrap.SRC.parent
NAVY, BLUE, SKY, RED, ORANGE, MUTED = "#1F2A44", "#2F6FB0", "#3E9BD6", "#C0392B", "#E08A2E", "#6B7280"
MM_PER_UNIT = 1850 / 160      # イラストの縮尺 (後ろから見た車の幅 160 単位 ≈ 車幅 1.85 m とみなす。表示用の目安)
COMFORT_TICKS = [(0.315, "少し不快"), (0.5, "かなり不快"), (0.8, "不快"), (1.25, "非常に不快"), (2.0, "極めて不快")]


def load_event(e, loader, raw, cfg, oc):
    """イベントの信号 (前輪の当たり時刻が 0)。動き (帯域通過) と ISO の 1 秒移動 RMS。"""
    fs = cfg["grid_hz"]
    T, P = oc["clip_s"], oc["pad_s"]
    m = loader.load(e.drive, int(e.segment), float(e.event_time_s), (-T - P, T + P))
    mot = dict(t=m["t"], heave=m["heave"], roll=m["roll"], pitch=m["pitch"], v=m["v"])
    for k in V.WHEELS:                         # 輪速の揺れ (帯域通過 2〜15 Hz、m/s → km/h) と、その 0.1 s の移動 RMS
        mot[k] = m[k] * 3.6
        mot[f"{k}_env"] = np.sqrt(pd.Series(mot[k] ** 2).rolling(int(0.1 * fs), center=True, min_periods=1).mean().to_numpy())
    # ISO: iso2631_events.py と同じ切り出し (前後 4 s) と計算
    t, x = raw.load(e.drive, int(e.segment), float(e.event_time_s), (-4.0, 4.0))
    aw = {k: ISO.weight(x[k], fs, w) for k, w in IE.AXES}
    rr = {k: ISO.running_rms(aw[k], fs) for k in aw}
    tot = np.sqrt(sum((ISO.K_SEAT[k] * rr[k]) ** 2 for k in rr))
    near = (t >= -1.0) & (t <= 2.0)
    mtvv = float(np.nanmax(tot[near]))
    if abs(mtvv - e.mtvv_total) > 0.01 * max(e.mtvv_total, 0.1):
        raise RuntimeError(f"{e.event_id}: MTVV が既存の値と合わない ({mtvv:.4f} vs {e.mtvv_total:.4f})")
    iso = dict(t=t, tot=tot, mtvv=mtvv, t_mtvv=float(t[near][np.nanargmax(tot[near])]))
    return mot, iso


def road_input(e, mot, oc):
    """路面の入力の模式に使う値。データから決めるもの: 前輪の当たり (イベントの時刻)、後輪の遅れ (ホイールベース / 車速)、
    左右 (側の指標。docs/vibration_physics_validation.md 4 章)、凸か凹か (上下加速度の最初の向き。final_qualitative.first_sign)。
    形 (長さ) と高さは決められないので模式。"""
    rc = oc["road_schematic"]
    side_tab = pd.read_csv(REPO / rc["side_index_csv"]).set_index("event_id")
    si = float(side_tab.loc[e.event_id, "side_index"])
    side = "右" if si <= -rc["side_threshold"] else ("左" if si >= rc["side_threshold"] else "両輪")
    sign, _ = FQ.first_sign(mot["heave"], mot["t"], 0.0)
    v = float(mot["v"])
    wb = rc["wheelbase_m"]
    return dict(side=side, side_index=si, h=rc["height_units"] * (1 if sign >= 0 else -1), len=rc["length_units"], show=True,
                v_ms=v, lag_s=wb / v, m_per_unit_side=wb / (SIDE_FRONT - SIDE_REAR),
                basis=f"側の指標 {si:+.2f}・上下加速度の最初の向き {'上' if sign >= 0 else '下'}")


def video_frames(e, loader, oc, tmp: Path):
    """前輪の当たりの前後 clip_s 秒の前方映像のフレーム (JPG) と、当たりからの時刻。"""
    refs = sorted(loader.refs[e.drive], key=lambda r: r.index)
    by = {r.index: r for r in refs}
    t0 = float(np.load(by[int(e.segment)].path / "processed_log/IMU/accelerometer/t").ravel()[0])
    t_hit = t0 + float(e.event_time_s)
    a, b = t_hit - oc["clip_s"], t_hit + oc["clip_s"]
    paths, times = [], []
    for k in (int(e.segment) - 1, int(e.segment), int(e.segment) + 1):
        if k not in by:
            continue
        ft = np.load(by[k].path / "global_pose/frame_times")
        idx = np.flatnonzero((ft >= a) & (ft <= b))
        if not len(idx):
            continue
        d = tmp / f"seg{k}"
        d.mkdir()
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-r", "20", "-i", str(by[k].path / "video.hevc"),
                        "-vf", f"select='between(n\\,{idx[0]}\\,{idx[-1]})'", "-fps_mode", "passthrough", "-q:v", "2",
                        str(d / "%05d.jpg")], check=True)
        fr = sorted(d.glob("*.jpg"))
        if len(fr) != len(idx):
            raise RuntimeError(f"{e.event_id} seg{k}: フレームの数が合わない ({len(fr)} vs {len(idx)})")
        paths += fr
        times += list(ft[idx] - t_hit)
    return paths, np.array(times)


# ---------------------------------------------------------------- 図
def timeseries_png(mot, iso, e, lim, width_px, out: Path, wheels=False):
    import matplotlib.pyplot as plt
    T = lim["T"]
    n = 5 if wheels else 4
    fig, axs = plt.subplots(n, 1, figsize=(width_px / 100, 4.6 + (1.0 if wheels else 0)), dpi=100, sharex=True)
    w = (mot["t"] >= -T) & (mot["t"] <= T)
    spec = [(mot["heave"], "上下加速度\n[m/s²]", BLUE, lim["heave"]), (mot["roll"], "ロール角速度\n[deg/s]", SKY, lim["rate"]),
            (mot["pitch"], "ピッチ角速度\n[deg/s]", NAVY, lim["rate"])]
    for ax, (y, lab, c, yl) in zip(axs, spec):
        ax.plot(mot["t"][w], y[w], color=c, lw=1.0)
        ax.set_ylim(-yl, yl)
        ax.set_ylabel(lab, fontsize=8)
    wi = (iso["t"] >= -T) & (iso["t"] <= T)
    axs[3].plot(iso["t"][wi], iso["tot"][wi], color=RED, lw=1.6)
    for th, name in COMFORT_TICKS:
        if th <= lim["iso"]:
            axs[3].axhline(th, color="#9AA5B4", lw=0.6, ls=":")
            axs[3].text(T - 0.02, th, name, fontsize=6.5, color=MUTED, ha="right", va="bottom")
    axs[3].set_ylim(0, lim["iso"])
    if wheels:
        for k, c, ls in (("FL", BLUE, "-"), ("FR", ORANGE, "-"), ("RL", BLUE, ":"), ("RR", ORANGE, ":")):
            axs[4].plot(mot["t"][w], mot[k][w], color=c, ls=ls, lw=0.9, label=k)
        axs[4].set_ylim(-lim["wheel"], lim["wheel"])
        axs[4].set_ylabel("輪速の揺れ\n[km/h]", fontsize=8)
        axs[4].legend(fontsize=7, ncol=4, loc="upper left", title="左 = 青、右 = 橙、前 = 実線、後 = 点線", title_fontsize=7)
    axs[3].set_ylabel("ISO 2631-1\n1 秒移動 RMS\n[m/s²]", fontsize=8)
    axs[3].plot([iso["t_mtvv"]], [iso["mtvv"]], "o", color=RED, ms=5)
    axs[3].text(iso["t_mtvv"] + 0.05, iso["mtvv"], f" MTVV {iso['mtvv']:.2f}", fontsize=7.5, color=RED, va="center")
    for ax in axs:
        ax.axvline(0, color="0.4", lw=0.8, ls="--")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-T, T)
    axs[0].set_title("縦の破線 = 前輪が当たった推定時刻 (0 s)。縦軸の範囲は全動画で共通", fontsize=8, color=MUTED, loc="left")
    axs[-1].set_xlabel("前輪が当たった推定時刻からの秒", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    H = img.shape[0]
    x0 = axs[0].transData.transform((-T, 0))[0]
    x1 = axs[0].transData.transform((T, 0))[0]
    mp = dict(x0=x0, x1=x1, ytop=H - axs[0].get_window_extent().y1, ybot=H - axs[-1].get_window_extent().y0, T=T)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return img, mp


def road_profile(x, h, length):
    """路面の入力の模式 (cos の山・谷)。x は入力の中心からの距離 (図の単位)。h > 0 は凸、h < 0 は凹。"""
    x = np.asarray(x, float)
    return np.where(np.abs(x) < length / 2, h * 0.5 * (1 + np.cos(2 * np.pi * x / length)), 0.0)


def body_rear(ax, z_px, ang_deg, lift_l, lift_r, road_l, road_r):
    """後ろから見た車。車体: 上下 z_px・ロール ang_deg (右下がりが正)。タイヤ: 路面の入力に沿って上下 (lift_*)。
    road_* はタイヤの下の路面の高さ (入力がタイヤの下にあるときだけ 0 でない)。"""
    from matplotlib.patches import Circle, Polygon, Rectangle
    ax.plot([-125, 125], [0, 0], color="#9AA5B4", lw=2)
    for xw, rh in ((-62, road_l), (62, road_r)):          # タイヤの下の路面の凸凹 (模式)
        if abs(rh) > 0.3:
            ax.add_patch(Rectangle((xw - 26, min(0, rh)), 52, abs(rh), fc="#E08A2E" if rh > 0 else "#F4F8FC",
                                   ec="#E08A2E", lw=1.5, zorder=1))
    wheel_y = {-62: 17 + lift_l, 62: 17 + lift_r}
    for xw, y in wheel_y.items():
        ax.add_patch(Circle((xw, y), 17, fc="#2B2F36", ec="none", zorder=3))
    pts = np.array([[-80, 52], [80, 52], [80, 92], [58, 94], [44, 132], [-44, 132], [-58, 94], [-80, 92]], float)
    c = np.array([0, 82.0])
    a = -math.radians(ang_deg)                 # 右下がり (正) = 画面で時計回り
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    off = np.array([0, z_px + 0.5 * (lift_l + lift_r) * 0.0])
    ax.add_patch(Polygon((pts - c) @ R.T + c + off, closed=True, fc="#DCE6F2", ec=NAVY, lw=2, zorder=4))
    win = np.array([[-38, 98], [38, 98], [30, 126], [-30, 126]], float)
    ax.add_patch(Polygon((win - c) @ R.T + c + off, closed=True, fc="#9DB8D6", ec="none", zorder=5))
    for xw, y in wheel_y.items():               # サスペンション (タイヤの中心と車体をつなぐ線)
        pb = (np.array([xw * 0.8, 52.0]) - c) @ R.T + c + off
        ax.plot([xw, pb[0]], [y, pb[1]], color=MUTED, lw=2, zorder=2)
    ax.text(-118, 150, "左", fontsize=9, color=MUTED)
    ax.text(108, 150, "右", fontsize=9, color=MUTED)
    ax.set_xlim(-125, 125); ax.set_ylim(-14, 165); ax.set_aspect("equal"); ax.axis("off")


SIDE_FRONT, SIDE_REAR = 86.0, -82.0               # 横から見た図の前輪・後輪の位置 (ホイールベース 168 単位)


def body_side(ax, z_px, ang_deg, x_in, road):
    """横から見た車 (右が前)。x_in は路面の入力の中心の位置 (図の座標、車とともに動く座標で後ろへ流れる)。"""
    from matplotlib.patches import Circle, Polygon
    xs = np.linspace(-155, 155, 400)
    prof = road_profile(xs - x_in, road["h"], road["len"]) if road["show"] else np.zeros_like(xs)
    ax.fill_between(xs, -14, prof, color="#EEF2F7", zorder=0)
    ax.plot(xs, prof, color="#9AA5B4", lw=2, zorder=1)
    if road["show"] and abs(x_in) < 160:
        ax.plot(xs[np.abs(xs - x_in) < road["len"] / 2], prof[np.abs(xs - x_in) < road["len"] / 2], color="#E08A2E", lw=3, zorder=1)
    wy = {}
    for xw in (SIDE_REAR, SIDE_FRONT):
        lift = float(road_profile(xw - x_in, road["h"], road["len"])) if road["show"] else 0.0
        wy[xw] = 17 + lift
        ax.add_patch(Circle((xw, wy[xw]), 17, fc="#2B2F36", ec="none", zorder=3))
    pts = np.array([[-135, 50], [135, 50], [138, 82], [70, 90], [38, 126], [-70, 126], [-112, 92], [-135, 90]], float)
    c = np.array([0, 80.0])
    a = math.radians(ang_deg)                  # 機首上げ (正) = 画面で反時計回り
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    off = np.array([0, z_px])
    ax.add_patch(Polygon((pts - c) @ R.T + c + off, closed=True, fc="#DCE6F2", ec=NAVY, lw=2, zorder=4))
    for xw, y in wy.items():
        pb = (np.array([xw, 50.0]) - c) @ R.T + c + off
        ax.plot([xw, pb[0]], [y, pb[1]], color=MUTED, lw=2, zorder=2)
    ax.annotate("", xy=(150, 150), xytext=(105, 150), arrowprops=dict(arrowstyle="-|>", color=MUTED))
    ax.text(103, 157, "進行方向", fontsize=8, color=MUTED)
    ax.set_xlim(-155, 155); ax.set_ylim(-14, 165); ax.set_aspect("equal"); ax.axis("off")


def wheel_color(q):
    """輪速の揺れの強さ q (0〜1) を、濃い灰 → オレンジ → 赤の色にする。"""
    q = float(np.clip(q, 0, 1))
    c0, c1, c2 = np.array([43, 47, 54]), np.array([224, 138, 46]), np.array([192, 57, 43])
    c = c0 + (c1 - c0) * min(q * 2, 1) if q < 0.5 else c1 + (c2 - c1) * (q - 0.5) * 2
    return tuple(c / 255)


def rigid_rear(ax, z_px, ang_deg, q_l, q_r, mark_l, mark_r):
    """後ろから見た車 (後輪 RL・RR)。タイヤも車体と一体で上下・ロールする。タイヤの色 = その輪の輪速の揺れ。
    mark_*: 路面の入力がその側の車輪の下にあるとき、地面に印 (高さは描かない)。"""
    from matplotlib.patches import Circle, Polygon, Rectangle
    ax.plot([-125, 125], [0, 0], color="#9AA5B4", lw=2)
    for xw, mk in ((-62, mark_l), (62, mark_r)):
        if mk:
            ax.add_patch(Rectangle((xw - 26, -5), 52, 5, fc=ORANGE, ec="none", zorder=1))
    c = np.array([0, 62.0])
    a = -math.radians(ang_deg)
    Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    tf = lambda P: (np.asarray(P, float) - c) @ Rm.T + c + [0, z_px]  # noqa: E731
    pts = [[-80, 32], [80, 32], [80, 72], [58, 74], [44, 112], [-44, 112], [-58, 74], [-80, 72]]
    for xw, q, name in ((-62, q_l, "RL"), (62, q_r, "RR")):
        ctr = tf([[xw, 17]])[0]
        ax.add_patch(Circle(ctr, 17, fc=wheel_color(q), ec="none", zorder=3))
        ax.text(ctr[0], ctr[1] - 30, name, fontsize=7, color=MUTED, ha="center")
    ax.add_patch(Polygon(tf(pts), closed=True, fc="#DCE6F2", ec=NAVY, lw=2, zorder=4))
    ax.add_patch(Polygon(tf([[-38, 78], [38, 78], [30, 106], [-30, 106]]), closed=True, fc="#9DB8D6", ec="none", zorder=5))
    ax.text(-118, 128, "左", fontsize=9, color=MUTED)
    ax.text(108, 128, "右", fontsize=9, color=MUTED)
    ax.set_xlim(-125, 125); ax.set_ylim(-24, 145); ax.set_aspect("equal"); ax.axis("off")


def rigid_side(ax, z_px, ang_deg, x_in, road, q_f, q_r):
    """横から見た車 (右が前。見えているのは右側の FR・RR)。タイヤも車体と一体。地面の印 = 路面の入力の位置 (高さは描かない)。"""
    from matplotlib.patches import Circle, Polygon, Rectangle
    ax.plot([-155, 155], [0, 0], color="#9AA5B4", lw=2)
    if road["show"] and abs(x_in) < 170:
        ax.add_patch(Rectangle((x_in - road["len"] / 2, -5), road["len"], 5, fc=ORANGE, ec="none", zorder=1))
    c = np.array([0, 60.0])
    a = math.radians(ang_deg)
    Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    tf = lambda P: (np.asarray(P, float) - c) @ Rm.T + c + [0, z_px]  # noqa: E731
    for xw, q, name in ((SIDE_REAR, q_r, "RR"), (SIDE_FRONT, q_f, "FR")):
        ctr = tf([[xw, 17]])[0]
        ax.add_patch(Circle(ctr, 17, fc=wheel_color(q), ec="none", zorder=5))
        ax.text(ctr[0], ctr[1] - 30, name, fontsize=7, color=MUTED, ha="center")
    pts = [[-135, 30], [135, 30], [138, 62], [70, 70], [38, 106], [-70, 106], [-112, 72], [-135, 70]]
    ax.add_patch(Polygon(tf(pts), closed=True, fc="#DCE6F2", ec=NAVY, lw=2, zorder=4))
    ax.annotate("", xy=(150, 128), xytext=(105, 128), arrowprops=dict(arrowstyle="-|>", color=MUTED))
    ax.text(103, 135, "進行方向", fontsize=8, color=MUTED)
    ax.set_xlim(-155, 155); ax.set_ylim(-24, 145); ax.set_aspect("equal"); ax.axis("off")


def road_state(road, tt):
    """時刻 tt (前輪の当たりからの秒) の、路面の入力の位置とタイヤの持ち上がり (模式)。"""
    x_in = SIDE_FRONT - road["v_ms"] * tt / road["m_per_unit_side"]          # 前輪の下に来るのが tt = 0
    lf = float(road_profile(SIDE_FRONT - x_in, road["h"], road["len"]))
    lr = float(road_profile(SIDE_REAR - x_in, road["h"], road["len"]))
    lift = lf if abs(lf) >= abs(lr) else lr                                  # 後ろから見ると前後の車輪が重なる
    on_l, on_r = road["side"] in ("左", "両輪"), road["side"] in ("右", "両輪")
    return x_in, (lift if on_l else 0.0), (lift if on_r else 0.0)


def panel_png(mot, iso, tt, e, gains, lim, w_px, h_px, out: Path, mode="road"):
    import matplotlib.pyplot as plt
    i = int(np.argmin(np.abs(mot["t"] - tt)))
    j = int(np.argmin(np.abs(iso["t"] - tt)))
    hv, rr_, pr = mot["heave"][i], mot["roll"][i], mot["pitch"][i]
    fig = plt.figure(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.text(0.03, 0.965, f"{e.event_id}  {e.role}", fontsize=10, fontweight="bold", color=NAVY, va="top")
    fig.text(0.03, 0.915, f"ISO 2631-1 の MTVV {e.mtvv_total:.2f} m/s² ({e.comfort})・普段の走行の {e.mtvv_total_normal_pct:.1f} パーセンタイル",
             fontsize=9, color=RED if e.cand else NAVY, va="top")
    road = e.road
    x_in, ll, lr_ = road_state(road, tt)
    a1 = fig.add_axes([0.02, 0.38, 0.46, 0.48])
    a2 = fig.add_axes([0.50, 0.38, 0.48, 0.48])
    if mode == "rigid":
        q = {k: mot[f"{k}_env"][i] / lim["wheel_env"] for k in V.WHEELS}
        under = lambda xw: road["show"] and abs(xw - x_in) < road["len"] / 2 + 17  # noqa: E731
        on = under(SIDE_FRONT) or under(SIDE_REAR)
        rigid_rear(a1, hv * gains["unit_per_ms2"], rr_ * gains["deg_per_dps"], q["RL"], q["RR"],
                   on and road["side"] in ("左", "両輪"), on and road["side"] in ("右", "両輪"))
        rigid_side(a2, hv * gains["unit_per_ms2"], pr * gains["deg_per_dps"], x_in, road, q["FR"], q["RR"])
        a1.set_title("後ろから見た図 (後輪 RL・RR): 上下とロール", fontsize=8.5, color=NAVY)
        a2.set_title("横から見た図 (右側 FR・RR): 上下とピッチ", fontsize=8.5, color=NAVY)
        ab = fig.add_axes([0.73, 0.245, 0.24, 0.085])             # 4 輪の輪速の揺れ (0.1 s の移動 RMS)
        vals = [mot[f"{k}_env"][i] for k in V.WHEELS]
        ab.barh(range(4), vals, color=[wheel_color(v / lim["wheel_env"]) for v in vals], height=0.7)
        ab.set_yticks(range(4)); ab.set_yticklabels(V.WHEELS, fontsize=6.5); ab.invert_yaxis()
        ab.set_xlim(0, lim["wheel_env"]); ab.tick_params(axis="x", labelsize=6)
        ab.set_title("輪速の揺れ [km/h] (0.1 s RMS)", fontsize=7, color=NAVY, pad=2)
        for sp_ in ("top", "right"):
            ab.spines[sp_].set_visible(False)
    else:
        body_rear(a1, hv * gains["unit_per_ms2"], rr_ * gains["deg_per_dps"], ll, lr_, ll, lr_)
        body_side(a2, hv * gains["unit_per_ms2"], pr * gains["deg_per_dps"], x_in, road)
        a1.set_title("後ろから見た図: 上下とロール", fontsize=9, color=NAVY)
        a2.set_title("横から見た図: 上下とピッチ", fontsize=9, color=NAVY)
    if mode == "rigid":
        fig.text(0.03, 0.335, f"いま: 上下 {hv:+5.2f} m/s²  ロール {rr_:+5.2f} deg/s  ピッチ {pr:+5.2f} deg/s", fontsize=8.6, color=NAVY)
        fig.text(0.03, 0.29, f"模式図: タイヤと車体は一体。ずれ ∝ 上下加速度 (1 m/s² → {gains['unit_per_ms2'] * MM_PER_UNIT / 10:.0f} cm 相当)、\n"
                 f"傾き ∝ 角速度 (1 deg/s → {gains['deg_per_dps']:.1f}°)。変位・角度ではない。タイヤの色 = その輪の輪速の揺れ",
                 fontsize=7.3, color=MUTED, va="center", linespacing=1.3)
    else:
        fig.text(0.03, 0.335, f"いま: 上下加速度 {hv:+5.2f} m/s²   ロール角速度 {rr_:+5.2f} deg/s   ピッチ角速度 {pr:+5.2f} deg/s",
                 fontsize=9, color=NAVY)
        fig.text(0.03, 0.29, f"模式図: 車体のずれは上下加速度に、傾きは角速度に比例 (1 m/s² → {gains['unit_per_ms2'] * MM_PER_UNIT / 10:.0f} cm 相当、"
                 f"1 deg/s → {gains['deg_per_dps']:.1f}°)。変位・角度ではない", fontsize=7.8, color=MUTED)
    # ISO メーター
    am = fig.add_axes([0.06, 0.10, 0.88, 0.07])
    cur = float(iso["tot"][j]) if np.isfinite(iso["tot"][j]) else 0.0
    am.barh([0], [lim["iso"]], color="#EEF2F7", height=0.8)
    am.barh([0], [min(cur, lim["iso"])], color=RED if cur >= 1.25 else (ORANGE if cur >= 0.8 else SKY), height=0.8)
    for th, name in COMFORT_TICKS:
        if th <= lim["iso"]:
            am.axvline(th, color="#9AA5B4", lw=0.8)
            am.text(th, 0.75, name, fontsize=6.5, color=MUTED, ha="center", va="bottom")
    am.axvline(e.mtvv_total, color=RED, lw=1.5, ls="--")
    am.set_xlim(0, lim["iso"]); am.set_ylim(-0.6, 1.4); am.set_yticks([]); am.tick_params(labelsize=7)
    for s in ("top", "right", "left"):
        am.spines[s].set_visible(False)
    fig.text(0.06, 0.205, f"ISO 2631-1 座面の合成 (5 軸) の 1 秒移動 RMS: いま {cur:.2f} m/s²  (赤の破線 = このイベントの MTVV)", fontsize=8.5, color=NAVY)
    if mode == "rigid":
        fig.text(0.03, 0.865, f"地面の橙の印 = 路面の入力の位置 ({road['side']}。側の指標 {road['side_index']:+.2f})。前輪 0 s・後輪 +{road['lag_s']:.2f} s で通る。"
                 "凸凹の形・高さは描かない", fontsize=7.6, color=ORANGE, va="top")
    else:
        fig.text(0.03, 0.865, f"路面の入力 (橙): {road['side']}・{'凸' if road['h'] > 0 else '凹'} ({road['basis']})。前輪 0 s・後輪 +{road['lag_s']:.2f} s で通る。形と高さは模式",
                 fontsize=7.6, color=ORANGE, va="top")
    fig.text(0.06, 0.02, "信号は帯域通過 (0.7 Hz〜) した IMU の値。IMU はフロントガラスの取付部 (座面ではない)。比例の係数は全動画で共通",
             fontsize=7, color=MUTED)
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs/vibration_clip_overlay.yaml")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tmp", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="このイベントだけ動画にする (倍率・縦軸は設定の全イベントで決める)")
    ap.add_argument("--illustration", choices=["road", "rigid"], default=None,
                    help="road: タイヤが路面の入力の模式に沿って動く / rigid: タイヤと車体が一体で、タイヤの色 = 輪速の揺れ")
    a = ap.parse_args()
    oc = yaml.safe_load(a.config.read_text())
    cfg = yaml.safe_load((REPO / oc["base_config"]).read_text())
    out = REPO / oc["out_dir"] / a.run_id
    if out.exists():
        raise SystemExit(f"{out} がすでにある。上書きしない")
    import matplotlib
    matplotlib.use("Agg")
    from near_miss.plotting import configure_japanese_font
    configure_japanese_font()
    from PIL import Image, ImageDraw, ImageFont

    ev = pd.read_csv(REPO / oc["events_csv"]).merge(pd.read_csv(REPO / oc["iso_csv"]), on="event_id")
    keys = ["strong_vertical", "strong_roll", "simultaneous_vertical_roll", "impulsive_response", "road_related_evidence"]
    ev["cand"] = ev[keys].all(axis=1)
    ev["n_conditions"] = ev[keys].sum(axis=1)
    loader = V.MeasuredLoader([REPO / r for r in oc["roots"]], cfg)
    raw = IE.RawLoader(loader, cfg["grid_hz"])
    sel = []
    for spec in oc["events"]:
        e = ev[ev.event_id == spec["id"]].iloc[0].copy()
        e["role"] = spec["role"]
        mot, iso = load_event(e, loader, raw, cfg, oc)
        e["road"] = road_input(e, mot, oc)
        sel.append((e, mot, iso))
    # 共通の倍率と縦軸
    T = oc["clip_s"]
    win = lambda m: (m["t"] >= -T) & (m["t"] <= T)  # noqa: E731
    max_rate = max(max(np.max(np.abs(m["roll"][win(m)])), np.max(np.abs(m["pitch"][win(m)]))) for _, m, _ in sel)
    max_h = max(np.max(np.abs(m["heave"][win(m)])) for _, m, _ in sel)
    # 模式図の比例の係数 (2 件で共通)。最大がこの見た目になるように、きりのよい値にする
    gains = dict(deg_per_dps=round(oc["display_max_angle_deg"] / max_rate * 2) / 2,
                 unit_per_ms2=round(oc["display_max_heave_px"] / max_h))
    lim = dict(T=T, heave=1.1 * max(np.max(np.abs(m["heave"][win(m)])) for _, m, _ in sel),
               rate=1.1 * max(max(np.max(np.abs(m["roll"][win(m)])), np.max(np.abs(m["pitch"][win(m)]))) for _, m, _ in sel),
               iso=max(2.0, 1.15 * max(i["mtvv"] for _, _, i in sel)))
    mode = a.illustration or oc.get("illustration", "road")
    lim["wheel"] = 1.1 * max(max(np.max(np.abs(m[k][win(m)])) for k in V.WHEELS) for _, m, _ in sel)
    lim["wheel_env"] = max(max(np.max(m[f"{k}_env"][win(m)]) for k in V.WHEELS) for _, m, _ in sel)
    if a.only:
        sel = [x for x in sel if x[0].event_id in a.only]
    (out / "plots").mkdir(parents=True)
    vc = oc["video"]
    W, cw = vc["width"], vc["camera_w"]
    font = ImageFont.truetype(vc["font"], 24)
    small = ImageFont.truetype(vc["font"], 19)
    rows = []
    for e, mot, iso in sel:
        with tempfile.TemporaryDirectory(dir=a.tmp) as td:
            tmp = Path(td)
            paths, times = video_frames(e, loader, oc, tmp)
            ch = int(round(cw * 874 / 1164 / 2) * 2)
            ts_img, mp = timeseries_png(mot, iso, e, lim, W, out / "plots" / f"{e.event_id}_timeseries.png", wheels=(mode == "rigid"))
            local = (pd.to_datetime(e.event_utc) - pd.Timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S")   # PDT (既存の資料と同じ)
            # 実時間 (カメラの各フレーム) のあとに、当たりの前後をスロー再生 (信号の刻みで描き、カメラは近いフレームを繰り返す)
            sm = oc["slow_motion"]
            plan = [(fp, tt, d_, False) for fp, tt, d_ in zip(paths, times, list(np.diff(times)) + [float(np.median(np.diff(times)))])]
            for tt in np.arange(sm["from_s"], sm["to_s"] + 1e-9, sm["step_s"]):
                plan.append((paths[int(np.argmin(np.abs(times - tt)))], float(tt), sm["step_s"] * sm["factor"], True))
            framelist, durs = [], []
            for k, (fp, tt, dur, slow) in enumerate(plan):
                pp = tmp / f"panel_{k:04d}.png"
                panel_png(mot, iso, tt, e, gains, lim, W - cw, ch, pp, mode)
                cam = Image.open(fp).convert("RGB").resize((cw, ch), Image.BILINEAR)
                ts_ = Image.fromarray(ts_img)
                canvas = Image.new("RGB", (W, ch + ts_.height + (ts_.height % 2)), "white")
                canvas.paste(cam, (0, 0))
                canvas.paste(Image.open(pp).convert("RGB"), (cw, 0))
                d = ImageDraw.Draw(ts_)
                x = mp["x0"] + (tt + T) / (2 * T) * (mp["x1"] - mp["x0"])
                d.line([(x, mp["ytop"]), (x, mp["ybot"])], fill=(220, 20, 60), width=3)
                canvas.paste(ts_, (0, ch))
                dc = ImageDraw.Draw(canvas, "RGBA")
                lines = [f"{e.event_id}  {local} PDT  {e.speed_kmh:.0f} km/h  地点 {int(e.site_id)}",
                         f"前輪が当たった推定時刻から {tt:+.2f} s"]
                dc.rectangle([0, 0, 640, 76], fill=(0, 0, 0, 150))
                dc.text((10, 6), lines[0], font=small, fill=(255, 255, 255))
                dc.text((10, 38), lines[1], font=font, fill=(255, 255, 255))
                tag = "異常振動ヒヤリハット候補" if e.cand else f"候補ではない (5 条件のうち {int(e.n_conditions)} つ)"
                tw = dc.textlength(tag, font=font)
                dc.rectangle([cw - tw - 28, 0, cw, 44], fill=(192, 57, 43, 220) if e.cand else (47, 111, 176, 220))
                dc.text((cw - tw - 14, 7), tag, font=font, fill=(255, 255, 255))
                if slow:
                    msg = f"スロー再生 1/{sm['factor']:.0f} (カメラは 20 Hz のフレームを繰り返す)"
                    tw2 = dc.textlength(msg, font=font)
                    dc.rectangle([0, ch - 48, tw2 + 28, ch], fill=(224, 138, 46, 230))
                    dc.text((14, ch - 42), msg, font=font, fill=(255, 255, 255))
                fpath = tmp / f"c_{k:04d}.jpg"
                canvas.save(fpath, quality=90)
                framelist.append(fpath)
                durs.append(dur)
            lf = tmp / "list.txt"
            with lf.open("w") as fh:
                for p_, d_ in zip(framelist, durs):
                    fh.write(f"file '{p_}'\nduration {d_:.6f}\n")
                fh.write(f"file '{framelist[-1]}'\n")
            mp4 = out / f"{e.event_id}_{'candidate' if e.cand else 'non_candidate'}.mp4"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lf), "-fps_mode", "vfr",
                            "-c:v", "libx264", "-crf", str(vc["crf"]), "-preset", vc["preset"], "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(mp4)], check=True)
            for name, tt_ in (("t0", 0.0), ("rear", float(e.road["lag_s"]))):
                kk = len(paths) + int(np.argmin(np.abs(np.arange(sm["from_s"], sm["to_s"] + 1e-9, sm["step_s"]) - tt_)))
                Image.open(framelist[kk]).save(out / f"{e.event_id}_still_{name}.jpg", quality=88)
        rows.append(dict(event_id=e.event_id, role=e.role, candidate=bool(e.cand), n_conditions=int(e.n_conditions), drive=e.drive,
                         segment=int(e.segment), event_time_s=e.event_time_s, local_time_pdt=local, speed_kmh=e.speed_kmh, site_id=int(e.site_id),
                         mtvv_total=e.mtvv_total, mtvv_recomputed=round(iso["mtvv"], 4), comfort=e.comfort,
                         mtvv_normal_pct=e.mtvv_total_normal_pct, heave_peak_abs=e.heave_peak_abs, roll_peak_abs=e.roll_peak_abs,
                         pitch_peak_abs=e.pitch_peak_abs, road_side=e.road["side"], side_index=e.road["side_index"],
                         road_polarity="凸" if e.road["h"] > 0 else "凹", rear_lag_s=round(e.road["lag_s"], 3), n_frames=len(times), video=mp4.name))
        print(e.event_id, rows[-1]["mtvv_total"], rows[-1]["mtvv_recomputed"], len(times))
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    (out / "manifest.json").write_text(json.dumps(dict(run_id=a.run_id, created_local=datetime.now().isoformat(timespec="seconds"),
                                                       script_sha256=sha(__file__), config=oc, config_sha256=sha(a.config), gains=gains,
                                                       illustration=mode, only=a.only,
                                                       limits={k: float(v) for k, v in lim.items()}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
