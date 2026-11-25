"""Udacity CH2_002 の 1 本の bag (既定 HMB_2) で、下り坂と反復制動の熱負荷、制動の応答の変化、前方映像を対応付ける。

入力: udacity_bag_extract.py の出力 (signals/*.csv.gz) と、映像用に元の bag。
手順:
1. 20 Hz にそろえる。車速 = 4 輪の輪速 (rad/s) の平均 × 実効半径 (steering_report.speed との比で決める)。
2. 勾配 = /fix の高度の差 ÷ 走行距離。気圧 (/pressure) の高度と、Xsens (/imu/data) の姿勢のピッチと比べる。
3. 力のつり合いの模型 a + g sin(theta) = kw*Tw - kb*Tb - c0 - c2*v^2 を 2 段で当てはめる (kw, c0, c2 はブレーキなしの時刻、kb は熱の少ない制動の時刻)。
   1 回目の熱の指標は減速度だけで出し (抵抗を無視)、cold を決めてから模型を当てはめ、熱の指標を出し直す。
4. 熱: ブレーキの仕事率 (W/kg) = kb*Tb*v。E60/E300 (直前の和) と H60/H300 (1 次遅れ)。
5. 1 回ごとの制動で、利き eta = (ブレーキによる減速度の実測) / (kb*Tb) と、ペダルからトルクへの比を測り、熱の指標と比べる。
   eta は cold で当てはめた模型を 1 とする比。下がっていてもフェードとは言わない (摩擦の変化と模型の誤差が区別できない)。
6. 熱の指標が最大の区間の前後で、前方映像と信号を並べた動画を作る (--video)。

    uv run --with rosbags python scripts/brake/udacity_hmb_heat.py --bag-name HMB_2 --run-id <新しい ID> [--video]
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from near_miss.plotting import configure_japanese_font

import brake_model as BM

OUT = Path("out/brake_fade")
CFG = Path("configs/brake/udacity_hmb.yaml")


def sig(ext: Path, topic: str) -> pd.DataFrame:
    return pd.read_csv(ext / "signals" / (topic.strip("/").replace("/", "__") + ".csv.gz"))


def interp(tg, D, col, t0):
    return np.interp(tg, D.t_header.to_numpy() - t0, D[col].to_numpy())


def movavg(x, n):
    n = max(1, int(n))
    return pd.Series(x).rolling(n, center=True, min_periods=1).mean().to_numpy()


def quat_pitch(x, y, z, w):
    """ROS の慣例 (x 前、y 左、z 上) で、前上がりを正にしたピッチ (rad)。"""
    return -np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))


def build(ext: Path, A: dict) -> tuple[pd.DataFrame, dict]:
    dt = A["grid_dt_s"]
    br, bi = sig(ext, "/vehicle/brake_report"), sig(ext, "/vehicle/brake_info_report")
    st, ws = sig(ext, "/vehicle/steering_report"), sig(ext, "/vehicle/wheel_speed_report")
    th, fx, pr, imu = sig(ext, "/vehicle/throttle_report"), sig(ext, "/fix"), sig(ext, "/pressure"), sig(ext, "/imu/data")
    t0 = max(D.t_header.min() for D in (br, bi, st, ws, th, fx, pr, imu))
    t1 = min(D.t_header.max() for D in (br, bi, st, ws, th, fx, pr, imu))
    tg = np.arange(0, t1 - t0, dt)
    om = np.mean([interp(tg, ws, c, t0) for c in ("front_left", "front_right", "rear_left", "rear_right")], axis=0)
    vs = interp(tg, st, "speed", t0)
    m = vs > 5
    r_eff = float(np.median(vs[m] / om[m]))
    v = movavg(om * r_eff, A["speed_smooth_s"] / dt)
    a = np.gradient(v, dt)
    s = np.cumsum(v) * dt
    alt = movavg(interp(tg, fx, "altitude", t0), A["grade"]["alt_smooth_s"] / dt)
    p = movavg(interp(tg, pr, "fluid_pressure", t0), A["grade"]["alt_smooth_s"] / dt)
    alt_baro = 44330.0 * (1 - (p / p[0]) ** (1 / 5.255))  # 走り始めを 0 とした気圧の高度 (標準大気の式)
    half = A["grade"]["ds_m"] / 2

    def grade(h):
        hp = np.interp(s + half, s, h)
        hm = np.interp(s - half, s, h)
        ds = np.interp(s + half, s, s) - np.interp(s - half, s, s)
        return np.where(ds > half, np.arcsin(np.clip((hp - hm) / np.maximum(ds, 1e-6), -0.3, 0.3)), np.nan)

    th_fix, th_baro = grade(alt), grade(alt_baro)
    q = [interp(tg, imu, f"orientation.{k}", t0) for k in "xyzw"]
    pitch = movavg(quat_pitch(*q), 1.0 / dt)
    ax_imu = movavg(interp(tg, imu, "linear_acceleration.x", t0), 1.0 / dt)
    D = pd.DataFrame(dict(
        t=tg, v=v, a=a, s=s, alt=alt, alt_baro=alt_baro, theta=th_fix if A["grade"]["source"] == "fix" else th_baro,
        theta_fix=th_fix, theta_baro=th_baro, pitch=pitch, ax_imu=ax_imu,
        pedal=interp(tg, br, "pedal_output", t0), torque_rep=interp(tg, br, "torque_output", t0),
        Tb=interp(tg, bi, "brake_torque_actual", t0), Treq=interp(tg, bi, "brake_torque_request", t0),
        Tw=interp(tg, bi, "wheel_torque_actual", t0), aog=interp(tg, bi, "accel_over_ground", t0),
        abs=interp(tg, bi, "abs_active", t0), thr=interp(tg, th, "pedal_output", t0),
        boo=interp(tg, br, "boo_output", t0) > 0.5))
    info = dict(t0_unix=float(t0), duration_s=float(tg[-1]), r_eff_m=r_eff)
    return D, info


def fit_model(D, nobrake, cold_brake, g):
    """2 段で当てはめる。1) ブレーキを使っていない時刻で y = kw*Tw - c0 - c2*v^2。
    2) 熱の少ない制動の時刻で、ブレーキによる減速度 d = kw*Tw - c0 - c2*v^2 - y を d = kb*Tb (原点を通る) に当てはめる。"""
    y = (D.a + g * np.sin(D.theta)).to_numpy()
    X = np.c_[D.Tw, -np.ones(len(D)), -D.v ** 2]
    ok1 = nobrake & np.isfinite(y)
    c1, *_ = np.linalg.lstsq(X[ok1], y[ok1], rcond=None)
    d = X @ c1 - y
    ok2 = cold_brake & np.isfinite(d)
    Tb = D.Tb.to_numpy()
    kb = float(np.sum(Tb[ok2] * d[ok2]) / np.sum(Tb[ok2] ** 2))
    pred = X @ c1 - kb * Tb
    return dict(kw=c1[0], kb=kb, c0=c1[1], c2=c1[2]), pred, ok1, ok2, d


def energy_budget(D, co, dt, g, t_max_search):
    """最も高い所から、その後 (t_max_search まで) の最も低い所までの下りで、位置エネルギーの行き先を分ける (J/kg)。"""
    i_top = int(D.alt.idxmax())
    sub = D[(D.t > D.t[i_top]) & (D.t < t_max_search)]
    if sub.empty:
        return None
    i_end = int(sub.alt.idxmin())
    W = D.loc[i_top:i_end]
    pt = co["kw"] * W.Tw * W.v * dt
    out = dict(t_top=float(W.t.iloc[0]), t_bottom=float(W.t.iloc[-1]), drop_m=float(W.alt.iloc[0] - W.alt.iloc[-1]),
               distance_m=float(W.s.iloc[-1] - W.s.iloc[0]),
               potential_Jkg=float(g * (W.alt.iloc[0] - W.alt.iloc[-1])), kinetic_gain_Jkg=float(0.5 * (W.v.iloc[-1] ** 2 - W.v.iloc[0] ** 2)),
               brake_Jkg=float((W.p_brake * dt).sum()), powertrain_negative_Jkg=float(-pt[pt < 0].sum()),
               powertrain_positive_Jkg=float(pt[pt > 0].sum()), resistance_Jkg=float(((co["c0"] + co["c2"] * W.v ** 2) * W.v * dt).sum()),
               brake_on_frac=float((W.p_brake > 0).mean()), mean_speed_kmh=float(W.v.mean() * 3.6))
    out["mean_grade_deg"] = float(np.degrees(np.arcsin(out["drop_m"] / max(out["distance_m"], 1.0))))
    out["residual_Jkg"] = (out["potential_Jkg"] + out["powertrain_positive_Jkg"] - out["powertrain_negative_Jkg"] - out["brake_Jkg"]
                           - out["resistance_Jkg"] - out["kinetic_gain_Jkg"])
    out["brake_share_of_potential"] = out["brake_Jkg"] / out["potential_Jkg"] if out["potential_Jkg"] > 0 else None
    return out


def events(D, on, E, dt):
    rows = []
    idx = np.where(np.diff(np.r_[0, on.astype(int), 0]))[0].reshape(-1, 2)
    for i0, i1 in idx:
        g = D.iloc[i0:i1]
        if (i1 - i0) * dt < E["min_duration_s"] or g.Tb.max() < E["min_peak_Nm"]:
            continue
        st = g[g.Tb >= E["steady_frac_of_peak"] * g.Tb.max()]
        if len(st) * dt < E["min_steady_s"]:
            continue
        rows.append(dict(i0=i0, i1=i1, t0=g.t.iloc[0], t1=g.t.iloc[-1], dur=(i1 - i0) * dt, v0=g.v.iloc[0], v1=g.v.iloc[-1],
                         Tb_peak=g.Tb.max(), Tb_steady=st.Tb.mean(), pedal_steady=st.pedal.mean(),
                         d_brake_obs=st.d_brake.mean(), d_brake_pred=st.d_pred_brake.mean(), theta_deg=np.degrees(st.theta.mean()),
                         H300_start=g.H300.iloc[0], E300_start=g.E300.iloc[0], H60_start=g.H60.iloc[0], abs_any=bool((g["abs"] > 0.5).any())))
    cols = ["i0", "i1", "t0", "t1", "dur", "v0", "v1", "Tb_peak", "Tb_steady", "pedal_steady", "d_brake_obs", "d_brake_pred",
            "theta_deg", "H300_start", "E300_start", "H60_start", "abs_any"]
    return pd.DataFrame(rows, columns=cols)


def overview_plot(D, Ev, S, path, win):
    configure_japanese_font()
    fig, ax = plt.subplots(6, 1, figsize=(12, 13), sharex=True)
    ax[0].plot(D.t, D.alt, label="高度 (/fix)", color="#2e86c1")
    ax[0].plot(D.t, D.alt_baro + D.alt.iloc[0], label="高度 (気圧から、走り始めを /fix にそろえた)", color="#888", lw=1)
    ax[0].set_ylabel("高度 [m]")
    ax[0].legend(fontsize=8)
    ax[1].plot(D.t, np.degrees(D.theta_fix), label="勾配 (/fix の高度 ÷ 距離)", color="#2e86c1")
    ax[1].plot(D.t, np.degrees(D.theta_baro), label="勾配 (気圧の高度)", color="#888", lw=1)
    ax[1].plot(D.t, np.degrees(D.pitch), label="車体のピッチ (Xsens の姿勢)", color="#c0392b", lw=1)
    ax[1].axhline(0, color="#aaa", lw=0.6)
    ax[1].set_ylabel("角度 [deg]\n上りが正")
    ax[1].legend(fontsize=8)
    ax[2].plot(D.t, D.v * 3.6, color="#222")
    ax[2].set_ylabel("車速 [km/h]")
    ax[3].plot(D.t, D.Tb, color="#c0392b", label="ブレーキトルク brake_torque_actual [Nm]")
    a3 = ax[3].twinx()
    a3.plot(D.t, D.pedal, color="#7f8c8d", lw=0.8)
    a3.set_ylabel("ペダル (単位なし)", color="#7f8c8d")
    ax[3].set_ylabel("Nm")
    ax[3].legend(fontsize=8, loc="upper left")
    ax[4].plot(D.t, -D.a, color="#222", lw=0.8, label="減速度 -dv/dt")
    ax[4].plot(D.t, D.d_brake, color="#c0392b", lw=0.8, label="ブレーキによる減速度 (勾配・駆動・抵抗を引いた実測)")
    ax[4].set_ylabel("m/s²")
    ax[4].legend(fontsize=8)
    ax[5].plot(D.t, D.E300, color="#c0392b", label="E300: 直前 300 s のブレーキの仕事")
    ax[5].plot(D.t, D.H300, color="#e67e22", label="H300: 1 次遅れ (時定数 300 s)")
    ax[5].axhline(S["heat"]["ref_stop_Jkg"], color="#888", ls="--", lw=0.8)
    ax[5].text(D.t.iloc[5], S["heat"]["ref_stop_Jkg"], " 100 km/h からの停止 1 回分", fontsize=8, va="bottom", color="#555")
    ax[5].axhline(S["heat"]["comma2k19_rav4_E300_max_Jkg"], color="#2e86c1", ls=":", lw=0.8)
    ax[5].text(D.t.iloc[5], S["heat"]["comma2k19_rav4_E300_max_Jkg"], " comma2k19 RAV4 の E300 の最大", fontsize=8, va="bottom", color="#2e86c1")
    ax[5].set_ylabel("J/kg")
    ax[5].legend(fontsize=8, loc="upper right")
    ax[5].set_xlabel("時刻 [s] (bag の始まりから)")
    for x in ax:
        x.axvspan(*win, color="#f9e79f", alpha=0.3, lw=0)
        x.grid(alpha=0.3)
    fig.suptitle("Udacity CH2_002 HMB_2: 峠を越えて下る間の制動と熱の指標 (黄色 = 動画の区間)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def eta_plot(Ev, path):
    configure_japanese_font()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    sc = ax[0].scatter(Ev.H300_start, Ev.eta, c=Ev.t0, cmap="viridis", s=30, edgecolor="white", linewidth=0.5)
    ax[0].axhline(1, color="#888", lw=0.8)
    ax[0].set_xlabel("制動の始まりの熱の指標 H300 [J/kg]")
    ax[0].set_ylabel("利き eta (ブレーキの減速度の実測 ÷ 模型)\ncold で当てはめた模型 = 1")
    fig.colorbar(sc, ax=ax[0], label="時刻 [s]")
    ax[1].scatter(Ev.H300_start, Ev.pedal_gain, c=Ev.t0, cmap="viridis", s=30, edgecolor="white", linewidth=0.5)
    ax[1].set_xlabel("制動の始まりの熱の指標 H300 [J/kg]")
    ax[1].set_ylabel("ペダルからトルクへの比 [Nm / ペダル]")
    for x in ax:
        x.grid(alpha=0.3)
    fig.suptitle("1 回ごとの制動の応答と熱の指標 (フェードの判定ではない)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def read_frames(bag: Path, topic: str, t0_unix: float, w0: float, w1: float):
    from PIL import Image
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    store = get_typestore(Stores.ROS1_NOETIC)
    out = []
    with Reader(bag) as r:
        conns = [c for c in r.connections if c.topic == topic]
        for c, ts, data in r.messages(connections=conns, start=int((t0_unix + w0 - 1) * 1e9), stop=int((t0_unix + w1 + 1) * 1e9)):
            msg = store.deserialize_ros1(data, c.msgtype)
            th = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 - t0_unix
            out.append((th, bytes(msg.data)))
    times = np.array([o[0] for o in out])
    return times, [o[1] for o in out], Image


RANK_COL = {"High": "#C0392B", "Medium": "#E08A2E", "Low": "#8A9BB0"}


def make_video(D, S, bag, cam_topic, win, V, path, Ev, risk=None, risk_rank=None, slope=True):
    """risk: fade_risk_index.py の時系列 (t, score, rank)。あれば右に 6 段目を足し、映像に順位の札と色の枠を付ける。
    slope=False: 左下の勾配のイラストを描かず、映像を左の列いっぱいに置く。"""
    times, jpgs, Image = read_frames(bag, cam_topic, S["t0_unix"], *win)
    configure_japanese_font()
    fig = plt.figure(figsize=(14.4, 7.2), dpi=100)
    nr = 6 if risk is not None else 5
    gs = fig.add_gridspec(nr, 2, width_ratios=[1.15, 1], hspace=0.5, wspace=0.18, left=0.03, right=0.97, top=0.93, bottom=0.07)
    axc = fig.add_subplot(gs[0:nr - 1, 0] if slope else gs[0:nr, 0])
    axs = fig.add_subplot(gs[nr - 1, 0]) if slope else None
    axp = [fig.add_subplot(gs[i, 1]) for i in range(nr)]
    W = D[(D.t >= win[0]) & (D.t <= win[1])]
    panels = [("高度 [m]", [(W.alt, "#2e86c1")]),
              ("車速 [km/h]", [(W.v * 3.6, "#222")]),
              ("ブレーキトルク [Nm]", [(W.Tb, "#c0392b")]),
              ("減速度 [m/s²]", [(-W.a, "#888"), (W.d_brake, "#c0392b")]),
              ("熱の指標 [J/kg]", [(W.E300, "#c0392b"), (W.H300, "#e67e22")])]
    cursors = []
    for ax, (lab, ser) in zip(axp, panels):
        for y, c in ser:
            ax.plot(W.t, y, color=c, lw=1)
        ax.set_ylabel(lab, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        cursors.append(ax.axvline(win[0], color="#000", lw=1))
    axp[4].axhline(S["heat"]["ref_stop_Jkg"], color="#888", ls="--", lw=0.7)
    axp[4].text(win[0], S["heat"]["ref_stop_Jkg"], "100 km/h 停止 1 回分", fontsize=7, va="bottom", color="#555")
    axp[3].text(0.01, 0.85, "灰: -dv/dt  赤: ブレーキによる分", transform=axp[3].transAxes, fontsize=7)
    axp[4].text(0.01, 0.85, "赤: E300  橙: H300", transform=axp[4].transAxes, fontsize=7)
    axp[-1].set_xlabel("時刻 [s]", fontsize=8)
    if risk is not None:
        Rw = risk[(risk.t >= win[0]) & (risk.t <= win[1])]
        ar = axp[5]
        ar.axhspan(risk_rank["high"], 100, color="#FADBD8", lw=0)
        ar.axhspan(risk_rank["medium"], risk_rank["high"], color="#FDEBD0", lw=0)
        ar.plot(Rw.t, Rw.score, color="#1F2A44", lw=1.2)
        ar.set_ylim(0, 100)
        ar.set_ylabel("フェードリスク\n[0〜100]", fontsize=8)
        ar.tick_params(labelsize=7)
        ar.grid(alpha=0.3)
        ar.text(0.01, 0.80, "High ≥ 60 / Medium ≥ 30 (確率ではない)", transform=ar.transAxes, fontsize=7)
        cursors.append(ar.axvline(win[0], color="#000", lw=1))
        for a in axp[:5]:
            a.set_xlabel("")
        rt, rsc, rrk = risk.t.to_numpy(), risk.score.to_numpy(), risk["rank"].to_numpy()
        badge = axc.text(0.99, 0.99, "", transform=axc.transAxes, va="top", ha="right", fontsize=14, color="white", fontweight="bold",
                         bbox=dict(facecolor="#8A9BB0", alpha=0.9, lw=0, boxstyle="round,pad=0.4"))
        border = plt.Rectangle((0, 0), 1, 1, transform=axc.transAxes, fill=False, lw=6, ec="#8A9BB0", clip_on=False)
        axc.add_patch(border)
    img = axc.imshow(np.zeros((480, 640, 3), np.uint8))
    axc.axis("off")
    txt = axc.text(0.01, 0.99, "", transform=axc.transAxes, va="top", fontsize=11, color="white",
                   bbox=dict(facecolor="black", alpha=0.55, lw=0))
    if slope:
        axs.set_xlim(-60, 60)
        axs.set_ylim(-12, 12)
        axs.set_aspect("equal", adjustable="datalim")  # 勾配の角度をそのままの見た目で描く
        axs.axis("off")
        road, = axs.plot([], [], color="#555", lw=3)
        car, = axs.plot([], [], color="#c0392b", lw=8, solid_capstyle="round")
        stxt = axs.text(0, -11, "", ha="center", fontsize=9)
    fig.suptitle("Udacity CH2_002 HMB_2 (前方カメラと制動の信号)。熱の指標は推定で、ブレーキ温度は記録されていない", fontsize=10)
    step = V["speedup"] / V["fps"]
    tt = np.arange(win[0], win[1], step)
    W2, H2 = fig.canvas.get_width_height()
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W2}x{H2}", "-r", str(V["fps"]),
                             "-i", "-", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "23", str(path)],
                            stdin=subprocess.PIPE)
    tv, thv, vv, Tbv, dv = D.t.to_numpy(), D.theta.to_numpy(), D.v.to_numpy(), D.Tb.to_numpy(), D.d_brake.to_numpy()
    E3, ped = D.E300.to_numpy(), D.pedal.to_numpy()
    for t in tt:
        k = int(np.argmin(np.abs(times - t)))
        img.set_data(np.asarray(Image.open(io.BytesIO(jpgs[k])).convert("RGB")))
        i = int(np.searchsorted(tv, t))
        i = min(i, len(tv) - 1)
        for c in cursors:
            c.set_xdata([t, t])
        ang = thv[i] if np.isfinite(thv[i]) else 0.0
        if slope:
            x = np.array([-60, 60])
            road.set_data(x, np.tan(ang) * x)
            cx = np.array([-6, 6]) * np.cos(ang)
            car.set_data(cx, np.tan(ang) * cx + 1.2)
            stxt.set_text(f"勾配 {np.degrees(ang):+.1f}° (実測、縦横比 1:1)")
        txt.set_text(f"t = {t:6.1f} s   {vv[i] * 3.6:5.1f} km/h\nペダル {ped[i]:.3f}   ブレーキトルク {Tbv[i]:4.0f} Nm\n"
                     f"ブレーキの減速度 {dv[i]:+.2f} m/s²   E300 {E3[i]:4.0f} J/kg"
                     + ("" if slope else f"\n勾配 {np.degrees(ang):+.1f}° (GPS の高度から)"))
        if risk is not None:
            j = min(int(np.searchsorted(rt, t)), len(rt) - 1)
            col = RANK_COL[rrk[j]]
            badge.set_text(f"フェードリスク {rsc[j]:.0f}  {rrk[j]}")
            badge.get_bbox_patch().set_facecolor(col)
            border.set_edgecolor(col)
        fig.canvas.draw()
        proc.stdin.write(fig.canvas.buffer_rgba().tobytes())
    proc.stdin.close()
    proc.wait()
    plt.close(fig)
    # 静止画 (代表の瞬間: E300 が最大になる時刻の少し前の、最も強い制動)
    return dict(n_frames=len(tt), camera_frames_in_window=len(times), camera_dt_median=float(np.median(np.diff(times))))


def snapshot(D, S, bag, cam_topic, t_pick, path):
    times, jpgs, Image = read_frames(bag, cam_topic, S["t0_unix"], t_pick - 0.5, t_pick + 0.5)
    k = int(np.argmin(np.abs(times - t_pick)))
    Image.open(io.BytesIO(jpgs[k])).save(path)
    return float(times[k])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag-name", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--risk-run", default=None, help="fade_risk_index.py の run (例 20261119_r31_fade_risk_index)。動画にフェードリスクを重ねる")
    ap.add_argument("--window", type=float, nargs=2, default=None, metavar=("START_S", "END_S"), help="動画の時間窓 [s] (既定: 熱の指標が最大の区間から自動で決める)")
    ap.add_argument("--speedup", type=float, default=None, help="動画の再生速度 (既定: udacity_hmb.yaml の video.speedup)")
    ap.add_argument("--no-slope", action="store_true", help="左下の勾配のイラストを描かない")
    ap.add_argument("--fps", type=float, default=None, help="動画の 1 秒あたりの枚数 (既定: udacity_hmb.yaml の video.fps)")
    a = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text())
    A = cfg["analysis"]
    name = a.bag_name or cfg["priority_bag"]
    ext = OUT / A["extract_run"] / name
    out = OUT / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    dt, g = A["grid_dt_s"], A["g"]
    D, info = build(ext, A)
    moving = D.v > A["min_speed_mps"]
    on = (D.Tb >= A["brake"]["torque_on_Nm"]) & moving
    blk = np.zeros(len(D), int)
    # 1 回目: 減速度だけの熱 (抵抗・駆動を無視) で cold を決める
    p0 = np.where(on, np.clip(-(D.a + g * np.sin(D.theta)), 0, None) * D.v, 0.0)
    H0 = BM.heat_indices(p0, blk, dt, [300.0], [])
    cold = (H0["H300"] < A["model"]["cold_max_Jkg"]) & moving.to_numpy()
    co, pred, ok1, ok2, dfit = fit_model(D, moving.to_numpy() & ~on.to_numpy(), cold & on.to_numpy(), g)
    y = (D.a + g * np.sin(D.theta)).to_numpy()
    fin = np.isfinite(y) & moving.to_numpy()
    r2_nobrake = BM.r2(y[ok1], pred[ok1])
    r2_cold_brake = BM.r2(dfit[ok2], co["kb"] * D.Tb.to_numpy()[ok2])
    hotb = fin & ~cold & on.to_numpy()
    r2_hot_brake = BM.r2(dfit[hotb], co["kb"] * D.Tb.to_numpy()[hotb]) if hotb.sum() > 50 else None
    r2_all = BM.r2(y[fin], pred[fin])
    # ブレーキによる減速度: 実測 = kw Tw - c0 - c2 v^2 - (a + g sin)、模型 = kb Tb
    D["d_brake"] = dfit
    D["d_pred_brake"] = co["kb"] * D.Tb
    power = np.where(on, co["kb"] * D.Tb * D.v, 0.0)
    D["p_brake"] = power
    H = BM.heat_indices(power, blk, dt, [float(x) for x in A["heat"]["taus_s"]], [float(x) for x in A["heat"]["windows_s"]])
    for k, v in H.items():
        D[k] = v
    p_dec = np.where(on, np.clip(D.d_brake, 0, None) * D.v, 0.0)
    E_dec = BM.heat_indices(p_dec, blk, dt, [], [300.0])["E300"]
    Ev = events(D, on.to_numpy(), A["events"], dt)
    Ev["eta"] = Ev.d_brake_obs / Ev.d_brake_pred
    Ev["pedal_gain"] = Ev.Tb_steady / (Ev.pedal_steady - A["brake"]["pedal_rest"])
    Ev.to_csv(out / "brake_events.csv", index=False)
    ref = 0.5 * (A["heat"]["ref_stop_kmh"] / 3.6) ** 2
    imax = int(np.argmax(D.E300))
    # 熱の指標が最大の区間 = E300 最大の時刻から 300 s 前まで (ただし制動のある範囲に縮める)
    w_on = D.t[on & (D.t > D.t[imax] - 300) & (D.t <= D.t[imax])]
    if w_on.empty:  # 制動がほとんど無い bag
        w_on = D.t[(D.t > D.t[imax] - 300) & (D.t <= D.t[imax])]
    de0 = energy_budget(D.assign(p_brake=power), co, dt, g, A["heat"].get("descent_search_until_s", {}).get(name, float(D.t.iloc[-1])))
    w_start = min(float(w_on.min()), de0["t_top"]) if de0 else float(w_on.min())  # 下りの頂上から映す
    win = (max(0.0, w_start - A["video"]["margin_s"]), min(float(D.t.iloc[-1]), float(D.t[imax]) + A["video"]["margin_s"]))
    seg = (D.t >= w_on.min()) & (D.t <= D.t[imax])
    alt_drop = float(D.alt[seg].max() - D.alt[D.t.between(D.t[seg].iloc[0], D.t[imax])].iloc[-1])
    cold_ev, hot_ev = Ev[Ev.H300_start < A["model"]["cold_max_Jkg"]], Ev[Ev.H300_start >= A["model"]["cold_max_Jkg"]]
    valid = np.isfinite(D.theta_fix) & np.isfinite(D.theta_baro) & moving
    S = dict(bag=name, **info, n_grid=len(D),
             grade_check=dict(
                 corr_fix_vs_baro=round(float(np.corrcoef(D.theta_fix[valid], D.theta_baro[valid])[0, 1]), 3),
                 corr_fix_vs_pitch=round(float(np.corrcoef(D.theta_fix[valid], D.pitch[valid])[0, 1]), 3),
                 median_abs_diff_fix_baro_deg=round(float(np.degrees(np.median(np.abs(D.theta_fix[valid] - D.theta_baro[valid])))), 2),
                 median_pitch_minus_grade_deg=round(float(np.degrees(np.median(D.pitch[valid] - D.theta_fix[valid]))), 2),
                 alt_range_fix_m=[round(float(D.alt.min()), 1), round(float(D.alt.max()), 1)],
                 alt_range_baro_m=[round(float(D.alt_baro.min()), 1), round(float(D.alt_baro.max()), 1)]),
             model=dict(form="a + g sin(theta) = kw*Tw - kb*Tb - c0 - c2*v^2", **{k: float(v) for k, v in co.items()},
                        kb_over_kw=float(co["kb"] / co["kw"]), implied_m_times_r_kgm=float(1 / co["kb"]),
                        implied_mass_kg_if_r_eff=float(1 / (co["kb"] * info["r_eff_m"])),
                        implied_mass_kg_from_kw=float(1 / (co["kw"] * info["r_eff_m"])),
                        r2_nobrake=round(float(r2_nobrake), 3), r2_brake_cold=round(float(r2_cold_brake), 3),
                        r2_brake_hot=None if r2_hot_brake is None else round(float(r2_hot_brake), 3), r2_all=round(float(r2_all), 3),
                        n_nobrake=int(ok1.sum()), n_brake_cold=int(ok2.sum()), n_brake_hot=int(hotb.sum())),
             heat=dict(ref_stop_Jkg=ref, comma2k19_rav4_E300_max_Jkg=A["heat"]["comma2k19_rav4_E300_max_Jkg"],
                       E300_max_Jkg=float(D.E300.max()), E300_max_in_ref_stops=round(float(D.E300.max() / ref), 2),
                       E300_max_t=float(D.t[imax]), E60_max_Jkg=float(D.E60.max()), H300_max_Jkg=float(D.H300.max()),
                       E300_max_decel_based_Jkg=float(E_dec.max()),
                       ratio_to_comma2k19_rav4_max=round(float(D.E300.max() / A["heat"]["comma2k19_rav4_E300_max_Jkg"]), 2),
                       brake_on_frac_in_window=round(float(on[seg].mean()), 3), alt_drop_in_window_m=round(alt_drop, 1),
                       potential_energy_drop_Jkg=round(g * alt_drop, 1), window_s=[float(w_on.min()), float(D.t[imax])],
                       mean_speed_window_kmh=round(float(D.v[seg].mean() * 3.6), 1)),
             events=dict(n=int(len(Ev)), n_abs=int(Ev.abs_any.sum()),
                         eta_median_cold=round(float(cold_ev.eta.median()), 3) if len(cold_ev) else None,
                         eta_median_hot=round(float(hot_ev.eta.median()), 3) if len(hot_ev) else None,
                         n_cold=int(len(cold_ev)), n_hot=int(len(hot_ev)),
                         spearman_eta_vs_H300=round(float(Ev.eta.rank().corr(Ev.H300_start.rank())), 3) if len(Ev) > 3 else None,
                         spearman_pedal_gain_vs_H300=round(float(Ev.pedal_gain.rank().corr(Ev.H300_start.rank())), 3) if len(Ev) > 3 else None,
                         pedal_gain_median_cold=round(float(cold_ev.pedal_gain.median()), 1) if len(cold_ev) else None,
                         pedal_gain_median_hot=round(float(hot_ev.pedal_gain.median()), 1) if len(hot_ev) else None),
             video_window_s=list(win))
    S["descent_energy"] = energy_budget(D, co, dt, g, A["heat"].get("descent_search_until_s", {}).get(name, float(D.t.iloc[-1])))
    overview_plot(D, Ev, S, out / "overview.png", win)
    eta_plot(Ev, out / "events_eta_vs_heat.png")
    D.iloc[::4].round(5).to_csv(out / "timeseries_5hz.csv.gz", index=False)
    bag = Path(cfg["bags_dir"]) / f"{name}.bag"
    # 代表の瞬間: 区間の中で熱の仕事率が最大の時刻
    t_rep = float(D.t[seg][D.p_brake[seg].idxmax()]) if seg.any() else float(D.t[imax])
    S["representative_moment"] = dict(t=t_rep, **{k: round(float(D.loc[D.t.sub(t_rep).abs().idxmin(), k]), 3)
                                                  for k in ("v", "pedal", "Tb", "d_brake", "E300", "H300", "theta", "alt")})
    if bag.exists():
        S["representative_moment"]["camera_t"] = snapshot(D, S, bag, cfg["camera_topic"], t_rep, out / "frame_representative.jpg")
        if a.video:
            risk, rr = None, None
            if a.risk_run:
                risk = pd.read_csv(OUT / a.risk_run / "udacity_hmb2_score.csv")[["t", "score", "rank"]]
                rr = yaml.safe_load(Path("configs/brake/fade_risk_index.yaml").read_text())["rank"]
                S["risk_run"] = a.risk_run
            vwin = tuple(a.window) if a.window else win
            V = dict(A["video"], **({"speedup": a.speedup} if a.speedup else {}), **({"fps": a.fps} if a.fps else {}))
            S["video"] = make_video(D, S, bag, cfg["camera_topic"], vwin, V, out / "scene.mp4", Ev, risk, rr, slope=not a.no_slope)
            S["video"].update(window_s=list(vwin), speedup=V["speedup"], fps=V["fps"], slope_illustration=not a.no_slope)
    (out / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    (out / "meta.json").write_text(json.dumps(dict(run_id=a.run_id, bag=name, config=CFG.as_posix(), analysis=A), ensure_ascii=False, indent=1))
    print(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    print(Ev.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
