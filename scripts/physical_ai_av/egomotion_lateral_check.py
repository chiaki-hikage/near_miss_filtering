#!/usr/bin/env python3
"""NVIDIA PhysicalAI-AV の推定済み Ego Motion に、横風などの外乱による短時間の横方向運動が残っているかを調べる。

チャンク 0 (100 クリップ、取得済み) だけを使う。追加の取得はしない。

  uv run --with pyarrow --with scipy --extra viz python scripts/physical_ai_av/egomotion_lateral_check.py \
      --out out/physical_ai_av/egomotion_check

前提 (docs/physical_ai_av.md、データで確認済み):
  - 時刻はクリップ開始からの μs。位置・速度・加速度は anchor frame (クリップ開始時の rig 座標、x 前 y 左 z 上)
  - 四元数は scipy の順 [qx, qy, qz, qw] (スカラーが最後)
  - 100 Hz なのは 20 秒のクリップの本体 (t = −0.2〜20 s) だけ。20 s より後 (最大 120 s) は平均 7 Hz 程度に
    間引かれている。100 Hz の等間隔に補間すると階段状の見かけの変動ができるので、本体だけを使う (WINDOW)

出力 (<out>/):
  clips.csv          クリップごとの走行条件
  stats.csv          高速 (> 15 m/s)・ほぼ直進の区間の、横速度・横滑り角・ヨーレート・横加速度の分布
  relations.csv      信号どうしの関係 (位置⇔速度、速度⇔加速度、曲率⇔ヨーレート/速度、横滑り角のモデル)
  spectra.csv        周波数帯ごとの RMS (NVIDIA と comma2k19 の生の IMU)
  offline_check.csv  online (100 Hz) と offline (10 Hz) のヨーレートの帯域ごとの一致
  events.csv         直進に近い区間の横方向の動きの候補 (点数の高い順)
  plots/             時系列とスペクトルの図
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.spatial.transform import Rotation as R, Slerp

warnings.filterwarnings("ignore", message=".*encountered in matmul")   # macOS の BLAS が出す誤報

FS = 100.0
DATA = Path("data/physical_ai_av")
EGO = DATA / "labels/egomotion/egomotion.chunk_0000.zip"
OFF = DATA / "labels/egomotion.offline/egomotion.offline.chunk_0000.zip"
HI_SPEED = 15.0          # m/s
STRAIGHT_R = np.radians(3)   # 直進とみなす: ±1.5 s の |ヨーレート| の最大が 3 deg/s 未満
STRAIGHT_WIN = 301
BANDS = [(0.1, 0.5), (0.5, 2), (2, 5), (5, 20), (20, 50)]
WINDOW = (-0.25, 20.05)  # 100 Hz で記録されている区間 [s]


def bp(x, lo, hi, fs=FS):
    b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def lp(x, fc, fs=FS):
    b, a = butter(2, fc / (fs / 2))
    return filtfilt(b, a, x)


def load_clip(zf: zipfile.ZipFile, name: str) -> dict | None:
    """1 クリップを 100 Hz の等間隔に直し、車体座標の量を作る。"""
    d = pd.read_parquet(io.BytesIO(zf.read(name))).sort_values("timestamp").drop_duplicates("timestamp")
    t_all = d.timestamp.to_numpy() / 1e6
    d = d[(t_all >= WINDOW[0]) & (t_all <= WINDOW[1])]
    t = d.timestamp.to_numpy() / 1e6
    if len(t) < 100:
        return None
    hz = (len(t) - 1) / (t[-1] - t[0])       # 区間の平均の行数 (間隔の中央値ではなく)
    g = np.arange(t[0], t[-1], 1 / FS)
    q = Slerp(t, R.from_quat(d[["qx", "qy", "qz", "qw"]].to_numpy()))(g)
    ip = lambda c: np.interp(g, t, d[c].to_numpy())
    pos = np.c_[ip("x"), ip("y"), ip("z")]
    v = np.c_[ip("vx"), ip("vy"), ip("vz")]
    a = np.c_[ip("ax"), ip("ay"), ip("az")]
    vb, ab = q.inv().apply(v), q.inv().apply(a)
    rel = (q[:-1].inv() * q[1:]).as_rotvec() * FS          # 車体座標の角速度
    r = np.r_[rel[:, 2], rel[-1, 2]]
    yaw = np.unwrap(q.as_euler("ZYX")[:, 0])
    return dict(clip=name.split(".")[0], t=g - g[0], t0=g[0], native_hz=hz, pos=pos, v=v, a=a, vb=vb, ab=ab, r=r, yaw=yaw,
                k=ip("curvature"), sp=vb[:, 0], beta=np.arctan2(vb[:, 1], vb[:, 0]),
                raw=dict(t=t, pos=d[["x", "y", "z"]].to_numpy(), v=d[["vx", "vy", "vz"]].to_numpy()))


def straight_mask(c):
    rmax = pd.Series(np.abs(lp(c["r"], 2))).rolling(STRAIGHT_WIN, center=True, min_periods=1).max().to_numpy()
    return (c["sp"] > HI_SPEED) & (rmax < STRAIGHT_R)


def segments(mask, minlen):
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(a, b) for a, b in zip(idx[::2], idx[1::2]) if b - a >= minlen]


# ---------------------------------------------------------------------------

def relations(c) -> dict:
    """信号どうしの関係。差が小さいほど、同じ推定の状態から導いた量の可能性が高い。"""
    raw = c["raw"]; m = c["sp"] > 8
    if m.sum() < 300:
        return {}
    vt = np.gradient(raw["pos"], raw["t"], axis=0)
    mr = np.hypot(raw["v"][:, 0], raw["v"][:, 1]) > 8
    e_pv = np.median(np.linalg.norm((vt - raw["v"])[mr, :2], axis=1))
    vs = pd.DataFrame(c["v"]).rolling(25, center=True, min_periods=1).mean().to_numpy()
    as_ = pd.DataFrame(c["a"]).rolling(25, center=True, min_periods=1).mean().to_numpy()
    e_va = np.median(np.linalg.norm((np.gradient(vs, c["t"], axis=0) - as_)[m, :2], axis=1))
    k_rv = np.where(m, c["r"] / np.where(m, c["sp"], 1.0), np.nan)
    e_k = np.sqrt(np.mean((c["k"] - k_rv)[m] ** 2)); k_rms = np.sqrt(np.mean(c["k"][m] ** 2))
    # 曲率の高い周波数: 曲率×速度 と ヨーレート の 2〜20 Hz 成分の比
    hf_ratio = np.std(bp(c["k"] * c["sp"], 2, 20)[m]) / max(np.std(bp(c["r"], 2, 20)[m]), 1e-12)
    # 横滑り角 = x·r/v + kc·a_y + b
    X = np.c_[c["r"][m] / c["sp"][m], c["ab"][m, 1], np.ones(m.sum())]
    coef, *_ = np.linalg.lstsq(X, c["beta"][m], rcond=None)
    res = c["beta"][m] - X @ coef
    # 横加速度 − v·r (= 横速度の変化 + 誤差)
    d_lat = c["ab"][:, 1] - c["sp"] * c["r"]
    return dict(clip=c["clip"], n=int(m.sum()), pos_vs_vel_ms=e_pv, vel_vs_acc_ms2=e_va,
                curv_vs_r_over_v=e_k, curv_rms=k_rms, curv_hf_ratio=hf_ratio,
                beta_x_m=coef[0], beta_kc_deg_per_ms2=np.degrees(coef[1]), beta_b_deg=np.degrees(coef[2]),
                beta_r2=1 - res.var() / c["beta"][m].var(),
                alat_minus_vr_lf=np.std(lp(d_lat, 0.5)[m]), alat_minus_vr_hf=np.std(bp(d_lat, 2, 20)[m]))


def stats(clips) -> pd.DataFrame:
    rows = {k: [] for k in ("横速度 [m/s]", "横滑り角 [deg]", "ヨーレート [deg/s]", "横加速度 [m/s2]", "横加速度 − v·r [m/s2]",
                            "ヨーレート 0.3〜3 Hz [deg/s]")}
    for c in clips:
        m = straight_mask(c)
        if m.sum() == 0:
            continue
        rows["横速度 [m/s]"].append(c["vb"][m, 1]); rows["横滑り角 [deg]"].append(np.degrees(c["beta"][m]))
        rows["ヨーレート [deg/s]"].append(np.degrees(c["r"][m])); rows["横加速度 [m/s2]"].append(c["ab"][m, 1])
        rows["横加速度 − v·r [m/s2]"].append((c["ab"][:, 1] - c["sp"] * c["r"])[m])
        rows["ヨーレート 0.3〜3 Hz [deg/s]"].append(np.degrees(bp(c["r"], 0.3, 3))[m])
    out = []
    for k, v in rows.items():
        x = np.concatenate(v)
        p = np.percentile(x, [1, 5, 50, 95, 99])
        out.append(dict(量=k, 標本数=len(x), 平均=x.mean(), 標準偏差=x.std(), p1=p[0], p5=p[1], p50=p[2], p95=p[3], p99=p[4]))
    return pd.DataFrame(out)


def spectra(clips, comma_root: Path, n_comma: int) -> tuple[pd.DataFrame, dict]:
    def collect(sigs):
        P = [welch(s - s.mean(), fs=FS, nperseg=256)[1] for s in sigs if len(s) >= 256]
        return np.median(P, axis=0) if P else None, len(P)
    nv = {"NVIDIA ヨーレート [rad/s]": [], "NVIDIA 横加速度 [m/s2]": [], "NVIDIA 横速度 [m/s]": []}
    for c in clips:
        if c["native_hz"] < 50:
            continue
        for a0, b0 in segments(straight_mask(c), 256):
            nv["NVIDIA ヨーレート [rad/s]"].append(c["r"][a0:b0]); nv["NVIDIA 横加速度 [m/s2]"].append(c["ab"][a0:b0, 1])
            nv["NVIDIA 横速度 [m/s]"].append(c["vb"][a0:b0, 1])
    cm = {"comma2k19 ジャイロ (ヨー軸) [rad/s]": [], "comma2k19 加速度 (横軸) [m/s2]": []}
    for p in sorted(glob.glob(str(comma_root / "*/*/processed_log/IMU/gyro/t")))[:n_comma]:
        s = p.rsplit("/processed_log", 1)[0]
        try:
            gt, gv = np.load(s + "/processed_log/IMU/gyro/t"), np.load(s + "/processed_log/IMU/gyro/value")
            at, av = np.load(s + "/processed_log/IMU/accelerometer/t"), np.load(s + "/processed_log/IMU/accelerometer/value")
            pt, pv = np.load(s + "/global_pose/frame_times"), np.load(s + "/global_pose/frame_velocities")
        except OSError:
            continue
        g = np.arange(max(gt[0], at[0], pt[0]), min(gt[-1], at[-1], pt[-1]), 1 / FS)
        if len(g) < 512:
            continue
        G = np.c_[[np.interp(g, gt, gv[:, i]) for i in range(3)]].T
        A = np.c_[[np.interp(g, at, av[:, i]) for i in range(3)]].T
        sp = np.interp(g, pt, np.linalg.norm(pv, axis=1))
        sm = pd.DataFrame(G).rolling(100, center=True, min_periods=1).mean()
        yaw_ax = int(np.argmax(sm.var().to_numpy()))           # 旋回でいちばん動く軸
        grav_ax = int(np.argmax(np.abs(A.mean(axis=0))))
        cand = [i for i in range(3) if i != grav_ax]
        lat_ax = max(cand, key=lambda i: np.corrcoef(pd.Series(A[:, i]).rolling(100, center=True, min_periods=1).mean(),
                                                      sm[yaw_ax] * sp)[0, 1] ** 2)
        st = (sp > HI_SPEED) & (sm[yaw_ax].abs().rolling(STRAIGHT_WIN, center=True, min_periods=1).max().to_numpy() < STRAIGHT_R)
        for a0, b0 in segments(st, 256):
            cm["comma2k19 ジャイロ (ヨー軸) [rad/s]"].append(G[a0:b0, yaw_ax]); cm["comma2k19 加速度 (横軸) [m/s2]"].append(A[a0:b0, lat_ax])
    f = welch(np.zeros(256), fs=FS, nperseg=256)[0]
    rows, curves = [], {}
    for name, sigs in {**nv, **cm}.items():
        P, k = collect(sigs)
        if P is None:
            continue
        curves[name] = P
        row = dict(信号=name, 区間数=k)
        for lo, hi in BANDS:
            row[f"RMS {lo}-{hi} Hz"] = float(np.sqrt(np.sum(P[(f >= lo) & (f < hi)]) * (f[1] - f[0])))
        rows.append(row)
    return pd.DataFrame(rows), dict(f=f, **curves)


def offline_check(clips) -> pd.DataFrame:
    """online (100 Hz) のヨーレートと、offline (10 Hz、別の最適化) のヨーレートの帯域ごとの一致。"""
    zf = zipfile.ZipFile(OFF)
    names = {n.split(".")[0]: n for n in zf.namelist()}
    rows = []
    for c in clips:
        if c["clip"] not in names or c["native_hz"] < 50:
            continue
        d = pd.read_parquet(io.BytesIO(zf.read(names[c["clip"]]))).sort_values("timestamp")
        to = d.timestamp.to_numpy() / 1e6
        qo = R.from_quat(d[["qx", "qy", "qz", "qw"]].to_numpy())
        ro = (qo[:-1].inv() * qo[1:]).as_rotvec()[:, 2] / np.diff(to)
        tm = (to[:-1] + to[1:]) / 2
        # online を offline と同じ 0.1 s 区間で平均する (同じ量にそろえる)
        tt = c["t"] + c["t0"]
        yaw_on = np.interp(to, tt, c["yaw"])
        ron = np.diff(yaw_on) / np.diff(to)
        spd = np.interp(tm, tt, c["sp"])
        m = spd > 8
        if m.sum() < 60:
            continue
        row = dict(clip=c["clip"], n=int(m.sum()))
        for lo, hi in [(0.05, 0.5), (0.5, 2), (2, 4.5)]:
            a = bp(ron, lo, hi, fs=10.0)[m]; b = bp(ro, lo, hi, fs=10.0)[m]
            row[f"相関 {lo}-{hi} Hz"] = float(np.corrcoef(a, b)[0, 1])
            row[f"RMS比 offline/online {lo}-{hi} Hz"] = float(np.std(b) / max(np.std(a), 1e-12))
        rows.append(row)
    return pd.DataFrame(rows)


EV_BAND = (0.3, 2.0)     # 突風への応答と修正操舵が入る帯域。7.5 Hz・14 Hz の振動が漏れないよう 4 次で切る
EV_WIN = 400             # 4 s 窓
EV_CONTEXT = 4.0         # 横移動は窓の前後 4 s を含めて測る (車線変更を見逃さないため)


def bp4(x, lo, hi, fs=FS):
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def lateral_in_frame(c, mid, sl):
    """sl の区間の位置を、mid の時刻の向きを x 軸にした横位置 [m] にする。"""
    ang = c["yaw"][mid]
    p = c["pos"][sl, :2] - c["pos"][mid, :2]
    return -np.sin(ang) * p[:, 0] + np.cos(ang) * p[:, 1], np.cos(ang) * p[:, 0] + np.sin(ang) * p[:, 1]


def events(clips) -> pd.DataFrame:
    """直進に近い高速の区間で、短時間の横方向の動きを拾う。

    指標 (いずれも推定値から作った量):
      r_bp       ヨーレートの 0.3〜2 Hz 成分の |最大| [deg/s]
      flips      r_bp が ±しきい値 (直進区間の 0.3〜2 Hz 成分の標準偏差の 2 倍) を超えて符号が入れ替わった回数
      ydev       窓の中の位置を 2 次式で当てはめた残りの横ずれの幅 [m]
      lat_ctx    窓の前後 4 s を含めた横移動の幅 [m] (車線変更なら 3 m 前後)
      alat       横加速度の 0.3〜2 Hz 成分の |最大| [m/s2]
    """
    usable = [c for c in clips if c["native_hz"] > 50]
    allbp = np.concatenate([np.degrees(bp4(c["r"], *EV_BAND))[straight_mask(c)] for c in usable if straight_mask(c).any()])
    thr = 2 * np.std(allbp)
    rows = []
    for c in usable:
        m = straight_mask(c)
        rb = np.degrees(bp4(c["r"], *EV_BAND)); ab = bp4(c["ab"][:, 1], *EV_BAND)
        n = len(c["t"]); ctx = int(EV_CONTEXT * FS)
        for a0, b0 in segments(m, EV_WIN):
            for s in range(a0, b0 - EV_WIN + 1, 100):     # 1 s ずつずらす
                w = slice(s, s + EV_WIN); mid = s + EV_WIN // 2
                x = rb[w]
                sig = np.sign(np.where(np.abs(x) > thr, x, 0)); sig = sig[sig != 0]
                flips = int(np.sum(sig[1:] != sig[:-1])) if len(sig) > 1 else 0
                lat, lon = lateral_in_frame(c, mid, w)
                ydev = float(np.ptp(lat - np.polyval(np.polyfit(lon, lat, 2), lon)))
                wc = slice(max(s - ctx, 0), min(s + EV_WIN + ctx, n))
                lat_c, _ = lateral_in_frame(c, mid, wc)
                rows.append(dict(clip=c["clip"], t_start=c["t"][s] + c["t0"], t_end=c["t"][s + EV_WIN - 1] + c["t0"],
                                 v=float(np.mean(c["sp"][w])), r_bp=float(np.max(np.abs(x))), flips=flips, ydev=ydev,
                                 lat_ctx=float(np.ptp(lat_c)), ctx_s=float((wc.stop - wc.start) / FS),
                                 alat=float(np.max(np.abs(ab[w]))), beta_ptp=float(np.degrees(np.ptp(c["beta"][w]))),
                                 flip_thr_deg_s=thr))
    e = pd.DataFrame(rows)
    if e.empty:
        return e
    keep = []
    for cid, g in e.sort_values("r_bp", ascending=False).groupby("clip", sort=False):
        taken = []
        for i, r in g.iterrows():
            if all(abs(r.t_start - t) >= 4 for t in taken):
                keep.append(i); taken.append(r.t_start)
    e = e.loc[keep].sort_values("r_bp", ascending=False).reset_index(drop=True)
    e["区分 (目安)"] = np.select([e.lat_ctx > 2.5, e.ydev > 0.5, e.flips >= 2],
                               ["車線変更か道路の曲がり (前後 4 s を含む横移動 > 2.5 m)", "横ずれあり (窓内 0.5 m 超)", "左右の振れ (横ずれは小さい)"],
                               "単発の小さな振れ")
    return e


# ---------------------------------------------------------------------------

def plot_all(out: Path, clips, ev, spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
    (out / "plots").mkdir(parents=True, exist_ok=True)
    C = {"main": "#2F6FB0", "sub": "#E07B39", "muted": "#9CA3AF"}

    # スペクトル: ヨーレートと横加速度 (NVIDIA と comma2k19 の生の IMU)
    f = spec["f"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (nv, cm, yl) in zip(axs, [("NVIDIA ヨーレート [rad/s]", "comma2k19 ジャイロ (ヨー軸) [rad/s]", "ヨーレート"),
                                      ("NVIDIA 横加速度 [m/s2]", "comma2k19 加速度 (横軸) [m/s2]", "横加速度")]):
        for k, col, lab in [(nv, C["main"], "NVIDIA Ego Motion (推定)"), (cm, C["sub"], "comma2k19 生の IMU (参考)")]:
            if k in spec:
                ax.loglog(f[1:], spec[k][1:], color=col, lw=2, label=lab)
        ax.set_title(f"{yl}のパワースペクトル (高速・ほぼ直進、区間の中央値)", fontsize=10)
        ax.set_xlabel("周波数 [Hz]"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out / "plots/spectra.png", dpi=150); plt.close(fig)

    # 時系列: 点数の上位と、代表的な直進区間
    byclip = {c["clip"]: c for c in clips}
    sel = list(ev.head(6).iterrows())
    med = ev.iloc[(ev.r_bp - ev.r_bp.median()).abs().argsort()[:1]]          # 典型例: r_bp が中央値に近い窓
    sel += [(f"typical", r) for _, r in med.iterrows()]
    for i, (lab, e) in enumerate(sel):
        c = byclip[e["clip"]]
        tt = c["t"] + c["t0"]
        w = (tt >= e.t_start - 4) & (tt <= e.t_end + 4)
        t = tt[w]
        fig, axs = plt.subplots(4, 1, figsize=(9, 7.5), sharex=True)
        series = [(np.degrees(c["r"][w]), "ヨーレート [deg/s]"), (c["ab"][w, 1], "横加速度 (車体) [m/s²]"),
                  (np.degrees(c["beta"][w]), "横滑り角 [deg]")]
        for ax, (y, yl) in zip(axs[:3], series):
            ax.plot(t, y, color=C["main"], lw=1)
            ax.axvspan(e.t_start, e.t_end, color="#E8EEF6", zorder=0); ax.set_ylabel(yl, fontsize=8); ax.grid(alpha=0.3)
        # 横ずれ: 窓の中心の向きでの横位置
        mid = int(np.argmin(np.abs(tt - (e.t_start + 2))))
        ang = c["yaw"][mid]; p = c["pos"][w, :2] - c["pos"][mid, :2]
        axs[3].plot(t, -np.sin(ang) * p[:, 0] + np.cos(ang) * p[:, 1], color=C["main"], lw=1)
        axs[3].axvspan(e.t_start, e.t_end, color="#E8EEF6", zorder=0); axs[3].set_ylabel("横位置 [m]", fontsize=8)
        axs[3].grid(alpha=0.3); axs[3].set_xlabel("クリップ開始からの時間 [s]")
        head = "典型例 (中央値に近い窓)" if lab == "typical" else f"上位 {i + 1}"
        fig.suptitle(f"{head}  {e['clip'][:8]}  v={e.v:.1f} m/s  ヨーレート(0.3–2 Hz)最大 {e.r_bp:.2f} deg/s  左右反転 {e.flips} 回  "
                     f"窓内の横ずれ {e.ydev:.2f} m  前後を含む横移動 {e.lat_ctx:.2f} m\n{e['区分 (目安)']}", fontsize=8)
        name = "typical" if lab == "typical" else f"event_{i + 1}"
        fig.tight_layout(); fig.savefig(out / f"plots/{name}_{e['clip'][:8]}.png", dpi=130); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("out/physical_ai_av/egomotion_check"))
    ap.add_argument("--comma-root", type=Path, default=Path("raw_data/Chunk_1"))
    ap.add_argument("--n-comma", type=int, default=120, help="参考に使う comma2k19 のセグメント数")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(EGO)
    clips = [c for c in (load_clip(zf, n) for n in zf.namelist()) if c is not None]
    meta = pd.read_parquet(DATA / "metadata/data_collection.parquet")
    cl = pd.DataFrame([dict(clip=c["clip"], native_hz=round(c["native_hz"], 1), dur_s=round(c["t"][-1], 1),
                            v_median=round(float(np.median(c["sp"])), 2), v_max=round(float(c["sp"].max()), 2),
                            hi_speed_s=round(float((c["sp"] > HI_SPEED).sum() / FS), 1),
                            straight_hi_s=round(float(straight_mask(c).sum() / FS), 1)) for c in clips])
    cl = cl.join(meta, on="clip")
    cl.to_csv(args.out / "clips.csv", index=False)

    st = stats(clips); st.round(5).to_csv(args.out / "stats.csv", index=False)
    rel = pd.DataFrame([x for x in (relations(c) for c in clips) if x]); rel.round(6).to_csv(args.out / "relations.csv", index=False)
    sp_tab, sp_curves = spectra(clips, args.comma_root, args.n_comma); sp_tab.to_csv(args.out / "spectra.csv", index=False)
    off = offline_check(clips); off.round(4).to_csv(args.out / "offline_check.csv", index=False)
    ev = events(clips); ev.round(4).to_csv(args.out / "events.csv", index=False)
    summary = dict(
        n_clips=len(clips), n_100hz=int((cl.native_hz > 50).sum()), hours=round(cl.dur_s.sum() / 3600, 2),
        hi_speed_hours=round(cl.hi_speed_s.sum() / 3600, 3), straight_hi_hours=round(cl.straight_hi_s.sum() / 3600, 3),
        relations_median=rel.drop(columns=["clip"]).median().round(5).to_dict(),
        offline_median=off.drop(columns=["clip"]).median().round(3).to_dict() if not off.empty else {},
        n_event_windows=len(ev), event_classes=ev["区分 (目安)"].value_counts().to_dict() if not ev.empty else {})
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.no_plots:
        plot_all(args.out, clips, ev, sp_curves)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
