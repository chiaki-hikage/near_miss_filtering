"""Civic の信号の確認 (符号・目盛り・ゼロ点)。結果は out/crosswind/civic_signal_check/ に書く。

基準: global_pose の ECEF 速度から求めた進行方向 (東西南北) とその変化率。
IMU は既存の振動解析と同じく、ドライブごとに平均の比力が真下を向くよう傾きだけを戻す。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
FS = 20.0

def load(d):
    t = np.load(d / "t").ravel().astype(float); v = np.load(d / "value").astype(float)
    t, i = np.unique(t, return_index=True); return t, v[i]

def rot_to_down(m):
    a = m / np.linalg.norm(m); b = np.array([0, 0, -1.0]); v = np.cross(a, b); c = a @ b
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)

def lp(x, hz):
    return sosfiltfilt(butter(2, hz, fs=FS, output="sos"), x)

def ecef_to_enu_R(p):
    x, y, z = p; lon = np.arctan2(y, x); lat = np.arctan2(z, np.hypot(x, y))
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])

sel = pd.read_csv(ROOT / "selection.csv")
rows, samples = [], []
for drive, g in sel.groupby("drive"):
    paths = [ROOT / drive / str(s) for s in sorted(g.seg)]
    acc = np.vstack([load(p / "processed_log/IMU/accelerometer")[1] for p in paths])
    m = acc.mean(0); R = rot_to_down(m)
    tilt = np.degrees(np.arccos(-m[2] / np.linalg.norm(m)))
    for p, seg in zip(paths, sorted(g.seg)):
        tg, gy = load(p / "processed_log/IMU/gyro"); gy = gy @ R.T
        ts, st = load(p / "processed_log/CAN/steering_angle")
        tv, v = load(p / "processed_log/CAN/speed"); v = v.ravel()
        tw, w = load(p / "processed_log/CAN/wheel_speed")
        ft = np.load(p / "global_pose/frame_times").ravel(); fp = np.load(p / "global_pose/frame_positions")
        fv = np.load(p / "global_pose/frame_velocities")
        enu = fv @ ecef_to_enu_R(fp[len(fp) // 2]).T
        lo, hi = max(tg[0], ts[0], tv[0], ft[0], tw[0]) + 1, min(tg[-1], ts[-1], tv[-1], ft[-1], tw[-1]) - 1
        if hi - lo < 20: continue
        t = np.arange(lo, hi, 1 / FS)
        gz = np.interp(t, tg, gy[:, 2]); vc = np.interp(t, tv, v); sa = np.interp(t, ts, st.ravel())
        wmean = np.interp(t, tw, w.mean(1))
        e, n = np.interp(t, ft, enu[:, 0]), np.interp(t, ft, enu[:, 1])
        vgp = np.hypot(e, n)
        hdg = np.unwrap(np.arctan2(e, n))                      # 方位角 (北から時計回り)
        hrate = np.gradient(lp(hdg, 1.0), 1 / FS)              # rad/s, 時計回り (右旋回) が正
        gzl = lp(gz, 1.0)
        mv = (vgp > 5) & (vc > 5)
        stop = (vc < 0.05) & (vgp < 0.1)
        rows.append(dict(drive=drive, seg=seg, tilt_deg=tilt, n_move=int(mv.sum()),
                         gz_vs_hrate_slope=np.polyfit(hrate[mv], gzl[mv], 1)[0] if mv.sum() > 100 else np.nan,
                         gz_hrate_corr=np.corrcoef(hrate[mv], gzl[mv])[0, 1] if mv.sum() > 100 else np.nan,
                         gz_stop_mean=np.degrees(gz[stop].mean()) if stop.sum() > 40 else np.nan,
                         gz_minus_hrate_med=np.degrees(np.median(gzl[mv] - hrate[mv])) if mv.sum() > 100 else np.nan,
                         v_ratio=np.median(vc[mv] / vgp[mv]) if mv.sum() > 100 else np.nan,
                         w_ratio=np.median(wmean[mv] / vc[mv]) if mv.sum() > 100 else np.nan))
        k = slice(None, None, 4)                               # 5 Hz に間引いて保存
        samples.append(pd.DataFrame(dict(drive=drive, seg=seg, t=t[k], v=vc[k], vgp=vgp[k], steer=sa[k],
                                         steer_lp=lp(sa, 1.0)[k], gz=gzl[k], hrate=hrate[k])))
seg = pd.DataFrame(rows); seg.to_csv(OUT / "per_segment.csv", index=False)
smp = pd.concat(samples); smp.to_csv(OUT / "samples_5hz.csv.gz", index=False)
print(len(seg), "segments")
