"""Civic: 20 Hz のグリッドに揃え、ジャイロの時変バイアスを推定して引いた時系列を作る。

前処理はパイプライン (configs/detection.yaml) に揃える: 20 Hz、車速 0.25 s・舵角 0.15 s の移動平均。
ジャイロ z も 0.15 s の移動平均。ヨーは左回り正 = -(gz - bias)。
バイアス = ドライブ内で連続した時刻の (gz - 方位変化率) の 60 s 移動中央値。
  方位変化率は global_pose の ECEF 速度の方位 (1 Hz 低域の後に微分)。停車中は真値 0 を使う。
出力: <out>/grid_20hz.csv.gz
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
FS = 20.0; BIAS_WIN_S = 60.0; BIAS_MIN_S = 15.0

def load(d):
    t = np.load(d / "t").ravel().astype(float); v = np.load(d / "value").astype(float)
    t, i = np.unique(t, return_index=True); return t, v[i]

def rot_to_down(m):
    a = m / np.linalg.norm(m); b = np.array([0, 0, -1.0]); v = np.cross(a, b); c = a @ b
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)

def ma(x, s):
    n = max(1, int(round(s * FS))); return pd.Series(x).rolling(n, center=True, min_periods=1).mean().to_numpy()

def lp(x, hz):
    return sosfiltfilt(butter(2, hz, fs=FS, output="sos"), x)

def enu_R(p):
    x, y, z = p; lon = np.arctan2(y, x); lat = np.arctan2(z, np.hypot(x, y))
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])

sel = pd.read_csv(ROOT / "selection.csv")
frames = []
for drive, g in sel.groupby("drive"):
    segs = sorted(g.seg); paths = [ROOT / drive / str(s) for s in segs]
    R = rot_to_down(np.vstack([load(p / "processed_log/IMU/accelerometer")[1] for p in paths]).mean(0))
    parts = []
    for p, seg in zip(paths, segs):
        tg, gy = load(p / "processed_log/IMU/gyro"); gz_raw = (gy @ R.T)[:, 2]
        ts, st = load(p / "processed_log/CAN/steering_angle"); tv, v = load(p / "processed_log/CAN/speed")
        ft = np.load(p / "global_pose/frame_times").ravel(); fp = np.load(p / "global_pose/frame_positions")
        fv = np.load(p / "global_pose/frame_velocities")
        okp = np.isfinite(fv).all(1) & np.isfinite(fp).all(1)
        lo = max(tg[0], ts[0], tv[0]) + 0.6; hi = min(tg[-1], ts[-1], tv[-1]) - 0.6
        if hi - lo < 10: continue
        t = np.arange(np.ceil(lo * FS) / FS, hi, 1 / FS)
        df = pd.DataFrame(dict(drive=drive, seg=seg, t=t))
        df["v"] = ma(np.interp(t, tv, v.ravel()), 0.25)
        df["steer"] = ma(np.interp(t, ts, st.ravel()), 0.15)
        # ジャイロは 100 Hz で 0.15 s 平均してから 20 Hz に落とす
        gz100 = pd.Series(gz_raw).rolling(15, center=True, min_periods=1).mean().to_numpy()
        df["gz"] = np.interp(t, tg, gz100)
        if okp.sum() > 100:
            enu = fv[okp] @ enu_R(fp[okp][okp.sum() // 2]).T
            e, n = np.interp(t, ft[okp], enu[:, 0]), np.interp(t, ft[okp], enu[:, 1])
            df["vgp"] = np.hypot(e, n)
            df["hrate"] = np.gradient(lp(np.unwrap(np.arctan2(e, n)), 1.0), 1 / FS)
        else:
            df["vgp"] = np.nan; df["hrate"] = np.nan
        parts.append(df)
    if not parts: continue
    d = pd.concat(parts).sort_values("t").reset_index(drop=True)
    gz_lp = np.concatenate([lp(x.to_numpy(), 1.0) if len(x) > 30 else x.to_numpy()
                            for _, x in d.groupby("seg")["gz"]])
    stop = (d.v < 0.05) & (d.vgp < 0.1)
    move = (d.v > 5) & (d.vgp > 5)
    obs = np.where(stop, d.gz, np.where(move, gz_lp - d.hrate, np.nan))
    # 時刻ベースの移動中央値 (セグメントの欠けは時刻で自然に扱われる)
    s = pd.Series(obs, index=pd.to_datetime(d.t, unit="s"))
    med = s.rolling(f"{int(BIAS_WIN_S)}s", center=True, min_periods=int(BIAS_MIN_S * FS)).median()
    cnt = s.rolling(f"{int(BIAS_WIN_S)}s", center=True).count()
    b = med.to_numpy()
    d["bias_n"] = cnt.to_numpy()
    # 推定できない時刻は、同じドライブの前後の推定値から時間で補間 (外挿は端の値)
    good = np.isfinite(b)
    d["bias_interp"] = ~good
    if good.sum() > 0:
        b = np.interp(d.t, d.t[good], b[good])
    d["bias"] = b
    d["stop"] = stop; d["bias_obs"] = obs
    frames.append(d)
out = pd.concat(frames)
out["yaw_dps"] = -np.degrees(out.gz - out.bias)          # 左回り正
out["yaw_raw_dps"] = -np.degrees(out.gz)                 # バイアス補正なし (比較用)
out["ay_kin"] = out.v * np.radians(out.yaw_dps)
out["lat_jerk"] = np.concatenate([ma(np.gradient(x.to_numpy(), 1 / FS), 0.25) for _, x in out.groupby(["drive", "seg"])["ay_kin"]])
out = out.merge(sel[["drive", "seg", "reason", "W", "G"]], on=["drive", "seg"])
out.to_csv(OUT / "grid_20hz.csv.gz", index=False, float_format="%.12g")
print(len(out), "rows", out.drive.nunique(), "drives")
