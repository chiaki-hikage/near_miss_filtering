"""横滑り角 β = 進路 − カメラ前方軸の方位 (水平面, 左正, deg) を 20 Hz のグリッドに付ける。

frame_orientations は Hamilton [w,x,y,z]。R(q) はカメラ [前,右,下] → ECEF (速度との一致で確認済み)。
出力: <out>/grid_beta.csv.gz (drive, seg, t, v, yaw_dps, steer, ay_kin, reason, beta)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
GRID, ROOT, OUT = map(Path, sys.argv[1:4])

def enu_R(p):
    x, y, z = p; lon = np.arctan2(y, x); lat = np.arctan2(z, np.hypot(x, y))
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])

d = pd.read_csv(GRID, usecols=["drive", "seg", "t", "v", "yaw_dps", "steer", "ay_kin", "reason", "W", "G"])
parts = []
for (dr, sg), x in d.groupby(["drive", "seg"]):
    p = ROOT / dr / str(sg) / "global_pose"
    ft = np.load(p / "frame_times").ravel(); q = np.load(p / "frame_orientations")
    fp = np.load(p / "frame_positions"); fv = np.load(p / "frame_velocities")
    ok = np.isfinite(q).all(1) & np.isfinite(fv).all(1) & np.isfinite(fp).all(1) & (np.linalg.norm(q, axis=1) > 0.5)
    if ok.sum() < 100:
        x = x.assign(beta=np.nan); parts.append(x); continue
    E = enu_R(fp[ok][ok.sum() // 2])
    fwd = Rot.from_quat(q[ok][:, [1, 2, 3, 0]]).apply([1.0, 0, 0]) @ E.T
    vel = fv[ok] @ E.T
    psi_body = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 1]))      # 方位 (北から時計回り)
    psi_vel = np.unwrap(np.arctan2(vel[:, 0], vel[:, 1]))
    b = -np.degrees(np.angle(np.exp(1j * (psi_vel - psi_body))))  # 左正: 進路が車体より左 → 正
    sp = np.linalg.norm(vel[:, :2], axis=1)
    b[sp < 3] = np.nan
    x = x.assign(beta=np.interp(x.t, ft[ok], b, left=np.nan, right=np.nan))
    parts.append(x)
out = pd.concat(parts); out.to_csv(OUT / "grid_beta.csv.gz", index=False, float_format="%.12g")
print(len(out), "rows; beta finite", np.isfinite(out.beta).mean().round(3))
