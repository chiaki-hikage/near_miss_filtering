"""両車種で同じ形の 20 Hz 時系列を作る。
  rav4 : 車速・舵角 = processed_log/CAN、ヨーレート・横加速度 = raw_can 0x024 (configs/vehicles/toyota_rav4.yaml で検証済み)
  civic: civic_v2/beta/grid_beta.csv.gz (ヨー = バイアス補正済みジャイロ, β) に、IMU 加速度を水平に直した横加速度を足す
共通の列: car, drive, seg, t, utc, v, steer, yaw, ay_meas, beta_raw, lat, lon, H
  yaw: 左回り正 deg/s。ay_meas: 左正 m/s²。beta_raw: 進路 − 車体 (カメラ) 方位、左正 deg、取付のずれは未補正
使い方: grid_common.py <car> <out.csv.gz> <roots...>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy.spatial.transform import Rotation as Rot
from concurrent.futures import ProcessPoolExecutor
REPO = Path.cwd(); sys.path.insert(0, str(REPO / "src"))
FS = 20.0

def ma(x, s):
    n = max(1, int(round(s * FS))); return pd.Series(x).rolling(n, center=True, min_periods=1).mean().to_numpy()

def ecef_llh(x, y, z):
    a, e2 = 6378137.0, 6.69437999014e-3
    lon = np.arctan2(y, x); p = np.hypot(x, y); lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2); lat = np.arctan2(z + e2 * n * np.sin(lat), p)
    return np.degrees(lat), np.degrees(lon)

def enu_R(p):
    x, y, z = p; lon = np.arctan2(y, x); lat = np.arctan2(z, np.hypot(x, y))
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])

def pose(seg_dir: Path, t: np.ndarray) -> pd.DataFrame:
    p = seg_dir / "global_pose"
    ft = np.load(p / "frame_times").ravel(); q = np.load(p / "frame_orientations")
    fp = np.load(p / "frame_positions"); fv = np.load(p / "frame_velocities"); gt = np.load(p / "frame_gps_times")
    ok = np.isfinite(q).all(1) & np.isfinite(fv).all(1) & np.isfinite(fp).all(1) & np.isfinite(gt).all(1) \
        & (np.linalg.norm(q, axis=1) > 0.5)
    out = pd.DataFrame(index=range(len(t)))
    if ok.sum() < 100:
        for c in ("utc", "lat", "lon", "H", "beta_raw", "vgp"): out[c] = np.nan
        return out
    ft, q, fp, fv, gt = ft[ok], q[ok], fp[ok], fv[ok], gt[ok]
    E = enu_R(fp[len(fp) // 2])
    fwd = Rot.from_quat(q[:, [1, 2, 3, 0]]).apply([1.0, 0, 0]) @ E.T
    vel = fv @ E.T
    psi_b = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 1])); psi_v = np.unwrap(np.arctan2(vel[:, 0], vel[:, 1]))
    b = -np.degrees(np.angle(np.exp(1j * (psi_v - psi_b)))); sp = np.hypot(vel[:, 0], vel[:, 1]); b[sp < 3] = np.nan
    lat, lon = ecef_llh(*fp.T)
    utc = 315964800 + gt[:, 0] * 604800 + gt[:, 1] - 18
    f = lambda y: np.interp(t, ft, y, left=np.nan, right=np.nan)
    out["utc"] = t + np.median(utc - ft)
    out["lat"], out["lon"] = f(lat), f(lon)
    out["H"] = np.degrees(f(np.unwrap(np.arctan2(vel[:, 0], vel[:, 1])))) % 360
    out["beta_raw"] = f(b); out["vgp"] = f(sp)
    return out

def rav4_seg(args):
    path, drive, seg = args
    from near_miss.io.comma2k19 import load_segment, SegmentRef
    from near_miss.config import VehicleConfig
    veh = VehicleConfig.from_dict(yaml.safe_load(open(REPO / "configs/vehicles/toyota_rav4.yaml")))
    ref = SegmentRef(path=Path(path), dongle_id=drive.split("|")[0], drive_id=drive, index=seg, dataset="comma2k19", platform="")
    s = load_segment(ref, veh, with_raw_can=True); ch = s.channels
    need = ("speed_mps", "steer_deg", "yaw_rate", "accel_y")
    if not all(k in ch for k in need): return None
    lo = max(ch[k].t[0] for k in need) + 0.6; hi = min(ch[k].t[-1] for k in need) - 0.6
    if hi - lo < 10: return None
    t = np.arange(np.ceil(lo * FS) / FS, hi, 1 / FS)
    g = lambda k: np.interp(t, *np.unique(np.asarray(ch[k].t, float), return_index=True)[0:1],
                            np.asarray(ch[k].v, float)[np.unique(np.asarray(ch[k].t, float), return_index=True)[1]])
    df = pd.DataFrame(dict(car="rav4", drive=drive, seg=seg, t=t))
    df["v"] = ma(g("speed_mps"), 0.25); df["steer"] = ma(g("steer_deg"), 0.15)
    df["yaw"] = ma(g("yaw_rate"), 0.15); df["ay_meas"] = ma(g("accel_y"), 0.15)
    return pd.concat([df, pose(Path(path), t)], axis=1)

def civic_ay(args):
    """IMU 加速度を重力方向で水平に直し、横成分を返す (向きと符号は後でドライブごとに v·r と合わせる)。"""
    path, t = args
    p = Path(path) / "processed_log/IMU/accelerometer"
    ta = np.load(p / "t").ravel().astype(float); a = np.load(p / "value").astype(float)
    ta, i = np.unique(ta, return_index=True); a = a[i]
    return ta, a, t

if __name__ == "__main__":
    car, out = sys.argv[1], Path(sys.argv[2]); roots = [Path(r) for r in sys.argv[3:]]
    if car == "rav4":
        jobs = [(str(s), s.parent.name, int(s.name)) for r in roots for s in sorted(r.glob("*/*")) if (s / "global_pose").exists()]
        with ProcessPoolExecutor(8) as ex: parts = [p for p in ex.map(rav4_seg, jobs, chunksize=4) if p is not None]
        d = pd.concat(parts, ignore_index=True)
    else:
        root = roots[0]
        g = pd.read_csv(REPO / "out/crosswind/civic_v2/beta/grid_beta.csv.gz")
        g = g.rename(columns={"yaw_dps": "yaw", "beta": "beta_raw"}).drop(columns=["ay_kin", "W", "G", "reason"])
        g.insert(0, "car", "civic")
        parts = []
        for drive, x in g.groupby("drive"):
            segs = sorted(x.seg.unique())
            acc = {s: civic_ay((root / drive / str(s), None)) for s in segs}
            m = np.vstack([acc[s][1] for s in segs]).mean(0)
            a_ = m / np.linalg.norm(m); b_ = np.array([0, 0, -1.0]); vv = np.cross(a_, b_); c = a_ @ b_
            vx = np.array([[0, -vv[2], vv[1]], [vv[2], 0, -vv[0]], [-vv[1], vv[0], 0]]); R = np.eye(3) + vx + vx @ vx / (1 + c)
            for s in segs:
                y = x[x.seg == s].copy(); ta, a, _ = acc[s]
                ah = a @ R.T
                a100 = pd.DataFrame(ah[:, :2]).rolling(15, center=True, min_periods=1).mean().to_numpy()
                y["ax_h"] = np.interp(y.t, ta, a100[:, 0]); y["ay_h"] = np.interp(y.t, ta, a100[:, 1])
                pz = pose(root / drive / str(s), y.t.to_numpy()); pz.index = y.index
                y = pd.concat([y.drop(columns=["beta_raw"]), pz], axis=1)
                parts.append(y)
        d = pd.concat(parts, ignore_index=True)
        # 水平加速度 (ax_h, ay_h) を回して、縦 = dv/dt、横 = v·r に合う向きと符号をドライブごとに決める
        d["dvdt"] = np.concatenate([np.gradient(z.to_numpy(), 1 / FS) for _, z in d.groupby(["drive", "seg"])["v"]])
        d["ay_kin"] = d.v * np.radians(d.yaw); d["ay_meas"] = np.nan; d["imu_rot_deg"] = np.nan
        for drive, x in d.groupby("drive"):
            mv = x[(x.v > 5)]
            A = np.c_[mv.ax_h, mv.ay_h]; Y = np.c_[mv.dvdt, mv.ay_kin]
            M, *_ = np.linalg.lstsq(A, Y, rcond=None)         # 2x2: 回転 (+ 反転) のはず
            U, _, Vt = np.linalg.svd(M); Rm = U @ Vt              # 最も近い直交行列
            lat_ = np.c_[x.ax_h, x.ay_h] @ Rm
            d.loc[x.index, "ay_meas"] = lat_[:, 1]
            d.loc[x.index, "imu_rot_deg"] = np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0])) + (1000 if np.linalg.det(Rm) < 0 else 0)
        d = d.drop(columns=["ax_h", "ay_h", "dvdt", "ay_kin"])
    d.to_csv(out, index=False, float_format="%.12g")
    mv = d[d.v > 10]
    print(len(d), "rows", d.drive.nunique(), "drives; corr(ay_meas, v·r) =",
          np.corrcoef(mv.ay_meas, mv.v * np.radians(mv.yaw))[0, 1].round(3),
          "beta finite", np.isfinite(d.beta_raw).mean().round(3))
    if "imu_rot_deg" in d: print(d.groupby("drive").imu_rot_deg.first().describe())
