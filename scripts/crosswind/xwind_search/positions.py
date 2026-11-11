"""全セグメントの 1 Hz の位置・方位・UTC を集める (RAV4 = Chunk_1/2, Civic = data/comma2k19_civic)。
出力: <out>/positions_1hz.csv.gz"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(sys.argv[1]); roots = [Path(p) for p in sys.argv[2:]]

def llh(x, y, z):
    a, e2 = 6378137.0, 6.69437999014e-3
    lon = np.arctan2(y, x); p = np.hypot(x, y); lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2); lat = np.arctan2(z + e2 * n * np.sin(lat), p)
    return np.degrees(lat), np.degrees(lon)

def enu_R(lat, lon):
    la, lo = np.radians(lat), np.radians(lon); sl, cl, so, co = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    return np.array([[-so, co, 0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])

rows = []
for root in roots:
    for gp in sorted(root.glob("*/*/global_pose")):
        seg = gp.parent; drive = seg.parent.name
        try:
            ft = np.load(gp / "frame_times").ravel(); gt = np.load(gp / "frame_gps_times")
            fp = np.load(gp / "frame_positions"); fv = np.load(gp / "frame_velocities")
        except Exception as e:
            print("skip", seg, e); continue
        ok = np.isfinite(fp).all(1) & np.isfinite(fv).all(1) & np.isfinite(gt).all(1)
        if ok.sum() < 20: continue
        ft, gt, fp, fv = ft[ok], gt[ok], fp[ok], fv[ok]
        utc = 315964800 + gt[:, 0] * 604800 + gt[:, 1] - 18
        lat, lon = llh(*fp.T); enu = fv @ enu_R(np.median(lat), np.median(lon)).T
        sel = np.arange(0, len(ft), 20)
        rows.append(pd.DataFrame(dict(car="rav4" if drive.startswith("b0c9") else "civic", drive=drive, seg=int(seg.name),
            t=ft[sel], utc=utc[sel], lat=lat[sel], lon=lon[sel], v=np.hypot(enu[sel, 0], enu[sel, 1]),
            H=np.degrees(np.arctan2(enu[sel, 0], enu[sel, 1])) % 360)))
d = pd.concat(rows); OUT.mkdir(parents=True, exist_ok=True)
d.to_csv(OUT / "positions_1hz.csv.gz", index=False, float_format="%.12g")
print(d.groupby("car").agg(n=("t", "size"), drives=("drive", "nunique"), segs=("seg", "size")))
print(d.groupby("car")[["lat", "lon"]].describe().T)
