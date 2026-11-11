"""Chunk_3〜10 の各セグメントの時刻 (GPS 時刻) と位置 (ECEF) だけを読み、先頭・中央・末尾を記録する。"""
import io, sys, threading, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from remote_zip import open_chunk

def llh(x, y, z):
    a, e2 = 6378137.0, 6.69437999014e-3
    lon = np.arctan2(y, x); p = np.hypot(x, y); lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2); lat = np.arctan2(z + e2 * n * np.sin(lat), p)
    return np.degrees(lat), np.degrees(lon)

tl = threading.local()
def zipf(n):
    if getattr(tl, 'n', None) != n:
        tl.f, tl.z = open_chunk(n); tl.n = n
    return tl.z

def one(n, seg_dir):
    z = zipf(n)
    g = np.load(io.BytesIO(z.read(seg_dir + '/global_pose/frame_gps_times')))
    p = np.load(io.BytesIO(z.read(seg_dir + '/global_pose/frame_positions')))
    unix = 315964800.0 + g[:, 0] * 604800 + g[:, 1] - 18.0     # GPS → UTC (2018 年のうるう秒 18 s)
    i = [0, len(p) // 2, len(p) - 1]
    lat, lon = llh(p[i, 0], p[i, 1], p[i, 2])
    _, drive, seg = seg_dir.split('/')
    return dict(chunk=n, drive=drive, seg=int(seg), t0=unix[0], t1=unix[-1], lat0=lat[0], lon0=lon[0], lat=lat[1], lon=lon[1], lat1=lat[2], lon1=lon[2])

out = []
for n in map(int, sys.argv[1:]):
    _, z = open_chunk(n)
    segs = sorted({nm.rsplit('/global_pose/', 1)[0] for nm in z.namelist() if nm.endswith('global_pose/frame_gps_times')})
    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(lambda s: one(n, s), segs))
    pd.DataFrame(rows).to_csv(f'remote_chunk_{n}.csv', index=False)
    print(n, len(rows), 'segments', flush=True)
