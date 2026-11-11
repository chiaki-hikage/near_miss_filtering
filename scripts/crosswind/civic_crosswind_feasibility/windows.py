"""手順 5: 直進・定常の 5 s 窓ごとに、ヨー残差 (横加速度換算) と風の横成分を並べる。

入力: out/crosswind/civic_bicycle_fit/grid_20hz.csv.gz, configs/crosswind/honda_civic.yaml,
      data/comma2k19_civic (global_pose), SFO の ASOS 1 分値と METAR。
出力: <out>/windows.csv
風の横成分 W_left = −W·sin(D − H)  (D: 風が吹いてくる方位, H: 進行方位)。左から吹くと正。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, yaml

GRID, CIVIC, ROOT, ASOS, METAR, OUT = map(Path, sys.argv[1:7]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path.cwd() / "src"))
from near_miss.features import bicycle_yaw_rate
KT = 0.514444; WIN_S = 5.0; FS = 20.0

def llh(x, y, z):
    a, e2 = 6378137.0, 6.69437999014e-3
    lon = np.arctan2(y, x); p = np.hypot(x, y); lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2); lat = np.arctan2(z + e2 * n * np.sin(lat), p)
    return np.degrees(lat), np.degrees(lon)

def enu_R(lat, lon):
    la, lo = np.radians(lat), np.radians(lon); sl, cl, so, co = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    return np.array([[-so, co, 0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])

g = yaml.safe_load(CIVIC.read_text())["geometry"]
d = pd.read_csv(GRID)
d["pred"] = bicycle_yaw_rate(d.v.to_numpy(), d.steer.to_numpy(), g["wheelbase_m"], g["steer_ratio"],
                             g["understeer_gradient"], g["steer_offset_deg"])
d["res"] = d.yaw_dps - d.pred
d["ay_res"] = d.v * np.radians(d.res)

# セグメントごとの位置・方位・UTC
pos = []
for (dr, sg), _ in d.groupby(["drive", "seg"]):
    p = ROOT / dr / str(sg) / "global_pose"
    ft = np.load(p / "frame_times").ravel(); gt = np.load(p / "frame_gps_times")
    fp = np.load(p / "frame_positions"); fv = np.load(p / "frame_velocities")
    ok = np.isfinite(fp).all(1) & np.isfinite(fv).all(1) & np.isfinite(gt).all(1)
    if ok.sum() < 100: continue
    ft, gt, fp, fv = ft[ok], gt[ok], fp[ok], fv[ok]
    utc = 315964800 + gt[:, 0] * 604800 + gt[:, 1] - 18
    lat, lon = llh(*fp.T); enu = fv @ enu_R(np.median(lat), np.median(lon)).T
    pos.append(pd.DataFrame(dict(drive=dr, seg=sg, ft=ft, utc_off=np.median(utc - ft), lat=lat, lon=lon,
                                 ve=enu[:, 0], vn=enu[:, 1])))
pos = pd.concat(pos)

rows = []
for (dr, sg), x in d.groupby(["drive", "seg"]):
    pp = pos[(pos.drive == dr) & (pos.seg == sg)]
    if pp.empty: continue
    k = ((x.t - x.t.min()) // WIN_S).astype(int)
    for _, w in x.groupby(k):
        if len(w) < WIN_S * FS * 0.9: continue
        # 定常: 1 s 平均の横加速度 (v·r) の窓内の幅が 0.5 m/s² 未満。
        # ジャイロ由来の横躍度は雑音が大きく、標本ごとの |横躍度| < 0.5 では大半が落ちるため
        ay1 = w.ay_kin.rolling(int(FS), center=True, min_periods=1).mean()
        ok = (w.v > 15).all() and (abs(w.yaw_dps.median()) < 0.5) and (ay1.max() - ay1.min() < 0.5) \
             and np.isfinite(w.res).all()
        if not ok: continue
        tm = w.t.mean()
        ve, vn = np.interp(w.t, pp.ft, pp.ve), np.interp(w.t, pp.ft, pp.vn)
        H = np.degrees(np.arctan2(ve.mean(), vn.mean())) % 360
        rows.append(dict(drive=dr, seg=sg, t=tm, utc=tm + pp.utc_off.iloc[0], reason=w.reason.iloc[0],
                         lat=np.interp(tm, pp.ft, pp.lat), lon=np.interp(tm, pp.ft, pp.lon), H=H,
                         v=w.v.mean(), steer=w.steer.mean(), yaw=w.yaw_dps.mean(), res=w.res.mean(),
                         ay_res=w.ay_res.mean(), bias_interp=w.bias_interp.mean(),
                         ay_res_std=w.ay_res.std(), steer_std=w.steer.std(), yaw_std=w.yaw_dps.std()))
win = pd.DataFrame(rows)

# 風: SFO の 1 分値。欠けていれば SFO の METAR (前後 60 分以内で最も近いもの)
a = pd.read_csv(ASOS); a["ts"] = pd.to_datetime(a["valid(UTC)"]).astype("int64") // 10**9
for c in ("sknt", "drct", "gust_sknt"): a[c] = pd.to_numeric(a[c], errors="coerce")
a = a.dropna(subset=["sknt", "drct"]).drop_duplicates("ts").set_index("ts")
m = pd.read_csv(METAR); m = m[m.station == "SFO"].copy(); m["ts"] = pd.to_datetime(m.valid).astype("int64") // 10**9
for c in ("sknt", "drct", "gust"): m[c] = pd.to_numeric(m[c], errors="coerce")
m = m.dropna(subset=["sknt", "drct"]).sort_values("ts")
minute = (win.utc // 60 * 60).astype(int)
hit = minute.isin(a.index)
win["W"] = np.where(hit, a.reindex(minute).sknt.to_numpy() * KT, np.nan)
win["D"] = np.where(hit, a.reindex(minute).drct.to_numpy(), np.nan)
win["Gst"] = np.where(hit, a.reindex(minute).gust_sknt.to_numpy() * KT, np.nan)
win["wind_src"] = np.where(hit, "asos1min", "")
j = np.searchsorted(m.ts.to_numpy(), win.utc.to_numpy()); j = np.clip(j, 1, len(m) - 1)
jj = np.where(np.abs(m.ts.to_numpy()[j] - win.utc) < np.abs(m.ts.to_numpy()[j - 1] - win.utc), j, j - 1)
near = np.abs(m.ts.to_numpy()[jj] - win.utc) <= 3600
fill = ~hit & near
win.loc[fill, "W"] = m.sknt.to_numpy()[jj][fill] * KT; win.loc[fill, "D"] = m.drct.to_numpy()[jj][fill]
win.loc[fill, "Gst"] = m.gust.to_numpy()[jj][fill] * KT; win.loc[fill, "wind_src"] = "metar"
win["W_left"] = -win.W * np.sin(np.radians(win.D - win.H))
win["W_head"] = win.W * np.cos(np.radians(win.D - win.H))
win.to_csv(OUT / "windows.csv", index=False, float_format="%.6g")
print(len(win), "windows;", win.wind_src.value_counts().to_dict())
