"""1 Hz の位置に SFO の風 (ASOS 1 分値、欠けたら前後 60 分以内の METAR) と、OSM の道路構造を付ける。
出力: <out>/positions_wind.csv.gz, <out>/drive_wind.csv"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
OUT, ASOS, METAR, OSM = map(Path, sys.argv[1:5]); KT = 0.514444
d = pd.read_csv(OUT / "positions_1hz.csv.gz")
d = d[np.isfinite(d.lat) & np.isfinite(d.v)].copy()
a = pd.read_csv(ASOS); a["ts"] = pd.to_datetime(a["valid(UTC)"]).astype("int64") // 10**9
for c in ("sknt", "drct", "gust_sknt"): a[c] = pd.to_numeric(a[c], errors="coerce")
a = a.dropna(subset=["sknt", "drct"]).drop_duplicates("ts").set_index("ts")
m = pd.read_csv(METAR); m = m[m.station == "SFO"].copy(); m["ts"] = pd.to_datetime(m.valid).astype("int64") // 10**9
for c in ("sknt", "drct", "gust"): m[c] = pd.to_numeric(m[c], errors="coerce")
m = m.dropna(subset=["sknt", "drct"]).sort_values("ts")
minute = (d.utc // 60 * 60).astype(int); hit = minute.isin(a.index).to_numpy()
r = a.reindex(minute)
d["W"] = np.where(hit, r.sknt.to_numpy() * KT, np.nan); d["D"] = np.where(hit, r.drct.to_numpy(), np.nan)
d["G"] = np.where(hit, r.gust_sknt.to_numpy() * KT, np.nan); d["wsrc"] = np.where(hit, "asos1min", "")
mt = m.ts.to_numpy(); j = np.clip(np.searchsorted(mt, d.utc.to_numpy()), 1, len(m) - 1)
jj = np.where(np.abs(mt[j] - d.utc) < np.abs(mt[j - 1] - d.utc), j, j - 1)
fill = ~hit & (np.abs(mt[jj] - d.utc.to_numpy()) <= 3600)
d.loc[fill, "W"] = m.sknt.to_numpy()[jj][fill] * KT; d.loc[fill, "D"] = m.drct.to_numpy()[jj][fill]
d.loc[fill, "G"] = m.gust.to_numpy()[jj][fill] * KT; d.loc[fill, "wsrc"] = "metar"
d["W_left"] = -d.W * np.sin(np.radians(d.D - d.H))
# OSM: 道路の点 (5 m 間隔に補間) に最も近いもの。進行方位と 35 度以内の向きの区間だけ
lat0 = 37.65; kx, ky = 111320 * np.cos(np.radians(lat0)), 110540
pts, info = [], []
for e in json.load(open(OSM))["elements"]:
    g = e.get("geometry");  tg = e["tags"]
    if not g or len(g) < 2: continue
    x = np.array([p["lon"] for p in g]) * kx; y = np.array([p["lat"] for p in g]) * ky
    for i in range(len(x) - 1):
        L = np.hypot(x[i+1]-x[i], y[i+1]-y[i]); n = max(int(L // 5), 1)
        brg = np.degrees(np.arctan2(x[i+1]-x[i], y[i+1]-y[i])) % 360
        for f in np.arange(n) / n:
            pts.append((x[i] + f*(x[i+1]-x[i]), y[i] + f*(y[i+1]-y[i])))
            info.append((e["id"], tg.get("highway"), tg.get("ref", ""), tg.get("name", ""), tg.get("bridge", ""),
                         tg.get("tunnel", ""), tg.get("bridge:name", ""), tg.get("layer", ""), brg, tg.get("oneway", "")))
pts = np.array(pts); info = pd.DataFrame(info, columns=["way", "hw", "ref", "name", "bridge", "tunnel", "bridge_name", "layer", "brg", "oneway"])
tree = cKDTree(pts); q = np.c_[d.lon * kx, d.lat * ky]
dist, idx = tree.query(q, k=12, distance_upper_bound=40)
best = np.full(len(d), -1); bd = np.full(len(d), np.nan)
H = d.H.to_numpy()
for k in range(12):
    ok = np.isfinite(dist[:, k]) & (best < 0)
    ii = np.where(ok, idx[:, k], 0)
    db = np.abs((info.brg.to_numpy()[ii] - H + 180) % 360 - 180)
    db = np.minimum(db, np.where(info.oneway.to_numpy()[ii] == "yes", 999, 180 - db))
    good = ok & (db < 35) & (d.v.to_numpy() > 3)
    best[good] = ii[good]; bd[good] = dist[good, k]
for c in ("way", "hw", "ref", "name", "bridge", "tunnel", "bridge_name"):
    d["osm_" + c] = np.where(best >= 0, info[c].to_numpy()[np.maximum(best, 0)], None)
d["osm_dist"] = bd
d.to_csv(OUT / "positions_wind.csv.gz", index=False, float_format="%.12g")
mv = d[d.v > 15]
g = mv.groupby(["car", "drive"]).agg(min_moving=("t", lambda s: len(s) / 60), W_med=("W", "median"), W_max=("W", "max"),
     G_max=("G", "max"), D_med=("D", "median"), wsrc=("wsrc", lambda s: (s == "asos1min").mean()),
     bridge_s=("osm_bridge", lambda s: (s == "yes").sum())).reset_index()
g.to_csv(OUT / "drive_wind.csv", index=False, float_format="%.4g")
print(g.sort_values("W_med", ascending=False).head(30).to_string())
print("moving >15 m/s hours by W bin:"); print(mv.groupby(["car", pd.cut(mv.W, [0, 5, 8, 10, 12, 20])]).size().unstack(0) / 3600)
print("matched frac", (d.osm_way.notna() & (d.v > 15)).sum() / (d.v > 15).sum())
print(mv[mv.osm_bridge == "yes"].groupby(["osm_ref", "osm_name", "osm_bridge_name"]).size().sort_values(ascending=False).head(30))
