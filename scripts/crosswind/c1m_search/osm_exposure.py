"""高速走行のセグメント (速度 > 18 m/s, 弦長 > 900 m) について、OSM から近くの橋・トンネル・海岸線を取る。
セグメントの始点・中点・終点を結んだ折れ線の周り (±400 m の箱) を 100 件ずつ 1 回の Overpass 問い合わせにまとめる。
出力: <out>/osm_features.json (要素の生データ), <out>/seg_exposure.csv (セグメントごとの最寄り距離)"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, requests
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
p = pd.read_csv("out/comma1m/regions.csv"); p = p[p.error.isna()]
hs = p[(p.speed_mid > 18) & (p.chord_m > 900)].reset_index(drop=True)
cache = OUT / "osm_features.json"
els = json.load(open(cache)) if cache.exists() else {}
def bbox(r, pad=0.004):
    la = [r.lat_start, r.lat_mid, r.lat_end]; lo = [r.lon_start, r.lon_mid, r.lon_end]
    return f"{min(la)-pad:.5f},{min(lo)-pad:.5f},{max(la)+pad:.5f},{max(lo)+pad:.5f}"
done = set(json.load(open(OUT / "osm_done.json"))) if (OUT / "osm_done.json").exists() else set()
todo = hs[~hs.segment_id.isin(done)]
for i in range(0, len(todo), 100):
    ch = todo.iloc[i:i + 100]
    parts = []
    for _, r in ch.iterrows():
        b = bbox(r)
        parts += [f'way["bridge"]["bridge"!="no"]["highway"~"^(motorway|trunk|primary|secondary)$"]({b});',
                  f'way["tunnel"]["tunnel"!="no"]["highway"~"^(motorway|trunk|primary|secondary)$"]({b});',
                  f'way["natural"="coastline"]({b});']
    q = "[out:json][timeout:180];(" + "".join(parts) + ");out tags geom;"
    for attempt in range(4):
        try:
            res = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=240,
                                headers={"User-Agent": "near-miss-research/0.1"})
            if res.status_code == 200: break
            print("status", res.status_code); time.sleep(30)
        except Exception as e:
            print("err", e); time.sleep(30)
    else:
        raise SystemExit("overpass failed")
    for e in res.json()["elements"]:
        els[str(e["id"])] = e
    done |= set(ch.segment_id)
    json.dump(els, open(cache, "w")); json.dump(sorted(done), open(OUT / "osm_done.json", "w"))
    print(f"{i + len(ch)}/{len(todo)}  要素 {len(els)}"); time.sleep(5)

# セグメントの折れ線と各要素の距離 (局所の平面近似)
def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay; L2 = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / np.where(L2 == 0, 1, L2), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))
feat = []
for e in els.values():
    g = e.get("geometry")
    if not g: continue
    la = np.array([q["lat"] for q in g]); lo = np.array([q["lon"] for q in g])
    k = 111320 * np.cos(np.radians(la.mean()))
    L = np.hypot(np.diff(lo) * k, np.diff(la) * 110540).sum()
    tg = e["tags"]
    kind = "coast" if tg.get("natural") == "coastline" else ("bridge" if tg.get("bridge") else "tunnel")
    feat.append(dict(id=e["id"], kind=kind, la=la, lo=lo, length=L, name=tg.get("name", ""), ref=tg.get("ref", ""),
                     bridge_name=tg.get("bridge:name", ""), hw=tg.get("highway", "")))
rows = []
for _, r in hs.iterrows():
    k = 111320 * np.cos(np.radians(r.lat_mid))
    P = np.array([[r.lon_start * k, r.lat_start * 110540], [r.lon_mid * k, r.lat_mid * 110540], [r.lon_end * k, r.lat_end * 110540]])
    best = {"bridge": (np.inf, None), "tunnel": (np.inf, None), "coast": (np.inf, None)}
    for f in feat:
        if abs(f["la"].mean() - r.lat_mid) > 0.05: continue
        if f["kind"] == "bridge" and f["length"] < 150: continue
        X, Y = f["lo"] * k, f["la"] * 110540
        dd = min(seg_dist(X, Y, *P[j], *P[j + 1]).min() for j in range(2))
        if dd < best[f["kind"]][0]: best[f["kind"]] = (dd, f)
    row = dict(segment_id=r.segment_id, country=r.country, admin1=r.admin1, place=r.place, lat=r.lat_mid, lon=r.lon_mid,
               speed=r.speed_mid)
    for kd, (dd, f) in best.items():
        row[f"{kd}_dist_m"] = dd
        if f is not None and kd != "coast":
            row[f"{kd}_len_m"] = f["length"]; row[f"{kd}_name"] = f["bridge_name"] or f["name"]; row[f"{kd}_ref"] = f["ref"]
    rows.append(row)
d = pd.DataFrame(rows); d.to_csv(OUT / "seg_exposure.csv", index=False, float_format="%.6g")
print(len(d), "segments")
print("橋 (150 m 以上) が 60 m 以内:", (d.bridge_dist_m < 60).sum(), " うち 500 m 以上:", ((d.bridge_dist_m < 60) & (d.bridge_len_m >= 500)).sum())
print("トンネル 60 m 以内:", (d.tunnel_dist_m < 60).sum(), " 海岸線 500 m 以内:", (d.coast_dist_m < 500).sum())
