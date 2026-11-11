"""風にさらされやすい長い通過 (橋の上 10 s 以上、または海岸線 500 m 以内のセグメントの橋) について、
通過中の 4 s 窓ごとに max|βc − 6 s 中央値| と横ずれの幅を取り、対照 (橋から離れた時刻の 4 s 窓) の分位で表す。
出力: <out>/exposed_scan.csv"""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(sys.argv[1]); FS = 20.0
fit = json.load(open(OUT / "beta_fit.json"))
pas = pd.read_csv(OUT / "passages_comp.csv")
ex = pas[(pas.dur_s >= 10) | (pas.coast_dist_m < 500)].drop_duplicates(["segment_id", "t_in"])
def prep(sid):
    d = pd.read_csv(glob.glob(str(OUT / "sig" / f"{sid}*.csv.gz"))[0])
    rr = np.radians(d.crs_rate); vv = d.v.clip(lower=3)
    d["bc"] = (d.beta1 - (fit["a_deg_per_radpm"] * rr / vv + fit["b"] * vv * rr + fit["c"])).where(d.v > 10)
    d["bc_dev"] = d.bc - d.bc.rolling(120, center=True, min_periods=40).median()
    return d
# 対照の 4 s 窓
nul = []
for f in glob.glob(str(OUT / "sig" / "*.csv.gz")):
    d = prep(Path(f).name.split(".")[0])
    for i in range(60, len(d) - 140, 40):
        w = d.iloc[i:i + 80]
        if d.near_m.iloc[i - 60:i + 140].min() < 100 or w.v.min() < 15 or w.crs_rate.abs().max() > 3: continue
        nul.append((w.bc_dev.abs().max(), np.ptp(w.xtrk.dropna()) if w.xtrk.notna().sum() > 10 else np.nan))
nul = np.array(nul); print("対照の窓", len(nul), "95%:", np.nanquantile(nul, 0.95, axis=0).round(3))
rows = []
for _, p in ex.iterrows():
    d = prep(p.segment_id); m = d.t.between(p.t_in - 2, p.t_out + 2)
    idx = np.flatnonzero(m)
    if len(idx) == 0: continue
    best = None
    for i in range(idx[0], max(idx[0] + 1, idx[-1] - 79), 10):
        w = d.iloc[i:i + 80]
        if len(w) < 40 or w.v.min() < 10: continue
        bd = w.bc_dev.abs().max(); xt = np.ptp(w.xtrk.dropna()) if w.xtrk.notna().sum() > 10 else np.nan
        pb = (nul[:, 0] < bd).mean(); px = np.nanmean(nul[:, 1] < xt) if np.isfinite(xt) else np.nan
        sc = np.nanmean([pb, px])
        if best is None or sc > best["score_w"]:
            best = dict(t_win=w.t.iloc[len(w) // 2], bdev=bd, xt=xt, p_bdev=pb, p_xt=px, score_w=sc, crs_max=w.crs_rate.abs().max())
    if best: rows.append(dict(segment_id=p.segment_id, name=p["name"], ref=p.ref, place=p.place, admin1=p.admin1, country=p.country,
                              dur_s=p.dur_s, coast_dist_m=p.coast_dist_m, v=p.v, **best))
r = pd.DataFrame(rows).sort_values("score_w", ascending=False); r.to_csv(OUT / "exposed_scan.csv", index=False, float_format="%.5g")
pd.set_option("display.width", 250); x = r.copy(); x["segment_id"] = x.segment_id.str[:8]; x["name"] = x.name.astype(str).str[:24]
print(len(r), "件"); print(x.round(3).to_string())
