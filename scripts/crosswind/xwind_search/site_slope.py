"""地点ごとに、ドライブ単位の「風に固有の残差」 wres と SFO の横風成分 W_left の傾きを出す。
地点 = 100 m マス × 方位 30 度 (同じ車種)。各ドライブがその地点を通ったときの wres の平均と W_left の平均。
wres = (yres1 − 基準) + κ(v)·(ayres1 − 基準)。横風なら傾き k < 0 (左から吹かれるとヨー残差が負)。
対照: 地点の中で W_left をドライブ間で入れ替えた (1000 回) ときの、全地点の t の最小値の分布 (多重比較の補正)。
出力: <out>/site_slope_<car>.csv, <out>/site_slope_perm.json"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, yaml
OUT = Path(sys.argv[1]); rng = np.random.default_rng(0)
CFG = {"civic": "configs/crosswind/honda_civic.yaml", "rav4": "configs/vehicles/toyota_rav4.yaml"}
lat0 = 37.65; kx, ky = 111320 * np.cos(np.radians(lat0)), 110540
res = {}
for car in ("civic", "rav4"):
    g = yaml.safe_load(open(CFG[car]))["geometry"]
    f = pd.read_csv(OUT / f"feat_{car}.csv.gz", usecols=["drive", "t", "lat", "lon", "H", "v", "yaw", "r_pred", "yres1", "base_yres1", "ayres1", "base_ayres1", "W", "W_left", "osm_bridge"])
    s = f[(f.v > 15) & (f.yaw.abs() < 3) & f.base_yres1.notna() & f.base_ayres1.notna() & f.W_left.notna()].copy()
    kap = g["understeer_gradient"] * s.v**2 / (g["wheelbase_m"] + g["understeer_gradient"] * s.v**2)
    s["e"] = (s.yres1 - s.base_yres1) + kap * (s.ayres1 - s.base_ayres1)
    s["cx"] = (s.lon * kx // 100).astype(int); s["cy"] = (s.lat * ky // 100).astype(int); s["ch"] = ((s.H + 15) % 360 // 30).astype(int)
    m = s.groupby(["cx", "cy", "ch", "drive"]).agg(e=("e", "mean"), Wl=("W_left", "mean"), W=("W", "mean"), n=("e", "size"),
                                                   lat=("lat", "mean"), lon=("lon", "mean"), H=("H", "median"),
                                                   bridge=("osm_bridge", lambda x: (x == "yes").mean())).reset_index()
    m = m[m.n >= 10]
    rows, perm_min = [], np.full(1000, np.inf)
    groups = [x for _, x in m.groupby(["cx", "cy", "ch"]) if len(x) >= 8 and x.Wl.std() > 2]
    def tstat(e, w):
        w0 = w - w.mean(); k = (w0 * (e - e.mean())).sum() / (w0 ** 2).sum()
        r = e - e.mean() - k * w0; se = np.sqrt((r ** 2).sum() / (len(e) - 2) / (w0 ** 2).sum()); return k, k / se
    for x in groups:
        e, w = x.e.to_numpy(), x.Wl.to_numpy()
        k, t = tstat(e, w)
        strong = x[x.W > 8]
        rows.append(dict(cx=x.cx.iloc[0], cy=x.cy.iloc[0], ch=x.ch.iloc[0], lat=x.lat.mean(), lon=x.lon.mean(), H=x.H.median(),
                         n_drives=len(x), k=k, t=t, W_left_sd=w.std(), bridge=x.bridge.max(),
                         e_strong_mean=strong.e.mean() if len(strong) else np.nan, n_strong=len(strong)))
        for i in range(1000):
            perm_min[i] = min(perm_min[i], tstat(e, rng.permutation(w))[1])
    r = pd.DataFrame(rows).sort_values("t"); r.to_csv(OUT / f"site_slope_{car}.csv", index=False, float_format="%.5g")
    q = np.quantile(perm_min, [0.05, 0.5]); n_obs = (r.t < q[0]).sum()
    # 片側の向き (k<0) と逆向き (k>0) の数を比べる (地点は重なりがあるので目安)
    res[car] = dict(n_sites=len(r), perm_min_t_q05=float(q[0]), perm_min_t_median=float(q[1]), n_sites_below_q05=int(n_obs),
                    obs_min_t=float(r.t.min()), frac_k_neg=float((r.k < 0).mean()), n_t_lt_m3=int((r.t < -3).sum()), n_t_gt_p3=int((r.t > 3).sum()),
                    p_min_t=float((perm_min <= r.t.min()).mean()))
    pd.set_option("display.width", 200); print(car, res[car]); print(r.head(12).round(4).to_string())
json.dump(res, open(OUT / "site_slope_perm.json", "w"), indent=1)
