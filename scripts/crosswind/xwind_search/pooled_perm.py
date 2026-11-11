"""地点の固定効果つきの傾き (全地点まとめ) を、ドライブ単位で風を入れ替える並べ替え検定で評価する。
各ドライブに別のドライブの (W, D) [ドライブの中央値] を割り当て、各地点の進行方位で W_left を作り直す。
出力: <out>/pooled_perm.json"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, yaml
OUT = Path(sys.argv[1]); rng = np.random.default_rng(1)
CFG = {"civic": "configs/crosswind/honda_civic.yaml", "rav4": "configs/vehicles/toyota_rav4.yaml"}
lat0 = 37.65; kx, ky = 111320 * np.cos(np.radians(lat0)), 110540
res = {}
for car in ("civic", "rav4"):
    g = yaml.safe_load(open(CFG[car]))["geometry"]
    f = pd.read_csv(OUT / f"feat_{car}.csv.gz", usecols=["drive", "lat", "lon", "H", "v", "yaw", "yres1", "base_yres1", "ayres1", "base_ayres1", "W", "D", "W_left"])
    s = f[(f.v > 15) & (f.yaw.abs() < 3) & f.base_yres1.notna() & f.base_ayres1.notna() & f.W_left.notna()].copy()
    kap = g["understeer_gradient"] * s.v**2 / (g["wheelbase_m"] + g["understeer_gradient"] * s.v**2)
    s["e"] = (s.yres1 - s.base_yres1) + kap * (s.ayres1 - s.base_ayres1)
    s["ey"] = s.yres1 - s.base_yres1
    s["cx"] = (s.lon * kx // 100).astype(int); s["cy"] = (s.lat * ky // 100).astype(int); s["ch"] = ((s.H + 15) % 360 // 30).astype(int)
    m = s.groupby(["cx", "cy", "ch", "drive"]).agg(e=("e", "mean"), ey=("ey", "mean"), Wl=("W_left", "mean"), H=("H", "mean"), n=("e", "size")).reset_index()
    m = m[m.n >= 10].copy()
    m["site"] = m.groupby(["cx", "cy", "ch"]).ngroup()
    cnt = m.groupby("site").size(); m = m[m.site.map(cnt) >= 5]
    dw = s.groupby("drive").agg(W=("W", "median"), D=("D", "median"))
    def pooled(Wl, col="e"):
        x = pd.DataFrame(dict(site=m.site, e=m[col], w=Wl))
        x["e0"] = x.e - x.groupby("site").e.transform("mean"); x["w0"] = x.w - x.groupby("site").w.transform("mean")
        return (x.w0 * x.e0).sum() / (x.w0 ** 2).sum()
    k_obs = pooled(m.Wl.to_numpy()); k_obs_y = pooled(m.Wl.to_numpy(), "ey")
    # 観測値は時刻ごとの風。並べ替えはドライブ中央値なので、比べるために中央値版の観測値も出す
    Wl_med = (-m.drive.map(dw.W) * np.sin(np.radians(m.drive.map(dw.D) - m.H))).to_numpy()
    k_obs_med = pooled(Wl_med)
    drives = dw.index.to_numpy(); perm = []
    for _ in range(2000):
        p = dict(zip(drives, rng.permutation(drives)))
        Wp = m.drive.map(lambda d: dw.W[p[d]]); Dp = m.drive.map(lambda d: dw.D[p[d]])
        perm.append(pooled((-Wp * np.sin(np.radians(Dp - m.H))).to_numpy()))
    perm = np.array(perm)
    res[car] = dict(n_sites=int(m.site.nunique()), n_rows=len(m), n_drives=int(m.drive.nunique()),
                    k_obs_wres=k_obs, k_obs_yres_only=k_obs_y, k_obs_wres_drive_median_wind=k_obs_med,
                    perm_q025=float(np.quantile(perm, 0.025)), perm_q975=float(np.quantile(perm, 0.975)),
                    p_one_sided=float((perm <= k_obs_med).mean()))
    print(car, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in res[car].items()})
json.dump(res, open(OUT / "pooled_perm.json", "w"), indent=1)
