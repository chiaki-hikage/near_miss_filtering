"""横外乱の特徴量を作り、同じ場所の他ドライブと比べて外れた山を広めに拾う。

特徴量 (20 Hz, 1 s 移動平均):
  yres  = v·(r − r_pred_lag) [m/s²]   r_pred = 単軌道モデル (車種ごとの当てはめ値)、
          r_pred_lag = 一次遅れ (時定数 tau) と遅延 (lag) を掛けた予測。tau/lag は弱風の直線外の点で残差分散最小に当てはめる
  beta  = beta_raw − ドライブごとの直進時の中央値 (取付の向き) [deg]
  ayres = ay_meas − v·r [m/s²] (= v·dβ/dt + 路面の傾き·g + センサのずれ)。ドライブごとの直進時の中央値を引く
場所の基準: 半径 R_SITE m・方位差 20 度以内にある、他ドライブ (同じ車種) の点の中央値 (3 ドライブ以上あるとき)。
z = (特徴量 − 基準) / 直進時のロバスト σ。
外乱の向き pol: 右へ押された (左から吹かれた) とき +。yres, beta, ayres はすべて負になるので z を符号反転して使う。
候補: v > 15 m/s で max(|z|) ≥ Z_MIN の山 (5 s 以上離す)。
出力: <out>/feat_<car>.csv.gz, <out>/cands_<car>.csv, <out>/fit_<car>.json
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy.spatial import cKDTree
from scipy.signal import lfilter
REPO = Path.cwd(); sys.path.insert(0, str(REPO / "src"))
from near_miss.features import bicycle_yaw_rate
car, OUT = sys.argv[1], Path(sys.argv[2])
P = dict(FS=20.0, R_SITE=20.0, DH_SITE=20.0, MIN_DRIVES=3, Z_MIN=3.0, V_MIN=15.0, SEP_S=5.0, CALM_W=5.0)
FS = P["FS"]
cfgp = {"rav4": "configs/vehicles/toyota_rav4.yaml", "civic": "configs/crosswind/honda_civic.yaml"}[car]
g = yaml.safe_load(open(REPO / cfgp))["geometry"]
d = pd.read_csv(OUT / f"grid_{car}.csv.gz")
pw = pd.read_csv(OUT / "positions_wind.csv.gz", usecols=["drive", "seg", "t", "W", "D", "G", "wsrc", "osm_ref", "osm_name", "osm_bridge", "osm_bridge_name", "osm_tunnel"])
d = d.sort_values(["drive", "seg", "t"]).reset_index(drop=True)
d["r_pred"] = bicycle_yaw_rate(d.v.to_numpy(), d.steer.to_numpy(), g["wheelbase_m"], g["steer_ratio"],
                               g["understeer_gradient"], g["steer_offset_deg"])
segkey = d.drive + "#" + d.seg.astype(str)

def lagged(x, tau, lag_n):
    out = np.empty_like(x)
    for _, idx in d.groupby(segkey).indices.items():
        y = x[idx].copy(); ok = np.isfinite(y); y[~ok] = 0
        if tau > 0:
            a = np.exp(-1 / (FS * tau)); y = lfilter([1 - a], [1, -a], y, zi=[y[0] * a])[0]
        if lag_n > 0: y = np.r_[np.full(lag_n, y[0]), y[:-lag_n]]
        y[~ok] = np.nan; out[idx] = y
    return out

# 一次遅れの当てはめ (風の弱い時間、15 m/s 超)。直線だけでは差が出ないので全点で
pw1 = pw.groupby("drive").W.median()
calm = d.drive.map(pw1).to_numpy() < 99
fitm = (d.v > P["V_MIN"]).to_numpy() & np.isfinite(d.r_pred.to_numpy()) & np.isfinite(d.yaw.to_numpy())
best = None
for tau in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
    for lag_n in (0, 1, 2, 3, 4):
        rp = lagged(d.r_pred.to_numpy(), tau, lag_n)
        s = np.nanstd((d.yaw.to_numpy() - rp)[fitm])
        if best is None or s < best[0]: best = (s, tau, lag_n)
s0 = np.nanstd((d.yaw - d.r_pred).to_numpy()[fitm])
print(f"{car}: 残差 σ 遅れなし {s0:.3f} → tau={best[1]} s, lag={best[2]/FS:.2f} s で {best[0]:.3f} deg/s")
d["r_pred_lag"] = lagged(d.r_pred.to_numpy(), best[1], best[2])

def roll1(col):
    return d.groupby(segkey)[col].transform(lambda s: s.rolling(int(FS), center=True, min_periods=int(FS * 0.8)).mean())

d["yres"] = d.v * np.radians(d.yaw - d.r_pred_lag)
straight = (d.v > P["V_MIN"]) & (d.yaw.abs() < 1.0) & (d.r_pred.abs() < 1.0)
d["beta"] = d.beta_raw - d.drive.map(d[straight].groupby("drive").beta_raw.median())
d["ayres"] = d.ay_meas - d.v * np.radians(d.yaw)
d["ayres"] = d.ayres - d.drive.map(d[straight].groupby("drive").ayres.median())
for c in ("yres", "beta", "ayres"): d[c + "1"] = roll1(c)

# 場所の基準 (2 Hz の点で作って 20 Hz に補間)
lat0 = 37.65; kx, ky = 111320 * np.cos(np.radians(lat0)), 110540
d["x"], d["y"] = d.lon * kx, d.lat * ky
sub = d[(d.index % 10 == 0) & (d.v > 8) & d.x.notna() & d.H.notna()].copy()
tree = cKDTree(np.c_[sub.x, sub.y])
drv = sub.drive.to_numpy(); Hs = sub.H.to_numpy()
codes = pd.factorize(drv)[0]
feats = ("yres1", "beta1", "ayres1")
vals = {c: sub[c].to_numpy() for c in feats}
base = {c: np.full(len(sub), np.nan) for c in feats}; ndr = np.zeros(len(sub), int)
nbrs = tree.query_ball_point(np.c_[sub.x, sub.y], r=P["R_SITE"])
for i, nb in enumerate(nbrs):
    nb = np.asarray(nb)
    if len(nb) == 0: continue
    nb = nb[(codes[nb] != codes[i]) & (np.abs((Hs[nb] - Hs[i] + 180) % 360 - 180) < P["DH_SITE"])]
    if len(nb) == 0: continue
    u = np.unique(codes[nb]); ndr[i] = len(u)
    if len(u) < P["MIN_DRIVES"]: continue
    for c in feats:
        # ドライブごとの中央値の中央値 (止まっているドライブの重みを増やさない)
        v_ = pd.Series(vals[c][nb]).groupby(codes[nb]).median()
        base[c][i] = np.nanmedian(v_)
sub["n_site_drives"] = ndr
for c in feats: sub["base_" + c] = base[c]
for c in ["n_site_drives"] + ["base_" + c for c in feats]:
    d[c] = np.nan
    for _, idx in d.groupby(segkey).indices.items():
        s_ = sub.loc[sub.index.intersection(d.index[idx])]
        if len(s_) < 2: continue
        d.loc[d.index[idx], c] = np.interp(d.t.to_numpy()[idx], s_.t, s_[c], left=np.nan, right=np.nan)
sig = {}
ok = (d.v > P["V_MIN"]) & straight
for c in feats:
    e = (d[c] - d["base_" + c])[ok].dropna()
    sig[c] = float(1.4826 * np.median(np.abs(e - e.median())))
    d["z_" + c[:-1]] = -(d[c] - d["base_" + c]) / sig[c]      # 右へ押されると + (pol 付きの z)
print("ロバスト σ (1 s 平均 − 場所の基準, 直進):", {k: round(v, 4) for k, v in sig.items()})

# 風・道路の付与 (1 Hz の表から最も近い時刻)
pw = pw.sort_values(["drive", "seg", "t"])
d = d.sort_values("t")
d = pd.merge_asof(d, pw.sort_values("t"), on="t", by=["drive", "seg"], direction="nearest", tolerance=1.0)
d = d.sort_values(["drive", "seg", "t"]).reset_index(drop=True)
d["W_left"] = -d.W * np.sin(np.radians(d.D - d.H))

# 候補の山
Z = np.c_[d.z_yres, d.z_beta, d.z_ayres]
d["zmax"] = np.nanmax(np.abs(Z), axis=1)
rows = []
for k, idx in d.groupby(segkey).indices.items():
    x = d.iloc[idx]
    m = (x.v > P["V_MIN"]).to_numpy() & (x.zmax >= P["Z_MIN"]).to_numpy() & (x.n_site_drives >= P["MIN_DRIVES"]).to_numpy()
    zz = np.where(m, x.zmax.to_numpy(), -1)
    used = np.zeros(len(x), bool)
    for j in np.argsort(-zz):
        if zz[j] < 0: break
        if used[max(0, j - int(P["SEP_S"] * FS)):j + int(P["SEP_S"] * FS)].any(): continue
        used[j] = True
        r = x.iloc[j]
        w = x.iloc[max(0, j - 60):j + 61]              # ±3 s
        zs = {c: float(r["z_" + c]) for c in ("yres", "beta", "ayres")}
        dom = max(zs, key=lambda c: abs(zs[c]))
        pol = np.sign(zs[dom])
        rows.append(dict(car=car, drive=r.drive, seg=r.seg, t=r.t, utc=r.utc, lat=r.lat, lon=r.lon, H=r.H, v=r.v,
            z_yres=zs["yres"], z_beta=zs["beta"], z_ayres=zs["ayres"], dom=dom, pol=pol,
            n_agree=int(sum(np.sign(v_) == pol and abs(v_) >= 1.5 for v_ in zs.values())),
            yaw=r.yaw, yaw_absmax3=float(w.yaw.abs().max()), rpred_absmax3=float(w.r_pred.abs().max()),
            steer_range3=float(w.steer.max() - w.steer.min()), n_site_drives=int(r.n_site_drives),
            W=r.W, G=r.G, D=r.D, W_left=r.W_left, wsrc=r.wsrc, osm_ref=r.osm_ref, osm_name=r.osm_name,
            osm_bridge=r.osm_bridge, osm_bridge_name=r.osm_bridge_name))
c = pd.DataFrame(rows)
c.to_csv(OUT / f"cands_{car}.csv", index=False, float_format="%.6g")
keep = ["car", "drive", "seg", "t", "utc", "lat", "lon", "H", "v", "steer", "yaw", "r_pred", "r_pred_lag", "ay_meas",
        "beta_raw", "beta", "yres1", "beta1", "ayres1", "base_yres1", "base_beta1", "base_ayres1", "n_site_drives",
        "z_yres", "z_beta", "z_ayres", "W", "D", "G", "W_left", "osm_ref", "osm_bridge"]
d[keep].to_csv(OUT / f"feat_{car}.csv.gz", index=False, float_format="%.10g")
json.dump(dict(params=P, tau_s=best[1], lag_s=best[2] / FS, resid_sd_nolag=s0, resid_sd_lag=best[0], robust_sigma=sig,
               n_cands=len(c)), open(OUT / f"fit_{car}.json", "w"), indent=1)
print(len(c), "候補;", c.dom.value_counts().to_dict())
