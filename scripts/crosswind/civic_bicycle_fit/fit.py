"""Civic の単軌道モデルの当てはめと評価。RAV4 (docs/signals_rav4.md) と同じ式・条件。

    δsw[rad] = a·(r/v) + b·(v·r) + c      a = SR·L, b = SR·Kus, c = δoffset
条件: v > 10 m/s, |横躍度| < 0.5 m/s³, |r| > 1 deg/s。当てはめは弱風 (calm_same_site) のセグメントだけ。
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

O = Path(sys.argv[1])
d = pd.read_csv(O / "grid_20hz.csv.gz")
RAV4 = dict(a=16.75 * 2.66, b=16.75 * 0.00235, c=0.911, sigma=0.341)

def cond(x, yaw="yaw_dps"):
    return (x.v > 10) & (x.lat_jerk.abs() < 0.5) & (x[yaw].abs() > 1) & np.isfinite(x[yaw]) & np.isfinite(x.steer)

def fit(x, yaw="yaw_dps"):
    r = np.radians(x[yaw].to_numpy()); v = x.v.to_numpy(); y = np.radians(x.steer.to_numpy())
    X = np.c_[r / v, v * r, np.ones_like(v)]
    a, b, c = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ [a, b, c]
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(a=a, b=b, c_deg=np.degrees(c), r2_steer=r2, n=len(v))

def predict_yaw(x, p):
    """舵角から予測したヨーレート [deg/s] (bicycle_yaw_rate と同じ式)。"""
    v = x.v.to_numpy(); dlt = np.radians(x.steer.to_numpy() - p["c_deg"])
    return np.degrees(v * dlt / (p["a"] + p["b"] * v ** 2))

def resid_stats(res):
    res = res[np.isfinite(res)]; a = np.abs(res)
    return dict(n=len(res), median=np.median(res), std=res.std(), mad_sigma=1.4826 * np.median(np.abs(res - np.median(res))),
                p99=np.percentile(a, 99), p999=np.percentile(a, 99.9), max=a.max())

calm = d[d.reason == "calm_same_site"]
m = cond(calm)
P = fit(calm[m])
P["SR_times_L"] = P["a"]; P["SR_times_Kus"] = P["b"]
res = calm.loc[m, "yaw_dps"].to_numpy() - predict_yaw(calm[m], P)
out = {"fit": P, "resid_fit_set": resid_stats(res)}

# 補正なしのジャイロで同じことをすると
m0 = cond(calm, "yaw_raw_dps"); P0 = fit(calm[m0], "yaw_raw_dps")
out["fit_no_bias_correction"] = P0
out["resid_no_bias_correction"] = resid_stats(calm.loc[m0, "yaw_raw_dps"].to_numpy() - predict_yaw(calm[m0], P0))

# 5 分割の交差検証 (ドライブ単位)
drives = np.array(sorted(calm.drive.unique())); rng = np.random.default_rng(0); rng.shuffle(drives)
folds = np.array_split(drives, 5); cv = []; cv_res = []
for k, te in enumerate(folds):
    tr = calm[~calm.drive.isin(te)]; ts = calm[calm.drive.isin(te)]
    p = fit(tr[cond(tr)]); mt = cond(ts)
    rr = ts.loc[mt, "yaw_dps"].to_numpy() - predict_yaw(ts[mt], p); cv_res.append(rr)
    cv.append(dict(fold=k, a=p["a"], b=p["b"], c_deg=p["c_deg"], n_test=int(mt.sum()), resid_std=rr.std()))
out["cv"] = cv; out["resid_cv_pooled"] = resid_stats(np.concatenate(cv_res))

# 見かけの SR·L (Kus を無視) の速度依存
x = calm[m].copy(); x["app"] = np.radians(x.steer - P["c_deg"]) / (np.radians(x.yaw_dps) / x.v)
x["vbin"] = pd.cut(x.v, [10, 15, 20, 25, 30, 35, 40])
app = x.groupby("vbin", observed=True).agg(n=("app", "size"), app_median=("app", "median"))
app["model"] = [P["a"] + P["b"] * ((iv.left + iv.right) / 2) ** 2 for iv in app.index]
out["apparent_SR_L_by_speed"] = {str(k): dict(n=int(r.n), median=float(r.app_median), model=float(r.model)) for k, r in app.iterrows()}

# 直進・定常 (v > 10, |r| < 1, |横躍度| < 0.5) の残差: 横風の検出で使う範囲
st = (calm.v > 10) & (calm.lat_jerk.abs() < 0.5) & (calm.yaw_dps.abs() <= 1)
rs = calm.loc[st, "yaw_dps"].to_numpy() - predict_yaw(calm[st], P)
out["resid_straight"] = resid_stats(rs)
v = calm.loc[st, "v"].to_numpy()
out["resid_straight_ay_equiv_mps2"] = resid_stats(np.radians(rs) * v)   # v·Δr = 横加速度に換算

# ドライブごとの残差の中央値 (バイアスの推定誤差と舵角ゼロ点の揺れの合計)
calm2 = calm[m | st].copy(); calm2["res"] = calm2.yaw_dps - predict_yaw(calm2, P)
dm = calm2.groupby("drive").res.median()
out["resid_drive_median"] = dict(std=dm.std(), min=dm.min(), max=dm.max(), n=len(dm))

# バイアスの推定の確認: 停車区間を外して推定した値と、停車中の実測を比べる
# (停車中の観測を使わない推定は build 側で別に作っていないので、停車の直前直後 30 s の走行中推定を使う)
chk = []
for (dr, sg), g in d.groupby(["drive", "seg"]):
    s = g[g.stop]
    if len(s) < 100: continue
    t0, t1 = s.t.min(), s.t.max()
    mv = g[(g.v > 5) & (g.vgp > 5) & (((g.t > t0 - 30) & (g.t < t0)) | ((g.t > t1) & (g.t < t1 + 30)))]
    if len(mv) < 100: continue
    est = np.median(mv.bias_obs.dropna()); true = s.gz.mean()
    chk.append(dict(drive=dr, seg=sg, stop_mean_dps=np.degrees(true), moving_est_dps=np.degrees(est)))
chk = pd.DataFrame(chk); chk["err"] = chk.moving_est_dps - chk.stop_mean_dps
chk.to_csv(O / "bias_check_stops.csv", index=False)
out["bias_check_vs_stop"] = dict(n=len(chk), err_median=chk.err.median(), err_mad_sigma=1.4826 * (chk.err - chk.err.median()).abs().median(),
                                 err_abs_p90=chk.err.abs().quantile(0.9))
out["bias_interp_fraction"] = float(d.bias_interp.mean())
out["rav4_reference"] = RAV4
(O / "fit_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
print(json.dumps(out, indent=1, ensure_ascii=False, default=lambda z: round(float(z), 4)))
