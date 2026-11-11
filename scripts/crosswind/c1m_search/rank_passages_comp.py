"""β から旋回の幾何の分を差し引いてから、橋の通過を対照と比べ直す。
  β ≈ a·(r/v) + b·(v·r) + c   (r: 進路変化率 [rad/s]。a は重心から後軸までの距離 [m] に相当、b はタイヤのたわみ)
  全セグメントの v>15 の点で当てはめ、βc = β − 当てはめ値 を使う。
指標は seg_signals.py と同じ (step, tr, xt)。対照は橋・トンネルから 100 m 以上離れた時刻。
出力: <out>/passages_comp.csv, <out>/null_comp.csv, <out>/beta_fit.json"""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(sys.argv[1]); FS = 20.0
P = json.load(open(OUT / "seg_signals_params.json"))
sig = {Path(f).name.split(".")[0]: pd.read_csv(f) for f in glob.glob(str(OUT / "sig" / "*.csv.gz"))}
allp = pd.concat([d.assign(seg=k) for k, d in sig.items()])
m = (allp.v > 15) & allp.beta1.notna()
r = np.radians(allp.crs_rate[m]); v = allp.v[m]
A = np.c_[r / v, v * r, np.ones(m.sum())]
coef, *_ = np.linalg.lstsq(A, allp.beta1[m], rcond=None)   # beta1 は deg
res = allp.beta1[m] - A @ coef
fit = dict(a_deg_per_radpm=coef[0], a_m=float(np.radians(coef[0])), b=coef[1], c=coef[2],
           r2=float(1 - res.var() / allp.beta1[m].var()), n=int(m.sum()))
print("当てはめ:", {k: round(v_, 4) for k, v_ in fit.items()})
json.dump(fit, open(OUT / "beta_fit.json", "w"), indent=1)
pas = pd.read_csv(OUT / "passages.csv")
rows, nulls = [], []
for sid, d in sig.items():
    rr = np.radians(d.crs_rate); vv = d.v.clip(lower=3)
    d["bc"] = (d.beta1 - (coef[0] * rr / vv + coef[1] * vv * rr + coef[2])).where(d.v > 10)
    d["bc_dev"] = d.bc - d.bc.rolling(int(6 * FS), center=True, min_periods=40).median()
    def metrics(i0, i1):
        hw = int(P["TR_HALF_S"] * FS); pre = d.bc.iloc[max(0, i0 - int(P["PRE_S"] * FS)):i0]; on = d.bc.iloc[i0:i1 + 1]
        a = slice(max(0, i0 - hw), i0 + hw); b = slice(max(0, i1 - hw), i1 + hw)
        return dict(step_c=on.median() - pre.median(), tr_c=max(d.bc_dev.iloc[a].abs().max(), d.bc_dev.iloc[b].abs().max()),
                    bc_on_absmax=on.abs().max())
    for _, p in pas[pas.segment_id.str[:len(sid)] == sid].iterrows():
        i0 = int(np.argmin(np.abs(d.t - p.t_in))); i1 = int(np.argmin(np.abs(d.t - p.t_out)))
        rows.append({**p.to_dict(), **metrics(i0, i1)})
    near = d.near_m.to_numpy(); L = int(4 * FS)
    for i0 in range(int(P["PRE_S"] * FS), len(d) - L - 60, int(2 * FS)):
        if near[max(0, i0 - 60):i0 + L + 60].min() < 100 or d.v.iloc[i0] < 15: continue
        mm = metrics(i0, i0 + L); mm["xt"] = np.ptp(d.xtrk.iloc[i0 - 60:i0 + L + 60].dropna()) if d.xtrk.iloc[i0-60:i0+L+60].notna().sum() > 5 else np.nan
        mm["crs_rate_max"] = d.crs_rate.iloc[i0 - 60:i0 + L + 60].abs().max(); nulls.append(mm)
p = pd.DataFrame(rows); n = pd.DataFrame(nulls)
p["xt"] = p[["xt_in", "xt_out"]].max(1)
def pct(x, ref):
    ref = np.sort(ref.dropna().to_numpy()); x = np.asarray(x, float)
    return np.where(np.isfinite(x), np.searchsorted(ref, np.nan_to_num(x)) / len(ref), np.nan)  # 欠損は欠損のまま
p["p_step"] = pct(p.step_c.abs(), n.step_c.abs()); p["p_tr"] = pct(p.tr_c, n.tr_c); p["p_xt"] = pct(p.xt, n.xt)
p["n_ind"] = p[["p_step", "p_tr", "p_xt"]].notna().sum(1)
p["score"] = p[["p_step", "p_tr", "p_xt"]].mean(1, skipna=True); p["curvy"] = p.crs_rate_max > 3
p = p.sort_values("score", ascending=False)
p.to_csv(OUT / "passages_comp.csv", index=False, float_format="%.5g"); n.to_csv(OUT / "null_comp.csv", index=False, float_format="%.5g")
print("対照 95%:", {c: round(float(n[c].abs().quantile(0.95)), 3) for c in ("step_c", "tr_c", "xt")}, "対照の数", len(n))
pd.set_option("display.width", 250)
c = ["segment_id", "name", "ref", "place", "admin1", "dur_s", "v", "step_c", "tr_c", "xt", "crs_rate_max", "p_step", "p_tr", "p_xt", "score", "curvy", "coast_dist_m"]
x = p[c].copy(); x["segment_id"] = x.segment_id.str[:8]; x["name"] = x.name.astype(str).str[:22]
print(x.head(20).round(3).to_string())
print("曲線でない通過:"); print(x[~p.curvy].head(12).round(3).to_string())
