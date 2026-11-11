"""直進窓の β (deg) を、ドライブ固定効果 + 升目×向き固定効果 + k·W_left で当てはめる。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr
W = Path(sys.argv[1]); G = Path(sys.argv[2]); O = Path(sys.argv[3])
w = pd.read_csv(W / "windows_scored.csv").dropna(subset=["W", "D"])
g = pd.read_csv(G / "grid_beta.csv.gz", usecols=["drive", "seg", "t", "beta"])
gb = {k: x for k, x in g.groupby(["drive", "seg"])}
def wmean(r):
    x = gb[(r.drive, r.seg)]; m = (x.t >= r.t - 2.5) & (x.t < r.t + 2.5)
    b = x.beta[m]; return pd.Series(dict(beta=b.mean(), beta_n=b.notna().sum(), beta_std=b.std()))
w = pd.concat([w, w.apply(wmean, axis=1)], axis=1)
w = w[w.beta_n >= 80].copy()

def fe_fit(x, use_cell=True, col="W_left"):
    """スパースな固定効果の最小二乗。k (col の係数) を返す。"""
    n = len(x); cols = []
    dcode = pd.factorize(x.drive)[0]
    cols.append(csr_matrix((np.ones(n), (np.arange(n), dcode)), shape=(n, dcode.max() + 1)))
    if use_cell:
        cc = pd.factorize(x.cell)[0]
        C = csr_matrix((np.ones(n), (np.arange(n), cc)), shape=(n, cc.max() + 1))[:, 1:]  # 1 つ落とす
        cols.append(C)
    cols.append(csr_matrix(x[col].to_numpy()[:, None]))
    A = hstack(cols).tocsr()
    sol = lsqr(A, x.beta.to_numpy(), atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    res = x.beta.to_numpy() - A @ sol
    return sol[-1], res

out = {"n_windows": len(w), "n_drives": int(w.drive.nunique())}
rng = np.random.default_rng(0); drives = w.drive.unique()
for name, use_cell in (("drive_fe", False), ("drive_and_site_fe", True)):
    for col in ("W_left", "vW"):
        if col == "vW": w["vW"] = w.v * w.W_left / 25.0
        k, res = fe_fit(w, use_cell, col)
        bs = []
        for _ in range(500):
            pick = rng.choice(drives, len(drives))
            b = pd.concat([w[w.drive == d_].assign(drive=f"{d_}#{i}") for i, d_ in enumerate(pick)])
            bs.append(fe_fit(b, use_cell, col)[0])
        out[f"{name}:{col}"] = dict(k_deg_per_mps=k, ci95=list(np.percentile(bs, [2.5, 97.5])), resid_std_deg=float(res.std()))
# ドライブの定数だけ引いた β を W_left で区分
w["b_d"] = w.beta - w.groupby("drive").beta.transform("median")
t = w.groupby(pd.cut(w.W_left, [-15, -8, -5, -1, 1, 5, 8, 15]), observed=True).agg(n=("b_d", "size"), drives=("drive", "nunique"), median=("b_d", "median"))
out["beta_minus_drive_median_by_Wleft"] = {str(k): {kk: float(vv) for kk, vv in r.items()} for k, r in t.iterrows()}
# 同じ窓の雑音
calm = w[w.W <= 3.1]; out["calm_window_beta_std_after_drive_median"] = float(calm.b_d.std())
out["window_internal_beta_std_median"] = float(w.beta_std.median())
w.to_csv(O / "windows_beta.csv", index=False, float_format="%.6g")
(O / "beta_wind.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print(json.dumps(out, indent=1, ensure_ascii=False, default=lambda z: round(float(z), 4)))
