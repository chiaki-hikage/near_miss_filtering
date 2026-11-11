"""候補に、同じ場所の他ドライブとの比較 (一意性)・風との整合を付け、出現率を風の強さ別に比べる。
一意性: 候補の地点 (半径 25 m・方位差 20 度) を通った他ドライブそれぞれについて、その範囲での
        max(pol·z_dom) を取り、候補の値以上だったドライブの割合 (frac_other_ge)。
出力: <out>/cands_ranked.csv, <out>/rates.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
OUT = Path(sys.argv[1])
lat0 = 37.65; kx, ky = 111320 * np.cos(np.radians(lat0)), 110540
allc, rates = [], []
for car in ("rav4", "civic"):
    f = pd.read_csv(OUT / f"feat_{car}.csv.gz", usecols=["drive", "seg", "t", "lat", "lon", "H", "v", "z_yres", "z_beta", "z_ayres", "W", "W_left"])
    c = pd.read_csv(OUT / f"cands_{car}.csv")
    f = f[(f.v > 15) & f.lat.notna()].reset_index(drop=True)
    tree = cKDTree(np.c_[f.lon * kx, f.lat * ky])
    fr, nd = [], []
    for _, r in c.iterrows():
        nb = np.asarray(tree.query_ball_point([r.lon * kx, r.lat * ky], 25.0), dtype=int)
        nb = nb[(f.drive.to_numpy()[nb] != r.drive) & (np.abs((f.H.to_numpy()[nb] - r.H + 180) % 360 - 180) < 20)]
        val = r.pol * f["z_" + r.dom].to_numpy()[nb]
        mx = pd.Series(val).groupby(f.drive.to_numpy()[nb]).max()
        cand_val = abs(r["z_" + r.dom])
        fr.append((mx >= cand_val).mean() if len(mx) else np.nan); nd.append(len(mx))
    c["frac_other_ge"] = fr; c["n_other"] = nd
    c["wind_cons"] = np.sign(c.W_left) == c.pol
    c["Wx"] = c.pol * c.W_left                       # 外乱の向きに沿った横風成分 (+ = 整合)
    allc.append(c)
    # 出現率: 直進に近い (±3 s の |r|, |r_pred| < 3 deg/s) 15 m/s 超の時間あたり、zmax≥4 かつ 2 指標以上一致
    f["Wbin"] = pd.cut(f.W, [0, 5, 8, 10, 20], labels=["≤5", "5–8", "8–10", ">10"])
    hrs = f.groupby("Wbin", observed=True).size() / 20 / 3600
    c["Wbin"] = pd.cut(c.W, [0, 5, 8, 10, 20], labels=["≤5", "5–8", "8–10", ">10"])
    st = c[(c.yaw_absmax3 < 3) & (c.rpred_absmax3 < 3) & (c[["z_yres", "z_beta", "z_ayres"]].abs().max(1) >= 4) & (c.n_agree >= 2)]
    for wb, h in hrs.items():
        s = st[st.Wbin == wb]
        rates.append(dict(car=car, W=wb, hours=h, n=len(s), per_h=len(s) / h if h else np.nan,
                          n_cons=int(s.wind_cons.sum()), n_opp=int((~s.wind_cons & s.W_left.abs().gt(1)).sum())))
c = pd.concat(allc, ignore_index=True)
c.to_csv(OUT / "cands_ranked.csv", index=False, float_format="%.6g")
r = pd.DataFrame(rates); r.to_csv(OUT / "rates.csv", index=False, float_format="%.4g"); print(r.to_string())
