"""進行方向の反転による検定 (ドライブ単位)。風に固有の残差 wres = e_y + κ·e_a 版。
直進 (|yaw|, |r_pred| < 1 deg/s, v > 15) の点で、e = yres1 − 場所の基準 を、横風成分 W_left の符号別に平均する。
横風なら  e(W_left>0) < 0 < e(W_left<0)  (左から吹かれると右へ押され、ヨー残差は負)。
舵角ゼロ点・ジャイロのずれなら、符号によらず同じ向き。
差 Δ = e(W_left<0) − e(W_left>0) を、両側の |W_left| 平均で割った感度 k [m/s² per m/s] も出す。
出力: <out>/reversal_<car>.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
OUT = Path(sys.argv[1])
CFG = {"civic": "configs/crosswind/honda_civic.yaml", "rav4": "configs/vehicles/toyota_rav4.yaml"}
for car in ("civic", "rav4"):
    f = pd.read_csv(OUT / f"feat_{car}.csv.gz", usecols=["drive", "seg", "t", "v", "yaw", "r_pred", "yres1", "base_yres1", "beta1", "base_beta1", "ayres1", "base_ayres1", "W", "W_left", "H"])
    g = yaml.safe_load(open(CFG[car]))["geometry"]
    s = f[(f.v > 15) & (f.yaw.abs() < 1) & (f.r_pred.abs() < 1) & f.base_yres1.notna() & f.W_left.notna()].copy()
    kap = g["understeer_gradient"] * s.v**2 / (g["wheelbase_m"] + g["understeer_gradient"] * s.v**2)
    # 路面の横断勾配ぶんを差し引いた「風に固有の残差」 (勾配: e_y ≈ −κ e_a、一定の横風: e_a ≈ 0)
    s["e"] = (s.yres1 - s.base_yres1) + kap * (s.ayres1 - s.base_ayres1); s["eb"] = s.beta1 - s.base_beta1
    s = s[s.W_left.abs() > 2].dropna(subset=["e"])
    rows = []
    for dr, x in s.groupby("drive"):
        p, n = x[x.W_left > 0], x[x.W_left < 0]
        if len(p) < 200 or len(n) < 200: continue          # 各向き 10 s 以上
        # 1 s ごとに間引いて、自己相関を弱めた点数で標準誤差を出す (目安)
        pe, ne = p.e.iloc[::20], n.e.iloc[::20]
        d_ = ne.mean() - pe.mean(); se = np.sqrt(pe.var() / len(pe) + ne.var() / len(ne))
        wl = (p.W_left.mean() - n.W_left.mean())
        rows.append(dict(drive=dr[-20:], W=x.W.median(), n_pos_s=len(p) / 20, n_neg_s=len(n) / 20,
                         e_windleft=pe.mean(), e_windright=ne.mean(), diff=d_, se=se, z=d_ / se, k=d_ / wl,
                         eb_windleft=p.eb.mean(), eb_windright=n.eb.mean()))
    r = pd.DataFrame(rows).sort_values("W", ascending=False); r.to_csv(OUT / f"reversal_wres_{car}.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 200); print(car); print(r.round(3).to_string())
