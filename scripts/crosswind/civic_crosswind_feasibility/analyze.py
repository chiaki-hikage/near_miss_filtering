"""手順 5: 横風で直進時のヨー残差 (横加速度換算) が動くかを見る。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
O = Path(sys.argv[1]); w = pd.read_csv(O / "windows.csv").dropna(subset=["W", "D"])
out = {"n_windows": len(w), "n_drives": int(w.drive.nunique()), "wind_src": w.wind_src.value_counts().to_dict()}
CALM_W = 3.1
# 同地点・同方向の基準: 50 m の升目 × 30° の向き。弱風 (W ≤ 3.1) の窓の中央値
R = 6371e3
w["cx"] = (np.radians(w.lon) * np.cos(np.radians(37.7)) * R // 50).astype(int)
w["cy"] = (np.radians(w.lat) * R // 50).astype(int)
w["hs"] = (w.H // 30).astype(int)
w["cell"] = w.cx.astype(str) + "_" + w.cy.astype(str) + "_" + w.hs.astype(str)
calm = w[w.W <= CALM_W]
base = calm.groupby("cell").agg(base=("ay_res", "median"), n_base=("ay_res", "size"), base_drives=("drive", "nunique"))
w = w.merge(base, on="cell", how="left")
w["d_ay"] = w.ay_res - w.base
w["Wl_bin"] = pd.cut(w.W_left, [-15, -8, -5, -3, -1, 1, 3, 5, 8, 15])

def tab(x, col):
    t = x.groupby("Wl_bin", observed=True).agg(n=(col, "size"), drives=("drive", "nunique"), median=(col, "median"),
                                              mean=(col, "mean"))
    return {str(k): {kk: round(float(vv), 4) for kk, vv in r.items()} for k, r in t.iterrows()}

out["raw_ay_by_Wleft"] = tab(w, "ay_res")
# 基準の窓を自分自身で作っていると差が 0 に寄るので、弱風の窓は自分の窓を除いた基準 (leave-one-drive-out) にする
def lodo_base(row):
    c = calm[(calm.cell == row.cell) & (calm.drive != row.drive)]
    return c.ay_res.median() if len(c) else np.nan
w["base_lodo"] = w.apply(lodo_base, axis=1)
w["d_ay_lodo"] = w.ay_res - w.base_lodo
s = w.dropna(subset=["d_ay_lodo"])
out["n_with_site_baseline"] = len(s); out["n_drives_with_site_baseline"] = int(s.drive.nunique())
out["site_diff_ay_by_Wleft"] = tab(s, "d_ay_lodo")

# 回帰 (差 ~ W_left, 差 ~ v·W_left)。ドライブ単位のブートストラップ
def slope(x, col, xcol):
    X = np.c_[np.ones(len(x)), x[xcol]]; return np.linalg.lstsq(X, x[col], rcond=None)[0][1]
s = s.assign(vW=s.v * s.W_left / 25.0)          # 車速 25 m/s で W_left と同じ目盛り
rng = np.random.default_rng(0); drv = s.drive.unique(); bs = {"W_left": [], "vW": []}
for _ in range(2000):
    pick = rng.choice(drv, len(drv)); b = pd.concat([s[s.drive == d_] for d_ in pick])
    for xc in bs: bs[xc].append(slope(b, "d_ay_lodo", xc))
for xc in bs:
    out[f"slope_d_ay_per_{xc}"] = dict(est=slope(s, "d_ay_lodo", xc), ci95=list(np.percentile(bs[xc], [2.5, 97.5])),
                                       corr=float(np.corrcoef(s[xc], s.d_ay_lodo)[0, 1]))
# 符号の検査: 同じ升目を逆向きに走った窓は W_left の符号が逆になる → 差の符号も逆になるはず
# 強風のドライブの中で、突風 ≥ 12.5 の窓とそれ以外
wd = w[w.reason == "windy_drive"]
out["windy_drive_gust_split"] = {k: dict(n=int(len(g)), median_ay=float(g.ay_res.median()), median_abs_Wleft=float(g.W_left.abs().median()),
                                        corr_ay_Wleft=float(np.corrcoef(g.ay_res, g.W_left)[0, 1]) if len(g) > 10 else None)
                                 for k, g in wd.groupby(wd.Gst >= 12.5)}
# 窓の値の散らばり (弱風) = 1 窓で検出するときの雑音
c2 = s[s.W <= CALM_W]
out["calm_window_noise_ay"] = dict(raw_std=float(calm.ay_res.std()), site_diff_std=float(c2.d_ay_lodo.std()),
                                   site_diff_mad_sigma=float(1.4826 * (c2.d_ay_lodo - c2.d_ay_lodo.median()).abs().median()),
                                   n=len(c2))
w.to_csv(O / "windows_scored.csv", index=False, float_format="%.6g")
(O / "feasibility.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print(json.dumps(out, indent=1, ensure_ascii=False, default=lambda z: round(float(z), 4)))
