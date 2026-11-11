"""候補の点数付け (順位付けの基準を先に決めて、全候補に同じ式を当てる)。
  S_beh  = Σ min(|z|, 6) [外乱の向き pol と同符号のもの]。重み: yres 1, ayres 1, beta 0.5
           (β は姿勢推定に由来し、単独で大きい値が出やすいので半分)
           ±3 s の |r| か |r_pred| が 3 deg/s を超える (カーブ・車線変更中) ときは ×0.5
  S_uniq = 1 − (同じ地点の他ドライブで、同じ向きに同じ大きさ以上が出た割合)。他ドライブが 5 本未満なら 0.5
  S_wind = clip(pol·W_left / 8 m/s, 0, 1)   (SFO の風の横成分が外乱の向きと合うほど大きい)
  修正操舵: 山の後 0.3〜2.5 s の舵角 − 山の前 1〜3 s の舵角の中央値。外乱を戻す向き (左に押されたら右へ) を + とした値 cs
  横方向のずれ lat_disp: ±5 s の位置を 2 次式で当てはめた残りの横ずれの幅 [m] (車線変更なら 3 m 前後)
  score = S_beh × S_uniq × S_wind
出力: <out>/cands_scored.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(sys.argv[1])
c = pd.read_csv(OUT / "cands_ranked.csv")
W8 = {"yres": 1.0, "ayres": 1.0, "beta": 0.5}
def sbeh(r):
    s = sum(w * min(abs(r["z_" + k]), 6) for k, w in W8.items() if np.sign(r["z_" + k]) == r.pol)
    return s * (0.5 if (r.yaw_absmax3 > 3 or r.rpred_absmax3 > 3) else 1.0)
c["S_beh"] = c.apply(sbeh, axis=1)
c["S_uniq"] = np.where(c.n_other >= 5, 1 - c.frac_other_ge.fillna(0.5), 0.5)
c["S_wind"] = np.clip(c.pol * c.W_left / 8.0, 0, 1).fillna(0)
c["score"] = c.S_beh * c.S_uniq * c.S_wind
lat0 = 37.65; kx, ky = 111320 * np.cos(np.radians(lat0)), 110540
cs, ld = [], []
for car, cc in c.groupby("car"):
    f = pd.read_csv(OUT / f"feat_{car}.csv.gz", usecols=["drive", "seg", "t", "lat", "lon", "steer"])
    for i, r in cc.iterrows():
        x = f[(f.drive == r.drive) & (f.t.between(r.t - 6, r.t + 6))]
        pre = x[x.t.between(r.t - 3, r.t - 1)].steer.median(); post = x[x.t.between(r.t + 0.3, r.t + 2.5)].steer
        dv = (post - pre).to_numpy()
        # 外乱を戻す向き: pol=+ (右へ押された) なら左 (+) に切る
        cs.append(float(np.nanmax(r.pol * dv)) if len(dv) else np.nan)
        y = x[x.t.between(r.t - 5, r.t + 5)].dropna(subset=["lat"])
        if len(y) > 50:
            X, Y = y.lon * kx, y.lat * ky; X, Y = X - X.mean(), Y - Y.mean()
            ang = np.arctan2(Y.iloc[-1] - Y.iloc[0], X.iloc[-1] - X.iloc[0])
            u = X * np.cos(ang) + Y * np.sin(ang); w = -X * np.sin(ang) + Y * np.cos(ang)
            res = w - np.polyval(np.polyfit(u, w, 2), u)
            ld.append(float(res.max() - res.min()))
        else: ld.append(np.nan)
        c.loc[i, "cs"] = cs[-1]; c.loc[i, "lat_disp"] = ld[-1]
c = c.sort_values("score", ascending=False)
c.to_csv(OUT / "cands_scored.csv", index=False, float_format="%.6g")
pd.set_option("display.width", 250)
cols = ["car", "drive", "seg", "t", "v", "z_yres", "z_ayres", "z_beta", "pol", "S_beh", "S_uniq", "n_other", "W", "G", "D", "H", "W_left", "score", "cs", "lat_disp", "yaw_absmax3", "osm_bridge"]
x = c[cols].copy(); x["drive"] = x.drive.str[-20:]
print(x.head(30).round(2).to_string())
# 風を見ない並び (挙動 × 一意性) の上位と、その風
c["score_nowind"] = c.S_beh * c.S_uniq
print(x.assign(sn=c.score_nowind).sort_values("sn", ascending=False).head(15).round(2).to_string())
