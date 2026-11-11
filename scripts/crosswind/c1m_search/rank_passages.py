"""橋・トンネルの通過を、対照 (橋から離れた時刻) の分布に対する分位で順位付けする。
  p_step = |step| の対照での分位、p_tr = max(tr_in, tr_out) の分位、p_xt = max(xt_in, xt_out) の分位
  曲線の補正: 通過区間の |進路変化率| の最大が 3 deg/s を超えるものは、幾何の影響を受けやすいので印を付ける
  score = mean(p_step, p_tr, p_xt)  (どれか 1 つでなく、複数が同時に大きいものを上に)
出力: <out>/passages_ranked.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(sys.argv[1])
p = pd.read_csv(OUT / "passages.csv"); n = pd.read_csv(OUT / "null_stats.csv")
n = n[n.crs_rate_max < 3]
def pct(x, ref): ref = np.sort(ref.dropna().to_numpy()); return np.searchsorted(ref, x) / len(ref)
p["tr"] = p[["tr_in", "tr_out"]].max(1); p["xt"] = p[["xt_in", "xt_out"]].max(1)
n["tr"] = n[["tr_in", "tr_out"]].max(1); n["xt"] = n[["xt_in", "xt_out"]].max(1)
p["p_step"] = pct(p.step.abs(), n.step.abs()); p["p_tr"] = pct(p.tr, n.tr); p["p_xt"] = pct(p.xt, n.xt)
p["curvy"] = p.crs_rate_max > 3
p["score"] = p[["p_step", "p_tr", "p_xt"]].mean(1)
p = p.sort_values("score", ascending=False); p.to_csv(OUT / "passages_ranked.csv", index=False, float_format="%.5g")
pd.set_option("display.width", 250)
print("対照の分位 (95%):", {c: round(float(n[c].abs().quantile(0.95)), 3) for c in ("step", "tr", "xt")})
c = ["segment_id", "kind", "name", "ref", "place", "admin1", "dur_s", "v", "step", "tr", "xt", "dbeta_max", "crs_rate_max", "p_step", "p_tr", "p_xt", "score", "curvy", "coast_dist_m"]
x = p[c].copy(); x["segment_id"] = x.segment_id.str[:8]; x["name"] = x.name.astype(str).str[:22]
print(x.head(25).round(3).to_string())
