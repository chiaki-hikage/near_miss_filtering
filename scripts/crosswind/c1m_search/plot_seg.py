"""セグメント全体の時系列 (速度, 進路変化率と車体ヨーレート, βc, dβ/dt, 横ずれ)。橋の上を塗る。
使い方: plot_seg.py <out> <segment_id 先頭> <png> [タイトル]"""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "scripts/vibration")
import validate_sites_with_fullcar as V; V.configure_japanese_font()
OUT, sid, png = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); title = sys.argv[4] if len(sys.argv) > 4 else sid
fit = json.load(open(OUT / "beta_fit.json"))
f = glob.glob(str(OUT / "sig" / f"{sid}*.csv.gz"))[0]; d = pd.read_csv(f)
rr = np.radians(d.crs_rate); vv = d.v.clip(lower=3)
d["bc"] = (d.beta1 - (fit["a_deg_per_radpm"] * rr / vv + fit["b"] * vv * rr + fit["c"])).where(d.v > 10)
t = d.t - d.t.iloc[0]
fig, ax = plt.subplots(5, 1, figsize=(10, 9), sharex=True)
ax[0].plot(t, d.v * 3.6, "k"); ax[0].set_ylabel("速度 [km/h]")
ax[1].plot(t, d.crs_rate, "k", lw=1, label="進路の変化率"); ax[1].plot(t, d.r_body, color="#e8740c", lw=1, label="車体のヨーレート"); ax[1].set_ylabel("[deg/s]"); ax[1].legend(fontsize=8)
ax[2].plot(t, d.bc, "k"); ax[2].set_ylabel("β (幾何の分を\n除いたもの) [deg]")
ax[3].plot(t, d.dbeta.rolling(10, center=True).mean(), "k", lw=0.8); ax[3].set_ylabel("dβ/dt [deg/s]")
ax[4].plot(t, d.xtrk, "k"); ax[4].set_ylabel("横ずれ [m]\n(±4 s の 2 次式から)")
onb = d.on_bridge.to_numpy()
for a in ax:
    a.grid(alpha=0.3)
    if onb.any():
        idx = np.flatnonzero(onb); runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for rn in runs: a.axvspan(t.iloc[rn[0]], t.iloc[rn[-1]], color="#cfe3f5", zorder=0)
ax[0].set_title(title + "  (青 = 橋の上)", fontsize=10); ax[-1].set_xlabel("セグメント先頭からの時刻 [s]  (t0 = %.1f)" % d.t.iloc[0])
fig.tight_layout(); png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(png, dpi=85); print(png)
