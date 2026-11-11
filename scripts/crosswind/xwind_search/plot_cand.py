"""候補の ±W s の時系列。各段に、同じ地点を通った他ドライブ (同じ車種) の 10〜90% の幅と中央値を重ねる。
使い方: plot_cand.py <out> <候補の csv (car, drive, t, png 列)> [half_window_s]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path.cwd() / "scripts/vibration")); 
try:
    import validate_sites_with_fullcar as V; V.configure_japanese_font()
except Exception: pass
OUT, LIST = Path(sys.argv[1]), Path(sys.argv[2])
HW = float(sys.argv[3]) if len(sys.argv) > 3 else 12
_F = {}
def plot_one(car, drive, t0, png):
  if car not in _F: _F[car] = pd.read_csv(OUT / f"feat_{car}.csv.gz")
  f = _F[car]
  lat0 = 37.65; kx, ky = 111320 * np.cos(np.radians(lat0)), 110540
  x = f[(f.drive == drive) & f.t.between(t0 - HW, t0 + HW)].sort_values("t")
  o = f[(f.drive != drive) & (f.v > 10) & f.lat.notna()].reset_index(drop=True)
  tree = cKDTree(np.c_[o.lon * kx, o.lat * ky])
  q = x.iloc[::5]
  band = {c: [] for c in ("yres1", "ayres1", "beta1", "steer", "yaw")}
  for _, r in q.iterrows():
      if not np.isfinite(r.lat):
          for c in band: band[c].append((np.nan,) * 3); continue
      nb = np.asarray(tree.query_ball_point([r.lon * kx, r.lat * ky], 20.0), dtype=int)
      nb = nb[np.abs((o.H.to_numpy()[nb] - r.H + 180) % 360 - 180) < 20]
      for c in band:
          v = pd.Series(o[c].to_numpy()[nb]).groupby(o.drive.to_numpy()[nb]).median()
          band[c].append(tuple(np.nanpercentile(v, [10, 50, 90])) if len(v) >= 3 else (np.nan,) * 3)
  nd = None
  tt = q.t - t0
  fig, ax = plt.subplots(5, 1, figsize=(9, 10), sharex=True)
  def bandplot(a, c, scale=1.0):
      b = np.array(band[c]) * scale
      a.fill_between(tt, b[:, 0], b[:, 2], color="0.85", lw=0, label="他ドライブ 10–90%")
      a.plot(tt, b[:, 1], color="0.5", lw=1, label="他ドライブ 中央値")
  X = x.t - t0
  bandplot(ax[0], "steer"); ax[0].plot(X, x.steer, "k", lw=1.2, label="舵角"); ax[0].set_ylabel("舵角 [deg]\n(左+)")
  bandplot(ax[1], "yaw"); ax[1].plot(X, x.yaw, "k", lw=1.2, label="ヨーレート 実測"); ax[1].plot(X, x.r_pred_lag, color="#e8740c", lw=1.2, label="舵角からの予測")
  ax[1].set_ylabel("ヨーレート\n[deg/s]")
  bandplot(ax[2], "yres1"); ax[2].plot(X, x.yres1, "k", lw=1.4); ax[2].set_ylabel("ヨー残差\nv·Δr [m/s²]")
  bandplot(ax[3], "ayres1"); ax[3].plot(X, x.ayres1, "k", lw=1.4); ax[3].set_ylabel("横加速度残差\nay−v·r [m/s²]")
  bandplot(ax[4], "beta1"); ax[4].plot(X, x.beta1, "k", lw=1.4); ax[4].set_ylabel("β [deg]")
  for a in ax: a.axvline(0, color="r", lw=0.6, ls=":"); a.grid(alpha=0.3); a.axhline(0, color="0.7", lw=0.6)
  ax[0].legend(fontsize=7, loc="upper left", ncol=3); ax[1].legend(fontsize=7, loc="upper left", ncol=4)
  r = x.iloc[np.argmin(np.abs(x.t - t0))]
  ax[0].set_title(f"{car} {drive[-20:]} seg{int(r.seg)} t={t0:.1f}  v={r.v:.1f} m/s  方位 {r.H:.0f}°  "
                  f"SFO 風 {r.W:.1f} m/s from {r.D:.0f}° (突風 {r.G:.1f}) 横成分(左+) {r.W_left:.1f}  {r.lat:.5f},{r.lon:.5f}", fontsize=8)
  ax[-1].set_xlabel("時刻 [s] (候補の山 = 0)")
  fig.tight_layout(); png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(png, dpi=90)

for _, r in pd.read_csv(LIST).iterrows():
    plot_one(r.car, r.drive, float(r.t), Path(r.png)); print(r.png)
