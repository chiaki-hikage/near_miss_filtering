"""強風イベントの前後 ±8 s の時系列図。符号は「風向と整合する外乱」が正になるようにそろえる。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
O = Path(sys.argv[1]); S = 8.0
meta = json.load(open(O / "rates.json")); sr, sb = meta["sig_ay_res_1s"], meta["sig_beta_1s"]
ev = pd.read_csv(O / "events.csv"); ev = ev[ev.set.str.startswith("strong")].sort_values("z_comb", ascending=False).reset_index(drop=True)
smp = pd.read_csv(O / "samples.csv.gz")
(O / "plots").mkdir(exist_ok=True)
for i, e in ev.iterrows():
    x = smp[(smp.drive == e.drive) & (smp.t >= e.t - S) & (smp.t <= e.t + S)].sort_values("t")
    s = np.sign(e.W_left) * (1 if e.set == "strong_consistent" else -1)
    tt = x.t - e.t
    f, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(tt, -s * x.ay_res_1s_site / sr, color="#1f6fb4", label="Δr（v·Δr, 1 s 平均, 場所の基準を引いた後）")
    ax[0].plot(tt, -s * x.beta_d_1s_site / sb, color="#d4571b", label="β（1 s 平均, 場所の基準を引いた後）")
    ax[0].axhline(3, color="0.6", lw=0.8, ls="--"); ax[0].axhline(0, color="0.8", lw=0.8)
    ax[0].set_ylabel("z（想定の向きが正）"); ax[0].legend(fontsize=8, loc="upper left")
    ax[1].plot(tt, x.steer, color="0.3"); ax[1].set_ylabel("舵角 [deg]（左が正）")
    ax[2].plot(tt, x.yaw_dps, color="#1f6fb4", label="実測 r"); ax[2].plot(tt, x.pred, color="0.5", ls="--", label="舵角からの予測 r_pred")
    ax[2].set_ylabel("ヨーレート [deg/s]（左が正）"); ax[2].legend(fontsize=8, loc="upper left"); ax[2].set_xlabel("山からの時刻 [s]")
    for a in ax: a.axvline(0, color="k", lw=0.6)
    side = "左" if e.W_left > 0 else "右"
    f.suptitle(f"#{i+1} {e.set}  {e.drive[-20:]} seg {e.seg}  v={e.v:.1f} m/s\n"
               f"風 {e.W:.1f} m/s・突風 {e.G:.1f} m/s・{side}から {abs(e.W_left):.1f} m/s  z={e.z_comb:.2f}", fontsize=10)
    f.tight_layout(); f.savefig(O / "plots" / f"event{i+1:02d}_{e.set}_{e.drive[-20:].replace('|','_')}_{e.seg}.png", dpi=110); plt.close(f)
print(len(ev), "plots")
