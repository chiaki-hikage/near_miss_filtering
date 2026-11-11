"""comma1M localizer の公開仕様に無い列 (姿勢・角速度・加速度) の意味を、データで検証する。
仮説 (openpilot の loc_kf の並び): 3:7 = ECEF→? 姿勢の四元数 [w,x,y,z]、10:13 = 車体角速度 [rad/s]、19:22 = 車体加速度 [m/s²]
検証:
  姿勢   R(q) で車体前方 [1,0,0] を ECEF に回し、水平面で速度の向きとの cos (Hamilton と共役の両方を試す)
  角速度 車体の上下軸成分 vs 速度方位の変化率 (相関・傾き)
  加速度 前後成分 vs dv/dt、横成分 vs v·(進路変化率)
出力: <out>/verify_cols.csv, 標準出力に要約"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
from safetensors.numpy import load_file
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import butter, sosfiltfilt
OUT = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
FS = 100.0
def enu_R(p):
    x, y, z = p; lon = np.arctan2(y, x); lat = np.arctan2(z, np.hypot(x, y))
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])
lp = lambda x, hz: sosfiltfilt(butter(2, hz, fs=FS, output="sos"), x)
rows = []
fs = [f for f in sorted(glob.glob("raw_data/comma1M/*.safetensors")) if "frame_info" not in f]
rng = np.random.default_rng(0)
for f in rng.choice(fs, size=min(N, len(fs)), replace=False):
    S = load_file(f)["states"]
    E = enu_R(S[len(S) // 2, 0:3]); v = S[:, 7:10] @ E.T; sp = np.hypot(v[:, 0], v[:, 1])
    mv = sp > 10
    if mv.sum() < 1000: continue
    q = S[:, 3:7]
    r = {"seg": Path(f).stem[:8], "n_moving": int(mv.sum())}
    for name, Rq in (("hamilton", Rot.from_quat(q[:, [1, 2, 3, 0]])), ("conj", Rot.from_quat(q[:, [1, 2, 3, 0]]).inv())):
        fwd = Rq.apply([1.0, 0, 0]) @ E.T
        c = (fwd[mv, 0] * v[mv, 0] + fwd[mv, 1] * v[mv, 1]) / (np.hypot(fwd[mv, 0], fwd[mv, 1]) * sp[mv])
        r[f"cos_{name}"] = float(np.median(c))
    Rb = Rot.from_quat(q[:, [1, 2, 3, 0]])                 # 車体 → ECEF (検証で採用されたほう)
    up_body = Rb.inv().apply(E[2])                           # ENU の上向きを車体座標で
    w = S[:, 10:13]; yaw_body = -(w * up_body).sum(1) * -1   # 上向き軸まわり (左回り正)
    yaw_body = (w * up_body).sum(1)
    crs = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))            # 東から反時計回り (左回り正)
    crs_rate = np.gradient(lp(crs, 2.0), 1 / FS)
    a = S[:, 19:22]
    dvdt = np.gradient(lp(sp, 2.0), 1 / FS)
    yb, cr = lp(yaw_body, 2.0)[mv], crs_rate[mv]
    r["yaw_corr"] = float(np.corrcoef(yb, cr)[0, 1]); r["yaw_slope"] = float(np.polyfit(cr, yb, 1)[0])
    for i, ax in enumerate("xyz"):
        ai = lp(a[:, i], 2.0)[mv]
        r[f"a{ax}_corr_dvdt"] = float(np.corrcoef(ai, dvdt[mv])[0, 1])
        r[f"a{ax}_corr_vr"] = float(np.corrcoef(ai, (sp * crs_rate)[mv])[0, 1])
    r["ay_slope_vr"] = float(np.polyfit((sp * crs_rate)[mv], lp(a[:, 1], 2.0)[mv], 1)[0])
    r["ax_slope_dvdt"] = float(np.polyfit(dvdt[mv], lp(a[:, 0], 2.0)[mv], 1)[0])
    r["yaw_std_dps"] = float(np.degrees(cr.std()))
    rows.append(r)
d = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True); d.to_csv(OUT / "verify_cols.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 250); print(d.round(3).to_string()); print(d.drop(columns="seg").median().round(3))
