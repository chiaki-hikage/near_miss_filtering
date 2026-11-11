"""風を受けやすい区間を通るセグメントについて、横滑り角 β と横ずれを作り、橋の出入りでの変化を測る。

使う列 (verify_cols.py でデータにより確認):
  0:3 ECEF 位置, 3:7 姿勢の四元数 [w,x,y,z] (車体→ECEF), 7:10 ECEF 速度, 10:13 車体角速度 [rad/s]
  (19:22 の加速度は v·r と相関 −0.999 で、独立した観測にならないので使わない)
信号 (20 Hz):
  beta  = 進路 − 車体の向き [deg] (左正)。セグメント内の直進 (|進路変化率|<1 deg/s, v>15) の中央値を引く
  dbeta = 進路変化率 − 車体ヨーレート [deg/s] (= dβ/dt)
  xtrk  = 横ずれ [m]: 位置を ±4 s の 2 次式で当てはめた残り (進行方向に直交する成分)
指標 (橋の通過ごと。通過 = 橋の線から 15 m 以内・向きの差 30 度以内が 2 s 以上続く区間):
  step   = 橋の上の β の中央値 − 手前 8 s の β の中央値
  tr_in, tr_out = 入口・出口 ±3 s の |β1 − 6 s 中央値| の最大
  xt_pp  = 入口・出口 ±3 s の横ずれの幅
対照 (null): 全セグメントの、橋から 100 m 以上離れた時刻で同じ指標を 2 s おきに計算した分布。各指標をその分位で表す。
出力: <out>/passages.csv, <out>/null_stats.csv, <out>/sig/<seg>.csv.gz"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, requests
from safetensors.numpy import load_file
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import butter, sosfiltfilt
sys.path.insert(0, "src")
from near_miss.io import comma1m
OUT = Path(sys.argv[1]); (OUT / "sig").mkdir(parents=True, exist_ok=True)
P = dict(BRIDGE_NEAR_M=15, MIN_PASS_S=2.0, PRE_S=8.0, TR_HALF_S=3.0, V_MIN=15.0, NULL_STEP_S=2.0, NULL_FAR_M=100)
ex = pd.read_csv(OUT / "seg_exposure.csv")
els = json.load(open(OUT / "osm_features.json"))
target = ex[(ex.bridge_dist_m < 60) | (ex.tunnel_dist_m < 60) | (ex.coast_dist_m < 500)]
print("対象セグメント", len(target))
sess = requests.Session()
def enu_R(p):
    x, y, z = p; lon = np.arctan2(y, x); lat = np.arctan2(z, np.hypot(x, y))
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])
def ecef2ll(x, y, z):
    from pymap3d import ecef2geodetic
    la, lo, _ = ecef2geodetic(x, y, z); return np.asarray(la), np.asarray(lo)
lp = lambda x, hz, fs: sosfiltfilt(butter(2, hz, fs=fs, output="sos"), x)
FS = 20.0
feats = []
for e in els.values():
    g = e.get("geometry"); tg = e["tags"]
    if not g or tg.get("natural") == "coastline": continue
    feats.append(dict(id=e["id"], kind="bridge" if tg.get("bridge") else "tunnel", la=np.array([q["lat"] for q in g]),
                      lo=np.array([q["lon"] for q in g]), name=tg.get("bridge:name") or tg.get("name", ""), ref=tg.get("ref", "")))
def near_mask(la, lo, H, f, k):
    """各点が要素 f の線から P 以内で、向きが揃っているか。距離も返す。"""
    X, Y = f["lo"] * k, f["la"] * 110540; px, py = lo * k, la * 110540
    dmin = np.full(len(la), np.inf); brg = np.zeros(len(la))
    for j in range(len(X) - 1):
        dx, dy = X[j + 1] - X[j], Y[j + 1] - Y[j]; L2 = dx * dx + dy * dy
        if L2 == 0: continue
        t = np.clip(((px - X[j]) * dx + (py - Y[j]) * dy) / L2, 0, 1)
        d = np.hypot(px - X[j] - t * dx, py - Y[j] - t * dy)
        b = d < dmin; dmin[b] = d[b]; brg[b] = np.degrees(np.arctan2(dx, dy)) % 360
    dh = np.abs((brg - H + 180) % 360 - 180); dh = np.minimum(dh, 180 - dh)
    return (dmin < P["BRIDGE_NEAR_M"]) & (dh < 30), dmin
passages, nulls = [], []
for _, r in target.iterrows():
    sid = r.segment_id
    try:
        path = comma1m.ensure_file(sess, sid)
        S = load_file(str(path))["states"]; t = load_file(str(path))["t"]
    except Exception as e:
        print("skip", sid, e); continue
    ok = np.isfinite(S).all(1) & (np.linalg.norm(S[:, 3:7], axis=1) > 0.5)
    S, t = S[ok], t[ok]
    if len(S) < 2000: continue
    E = enu_R(S[len(S) // 2, 0:3]); v = S[:, 7:10] @ E.T; sp = np.hypot(v[:, 0], v[:, 1])
    fwd = Rot.from_quat(S[:, [4, 5, 6, 3]]).apply([1.0, 0, 0]) @ E.T
    w = S[:, 10:13]; up_b = Rot.from_quat(S[:, [4, 5, 6, 3]]).inv().apply(np.tile(E[2], (len(S), 1)))
    r_body = np.degrees((w * up_b).sum(1))
    crs = np.unwrap(np.arctan2(v[:, 1], v[:, 0])); hdg = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    beta = np.degrees(np.angle(np.exp(1j * (crs - hdg))))
    # 20 Hz へ (100 Hz を 0.1 s 平均してから間引き)
    tg_ = np.arange(t[0] + 0.5, t[-1] - 0.5, 1 / FS)
    sm = lambda x: np.interp(tg_, t, pd.Series(x).rolling(10, center=True, min_periods=1).mean().to_numpy())
    la, lo = ecef2ll(*S[:, 0:3].T)
    d = pd.DataFrame(dict(t=tg_, v=sm(sp), beta_raw=sm(beta), r_body=sm(r_body),
                          crs_rate=np.gradient(lp(np.degrees(np.interp(tg_, t, crs)), 1.0, FS), 1 / FS),
                          lat=np.interp(tg_, t, la), lon=np.interp(tg_, t, lo),
                          e=np.interp(tg_, t, (S[:, 0:3] @ E.T)[:, 0]), n=np.interp(tg_, t, (S[:, 0:3] @ E.T)[:, 1])))
    d.loc[d.v < 3, "beta_raw"] = np.nan
    d["H"] = (90 - np.degrees(np.interp(tg_, t, crs))) % 360
    st = (d.v > P["V_MIN"]) & (d.crs_rate.abs() < 1)
    if st.sum() < 100: continue
    d["beta"] = d.beta_raw - d.beta_raw[st].median()
    d["beta1"] = d.beta.rolling(int(FS), center=True, min_periods=10).mean()
    d["beta_dev"] = d.beta1 - d.beta1.rolling(int(6 * FS), center=True, min_periods=40).median()
    d["dbeta"] = d.crs_rate - d.r_body
    # 横ずれ: ±4 s の 2 次式からの残り (進行方向に直交)
    xt = np.full(len(d), np.nan); hw = int(4 * FS)
    E_, N_ = d.e.to_numpy(), d.n.to_numpy()
    for i in range(hw, len(d) - hw, 2):
        sl = slice(i - hw, i + hw + 1); ex_, ny_ = E_[sl] - E_[i], N_[sl] - N_[i]
        h = np.arctan2(ny_[-1] - ny_[0], ex_[-1] - ex_[0]); u = ex_ * np.cos(h) + ny_ * np.sin(h); wv = -ex_ * np.sin(h) + ny_ * np.cos(h)
        if np.ptp(u) < 50: continue
        c = np.polyfit(u, wv, 2); xt[i] = wv[hw] - c[2]
    d["xtrk"] = pd.Series(xt).interpolate(limit=3).to_numpy()
    k = 111320 * np.cos(np.radians(r.lat))
    near_any = np.full(len(d), np.inf); onb = np.zeros(len(d), bool)
    segp = []
    for f in feats:
        if abs(f["la"].mean() - r.lat) > 0.05 or abs(f["lo"].mean() - r.lon) > 0.07: continue
        m, dm = near_mask(d.lat.to_numpy(), d.lon.to_numpy(), d.H.to_numpy(), f, k)
        near_any = np.minimum(near_any, dm)
        if m.sum() < P["MIN_PASS_S"] * FS: continue
        idx = np.flatnonzero(m)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 5) + 1)
        for run in runs:
            if len(run) < P["MIN_PASS_S"] * FS: continue
            i0, i1 = run[0], run[-1]; onb[i0:i1 + 1] |= f["kind"] == "bridge"
            segp.append((f, i0, i1))
    d["on_bridge"] = onb; d["near_m"] = near_any
    d.to_csv(OUT / "sig" / f"{sid}.csv.gz", index=False, float_format="%.6g")
    def metrics(i0, i1):
        pre = d.beta1.iloc[max(0, i0 - int(P["PRE_S"] * FS)):i0]; on = d.beta1.iloc[i0:i1 + 1]
        hw_ = int(P["TR_HALF_S"] * FS)
        a = slice(max(0, i0 - hw_), i0 + hw_); b = slice(max(0, i1 - hw_), i1 + hw_)
        return dict(step=on.median() - pre.median(), tr_in=d.beta_dev.iloc[a].abs().max(), tr_out=d.beta_dev.iloc[b].abs().max(),
                    xt_in=np.ptp(d.xtrk.iloc[a].dropna()) if d.xtrk.iloc[a].notna().sum() > 5 else np.nan,
                    xt_out=np.ptp(d.xtrk.iloc[b].dropna()) if d.xtrk.iloc[b].notna().sum() > 5 else np.nan,
                    dbeta_max=d.dbeta.iloc[min(a.start, b.start):b.stop].abs().max(),
                    crs_rate_max=d.crs_rate.iloc[min(a.start, b.start):b.stop].abs().max(), v=d.v.iloc[i0:i1 + 1].mean())
    for f, i0, i1 in segp:
        passages.append(dict(segment_id=sid, kind=f["kind"], osm_id=f["id"], name=f["name"], ref=f["ref"], t_in=d.t.iloc[i0], t_out=d.t.iloc[i1],
                             dur_s=(i1 - i0) / FS, lat=d.lat.iloc[i0], lon=d.lon.iloc[i0], H=d.H.iloc[i0], country=r.country, admin1=r.admin1,
                             place=r.place, coast_dist_m=r.coast_dist_m, **metrics(i0, i1)))
    # 対照: 橋・トンネルから離れた時刻で、長さ 4 s の仮の通過
    L = int(4 * FS)
    for i0 in range(int(P["PRE_S"] * FS), len(d) - L - int(P["TR_HALF_S"] * FS), int(P["NULL_STEP_S"] * FS)):
        if near_any[max(0, i0 - 60):i0 + L + 60].min() < P["NULL_FAR_M"] or d.v.iloc[i0] < P["V_MIN"]: continue
        nulls.append(dict(segment_id=sid, t=d.t.iloc[i0], coast_dist_m=r.coast_dist_m, **metrics(i0, i0 + L)))
    print(sid[:8], "passages", len(segp))
pd.DataFrame(passages).to_csv(OUT / "passages.csv", index=False, float_format="%.6g")
pd.DataFrame(nulls).to_csv(OUT / "null_stats.csv", index=False, float_format="%.6g")
json.dump(P, open(OUT / "seg_signals_params.json", "w"))
print("通過", len(passages), "対照", len(nulls))
