"""強風時間帯のイベント単位の探索。風向と整合する横外乱 (ヨー残差・横滑り角) の山を拾ってランキングする。

入力: <v2>/fit/grid_20hz.csv.gz, <v2>/beta/grid_beta.csv.gz, configs/crosswind/honda_civic.yaml,
      data/comma2k19_civic (global_pose), SFO の ASOS 1 分値・METAR
出力: <out>/samples.csv.gz (20 Hz, 走行中), events.csv, rates.json

符号: s = sign(W_left)。左から (s=+1) の突風では Δr<0, β<0, その後 舵角が増える (左へ当て舵)。
  D_r = −s·(v·Δr の 1 s 平均 − 場所の基準) / σ_r,  D_b = −s·(β の 1 s 平均 − 場所の基準) / σ_b
  場所の基準 = 同じ 50 m 升目 × 30° 向きの、別ドライブの弱風 (W ≤ 3.1) の標本の中央値。無ければ 0
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path.cwd() / "src"))
from near_miss.features import bicycle_yaw_rate

V, CIVIC, ROOT, ASOS, METAR, OUT = map(Path, sys.argv[1:7]); OUT.mkdir(parents=True, exist_ok=True)
# 第 7 引数 strict: 山の前後 3 s に |r|, |r_pred| < 1 deg/s を課し、しきい値を 2.5 にする (ほぼ直線の道路だけ)
STRICT = len(sys.argv) > 7 and sys.argv[7] == "strict"
FS = 20; KT = 0.514444; STRONG_G = 12.5; CALM_W = 3.1; Z_PEAK = 2.5 if STRICT else 3.0

def llh(x, y, z):
    a, e2 = 6378137.0, 6.69437999014e-3
    lon = np.arctan2(y, x); p = np.hypot(x, y); lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2); lat = np.arctan2(z + e2 * n * np.sin(lat), p)
    return np.degrees(lat), np.degrees(lon)

def enu_R(lat, lon):
    la, lo = np.radians(lat), np.radians(lon); sl, cl, so, co = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    return np.array([[-so, co, 0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])

def ma(x, n):
    return pd.Series(x).rolling(n, center=True, min_periods=n // 2).mean().to_numpy()

g = yaml.safe_load(CIVIC.read_text())["geometry"]
d = pd.read_csv(V / "fit/grid_20hz.csv.gz")
bt = pd.read_csv(V / "beta/grid_beta.csv.gz", usecols=["drive", "seg", "t", "beta"])
d = d.merge(bt, on=["drive", "seg", "t"], how="left")
d["pred"] = bicycle_yaw_rate(d.v.to_numpy(), d.steer.to_numpy(), g["wheelbase_m"], g["steer_ratio"],
                             g["understeer_gradient"], g["steer_offset_deg"])
d["res"] = d.yaw_dps - d.pred
d["ay_res"] = d.v * np.radians(d.res)

# 位置・進路・UTC を各標本に
parts = []
for (dr, sg), x in d.groupby(["drive", "seg"], sort=False):
    p = ROOT / dr / str(sg) / "global_pose"
    ft = np.load(p / "frame_times").ravel(); gt = np.load(p / "frame_gps_times")
    fp = np.load(p / "frame_positions"); fv = np.load(p / "frame_velocities")
    ok = np.isfinite(fp).all(1) & np.isfinite(fv).all(1) & np.isfinite(gt).all(1)
    x = x.copy()
    if ok.sum() < 100:
        for c in ("utc", "lat", "lon", "H"): x[c] = np.nan
        parts.append(x); continue
    ft, gt, fp, fv = ft[ok], gt[ok], fp[ok], fv[ok]
    utc = 315964800 + gt[:, 0] * 604800 + gt[:, 1] - 18
    lat, lon = llh(*fp.T); enu = fv @ enu_R(np.median(lat), np.median(lon)).T
    x["utc"] = x.t + np.median(utc - ft)
    x["lat"] = np.interp(x.t, ft, lat); x["lon"] = np.interp(x.t, ft, lon)
    ve, vn = np.interp(x.t, ft, enu[:, 0]), np.interp(x.t, ft, enu[:, 1])
    x["H"] = np.degrees(np.arctan2(ve, vn)) % 360
    parts.append(x)
d = pd.concat(parts, ignore_index=True)

# 風 (1 分値、無ければ METAR ±60 分)
a = pd.read_csv(ASOS); a["ts"] = pd.to_datetime(a["valid(UTC)"]).astype("int64") // 10**9
for c in ("sknt", "drct", "gust_sknt"): a[c] = pd.to_numeric(a[c], errors="coerce")
a = a.dropna(subset=["sknt", "drct"]).drop_duplicates("ts").set_index("ts")
m = pd.read_csv(METAR); m = m[m.station == "SFO"].copy(); m["ts"] = pd.to_datetime(m.valid).astype("int64") // 10**9
for c in ("sknt", "drct", "gust"): m[c] = pd.to_numeric(m[c], errors="coerce")
m = m.dropna(subset=["sknt", "drct"]).sort_values("ts")
mi = (d.utc // 60 * 60).fillna(-1).astype(np.int64)
hit = mi.isin(a.index).to_numpy()
d["Wm"] = np.where(hit, a.reindex(mi).sknt.to_numpy() * KT, np.nan)
d["Dm"] = np.where(hit, a.reindex(mi).drct.to_numpy(), np.nan)
d["Gm"] = np.where(hit, a.reindex(mi).gust_sknt.to_numpy() * KT, np.nan)
d["wind_src"] = np.where(hit, "asos1min", "")
mt = m.ts.to_numpy(); u = d.utc.fillna(0).to_numpy()
j = np.clip(np.searchsorted(mt, u), 1, len(m) - 1); jj = np.where(np.abs(mt[j] - u) < np.abs(mt[j - 1] - u), j, j - 1)
fill = (~hit) & (np.abs(mt[jj] - u) <= 3600) & d.utc.notna().to_numpy()
d.loc[fill, "Wm"] = m.sknt.to_numpy()[jj][fill] * KT; d.loc[fill, "Dm"] = m.drct.to_numpy()[jj][fill]
d.loc[fill, "Gm"] = m.gust.to_numpy()[jj][fill] * KT; d.loc[fill, "wind_src"] = "metar"
d["W_left"] = -d.Wm * np.sin(np.radians(d.Dm - d.H))

# 取付の向き (ドライブの直進時 β の中央値) を引く。1 s 平均
st = (d.v > 15) & (d.yaw_dps.abs() < 0.5)
d["beta_d"] = d.beta - d.drive.map(d[st].groupby("drive").beta.median())
key = ["drive", "seg"]
for c in ("ay_res", "beta_d", "steer", "yaw_dps", "pred"):
    d[c + "_1s"] = d.groupby(key)[c].transform(lambda s: ma(s.to_numpy(), FS))

# 場所の基準 (自分のドライブを除く弱風の中央値)
R = 6371e3
d["cell"] = ((np.radians(d.lon) * np.cos(np.radians(37.7)) * R // 50).astype("Int64").astype(str) + "_" +
             (np.radians(d.lat) * R // 50).astype("Int64").astype(str) + "_" + (d.H // 30).astype("Int64").astype(str))
run = (d.v > 15) & d.utc.notna()
calm = d[run & (d.Wm <= CALM_W)]
for c in ("ay_res_1s", "beta_d_1s"):
    tot = calm.groupby("cell")[c].median()
    per = calm.groupby(["cell", "drive"])[c].median().rename("own").reset_index()
    # 自分のドライブを除いた中央値: セル内の他ドライブの値の中央値で近似 (ドライブごとの中央値の中央値)
    def lodo(x):
        vals = x.own.to_numpy(); out = []
        for i in range(len(vals)):
            o = np.delete(vals, i); out.append(np.median(o) if len(o) else np.nan)
        return pd.Series(out, index=x.drive.to_numpy())
    lo = per.groupby("cell").apply(lodo).rename("b").reset_index()
    lo.columns = ["cell", "drive", "b"]
    d = d.merge(lo.rename(columns={"b": c + "_base_own"}), on=["cell", "drive"], how="left")
    allb = calm.groupby(["cell", "drive"])[c].median().groupby("cell").median().rename(c + "_base_all")
    d = d.merge(allb, left_on="cell", right_index=True, how="left")
    # 自分のドライブが弱風の基準に入っていない場合は全体の基準をそのまま使う
    d[c + "_base"] = d[c + "_base_own"].where(d[c + "_base_own"].notna() | d.drive.isin([]), d[c + "_base_all"])
    own_in = d.set_index(["cell", "drive"]).index.isin(per.set_index(["cell", "drive"]).index)
    d.loc[~own_in, c + "_base"] = d.loc[~own_in, c + "_base_all"]
    d[c + "_site"] = d[c] - d[c + "_base"].fillna(0)
d["has_base"] = d.ay_res_1s_base.notna() & d.beta_d_1s_base.notna()

# 雑音の大きさ (弱風・走行中・旋回でない標本のロバスト σ)
quiet = run & (d.Wm <= CALM_W) & (d.yaw_dps_1s.abs() < 3) & (d.pred_1s.abs() < 3)
def rsig(x): x = x.dropna(); return 1.4826 * np.median(np.abs(x - np.median(x)))
sig_r = rsig(d.loc[quiet, "ay_res_1s_site"]); sig_b = rsig(d.loc[quiet, "beta_d_1s_site"])
# 当て舵: 山の後 2 s の舵角の変化 (山の 0.5 s 前を基準)。雑音は弱風での同じ量の σ
d = d.sort_values(["drive", "seg", "t"]).reset_index(drop=True)
grp = d.groupby(key)
def shift(c, k): return grp[c].shift(-k)
d["steer_fwd2"] = pd.concat([shift("steer_1s", k) for k in range(0, 2 * FS + 1)], axis=1).sub(shift("steer_1s", -FS // 2), axis=0).abs().max(axis=1)
sig_s_samples = (pd.concat([shift("steer_1s", k) for k in range(0, 2 * FS + 1, 4)], axis=1).sub(shift("steer_1s", -FS // 2), axis=0))
d.to_csv(OUT / "samples.csv.gz", index=False, float_format="%.12g",
         columns=["drive", "seg", "t", "utc", "reason", "lat", "lon", "H", "v", "steer", "yaw_dps", "pred", "res", "ay_res",
                  "beta", "beta_d", "ay_res_1s_site", "beta_d_1s_site", "steer_1s", "yaw_dps_1s", "pred_1s",
                  "Wm", "Dm", "Gm", "W_left", "wind_src", "bias_interp", "has_base", "cell"])
meta = dict(sig_ay_res_1s=sig_r, sig_beta_1s=sig_b)

def detect(x, s):
    """x: 1 ドライブ・1 セグメントの連続標本。s: 符号の配列 (風向)。整合する山を拾う。"""
    zr = -s * x.ay_res_1s_site.to_numpy() / sig_r
    zb = -s * x.beta_d_1s_site.to_numpy() / sig_b
    if STRICT:
        turn = pd.Series(np.fmax(np.abs(x.yaw_dps_1s.to_numpy()), np.abs(x.pred_1s.to_numpy())))
        ok = (x.v.to_numpy() > 15) & (turn.rolling(6 * FS + 1, center=True, min_periods=1).max().to_numpy() < 1.0)
    else:
        ok = (x.v.to_numpy() > 15) & (np.abs(x.yaw_dps_1s.to_numpy()) < 3) & (np.abs(x.pred_1s.to_numpy()) < 3)
    comb = np.where(ok, np.fmin(zr, zb), np.nan)    # 両方が整合して大きいこと
    ev = []; i = 0; n = len(x); t = x.t.to_numpy()
    while i < n:
        if not (comb[i] >= Z_PEAK):
            i += 1; continue
        j = i
        while j + 1 < n and (comb[j + 1] >= Z_PEAK or (t[j + 1] - t[j] < 0.06 and np.nanmax(comb[j + 1:j + 2]) >= Z_PEAK * 0.5)):
            j += 1
        k = i + int(np.nanargmax(comb[i:j + 1]))
        ev.append((i, k, j)); i = j + int(3 * FS)
    return ev, zr, zb

def describe(x, ev, zr, zb, s):
    rows = []
    t = x.t.to_numpy(); st = x.steer_1s.to_numpy(); H = x.H.to_numpy(); v = x.v.to_numpy()
    for i, k, j in ev:
        w = lambda a, b: slice(max(0, k + int(a * FS)), min(len(x), k + int(b * FS)))
        sk = s[k]
        pk_r = np.nanmax(zr[w(-1, 1)]); pk_b = np.nanmax(zb[w(-1, 1)])
        tr = t[w(-1, 1)][np.nanargmax(zr[w(-1, 1)])] - t[k]; tb = t[w(-1, 1)][np.nanargmax(zb[w(-1, 1)])] - t[k]
        base = st[max(0, k - FS // 2)]
        corr = np.nanmax(sk * (st[w(0, 2)] - base)) if len(st[w(0, 2)]) else np.nan        # 風上への当て舵 [deg]
        pre = sk * (st[max(0, k - FS // 2)] - st[max(0, k - int(2.5 * FS))])               # 山の前の操舵 (風上が正)
        # 前後 3 s の横移動 (山の 3 s 前の進路に対する横ずれ)
        a0 = max(0, k - 3 * FS); a1 = min(len(x), k + 3 * FS)
        # 進路の向きから直線 (カーブで一定に曲がる分) を除いた残りで横移動を積分する。
        # 車線変更なら S 字に約 3.5 m ずれる。最大と最小の差を取る
        hh = np.unwrap(np.radians(H[a0:a1])); tt = t[a0:a1] - t[k]
        okh = np.isfinite(hh)
        if okh.sum() > 20:
            hd = hh - np.polyval(np.polyfit(tt[okh], hh[okh], 1), tt)
            y = np.nancumsum(v[a0:a1] * np.sin(hd)) / FS; lat_disp = float(np.nanmax(y) - np.nanmin(y))
        else:
            lat_disp = np.nan
        row = x.iloc[k]
        rows.append(dict(drive=row.drive, seg=int(row.seg), t=t[k], utc=row.utc, reason=row.reason, lat=row.lat, lon=row.lon,
                         v=row.v, W=row.Wm, D=row.Dm, G=row.Gm, W_left=row.W_left, wind_src=row.wind_src,
                         z_comb=np.nanmax(np.fmin(zr, zb)[i:j + 1]), z_r=pk_r, z_b=pk_b, dt_r=tr, dt_b=tb,
                         dur_s=(t[j] - t[i]) + 1 / FS,
                         ay_res_peak=float(x.ay_res_1s_site.iloc[k]), beta_peak_deg=float(x.beta_d_1s_site.iloc[k]),
                         steer_corr_deg=corr, steer_pre_deg=pre, lat_disp_m=lat_disp,
                         has_base=bool(row.has_base), bias_interp=float(row.bias_interp)))
    return rows

events = []; hours = {}
for (dr, sg), x in d.groupby(key, sort=False):
    x = x.reset_index(drop=True)
    if x.utc.isna().all() or x.W_left.isna().all(): continue
    strong = (x.Gm >= STRONG_G).to_numpy(); calm_m = (x.Wm <= CALM_W).to_numpy()
    # 同じ強風のドライブの、突風が 12.5 m/s 未満の時間 (道路は同じで風だけ弱い対照)
    mid = ((x.reason == "windy_drive") & (x.Gm < STRONG_G)).to_numpy()
    sgn = np.sign(x.W_left.fillna(0).to_numpy())
    for label, mask, s in (("strong_consistent", strong, sgn), ("strong_opposite", strong, -sgn),
                           ("calm_pos", calm_m, np.ones(len(x))), ("calm_neg", calm_m, -np.ones(len(x))),
                           ("windydrive_notstrong_consistent", mid, sgn), ("windydrive_notstrong_opposite", mid, -sgn)):
        if mask.sum() == 0: continue
        xx = x.copy(); ss = np.where(mask, s, np.nan)
        for c in ("ay_res_1s_site", "beta_d_1s_site"): xx.loc[~mask, c] = np.nan
        ev, zr, zb = detect(xx, ss)
        for r in describe(xx, ev, zr, zb, ss): r["set"] = label; events.append(r)
        if STRICT:
            turn = pd.Series(np.fmax(xx.yaw_dps_1s.abs().to_numpy(), xx.pred_1s.abs().to_numpy()))
            elig = (xx.v.to_numpy() > 15) & (turn.rolling(6 * FS + 1, center=True, min_periods=1).max().to_numpy() < 1.0)
        else:
            elig = (xx.v > 15).to_numpy()
        hours[label] = hours.get(label, 0) + (elig & mask).sum() / FS / 3600
ev = pd.DataFrame(events)
ev.to_csv(OUT / "events.csv", index=False, float_format="%.12g")
rates = {k: dict(hours=h, n=int((ev.set == k).sum()) if len(ev) else 0) for k, h in hours.items()}
for k in rates: rates[k]["per_hour"] = rates[k]["n"] / rates[k]["hours"] if rates[k]["hours"] else None
meta["rates"] = rates
(OUT / "rates.json").write_text(json.dumps(meta, indent=1, default=float))
print(json.dumps(meta, indent=1, default=lambda z: round(float(z), 4)))
