"""NCST の重量車のデータで、実測のブレーキ温度と制動の履歴の関係を確かめ、熱の模型がどこまで成り立つかを検証する。

データ (data/brake/ncst_hdv_brake/、scripts/brake/fetch_ncst.py で取得、CC0):
  ブレーキの記録 DATAxx.CSV: 1 Hz。Time、T (冷接点の温度)、CJC (ブレーキの温度 °C。README のとおり T と CJC の名前が逆)、PV、PSI (後輪の油圧ではなくエア圧、psi)
     温度の 10001・10002 は抜け (報告書の "temperature drop out")
  台上試験の車速: 動力計の記録 2021021xxx_xxxx.csv (1 Hz、MPH、気象計の気温 WS Tempera)
  路上の車速: HEM の記録 1C16xxxx.csv (J1939 の車輪の車速 kph、GPS)。抜けが多い

組 (時刻の重なりで対応づけた):
  CBD0   DATA28 + 2021021505_1905 (台上、CBD の暖機)
  CBD1   DATA32 + 2021021518_2217 (台上、CBD の暖機) ─┐ 31 s の隙間で続く
  CBDx3  DATA33 + 2021021530_2229 (台上、CBD × 3)    ─┘
  UDDS   DATA35 + 2021021621_2305 (台上、UDDS × 3)
  LOCAL  DATA42 + 1C161614 (路上、市街地)
  FWY    DATA43 + 1C161734 (路上、Riverside〜Victorville の高速道路、下り坂を含む)

熱の模型 (1 次の熱容量、離散 1 s):
  T[k+1] = T[k] + dt (a P[k] - (c0 + c1 v[k]) (T[k] - T_amb))
  P: 制動の仕事率の代わり。2 通り比べる
    P_pv = max(psi - psi0, 0) · v       (ブレーキの力は圧力に比例、仕事率は力 × 速度)
    P_ke = max(-dv/dt, 0) · v · 1(制動中) (車速だけから作る。comma2k19 の熱の指標と同じ考え。リターダ・抵抗の分を区別しない)
  T は最初の有効な温度から始めて模型だけで進める (途中の実測で直さない)。評価は抜けを除いた実測の温度との差

    uv run python scripts/brake/ncst_thermal.py --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from near_miss.plotting import configure_japanese_font

DATA = Path("data/brake/ncst_hdv_brake")
OUT = Path("out/brake_fade")
MPH = 0.44704
# 報告書 (NCST-UCR-RR-22-03 p.5): ブレーキの記録器は NIST の時刻より台上試験で 3 s、路上で 9 s 進んでいた
LOGGER_AHEAD_S = {"chassis": 3.0, "road": 9.0}
PAIRS = {
    "CBD0": ("DATA28.CSV", "2021021505_1905.csv", "chassis"),
    "CBD1": ("DATA32_CBD_Cycle_Warm_Up.CSV", "2021021518_2217.csv", "chassis"),
    "CBDx3": ("DATA33_CBDx3.CSV", "2021021530_2229.csv", "chassis"),
    "UDDS": ("DATA35_UDDSx3.CSV", "2021021621_2305.csv", "chassis"),
    "LOCAL": ("DATA42.CSV", "1C161614.csv", "road"),
    "FWY": ("DATA43.CSV", "1C161734.csv", "road"),
}


def read_logger(name: str, kind: str) -> pd.DataFrame:
    L = open(DATA / name).read().splitlines()
    day = dt.datetime.strptime(L[0].split(",")[-1].strip(), "%d:%m:%Y").date()
    rows = [r.split(",") for r in L[1:] if r.strip()]
    t = [dt.datetime.combine(day, dt.datetime.strptime(r[0].strip(), "%H:%M:%S").time()).timestamp() for r in rows]
    df = pd.DataFrame(dict(t=np.array(t) - LOGGER_AHEAD_S[kind], t_cj=[float(r[1]) for r in rows], T=[float(r[2]) for r in rows],
                           psi=[float(r[4]) for r in rows]))
    df.loc[(df["T"] > 1000) | (df["T"] <= 0), "T"] = np.nan      # 10001・10002 と 0 は抜け
    return df


def read_dyno(name: str) -> pd.DataFrame:
    L = open(DATA / name, encoding="latin1").read().splitlines()
    t0 = dt.datetime.strptime(next(x for x in L if x.startswith("Date/Time Tested")).split(": ", 1)[1].strip(), "%Y-%m-%d %H:%M:%S").timestamp()
    h = next(i for i, x in enumerate(L) if x.strip().startswith("Time,"))
    cols = [c.strip() for c in L[h].split(",")]
    rows = []
    for x in L[h + 3:]:
        p = [c.strip() for c in x.split(",")]
        try:
            rows.append([float(p[0]), float(p[cols.index("Speed")]), float(p[cols.index("WS Tempera")])])
        except (ValueError, IndexError):
            continue
    a = np.array(rows)
    return pd.DataFrame(dict(t=t0 + a[:, 0], v=a[:, 1] * MPH, t_amb=(a[:, 2] - 32) / 1.8 if np.nanmedian(a[:, 2]) > 30 else a[:, 2]))


def read_hem(name: str) -> pd.DataFrame:
    L = open(DATA / name, encoding="latin1").read().splitlines()
    cols = L[2].split(",")
    rows = [r.split(",") for r in L[3:]]

    def col(c):
        i = cols.index(c)
        out = []
        for r in rows:
            try:
                out.append(float(r[i]))
            except (ValueError, IndexError):
                out.append(np.nan)
        return np.array(out)

    t = np.array([dt.datetime.strptime(r[0].strip(), "%m/%d/%y %H:%M:%S").timestamp() for r in rows])
    v = col("Wheel-Based Vehicle Speed (kph)") / 3.6
    v[v > 70] = np.nan                        # 255.996 kph は無効
    lat, lon, alt = col("Latitude"), col("Longitude"), col("Altitude")
    bad = ~np.isfinite(lat) | (lat == 0)
    lat[bad] = lon[bad] = alt[bad] = np.nan
    amb = col("Ambient Air Temperature (C)")
    amb[(amb < -40) | (amb > 60)] = np.nan
    df = pd.DataFrame(dict(t=t, v_wheel=v, lat=lat, lon=lon, alt=alt, t_amb=amb,
                           retarder=col("Actual Retarder - Percent Torque (%)"), abs_flag=col("Anti-Lock Braking (ABS) Active (bit)")))
    df = df.groupby("t").first().reset_index()    # 同じ秒の重複は先頭
    # GPS の速度 (1 s ごとの位置の差)
    la, lo = np.radians(df.lat.to_numpy()), np.radians(df.lon.to_numpy())
    d = 6371000 * np.hypot(np.diff(la), np.diff(lo) * np.cos(la[1:]))
    dtt = np.diff(df.t.to_numpy())
    vg = np.r_[np.nan, d / np.where(dtt > 0, dtt, np.nan)]
    vg[np.r_[True, dtt > 1.5]] = np.nan
    vg[vg > 40] = np.nan
    df["v_gps"] = vg
    return df


def best_lag(t, psi, tv, v, search=60) -> tuple[float, float]:
    """油圧の立ち上がりと車速の減り (減速度) が一番そろう時刻のずれ (s)。正なら車速の時計を後ろへずらす。"""
    grid = np.arange(max(t.min(), tv.min()), min(t.max(), tv.max()), 1.0)
    p = np.interp(grid, t, np.clip(psi, 0, None))
    best = (0.0, -1.0)
    for lag in range(-search, search + 1):
        vv = np.interp(grid, tv + lag, v, left=np.nan, right=np.nan)
        dec = -np.gradient(pd.Series(vv).rolling(3, center=True, min_periods=1).mean().to_numpy())
        ok = np.isfinite(dec)
        if ok.sum() < 100:
            continue
        r = np.corrcoef(p[ok], dec[ok])[0, 1]
        if r > best[1]:
            best = (float(lag), float(r))
    return best


def build(name: str, lag_override: dict | None = None) -> tuple[pd.DataFrame, dict]:
    lf, sf, kind = PAIRS[name]
    B = read_logger(lf, kind)
    info = dict(pair=name, logger=lf, speed_file=sf, kind=kind, logger_rows=len(B), temp_valid_frac=round(float(B["T"].notna().mean()), 3))
    if kind == "chassis":
        S = read_dyno(sf)
        tv, v = S.t.to_numpy(), S.v.to_numpy()
        amb_t, amb = S.t.to_numpy(), S.t_amb.to_numpy()
    else:
        S = read_hem(sf)
        v = S.v_wheel.to_numpy().copy()
        use_gps = ~np.isfinite(v) & np.isfinite(S.v_gps.to_numpy())
        v[use_gps] = S.v_gps.to_numpy()[use_gps]
        tv = S.t.to_numpy()
        info.update(speed_wheel_valid=round(float(np.isfinite(S.v_wheel).mean()), 3), speed_gps_valid=round(float(np.isfinite(S.v_gps).mean()), 3),
                    speed_any_valid=round(float(np.isfinite(v).mean()), 3), abs_active_rows=int((S.abs_flag == 1).sum()),
                    retarder_active_rows=int(((S.retarder < 0) & (S.retarder > -125)).sum()), alt_range_m=[float(np.nanmin(S.alt)), float(np.nanmax(S.alt))])
        amb_t, amb = S.t.to_numpy(), S.t_amb.to_numpy()
    ok = np.isfinite(v)
    psi0 = float(np.nanmedian(B.psi[B.psi < np.nanpercentile(B.psi, 50)]))
    lag, r = best_lag(B.t.to_numpy(), B.psi.to_numpy() - psi0, tv[ok], v[ok])
    info.update(psi0=round(psi0, 3), clock_lag_est_s=lag, clock_lag_r=round(r, 3))
    if r < 0.4:          # 推定が弱い (路上は車速の抜けが多い) ときは、報告書の補正だけを使う
        lag = 0.0
    if lag_override and name in lag_override:
        lag = lag_override[name]
    info["clock_lag_used_s"] = lag
    # 1 s の格子 (ブレーキの記録の時刻)
    t = B.t.to_numpy()
    vv = np.interp(t, tv[ok] + lag, v[ok], left=np.nan, right=np.nan)
    # 10 s より長い抜けは補わない
    gap = np.interp(t, tv[ok] + lag, np.r_[0, np.diff(tv[ok])], left=np.nan, right=np.nan)
    tok = tv[ok] + lag
    idx = np.searchsorted(tok, t)
    gap_len = np.where((idx > 0) & (idx < len(tok)), tok[np.clip(idx, 0, len(tok) - 1)] - tok[np.clip(idx - 1, 0, len(tok) - 1)], np.inf)
    v_meas = vv.copy()
    v_meas[gap_len > 10] = np.nan
    info["speed_filled_frac"] = round(float(np.mean(~np.isfinite(v_meas) & np.isfinite(vv))), 3)
    a_amb = np.interp(t, amb_t + lag, pd.Series(amb).interpolate(limit_direction="both").to_numpy(), left=np.nan, right=np.nan)
    df = pd.DataFrame(dict(t=t, T=B["T"], t_cj=B.t_cj, psi=np.clip(B.psi.to_numpy() - psi0, 0, None), v=vv, v_meas=v_meas,
                           t_amb=np.where(np.isfinite(a_amb), a_amb, B.t_cj.to_numpy())))
    dv = -np.gradient(pd.Series(df.v).rolling(3, center=True, min_periods=1).mean().to_numpy())
    braking = df.psi > 1.0
    df["P_ke"] = np.where(braking, np.clip(dv, 0, None) * df.v, 0.0)
    info.update(brake_time_frac=round(float(braking.mean()), 3), brake_time_with_speed_frac=round(float((braking & df.v.notna()).sum() / max(braking.sum(), 1)), 3),
                T_min=float(np.nanmin(df["T"])), T_max=float(np.nanmax(df["T"])), psi_max=float(df.psi.max()), v_max=float(np.nanmax(df.v)),
                duration_min=round((t[-1] - t[0]) / 60, 1))
    if kind == "road":
        df["alt"] = np.interp(t, S.t.to_numpy() + lag, S.alt.interpolate(limit=5).to_numpy(), left=np.nan, right=np.nan)
    return df, info


def power(df: pd.DataFrame, pkey: str, p_th: float) -> np.ndarray:
    """制動の仕事率の代わり。p_th は押し出し圧 (これより低い圧力ではブレーキの力が出ない) で、当てはめで決める。"""
    psi = df.psi.to_numpy()
    v = df.v.to_numpy()
    if pkey == "P_pv":          # 圧力 × 車速
        return np.nan_to_num(np.clip(psi - p_th, 0, None) * v)
    if pkey == "P_p":           # 圧力だけ (車速を一定とみなす。車速の記録が無くても使える)
        return np.clip(psi - p_th, 0, None) * 15.0
    if pkey == "P_ke":          # 車速の減りだけ (comma2k19 と同じ考え)
        return np.nan_to_num(np.where(psi > p_th, df.P_ke.to_numpy(), 0.0))
    raise KeyError(pkey)


def simulate(params, df, pkey):
    """2 段: ドラムの温度 Td (熱の入力と冷え) と、センサーの読み Tm (Td に時定数 tau_s で遅れて追う)。"""
    a, c0, c1, p_th, tau_s = params
    P = power(df, pkey, p_th)
    v = np.nan_to_num(df.v.to_numpy(), nan=float(np.nanmedian(df.v)) if df.v.notna().any() else 0.0)
    Ta = df.t_amb.to_numpy()
    T = df["T"].to_numpy()
    i0 = int(np.argmax(np.isfinite(T)))
    Td = Tm = T[i0]
    out = np.full(len(T), np.nan)
    for k in range(i0, len(T)):
        out[k] = Tm
        Td = Td + a * P[k] - (c0 + c1 * v[k]) * (Td - Ta[k])
        Tm = Tm + (Td - Tm) / max(tau_s, 1.0)
    return out


X0 = np.array([2e-4, 2e-4, 1e-5, 2.0, 60.0])
LB = np.array([0, 0, 0, 0, 1.0])
UB = np.array([1.0, 0.05, 0.01, 10.0, 1800.0])


def fit(dfs: list[pd.DataFrame], pkey: str, use_c1: bool = True):
    free = np.array([True, True, use_c1, True, True])

    def full(x):
        p = X0.copy()
        p[free] = x
        if not use_c1:
            p[2] = 0.0
        return p

    def resid(x):
        out = []
        for df in dfs:
            sim = simulate(full(x), df, pkey)
            T = df["T"].to_numpy()
            m = np.isfinite(T) & np.isfinite(sim)
            out.append(sim[m] - T[m])
        return np.concatenate(out)
    r = least_squares(resid, X0[free], bounds=(LB[free], UB[free]), x_scale=X0[free])
    return full(r.x)


def evaluate(df, x, pkey):
    sim = simulate(x, df, pkey)
    T = df["T"].to_numpy()
    m = np.isfinite(T) & np.isfinite(sim)
    e = sim[m] - T[m]
    return dict(rmse=float(np.sqrt(np.mean(e**2))), mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
                r2=float(1 - np.sum(e**2) / np.sum((T[m] - T[m].mean()) ** 2)),
                peak_meas=float(np.nanmax(T[m])), peak_sim=float(np.max(sim[m])), n=int(m.sum())), sim


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--lag", default="", help="時計のずれを指定する (例 LOCAL=5,FWY=16)。路上は ncst_scene_illustration.pedal_lag で求めた値")
    a = ap.parse_args()
    lag_override = {k: float(v) for k, v in (x.split("=") for x in a.lag.split(",") if x)}
    out = OUT / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    S: dict = {}
    D, info = {}, {}
    for k in PAIRS:
        D[k], info[k] = build(k, lag_override)
        D[k].to_csv(out / f"aligned_{k}.csv", index=False)
    S["pairs"] = info
    # CBD1 と CBDx3 は 31 s の隙間で続くので 1 本にする (隙間は熱の入力 0)
    cbd = pd.concat([D["CBD1"], D["CBDx3"]], ignore_index=True)
    traj = {"CBD0": D["CBD0"], "CBD": cbd, "UDDS": D["UDDS"], "LOCAL": D["LOCAL"], "FWY": D["FWY"]}
    chassis, road = ["CBD0", "CBD", "UDDS"], ["LOCAL", "FWY"]
    rows = []
    sims = {}
    for pkey in ("P_pv", "P_p", "P_ke"):
        for use_c1 in (True, False):
            # 1) 台上で当てはめ -> 路上で評価、2) 路上で当てはめ -> 台上で評価、3) 1 本ずつ抜いて評価
            splits = [("fit_chassis", chassis, road), ("fit_road", road, chassis)] + [(f"loo_{k}", [j for j in traj if j != k], [k]) for k in traj]
            for sname, tr, te in splits:
                x = fit([traj[j] for j in tr], pkey, use_c1)
                for j in te:
                    ev, sim = evaluate(traj[j], x, pkey)
                    rows.append(dict(power=pkey, speed_cooling=use_c1, split=sname, test=j, a=x[0], c0=x[1], c1=x[2], p_th=x[3], tau_sensor_s=x[4],
                                     tau_cool_s_at_10mps=float(1 / (x[1] + 10 * x[2])) if (x[1] + 10 * x[2]) > 0 else np.inf, **ev))
                    if sname == "fit_chassis" and use_c1:
                        sims[(pkey, j)] = sim
                    if sname.startswith("loo") and use_c1:
                        sims[(pkey + "_loo", j)] = sim
    R = pd.DataFrame(rows)
    R.to_csv(out / "thermal_model_eval.csv", index=False)
    # 何もしない比較: 温度を一定 (当てはめ側の平均) とみなす
    base = []
    for j in traj:
        T = traj[j]["T"].to_numpy()
        tr = np.concatenate([traj[k]["T"].dropna().to_numpy() for k in traj if k != j])
        e = T[np.isfinite(T)] - tr.mean()
        base.append(dict(test=j, rmse_const=float(np.sqrt(np.mean(e**2))), T_sd=float(np.nanstd(T))))
    S["baseline_constant"] = base
    S["eval"] = R.round(5).to_dict("records")
    # 温度の上がり方と制動のエネルギーの関係 (報告書は 2.3e-5 °C/J と報告)
    S["energy_slope"] = {}
    for j in ("CBD", "UDDS"):
        df = traj[j]
        E = np.cumsum(df.P_ke.fillna(0).to_numpy())     # J/kg
        Tm = df["T"].to_numpy()
        m = np.isfinite(Tm)
        k = np.polyfit(E[m], Tm[m], 1)
        S["energy_slope"][j] = dict(degC_per_J_per_kg=float(k[0]), degC_per_J_at_29484kg=float(k[0] / 29484.0),
                                    note="総質量 65,000 lb = 29,484 kg (報告書)。冷えを無視した直線の当てはめ。報告書の 2.3e-5 °C/J は制動 1 回ごとの定義で、そのまま比べられない")
    # 台上試験での利きと温度: 1 回の制動ごとの 減速度 / (圧力 - 押し出し圧)。台上は車速 (動力計) がそろっている
    pth = float(R[(R.power == "P_pv") & R.speed_cooling & (R.split == "fit_chassis")].p_th.iloc[0])
    ev = []
    for j in ("CBD0", "CBD", "UDDS"):
        df = traj[j]
        v = df.v.to_numpy()
        dec = -np.gradient(pd.Series(v).rolling(3, center=True, min_periods=1).mean().to_numpy())
        on = (df.psi.to_numpy() > 5) & (v > 2)
        run = np.cumsum(np.r_[True, on[1:] != on[:-1]])
        Tm = df["T"].interpolate(limit=5).to_numpy()
        for rr in np.unique(run[on]):
            i = np.where(run == rr)[0]
            if len(i) < 3:
                continue
            ps = df.psi.to_numpy()[i]
            st = i[ps >= 0.5 * ps.max()]
            Tt = np.nanmean(Tm[i])
            if len(st) < 2 or not np.isfinite(Tt):
                continue
            ev.append(dict(traj=j, t_min=(df.t.iat[i[0]] - df.t.iat[0]) / 60, v0=float(v[i[0]]), psi=float(np.mean(ps[ps >= 0.5 * ps.max()])),
                           decel=float(np.mean(dec[st])), gain=float(np.mean(dec[st]) / max(np.mean(df.psi.to_numpy()[st]) - pth, 0.5)), T=float(Tt)))
    EV = pd.DataFrame(ev)
    EV.to_csv(out / "chassis_brake_events.csv", index=False)
    S["chassis_gain_vs_temp"] = {}
    for j, g in EV.groupby("traj"):
        g = g[(g.decel > 0.2)]
        if len(g) > 5:
            k = np.polyfit(g["T"], g.gain / g.gain.median(), 1)
            S["chassis_gain_vs_temp"][j] = dict(n=int(len(g)), T_range=[round(float(g["T"].min()), 1), round(float(g["T"].max()), 1)],
                                                gain_median=round(float(g.gain.median()), 4), gain_robust_cv=round(float((g.gain.quantile(.75) - g.gain.quantile(.25)) / 1.349 / g.gain.median()), 3),
                                                rel_change_per_100C=round(float(k[0] * 100), 3), spearman=round(float(g["T"].rank().corr(g.gain.rank())), 3))
    g = EV[EV.decel > 0.2]
    k = np.polyfit(g["T"], g.gain / g.gain.median(), 1)
    S["chassis_gain_vs_temp"]["all"] = dict(n=int(len(g)), rel_change_per_100C=round(float(k[0] * 100), 3), spearman=round(float(g["T"].rank().corr(g.gain.rank())), 3),
                                            T_range=[round(float(g["T"].min()), 1), round(float(g["T"].max()), 1)])
    # 図
    configure_japanese_font()
    fig, axs = plt.subplots(len(traj), 1, figsize=(13, 2.6 * len(traj)))
    for ax, j in zip(axs, traj):
        df = traj[j]
        T = df["T"].to_numpy()
        tt = (df.t.to_numpy() - df.t.iat[0]) / 60
        ax.plot(tt, T, ".", ms=2, color="#555", label="実測 (抜けは除く)")
        if ("P_pv", j) in sims:
            ax.plot(tt, sims[("P_pv", j)], color="#c0392b", lw=1.2, label="模型 P=圧力×速度 (台上で当てはめ)")
        if ("P_pv_loo", j) in sims:
            ax.plot(tt, sims[("P_pv_loo", j)], color="#2e86c1", lw=1.0, ls="--", label="模型 (この 1 本を除いて当てはめ)")
        ax2 = ax.twinx()
        ax2.fill_between(tt, 0, df.psi.to_numpy(), color="#e67e22", alpha=0.3, lw=0)
        if "v_meas" in df:
            ax2.plot(tt, np.where(df.v_meas.notna(), -2, np.nan), color="#27ae60", lw=3, label="車速の記録あり")
        ax2.set_ylabel("圧力 [psi]", color="#e67e22")
        ax.set_ylabel("温度 [°C]")
        ax.set_title(f"{j}")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8, loc="upper left")
    axs[-1].set_xlabel("経過時間 [分]")
    fig.tight_layout()
    fig.savefig(out / "thermal_model.png", dpi=110)
    plt.close(fig)
    (out / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    (out / "meta.json").write_text(json.dumps(dict(run_id=a.run_id, data=str(DATA), pairs=PAIRS, logger_ahead_s=LOGGER_AHEAD_S), ensure_ascii=False, indent=1))
    print(json.dumps({k: S[k] for k in ("energy_slope", "baseline_constant")}, ensure_ascii=False, indent=1, default=float))
    for k, v in S["pairs"].items():
        print(k, {kk: v.get(kk) for kk in ("clock_lag_est_s", "clock_lag_r", "clock_lag_used_s", "temp_valid_frac", "speed_any_valid", "speed_filled_frac", "T_min", "T_max")})
    print(R[(R.speed_cooling)].groupby(["power", "split", "test"])[["rmse", "r2", "bias", "peak_meas", "peak_sim", "p_th", "tau_sensor_s", "tau_cool_s_at_10mps"]].first().round(3).to_string())
    print(json.dumps(S.get("chassis_gain_vs_temp"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
