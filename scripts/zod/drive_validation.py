#!/usr/bin/env python3
"""ZOD の走行全体の車両データ (mini で 55 分・98 分) の調査と、外乱推定モデルの確かめ。ネットワークは使わない。

  uv run --offline --extra viz --with "zod==0.8.0" --with "dataclass-wizard==0.39.1" \\
      python scripts/zod/drive_validation.py --root raw_data/zod --version mini

  (プロジェクトの環境に zod を上乗せする。pyproject.toml・uv.lock は変わらない)

設定: configs/zod/signal_survey.yaml (前処理) と configs/zod/drive_validation.yaml (この調査の分)。
説明: docs/zod_drive_validation.md。横風の正解は無いので、残差の大きさだけで横風とは判断しない。

出力 (<out>/、既定 out/zod/drive_validation/<version>/):
  coverage.csv          信号ごとの連続記録の区間・途切れ (走行全体)
  gnss.csv              衛星測位 (車両データ) の UTC・位置の範囲、更新の間隔、欠損、HDOP・衛星数
  gnss_vs_oxts.csv      OXTS の 50 s での、衛星の位置・速さ・方位と OXTS の差
  window_cv.csv         OXTS の 50 s のヨー残差: 時間で分けた交差検証、舵角の遅れ・応答の遅れの有無
  offsets.csv           舵角の中立のずれの推定 (方法ごと)
  classes_window.csv    OXTS の 50 s: 直進・旋回・方向指示器の前後でのヨー残差の分布
  classes_drive.csv     走行全体: 区間の種類 × 速度ごとの横加速度の残差 (0.2〜2 Hz) の分布 (車両ごとの正常の分布)
  drive_split.csv       走行全体のモデル: 前半で当てはめて後半で測る (逆も)
  vehicle_only.csv      車両データだけの残差と、OXTS を使った残差の一致 (50 s)、衛星の方位からのヨーレート
  ay_terms.csv          0.2〜2 Hz の横加速度の内訳 (ロールの重力成分・横滑りの変化) とヨー残差の大きさ
  slip.csv              横滑り角 (OXTS slipAngle) の性質: 直進での散らばり、r/v との関係、速度の精度からの不確かさ
  independence.csv      OXTS と車両 IMU の帯域ごとのコヒーレンス
  detections.csv        車両データだけで残差が閾値を超えた時刻 (確認用の候補)
  plots/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import coherence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signal_survey as S  # noqa: E402

K_LAT = "latposn/nanodegrees/value"
K_LON = "longposn/nanodegrees/value"
K_HDOP = "dilution_of_precision/horizontal/unitless/value"
K_NSAT = "nrof_satellites/unitless/value"
K_SSPD = "speed/meters_per_second/value"
K_SHDG = "heading/degrees/value"
K_VALID = "valid/unitless/value"
UTC_KEYS = ["utctime/year/year/value", "utctime/month/month/value", "utctime/day/day/value",
            "utctime/hour/hour/value", "utctime/minute/minute/value", "utctime/sec/second/value"]


def utc(t):
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat(timespec="seconds")


def dist_m(lat1, lon1, lat2, lon2):
    k = np.pi / 180
    x = (lon2 - lon1) * k * np.cos(0.5 * (lat1 + lat2) * k)
    y = (lat2 - lat1) * k
    return 6371000.0 * np.hypot(x, y)


# ---------------------------------------------------------------- 1. 連続記録
def coverage(sid, vd, cfg) -> list[dict]:
    cc = cfg["coverage"]
    rows = []
    for name, grp, key in [("steering_angle", S.CTRL, S.K_STEER), ("steering_torque", S.CTRL, S.K_TORQUE),
                           ("turn_indicator", S.CTRL, S.K_TURN), ("lat_acc", S.EGO, S.K_LATA),
                           ("lon_vel", S.EGO, S.K_LONV), ("roll_rate", S.EGO, S.K_ROLL)]:
        d = vd[grp]
        t = d.t.to_numpy()
        x = d[key].to_numpy(float)
        dt = np.diff(t)
        med = np.median(dt)
        brk = np.where(dt > cc["gap_factor"] * med)[0]
        starts = np.r_[0, brk + 1]
        ends = np.r_[brk, len(t) - 1]
        seg = t[ends] - t[starts]
        seg = seg[seg >= cc["min_segment_s"]]
        gaps = dt[brk]
        rows.append(dict(id=sid, signal=name, rate_hz=round(1 / med, 2), span_min=round((t[-1] - t[0]) / 60, 2),
                         n_segments=int(len(seg)), longest_segment_min=round(seg.max() / 60, 2),
                         median_segment_min=round(np.median(seg) / 60, 2), n_gaps=int(len(gaps)),
                         gap_total_s=round(float(gaps.sum()), 2), longest_gap_s=round(float(gaps.max()) if len(gaps) else 0.0, 3),
                         coverage_frac=round(float(1 - gaps.sum() / (t[-1] - t[0])), 5),
                         n_nan=int(np.isnan(x).sum()), t_start_utc=utc(t[0]), t_end_utc=utc(t[-1])))
    return rows


# ---------------------------------------------------------------- 2. 衛星測位 (車両データ)
def gnss_updates(sat) -> pd.DataFrame:
    """行が新しい測位かどうか: 位置・速さ・方位のどれかが前の行と違えば新しいとみなす。"""
    lat = sat[K_LAT].to_numpy() / 1e9
    lon = sat[K_LON].to_numpy() / 1e9
    key = np.c_[lat, lon, sat[K_SSPD].to_numpy(), sat[K_SHDG].to_numpy()]
    new = np.r_[True, np.any(np.diff(key, axis=0) != 0, axis=1)]
    d = pd.DataFrame(dict(t=sat.t.to_numpy(), lat=lat, lon=lon, speed=sat[K_SSPD].to_numpy(), heading=sat[K_SHDG].to_numpy(),
                          hdop=sat[K_HDOP].to_numpy(), nsat=sat[K_NSAT].to_numpy(), valid=sat[K_VALID].to_numpy()))
    return d[new].reset_index(drop=True)


def gnss_summary(sid, vd, cfg) -> tuple[dict, pd.DataFrame]:
    g = cfg["gnss"]
    sat = vd["satellite"]
    u = gnss_updates(sat)
    t = u.t.to_numpy()
    dt = np.diff(t)
    span = t[-1] - t[0]
    stale = dt[dt > g["stale_s"]]
    # UTC の列 (秒まで) と記録の時刻の差
    Y, M, D, h, m, s = (sat[k].to_numpy().astype(int) for k in UTC_KEYS)
    ok = (M > 0) & (D > 0)
    ut = np.array([datetime(2000 + y if y < 100 else y, mm, dd, hh, mi, ss, tzinfo=timezone.utc).timestamp()
                   for y, mm, dd, hh, mi, ss in zip(Y[ok], M[ok], D[ok], h[ok], m[ok], s[ok])])
    off = sat.t.to_numpy()[ok] - ut
    bad = (u.hdop == 0) | (u.nsat < 4) | (u.valid == 0)
    path = float(np.nansum(dist_m(u.lat.to_numpy()[:-1], u.lon.to_numpy()[:-1], u.lat.to_numpy()[1:], u.lon.to_numpy()[1:])))
    return dict(id=sid, rows=int(len(sat)), unique_timestamps=int(sat.t.nunique()), position_updates=int(len(u)),
                update_interval_s_median=round(float(np.median(dt)), 3), update_interval_s_p99=round(float(np.percentile(dt, 99)), 3),
                span_min=round(span / 60, 2), first_utc=utc(t[0]), last_utc=utc(t[-1]),
                n_stale_gaps=int(len(stale)), stale_total_s=round(float(stale.sum()), 1),
                longest_gap_s=round(float(dt.max()), 2), missing_frac_1hz=round(float(max(0.0, 1 - len(u) / max(span, 1))), 4),
                utc_fields_valid_frac=round(float(ok.mean()), 4),
                timestamp_minus_utcfield_s_p5_p50_p95="/".join(f"{x:.2f}" for x in np.percentile(off, [5, 50, 95])),
                hdop_p50=round(float(u.hdop.median()), 2), hdop_p95=round(float(u.hdop.quantile(0.95)), 2),
                hdop_zero_frac=round(float((u.hdop == 0).mean()), 4), nsat_p5=float(u.nsat.quantile(0.05)),
                nsat_p50=float(u.nsat.median()), questionable_frac=round(float(bad.mean()), 4),
                lat_range=f"{u.lat.min():.4f}..{u.lat.max():.4f}", lon_range=f"{u.lon.min():.4f}..{u.lon.max():.4f}",
                path_km=round(path / 1000, 1)), u


def gnss_vs_oxts(sid, u, ox, cfg) -> dict:
    g = cfg["gnss"]
    t = ox.t.to_numpy()
    w = u[(u.t > t[0] + g["shift_range_s"]) & (u.t < t[-1] - g["shift_range_s"])]
    shifts = np.arange(-g["shift_range_s"], g["shift_range_s"] + 1e-9, g["shift_step_s"])
    med = []
    for sh in shifts:
        la = np.interp(w.t + sh, t, ox.posLat)
        lo = np.interp(w.t + sh, t, ox.posLon)
        med.append(np.median(dist_m(w.lat, w.lon, la, lo)))
    i = int(np.argmin(med))
    sh = shifts[i]
    la, lo = np.interp(w.t + sh, t, ox.posLat), np.interp(w.t + sh, t, ox.posLon)
    d = dist_m(w.lat, w.lon, la, lo)
    sp = np.interp(w.t + sh, t, ox.speed)
    hd = np.interp(w.t + sh, t, np.degrees(np.unwrap(np.radians(ox.heading))))
    hdiff = (w.heading - hd + 180) % 360 - 180
    m = w.speed > g["min_speed_heading"]
    return dict(id=sid, n_updates=int(len(w)), oxts_pos_std_m_median=round(float(np.median(np.hypot(ox.stdDevPosNorth, ox.stdDevPosEast))), 3),
                best_time_shift_s=round(float(sh), 2), dist_m_at_shift0_median=round(float(med[np.argmin(np.abs(shifts))]), 2),
                dist_m_median=round(float(np.median(d)), 2), dist_m_p95=round(float(np.percentile(d, 95)), 2),
                speed_diff_ms_median=round(float(np.median(w.speed - sp)), 3), speed_diff_ms_p95abs=round(float(np.percentile(np.abs(w.speed - sp), 95)), 3),
                heading_diff_deg_median=round(float(np.median(hdiff[m])), 2) if m.any() else np.nan,
                heading_diff_deg_p95abs=round(float(np.percentile(np.abs(hdiff[m]), 95)), 2) if m.any() else np.nan,
                hdop_median=round(float(w.hdop.median()), 2), nsat_median=float(w.nsat.median()))


# ---------------------------------------------------------------- 3. OXTS の 50 s のヨー残差の確かめ
def window_arrays(ox, vd, cfg, delay=0.0):
    t = ox.t.to_numpy()
    c = vd[S.CTRL]
    st = S.moving(c[S.K_STEER].to_numpy(float), S.sm_n(c, cfg["smooth_s"]["steer"]))
    steer = np.interp(t - delay, c.t, st)                      # delay > 0: 舵角を遅らせる
    v = S.moving(ox.speed.to_numpy(), cfg["smooth_s"]["speed"] * 100)
    r = np.radians(-S.moving(ox.angularRateZ.to_numpy(), cfg["smooth_s"]["steer"] * 100))   # 左回りが正
    turn = np.interp(t, c.t, c[S.K_TURN].to_numpy(float)).round()
    return t, steer, v, r, turn


def first_order(x, tau, dt):
    if tau <= 0:
        return x
    a = dt / (tau + dt)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def fit_abc(steer, v, r, m):
    X = np.c_[r / v, v * r, np.ones(len(v))]
    with np.errstate(all="ignore"):
        co, *_ = np.linalg.lstsq(X[m], steer[m], rcond=None)
    return co


def predict_r(steer, v, co, tau, dt):
    with np.errstate(all="ignore"):
        rss = v * (steer - co[2]) / (co[0] + co[1] * v ** 2)
    return first_order(rss, tau, dt)


def window_cv(sid, ox, vd, cfg) -> tuple[list[dict], np.ndarray]:
    wc = cfg["window_cv"]
    rows = []
    best = None
    for delay in wc["steer_delay_s"]:
        t, steer, v, r, _ = window_arrays(ox, vd, cfg, delay)
        dt = float(np.median(np.diff(t)))
        mv = v > cfg["fit"]["min_speed"]
        n = len(t)
        for tau in wc["yaw_tau_s"]:
            # (a) 5 つのブロック: 1 つを評価、前後 buffer を除いた残りで当てはめ
            dr = np.full(n, np.nan)
            edges = np.linspace(0, n, wc["blocks"] + 1).astype(int)
            buf = int(wc["buffer_s"] / dt)
            for k in range(wc["blocks"]):
                te = np.zeros(n, bool); te[edges[k]:edges[k + 1]] = True
                tr = np.ones(n, bool); tr[max(0, edges[k] - buf):min(n, edges[k + 1] + buf)] = False
                co = fit_abc(steer, v, r, mv & tr)
                # 当てはめは定常の式 (τ = 0) で行い、予測にだけ遅れを入れる
                rp = predict_r(steer, v, co, tau, dt)
                dr[te] = np.degrees(r - rp)[te]
            # (b) 前半 → 後半 (時間の順を守る)
            half = np.arange(n) < n // 2
            co = fit_abc(steer, v, r, mv & half)
            fwd = np.degrees(r - predict_r(steer, v, co, tau, dt))[~half]
            row = dict(id=sid, steer_delay_s=delay, yaw_tau_s=tau,
                       blocked_cv_std_deg_s=round(float(np.nanstd(dr[mv])), 4),
                       blocked_cv_p99_abs=round(float(np.nanpercentile(np.abs(dr[mv]), 99)), 4),
                       forward_std_deg_s=round(float(np.nanstd(fwd)), 4),
                       forward_c_deg=round(float(np.degrees(co[2])), 3))
            rows.append(row)
            if delay == 0 and tau == 0:
                best = dr
    return rows, best


def offsets(sid, ox, vd, g, model_window, cfg) -> list[dict]:
    t, steer, v, r, turn = window_arrays(ox, vd, cfg)
    rs = np.degrees(S.lowpass(r, 100, 1.0))
    m = (np.abs(rs) < 0.3) & (turn == 0)
    hs = (g.v > 25) & (g.turn == 0)
    return [dict(id=sid, method="OXTS の 50 s で単軌道モデルの c", offset_deg=model_window["c_offset_deg"], n=model_window["n_points"]),
            dict(id=sid, method="OXTS の 50 s で |r| < 0.3 deg/s の舵角の中央値", offset_deg=round(float(np.degrees(np.median(steer[m]))), 3), n=int(m.sum())),
            dict(id=sid, method="走行全体で 90 km/h 超・方向指示器なしの舵角の中央値 (路面の傾き・カーブを含む)",
                 offset_deg=round(float(np.degrees(np.nanmedian(g.steer[hs]))), 3), n=int(hs.sum()))]


def steer_lags(sid, ox, vd, g, cfg) -> list[dict]:
    """舵角 → ヨーレート (OXTS)、舵角 → 横加速度 (車両 IMU、走行全体) の遅れ。0.1〜2 Hz で比べる。"""
    t, steer, v, r, _ = window_arrays(ox, vd, cfg)
    a = S.bandpass(r, 100, [0.1, 2.0]); b = S.bandpass(steer, 100, [0.1, 2.0])
    _, _, lag1, c1, _ = S.lag_scan(t, a, t, b, 0.5, 0.01)
    ok = g[["ay", "steer", "v"]].notna().all(axis=1) & (g.v > 10)
    gg = g[ok]
    fs = cfg["grid_hz"]
    a2 = S.bandpass(gg.ay.to_numpy(), fs, [0.1, 2.0]); b2 = S.bandpass((gg.steer * gg.v ** 2).to_numpy(), fs, [0.1, 2.0])
    tt = np.arange(len(gg)) / fs
    _, _, lag2, c2, _ = S.lag_scan(tt, a2, tt, b2, 0.5, 1 / fs)
    return [dict(id=sid, pair="舵角 → OXTS ヨーレート (50 s)", lag_s=round(-lag1, 3), corr=round(c1, 4)),
            dict(id=sid, pair="舵角 × v² → 車両の横加速度 (走行全体、10 m/s 超)", lag_s=round(-lag2, 3), corr=round(c2, 4))]


# ---------------------------------------------------------------- 4. 区間の種類ごとの残差
def classes_window(sid, ox, vd, dr, cfg) -> list[dict]:
    cl = cfg["classes"]
    t, steer, v, r, turn = window_arrays(ox, vd, cfg)
    rs = np.abs(np.degrees(S.lowpass(r, 100, 1.0)))
    ind = pd.Series(turn != 0).rolling(int(2 * cl["indicator_margin_s"] * 100) + 1, center=True, min_periods=1).max().astype(bool).to_numpy()
    cls = np.where(ind, "方向指示器の前後", np.where(rs < cl["straight_max_yaw_deg_s"], "直進",
                   np.where(rs >= cl["turn_min_yaw_deg_s"], "旋回", "その間")))
    rows = []
    for k in ["直進", "その間", "旋回", "方向指示器の前後"]:
        x = np.abs(dr[(cls == k) & np.isfinite(dr)])
        rows.append(dict(id=sid, cls=k, seconds=round(len(x) / 100, 1),
                         abs_resid_p50=round(float(np.median(x)), 3) if len(x) else np.nan,
                         abs_resid_p95=round(float(np.percentile(x, 95)), 3) if len(x) else np.nan,
                         abs_resid_max=round(float(x.max()), 3) if len(x) else np.nan))
    return rows


def drive_residual(g, model, cfg):
    g = g.copy()
    g["ay_pred"] = S.predict_ay(g, model)
    g["d_ay"] = g.ay - g.ay_pred
    fs = cfg["grid_hz"]
    ok = g.d_ay.notna().to_numpy()
    band = S.bandpass(g.d_ay.interpolate(limit_direction="both").to_numpy(), fs, cfg["straight_highspeed"]["band_hz"])
    near = pd.Series(~ok).rolling(2 * int(cfg["gap_margin_s"] * fs) + 1, center=True, min_periods=1).max().astype(bool).to_numpy()
    band[near] = np.nan
    g["d_ay_band"] = band
    return g


def drive_classes(g, model, cfg):
    cl = cfg["classes"]
    fs = cfg["grid_hz"]
    ind = pd.Series(g.turn != 0).rolling(int(2 * cl["indicator_margin_s"] * fs) + 1, center=True, min_periods=1).max().astype(bool)
    curve = (g.steer - np.radians(model["c_offset_deg"])).abs() >= np.radians(cl["curve_min_steer_deg"])
    return np.where(ind, "方向指示器の前後", np.where(curve, "曲がり (舵角)", "直進 (舵角)"))


def classes_drive(sid, g, cls, cfg) -> list[dict]:
    bins = cfg["classes"]["speed_bins_kmh"]
    rows = []
    kmh = g.v * 3.6
    for k in ["直進 (舵角)", "曲がり (舵角)", "方向指示器の前後"]:
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (cls == k) & (kmh >= lo) & (kmh < hi) & g.d_ay_band.notna()
            x = np.abs(g.d_ay_band[m])
            if len(x) < cfg["grid_hz"] * 10:
                continue
            rows.append(dict(id=sid, cls=k, speed_kmh=f"{lo}-{hi}", minutes=round(len(x) / cfg["grid_hz"] / 60, 1),
                             p50=round(float(x.median()), 4), p95=round(float(x.quantile(0.95)), 4),
                             p99=round(float(x.quantile(0.99)), 4), p999=round(float(x.quantile(0.999)), 4),
                             torque_std=round(float(g.torque[m].std()), 3)))
    return rows


def drive_split(sid, g, cfg) -> list[dict]:
    """走行全体の横加速度のモデル: 前半で当てはめて後半で測る、と逆。時間で分ける。"""
    n = len(g)
    first = g.iloc[: n // 2]
    second = g.iloc[n // 2:]
    rows = []
    for name, tr, te in [("前半 → 後半", first, second), ("後半 → 前半", second, first), ("全体 → 全体 (参考)", g, g)]:
        m = S.fit_model([tr], cfg, name)
        gg = drive_residual(te, m, cfg)
        cls = drive_classes(gg, m, cfg)
        st = (cls == "直進 (舵角)") & (gg.v > cfg["straight_highspeed"]["min_speed"])
        rows.append(dict(id=sid, split=name, a=m["a_SRxL_m"], b=m["b_SRxKus_s2"], c_deg=m["c_offset_deg"], fit_R2=m["steer_R2"],
                         test_straight_hi_min=round(st.sum() / cfg["grid_hz"] / 60, 1),
                         test_band_std=round(float(gg.d_ay_band[st].std()), 4),
                         test_band_p99abs=round(float(gg.d_ay_band[st].abs().quantile(0.99)), 4),
                         test_lowfreq_resid_std=round(float((gg.d_ay - gg.d_ay_band)[st].std()), 4)))
    return rows


# ---------------------------------------------------------------- 5. 車両データだけで足りるか
def vehicle_only(sid, ox, vd, g, model_drive, model_win, u, cfg) -> dict:
    t, steer, v, r, _ = window_arrays(ox, vd, cfg)
    co = (model_win["a_SRxL_m"], model_win["b_SRxKus_s2"], np.radians(model_win["c_offset_deg"]))
    r_pred = predict_r(steer, v, (co[0], co[1], co[2]), 0.0, 0.01)
    ref = S.bandpass(v * (r - r_pred), 100, cfg["straight_highspeed"]["band_hz"])     # OXTS を使った残差 [m/s²]
    gg = drive_residual(g, model_drive, cfg)
    w = (gg.t > t[0]) & (gg.t < t[-1]) & gg.d_ay_band.notna()
    veh = np.interp(gg.t[w], t, ref)
    c1 = np.corrcoef(gg.d_ay_band[w], veh)[0, 1]
    # OXTS の横加速度で同じ残差を作ったもの (センサだけ替える)
    ay_ox = -S.lowpass(ox.accelerationY.to_numpy(), 100, 5)
    ay_pred = v ** 2 * (steer - np.radians(model_drive["c_offset_deg"])) / (model_drive["a_SRxL_m"] + model_drive["b_SRxKus_s2"] * v ** 2)
    ox_band = S.bandpass(ay_ox - ay_pred, 100, cfg["straight_highspeed"]["band_hz"])
    c2 = np.corrcoef(gg.d_ay_band[w], np.interp(gg.t[w], t, ox_band))[0, 1]
    # 衛星の方位 (約 1 Hz) から求めたヨーレートと OXTS
    gm = cfg["gnss"]["min_speed_heading"]
    uu = u[u.speed > gm]
    hd = np.unwrap(np.radians(uu.heading.to_numpy()))
    tu = uu.t.to_numpy()
    rate = np.gradient(hd, tu)                                  # rad/s、右回りが正
    ww = (tu > t[0] + 1) & (tu < t[-1] - 1)
    ox_rate = np.interp(tu[ww], t, np.radians(S.lowpass(ox.angularRateZ.to_numpy(), 100, 0.3)))
    c3 = np.corrcoef(rate[ww], ox_rate)[0, 1] if ww.sum() > 5 else np.nan
    rms = float(np.degrees(np.sqrt(np.mean((rate[ww] - ox_rate) ** 2)))) if ww.sum() > 5 else np.nan
    return dict(id=sid, corr_vehicle_band_vs_oxts_yaw_band=round(float(c1), 3),
                std_vehicle_band=round(float(gg.d_ay_band[w].std()), 4), std_oxts_yaw_band=round(float(np.std(veh)), 4),
                corr_vehicle_band_vs_oxts_ay_band=round(float(c2), 3),
                gnss_heading_rate_vs_oxts_corr=round(float(c3), 3), gnss_heading_rate_rms_deg_s=round(rms, 3),
                gnss_heading_rate_std_oxts_deg_s=round(float(np.degrees(np.std(ox_rate))), 3), n_gnss=int(ww.sum()))


def ay_terms(sid, ox, vd, cfg) -> list[dict]:
    """OXTS の 50 s で、0.2〜2 Hz の横加速度を分ける: ay − v·r には、ロールによる重力の成分 (g·φ) と
    横滑り角の変化 (v·dβ/dt) が入る。ヨー残差 v·Δr と大きさを比べる (車両データだけで足りるかの材料)。"""
    t, steer, v, r, _ = window_arrays(ox, vd, cfg)
    mw, _ = S.fit_yaw_window(sid, ox, vd, cfg)
    rp = predict_r(steer, v, (mw["a_SRxL_m"], mw["b_SRxKus_s2"], np.radians(mw["c_offset_deg"])), 0.0, 0.01)
    band = cfg["straight_highspeed"]["band_hz"]
    B = lambda x: S.bandpass(x, 100, band)                     # noqa: E731
    e = vd[S.EGO]
    ay = np.interp(t, e.t, e[S.K_LATA].to_numpy(float))
    phi_v = np.cumsum(np.interp(t, e.t, e[S.K_ROLL].to_numpy(float))) * float(np.median(np.diff(t)))  # 車両のロール角速度の積分
    beta = np.radians(-ox.slipAngle.to_numpy())                # 左を正
    terms = [("ヨー残差 v·Δr (OXTS のヨーレート)", v * (r - rp)),
             ("車両 ay − v·r", ay - v * r),
             ("車両 ay − v·r − g·φ (φ: 車両のロール角速度の積分)", ay - v * r - 9.81 * phi_v),
             ("参考: v·dβ/dt (OXTS の横滑り角)", v * np.gradient(beta, t))]
    return [dict(id=sid, term=k, band_std=round(float(B(x).std()), 4)) for k, x in terms]


def detections(sid, g, cls, cfg, cam_t) -> tuple[list[dict], float]:
    dc = cfg["detection"]
    fs = cfg["grid_hz"]
    st = (cls == "直進 (舵角)") & (g.v > cfg["straight_highspeed"]["min_speed"]) & g.d_ay_band.notna()
    thr = float(g.d_ay_band[st].abs().quantile(dc["quantile"]))
    over = (g.d_ay_band.abs() > thr) & g.d_ay_band.notna() & (g.v > 10)
    idx = np.where(over)[0]
    rows = []
    if len(idx):
        groups = np.split(idx, np.where(np.diff(idx) > dc["merge_s"] * fs)[0] + 1)
        for gr in groups:
            i = gr[np.argmax(g.d_ay_band.abs().to_numpy()[gr])]
            s = slice(max(0, gr[0] - 2 * fs), gr[-1] + 2 * fs)
            rows.append(dict(id=sid, t_utc=utc(g.t.iloc[i]), minute=round((g.t.iloc[i] - g.t.iloc[0]) / 60, 2),
                             speed_kmh=round(g.v.iloc[i] * 3.6, 1), d_ay_band=round(float(g.d_ay_band.iloc[i]), 3),
                             duration_s=round(len(gr) / fs, 2), cls=cls[i],
                             steer_range_deg=round(float(np.degrees(np.ptp(g.steer.iloc[s]))), 2),
                             torque_peak_nm=round(float(g.torque.iloc[s].abs().max()), 2),
                             in_camera_window=bool(cam_t[0] <= g.t.iloc[i] <= cam_t[1])))
    return rows, thr


# ---------------------------------------------------------------- 6. 横滑り角
def slip(sid, ox, cfg) -> dict:
    v = ox.speed.to_numpy()
    beta = ox.slipAngle.to_numpy()                               # deg
    r = np.radians(-S.lowpass(ox.angularRateZ.to_numpy(), 100, 2))     # 左回りが正
    b = -S.lowpass(beta, 100, 2)                                 # 左を正にそろえる (velLateral は右が正)
    m = v > 5
    X = np.c_[r[m] / v[m], np.ones(m.sum())]
    co, *_ = np.linalg.lstsq(X, np.radians(b[m]), rcond=None)
    pred = X @ co
    r2 = 1 - np.sum((np.radians(b[m]) - pred) ** 2) / np.sum((np.radians(b[m]) - np.radians(b[m]).mean()) ** 2)
    st = m & (np.abs(np.degrees(r)) < 0.5)
    sig_beta = np.degrees(np.hypot(ox.stdDevVelNorth, ox.stdDevVelEast).to_numpy() / np.maximum(v, 1))
    return dict(id=sid, beta_deg_range=f"{beta.min():.2f}..{beta.max():.2f}",
                fit_beta_vs_r_over_v_slope_m=round(float(co[0]), 3), fit_beta_offset_deg=round(float(np.degrees(co[1])), 3),
                fit_R2=round(float(r2), 3), straight_beta_std_deg=round(float(np.std(beta[st])), 3) if st.any() else np.nan,
                straight_seconds=round(st.sum() / 100, 1),
                beta_uncertainty_from_vel_std_deg_median=round(float(np.median(sig_beta[m])), 3),
                stdDevHeading_deg_median=round(float(ox.stdDevHeading.median()), 3))


# ---------------------------------------------------------------- 7. OXTS と車両 IMU
def independence(sid, ox, vd, cfg) -> list[dict]:
    t = ox.t.to_numpy()
    e = vd[S.EGO]
    fs = 100
    rows = []
    for name, a, key, sign, scale in [("横加速度", -ox.accelerationY.to_numpy(), S.K_LATA, 1, 1.0),
                                      ("ロール角速度", ox.angularRateX.to_numpy(), S.K_ROLL, 1, 57.29578),
                                      ("縦加速度", ox.accelerationX.to_numpy(), S.K_LONA, 1, 1.0)]:
        b = np.interp(t, e.t, e[key].to_numpy(float)) * sign * scale
        f, C = coherence(a - a.mean(), b - b.mean(), fs=fs, nperseg=int(cfg["independence"]["nperseg_s"] * fs))
        for lo, hi in cfg["independence"]["bands_hz"]:
            m = (f >= lo) & (f < hi)
            rows.append(dict(id=sid, signal=name, band_hz=f"{lo}-{hi}", coherence_mean=round(float(C[m].mean()), 3)))
        # 高い周波数の強さの比 (車両 IMU が平滑化されているか)
        from scipy.signal import welch
        fa, pa = welch(a, fs=fs, nperseg=512); fb, pb = welch(np.interp(t, e.t, e[key].to_numpy(float)) * scale, fs=fs, nperseg=512)
        hb = (fa >= 5) & (fa < 20)
        rows.append(dict(id=sid, signal=name, band_hz="5-20 の強さの比 (車両 / OXTS)", coherence_mean=round(float(np.sqrt(pb[hb].sum() / pa[hb].sum())), 3)))
    return rows


# ---------------------------------------------------------------- 図
def plot_drive(sid, g, u, ox, cam_t, thr, plots: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    t0 = g.t.iloc[0]
    tm = (g.t - t0) / 60
    ax[0].plot(tm, g.v * 3.6, lw=0.5); ax[0].set_ylabel("速さ [km/h]")
    ax[1].plot(tm, np.degrees(g.steer), lw=0.5); ax[1].set_ylim(-40, 40); ax[1].set_ylabel("舵角 [deg]")
    ax[2].plot(tm, g.d_ay_band, lw=0.4, color="gray"); ax[2].axhline(thr, color="tab:red", lw=0.6); ax[2].axhline(-thr, color="tab:red", lw=0.6)
    ax[2].set_ylim(-1.5, 1.5); ax[2].set_ylabel("横加速度の残差\n0.2–2 Hz [m/s²]")
    ut = (u.t - t0) / 60
    ax[3].plot(ut, u.hdop, ".", ms=1); ax[3].set_ylabel("HDOP"); ax[3].set_ylim(0, 5)
    a3 = ax[3].twinx(); a3.plot(ut, u.nsat, ".", ms=1, color="tab:orange"); a3.set_ylabel("衛星数", color="tab:orange")
    dt = np.r_[np.nan, np.diff(u.t)]
    ax[4].plot(ut, dt, ".", ms=1); ax[4].set_yscale("log"); ax[4].set_ylabel("測位の間隔 [s]")
    for a in ax:
        a.axvspan((ox.t.iloc[0] - t0) / 60, (ox.t.iloc[-1] - t0) / 60, color="tab:green", alpha=0.3)
    ax[4].set_xlabel("車両データの最初からの時刻 [分] (緑: OXTS の 50 s)")
    fig.suptitle(f"{sid}: 走行全体の車両データと衛星測位"); fig.tight_layout()
    fig.savefig(plots / f"{sid}_drive_gnss.png", dpi=100); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(u.lon, u.lat, lw=0.6)
    w = (u.t >= ox.t.iloc[0]) & (u.t <= ox.t.iloc[-1])
    ax.plot(u.lon[w], u.lat[w], color="tab:red", lw=3, label="OXTS の 50 s")
    ax.plot(u.lon.iloc[0], u.lat.iloc[0], "go", label="始め"); ax.plot(u.lon.iloc[-1], u.lat.iloc[-1], "ks", label="終わり")
    ax.set_xlabel("経度 [deg]"); ax.set_ylabel("緯度 [deg]"); ax.legend(fontsize=8); ax.set_aspect(1 / np.cos(np.radians(u.lat.mean())))
    fig.suptitle(f"{sid}: 衛星測位の軌跡 (車両データ)"); fig.tight_layout()
    fig.savefig(plots / f"{sid}_track.png", dpi=100); plt.close(fig)


def plot_classes(rows, plots: Path):
    import matplotlib.pyplot as plt
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, len(d.id.unique()), figsize=(12, 3.8), sharey=True)
    ax = np.atleast_1d(ax)
    for a, (sid, x) in zip(ax, d.groupby("id")):
        for k, xx in x.groupby("cls"):
            a.plot(xx.speed_kmh, xx.p99, "o-", label=k)
        a.set_title(sid); a.set_xlabel("速さ [km/h]"); a.legend(fontsize=8)
    ax[0].set_ylabel("|横加速度の残差 0.2–2 Hz| の 99% 点 [m/s²]")
    fig.tight_layout(); fig.savefig(plots / "classes_drive.png", dpi=100); plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("raw_data/zod"))
    ap.add_argument("--version", choices=["mini", "full"], default="mini")
    ap.add_argument("--base-config", type=Path, default=Path("configs/zod/signal_survey.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/zod/drive_validation.yaml"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    cfg.update(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    out = args.out or Path(f"out/zod/drive_validation/{args.version}")
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]

    from zod import ZodSequences
    ds = ZodSequences(str(args.root), args.version, mp=False)
    R = {k: [] for k in ["coverage", "gnss", "gnss_vs_oxts", "window_cv", "offsets", "lags", "classes_window",
                         "classes_drive", "drive_split", "vehicle_only", "ay_terms", "slip", "independence", "detections"]}
    thresholds = {}
    for sid in sorted(ds.get_all_ids()):
        info = ds[sid].info
        ox = S.load_oxts(Path(info.oxts_path))
        vd = S.load_vehicle(Path(info.vehicle_data_path))
        fr = info.camera_frames["front_blur"]
        cam_t = (fr[0].time.timestamp(), fr[-1].time.timestamp())
        g = S.drive_grid(vd, cfg)
        model_drive = S.fit_model([g], cfg, f"{sid} 走行全体")
        model_win, _ = S.fit_yaw_window(sid, ox, vd, cfg)

        R["coverage"] += coverage(sid, vd, cfg)
        gs, u = gnss_summary(sid, vd, cfg)
        gs["oxts_window_start_min"] = round((ox.t.iloc[0] - u.t.iloc[0]) / 60, 2)
        gs["oxts_window_start_utc"] = utc(ox.t.iloc[0])
        R["gnss"].append(gs)
        R["gnss_vs_oxts"].append(gnss_vs_oxts(sid, u, ox, cfg))
        cv_rows, dr = window_cv(sid, ox, vd, cfg)
        R["window_cv"] += cv_rows
        R["offsets"] += offsets(sid, ox, vd, g, model_win, cfg)
        R["lags"] += steer_lags(sid, ox, vd, g, cfg)
        R["classes_window"] += classes_window(sid, ox, vd, dr, cfg)
        gg = drive_residual(g, model_drive, cfg)
        cls = drive_classes(gg, model_drive, cfg)
        R["classes_drive"] += classes_drive(sid, gg, cls, cfg)
        R["drive_split"] += drive_split(sid, g, cfg)
        R["vehicle_only"].append(vehicle_only(sid, ox, vd, g, model_drive, model_win, u, cfg))
        R["ay_terms"] += ay_terms(sid, ox, vd, cfg)
        R["slip"].append(slip(sid, ox, cfg))
        R["independence"] += independence(sid, ox, vd, cfg)
        det, thr = detections(sid, gg, cls, cfg, cam_t)
        thresholds[sid] = round(thr, 4)
        R["detections"] += det
        plot_drive(sid, gg, u, ox, cam_t, thr, plots)
        print(sid, "済み")

    plot_classes(R["classes_drive"], plots)
    for k, rows in R.items():
        pd.DataFrame(rows).to_csv(out / f"{k}.csv", index=False)
    (out / "summary.json").write_text(json.dumps(dict(detection_thresholds=thresholds,
                                                      n_detections={s: sum(1 for r in R['detections'] if r['id'] == s) for s in thresholds}),
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ["coverage", "gnss", "gnss_vs_oxts", "window_cv", "offsets", "lags", "classes_window", "drive_split", "vehicle_only", "ay_terms", "slip", "independence"]:
        print(f"== {k}"); print(pd.DataFrame(R[k]).to_string(index=False))
    print("== thresholds", thresholds, " detections", {s: sum(1 for r in R['detections'] if r['id'] == s) for s in thresholds})
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
