#!/usr/bin/env python3
"""ZOD Sequences full (構成 A: 車両データ・OXTS・infos) の全シーケンスの品質確認とインベントリ。Phase 1。
取得済みのファイルだけを読む。ネットワークは使わない。モデルの学習や閾値の決定はしない。

  uv run --offline --extra viz --with "zod==0.8.0" --with "dataclass-wizard==0.39.1" \\
      python scripts/zod/full_inventory.py --root raw_data/zod

設定: configs/zod/full_inventory.yaml (+ signal_survey.yaml, drive_validation.yaml)。説明: docs/zod_full_inventory.md

出力 (out/zod/full_inventory/<run_id>/):
  sequences.csv          シーケンスごと: 日時・位置・車両・OXTS の品質・高速/直進の時間・符号と単位の確かめ・ヨー残差
  drives.csv             走行 (同じ車両データのファイル) ごと: 長さ・途切れ・高速/直進の時間・GNSS
  validation_by_car.csv  車両ごとの符号・単位・遅れ・ゼロ点 (操舵の動きが十分な窓だけ)
  oxts_mode_codes.csv    OXTS の positionMode などのコードの分布 (意味は解釈しない)
  window_overlaps.csv    同じ走行のシーケンスどうしで OXTS の 50 s が重なる組
  summary.json           件数・時間の集計
  errors.csv             読めなかったシーケンス
  manifest.json          再現用 (設定の写し・スクリプトの sha256・入力)
  plots/
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_validation as D  # noqa: E402
import signal_survey as S  # noqa: E402

CFG: dict = {}
ROOT: Path = Path()
SPLIT: dict = {}


# ---------------------------------------------------------------- 小道具
def r3(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def indicator_mask(turn: np.ndarray, fs: float, margin_s: float) -> np.ndarray:
    on = pd.Series(np.nan_to_num(turn) != 0)
    return on.rolling(2 * int(margin_s * fs) + 1, center=True, min_periods=1).max().astype(bool).to_numpy()


def fill_nan(x: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, float)
    bad = ~np.isfinite(x)
    if bad.all():
        return x, 1.0
    if bad.any():
        i = np.arange(len(x))
        x = x.copy()
        x[bad] = np.interp(i[bad], i[~bad], x[~bad])
    return x, float(bad.mean())


def intervals(t, m) -> list[tuple[float, float]]:
    """真の区間の [開始, 終了) の時刻 (重なりを除いて時間を数えるため)。"""
    m = np.r_[False, np.asarray(m, bool), False]
    d = np.diff(m.astype(int))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    dt = float(np.median(np.diff(t)))
    return [(float(t[a]), float(t[b - 1] + dt)) for a, b in zip(s, e)]


def slope_corr(a, b):
    """b ≈ k·a の最小二乗の傾き k と相関。"""
    a = a - a.mean(); b = b - b.mean()
    den = float(np.dot(a, a))
    if den == 0:
        return np.nan, np.nan
    return float(np.dot(a, b) / den), float(np.corrcoef(a, b)[0, 1])


def blocked_dr(steer, v, r, mv, blocks, buf):
    """5 ブロックの交差検証のヨー残差 [deg/s] (drive_validation.window_cv の τ = 0 と同じ手順)。"""
    n = len(v)
    dr = np.full(n, np.nan)
    edges = np.linspace(0, n, blocks + 1).astype(int)
    for k in range(blocks):
        te = np.zeros(n, bool); te[edges[k]:edges[k + 1]] = True
        tr = np.ones(n, bool); tr[max(0, edges[k] - buf):min(n, edges[k + 1] + buf)] = False
        if (mv & tr).sum() < 300:
            continue
        co = D.fit_abc(steer, v, r, mv & tr)
        dr[te] = np.degrees(r - D.predict_r(steer, v, co, 0.0, 0.01))[te]
    return dr


# ---------------------------------------------------------------- OXTS の 50 s
def oxts_part(ox, vd, cfg) -> dict:
    q, sp, st = cfg["quality"], cfg["speed"], cfg["straight"]
    t = ox.t.to_numpy()
    dt = np.diff(t)
    v = ox.speed.to_numpy(float)
    rz, rz_nan = fill_nan(ox.angularRateZ.to_numpy())
    r_lp = -np.degrees(np.radians(S.lowpass(rz, 100, 1.0)))   # deg/s、左回りが正
    c = vd[S.CTRL]
    turn = np.interp(t, c.t, c[S.K_TURN].to_numpy(float)).round() if S.K_TURN in c else np.zeros(len(t))
    ind = indicator_mask(turn, 100, st["indicator_margin_s"])
    hi = v * 3.6 > sp["high_kmh"]
    straight = hi & (np.abs(r_lp) < st["oxts_max_yaw_deg_s"]) & ~ind
    vel_std = np.hypot(ox.stdDevVelNorth, ox.stdDevVelEast).to_numpy(float)
    pos_std = np.hypot(ox.stdDevPosNorth, ox.stdDevPosEast).to_numpy(float)
    beta_unc = np.degrees(vel_std / np.maximum(v, 1.0))
    m15 = v > q["beta_unc_min_speed_ms"]
    hd = np.degrees(np.unwrap(np.radians(ox.heading.to_numpy(float))))
    out = dict(oxts_rows=int(len(t)), oxts_span_s=r3(t[-1] - t[0], 2), oxts_dt_median_s=r3(np.median(dt), 4),
               oxts_max_gap_s=r3(dt.max(), 3), oxts_completeness=r3(len(t) / ((t[-1] - t[0]) * 100 + 1), 4),
               oxts_yaw_nan_frac=r3(rz_nan, 4), oxts_slip_nan_frac=r3(float(np.isnan(ox.slipAngle).mean()), 4),
               oxts_speed_kmh_p50=r3(np.median(v) * 3.6, 1), oxts_speed_kmh_min=r3(v.min() * 3.6, 1), oxts_speed_kmh_max=r3(v.max() * 3.6, 1),
               oxts_heading_change_deg=r3(np.ptp(hd), 1), oxts_yaw_abs_p99_deg_s=r3(np.percentile(np.abs(r_lp), 99), 2),
               oxts_hi_s=r3(hi.sum() / 100, 2), oxts_hi_straight_s=r3(straight.sum() / 100, 2),
               oxts_indicator_s=r3((turn != 0).sum() / 100, 2),
               oxts_pos_std_m_p50=r3(np.median(pos_std)), oxts_pos_std_m_p95=r3(np.percentile(pos_std, 95)),
               oxts_pos_good_frac=r3((pos_std <= q["pos_std_good_m"]).mean(), 4),
               oxts_vel_std_ms_p50=r3(np.median(vel_std), 4), oxts_heading_std_deg_p50=r3(float(ox.stdDevHeading.median()), 4),
               oxts_sats_p50=r3(float(ox.satellites.median()), 1),
               beta_unc_deg_p50=r3(np.median(beta_unc[m15]), 4) if m15.any() else None,
               beta_unc_deg_p95=r3(np.percentile(beta_unc[m15], 95), 4) if m15.any() else None,
               beta_usable_frac=r3((beta_unc[m15] <= q["beta_unc_good_deg"]).mean(), 4) if m15.any() else None,
               beta_usable_hi_straight_s=r3((straight & (beta_unc <= q["beta_unc_good_deg"])).sum() / 100, 2),
               oxts_start_utc=D.utc(t[0]), oxts_t0=float(t[0]), oxts_t1=float(t[-1]),
               _iv_hi=intervals(t, hi), _iv_hi_straight=intervals(t, straight),
               _iv_beta_hi_straight=intervals(t, straight & (beta_unc <= q["beta_unc_good_deg"])))
    for k in ["positionMode", "velocityMode", "orientationMode"]:
        x = ox[k].to_numpy(float) if k in ox else np.full(len(t), np.nan)
        ok = np.isfinite(x)
        out[f"{k}_nan_frac"] = r3(1 - ok.mean(), 4)
        out[f"{k}_mode"] = int(Counter(x[ok].astype(int)).most_common(1)[0][0]) if ok.any() else None
    return out


# ---------------------------------------------------------------- 符号・単位・遅れ・ゼロ点 (OXTS の 50 s)
def checks_part(ox, vd, cfg) -> tuple[dict, dict]:
    va = cfg["validation"]
    t, steer, v, r, _ = D.window_arrays(ox, vd, cfg)
    mv = v > cfg["speed"]["fit_min_ms"]
    out = dict(moving_s=r3(mv.sum() / 100, 1))
    e = vd[S.EGO]
    e = e[(e.t > t[0] - 3) & (e.t < t[-1] + 3)]
    c = vd[S.CTRL]
    c = c[(c.t > t[0] - 3) & (c.t < t[-1] + 3)]
    # 車両データの窓内の取得率
    for name, src, key in [("steer", c, S.K_STEER), ("torque", c, S.K_TORQUE), ("lat_acc", e, S.K_LATA),
                           ("lon_vel", e, S.K_LONV), ("roll_rate", e, S.K_ROLL)]:
        w = src[(src.t >= t[0]) & (src.t <= t[-1])]
        rate = 1 / np.median(np.diff(src.t)) if len(src) > 2 else np.nan
        out[f"vd_{name}_cov"] = r3(min(1.0, w[key].notna().sum() / ((t[-1] - t[0]) * rate)), 4) if len(w) else 0.0
    if mv.sum() < va["min_moving_s"] * 100 or len(e) < 100:
        out["excited"] = False
        return out, {}
    fs_e = 1 / np.median(np.diff(e.t))
    band = va["band_hz"]
    # 速さ: 車両の lon_vel / OXTS speed
    lv = np.interp(t, e.t, e[S.K_LONV].to_numpy(float))
    out["speed_ratio_vd_over_oxts"] = r3(np.median(lv[mv] / v[mv]), 4)
    # 横加速度: 車両 lat_acc と OXTS −accelerationY (どちらも左が正・重力を含む、mini で確認)
    ay_ox = S.lowpass(-ox.accelerationY.to_numpy(float), 100, va["lowpass_hz"])
    ay_vd_raw = S.lowpass(e[S.K_LATA].to_numpy(float), fs_e, min(va["lowpass_hz"], 0.4 * fs_e))
    k, rr = slope_corr(ay_ox[mv], np.interp(t, e.t, ay_vd_raw)[mv])
    out.update(latacc_slope=r3(k, 3), latacc_corr=r3(rr, 3))
    a_b = S.bandpass(-ox.accelerationY.to_numpy(float), 100, band)
    b_b = S.bandpass(e[S.K_LATA].to_numpy(float), fs_e, band)
    _, _, lag, cc, _ = S.lag_scan(t, a_b, e.t.to_numpy(), b_b, va["lag_range_s"], va["lag_step_s"])
    out.update(latacc_lag_vd_minus_oxts_s=r3(lag, 3), latacc_lag_corr=r3(cc, 3))
    # ロール角速度: deg(車両 roll_rate [rad/s]) と OXTS angularRateX [deg/s]
    rx = S.bandpass(ox.angularRateX.to_numpy(float), 100, band)
    rv = S.bandpass(np.degrees(e[S.K_ROLL].to_numpy(float)), fs_e, band)
    k, rr = slope_corr(rx[mv], np.interp(t, e.t, rv)[mv])
    out.update(roll_slope_deg=r3(k, 3), roll_corr=r3(rr, 3))
    # 舵角 → ヨーレート
    sb = S.bandpass(steer, 100, band)
    rb = S.bandpass(r, 100, band)
    _, _, lag, cc, _ = S.lag_scan(t, rb, t, sb, va["lag_range_s"], va["lag_step_s"])
    out.update(steer_to_yaw_lag_s=r3(-lag, 3), steer_to_yaw_corr=r3(cc, 3))
    out["steer_band_std_deg"] = r3(np.degrees(np.std(sb[mv])), 3)
    out["ay_band_std"] = r3(np.std(a_b[mv]), 3)
    out["excited"] = bool(out["steer_band_std_deg"] >= va["min_steer_band_deg"] and out["ay_band_std"] >= va["min_ay_band"])
    # 単軌道モデル δsw = a·(r/v) + b·(v·r) + c (50 s 全体で当てはめ、参考) と交差検証のヨー残差
    co = D.fit_abc(steer, v, r, mv)
    rp = D.predict_r(steer, v, co, 0.0, 0.01)
    res = np.degrees(r - rp)[mv]
    out.update(model_a=r3(co[0], 2), model_b=r3(co[1], 4), model_c_deg=r3(np.degrees(co[2]), 3),
               yaw_fit_R2=r3(1 - np.var(res) / np.var(np.degrees(r[mv])), 3))
    best = (np.inf, None, None)
    for d in va["steer_delay_s"]:
        _, st_d, _, _, _ = D.window_arrays(ox, vd, cfg, d)
        dr = blocked_dr(st_d, v, r, mv, cfg["window_cv"]["blocks"], int(cfg["window_cv"]["buffer_s"] * 100))
        s = np.nanstd(dr[mv])
        out[f"cv_std_delay_{d:.2f}"] = r3(s, 4)
        if s < best[0]:
            best = (s, d, dr)
    out.update(cv_best_delay_s=best[1], cv_std_deg_s=r3(best[0], 4))
    dr = best[2]
    hi = v * 3.6 > cfg["speed"]["high_kmh"]
    r_lp = np.degrees(S.lowpass(r, 100, 1.0))
    msk = hi & (np.abs(r_lp) < cfg["straight"]["oxts_max_yaw_deg_s"]) & np.isfinite(dr)
    out["cv_std_hi_straight_deg_s"] = r3(np.std(dr[msk]), 4) if msk.sum() > 200 else None
    out["cv_p99_hi_straight_deg_s"] = r3(np.percentile(np.abs(dr[msk]), 99), 4) if msk.sum() > 200 else None
    with np.errstate(all="ignore"):
        sl = D.slip("", ox, cfg)
    out.update(slip_straight_std_deg=sl["straight_beta_std_deg"], slip_fit_R2=sl["fit_R2"])
    return out, {}


# ---------------------------------------------------------------- 走行全体の車両データ
def drive_part(sid, vd, cfg) -> dict:
    st, sp = cfg["straight"], cfg["speed"]
    c = vd[S.CTRL]
    out = dict(drive_t0=float(c.t.iloc[0]), drive_t1=float(c.t.iloc[-1]), drive_ctrl_rows=int(len(c)),
               drive_min=r3((c.t.iloc[-1] - c.t.iloc[0]) / 60, 2), drive_start_utc=D.utc(c.t.iloc[0]),
               has_turn_indicator=S.K_TURN in c)
    for row in D.coverage(sid, vd, cfg) if S.K_TURN in c else []:
        out[f"cov_{row['signal']}"] = row["coverage_frac"]
        out[f"gaps_{row['signal']}"] = row["n_gaps"]
        out[f"rate_hz_{row['signal']}"] = row["rate_hz"]
    g = S.drive_grid(vd, cfg)
    fs = cfg["grid_hz"]
    ok = g[["v", "steer"]].notna().all(axis=1).to_numpy()
    base = ok & (g.turn.to_numpy() == 0)
    hs = base & (g.v.to_numpy() > st["drive_offset_min_speed_ms"])
    cdrive = float(np.median(g.steer[hs])) if hs.sum() > 100 else np.nan
    ind = indicator_mask(g.turn.to_numpy(), fs, st["indicator_margin_s"])
    hi = ok & (g.v.to_numpy() * 3.6 > sp["high_kmh"])
    straight = hi & ~ind & (np.abs(g.steer.to_numpy() - cdrive) < np.radians(st["drive_max_steer_deg"]))
    tq = g.torque.to_numpy()
    out.update(drive_offset_deg=r3(np.degrees(cdrive), 3), drive_hi_min=r3(hi.sum() / fs / 60, 2),
               drive_hi_straight_min=r3(straight.sum() / fs / 60, 2),
               drive_speed_kmh_p50=r3(np.nanmedian(g.v) * 3.6, 1), drive_speed_kmh_p95=r3(np.nanpercentile(g.v, 95) * 3.6, 1),
               drive_indicator_frac=r3(float((g.turn != 0).mean()), 4),
               drive_torque_valid_frac=r3(float(np.isfinite(tq).mean()), 4),
               drive_torque_std_hi_straight_nm=r3(np.nanstd(tq[straight]), 3) if straight.sum() > 100 else None,
               drive_torque_p50_hi_straight_nm=r3(np.nanmedian(tq[straight]), 3) if straight.sum() > 100 else None)
    out.update(gnss_part(vd))
    return out


K_HSTD = "horizontal_standard_deviation/meters/value"


def gnss_part(vd) -> dict:
    """車両データの衛星測位。品質の列は 2 通り (HDOP を持つもの / 水平位置の標準偏差を持つもの) あるので、あるほうを記録する。"""
    sat = vd["satellite"]
    lat = sat[D.K_LAT].to_numpy() / 1e9
    lon = sat[D.K_LON].to_numpy() / 1e9
    key = np.c_[lat, lon, sat[D.K_SSPD].to_numpy(), sat[D.K_SHDG].to_numpy()]
    new = np.r_[True, np.any(np.diff(key, axis=0) != 0, axis=1)]
    t = sat.t.to_numpy()[new]
    la, lo = lat[new], lon[new]
    span = t[-1] - t[0]
    path = float(np.nansum(D.dist_m(la[:-1], lo[:-1], la[1:], lo[1:])))
    out = dict(gnss_schema="hdop" if D.K_HDOP in sat else ("horizontal_std" if K_HSTD in sat else "none"),
               gnss_first_utc=D.utc(t[0]), gnss_last_utc=D.utc(t[-1]), gnss_path_km=r3(path / 1000, 1),
               gnss_missing_frac_1hz=r3(max(0.0, 1 - len(t) / max(span, 1)), 4),
               gnss_nsat_p50=r3(float(np.median(sat[D.K_NSAT].to_numpy()[new])), 1),
               gnss_invalid_frac=r3(float((sat[D.K_VALID].to_numpy()[new] == 0).mean()), 4),
               gnss_lat_range=f"{np.nanmin(la):.4f}..{np.nanmax(la):.4f}", gnss_lon_range=f"{np.nanmin(lo):.4f}..{np.nanmax(lo):.4f}")
    out["gnss_hdop_p50"] = r3(float(np.median(sat[D.K_HDOP].to_numpy()[new])), 2) if D.K_HDOP in sat else None
    out["gnss_hstd_m_p50"] = r3(float(np.median(sat[K_HSTD].to_numpy()[new])), 2) if K_HSTD in sat else None
    return out


# ---------------------------------------------------------------- 1 シーケンス
def init(cfg, root, split):
    global CFG, ROOT, SPLIT
    CFG, ROOT, SPLIT = cfg, root, split


def process(sid: str) -> dict:
    try:
        d = ROOT / "sequences" / sid
        meta = json.loads((d / "metadata.json").read_text())
        st = datetime.fromisoformat(meta["start_time"].replace("Z", "+00:00"))
        tz = S.TZ.get(meta["country_code"])
        row = dict(id=sid, split=SPLIT.get(sid, "不明"), car=meta["collection_car"], country=meta["country_code"],
                   cam_start_utc=meta["start_time"], cam_end_utc=meta["end_time"],
                   local_time=st.astimezone(ZoneInfo(tz)).isoformat(timespec="minutes") if tz else "",
                   lat=round(meta["latitude"], 5), lon=round(meta["longitude"], 5))
        ox = S.load_oxts(d / "oxts.hdf5")
        vd = S.load_vehicle(d / "vehicle_data.hdf5")
        row.update(oxts_part(ox, vd, CFG))
        row["oxts_start_minus_cam_s"] = r3(row["oxts_t0"] - st.timestamp(), 2)
        chk, _ = checks_part(ox, vd, CFG)
        row.update(chk)
        row.update(drive_part(sid, vd, CFG))
        row["oxts_offset_in_drive_min"] = r3((row["oxts_t0"] - row["drive_t0"]) / 60, 2)
        return dict(ok=True, row=row)
    except Exception as ex:  # noqa: BLE001 — 読めないシーケンスは記録して続ける
        return dict(ok=False, id=sid, error=f"{type(ex).__name__}: {ex}", tb=traceback.format_exc(limit=3))


# ---------------------------------------------------------------- 集計
def group_drives(df: pd.DataFrame, tol: float) -> pd.DataFrame:
    df = df.sort_values(["car", "drive_t0", "drive_t1"]).copy()
    ids, prev, k = [], None, 0
    for _, r in df.iterrows():
        if prev is None or r.car != prev.car or abs(r.drive_t0 - prev.drive_t0) > tol or abs(r.drive_t1 - prev.drive_t1) > tol:
            k += 1
            prev = r
        ids.append(f"{r.car}_{datetime.fromtimestamp(prev.drive_t0, timezone.utc):%Y%m%dT%H%M%S}")
    df["drive_id"] = ids
    # 同じ車両で、別のファイルなのに時間が重なる走行 (部分的な重複)
    spans = df.groupby("drive_id").agg(car=("car", "first"), t0=("drive_t0", "min"), t1=("drive_t1", "max")).sort_values(["car", "t0"])
    overl = set()
    for car, s in spans.groupby("car"):
        s = s.sort_values("t0")
        for a, b in zip(s.index[:-1], s.index[1:]):
            if s.loc[b, "t0"] < s.loc[a, "t1"]:
                overl.update([a, b])
    df["drive_partial_overlap"] = df.drive_id.isin(overl)
    return df.sort_values("id")


def window_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for did, s in df.groupby("drive_id"):
        s = s.sort_values("oxts_t0")
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                a, b = s.iloc[i], s.iloc[j]
                ov = min(a.oxts_t1, b.oxts_t1) - max(a.oxts_t0, b.oxts_t0)
                if ov > 0:
                    rows.append(dict(drive_id=did, id_a=a.id, id_b=b.id, overlap_s=round(ov, 2)))
    return pd.DataFrame(rows, columns=["drive_id", "id_a", "id_b", "overlap_s"])


def union_seconds(iv) -> float:
    tot, cur = 0.0, None
    for a, b in sorted(iv):
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return tot + (cur[1] - cur[0] if cur else 0.0)


VAL_COLS = ["speed_ratio_vd_over_oxts", "latacc_slope", "latacc_corr", "latacc_lag_vd_minus_oxts_s", "roll_slope_deg", "roll_corr",
            "steer_to_yaw_lag_s", "steer_to_yaw_corr", "model_a", "model_b", "model_c_deg", "yaw_fit_R2", "cv_best_delay_s",
            "cv_std_deg_s", "cv_std_hi_straight_deg_s", "drive_offset_deg", "slip_straight_std_deg", "beta_unc_deg_p50"]


def validation_by_car(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (car, sub) in [(c, df[df.car == c]) for c in sorted(df.car.unique())]:
        ex = sub[sub.excited == True]  # noqa: E712
        for col in VAL_COLS:
            x = pd.to_numeric(ex[col], errors="coerce").dropna()
            rows.append(dict(car=car, metric=col, n_windows=int(len(x)), n_excluded_low_excitation=int(len(sub) - len(ex)),
                             p05=r3(x.quantile(0.05), 4) if len(x) else None, p25=r3(x.quantile(0.25), 4) if len(x) else None,
                             p50=r3(x.median(), 4) if len(x) else None, p75=r3(x.quantile(0.75), 4) if len(x) else None,
                             p95=r3(x.quantile(0.95), 4) if len(x) else None,
                             frac_negative=r3((x < 0).mean(), 3) if len(x) else None))
    return pd.DataFrame(rows)


def mode_codes(root: Path, ids, cars) -> pd.DataFrame:
    import h5py
    cnt = Counter()
    for sid, car in zip(ids, cars):
        with h5py.File(root / "sequences" / sid / "oxts.hdf5", "r") as f:
            for k in ["positionMode", "velocityMode", "orientationMode", "status"]:
                x = f[k][()]
                u, n = np.unique(np.where(np.isfinite(x), x, -1).astype(int), return_counts=True)
                for a, b in zip(u, n):
                    cnt[(car, k, "NaN" if a == -1 else int(a))] += int(b)
    d = pd.DataFrame([dict(car=c, field=k, code=v, samples=n) for (c, k, v), n in cnt.items()])
    d["frac_within_car_field"] = d.samples / d.groupby(["car", "field"]).samples.transform("sum")
    return d.sort_values(["field", "car", "samples"], ascending=[True, True, False])


def summarize(df: pd.DataFrame, drives: pd.DataFrame, ov: pd.DataFrame, cfg) -> dict:
    hi = cfg["speed"]["high_kmh"]
    s = dict(n_sequences=int(len(df)), n_drives=int(df.drive_id.nunique()),
             by_car=df.car.value_counts().to_dict(), by_country=df.country.value_counts().to_dict(),
             by_split=df.split.value_counts().to_dict(),
             drives_by_car=drives.car.value_counts().to_dict(),
             sequences_per_drive=dict(df.drive_id.value_counts().describe().round(2)),
             date_range_utc=[df.cam_start_utc.min(), df.cam_start_utc.max()],
             n_dates=int(df.cam_start_utc.str[:10].nunique()),
             drives_with_partial_overlap=int(drives.drive_partial_overlap.sum()),
             oxts_window_overlap_pairs=int(len(ov)))
    # OXTS の 50 s: 高速・直進
    w = {}
    for name, col in [(f"over_{hi}kmh", "oxts_hi_s"), (f"over_{hi}kmh_straight", "oxts_hi_straight_s"),
                      ("beta_usable_hi_straight", "beta_usable_hi_straight_s")]:
        x = df[col].fillna(0)
        w[name] = dict(seq_ge_1s=int((x >= 1).sum()), seq_ge_10s=int((x >= 10).sum()), seq_ge_30s=int((x >= 30).sum()),
                       seq_ge_45s=int((x >= 45).sum()), total_min=round(float(x.sum() / 60), 1),
                       by_car_min={c: round(float(v / 60), 1) for c, v in df.groupby("car")[col].sum().items()})
    # 同じ走行の重なりを除いた OXTS の時間
    u = sum(union_seconds(list(zip(g.oxts_t0, g.oxts_t1))) for _, g in df.groupby("drive_id"))
    w["oxts_total_min_raw"] = round(float(df.oxts_span_s.sum() / 60), 1)
    w["oxts_total_min_unique"] = round(float(u / 60), 1)
    for k in ["hi", "hi_straight", "beta_hi_straight"]:
        col = f"_iv_{k}"
        per_car = {}
        for car, gc in df.groupby("car"):
            per_car[car] = round(sum(union_seconds([iv for ivs in g[col] for iv in ivs]) for _, g in gc.groupby("drive_id")) / 60, 1)
        w[f"unique_min_{k}"] = dict(total=round(sum(per_car.values()), 1), by_car=per_car)
    s["oxts_windows"] = w
    # 走行全体 (同じ走行は 1 回だけ数える)
    s["drives"] = dict(total_h=round(float(drives.drive_min.sum() / 60), 1),
                       hi_h=round(float(drives.drive_hi_min.sum() / 60), 1),
                       hi_straight_h=round(float(drives.drive_hi_straight_min.sum() / 60), 1),
                       by_car_hi_straight_h={c: round(float(v / 60), 1) for c, v in drives.groupby("car").drive_hi_straight_min.sum().items()},
                       drive_min_quantiles=dict(drives.drive_min.quantile([0, 0.1, 0.5, 0.9, 1]).round(1)))
    # 品質
    q = {}
    for col in ["oxts_completeness", "oxts_pos_std_m_p50", "beta_unc_deg_p50", "beta_usable_frac", "cv_std_deg_s",
                "cv_std_hi_straight_deg_s", "oxts_start_minus_cam_s"]:
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        q[col] = dict(n=int(len(x)), **{f"p{int(p * 100):02d}": r3(x.quantile(p), 4) for p in [0.05, 0.25, 0.5, 0.75, 0.95]})
    q["n_excited_windows"] = int((df.excited == True).sum())  # noqa: E712
    q["n_beta_usable_frac_ge_0.9"] = int((df.beta_usable_frac.fillna(0) >= 0.9).sum())
    s["quality"] = q
    return {k: (v if not isinstance(v, dict) else json.loads(json.dumps(v, default=str))) for k, v in s.items()}


def plots(df: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
    out.mkdir(exist_ok=True)
    cars = sorted(df.car.unique())
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    for (col, title, rng), a in zip([("cv_std_deg_s", "ヨー残差 (交差検証) の標準偏差 [deg/s]", (0, 1.0)),
                                     ("beta_unc_deg_p50", "β の不確かさの中央値 [deg]", (0, 2.0)),
                                     ("oxts_hi_straight_s", "OXTS 50 s のうち 80 km/h 超・直進 [s]", (0, 50)),
                                     ("model_c_deg", "舵角のゼロ点 c [deg] (動きのある窓)", (-10, 10))], ax.ravel()):
        for c in cars:
            sub = df[df.car == c]
            if col == "model_c_deg":
                sub = sub[sub.excited == True]  # noqa: E712
            x = pd.to_numeric(sub[col], errors="coerce").dropna().clip(*rng)
            a.hist(x, bins=40, range=rng, histtype="step", lw=1.5, label=f"{c} (n={len(x)})")
        a.set_title(title, fontsize=10)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "quality_by_car.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("raw_data/zod"))
    ap.add_argument("--config", type=Path, default=Path("configs/zod/full_inventory.yaml"))
    ap.add_argument("--out", type=Path, default=Path("out/zod/full_inventory"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=None, help="試しに先頭の何件だけ")
    args = ap.parse_args()

    base = yaml.safe_load(Path("configs/zod/signal_survey.yaml").read_text())
    base.update(yaml.safe_load(Path("configs/zod/drive_validation.yaml").read_text()))
    own = yaml.safe_load(args.config.read_text())
    cfg = copy.deepcopy(base); cfg.update(own)

    run_id = args.run_id or datetime.now().strftime("%Y%m%dT%H%M%S")
    out = args.out / run_id
    if out.exists():
        print(f"{out} がすでにある。上書きしない", file=sys.stderr)
        return 1
    out.mkdir(parents=True)

    split = {}
    for k, v in json.loads((args.root / "trainval-sequences-full.json").read_text()).items():
        for it in v:
            split[it["id"]] = k
    ids = sorted(p.name for p in (args.root / "sequences").iterdir() if (p / "oxts.hdf5").exists())
    if args.limit:
        ids = ids[: args.limit]

    t0 = time.time()
    with Pool(cfg["workers"], initializer=init, initargs=(cfg, args.root, split)) as pool:
        res = list(pool.imap(process, ids, chunksize=4))
    el = time.time() - t0

    rows = [r["row"] for r in res if r["ok"]]
    errs = [dict(id=r["id"], error=r["error"], tb=r["tb"]) for r in res if not r["ok"]]
    pd.DataFrame(errs, columns=["id", "error", "tb"]).to_csv(out / "errors.csv", index=False)
    df = group_drives(pd.DataFrame(rows), cfg["duplicates"]["same_drive_tolerance_s"])
    ov = window_overlaps(df)
    dcols = ["drive_id", "car", "drive_partial_overlap"] + [c for c in df.columns if c.startswith(("drive_", "cov_", "gaps_", "rate_hz_", "gnss_"))
                                                          and c not in ("drive_id", "drive_partial_overlap")] + ["has_turn_indicator"]
    drives = df.groupby("drive_id").first().reset_index()[dcols]
    drives["n_sequences"] = drives.drive_id.map(df.drive_id.value_counts())
    drives["countries"] = drives.drive_id.map(df.groupby("drive_id").country.agg(lambda x: ",".join(sorted(set(x)))))
    # 走行ごとの時計のずれ・舵角のゼロ点 (操舵の動きが十分な窓の中央値) と、ロール角速度が 0 のままの走行
    ex = df[df.excited == True]  # noqa: E712
    for col, name in [("latacc_lag_vd_minus_oxts_s", "clock_vd_minus_oxts_s_median"), ("steer_to_yaw_lag_s", "steer_to_yaw_lag_s_median"),
                      ("model_c_deg", "window_c_deg_median"), ("cv_std_deg_s", "cv_std_deg_s_median")]:
        drives[name] = drives.drive_id.map(ex.groupby("drive_id")[col].median())
    drives["n_excited_windows"] = drives.drive_id.map(ex.drive_id.value_counts()).fillna(0).astype(int)
    drives["vd_roll_rate_all_zero"] = drives.drive_id.map(df.groupby("drive_id").roll_slope_deg.apply(lambda x: bool((x.dropna() == 0).all() and x.notna().any())))
    for k in ["hi", "hi_straight", "beta_hi_straight"]:
        drives[f"oxts_unique_s_{k}"] = drives.drive_id.map(df.groupby("drive_id")[f"_iv_{k}"].apply(
            lambda s: round(union_seconds([iv for ivs in s for iv in ivs]), 2)))
    df.drop(columns=[c for c in df.columns if c.startswith("_iv_")]).to_csv(out / "sequences.csv", index=False)
    drives.to_csv(out / "drives.csv", index=False)
    ov.to_csv(out / "window_overlaps.csv", index=False)
    validation_by_car(df).to_csv(out / "validation_by_car.csv", index=False)
    mode_codes(args.root, df.id, df.car).to_csv(out / "oxts_mode_codes.csv", index=False)
    summ = summarize(df, drives, ov, cfg)
    summ["n_errors"] = len(errs)
    summ["elapsed_s"] = round(el, 1)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=str))
    plots(df, out / "plots")

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        head = ""
    scripts = [Path(__file__), Path(__file__).parent / "signal_survey.py", Path(__file__).parent / "drive_validation.py"]
    manifest = dict(run_id=run_id, created_local=datetime.now().isoformat(timespec="seconds"), git_head=head,
                    note="スクリプトは未コミット。sha256 で版を特定する", scripts={str(p): sha(p) for p in scripts},
                    configs={str(p): sha(p) for p in [args.config, Path("configs/zod/signal_survey.yaml"), Path("configs/zod/drive_validation.yaml")]},
                    config_effective=cfg, input_root=str(args.root), n_input_sequences=len(ids),
                    trainval_json_sha256=sha(args.root / "trainval-sequences-full.json"),
                    python=sys.version, numpy=np.__version__, pandas=pd.__version__, elapsed_s=round(el, 1))
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    print(f"{len(rows)} 件 / 失敗 {len(errs)} 件、{el:.0f} s → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
