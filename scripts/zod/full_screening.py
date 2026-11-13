#!/usr/bin/env python3
"""ZOD Sequences full: OXTS が使える高速・直進の区間で、操舵では説明しにくいヨーの動き (外乱の候補) を抽出する。Phase 2。
取得済みのファイルと Phase 1 のインベントリだけを使う。ネットワークは使わない。横風かどうかは判断しない。

  uv run --offline --extra viz --with "zod==0.8.0" --with "dataclass-wizard==0.39.1" \\
      python scripts/zod/full_screening.py --run-id <id>

設定: configs/zod/full_screening.yaml。説明: docs/zod_full_screening.md

手順
  1. 走行ごとに、シーケンスの OXTS を時刻で重複を除いてつなぐ (同じ時刻の値は同じことを確認済み)
  2. 時計のずれ (車両データ − OXTS) を、同じ物理量どうし (横加速度・ロール角速度) の相互相関で推定。
     舵角 → 横加速度の遅れを OXTS 側・車両 IMU 側の両方で測り、同期の誤差と応答の遅れを分ける
  3. 車両ごとに走行の単位で k 分割。学習側の走行で a, b, 舵角の遅れ d, 車両の基準のゼロ点 c_car、閾値を決める
  4. 評価する走行のゼロ点の補正量 Δc は、その走行の車両データのうち OXTS の区間から離れた所 (衛星測位の方位) で決める
  5. ヨー残差 Δr = r − r_pred (補正前: c_car、補正後: c_car + Δc) から、突発型と持続型の閾値超過を拾い、特徴量と仮説を付ける
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool

from scipy.signal import lfilter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_validation as D  # noqa: E402
import event_time as ET  # noqa: E402
import signal_survey as S  # noqa: E402

G = 9.80665
OX_COLS = ["t", "timestamp", "speed", "angularRateZ", "angularRateX", "accelerationY", "roll", "slipAngle", "heading",
           "stdDevVelNorth", "stdDevVelEast", "stdDevPosNorth", "stdDevPosEast", "stdDevHeading", "posLat", "posLon"]
CFG: dict = {}
ROOT = Path()


# ================================================================ 小道具
def r3(x, n=3):
    try:
        return None if x is None or not np.isfinite(x) else round(float(x), n)
    except TypeError:
        return None


def utc(t):
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat(timespec="milliseconds")


def runs(m: np.ndarray) -> list[tuple[int, int]]:
    m = np.r_[False, np.asarray(m, bool), False]
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def roll_mean(x, n):
    return pd.Series(x).rolling(max(int(n), 1), center=True, min_periods=1).mean().to_numpy()


def dilate(m, n):
    return pd.Series(np.asarray(m, bool)).rolling(2 * int(n) + 1, center=True, min_periods=1).max().astype(bool).to_numpy()


def resample(src_t, x, tq, max_gap, smooth_n=1):
    """不等間隔の車両データを tq に。間隔が max_gap を超える所と範囲外は無効 (False)。"""
    x = np.asarray(x, float)
    if smooth_n > 1:
        x = S.moving(x, smooth_n)
    y = np.interp(tq, src_t, x)
    i = np.searchsorted(src_t, tq).clip(1, len(src_t) - 1)
    ok = (src_t[i] - src_t[i - 1] <= max_gap) & (tq >= src_t[0]) & (tq <= src_t[-1]) & np.isfinite(y)
    return y, ok


def band(x, fs, b):
    x = np.asarray(x, float)
    if len(x) < 3 * fs / b[0]:
        return None
    return S.bandpass(x, fs, b)


# ================================================================ 1. 走行の読み込みと時計のずれ
def pooled_lag(blocks_a, b_fun, cfg_s):
    """block ごとの a(t) と b(t + L) の相関を、和を足し合わせて全 block で求める。L > 0 なら b の記録が遅い。
    戻り値: (L の推定 [s], 相関, jackknife の標準誤差 [s], block の数)"""
    lags = np.arange(-cfg_s["lag_range_s"], cfg_s["lag_range_s"] + 1e-9, cfg_s["lag_step_s"])
    sums = []
    for t, a in blocks_a:
        rows = []
        for L in lags:
            b = b_fun(t + L)
            m = np.isfinite(a) & np.isfinite(b)
            aa, bb = a[m], b[m]
            rows.append([m.sum(), aa.sum(), bb.sum(), (aa * aa).sum(), (bb * bb).sum(), (aa * bb).sum()])
        sums.append(np.array(rows))
    if not sums:
        return np.nan, np.nan, np.nan, 0
    sums = np.array(sums)                                   # block × lag × 6

    def peak(s):
        n, sa, sb, saa, sbb, sab = s.T
        with np.errstate(all="ignore"):
            c = (sab - sa * sb / n) / np.sqrt((saa - sa ** 2 / n) * (sbb - sb ** 2 / n))
        if not np.isfinite(c).any():
            return np.nan, np.nan
        i = int(np.nanargmax(c))
        L = lags[i]
        if 0 < i < len(c) - 1 and np.all(np.isfinite(c[i - 1:i + 2])):   # 放物線で細かく
            y0, y1, y2 = c[i - 1:i + 2]
            den = y0 - 2 * y1 + y2
            if den < 0:
                L = L + 0.5 * (y0 - y2) / den * cfg_s["lag_step_s"]
        return float(L), float(c[i])

    L, c = peak(sums.sum(0))
    nb = len(sums)
    se = np.nan
    if nb >= cfg_s["min_blocks_jackknife"]:
        jk = np.array([peak(sums.sum(0) - sums[j])[0] for j in range(nb)])
        se = float(np.sqrt((nb - 1) / nb * np.nansum((jk - np.nanmean(jk)) ** 2)))
    return L, c, se, nb


def oxts_union(ids):
    fr = []
    for sid in ids:
        o = S.load_oxts(ROOT / "sequences" / sid / "oxts.hdf5")
        fr.append(o[[c for c in OX_COLS if c in o]])
    ox = pd.concat(fr).drop_duplicates("timestamp").sort_values("t").reset_index(drop=True)
    return ox


def sync_drive(ox, vd, blk, cfg):
    """時計のずれと遅れ。ox は 1 走行の OXTS (block の番号 blk つき)、vd は車両データ (補正前の時刻)。"""
    sc, gc = cfg["sync"], cfg["grid"]
    bnd = sc["band_hz"]
    c, e = vd[S.CTRL], vd[S.EGO]
    # 車両データを 100 Hz の等間隔に (走行全体)、帯域を通す
    def vgrid(src, key, scale=1.0):
        ts = src.t.to_numpy()
        tg = np.arange(ts[0], ts[-1], 0.01)
        y, ok = resample(ts, src[key].to_numpy(float) * scale, tg, gc["vd_max_gap_s"])
        y = np.where(ok, y, np.nan)
        yi = pd.Series(y).interpolate(limit_direction="both").to_numpy()
        bb = S.bandpass(yi, 100, bnd)
        bb[dilate(~ok, 100)] = np.nan
        return lambda tq: np.interp(tq, tg, bb, left=np.nan, right=np.nan)
    f_ay = vgrid(e, S.K_LATA)
    f_roll = vgrid(e, S.K_ROLL, 180 / np.pi)
    f_steer = vgrid(c, S.K_STEER)
    roll_zero = bool(np.nanmax(np.abs(e[S.K_ROLL].to_numpy(float))) == 0)
    blocks = {"ay": [], "roll": [], "yaw": [], "vd_ay": []}
    t_all = ox.t.to_numpy()
    for k in np.unique(blk):
        m = blk == k
        t = t_all[m]
        if t[-1] - t[0] < sc["min_block_s"]:
            continue
        v = ox.speed.to_numpy()[m]
        mv = v > sc["min_speed_ms"]
        ay = band(-ox.accelerationY.to_numpy()[m], 100, bnd)
        rx = band(ox.angularRateX.to_numpy()[m], 100, bnd)
        rz = band(-ox.angularRateZ.to_numpy()[m], 100, bnd)
        if ay is None:
            continue
        blocks["ay"].append((t[mv], ay[mv]))
        blocks["roll"].append((t[mv], rx[mv]))
        blocks["yaw"].append((t[mv], rz[mv]))
        blocks["vd_ay"].append((t[mv], f_ay(t[mv])))           # 車両の時計のままの時刻で並べる (舵角 → 車両の横加速度)
    out = {}
    L, cc, se, nb = pooled_lag(blocks["ay"], f_ay, sc)
    out.update(clock_ay_s=L, clock_ay_corr=cc, clock_ay_se_s=se, n_blocks=nb)
    if roll_zero:
        out.update(clock_roll_s=np.nan, clock_roll_corr=np.nan, clock_roll_se_s=np.nan)
    else:
        L, cc, se, _ = pooled_lag(blocks["roll"], f_roll, sc)
        out.update(clock_roll_s=L, clock_roll_corr=cc, clock_roll_se_s=se)
    # 舵角 → 物理量 (舵角は車両データの操作の系列)。L は「舵角の記録が遅い」側が正なので、舵角が先なら負 → 遅れ = −L
    for name, bl in [("steer_to_oxts_yaw", blocks["yaw"]), ("steer_to_oxts_ay", blocks["ay"]), ("steer_to_vd_ay", blocks["vd_ay"])]:
        L, cc, se, _ = pooled_lag(bl, f_steer, sc)
        out.update({f"{name}_lag_s": -L if np.isfinite(L) else np.nan, f"{name}_corr": cc, f"{name}_se_s": se})
    out["vd_roll_all_zero"] = roll_zero
    return out


# ================================================================ 2. 100 Hz の配列と、ゼロ点の校正用の区間
def build_arrays(ox, vd, blk, shift, cfg):
    """shift: 車両データの時刻に足す量 (= −clock)。OXTS の時刻にそろえる。"""
    gc = cfg["grid"]
    t = ox.t.to_numpy()
    c, e = vd[S.CTRL], vd[S.EGO]
    tc, te = c.t.to_numpy() + shift, e.t.to_numpy() + shift
    nst = S.sm_n(c, gc["smooth_steer_s"])
    A = dict(t=t, blk=blk.astype(np.int32))
    v = np.empty_like(t); r = np.empty_like(t)
    for k in np.unique(blk):
        m = blk == k
        v[m] = S.moving(ox.speed.to_numpy()[m], gc["smooth_speed_s"] * 100)
        r[m] = np.radians(-S.moving(ox.angularRateZ.to_numpy()[m], gc["smooth_yaw_s"] * 100))   # 左回りが正
    A["v"], A["r"] = v, r
    ok_all = np.ones(len(t), bool)
    delays = cfg["model"]["steer_delay_s"]
    st = np.empty((len(delays), len(t)))
    for j, d in enumerate(delays):
        st[j], okj = resample(tc, c[S.K_STEER].to_numpy(float), t - d, gc["vd_max_gap_s"], nst)
        ok_all &= okj
    A["steer_d"] = st
    A["steer"], _ = resample(tc, c[S.K_STEER].to_numpy(float), t, gc["vd_max_gap_s"], nst)
    A["torque"], ok_tq = resample(tc, c[S.K_TORQUE].to_numpy(float), t, gc["vd_max_gap_s"], nst)
    A["turn"] = np.interp(t, tc, c[S.K_TURN].to_numpy(float)).round() if S.K_TURN in c else np.zeros(len(t))
    A["vd_ay"], ok_ay = resample(te, e[S.K_LATA].to_numpy(float), t, gc["vd_max_gap_s"])
    ok_all &= ok_tq & ok_ay
    A["vd_ok"] = ~dilate(~ok_all, gc["vd_gap_margin_s"] * 100)
    A["ox_ay"] = -ox.accelerationY.to_numpy()                           # 左が正、重力を含む
    A["roll"] = ox.roll.to_numpy()                                      # deg (OXTS)
    A["beta"] = -ox.slipAngle.to_numpy()                                # deg、左が正
    vel_std = np.hypot(ox.stdDevVelNorth, ox.stdDevVelEast).to_numpy()
    A["beta_unc"] = np.degrees(vel_std / np.maximum(ox.speed.to_numpy(), 1.0))
    A["pos_std"] = np.hypot(ox.stdDevPosNorth, ox.stdDevPosEast).to_numpy()
    A["hdg_std"] = ox.stdDevHeading.to_numpy()
    A["lat"], A["lon"], A["heading"] = ox.posLat.to_numpy(), ox.posLon.to_numpy(), ox.heading.to_numpy()
    lp = cfg["grid"]["lowpass_hz"]
    for k in np.unique(blk):                                            # block ごとに低域 (長さが足りなければ移動平均)
        m = np.where(blk == k)[0]
        for name in ["vd_ay", "ox_ay", "beta"]:
            x = pd.Series(A[name][m]).interpolate(limit_direction="both").to_numpy()
            A[name][m] = S.lowpass(x, 100, lp) if len(m) > 50 else x
        A.setdefault("r_lp", np.empty_like(t))[m] = S.lowpass(r[m], 100, lp) if len(m) > 50 else r[m]
    for kk in list(A):
        if isinstance(A[kk], np.ndarray) and A[kk].dtype == np.float64 and kk != "t":
            A[kk] = A[kk].astype(np.float32)
    return A


def calib_windows(vd, ox_blocks_iv, cfg):
    """ゼロ点の校正用: 車両データの走行全体を window_s ごとに区切り、OXTS の区間 ±margin を除いた、高速・直進の区間の
    δ の平均・速さ・衛星の方位の変化率 (r_gnss) を出す。時刻は車両データのまま (同じ時計どうしなので補正は要らない)。"""
    oc = cfg["offset_calib"]
    c, e, sat = vd[S.CTRL], vd[S.EGO], vd["satellite"]
    u = gnss_updates_any(sat)
    tw0 = np.arange(c.t.iloc[0], c.t.iloc[-1] - oc["window_s"], oc["window_s"])
    st_t, st = c.t.to_numpy(), c[S.K_STEER].to_numpy(float)
    tn = c[S.K_TURN].to_numpy(float) if S.K_TURN in c else np.zeros(len(c))
    et, ev, eay = e.t.to_numpy(), e[S.K_LONV].to_numpy(float), e[S.K_LATA].to_numpy(float)
    ut, uh = u.t.to_numpy(), np.degrees(np.unwrap(np.radians(u.heading.to_numpy())))
    rows = []
    margin = oc["exclude_oxts_margin_s"]
    for a in tw0:
        b = a + oc["window_s"]
        if any(a < t1 + margin and b > t0 - margin for t0, t1 in ox_blocks_iv):
            continue
        i0, i1 = np.searchsorted(st_t, [a - 3, b + 3])
        if i1 - i0 < 100 or np.any(tn[i0:i1] != 0):
            continue
        j0, j1 = np.searchsorted(et, [a, b])
        if j1 - j0 < 0.9 * oc["window_s"] * 45:
            continue
        vv = ev[j0:j1]
        if vv.min() < oc["min_speed_ms"]:
            continue
        k0, k1 = np.searchsorted(ut, [a, b])
        if k1 - k0 < oc["min_gnss_updates"]:
            continue
        p = np.polyfit(ut[k0:k1] - a, uh[k0:k1], 1)
        r_g = -np.radians(p[0])                     # 方位は右回りで増える → 左回りを正に (OXTS と同じ向き。build で照合)
        s0, s1 = np.searchsorted(st_t, [a, b])
        rows.append(dict(t0=a, v=float(vv.mean()), steer=float(st[s0:s1].mean()), ay=float(eay[j0:j1].mean()),
                         r_gnss=float(r_g), heading=float(np.mean(uh[k0:k1]) % 360)))
    return pd.DataFrame(rows)


def gnss_updates_any(sat):
    lat = sat[D.K_LAT].to_numpy() / 1e9
    lon = sat[D.K_LON].to_numpy() / 1e9
    key = np.c_[lat, lon, sat[D.K_SSPD].to_numpy(), sat[D.K_SHDG].to_numpy()]
    new = np.r_[True, np.any(np.diff(key, axis=0) != 0, axis=1)]
    return pd.DataFrame(dict(t=sat.t.to_numpy()[new], heading=sat[D.K_SHDG].to_numpy()[new], speed=sat[D.K_SSPD].to_numpy()[new]))


def gnss_sign_check(vd, ox, cfg):
    """衛星の方位の変化率と OXTS の r (左が正) の向きの照合 (OXTS の区間の中、window_s ごと)。"""
    sat = vd["satellite"]
    u = gnss_updates_any(sat)
    ut, uh = u.t.to_numpy(), np.degrees(np.unwrap(np.radians(u.heading.to_numpy())))
    t, r = ox.t.to_numpy(), -ox.angularRateZ.to_numpy()
    xs, ys = [], []
    for a in np.arange(t[0], t[-1] - 10, 10):
        k0, k1 = np.searchsorted(ut, [a, a + 10])
        i0, i1 = np.searchsorted(t, [a, a + 10])
        if k1 - k0 >= 8 and i1 - i0 > 900 and ox.speed.iloc[i0:i1].min() > 5:
            xs.append(-np.polyfit(ut[k0:k1] - a, uh[k0:k1], 1)[0]); ys.append(r[i0:i1].mean())
    return xs, ys


def process_drive(arg):
    did, ids, car = arg
    try:
        ox = oxts_union(ids)
        vd = S.load_vehicle(ROOT / "sequences" / ids[0] / "vehicle_data.hdf5")
        t = ox.t.to_numpy()
        blk = np.r_[0, np.cumsum(np.diff(t) > CFG["grid"]["block_gap_s"])]
        sy = sync_drive(ox, vd, blk, CFG)
        iv = [(t[blk == k][0], t[blk == k][-1]) for k in np.unique(blk)]
        cw = calib_windows(vd, [(a - 0, b) for a, b in iv], CFG)     # 車両の時刻 ≈ OXTS の時刻 (ずれ 0.1 s 程度は 30 s の余白に入る)
        gx, gy = gnss_sign_check(vd, ox, CFG)
        return dict(ok=True, drive_id=did, car=car, ids=ids, sync=sy, calib=cw, gsign=(gx, gy), iv=iv)
    except Exception as ex:  # noqa: BLE001
        import traceback
        return dict(ok=False, drive_id=did, error=f"{type(ex).__name__}: {ex}", tb=traceback.format_exc(limit=4))


def finish_drive(arg):
    """時計の補正量が決まった後に、ファイルを読み直して 100 Hz の配列を作る。"""
    did, ids, shift = arg
    ox = oxts_union(ids)
    vd = S.load_vehicle(ROOT / "sequences" / ids[0] / "vehicle_data.hdf5")
    t = ox.t.to_numpy()
    blk = np.r_[0, np.cumsum(np.diff(t) > CFG["grid"]["block_gap_s"])]
    return did, build_arrays(ox, vd, blk, shift, CFG)


def init(cfg, root):
    global CFG, ROOT
    CFG, ROOT = cfg, root


# ================================================================ 3. 交差検証のモデル
def fit_mask(A, cfg):
    mc = cfg["model"]
    ind = dilate(A["turn"] != 0, mc["indicator_margin_s"] * 100)
    edge = np.zeros(len(A["t"]), bool)
    for k in np.unique(A["blk"]):
        m = np.where(A["blk"] == k)[0]
        e = int(cfg["grid"]["edge_s"] * 100)
        edge[m[:e]] = True; edge[m[-e:]] = True
    return (A["v"] > mc["min_speed_ms"]) & ~ind & A["vd_ok"] & (np.abs(np.degrees(A["r"])) < mc["max_abs_yaw_deg_s"]) & ~edge


def fit_fold(arrs: dict, train: list, cfg):
    """学習側の走行で a, b, 遅れ d、走行ごとの c_d (固定効果) を求める。c_car = c_d の中央値。
    遅れは、学習側のヨー残差の標準偏差 (走行ごとの c_d を使う) が一番小さいものを選ぶ。"""
    mc = cfg["model"]
    V, Rr, Dd, M = [], [], [], []
    for did in train:
        A = arrs[did]
        m = A["fit_mask"].copy()
        m[np.arange(len(m)) % mc["fit_decimate"] != 0] = False
        M.append(m); V.append(A["v"][m].astype(float)); Rr.append(A["r"][m].astype(float)); Dd.append(np.full(m.sum(), did, object))
    v, r, dd = map(np.concatenate, (V, Rr, Dd))
    best = None
    for j, d in enumerate(mc["steer_delay_s"]):
        y = np.concatenate([arrs[did]["steer_d"][j][m].astype(float) for did, m in zip(train, M)])
        df = pd.DataFrame(dict(x1=r / v, x2=v * r, y=y, d=dd))
        w = df[["x1", "x2", "y"]] - df.groupby("d")[["x1", "x2", "y"]].transform("mean")
        with np.errstate(all="ignore"):
            (a, b), *_ = np.linalg.lstsq(w[["x1", "x2"]].to_numpy(), w.y.to_numpy(), rcond=None)
        g = df.groupby("d")
        cd = g.y.mean() - a * g.x1.mean() - b * g.x2.mean()
        cc = df.d.map(cd).to_numpy(float)
        with np.errstate(all="ignore"):
            rp = v * (y - cc) / (a + b * v ** 2)
        s = float(np.nanstd(np.degrees(r - rp)))
        if best is None or s < best["train_yaw_std"]:
            best = dict(delay_idx=j, delay_s=d, a=float(a), b=float(b), c_drive_fit=cd.to_dict(), c_car=float(np.median(cd.to_numpy())),
                        train_yaw_std=s, n_train_samples=int(len(y)))
    return best


def first_order_blocks(x, blk, tau):
    """block ごとに 1 次遅れ (時定数 tau [s]、100 Hz)。"""
    if tau <= 0:
        return x
    al = 0.01 / (tau + 0.01)
    y = np.empty_like(x)
    for k in np.unique(blk):
        m = np.where(blk == k)[0]
        xx = x[m]
        y[m] = lfilter([al], [1, -(1 - al)], xx, zi=[(1 - al) * xx[0]])[0]
    return y


def select_tau(arrs, train, mdl, cfg):
    """遅れ d と係数を固定し、学習側の走行 (全 100 Hz、fit_mask) でヨーの 1 次遅れ τ を選ぶ。"""
    best = None
    for tau in cfg["model"]["yaw_tau_s"]:
        e = []
        for did in train:
            if did not in mdl["c_drive_fit"]:          # 当てはめに使える区間が無い走行
                continue
            A = arrs[did]
            v = A["v"].astype(float)
            with np.errstate(all="ignore"):
                rss = v * (A["steer_d"][mdl["delay_idx"]].astype(float) - mdl["c_drive_fit"][did]) / (mdl["a"] + mdl["b"] * v ** 2)
            rss = np.where(np.isfinite(rss), rss, 0.0)
            rp = first_order_blocks(rss, A["blk"], tau)
            e.append(np.degrees(A["r"].astype(float) - rp)[A["fit_mask"]])
        s = float(np.nanstd(np.concatenate(e)))
        if best is None or s < best[1]:
            best = (tau, s)
    return best


def calib_offset(cw: pd.DataFrame, a, b, c_car, cfg, rng):
    """Δc (走行ごと)。r_gnss で直進の区間だけ。方位依存 (c0 + A sin H + B cos H) の振幅も。"""
    oc = cfg["offset_calib"]
    out = dict(calib_windows=0, delta_c_deg=np.nan, delta_c_ci_lo=np.nan, delta_c_ci_hi=np.nan, c_drive_ay_deg=np.nan,
               heading_amp_deg=np.nan, heading_amp_p=np.nan, calib_t0=None, calib_t1=None, calib_minutes=0.0)
    if cw is None or len(cw) == 0:
        return out
    w = cw[np.abs(np.degrees(cw.r_gnss)) < oc["max_abs_yaw_deg_s"]]
    out["calib_windows"] = int(len(w))
    out["calib_minutes"] = round(len(w) * oc["window_s"] / 60, 1)
    if len(w) < oc["min_windows"]:
        return out
    off = w.steer - a * w.r_gnss / w.v - b * w.v * w.r_gnss
    off_ay = w.steer - a * w.ay / w.v ** 2 - b * w.ay
    c_d = float(np.median(off))
    boots = [np.median(rng.choice(off.to_numpy(), len(off))) for _ in range(oc["n_boot"])]
    out.update(delta_c_deg=np.degrees(c_d - c_car), delta_c_ci_lo=np.degrees(np.percentile(boots, 2.5) - c_car),
               delta_c_ci_hi=np.degrees(np.percentile(boots, 97.5) - c_car), c_drive_ay_deg=np.degrees(np.median(off_ay)),
               calib_t0=utc(w.t0.min()), calib_t1=utc(w.t0.max() + oc["window_s"]))
    H = np.radians(w.heading.to_numpy())
    X = np.c_[np.ones(len(H)), np.sin(H), np.cos(H)]
    y = np.degrees(off.to_numpy())

    def amp(yy):
        co, *_ = np.linalg.lstsq(X, yy, rcond=None)
        return float(np.hypot(co[1], co[2]))
    a0 = amp(y)
    perm = [amp(rng.permutation(y)) for _ in range(oc["heading_perm"])]
    out.update(heading_amp_deg=a0, heading_amp_p=float((np.sum(np.array(perm) >= a0) + 1) / (len(perm) + 1)),
               heading_range_deg=float(np.ptp(w.heading)))
    return out


def residuals(A, mdl, c_unc, c_cor, k_roll):
    j = mdl["delay_idx"]
    v = A["v"].astype(float)
    st = A["steer_d"][j].astype(float)
    out = {}
    for name, c in [("unc", c_unc), ("cor", c_cor)]:
        with np.errstate(all="ignore"):
            rp = v * (st - c) / (mdl["a"] + mdl["b"] * v ** 2)
        rp = first_order_blocks(np.where(np.isfinite(rp), rp, 0.0), A["blk"], mdl["tau_s"])
        out[f"rp_{name}"] = rp
        dr = np.degrees(A["r"].astype(float) - rp)
        lp = np.empty_like(dr)
        for k in np.unique(A["blk"]):
            m = np.where(A["blk"] == k)[0]
            x = pd.Series(dr[m]).interpolate(limit_direction="both").to_numpy()
            lp[m] = S.lowpass(x, 100, CFG["grid"]["lowpass_hz"]) if len(m) > 50 else x
        out[f"dr_{name}"] = lp
        grav = k_roll * G * np.sin(np.radians(A["roll"].astype(float)))
        out[f"day_{name}"] = A["ox_ay"] - grav - v * rp                  # OXTS の横加速度 − 重力 (ロール角) − v·r_pred
        out[f"dayvd_{name}"] = A["vd_ay"] - grav - v * rp                # 車両 IMU で同じもの (独立のセンサー)
    return out


def screen_mask(A, speed_kmh, cfg):
    sc = cfg["screen"]
    r10 = np.empty(len(A["t"]))
    for k in np.unique(A["blk"]):
        m = np.where(A["blk"] == k)[0]
        r10[m] = roll_mean(np.degrees(A["r_lp"][m].astype(float)), sc["straight_window_s"] * 100)
    ind = dilate(A["turn"] != 0, sc["indicator_margin_s"] * 100)
    edge = ~A["fit_mask_edge_ok"]
    return (A["v"] * 3.6 >= speed_kmh) & (np.abs(r10) < sc["straight_max_yaw_deg_s"]) & ~ind & A["vd_ok"] & ~edge, r10


def fast_slow(dr, blk, cfg):
    sc = cfg["screen"]
    fast, slow = np.empty_like(dr), np.empty_like(dr)
    for k in np.unique(blk):
        m = np.where(blk == k)[0]
        fast[m] = dr[m] - roll_mean(dr[m], sc["fast_detrend_s"] * 100)
        slow[m] = roll_mean(dr[m], sc["slow_window_s"] * 100)
    return fast, slow


def quiet_prior(A, cfg):
    """各時刻の前 quiet_pre_s の |dδ/dt| の最大が quiet_steer_rate_deg_s 未満か。"""
    sc = cfg["screen"]
    q = np.zeros(len(A["t"]), bool)
    n = int(sc["quiet_pre_s"] * 100)
    for k in np.unique(A["blk"]):
        m = np.where(A["blk"] == k)[0]
        sr = np.abs(np.gradient(np.degrees(A["steer"][m].astype(float)), 0.01))
        mx = pd.Series(sr).rolling(n + 1, min_periods=1).max().to_numpy()
        q[m] = mx < sc["quiet_steer_rate_deg_s"]
    return q


def detect(x, mask, thr, min_dur_s, merge_s, blk):
    ev = []
    hit = mask & (np.abs(x) > thr)
    for i0, i1 in runs(hit):
        ev.append([i0, i1])
    # 同じ block の中で merge_s 以内のものをつなぐ
    merged = []
    for e in ev:
        if merged and blk[e[0]] == blk[merged[-1][1] - 1] and (e[0] - merged[-1][1]) <= merge_s * 100:
            merged[-1][1] = e[1]
        else:
            merged.append(e)
    return [(a, b) for a, b in merged if (b - a) >= min_dur_s * 100]


# ================================================================ 4. 事象の特徴量と仮説
def pk(x):
    """|x| が最大の所の値 (全部 NaN なら NaN)。"""
    x = np.asarray(x, float)
    if not len(x) or not np.isfinite(x).any():
        return np.nan
    return float(x[int(np.nanargmax(np.abs(x)))])


def event_features(A, R, i0, i1, kind, thr, r10, cfg, ds="cor", mask=None):
    """候補の特徴量。時刻の定義は event_time.py (推定開始・閾値超過区間・ピークを別々に持つ)。
    mask: 判定の対象のマスク (閾値超過区間の端が打ち切りかを判断する)。"""
    ec = cfg["event"]
    t = A["t"]
    blk = A["blk"]
    bm = np.where(blk == blk[i0])[0]
    lo, hi = bm[0], bm[-1] + 1
    dr = R[f"dr_{ds}"]
    x = R[f"{kind}_{ds}"]
    ip = i0 + int(np.nanargmax(np.abs(x[i0:i1])))
    sgn = np.sign(x[ip])
    # 始まり: ピークより前で、|x| が onset_frac·|ピーク| を最後に下回った直後
    j = ip
    while j > lo and np.abs(x[j]) >= ec["onset_frac"] * np.abs(x[ip]):
        j -= 1
    ion = j + 1
    ton = t[ion]
    # 推定開始の打ち切り: 半分を下回る前に区間 (block) の先頭に達した
    rise_cens = bool(j == lo and np.abs(x[lo]) >= ec["onset_frac"] * np.abs(x[ip]))
    times = ET.interval_fields(t, blk, np.ones(len(t), bool) if mask is None else mask, int(i0), int(i1), int(ion), int(ip),
                               rise_censored=rise_cens)

    def idx(sec):
        return int(np.clip(ion + sec * 100, lo, hi - 1))
    # 操舵: 基準 (始まりの 2〜0.3 s 前の中央値) から steer_move_deg 動いた最初の時刻
    st = np.degrees(A["steer"].astype(float))
    base = np.nanmedian(st[idx(-2.0):idx(-0.3)]) if idx(-0.3) > idx(-2.0) else st[ion]
    w = np.arange(idx(-1.0), idx(3.0))
    moved = np.where(np.abs(st[w] - base) > ec["steer_move_deg"])[0]
    lag_steer = float(t[w[moved[0]]] - ton) if len(moved) else np.nan
    seg = st[idx(0):idx(3.0)] - base
    dsteer = pk(seg)
    opposite = bool(np.sign(dsteer) == -sgn) if np.isfinite(dsteer) and abs(dsteer) > ec["steer_move_deg"] else False
    srate = np.gradient(st, 0.01)
    pre_rate = float(np.nanmax(np.abs(srate[idx(-1.5):idx(0)]))) if idx(0) > idx(-1.5) else np.nan
    tq = A["torque"].astype(float)
    tq_base = np.nanmedian(tq[idx(-2.0):idx(-0.3)]) if idx(-0.3) > idx(-2.0) else tq[ion]
    tseg = tq[idx(0):idx(3.0)] - tq_base
    dtq = pk(tseg)
    tw = np.arange(idx(-1.0), idx(3.0))
    tmv = np.where(np.abs(tq[tw] - tq_base) > ec["torque_move_nm"])[0]
    lag_torque = float(t[tw[tmv[0]]] - ton) if len(tmv) else np.nan
    # 残差の動きは実測の r と予測 r_pred のどちらから来たか (基準: 始まりの 1〜0.3 s 前の中央値、比べるのはピーク)
    rr = np.degrees(A["r_lp"].astype(float)); pp = np.degrees(R[f"rp_{ds}"].astype(float))
    b0, b1 = idx(-1.0), idx(-0.3)
    dR = float(rr[ip] - np.nanmedian(rr[b0:b1])) if b1 > b0 else np.nan
    dP = float(pp[ip] - np.nanmedian(pp[b0:b1])) if b1 > b0 else np.nan
    share = abs(dR) / (abs(dR) + abs(dP)) if np.isfinite(dR) and np.isfinite(dP) and (abs(dR) + abs(dP)) > 0 else np.nan
    from_r = bool(np.isfinite(share) and share >= ec["measured_share_min"])
    quiet = bool(R["quiet"][ion])
    # 横加速度の残差、β
    sl = slice(idx(-0.5), idx(3.0))
    day = R[f"day_{ds}"][sl]; dayvd = R[f"dayvd_{ds}"][sl]
    beta = A["beta"].astype(float); bunc = A["beta_unc"].astype(float)
    beta_block_ok = bool(A["beta_block_ok"][ion])
    bq = bool(np.nanmax(bunc[idx(-3):idx(3)]) <= cfg["screen"]["beta_unc_good_deg"]) and beta_block_ok
    bbase = np.nanmedian(beta[idx(-3.0):idx(-0.5)]) if idx(-0.5) > idx(-3.0) else beta[ion]
    bseg = beta[idx(0):idx(3.0)] - bbase
    dbeta = pk(bseg)
    # センサーの照合 (車両 IMU と OXTS の横加速度、0.1〜2 Hz、±3 s)
    w3 = slice(idx(-3), idx(3))
    a1 = A["ox_ay"][w3].astype(float); a2 = A["vd_ay"][w3].astype(float)
    a1 = a1 - np.nanmean(a1); a2 = a2 - np.nanmean(a2)
    ay_move = float(np.nanstd(a1))
    okc = np.isfinite(a1) & np.isfinite(a2)
    ay_corr = float(np.corrcoef(a1[okc], a2[okc])[0, 1]) if okc.sum() > 50 and ay_move > 0 and np.nanstd(a2) > 0 else np.nan
    yaw_acc = float(np.nanmax(np.abs(np.gradient(np.degrees(A["r"][idx(-1):idx(2)].astype(float)), 0.01)))) if idx(2) > idx(-1) + 2 else np.nan
    vd_bad = bool((~A["vd_ok"][idx(-3):idx(3)]).any())
    ind = bool((A["turn"][idx(-ec["indicator_pre_s"]):idx(ec["indicator_post_s"])] != 0).any())
    curv_change = float(abs(np.nanmean(np.abs(r10[idx(2):idx(6)])) - np.nanmean(np.abs(r10[idx(-6):idx(-2)])))) if idx(6) > idx(2) and idx(-2) > idx(-6) else np.nan
    # 進路の変化 (推定): 始まりの前の位置に直線を当て、その後の横のずれ (OXTS の位置)
    p0, p1 = idx(ec["path_fit_s"][0]), idx(ec["path_fit_s"][1])
    path_dev = np.nan
    if p1 - p0 > 100:
        lat0 = float(A["lat"][ion])
        xE = np.radians(A["lon"].astype(np.float64) - A["lon"][ion]) * 6371000 * np.cos(np.radians(lat0))
        yN = np.radians(A["lat"].astype(np.float64) - lat0) * 6371000
        P = np.c_[xE[p0:p1], yN[p0:p1]]
        dvec = P[-1] - P[0]
        if np.linalg.norm(dvec) > 5:
            dvec /= np.linalg.norm(dvec)
            nvec = np.array([-dvec[1], dvec[0]])                      # 左が正
            q = np.c_[xE[idx(0):idx(ec["path_eval_s"])], yN[idx(0):idx(ec["path_eval_s"])]] - P.mean(0)
            dev = q @ nvec
            path_dev = float(dev[int(np.argmax(np.abs(dev)))]) if len(dev) else np.nan
    # 説明されないヨーの積分 (始まりから 3 s): 方位のずれ [deg] と、それによる横のずれ [m] (推定)
    s3 = slice(ion, idx(3.0))
    psi = np.cumsum(dr[s3]) * 0.01
    yerr = float(np.sum(A["v"][s3].astype(float) * np.radians(psi)) * 0.01) if len(psi) else np.nan
    reasons = []
    if vd_bad:
        reasons.append("車両データの途切れ")
    if np.isfinite(yaw_acc) and yaw_acc > ec["yaw_accel_max_deg_s2"]:
        reasons.append("ヨーレートのとび")
    if ay_move >= ec["sensor_min_ay_band"] and np.isfinite(ay_corr) and ay_corr < ec["sensor_ay_corr_min"]:
        reasons.append("2 つの横加速度の不一致")
    if not beta_block_ok:
        reasons.append("OXTS の β・姿勢の異常 (区間の |β| の中央値が大きい)")
    sensor = bool(reasons)
    if ind:
        hyp = "車線変更など (方向指示器)"
    elif sensor:
        hyp = "センサー異常の疑い"
    elif kind == "slow":
        hyp = "持続的なずれ (当て舵・路面の傾き・ゼロ点・横風)"
    elif np.isfinite(curv_change) and curv_change >= ec["curvature_change_deg_s"]:
        hyp = "道路の曲がりの変化"
    elif from_r and (quiet or not np.isfinite(lag_steer) or lag_steer > ec["driver_first_max_lag_s"]):
        hyp = "ヨーの変化が先 (外乱の候補)"
    else:
        hyp = "操舵が先 (運転者の操作・モデルの誤差)"
    return dict(t_onset_utc=utc(ton), t_onset=float(ton), t_peak=float(t[ip]), duration_s=r3((i1 - i0) / 100, 2),
                t_rise_utc=utc(times["t_rise"]), t_exceed_start_utc=utc(times["t_exceed_start"]), t_exceed_end_utc=utc(times["t_exceed_end"]),
                t_peak_utc=utc(times["t_peak"]), **{k: v for k, v in times.items() if k != "t_peak"},
                lat=r3(A["lat"][ip], 6), lon=r3(A["lon"][ip], 6), speed_kmh=r3(A["v"][ip] * 3.6, 1),
                heading_deg=r3(A["heading"][ip], 1), peak_value=r3(x[ip], 3), peak_over_thr=r3(abs(x[ip]) / thr, 2),
                dr_peak_deg_s=r3(dr[ip], 3), dr_unc_at_peak=r3(R["dr_unc"][ip], 3), dr_cor_at_peak=r3(R["dr_cor"][ip], 3),
                day_peak=r3(pk(day), 3), dayvd_peak=r3(pk(dayvd), 3),
                beta_quality_ok=bq, dbeta_deg=r3(dbeta, 3), beta_unc_max=r3(np.nanmax(bunc[idx(-3):idx(3)]), 3),
                dyaw_meas_deg_s=r3(dR, 3), dyaw_pred_deg_s=r3(dP, 3), measured_share=r3(share, 3), driven_by_measured_yaw=from_r,
                sensor_reasons=" / ".join(reasons), beta_block_median_abs_deg=r3(float(np.nanmedian(np.abs(beta[lo:hi]))), 2), steer_quiet_before=quiet,
                steer_lag_after_onset_s=r3(lag_steer, 2), dsteer_deg=r3(dsteer, 3), steer_opposes=opposite,
                pre_steer_rate_max_deg_s=r3(pre_rate, 2), dtorque_nm=r3(dtq, 3), torque_lag_after_onset_s=r3(lag_torque, 2),
                torque_before_steer=bool(np.isfinite(lag_torque) and (not np.isfinite(lag_steer) or lag_torque < lag_steer - 0.1)),
                yaw_curv_change_deg_s=r3(curv_change, 3), r10_at_onset=r3(r10[ion], 3), indicator_near=ind,
                vd_gap_near=vd_bad, ay_corr_vd_oxts=r3(ay_corr, 3), ay_band_std=r3(ay_move, 3), yaw_accel_max_deg_s2=r3(yaw_acc, 1),
                heading_err_3s_deg=r3(psi[-1] if len(psi) else np.nan, 3), lateral_err_3s_m=r3(yerr, 3),
                path_dev_4s_m=r3(path_dev, 3), pos_std_m=r3(A["pos_std"][ip], 3), hypothesis=hyp, i0=int(i0), i1=int(i1), ion=int(ion), ip=int(ip))


UNCERT = {
    "車線変更など (方向指示器)": "方向指示器の前後。車線変更の動きでモデルが外れたとみられる",
    "センサー異常の疑い": "車両データの途切れ・ヨーレートのとび・2 つの横加速度の不一致のどれか。物理の動きか確かめられない",
    "持続的なずれ (当て舵・路面の傾き・ゼロ点・横風)": "数秒以上続くずれ。路面の傾き (カント・勾配の横成分)・ゼロ点の補正の誤差・横風を、車両の信号だけでは分けられない",
    "道路の曲がりの変化": "10 s 平均のヨーが前後で変わる。カーブの出入りでモデル (定常の式) が外れたとみられる",
    "操舵が先 (運転者の操作・モデルの誤差)": "操舵が残差の始まりと同時か先に動く。運転者の操作に対する応答の遅れをモデルが表せていないとみられる",
    "ヨーの変化が先 (外乱の候補)": "操舵が動く前にヨーが変わった。横風・路面 (わだち・継ぎ目)・追い越しの車の風圧などの外力の可能性。原因は特定できない",
}


# ================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("raw_data/zod"))
    ap.add_argument("--config", type=Path, default=Path("configs/zod/full_screening.yaml"))
    ap.add_argument("--out", type=Path, default=Path("out/zod/full_screening"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--cars", nargs="*", default=None)
    ap.add_argument("--max-drives", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path("configs/zod/signal_survey.yaml").read_text())
    cfg.update(yaml.safe_load(Path("configs/zod/drive_validation.yaml").read_text()))
    cfg.update(yaml.safe_load(args.config.read_text()))
    run_id = args.run_id or datetime.now().strftime("%Y%m%dT%H%M%S")
    out = args.out / run_id
    if out.exists():
        print(f"{out} がすでにある。上書きしない", file=sys.stderr)
        return 1
    (out / "plots").mkdir(parents=True)
    (out / "residuals").mkdir()
    inv = Path(cfg["inventory_run"])
    seqs = pd.read_csv(inv / "sequences.csv", dtype={"id": str})
    drives = pd.read_csv(inv / "drives.csv")
    if args.cars:
        drives = drives[drives.car.isin(args.cars)]
    if args.max_drives:
        drives = drives.groupby("car").head(args.max_drives)
    jobs = [(d, sorted(seqs[seqs.drive_id == d].id), c) for d, c in zip(drives.drive_id, drives.car)]
    t_start = time.time()
    log = {}

    # ---------------- 1. 読み込み・時計のずれ
    with Pool(cfg["workers"], initializer=init, initargs=(cfg, args.root)) as pool:
        res = list(pool.imap(process_drive, jobs, chunksize=1))
    errs = [r for r in res if not r["ok"]]
    res = [r for r in res if r["ok"]]
    log["load_s"] = round(time.time() - t_start, 1)
    syn = pd.DataFrame([dict(drive_id=r["drive_id"], car=r["car"], **r["sync"]) for r in res])
    # 補正量: 横加速度の推定。block が足りない・相関が低いときは車両の中央値を使う (記録する)
    good = (syn.clock_ay_corr > 0.6) & syn.clock_ay_s.notna()
    car_med = syn[good].groupby("car").clock_ay_s.median()
    syn["clock_used_s"] = np.where(good, syn.clock_ay_s, syn.car.map(car_med))
    syn["clock_method"] = np.where(good, "横加速度の相互相関 (この走行の OXTS 区間すべて)", "車両の中央値 (この走行では推定できない)")
    car_sd = syn[good].groupby("car").clock_ay_s.std()
    syn["clock_uncert_s"] = np.where(good, syn.clock_ay_se_s.fillna(cfg["sync"]["lag_step_s"]), syn.car.map(car_sd))
    syn["clock_uncert_note"] = np.where(good & syn.clock_ay_se_s.isna(), "block が少なく jackknife できない。刻み幅を仮に置いた",
                                        np.where(good, "jackknife (block を 1 つずつ除く) の標準誤差", "車両内の走行間の標準偏差"))
    syn["roll_minus_ay_s"] = syn.clock_roll_s - syn.clock_ay_s
    syn["steer_vd_minus_oxts_ay_s"] = syn.steer_to_vd_ay_lag_s - syn.steer_to_oxts_ay_lag_s
    syn["steer_to_oxts_yaw_lag_corrected_s"] = syn.steer_to_oxts_yaw_lag_s + syn.clock_used_s
    syn["drive_start_utc"] = syn.drive_id.map(drives.set_index("drive_id").drive_start_utc)
    syn["oxts_seconds"] = [sum(b - a for a, b in r["iv"]) for r in res]
    syn["calib_windows_all"] = [len(r["calib"]) for r in res]
    gx = np.concatenate([np.array(r["gsign"][0]) for r in res if len(r["gsign"][0])])
    gy = np.concatenate([np.array(r["gsign"][1]) for r in res if len(r["gsign"][1])])
    gnss_r_slope = float(np.polyfit(gx, gy, 1)[0])
    log["gnss_heading_rate_vs_oxts_r"] = dict(n=int(len(gx)), slope=gnss_r_slope, corr=float(np.corrcoef(gx, gy)[0, 1]))
    if gnss_r_slope < 0:
        raise SystemExit("衛星の方位の変化率と OXTS の r の向きが合わない。calib_windows の符号を確かめる")

    # ---------------- 2. 100 Hz の配列 (時計を補正)
    shifts = dict(zip(syn.drive_id, -syn.clock_used_s))
    with Pool(cfg["workers"], initializer=init, initargs=(cfg, args.root)) as pool:
        arrs = dict(pool.imap(finish_drive, [(r["drive_id"], r["ids"], shifts[r["drive_id"]]) for r in res], chunksize=1))
    meta = {r["drive_id"]: r for r in res}
    init(cfg, args.root)
    for did, A in arrs.items():
        A["fit_mask"] = fit_mask(A, cfg)
        e = int(cfg["grid"]["edge_s"] * 100)
        ok = np.ones(len(A["t"]), bool)
        for k in np.unique(A["blk"]):
            m = np.where(A["blk"] == k)[0]
            ok[m[:e]] = False; ok[m[-e:]] = False
        A["fit_mask_edge_ok"] = ok
        A["quiet"] = quiet_prior(A, cfg)
        bok = np.ones(len(A["t"]), bool)
        for k in np.unique(A["blk"]):
            m = np.where(A["blk"] == k)[0]
            bok[m] = np.nanmedian(np.abs(A["beta"][m])) <= cfg["event"]["beta_block_abs_max_deg"]
        A["beta_block_ok"] = bok
    log["arrays_s"] = round(time.time() - t_start, 1)

    # ロールの重力成分の向き (OXTS の roll と −accelY − v·r の関係を、直進・定速で測る)
    xs, ys = [], []
    for A in arrs.values():
        m = A["fit_mask"] & (np.abs(np.degrees(A["r_lp"])) < 0.3) & (A["v"] > 15)
        m[np.arange(len(m)) % 10 != 0] = False
        xs.append(G * np.sin(np.radians(A["roll"][m].astype(float)))); ys.append((A["ox_ay"][m] - A["v"][m] * A["r"][m]).astype(float))
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    k_roll = float(np.polyfit(xs, ys, 1)[0])
    log["roll_gravity"] = dict(slope=k_roll, corr=float(np.corrcoef(xs, ys)[0, 1]), n=int(len(xs)),
                               note="−accelY − v·r = k·g·sin(roll)。|k| が 0.7〜1.3 を外れたら重力の補正をしない")
    if not 0.7 <= abs(k_roll) <= 1.3:
        k_roll = 0.0

    # ---------------- 3. 車両ごとの k 分割
    rng = np.random.default_rng(cfg["seed"])
    models, calibs, thr_rows, oof = [], [], [], {}
    ev_rows, exc_rows = [], []
    for car in sorted(syn.car.unique()):
        dl = sorted(syn[syn.car == car].drive_id)
        perm = rng.permutation(len(dl))
        k = min(cfg["model"]["kfold"], len(dl))
        fold_of = {dl[p]: i % k for i, p in enumerate(perm)}
        for f in range(k):
            test = [d for d in dl if fold_of[d] == f]
            train = [d for d in dl if fold_of[d] != f]
            mdl = fit_fold(arrs, train, cfg)
            mdl["tau_s"], mdl["train_yaw_std_tau"] = select_tau(arrs, train, mdl, cfg)
            models.append(dict(car=car, fold=f, n_train_drives=len(train), n_test_drives=len(test), delay_s=mdl["delay_s"], tau_s=mdl["tau_s"],
                               train_yaw_std_with_tau=r3(mdl["train_yaw_std_tau"], 4),
                               a=r3(mdl["a"], 3), b=r3(mdl["b"], 5), c_car_deg=r3(np.degrees(mdl["c_car"]), 3),
                               c_train_drives_deg_p10_p90="/".join(f"{np.degrees(x):.2f}" for x in np.percentile(list(mdl["c_drive_fit"].values()), [10, 90])),
                               train_yaw_std_deg_s=r3(mdl["train_yaw_std"], 4), n_train_samples=mdl["n_train_samples"],
                               train_drives=" ".join(train), test_drives=" ".join(test)))
            # 全走行 (学習側は閾値用、評価側は判定用) の残差
            fold_R = {}
            for did in dl:
                cal = calib_offset(meta[did]["calib"], mdl["a"], mdl["b"], mdl["c_car"], cfg, rng)
                dc = np.radians(cal["delta_c_deg"]) if np.isfinite(cal["delta_c_deg"]) else 0.0
                R = residuals(arrs[did], mdl, mdl["c_car"], mdl["c_car"] + dc, k_roll)
                for ds in ["unc", "cor"]:
                    R[f"fast_{ds}"], R[f"slow_{ds}"] = fast_slow(R[f"dr_{ds}"], arrs[did]["blk"], cfg)
                    R[f"quiet_{ds}"] = R[f"fast_{ds}"]
                R["quiet"] = arrs[did]["quiet"]
                fold_R[did] = (R, cal)
                if did in test:
                    calibs.append(dict(car=car, fold=f, drive_id=did, c_car_deg=r3(np.degrees(mdl["c_car"]), 3),
                                       **{kk: (r3(vv, 4) if isinstance(vv, float) else vv) for kk, vv in cal.items()},
                                       applied=bool(np.isfinite(cal["delta_c_deg"]))))
            for spd in cfg["screen"]["speeds_kmh"]:
                masks = {did: screen_mask(arrs[did], spd, cfg) for did in dl}
                th = {}
                for ds in ["unc", "cor"]:
                    for kind, q in [("fast", cfg["screen"]["q_fast"]), ("slow", cfg["screen"]["q_slow"]), ("quiet", cfg["screen"]["q_quiet"])]:
                        vals = np.concatenate([np.abs(fold_R[d][0][f"{kind}_{ds}"][masks[d][0] & (arrs[d]["quiet"] if kind == "quiet" else True)]) for d in train])
                        vals = vals[np.isfinite(vals)]
                        th[(ds, kind)] = float(np.quantile(vals, q)) if len(vals) > 1000 else np.nan
                        thr_rows.append(dict(car=car, fold=f, speed_kmh=spd, residual=ds, kind=kind, quantile=q, threshold_deg_s=r3(th[(ds, kind)], 4),
                                             train_seconds=round(len(vals) / 100, 1)))
                for did in test:
                    A = arrs[did]
                    R, cal = fold_R[did]
                    msk, r10 = masks[did]
                    if spd == cfg["screen"]["speeds_kmh"][0]:
                        oof[did] = dict(R=R, mask80=msk, r10=r10, fold=f, model=mdl, cal=cal, thr=th)
                    else:
                        oof[did]["mask70"] = msk
                        oof[did]["thr70"] = th
                    for ds in ["unc", "cor"]:
                        for kind, md in [("fast", cfg["screen"]["min_dur_fast_s"]), ("slow", cfg["screen"]["min_dur_slow_s"]), ("quiet", cfg["screen"]["min_dur_fast_s"])]:
                            thr = th[(ds, kind)]
                            if not np.isfinite(thr):
                                continue
                            evs = detect(R[f"{kind}_{ds}"], msk, thr, md, cfg["screen"]["merge_s"], A["blk"])
                            if kind == "quiet":                 # 超え始めの時刻の前 1 s に操舵が静かだったものだけ
                                evs = [(a, b) for a, b in evs if A["quiet"][a]]
                            exc_rows.append(dict(car=car, fold=f, drive_id=did, speed_kmh=spd, residual=ds, kind=kind,
                                                 screened_s=round(msk.sum() / 100, 1), n_events=len(evs)))
                            for i0, i1 in evs:
                                feat = event_features(A, R, i0, i1, kind, thr, r10, cfg, ds, mask=msk)
                                ev_rows.append(dict(car=car, drive_id=did, fold=f, speed_kmh_setting=spd, residual=ds, kind=kind,
                                                    threshold_deg_s=r3(thr, 4), delta_c_deg=r3(cal["delta_c_deg"], 3), **feat))
        log[f"model_{car}_s"] = round(time.time() - t_start, 1)

    # ---------------- 4. まとめ
    syn.to_csv(out / "clock_sync_by_drive.csv", index=False)
    pd.DataFrame(models).to_csv(out / "model_by_fold.csv", index=False)
    pd.DataFrame(calibs).to_csv(out / "offset_calib_by_drive.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(out / "thresholds_by_fold.csv", index=False)
    exc = pd.DataFrame(exc_rows)
    exc.to_csv(out / "exceedance_by_drive.csv", index=False)
    ev = pd.DataFrame(ev_rows)
    seq_iv = seqs[["id", "drive_id", "oxts_t0", "oxts_t1", "cam_start_utc", "cam_end_utc", "split"]].copy()
    seq_iv["cam_t0"] = pd.to_datetime(seq_iv.cam_start_utc, format="ISO8601").astype("int64") / 1e9
    seq_iv["cam_t1"] = pd.to_datetime(seq_iv.cam_end_utc, format="ISO8601").astype("int64") / 1e9
    if len(ev):
        def link(r):
            s = seq_iv[(seq_iv.drive_id == r.drive_id) & (seq_iv.oxts_t0 <= r.t_peak) & (seq_iv.oxts_t1 >= r.t_peak)]
            # カメラ: 閾値超過区間とカメラの区間が重なるシーケンス (event_time の定義。推定開始 + 継続時間は使わない)
            a, b = ET.event_interval(r)
            d = seq_iv[seq_iv.drive_id == r.drive_id]
            ov = np.array([ET.overlap(a, b, x, y) for x, y in zip(d.cam_t0, d.cam_t1)])
            cam = d[ov > 0]
            best = d.iloc[int(np.argmax(ov))] if len(d) and ov.max() > 0 else None
            rel = ET.relation_to_window(a, b, best.cam_t0, best.cam_t1) if best is not None else ""
            return pd.Series(dict(sequence_ids=" ".join(s.id), in_camera_sequences=" ".join(cam.id), split=",".join(sorted(set(s.split))),
                                  camera_overlap_s=round(float(ov.max()), 3) if len(ov) else 0.0, camera_relation=rel,
                                  peak_in_camera=bool(((d.cam_t0 <= r.t_peak) & (d.cam_t1 >= r.t_peak)).any())))
        ev = pd.concat([ev, ev.apply(link, axis=1)], axis=1)
        ev["uncertainty"] = ev.hypothesis.map(UNCERT)
        ev["event_id"] = [f"E{i:04d}" for i in range(len(ev))]
    # 走行ごとの残差のばらつき (80 km/h、補正後) と、車両の中央値との比
    dstd = {d: float(np.nanstd(o["R"]["dr_cor"][o["mask80"] | o["mask70"]])) if (o["mask80"] | o["mask70"]).sum() > 500 else np.nan for d, o in oof.items()}
    dsd = pd.DataFrame(dict(drive_id=list(dstd), dr_std=list(dstd.values())))
    dsd["car"] = dsd.drive_id.map(lambda d: meta[d]["car"])
    dsd["dr_std_ratio_to_car_median"] = dsd.dr_std / dsd.groupby("car").dr_std.transform("median")
    dsd.to_csv(out / "drive_residual_spread.csv", index=False)
    if len(ev):
        ev["drive_dr_std_ratio"] = ev.drive_id.map(dsd.set_index("drive_id").dr_std_ratio_to_car_median).round(2)
    ev.to_csv(out / "events_all.csv", index=False)
    cand = ev[(ev.residual == "cor")].copy() if len(ev) else ev
    cand.to_csv(out / "candidates.csv", index=False)
    uniq = unique_candidates(cand)
    rec = recurrence(uniq, arrs, oof, meta, cfg)
    if len(rec):
        agg = rec.groupby("event_id").agg(n_other_passes=("drive_id", "size"), n_other_drives=("drive_id", "nunique"),
                                          other_pass_slow_mean_median=("slow_cor_mean", "median"),
                                          n_other_same_sign_over_half_thr=("same_sign_over_half_thr", "sum"))
        uniq = uniq.merge(agg, left_on="event_id", right_index=True, how="left")
    rec.to_csv(out / "recurrence.csv", index=False)
    uniq.to_csv(out / "candidates_unique.csv", index=False)

    # 残差の分布 (交差検証の評価側、高速・直進)
    dist = []
    for car in sorted(syn.car.unique()):
        for spd, mk in [(80, "mask80"), (70, "mask70")]:
            for ds in ["unc", "cor"]:
                for kind in ["dr", "fast", "slow", "day", "dayvd"]:
                    vals = np.concatenate([oof[d]["R"][f"{kind}_{ds}"][oof[d][mk]] for d in oof if meta[d]["car"] == car])
                    vals = vals[np.isfinite(vals)]
                    if not len(vals):
                        continue
                    a = np.abs(vals)
                    dist.append(dict(car=car, speed_kmh=spd, residual=ds, signal=kind, seconds=round(len(vals) / 100, 1), mean=r3(vals.mean(), 4),
                                     std=r3(vals.std(), 4), abs_p50=r3(np.quantile(a, .5), 4), abs_p95=r3(np.quantile(a, .95), 4),
                                     abs_p99=r3(np.quantile(a, .99), 4), abs_p999=r3(np.quantile(a, .999), 4), abs_max=r3(a.max(), 4)))
    pd.DataFrame(dist).to_csv(out / "residual_distribution.csv", index=False)

    # 時系列 (20 Hz、補正前・補正後) を走行ごとに保存
    for did, o in oof.items():
        A, R = arrs[did], o["R"]
        sl = slice(None, None, 5)
        np.savez_compressed(out / "residuals" / f"{did}.npz", t=A["t"][sl], blk=A["blk"][sl], v=A["v"][sl], r_deg_s=np.degrees(A["r"][sl]),
                            r_pred_unc=np.degrees(R["rp_unc"][sl]).astype(np.float32), r_pred_cor=np.degrees(R["rp_cor"][sl]).astype(np.float32),
                            dr_unc=R["dr_unc"][sl].astype(np.float32), dr_cor=R["dr_cor"][sl].astype(np.float32),
                            day_cor=R["day_cor"][sl].astype(np.float32), dayvd_cor=R["dayvd_cor"][sl].astype(np.float32),
                            beta=A["beta"][sl], beta_unc=A["beta_unc"][sl], steer_deg=np.degrees(A["steer"][sl]).astype(np.float32),
                            torque=A["torque"][sl], turn=A["turn"][sl].astype(np.int8), mask80=o["mask80"][sl], mask70=o["mask70"][sl])

    # 対象の時間 (重なりは除いてある: 走行ごとに時刻で 1 本にした)
    cov = []
    for car in sorted(syn.car.unique()):
        ds_ = [d for d in oof if meta[d]["car"] == car]
        cov.append(dict(car=car, drives=len(ds_), oxts_unique_min=round(sum(len(arrs[d]["t"]) for d in ds_) / 6000, 1),
                        screened_min_80=round(sum(oof[d]["mask80"].sum() for d in ds_) / 6000, 1),
                        screened_min_70=round(sum(oof[d]["mask70"].sum() for d in ds_) / 6000, 1),
                        drives_with_80=int(sum(oof[d]["mask80"].sum() > 0 for d in ds_)),
                        beta_ok_min_80=round(sum((oof[d]["mask80"] & (arrs[d]["beta_unc"] <= cfg["screen"]["beta_unc_good_deg"])).sum() for d in ds_) / 6000, 1)))
    cov = pd.DataFrame(cov)
    cov.to_csv(out / "screened_time.csv", index=False)

    # 上位候補と対照、図
    top, ctrl = select_top(ev, arrs, oof, meta, cfg, rng)
    top.to_csv(out / "top_candidates.csv", index=False)
    ctrl.to_csv(out / "controls.csv", index=False)
    plot_all(top, ctrl, arrs, oof, cfg, out / "plots")
    plot_sync(syn, out / "plots")
    plot_dist(oof, arrs, meta, out / "plots")
    phase3(ev, top, cfg).to_csv(out / "phase3_estimate.csv", index=False)

    summ = dict(n_drives=len(res), n_errors=len(errs), errors=[dict(drive_id=e["drive_id"], error=e["error"]) for e in errs],
                screened=cov.to_dict("records"), log=log, k_roll_used=k_roll)
    if len(ev):
        summ["events"] = ev.groupby(["speed_kmh_setting", "residual", "kind", "hypothesis"]).size().reset_index(name="n").to_dict("records")
        e80 = exc[(exc.residual == "cor") & (exc.speed_kmh == 80)]
        summ["exceedance_per_hour_80_cor"] = (e80.groupby(["car", "kind"]).n_events.sum() / (e80.groupby(["car", "kind"]).screened_s.sum() / 3600)).round(2).reset_index(name="per_hour").to_dict("records")
    summ["elapsed_s"] = round(time.time() - t_start, 1)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=str))

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    here = Path(__file__).parent
    manifest = dict(run_id=run_id, created_local=datetime.now().isoformat(timespec="seconds"), git_head=head,
                    note="スクリプト・設定は未コミット。sha256 で版を特定する。モデルの版 = full_screening.py の sha256",
                    scripts={str(p): sha(p) for p in [Path(__file__), here / "signal_survey.py", here / "drive_validation.py"]},
                    configs={str(p): sha(p) for p in [args.config, Path("configs/zod/signal_survey.yaml"), Path("configs/zod/drive_validation.yaml")]},
                    inventory_run=str(inv), inventory_sha256={f: sha(inv / f) for f in ["sequences.csv", "drives.csv"]},
                    config_effective=cfg, args=vars(args), k_roll=k_roll, python=sys.version, numpy=np.__version__, pandas=pd.__version__)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    print(f"{len(res)} 走行 / 失敗 {len(errs)}、事象 {len(ev)}、{summ['elapsed_s']} s → {out}")
    return 0


# ================================================================ 上位候補・対照・図・Phase 3
def recurrence(uniq, arrs, oof, meta, cfg):
    """候補の地点 (半径 radius_m) を同じ向きに通ったほかの走行の、持続成分 (補正後) の平均。"""
    rc = cfg["recurrence"]
    rows = []
    for _, r in uniq.iterrows():
        ip = int(r.ip)
        A0 = arrs[r.drive_id]
        h0 = float(A0["heading"][ip])
        thr = oof[r.drive_id]["thr"][("cor", "slow")]
        for d, o in oof.items():
            A = arrs[d]
            m = (o["mask80"] | o["mask70"])
            dist = D.dist_m(r.lat, r.lon, A["lat"].astype(float), A["lon"].astype(float))
            dh = np.abs((A["heading"].astype(float) - h0 + 180) % 360 - 180)
            sel = m & (dist <= rc["radius_m"]) & (dh <= rc["same_direction_deg"])
            if d == r.drive_id:                     # 同じ走行では、候補の時刻の前後 60 s を除く (別の通過だけ)
                sel &= np.abs(A["t"] - A["t"][ip]) > 60
            if sel.sum() < rc["min_samples"]:
                continue
            sm = o["R"]["slow_cor"][sel]
            rows.append(dict(event_id=r.event_id, event_drive_id=r.drive_id, drive_id=d, car=meta[d]["car"], date=utc(A["t"][sel][0])[:10],
                             seconds=round(sel.sum() / 100, 1), slow_cor_mean=r3(np.nanmean(sm), 3), slow_cor_absmax=r3(np.nanmax(np.abs(sm)), 3),
                             event_slow_sign=int(np.sign(r.dr_peak_deg_s)),
                             same_sign_over_half_thr=bool(np.sign(np.nanmean(sm)) == np.sign(r.dr_peak_deg_s) and abs(np.nanmean(sm)) > 0.5 * thr)))
    return pd.DataFrame(rows)


def unique_candidates(c: pd.DataFrame) -> pd.DataFrame:
    """補正後の事象を、同じ走行で始まりが 2 s 以内のものを 1 つにまとめる (80 の設定・閾値比の大きいものを代表に)。
    どの設定・種類で検出されたかを列に残す。"""
    if not len(c):
        return c
    c = c.sort_values(["drive_id", "t_onset"]).copy()
    gid, last = [], {}
    k = -1
    for _, r in c.iterrows():
        if r.drive_id in last and r.t_onset - last[r.drive_id][1] <= 2.0:
            g = last[r.drive_id][0]
        else:
            k += 1
            g = k
        last[r.drive_id] = (g, r.t_onset)
        gid.append(g)
    c["group"] = gid
    found = c.groupby("group").apply(lambda g: " ".join(sorted({f"{int(a)}:{b}" for a, b in zip(g.speed_kmh_setting, g.kind)})), include_groups=False)
    rep = c.sort_values(["speed_kmh_setting", "peak_over_thr"], ascending=[False, False]).drop_duplicates("group").copy()
    rep["detected_as"] = rep.group.map(found)
    rep["only_at_70_setting"] = ~rep.detected_as.str.contains("80:")
    rep["below_80kmh"] = rep.speed_kmh < 80
    return rep.sort_values("t_onset")



def select_top(ev, arrs, oof, meta, cfg, rng):
    tc = cfg["top"]
    if not len(ev):
        return pd.DataFrame(), pd.DataFrame()
    c = ev[(ev.residual == "cor")].copy()
    # 同じ時刻の事象 (80 と 70、突発と持続) は 1 つに: 走行と始まりの時刻 (1 s) でまとめ、80 の設定を優先
    c["key"] = c.drive_id + "_" + (c.t_onset.round(0)).astype(str)
    c = c.sort_values(["speed_kmh_setting", "peak_over_thr"], ascending=[False, False]).drop_duplicates("key")
    pri = {"ヨーの変化が先 (外乱の候補)": 0, "持続的なずれ (当て舵・路面の傾き・ゼロ点・横風)": 1}
    c["priority"] = c.hypothesis.map(pri).fillna(9)
    top = c[c.priority < 9].sort_values(["priority", "peak_over_thr"], ascending=[True, False]).head(tc["n"]).copy()
    top["rank"] = range(1, len(top) + 1)
    rows = []
    for _, r in top.iterrows():
        car = r.car
        o = oof[r.drive_id]
        thr = o["thr"][("cor", "fast")]
        best = None
        cands = [d for d in oof if meta[d]["car"] == car and d != r.drive_id] or [r.drive_id]
        for d in rng.permutation(cands):
            A, R, m = arrs[d], oof[d]["R"], oof[d]["mask80"] | oof[d].get("mask70", oof[d]["mask80"])
            n = int(tc["control_len_s"] * 100)
            ok = pd.Series(m.astype(float)).rolling(n).min().to_numpy() == 1
            sp = pd.Series(np.abs(A["v"] * 3.6 - r.speed_kmh) <= tc["control_speed_tol_kmh"]).rolling(n).min().to_numpy() == 1
            fx = pd.Series(np.abs(R["fast_cor"])).rolling(n).max().to_numpy() < tc["control_max_fast_ratio"] * thr
            idx = np.where(ok & sp & fx)[0]
            if len(idx):
                j = int(rng.choice(idx)) - n // 2
                best = (d, j)
                break
        if best:
            d, j = best
            A = arrs[d]
            R = oof[d]["R"]
            sl = slice(j - n // 2, j + n // 2)
            rows.append(dict(rank=r["rank"], event_id=r.event_id, control_drive_id=d, control_t_center_utc=utc(A["t"][j]), control_t_center=float(A["t"][j]),
                             control_idx=j, speed_kmh=r3(A["v"][j] * 3.6, 1), dr_cor_abs_max=r3(np.nanmax(np.abs(R["dr_cor"][sl])), 3),
                             fast_cor_abs_max=r3(np.nanmax(np.abs(R["fast_cor"][sl])), 3), steer_range_deg=r3(np.degrees(np.ptp(A["steer"][sl])), 3),
                             torque_range_nm=r3(np.ptp(A["torque"][sl]), 3)))
    return top, pd.DataFrame(rows)


def plot_window(ax, A, R, i_c, cfg, title, thr=None, ion=None, run=None):
    """run = (i0, i1): 閾値超過区間 (帯で描く)。ion: 推定開始 (赤い破線)。"""
    pre, post = int(cfg["top"]["plot_pre_s"] * 100), int(cfg["top"]["plot_post_s"] * 100)
    bm = np.where(A["blk"] == A["blk"][i_c])[0]
    sl = slice(max(bm[0], i_c - pre), min(bm[-1] + 1, i_c + post))
    tt = A["t"][sl] - A["t"][i_c]
    ax[0].plot(tt, A["v"][sl] * 3.6, color="#333"); ax[0].set_ylabel("速さ\nkm/h")
    ax[1].plot(tt, np.degrees(A["r"][sl]), label="r (OXTS)", color="#1f4e9c")
    ax[1].plot(tt, np.degrees(R["rp_cor"][sl]), label="r_pred (補正後)", color="#c0392b", lw=1)
    ax[1].plot(tt, np.degrees(R["rp_unc"][sl]), label="r_pred (補正前)", color="#e59866", lw=0.8, ls="--")
    ax[1].set_ylabel("ヨー\ndeg/s"); ax[1].legend(fontsize=7, loc="upper right")
    ax[2].plot(tt, R["dr_cor"][sl], label="Δr 補正後", color="#1f4e9c")
    ax[2].plot(tt, R["dr_unc"][sl], label="Δr 補正前", color="#7f8c8d", lw=0.8, ls="--")
    ax[2].plot(tt, R["fast_cor"][sl], label="突発成分", color="#16a085", lw=0.8)
    if thr is not None:
        for s in (1, -1):
            ax[2].axhline(s * thr, color="#c0392b", lw=0.6, ls=":")
    ax[2].set_ylabel("ヨー残差\ndeg/s"); ax[2].legend(fontsize=7, loc="upper right")
    ax[3].plot(tt, R["day_cor"][sl], label="Δay OXTS", color="#1f4e9c")
    ax[3].plot(tt, R["dayvd_cor"][sl], label="Δay 車両 IMU", color="#e67e22", lw=0.8)
    ax[3].set_ylabel("横加速度の\n残差 m/s²"); ax[3].legend(fontsize=7, loc="upper right")
    b = A["beta"][sl].astype(float); u = A["beta_unc"][sl].astype(float)
    ax[4].plot(tt, b, color="#8e44ad"); ax[4].fill_between(tt, b - u, b + u, color="#8e44ad", alpha=0.2, lw=0)
    ax[4].set_ylabel("β deg\n(±不確かさ)")
    st = np.degrees(A["steer"][sl].astype(float))
    ax[5].plot(tt, st - np.nanmedian(st), color="#2c3e50", label="舵角 − 中央値 [deg]")
    ax5 = ax[5].twinx(); ax5.plot(tt, A["torque"][sl], color="#d35400", lw=0.8); ax5.set_ylabel("トルク N·m", fontsize=8, color="#d35400")
    on = A["turn"][sl] != 0
    if on.any():
        ax[5].fill_between(tt, *ax[5].get_ylim(), where=on, color="#f1c40f", alpha=0.3, label="方向指示器")
    ax[5].set_ylabel("舵角\ndeg"); ax[5].legend(fontsize=7, loc="upper left")
    ax[5].set_xlabel("基準の時刻からの秒")
    for a in ax:
        a.axvline(0, color="#999", lw=0.6)
        if ion is not None:
            a.axvline(A["t"][ion] - A["t"][i_c], color="#c0392b", lw=0.6, ls="--")
        if run is not None:
            a.axvspan(A["t"][run[0]] - A["t"][i_c], A["t"][run[1] - 1] + ET.DT - A["t"][i_c], color="#c0392b", alpha=0.12, lw=0)
        a.grid(alpha=0.3)
    ax[0].set_title(title, fontsize=9)


def plot_all(top, ctrl, arrs, oof, cfg, pdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
    for _, r in top.iterrows():
        A, R = arrs[r.drive_id], oof[r.drive_id]["R"]
        c = ctrl[ctrl["rank"] == r["rank"]]
        fig, ax = plt.subplots(6, 2 if len(c) else 1, figsize=(14 if len(c) else 8, 12), sharex="col", squeeze=False)
        thr = r.threshold_deg_s if r.kind == "fast" else None
        plot_window(ax[:, 0], A, R, int(r.ip), cfg, f"#{r['rank']} {r.event_id} {r.car} {r.drive_id}\n{r.t_onset_utc} {r.speed_kmh} km/h  {r.kind}  {r.hypothesis}\n帯: 閾値超過区間  破線: 推定開始", thr, int(r.ion), (int(r.i0), int(r.i1)))
        if len(c):
            c = c.iloc[0]
            plot_window(ax[:, 1], arrs[c.control_drive_id], oof[c.control_drive_id]["R"], int(c.control_idx), cfg,
                        f"対照 {c.control_drive_id}\n{c.control_t_center_utc} {c.speed_kmh} km/h", r.threshold_deg_s if r.kind == "fast" else None)
        fig.tight_layout()
        fig.savefig(pdir / f"top{int(r['rank']):02d}_{r.event_id}.png", dpi=100)
        plt.close(fig)


def plot_sync(syn, pdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for car, s in syn.groupby("car"):
        ax[0].errorbar(s.clock_ay_s, s.steer_to_oxts_yaw_lag_s, xerr=s.clock_ay_se_s.fillna(0), fmt="o", ms=4, label=car, alpha=0.7)
        ax[1].scatter(s.clock_ay_s, s.steer_to_vd_ay_lag_s, s=14, label=car, alpha=0.7)
    ax[0].set_xlabel("時計のずれ (車両 − OXTS、横加速度) [s]"); ax[0].set_ylabel("舵角 → OXTS のヨーの遅れ [s]")
    ax[1].set_xlabel("時計のずれ (車両 − OXTS、横加速度) [s]"); ax[1].set_ylabel("舵角 → 車両 IMU の横加速度の遅れ [s]")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    ax[0].set_title("OXTS 側の遅れは時計のずれに沿って動く", fontsize=10)
    ax[1].set_title("車両データの中だけの遅れは時計のずれと関係しない", fontsize=10)
    fig.tight_layout(); fig.savefig(pdir / "clock_sync.png", dpi=110); plt.close(fig)


def plot_dist(oof, arrs, meta, pdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
    cars = sorted({meta[d]["car"] for d in oof})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for car in cars:
        for a, (key, ds, rng_) in zip(ax, [("dr", "cor", 1.5), ("fast", "cor", 1.0), ("slow", "unc", 1.5)]):
            v = np.concatenate([oof[d]["R"][f"{key}_{ds}"][oof[d]["mask80"]] for d in oof if meta[d]["car"] == car])
            v = v[np.isfinite(v)]
            a.hist(np.clip(v, -rng_, rng_), bins=120, range=(-rng_, rng_), histtype="step", lw=1.3, density=True, label=f"{car} ({len(v) / 6000:.0f} 分)")
            a.set_yscale("log")
    for a, t in zip(ax, ["ヨー残差 (補正後) [deg/s]", "突発成分 (補正後) [deg/s]", "持続成分 5 s 平均 (補正前) [deg/s]"]):
        a.set_title(t + "\n80 km/h 超・直進、交差検証の評価側", fontsize=9); a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(pdir / "residual_distribution.png", dpi=110); plt.close(fig)


def phase3(ev, top, cfg):
    """Phase 3 で要る観測所・日時・量の見積もり (手元の観測所一覧だけを使う。取得はしない)。"""
    import re
    rows = []
    if not len(ev):
        return pd.DataFrame()
    c = ev[ev.residual == "cor"].drop_duplicates(["drive_id", "t_onset"])
    j = json.loads(Path("data/zod_weather/smhi/parameter_4.json").read_text())
    t22a, t22b = datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000, datetime(2022, 9, 30, tzinfo=timezone.utc).timestamp() * 1000
    sm = pd.DataFrame([dict(sid=str(s["key"]), lat=s["latitude"], lon=s["longitude"]) for s in j["station"] if s["from"] < t22a and s["to"] > t22b])
    dw = []
    for line in Path("data/zod_weather/dwd/zehn_min_ff_Beschreibung_Stationen.txt").read_bytes().decode("latin-1").splitlines()[2:]:
        m = re.match(r"\s*(\d+)\s+(\d{8})\s+(\d{8})\s+(-?\d+)\s+([\d.]+)\s+([\d.]+)", line)
        if m and m.group(2) < "20220101" and m.group(3) > "20220930":
            dw.append(dict(sid=m.group(1).zfill(5), lat=float(m.group(5)), lon=float(m.group(6))))
    dw = pd.DataFrame(dw)
    # 手元に取得済みの観測所 (前回の PoC。data/zod_weather/ のファイル名から)
    have = {("SMHI", p.name.split("_s")[1].split("_")[0]) for p in Path("data/zod_weather/smhi").glob("p4_s*_corrected-archive.csv")}
    have |= {("DWD", p.name.split("_")[2]) for p in Path("data/zod_weather/dwd").glob("10minutenwerte_wind_*_hist.zip")}
    allst = pd.concat([sm.assign(src="SMHI"), dw.assign(src="DWD")])
    loc = allst[[(a, b) in have for a, b in zip(allst.src, allst.sid)]]
    for _, r in c.iterrows():
        dl = D.dist_m(r.lat, r.lon, loc.lat.to_numpy(), loc.lon.to_numpy()) / 1000
        i = int(np.argmin(dl))
        rows.append(dict(event_id=r.event_id, hypothesis=r.hypothesis, in_top=r.event_id in set(top.event_id), date_utc=r.t_onset_utc[:10],
                         source="取得済み (" + loc.src.iloc[i] + ")", radius_km=None, n_stations=None, nearest_km=round(float(dl[i]), 1),
                         station_ids=loc.sid.iloc[i]))
        for src, st in [("SMHI", sm), ("DWD", dw)]:
            d = D.dist_m(r.lat, r.lon, st.lat.to_numpy(), st.lon.to_numpy()) / 1000
            for rad in cfg["phase3"]["radii_km"]:
                ids = st.sid[d <= rad].tolist()
                if ids:
                    rows.append(dict(event_id=r.event_id, hypothesis=r.hypothesis, in_top=r.event_id in set(top.event_id), date_utc=r.t_onset_utc[:10],
                                     source=src, radius_km=rad, n_stations=len(ids), nearest_km=round(float(d.min()), 1), station_ids=" ".join(ids)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.exit(main())
