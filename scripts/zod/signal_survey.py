#!/usr/bin/env python3
"""ZOD Sequences (mini) の車両データ・OXTS・前方カメラについて、横風の外乱推定に使う前に
単位・座標系・周期・時刻の対応を、信号どうしの照合で確かめる。ネットワークは使わない。

  uv run --offline --extra viz --with "zod==0.8.0" --with "dataclass-wizard==0.39.1" \\
      python scripts/zod/signal_survey.py --root raw_data/zod --version mini

  (プロジェクトの環境に zod を上乗せする。pyproject.toml・uv.lock は変わらない)

設定: configs/zod/signal_survey.yaml。結果の説明: docs/zod_signals.md

出力 (<out>/、既定 out/zod/signal_survey/<version>/):
  relations.csv     信号の組ごとの相関・傾き・時刻のずれ (単位・符号・時計の確認)
  camera_sync.csv   画面の横の流れと OXTS のヨーレートの相互相関 (カメラの時刻の確認)
  model_fit.json    走行全体の車両データで当てはめた単軌道モデル (舵角 → 横加速度)
  residuals.csv     シーケンスごとの残差の大きさ (OXTS の 50 s: ヨー残差、走行全体: 横加速度の残差)
  conditions.csv    シーケンスごとの走行条件 (日時・場所・速度・走行全体の長さなど)
  top_events.csv    走行全体で横加速度の残差が大きい区間 (確認用。横風とは判断していない)
  plots/            時系列の図、カメラの静止画
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, sosfiltfilt

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc).timestamp()
TZ = {"SE": "Europe/Stockholm", "DE": "Europe/Berlin", "FR": "Europe/Paris", "NO": "Europe/Oslo",
      "FI": "Europe/Helsinki", "DK": "Europe/Copenhagen", "PL": "Europe/Warsaw", "GB": "Europe/London",
      "IT": "Europe/Rome", "ES": "Europe/Madrid", "NL": "Europe/Amsterdam", "BE": "Europe/Brussels",
      "AT": "Europe/Vienna", "CH": "Europe/Zurich", "HU": "Europe/Budapest", "CZ": "Europe/Prague"}
CTRL, EGO = "ego_vehicle_controls", "ego_vehicle_data"
K_STEER = "steering_wheel_angle/angle/radians/value"
K_TORQUE = "steer_wheel_torque/torque/newton_meters/value"
K_TURN = "turn_indicator_status/state"
K_LONV = "lon_vel_data/velocity/meters_per_second/value"
K_LATV = "lat_vel_data/velocity/meters_per_second/value"
K_LATA = "lat_acc_data/acceleration/meters_per_second2/value"
K_LONA = "lon_acc_data/acceleration/meters_per_second2/value"
K_ROLL = "roll_rate_data/angle_rate/radians_per_second/value"
K_TS = "timestamp/nanoseconds/value"


# ---------------------------------------------------------------- 読み込み
def load_oxts(path: Path) -> pd.DataFrame:
    import h5py
    with h5py.File(path, "r") as f:
        cols = {k: f[k][()] for k in f if isinstance(f[k], h5py.Dataset) and f[k].shape == f["timestamp"].shape
                and f[k].dtype.kind in "fiu"}
        d = pd.DataFrame(cols)
    d["t"] = GPS_EPOCH + d.timestamp + d.leapSeconds         # SDK (zod/data_classes/ego_motion.py) と同じ式
    return d


def load_vehicle(path: Path) -> dict[str, pd.DataFrame]:
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        for g in (CTRL, EGO, "satellite"):
            grp = f[g]
            cols = {}
            grp.visititems(lambda n, o: cols.__setitem__(n, o[()]) if isinstance(o, h5py.Dataset) else None)
            n = len(cols[K_TS])
            d = pd.DataFrame({k: v for k, v in cols.items() if len(v) == n})
            d["t"] = d[K_TS].astype(np.float64) / 1e9
            out[g] = d
    return out


# ---------------------------------------------------------------- 小道具
def lowpass(x, fs, fc, order=2):
    return sosfiltfilt(butter(order, fc, fs=fs, output="sos"), x)


def bandpass(x, fs, band, order=2):
    return sosfiltfilt(butter(order, band, btype="band", fs=fs, output="sos"), x)


def lag_scan(t, a, tb, b, lag_range, step):
    """b(t + lag) と a(t) の相関を lag ごとに求める。lag > 0 なら b の記録が a より遅れている。"""
    lags = np.arange(-lag_range, lag_range + step / 2, step)
    c = []
    for L in lags:
        bb = np.interp(t + L, tb, b, left=np.nan, right=np.nan)
        m = np.isfinite(bb)
        c.append(np.corrcoef(a[m], bb[m])[0, 1])
    c = np.array(c)
    i = int(np.nanargmax(np.abs(c)))
    return lags, c, float(lags[i]), float(c[i]), float(c[np.argmin(np.abs(lags))])


def moving(x, n):
    n = max(int(round(n)), 1)
    return pd.Series(x).rolling(n, center=True, min_periods=1).mean().to_numpy()


# ---------------------------------------------------------------- 1. 信号どうしの関係
def relations(sid, ox, vd, cfg) -> list[dict]:
    t = ox.t.to_numpy()
    fs = 100.0
    c, e = vd[CTRL], vd[EGO]
    rows = []

    def add(name, a, tb, b, fc, unit_note, lag_range=1.0):
        a_f = lowpass(a, fs, fc)
        b_on_t = np.interp(t, tb, b)
        b_f = lowpass(b_on_t, fs, fc)
        _, _, lag, cbest, c0 = lag_scan(t, a_f, t, b_f, lag_range, 0.01)
        slope = np.polyfit(b_f, a_f, 1)[0]
        rows.append(dict(id=sid, relation=name, lowpass_hz=fc, corr_at_0=round(c0, 4), best_lag_s=lag,
                         corr_at_best=round(cbest, 4), slope_a_per_b=round(float(slope), 4), note=unit_note))

    hd = np.degrees(np.unwrap(np.radians(ox.heading.to_numpy())))
    add("OXTS d(heading)/dt [deg/s] ~ OXTS angularRateZ", np.gradient(hd, t), t, ox.angularRateZ.to_numpy(), 5,
        "傾き 1・相関 +1 なら angularRateZ は deg/s で、方位 (北から時計回り) の増える向き = 右回りが正")
    add("OXTS slipAngle [deg] ~ deg(atan2(velLateral, velForward))", ox.slipAngle.to_numpy(), t,
        np.degrees(np.arctan2(ox.velLateral, ox.velForward)).to_numpy(), 5, "slipAngle は横速度と前後速度の角 [deg]")
    add("OXTS speed ~ hypot(velForward, velLateral)", ox.speed.to_numpy(), t,
        np.hypot(ox.velForward, ox.velLateral).to_numpy(), 5, "")
    add("OXTS accelerationY ~ velForward·rad(angularRateZ)", ox.accelerationY.to_numpy(), t,
        (ox.velForward * np.radians(ox.angularRateZ)).to_numpy(), 1, "正の相関なら y は右向き (z 下向きの右手系)")
    add("OXTS accelerationX ~ d(velForward)/dt", ox.accelerationX.to_numpy(), t,
        np.gradient(ox.velForward.to_numpy(), t), 1, "")
    add("vehicle lon_vel ~ OXTS speed", ox.speed.to_numpy(), e.t.to_numpy(), e[K_LONV].to_numpy(float), 2,
        "時計の対応と速度の倍率")
    add("vehicle lat_acc ~ OXTS accelerationY", ox.accelerationY.to_numpy(), e.t.to_numpy(), e[K_LATA].to_numpy(float), 2,
        "負の相関なら車両データの横は左向きが正")
    add("vehicle lon_acc ~ OXTS accelerationX", ox.accelerationX.to_numpy(), e.t.to_numpy(), e[K_LONA].to_numpy(float), 2, "")
    add("vehicle roll_rate [rad/s] ~ OXTS angularRateX", ox.angularRateX.to_numpy(), e.t.to_numpy(), e[K_ROLL].to_numpy(float), 3,
        "傾き 57.3 なら OXTS は deg/s、車両データは rad/s")
    add("vehicle lat_vel ~ OXTS velLateral", ox.velLateral.to_numpy(), e.t.to_numpy(), e[K_LATV].to_numpy(float), 2,
        "相関が低ければ車両データの横速度は OXTS と別物 (推定値の可能性)")
    add("vehicle steering_wheel_angle ~ OXTS angularRateZ", ox.angularRateZ.to_numpy(), c.t.to_numpy(), c[K_STEER].to_numpy(float), 2,
        "負の相関なら舵角は左が正 (SDK のコメント「反時計回りが正」と合う)。lag < 0 ならヨーが舵角より遅れる")
    return rows


# ---------------------------------------------------------------- 2. カメラの時刻
def load_strips(frames, ca):
    """各フレームの地平線のまわりの帯を、1/4 の大きさで読み、横方向の輝度の勾配にする (明るさの変化に強くするため)。"""
    from PIL import Image
    out, w = [], None
    for fr in frames:
        im = Image.open(fr.filepath)
        im.draft("L", tuple(ca["draft_size"]))
        a = np.asarray(im.convert("L"), dtype=np.float32)
        h, w = a.shape
        a = a[int(ca["rows"][0] * h): int(ca["rows"][1] * h), int(ca["cols"][0] * w): int(ca["cols"][1] * w)]
        gx = np.diff(a, axis=1)[:-1]
        out.append(gx - gx.mean())
    return out, w


def phase_shift(a, b, lim):
    """b に対する a の横のずれ [px] (位相相関。探す範囲を ±lim に限る)。"""
    win = np.hanning(a.shape[1])[None, :] * np.hanning(a.shape[0])[:, None]
    F = np.fft.fft2(a * win) * np.conj(np.fft.fft2(b * win))
    F /= np.abs(F) + 1e-9
    prof = np.real(np.fft.ifft2(F)).sum(axis=0)
    cand = np.r_[0:lim + 1, len(prof) - lim:len(prof)]
    k = int(cand[np.argmax(prof[cand])])
    km, kp = prof[k - 1], prof[(k + 1) % len(prof)]
    d = km - 2 * prof[k] + kp
    s = k + (0.5 * (km - kp) / d if d != 0 else 0.0)
    return s - len(prof) if s > len(prof) / 2 else s


def camera_sync(sid, info, ox, calib, cfg, plots: Path) -> dict:
    """k フレーム離れた画像の横のずれと、同じ区間の OXTS の方位の変化を、時刻をずらしながら比べる。"""
    ca = cfg["camera_alignment"]
    frames = info.camera_frames["front_blur"]
    tf = np.array([f.time.timestamp() for f in frames])
    k = int(ca["frame_baseline"])
    strips, w_small = load_strips(frames, ca)
    sh_raw = np.array([phase_shift(strips[i], strips[i + k], int(ca["max_shift_px"])) for i in range(len(strips) - k)])
    from scipy.signal import medfilt
    sh = medfilt(sh_raw, int(ca["median_filter"]))             # 位相相関の外れたピーク (1〜2 点の飛び) を除く
    ta, tb = tf[:-k], tf[k:]
    t = ox.t.to_numpy()
    psi = np.unwrap(np.radians(ox.heading.to_numpy()))          # 方位 [rad]、右回りが正
    lags = np.arange(-ca["lag_range_s"], ca["lag_range_s"] + ca["lag_step_s"] / 2, ca["lag_step_s"])
    cs = np.array([np.corrcoef(sh, np.interp(tb + L, t, psi) - np.interp(ta + L, t, psi))[0, 1] for L in lags])
    i = int(np.nanargmax(np.abs(cs)))
    lag = float(lags[i])
    dpsi = np.interp(tb + lag, t, psi) - np.interp(ta + lag, t, psi)
    slope = float(np.polyfit(dpsi, sh, 1)[0])
    near = lags[np.abs(cs) >= np.abs(cs[i]) - 0.01]
    fx_small = calib["FC"]["intrinsics"][0][0] * w_small / calib["FC"]["image_dimensions"][0]
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
        ax[0].plot(ta - tf[0], sh / fx_small, label="画面の横のずれ / fx [rad]")
        ax[0].plot(ta - tf[0], np.interp(tb, t, psi) - np.interp(ta, t, psi), label="OXTS の方位の変化 (ずれ 0)", alpha=0.8)
        ax[0].set_xlabel("カメラの最初のフレームからの時刻 [s]"); ax[0].set_ylabel(f"{k} フレームの間の変化 [rad]"); ax[0].legend(fontsize=8)
        ax[1].plot(lags, cs); ax[1].axvline(0, color="gray", lw=0.5)
        ax[1].set_xlabel("ずれ [s] (正: OXTS の記録が映像より遅れる)"); ax[1].set_ylabel("相関")
        fig.suptitle(f"{sid}: 映像と OXTS の時刻の対応")
        fig.tight_layout(); fig.savefig(plots / f"{sid}_camera_sync.png", dpi=110); plt.close(fig)
    except Exception as ex:
        print("図を描けなかった:", ex)
    return dict(id=sid, n_frames=len(frames), frame_rate_hz=round(1 / np.median(np.diff(tf)), 4),
                frame_baseline=k, heading_change_std_deg=round(float(np.degrees(dpsi.std())), 3),
                n_outlier_frames=int((np.abs(sh_raw - sh) > 0.02 * fx_small).sum()),
                corr_at_0=round(float(cs[np.argmin(np.abs(lags))]), 4), best_lag_s=round(lag, 3),
                corr_at_best=round(float(cs[i]), 4), lags_within_0p01_of_best=f"{near.min():+.2f}..{near.max():+.2f}",
                shift_px_per_rad=round(slope, 1), expected_px_per_rad_fx=round(fx_small, 1),
                camera_t0_minus_oxts_t0_s=round(tf[0] - t[0], 3), oxts_t1_minus_camera_t1_s=round(t[-1] - tf[-1], 3))


# ---------------------------------------------------------------- 3. 単軌道モデル (走行全体の車両データ)
def drive_grid(vd, cfg) -> pd.DataFrame:
    """走行全体の車両データを等間隔にそろえる。間隔が長く空いた所は NaN。"""
    fs = cfg["grid_hz"]
    c, e = vd[CTRL], vd[EGO]
    t0, t1 = max(c.t.iloc[0], e.t.iloc[0]), min(c.t.iloc[-1], e.t.iloc[-1])
    tg = np.arange(t0, t1, 1 / fs)
    g = pd.DataFrame({"t": tg})

    def put(src, key, name, smooth):
        ts = src.t.to_numpy()
        x = moving(src[key].to_numpy(float), smooth * (1 / np.median(np.diff(ts))))
        g[name] = np.interp(tg, ts, x)
        i = np.searchsorted(ts, tg).clip(1, len(ts) - 1)
        gap = ts[i] - ts[i - 1]
        g.loc[gap > cfg["max_gap_s"], name] = np.nan

    sm = cfg["smooth_s"]
    put(e, K_LONV, "v", sm["speed"])
    put(e, K_LATA, "ay", sm["accel"])
    put(c, K_STEER, "steer", sm["steer"])
    put(c, K_TORQUE, "torque", sm["steer"])
    g["turn"] = np.interp(tg, c.t, c[K_TURN].to_numpy(float)).round()
    g["jerk"] = np.gradient(g.ay.to_numpy(), 1 / fs)
    return g


def fit_model(grids: list[pd.DataFrame], cfg, note: str) -> dict:
    """δsw = a·(ay/v²) + b·ay + c  (docs/crosswind_civic_feasibility.md (4.1) の r = ay/v 版)。"""
    f = cfg["fit"]
    d = pd.concat(grids)
    m = (d.v > f["min_speed"]) & (d.jerk.abs() < f["max_jerk"]) & (d.ay.abs() > f["min_abs_ay"]) & (d.turn == 0)
    d = d[m].dropna(subset=["v", "ay", "steer"])
    X = np.c_[d.ay / d.v ** 2, d.ay, np.ones(len(d))].astype(np.float64)
    y = d.steer.to_numpy(np.float64)
    with np.errstate(all="ignore"):                            # macOS の Accelerate が出す見かけの警告を抑える
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
    r2 = 1 - np.sum((d.steer - pred) ** 2) / np.sum((d.steer - d.steer.mean()) ** 2)
    a, b, c = (float(x) for x in coef)
    return dict(a_SRxL_m=round(a, 3), b_SRxKus_s2=round(b, 5), c_offset_deg=round(np.degrees(c), 3),
                steer_R2=round(float(r2), 4), n_points=int(len(d)), speed_kmh_range=f"{d.v.min() * 3.6:.0f}..{d.v.max() * 3.6:.0f}", note=note)


def fit_yaw_window(sid, ox, vd, cfg) -> tuple[dict, dict]:
    """OXTS がある 50 s で、Civic と同じ式 δsw = a·(r/v) + b·(v·r) + c を当てはめる
    (docs/crosswind_civic_feasibility.md (4.1)。r は OXTS のヨーレート、左回りが正)。
    前半で当てはめて後半で残差を測り、逆も行う (2 分割の交差検証)。"""
    t = ox.t.to_numpy()
    c = vd[CTRL]
    steer = np.interp(t, c.t, moving(c[K_STEER].to_numpy(float), sm_n(c, cfg["smooth_s"]["steer"])))
    v = moving(ox.speed.to_numpy(), cfg["smooth_s"]["speed"] * 100)
    r = np.radians(-moving(ox.angularRateZ.to_numpy(), cfg["smooth_s"]["steer"] * 100))
    m = v > cfg["fit"]["min_speed"]
    X = np.c_[r / v, v * r, np.ones(len(t))]
    half = np.arange(len(t)) < len(t) // 2
    dr = np.full(len(t), np.nan)
    with np.errstate(all="ignore"):
        coef_all, *_ = np.linalg.lstsq(X[m], steer[m], rcond=None)
        for tr in (half, ~half):
            co, *_ = np.linalg.lstsq(X[m & tr], steer[m & tr], rcond=None)
            te = m & ~tr
            r_pred = v[te] * (steer[te] - co[2]) / (co[0] + co[1] * v[te] ** 2)
            dr[te] = np.degrees(r[te] - r_pred)
        pred = X[m] @ coef_all
    r2 = 1 - np.sum((steer[m] - pred) ** 2) / np.sum((steer[m] - steer[m].mean()) ** 2)
    a, b, c0 = (float(x) for x in coef_all)
    model = dict(a_SRxL_m=round(a, 3), b_SRxKus_s2=round(b, 5), c_offset_deg=round(np.degrees(c0), 3), steer_R2=round(float(r2), 4),
                 n_points=int(m.sum()), speed_kmh_range=f"{v[m].min() * 3.6:.0f}..{v[m].max() * 3.6:.0f}",
                 note=f"{sid} の OXTS の 50 s。速度の幅が狭く、b はほとんど決まらない")
    r_pred_all = np.degrees(v * (steer - c0) / (a + b * v ** 2))
    stats = dict(oxts_yaw_residual_cv_std_deg_s=round(float(np.nanstd(dr)), 3),
                 oxts_yaw_residual_cv_p99_abs_deg_s=round(float(np.nanpercentile(np.abs(dr), 99)), 3),
                 oxts_yaw_rate_std_deg_s=round(float(np.degrees(np.std(r))), 3))
    return model, dict(stats=stats, t=t, r_left=np.degrees(r), r_pred=r_pred_all, steer=steer, v=v)


def predict_ay(g, model):
    a, b, c = model["a_SRxL_m"], model["b_SRxKus_s2"], np.radians(model["c_offset_deg"])
    return g.v ** 2 * (g.steer - c) / (a + b * g.v ** 2)


def residuals(sid, g, model, yaw, cfg) -> tuple[dict, pd.DataFrame]:
    """走行全体: 舵角から予測した横加速度との差。横加速度には路面の横の傾き (重力の成分) も入り、
    舵角だけでは分けられないので、ゆっくりした変化を除いた帯域 (band_hz) だけを見る。"""
    fs = cfg["grid_hz"]
    sh = cfg["straight_highspeed"]
    g = g.copy()
    g["ay_pred"] = predict_ay(g, model)
    g["d_ay"] = g.ay - g.ay_pred
    ok = g.d_ay.notna()
    band = np.full(len(g), np.nan)
    if ok.sum() > 10 * fs:
        # 途切れた所は前後の値を線でつないでから帯域を通し、途切れの前後 gap_margin_s は使わない
        # (0 で埋めると、途切れの前後に見かけの山ができる)
        band = bandpass(g.d_ay.interpolate(limit_direction="both").to_numpy(), fs, sh["band_hz"])
        near_gap = pd.Series(~ok.to_numpy()).rolling(2 * int(cfg["gap_margin_s"] * fs) + 1, center=True, min_periods=1).max().astype(bool).to_numpy()
        band[near_gap] = np.nan
        ok = ok & ~near_gap
    g["d_ay_band"] = band
    st = (g.v > sh["min_speed"]) & ((g.steer - np.radians(model["c_offset_deg"])).abs() < np.radians(sh["max_abs_steer_deg"])) & (g.turn == 0)
    out = dict(id=sid, **yaw["stats"],
               drive_minutes=round((g.t.iloc[-1] - g.t.iloc[0]) / 60, 1),
               drive_valid_fraction=round(float(ok.mean()), 4),
               drive_ay_residual_std=round(float(g.d_ay[ok & (g.v > cfg["fit"]["min_speed"])].std()), 4),
               straight_hi_minutes=round(st.sum() / fs / 60, 2),
               straight_hi_ay_residual_band_std=round(float(np.nanstd(g.d_ay_band[st])), 4),
               straight_hi_ay_residual_band_p99_abs=round(float(np.nanpercentile(np.abs(g.d_ay_band[st]), 99)), 4) if st.any() else np.nan,
               straight_hi_torque_std_nm=round(float(g.torque[st].std()), 3))
    g["straight_hi"] = st
    g.attrs["oxts"] = yaw
    return out, g


def sm_n(df, sec):
    return sec / np.median(np.diff(df.t.to_numpy()))


def top_events(sid, g, cfg, cam_t) -> list[dict]:
    """帯域を通した横加速度の残差の大きい区間 (高速・直進のうち)。横風とは判断しない。"""
    fs = cfg["grid_hz"]
    w = int(cfg["event_window_s"] * fs)
    x = np.where(g.straight_hi, np.abs(g.d_ay_band), np.nan)
    rows, used = [], np.zeros(len(g), bool)
    order = np.argsort(-np.nan_to_num(x, nan=-1))
    for i in order:
        if len(rows) >= cfg["top_events"] or not np.isfinite(x[i]):
            break
        if used[max(0, i - w): i + w].any():
            continue
        used[max(0, i - w): i + w] = True
        s = slice(max(0, i - w), i + w)
        rows.append(dict(id=sid, t_utc=datetime.fromtimestamp(g.t.iloc[i], timezone.utc).isoformat(timespec="milliseconds"),
                         speed_kmh=round(g.v.iloc[i] * 3.6, 1), d_ay_band=round(float(g.d_ay_band.iloc[i]), 3),
                         steer_range_deg=round(float(np.degrees(np.ptp(g.steer.iloc[s]))), 2),
                         torque_range_nm=round(float(np.ptp(g.torque.iloc[s])), 2),
                         turn_indicator_on=bool((g.turn.iloc[s] != 0).any()),
                         in_camera_window=bool(cam_t[0] <= g.t.iloc[i] <= cam_t[1])))
    return rows


# ---------------------------------------------------------------- 4. 走行条件
def conditions(sid, meta, ox, g, vd) -> dict:
    st = datetime.fromisoformat(meta["start_time"].replace("Z", "+00:00"))
    tz = TZ.get(meta["country_code"])
    loc = st.astimezone(ZoneInfo(tz)).isoformat(timespec="minutes") if tz else ""
    v = g.v.dropna()
    sat = vd["satellite"]
    ts = sat.t.to_numpy()
    return dict(id=sid, start_utc=meta["start_time"], local_time=loc, local_tz=tz or "不明", country=meta["country_code"],
                car=meta["collection_car"], lat=round(meta["latitude"], 5), lon=round(meta["longitude"], 5),
                seq_speed_kmh_min=round(ox.speed.iloc[int(len(ox) * 0.1):int(len(ox) * 0.5)].min() * 3.6, 1),
                seq_speed_kmh_max=round(ox.speed.iloc[int(len(ox) * 0.1):int(len(ox) * 0.5)].max() * 3.6, 1),
                oxts_speed_kmh_range=f"{ox.speed.min() * 3.6:.1f}..{ox.speed.max() * 3.6:.1f}",
                oxts_heading_change_deg=round(float(np.ptp(np.degrees(np.unwrap(np.radians(ox.heading))))), 1),
                oxts_alt_m_range=f"{ox.posAlt.min():.0f}..{ox.posAlt.max():.0f}",
                oxts_pos_std_m_median=round(float(np.median(np.hypot(ox.stdDevPosNorth, ox.stdDevPosEast))), 3),
                drive_minutes=round((g.t.iloc[-1] - g.t.iloc[0]) / 60, 1),
                drive_speed_kmh_p50=round(v.median() * 3.6, 1), drive_speed_kmh_p95=round(v.quantile(0.95) * 3.6, 1),
                drive_min_over_80kmh=round((v > 22.2).sum() / 20 / 60, 1),
                drive_min_over_100kmh=round((v > 27.8).sum() / 20 / 60, 1),
                drive_seq_offset_min=round((st.timestamp() - g.t.iloc[0]) / 60, 1),
                sat_unique_timestamps=int(len(np.unique(ts))), sat_rows=int(len(ts)))


# ---------------------------------------------------------------- 図
def plot_sequence(sid, g, ox, plots: Path, cam_t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    o = g.attrs["oxts"]
    t0 = cam_t[0]
    t = o["t"] - t0
    gw = g[(g.t > o["t"][0]) & (g.t < o["t"][-1])]
    tg = gw.t - t0
    fig, ax = plt.subplots(5, 1, figsize=(11, 10), sharex=True)
    ax[0].plot(t, o["v"] * 3.6); ax[0].set_ylabel("速さ [km/h]")
    ax[1].plot(t, np.degrees(o["steer"])); ax[1].set_ylabel("舵角 [deg]\n(左が正)")
    ax[2].plot(t, o["r_left"], label="OXTS (−angularRateZ)"); ax[2].plot(t, o["r_pred"], label="舵角からの予測", alpha=0.8)
    ax[2].set_ylabel("ヨーレート [deg/s]\n(左回りが正)"); ax[2].legend(fontsize=8)
    ax[3].plot(tg, gw.ay, label="車両データ lat_acc"); ax[3].plot(tg, gw.ay_pred, label="舵角からの予測", alpha=0.8)
    ax[3].plot(t, -lowpass(ox.accelerationY.to_numpy(), 100, 3), label="OXTS −accelerationY", alpha=0.6)
    ax[3].set_ylabel("横加速度 [m/s²]\n(左が正)"); ax[3].legend(fontsize=8)
    ax[4].plot(tg, gw.torque); ax[4].set_ylabel("操舵トルク [N·m]")
    for a in ax:
        a.axvspan(0, cam_t[1] - t0, color="tab:green", alpha=0.08)
    ax[4].set_xlabel("カメラの最初のフレームからの時刻 [s] (緑: カメラがある区間)")
    fig.suptitle(f"{sid}: OXTS がある 50 s")
    fig.tight_layout(); fig.savefig(plots / f"{sid}_sequence.png", dpi=100); plt.close(fig)

    fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    tm = (g.t - g.t.iloc[0]) / 60
    ax[0].plot(tm, g.v * 3.6, lw=0.5); ax[0].set_ylabel("速さ [km/h]")
    ax[1].plot(tm, np.degrees(g.steer), lw=0.5); ax[1].set_ylabel("舵角 [deg]"); ax[1].set_ylim(-60, 60)
    ax[2].plot(tm, g.d_ay_band, lw=0.4, color="gray", label="全体")
    ax[2].plot(tm.where(g.straight_hi), g.d_ay_band.where(g.straight_hi), lw=0.5, color="tab:red", label="高速・直進")
    ax[2].set_ylabel("横加速度の残差\n0.2–2 Hz [m/s²]"); ax[2].set_ylim(-1.5, 1.5); ax[2].legend(fontsize=8)
    ax[3].plot(tm, g.torque, lw=0.4); ax[3].set_ylabel("操舵トルク [N·m]")
    for a in ax:
        a.axvspan((cam_t[0] - g.t.iloc[0]) / 60, (cam_t[1] - g.t.iloc[0]) / 60, color="tab:green", alpha=0.3)
    ax[3].set_xlabel("車両データの最初からの時刻 [分] (緑: シーケンス 20 s)")
    fig.suptitle(f"{sid}: 車両データの走行全体")
    fig.tight_layout(); fig.savefig(plots / f"{sid}_drive.png", dpi=100); plt.close(fig)


def contact_sheet(sid, info, plots: Path, n=6):
    from PIL import Image
    fr = info.camera_frames["front_blur"]
    pick = [fr[int(i)] for i in np.linspace(0, len(fr) - 1, n)]
    tiles = []
    for f in pick:
        im = Image.open(f.filepath); im.draft("RGB", (962, 542)); tiles.append(im.convert("RGB").resize((640, 361)))
    sheet = Image.new("RGB", (640 * 3, 361 * 2))
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % 3) * 640, (k // 3) * 361))
    sheet.save(plots / f"{sid}_frames.jpg", quality=85)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("raw_data/zod"))
    ap.add_argument("--version", choices=["mini", "full"], default="mini")
    ap.add_argument("--config", type=Path, default=Path("configs/zod/signal_survey.yaml"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out = args.out or Path(f"out/zod/signal_survey/{args.version}")
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]

    from zod import ZodSequences
    ds = ZodSequences(str(args.root), args.version, mp=False)
    ids = sorted(ds.get_all_ids())
    data = {}
    rel, cams, conds = [], [], []
    for sid in ids:
        info = ds[sid].info
        ox = load_oxts(Path(info.oxts_path))
        vd = load_vehicle(Path(info.vehicle_data_path))
        meta = json.loads(Path(info.metadata_path).read_text())
        calib = json.loads(Path(info.calibration_path).read_text())
        g = drive_grid(vd, cfg)
        data[sid] = (info, ox, vd, meta, g)
        rel += relations(sid, ox, vd, cfg)
        cams.append(camera_sync(sid, info, ox, calib, cfg, plots))
        conds.append(conditions(sid, meta, ox, g, vd))
        contact_sheet(sid, info, plots)
        print(sid, "読み込み済み")

    models = {f"{sid}_drive_lat_acc": fit_model([d[4]], cfg, f"{sid} の走行全体 (車両 {d[3]['collection_car']})。"
                                                 "横加速度には路面の横の傾きが入るので、a, b, c は見かけの値") for sid, d in data.items()}
    res, ev = [], []
    for sid, (info, ox, vd, meta, g) in data.items():
        model = models[f"{sid}_drive_lat_acc"]
        ym, yaw = fit_yaw_window(sid, ox, vd, cfg)
        models[f"{sid}_oxts_yaw"] = ym
        fr = info.camera_frames["front_blur"]
        cam_t = (fr[0].time.timestamp(), fr[-1].time.timestamp())
        r, g2 = residuals(sid, g, model, yaw, cfg)
        res.append(r)
        ev += top_events(sid, g2, cfg, cam_t)
        plot_sequence(sid, g2, ox, plots, cam_t)

    pd.DataFrame(rel).to_csv(out / "relations.csv", index=False)
    pd.DataFrame(cams).to_csv(out / "camera_sync.csv", index=False)
    pd.DataFrame(conds).to_csv(out / "conditions.csv", index=False)
    pd.DataFrame(res).to_csv(out / "residuals.csv", index=False)
    pd.DataFrame(ev).to_csv(out / "top_events.csv", index=False)
    (out / "model_fit.json").write_text(json.dumps(dict(models=models, config=cfg), ensure_ascii=False, indent=1), encoding="utf-8")
    for name, df in (("relations", rel), ("camera_sync", cams), ("conditions", conds), ("residuals", res)):
        print(f"== {name}"); print(pd.DataFrame(df).to_string(index=False))
    print("== models", json.dumps(models, ensure_ascii=False, indent=1))
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
