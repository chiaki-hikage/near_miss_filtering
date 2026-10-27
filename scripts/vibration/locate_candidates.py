"""路面起因と思われる振動地点を、位置・再現性・映像と合わせて一覧にする (comma2k19 RAV4)。

振動イベントの検出は score_longtail.py と同じ (A = sqrt(上下比 × max(ロール比, ピッチ比)) ≥ a_min)。
その上に次を足す。凹凸の高さは推定しない。

  帯域     上下・ロールを 3〜8 Hz と 8〜15 Hz に分けた普段比。8〜15 Hz だけ高いものに印を付ける
  位置     前輪が当たった時刻の global_pose (カメラ位置) を前輪まで進めた緯度経度と、誤差の目安
  地点     半径・進行方向で同じ地点にまとめる
  通過     地点を通った全走行で A・上下比・帯域比・輪速比を取り、閾値未満の通過も分布として残す。
           無作為な場所の同じ長さの窓と比べ、「通るたびに揺れるか」を二項分布の上側確率で表す
  確認用   Google Maps / Street View のリンク、波形、通過ごとの分布、前方映像のフレームと動画

結果は「路面起因の振動地点の候補」であり、ヒヤリハットとは判定しない。

    uv run python scripts/vibration/locate_candidates.py raw_data/Chunk_1 raw_data/Chunk_2
"""

from __future__ import annotations

import argparse
import bz2
import os
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import binom

import find_roll_bumps as F
import score_longtail as L
from near_miss.config import find_vehicle_config, load_vehicle_configs, load_yaml
from near_miss.io.comma2k19 import find_segments
from near_miss.plotting import configure_japanese_font

REPO = _bootstrap.SRC.parent
WHEELS = L.WHEELS
GPS_UNIX_OFFSET_S = 315964800.0   # 1980-01-06 (GPS 時刻の原点) の unix 時刻
GPS_LEAP_S = 18.0                 # 2018 年時点の GPS − UTC
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON_EQ = 111320.0
IMG_CENTER_COL = 582              # 前方カメラ 1164 列の中央


# ---------------------------------------------------------------------------
# 座標
# ---------------------------------------------------------------------------

def ecef_to_llh_vec(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ECEF [m] (N, 3) → 緯度・経度 [deg]。find_roll_bumps.ecef_to_llh のベクトル版。"""
    a, e2 = 6378137.0, 6.69437999014e-3
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    lon = np.arctan2(y, x)
    r = np.hypot(x, y)
    lat = np.arctan2(z, r * (1 - e2))
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = r / np.cos(lat) - N
        lat = np.arctan2(z, r * (1 - e2 * N / (N + h)))
    return np.rad2deg(lat), np.rad2deg(lon)


class LocalFrame:
    """緯度経度 ⇔ 局所平面 (東・北 [m])。CA-280 沿いの 20 km 程度なら縮尺の誤差は 0.1% 未満。"""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = M_PER_DEG_LON_EQ * np.cos(np.deg2rad(lat0))

    def xy(self, lat, lon) -> np.ndarray:
        return np.c_[(np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * M_PER_DEG_LAT]

    def latlon(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.atleast_2d(xy)
        return self.lat0 + xy[:, 1] / M_PER_DEG_LAT, self.lon0 + xy[:, 0] / self.kx


def heading_diff(a, b) -> np.ndarray:
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def unit(hdg_deg) -> np.ndarray:
    """北から時計回りの方位 → (東, 北) の単位ベクトル。"""
    h = np.deg2rad(hdg_deg)
    return np.c_[np.sin(h), np.cos(h)]


def ecef_velocity_heading(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    lat, lon = (np.deg2rad(a) for a in ecef_to_llh_vec(pos))
    east = -np.sin(lon) * vel[:, 0] + np.cos(lon) * vel[:, 1]
    north = (-np.sin(lat) * np.cos(lon) * vel[:, 0] - np.sin(lat) * np.sin(lon) * vel[:, 1]
             + np.cos(lat) * vel[:, 2])
    return np.rad2deg(np.arctan2(east, north)) % 360


def gps_to_unix(week: np.ndarray, sec: np.ndarray) -> np.ndarray:
    return GPS_UNIX_OFFSET_S + week * 604800.0 + sec - GPS_LEAP_S


def attach_track(run, frame: LocalFrame, camera_behind_m: float) -> None:
    """run.t の格子に、前輪の位置 (東・北)・方位・走行距離・unix 時刻と、最寄りの GPS 点との時間差を付ける。

    global_pose はカメラの位置なので、進行方向にカメラと前輪の距離だけ進める。
    """
    t, p, v, g = [], [], [], []
    for r in run.refs:
        t.append(np.load(r.path / "global_pose/frame_times"))
        p.append(np.load(r.path / "global_pose/frame_positions"))
        v.append(np.load(r.path / "global_pose/frame_velocities"))
        g.append(np.load(r.path / "global_pose/frame_gps_times"))
    t, p, v, g = (np.concatenate(a) for a in (t, p, v, g))
    t, i = np.unique(t, return_index=True)
    p, v, g = p[i], v[i], g[i]
    xy_cam = frame.xy(*ecef_to_llh_vec(p))
    hdg = ecef_velocity_heading(p, v)
    u = unit(hdg)
    ux, uy = np.interp(run.t, t, u[:, 0]), np.interp(run.t, t, u[:, 1])
    run.hdg = np.rad2deg(np.arctan2(ux, uy)) % 360
    run.xy = np.c_[np.interp(run.t, t, xy_cam[:, 0]), np.interp(run.t, t, xy_cam[:, 1])] \
        + camera_behind_m * np.c_[ux, uy] / np.maximum(np.hypot(ux, uy), 1e-9)[:, None]
    out = (run.t < t[0]) | (run.t > t[-1])
    run.xy[out] = np.nan
    j = np.clip(np.searchsorted(t, run.t), 1, len(t) - 1)
    run.gps_dt = np.minimum(np.abs(run.t - t[j - 1]), np.abs(t[j] - run.t))
    run.unix = np.interp(run.t, t, gps_to_unix(g[:, 0], g[:, 1]))
    dt = np.diff(run.t, prepend=run.t[0])
    run.dist = np.cumsum(np.nan_to_num(run.v) * dt)


# ---------------------------------------------------------------------------
# GNSS の自己申告精度 (raw_log の gpsLocationExternal.horizontalAccuracy)
# ---------------------------------------------------------------------------

def _gnss_accuracy_one(seg_dir: str) -> tuple[str, float, float, int]:
    from near_miss.io.rlog import load_schema

    log = load_schema()
    acc = []
    try:
        data = bz2.open(Path(seg_dir) / "raw_log.bz2").read()
        for m in log.Event.read_multiple_bytes(data):
            try:
                if m.which() == "gpsLocationExternal":
                    acc.append(m.gpsLocationExternal.horizontalAccuracy)
            except Exception:  # 古いログに現行スキーマに無い種別が混じる
                continue
    except (OSError, EOFError):
        return seg_dir, np.nan, np.nan, 0
    a = np.asarray(acc, float)
    if a.size == 0:
        return seg_dir, np.nan, np.nan, 0
    return seg_dir, float(np.median(a)), float(np.percentile(a, 95)), int(a.size)


def gnss_accuracy(refs: list, cache: Path, workers: int) -> pd.DataFrame:
    have = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["seg_dir"])
    todo = [str(r.path) for r in refs if str(r.path) not in set(have.seg_dir)]
    if todo:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_gnss_accuracy_one, todo, chunksize=4))
        new = pd.DataFrame(rows, columns=["seg_dir", "acc_median_m", "acc_p95_m", "n_fix"])
        have = pd.concat([have, new], ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        have.to_csv(cache, index=False)
    return have.set_index("seg_dir")


# ---------------------------------------------------------------------------
# 帯域別の普段比
# ---------------------------------------------------------------------------

def add_band_envelopes(run, cfg: dict, loc: dict) -> None:
    """上下・ロールの帯域別の移動 RMS を run.env に足す (キーは heave_road など)。"""
    fs = cfg["grid_hz"]
    n_env = max(1, int(round(cfg["envelope_s"] * fs)))
    trim = int(F.EDGE_TRIM_S * fs)
    for key in ("heave", "roll"):
        x = L._fill(run.raw[key])
        for name, (lo, hi) in loc["bands_hz"].items():
            e = F._mrms(F._bandpass(x, lo, hi, fs), n_env)
            e[:trim] = np.nan
            e[-trim:] = np.nan
            run.env[f"{key}_{name}"] = e


def run_ratios(run, base: pd.DataFrame, cfg: dict, fast: bool) -> dict[str, np.ndarray]:
    r = F.ratios(run, base, cfg)
    if not fast:  # ジャイロが実質 10 Hz の走行では、ロールとピッチの振動は見えない
        for k in [k for k in r if k.startswith(("roll", "pitch"))]:
            r[k] = np.full_like(r[k], np.nan)
    hv = np.nan_to_num(r["heave"])
    r["A"] = np.sqrt(hv * np.maximum(np.nan_to_num(r["roll"]), np.nan_to_num(r["pitch"]))) if fast \
        else np.full_like(hv, np.nan)
    r["A"][~np.isfinite(r["heave"])] = np.nan
    return r


# ---------------------------------------------------------------------------
# イベント
# ---------------------------------------------------------------------------

def hit_index(run, e, cfg: dict, loc: dict) -> int:
    """前輪が当たった時刻: ピーク付近で上下の絶対値が最大値の frac 倍を最初に超えた点。"""
    fs = cfg["grid_hz"]
    a = max(0, e.i_start - int(0.2 * fs))
    b = min(len(run.t), e.i_end + int(0.2 * fs) + 1)
    x = np.abs(run.bp["heave"])
    th = loc["hit"]["frac"] * np.max(x[a:b])
    s0 = max(0, e.i_peak + int(loc["hit"]["search_s"][0] * fs))
    s1 = min(len(run.t), e.i_peak + int(loc["hit"]["search_s"][1] * fs) + 1)
    over = np.flatnonzero(x[s0:s1] >= th)
    return int(s0 + over[0]) if over.size else int(e.i_peak)


def describe_events(ev: pd.DataFrame, runs: list, ratios: list, gnss: pd.DataFrame,
                    frame: LocalFrame, cfg: dict, loc: dict) -> pd.DataFrame:
    fs = cfg["grid_hz"]
    pos = loc["position"]
    rows = []
    for _, e in ev.iterrows():
        run, r = runs[e.run], ratios[e.run]
        k = hit_index(run, e, cfg, loc)
        a = max(0, e.i_start - int(0.2 * fs))
        b = min(len(run.t), e.i_end + int(0.2 * fs) + 1)
        band = {key: float(np.nanmax(r[key][a:b])) for key in r if "_" in key and key.split("_")[0] in ("heave", "roll")}
        ref = next(x for x in run.refs if x.index == e.segment)
        acc = gnss.loc[str(ref.path)] if str(ref.path) in gnss.index else None
        acc_m = float(acc.acc_median_m) if acc is not None else np.nan
        v = float(run.v[k])
        along = float(np.sqrt(acc_m ** 2 + (v * pos["hit_time_unc_s"]) ** 2 + pos["camera_offset_unc_m"] ** 2))
        lat, lon = frame.latlon(run.xy[k])
        c0, c1 = max(0, k - int(fs)), min(len(run.t), k + int(fs))
        wmax = max(e[f"wheel_ratio_{w}"] for w in WHEELS)
        rows.append(dict(
            t_hit_s=round(float(run.t[k] - run.seg_t0[e.segment]), 2),
            event_unix=round(float(run.unix[k]), 2),
            event_utc=pd.to_datetime(run.unix[k], unit="s").strftime("%Y-%m-%d %H:%M:%S"),
            lat=round(float(lat[0]), 6), lon=round(float(lon[0]), 6), heading_deg=round(float(run.hdg[k]), 1),
            x_m=round(float(run.xy[k, 0]), 2), y_m=round(float(run.xy[k, 1]), 2),
            gps_dt_s=round(float(run.gps_dt[k]), 3), gnss_acc_m=round(acc_m, 2),
            pos_err_along_m=round(along, 2), pos_err_cross_m=round(acc_m, 2),
            wheel_max_ratio=round(wmax, 1),
            wheel_lag_match=bool(abs(e.wheel_front_rear_lag_s - e.wheel_lag_expected_s) <= loc["lag_tol_s"]),
            **{f"{key}_ratio": round(v_, 2) for key, v_ in band.items()},
            brake_any=bool(np.nansum(run.brake[c0:c1] > 0.5) > 0),
            ay_range_mps2=round(float(np.nanmax(run.ay[c0:c1]) - np.nanmin(run.ay[c0:c1])), 2),
        ))
    out = pd.concat([ev.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    bmin = loc["band_ratio_min"]
    out["wheelhop_only"] = (out.heave_wheelhop_ratio >= bmin) & (out.heave_road_ratio < bmin)
    out["road_band"] = out.heave_road_ratio >= bmin
    out["one_sided"] = ((out.wheel_lr_log2.abs() >= cfg["longtail"]["one_sided_lr_log2"])
                        & (out.roll_over_pitch >= cfg["event"]["roll_heave"]["roll_over_pitch_min"]))
    out["frequency_features"] = [
        f"上下 3-8Hz×{x.heave_road_ratio:.1f} / 8-15Hz×{x.heave_wheelhop_ratio:.1f}; "
        f"ロール 3-8Hz×{x.roll_road_ratio:.1f} / 8-15Hz×{x.roll_wheelhop_ratio:.1f}; 主周波数 {x.heave_dom_hz:.1f} Hz"
        for x in out.itertuples()]
    # 振動の強さ: A・輪速・3〜8 Hz の上下比の、全イベント中の順位 (0〜100) の平均
    out["vibration_score"] = (out[["A", "wheel_max_ratio", "heave_road_ratio"]].rank(pct=True).mean(axis=1) * 100).round(1)
    return out


# ---------------------------------------------------------------------------
# 地点と通過
# ---------------------------------------------------------------------------

def circular_mean(deg: np.ndarray) -> float:
    u = unit(deg).mean(axis=0)
    return float(np.rad2deg(np.arctan2(u[0], u[1])) % 360)


def cluster_sites(xy: np.ndarray, hdg: np.ndarray, strength: np.ndarray, radius: float, heading_tol: float,
                  lane_half: float) -> np.ndarray:
    """強い順に種を取り、種の進行方向に ±radius・横に ±lane_half (同じ車線)・方位差以内の未割り当てを同じ地点にする。"""
    site = -np.ones(len(xy), int)
    n = 0
    for i in np.argsort(-strength):
        if site[i] >= 0:
            continue
        u = unit(hdg[i])[0]
        dx = xy - xy[i]
        along, cross = dx @ u, dx @ np.array([u[1], -u[0]])
        site[(np.abs(along) <= radius) & (np.abs(cross) <= lane_half)
             & (heading_diff(hdg, hdg[i]) <= heading_tol) & (site < 0)] = n
        n += 1
    return site


def split_passes(mask: np.ndarray, t: np.ndarray, gap_s: float = 2.0) -> list[np.ndarray]:
    """mask が真の連続区間 (gap_s 以内の切れ目はつなぐ) ごとの添字の配列。"""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cut = np.flatnonzero(np.diff(t[idx]) > gap_s) + 1
    return np.split(idx, cut)


def window_around(dist: np.ndarray, k: int, half_m: float) -> slice:
    a = int(np.searchsorted(dist, dist[k] - half_m, side="left"))
    b = int(np.searchsorted(dist, dist[k] + half_m, side="right"))
    return slice(a, max(b, a + 1))


def pass_metrics(run, r: dict, w: slice, k: int, flank: tuple[float, float] = (11.0, 31.0)) -> dict:
    """窓 w の中の各指標の最大値。A と上下比は、最大になった点の k からの走行距離 (off) も返す。

    窓の端で最大になる通過は、窓の外にある別の凹凸の立ち上がりを拾っているだけのことがあるので、
    再現の判定では off が窓の内側にあるものだけを数える (interior_hit)。
    """
    def mx(key):
        x = r[key][w]
        return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan

    def off(key):
        x = r[key][w]
        if not np.isfinite(x).any():
            return np.nan
        return float(run.dist[w][int(np.nanargmax(x))] - run.dist[k])

    # 前後 flank m の範囲 (地点の外) の中央値に対する、窓の中の最大値の比。
    # 荒れた区間の途中では 1 に近く、マンホールや継ぎ目のような点状の凹凸では大きくなる
    rel = run.dist - run.dist[k]
    lo = max(0, int(np.searchsorted(run.dist, run.dist[k] - flank[1])))
    hi = int(np.searchsorted(run.dist, run.dist[k] + flank[1]))
    fm = np.zeros(len(run.dist), bool)
    fm[lo:hi] = np.abs(rel[lo:hi]) >= flank[0]

    def contrast(key):
        f = r[key][fm]
        m = mx(key)
        if not np.isfinite(f).any() or not np.isfinite(m):
            return np.nan
        return float(m / max(np.nanmedian(f), 1e-6))

    wr = {k: mx(k) for k in WHEELS}
    left, right = np.nanmax([wr["FL"], wr["RL"]]), np.nanmax([wr["FR"], wr["RR"]])
    return dict(
        A_max=mx("A"), A_off_m=off("A"), A_contrast=contrast("A"),
        heave_ratio_max=mx("heave"), heave_off_m=off("heave"), heave_contrast=contrast("heave"), roll_ratio_max=mx("roll"), pitch_ratio_max=mx("pitch"),
        heave_road_max=mx("heave_road"), heave_wheelhop_max=mx("heave_wheelhop"),
        roll_road_max=mx("roll_road"), roll_wheelhop_max=mx("roll_wheelhop"),
        wheel_max_ratio=float(np.nanmax(list(wr.values()))),
        wheel_lr_log2=float(np.log2(left / right)) if left > 0 and right > 0 else np.nan,
        **{f"wheel_ratio_{k}": v for k, v in wr.items()},
    )


def site_passes(sites: pd.DataFrame, runs: list, ratios: list, fast: list[bool], ev: pd.DataFrame,
                cfg: dict, loc: dict) -> pd.DataFrame:
    """各地点を通った全走行の指標。イベントの有無に関係なく、閾値未満の通過も残す。"""
    R, H = loc["site"]["radius_m"], loc["site"]["heading_deg"]
    flank = (R + loc["repeat"]["flank_m"][0], R + loc["repeat"]["flank_m"][1])
    rows = []
    for s in sites.itertuples():
        c = np.array([s.x_m, s.y_m])
        u = unit(s.heading_deg)[0]
        n_right = np.array([u[1], -u[0]])
        for ri, run in enumerate(runs):
            d = np.hypot(*(run.xy - c).T)
            m = (d <= R) & (heading_diff(run.hdg, s.heading_deg) <= H) & (run.v >= cfg["min_speed_mps"])
            for g in split_passes(np.nan_to_num(m, nan=0).astype(bool), run.t):
                k = int(g[np.nanargmin(d[g])])
                w = window_around(run.dist, k, R)
                pm = pass_metrics(run, ratios[ri], w, k, flank)
                along = (run.xy[w, 0] - c[0]) * u[0] + (run.xy[w, 1] - c[1]) * u[1]
                key = "A" if fast[ri] else "heave"
                j = int(np.nanargmax(np.nan_to_num(ratios[ri][key][w], nan=-1)))
                evs = ev[(ev.run_global == ri) & (ev.i_hit >= w.start) & (ev.i_hit < w.stop)]
                seg = int(run.seg[k])
                rows.append(dict(
                    site_id=s.site_id, drive_id=run.drive_id, segment=seg,
                    t_closest_s=round(float(run.t[k] - run.seg_t0[seg]), 2),
                    event_utc=pd.to_datetime(run.unix[k], unit="s").strftime("%Y-%m-%d %H:%M"),
                    gyro_fast=fast[ri], speed_kmh=round(float(run.v[k]) * 3.6, 1),
                    closest_m=round(float(d[k]), 2), cross_m=round(float((run.xy[k] - c) @ n_right), 2),
                    same_lane=bool(abs(float((run.xy[k] - c) @ n_right)) <= loc["site"]["lane_half_m"]),
                    peak_along_m=round(float(along[j]), 1),
                    event_id=";".join(evs.event_id) if len(evs) else "",
                    run=ri, i_closest=k,
                    **{k_: (round(v_, 2) if np.isfinite(v_) else np.nan) for k_, v_ in pm.items()},
                ))
    return pd.DataFrame(rows)


def null_windows(runs: list, ratios: list, fast: list[bool], cfg: dict, loc: dict) -> pd.DataFrame:
    """無作為な場所で、地点の判定と同じ長さ (前後 R m) の窓を取り、通過ごとの指標の基準分布にする。"""
    rep = loc["repeat"]
    rng = np.random.default_rng(rep["null_seed"])
    R = loc["site"]["radius_m"]
    flank = (R + rep["flank_m"][0], R + rep["flank_m"][1])
    cand = [(ri, i) for ri, run in enumerate(runs)
            for i in np.flatnonzero((run.v >= cfg["min_speed_mps"]) & np.isfinite(run.env["heave"]))[::20]]
    pick = rng.choice(len(cand), size=min(rep["null_windows"], len(cand)), replace=False)
    rows = []
    for p in pick:
        ri, i = cand[p]
        pm = pass_metrics(runs[ri], ratios[ri], window_around(runs[ri].dist, i, R), i, flank)
        rows.append(dict(run=ri, gyro_fast=fast[ri], **pm))
    return pd.DataFrame(rows)


def binom_surprise(k: int, n: int, p0: float) -> float:
    """n 回の通過のうち k 回以上が閾値を超える確率 (偶然なら) の -log10。"""
    if n <= 0:
        return 0.0
    return float(-np.log10(max(binom.sf(k - 1, n, p0), 1e-300)))


def repeat_hits(d: pd.DataFrame, loc: dict) -> tuple[pd.Series, pd.Series]:
    """通過ごとの再現の判定 (A と上下比)。閾値以上で、最大の点が窓の内側にあり、前後より際立っているもの。"""
    rep = loc["repeat"]
    inner = loc["site"]["radius_m"] - rep["edge_margin_m"]
    a = (d.A_max >= rep["a_min"]) & (d.A_off_m.abs() <= inner) & (d.A_contrast >= rep["contrast_min"])
    h = (d.heave_ratio_max >= rep["heave_ratio_min"]) & (d.heave_off_m.abs() <= inner) & (d.heave_contrast >= rep["contrast_min"])
    return a, h


def summarize_sites(sites: pd.DataFrame, passes: pd.DataFrame, null: pd.DataFrame, loc: dict) -> pd.DataFrame:
    rep = loc["repeat"]
    hitA = lambda d: repeat_hits(d, loc)[0]
    hitH = lambda d: repeat_hits(d, loc)[1]
    nf = null[null.gyro_fast]
    p0_A = float(hitA(nf).mean())
    p0_h = float(hitH(null).mean())
    rows = []
    for s in sites.itertuples():
        g_all = passes[passes.site_id == s.site_id]
        other = g_all[~g_all.same_lane]
        g = g_all[g_all.same_lane].copy()   # 再現は同じ車線を通った回だけで数える
        g["strength"] = np.nan_to_num(np.where(g.gyro_fast, g.A_max, g.heave_ratio_max), nan=-1.0)
        # 地点を選ぶきっかけになった最も強い通過は除いて、残りの通過で再現を数える
        rest = g.drop(g.strength.idxmax()) if len(g) else g
        rf = rest[rest.gyro_fast]
        kA, nA = int(hitA(rf).sum()), len(rf)
        kh, nh = int(hitH(rest).sum()), len(rest)
        hit = g[(g.gyro_fast & hitA(g)) | hitH(g)]
        sides = np.sign(hit.wheel_lr_log2.dropna())
        major = (np.sign(sides.sum()) if sides.sum() != 0 else 0)
        rows.append(dict(
            site_id=s.site_id,
            n_pass=len(g), n_pass_drives=g.drive_id.nunique(), n_pass_fast=int(g.gyro_fast.sum()),
            n_pass_other_lane=len(other),
            other_lane_repeat=int((repeat_hits(other, loc)[0] | repeat_hits(other, loc)[1]).sum()),
            repeat_A_k=kA, repeat_A_n=nA, repeat_A_surprise=round(binom_surprise(kA, nA, p0_A), 2),
            repeat_heave_k=kh, repeat_heave_n=nh, repeat_heave_surprise=round(binom_surprise(kh, nh, p0_h), 2),
            A_pass_median=round(float(g[g.gyro_fast].A_max.median()), 2) if g.gyro_fast.any() else np.nan,
            A_pass_values=" ".join(f"{x:.1f}" for x in sorted(g[g.gyro_fast].A_max.dropna(), reverse=True)),
            heave_pass_median=round(float(g.heave_ratio_max.median()), 2),
            heave_road_pass_median=round(float(g.heave_road_max.median()), 2),
            heave_wheelhop_pass_median=round(float(g.heave_wheelhop_max.median()), 2),
            wheel_pass_median=round(float(g.wheel_max_ratio.median()), 2),
            side_major=("左" if major > 0 else "右" if major < 0 else "—"),
            side_consistency=round(float((sides == major).mean()), 2) if len(sides) and major != 0 else np.nan,
            contrast_median=round(float(np.nanmedian(np.where(g.gyro_fast, g.A_contrast, g.heave_contrast))), 2)
            if len(g) else np.nan,
            peak_along_std_m=round(float(hit.peak_along_m.std()), 2) if len(hit) >= 3 else np.nan,
            cross_range_m=round(float(g.cross_m.max() - g.cross_m.min()), 2) if len(g) else np.nan,
        ))
    out = pd.DataFrame(rows)
    out.attrs.update(p0_A=p0_A, p0_heave=p0_h)
    return out


def stretches(xy: np.ndarray, hdg: np.ndarray, link_m: float, heading_tol: float) -> np.ndarray:
    """同じ向きで link_m 以内につながる地点を 1 つの区間にする (単連結)。荒れた区間・橋の継ぎ目の連なりを見分ける。"""
    n = len(xy)
    lab = -np.ones(n, int)
    k = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        lab[i] = k
        stack = [i]
        while stack:
            j = stack.pop()
            d = np.hypot(*(xy - xy[j]).T)
            nb = np.flatnonzero((d <= link_m) & (heading_diff(hdg, hdg[j]) <= heading_tol) & (lab < 0))
            lab[nb] = k
            stack += nb.tolist()
        k += 1
    return lab


def opposite_sites(s: pd.DataFrame, within_m: float) -> list[str]:
    """反対向き (方位差 150° 以上) で within_m 以内にある地点。道路を横切る継ぎ目などの手がかり。"""
    xy = s[["x_m", "y_m"]].to_numpy()
    out = []
    for i in range(len(s)):
        d = np.hypot(*(xy - xy[i]).T)
        m = (d <= within_m) & (heading_diff(s.heading_deg, s.heading_deg.iloc[i]) >= 150)
        out.append(";".join(str(x) for x in s.site_id[m]))
    return out


def pick_top(s: pd.DataFrame, n: int, per_stretch: int) -> pd.DataFrame:
    """順位の高い順に、同じ区間からは per_stretch 件までにして n 件を選ぶ。"""
    taken = s.groupby("stretch_id").cumcount()
    return s[taken < per_stretch].head(n)


def flag_and_rank(sites: pd.DataFrame, summ: pd.DataFrame, ev: pd.DataFrame, loc: dict) -> pd.DataFrame:
    rep = loc["repeat"]
    s = sites.merge(summ, on="site_id")
    best = ev.loc[ev.groupby("site_id").A.idxmax()].set_index("site_id")
    s = s.join(best[["event_id", "A", "wheel_max_ratio", "wheel_lr_log2", "wheel_side", "one_sided",
                     "heave_road_ratio", "heave_wheelhop_ratio", "wheelhop_only", "wheel_lag_match",
                     "vibration_score", "speed_kmh", "heave_peak_mps2", "roll_rate_peak_dps", "spike_like"]]
                .rename(columns={"event_id": "best_event_id", "A": "A_best"}), on="site_id")
    s["flag_A"] = s.wheel_max_ratio >= loc["wheel_ratio_min"]
    s["flag_B"] = ~s.wheelhop_only & (s.heave_road_ratio >= loc["band_ratio_min"])
    s["flag_C"] = (((s.repeat_A_n >= rep["min_passes"]) & (s.repeat_A_surprise >= rep["surprise_min"]))
                   | ((s.repeat_heave_n >= rep["min_passes"]) & (s.repeat_heave_surprise >= rep["surprise_min"])))
    s["flag_D"] = ""   # 映像・地図の目視で埋める
    reproduced = s.repeat_A_k + s.repeat_heave_k > 0
    s["flag_E"] = s.one_sided & (~reproduced | s.side_consistency.isna()
                                 | (s.side_consistency >= rep["side_consistency_min"]))
    s.loc[s.spike_like, ["flag_A", "flag_B"]] = False
    s["n_flags"] = s[["flag_A", "flag_B", "flag_C", "flag_E"]].sum(axis=1)
    # 点状 (マンホール・継ぎ目・段差など) か、荒れた区間の一部か。地図・映像での照合のしやすさが違う
    s["feature_type"] = np.where(s.contrast_median >= loc["site"]["point_contrast_min"], "点状", "区間")
    s["repeat_surprise"] = s[["repeat_A_surprise", "repeat_heave_surprise"]].max(axis=1)
    s["is_point"] = s.feature_type == "点状"
    s = s.sort_values(["n_flags", "is_point", "repeat_surprise", "vibration_score"], ascending=False).reset_index(drop=True)
    s.insert(0, "rank", range(1, len(s) + 1))
    s["stretch_id"] = stretches(s[["x_m", "y_m"]].to_numpy(), s.heading_deg.to_numpy(),
                                loc["stretch"]["link_m"], loc["site"]["heading_deg"])
    s["stretch_n_sites"] = s.groupby("stretch_id").site_id.transform("size")
    s["opposite_site_ids"] = opposite_sites(s, loc["stretch"]["opposite_m"])
    s["google_maps"] = [f"https://www.google.com/maps/search/?api=1&query={a:.6f},{b:.6f}" for a, b in zip(s.lat, s.lon)]
    s["street_view"] = [f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={a:.6f},{b:.6f}&heading={h:.0f}&pitch=-15"
                        for a, b, h in zip(s.lat, s.lon, s.heading_deg)]
    return s


# ---------------------------------------------------------------------------
# 位置誤差の実測 (地点 19 の横断歩道)
# ---------------------------------------------------------------------------

def landmark_check(runs: list, frame: LocalFrame, loc: dict, pav: dict) -> tuple[pd.DataFrame, dict]:
    """映像から求めた横断歩道を前輪が通過した時刻に、GPS で前輪がどこにいたかを走行ごとに並べる。

    同じ線なので本当の位置は全走行で同じ。進行方向のばらつきが、位置 + 時刻の誤差の実測値になる。
    線の通過時刻そのものにも ±0.05 s (約 0.7 m) の誤差がある (docs/vibration_pavement_site19.md 1.2)。
    """
    lm = pd.read_csv(REPO / loc["landmark_passes"])
    lat0, lon0 = pav["site19"]["center_latlon"]
    c = frame.xy([lat0], [lon0])[0]
    u = unit(pav["site19"]["heading_deg"])[0]
    n_right = np.array([u[1], -u[0]])
    rows = []
    for x in lm.itertuples():
        run = next((r for r in runs if r.drive_id.endswith(x.drive) and x.segment in r.seg_t0), None)
        if run is None:
            continue
        for line, ts in (("入口", x.entry_s), ("出口", x.exit_s)):
            if not np.isfinite(ts):
                continue
            t_abs = run.seg_t0[x.segment] + ts
            k = int(np.clip(np.searchsorted(run.t, t_abs), 0, len(run.t) - 1))
            rows.append(dict(drive=x.drive, segment=x.segment, line=line, t_s=ts, speed_kmh=round(float(run.v[k]) * 3.6, 1),
                             along_m=round(float((run.xy[k] - c) @ u), 2), cross_m=round(float((run.xy[k] - c) @ n_right), 2),
                             gps_dt_s=round(float(run.gps_dt[k]), 3)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}
    df["along_dev_m"] = df.along_m - df.groupby("line").along_m.transform("median")
    pooled = float(np.sqrt((df.groupby("line").along_m.transform(lambda a: a - a.mean()) ** 2).sum()
                           / max(len(df) - df.line.nunique(), 1)))
    length = df.pivot_table(index=["drive", "segment"], columns="line", values="along_m")
    summary = dict(
        passes=int(df[["drive", "segment"]].drop_duplicates().shape[0]), lines=int(len(df)),
        along_std_pooled_m=round(pooled, 2),
        along_dev_absmax_m=round(float(df.along_dev_m.abs().max()), 2),
        crossing_len_gps_m=(length["出口"] - length["入口"]).round(2).dropna().tolist() if {"入口", "出口"} <= set(length) else [],
        crossing_len_can_m=lm.crossing_length_m.dropna().tolist(),
        suggested_radius_m=float(np.clip(np.ceil(2 * np.sqrt(2) * pooled), 5, 15)),
        rule="半径 = clip(ceil(2√2 × 進行方向の誤差の標準偏差), 5, 15)。2 回の通過の位置の差の 95% 程度を覆う",
    )
    return df, summary


def event_scatter(ev: pd.DataFrame, sites: pd.DataFrame, pos: dict) -> dict:
    """同じ地点に入った別ドライブのイベントの、進行方向・横方向の位置のばらつき。

    当たり時刻の推定のばらつきを含む。半径の内側でしか数えないので、半径が小さいほど小さく出る。
    地点 19 の誤差 (GPS + 線の時刻) と合わせて、σ_total = sqrt(σ_landmark² + σ_event²) から半径を提案する。
    """
    h = sites.set_index("site_id").heading_deg
    a_std, c_std = [], []
    for sid, g in ev.groupby("site_id"):
        if g.drive_id.nunique() < 2:
            continue
        u = unit(h[sid])[0]
        d = g[["x_m", "y_m"]].to_numpy()
        a_std.append(np.std(d @ u, ddof=1))
        c_std.append(np.std(d @ np.array([u[1], -u[0]]), ddof=1))
    if not a_std:
        return {}
    se = float(np.median(a_std))
    total = float(np.hypot(pos.get("along_std_pooled_m", 0.0), se))
    return dict(event_sites=len(a_std), event_along_std_median_m=round(se, 2),
                event_along_std_p90_m=round(float(np.percentile(a_std, 90)), 2),
                event_cross_std_median_m=round(float(np.median(c_std)), 2),
                sigma_total_m=round(total, 2),
                suggested_radius_total_m=float(np.clip(np.ceil(2 * np.sqrt(2) * total), 5, 15)))


# ---------------------------------------------------------------------------
# 図・映像・一覧
# ---------------------------------------------------------------------------

def plot_event(run, r: dict, e: pd.Series, cfg: dict, loc: dict, out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fs = cfg["grid_hz"]
    T = loc["output"]["clip_s"]
    k = int(e.i_hit)
    a, b = max(0, k - int(T * fs)), min(len(run.t), k + int(T * fs))
    t = run.t[a:b] - run.t[k]
    lag = e.wheel_lag_expected_s
    fig, axs = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    axs[0].plot(t, run.bp["heave"][a:b], lw=0.8, color="tab:blue")
    axs[0].set_ylabel("上下加速度 [m/s²]\n(0.7〜20 Hz)")
    axs[1].plot(t, run.bp["roll"][a:b], lw=0.8, color="tab:green", label="ロール (正 = 右下がり)")
    axs[1].plot(t, run.bp["pitch"][a:b], lw=0.8, color="tab:purple", alpha=0.8, label="ピッチ (正 = 機首上げ)")
    axs[1].set_ylabel("角速度 [deg/s]\n(0.7〜15 Hz)")
    axs[1].legend(fontsize=8, loc="upper right")
    for w, c, ls in (("FL", "tab:blue", "--"), ("RL", "tab:blue", "-"), ("FR", "tab:red", "--"), ("RR", "tab:red", "-")):
        axs[2].plot(t, run.bp[w][a:b] * 3.6, lw=0.8, color=c, ls=ls, label=w)
    axs[2].set_ylabel("輪速の揺れ [km/h]\n(2〜15 Hz)")
    axs[2].legend(fontsize=8, ncol=4, loc="upper right")
    axs[3].plot(t, r["heave_road"][a:b], color="tab:orange", label="上下 3〜8 Hz")
    axs[3].plot(t, r["heave_wheelhop"][a:b], color="tab:gray", label="上下 8〜15 Hz")
    axs[3].plot(t, r["roll_road"][a:b], color="tab:green", ls="--", label="ロール 3〜8 Hz")
    axs[3].axhline(loc["band_ratio_min"], color="k", lw=0.6, ls=":")
    axs[3].set_ylabel("帯域別の普段比 [倍]")
    axs[3].legend(fontsize=8, loc="upper right")
    axs[4].plot(t, r["A"][a:b], color="k", lw=1.0, label="A")
    axs[4].plot(t, np.nanmax(np.c_[[r[w][a:b] for w in WHEELS]], axis=0), color="tab:red", lw=0.8, label="輪速 (4 輪の最大)")
    axs[4].axhline(cfg["longtail"]["a_min"], color="k", lw=0.6, ls=":")
    axs[4].set_ylabel("普段比 [倍]")
    axs[4].legend(fontsize=8, loc="upper right")
    for ax in axs:
        ax.axvline(0, color="tab:red", lw=0.8, alpha=0.6)
        ax.axvline(lag, color="tab:red", lw=0.8, ls="--", alpha=0.6)
        ax.grid(alpha=0.3)
    axs[-1].set_xlabel(f"前輪が当たった推定時刻からの時間 [s] (実線 = 前輪、破線 = 後輪 L/v = {lag:.2f} s 後)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def plot_passes(site, passes: pd.DataFrame, runs: list, ratios: list, loc: dict, out: Path) -> None:
    """地点を通った全走行について、地点からの距離に対する A と上下比を重ねる。"""
    import matplotlib.pyplot as plt

    R = loc["site"]["radius_m"]
    c = np.array([site.x_m, site.y_m])
    u = unit(site.heading_deg)[0]
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for p in passes.itertuples():
        run, r = runs[p.run], ratios[p.run]
        w = window_around(run.dist, p.i_closest, R + 30)
        along = (run.xy[w, 0] - c[0]) * u[0] + (run.xy[w, 1] - c[1]) * u[1]
        ev = bool(p.event_id)
        col = "tab:red" if ev else ("0.35" if p.gyro_fast else "tab:cyan")
        if not p.same_lane:   # 隣の車線を通った回は薄く描く (再現の判定には使わない)
            col = "tab:olive"
        lw = 1.4 if ev else 0.8
        lab = p.drive_id.split("|")[1][5:16]
        if p.gyro_fast:
            axs[0].plot(along, r["A"][w], color=col, lw=lw)
        axs[1].plot(along, r["heave"][w], color=col, lw=lw, ls="-" if p.gyro_fast else "--")
        axs[2].plot(along, np.nanmax(np.c_[[r[k][w] for k in WHEELS]], axis=0), color=col, lw=lw,
                    ls="-" if p.gyro_fast else "--", label=lab if ev else None)
    for ax, th, yl in ((axs[0], loc["repeat"]["a_min"], "A [倍]\n(ジャイロ 104 Hz の走行)"),
                       (axs[1], loc["repeat"]["heave_ratio_min"], "上下の普段比 [倍]\n(破線 = ジャイロ 10 Hz の走行)"),
                       (axs[2], loc["wheel_ratio_min"], "輪速の揺れの普段比 [倍]\n(4 輪の最大)")):
        ax.axvspan(-R, R, color="gold", alpha=0.15)
        ax.axhline(th, color="k", lw=0.6, ls=":")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axs[2].legend(fontsize=7, ncol=4, title="赤 = イベント検出あり、黄緑 = 隣の車線 (判定に使わない)", title_fontsize=7)
    axs[-1].set_xlabel(f"地点の中心からの進行方向の距離 [m] (黄 = 同じ地点とみなす ±{R:g} m)")
    fig.suptitle(f"地点 {site.site_id}: 同じ車線の通過 {site.n_pass} 回 (ジャイロ 104 Hz {site.n_pass_fast} 回)、"
                 f"隣の車線 {site.n_pass_other_lane} 回、イベントあり {site.n_event_drives} ドライブ", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def frame_for(run, t_abs: float):
    """t_abs に最も近い前方映像のフレーム (隣のセグメントもまたいで探す)。"""
    import final_qualitative as FQ

    best = None
    for r in run.refs:
        ft = np.load(r.path / "global_pose/frame_times")
        i = int(np.argmin(np.abs(ft - t_abs)))
        if best is None or abs(ft[i] - t_abs) < best[0]:
            best = (abs(ft[i] - t_abs), r)
    img, t_frame = FQ.frame_at(best[1].path, 0.0, t_abs)
    return img, t_frame, best[1].index


def plot_frames(run, e: pd.Series, cam: dict, track: float, loc: dict, out: Path) -> None:
    """前輪が当たる地点が前方 12 m・6 m に見えるフレームに、左右の車輪の通り道と推定位置を重ねる。"""
    import matplotlib.pyplot as plt

    f, yh, K = cam["focal_px"], cam["horizon_row"], cam["K"]
    behind = cam["camera_behind_front_axle_m"]
    v = max(float(e.speed_kmh) / 3.6, 1.0)
    t_hit = run.t[int(e.i_hit)]
    dists = loc["output"]["frame_ahead_m"]
    fig, axs = plt.subplots(1, len(dists), figsize=(7.2 * len(dists), 4.6))
    for ax, ahead in zip(np.atleast_1d(axs), dists):
        img, t_frame, seg = frame_for(run, t_hit - ahead / v)
        ax.imshow(img)
        d_on = v * (t_hit - t_frame) + behind     # カメラから当たる地点までの距離
        dd = np.linspace(3.5, 45, 200)
        for y_lat, c, lab in ((-track / 2, "tab:blue", "左輪の通り道"), (track / 2, "tab:red", "右輪の通り道")):
            ax.plot(IMG_CENTER_COL + f * y_lat / dd, yh + K / dd, color=c, lw=1.2, ls="--", label=lab)
            ax.plot(IMG_CENTER_COL + f * y_lat / d_on, yh + K / d_on, "o", ms=10, mfc="none", mec=c, mew=2)
        # 進行方向の位置の誤差 (当たり時刻 ±0.05 s とカメラ位置の仮定) を帯で示す
        unc = np.hypot(v * loc["position"]["hit_time_unc_s"], loc["position"]["camera_offset_unc_m"])
        lo, hi = max(d_on - unc, 3.0), d_on + unc
        ax.axhspan(yh + K / lo, yh + K / hi, color="yellow", alpha=0.18)
        ax.set_xlim(250, 1100)
        ax.set_ylim(720, 420)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"seg{seg}  当たる {t_hit - t_frame:.2f} s 前 (約 {d_on:.1f} m 先)", fontsize=9)
    np.atleast_1d(axs)[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("○ = 推定の当たり位置 (平坦・直進・カメラは車両中心と仮定。黄色の帯 = 進行方向の誤差の目安、横は ±0.3 m 程度)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=95)
    plt.close(fig)


def daytime_event(g: pd.DataFrame, loc: dict, default: pd.Series) -> pd.Series:
    """映像の確認用に、昼間 (現地時刻) のイベントのうち A が最大のものを選ぶ。無ければ default。"""
    o = loc["output"]
    h = (pd.to_datetime(g.event_unix, unit="s") + pd.Timedelta(hours=o["local_utc_offset_h"])).dt.hour
    day = g[(h >= o["daylight_local_h"][0]) & (h < o["daylight_local_h"][1])]
    return day.loc[day.A.idxmax()] if len(day) else default


def attach_review(top: pd.DataFrame, review: Path, frame: LocalFrame, loc: dict) -> pd.DataFrame:
    """目視確認の CSV (lat, lon, heading_deg, d_video, d_streetview, note) を、位置と向きで地点に対応付ける。

    地点の番号は設定を変えると振り直されるので、番号ではなく位置で照合する。
    """
    top = top.copy()
    for c in ("d_video", "d_streetview", "d_note"):
        top[c] = ""
    if not review.exists():
        return top
    rv = pd.read_csv(review).fillna("")
    rxy = frame.xy(rv.lat.to_numpy(), rv.lon.to_numpy())
    for i, s in top.iterrows():
        d = np.hypot(*(rxy - np.array([s.x_m, s.y_m])).T)
        m = (d <= loc["site"]["radius_m"]) & (heading_diff(rv.heading_deg, s.heading_deg) <= loc["site"]["heading_deg"])
        if m.any():
            j = int(np.argmin(np.where(m, d, np.inf)))
            top.loc[i, ["d_video", "d_streetview", "d_note"]] = [rv.d_video.iloc[j], rv.d_streetview.iloc[j], rv.note.iloc[j]]
    top["flag_D"] = top.d_video.str.startswith("一致") | top.d_streetview.str.startswith("一致")
    return top


def write_markdown(top: pd.DataFrame, meta: dict, pos: dict, out: Path, rel: str) -> None:
    yes = lambda b: "○" if bool(b) else "—"
    dmark = lambda s: ("○" if s.flag_D is True else "△" if str(s.d_video).startswith("候補") else "?" if s.d_video else "")
    L_ = [
        "# 路面起因と思われる振動地点 上位候補 (comma2k19 RAV4 Chunk_1・2)",
        "",
        "ヒヤリハットとは判定していない。凹凸の高さは推定していない。",
        "D (映像・地図上の路面特徴) は目視で埋める欄。Street View は撮影時期が 2018 年の走行と違う可能性があるので、"
        "地図との一致だけで路面起因と断定しない。一次確認は前方映像で行う。",
        "",
        f"- セグメント {meta['segments_fast']} 本 (ジャイロ 104 Hz) + {meta['segments_slow']} 本 (10 Hz、上下と輪速だけ使う)、"
        f"イベント {meta['events']} 件、地点 {meta['sites']} か所",
        f"- 同じ地点の判定: 半径 {meta['radius_m']:g} m・方位差 {meta['heading_deg']:g}°。"
        f"地点 19 の横断歩道で測った進行方向の位置誤差 σ = {pos.get('along_std_pooled_m', float('nan'))} m",
        f"- 無作為な場所で A ≥ {meta['repeat_a_min']:g} になる割合 {meta['p0_A']:.3f}、上下比 ≥ {meta['repeat_heave_min']:g} は {meta['p0_heave']:.3f}",
        "",
        "| No. | 全体順位 | 地点 | 形 (前後比) | A | B | C | E | D | 再現 A (k/n) | 再現 上下 (k/n) | 強い側 (一致率) | 最大 A | 輪速 | 3-8 Hz | 車速 | 区間 (地点数) | 反対向き | 位置 | 地図 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for no, s in enumerate(top.itertuples(), 1):
        L_.append(
            f"| {no} | {s.rank} | {s.site_id} | {s.feature_type} ({s.contrast_median:.1f}) | {yes(s.flag_A)} | {yes(s.flag_B)} | {yes(s.flag_C)} | {yes(s.flag_E)} | {dmark(s)} |"
            f" {s.repeat_A_k}/{s.repeat_A_n} | {s.repeat_heave_k}/{s.repeat_heave_n} |"
            f" {s.side_major} ({'' if pd.isna(s.side_consistency) else f'{s.side_consistency:.2f}'}) |"
            f" {s.A_best:.1f} | {s.wheel_max_ratio:.1f} | {s.heave_road_ratio:.1f} | {s.speed_kmh:.0f} |"
            f" {s.stretch_id} ({s.stretch_n_sites}) | {s.opposite_site_ids if isinstance(s.opposite_site_ids, str) and s.opposite_site_ids else '—'} |"
            f" {s.lat:.6f}, {s.lon:.6f} ({s.heading_deg:.0f}°) | [Maps]({s.google_maps}) / [SV]({s.street_view}) |")
    L_ += ["", "D: 前方映像で、推定の当たり位置に路面の特徴が見えるか (○ 一致、△ 候補、? 不明瞭・判別不能、空欄 未確認)。"
           "目視の暫定判定で、review_video.csv を直して再実行すると反映される。"
           "A: CAN/IMU で明確 (A ≥ 3 かつ輪速の普段比 ≥ 3)。B: 3〜8 Hz の上下も普段の 2 倍以上 (8〜15 Hz だけではない)。"
           "C: 同じ地点の他の通過でも再現 (最も強い通過を除き、偶然なら 1% 未満)。E: 輪速とロールから片側入力。"
           "再現 k/n は、最も強い通過を除いた n 回のうち閾値を超えた回数。"
           f"区間は同じ向きで {meta['stretch_link_m']:g} m 以内につながる地点のまとまりで、一覧には同じ区間から {meta['max_per_stretch']} 件までを載せた。"
           "反対向きは、逆方向の走行で 20 m 以内に振動地点があるもの (道路を横切る継ぎ目などの手がかり)。"
           "形は、地点の中 (±半径) の最大値と前後 (地点の外 5〜25 m) の中央値の比の、通過ごとの中央値。"
           f"{meta['point_contrast_min']:g} 以上を点状、未満を荒れた区間の一部とした。同じ印の数なら点状を先に並べた。", ""]
    for no, s in enumerate(top.itertuples(), 1):
        L_ += [f"## {no}. 地点 {s.site_id} (全体順位 {s.rank})  ({s.lat:.6f}, {s.lon:.6f}、進行方向 {s.heading_deg:.0f}°)", "",
               f"- 代表イベント {s.best_event_id}: {s.speed_kmh:.0f} km/h、上下 {s.heave_peak_mps2:.1f} m/s²、"
               f"ロール {s.roll_rate_peak_dps:.1f} deg/s、輪速の揺れ {s.wheel_side}側が強い (log2 左/右 = {s.wheel_lr_log2:+.2f})、"
               f"前→後輪の時間差 {'一致' if s.wheel_lag_match else '不一致'}",
               f"- 同じ車線の通過 {s.n_pass} 回 ({s.n_pass_drives} ドライブ)、隣の車線 {s.n_pass_other_lane} 回 "
               f"(うち閾値超え {s.other_lane_repeat} 回)、イベントあり {s.n_event_drives} ドライブ。"
               f"通過ごとの A: {s.A_pass_values or '—'}",
               f"- 位置の誤差の目安: 進行方向 ±{s.pos_err_along_m:.1f} m、横 ±{s.pos_err_cross_m:.1f} m (自己申告と仮定からの合成)",
               f"- [Google Maps]({s.google_maps}) / [Street View]({s.street_view})",
               f"- 映像の確認 (D、暫定): {s.d_video or '未確認'}{'。' + s.d_note if s.d_note else ''}",
               f"- Street View の確認: {s.d_streetview or '未確認'}",
               "",
               f"![波形]({rel}/site{s.site_id:03d}_wave.png)", "",
               f"![通過ごと]({rel}/site{s.site_id:03d}_passes.png)", "",
               f"映像 ({s.frame_event_id}{'、昼間の通過' if s.frame_event_id != s.best_event_id else ''}):", "",
               f"![映像]({rel}/site{s.site_id:03d}_frames.png)", "",
               f"動画: `{rel}/site{s.site_id:03d}_clip.mp4`", ""]
    out.write_text("\n".join(L_))


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--config", type=Path, default=REPO / "configs/vibration_locate.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "out/vibration/locate")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-media", action="store_true", help="図・フレーム・動画を作らない")
    args = ap.parse_args()

    loc = yaml.safe_load(args.config.read_text())
    cfg = yaml.safe_load((REPO / loc["base_config"]).read_text())
    pav = yaml.safe_load((REPO / loc["pavement_config"]).read_text())
    det_cfg = load_yaml(REPO / "configs/detection.yaml")
    args.out.mkdir(parents=True, exist_ok=True)
    media = args.out / "media"

    refs = [r for root in args.roots for r in find_segments(root)]
    veh = find_vehicle_config(refs[0].dongle_id, load_vehicle_configs())
    wheelbase = veh.geometry["wheelbase_m"]
    is_fast = {str(r.path): F.gyro_rate_hz(r.path) >= cfg["min_gyro_hz"] for r in refs}
    fast_refs = [r for r in refs if is_fast[str(r.path)]]
    slow_refs = [r for r in refs if not is_fast[str(r.path)]]
    print(f"セグメント: ジャイロ 104 Hz {len(fast_refs)} 本 / 10 Hz {len(slow_refs)} 本")

    runs_fast = L.build_runs(fast_refs, veh, cfg, det_cfg)
    runs_slow = L.build_runs(slow_refs, veh, cfg, det_cfg) if slow_refs else []
    runs = runs_fast + runs_slow
    fast = [True] * len(runs_fast) + [False] * len(runs_slow)
    for run in runs:
        add_band_envelopes(run, cfg, loc)
    # 普段比の基準は、ジャイロの周期が同じ走行どうしで取る
    base_fast = F.baselines(runs_fast, cfg)
    base_slow = F.baselines(runs_slow, cfg) if runs_slow else base_fast
    ratios = [run_ratios(run, base_fast if f else base_slow, cfg, f) for run, f in zip(runs, fast)]

    p_ref = loc["position"]["ref_latlon"]
    frame = LocalFrame(*p_ref)
    cam = dict(yaml.safe_load((REPO / loc["camera_fit"]).read_text()))
    cam["focal_px"] = pav["camera"]["focal_px"]
    for run in runs:
        attach_track(run, frame, cam["camera_behind_front_axle_m"])
    gnss = gnss_accuracy(refs, args.out / "gnss_accuracy.csv", args.workers)

    # 振動イベント (score_longtail.py と同じ検出)
    ev = L.find_events(runs_fast, base_fast, cfg, wheelbase)
    ev["run_global"] = ev.run
    ev = describe_events(ev, runs, ratios, gnss, frame, cfg, loc)
    ev["i_hit"] = [hit_index(runs[e.run], e, cfg, loc) for _, e in ev.iterrows()]
    ev.insert(0, "event_id", [f"E{i:04d}" for i in range(1, len(ev) + 1)])
    ev.insert(1, "dataset", "comma2k19")
    ev.insert(2, "chunk", [next(x for x in runs[e.run].refs if x.index == e.segment).path.parts[-3] for _, e in ev.iterrows()])

    # 位置誤差の実測 → 同じ地点の半径
    lmk, pos = landmark_check(runs, frame, loc, pav)
    lmk.to_csv(args.out / "position_check_site19.csv", index=False)
    print("地点 19 の位置誤差:", json.dumps(pos, ensure_ascii=False))

    R, H = loc["site"]["radius_m"], loc["site"]["heading_deg"]
    xy = ev[["x_m", "y_m"]].to_numpy()
    ev["site_id"] = cluster_sites(xy, ev.heading_deg.to_numpy(), ev.A.to_numpy(), R, H, loc["site"]["lane_half_m"])
    g = ev.groupby("site_id")
    sites = pd.DataFrame(dict(
        x_m=g.x_m.mean(), y_m=g.y_m.mean(),
        heading_deg=g.heading_deg.agg(lambda h: circular_mean(h.to_numpy())),
        n_events=g.size(), n_event_drives=g.drive_id.nunique(),
        pos_err_along_m=g.pos_err_along_m.median(), pos_err_cross_m=g.pos_err_cross_m.median(),
        event_spread_m=g.apply(lambda d: float(np.hypot(d.x_m - d.x_m.mean(), d.y_m - d.y_m.mean()).max())),
    )).reset_index()
    sites["lat"], sites["lon"] = (np.round(a, 6) for a in frame.latlon(sites[["x_m", "y_m"]].to_numpy()))
    ev["repeat_count"] = ev.site_id.map(sites.set_index("site_id").n_event_drives)
    pos.update(event_scatter(ev, sites, pos))
    print("同じ地点のイベントのばらつき:", {k: pos[k] for k in ("event_along_std_median_m", "event_cross_std_median_m",
                                                   "sigma_total_m", "suggested_radius_total_m")})

    passes = site_passes(sites, runs, ratios, fast, ev, cfg, loc)
    passes["repeat_hit_A"], passes["repeat_hit_heave"] = repeat_hits(passes, loc)
    null = null_windows(runs, ratios, fast, cfg, loc)
    summ = summarize_sites(sites, passes, null, loc)
    ranked = flag_and_rank(sites, summ, ev, loc)

    # 出力
    keep = ["event_id", "dataset", "chunk", "drive_id", "segment", "t_hit_s", "t_peak_s", "t_start_s", "t_end_s",
            "event_utc", "lat", "lon", "heading_deg", "speed_kmh",
            "heave_peak_mps2", "heave_ratio", "roll_rate_peak_dps", "roll_ratio", "pitch_rate_peak_dps", "pitch_ratio",
            "A", "roll_over_pitch", *[f"wheel_ratio_{w}" for w in WHEELS], "wheel_max_ratio", "wheel_lr_log2", "wheel_side",
            "wheel_front_rear_lag_s", "wheel_lag_expected_s", "wheel_lag_match",
            "heave_road_ratio", "heave_wheelhop_ratio", "roll_road_ratio", "roll_wheelhop_ratio", "heave_dom_hz",
            "wheelhop_only", "road_band", "one_sided", "frequency_features", "vibration_score",
            "site_id", "repeat_count", "gps_dt_s", "gnss_acc_m", "pos_err_along_m", "pos_err_cross_m",
            "spike_like", "brake_any", "ay_range_mps2"]
    # ご依頼の列名に合わせる (単位は列名の末尾)
    ev[keep].rename(columns={"drive_id": "drive", "t_hit_s": "event_time_s", "heave_peak_mps2": "vertical_accel_peak_mps2",
                             "roll_ratio": "roll_metric", "pitch_ratio": "pitch_metric", "wheel_max_ratio": "wheel_metric"}
                    ).to_csv(args.out / "candidates.csv", index=False)
    passes.drop(columns=["run", "i_closest"]).to_csv(args.out / "site_passes.csv", index=False)
    null.drop(columns=["run"]).to_csv(args.out / "null_windows.csv", index=False)
    ranked.to_csv(args.out / "sites.csv", index=False)
    top = pick_top(ranked, loc["output"]["top_n"], loc["stretch"]["max_per_stretch"])
    top.to_csv(args.out / "top20.csv", index=False)

    meta = dict(
        roots=[str(r) for r in args.roots], config=os.path.relpath(args.config, REPO),
        config_sha1=hashlib.sha1(args.config.read_bytes()).hexdigest()[:10],
        base_config_sha1=hashlib.sha1((REPO / loc["base_config"]).read_bytes()).hexdigest()[:10],
        segments_fast=len(fast_refs), segments_slow=len(slow_refs), runs=len(runs),
        events=len(ev), sites=len(sites), sites_multi_event_drive=int((sites.n_event_drives >= 2).sum()),
        radius_m=R, heading_deg=H, point_contrast_min=loc["site"]["point_contrast_min"], stretch_link_m=loc["stretch"]["link_m"], max_per_stretch=loc["stretch"]["max_per_stretch"], repeat_a_min=loc["repeat"]["a_min"], repeat_heave_min=loc["repeat"]["heave_ratio_min"],
        p0_A=summ.attrs["p0_A"], p0_heave=summ.attrs["p0_heave"],
        flags={k: int(ranked[k].sum()) for k in ("flag_A", "flag_B", "flag_C", "flag_E")},
        n_flags=ranked.n_flags.value_counts().sort_index().to_dict(),
        position_check=pos,
    )
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float))
    print(json.dumps({k: v for k, v in meta.items() if k != "position_check"}, ensure_ascii=False, indent=1, default=float))

    if not args.no_media:
        matplotlib.use("Agg")
        configure_japanese_font()
        media.mkdir(exist_ok=True)
        # 映像の車輪の通り道は final_qualitative.py と同じカタログのトレッドを使う
        track = yaml.safe_load((REPO / "configs/vibration_fullcar.yaml").read_text())["vehicle"]["track_front_m"]["value"]
        T = loc["output"]["clip_s"]
        top = top.copy()
        top["frame_event_id"] = ""
        for s in top.itertuples():
            e = ev[ev.event_id == s.best_event_id].iloc[0]
            run, r = runs[e.run], ratios[e.run]
            stem = media / f"site{s.site_id:03d}"
            title = (f"{s.rank}. 地点 {s.site_id}  {e.event_id}  {e.drive_id.split('|')[1]} seg{e.segment} {e.t_hit_s:.2f} s  "
                     f"{e.speed_kmh:.0f} km/h  A={e.A:.1f}  輪速 {e.wheel_side}側 ×{e.wheel_max_ratio:.1f}\n{e.frequency_features}")
            plot_event(run, r, e, cfg, loc, Path(f"{stem}_wave.png"), title)
            plot_passes(s, passes[passes.site_id == s.site_id], runs, ratios, loc, Path(f"{stem}_passes.png"))
            ef = daytime_event(ev[ev.site_id == s.site_id], loc, e)
            top.loc[s.Index, "frame_event_id"] = ef.event_id
            plot_frames(runs[ef.run], ef, cam, track, loc, Path(f"{stem}_frames.png"))
            if loc["output"]["clips"]:
                t_hit = run.t[int(e.i_hit)]
                L.clip_case(run, max(t_hit - T, run.t[0]), min(t_hit + T, run.t[-1]), Path(f"{stem}_clip.mp4"))
        top = attach_review(top, REPO / loc["output"]["review_csv"], frame, loc)
        top.to_csv(args.out / "top20.csv", index=False)
        write_markdown(top, meta, pos, args.out / "top20.md", "media")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
