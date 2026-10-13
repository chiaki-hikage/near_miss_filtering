"""路面の凹凸・段差による振動のうち、車両挙動に影響した可能性のある区間を探す。

振動イベントごとに独立した 3 つの指標を出す。

  A  路面外乱の異常度   上下・ロール・ピッチが同じ車速帯の普段の何倍か
  B  車両挙動の異常度   振動の後にヨーレート残差・操舵・横ずれが、振動前と比べて乱れたか
  C  同種イベントの希少性  (A, B) の組み合わせが全振動イベントの中でどれだけ珍しいか

B は正常走行から無作為に取った疑似イベントにも同じ計算をかけ、その分布の上位何 % に
あたるかを -log10 で表す (2.0 = 上位 1%)。振動中は IMU と輪速自体が揺れて
挙動の指標を汚すので、挙動は振動が終わった後の窓だけで評価する。

結果はいずれも「ロングテール候補」であり、ヒヤリハットとは判定しない。

    uv run python scripts/vibration/score_longtail.py raw_data/Chunk_1 --clips 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import _bootstrap  # noqa: F401

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks

import find_roll_bumps as F
from near_miss.config import find_vehicle_config, load_vehicle_configs, load_yaml
from near_miss.features import bicycle_yaw_rate, find_lane_changes, rolling, window_samples
from near_miss.io.comma2k19 import find_segments, load_segment
from near_miss.plotting import configure_japanese_font

REPO = _bootstrap.SRC.parent
WHEELS = ("FL", "FR", "RL", "RR")


@dataclass
class Run:
    """同じドライブの連番セグメントを 1 本につないだもの。"""

    drive_id: str
    refs: list
    seg_t0: dict[int, float]   # セグメント番号 → IMU 先頭の boot time
    t: np.ndarray              # boot time [s]
    seg: np.ndarray
    v: np.ndarray
    steer: np.ndarray
    raw: dict
    bp: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    yaw: np.ndarray = None       # CAN YAW_RATE を 2 Hz で低域通過 [deg/s] 左旋回正
    yaw_model: np.ndarray = None # 舵角から単軌道モデルで予測したヨーレート
    yaw_res: np.ndarray = None   # 実測 − 単軌道モデル、帯域通過後
    yaw_road: np.ndarray = None  # 実測 − 10 秒移動中央値 (道路の曲率を除いた偏差)
    steer_rate: np.ndarray = None
    ay: np.ndarray = None
    brake: np.ndarray = None
    gvc: np.ndarray = None
    gps_t: np.ndarray = None
    gps_en: np.ndarray = None    # 東・北 [m] (ドライブ先頭基準の局所平面)
    gps_hdg: np.ndarray = None   # 北から時計回り [deg]
    lane_changes: list = field(default_factory=list)


def _interp(t_src: np.ndarray, v_src: np.ndarray, t: np.ndarray) -> np.ndarray:
    if t_src.size == 0:
        return np.full(t.shape, np.nan)
    out = np.interp(t, t_src, v_src)
    out[(t < t_src[0]) | (t > t_src[-1])] = np.nan
    return out


def _fill(x: np.ndarray) -> np.ndarray:
    """フィルタに通すためだけの穴埋め。結果は元の NaN 位置で NaN に戻す。"""
    ok = np.isfinite(x)
    if ok.all() or not ok.any():
        return np.nan_to_num(x)
    i = np.arange(len(x))
    return np.interp(i, i[ok], x[ok])


def ecef_to_enu(p: np.ndarray, ref: np.ndarray) -> np.ndarray:
    lat, lon = (np.deg2rad(a) for a in F.ecef_to_llh(*ref))
    R = np.array([
        [-np.sin(lon), np.cos(lon), 0.0],
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
    ])
    return np.einsum("ij,nj->ni", R, p - ref)


def build_runs(refs: list, veh, cfg: dict, det_cfg: dict) -> list[Run]:
    fs = cfg["grid_hz"]
    geo = veh.geometry
    lt = cfg["longtail"]
    by_drive: dict[str, list] = {}
    for r in refs:
        by_drive.setdefault(r.drive_id, []).append(r)
    runs = []
    for drive, rs in by_drive.items():
        rs = sorted(rs, key=lambda r: r.index)
        rot = F.drive_rotation([r.path for r in rs])
        groups, cur = [], [rs[0]]
        for r in rs[1:]:
            if r.index == cur[-1].index + 1:
                cur.append(r)
            else:
                groups.append(cur)
                cur = [r]
        groups.append(cur)
        ecef_ref = np.load(rs[0].path / "global_pose/frame_positions")[0]
        for grp in groups:
            parts = dict(t=[], seg=[], v=[], steer=[], yaw=[], brake=[], gvc=[],
                         gt=[], gp=[], gv=[], **{k: [] for k in ("heave", "roll", "pitch", *WHEELS)})
            seg_t0 = {}
            last = -np.inf
            for r in grp:
                s = F.load_seg(r, rot, cfg)
                seg_t0[r.index] = s.t0
                t = s.t + s.t0
                keep = t > last
                t = t[keep]
                last = t[-1]
                parts["t"].append(t)
                parts["seg"].append(np.full(t.size, r.index))
                parts["v"].append(s.v[keep])
                parts["steer"].append(s.steer[keep])
                for k in ("heave", "roll", "pitch", *WHEELS):
                    parts[k].append(s.raw[k][keep])
                d = load_segment(r, veh, with_raw_can=True).channels
                for name, key in (("yaw_rate", "yaw"), ("gvc_mps2", "gvc"), ("brake_pressed", "brake")):
                    c = d.get(name)
                    parts[key].append(_interp(c.t, c.v.astype(float), t) if c is not None else np.full(t.size, np.nan))
                parts["gt"].append(np.load(r.path / "global_pose/frame_times"))
                parts["gp"].append(np.load(r.path / "global_pose/frame_positions"))
                parts["gv"].append(np.load(r.path / "global_pose/frame_velocities"))
            cat = {k: np.concatenate(v) for k, v in parts.items()}
            run = Run(drive_id=drive, refs=grp, seg_t0=seg_t0, t=cat["t"], seg=cat["seg"], v=cat["v"],
                      steer=cat["steer"], raw={k: cat[k] for k in ("heave", "roll", "pitch", *WHEELS)})
            n_env = max(1, int(round(cfg["envelope_s"] * fs)))
            trim = int(F.EDGE_TRIM_S * fs)
            for k, x in run.raw.items():
                band = cfg["band_hz"]["wheel" if k in WHEELS else k]
                run.bp[k] = F._bandpass(_fill(x), band[0], band[1], fs)
                e = F._mrms(run.bp[k], n_env)
                e[:trim] = np.nan
                e[-trim:] = np.nan
                run.env[k] = e
            yaw_raw = cat["yaw"]
            run.yaw = F._lowpass(_fill(yaw_raw), 2.0, fs)
            run.yaw[~np.isfinite(yaw_raw)] = np.nan
            model = bicycle_yaw_rate(run.v, run.steer, geo["wheelbase_m"], geo["steer_ratio"],
                                     geo["understeer_gradient"], geo["steer_offset_deg"])
            run.yaw_model = model
            run.yaw_road = run.yaw - rolling(run.yaw, window_samples(det_cfg["lane_change"]["curvature_window_s"], fs), "median")
            res = run.yaw - model
            yb = lt["yaw_band_hz"]
            run.yaw_res = F._bandpass(_fill(res), yb[0], yb[1], fs)
            run.yaw_res[~np.isfinite(res)] = np.nan
            run.steer_rate = np.gradient(F._lowpass(_fill(run.steer), lt["steer_lp_hz"], fs), 1 / fs)
            run.ay = run.v * np.deg2rad(run.yaw)
            run.brake = cat["brake"]
            run.gvc = cat["gvc"]
            order = np.argsort(cat["gt"])
            run.gps_t = cat["gt"][order]
            run.gps_en = ecef_to_enu(cat["gp"][order], ecef_ref)
            gv = cat["gv"][order]
            en_v = ecef_to_enu(cat["gp"][order] + gv, ecef_ref) - run.gps_en
            run.gps_hdg = np.rad2deg(np.arctan2(en_v[:, 0], en_v[:, 1])) % 360
            run.lane_changes = find_lane_changes(run.t, run.v, np.nan_to_num(run.yaw), fs, det_cfg)
            runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# B: 車両挙動
# ---------------------------------------------------------------------------

def _lateral(run: Run, dev_dps: np.ndarray, i0: int, i1: int) -> np.ndarray:
    """ヨーレートの偏差 dev を i0 から i1 まで積分した横ずれ [m]。左正。"""
    dt = np.diff(run.t[i0:i1 + 1], prepend=run.t[i0])
    psi = np.cumsum(np.deg2rad(np.nan_to_num(dev_dps[i0:i1 + 1])) * dt)
    return np.cumsum(np.nan_to_num(run.v[i0:i1 + 1]) * np.sin(psi) * dt)


def lateral_vs_model(run: Run, i0: int, i1: int, j0: int, j1: int) -> np.ndarray:
    """舵角から予測した進路に対する横ずれ。[j0, j1) の残差平均 (モデルの定常誤差) は差し引く。"""
    res = run.yaw - run.yaw_model
    return _lateral(run, res - np.nanmean(res[j0:j1]), i0, i1)


def _gps_lateral(run: Run, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray]:
    """GPS の位置を、i0 時点から舵角どおりに進んだ場合の予測進路と比べた横ずれ [m]。左正。"""
    ta, tb = run.t[i0], run.t[i1]
    k0 = int(np.clip(np.searchsorted(run.gps_t, ta), 0, len(run.gps_t) - 1))
    m = (run.gps_t > ta) & (run.gps_t <= tb)
    if not m.any():
        return np.array([]), np.array([])
    dt = np.diff(run.t[i0:i1 + 1], prepend=run.t[i0])
    # 方位は北から時計回り。ヨーレートは左旋回正なので符号を反転して積分する
    h = np.deg2rad(run.gps_hdg[k0]) - np.cumsum(np.deg2rad(np.nan_to_num(run.yaw_model[i0:i1 + 1])) * dt)
    v = np.nan_to_num(run.v[i0:i1 + 1])
    path = run.gps_en[k0] + np.c_[np.cumsum(v * np.sin(h) * dt), np.cumsum(v * np.cos(h) * dt)]
    tt = run.t[i0:i1 + 1]
    gt = run.gps_t[m]
    j = np.clip(np.searchsorted(tt, gt), 0, len(tt) - 1)
    n = np.c_[-np.cos(h[j]), np.sin(h[j])]
    return gt, np.sum((run.gps_en[m] - path[j]) * n, axis=1)


def behavior(run: Run, i_start: int, i_end: int, cfg: dict) -> dict | None:
    fs = cfg["grid_hz"]
    lt = cfg["longtail"]
    p0 = i_start + int(lt["pre_s"][0] * fs)
    p1 = i_start + int(lt["pre_s"][1] * fs)
    q1 = i_end + int(lt["post_s"] * fs)
    a0 = i_start + int(lt["avoid_pre_s"][0] * fs)
    h0 = i_start - int(lt["heading_ref_s"] * fs)
    if min(p0, a0) < 0 or q1 >= len(run.t):
        return None
    if np.nanmin(run.v[min(p0, a0):q1]) < cfg["min_speed_mps"]:
        return None
    if not np.isfinite(run.yaw_road[a0:i_start]).all():
        return None
    yr_pre = np.sqrt(np.nanmean(run.yaw_res[p0:p1] ** 2))
    yr_post = np.sqrt(np.nanmean(run.yaw_res[i_end:q1] ** 2))
    sr_pre = np.nanmax(np.abs(run.steer_rate[p0:p1]))
    sr_post = np.nanmax(np.abs(run.steer_rate[i_end:q1]))
    lat_post = lateral_vs_model(run, i_end, q1, h0, i_start)
    lat_avoid = _lateral(run, run.yaw_road, a0, i_start)
    _, gps_post = _gps_lateral(run, i_end, q1)
    return dict(
        yaw_res_pre_dps=yr_pre, yaw_res_post_dps=yr_post,
        yaw_res_post_max_dps=float(np.nanmax(np.abs(run.yaw_res[i_end:q1]))),
        b_yaw_ratio=yr_post / max(yr_pre, 1e-3),
        steer_rate_pre_dps=sr_pre, steer_rate_post_dps=sr_post,
        b_steer_ratio=sr_post / max(sr_pre, lt["steer_rate_floor_dps"]),
        steer_reversals_post=int(np.sum(np.diff(np.sign(run.steer_rate[i_end:q1][
            np.abs(run.steer_rate[i_end:q1]) > lt["steer_rate_floor_dps"]])) != 0)),
        b_lat_m=float(np.max(np.abs(lat_post))),
        b_lat_gps_m=float(np.max(np.abs(gps_post))) if gps_post.size else np.nan,
        avoid_lat_m=float(np.max(np.abs(lat_avoid))),
        dv_kmh=float((np.nanmean(run.v[i_end:q1]) - np.nanmean(run.v[p0:p1])) * 3.6),
    )


B_KEYS = ("b_yaw_ratio", "b_steer_ratio", "b_lat_m", "avoid_lat_m")


def null_behavior(runs: list[Run], events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """振動の無い正常走行から疑似イベントを取り、B の比較基準にする。"""
    fs = cfg["grid_hz"]
    lt = cfg["longtail"]
    rng = np.random.default_rng(lt["null_seed"])
    quiet = int(lt["null_quiet_s"] * fs)
    cand = []
    for ri, run in enumerate(runs):
        busy = np.zeros(len(run.t), bool)
        for _, e in events[events.run == ri].iterrows():
            busy[max(0, e.i_start - quiet):e.i_end + quiet] = True
        ok = np.flatnonzero(~busy & (run.v >= cfg["min_speed_mps"]))
        cand += [(ri, i) for i in ok[::10]]
    pick = rng.choice(len(cand), size=min(lt["null_samples"] * 2, len(cand)), replace=False)
    rows = []
    for k in pick:
        ri, i = cand[k]
        b = behavior(runs[ri], i, i + int(0.5 * fs), cfg)
        if b is not None:
            rows.append(b)
        if len(rows) >= lt["null_samples"]:
            break
    return pd.DataFrame(rows)


def surprise(x: np.ndarray, null: np.ndarray) -> np.ndarray:
    null = np.sort(null[np.isfinite(null)])
    ge = len(null) - np.searchsorted(null, x, side="left")
    return -np.log10((ge + 1) / (len(null) + 1))


# ---------------------------------------------------------------------------
# A: 路面外乱
# ---------------------------------------------------------------------------

def find_events(runs: list[Run], base: pd.DataFrame, cfg: dict, wheelbase: float) -> pd.DataFrame:
    fs = cfg["grid_hz"]
    lt = cfg["longtail"]
    rows = []
    for ri, run in enumerate(runs):
        r = F.ratios(run, base, cfg)
        A = np.sqrt(np.nan_to_num(r["heave"]) * np.maximum(np.nan_to_num(r["roll"]), np.nan_to_num(r["pitch"])))
        pk, _ = find_peaks(A, height=lt["a_min"], distance=int(cfg["event"]["min_separation_s"] * fs))
        for i in pk:
            a, b = F._span(A, i, cfg["event"]["half_width_frac"])
            pad = int(0.2 * fs)
            w = slice(max(0, a - pad), min(len(run.t), b + pad + 1))
            peak = {k: float(np.max(np.abs(run.bp[k][w]))) for k in ("heave", "roll", "pitch")}
            rs = slice(max(0, i - int(0.3 * fs)), min(len(run.t), i + int(2.0 * fs)))
            n_reb, decay = F.rebound(run.bp["roll"][rs], cfg["rebound"]["extremum_frac"])
            nmax = cfg["spike"]["max_samples_above_half"]
            spike = all(int((np.abs(run.bp[k][w]) > 0.5 * peak[k]).sum()) <= nmax for k in ("heave", "roll"))
            v_pk = float(run.v[i])
            lag_exp = wheelbase / max(v_pk, 0.1)
            wsl = slice(max(0, i - int(0.3 * fs)), min(len(run.t), i + int((0.7 + lag_exp) * fs)))
            wr = {k: float(np.nanmax(r[k][wsl])) for k in WHEELS}
            wt = {k: float(run.t[wsl][np.nanargmax(r[k][wsl])] - run.t[a]) for k in WHEELS}
            left, right = max(wr["FL"], wr["RL"]), max(wr["FR"], wr["RR"])
            side = ("FL", "RL") if left >= right else ("FR", "RR")
            seg = int(run.seg[i])
            t0 = run.seg_t0[seg]
            rows.append(dict(
                run=ri, i_start=a, i_end=b, i_peak=i,
                drive_id=run.drive_id, segment=seg,
                t_start_s=round(run.t[a] - t0, 2), t_end_s=round(run.t[b] - t0, 2), t_peak_s=round(run.t[i] - t0, 2),
                speed_kmh=round(v_pk * 3.6, 1),
                heave_peak_mps2=round(peak["heave"], 2), roll_rate_peak_dps=round(peak["roll"], 2),
                pitch_rate_peak_dps=round(peak["pitch"], 2),
                heave_ratio=round(float(r["heave"][i]), 2), roll_ratio=round(float(r["roll"][i]), 2),
                pitch_ratio=round(float(r["pitch"][i]), 2), A=round(float(A[i]), 2),
                roll_over_pitch=round(float(r["roll"][i] / r["pitch"][i]), 2),
                roll_rebounds=n_reb, roll_decay=round(float(decay), 2) if np.isfinite(decay) else np.nan,
                heave_dom_hz=round(F.dominant_hz(run.bp["heave"][w], fs), 1),
                spike_like=spike,
                **{f"wheel_ratio_{k}": round(wr[k], 1) for k in WHEELS},
                wheel_lr_log2=round(float(np.log2(left / right)), 2),
                wheel_side=("左" if side[0] == "FL" else "右"),
                wheel_front_rear_lag_s=round(wt[side[1]] - wt[side[0]], 2),
                wheel_lag_expected_s=round(lag_exp, 2),
            ))
    ev = pd.DataFrame(rows)
    merge = int(0.5 * fs)
    keep = []
    for _, g in ev.sort_values(["run", "i_start"]).groupby("run"):
        cur = None
        for idx, e in g.iterrows():
            if cur is not None and e.i_start <= ev.at[cur, "i_end"] + merge:
                if e.A > ev.at[cur, "A"]:
                    keep[-1] = idx
                    cur = idx
                continue
            keep.append(idx)
            cur = idx
    return ev.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# C: 希少性と地点
# ---------------------------------------------------------------------------

def assign_sites(ev: pd.DataFrame, runs: list[Run], cfg: dict) -> pd.DataFrame:
    lt = cfg["longtail"]
    ev = ev.copy()
    lla, hdg = [], []
    for _, e in ev.iterrows():
        run = runs[e.run]
        t_pk = run.t[e.i_peak]
        k = int(np.clip(np.searchsorted(run.gps_t, t_pk), 0, len(run.gps_t) - 1))
        seg_dir = next(r.path for r in run.refs if r.index == e.segment)
        ft = np.load(seg_dir / "global_pose/frame_times")
        j = int(np.clip(np.searchsorted(ft, t_pk), 0, len(ft) - 1))
        lla.append(F.ecef_to_llh(*np.load(seg_dir / "global_pose/frame_positions")[j]))
        hdg.append(run.gps_hdg[k])
    ev["lat"] = [round(a, 6) for a, _ in lla]
    ev["lon"] = [round(b, 6) for _, b in lla]
    ev["heading_deg"] = np.round(hdg, 1)
    lat0 = np.deg2rad(ev.lat.mean())
    xy = np.c_[ev.lon * 111320 * np.cos(lat0), ev.lat * 110540]
    site = -np.ones(len(ev), int)
    order = np.argsort(-ev.A.to_numpy())
    n = 0
    for i in order:
        if site[i] >= 0:
            continue
        d = np.hypot(*(xy - xy[i]).T)
        dh = np.abs((ev.heading_deg.to_numpy() - ev.heading_deg.iloc[i] + 180) % 360 - 180)
        site[(d <= lt["site_radius_m"]) & (dh <= lt["site_heading_deg"]) & (site < 0)] = n
        n += 1
    ev["site_id"] = site
    # 地点ごとに、何ドライブが通過し、そのうち何ドライブで振動イベントが出たか
    tracks = []
    for run in runs:
        ll = np.array([F.ecef_to_llh(*p) for p in
                       np.vstack([np.load(r.path / "global_pose/frame_positions")[::10] for r in run.refs])])
        tracks.append((run.drive_id, np.c_[ll[:, 1] * 111320 * np.cos(lat0), ll[:, 0] * 110540], run.gps_hdg[::10]))
    passes = {}
    for sid, g in ev.groupby("site_id"):
        c = np.c_[g.lon * 111320 * np.cos(lat0), g.lat * 110540].mean(axis=0)
        h = g.heading_deg.iloc[0]
        drives = set()
        for drive, txy, th in tracks:
            m = min(len(th), len(txy))
            d = np.hypot(*(txy[:m] - c).T)
            th = th[:m]
            dh = np.abs((th - h + 180) % 360 - 180)
            if np.any((d <= lt["site_radius_m"]) & (dh <= lt["site_heading_deg"])):
                drives.add(drive)
        passes[sid] = drives
    ev["site_events"] = ev.groupby("site_id").site_id.transform("size")
    ev["site_drives_event"] = ev.groupby("site_id").drive_id.transform("nunique")
    ev["site_drives_pass"] = ev.site_id.map(lambda s: len(passes[s] | set(ev.drive_id[ev.site_id == s])))
    ev["site_hit_rate"] = (ev.site_drives_event / ev.site_drives_pass).round(2)
    return ev


# ---------------------------------------------------------------------------
# 優先順位と説明
# ---------------------------------------------------------------------------

PRIORITY_LABEL = {
    1: "片輪段差の疑い + ロール・軌跡の乱れ",
    2: "路面衝撃後の操舵補正",
    3: "制動・旋回中の異常振動",
    4: "回避的な軌跡変化の可能性",
    5: "著しく大きな振動",
}

AD_NEEDS = {
    1: "片側入力による横ずれ・ヨー外乱への車線維持制御の外乱抑制。路面の段差を事前に検知して進路・速度を選ぶ判断",
    2: "衝撃で操舵が乱れた後の補正操舵の量と速さ。衝撃直後の過剰な補正を避ける制御",
    3: "制動・旋回で荷重が偏った状態での路面入力への耐性。振動中のセンサ値を鵜呑みにしない状態推定",
    4: "前方の路面凹凸を認識して避けるか、そのまま通過するかの判断",
    5: "強い路面入力を受けた際の乗り心地・センサ健全性の判定と速度調整",
}


def classify(ev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    lt = cfg["longtail"]
    hi = lt["surprise_hi"]
    ev = ev.copy()
    ev["one_sided"] = (ev.wheel_lr_log2.abs() >= lt["one_sided_lr_log2"]) & (ev.roll_over_pitch >= cfg["event"]["roll_heave"]["roll_over_pitch_min"])
    ev["braking"] = ev.brake_any | (ev.gvc_min <= lt["braking_gvc_mps2"])
    ev["turning"] = ev.ay_absmax >= lt["turning_ay_mps2"]
    b_ok = ~ev.lane_change_overlap
    p = {
        1: ev.one_sided & b_ok & ((ev.s_yaw >= hi) | (ev.s_lat >= hi)),
        2: b_ok & (ev.s_steer >= hi),
        3: (ev.braking | ev.turning) & (ev.A_pct >= lt["p3_a_pct"]),
        4: b_ok & (ev.s_avoid >= hi),
        5: ev.A_pct >= lt["a_extreme_pct"],
    }
    ev["priorities"] = [";".join(str(k) for k in p if p[k].iloc[j]) for j in range(len(ev))]
    ev["priority"] = [min((k for k in p if p[k].iloc[j]), default=9) for j in range(len(ev))]
    ev.loc[ev.spike_like, "priority"] = 9
    ev["priority_label"] = ev.priority.map(lambda k: PRIORITY_LABEL.get(k, "対象外"))
    ev["S"] = (ev.A_s + ev.B + ev.C).round(2)
    return ev


def describe(e: pd.Series, cfg: dict) -> dict:
    hi = cfg["longtail"]["surprise_hi"]
    obs = [f"{e.speed_kmh:.0f} km/h で上下 {e.heave_peak_mps2:.1f} m/s²・ロール {e.roll_rate_peak_dps:.1f} deg/s・"
           f"ピッチ {e.pitch_rate_peak_dps:.1f} deg/s の振動 (普段比 上下 {e.heave_ratio:.1f}・ロール {e.roll_ratio:.1f}・ピッチ {e.pitch_ratio:.1f} 倍)"]
    obs.append(f"振動後 {cfg['longtail']['post_s']:.0f} 秒間: 操舵速度最大 {e.steer_rate_post_dps:.1f} deg/s (前の {e.b_steer_ratio:.1f} 倍)、"
               f"ヨーレート残差 RMS {e.yaw_res_post_dps:.2f} deg/s (前の {e.b_yaw_ratio:.1f} 倍)、"
               f"推測航法の横ずれ {e.b_lat_m:.2f} m (GPS {e.b_lat_gps_m:.2f} m)、車速変化 {e.dv_kmh:+.1f} km/h")
    if e.braking:
        obs.append("制動中")
    if e.turning:
        obs.append(f"旋回中 (|横加速度| 最大 {e.ay_absmax:.1f} m/s²)")
    cause = []
    if abs(e.wheel_lr_log2) >= 0.5:
        cause.append(f"輪速 (CAN) の揺れが{e.wheel_side}側で {2 ** abs(e.wheel_lr_log2):.1f} 倍大きい")
    if abs(e.wheel_front_rear_lag_s - e.wheel_lag_expected_s) <= 0.05:
        cause.append(f"{e.wheel_side}前輪→後輪の揺れの時間差 {e.wheel_front_rear_lag_s:.2f} s がホイールベース/車速 {e.wheel_lag_expected_s:.2f} s と一致")
    if e.site_drives_event >= 2:
        cause.append(f"同じ地点・向きで {int(e.site_drives_event)}/{int(e.site_drives_pass)} ドライブに振動イベント")
    effect = []
    if e.s_yaw >= hi:
        effect.append("振動後にヨーレート残差が正常走行の上位 1% を超えて増加")
    if e.s_steer >= hi:
        effect.append("振動後の操舵速度が正常走行の上位 1% を超える")
    if e.s_lat >= hi:
        effect.append("振動後の横ずれが正常走行の上位 1% を超える")
    if e.s_avoid >= hi:
        effect.append("振動の前に進路の横ずれ (回避的な動きの可能性)")
    if not effect:
        effect.append("振動後の挙動変化は正常走行の範囲内")
    usable = ("評価データ候補 (挙動変化あり)" if e.B >= hi else "学習データ候補 (路面入力の多様性)")
    if e.site_drives_event >= 2:
        usable += "。同一地点の繰り返しがあり、同じ路面での再現性評価に使える"
    unconf = ["映像での路面状態の確認", "openpilot の作動状態 (操舵が運転者か制御か)"]
    if e.lane_change_overlap:
        unconf.append("車線変更と重なっており、挙動変化が振動によるものか不明")
    return dict(
        observed="。".join(obs), cause_evidence="。".join(cause) or "IMU 以外の裏付けなし",
        vehicle_effect="。".join(effect), ad_needs=AD_NEEDS.get(e.priority, ""),
        data_use=usable, unconfirmed="。".join(unconf),
    )


# ---------------------------------------------------------------------------
# 図と映像
# ---------------------------------------------------------------------------

def plot_case(run: Run, e: pd.Series, a: int, b: int, out: Path, title: str, cfg: dict) -> None:
    import matplotlib.pyplot as plt

    fs = cfg["grid_hz"]
    lt = cfg["longtail"]
    t0 = run.t[e.i_start]
    t = run.t[a:b] - t0
    end = run.t[e.i_end] - t0
    fig, axs = plt.subplots(7, 1, figsize=(12, 15), sharex=True)
    axs[0].plot(t, run.bp["heave"][a:b], lw=0.8, color="tab:blue")
    axs[0].set_ylabel("上下加速度 [m/s²]\n(0.7〜20 Hz)", fontsize=9)
    axs[1].plot(t, run.bp["roll"][a:b], lw=0.8, color="tab:green", label="ロール (正=右下がり)")
    axs[1].plot(t, run.bp["pitch"][a:b], lw=0.8, color="tab:purple", alpha=0.8, label="ピッチ (正=機首上げ)")
    axs[1].set_ylabel("角速度 [deg/s]\n(0.7〜15 Hz)", fontsize=9)
    axs[1].legend(loc="upper right", fontsize=8)
    axs[2].plot(t, run.v[a:b] * 3.6, color="tab:red", lw=0.9)
    axs[2].set_ylabel("車速 [km/h]", fontsize=9)
    tw = axs[2].twinx()
    tw.plot(t, run.gvc[a:b], color="0.4", lw=0.8, label="GVC 前後G")
    tw.fill_between(t, 0, np.nan_to_num(run.brake[a:b]) > 0.5, color="tab:red", alpha=0.12,
                    transform=tw.get_xaxis_transform(), label="ブレーキ")
    tw.set_ylabel("前後G [m/s²]", fontsize=9)
    tw.legend(loc="upper right", fontsize=8)
    axs[3].plot(t, run.steer[a:b], color="tab:brown", lw=0.9, label="ハンドル角")
    axs[3].set_ylabel("ハンドル角 [deg]", fontsize=9)
    tw = axs[3].twinx()
    tw.plot(t, run.steer_rate[a:b], color="tab:orange", lw=0.7, alpha=0.8, label="操舵速度")
    tw.set_ylabel("操舵速度 [deg/s]", fontsize=9)
    q1 = e.i_end + int(lt["post_s"] * fs)
    k = e.i_end + int(np.nanargmax(np.abs(run.steer_rate[e.i_end:q1])))
    tw.plot(run.t[k] - t0, run.steer_rate[k], "v", color="k", ms=7, label=f"振動後の最大 {abs(run.steer_rate[k]):.1f}")
    tw.legend(loc="upper right", fontsize=8)
    axs[3].legend(loc="upper left", fontsize=8)
    model = bicycle_yaw_rate(run.v, run.steer, **cfg["_geo"])
    axs[4].plot(t, run.yaw[a:b], color="tab:cyan", lw=0.9, label="ヨーレート (CAN)")
    axs[4].plot(t, model[a:b], color="0.5", lw=0.8, ls="--", label="単軌道モデル")
    axs[4].plot(t, run.yaw_res[a:b], color="tab:pink", lw=0.9, label="残差 (0.2〜3 Hz)")
    k = e.i_end + int(np.nanargmax(np.abs(run.yaw_res[e.i_end:q1])))
    axs[4].plot(run.t[k] - t0, run.yaw_res[k], "v", color="k", ms=7)
    axs[4].set_ylabel("ヨーレート [deg/s]", fontsize=9)
    axs[4].legend(loc="upper right", fontsize=8, ncol=3)
    for w, c in zip(WHEELS, ("tab:blue", "tab:red", "tab:cyan", "tab:orange")):
        axs[5].plot(t, run.bp[w][a:b], lw=0.7, color=c, label=w)
    axs[5].set_ylabel("輪速の揺れ [m/s]\n(2〜15 Hz)", fontsize=9)
    axs[5].legend(loc="upper right", fontsize=8, ncol=4)
    h0 = e.i_start - int(lt["heading_ref_s"] * fs)
    lat = lateral_vs_model(run, e.i_end, q1, h0, e.i_start)
    axs[6].plot(run.t[e.i_end:q1 + 1] - t0, lat, color="tab:olive", lw=1.2, label="振動後: 舵角どおりの進路からのずれ (CAN)")
    gt, gl = _gps_lateral(run, e.i_end, q1)
    if gt.size:
        axs[6].plot(gt - t0, gl, "o-", ms=2, lw=0.8, color="tab:gray", label="同 (GPS)")
    a0 = e.i_start + int(lt["avoid_pre_s"][0] * fs)
    pre = _lateral(run, run.yaw_road, a0, e.i_start)
    axs[6].plot(run.t[a0:e.i_start + 1] - t0, pre, color="tab:olive", lw=1.0, ls=":", label="振動前: 道路の曲率を除いた横の動き")
    axs[6].axhline(0, color="k", lw=0.4)
    axs[6].set_ylabel("横ずれ [m]\n(左正)", fontsize=9)
    axs[6].legend(loc="upper left", fontsize=8)
    p0, p1 = (x for x in lt["pre_s"])
    for ax in axs:
        ax.axvspan(0, end, color="orange", alpha=0.25)
        ax.axvspan(p0, p1, color="0.85", alpha=0.35)
        ax.axvspan(end, end + lt["post_s"], color="tab:blue", alpha=0.06)
        ax.axvline(0, color="red", lw=0.8)
        ax.grid(alpha=0.3)
    for w, c in zip(WHEELS, ("tab:blue", "tab:red", "tab:cyan", "tab:orange")):
        lag = run.t[e.i_start:q1][np.nanargmax(np.abs(run.bp[w][e.i_start:q1]))] - t0
        axs[5].axvline(lag, color=c, lw=0.6, ls="--")
    axs[-1].set_xlim(t[0], t[-1])
    axs[-1].set_xlabel("振動開始からの時刻 [s]   (灰 = 振動前の基準窓 / 橙 = 振動 / 青 = 振動後の評価窓)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def clip_case(run: Run, a_abs: float, b_abs: float, out: Path) -> dict:
    parts = [SimpleNamespace(path=r.path, index=r.index) for r in run.refs]
    return F.clip_video(parts, SimpleNamespace(t0=0.0), a_abs, b_abs, out)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root", type=Path)
    ap.add_argument("--config", type=Path, default=REPO / "configs/vibration.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "out/vibration/longtail")
    ap.add_argument("--clips", type=int, default=0, help="上位この件数 (独立地点) の動画とグラフを作る")
    ap.add_argument("--video-dir", type=Path, default=REPO / "out/vibration/videos")
    ap.add_argument("--plot-dir", type=Path, default=REPO / "out/vibration/plots")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    lt = cfg["longtail"]
    det_cfg = load_yaml(REPO / "configs/detection.yaml")
    args.out.mkdir(parents=True, exist_ok=True)

    refs = find_segments(args.data_root)
    veh = find_vehicle_config(refs[0].dongle_id, load_vehicle_configs())
    geo = veh.geometry
    cfg["_geo"] = dict(wheelbase_m=geo["wheelbase_m"], steer_ratio=geo["steer_ratio"],
                understeer_gradient=geo["understeer_gradient"], steer_offset_deg=geo["steer_offset_deg"])
    use = [r for r in refs if F.gyro_rate_hz(r.path) >= cfg["min_gyro_hz"]]
    runs = build_runs(use, veh, cfg, det_cfg)
    print(f"セグメント {len(use)}/{len(refs)} 本、連続区間 {len(runs)} 本")

    base = F.baselines(runs, cfg)
    ev = find_events(runs, base, cfg, geo["wheelbase_m"])

    fs = cfg["grid_hz"]
    beh, ctx = [], []
    for _, e in ev.iterrows():
        run = runs[e.run]
        beh.append(behavior(run, e.i_start, e.i_end, cfg) or {})
        c0 = max(0, e.i_start - int(fs))
        c1 = min(len(run.t), e.i_end + int(fs))
        q1 = e.i_end + int(lt["post_s"] * fs)
        lc = any(x.i_start <= q1 and x.i_end >= e.i_start - 2 * fs for x in run.lane_changes)
        ctx.append(dict(brake_any=bool(np.nansum(run.brake[c0:c1] > 0.5) > 0),
                        gvc_min=float(np.nanmin(run.gvc[c0:c1])) if np.isfinite(run.gvc[c0:c1]).any() else np.nan,
                        ay_absmax=float(np.nanmax(np.abs(run.ay[c0:e.i_end]))),
                        lane_change_overlap=lc))
    ev = pd.concat([ev, pd.DataFrame(beh), pd.DataFrame(ctx)], axis=1)
    n_all = len(ev)
    ev = ev[ev.b_yaw_ratio.notna()].reset_index(drop=True)

    null = null_behavior(runs, ev, cfg)
    null.to_csv(args.out / "null_behavior.csv", index=False)
    for k, s in (("b_yaw_ratio", "s_yaw"), ("b_steer_ratio", "s_steer"), ("b_lat_m", "s_lat"), ("avoid_lat_m", "s_avoid")):
        ev[s] = surprise(ev[k].to_numpy(), null[k].to_numpy()).round(2)
    ev["B"] = ev[["s_yaw", "s_steer", "s_lat"]].max(axis=1)
    ev["A_pct"] = ev.A.rank(pct=True).round(3)
    ev["A_s"] = (-np.log10(1 - ev.A_pct + 1 / (len(ev) + 1))).round(2)
    a, b = ev.A.to_numpy(), ev.B.to_numpy()
    ev["C"] = [round(-np.log10(((a >= a[i]) & (b >= b[i])).sum() / len(ev)), 2) for i in range(len(ev))]
    ev = assign_sites(ev, runs, cfg)
    ev = classify(ev, cfg)
    ev = ev.sort_values(["priority", "S"], ascending=[True, False]).reset_index(drop=True)
    text = pd.DataFrame([describe(e, cfg) for _, e in ev.iterrows()])
    ev = pd.concat([ev, text], axis=1)
    ev.insert(0, "label", "ロングテール候補")

    drop = ["run", "i_start", "i_end", "i_peak"]
    ev.drop(columns=drop).to_csv(args.out / "events_all.csv", index=False)
    top = ev[ev.priority < 9].head(lt["top_n"]).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top.drop(columns=drop).to_csv(args.out / "longtail_top20.csv", index=False)
    sites = (ev[ev.priority < 9].groupby("site_id")
             .agg(lat=("lat", "mean"), lon=("lon", "mean"), heading_deg=("heading_deg", "first"),
                  events=("A", "size"), drives_event=("site_drives_event", "first"),
                  drives_pass=("site_drives_pass", "first"), best_priority=("priority", "min"),
                  best_S=("S", "max"))
             .sort_values(["best_priority", "best_S"], ascending=[True, False]))
    sites.to_csv(args.out / "sites.csv")

    clips = []
    if args.clips:
        matplotlib.use("Agg")
        configure_japanese_font()
        args.video_dir.mkdir(parents=True, exist_ok=True)
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        reps = top.drop_duplicates("site_id").head(args.clips)
        for n, (_, e) in enumerate(reps.iterrows(), 1):
            run = runs[e.run]
            a_abs = max(run.t[e.i_start] - 5.0, run.t[0])
            b_abs = min(run.t[e.i_end] + 5.0, run.t[-1])
            t0 = run.seg_t0[e.segment]
            stem = (f"lt{n:02d}_rank{int(e['rank']):02d}_{F.drive_tag(e.drive_id)}_seg{e.segment}"
                    f"_{a_abs - t0:.1f}-{b_abs - t0:.1f}")
            info = clip_case(run, a_abs, b_abs, args.video_dir / f"{stem}.mp4")
            i0 = int(np.searchsorted(run.t, a_abs))
            i1 = int(np.searchsorted(run.t, b_abs))
            title = (f"ロングテール候補 #{n} (順位 {int(e['rank'])}) 優先 {e.priority}: {e.priority_label}\n"
                     f"{e.drive_id} seg{e.segment} {e.t_start_s:.2f}–{e.t_end_s:.2f} s  {e.speed_kmh:.0f} km/h  "
                     f"A={e.A:.1f} B={e.B:.1f} C={e.C:.1f}  地点 {int(e.site_id)} ({int(e.site_drives_event)}/{int(e.site_drives_pass)} ドライブ)")
            plot_case(run, e, i0, i1, args.plot_dir / f"{stem}.png", title, cfg)
            clips.append(dict(case=n, rank=int(e["rank"]), site_id=int(e.site_id), stem=stem,
                              clip_start_s=round(a_abs - t0, 2), clip_end_s=round(b_abs - t0, 2), **info))
        pd.DataFrame(clips).to_csv(args.out / "clips.csv", index=False)

    meta = dict(
        config=str(args.config.relative_to(REPO)),
        config_sha1=hashlib.sha1(args.config.read_bytes()).hexdigest()[:10],
        segments_used=len(use), runs=len(runs),
        events_detected=n_all, events_scored=len(ev), null_samples=len(null),
        priority_counts=ev.priority.value_counts().sort_index().to_dict(),
        sites_ranked=int(ev[ev.priority < 9].site_id.nunique()),
        top_sites=int(top.site_id.nunique()),
    )
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=int))
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=int))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
