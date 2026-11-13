#!/usr/bin/env python3
"""ZOD full Phase 3: Phase 2 の外乱候補 (14 件) と、比べる区間を、SMHI・DWD の観測と照らし合わせる。

  uv run --offline --extra viz --with "zod==0.8.0" --with "dataclass-wizard==0.39.1" python scripts/zod/full_weather.py plan
  uv run --extra viz --with "zod==0.8.0" --with "dataclass-wizard==0.39.1" python scripts/zod/full_weather.py fetch      # ネットワーク (SMHI・DWD だけ)
  uv run --offline --extra viz --with "zod==0.8.0" --with "dataclass-wizard==0.39.1" python scripts/zod/full_weather.py analyze --run-id <id>

設定: configs/zod/full_weather.yaml (判定の規則は取得の前に書いた)。説明: docs/zod_full_weather.md
Phase 2 のモデル・閾値は変えない (Phase 2 の出力の残差をそのまま使う)。観測は車両の位置の風の真値ではない。

横風成分 W_left = −W·sin(D − H): 左から吹くと正 (D: 風が吹いてくる方位、H: 進む方位、北から時計回り)。
風のベクトル (吹いていく向き) u = −W·sin D (東)、v = −W·cos D (北)。W_left = u·cos H − v·sin H。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_validation as D  # noqa: E402
import full_screening as F  # noqa: E402
import weather_common as W  # noqa: E402
import weather_match as WM  # noqa: E402
import event_time as ET  # noqa: E402

G = 9.80665
OUT_PLAN = Path("out/zod/full_weather/plan")


def r3(x, n=3):
    try:
        return None if x is None or not np.isfinite(x) else round(float(x), n)
    except TypeError:
        return None


def load_cfg(p):
    return yaml.safe_load(Path(p).read_text())


# ================================================================ 観測所の一覧 (手元のもの)
def station_lists(cfg):
    sc = cfg["stations"]
    a = datetime.fromisoformat(sc["active_from"]).replace(tzinfo=timezone.utc).timestamp() * 1000
    b = datetime.fromisoformat(sc["active_to"]).replace(tzinfo=timezone.utc).timestamp() * 1000
    sets = []
    for p in cfg["fetch"]["smhi_params"]:
        j = json.loads(Path(f"data/zod_weather/smhi/parameter_{p}.json").read_text())
        sets.append({str(s["key"]): (s["latitude"], s["longitude"], s.get("height") or 0.0, s["name"]) for s in j["station"] if s["from"] < a and s["to"] > b})
    common = set.intersection(*[set(s) for s in sets])
    sm = pd.DataFrame([dict(source="SMHI", sid=k, lat=sets[0][k][0], lon=sets[0][k][1], height_m=float(sets[0][k][2]), name=sets[0][k][3]) for k in sorted(common)])
    rows = {}
    for fn in ["zehn_min_ff_Beschreibung_Stationen.txt", "zehn_min_fx_Beschreibung_Stationen.txt"]:
        ids = {}
        for line in Path(f"data/zod_weather/dwd/{fn}").read_bytes().decode("latin-1").splitlines()[2:]:
            m = re.match(r"\s*(\d+)\s+(\d{8})\s+(\d{8})\s+(-?\d+)\s+([\d.]+)\s+([\d.]+)\s+(.*)$", line)
            if m and m.group(2) <= sc["active_from"].replace("-", "") and m.group(3) >= sc["active_to"].replace("-", ""):
                ids[m.group(1).zfill(5)] = (float(m.group(5)), float(m.group(6)), re.split(r"\s{2,}", m.group(7).strip())[0], float(m.group(4)))
        rows[fn] = ids
    dw_ids = set(rows["zehn_min_ff_Beschreibung_Stationen.txt"])
    ff = rows["zehn_min_ff_Beschreibung_Stationen.txt"]
    dw = pd.DataFrame([dict(source="DWD", sid=k, lat=ff[k][0], lon=ff[k][1], name=ff[k][2], height_m=ff[k][3],
                            has_fx=k in rows["zehn_min_fx_Beschreibung_Stationen.txt"]) for k in sorted(dw_ids)])
    return pd.concat([sm, dw], ignore_index=True)


def nearby(st, lat, lon, cfg):
    d = D.dist_m(lat, lon, st.lat.to_numpy(), st.lon.to_numpy()) / 1000
    o = st.assign(dist_km=d)
    o = o[o.dist_km <= cfg["stations"]["radius_km"]].sort_values("dist_km")
    return o.head(cfg["stations"]["max_per_point"])


# ================================================================ 位置・時刻 (候補・対照・比べる区間)
def drive_tracks(cfg, drives):
    """走行ごとの 20 Hz の時系列 (Phase 2 の残差 + OXTS の位置・方位・ロール・横加速度)。"""
    scr = Path(cfg["screening_run"])
    inv = Path(cfg["inventory_run"])
    seqs = pd.read_csv(inv / "sequences.csv", dtype={"id": str})
    F.ROOT = Path("raw_data/zod")
    out = {}
    for d in drives:
        z = dict(np.load(scr / "residuals" / f"{d}.npz"))
        ox = F.oxts_union(sorted(seqs[seqs.drive_id == d].id)).iloc[::5].reset_index(drop=True)
        if len(ox) != len(z["t"]) or np.max(np.abs(ox.t.to_numpy() - z["t"])) > 1e-3:
            raise RuntimeError(f"{d}: 残差と OXTS の時刻が合わない")
        z.update(lat=ox.posLat.to_numpy(), lon=ox.posLon.to_numpy(), heading=ox.heading.to_numpy(), roll=ox.roll.to_numpy(),
                 ox_ay_raw=-ox.accelerationY.to_numpy())
        out[d] = z
    return out


def slow20(dr, blk, n=100):
    """Phase 2 の持続成分 (5 s の移動平均) を 20 Hz で作る (100 点 = 5 s)。"""
    s = np.empty_like(dr, dtype=float)
    for k in np.unique(blk):
        m = np.where(blk == k)[0]
        s[m] = pd.Series(dr[m].astype(float)).rolling(n, center=True, min_periods=1).mean().to_numpy()
    return s


def segments(tracks, cand_all, cfg):
    """判定の対象 (80 km/h、Phase 2 の mask80) を 60 s ずつに区切る。候補 (全 67 件) の ±30 s を含む区間は印を付ける。"""
    cc = cfg["compare"]
    rows = []
    for d, z in tracks.items():
        m = z["mask80"].astype(bool)
        t = z["t"]
        civ = [ET.event_interval(r) for _, r in cand_all[cand_all.drive_id == d].iterrows()]   # 閾値超過区間 (event_time)
        for k in np.unique(z["blk"]):
            bi = np.where(z["blk"] == k)[0]
            for s0 in np.arange(t[bi[0]], t[bi[-1]], cc["segment_s"]):
                w = bi[(t[bi] >= s0) & (t[bi] < s0 + cc["segment_s"])]
                w = w[m[w]]
                if len(w) < cc["min_screened_s"] * 20:
                    continue
                i = w[len(w) // 2]
                rows.append(dict(drive_id=d, t0=float(t[w[0]]), t1=float(t[w[-1]]), t=float(t[i]), lat=float(z["lat"][i]), lon=float(z["lon"][i]),
                                 heading=float(z["heading"][i]), speed_kmh=float(np.mean(z["v"][w]) * 3.6), screened_s=len(w) / 20,
                                 idx=w, has_candidate=any(ET.overlap(a_, b_, t[w[0]] - 30, t[w[-1]] + 30) > 0 for a_, b_ in civ)))
    return pd.DataFrame(rows)


# ================================================================ plan
def plan(cfg):
    scr = Path(cfg["screening_run"])
    u = pd.read_csv(scr / "candidates_unique.csv", dtype={"in_camera_sequences": str, "sequence_ids": str})
    c14 = u[u.hypothesis.isin(cfg["candidate_hypotheses"])]
    ctrl = pd.read_csv(scr / "controls.csv")
    st = station_lists(cfg)
    drives = sorted(p.stem for p in (scr / "residuals").glob("*.npz"))
    tracks = drive_tracks(cfg, drives)
    seg = segments(tracks, u, cfg)
    need = []
    for _, r in c14.iterrows():
        for _, s in nearby(st, r.lat, r.lon, cfg).iterrows():
            need.append(dict(purpose="候補", ref=r.event_id, date=r.t_onset_utc[:10], source=s.source, sid=s.sid, dist_km=round(s.dist_km, 1)))
    for _, r in ctrl[ctrl.event_id.isin(c14.event_id)].iterrows():
        z = tracks[r.control_drive_id]
        i = int(np.argmin(np.abs(z["t"] - r.control_t_center)))
        for _, s in nearby(st, z["lat"][i], z["lon"][i], cfg).iterrows():
            need.append(dict(purpose="対照 (Phase 2)", ref=r.event_id, date=r.control_t_center_utc[:10], source=s.source, sid=s.sid, dist_km=round(s.dist_km, 1)))
    for _, r in seg.iterrows():
        for _, s in nearby(st, r.lat, r.lon, cfg).iterrows():
            need.append(dict(purpose="比べる区間", ref=f"{r.drive_id}@{int(r.t)}", date=datetime.fromtimestamp(r.t, timezone.utc).date().isoformat(),
                             source=s.source, sid=s.sid, dist_km=round(s.dist_km, 1)))
    need = pd.DataFrame(need)
    fc = cfg["fetch"]

    def have(src, sid):
        if src == "SMHI":
            return all(any((Path(dd) / "smhi" / f"p{p}_s{sid}_corrected-archive.csv").exists() for dd in fc["reuse_dirs"] + [fc["out_dir"]]) for p in fc["smhi_params"])
        return any(any((Path(dd) / "dwd").glob(f"10minutenwerte_wind_{sid}_*_hist.zip")) for dd in fc["reuse_dirs"] + [fc["out_dir"]])

    stn = need.drop_duplicates(["source", "sid"])[["source", "sid"]].copy()
    stn["already"] = [have(a, b) for a, b in zip(stn.source, stn.sid)]
    stn["mb_est"] = np.where(stn.already, 0.0, np.where(stn.source == "SMHI", fc["smhi_mb_per_file_est"] * len(fc["smhi_params"]), fc["dwd_mb_per_station_est"]))
    pri = {"候補": 0, "対照 (Phase 2)": 1, "比べる区間": 2}
    first = need.assign(p=need.purpose.map(pri)).sort_values("p").drop_duplicates(["source", "sid"])[["source", "sid", "purpose"]]
    stn = stn.merge(first, on=["source", "sid"])
    # 比べる区間の観測所は、上限に収まるように、多くの区間に使われるものから選ぶ
    use = need.groupby(["source", "sid"]).ref.nunique().rename("n_points")
    stn = stn.merge(use, left_on=["source", "sid"], right_index=True)
    stn["p"] = stn.purpose.map(pri)
    stn = stn.sort_values(["p", "n_points"], ascending=[True, False])
    stn["cum_mb"] = stn.mb_est.cumsum()
    stn["selected"] = (stn.p < 2) | (stn.cum_mb <= fc["max_total_mb"])
    OUT_PLAN.mkdir(parents=True, exist_ok=True)
    need.to_csv(OUT_PLAN / "station_needs.csv", index=False)
    stn.drop(columns="p").to_csv(OUT_PLAN / "fetch_plan.csv", index=False)
    seg.drop(columns="idx").to_csv(OUT_PLAN / "segments.csv", index=False)
    sel = stn[stn.selected]
    covered = need.merge(sel[["source", "sid"]], on=["source", "sid"])
    summ = dict(n_candidates=len(c14), n_segments=len(seg), n_segment_drives=int(seg.drive_id.nunique()),
                stations_total=len(stn), stations_selected=len(sel), stations_already=int(sel.already.sum()),
                mb_new_selected=round(float(sel.mb_est.sum()), 1), mb_new_all=round(float(stn.mb_est.sum()), 1),
                by_source=sel.groupby("source").agg(n=("sid", "size"), mb=("mb_est", "sum")).round(1).to_dict("index"),
                segments_with_station=int(covered[covered.purpose == "比べる区間"].ref.nunique()),
                candidates_with_station=int(covered[covered.purpose == "候補"].ref.nunique()),
                dates=sorted(need[need.source.isin(sel.source)].date.unique().tolist()))
    (OUT_PLAN / "plan_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summ.items() if k != "dates"}, ensure_ascii=False, indent=1))
    return 0


# ================================================================ fetch
def fetch(cfg):
    fc = cfg["fetch"]
    lic = load_cfg("configs/zod/weather.yaml")
    W.DATA = Path(fc["out_dir"])                       # 記録は data/zod_weather/phase3/fetch_log.csv (新しいファイル)
    W.LOG = W.DATA / "fetch_log.csv"
    plan_ = pd.read_csv(OUT_PLAN / "fetch_plan.csv", dtype={"sid": str})
    plan_ = plan_[plan_.selected & ~plan_.already]
    t0 = time.time()
    got = []
    idx = {}
    for prod, folder in fc["dwd_products"].items():
        cache = Path("data/zod_weather/dwd") / f"index_{folder.replace('/', '_')}.txt"   # 前回の取得で保存したファイル名の一覧 (再利用)
        idx[prod] = (folder, cache.read_text().split())
    for _, r in plan_.iterrows():
        if r.source == "SMHI":
            for p in fc["smhi_params"]:
                url = f"{lic['smhi']['base']}/parameter/{p}/station/{r.sid}/period/corrected-archive/data.csv"
                f = W.fetch(url, W.DATA / "smhi" / f"p{p}_s{r.sid}_corrected-archive.csv", "SMHI", lic["smhi"]["license"], fc["max_mb_per_file"], 180)
                got.append(dict(source="SMHI", sid=r.sid, product=f"p{p}", path=str(f) if f else "", ok=f is not None))
        else:
            for prod, (folder, names) in idx.items():
                for n in names:
                    m = re.search(rf"_{r.sid}_(\d{{8}})_(\d{{8}})_hist\.zip$", n)
                    if m and m.group(1) <= fc["date_to"] and m.group(2) >= fc["date_from"]:
                        url = f"{lic['dwd']['base']}/{folder}/{n}"
                        f = W.fetch(url, W.DATA / "dwd" / n, "DWD", lic["dwd"]["license"], fc["max_mb_per_file"], 180)
                        got.append(dict(source="DWD", sid=r.sid, product=prod, path=str(f) if f else "", ok=f is not None))
    g = pd.DataFrame(got)
    g.to_csv(W.DATA / "fetched_files.csv", index=False)
    sz = sum(Path(p).stat().st_size for p in g.path if p) / 1e6
    print(f"{int(g.ok.sum())} / {len(g)} ファイル、{sz:.1f} MB、{time.time() - t0:.0f} s")
    return 0


# ================================================================ analyze: 観測の読み込み
def find_files(cfg, src, sid):
    fc = cfg["fetch"]
    dirs = [Path(d) for d in fc["reuse_dirs"] + [fc["out_dir"]]]
    if src == "SMHI":
        out = []
        for p in fc["smhi_params"]:
            for d in dirs:
                f = d / "smhi" / f"p{p}_s{sid}_corrected-archive.csv"
                if f.exists():
                    out.append((p, f))
                    break
        return out
    fs = []
    for d in dirs:
        fs += list((d / "dwd").glob(f"10minutenwerte_wind_{sid}_*_hist.zip")) + list((d / "dwd").glob(f"10minutenwerte_extrema_wind_{sid}_*_hist.zip"))
    return fs


def load_obs(cfg, stations):
    """観測所ごとに、2022 年の speed・dir・gust (DWD の FX_10) の時系列 (UNIX 秒) を返す。"""
    obs = {}
    for _, s in stations.iterrows():
        parts = []
        if s.source == "SMHI":
            for p, f in find_files(cfg, "SMHI", s.sid):
                parts.append(WM.read_smhi(f, p))
        else:
            for f in find_files(cfg, "DWD", s.sid):
                parts.append(WM.read_dwd(f))
        if not parts:
            continue
        d = pd.concat(parts)
        d = d[(d.t >= "2022-01-01") & (d.t < "2023-01-01")]
        d = d[d.value.notna()]
        out = {}
        for q in ["speed", "dir", "gust"]:
            x = d[d.quantity == q].sort_values("t")
            out[q] = (x.t.astype("int64").to_numpy() / 1e9, x.value.to_numpy(float))
        out["step_min"] = float(np.median(np.diff(out["speed"][0])) / 60) if len(out["speed"][0]) > 2 else np.nan
        obs[(s.source, s.sid)] = out
    return obs


def at(series, t, max_gap_min, end_label=True):
    """時刻 t に最も近い観測。DWD の 10 分値・SMHI の値は、その時刻までの 10 分の平均 (区間の終わりの時刻)。
    区間の中心に合わせるため、比べる時刻は t + 5 分にする。"""
    ts, v = series
    if not len(ts):
        return np.nan, np.nan
    tt = t + 300 if end_label else t
    i = int(np.argmin(np.abs(ts - tt)))
    gap = (ts[i] - tt) / 60
    return (float(v[i]), gap) if abs(gap) <= max_gap_min else (np.nan, gap)


def wind_point(obs, st, lat, lon, t, H, cfg):
    """1 地点・1 時刻の風: 観測所ごとの値、最寄り、IDW、ばらつき、1 つ抜きの誤差。"""
    mc = cfg["match"]
    near = nearby(st, lat, lon, cfg)
    rows = []
    for _, s in near.iterrows():
        o = obs.get((s.source, s.sid))
        if o is None:
            continue
        gap = mc["dwd_max_gap_min"] if s.source == "DWD" else mc["smhi_max_gap_min"]
        w, g1 = at(o["speed"], t, gap)
        dd, _ = at(o["dir"], t, gap)
        gu, _ = at(o["gust"], t, gap) if s.source == "DWD" else (np.nan, np.nan)
        ts, vs = o["speed"]
        win = (ts >= t - mc["temporal_window_min"] * 60) & (ts <= t + mc["temporal_window_min"] * 60)
        tsd = float(np.std(vs[win])) if win.sum() >= 3 else np.nan
        if not (np.isfinite(w) and np.isfinite(dd)):
            continue
        u, v = -w * np.sin(np.radians(dd)), -w * np.cos(np.radians(dd))
        rows.append(dict(source=s.source, sid=s.sid, dist_km=s.dist_km, speed=w, dir=dd, gust10=gu, gap_min=g1, step_min=o["step_min"],
                         temporal_std=tsd, u=u, v=v, w_left=-w * np.sin(np.radians(dd - H))))
    st_ = pd.DataFrame(rows)
    res = dict(n_stations=len(st_))
    if not len(st_):
        return res, st_
    n0 = st_.iloc[0]
    res.update(nearest_sid=f"{n0.source}:{n0.sid}", nearest_km=r3(n0.dist_km, 1), nearest_speed=r3(n0.speed, 2), nearest_dir=r3(n0.dir, 0),
               nearest_w_left=r3(n0.w_left, 2), nearest_gap_min=r3(n0.gap_min, 1), nearest_step_min=r3(n0.step_min, 0),
               gust10_max=r3(st_.gust10.max(), 2) if st_.gust10.notna().any() else None,
               temporal_std_median=r3(st_.temporal_std.median(), 2), stations=" ".join(f"{a}:{b}@{c:.0f}km" for a, b, c in zip(st_.source, st_.sid, st_.dist_km)))
    if len(st_) >= mc["idw_min_stations"]:
        wgt = 1 / np.maximum(st_.dist_km.to_numpy(), 0.5) ** mc["idw_power"]
        u, v = np.average(st_.u, weights=wgt), np.average(st_.v, weights=wgt)
        res.update(idw_u=r3(u, 2), idw_v=r3(v, 2), idw_speed=r3(np.hypot(u, v), 2), idw_w_left=r3(u * np.cos(np.radians(H)) - v * np.sin(np.radians(H)), 2),
                   spread_w_left=r3(st_.w_left.std(ddof=1), 2), spread_speed=r3(st_.speed.std(ddof=1), 2))
        if len(st_) >= 3:                                   # 1 つ抜き: ほかの観測所の IDW で、抜いた観測所の横風成分を当てる
            err = []
            for i in range(len(st_)):
                o = st_.drop(st_.index[i])
                ww = 1 / np.maximum(o.dist_km.to_numpy(), 0.5) ** mc["idw_power"]
                err.append(np.average(o.w_left, weights=ww) - st_.w_left.iloc[i])
            res["loo_rmse_w_left"] = r3(np.sqrt(np.mean(np.square(err))), 2)
    return res, st_


def uncert(res, cfg):
    """横風成分の不確かさ (1σ の目安): 観測所の間のばらつきと 1 つ抜きの誤差の二乗和。観測所が 1 つなら、mini の観測所間の差の上限 (4.5 m/s) を置く。"""
    s, l = res.get("spread_w_left"), res.get("loo_rmse_w_left")
    parts = [x for x in (s, l) if x is not None]
    if res.get("n_stations", 0) < 2 or not parts:
        return cfg["match"].get("single_station_sigma", 4.5), "観測所 1 つ (mini の観測所間の差の上限を仮に置く)"
    return float(np.sqrt(np.sum(np.square(parts)))), "観測所間のばらつきと 1 つ抜きの誤差" if l is not None else "観測所間のばらつき (2 か所)"


# ================================================================ analyze: 候補の物理の確かめ
def event_physics(z, ev, cfg, tq_sign):
    """持続型・突発型の候補の、舵角・トルク・β・ロール (横断勾配)・横加速度の向き。z は 20 Hz の時系列。"""
    t = z["t"]
    # 見る区間 = 閾値超過区間 (event_time)。2 s より短ければ 2 s まで延ばす (従来の規則)。
    # 基準 (β・局所のロール) = 推定開始の 10〜0.5 s 前 (従来どおり。推定開始が OXTS の頭で打ち切られていれば基準は無い)
    t0, t1 = ET.event_interval(ev)
    t1 = max(t1, t0 + 2.0)
    tr = float(ev.t_rise)
    blk0 = z["blk"][int(np.argmin(np.abs(t - t0)))]
    w = (t >= t0) & (t <= t1) & (z["blk"] == blk0)
    pre = (t >= tr - 10) & (t < tr - 0.5) & (z["blk"] == blk0)
    base = z["mask80"].astype(bool) | z["mask70"].astype(bool)
    v = z["v"][w].astype(float)
    dr = np.radians(z["dr_cor"][w].astype(float))
    a_steer = float(np.mean(-v * dr))                              # 舵角が求める横加速度の異常 (左が正) = v·(r_pred − r)
    gs = G * np.sin(np.radians(z["roll"].astype(float)))
    a_bank = float(np.mean(gs[w]) - np.median(gs[base]))           # ロール (横断勾配 + 車体のロール) による重力の横成分の、走行の中央値からの差
    a_bank_pre = float(np.mean(gs[w]) - np.median(gs[pre])) if pre.sum() > 20 else np.nan
    ay_raw = z["ox_ay_raw"].astype(float)
    spec = float(np.mean(ay_raw[w] - v * np.radians(z["r_pred_cor"][w].astype(float))) - np.median((ay_raw - z["v"] * np.radians(z["r_pred_cor"]))[base]))
    beta = z["beta"].astype(float)
    bq = bool(np.nanmax(z["beta_unc"][w]) <= 0.1) and bool(np.nanmedian(np.abs(beta[(z["blk"] == z["blk"][w][0])])) <= 3.0) if w.any() else False
    dbeta = float(np.mean(beta[w]) - np.median(beta[pre])) if pre.sum() > 20 else np.nan
    tq = z["torque"].astype(float)
    dtq = float(np.mean(tq[w]) - np.median(tq[base]))
    steer_anom = float(np.mean(-dr))                              # 当て舵の向き: r_pred − r > 0 なら、実際の動きに対して舵角が左に余分 (= −Δr)
    ratio = a_bank / a_steer if abs(a_steer) > 1e-3 else np.nan
    exp_sign_force = -np.sign(np.mean(dr))                        # Δr < 0 → 右向きの外力 → 期待する横風は左から (W_left > 0)
    return dict(window_s=r3(t1 - t0, 1), a_steer=r3(a_steer, 3), a_bank=r3(a_bank, 3), a_bank_local=r3(a_bank_pre, 3), bank_ratio=r3(ratio, 2),
                spec_force_resid=r3(spec, 3), steer_left=bool(steer_anom > 0), torque_anom_nm=r3(dtq, 2),
                torque_left=bool(np.sign(dtq) * tq_sign > 0), torque_agrees_steer=bool((np.sign(dtq) * tq_sign > 0) == (steer_anom > 0)),
                dbeta_deg=r3(dbeta, 3), beta_ok=bq, beta_agrees=bool(np.isfinite(dbeta) and np.sign(dbeta) == np.sign(np.mean(dr))) if bq else None,   # 右向きの外力 (Δr < 0) なら β も負に
                expected_w_left_sign=int(exp_sign_force), mean_dr_deg_s=r3(np.degrees(np.mean(dr)), 3), speed_ms=r3(np.mean(v), 2))


def required_wind(a_steer, v, cfg):
    p, h = cfg["physics"], cfg["match"]["vehicle_height_factor"]
    a = abs(a_steer)
    lo = 2 * p["mass_kg"] * a / (p["rho"] * p["area_m2"] * max(p["cy_per_rad"]) * v)
    hi = 2 * p["mass_kg"] * a / (p["rho"] * p["area_m2"] * min(p["cy_per_rad"]) * v)
    return lo / max(h), hi / min(h)                                # 10 m の高さの値に直す


def judge_sustained(ph, res, sig, req, cfg):
    j, pc = cfg["judgement"], cfg["physics"]
    reasons = []
    w = res.get("idw_w_left", res.get("nearest_w_left"))
    if w is None:
        return "不明", "風の観測なし"
    we = ph["expected_w_left_sign"] * w                           # 期待の向きを正にした横風成分
    lo, hi = we - sig, we + sig
    r = ph["bank_ratio"]
    bank_explains = r is not None and pc["bank_ratio_explains"][0] <= r <= pc["bank_ratio_explains"][1]
    bank_small = r is not None and abs(r) < pc["bank_ratio_small"]
    if bank_explains:
        reasons.append(f"ロールの変化 (横断勾配) で舵角の異常の {r:.0%} を説明できる")
    if hi < 0:
        reasons.append(f"推定の横風 ({w:+.1f} ± {sig:.1f} m/s) は期待と逆向き")
    if hi < j["weak_upper_fraction"] * req[0]:
        reasons.append(f"推定の上限 {max(hi, 0):.1f} m/s は必要量の下限 {req[0]:.1f} m/s の 1/3 未満")
    if reasons:
        return "反証", " / ".join(reasons)
    if we > 0 and hi >= req[0] and bank_small:
        return "支持", f"期待の向きの横風 {we:.1f} ± {sig:.1f} m/s が必要量 {req[0]:.1f}〜{req[1]:.1f} m/s に届きうる。傾きでは説明できない (比 {r:.2f})"
    return "不明", f"期待の向き {we:+.1f} ± {sig:.1f} m/s、必要量 {req[0]:.1f}〜{req[1]:.1f} m/s、傾きの比 {r if r is not None else float('nan'):.2f}"


def judge_transient(res, cfg):
    g = res.get("gust10_max")
    if g is None:
        return "不明", "10 分の最大瞬間風速の観測なし (SMHI は時刻の分からない 1 時間値しか無いので使わない)"
    if g < cfg["judgement"]["gust_needed_ms"]:
        return "反証", f"近くの観測所の 10 分の最大瞬間風速 {g:.1f} m/s は検出限界の目安 {cfg['judgement']['gust_needed_ms']} m/s 未満"
    return "不明", f"10 分の最大瞬間風速 {g:.1f} m/s は検出限界の目安を超えうるが、その 10 分の中の時刻は分からない"


# ================================================================ analyze
def analyze(cfg, run_id):
    out = Path("out/zod/full_weather") / run_id
    if out.exists():
        print(f"{out} がすでにある。上書きしない", file=sys.stderr)
        return 1
    (out / "plots").mkdir(parents=True)
    scr = Path(cfg["screening_run"])
    u = pd.read_csv(scr / "candidates_unique.csv", dtype={"in_camera_sequences": str, "sequence_ids": str})
    c14 = u[u.hypothesis.isin(cfg["candidate_hypotheses"])].copy()
    ctrl = pd.read_csv(scr / "controls.csv")
    calib = pd.read_csv(scr / "offset_calib_by_drive.csv")
    thr = pd.read_csv(scr / "thresholds_by_fold.csv")
    st = station_lists(cfg)
    plan_ = pd.read_csv(OUT_PLAN / "fetch_plan.csv", dtype={"sid": str})
    sel = plan_[plan_.selected]
    st = st.merge(sel[["source", "sid"]], on=["source", "sid"])     # 取得した (または取得済みの) 観測所だけ
    obs = load_obs(cfg, st)
    st = st[[(a, b) in obs for a, b in zip(st.source, st.sid)]]
    excluded_high = st[st.height_m > cfg["stations"]["max_station_height_m"]][["source", "sid", "name", "height_m"]].to_dict("records")
    st = st[st.height_m <= cfg["stations"]["max_station_height_m"]]
    drives = sorted(p.stem for p in (scr / "residuals").glob("*.npz"))
    tracks = drive_tracks(cfg, drives)
    seg = segments(tracks, u, cfg)

    # トルクの向き: 高速でのトルクと v·r (左が正) の相関の符号
    xs, ys = [], []
    for z in tracks.values():
        m = (z["v"] > 15) & z["mask70"].astype(bool) | ((z["v"] > 15) & (np.abs(z["r_deg_s"]) > 1))
        xs.append(z["torque"][m].astype(float) - np.median(z["torque"][m])); ys.append((z["v"] * np.radians(z["r_deg_s"]))[m])
    tq_corr = float(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1])
    tq_sign = 1.0 if tq_corr > 0 else -1.0
    log = dict(torque_vs_lateral_acc_corr=tq_corr, torque_sign_used=tq_sign, excluded_high_stations=excluded_high,
               n_stations_loaded=len(obs), stations=sorted(f"{a}:{b}" for a, b in obs))

    # ---- 1. 候補 14 件
    rows, st_rows = [], []
    for _, ev in c14.iterrows():
        z = tracks[ev.drive_id]
        a_, b_ = ET.event_interval(ev)
        tw = 0.5 * (a_ + b_)                                       # 風を見る時刻・位置・方位 = 閾値超過区間の中央
        i = int(np.argmin(np.abs(z["t"] - tw)))
        H = float(z["heading"][i])
        res, sdf = wind_point(obs, st, float(z["lat"][i]), float(z["lon"][i]), tw, H, cfg)
        sig, sig_note = uncert(res, cfg)
        ph = event_physics(z, ev, cfg, tq_sign)
        req = required_wind(ph["a_steer"], ph["speed_ms"], cfg)
        if ev.kind == "slow" or "持続" in ev.hypothesis:
            verdict, why = judge_sustained(ph, res, sig, req, cfg)
            kind = "持続型"
        else:
            verdict, why = judge_transient(res, cfg)
            kind = "突発型"
        rows.append(dict(event_id=ev.event_id, kind=kind, car=ev.car, drive_id=ev.drive_id, t_onset_utc=ev.t_onset_utc,
                         t_rise_utc=ev.t_rise_utc, rise_censored=ev.rise_censored, t_exceed_start_utc=ev.t_exceed_start_utc, t_exceed_end_utc=ev.t_exceed_end_utc,
                         exceed_duration_s=ev.exceed_duration_s, exceed_start_censored=ev.exceed_start_censored, exceed_end_censored=ev.exceed_end_censored,
                         t_peak_utc=ev.t_peak_utc, wind_time_utc=datetime.fromtimestamp(tw, timezone.utc).isoformat(timespec="seconds"),
                         camera_overlap_s=ev.get("camera_overlap_s"), camera_relation=ev.get("camera_relation"),
                         lat=r3(float(z["lat"][i]), 6), lon=r3(float(z["lon"][i]), 6),
                         speed_kmh=ev.speed_kmh, heading_deg=r3(H, 1), dr_peak_deg_s=ev.dr_peak_deg_s, **res, w_left_sigma=r3(sig, 2), sigma_basis=sig_note,
                         **ph, req_w_left_10m_lo=r3(req[0], 1), req_w_left_10m_hi=r3(req[1], 1), verdict=verdict, verdict_reason=why,
                         in_camera_sequences=ev.in_camera_sequences))
        for _, s in sdf.iterrows():
            st_rows.append(dict(event_id=ev.event_id, **{k: (r3(v, 3) if isinstance(v, float) else v) for k, v in s.items()}))
    cand = pd.DataFrame(rows)
    cand.to_csv(out / "candidates_weather.csv", index=False)
    pd.DataFrame(st_rows).to_csv(out / "candidates_station_values.csv", index=False)

    # ---- 2. 区間ごとの風と車両の指標
    fold = calib.set_index("drive_id").fold.to_dict()
    car = calib.set_index("drive_id").car.to_dict()
    tsl = thr[(thr.speed_kmh == 80) & (thr.residual == "cor") & (thr.kind == "slow")].set_index(["car", "fold"]).threshold_deg_s
    srows = []
    for _, s in seg.iterrows():
        z = tracks[s.drive_id]
        res, _ = wind_point(obs, st, s.lat, s.lon, s.t, s.heading, cfg)
        sl = slow20(z["dr_cor"], z["blk"])[s.idx]
        slu = slow20(z["dr_unc"], z["blk"])[s.idx]
        th = tsl.get((car[s.drive_id], fold[s.drive_id]), np.nan)
        sig, _ = uncert(res, cfg)
        srows.append(dict(drive_id=s.drive_id, car=car[s.drive_id], t_utc=datetime.fromtimestamp(s.t, timezone.utc).isoformat(timespec="seconds"),
                          t=s.t, lat=r3(s.lat, 5), lon=r3(s.lon, 5), heading=r3(s.heading, 1), speed_kmh=r3(s.speed_kmh, 1), screened_s=s.screened_s,
                          has_candidate=s.has_candidate, slow_mean=r3(np.nanmean(sl), 4), slow_mean_unc=r3(np.nanmean(slu), 4), slow_absmax=r3(np.nanmax(np.abs(sl)), 4),
                          slow_thr=r3(th, 4), torque_mean=r3(np.nanmean(z["torque"][s.idx]), 3), w_left_sigma=r3(sig, 2),
                          **{k: res.get(k) for k in ["n_stations", "nearest_km", "nearest_speed", "nearest_w_left", "nearest_gap_min", "idw_speed",
                                                     "idw_w_left", "spread_w_left", "loo_rmse_w_left", "gust10_max", "temporal_std_median"]}))
    sg = pd.DataFrame(srows)
    sg["w_left_est"] = sg.idw_w_left.fillna(sg.nearest_w_left)
    sg["speed_est"] = sg.idw_speed.fillna(sg.nearest_speed)
    cc = cfg["compare"]
    sg["cross_class"] = np.select([sg.w_left_est.abs() >= cc["strong_cross_ms"], sg.w_left_est.abs() < cc["weak_cross_ms"]], ["強い横風", "弱い横風"], "中間")
    sg.loc[sg.w_left_est.isna(), "cross_class"] = "観測なし"
    sg["wind_class"] = np.where(sg.speed_est >= cc["strong_speed_ms"], "強風", np.where(sg.speed_est.notna(), "強風でない", "観測なし"))
    sg.to_csv(out / "segments_weather.csv", index=False)

    # ---- 3. 比べる: 候補 vs 似た非候補区間 (走行の単位)
    rng = np.random.default_rng(cc["seed"])
    comp = []
    for _, r in cand.iterrows():
        pool = sg[(sg.car == r.car) & ~sg.has_candidate & (np.abs(sg.speed_kmh - r.speed_kmh) <= cc["match_speed_tol_kmh"]) & sg.w_left_est.notna()]
        exp = r.expected_w_left_sign
        x = pool.assign(we=exp * pool.w_left_est).groupby("drive_id").we.mean()          # 走行ごとの平均 (同じ走行の区間は 1 つにまとめる)
        we = exp * (r.idw_w_left if pd.notna(r.get("idw_w_left")) else r.nearest_w_left) if pd.notna(r.get("nearest_w_left")) else np.nan
        comp.append(dict(event_id=r.event_id, kind=r.kind, expected_dir_w_left=r3(we, 2), n_match_drives=len(x), n_match_segments=len(pool),
                         match_drive_mean_p50=r3(x.median(), 2) if len(x) else None,
                         percentile_among_drives=r3((x < we).mean() * 100, 0) if len(x) and np.isfinite(we) else None))
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "candidates_vs_matched.csv", index=False)
    # Phase 2 の対照の区間の風
    crow = []
    for _, r in ctrl[ctrl.event_id.isin(c14.event_id)].iterrows():
        z = tracks[r.control_drive_id]
        i = int(np.argmin(np.abs(z["t"] - r.control_t_center)))
        res, _ = wind_point(obs, st, float(z["lat"][i]), float(z["lon"][i]), float(r.control_t_center), float(z["heading"][i]), cfg)
        crow.append(dict(event_id=r.event_id, control_drive_id=r.control_drive_id, control_t_center_utc=r.control_t_center_utc,
                         **{k: res.get(k) for k in ["n_stations", "nearest_km", "nearest_speed", "nearest_w_left", "idw_speed", "idw_w_left", "spread_w_left"]}))
    pd.DataFrame(crow).to_csv(out / "controls_weather.csv", index=False)

    # ---- 4. 強風から逆に: 横風の強さと、持続成分・閾値超過 (走行の単位)
    ok = sg[sg.w_left_est.notna()].copy()
    ok["slow_signed"] = ok.slow_mean * np.sign(ok.w_left_est)                  # 期待: 負 (左からの風 → Δr < 0)
    ok["slow_signed_unc"] = ok.slow_mean_unc * np.sign(ok.w_left_est)
    ok["exceed"] = ok.slow_absmax > ok.slow_thr
    rev = []
    for name, col, a, b in [("横風成分", "cross_class", "強い横風", "弱い横風"), ("風速", "wind_class", "強風", "強風でない")]:
        A, B = ok[ok[col] == a], ok[ok[col] == b]
        da, db = A.groupby("drive_id"), B.groupby("drive_id")

        def stat(Dd):
            g = Dd.agg(ex=("exceed", "mean"), ss=("slow_signed", "mean"), n=("exceed", "size"))
            return g
        ga, gb = stat(da), stat(db)
        boots = []
        for _ in range(cc["bootstrap"]):
            ia = rng.choice(len(ga), len(ga)) if len(ga) else []
            ib = rng.choice(len(gb), len(gb)) if len(gb) else []
            if len(ga) and len(gb):
                boots.append((ga.ex.iloc[ia].mean() - gb.ex.iloc[ib].mean(), ga.ss.iloc[ia].mean() - gb.ss.iloc[ib].mean()))
        boots = np.array(boots) if boots else np.full((1, 2), np.nan)
        rev.append(dict(split=name, group_a=a, group_b=b, a_segments=len(A), a_drives=len(ga), b_segments=len(B), b_drives=len(gb),
                        a_exceed_rate=r3(ga.ex.mean(), 3) if len(ga) else None, b_exceed_rate=r3(gb.ex.mean(), 3) if len(gb) else None,
                        diff_exceed=r3(ga.ex.mean() - gb.ex.mean(), 3) if len(ga) and len(gb) else None,
                        diff_exceed_ci95=f"{np.nanpercentile(boots[:, 0], 2.5):.3f}..{np.nanpercentile(boots[:, 0], 97.5):.3f}",
                        a_slow_signed=r3(ga.ss.mean(), 4) if len(ga) else None, b_slow_signed=r3(gb.ss.mean(), 4) if len(gb) else None,
                        diff_slow_signed=r3(ga.ss.mean() - gb.ss.mean(), 4) if len(ga) and len(gb) else None,
                        diff_slow_signed_ci95=f"{np.nanpercentile(boots[:, 1], 2.5):.4f}..{np.nanpercentile(boots[:, 1], 97.5):.4f}"))
    # 走行の中の関係: 持続成分の平均 ~ 横風成分 (走行ごとに平均を引く)。走行の単位のブートストラップ
    wd = ok.copy()
    wd["xc"] = wd.w_left_est - wd.groupby("drive_id").w_left_est.transform("mean")
    wd["yc"] = wd.slow_mean - wd.groupby("drive_id").slow_mean.transform("mean")
    dl = wd.drive_id.unique()

    def slope(dd):
        x, y = dd.xc.to_numpy(), dd.yc.to_numpy()
        return float(np.sum(x * y) / np.sum(x * x)) if np.sum(x * x) > 0 else np.nan
    sl0 = slope(wd)
    bs = []
    groups = {d: g for d, g in wd.groupby("drive_id")}
    for _ in range(cc["bootstrap"]):
        pick = rng.choice(dl, len(dl))
        bs.append(slope(pd.concat([groups[d] for d in pick])))
    p = cfg["physics"]
    exp_slope = [-np.degrees(p["rho"] * p["area_m2"] * c * h / (2 * p["mass_kg"])) for c in p["cy_per_rad"] for h in cfg["match"]["vehicle_height_factor"]]
    within = dict(n_segments=len(wd), n_drives=len(dl), slope_deg_s_per_ms=r3(sl0, 4),
                  slope_ci95=f"{np.nanpercentile(bs, 2.5):.4f}..{np.nanpercentile(bs, 97.5):.4f}",
                  expected_slope_range=f"{min(exp_slope):.3f}..{max(exp_slope):.3f}",
                  note="期待の傾き = −ρ·A·C_Y·k/(2m) [rad/s per m/s] を deg に (k: 高さの比)。作用点が中立操舵点の近くなら 0 に近い")
    # 感度分析: 候補を含む区間を除く (観測所を候補の近くから選んだための偏りを確かめる)
    nc = ok[~ok.has_candidate]
    for name, col, a, b in [("横風成分 (候補を含む区間を除く)", "cross_class", "強い横風", "弱い横風"), ("風速 (候補を含む区間を除く)", "wind_class", "強風", "強風でない")]:
        ga = nc[nc[col] == a].groupby("drive_id").slow_signed.mean()
        gb = nc[nc[col] == b].groupby("drive_id").slow_signed.mean()
        bt = [ga.iloc[rng.choice(len(ga), len(ga))].mean() - gb.iloc[rng.choice(len(gb), len(gb))].mean() for _ in range(cc["bootstrap"])] if len(ga) and len(gb) else [np.nan]
        rev.append(dict(split=name, group_a=a, group_b=b, a_segments=int((nc[col] == a).sum()), a_drives=len(ga), b_segments=int((nc[col] == b).sum()), b_drives=len(gb),
                        a_slow_signed=r3(ga.mean(), 4), b_slow_signed=r3(gb.mean(), 4), diff_slow_signed=r3(ga.mean() - gb.mean(), 4),
                        diff_slow_signed_ci95=f"{np.nanpercentile(bt, 2.5):.4f}..{np.nanpercentile(bt, 97.5):.4f}"))
    wn = nc.copy()
    wn["xc"] = wn.w_left_est - wn.groupby("drive_id").w_left_est.transform("mean")
    wn["yc"] = wn.slow_mean - wn.groupby("drive_id").slow_mean.transform("mean")
    gn = {d: g for d, g in wn.groupby("drive_id")}
    dn = list(gn)
    bsn = [slope(pd.concat([gn[d] for d in rng.choice(dn, len(dn))])) for _ in range(cc["bootstrap"])]
    within["without_candidate_segments"] = dict(n_segments=len(wn), n_drives=len(dn), slope=r3(slope(wn), 4),
                                                slope_ci95=f"{np.nanpercentile(bsn, 2.5):.4f}..{np.nanpercentile(bsn, 97.5):.4f}")
    # 補正前の残差で同じ比較 (走行ごとのゼロ点の補正が、横風の当て舵を吸収したかを見る)
    for name, col, a, b in [("横風成分 (補正前の残差)", "cross_class", "強い横風", "弱い横風"), ("風速 (補正前の残差)", "wind_class", "強風", "強風でない")]:
        ga = ok[ok[col] == a].groupby("drive_id").slow_signed_unc.mean()
        gb = ok[ok[col] == b].groupby("drive_id").slow_signed_unc.mean()
        bt = [ga.iloc[rng.choice(len(ga), len(ga))].mean() - gb.iloc[rng.choice(len(gb), len(gb))].mean() for _ in range(cc["bootstrap"])] if len(ga) and len(gb) else [np.nan]
        rev.append(dict(split=name, group_a=a, group_b=b, a_segments=int((ok[col] == a).sum()), a_drives=len(ga), b_segments=int((ok[col] == b).sum()), b_drives=len(gb),
                        a_slow_signed=r3(ga.mean(), 4), b_slow_signed=r3(gb.mean(), 4), diff_slow_signed=r3(ga.mean() - gb.mean(), 4),
                        diff_slow_signed_ci95=f"{np.nanpercentile(bt, 2.5):.4f}..{np.nanpercentile(bt, 97.5):.4f}"))
    # 走行ごとのゼロ点の補正量 Δc と、その走行の区間の平均の横風成分 (左からの風なら左に当て舵 → Δc > 0 が期待)
    dc = calib.set_index("drive_id").delta_c_deg
    dd_ = ok.groupby("drive_id").agg(w_left_mean=("w_left_est", "mean"), n=("t", "size"), heading_sd=("heading", "std"))
    dd_["delta_c_deg"] = dd_.index.map(dc)
    dd_ = dd_.dropna(subset=["delta_c_deg"])
    cor0 = float(np.corrcoef(dd_.w_left_mean, dd_.delta_c_deg)[0, 1]) if len(dd_) > 3 else np.nan
    bc = []
    for _ in range(cc["bootstrap"]):
        ii = rng.choice(len(dd_), len(dd_))
        x = dd_.iloc[ii]
        bc.append(np.corrcoef(x.w_left_mean, x.delta_c_deg)[0, 1] if x.w_left_mean.std() > 0 else np.nan)
    within["delta_c_vs_drive_mean_w_left"] = dict(n_drives=len(dd_), corr=r3(cor0, 3), corr_ci95=f"{np.nanpercentile(bc, 2.5):.3f}..{np.nanpercentile(bc, 97.5):.3f}",
                                                  note="Δc は OXTS の区間の外 (±30 s) から求めたので、比べる区間の時刻とは同じでない")
    dd_.to_csv(out / "drive_delta_c_vs_wind.csv")
    pd.DataFrame(rev).to_csv(out / "reverse_search.csv", index=False)
    # 強い横風の区間の一覧 (逆方向の探索)
    strong = ok[ok.cross_class == "強い横風"].sort_values("w_left_est", key=np.abs, ascending=False)
    strong.to_csv(out / "strong_cross_segments.csv", index=False)

    plots(cand, ok, out / "plots")
    summ = dict(log=log, n_candidates=len(cand), verdicts=cand.groupby(["kind", "verdict"]).size().reset_index(name="n").to_dict("records"),
                segments=dict(total=len(sg), with_weather=int(sg.w_left_est.notna().sum()), drives=int(sg.drive_id.nunique()),
                              drives_with_weather=int(sg[sg.w_left_est.notna()].drive_id.nunique()),
                              by_cross_class=sg.cross_class.value_counts().to_dict(), by_wind_class=sg.wind_class.value_counts().to_dict(),
                              strong_cross_drives=int(sg[sg.cross_class == "強い横風"].drive_id.nunique())),
                reverse=rev, within_drive=within)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=str))

    def sha(pth):
        return hashlib.sha256(Path(pth).read_bytes()).hexdigest()
    here = Path(__file__).parent
    man = dict(run_id=run_id, created_local=datetime.now().isoformat(timespec="seconds"),
               scripts={str(x): sha(x) for x in [Path(__file__), here / "full_screening.py", here / "weather_match.py", here / "weather_common.py"]},
               config=cfg, config_sha256=sha("configs/zod/full_weather.yaml"),
               inputs={str(x): sha(x) for x in [scr / "candidates_unique.csv", scr / "controls.csv", OUT_PLAN / "fetch_plan.csv"]},
               observation_files=sorted(str(f) for (s_, i_) in obs for f in (find_files(cfg, s_, i_) if s_ == "DWD" else [p_ for _, p_ in find_files(cfg, s_, i_)])))
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({k: summ[k] for k in ["verdicts", "segments", "reverse", "within_drive"]}, ensure_ascii=False, indent=1, default=str))
    return 0


def plots(cand, ok, pdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    a = ax[0]
    a.scatter(ok.w_left_est, ok.slow_mean, s=8, alpha=0.35, color="#7f8c8d", label=f"比べる区間 ({len(ok)}、{ok.drive_id.nunique()} 走行)")
    for kind, mk, col in [("持続型", "o", "#c0392b"), ("突発型", "^", "#1f4e9c")]:
        c = cand[cand.kind == kind]
        w = c.idw_w_left.fillna(c.nearest_w_left) if "idw_w_left" in c else c.nearest_w_left
        a.errorbar(w, c.mean_dr_deg_s, xerr=c.w_left_sigma, fmt=mk, color=col, ms=6, lw=0.8, label=f"候補 {kind}")
    a.axhline(0, color="#999", lw=0.6); a.axvline(0, color="#999", lw=0.6)
    a.set_xlabel("推定の横風成分 W_left [m/s] (10 m、左から正)"); a.set_ylabel("ヨー残差の持続成分 [deg/s]")
    a.set_title("横風が左から (右) なら、期待はヨー残差が負 (正) = 第 2・第 4 象限", fontsize=9); a.legend(fontsize=7); a.grid(alpha=0.3)
    b = ax[1]
    for cls, col in [("弱い横風", "#2e86c1"), ("中間", "#aab7b8"), ("強い横風", "#c0392b")]:
        x = ok[ok.cross_class == cls].groupby("drive_id").slow_signed.mean()
        if len(x):
            b.hist(x, bins=25, histtype="step", lw=1.5, color=col, label=f"{cls} ({len(x)} 走行)", density=True)
    b.set_xlabel("走行ごとの平均: 持続成分 × sign(W_left) [deg/s] (期待: 強い横風で負)"); b.legend(fontsize=8); b.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(pdir / "wind_vs_residual.png", dpi=110); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "fetch", "analyze"])
    ap.add_argument("--config", default="configs/zod/full_weather.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--screening-run", default=None, help="Phase 2 の実行 (設定の screening_run を上書き。manifest に残る)")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.screening_run:
        cfg["screening_run"] = a.screening_run
    if a.mode == "plan":
        return plan(cfg)
    if a.mode == "fetch":
        return fetch(cfg)
    return analyze(cfg, a.run_id or datetime.now().strftime("%Y%m%dT%H%M%S"))


if __name__ == "__main__":
    sys.exit(main())
