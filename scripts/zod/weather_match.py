#!/usr/bin/env python3
"""ZOD mini の 2 走行を、近くの観測所の風の観測 (weather_fetch.py で取得済み) と照らし合わせる。ネットワークは使わない。

  uv run --offline --extra viz --with "zod==0.8.0" --with "dataclass-wizard==0.39.1" \\
      python scripts/zod/weather_match.py --root raw_data/zod --version mini

観測は車両の位置での風の真値ではない。観測所の値を、走行の時刻と進行方位に当てはめた「目安」として扱う。
横風成分: W_left = −W·sin(D − H) (左から吹くと正。docs/crosswind_civic_feasibility.md (5.2) と同じ)。
D は風が吹いてくる方位、H は車の進む方位 (北から時計回り)。

出力 (out/zod/weather/<version>/):
  observations.csv       走行の前後 3 時間の観測 (観測所・量・時刻・値・品質)。時間の分解能を列で区別する
  availability.csv       観測所 × 量: 走行の前後 3 時間と 2022 年全体の欠測率、品質の分布、観測の間隔
  match_blocks.csv       走行を 10 分ずつに分け、観測所ごとの風・横風成分と車両の指標を並べたもの
  match_oxts.csv         OXTS の 50 s: 観測所ごとの風・横風成分、ヨー残差、横滑り角とその品質、操舵
  station_pairs.csv      観測所の組ごとの、走行の前後 3 時間の風速・風向・突風の一致 (空間の代表性の目安)
  block_spread.csv       10 分の区間ごとの、観測所の間の横風成分の広がり
  time_near_station.csv  走行時間のうち、照合した観測所から 1/3/5/10/20 km 以内にいた割合
  plots/                 風と横風成分と車両の指標の時系列
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weather_common as W  # noqa: E402
import signal_survey as S  # noqa: E402
import drive_validation as D  # noqa: E402

# 量の名前、単位、時間の分解能 (配布元の説明から)
SMHI_PARAM = {4: ("speed", "10 分平均 (毎正時の 1 値)"), 3: ("dir", "10 分平均 (毎正時の 1 値)"),
              21: ("gust", "最大瞬間風速 (1 時間に 1 値。対象の時間幅は配布元の説明 'max, 1 gång/tim' だけ)"),
              25: ("max10min_3h", "直前 3 時間の 10 分平均の最大 (1 時間に 1 値)")}
DWD_COLS = {"FF_10": ("speed", "直前 10 分の平均"), "DD_10": ("dir", "直前 10 分の平均"),
            "FX_10": ("gust", "直前 10 分の最大瞬間風速"), "DX_10": ("gust_dir", "最大瞬間風速の風向"),
            "FMX_10": ("max_1min_mean", "直前 10 分の 1 分平均の最大 (DWD の FMX_10)"),
            "F": ("speed_hourly", "1 時間値 (毎正時の 10 分平均)"), "D": ("dir_hourly", "1 時間値")}


# ---------------------------------------------------------------- 観測の読み込み
def read_smhi(path: Path, p: int) -> pd.DataFrame:
    txt = path.read_text(encoding="utf-8-sig").splitlines()
    i = next(k for k, l in enumerate(txt) if l.startswith("Datum;Tid (UTC)"))
    rows = []
    for l in txt[i + 1:]:
        f = l.split(";")
        if len(f) < 4 or not f[0]:
            continue
        rows.append((f"{f[0]} {f[1]}", float(f[2]), f[3]))
    d = pd.DataFrame(rows, columns=["t", "value", "quality"])
    d["t"] = pd.to_datetime(d.t, utc=True)
    height = txt[1].split(";")[3]
    d["quantity"], d["resolution"] = SMHI_PARAM[p]
    d["sensor_height_m"] = float(height)
    return d


def read_dwd(path: Path) -> pd.DataFrame:
    z = zipfile.ZipFile(path)
    name = next(n for n in z.namelist() if n.startswith("produkt"))
    d = pd.read_csv(io.BytesIO(z.read(name)), sep=";", skipinitialspace=True, dtype=str)
    d.columns = [c.strip() for c in d.columns]
    fmt = "%Y%m%d%H%M" if len(d.MESS_DATUM.iloc[0].strip()) == 12 else "%Y%m%d%H"
    d["t"] = pd.to_datetime(d.MESS_DATUM.str.strip(), format=fmt, utc=True)
    qcol = "QN" if "QN" in d else "QN_3"
    out = []
    for c, (q, res) in DWD_COLS.items():
        if c in d:
            v = pd.to_numeric(d[c], errors="coerce")
            out.append(pd.DataFrame(dict(t=d.t, value=v.where(v > -998), quality=d[qcol].str.strip(), quantity=q, resolution=res)))
    return pd.concat(out)


def load_observations(fetched: pd.DataFrame, tracks: dict, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs, avail = [], []
    h = cfg["match_window_h"]
    for _, r in fetched.iterrows():
        sid, src, stn = r["id"], r.source, str(r.station_id)
        u = tracks[sid]["track"]
        t0 = pd.Timestamp(u.t.iloc[0], unit="s", tz="UTC") - pd.Timedelta(hours=h)
        t1 = pd.Timestamp(u.t.iloc[-1], unit="s", tz="UTC") + pd.Timedelta(hours=h)
        for fn in str(r.files).split(";"):
            if not fn or fn == "nan":
                continue
            if src == "SMHI":
                p = int(re.match(r"p(\d+)_", fn).group(1))
                d = read_smhi(W.DATA / "smhi" / fn, p)
                kind = "SMHI 1 時間ごと"
            else:
                d = read_dwd(W.DATA / "dwd" / fn)
                d["sensor_height_m"] = 10.0                       # DWD のメタデータ (Geberhoehe) で 2022 年は 10 m
                kind = "DWD 1 時間値" if fn.startswith("stunden") else "DWD 10 分値"
            for q, g in d.groupby("quantity"):
                y22 = g[(g.t >= "2022-01-01") & (g.t < "2023-01-01")]
                w = g[(g.t >= t0) & (g.t <= t1)]
                step = y22.t.diff().median()
                expected = int(pd.Timedelta(days=365) / step) if pd.notna(step) else np.nan
                ew = int((t1 - t0) / step) + 1 if pd.notna(step) else np.nan
                avail.append(dict(id=sid, source=src, station_id=stn, product=kind, quantity=q, resolution=g.resolution.iloc[0],
                                  step_min=step.total_seconds() / 60 if pd.notna(step) else np.nan,
                                  window_expected=ew, window_present=int(w.value.notna().sum()),
                                  window_missing_frac=round(1 - w.value.notna().sum() / ew, 4) if ew else np.nan,
                                  year2022_expected=expected, year2022_present=int(y22.value.notna().sum()),
                                  year2022_missing_frac=round(1 - y22.value.notna().sum() / expected, 4) if expected else np.nan,
                                  quality_codes_window=",".join(f"{k}:{v}" for k, v in w.quality.value_counts().items()),
                                  sensor_height_m=g.sensor_height_m.iloc[0] if "sensor_height_m" in g else np.nan))
                ww = w.copy()
                ww["id"], ww["source"], ww["station_id"], ww["product"] = sid, src, stn, kind
                obs.append(ww)
    return pd.concat(obs, ignore_index=True), pd.DataFrame(avail)


# ---------------------------------------------------------------- 照合
def nearest_obs(obs_st: pd.DataFrame, quantity: str, t: float, max_gap_min: float):
    o = obs_st[(obs_st.quantity == quantity) & obs_st.value.notna()]
    if o.empty:
        return np.nan, np.nan
    ts = o.t.astype("int64").to_numpy() / 1e9
    i = int(np.argmin(np.abs(ts - t)))
    gap = (ts[i] - t) / 60
    return (float(o.value.iloc[i]), gap) if abs(gap) <= max_gap_min else (np.nan, gap)


def wind_at(obs_st, t, prod):
    """観測所の、時刻 t に最も近い風速・風向・突風。10 分値なら 10 分以内、1 時間値なら 60 分以内のもの。"""
    gap = 10 if "10 分" in prod else 60
    spd, g1 = nearest_obs(obs_st, "speed", t, gap)
    if np.isnan(spd):
        spd, g1 = nearest_obs(obs_st, "speed_hourly", t, 60)
    dr, _ = nearest_obs(obs_st, "dir", t, gap)
    if np.isnan(dr):
        dr, _ = nearest_obs(obs_st, "dir_hourly", t, 60)
    gu, gg = nearest_obs(obs_st, "gust", t, gap)
    return spd, dr, gu, g1, gg


def w_left(W_, Dd, H):
    return -W_ * np.sin(np.radians(Dd - H))


def blocks(sid, tr, obs, stations, cfg, model_drive, c_deg) -> list[dict]:
    """走行全体を 10 分ずつに分けて、観測所ごとの横風成分と、車両の指標を並べる。"""
    cw = cfg["crosswind"]
    u = tr["track"]
    g = D.drive_residual(S.drive_grid(tr["vd"], CFG_SIG), model_drive, CFG_SIG)
    cls = D.drive_classes(g, model_drive, CFG_SIG)
    st_hi = (cls == "直進 (舵角)") & (g.v > cw["straight_hi_min_speed"])
    phi = np.cumsum(np.interp(g.t, tr["vd"][S.EGO].t, tr["vd"][S.EGO][S.K_ROLL].to_numpy(float))) / CFG_SIG["grid_hz"]
    edges = np.arange(u.t.iloc[0], u.t.iloc[-1] + 1, cw["block_min"] * 60)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        uu = u[(u.t >= a) & (u.t < b) & (u.speed > cw["min_speed_for_heading"])]
        gm = (g.t >= a) & (g.t < b)
        sm = gm & st_hi
        base = dict(id=sid, block_start_utc=datetime.fromtimestamp(a, timezone.utc).isoformat(timespec="minutes"),
                    minutes_moving=round(len(uu) / 60, 1), speed_kmh_median=round(float(g.v[gm].median() * 3.6), 1),
                    straight_hi_min=round(sm.sum() / CFG_SIG["grid_hz"] / 60, 1),
                    steer_minus_c_deg_median=round(float(np.degrees(g.steer[sm].median()) - c_deg), 3) if sm.any() else np.nan,
                    torque_mean_nm=round(float(g.torque[sm].mean()), 3) if sm.any() else np.nan,
                    ay_resid_lowfreq_median=round(float((g.d_ay - g.d_ay_band)[sm].median()), 4) if sm.any() else np.nan,
                    ay_resid_band_std=round(float(g.d_ay_band[sm].std()), 4) if sm.sum() > 100 else np.nan,
                    torque_std_nm=round(float(g.torque[sm].std()), 3) if sm.sum() > 100 else np.nan)
        if len(uu) < 30:
            continue
        H = uu.heading.to_numpy()
        tm = float(uu.t.median())
        for _, s in stations.iterrows():
            o = obs[(obs.id == sid) & (obs.station_id == s.station_id)]
            if o.empty:
                continue
            spd, dr, gu, g1, gg = wind_at(o, tm, o["product"].iloc[0])
            dist = W.dist_km(uu.lat, uu.lon, s.lat, s.lon)
            wl = w_left(spd, dr, H)
            rows.append(dict(base, station_id=s.station_id, station=s["name"], product=o["product"].iloc[0],
                             dist_km_median=round(float(np.median(dist)), 1), obs_time_gap_min=round(g1, 1) if np.isfinite(g1) else np.nan,
                             wind_speed=spd, wind_dir=dr, gust=gu,
                             w_left_mean=round(float(np.mean(wl)), 2) if np.isfinite(spd) and np.isfinite(dr) else np.nan,
                             w_cross_abs_mean=round(float(np.mean(np.abs(wl))), 2) if np.isfinite(spd) and np.isfinite(dr) else np.nan,
                             gust_cross_abs_mean=round(float(np.mean(np.abs(w_left(gu, dr, H)))), 2) if np.isfinite(gu) and np.isfinite(dr) else np.nan,
                             heading_median=round(float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(H)))))) % 360, 1)))
    return rows


def oxts_match(sid, tr, obs, stations, cfg) -> list[dict]:
    ox, vd = tr["ox"], tr["vd"]
    mw, yaw = S.fit_yaw_window(sid, ox, vd, CFG_SIG)
    tm = float(ox.t.median())
    H = ox.heading.to_numpy()
    beta = ox.slipAngle.to_numpy()
    sig_b = np.degrees(np.hypot(ox.stdDevVelNorth, ox.stdDevVelEast).to_numpy() / np.maximum(ox.speed.to_numpy(), 1))
    c = vd[S.CTRL]
    m = (c.t >= ox.t.iloc[0]) & (c.t <= ox.t.iloc[-1])
    base = dict(id=sid, oxts_start_utc=datetime.fromtimestamp(ox.t.iloc[0], timezone.utc).isoformat(timespec="seconds"),
                speed_kmh=f"{ox.speed.min() * 3.6:.0f}..{ox.speed.max() * 3.6:.0f}",
                heading_deg_range=f"{np.min(H):.0f}..{np.max(H):.0f}",
                yaw_resid_cv_std=yaw["stats"]["oxts_yaw_residual_cv_std_deg_s"],
                beta_mean_deg=round(float(beta.mean()), 3), beta_std_deg=round(float(beta.std()), 3),
                beta_uncertainty_deg_median=round(float(np.median(sig_b)), 3),
                beta_usable=bool(np.median(sig_b) < 0.2),
                steer_minus_c_deg_mean=round(float(np.degrees(c[S.K_STEER][m].mean()) - mw["c_offset_deg"]), 3),
                torque_mean_nm=round(float(c[S.K_TORQUE][m].mean()), 3), torque_std_nm=round(float(c[S.K_TORQUE][m].std()), 3),
                oxts_pos_std_m=round(float(np.median(np.hypot(ox.stdDevPosNorth, ox.stdDevPosEast))), 3))
    rows = []
    for _, s in stations.iterrows():
        o = obs[(obs.id == sid) & (obs.station_id == s.station_id)]
        if o.empty:
            continue
        spd, dr, gu, g1, gg = wind_at(o, tm, o["product"].iloc[0])
        wl = w_left(spd, dr, H)
        rows.append(dict(base, station_id=s.station_id, station=s["name"], product=o["product"].iloc[0],
                         dist_km=round(float(W.dist_km(ox.posLat.mean(), ox.posLon.mean(), s.lat, s.lon)), 1),
                         obs_time_gap_min=round(g1, 1) if np.isfinite(g1) else np.nan, wind_speed=spd, wind_dir=dr, gust=gu,
                         w_left_mean=round(float(np.mean(wl)), 2) if np.isfinite(spd) and np.isfinite(dr) else np.nan,
                         w_left_range=f"{np.min(wl):.1f}..{np.max(wl):.1f}" if np.isfinite(spd) and np.isfinite(dr) else ""))
    return rows


def plot(sid, obs, mb, tr, plots: Path):
    import matplotlib.pyplot as plt
    u = tr["track"]
    t0 = pd.Timestamp(u.t.iloc[0], unit="s", tz="UTC")
    t1 = pd.Timestamp(u.t.iloc[-1], unit="s", tz="UTC")
    ox = tr["ox"]
    o = obs[obs.id == sid]
    fig, ax = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    for (stn, prod), g in o.groupby(["station_id", "product"]):
        lab = f"{stn} ({prod})"
        for q, a, sty in [("speed", 0, "-o"), ("speed_hourly", 0, "--"), ("gust", 1, "-o"), ("dir", 2, "o"), ("dir_hourly", 2, "x")]:
            x = g[g.quantity == q]
            if not x.empty:
                ax[a].plot(x.t, x.value, sty, ms=3, lw=0.8, label=lab)
    m = mb[mb.id == sid].copy()
    m["t"] = pd.to_datetime(m.block_start_utc, utc=True) + pd.Timedelta(minutes=5)
    for stn, g in m.groupby("station_id"):
        ax[3].plot(g.t, g.w_left_mean, "-o", ms=3, label=f"{stn}: W_left")
    m1 = m.drop_duplicates("block_start_utc")
    ax4 = ax[4]
    ax4.plot(m1.t, m1.steer_minus_c_deg_median, "-o", ms=3, label="舵角 − c [deg] (高速・直進の中央値)")
    ax4.plot(m1.t, m1.torque_mean_nm, "-s", ms=3, label="操舵トルクの平均 [N·m]")
    ax4.plot(m1.t, m1.ay_resid_lowfreq_median * 10, "-^", ms=3, label="横加速度の残差 (低域) × 10 [m/s²]")
    ax[0].set_ylabel("風速 [m/s]"); ax[1].set_ylabel("最大瞬間風速 [m/s]"); ax[2].set_ylabel("風向 [deg]\n(吹いてくる方位)")
    ax[3].set_ylabel("横風成分 W_left [m/s]\n(左から正)"); ax[3].axhline(0, color="gray", lw=0.5)
    ax4.set_ylabel("車両の指標"); ax4.axhline(0, color="gray", lw=0.5)
    for a in ax:
        a.axvspan(t0, t1, color="tab:blue", alpha=0.06)
        a.axvspan(pd.Timestamp(ox.t.iloc[0], unit="s", tz="UTC"), pd.Timestamp(ox.t.iloc[-1], unit="s", tz="UTC"), color="tab:red", alpha=0.5)
        a.legend(fontsize=7, ncol=2)
    ax[-1].set_xlabel("UTC (青: 走行、赤: OXTS の 50 s)")
    fig.suptitle(f"{sid}: 近くの観測所の風と、車両の指標 (観測は車の位置の真値ではない)")
    fig.tight_layout(); fig.savefig(plots / f"{sid}_weather.png", dpi=100); plt.close(fig)


def spatial_consistency(obs, mb, tracks, stations) -> tuple[pd.DataFrame, pd.DataFrame]:
    """観測所の間の一致 (空間の代表性の目安) と、走行時間のうち観測所の近くにいた割合。"""
    rows = []
    for sid, o in obs.groupby("id"):
        for q in ("speed", "dir", "gust"):
            x = o[(o.quantity == q) & (o["product"].str.contains("10 分|1 時間ごと"))].pivot_table(index="t", columns="station_id", values="value")
            st = list(x.columns)
            for i in range(len(st)):
                for j in range(i + 1, len(st)):
                    a, b = x[st[i]], x[st[j]]
                    m = a.notna() & b.notna()
                    if m.sum() < 5:
                        continue
                    if q == "dir":
                        d = ((a[m] - b[m] + 180) % 360 - 180).abs()
                        rows.append(dict(id=sid, quantity=q, pair=f"{st[i]}-{st[j]}", n=int(m.sum()), corr=np.nan,
                                         mean_abs_diff=round(float(d.mean()), 1)))
                    else:
                        rows.append(dict(id=sid, quantity=q, pair=f"{st[i]}-{st[j]}", n=int(m.sum()), corr=round(float(np.corrcoef(a[m], b[m])[0, 1]), 3),
                                         mean_abs_diff=round(float((a[m] - b[m]).abs().mean()), 2)))
    pairs = pd.DataFrame(rows)
    # 10 分の区間ごとの、観測所の間の横風成分の広がり
    spread = mb.groupby(["id", "block_start_utc"]).agg(n_stations=("station_id", "nunique"), w_left_min=("w_left_mean", "min"),
                                                       w_left_max=("w_left_mean", "max"), speed_min=("wind_speed", "min"),
                                                       speed_max=("wind_speed", "max"), nearest_km=("dist_km_median", "min")).reset_index()
    spread["w_left_range"] = spread.w_left_max - spread.w_left_min
    near = []
    for sid, tr in tracks.items():
        u = tr["track"]
        st = stations[stations.id == sid]
        dmin = np.min(np.vstack([W.dist_km(u.lat, u.lon, r.lat, r.lon) for r in st.itertuples()]), axis=0)
        near.append(dict(id=sid, **{f"frac_time_within_{k}km": round(float((dmin <= k).mean()), 3) for k in (1, 3, 5, 10, 20)},
                         median_dist_to_nearest_km=round(float(np.median(dmin)), 1)))
    return pairs, spread, pd.DataFrame(near)


CFG_SIG = None


def main() -> int:
    global CFG_SIG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("raw_data/zod"))
    ap.add_argument("--version", default="mini")
    ap.add_argument("--config", type=Path, default=Path("configs/zod/weather.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    CFG_SIG = yaml.safe_load(Path("configs/zod/signal_survey.yaml").read_text(encoding="utf-8"))
    CFG_SIG.update(yaml.safe_load(Path("configs/zod/drive_validation.yaml").read_text(encoding="utf-8")))
    out = Path(f"out/zod/weather/{args.version}")
    (out / "plots").mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]

    tracks = W.load_tracks(args.root, args.version)
    fetched = pd.read_csv(out / "fetched_files.csv", dtype={"station_id": str, "id": str})
    stations = pd.read_csv(out / "stations.csv", dtype={"station_id": str, "id": str}).drop_duplicates(["id", "station_id"])
    # OXTS の区間に最も近い観測所は 20 km の外なので stations.csv に無い。一覧から位置を足す
    extra = []
    for f in (W.DATA / "smhi").glob("p4_s*_corrected-archive.csv"):
        txt = f.read_text(encoding="utf-8-sig").splitlines()
        stn = txt[1].split(";")[1]
        k = next(i for i, l in enumerate(txt) if l.startswith("Tidsperiod (fr.o.m)"))
        per = []
        for l in txt[k + 1:]:
            if not l.strip():
                break
            per.append(l)
        last = per[-1].split(";")                                 # 最後の期間 (2022 年を含む) の位置
        extra.append(dict(id="000000", station_id=stn, name=txt[1].split(";")[0], lat=float(last[3]), lon=float(last[4])))
    for f in (W.DATA / "dwd").glob("meta_*/Metadaten_Geographie_*.txt"):
        ln = [l for l in f.read_bytes().decode("latin-1").splitlines() if re.match(r"\s*\d+;", l)][-1].split(";")
        extra.append(dict(id="000002", station_id=ln[0].strip().zfill(5), name=ln[6].strip(), lat=float(ln[2]), lon=float(ln[3])))
    for f in (W.DATA / "dwd").glob("meta_*/Metadaten_Geraete_Windgeschwindigkeit_*.txt"):
        if "Mittel" in f.name:
            continue
        ln = [l for l in f.read_bytes().decode("latin-1").splitlines() if re.match(r"\s*\d+;", l)][-1].split(";")
        extra.append(dict(id="000002", station_id=ln[0].strip().zfill(5), name=ln[1].strip(), lat=float(ln[3]), lon=float(ln[2])))
    stations = pd.concat([stations[["id", "station_id", "name", "lat", "lon"]], pd.DataFrame(extra)]).drop_duplicates(["id", "station_id"])

    obs, avail = load_observations(fetched, tracks, cfg)
    obs.to_csv(out / "observations.csv", index=False)
    avail.to_csv(out / "availability.csv", index=False)

    mb, mo = [], []
    for sid, tr in tracks.items():
        g = S.drive_grid(tr["vd"], CFG_SIG)
        model_drive = S.fit_model([g], CFG_SIG, sid)
        mw, _ = S.fit_yaw_window(sid, tr["ox"], tr["vd"], CFG_SIG)
        st = stations[stations.id == sid]
        mb += blocks(sid, tr, obs, st, cfg, model_drive, mw["c_offset_deg"])
        mo += oxts_match(sid, tr, obs, st, cfg)
    mb, mo = pd.DataFrame(mb), pd.DataFrame(mo)
    mb.to_csv(out / "match_blocks.csv", index=False)
    mo.to_csv(out / "match_oxts.csv", index=False)
    for sid, tr in tracks.items():
        plot(sid, obs, mb, tr, out / "plots")
    pairs, spread, near = spatial_consistency(obs, mb, tracks, stations)
    pairs.to_csv(out / "station_pairs.csv", index=False)
    spread.to_csv(out / "block_spread.csv", index=False)
    near.to_csv(out / "time_near_station.csv", index=False)
    print(pairs.to_string(index=False)); print(spread.to_string(index=False)); print(near.to_string(index=False))
    print(avail.to_string(index=False))
    print(mo.to_string(index=False))
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
