#!/usr/bin/env python3
"""ZOD mini の走行軌跡の近くにある気象観測所 (SMHI・DWD) を調べる。

  uv run --extra viz --with "zod==0.8.0" --with "dataclass-wizard==0.39.1" \\
      python scripts/zod/weather_stations.py --root raw_data/zod --version mini

ネットワークに出るのは、SMHI Open Data と DWD CDC の観測所の一覧 (小さいテキスト・JSON) だけ。
取得したものは data/zod_weather/ に置き、data/zod_weather/fetch_log.csv に取得元・時刻・sha256・利用条件を残す。

出力 (out/zod/weather/<version>/):
  stations.csv   観測所 × 走行: 位置・標高・観測期間・2022 年の走行日に動いていたか・走行軌跡までの最短距離と中央値など
  tracks.csv     走行ごとの軌跡の要約 (国・日時・範囲)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weather_common as W  # noqa: E402


def smhi_stations(cfg) -> pd.DataFrame:
    sm = cfg["smhi"]
    rows = []
    for p, (name, res) in sm["parameters"].items():
        f = W.fetch(f"{sm['base']}/parameter/{p}.json", W.DATA / "smhi" / f"parameter_{p}.json", "SMHI", sm["license"], 10)
        if f is None:
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        for s in j["station"]:
            rows.append(dict(source="SMHI", product=f"{p}:{name}", resolution=res, station_id=str(s["key"]), name=s["name"],
                             lat=s["latitude"], lon=s["longitude"], height_m=s.get("height"), active_now=s.get("active"),
                             from_utc=datetime.fromtimestamp(s["from"] / 1000, timezone.utc), to_utc=datetime.fromtimestamp(s["to"] / 1000, timezone.utc),
                             owner=s.get("owner", "")))
    return pd.DataFrame(rows)


def dwd_stations(cfg) -> pd.DataFrame:
    dw = cfg["dwd"]
    rows = []
    for prod, (folder, fname) in dw["products"].items():
        f = W.fetch(f"{dw['base']}/{folder}/{fname}", W.DATA / "dwd" / fname, "DWD", dw["license"], 10)
        if f is None:
            continue
        for line in f.read_bytes().decode("latin-1").splitlines()[2:]:
            m = re.match(r"\s*(\d+)\s+(\d{8})\s+(\d{8})\s+(-?\d+)\s+([\d.]+)\s+([\d.]+)\s+(.*)$", line)
            if not m:
                continue
            rest = re.split(r"\s{2,}", m.group(7).strip())
            rows.append(dict(source="DWD", product=prod, resolution=folder, station_id=m.group(1).zfill(5), name=rest[0],
                             lat=float(m.group(5)), lon=float(m.group(6)), height_m=float(m.group(4)),
                             active_now=None, from_utc=datetime.strptime(m.group(2), "%Y%m%d").replace(tzinfo=timezone.utc),
                             to_utc=datetime.strptime(m.group(3), "%Y%m%d").replace(tzinfo=timezone.utc),
                             owner=rest[1] if len(rest) > 1 else ""))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("raw_data/zod"))
    ap.add_argument("--version", default="mini")
    ap.add_argument("--config", type=Path, default=Path("configs/zod/weather.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out = Path(f"out/zod/weather/{args.version}")
    out.mkdir(parents=True, exist_ok=True)

    tracks = W.load_tracks(args.root, args.version)
    st = pd.concat([smhi_stations(cfg), dwd_stations(cfg)], ignore_index=True)
    rows, trows = [], []
    for sid, d in tracks.items():
        u = d["track"]
        ox = d["ox"]
        country = d["meta"]["country_code"]
        t0, t1 = datetime.fromtimestamp(u.t.iloc[0], timezone.utc), datetime.fromtimestamp(u.t.iloc[-1], timezone.utc)
        lat_c, lon_c = float(ox.posLat.mean()), float(ox.posLon.mean())
        trows.append(dict(id=sid, country=country, start_utc=t0.isoformat(timespec="seconds"), end_utc=t1.isoformat(timespec="seconds"),
                          n_points=len(u), lat_range=f"{u.lat.min():.4f}..{u.lat.max():.4f}", lon_range=f"{u.lon.min():.4f}..{u.lon.max():.4f}",
                          oxts_lat=round(lat_c, 5), oxts_lon=round(lon_c, 5), oxts_start_utc=datetime.fromtimestamp(ox.t.iloc[0], timezone.utc).isoformat(timespec="seconds")))
        src = "SMHI" if country == "SE" else "DWD" if country == "DE" else None
        cand = st[st.source == src]
        # 粗く絞る (軌跡の範囲 ± 0.5 度)
        cand = cand[(cand.lat > u.lat.min() - 0.5) & (cand.lat < u.lat.max() + 0.5) & (cand.lon > u.lon.min() - 0.8) & (cand.lon < u.lon.max() + 0.8)]
        for _, s in cand.iterrows():
            dk = W.dist_km(u.lat, u.lon, s.lat, s.lon)
            i = int(np.argmin(dk))
            rows.append(dict(id=sid, source=s.source, product=s["product"], resolution=s.resolution, station_id=s.station_id, name=s["name"],
                             lat=s.lat, lon=s.lon, height_m=s.height_m, owner=s.owner,
                             period=f"{s.from_utc:%Y-%m-%d}..{s.to_utc:%Y-%m-%d}", active_now=s.active_now,
                             covers_drive_day=bool(s.from_utc <= t0 and s.to_utc >= t1),
                             min_dist_km=round(float(dk.min()), 2), median_dist_km=round(float(np.median(dk)), 2),
                             max_dist_km=round(float(dk.max()), 2),
                             dist_at_oxts_km=round(float(W.dist_km(lat_c, lon_c, s.lat, s.lon)), 2),
                             closest_utc=datetime.fromtimestamp(u.t.iloc[i], timezone.utc).isoformat(timespec="seconds"),
                             frac_track_within_10km=round(float((dk <= 10).mean()), 3),
                             frac_track_within_5km=round(float((dk <= 5).mean()), 3)))
    df = pd.DataFrame(rows)
    df = df[df.min_dist_km <= cfg["search_radius_km"]].sort_values(["id", "min_dist_km", "product"])
    df.to_csv(out / "stations.csv", index=False)
    pd.DataFrame(trows).to_csv(out / "tracks.csv", index=False)

    # 距離の区分ごとの数 (2022 年の走行日に動いていた観測所、製品を問わず観測所の数)
    summ = []
    for sid, g in df.groupby("id"):
        act = g[g.covers_drive_day]
        for b in cfg["distance_bins_km"]:
            summ.append(dict(id=sid, within_km=b, stations_active_on_drive_day=act[act.min_dist_km <= b].station_id.nunique(),
                             stations_listed_any_period=g[g.min_dist_km <= b].station_id.nunique(),
                             stations_active_at_oxts=act[act.dist_at_oxts_km <= b].station_id.nunique()))
    pd.DataFrame(summ).to_csv(out / "station_counts.csv", index=False)
    print(pd.DataFrame(summ).to_string(index=False))
    show = df[df.covers_drive_day].drop_duplicates(["id", "station_id", "product"])
    print(show[["id", "source", "product", "station_id", "name", "height_m", "period", "min_dist_km", "median_dist_km", "dist_at_oxts_km"]].to_string(index=False))
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
