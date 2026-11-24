"""NCST の高速道路の走行の、GPS が切れた後の経路 (I-15 南行き) の形と標高を公開地図から取得する。

- 道の形: OpenStreetMap (ODbL) を Overpass API で。I-15 の本線 (motorway) の way を形つきで取る
- 標高: USGS 3DEP の Elevation Point Query Service (公有)。経路に沿って step_m ごとに 1 点ずつ問い合わせる
送るのは公開データの経路の座標だけ。取得したものは data_dir に置き、取得の記録を _fetch_log.csv に残す。
すでにファイルがあれば取得しない (再現のため)。

    uv run python scripts/brake/ncst_route_fetch.py
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yaml

CFG = Path("configs/brake/ncst_route.yaml")
R_EARTH = 6371008.8


def log_fetch(d: Path, url: str, path: Path):
    new = not (d / "_fetch_log.csv").exists()
    with open(d / "_fetch_log.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "url", "file", "bytes", "sha256"])
        b = path.read_bytes()
        w.writerow([dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), url, path.name, len(b), hashlib.sha256(b).hexdigest()])


def http(url: str, ua: str, data: bytes | None = None, timeout=120) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl, dp = np.radians(lon2 - lon1), p2 - p1
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def last_fix(cfg) -> dict:
    """HEM の最後の有効な GPS の点。時刻は ncst_scene_illustration と同じ「logger の始まりからの分」
    (logger は報告書の 9 s 進みを直し、HEM は hem_lag_s ずらす)。"""
    import ncst_thermal as NT

    B = NT.read_logger(cfg["logger_file"], "road")
    t0 = B.t.iat[0]
    D = pd.read_csv(NT.DATA / cfg["hem_file"], skiprows=2, encoding="latin1", low_memory=False)
    t = np.array([dt.datetime.strptime(x.strip(), "%m/%d/%y %H:%M:%S").timestamp() for x in D.Time])
    m = (t + cfg["hem_lag_s"] - t0) / 60
    lat, lon = pd.to_numeric(D.Latitude, errors="coerce"), pd.to_numeric(D.Longitude, errors="coerce")
    ok = (lat.abs() > 1).to_numpy()
    i = int(np.where(ok)[0][-1])
    hd = pd.to_numeric(D.Heading, errors="coerce")
    return dict(lat=float(lat.iat[i]), lon=float(lon.iat[i]), min=float(m[i]), heading=float(hd.iat[i]))


def chain_route(ways: list, fix: dict, O: dict) -> list[tuple[float, float]]:
    """南行きの本線の way を、GPS の最後の点から順につなぐ。"""
    W = [w for w in ways if w["type"] == "way" and "geometry" in w]
    best, best_d = None, 1e18
    for w in W:
        g = w["geometry"]
        for k in range(len(g) - 1):
            d = haversine(fix["lat"], fix["lon"], g[k]["lat"], g[k]["lon"])
            b = bearing(g[k]["lat"], g[k]["lon"], g[k + 1]["lat"], g[k + 1]["lon"])
            if d < best_d and abs((b - O["start_bearing_deg"] + 180) % 360 - 180) < O["bearing_tol_deg"]:
                best, best_d, best_k = w, d, k
    pts = [(p["lat"], p["lon"]) for p in best["geometry"][best_k:]]
    used = {best["id"]}
    cur = best
    while pts[-1][0] > O["stop_lat"]:
        end = cur["nodes"][-1]
        nxt = [w for w in W if w["id"] not in used and w["nodes"][0] == end]
        if not nxt:  # 端点の node が違うときは距離で
            nxt = [w for w in W if w["id"] not in used and
                   haversine(pts[-1][0], pts[-1][1], w["geometry"][0]["lat"], w["geometry"][0]["lon"]) < O["min_continue_m"]]
        if not nxt:
            break
        # 進む向きが近いものを選ぶ
        b0 = bearing(*pts[-2], *pts[-1])
        nxt.sort(key=lambda w: abs((bearing(w["geometry"][0]["lat"], w["geometry"][0]["lon"], w["geometry"][min(1, len(w["geometry"]) - 1)]["lat"],
                                            w["geometry"][min(1, len(w["geometry"]) - 1)]["lon"]) - b0 + 180) % 360 - 180))
        cur = nxt[0]
        used.add(cur["id"])
        pts += [(p["lat"], p["lon"]) for p in cur["geometry"][1:]]
    return pts, best_d


def resample(pts, step):
    lat = np.array([p[0] for p in pts])
    lon = np.array([p[1] for p in pts])
    seg = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    s = np.r_[0, np.cumsum(seg)]
    keep = np.r_[True, seg > 0.01]
    s, lat, lon = s[keep], lat[keep], lon[keep]
    sg = np.arange(0, s[-1], step)
    return pd.DataFrame(dict(s_m=sg, lat=np.interp(sg, s, lat), lon=np.interp(sg, s, lon)))


def main() -> int:
    cfg = yaml.safe_load(CFG.read_text())
    d = Path(cfg["data_dir"])
    d.mkdir(parents=True, exist_ok=True)
    ua = cfg["user_agent"]
    fix = last_fix(cfg)
    print("最後の GPS の点:", fix)
    O = cfg["osm"]
    osm_path = d / "osm_i15_motorway.json"
    if not osm_path.exists():
        s, w, n, e = O["bbox"]
        q = f'[out:json][timeout:120];way["highway"="motorway"]["ref"~"{O["ref_regex"]}"]({s},{w},{n},{e});out body geom;'
        osm_path.write_bytes(http(O["url"], ua, data=urllib.parse.urlencode({"data": q}).encode()))
        log_fetch(d, O["url"] + " (query: " + q + ")", osm_path)
    ways = json.loads(osm_path.read_text())["elements"]
    pts, d0 = chain_route(ways, fix, O)
    R = resample(pts, cfg["elevation"]["step_m"])
    print(f"経路 {R.s_m.iloc[-1] / 1000:.1f} km、始点と GPS の最後の点の距離 {d0:.0f} m、点 {len(R)}")
    el_path = d / "usgs_epqs_elevation.csv"
    done = pd.read_csv(el_path) if el_path.exists() else pd.DataFrame(columns=["lat", "lon", "elev_m", "resolution_m"])
    have = {(round(a, 6), round(b, 6)) for a, b in zip(done.lat, done.lon)}
    new = []
    E = cfg["elevation"]
    for _, r in R.iterrows():
        key = (round(r.lat, 6), round(r.lon, 6))
        if key in have:
            continue
        url = E["url"] + "?" + urllib.parse.urlencode(dict(x=key[1], y=key[0], units="Meters", wkid=4326, includeDate="false"))
        for attempt in range(4):
            try:
                j = json.loads(http(url, ua, timeout=60))
                new.append(dict(lat=key[0], lon=key[1], elev_m=float(j["value"]), resolution_m=j.get("resolution")))
                break
            except Exception as ex:  # noqa: BLE001
                time.sleep(2 + 2 * attempt)
                if attempt == 3:
                    print("取得できない点", key, ex)
        time.sleep(E["sleep_s"])
        if len(new) % 50 == 0 and new:
            print(len(new), "点", flush=True)
    if new:
        done = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
        done.to_csv(el_path, index=False)
        log_fetch(d, E["url"] + f" ({len(new)} 点を 1 点ずつ)", el_path)
    R.to_csv(d / "route_points.csv", index=False)
    (d / "last_fix.json").write_text(json.dumps(dict(fix, route_start_offset_m=d0), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
