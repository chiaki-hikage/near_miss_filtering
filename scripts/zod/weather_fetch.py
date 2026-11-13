#!/usr/bin/env python3
"""ZOD mini の走行の近くの観測所 (weather_stations.py の結果) の風の観測を、公開元から取る。

  uv run --extra viz --with "zod==0.8.0" --with "dataclass-wizard==0.39.1" \\
      python scripts/zod/weather_fetch.py --version mini

- SMHI: パラメータ 4 (風速 10 分平均、1 時間ごと)、3 (風向 10 分平均)、21 (最大瞬間風速、1 時間ごと)、
  25 (3 時間の 10 分平均の最大) の corrected-archive (検査済みの過去分の CSV)。1 分値 (47・48) は
  過去分が提供されていない (latest-hour / latest-day だけ) ので取らない
- DWD: 10 分値の風 (FF_10, DD_10)、10 分値の極値 (FX_10 など)、1 時間値の風。観測所ごと・期間ごとの zip
- 1 ファイル 30 MB (configs/zod/weather.yaml の fetch_max_mb_per_file) を超えるものは取らない
- 取ったものは data/zod_weather/、記録は data/zod_weather/fetch_log.csv。失敗は記録し、繰り返さない

選ぶ観測所: 走行日に動いていて、軌跡から 20 km 以内のもの (近い順に fetch_max_stations_per_drive まで) と、
OXTS の区間に最も近い観測所 (20 km を超えても)。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weather_common as W  # noqa: E402

EXTRA_NEAREST_TO_OXTS = {"000000": [("SMHI", "73480")], "000002": [("DWD", "07367")]}   # weather_stations の調べで、OXTS の区間に最も近い観測所


def smhi_fetch(station: str, cfg) -> list[Path]:
    sm = cfg["smhi"]
    got = []
    for p in (4, 3, 21, 25):
        meta = W.fetch(f"{sm['base']}/parameter/{p}/station/{station}.json", W.DATA / "smhi" / f"p{p}_s{station}.json", "SMHI", sm["license"], 1)
        if meta is None:
            continue
        periods = [x["key"] for x in json.loads(meta.read_text(encoding="utf-8")).get("period", [])]
        if "corrected-archive" not in periods:
            print(f"SMHI {station} p{p}: corrected-archive が無い ({periods})")
            continue
        f = W.fetch(f"{sm['base']}/parameter/{p}/station/{station}/period/corrected-archive/data.csv",
                    W.DATA / "smhi" / f"p{p}_s{station}_corrected-archive.csv", "SMHI", sm["license"], cfg["fetch_max_mb_per_file"], 120)
        if f is not None:
            got.append(f)
    return got


def dwd_list(url: str) -> list[str]:
    req = urllib.request.Request(url, headers={"User-Agent": W.UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode("latin-1")
    return re.findall(r'href="([^"]+\.zip)"', html)


def dwd_fetch(station: str, day: str, cfg) -> list[Path]:
    dw = cfg["dwd"]
    got = []
    for prod, (folder, _) in dw["products"].items():
        url = f"{dw['base']}/{folder}/"
        cache = W.DATA / "dwd" / f"index_{folder.replace('/', '_')}.txt"
        if cache.exists():
            names = cache.read_text().split()
        else:
            try:
                names = dwd_list(url)
            except Exception as ex:
                W.log_fetch(url, cache, 0, "", f"failed: {type(ex).__name__}", "DWD", dw["license"], "directory listing")
                print("一覧を取れなかった:", url)
                continue
            cache.write_text("\n".join(names))
            W.log_fetch(url, cache, len("\n".join(names)), "", "ok", "DWD", dw["license"], "directory listing (file names only)")
        for n in names:
            m = re.search(rf"_{station}_(\d{{8}})_(\d{{8}})_hist\.zip$", n)
            if m and m.group(1) <= day <= m.group(2):
                f = W.fetch(url + n, W.DATA / "dwd" / n, "DWD", dw["license"], cfg["fetch_max_mb_per_file"], 180)
                if f is not None:
                    got.append(f)
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="mini")
    ap.add_argument("--config", type=Path, default=Path("configs/zod/weather.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    st = pd.read_csv(f"out/zod/weather/{args.version}/stations.csv", dtype={"station_id": str, "id": str})
    tr = pd.read_csv(f"out/zod/weather/{args.version}/tracks.csv", dtype={"id": str}).set_index("id")
    rows = []
    for sid, g in st[st.covers_drive_day].groupby("id"):
        day = tr.loc[sid, "start_utc"][:10].replace("-", "")
        pick = (g.sort_values("min_dist_km").drop_duplicates("station_id")[["source", "station_id"]]
                .head(cfg["fetch_max_stations_per_drive"]).itertuples(index=False, name=None))
        pick = list(dict.fromkeys(list(pick) + EXTRA_NEAREST_TO_OXTS.get(sid, [])))
        for src, stn in pick:
            files = smhi_fetch(stn, cfg) if src == "SMHI" else dwd_fetch(stn.zfill(5), day, cfg)
            rows.append(dict(id=sid, source=src, station_id=stn, files=";".join(p.name for p in files)))
            print(sid, src, stn, [p.name for p in files])
    out = Path(f"out/zod/weather/{args.version}")
    pd.DataFrame(rows).to_csv(out / "fetched_files.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
