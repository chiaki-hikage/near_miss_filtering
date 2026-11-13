"""ZOD の気象照合 PoC の共通部分: 公開気象データの取得 (記録つき) と、走行軌跡の読み込み。

取得先は SMHI Open Data と DWD Climate Data Center だけ。ZOD の Dropbox には触れない。
取得は 1 回だけ試し、失敗しても繰り返さない (失敗は記録して止める)。
"""

from __future__ import annotations

import csv
import hashlib
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signal_survey as S  # noqa: E402
import drive_validation as D  # noqa: E402

DATA = Path("data/zod_weather")
LOG = DATA / "fetch_log.csv"
UA = "near-miss-filtering-research/0.1 (python urllib)"


def log_fetch(url, path, nbytes, sha, status, source, license_, note=""):
    DATA.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["fetched_at_utc", "source", "url", "path", "bytes", "sha256", "status", "license", "note"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), source, url, str(path), nbytes, sha, status, license_, note])


def fetch(url: str, dest: Path, source: str, license_: str, max_mb: float = 30.0, timeout: int = 60) -> Path | None:
    """1 回だけ取る。既にあれば取らない。大きさの上限を超えるなら取らない。"""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            n = r.headers.get("Content-Length")
            if n is not None and int(n) > max_mb * 1e6:
                log_fetch(url, dest, int(n), "", "skipped_too_large", source, license_)
                print(f"大きすぎるので取らない ({int(n) / 1e6:.1f} MB): {url}")
                return None
            data = r.read(int(max_mb * 1e6) + 1)
    except Exception as ex:  # 取得の失敗は記録して返す (繰り返さない)
        log_fetch(url, dest, 0, "", f"failed: {type(ex).__name__}: {ex}"[:200], source, license_)
        print(f"取得できなかった: {url} ({type(ex).__name__})")
        return None
    if len(data) > max_mb * 1e6:
        log_fetch(url, dest, len(data), "", "skipped_too_large", source, license_)
        return None
    dest.write_bytes(data)
    log_fetch(url, dest, len(data), hashlib.sha256(data).hexdigest(), "ok", source, license_)
    return dest


def load_tracks(root: Path, version: str = "mini") -> dict:
    """シーケンスごとに、走行全体の衛星測位 (約 1 Hz、車両データ) と OXTS の 50 s を返す。"""
    from zod import ZodSequences
    ds = ZodSequences(str(root), version, mp=False)
    out = {}
    for sid in sorted(ds.get_all_ids()):
        info = ds[sid].info
        vd = S.load_vehicle(Path(info.vehicle_data_path))
        ox = S.load_oxts(Path(info.oxts_path))
        u = D.gnss_updates(vd["satellite"])
        meta = pd.Series(__import__("json").loads(Path(info.metadata_path).read_text()))
        out[sid] = dict(info=info, vd=vd, ox=ox, track=u, meta=meta)
    return out


def dist_km(lat1, lon1, lat2, lon2):
    return D.dist_m(np.asarray(lat1, float), np.asarray(lon1, float), np.asarray(lat2, float), np.asarray(lon2, float)) / 1000.0
