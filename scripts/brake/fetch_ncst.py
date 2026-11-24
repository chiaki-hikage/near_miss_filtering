"""NCST の重量車のブレーキの実走行データ (Zenodo 5839097、DOI 10.6086/D14H5C、CC0) を取得する。

公式の配布元 (Zenodo の API) から、全ファイル (31、約 17 MB) を落とし、md5 を確かめて記録を残す。
すでに同じ md5 のファイルがあれば落とさない。

    uv run python scripts/brake/fetch_ncst.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import urllib.request
from pathlib import Path

RECORD = "https://zenodo.org/api/records/5839097"
UA = {"User-Agent": "near-miss-filtering-research/1.0"}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=Path("data/brake/ncst_hdv_brake"))
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)
    rec = json.load(urllib.request.urlopen(urllib.request.Request(RECORD, headers=UA)))
    (a.dest / "_zenodo_record.json").write_text(json.dumps(rec, indent=1))
    rows = []
    for f in rec["files"]:
        p = a.dest / f["key"]
        want = f["checksum"].split(":", 1)[1]
        if not (p.exists() and md5(p) == want):
            with urllib.request.urlopen(urllib.request.Request(f["links"]["self"], headers=UA)) as r:
                p.write_bytes(r.read())
        got = md5(p)
        rows.append(dict(file=f["key"], bytes=p.stat().st_size, md5=got, md5_ok=got == want, url=f["links"]["self"],
                         fetched_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")))
        print(f["key"], p.stat().st_size, "ok" if got == want else "MD5 MISMATCH", flush=True)
    with open(a.dest / "_fetch_log.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lic = rec["metadata"].get("license", {}).get("id")
    print(f"{len(rows)} files, {sum(r['bytes'] for r in rows) / 1e6:.2f} MB, license={lic}, all md5 ok={all(r['md5_ok'] for r in rows)}")
    return 0 if all(r["md5_ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
