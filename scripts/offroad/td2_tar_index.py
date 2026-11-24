"""KITTI 形式の画像の tar (image_left_color.tar、1 走行 約 5 GB) の、中のファイルの位置の一覧を作る。

tar の中のファイルは時刻の順に並んでいないので、先頭から 512 byte のヘッダを順に範囲指定で読み、
(名前, 本体の位置, 大きさ) を data/offroad/tartandrive2/tar_index/<run>__<tar>.csv に書く。
本体は読まないので、取る量はヘッダの分 (ファイル数 × 512 byte) だけ。1 つの接続を使い回す。

    uv run python scripts/offroad/td2_tar_index.py --runs <走行> [<走行> ...]
"""

from __future__ import annotations

import argparse
import csv
import http.client
import time
import urllib.parse
from pathlib import Path

import yaml


def walk(host: str, path: str, max_members: int | None = None) -> tuple[list[tuple[str, int, int]], int]:
    conn = http.client.HTTPSConnection(host, timeout=60)
    pos, out, got = 0, [], 0
    while True:
        for attempt in range(3):
            try:
                conn.request("GET", path, headers={"Range": f"bytes={pos}-{pos + 511}"})
                r = conn.getresponse()
                h = r.read()
                break
            except (http.client.HTTPException, OSError):
                conn.close()
                time.sleep(2)
                conn = http.client.HTTPSConnection(host, timeout=60)
        else:
            raise RuntimeError(f"ヘッダを読めない (位置 {pos})")
        got += len(h)
        if r.status != 206 or len(h) < 512 or h == b"\0" * 512:
            break
        name = h[:100].rstrip(b"\0").decode(errors="replace")
        size = int(h[124:136].rstrip(b"\0 ").decode() or "0", 8)
        typ = h[156:157]
        if typ in (b"L", b"x", b"g"):
            raise NotImplementedError(f"長い名前・拡張ヘッダ ({typ!r}) には対応していない")
        out.append((name, pos + 512, size))
        pos += 512 + ((size + 511) // 512) * 512
        if max_members and len(out) >= max_members:
            break
    conn.close()
    return out, got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="configs/offroad/tartandrive2.yaml")
    ap.add_argument("--tar", default="image_left_color.tar")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--max-members", type=int, default=None)
    a = ap.parse_args()
    src = yaml.safe_load(open(a.source))
    u = urllib.parse.urlparse(src["source"]["endpoint"])
    od = Path(src["data_dir"]) / "tar_index"
    od.mkdir(parents=True, exist_ok=True)
    for run in a.runs:
        dst = od / f"{run}__{a.tar}.csv"
        if dst.exists() and not a.max_members:
            print(run, "すでにある")
            continue
        path = f"{u.path}/{src['source']['bucket']}/{urllib.parse.quote(f'kitti/all_topics/{run}/{a.tar}')}"
        t0 = time.time()
        mem, got = walk(u.netloc, path, a.max_members)
        with open(dst.with_suffix(".part"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "offset", "size"])
            w.writerows(mem)
        dst.with_suffix(".part").rename(dst)
        print(f"{run} {len(mem)} ファイル、ヘッダ {got / 1e6:.2f} MB、{time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
