"""TartanDrive 2.0 の rosbag から、イベントの前後のカメラ画像だけを範囲指定で取り出す (確認用。特徴抽出には使わない)。

KITTI 形式の画像 (image_left_color.tar、1 走行約 5 GB) は中のファイルの並びが時刻順でないので、rosbag を使う。
1. data_extraction.log の話題ごとの時刻の範囲から bag の始まりを、*_topics.yaml の IMU の数と周期から分割ファイルごとの長さを出し、
   イベントの時刻を含む分割ファイルを決める (境目に近ければ両方)
2. その分割ファイルの末尾の索引 (約 2 MB) を読み、画像の話題を含み、目的の時刻を含むチャンク (約 4 MB) だけを取る
取った量は manifest.csv に書き、設定の上限を超えそうなら止める。
取ったチャンクの中の racepak・IMU・画像のメッセージの、ヘッダの時刻と bag の記録時刻を timing.csv に書く (時刻のずれの原因を調べるため)。
--events は event_id・run・t_abs の列があれば、PoC の events.csv でなくてもよい。

    uv run python scripts/offroad/td2_event_frames.py --events out/offroad/tartandrive2/<id>/events.csv \
        --ids T000,T001 --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
import urllib.parse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yaml
from PIL import Image

import rosbag_range as RB


def bag_start(log: Path) -> float:
    ts = [float(m.group(1)) for m in re.finditer(r"^\s+/\S+\s+\t?\s*\d+\s+\t?\s*([\d.]+) - [\d.]+", log.read_text(), re.M)]
    return min(ts)


def split_files(bag_dir: Path) -> list[tuple[str, float]]:
    """(分割ファイルの名前, 長さ s) を番号の順に。"""
    out = []
    for f in sorted(bag_dir.glob("*_topics.yaml"), key=lambda p: int(re.search(r"_(\d+)_topics", p.name).group(1))):
        d = yaml.safe_load(f.read_text()) or {}
        imu = d.get("/novatel/imu/data") or d.get("/multisense/imu/imu_data")
        ok = imu and imu.get("frequency") and imu.get("message_count")
        out.append((f.name.replace("_topics.yaml", ".bag"), imu["message_count"] / imu["frequency"] if ok else 0.0))
    return out


def decode_image(md: bytes) -> tuple[float, np.ndarray]:
    stamp, p = RB.decode_header(md)
    h, w = struct.unpack_from("<II", md, p)
    p += 8
    n = struct.unpack_from("<I", md, p)[0]
    enc = md[p + 4:p + 4 + n].decode()
    p += 4 + n + 1                       # encoding, is_bigendian
    step = struct.unpack_from("<I", md, p)[0]
    p += 4
    n = struct.unpack_from("<I", md, p)[0]
    a = np.frombuffer(md, np.uint8, n, p + 4).reshape(h, step)[:, :w * 3].reshape(h, w, 3)
    if enc.lower().startswith("bgr"):
        a = a[:, :, ::-1]
    return stamp, a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offroad/td2_vibration_poc.yaml")
    ap.add_argument("--source", default="configs/offroad/tartandrive2.yaml")
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))["frames"]
    src = yaml.safe_load(open(a.source))
    root = Path(src["data_dir"])
    RB.CACHE_DIR = root / "bag_cache"      # 取ったチャンクと索引を残す (取り直しを避ける)
    out = Path("out/offroad/tartandrive2") / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    E = pd.read_csv(a.events).set_index("event_id")
    used, man = 0, []
    idx_cache: dict[str, RB.BagIndex] = {}

    def fetch(n_bytes: int, what: str):
        nonlocal used
        used += n_bytes
        man.append(dict(what=what, bytes=n_bytes))
        if used > cfg["max_download_mb"] * 1e6:
            raise SystemExit(f"取得量が上限 {cfg['max_download_mb']} MB を超えた ({used / 1e6:.0f} MB)。止める")

    timing = []          # 取ったチャンクの中の、ヘッダの時刻と記録時刻 (時刻のずれの原因を調べるため)

    def index_of(run: str, nm: str) -> tuple[str, RB.BagIndex]:
        key = f"bags/{run}/{nm}"
        if key not in idx_cache:
            url = f"{src['source']['endpoint']}/{src['source']['bucket']}/{urllib.parse.quote(key)}"
            was = RB.cached(url, 0, 4095 + 13)
            bi = RB.read_index(url)
            fetch(0 if was else bi.index_bytes, f"index {key}" + (" (キャッシュ)" if was else ""))
            idx_cache[key] = bi
        return key, idx_cache[key]

    for eid in a.ids.split(","):
        e = E.loc[eid]
        t0 = bag_start(root / "kitti/all_topics" / e.run / "data_extraction.log")
        files = split_files(root / "bags" / e.run)
        edges = np.r_[0, np.cumsum([d for _, d in files])] + t0
        targets = [float(e.t_abs) + o for o in cfg["offsets_s"]]
        got, seen = {}, {}
        for t in targets:
            # 分割ファイルの境目は *_topics.yaml の数と周期からの推定で、10 s 近くずれることがある (T003 で確認)。
            # 推定のファイルの索引を読み、目的の時刻がその実際の範囲の外なら隣のファイルへ移る
            k = min(max(int(np.searchsorted(edges, t, side="right")) - 1, 0), len(files) - 1)
            for _hop in range(len(files)):
                key, bi = index_of(e.run, files[k][0])
                lo_t, hi_t = min(c["start"] for c in bi.chunks), max(c["end"] for c in bi.chunks)
                if t < lo_t and k > 0:
                    k -= 1
                elif t > hi_t and k < len(files) - 1:
                    k += 1
                else:
                    break
            ids = set(bi.topic_ids(cfg["topic"]))
            if not ids:
                man.append(dict(what=f"{eid}: {key} に {cfg['topic']} が無い", bytes=0))
                continue
            typ = bi.conns[next(iter(ids))]["type"]
            if typ != "sensor_msgs/Image":
                raise SystemExit(f"{cfg['topic']} の型が {typ} (sensor_msgs/Image を想定)")
            tids = {i: c["topic"] for i, c in bi.conns.items() if c["topic"] in cfg["timing_topics"]}
            cs = [c for c in bi.chunks if c["start"] - 0.2 <= t <= c["end"] + 0.2 and ids & set(c["counts"])]
            best = None
            # 近いチャンクから 1 つずつ取り、許容の範囲の画像が見つかったら止める (取る量を抑える)
            for c in sorted(cs, key=lambda c: abs((c["start"] + c["end"]) / 2 - t))[:3]:
                if (key, c["pos"]) not in seen:
                    nxt = [x["pos"] for x in bi.chunks if x["pos"] > c["pos"]]
                    end = (min(nxt) if nxt else bi.size) - 1
                    was = RB.cached(bi.url, c["pos"], end)
                    fetch(0 if was else end + 1 - c["pos"], f"chunk {key}@{c['pos']}" + (" (キャッシュ)" if was else ""))
                    msgs = RB.read_chunk_messages(bi, c, ids | set(tids))
                    seen[(key, c["pos"])] = [decode_image(md) for cid, _rt, md in msgs if cid in ids]
                    for cid, rt, md in msgs:
                        if cid in tids:
                            timing.append(dict(item=eid, run=e.run, file=files[k][0], chunk=c["pos"], topic=tids[cid],
                                               record_t=rt, header_t=RB.decode_header(md)[0]))
                for st, img in seen[(key, c["pos"])]:
                    if best is None or abs(st - t) < abs(best[0] - t):
                        best = (st, img)
                if best is not None and abs(best[0] - t) <= cfg["max_frame_offset_s"]:
                    break
            if best is not None and abs(best[0] - t) <= cfg["max_frame_offset_s"]:
                got[t] = best
        for o, t in zip(cfg["offsets_s"], targets):
            if t in got:
                st, img = got[t]
                Image.fromarray(img).save(out / f"{eid}_{o:+.1f}s.jpg", quality=88)
                man.append(dict(what=f"frame {eid} offset {o:+.1f} s (画像の時刻 - 目的 {st - t:+.3f} s)", bytes=0))
        print(eid, e.run, "frames", len(got), f"累計 {used / 1e6:.0f} MB", flush=True)
        # 途中で止まっても残るよう、1 件ごとに書く
        pd.DataFrame(timing).to_csv(out / "timing.csv", index=False)
        pd.DataFrame(man).to_csv(out / "manifest.csv", index=False)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["what", "bytes"])
        w.writeheader()
        w.writerows(man)
    print(f"取得 {used / 1e6:.1f} MB (索引を含む) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
