"""Udacity CH2 の ROS1 bag から、話題の一覧・型の定義・数値の信号を取り出す (元の bag は書き換えない)。

出力 (out/brake_fade/<run-id>/<bag 名>/):
- topics.csv: 話題・型・件数・最初と最後の時刻・平均の頻度
- msgdefs/<型>.txt: bag に入っている型の定義そのもの (単位のコメントを確かめるため)
- signals/<話題>.csv.gz: t_header (header.stamp、無ければ空)・t_bag と、数値・真偽の項目 (1 段だけ平らにする)
- camera_times.csv: 前方カメラの各こまの時刻 (画像は書き出さない)

    uv run --with rosbags python scripts/brake/udacity_bag_extract.py --bag data/udacity_ch2/extracted/HMB_2.bag --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yaml
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

OUT = Path("out/brake_fade")
CFG = Path("configs/brake/udacity_hmb.yaml")


def split_msgdef(text: str, name: str) -> dict:
    """ROS1 の連結された定義 (=== と MSG: で区切る) を型ごとに分ける。"""
    parts = {}
    cur_name, buf = name, []
    for line in text.splitlines():
        if line.startswith("====="):
            parts[cur_name] = "\n".join(buf)
            cur_name, buf = None, []
            continue
        if cur_name is None and line.startswith("MSG: "):
            cur_name = line[5:].strip()
            continue
        buf.append(line)
    if cur_name:
        parts[cur_name] = "\n".join(buf)
    return parts


def register(reader, store):
    add, raw = {}, {}
    for c in reader.connections:
        for n, body in split_msgdef(c.msgdef.data if hasattr(c.msgdef, "data") else c.msgdef, c.msgtype).items():
            n = n if "/msg/" in n else n.replace("/", "/msg/")
            raw[n] = body
            try:
                add.update(get_types_from_msg(body, n))
            except Exception:  # noqa: BLE001  (Header など既知の型は登録済み)
                pass
    store.register(add)
    return raw


def flatten(msg, prefix="") -> dict:
    out = {}
    for k in getattr(msg, "__dataclass_fields__", {}):
        if k == "header":
            continue
        v = getattr(msg, k)
        name = f"{prefix}{k}"
        if isinstance(v, (bool, np.bool_)):
            out[name] = int(v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[name] = float(v)
        elif isinstance(v, np.ndarray) and v.size <= 9 and v.dtype.kind in "fiu":
            for i, x in enumerate(v.ravel()):
                out[f"{name}_{i}"] = float(x)
        elif hasattr(v, "__dataclass_fields__") and not prefix:
            out.update(flatten(v, prefix=f"{name}."))
    return out


def stamp(msg):
    h = getattr(msg, "header", None)
    return h.stamp.sec + h.stamp.nanosec * 1e-9 if h is not None else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text())
    out = OUT / a.run_id / a.bag.stem
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    (out / "msgdefs").mkdir(parents=True, exist_ok=True)
    (out / "signals").mkdir(exist_ok=True)
    store = get_typestore(Stores.ROS1_NOETIC)
    want = set(cfg["topics"]) | {cfg["camera_topic"]}
    rows = {t: [] for t in want}
    with Reader(a.bag) as r:
        raw = register(r, store)
        for n, body in raw.items():
            if n.startswith(("dbw_mkz_msgs", "sensor_msgs/msg/Imu", "sensor_msgs/msg/NavSatFix", "sensor_msgs/msg/FluidPressure")):
                (out / "msgdefs" / (n.replace("/", "__") + ".txt")).write_text(body)
        T = []
        for c in r.connections:
            T.append(dict(topic=c.topic, type=c.msgtype, count=c.msgcount))
        conns = [c for c in r.connections if c.topic in want]
        for c, ts, data in r.messages(connections=conns):
            tb = ts * 1e-9
            if c.topic == cfg["camera_topic"]:
                msg = store.deserialize_ros1(data, c.msgtype)
                rows[c.topic].append(dict(t_header=stamp(msg), t_bag=tb, nbytes=len(msg.data)))
                continue
            msg = store.deserialize_ros1(data, c.msgtype)
            d = dict(t_header=stamp(msg), t_bag=tb)
            if c.msgtype == "std_msgs/msg/Float64":
                d["data"] = float(msg.data)
            else:
                d.update(flatten(msg))
            rows[c.topic].append(d)
        start, end = r.start_time * 1e-9, r.end_time * 1e-9
    TT = pd.DataFrame(T).groupby(["topic", "type"], as_index=False)["count"].sum()
    TT["rate_hz"] = TT["count"] / (end - start)
    TT.sort_values("topic").to_csv(out / "topics.csv", index=False)
    summ = dict(bag=a.bag.as_posix(), start=start, end=end, duration_s=end - start, topics={})
    for t, rr in rows.items():
        D = pd.DataFrame(rr)
        if t == cfg["camera_topic"]:
            D.to_csv(out / "camera_times.csv", index=False)
        else:
            D.to_csv(out / "signals" / (t.strip("/").replace("/", "__") + ".csv.gz"), index=False)
        summ["topics"][t] = dict(n=len(D), columns=[c for c in D.columns if c not in ("t_header", "t_bag")])
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    meta = OUT / a.run_id / "meta.json"
    if not meta.exists():  # 同じ run に別の bag を足すときは最初の meta を残す
        meta.write_text(json.dumps(dict(run_id=a.run_id, config=CFG.as_posix(), cfg=cfg), ensure_ascii=False, indent=1))
    print(TT.sort_values("topic").to_string(index=False))
    print(json.dumps({k: v["n"] for k, v in summ["topics"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
