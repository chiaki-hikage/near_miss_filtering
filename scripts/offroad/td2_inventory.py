"""TartanDrive 2.0 の走行ごとの長さ・KITTI 形式の切り出しの範囲・信号の品質をまとめる。

- rosbag の長さ: bags/<run>/*_topics.yaml (話題ごとの message_count / frequency) から。bag 本体は読まない
- KITTI の切り出しの範囲: kitti/all_topics/<run>/data_extraction.log の「starting time / ending time」と、
  話題ごとの時刻の範囲 (切り出しの始まり = 話題の始まりの最大、終わり = 終わりの最小、のどれが決めたか)
- 信号の品質: 周期・時刻の抜け・逆行・同じ値の行の続き・外れ値・ストロークの値の範囲

    uv run python scripts/offroad/td2_inventory.py --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SIGNALS = {  # フォルダ: (ファイル, 列名)
    "novatel_imu": ("imu.npy", ["wx", "wy", "wz", "ax", "ay", "az"]),
    "multisense_imu": ("imu.npy", ["wx", "wy", "wz", "ax", "ay", "az"]),
    "shock_pos": ("shock.npy", ["FL", "FR", "RL", "RR"]),
    "wheel_rpm": ("encoder.npy", ["FL", "FR", "RL", "RR"]),
    "controls": ("control.npy", ["throttle", "brake"]),
    "steering_angle": ("float.npy", ["steer"]),
    "gps_odom": ("odometry.npy", None),
}


def bag_durations(bag_dir: Path) -> dict:
    """bag の分割ファイルごとの *_topics.yaml から、IMU の話題の数と周期で長さを出す。"""
    tot, n_files, topics = 0.0, 0, set()
    for f in sorted(bag_dir.glob("*_topics.yaml")):
        d = yaml.safe_load(f.read_text()) or {}
        topics |= set(d)
        imu = d.get("/novatel/imu/data") or d.get("/multisense/imu/imu_data")
        if imu and imu.get("frequency"):
            tot += imu["message_count"] / imu["frequency"]
        n_files += 1
    return dict(bag_files=n_files, bag_s=tot, has_shock=("/shock_pos" in topics), has_novatel=("/novatel/imu/data" in topics))


LOG_TOPIC = re.compile(r"^\s+(/\S+)\s+\t?\s*(\d+)\s+\t?\s*([\d.]+) - ([\d.]+)")


def parse_log(p: Path) -> dict:
    txt = p.read_text(errors="replace")
    topics = {}
    for line in txt.splitlines():
        m = LOG_TOPIC.match(line)
        if m and "starting time" not in line:
            topics.setdefault(m.group(1), (int(m.group(2)), float(m.group(3)), float(m.group(4))))
    m = re.search(r"starting time ([\d.]+), ending time ([\d.]+), duration ([\d.]+)", txt)
    out = dict(kitti_start=float(m.group(1)), kitti_end=float(m.group(2)), kitti_s=float(m.group(3))) if m else {}
    if topics and m:
        bag_start = min(v[1] for v in topics.values())
        bag_end = max(v[2] for v in topics.values())
        out.update(log_bag_s=bag_end - bag_start,
                   start_limited_by=",".join(t for t, v in topics.items() if abs(v[1] - out["kitti_start"]) < 0.05),
                   end_limited_by=",".join(t for t, v in topics.items() if abs(v[2] - out["kitti_end"]) < 0.05))
    out["filled_frames"] = sum(int(x) for x in re.findall(r"filling missing (\d+) frames", txt))
    return out


def signal_qc(run_dir: Path) -> dict:
    q = {}
    for sub, (fn, cols) in SIGNALS.items():
        f, tf = run_dir / sub / fn, run_dir / sub / "timestamps.txt"
        if not f.exists() or not tf.exists():
            q[f"{sub}_present"] = False
            continue
        a = np.load(f, allow_pickle=False)
        t = np.loadtxt(tf)
        a2 = a.reshape(len(a), -1)
        dt = np.diff(t)
        q.update({f"{sub}_present": True, f"{sub}_n": len(a), f"{sub}_hz": 1 / np.median(dt) if len(dt) else np.nan,
                  f"{sub}_gap_max_s": float(dt.max()) if len(dt) else np.nan, f"{sub}_backward": int((dt < 0).sum()),
                  f"{sub}_dup_rows_frac": float(np.mean(np.all(np.diff(a2, axis=0) == 0, axis=1))) if len(a2) > 1 else np.nan,
                  f"{sub}_nan": int(np.isnan(a2).sum())})
        if cols and a2.shape[1] == len(cols):
            for i, c in enumerate(cols):
                v = a2[:, i]
                q[f"{sub}_{c}_p001"], q[f"{sub}_{c}_p50"], q[f"{sub}_{c}_p999"] = np.nanpercentile(v, [0.1, 50, 99.9])
                q[f"{sub}_{c}_min"], q[f"{sub}_{c}_max"] = np.nanmin(v), np.nanmax(v)
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/offroad/tartandrive2.yaml")
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    data = Path(cfg["data_dir"])
    out = Path("out/offroad/tartandrive2") / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted({p.name for p in (data / "bags").iterdir()} | {p.name for p in (data / "kitti/all_topics").iterdir()})
    rows = []
    for r in runs:
        row = dict(run=r)
        bd = data / "bags" / r
        if bd.exists():
            row.update(bag_durations(bd))
        kd = data / "kitti/all_topics" / r
        row["kitti"] = kd.exists()
        if (kd / "data_extraction.log").exists():
            row.update(parse_log(kd / "data_extraction.log"))
        if kd.exists():
            row.update(signal_qc(kd))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "inventory.csv", index=False)
    summ = dict(runs_bag=int(df.bag_files.notna().sum()) if "bag_files" in df else 0, runs_kitti=int(df.kitti.sum()),
                bag_h=float(df.get("bag_s", pd.Series(dtype=float)).sum() / 3600),
                bag_h_with_kitti=float(df.loc[df.kitti, "bag_s"].sum() / 3600) if "bag_s" in df else None,
                kitti_h=float(df.get("kitti_s", pd.Series(dtype=float)).sum() / 3600))
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    print(json.dumps(summ, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
