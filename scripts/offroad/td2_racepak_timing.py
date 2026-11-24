"""racepak の時刻のずれの原因を調べる: 60 s の block ごとに、rosbag のチャンクを 1 つ取り、
racepak (/shock_pos) のヘッダの時刻と bag の記録時刻の差を測り、包絡線の相互相関で推定した遅れ (td2_vibration_poc.py の lags.csv) と並べる。

予想 (反証できる形): ヘッダ − 記録時刻 が +1 s の block では遅れが約 1.4 s、0 の block では約 0.4 s。

    uv run python scripts/offroad/td2_racepak_timing.py --lags out/offroad/tartandrive2/20261117_r3/lags.csv \
        --run <走行> --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yaml

import rosbag_range as RB
import td2_event_frames as F


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="configs/offroad/tartandrive2.yaml")
    ap.add_argument("--config", default="configs/offroad/td2_vibration_poc.yaml")
    ap.add_argument("--lags", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    src = yaml.safe_load(open(a.source))
    tc = yaml.safe_load(open(a.config))["racepak_timing"]
    root = Path(src["data_dir"])
    RB.CACHE_DIR = root / "bag_cache"
    out = Path("out/offroad/tartandrive2") / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    L = pd.read_csv(a.lags)
    L = L[L.run == a.run]
    kd = root / "kitti/all_topics" / a.run
    t_k0 = np.loadtxt(kd / "novatel_imu/timestamps.txt")[0]       # lags.csv の block_start_s の原点 (IMU の格子の先頭)
    idx = []
    for nm, _ in F.split_files(root / "bags" / a.run):
        url = f"{src['source']['endpoint']}/{src['source']['bucket']}/{urllib.parse.quote(f'bags/{a.run}/{nm}')}"
        idx.append((nm, RB.read_index(url)))
    used, rows, msgs_all = 0, [], []
    for _, b in L.iterrows():
        t = t_k0 + b.block_start_s + 30.0                              # block の真ん中
        for nm, bi in idx:
            sid = set(bi.topic_ids("/shock_pos"))
            cs = [c for c in bi.chunks if c["start"] <= t <= c["end"] + 0.5 and sid & set(c["counts"])]
            if cs:
                break
        if not cs:
            rows.append(dict(block_start_s=b.block_start_s, lag_s=b.lag_s, corr=b["corr"], note="チャンクなし"))
            continue
        c = min(cs, key=lambda c: abs(c["start"] - t))
        nxt = [x["pos"] for x in bi.chunks if x["pos"] > c["pos"]]
        end = (min(nxt) if nxt else bi.size) - 1
        if not RB.cached(bi.url, c["pos"], end):
            used += end + 1 - c["pos"]
        if used > tc["max_download_mb"] * 1e6:
            raise SystemExit(f"上限 {tc['max_download_mb']} MB を超えた")
        ids = {i: x["topic"] for i, x in bi.conns.items() if x["topic"] in tc["topics"]}
        ms = RB.read_chunk_messages(bi, c, set(ids))
        d = {}
        for cid, rt, md in ms:
            ht = RB.decode_header(md)[0]
            d.setdefault(ids[cid], []).append(ht - rt)
            msgs_all.append(dict(block_start_s=b.block_start_s, file=nm, topic=ids[cid], record_t=rt, header_t=ht))
        rows.append(dict(block_start_s=b.block_start_s, file=nm, lag_s=b.lag_s, corr=b["corr"], speed_med=b.speed_med,
                         **{f"hdr_minus_rec_{k.strip('/').replace('/', '_')}": float(np.median(v)) for k, v in d.items()},
                         n_shock=len(d.get("/shock_pos", []))))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / "blocks.csv", index=False)
    pd.DataFrame(msgs_all).to_csv(out / "messages.csv", index=False)
    (out / "meta.json").write_text(json.dumps(dict(run=a.run, lags=str(a.lags), downloaded_mb=used / 1e6, config=tc), ensure_ascii=False, indent=2))
    print(f"取得 {used / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
