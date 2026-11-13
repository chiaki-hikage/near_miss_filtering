#!/usr/bin/env python3
"""ZOD Sequences の時刻の対応の確かめ: カメラ・LiDAR・ego_motion・OXTS・車両データの区間と、Phase 2 の候補の時刻。
ネットワークは使わない。

  uv run --offline --extra viz --with "zod==0.8.0" --with "dataclass-wizard==0.39.1" python scripts/zod/time_alignment.py --run-id <id>

出力 (out/zod/time_alignment/<run_id>/):
  sequence_times.csv     シーケンスごと: info.json (start/end/keyframe)、カメラ・LiDAR の最初と最後のフレーム、metadata.json、
                         ego_motion.json、OXTS (SDK と同じ変換)、車両データの区間と差。ego_motion と OXTS の中身の一致
  camera_sync_wide.csv   mini の 2 シーケンス: 画像の横の流れと OXTS の方位の変化の相関を ±10 s で (前回は ±1 s)
  candidate_intervals.csv  Phase 2 の候補の、閾値を超えた区間の正しい時刻 (配列の位置 i0, i1 から) と、前の計算 (始まり + 持続時間) との違い
  summary.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_screening as F  # noqa: E402
import signal_survey as S  # noqa: E402


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def seq_times(root: Path, sid: str) -> dict:
    d = root / "sequences" / sid
    info = json.loads((d / "info.json").read_text())
    meta = json.loads((d / "metadata.json").read_text())
    em = json.loads((d / "ego_motion.json").read_text())
    cam = [ts(f["time"]) for f in info["camera_frames"]["front_blur"]]
    lid = [ts(f["time"]) for f in info["lidar_frames"]["velodyne"]]
    te = np.array(em["timestamps"])
    ox = S.load_oxts(d / "oxts.hdf5")                       # SDK の EgoMotion.from_oxts_path と同じ式 (GPS 元期 + timestamp + leapSeconds)
    t = ox.t.to_numpy()
    import h5py
    with h5py.File(d / "vehicle_data.hdf5", "r") as f:
        vt = f["ego_vehicle_controls/timestamp/nanoseconds/value"][()] / 1e9
    wz = np.array(em["angular_rates"])[:, 2]
    vx = np.array(em["velocities"])[:, 0]
    inside = (te >= t[0]) & (te <= t[-1])
    dz = np.abs(np.interp(te[inside], t, ox.angularRateZ.to_numpy()) - wz[inside])
    dv = np.abs(np.interp(te[inside], t, ox.velForward.to_numpy()) - vx[inside])
    union = np.unique(np.round(np.r_[cam, lid], 4))
    c0, c1 = cam[0], cam[-1]
    return dict(id=sid, car=meta["collection_car"],
                info_start_minus_cam0_s=round(ts(info["start_time"]) - c0, 4), info_end_minus_cam1_s=round(ts(info["end_time"]) - c1, 4),
                keyframe_minus_cam0_s=round(ts(info["keyframe_time"]) - c0, 3),
                meta_start_minus_cam0_s=round(ts(meta["start_time"]) - c0, 4), meta_end_minus_cam1_s=round(ts(meta["end_time"]) - c1, 4),
                cam_n=len(cam), cam_span_s=round(c1 - c0, 3), lidar_n=len(lid), lidar0_minus_cam0_s=round(lid[0] - c0, 4), lidar1_minus_cam1_s=round(lid[-1] - c1, 4),
                ego_n=len(te), ego0_minus_cam0_s=round(te[0] - c0, 4), ego1_minus_cam1_s=round(te[-1] - c1, 4),
                ego_ts_equals_cam_lidar_union=bool(len(union) == len(te) and np.allclose(np.sort(te), union, atol=1e-3)),
                ego_inside_oxts_frac=round(float(inside.mean()), 4),
                ego_vs_oxts_wz_maxabs=float(dz.max()) if len(dz) else np.nan, ego_vs_oxts_vx_maxabs=float(dv.max()) if len(dv) else np.nan,
                oxts_span_s=round(t[-1] - t[0], 3), cam0_minus_oxts0_s=round(c0 - t[0], 3), oxts1_minus_cam1_s=round(t[-1] - c1, 3),
                vehicle_span_min=round((vt[-1] - vt[0]) / 60, 2), vehicle_covers_camera=bool(vt[0] <= c0 and vt[-1] >= c1),
                cam0_minus_vehicle0_min=round((c0 - vt[0]) / 60, 2))


def camera_wide(root: Path, sid: str, lag_range: float):
    from zod import ZodSequences
    cfg = yaml.safe_load(Path("configs/zod/signal_survey.yaml").read_text())
    cfg = copy.deepcopy(cfg)
    cfg["camera_alignment"]["lag_range_s"] = lag_range
    cfg["camera_alignment"]["lag_step_s"] = 0.05
    ds = ZodSequences(str(root), "mini", mp=False)
    seq = ds[sid]
    ox = S.load_oxts(Path(seq.info.oxts_path))
    calib = json.loads(Path(seq.info.calibration_path).read_text())
    out = Path("/private/tmp") if False else None
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        r = S.camera_sync(sid, seq.info, ox, calib, cfg, Path(td))
    return r


def candidate_intervals(root: Path, cfg_scr: dict):
    """Phase 2 の候補の i0, i1, ion, ip (走行ごとの OXTS を時刻で重ねずにつないだ 100 Hz の配列の位置) から時刻を出す。"""
    scr = Path(cfg_scr["screening_run"])
    inv = Path(cfg_scr["inventory_run"])
    seqs = pd.read_csv(inv / "sequences.csv", dtype={"id": str})
    u = pd.read_csv(scr / "candidates_unique.csv")
    F.ROOT = root
    rows = []
    for did, g in u.groupby("drive_id"):
        ox = F.oxts_union(sorted(seqs[seqs.drive_id == did].id))
        t = ox.t.to_numpy()
        blk = np.r_[0, np.cumsum(np.diff(t) > 0.05)]
        for _, r in g.iterrows():
            i0, i1, ion, ip = int(r.i0), int(r.i1), int(r.ion), int(r.ip)
            bm = np.where(blk == blk[i0])[0]
            run0, run1 = t[i0], t[i1 - 1]
            # このカメラ区間 (同じ走行で OXTS が run を含むシーケンス)
            s = seqs[(seqs.drive_id == did)]
            cams = []
            for _, q in s.iterrows():
                c0, c1 = ts(q.cam_start_utc), ts(q.cam_end_utc)
                ov = max(0.0, min(run1, c1) - max(run0, c0))
                if q.oxts_t0 <= t[ip] <= q.oxts_t1 or ov > 0:
                    cams.append((q.id, c0, c1, ov))
            best = max(cams, key=lambda x: (x[3], -abs(x[1] - run0))) if cams else (None, np.nan, np.nan, 0.0)
            rows.append(dict(event_id=r.event_id, drive_id=did, kind=r.kind, hypothesis=r.hypothesis,
                             check_onset_matches=bool(abs(t[ion] - r.t_onset) < 1e-3), check_peak_matches=bool(abs(t[ip] - r.t_peak) < 1e-3),
                             onset_utc=r.t_onset_utc, onset_at_block_start=bool(abs(t[ion] - t[bm[0]]) < 0.05),
                             block_start_utc=datetime.fromtimestamp(t[bm[0]], timezone.utc).isoformat(timespec="milliseconds"),
                             run_start_utc=datetime.fromtimestamp(run0, timezone.utc).isoformat(timespec="milliseconds"),
                             run_end_utc=datetime.fromtimestamp(run1, timezone.utc).isoformat(timespec="milliseconds"),
                             onset_minus_run_start_s=round(t[ion] - run0, 2), duration_s=r.duration_s,
                             old_end_minus_true_end_s=round((r.t_onset + r.duration_s) - run1, 2),
                             camera_sequence=best[0], run_start_rel_cam_s=round(run0 - best[1], 2) if best[0] else None,
                             run_end_rel_cam_s=round(run1 - best[1], 2) if best[0] else None, peak_rel_cam_s=round(t[ip] - best[1], 2) if best[0] else None,
                             true_overlap_with_camera_s=round(best[3], 2)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="raw_data/zod")
    ap.add_argument("--run-id", default=datetime.now().strftime("%Y%m%dT%H%M%S"))
    a = ap.parse_args()
    out = Path("out/zod/time_alignment") / a.run_id
    if out.exists():
        print("すでにある。上書きしない", file=sys.stderr)
        return 1
    out.mkdir(parents=True)
    root = Path(a.root)
    ids = sorted(p.name for p in (root / "sequences").iterdir() if (p / "ego_motion.json").exists())
    st = pd.DataFrame([seq_times(root, s) for s in ids])
    st.to_csv(out / "sequence_times.csv", index=False)
    cw = pd.DataFrame([camera_wide(root, s, 10.0) for s in ["000000", "000002"]])
    cw.to_csv(out / "camera_sync_wide.csv", index=False)
    cfg_scr = yaml.safe_load(Path("configs/zod/full_screening.yaml").read_text())
    cfg_scr["screening_run"] = "out/zod/full_screening/20261112_r6"
    ci = candidate_intervals(root, cfg_scr)
    ci.to_csv(out / "candidate_intervals.csv", index=False)

    def q(c):
        x = st[c].astype(float)
        return dict(min=round(x.min(), 4), p50=round(x.median(), 4), max=round(x.max(), 4))
    summ = dict(n=len(st), **{c: q(c) for c in ["info_start_minus_cam0_s", "info_end_minus_cam1_s", "meta_start_minus_cam0_s", "lidar0_minus_cam0_s",
                                                  "ego0_minus_cam0_s", "ego1_minus_cam1_s", "cam_span_s", "oxts_span_s", "cam0_minus_oxts0_s", "oxts1_minus_cam1_s",
                                                  "ego_vs_oxts_wz_maxabs", "ego_vs_oxts_vx_maxabs", "vehicle_span_min", "cam0_minus_vehicle0_min"]},
                ego_ts_equals_cam_lidar_union=int(st.ego_ts_equals_cam_lidar_union.sum()), ego_inside_oxts_all=int((st.ego_inside_oxts_frac == 1).sum()),
                vehicle_covers_camera=int(st.vehicle_covers_camera.sum()),
                camera_sync_wide=cw[["id", "corr_at_0", "best_lag_s", "corr_at_best", "lags_within_0p01_of_best"]].to_dict("records"),
                candidates=dict(n=len(ci), onset_at_block_start=int(ci.onset_at_block_start.sum()),
                                index_checks_ok=int((ci.check_onset_matches & ci.check_peak_matches).sum())))
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
