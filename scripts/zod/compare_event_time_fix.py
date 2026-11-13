#!/usr/bin/env python3
"""候補の時刻の定義の修正 (scripts/zod/event_time.py、2026-11-13) の前後の比較。ネットワークは使わない。

  uv run --offline python scripts/zod/compare_event_time_fix.py --run-id <id>

比べるもの (既定の実行):
  Phase 2  out/zod/full_screening/20261112_r6 (前) → 20261113_r7 (後)
  Phase 3  out/zod/full_weather/20261112_r3 (前) → 20261113_r4 (後)
  映像     out/zod/video_review/20261113_r1 (前) → 20261113_r2 (後)
出力: out/zod/event_time_fix/<run_id>/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["drive_id", "residual", "kind", "speed_kmh_setting", "i0", "i1"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--p2", nargs=2, default=["out/zod/full_screening/20261112_r6", "out/zod/full_screening/20261113_r7"])
    ap.add_argument("--p3", nargs=2, default=["out/zod/full_weather/20261112_r3", "out/zod/full_weather/20261113_r4"])
    ap.add_argument("--video", nargs=2, default=["out/zod/video_review/20261113_r1", "out/zod/video_review/20261113_r2"])
    a = ap.parse_args()
    out = Path("out/zod/event_time_fix") / a.run_id
    if out.exists():
        print("すでにある。上書きしない", file=sys.stderr)
        return 1
    out.mkdir(parents=True)
    summ = {}

    # ---- Phase 2
    e0 = pd.read_csv(Path(a.p2[0]) / "events_all.csv")
    e1 = pd.read_csv(Path(a.p2[1]) / "events_all.csv")
    m = e0.merge(e1, on=KEY, how="outer", suffixes=("_old", "_new"), indicator=True)
    same_cols = ["event_id", "hypothesis", "peak_over_thr", "t_onset", "t_peak", "duration_s", "dr_peak_deg_s", "lat", "lon"]
    both = m[m._merge == "both"]
    summ["phase2"] = dict(events_old=len(e0), events_new=len(e1), matched=int((m._merge == "both").sum()),
                          only_old=int((m._merge == "left_only").sum()), only_new=int((m._merge == "right_only").sum()),
                          unchanged_columns={c: bool(np.allclose(both[f"{c}_old"], both[f"{c}_new"], equal_nan=True)) if both[f"{c}_old"].dtype.kind in "fi"
                                             else bool((both[f"{c}_old"].astype(str) == both[f"{c}_new"].astype(str)).all()) for c in same_cols})
    u0 = pd.read_csv(Path(a.p2[0]) / "candidates_unique.csv", dtype={"in_camera_sequences": str})
    u1 = pd.read_csv(Path(a.p2[1]) / "candidates_unique.csv", dtype={"in_camera_sequences": str})
    summ["phase2"]["unique_old"] = len(u0)
    summ["phase2"]["unique_new"] = len(u1)
    summ["phase2"]["unique_same_ids"] = bool(sorted(u0.event_id) == sorted(u1.event_id))
    u = u1.merge(u0[["event_id", "in_camera_sequences"]], on="event_id", suffixes=("", "_old"))
    u["old_interval_start"] = u.t_onset
    u["old_interval_end"] = u.t_onset + u.duration_s
    u["start_shift_s"] = (u.t_exceed_start - u.old_interval_start).round(2)
    u["end_shift_s"] = (u.t_exceed_end - u.old_interval_end).round(2)
    u["in_camera_old"] = u.in_camera_sequences_old.notna() & (u.in_camera_sequences_old.astype(str) != "nan")
    u["in_camera_new"] = u.camera_overlap_s > 0
    cols = ["event_id", "car", "drive_id", "kind", "hypothesis", "t_rise_utc", "rise_censored", "t_exceed_start_utc", "t_exceed_end_utc", "exceed_duration_s",
            "exceed_start_censored", "exceed_start_censor_reason", "exceed_end_censored", "exceed_end_censor_reason", "t_peak_utc",
            "start_shift_s", "end_shift_s", "in_camera_sequences_old", "in_camera_sequences", "camera_overlap_s", "camera_relation", "peak_in_camera"]
    u[cols].to_csv(out / "phase2_candidates_intervals.csv", index=False)
    summ["phase2"].update(start_shift_s=u.start_shift_s.describe().round(2).to_dict(), end_shift_s=u.end_shift_s.describe().round(2).to_dict(),
                          rise_censored=int(u.rise_censored.sum()), exceed_start_censored=int(u.exceed_start_censored.sum()),
                          exceed_end_censored=int(u.exceed_end_censored.sum()),
                          in_camera_old=int(u.in_camera_old.sum()), in_camera_new=int(u.in_camera_new.sum()),
                          camera_relation=u.camera_relation.replace("", "カメラなし").value_counts().to_dict())

    # ---- Phase 3
    w0 = pd.read_csv(Path(a.p3[0]) / "candidates_weather.csv")
    w1 = pd.read_csv(Path(a.p3[1]) / "candidates_weather.csv")
    pc = ["idw_w_left", "nearest_w_left", "w_left_sigma", "idw_speed", "gust10_max", "window_s", "mean_dr_deg_s", "a_steer", "a_bank", "bank_ratio",
          "torque_anom_nm", "torque_agrees_steer", "dbeta_deg", "beta_agrees", "req_w_left_10m_lo", "req_w_left_10m_hi", "expected_w_left_sign", "verdict", "verdict_reason"]
    w = w1[["event_id", "kind", "t_exceed_start_utc", "t_exceed_end_utc", "wind_time_utc", "camera_overlap_s"] + pc].merge(
        w0[["event_id", "t_onset_utc"] + pc], on="event_id", suffixes=("_new", "_old"))
    w["verdict_changed"] = w.verdict_new != w.verdict_old
    w["wind_time_shift_s"] = (pd.to_datetime(w.wind_time_utc) - pd.to_datetime(w.t_onset_utc, format="ISO8601")).dt.total_seconds().round(1)
    w.to_csv(out / "phase3_candidates_before_after.csv", index=False)
    r0 = pd.read_csv(Path(a.p3[0]) / "reverse_search.csv")
    r1 = pd.read_csv(Path(a.p3[1]) / "reverse_search.csv")
    s0 = json.loads((Path(a.p3[0]) / "summary.json").read_text())
    s1 = json.loads((Path(a.p3[1]) / "summary.json").read_text())
    summ["phase3"] = dict(verdict_old=w0.groupby(["kind", "verdict"]).size().to_dict(), verdict_new=w1.groupby(["kind", "verdict"]).size().to_dict(),
                          verdict_changed=w[w.verdict_changed][["event_id", "verdict_old", "verdict_new"]].to_dict("records"),
                          reverse_search_identical=bool(r0.equals(r1)),
                          within_drive_identical=s0["within_drive"] == s1["within_drive"])
    summ["phase3"] = {k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in summ["phase3"].items()}

    # ---- 映像の一覧
    v0 = pd.read_csv(Path(a.video[0]) / "index.csv", dtype={"sequence_id": str})
    v1 = pd.read_csv(Path(a.video[1]) / "index.csv", dtype={"sequence_id": str})
    vf = v1[v1.is_focus == True][["sequence_id", "event_id", "detect_rel_camera_s", "overlap_with_camera_s", "camera_relation", "rise_rel_camera_s", "rise_censored",  # noqa: E712
                                 "exceed_start_censored", "exceed_end_censored", "peak_rel_camera_s", "verdict"]].merge(
        v0[v0.is_focus == True][["sequence_id", "event_id", "detect_rel_camera_s", "overlap_with_camera_s", "verdict"]],  # noqa: E712
        on=["sequence_id", "event_id"], how="outer", suffixes=("_new", "_old"))
    vf.to_csv(out / "video_focus_before_after.csv", index=False)
    summ["video"] = dict(focus_events=len(vf), in_camera_old=int((vf.overlap_with_camera_s_old > 0).sum()), in_camera_new=int((vf.overlap_with_camera_s_new > 0).sum()))
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
