"""地点 19: 走行位置 (車線と横位置) の不確かさが、新 B モデルの予測に与える影響を調べる。

site19_validate.py で校正した路面 (calibration.json) をそのまま使い、
対象走行の横位置だけを変えて予測する。路面のパラメータは変えない。

    uv run python scripts/vibration/site19_lane_sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yaml

import fullcar as FC
import site19_validate as V
from near_miss.io.comma2k19 import find_segments

REPO = _bootstrap.SRC.parent


def main() -> int:
    out = REPO / "out/vibration/pavement"
    cfg = yaml.safe_load((REPO / "configs/vibration_pavement.yaml").read_text())
    vib = dict(yaml.safe_load((REPO / "configs/vibration.yaml").read_text()))
    vib["grid_hz"] = V.FS
    p = FC.build_params(FC.load_config(REPO / "configs/vibration_fullcar.yaml"))
    calib = json.loads((out / "calibration.json").read_text())["calibration"]
    seeds = list(range(cfg["road"]["seeds"]))
    refs = find_segments(REPO / "raw_data/Chunk_1")
    rows = []
    for _, r in pd.read_csv(out / "site19_passes.csv").iterrows():
        P = V.Pass(r, refs, vib)
        cands = [P.y_lane - 0.5, P.y_lane, P.y_lane + 0.5]
        if P.tag == "08-03 seg16":
            cands += [V.LANE_WIDTH_M - 0.5, V.LANE_WIDTH_M, V.LANE_WIDTH_M + 0.5]
        for y in cands:
            P.y_lane = y
            ms = [V.metrics(P, V.simulate(P, p, V.build("新B", calib, cfg, sd))) for sd in seeds]
            med = pd.DataFrame(ms).median(numeric_only=True)
            rows.append(dict(tag=P.tag, lane_by_gps=P.lane, y_lane_m=y, env_r2=round(med.env_r2, 3),
                             env_corr=round(med.env_corr, 3), onset_err_m=round(med.onset_err_m, 2),
                             heave_8_15_logerr=round(med["heave_8-15_logerr"], 3),
                             heave_3_8_logerr=round(med["heave_3-8_logerr"], 3),
                             roll_rms_ratio=round(med.get("roll_rms_ratio", np.nan), 2)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "lane_sensitivity.csv", index=False)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
