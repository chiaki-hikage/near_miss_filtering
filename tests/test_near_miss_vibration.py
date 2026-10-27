import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "vibration"))
sys.path.insert(0, str(REPO / "scripts"))

import rank_near_miss_vibration as R  # noqa: E402

NM = yaml.safe_load((REPO / "configs/vibration_near_miss.yaml").read_text())
FS = 100.0
T = np.arange(-2.0, 2.5, 1 / FS)


def _sig(heave, roll):
    return dict(t=T, heave=heave, roll=roll, pitch=0.5 * heave)


def test_short_burst_is_concentrated_and_fast():
    burst = np.exp(-((T - 0.1) / 0.08) ** 2) * np.sin(2 * np.pi * 11 * T)
    f = R.response_features(_sig(3 * burst, 5 * burst), FS, NM)
    assert f["concentration_heave"] > 0.9
    assert f["rise_s"] <= NM["impulsive"]["max_rise_s"]
    assert abs(f["heave_roll_dt_s"]) < 0.05


def test_long_rough_road_is_not_concentrated():
    long = np.sin(2 * np.pi * 9 * T) * (np.abs(T) < 2.0)
    f = R.response_features(_sig(2 * long, 3 * long), FS, NM)
    # 前後 1.5 s のうち前後 0.5 s は 1/3 なので、エネルギーの割合も 1/3 程度になる
    assert f["concentration_heave"] == pytest.approx(1 / 3, abs=0.05)
    assert f["concentration_heave"] < NM["impulsive"]["min_concentration"]


def test_roll_peak_time_offset_is_measured():
    h = np.exp(-((T - 0.0) / 0.05) ** 2)
    r = np.exp(-((T - 0.4) / 0.05) ** 2)
    f = R.response_features(_sig(h, r), FS, NM)
    assert f["heave_roll_dt_s"] == pytest.approx(0.4, abs=0.02)
    assert f["heave_peak_abs"] == pytest.approx(1.0, abs=1e-6)
