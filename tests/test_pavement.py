import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "vibration"))

import pavement as PV  # noqa: E402


def test_band_rms_matches_numeric_integral():
    g = PV.ISO8608_GD0["B"]
    n = np.linspace(0.2, 10, 200000)
    num = np.sqrt(np.trapezoid(PV.gd(n, g), n))
    assert PV.band_rms(g, 0.2, 10) == pytest.approx(num, rel=1e-4)
    # 帯域を狭めれば RMS は小さくなる (路面高さの RMS は帯域と組で意味を持つ)
    assert PV.band_rms(g, 0.5, 1.0) < PV.band_rms(g, 0.2, 10)


def test_profile_psd_follows_iso_definition():
    r = PV.Roughness((0.2, 10.0), seed=3, n_comp=600)
    s = np.arange(0, 400, 0.02)
    h = np.sqrt(64e-6) * r.unit(s, 0.0, common=False)
    assert np.std(h) == pytest.approx(PV.band_rms(64e-6, 0.2, 10), rel=0.15)


def test_common_component_is_identical_left_right():
    r = PV.Roughness((0.2, 10.0), seed=1)
    s = np.arange(0, 30, 0.01)
    m = PV.PavementModel("B", [PV.Zone((0, 30), (-10, 10), 1e-4)], r, 0.0, common_fraction=1.0)
    assert np.allclose(m.height(s, -0.8), m.height(s, 0.8))


def test_partial_correlation_decreases_with_separation():
    r = PV.Roughness((0.2, 10.0), seed=2, n_comp=800)
    s = np.arange(0, 300, 0.01)
    base = r.unit(s, 0.0, common=False)
    c_near = np.corrcoef(base, r.unit(s, 0.2, common=False))[0, 1]
    c_far = np.corrcoef(base, r.unit(s, 1.6, common=False))[0, 1]
    assert c_near > c_far


def test_zone_confines_roughness():
    r = PV.Roughness((0.2, 10.0), seed=0)
    m = PV.PavementModel("A", [PV.Zone((10, 14), (-2, 2), 1e-3)], r, gd0_outside=0.0)
    s = np.arange(0, 30, 0.01)
    h = m.height(s, 0.0)
    assert np.all(h[(s < 10) | (s > 14)] == 0)
    assert np.std(h[(s > 10.5) & (s < 13.5)]) > 0
    assert np.all(m.height(s, 5.0) == 0)   # 横方向の範囲の外


def test_rear_wheels_follow_front_by_wheelbase():
    bump = lambda s, y: np.where((s > 5) & (s < 5.5), 0.01, 0.0)
    s_front = np.linspace(0, 20, 4001)
    zr = PV.track_inputs(bump, s_front, 0.0, 2.66, 1.57, envelope_m=0.005)
    i_front = np.argmax(zr[:, 0] > 0.005)
    i_rear = np.argmax(zr[:, 2] > 0.005)
    assert s_front[i_rear] - s_front[i_front] == pytest.approx(2.66, abs=0.01)
