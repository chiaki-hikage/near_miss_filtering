import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "vibration"))
sys.path.insert(0, str(REPO / "scripts"))

import locate_candidates as LC  # noqa: E402

LOC = dict(site=dict(radius_m=6.0, heading_deg=30.0, lane_half_m=1.5),
           repeat=dict(a_min=3.0, heave_ratio_min=3.0, edge_margin_m=1.0, contrast_min=1.3))


def _llh_to_ecef(lat, lon, h=0.0):
    a, e2 = 6378137.0, 6.69437999014e-3
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    N = a / np.sqrt(1 - e2 * np.sin(la) ** 2)
    return np.array([(N + h) * np.cos(la) * np.cos(lo), (N + h) * np.cos(la) * np.sin(lo), (N * (1 - e2) + h) * np.sin(la)])


def test_ecef_round_trip_and_local_frame():
    lat, lon = 37.726831, -122.471983
    la, lo = LC.ecef_to_llh_vec(_llh_to_ecef(lat, lon, 30.0)[None, :])
    assert la[0] == pytest.approx(lat, abs=1e-7) and lo[0] == pytest.approx(lon, abs=1e-7)
    f = LC.LocalFrame(37.65, -122.44)
    xy = f.xy([lat], [lon])
    back = f.latlon(xy)
    assert back[0][0] == pytest.approx(lat, abs=1e-9) and back[1][0] == pytest.approx(lon, abs=1e-9)
    # 北へ 100 m
    d = f.xy([lat + 100 / LC.M_PER_DEG_LAT], [lon]) - xy
    assert d[0] == pytest.approx([0.0, 100.0], abs=1e-6)


def test_heading_from_ecef_velocity_points_north_and_east():
    p = _llh_to_ecef(37.7, -122.47)[None, :]
    north = _llh_to_ecef(37.7 + 1e-4, -122.47)[None, :] - p
    east = _llh_to_ecef(37.7, -122.47 + 1e-4)[None, :] - p
    assert LC.ecef_velocity_heading(p, north)[0] == pytest.approx(0.0, abs=0.1) or \
        LC.ecef_velocity_heading(p, north)[0] == pytest.approx(360.0, abs=0.1)
    assert LC.ecef_velocity_heading(p, east)[0] == pytest.approx(90.0, abs=0.1)


def test_gps_time_to_unix_matches_drive_name():
    # docs/datasets.md: 週 2011 + 454455.3 s = UTC 2018-07-27 06:13:57
    t = pd.to_datetime(LC.gps_to_unix(np.array([2011.0]), np.array([454455.3]))[0], unit="s")
    assert t.strftime("%Y-%m-%d %H:%M:%S") == "2018-07-27 06:13:57"


def test_cluster_sites_separates_lanes_but_merges_along_track():
    # 北向き。同じ車線で進行方向に 4 m 離れた 2 件はまとめ、横に 3 m (隣の車線) は分ける
    xy = np.array([[0.0, 0.0], [0.0, 4.0], [3.0, 0.0], [0.0, 30.0]])
    hdg = np.zeros(4)
    site = LC.cluster_sites(xy, hdg, np.array([5.0, 4.0, 3.0, 2.0]), 6.0, 30.0, 1.5)
    assert site[0] == site[1]
    assert len({site[0], site[2], site[3]}) == 3


def test_cluster_sites_separates_opposite_directions():
    xy = np.zeros((2, 2))
    site = LC.cluster_sites(xy, np.array([0.0, 180.0]), np.array([2.0, 1.0]), 6.0, 30.0, 1.5)
    assert site[0] != site[1]


def test_split_passes_breaks_on_time_gap():
    t = np.arange(0, 20, 0.01)
    m = ((t > 1) & (t < 2)) | ((t > 10) & (t < 11))
    g = LC.split_passes(m, t)
    assert len(g) == 2 and t[g[1][0]] > 10


def test_repeat_hits_requires_interior_peak_and_contrast():
    d = pd.DataFrame(dict(
        A_max=[4.0, 4.0, 4.0, 2.0], A_off_m=[0.5, 5.5, 0.5, 0.0], A_contrast=[2.0, 2.0, 1.1, 3.0],
        heave_ratio_max=[4.0] * 4, heave_off_m=[0.5] * 4, heave_contrast=[2.0] * 4))
    a, h = LC.repeat_hits(d, LOC)
    # 内側で際立つ / 窓の端 / 前後と同じ高さ (荒れた区間の途中) / 閾値未満
    assert a.tolist() == [True, False, False, False]
    assert h.all()


def test_binom_surprise_grows_with_hits():
    assert LC.binom_surprise(0, 5, 0.1) == pytest.approx(0.0)
    assert LC.binom_surprise(5, 5, 0.1) == pytest.approx(5.0)
    assert LC.binom_surprise(3, 8, 0.05) > LC.binom_surprise(2, 8, 0.05)
    assert LC.binom_surprise(1, 0, 0.1) == 0.0


def test_stretches_link_chain_of_sites():
    xy = np.array([[0.0, 0.0], [0.0, 40.0], [0.0, 80.0], [0.0, 300.0]])
    lab = LC.stretches(xy, np.zeros(4), 50.0, 30.0)
    assert lab[0] == lab[1] == lab[2] != lab[3]


def test_pick_top_caps_sites_per_stretch():
    s = pd.DataFrame(dict(site_id=range(5), stretch_id=[0, 0, 0, 1, 2]))
    assert LC.pick_top(s, 10, 2).site_id.tolist() == [0, 1, 3, 4]
