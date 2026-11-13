"""ZOD の外乱候補の時刻 (推定開始・閾値超過区間・ピーク) の回帰テスト。docs/zod_event_time_fix.md

2026-11-13 に見つけた誤り: Phase 2 の「推定開始」(ピークから半分まで戻った時刻) に「閾値超過の長さ」を足して検出区間とし、
持続型の候補の区間が 1〜21 s 前にずれた (Phase 3 の物理の確かめ・カメラとの重なりが誤った)。同じ誤りが戻らないことを確かめる。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ZOD = Path(__file__).resolve().parents[1] / "scripts" / "zod"
sys.path.insert(0, str(ZOD))

import event_time as ET  # noqa: E402


def _arrays(n=3000, dt=0.01):
    t = 1_650_000_000.0 + np.arange(n) * dt
    return t, np.zeros(n, int), np.ones(n, bool)


# ---------------------------------------------------------------- event_time
def test_interval_fields_basic():
    t, blk, mask = _arrays()
    mask[:200] = False
    f = ET.interval_fields(t, blk, mask, i0=500, i1=800, ion=300, ip=700)
    assert f["t_exceed_start"] == pytest.approx(t[500])
    assert f["t_exceed_end"] == pytest.approx(t[799] + 0.01)
    assert f["exceed_duration_s"] == pytest.approx(3.0)
    assert f["t_rise"] == pytest.approx(t[300])
    assert f["t_peak"] == pytest.approx(t[700])
    assert not f["rise_censored"] and not f["exceed_start_censored"] and not f["exceed_end_censored"]


def test_rise_plus_duration_is_not_the_interval():
    """今回の誤りの形: 推定開始が閾値超過の始まりより前 → 推定開始 + 長さ は閾値超過の終わりではない。"""
    t, blk, mask = _arrays()
    f = ET.interval_fields(t, blk, mask, i0=2000, i1=2300, ion=500, ip=2200)
    wrong_end = f["t_rise"] + f["exceed_duration_s"]
    assert f["t_exceed_end"] - wrong_end == pytest.approx(15.0)
    row = pd.Series(f)
    assert ET.event_interval(row) == (pytest.approx(t[2000]), pytest.approx(t[2299] + 0.01))
    # カメラが閾値超過区間にかかるとき、重なりは閾値超過区間から出す (誤った区間では 0 になる)
    cam0, cam1 = t[2100], t[2100] + 20
    assert ET.overlap(*ET.event_interval(row), cam0, cam1) == pytest.approx(2.0)
    assert ET.overlap(f["t_rise"], wrong_end, cam0, cam1) == 0.0


def test_censoring_flags():
    t, blk, mask = _arrays()
    blk[1500:] = 1                                           # 2 つ目の OXTS の区間
    f = ET.interval_fields(t, blk, mask, i0=1500, i1=1600, ion=1500, ip=1550)
    assert f["rise_censored"] and f["exceed_start_censored"] and "先頭" in f["exceed_start_censor_reason"]
    f = ET.interval_fields(t, blk, mask, i0=1400, i1=1500, ion=1400, ip=1450)
    assert f["exceed_end_censored"] and "末尾" in f["exceed_end_censor_reason"]
    mask[1699] = False                                       # 判定の対象の外 (方向指示器など) で切れる
    f = ET.interval_fields(t, blk, mask, i0=1700, i1=1800, ion=1650, ip=1750)
    assert f["exceed_start_censored"] and "判定の対象の外" in f["exceed_start_censor_reason"]
    assert not f["rise_censored"]


def test_invalid_positions_rejected():
    t, blk, mask = _arrays()
    with pytest.raises(ValueError):
        ET.interval_fields(t, blk, mask, i0=500, i1=400, ion=300, ip=450)
    with pytest.raises(ValueError):
        ET.interval_fields(t, blk, mask, i0=500, i1=600, ion=650, ip=550)
    blk[550:] = 1
    with pytest.raises(ValueError):
        ET.interval_fields(t, blk, mask, i0=500, i1=600, ion=500, ip=520)


def test_old_rows_are_not_converted():
    """旧い結果 (t_onset と duration_s だけ) を、推定開始 + 長さ で黙って区間にしない。"""
    with pytest.raises(KeyError):
        ET.event_interval(pd.Series(dict(t_onset=100.0, duration_s=3.0)))


def test_relation_to_window():
    assert ET.relation_to_window(0, 1, 2, 3) == "窓の前"
    assert ET.relation_to_window(4, 5, 2, 3) == "窓の後"
    assert ET.relation_to_window(1, 2.5, 2, 3) == "窓の前から続く"
    assert ET.relation_to_window(2.5, 4, 2, 3) == "窓の後まで続く"
    assert ET.relation_to_window(1, 4, 2, 3) == "窓の前から後まで"
    assert ET.relation_to_window(2.2, 2.8, 2, 3) == "窓の中"


# ---------------------------------------------------------------- Phase 2: 検出 → 特徴量
@pytest.fixture(scope="module")
def screening_cfg():
    cfg = yaml.safe_load((ZOD.parents[1] / "configs/zod/full_screening.yaml").read_text())
    return cfg


def _synthetic_drive(n=6000, start_level=0.0):
    """100 Hz・60 s の 1 区間。20〜35 s に、ゆっくり立ち上がる持続的なヨー残差 (閾値の前から高い)。"""
    t = 1_650_000_000.0 + np.arange(n) * 0.01
    x = np.zeros(n)
    rise = (t - t[0] >= 20) & (t - t[0] < 35)
    x[rise] = np.sin(np.linspace(0, np.pi, rise.sum())) * 1.2
    if start_level:
        x[: int(15 * 100)] = start_level                   # 区間の頭から高い (打ち切りの形)
    A = dict(t=t, blk=np.zeros(n, np.int32), v=np.full(n, 30.0, np.float32), r=np.zeros(n, np.float32), r_lp=np.zeros(n, np.float32),
             steer=np.zeros(n, np.float32), torque=np.zeros(n, np.float32), turn=np.zeros(n, np.float32), vd_ok=np.ones(n, bool),
             ox_ay=np.zeros(n, np.float32), vd_ay=np.zeros(n, np.float32), roll=np.zeros(n, np.float32), beta=np.zeros(n, np.float32),
             beta_unc=np.full(n, 0.03, np.float32), pos_std=np.full(n, 0.02, np.float32), hdg_std=np.zeros(n, np.float32),
             lat=np.full(n, 57.7) + np.arange(n) * 2.7e-6, lon=np.full(n, 12.0), heading=np.zeros(n, np.float32),
             beta_block_ok=np.ones(n, bool), quiet=np.ones(n, bool))
    R = dict(dr_cor=x, dr_unc=x, rp_cor=np.zeros(n), rp_unc=np.zeros(n), day_cor=np.zeros(n), dayvd_cor=np.zeros(n), day_unc=np.zeros(n),
             dayvd_unc=np.zeros(n), quiet=A["quiet"])
    return A, R


@pytest.mark.parametrize("start_level", [0.0, 0.9])
def test_screening_event_times_match_detection(screening_cfg, start_level):
    import full_screening as F
    F.CFG = screening_cfg
    A, R = _synthetic_drive(start_level=start_level)
    R["fast_cor"], R["slow_cor"] = F.fast_slow(R["dr_cor"], A["blk"], screening_cfg)
    mask = np.ones(len(A["t"]), bool)
    mask[:200] = False
    mask[-200:] = False
    thr = 0.8
    evs = F.detect(R["slow_cor"], mask, thr, screening_cfg["screen"]["min_dur_slow_s"], screening_cfg["screen"]["merge_s"], A["blk"])
    assert evs, "合成の持続的な残差を検出できない"
    r10 = np.zeros(len(A["t"]))
    for i0, i1 in evs:
        f = F.event_features(A, R, i0, i1, "slow", thr, r10, screening_cfg, "cor", mask=mask)
        # 閾値超過区間は検出の run そのもの
        assert f["t_exceed_start"] == pytest.approx(A["t"][i0])
        assert f["t_exceed_end"] == pytest.approx(A["t"][i1 - 1] + 0.01)
        assert f["exceed_duration_s"] == pytest.approx(f["duration_s"], abs=0.011)
        # 推定開始 ≤ 閾値超過の始まり ≤ ピーク ≤ 閾値超過の終わり
        assert f["t_rise"] <= f["t_exceed_start"] <= f["t_peak"] <= f["t_exceed_end"]
        assert f["t_onset"] == pytest.approx(f["t_rise"])
        if start_level and i0 == 200:                       # 判定の対象の始まり (区間の頭の 2 s の後) から超えている検出
            assert f["rise_censored"], "区間の頭から高い残差は、推定開始を打ち切りとする"
            assert f["exceed_start_censored"], "判定の対象の端から始まる閾値超過は、始まりを打ち切りとする"
    if start_level:
        assert any(i0 == 200 for i0, _ in evs)
        # 閾値超過区間の中の値は、端の近くで閾値を超えている
        assert np.abs(R["slow_cor"][i0]) > thr and np.abs(R["slow_cor"][i1 - 1]) > thr


# ---------------------------------------------------------------- Phase 3: 物理の確かめの区間
def test_weather_physics_window_uses_exceedance():
    import full_weather as FW
    n = 1200                                               # 20 Hz・60 s
    t = 1_650_000_000.0 + np.arange(n) * 0.05
    z = dict(t=t, blk=np.zeros(n, int), v=np.full(n, 30.0), dr_cor=np.zeros(n), r_pred_cor=np.zeros(n), roll=np.zeros(n),
             ox_ay_raw=np.zeros(n), beta=np.zeros(n), beta_unc=np.full(n, 0.03), torque=np.zeros(n), mask80=np.ones(n, bool), mask70=np.ones(n, bool))
    a, b = t[600], t[700]                                  # 閾値超過区間 30〜35 s
    z["dr_cor"][(t >= a) & (t <= b)] = -1.0                # その区間だけ残差がある
    ev = pd.Series(dict(t_rise=t[100], t_onset=t[100], duration_s=5.0, t_exceed_start=a, t_exceed_end=b))
    cfg = yaml.safe_load((ZOD.parents[1] / "configs/zod/full_weather.yaml").read_text())
    ph = FW.event_physics(z, ev, cfg, 1.0)
    assert ph["window_s"] == pytest.approx(5.0, abs=0.06)
    assert ph["mean_dr_deg_s"] == pytest.approx(-1.0, abs=0.05), "残差のある閾値超過区間を見ていない (推定開始 + 長さ の区間を見ている)"


# ---------------------------------------------------------------- 映像: カメラとの重なり
def test_video_overlap_uses_exceedance():
    import video_review as VR
    c0 = 1_650_000_100.0
    cand = pd.DataFrame([dict(event_id="E1", drive_id="d", t_rise=c0 - 20, t_onset=c0 - 20, duration_s=4.0,
                              t_exceed_start=c0 + 1.0, t_exceed_end=c0 + 5.0, t_peak=c0 + 3.0, hypothesis="持続的なずれ")])
    cfg = dict(plot=dict(search_events_s=[-15, 5]))
    ev = VR.sequence_events(pd.Series(dict(drive_id="d")), cand, c0, c0 + 19.8, cfg)
    assert len(ev) == 1
    assert ev.overlap_s.iloc[0] == pytest.approx(4.0)
    assert ev.camera_relation.iloc[0] == "窓の中"
    # カメラの後 (OXTS だけの区間) にある注目の候補も、一覧に残して「窓の後」と出す
    cand2 = cand.assign(event_id="E2", t_exceed_start=c0 + 28, t_exceed_end=c0 + 39, t_peak=c0 + 35)
    ev2 = VR.sequence_events(pd.Series(dict(drive_id="d")), cand2, c0, c0 + 19.8, cfg, focus=["E2"])
    assert ev2.overlap_s.iloc[0] == 0.0 and ev2.camera_relation.iloc[0] == "窓の後"
