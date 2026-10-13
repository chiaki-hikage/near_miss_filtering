import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "vibration"))

import fullcar as FC  # noqa: E402


@pytest.fixture(scope="module")
def p():
    return FC.build_params(FC.load_config(REPO / "configs/vibration_fullcar.yaml"))


def _run(p, key, v=20.0, dur=4.0, **kw):
    t = np.arange(0, dur, 0.001)
    case = FC.road_cases(**kw)[key]
    return FC.simulate(p, t, FC.wheel_inputs(case, p, v, t))


def test_symmetric_step_has_no_roll(p):
    r = _run(p, "D")
    assert np.max(np.abs(r["roll_rate"])) < 1e-6
    assert np.max(np.abs(r["pitch_rate"])) > 0.1


def test_right_side_input_rolls_body(p):
    r = _run(p, "B")
    assert np.max(np.abs(r["roll_rate"])) > 0.5


def test_total_tire_load_equals_weight_at_rest(p):
    r = _run(p, "F", dur=8.0)
    total = r["tire_load"][-1].sum()
    weight = (p.m_s + p.m_u.sum()) * FC.G
    assert total == pytest.approx(weight, rel=1e-4)


def test_steady_roll_gradient_matches_target(p):
    t = np.arange(0, 6, 0.001)
    ay = np.full(len(t), FC.G)            # 左旋回 1 g 相当
    r = FC.simulate(p, t, np.zeros((len(t), 4)), ay=ay)
    # 設定の 5.5 deg/g はサスペンションだけの勾配。タイヤのたわみが直列に加わるぶん大きくなる
    assert r["roll"][-1] > 0
    assert 5.5 <= r["roll"][-1] <= 7.0


def test_braking_pitches_nose_down(p):
    t = np.arange(0, 4, 0.001)
    r = FC.simulate(p, t, np.zeros((len(t), 4)), ax=np.full(len(t), -5.0))
    assert r["pitch"][-1] < 0


def test_modal_frequencies_are_plausible(p):
    m = FC.modes(p)
    f = [x["freq_hz"] for x in m]
    assert 0.9 < f[0] < 1.8
    hop = [x["freq_hz"] for x in m if "ばね下" in x["dominant"]]
    assert all(10.0 < h < 13.5 for h in hop)


def test_suspension_force_is_not_tire_load(p):
    r = _run(p, "B")
    # 接地荷重は静的荷重を含み、サスペンション力 (ばね・ダンパー、つり合いからの変化分) とは別物
    assert np.allclose(r["tire_load"][0], p.static_load)
    assert np.allclose(r["susp_force"][0], 0.0)


def test_unilateral_equals_linear_without_lift_off(p):
    t = np.arange(0, 4, 0.001)
    case = FC.road_cases(height=0.01)["C"]
    zr = FC.wheel_inputs(case, p, 20.0, t)
    a = FC.simulate(p, t, zr, contact="unilateral")
    b = FC.simulate(p, t, zr, contact="linear")
    assert not bool(a["lifted_linear"])
    assert np.allclose(a["imu_heave"], b["imu_heave"])


def test_unilateral_contact_never_pulls(p):
    t = np.arange(0, 4, 0.001)
    case = FC.road_cases(height=0.05)["C"]
    zr = FC.wheel_inputs(case, p, 22.0, t)
    lin = FC.simulate(p, t, zr, contact="linear")
    uni = FC.simulate(p, t, zr, contact="unilateral")
    assert lin["tire_load"].min() < 0
    assert uni["tire_load"].min() >= -1e-6
    assert uni["lift_off"].any()


def test_unilateral_rk4_matches_lsim_when_in_contact(p):
    # 独立した積分法 (RK4) と lsim が、接地したままの入力で一致することを確かめる
    t = np.arange(0, 3, 0.001)
    case = FC.road_cases(height=0.01)["B"]
    zr = FC.wheel_inputs(case, p, 20.0, t)
    sys, m = FC.state_space(p)
    q, _ = FC._integrate_unilateral(p, m, t, zr, np.gradient(zr, 0.001, axis=0), np.zeros((len(t), 2)))
    ref = FC.simulate(p, t, zr, contact="linear")
    assert np.max(np.abs(np.rad2deg(q[:, 2]) - ref["roll"])) < 1e-3 * np.max(np.abs(ref["roll"])) + 1e-9


def test_rough_patch_is_confined_and_scales():
    s = np.arange(0, 40, 0.01)
    h1 = FC.rough_patch(s, 10.0, 15.0, FC.ISO8608_GD0["B"], seed=1)
    h4 = FC.rough_patch(s, 10.0, 15.0, FC.ISO8608_GD0["C"], seed=1)
    assert np.all(h1[(s < 10) | (s > 25)] == 0)
    assert np.std(h4) == pytest.approx(2 * np.std(h1), rel=1e-6)
