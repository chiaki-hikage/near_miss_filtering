import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "vibration"))
sys.path.insert(0, str(REPO / "scripts"))

import compare_fullcar as CF  # noqa: E402
import fullcar as FC  # noqa: E402
import validate_sites_with_fullcar as V  # noqa: E402

PV = yaml.safe_load((REPO / "configs/vibration_physics_validation.yaml").read_text())
CFG = yaml.safe_load((REPO / "configs/vibration.yaml").read_text())
P = FC.build_params(FC.load_config(REPO / "configs/vibration_fullcar.yaml"))


@pytest.mark.parametrize("label,feature,side", [
    ("一致: 補修跡・ひび (右輪の通り道)", "patch", "R"),
    ("一致: 横断方向の舗装継ぎ目 (両輪)", "joint", "B"),
    ("一致: 亀裂の多い荒れた路面 (両輪)", "rough", "B"),
    ("不明瞭", "unknown", "?"),
])
def test_parse_video(label, feature, side):
    d = V.parse_video(label, PV["video_map"])
    assert d["road_feature"] == feature and d["video_side"] == side


def test_shapes_have_expected_sign_and_extent():
    s = np.linspace(0, 10, 10001)
    dip = V.shape_fn(PV["shapes"]["dip"], 0.02, 5.0)(s)
    assert dip.min() == pytest.approx(-0.02, abs=1e-4) and dip[s < 5].max() == 0 and dip[s > 5.5].max() == 0
    step = V.shape_fn(PV["shapes"]["step_up"], 0.02, 5.0)(s)
    assert step[-1] == pytest.approx(0.02)
    sine = V.shape_fn(PV["shapes"]["sine"], 0.02, 5.0)(s)
    assert sine.max() == pytest.approx(0.02, abs=1e-4) and sine.min() == pytest.approx(-0.02, abs=1e-4)


def _side_index(side: str, shape: str, v_kmh: float) -> float:
    v = v_kmh / 3.6
    s0 = v * V.LEAD_S
    f = V.shape_fn(PV["shapes"][shape], 0.01, s0)
    zero = lambda s: np.zeros_like(s)
    left = f if side in ("L", "B") else zero
    right = f if side in ("R", "B") else zero
    m = V.run_model(P, v, left, right, V.LEAD_S + P.wheelbase / v + 3, CFG, PV["bands_hz"]["wheel"])
    t = m["t"] - V.LEAD_S
    return V.metrics(m, t, v, P.wheelbase, CFG, PV)["side_index"]


@pytest.mark.parametrize("shape", ["dip", "bump", "narrow_dip", "step_up"])
@pytest.mark.parametrize("v_kmh", [40, 100])
def test_side_index_sign_does_not_depend_on_shape_or_speed(shape, v_kmh):
    # 右だけの入力でロール角速度と上下速度は逆相、左だけで同相 (凹み/突起・車速によらない)
    r = _side_index("R", shape, v_kmh)
    assert r < -0.5
    assert _side_index("L", shape, v_kmh) == pytest.approx(-r, abs=1e-6)


def test_side_index_is_zero_for_symmetric_input():
    assert abs(_side_index("B", "dip", 80)) < 1e-6


def test_model_response_is_linear_in_height():
    # 比較に使う比・符号・時刻が高さによらないことの前提
    v = 25.0
    s0 = v * V.LEAD_S
    zero = lambda s: np.zeros_like(s)
    a = V.run_model(P, v, zero, V.shape_fn(PV["shapes"]["dip"], 0.01, s0), 4.0, CFG, PV["bands_hz"]["wheel"])
    b = V.run_model(P, v, zero, V.shape_fn(PV["shapes"]["dip"], 0.03, s0), 4.0, CFG, PV["bands_hz"]["wheel"])
    for k in ("heave", "roll", "pitch", "FR"):
        assert np.allclose(b[k], 3 * a[k], rtol=1e-6, atol=1e-9)


def test_rear_wheel_follows_front_by_wheelbase_over_speed():
    v = 20.0
    s0 = v * V.LEAD_S
    zero = lambda s: np.zeros_like(s)
    m = V.run_model(P, v, zero, V.shape_fn(PV["shapes"]["narrow_bump"], 0.01, s0), 4.0, CFG, PV["bands_hz"]["wheel"])
    lag = (np.argmax(np.abs(m["RR"])) - np.argmax(np.abs(m["FR"]))) / CFG["grid_hz"]
    assert lag == pytest.approx(P.wheelbase / v, abs=0.03)
    assert np.abs(m["FL"]).max() < 0.05 * np.abs(m["FR"]).max()


def test_agg_and_overall():
    assert V.agg(["true", "true", "false"]) == "true"
    assert V.agg(["true", "false"]) == "uncertain"
    assert V.agg(["—"]) == "—"
    strong = dict(P1="true", P2="true", P3="true", P4="true", P5="uncertain", P6="false")
    assert V.overall(strong) == "Strong physical consistency"
    assert V.overall(dict(strong, P1="false", P3="false")) == "Not explained"
    assert V.overall(dict(strong, P3="uncertain")) == "Partial consistency"
