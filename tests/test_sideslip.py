"""横滑り 2 段フィルタの単体試験。

合成信号で「通るべきものが通り、落ちるべきものが落ちる」ことを確かめる。
実データでの再現率は scripts/validate_sideslip_filter.py が KIT MSDM で測る。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from near_miss.config import load_vehicle_configs, load_yaml, DEFAULT_DETECTION, DEFAULT_VEHICLE_DIR
from near_miss.features import sideslip_expected_deg, sideslip_model_deg, sideslip_noise_deg
from near_miss.signals import GriddedSignals
from near_miss.sideslip import (
    find_sideslip_candidates,
    stage1_mask,
    stage2_runs,
)

RATE = 20.0
L_R = 1.5065
K = 0.006573


@pytest.fixture(scope="module")
def cfg():
    return load_yaml(DEFAULT_DETECTION)


def make_gs(n=400, v=20.0, yaw_dps=0.0, ay_ratio=1.0, **cols) -> GriddedSignals:
    """合成の走行を組む。

    横加速度は 2 系統とも v * yaw_rate から作る。ay_ratio は横加速度計が
    v*yaw に対して何倍を読むかで、これが 1 から離れるほど beta が大きくなる
    (定常でない横運動が起きていることに当たる)。実測の中央値は 0.79。
    """
    t = np.arange(n) / RATE
    yaw = np.asarray(yaw_dps, dtype=float) if np.ndim(yaw_dps) else np.full(n, float(yaw_dps))
    speed = np.asarray(v, dtype=float) if np.ndim(v) else np.full(n, float(v))
    ay_kin = speed * np.deg2rad(yaw)
    base = {
        "t": t,
        "v_mps": speed,
        "ax_mps2": np.zeros(n),
        "yaw_rate_dps": yaw,
        "ay_kin_mps2": ay_kin,
        "ay_can_mps2": ay_kin * ay_ratio,
        "yaw_residual_sigma": np.zeros(n),
    }
    for k, val in cols.items():
        base[k] = np.asarray(val, dtype=float) if np.ndim(val) else np.full(n, float(val))
    df = pd.DataFrame(base)
    df["beta_model_deg"] = sideslip_model_deg(
        df.v_mps.to_numpy(), df.yaw_rate_dps.to_numpy(), df.ay_can_mps2.to_numpy(), L_R, K, 3.0
    )
    df["beta_noise_deg"] = sideslip_noise_deg(df.v_mps.to_numpy(), L_R, K, 0.32, 0.21)
    df["beta_sigma"] = df.beta_model_deg / df.beta_noise_deg
    if "beta_excess_deg" not in df:
        # 既定では「舵ではまったく説明できない」ことにする。
        df["beta_excess_deg"] = df["beta_model_deg"]
    return GriddedSignals(
        df=df, rate_hz=RATE, segment_id="synthetic", drive_id="synthetic",
        vehicle="test", raw_can_loaded=False, meta={},
    )


def test_noise_model_falls_with_speed():
    """beta の雑音は 1/v で効く項を持つので、低速ほど大きい。"""
    v = np.array([5.0, 10.0, 30.0])
    s = sideslip_noise_deg(v, L_R, K, 0.32, 0.21)
    assert s[0] > s[1] > s[2] > 0
    # 高速側は横加速度センサの雑音で頭打ちになる
    assert np.isclose(sideslip_noise_deg(np.array([1e6]), L_R, K, 0.32, 0.21)[0],
                      np.degrees(K * 0.21), rtol=1e-6)


def test_expected_sideslip_vanishes_at_critical_speed():
    """v = sqrt(l_r/k) では舵から期待される beta が 0 になる。"""
    v_crit = np.sqrt(L_R / K)
    out = sideslip_expected_deg(np.array([v_crit]), np.array([20.0]), L_R, K, 3.0)
    assert abs(float(out[0])) < 1e-9


def test_stage1_ignores_low_speed(cfg):
    """適用範囲より遅い区間は、beta がいくら大きくても 1 次を通さない。"""
    gs = make_gs(v=4.0, yaw_dps=50.0, ay_ratio=0.6)
    assert abs(gs.df.beta_model_deg.iloc[0]) > 5.0     # beta 自体は大きい
    mask, _ = stage1_mask(gs.df, cfg)
    assert not mask.any()


def test_stage1_passes_large_sideslip(cfg):
    gs = make_gs(v=10.0, yaw_dps=50.0, ay_ratio=0.6)
    assert abs(gs.df.beta_model_deg.iloc[0]) > 5.0
    mask, reasons = stage1_mask(gs.df, cfg)
    assert mask.all()
    assert reasons["beta"].all()


def test_stage1_direct_flag_without_beta(cfg):
    """VSC が立っていれば beta が小さくても 1 次は通す。"""
    gs = make_gs(v=20.0, vsc_active_flag=1.0)
    mask, reasons = stage1_mask(gs.df, cfg)
    assert mask.all() and reasons["flag"].all() and not reasons["beta"].any()


def test_steering_explained_still_passes_when_beta_is_large(cfg):
    """舵で説明できてしまう定常ドリフトでも、beta が大きければ通す。

    KIT の実測で、定常のドリフト中は舵と車両の応答がつり合い、
    本物の横滑りでも beta_excess が 0 付近に落ちることが分かっている。
    ここを必須条件にしていた版はこの形を落としていた。
    """
    gs = make_gs(v=10.0, yaw_dps=50.0, ay_ratio=0.6)
    gs.df["beta_excess_deg"] = 0.0
    mask, _ = stage1_mask(gs.df, cfg)
    passed, _ = stage2_runs(gs.df.t.to_numpy(), gs.df, mask, cfg, RATE)
    assert len(passed) == 1
    assert set(passed[0][2]["items"]) == {"strong_beta", "lateral_force"}


def test_low_lateral_force_still_passes_when_beta_is_large(cfg):
    """横加速度が小さい低速の滑りでも、beta が大きければ通す。

    KIT で 2 件の実測横滑り (v = 4 m/s / |ay_kin| 0.7〜1.1) が
    横力の条件だけで落ちていた。
    """
    gs = make_gs(v=10.0, yaw_dps=50.0, ay_ratio=0.6, ay_kin_mps2=0.0)
    mask, _ = stage1_mask(gs.df, cfg)
    passed, _ = stage2_runs(gs.df.t.to_numpy(), gs.df, mask, cfg, RATE)
    assert len(passed) == 1
    assert set(passed[0][2]["items"]) == {"strong_beta", "unexplained"}


def test_single_confidence_item_is_rejected(cfg):
    """信頼度が 1 項目しか立たない区間は通さない。

    beta は 1 次を通る大きさだが 3 deg には届かず、舵で説明でき、
    急でもない。横力だけが立つ形 = 通常のきつい旋回。
    """
    gs = make_gs(v=10.0, yaw_dps=13.5, ay_ratio=0.6)
    gs.df["beta_excess_deg"] = 0.0
    b = abs(gs.df.beta_model_deg.iloc[0])
    assert cfg["sideslip"]["stage1"]["min_beta_deg"] <= b < cfg["sideslip"]["stage2"]["confidence"]["strong_beta"]["threshold"]
    mask, _ = stage1_mask(gs.df, cfg)
    assert mask.any()
    passed, reject = stage2_runs(gs.df.t.to_numpy(), gs.df, mask, cfg, RATE)
    assert not passed and reject.get("confidence") == 1


def test_transient_counts_as_confidence(cfg):
    """beta の急な立ち上がりは、それ自体が信頼度の 1 項目になる。"""
    gs = make_gs(v=10.0, yaw_dps=13.5, ay_ratio=0.6, ay_kin_mps2=0.0)
    gs.df["beta_excess_deg"] = 0.0
    gs.df["beta_rate_dps"] = 0.0
    mask, _ = stage1_mask(gs.df, cfg)
    assert not stage2_runs(gs.df.t.to_numpy(), gs.df, mask, cfg, RATE)[0]
    gs.df["beta_rate_dps"] = 12.0          # 公道では v>=10 で最大 3.03 deg/s
    passed, _ = stage2_runs(gs.df.t.to_numpy(), gs.df, mask, cfg, RATE)
    assert not passed        # 過渡だけでは 1 項目なので通らない
    gs.df["beta_excess_deg"] = 1.0
    passed, _ = stage2_runs(gs.df.t.to_numpy(), gs.df, mask, cfg, RATE)
    assert len(passed) == 1
    assert set(passed[0][2]["items"]) == {"unexplained", "transient"}


def test_stage2_rejects_inconsistent_lateral_channels(cfg):
    """横加速度 2 系統の符号が逆なら、信号がおかしいので落とす。"""
    gs = make_gs(v=10.0, yaw_dps=50.0, ay_ratio=-0.6)
    mask, _ = stage1_mask(gs.df, cfg)
    passed, reject = stage2_runs(gs.df.t.to_numpy(), gs.df, mask, cfg, RATE)
    assert not passed and reject.get("signal_sane") == 1     # 必須条件のまま


def test_stage2_rejects_short_spike(cfg):
    """1 サンプルだけの尖りは持続時間で落ちる。"""
    n = 400
    yaw = np.zeros(n)
    yaw[100] = 50.0
    gs = make_gs(n=n, v=10.0, yaw_dps=yaw, ay_ratio=0.6)
    mask, _ = stage1_mask(gs.df, cfg)
    assert mask.any()
    passed, reject = stage2_runs(gs.df.t.to_numpy(), gs.df, mask, cfg, RATE)
    assert not passed and reject.get("duration") == 1


def test_full_chain_on_synthetic_slide(cfg):
    """すべての条件を満たす合成の横滑りが最終候補まで残る。"""
    n = 400
    yaw = np.zeros(n)
    yaw[100:180] = 50.0          # 4 s 続くヨー
    gs = make_gs(n=n, v=10.0, yaw_dps=yaw, ay_ratio=0.6)
    cands, counts = find_sideslip_candidates(gs, cfg)
    assert counts.n_samples == n
    assert counts.n_stage1 >= counts.n_stage2 > 0
    assert len(cands) == 1
    c = cands[0]
    assert abs(c.beta_peak_deg) > cfg["sideslip"]["final"]["grade_review_deg"]
    assert c.grade in ("A_大横滑り", "B_要確認")
    assert c.t_start < gs.df.t.iloc[100] and c.t_end > gs.df.t.iloc[179]


def test_counts_are_monotone(cfg):
    """段を下るほどサンプル数が減ることを崩さない。"""
    n = 600
    rng = np.random.default_rng(0)
    yaw = rng.normal(0, 3, n)
    yaw[200:260] = 50.0
    gs = make_gs(n=n, v=10.0, yaw_dps=yaw, ay_ratio=0.6)
    _, counts = find_sideslip_candidates(gs, cfg)
    assert counts.n_samples >= counts.n_beta_valid >= counts.n_in_range
    assert counts.n_in_range >= counts.n_stage1 >= counts.n_stage2


def test_kit_vehicle_config_matches_published_coefficient():
    """KIT の諸元から出す k が、データセット公称の 0.005018 と一致する。"""
    v = [c for c in load_vehicle_configs(DEFAULT_VEHICLE_DIR) if c.name == "kit_msdm"]
    assert v, "configs/vehicles/kit_msdm.yaml がありません"
    assert v[0].sideslip_ay_coeff() == pytest.approx(0.005018, rel=1e-4)
    assert v[0].center_to_rear_m() == pytest.approx(1.437, rel=1e-6)


def test_lat_dynamics_branch_catches_beta_null(cfg):
    """beta が構造的に 0 になる速度でも、横速度の変化があれば 1 次を通す。

    v = sqrt(l_r/k) では beta の第 1 項の係数が消えるので、
    どれだけ滑っていても beta は a_y − v·r のぶんしか出ない。
    """
    v_c = np.sqrt(L_R / K)
    # a_y が v*r から 2.5 m/s² ずれている状態を作る (= 横速度が変化している)
    n = 400
    yaw = np.full(n, 30.0)
    ay_kin = v_c * np.deg2rad(yaw)
    gs = make_gs(n=n, v=v_c, yaw_dps=yaw)
    gs.df["ay_can_mps2"] = ay_kin - 2.5
    gs.df["ay_residual_mps2"] = gs.df.ay_can_mps2 - gs.df.ay_kin_mps2
    gs.df["beta_model_deg"] = sideslip_model_deg(
        gs.df.v_mps.to_numpy(), gs.df.yaw_rate_dps.to_numpy(),
        gs.df.ay_can_mps2.to_numpy(), L_R, K, 3.0,
    )
    # beta の第 1 項が消えているので、残るのは -k*ay_residual だけ
    assert abs(gs.df.beta_model_deg.iloc[0] - np.degrees(K * 2.5)) < 1e-9
    mask, reasons = stage1_mask(gs.df, cfg)
    assert mask.all()
    assert reasons["lat_dynamics"].all()
