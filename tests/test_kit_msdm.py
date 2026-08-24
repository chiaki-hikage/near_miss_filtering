"""KIT Multi-Surface Driving Maneuvers の読み出しの確認。

実データを置いていない環境でも回るように、MAT ファイルそのものは読まない。
バイト列の走査と幾何変換、parameter.m の解釈だけを見る。
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from near_miss.io import kit_msdm as kit


def _element(tag: int, payload: bytes) -> bytes:
    """MAT-file の 1 要素 (8 バイトのタグ + 8 バイト境界に揃えた本体)。"""
    pad = (-len(payload)) % 8
    return struct.pack("<II", tag, len(payload)) + payload + b"\x00" * pad


def test_iter_double_arrays_descends_into_matrix():
    """miMATRIX は入れ物なので中へ降り、長い double だけを拾う。"""
    long_a = np.arange(2000, dtype="<f8")
    long_b = (np.arange(2000, dtype="<f8") * -1.5)
    short = np.arange(3, dtype="<f8")            # 短いので拾わない
    buf = (
        _element(kit.MI_MATRIX, b"")             # 中身へ降りる
        + _element(kit.MI_DOUBLE, long_a.tobytes())
        + _element(kit.MI_DOUBLE, short.tobytes())
        + _element(kit.MI_DOUBLE, long_b.tobytes())
    )
    got = kit._iter_double_arrays(buf, min_bytes=8000)
    assert len(got) == 2
    assert np.array_equal(got[0], long_a)
    assert np.array_equal(got[1], long_b)


def test_iter_double_arrays_skips_other_types():
    other = np.arange(4000, dtype="<u4")
    buf = _element(6, other.tobytes()) + _element(kit.MI_DOUBLE, np.ones(2000).tobytes())
    got = kit._iter_double_arrays(buf, min_bytes=8000)
    assert len(got) == 1


def test_translate_velocity_pure_rotation():
    """原点で静止して回っている剛体は、前方 r_x の点で横速度 yaw_rate*r_x を持つ。"""
    n = 5
    zeros = np.zeros(n)
    yaw = np.full(n, 2.0)                        # rad/s、左回りが正
    vx, vy = kit.translate_velocity(zeros, zeros, yaw, np.array([1.5, 0.0, 0.0]))
    assert np.allclose(vx, 0.0)
    assert np.allclose(vy, 3.0)


def test_translate_velocity_lateral_offset_affects_longitudinal():
    n = 3
    yaw = np.full(n, 1.0)
    vx, vy = kit.translate_velocity(np.zeros(n), np.zeros(n), yaw, np.array([0.0, 2.0, 0.0]))
    assert np.allclose(vx, -2.0)
    assert np.allclose(vy, 0.0)


def test_load_parameters(tmp_path):
    p = tmp_path / "parameter.m"
    p.write_text(
        "params.vehicle.l = 3;      % Wheelbase in m\n"
        "params.vehicle.tw = 1.65;\n"
        "params.vehicle.m = 2254;\n"
        "params.vehicle.l_f = 1.563;\n"
        "params.vehicle.l_r = 1.437;\n"
        "params.tire.C_f = 234008;\n"
        "params.tire.C_r = 234008;\n"
        "params.tf.trvec_cor_ra = [0.958, -0.159, 0];\n"
        "params.mu.asphalt_a = 1.1;\n",
        encoding="utf-8",
    )
    par = kit.load_parameters(p)
    assert par.wheelbase_m == 3.0
    assert par.mass_kg == 2254.0
    assert par.mu("asphalt_a") == 1.1
    assert par.mu("ice") is None
    assert np.allclose(par.trvec("cor_ra"), [0.958, -0.159, 0.0])


def test_understeer_gradient_sign_matches_weight_distribution():
    """重心が後ろ寄り (l_f > l_r) で前後の剛性が同じならオーバーステア傾向 (Kus < 0)。"""
    par = kit.VehicleParams(raw={
        "vehicle.l": 3.0, "vehicle.m": 2254.0,
        "vehicle.l_f": 1.563, "vehicle.l_r": 1.437,
        "tire.C_f": 234008.0, "tire.C_r": 234008.0,
    })
    assert par.understeer_gradient() < 0


def test_classify_run_names():
    assert kit._classify("dynamic_driving_asphalt_a_1") == ("dynamic", "asphalt_a")
    assert kit._classify("dynamic_driving_asphalt_a_cobble_1") == ("dynamic", "asphalt_a+cobble")
    # 綴りが揺れている 1 件も dynamic として拾う
    assert kit._classify("dynmic_asphalt_b_cobble_1")[0] == "dynamic"
    assert kit._classify("standstill_plates_2") == ("standstill", "plates")
    assert kit._classify("parking_cobble_1") == ("parking", "cobble")


def test_sideslip_masks_low_speed():
    n = 100
    vx = np.linspace(0.0, 10.0, n)
    run = kit.Run(
        name="x", path=None, t=np.arange(n) / kit.RATE_HZ,
        channels={"v_x_cor_mps": vx, "v_y_cor_mps": np.full(n, 1.0),
                  "w_z_cor_radps": np.zeros(n)},
        surface="asphalt_a", kind="dynamic",
    )
    beta = kit.sideslip_deg(run, min_speed_mps=2.0)
    assert np.isnan(beta[vx < 2.0]).all()
    assert np.isfinite(beta[vx >= 2.0]).all()
    # 左向きの横速度は正の beta
    assert (beta[vx >= 2.0] > 0).all()


def test_sideslip_requires_params_for_other_points():
    run = kit.Run(name="x", path=None, t=np.zeros(2),
                  channels={"v_x_cor_mps": np.ones(2), "v_y_cor_mps": np.zeros(2),
                            "w_z_cor_radps": np.zeros(2)},
                  surface="asphalt_a", kind="dynamic")
    with pytest.raises(ValueError):
        kit.sideslip_deg(run, at="cog")


def test_sideslip_coefficient_matches_kit_published_parameters():
    """k = l_f*Kus/(l_r-l_f) が、KIT の公称諸元から出る m*l_f/(C_r*L) と一致すること。

    質量とコーナリング剛性を知らない車種でも、当てはめ済みの Kus だけで
    横滑り角の式を組み立てられる、という前提の裏取り。
    """
    from near_miss.config import VehicleConfig

    par = kit.VehicleParams(raw={
        "vehicle.l": 3.0, "vehicle.m": 2254.0,
        "vehicle.l_f": 1.563, "vehicle.l_r": 1.437,
        "tire.C_f": 234008.0, "tire.C_r": 234008.0,
    })
    k_published = par.mass_kg * par.l_f / (par.c_r * par.wheelbase_m)

    veh = VehicleConfig.from_dict({
        "name": "kit_ioniq5",
        "geometry": {
            "wheelbase_m": 3.0,
            "center_to_front_m": 1.563,
            "understeer_gradient": par.understeer_gradient(),
        },
    })
    assert veh.center_to_rear_m() == pytest.approx(1.437)
    assert veh.sideslip_ay_coeff() == pytest.approx(k_published, rel=1e-3)
