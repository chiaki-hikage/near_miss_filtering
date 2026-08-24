"""commaCarSegments 側の入力層と、データセット共通化まわりの検証。"""

import numpy as np
import pytest

from near_miss.config import RadarSpec, SignalSpec, VehicleConfig, unit_scale
from near_miss.io.can_decode import build_derived, decode_channels, extract_bits
from near_miss.io.can_radar import radar_from_can
from near_miss.io.canonical import Channel, RawCanFrames, SegmentData, SegmentRef
from near_miss.io.comma_car_segments import SegmentName, _longest_run
from near_miss.io.rlog import _payloads_to_u64
from near_miss.signals import build_grid


# ---------------------------------------------------------------------------
# セグメント名
# ---------------------------------------------------------------------------
def test_segment_name_parses_database_entry():
    sn = SegmentName.parse("d0e756c10c7adf7a/00000004--a4e9cb8af7/36/s")
    assert sn.dongle_id == "d0e756c10c7adf7a"
    assert sn.route_id == "00000004--a4e9cb8af7"
    assert sn.index == 36
    # comma2k19 の '<dongle>|<日時>' と同じ形にそろえて、下流を共通化できるようにする
    assert sn.drive_id == "d0e756c10c7adf7a|00000004--a4e9cb8af7"
    assert sn.url.endswith("/segments/d0e756c10c7adf7a/00000004--a4e9cb8af7/36/rlog.zst")


def test_segment_name_rejects_unknown_form():
    with pytest.raises(ValueError):
        SegmentName.parse("not-a-segment")


def test_longest_run_picks_consecutive_block():
    names = [f"a" * 16 + f"/route/{i}/s" for i in (1, 2, 3, 7, 8, 20)]
    run = _longest_run(names)
    assert [SegmentName.parse(n).index for n in run] == [1, 2, 3]


# ---------------------------------------------------------------------------
# ペイロードの詰め方
# ---------------------------------------------------------------------------
def test_short_payload_is_left_aligned():
    """8 バイト未満のメッセージは後ろを 0 で埋める。

    DBC の Motorola 表記は先頭バイトを byte 0 として数えるので、
    右詰めにすると全ビット位置がずれる。
    """
    u = _payloads_to_u64([bytes([0xAB, 0xCD])])
    assert u[0] == 0xABCD000000000000
    # 先頭バイトの上位 8 bit が start_bit 7 から取れること
    assert extract_bits(u, 7, 8, False)[0] == 0xAB


# ---------------------------------------------------------------------------
# 単位換算と派生チャネル
# ---------------------------------------------------------------------------
def test_unit_scale_converts_kph_to_mps():
    assert unit_scale("km/h", "m/s") == pytest.approx(1 / 3.6)
    assert unit_scale("m/s", "m/s") == 1.0
    with pytest.raises(ValueError):
        unit_scale("furlong/fortnight", "m/s")


def _u64(values: dict[tuple[int, int], int]) -> int:
    """(start_bit, length) -> 値 から Motorola 表記でペイロードを組み立てる。"""
    out = 0
    for (start_bit, length), v in values.items():
        byte_idx, bit_in_byte = divmod(start_bit, 8)
        msb = byte_idx * 8 + (7 - bit_in_byte)
        shift = 64 - msb - length
        out |= (int(v) & ((1 << length) - 1)) << shift
    return out


def _vehicle(signals: dict, derived: dict | None = None, radar: dict | None = None) -> VehicleConfig:
    return VehicleConfig.from_dict(
        {"name": "test", "signals": signals, "derived": derived or {}, "radar": radar}
    )


def test_decode_applies_unit_conversion():
    """DBC の factor/offset をそのまま書き、単位換算は unit_in/unit で行う。"""
    payload = _u64({(47, 16): 3600})  # 0.01 km/h 刻みで 36.00 km/h
    raw = RawCanFrames(
        t=np.array([0.0]), address=np.array([180]),
        payload_u64=np.array([payload], dtype=np.uint64), src=np.array([0]),
    )
    veh = _vehicle({
        "SPEED": {"can_id": 180, "start_bit": 47, "length": 16, "factor": 0.01,
                  "bus": 0, "unit_in": "km/h", "unit": "m/s", "channel": "speed_mps"}
    })
    ch, notes = decode_channels(raw, veh)
    assert ch["speed_mps"].v[0] == pytest.approx(10.0)   # 36 km/h = 10 m/s
    assert ch["speed_mps"].unit == "m/s"
    assert notes == []


def test_derived_sum_and_mean():
    ch = {
        "a": Channel(t=np.array([0.0, 1.0]), v=np.array([1.0, 2.0]), unit="deg", kind="continuous"),
        "b": Channel(t=np.array([0.0, 1.0]), v=np.array([0.1, 0.2]), unit="deg", kind="continuous"),
        "c": Channel(t=np.array([0.0, 1.0]), v=np.array([3.0, 4.0]), unit="deg", kind="continuous"),
    }
    veh = _vehicle({}, derived={"s": {"sum": ["a", "b"]}, "m": {"mean": ["a", "c"]}})
    out, notes = build_derived(ch, veh)
    assert out["s"].v == pytest.approx([1.1, 2.2])
    assert out["m"].v == pytest.approx([2.0, 3.0])
    assert notes == []


def test_derived_reports_missing_part():
    veh = _vehicle({}, derived={"s": {"sum": ["a", "missing"]}})
    out, notes = build_derived(
        {"a": Channel(np.array([0.0]), np.array([1.0]), "deg", "continuous")}, veh
    )
    assert out == {}
    assert any("missing" in n for n in notes)


# ---------------------------------------------------------------------------
# レーダ
# ---------------------------------------------------------------------------
RADAR_CFG = {
    "bus": 1, "track_first_id": 0x180, "track_count": 2, "lateral_sign": -1.0,
    "max_distance_m": 254.0,
    "signals": {
        "long_dist": {"start_bit": 15, "length": 15, "signed": False, "factor": 0.01},
        "lat_dist": {"start_bit": 31, "length": 11, "signed": True, "factor": 0.04},
        "rel_speed": {"start_bit": 47, "length": 12, "signed": True, "factor": 0.025},
        "new_track": {"start_bit": 36, "length": 1},
        "valid": {"start_bit": 48, "length": 1},
    },
    "score": {"first_id": 0x190, "signal": {"start_bit": 23, "length": 8}, "min_score": 50.0},
}


def _radar_frames(rows):
    """rows = [(t, address, payload_u64)] から RawCanFrames を作る。"""
    return RawCanFrames(
        t=np.array([r[0] for r in rows], dtype=float),
        address=np.array([r[1] for r in rows], dtype=np.int64),
        payload_u64=np.array([r[2] for r in rows], dtype=np.uint64),
        src=np.array([1] * len(rows), dtype=np.int64),
    )


def test_radar_decodes_track_and_flips_lateral_sign():
    """DBC の LAT_DIST は右が正。正規化では左を正にそろえる。"""
    a = _u64({(15, 15): 4000, (31, 11): 50, (47, 12): -80 & 0xFFF, (48, 1): 1})
    raw = _radar_frames([(0.0, 0x180, a)])
    tracks, notes = radar_from_can(raw, RadarSpec.from_dict(RADAR_CFG))
    assert tracks.distance_m[0] == pytest.approx(40.0)
    assert tracks.lateral_m[0] == pytest.approx(-2.0)   # DBC 上は右 +2.0 m
    assert tracks.vrel_mps[0] == pytest.approx(-2.0)    # 負が接近
    assert tracks.track_id[0] == 0x180


def test_radar_keeps_invalid_point_only_when_score_is_high():
    """VALID が落ちていても、スコアが閾値を超えていれば残す (openpilot と同じ扱い)。"""
    body = {(15, 15): 4000, (31, 11): 0, (47, 12): 0, (48, 1): 0}
    raw = _radar_frames([
        (0.0, 0x190, _u64({(23, 8): 10})),    # スコア低
        (0.0, 0x180, _u64(body)),
        (1.0, 0x191, _u64({(23, 8): 90})),    # スコア高
        (1.0, 0x181, _u64(body)),
    ])
    tracks, _ = radar_from_can(raw, RadarSpec.from_dict(RADAR_CFG))
    assert tracks.track_id.tolist() == [0x181]


def test_radar_ignores_other_buses():
    a = _u64({(15, 15): 4000, (48, 1): 1})
    raw = _radar_frames([(0.0, 0x180, a)])
    raw.src[:] = 0     # レーダは バス 1。バス 0 の同一アドレスは別物
    tracks, notes = radar_from_can(raw, RadarSpec.from_dict(RADAR_CFG))
    assert tracks is None
    assert any("no_frames" in n for n in notes)


# ---------------------------------------------------------------------------
# 解析窓の安定性
# ---------------------------------------------------------------------------
def _seg(channels: dict[str, Channel]) -> SegmentData:
    ref = SegmentRef(path=None, dongle_id="x", drive_id="d", index=0)
    return SegmentData(ref=ref, vehicle="test", channels=channels)


def test_t_span_ignores_auxiliary_channels():
    """付随信号を 1 本足しただけで解析窓が動かないこと。

    アクセル開度のような信号は車種や年式で受信の始まりが数十 ms ずれる。
    そこに解析窓を合わせると、信号を足すだけで検出結果が変わってしまう。
    """
    primary = {
        "speed_mps": Channel(np.arange(0.0, 10.0, 0.1), np.zeros(100), "m/s", "continuous"),
        "yaw_rate": Channel(np.arange(0.0, 10.0, 0.1), np.zeros(100), "deg/s", "continuous"),
    }
    before = _seg(dict(primary)).t_span
    with_aux = dict(primary)
    with_aux["gas_pedal_pct"] = Channel(np.arange(2.0, 8.0, 0.1), np.zeros(60), "%", "continuous")
    assert _seg(with_aux).t_span == before


def test_grid_phase_is_fixed_to_absolute_time():
    """開始が少し動いてもサンプル位置がずれないこと。"""
    g1 = build_grid(100.000, 110.0, 20.0, 0.6)
    g2 = build_grid(100.028, 110.0, 20.0, 0.6)
    common = np.intersect1d(np.round(g1, 6), np.round(g2, 6))
    # 端の 1 点を除いてすべて共通のはず
    assert len(common) >= len(g2) - 1
    assert np.allclose(np.round(g1 * 20.0) / 20.0, g1)


# ---------------------------------------------------------------------------
# 輪速の異常
# ---------------------------------------------------------------------------
from near_miss.features import (  # noqa: E402
    find_pedal_panic_brakes,
    find_wheel_speed_anomalies,
    wheel_speed_excess,
)

WS_CFG = {
    "wheel_speed": {
        "min_excess_mps": 0.4,
        "min_speed_mps": 5.0,
        "tolerance_s": 0.5,
        "corroborate_ax_mps2": 1.5,
        "corroborate_yaw_sigma": 3.0,
    }
}


def test_steady_cornering_does_not_look_like_slip():
    """定常旋回では外輪と内輪の差が開くが、これは幾何で説明できる。"""
    yaw = np.full(200, 10.0)                      # deg/s
    track = 1.636
    spread = np.abs(np.deg2rad(yaw)) * track      # 幾何どおりのばらつき
    excess, expected = wheel_speed_excess(spread, yaw, track)
    assert np.allclose(expected, spread)
    assert np.abs(excess).max() < 1e-9


def test_wheel_speed_anomaly_needs_corroboration():
    """超過だけでは発火しない。加減速・ヨー乖離・ABS のどれかが要る。"""
    n = 200
    t = np.arange(n) / 20.0
    excess = np.zeros(n)
    excess[100:120] = 1.0                          # 1 秒間の大きな超過
    v = np.full(n, 25.0)
    quiet = find_wheel_speed_anomalies(t, excess, v, 20.0, WS_CFG)
    assert quiet.max() == 0.0

    ax = np.zeros(n)
    ax[105:115] = -3.0                             # 強い制動が重なる
    supported = find_wheel_speed_anomalies(t, excess, v, 20.0, WS_CFG, ax_mps2=ax)
    assert supported[100:120].max() == 1.0


def test_wheel_speed_anomaly_is_not_triggered_by_single_sample_noise():
    """1 サンプルだけの尖りは拾わない。

    生の輪速は 1 サンプルごとの符号反転率が 62〜67% (白色雑音なら 50%) あり、
    平滑化しないと 0.4 m/s を超える尖りが常時出る。滑りは 1 サンプルでは終わらない。
    """
    rng = np.random.default_rng(0)
    n = 400
    t = np.arange(n) / 20.0
    excess = rng.normal(0.0, 0.05, n)
    excess[::37] = 1.2                             # 単発のスパイクを散らす
    v = np.full(n, 25.0)
    ax = np.full(n, -2.0)                          # 裏付けは常に立っている状態
    active = find_wheel_speed_anomalies(t, excess, v, 20.0, WS_CFG, ax_mps2=ax)
    # 発火はするが、いずれも 1 サンプルで終わる。継続時間の条件で落ちること
    runs = np.diff(np.r_[0, active, 0])
    lengths = np.flatnonzero(runs == -1) - np.flatnonzero(runs == 1)
    assert lengths.max() <= 1


def test_wheel_speed_anomaly_ignores_low_speed():
    n = 200
    t = np.arange(n) / 20.0
    excess = np.full(n, 1.0)
    ax = np.full(n, -3.0)
    v = np.full(n, 2.0)
    assert find_wheel_speed_anomalies(t, excess, v, 20.0, WS_CFG, ax_mps2=ax).max() == 0.0


# ---------------------------------------------------------------------------
# アクセル急 OFF からの制動
# ---------------------------------------------------------------------------
PB_CFG = {
    "panic_brake": {
        "pedal": {
            "min_gas_pct": 5.0,
            "released_pct": 1.0,
            "release_window_s": 0.5,
            "reaction_window_s": 1.5,
            "min_brake_rise": 8.0,
            "brake_threshold": -2.5,
        }
    }
}


def _pedal_case(gas_before, ax_after, brake_rise):
    n = 200
    t = np.arange(n) / 20.0
    gas = np.zeros(n)
    gas[:100] = gas_before
    ax = np.zeros(n)
    ax[105:125] = ax_after
    brake = np.zeros(n)
    brake[105:125] = brake_rise
    return t, gas, ax, brake


def test_pedal_panic_brake_fires_on_release_then_brake():
    t, gas, ax, brake = _pedal_case(30.0, -4.0, 20.0)
    out = find_pedal_panic_brakes(t, gas, ax, 20.0, PB_CFG, brake_level=brake)
    assert out[100:130].max() == 1.0


def test_pedal_panic_brake_ignores_gentle_deceleration():
    """アクセルを離しただけで減速が弱いものは拾わない (engine brake)。"""
    t, gas, ax, brake = _pedal_case(30.0, -0.8, 20.0)
    assert find_pedal_panic_brakes(t, gas, ax, 20.0, PB_CFG, brake_level=brake).max() == 0.0


def test_pedal_panic_brake_ignores_braking_without_release():
    """もともと踏んでいなければ「急に戻した」ではない。"""
    t, gas, ax, brake = _pedal_case(0.0, -4.0, 20.0)
    assert find_pedal_panic_brakes(t, gas, ax, 20.0, PB_CFG, brake_level=brake).max() == 0.0


# --- 定常単軌道モデルによる横滑り角 -----------------------------------------

def test_sideslip_model_reproduces_a_steady_turn():
    """定常円旋回を作って、式が定義どおりの値を返すことを見る。"""
    from near_miss.features import sideslip_model_deg

    l_r, k = 1.5065, 0.006573
    v = np.full(10, 20.0)
    yaw_dps = np.full(10, 10.0)
    a_y = np.full(10, 3.5)
    got = sideslip_model_deg(v, yaw_dps, a_y, l_r, k, min_speed_mps=3.0)
    want = np.degrees(l_r * np.deg2rad(10.0) / 20.0 - k * 3.5)
    assert np.allclose(got, want)


def test_sideslip_model_is_nan_below_min_speed():
    """第 1 項が発散する低速では NaN。埋めない。"""
    from near_miss.features import sideslip_model_deg

    v = np.array([0.0, 1.0, 2.9, 3.0, 10.0])
    out = sideslip_model_deg(v, np.full(5, 5.0), np.zeros(5), 1.5, 0.0065, min_speed_mps=3.0)
    assert np.isnan(out[:3]).all()
    assert np.isfinite(out[3:]).all()


def test_sideslip_model_sign_left_turn():
    """左旋回 (ヨーレート正・横加速度正) では、幾何の項と横力の項が逆に効く。"""
    from near_miss.features import sideslip_model_deg

    v, yaw, ay = np.array([20.0]), np.array([10.0]), np.array([3.5])
    geom_only = sideslip_model_deg(v, yaw, ay, 1.5, 0.0, min_speed_mps=3.0)
    with_tire = sideslip_model_deg(v, yaw, ay, 1.5, 0.0065, min_speed_mps=3.0)
    assert geom_only[0] > 0
    assert with_tire[0] < geom_only[0]


def test_sideslip_model_column_appears_only_with_geometry(monkeypatch):
    """諸元が無い車種では列を作らない。"""
    from near_miss.config import VehicleConfig

    bare = VehicleConfig.from_dict({"name": "unknown", "geometry": {}})
    assert bare.sideslip_ay_coeff() is None
    assert bare.center_to_rear_m() is None
