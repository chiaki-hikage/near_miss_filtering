"""comma1M の入力層の確認。

localizer の読み出しと、自己運動から正規化信号を作る部分を見る。
通信は行わない。safetensors のヘッダ解釈は手で組み立てたバイト列で確かめる。
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from near_miss.io import comma1m
from near_miss.io.canonical import SegmentRef


def _build_safetensors(tensors: dict[str, np.ndarray]) -> bytes:
    """safetensors のバイト列を組み立てる (テスト用の最小実装)。"""
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype="<f8")
        header[name] = {"dtype": "F64", "shape": list(a.shape),
                        "data_offsets": [offset, offset + a.nbytes]}
        blobs.append(a.tobytes())
        offset += a.nbytes
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"".join(blobs)


def test_parse_safeheader_gives_offsets():
    buf = _build_safetensors({"states": np.zeros((3, 43)), "t": np.zeros(3)})
    hdr = comma1m.parse_safeheader(buf[:4096])
    assert hdr.shape("states") == [3, 43]
    a, b = hdr.offset("states")
    assert b - a == 3 * 43 * 8
    assert buf[a:b] == np.zeros((3, 43), dtype="<f8").tobytes()


def test_parse_safeheader_rejects_truncated_header():
    buf = _build_safetensors({"states": np.zeros((3, 43))})
    with pytest.raises(ValueError):
        comma1m.parse_safeheader(buf[:16])


def test_ecef_to_enu_velocity_at_equator_prime_meridian():
    """緯度 0 / 経度 0 では ECEF の x が Up、y が East、z が North になる。"""
    vel = np.array([[1.0, 2.0, 3.0]])
    enu = comma1m.ecef_to_enu_velocity(vel, np.array([0.0]), np.array([0.0]))
    assert np.allclose(enu[0], [2.0, 3.0, 1.0])


def test_ecef_to_enu_velocity_preserves_norm():
    rng = np.random.default_rng(0)
    vel = rng.normal(size=(50, 3))
    lat = rng.uniform(-80, 80, 50)
    lon = rng.uniform(-180, 180, 50)
    enu = comma1m.ecef_to_enu_velocity(vel, lat, lon)
    assert np.allclose(np.linalg.norm(enu, axis=1), np.linalg.norm(vel, axis=1))


def test_course_rate_left_turn_is_positive():
    """左旋回 (反時計回り) で正になること。"""
    t = np.arange(0, 5, 0.01)
    omega = np.deg2rad(10.0)          # 10 deg/s で左へ回る
    speed = 20.0
    ang = omega * t
    vel = np.column_stack([speed * np.cos(ang), speed * np.sin(ang), np.zeros_like(t)])
    rate, ground = comma1m.course_rate_dps(t, vel, smooth_window_s=0.1, min_speed_mps=2.0)
    mid = slice(100, -100)            # 端は平滑化の影響を受ける
    assert np.allclose(rate[mid], 10.0, atol=0.1)
    assert np.allclose(ground, speed)


def test_course_rate_is_nan_below_min_speed():
    t = np.arange(0, 2, 0.01)
    vel = np.column_stack([np.full_like(t, 0.5), np.zeros_like(t), np.zeros_like(t)])
    rate, _ = comma1m.course_rate_dps(t, vel, smooth_window_s=0.1, min_speed_mps=2.0)
    assert np.isnan(rate).all()


def _fake_localizer(n: int = 500) -> comma1m.Localizer:
    t = np.arange(n) * 0.01
    lat = np.full(n, 40.0)
    lon = np.linspace(-75.0, -74.999, n)
    return comma1m.Localizer(
        segment_id="x", t=t,
        ecef=np.zeros((n, 3)),
        velocity=np.column_stack([np.full(n, 10.0), np.zeros(n), np.zeros(n)]),
        lat=lat, lon=lon, alt=np.full(n, 100.0),
    )


def test_segment_data_channels_and_units():
    loc = _fake_localizer()
    ref = SegmentRef(path=None, dongle_id="", drive_id="x", index=0, dataset="comma1M")
    seg = comma1m.segment_data(loc, ref)
    assert set(seg.channels) == {"speed_mps", "yaw_rate", "lat_deg", "lon_deg", "alt_m"}
    assert seg.channels["speed_mps"].unit == "m/s"
    assert seg.channels["yaw_rate"].unit == "deg/s"
    assert not seg.raw_can_loaded
    # 補助チャネルを足しても解析範囲は主系列だけで決まる
    assert np.isfinite(seg.t_span).all()


def test_segment_data_notes_record_course_rate_caveat():
    """yaw_rate が車両のヨーレートでないことを、必ず記録に残す。"""
    loc = _fake_localizer()
    ref = SegmentRef(path=None, dongle_id="", drive_id="x", index=0, dataset="comma1M")
    seg = comma1m.segment_data(loc, ref)
    assert any("course rate" in n for n in seg.notes)


def test_rate_hz_is_measured_not_assumed():
    loc = _fake_localizer()
    assert loc.rate_hz == pytest.approx(100.0)
