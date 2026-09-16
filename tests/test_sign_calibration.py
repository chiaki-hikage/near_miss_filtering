"""加速度センサの符号の判定と補正 (sign_calibration.py) の試験。

確かめること
  1. 独立な基準 (Δv, v·yaw) との相関で +1 / -1 / 不明 が決まる
  2. 確信が持てないとき (相関が弱い・対が少ない) は不明にする
  3. 補正は反転・除去・何もしない を表どおりに行い、何をしたかを残す
  4. 符号表の中身が変われば config_hash も変わる
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from near_miss.config import DEFAULT_DETECTION, VehicleConfig, config_hash, load_yaml
from near_miss.io.canonical import Channel
from near_miss.sign_calibration import (
    STATUS_FLIPPED,
    STATUS_OK,
    STATUS_UNKNOWN,
    VehicleSignStats,
    apply_accel_signs,
    decide,
    load_sign_table,
    segment_pairs,
)


@pytest.fixture(scope="module")
def cfg():
    return load_yaml(DEFAULT_DETECTION)["accel_sign"]


def drive(sign_ax: float = 1.0, sign_ay: float = 1.0, noise_only: bool = False,
          seconds: float = 120.0, seed: int = 0) -> dict[str, Channel]:
    """加減速と旋回を繰り返す合成走行。加速度センサに符号を掛けて返す。"""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, 0.02)
    a_true = 1.5 * np.sin(2 * np.pi * t / 17.0)
    v = 15.0 + np.cumsum(a_true) * 0.02
    yaw = 8.0 * np.sin(2 * np.pi * t / 11.0)                 # deg/s
    ay_true = v * np.deg2rad(yaw)
    if noise_only:
        ax_meas = rng.normal(0, 1.0, t.size)
        ay_meas = rng.normal(0, 1.0, t.size)
    else:
        ax_meas = sign_ax * a_true + rng.normal(0, 0.1, t.size)
        ay_meas = sign_ay * ay_true + rng.normal(0, 0.1, t.size)
    return {
        "speed_mps": Channel(t, v, "m/s", "continuous"),
        "yaw_rate": Channel(t, yaw, "deg/s", "continuous"),
        "accel_x": Channel(t, ax_meas, "m/s^2", "continuous"),
        "accel_y": Channel(t, ay_meas, "m/s^2", "continuous"),
    }


def stats_for(channels: dict[str, Channel], cfg, n_segments: int = 1) -> dict[str, Any]:
    st = VehicleSignStats(dongle_id="d")
    for _ in range(n_segments):
        st.add(segment_pairs(channels, cfg), vin="JTDXXXXXXXX")
    return decide(st, cfg)


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------
def test_正しい向きは_plus1(cfg):
    row = stats_for(drive(1, 1), cfg)
    assert row["accel_x_sign"] == 1 and row["accel_x_status"] == STATUS_OK
    assert row["accel_y_sign"] == 1 and row["accel_y_status"] == STATUS_OK


def test_逆向きは_minus1(cfg):
    row = stats_for(drive(-1, -1), cfg)
    assert row["accel_x_sign"] == -1 and row["accel_x_status"] == STATUS_FLIPPED
    assert row["accel_y_sign"] == -1 and row["accel_y_status"] == STATUS_FLIPPED


def test_軸ごとに独立に判定する(cfg):
    """実データでは片方だけ逆の車両は無かったが、判定は軸ごとに行う。"""
    row = stats_for(drive(1, -1), cfg)
    assert (row["accel_x_sign"], row["accel_y_sign"]) == (1, -1)


def test_相関が弱ければ不明(cfg):
    row = stats_for(drive(noise_only=True), cfg)
    assert row["accel_x_sign"] == 0 and row["accel_x_status"] == STATUS_UNKNOWN
    assert row["accel_y_sign"] == 0


def test_対が少なければ不明(cfg):
    """短い走行 1 本では、相関が強くても符号を決めない。"""
    row = stats_for(drive(-1, -1, seconds=20.0), cfg)
    assert row["accel_x_sign"] == 0
    assert row["accel_x_pairs"] < cfg["min_pairs_ax"]


def test_停止中は判定に使わない(cfg):
    ch = drive(1, 1)
    ch["speed_mps"] = Channel(ch["speed_mps"].t, np.zeros_like(ch["speed_mps"].v), "m/s", "continuous")
    p = segment_pairs(ch, cfg)
    assert p["accel_x"].shape[0] == 0 and p["accel_y"].shape[0] == 0


# ---------------------------------------------------------------------------
# 補正
# ---------------------------------------------------------------------------
@dataclass
class _Ref:
    dongle_id: str


@dataclass
class _Seg:
    ref: _Ref
    channels: dict[str, Channel]
    notes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Vehicle:
    raw: dict[str, Any]


@pytest.fixture()
def table(tmp_path):
    p = tmp_path / "signs.csv"
    pd.DataFrame([
        {"dongle_id": "aaaa", "accel_x_sign": 1, "accel_y_sign": 1},
        {"dongle_id": "bbbb", "accel_x_sign": -1, "accel_y_sign": -1},
        {"dongle_id": "cccc", "accel_x_sign": 0, "accel_y_sign": 1},
    ]).to_csv(p, index=False)
    load_sign_table.cache_clear()
    return p


def _seg(dongle: str) -> _Seg:
    t = np.arange(5.0)
    return _Seg(_Ref(dongle), {"accel_x": Channel(t, t + 1, "m/s^2", "continuous"),
                               "accel_y": Channel(t, -t, "m/s^2", "continuous"),
                               "yaw_rate": Channel(t, t, "deg/s", "continuous")})


def test_反転する車両は値に符号を掛ける(table):
    s = _seg("bbbb")
    apply_accel_signs(s, _Vehicle({"calibration": {"accel_sign": str(table)}}), Path("/"))
    assert np.array_equal(s.channels["accel_x"].v, -(np.arange(5.0) + 1))
    assert np.array_equal(s.channels["accel_y"].v, np.arange(5.0))
    assert np.array_equal(s.channels["yaw_rate"].v, np.arange(5.0))   # 他のチャネルは触らない
    assert s.meta["accel_sign"]["status"] == "applied"
    assert "accel_sign:accel_x:flipped" in s.notes


def test_正しい向きの車両は変えない(table):
    s = _seg("aaaa")
    apply_accel_signs(s, _Vehicle({"calibration": {"accel_sign": str(table)}}), Path("/"))
    assert np.array_equal(s.channels["accel_x"].v, np.arange(5.0) + 1)
    assert not [n for n in s.notes if "flipped" in n]


def test_不明の軸はチャネルを取り除く(table):
    """NaN にせず取り除く。後段は列の有無でその特徴量を作るかを決める。"""
    s = _seg("cccc")
    apply_accel_signs(s, _Vehicle({"calibration": {"accel_sign": str(table)}}), Path("/"))
    assert "accel_x" not in s.channels
    assert "accel_y" in s.channels
    assert "accel_sign:accel_x:unknown:dropped" in s.notes


def test_表に無い車両は既定で加速度を使わない(table):
    s = _seg("zzzz")
    apply_accel_signs(s, _Vehicle({"calibration": {"accel_sign": str(table)}}), Path("/"))
    assert "accel_x" not in s.channels and "accel_y" not in s.channels
    assert s.meta["accel_sign"]["status"] == "missing:drop"


def test_表に無い車両を残す設定もできる(table):
    s = _seg("zzzz")
    apply_accel_signs(s, _Vehicle({"calibration": {"accel_sign": str(table),
                                                   "on_missing": "keep"}}), Path("/"))
    assert "accel_x" in s.channels
    assert "accel_sign:missing_vehicle:kept_uncorrected" in s.notes


def test_設定が無ければ何もしない():
    s = _seg("bbbb")
    apply_accel_signs(s, _Vehicle({}), Path("/"))
    assert np.array_equal(s.channels["accel_x"].v, np.arange(5.0) + 1)
    assert "accel_sign" not in s.meta


def test_表が見つからなければ止める(tmp_path):
    s = _seg("bbbb")
    with pytest.raises(FileNotFoundError):
        apply_accel_signs(s, _Vehicle({"calibration": {"accel_sign": str(tmp_path / "no.csv")}}),
                          Path("/"))


# ---------------------------------------------------------------------------
# 追跡性
# ---------------------------------------------------------------------------
def test_符号表を作り直すと_config_hash_が変わる(tmp_path):
    table = tmp_path / "signs.csv"
    yml = tmp_path / "v.yaml"
    yml.write_text(f"name: v\nsignals: {{}}\ncalibration:\n  accel_sign: {table}\n", encoding="utf-8")

    table.write_text("dongle_id,accel_x_sign,accel_y_sign\naaaa,1,1\n", encoding="utf-8")
    h1 = config_hash(VehicleConfig.load(yml).raw)
    table.write_text("dongle_id,accel_x_sign,accel_y_sign\naaaa,-1,-1\n", encoding="utf-8")
    h2 = config_hash(VehicleConfig.load(yml).raw)
    assert h1 != h2


# ---------------------------------------------------------------------------
# 2 段目 (もう一方の軸が決まっているときの弱い証拠)
# ---------------------------------------------------------------------------
def _d(rx, ry, nx=5000, ny=40000):
    from near_miss.sign_calibration import decide_from_corr
    c = load_yaml(DEFAULT_DETECTION)["accel_sign"]
    return decide_from_corr({"accel_x": rx, "accel_y": ry}, {"accel_x": nx, "accel_y": ny}, c)


def test_弱い軸も同じ向きの証拠があれば決める():
    """Highlander で実際にあった形: 前後 -0.8 / 横 -0.45。"""
    r = _d(-0.8, -0.45)
    assert r["accel_x"][0] == -1 and r["accel_y"][0] == -1


def test_弱い軸が逆向きなら決めない():
    r = _d(-0.8, 0.45)
    assert r["accel_x"][0] == -1
    assert r["accel_y"][0] == 0


def test_弱い軸の証拠が薄すぎれば決めない():
    """もう一方が決まっていても、この軸自体に |r| >= 0.3 の証拠が要る。"""
    r = _d(-0.8, -0.1)
    assert r["accel_y"][0] == 0


def test_両方弱ければどちらも決めない():
    r = _d(-0.45, -0.45)
    assert r["accel_x"][0] == 0 and r["accel_y"][0] == 0


def test_弱い軸でも対が足りなければ決めない():
    r = _d(-0.8, -0.45, ny=10)
    assert r["accel_y"][0] == 0
