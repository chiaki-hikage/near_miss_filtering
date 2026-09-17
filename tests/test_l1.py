"""L1 軽量フィルタの固定条件 (near_miss/l1.py) の単体試験。

確かめるのは 4 点。
  1. 窓の割り方がセグメント先頭起点の floor(t / 2) と同じであること
  2. 各チャネルを**自分のタイムスタンプだけ**で集計すること (時刻同期しない)
  3. 制御・安全系フラグが立った窓は速度・加速度に関係なく通ること
  4. 定常判定が車速の max/min だけで決まること
"""

from __future__ import annotations

import numpy as np
import pytest

from near_miss import l1


class Ch:
    """load_segment が返すチャネルの最小形。"""

    def __init__(self, t, v):
        self.t = np.asarray(t, dtype=float)
        self.v = np.asarray(v, dtype=float)


def const(t0: float, t1: float, value: float, rate: float = 20.0) -> Ch:
    t = np.arange(t0, t1, 1.0 / rate)
    return Ch(t, np.full(t.size, value))


def test_window_edges_starts_at_segment_head():
    e = l1.window_edges(100.0, 106.0)
    assert e is not None
    np.testing.assert_allclose(e[:4], [100.0, 102.0, 104.0, 106.0])


@pytest.mark.parametrize("t0,t1", [(0.0, 0.0), (5.0, 1.0), (np.nan, 10.0)])
def test_window_edges_rejects_bad_span(t0, t1):
    assert l1.window_edges(t0, t1) is None


def test_window_minmax_bins_by_own_timestamp():
    edges = np.array([0.0, 2.0, 4.0])
    t = np.array([0.5, 1.5, 2.5, 3.5])
    v = np.array([1.0, -3.0, 7.0, 2.0])
    lo, hi = l1.window_minmax(t, v, edges)
    np.testing.assert_allclose(lo, [-3.0, 2.0])
    np.testing.assert_allclose(hi, [1.0, 7.0])


def test_window_minmax_empty_window_is_nan():
    edges = np.array([0.0, 2.0, 4.0])
    lo, hi = l1.window_minmax(np.array([0.5]), np.array([1.0]), edges)
    assert np.isnan(lo[1]) and np.isnan(hi[1])


def test_channels_are_not_time_synced():
    """ax と速度のサンプル時刻がずれていても、窓の中で揃っていれば通る。"""
    edges = l1.window_edges(0.0, 2.0)
    ch = {"accel_x": Ch([0.1], [-4.0]),       # 窓の先頭寄り
          "speed_mps": Ch([1.9], [20.0])}     # 窓の末尾寄り
    _, tot = l1.window_pass(l1.window_stats(ch, edges))
    assert tot[0]


def test_base_condition_needs_speed_gate():
    edges = l1.window_edges(0.0, 2.0)
    slow = {"accel_x": const(0.0, 2.0, -4.0), "speed_mps": const(0.0, 2.0, 2.0)}
    fast = {"accel_x": const(0.0, 2.0, -4.0), "speed_mps": const(0.0, 2.0, 20.0)}
    assert not l1.window_pass(l1.window_stats(slow, edges))[1][0]
    assert l1.window_pass(l1.window_stats(fast, edges))[1][0]


def test_lateral_branch_uses_absolute_value():
    edges = l1.window_edges(0.0, 2.0)
    for sign in (+1.0, -1.0):
        ch = {"accel_y": const(0.0, 2.0, sign * 2.5), "speed_mps": const(0.0, 2.0, 20.0)}
        assert l1.window_pass(l1.window_stats(ch, edges))[1][0]


def test_below_threshold_does_not_pass():
    edges = l1.window_edges(0.0, 2.0)
    ch = {"accel_x": const(0.0, 2.0, -2.0), "accel_y": const(0.0, 2.0, 1.0),
          "speed_mps": const(0.0, 2.0, 20.0)}
    assert not l1.window_pass(l1.window_stats(ch, edges))[1][0]


@pytest.mark.parametrize("flag", l1.PASS_FLAGS)
def test_control_flag_passes_unconditionally(flag):
    """停車中でも加速度が小さくても、制御が作動した窓は落とさない。"""
    edges = l1.window_edges(0.0, 2.0)
    ch = {"speed_mps": const(0.0, 2.0, 0.0), "accel_x": const(0.0, 2.0, 0.0),
          flag: const(0.0, 2.0, 1.0)}
    kin, tot = l1.window_pass(l1.window_stats(ch, edges))
    assert not kin[0] and tot[0]


def test_slip_warn_lamp_test_does_not_pass():
    """3 は始動時のランプ試験。4 だけが実作動。"""
    edges = l1.window_edges(0.0, 2.0)
    base = {"speed_mps": const(0.0, 2.0, 0.0)}
    assert not l1.window_pass(l1.window_stats(
        {**base, "slip_warn": const(0.0, 2.0, 3.0)}, edges))[1][0]
    assert l1.window_pass(l1.window_stats(
        {**base, "slip_warn": const(0.0, 2.0, 4.0)}, edges))[1][0]
    # 7 は継続点灯。車種で意味が違うので通過条件には使わない。
    assert not l1.window_pass(l1.window_stats(
        {**base, "slip_warn": const(0.0, 2.0, 7.0)}, edges))[1][0]


def test_watch_flags_are_not_pass_flags():
    """tc_disabled は「切られている」、abs_fault は「故障」で、どちらも作動ではない。"""
    for f in l1.WATCH_FLAGS:
        assert f not in l1.PASS_FLAGS


def test_missing_channels_do_not_pass():
    """判定できないことを「条件を満たす」にしない。"""
    edges = l1.window_edges(0.0, 2.0)
    kin, tot = l1.window_pass(l1.window_stats({}, edges))
    assert not kin.any() and not tot.any()


def test_steady_mask_flags_constant_speed():
    edges = l1.window_edges(0.0, 20.0)
    st = l1.window_stats({"speed_mps": const(0.0, 20.0, 25.0)}, edges)
    assert l1.steady_mask(st).all()


def test_steady_mask_clears_on_speed_change():
    edges = l1.window_edges(0.0, 20.0)
    t = np.arange(0.0, 20.0, 0.05)
    ch = {"speed_mps": Ch(t, np.linspace(5.0, 25.0, t.size))}
    assert not l1.steady_mask(l1.window_stats(ch, edges)).any()


def test_base_is_the_koatsushuku_condition():
    """基本候補が高圧縮のまま固定されていること。"""
    assert l1.BASE_NAME == "高圧縮"
    assert l1.BASE == {"ax": -2.5, "ay": 2.0, "v_min": 5.0}
    assert set(l1.ALTERNATIVES) == {"標準", "1段階厳"}
