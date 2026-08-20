"""再サンプル層の検証。"""

import numpy as np

from near_miss.signals import (
    build_grid,
    moving_average,
    resample_continuous,
    resample_flag,
    resample_occupancy,
)


def test_grid_is_uniform_and_trimmed():
    g = build_grid(100.0, 160.0, rate_hz=20.0, edge_trim_s=0.5)
    assert g[0] >= 100.5 and g[-1] <= 159.5
    assert np.allclose(np.diff(g), 0.05)


def test_continuous_resample_marks_gaps_as_nan():
    # 2.0 秒の欠測を挟んだ時系列
    t = np.array([0.0, 0.1, 0.2, 2.2, 2.3, 2.4])
    v = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    grid = np.arange(0.0, 2.45, 0.05)
    out = resample_continuous(t, v, grid, max_gap_s=0.5)
    assert np.isfinite(out[grid <= 0.2]).all()
    assert np.isnan(out[(grid > 0.25) & (grid < 2.15)]).all()


def test_flag_uses_zero_order_hold():
    t = np.array([0.0, 1.0, 2.0])
    v = np.array([0.0, 1.0, 0.0])
    grid = np.array([0.5, 1.5, 2.5])
    out = resample_flag(t, v, grid, max_gap_s=1.0)
    # 内挿されて 0.5 のような中間値にならないこと
    assert set(np.unique(out[np.isfinite(out)])) <= {0.0, 1.0}
    assert out[1] == 1.0


def test_occupancy_marks_bins_with_any_event():
    grid = np.arange(0.0, 1.0, 0.1)
    t = np.array([0.35, 0.36, 0.81])
    out = resample_occupancy(t, grid)
    assert out[3] == 1.0 and out[8] == 1.0
    assert out.sum() == 2.0


def test_moving_average_keeps_nan_at_edges_only():
    x = np.arange(100.0)
    out = moving_average(x, 5)
    assert np.isnan(out[:2]).all() and np.isnan(out[-2:]).all()
    assert np.allclose(out[2:-2], x[2:-2])


def test_resample_then_differentiate_is_robust_to_timestamp_jitter():
    """実データで見つかった問題の再現。

    時刻が最小 0.2 ms まで詰まる列を生の時刻で微分すると加速度が発散するが、
    一様グリッドへ載せてから微分すれば妥当な値に収まる。
    """
    rng = np.random.default_rng(0)
    n = 5000
    dt = np.clip(rng.normal(0.012, 0.006, n), 0.0002, None)
    t = np.cumsum(dt)
    v_true = 20.0 + 0.5 * t                      # 一定加速 0.5 m/s^2
    v = np.round(v_true / (0.01 / 3.6)) * (0.01 / 3.6)  # 0.01 km/h 量子化

    naive = np.gradient(v, t)
    # 真値 0.5 m/s^2 に対し、生の時刻で微分すると 10 m/s^2 を超える値が出る
    assert np.abs(naive).max() > 10.0

    grid = np.arange(t[0] + 1.0, t[-1] - 1.0, 0.05)
    v_grid = resample_continuous(t, v, grid, max_gap_s=0.5)
    a = np.gradient(moving_average(v_grid, 5), grid)
    assert np.nanmax(np.abs(a - 0.5)) < 0.5      # グリッド化すれば真値近傍に収まる


def test_frame_mask_separates_buses_and_drops_tx():
    """同じアドレスが複数バスに別内容で流れる場合に取り違えないこと。

    実測で 0x224 はバス 0 が BRAKE_MODULE、バス 1 は先頭バイトがカウンタの
    別メッセージだった。混ぜるとフラグがチャタリングする。
    """
    from near_miss.io.can_decode import frame_mask

    address = np.array([548, 548, 548, 548, 36])
    src = np.array([0, 1, 0x80, 0x81, 0])     # bus0受信 / bus1受信 / bus0送信 / bus1送信 / bus0受信

    m = frame_mask(address, src, can_id=548, bus=0)
    assert list(m) == [True, False, False, False, False]

    m = frame_mask(address, src, can_id=548, bus=None)
    assert list(m) == [True, True, False, False, False]   # 送信は既定で除く

    m = frame_mask(address, src, can_id=548, bus=0, include_tx=True)
    assert list(m) == [True, False, True, False, False]

    # src が無ければアドレスだけで選ぶ
    assert list(frame_mask(address, None, can_id=548, bus=0)) == [True, True, True, True, False]
