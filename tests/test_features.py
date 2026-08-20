"""特徴量層の検証。"""

import numpy as np

from near_miss.features import (
    count_reversals,
    derivative,
    lead_from_tracks,
    time_headway,
    time_to_collision,
)


def test_derivative_on_uniform_grid():
    t = np.arange(0.0, 10.0, 0.05)
    x = 3.0 * t
    d = derivative(x, 0.05)
    assert np.allclose(d[1:-1], 3.0)
    assert np.isnan(d[0]) and np.isnan(d[-1])


def test_ttc_is_defined_only_when_approaching():
    d = np.array([50.0, 50.0, 50.0])
    vrel = np.array([-10.0, 0.0, 5.0])
    ttc = time_to_collision(d, vrel)
    assert ttc[0] == 5.0
    assert np.isnan(ttc[1]) and np.isnan(ttc[2])


def test_thw_not_computed_below_min_speed():
    d = np.array([20.0, 20.0])
    v = np.array([2.0, 20.0])
    thw = time_headway(d, v, min_ego_speed=5.0)
    assert np.isnan(thw[0])
    assert thw[1] == 1.0


LEAD_CFG = {
    "lane_half_width_m": 1.8,
    "min_distance_m": 1.0,
    "max_distance_m": 150.0,
    "max_hold_s": 0.5,
    "min_target_speed_mps": 2.0,
    "min_target_speed_ratio": 0.2,
    "curvature_compensation": False,
    "curvature_min_speed_mps": 3.0,
}


def test_lead_picks_nearest_in_lane_track():
    grid = np.arange(0.0, 1.0, 0.1)
    # 3 トラック: 自車レーン 30m / 自車レーン 60m / 隣レーン 10m
    tt = np.repeat(grid, 3)
    dist = np.tile(np.array([30.0, 60.0, 10.0]), grid.size)
    lat = np.tile(np.array([0.5, -0.3, 3.5]), grid.size)
    vrel = np.tile(np.array([-1.0, -2.0, -9.0]), grid.size)
    ego = np.full(grid.shape, 25.0)
    d, v, ts, _ = lead_from_tracks(grid, tt, dist, lat, vrel, LEAD_CFG, ego_v=ego)
    assert np.allclose(d, 30.0)      # 隣レーンの 10m は選ばない
    assert np.allclose(v, -1.0)
    assert np.allclose(ts, 24.0)     # 先行車の絶対速度 = 25 - 1


def test_lead_reports_track_id():
    """割り込みの検出にトラック ID が要る。"""
    grid = np.arange(0.0, 1.0, 0.1)
    tt = np.repeat(grid, 2)
    dist = np.tile(np.array([30.0, 60.0]), grid.size)
    lat = np.tile(np.array([0.5, -0.3]), grid.size)
    vrel = np.tile(np.array([-1.0, -2.0]), grid.size)
    tid = np.tile(np.array([531.0, 528.0]), grid.size)
    ego = np.full(grid.shape, 25.0)
    d, _, _, ids = lead_from_tracks(grid, tt, dist, lat, vrel, LEAD_CFG, ego_v=ego, track_id=tid)
    assert np.allclose(d, 30.0)
    assert np.allclose(ids, 531.0)      # 選ばれたトラックの ID が返る


def test_stationary_object_is_not_a_lead():
    """路側の静止物を先行車にしない。

    静止物は相対速度が -自車速 になり、TTC = 距離 / 自車速 で恒常的に
    小さい値を出す。Chunk_1 では low_ttc の 25 件すべてがこれだった。
    """
    grid = np.arange(0.0, 1.0, 0.1)
    tt = np.repeat(grid, 2)
    ego_speed = 25.0
    # 20m 先の静止物 (相対速度 = -自車速) と 50m 先の実在の先行車
    dist = np.tile(np.array([20.0, 50.0]), grid.size)
    lat = np.tile(np.array([0.3, -0.2]), grid.size)
    vrel = np.tile(np.array([-ego_speed, -1.0]), grid.size)
    ego = np.full(grid.shape, ego_speed)

    d, v, ts, _ = lead_from_tracks(grid, tt, dist, lat, vrel, LEAD_CFG, ego_v=ego)
    assert np.allclose(d, 50.0)      # 静止物ではなく実在の先行車を選ぶ
    assert np.allclose(ts, 24.0)

    # 足切りを外すと、より近い静止物のほうを選んでしまう
    off = dict(LEAD_CFG, min_target_speed_mps=0.0, min_target_speed_ratio=0.0)
    d_off, _, _, _ = lead_from_tracks(grid, tt, dist, lat, vrel, off, ego_v=ego)
    assert np.allclose(d_off, 20.0)


def test_curvature_compensation_follows_the_turn():
    """旋回中は帯を進路の曲がりに合わせてずらす。

    左旋回では、自車レーンの先行車は車両座標で左へずれて見える。
    補正が無いと帯から外れ、代わりに路側が帯に入る。
    """
    grid = np.arange(0.0, 1.0, 0.1)
    ego_speed, yaw = 25.0, 6.0            # 6 deg/s の左旋回
    dist = np.full(grid.size, 50.0)
    # 50m 先で進路が横にずれる量
    kappa = np.deg2rad(yaw) / ego_speed
    y_path = 0.5 * kappa * 50.0 ** 2
    lat = np.full(grid.size, y_path)      # 進路上にいる先行車
    vrel = np.full(grid.size, -1.0)
    ego = np.full(grid.shape, ego_speed)
    yaws = np.full(grid.shape, yaw)

    assert abs(y_path) > 1.8              # 補正なしでは帯の外
    d_off, _, _, _ = lead_from_tracks(grid, grid, dist, lat, vrel, LEAD_CFG,
                                      ego_v=ego, ego_yaw_dps=yaws)
    assert np.isnan(d_off).all()

    on = dict(LEAD_CFG, curvature_compensation=True)
    d_on, _, _, _ = lead_from_tracks(grid, grid, dist, lat, vrel, on,
                                     ego_v=ego, ego_yaw_dps=yaws)
    assert np.allclose(d_on, 50.0)


def test_reversal_count_ignores_small_oscillation():
    t = np.arange(0.0, 20.0, 0.05)
    small = 0.5 * np.sin(2 * np.pi * t)          # 振幅 0.5 deg
    large = 5.0 * np.sin(2 * np.pi * t / 4.0)    # 振幅 5 deg, 4 秒周期
    assert count_reversals(t, small, amplitude=3.0, window_s=10.0).max() == 0
    assert count_reversals(t, large, amplitude=3.0, window_s=10.0).max() >= 4


def test_ttc_not_computed_when_ego_stopped():
    """自車停止中は TTC を出さない。

    停止中に相対速度が負になるのは相手が近づいている状態で、
    自車側のヒヤリハットではない。Chunk_1 で v ≈ 0 の誤検出として出た。
    """
    d = np.array([20.0, 20.0, 20.0])
    vrel = np.array([-5.0, -5.0, -5.0])
    ego = np.array([0.0, 3.0, 25.0])
    ttc = time_to_collision(d, vrel, ego, min_ego_speed=5.0)
    assert np.isnan(ttc[0]) and np.isnan(ttc[1])
    assert ttc[2] == 4.0


def test_rolling_window_smooths_single_sample_spikes():
    """躍度のような 1 サンプルの跳ねを、補助特徴量では均すこと。"""
    from near_miss.features import rolling

    x = np.zeros(41)
    x[20] = -10.0                      # 単発スパイク
    assert rolling(x, 7, "min")[20] == -10.0        # 最小値はスパイクを残す
    assert abs(rolling(x, 7, "mean")[20] + 10.0 / 7) < 1e-9   # 平均は薄まる
    x2 = np.full(41, -1.0)
    x2[18:23] = -6.0                   # 0.25 秒相当の持続的な落ち込み
    assert rolling(x2, 7, "mean")[20] < -4.0        # 持続していれば平均も落ちる


LANE_CHANGE_CFG = {
    "lane_change": {
        "curvature_window_s": 10.0,
        "enter_dps": 1.0,
        "lobe_merge_gap_s": 0.5,
        "pair_max_gap_s": 2.0,
        "min_lobe_heading_deg": 1.5,
        "max_net_heading_deg": 2.5,
        "min_duration_s": 1.0,
        "max_duration_s": 8.0,
        "min_offset_m": 2.0,
        "max_offset_m": 6.0,
        "min_speed_mps": 10.0,
    }
}


def _s_shape(t, t0, amp_dps, period_s):
    """t0 から 1 周期ぶんの正弦。左右一組の振れになる。"""
    y = np.zeros_like(t)
    m = (t >= t0) & (t < t0 + period_s)
    y[m] = amp_dps * np.sin(2 * np.pi * (t[m] - t0) / period_s)
    return y


def test_lane_change_found_for_s_shaped_yaw():
    """左右一組の振りで、方位角が元へ戻り、横変位が 1 車線ぶんなら候補になる。"""
    from near_miss.features import find_lane_changes

    t = np.arange(0.0, 40.0, 0.05)
    v = np.full_like(t, 30.0)
    yaw = _s_shape(t, 15.0, 3.2, 4.0)
    got = find_lane_changes(t, v, yaw, 20.0, LANE_CHANGE_CFG)
    assert len(got) == 1
    assert 2.0 <= abs(got[0].offset_m) <= 6.0
    assert abs(got[0].net_heading_deg) < 2.5      # 向きが戻っている


def test_steady_curve_is_not_a_lane_change():
    """定常カーブは方位角が戻らないので候補にしない。"""
    from near_miss.features import find_lane_changes

    t = np.arange(0.0, 40.0, 0.05)
    v = np.full_like(t, 30.0)
    yaw = np.where((t > 10.0) & (t < 20.0), 4.0, 0.0)
    assert find_lane_changes(t, v, yaw, 20.0, LANE_CHANGE_CFG) == []


def test_small_wobble_is_not_a_lane_change():
    """車線内のふらつきは横変位が足りないので候補にしない。"""
    from near_miss.features import find_lane_changes

    t = np.arange(0.0, 40.0, 0.05)
    v = np.full_like(t, 30.0)
    yaw = _s_shape(t, 15.0, 1.2, 2.0)
    assert find_lane_changes(t, v, yaw, 20.0, LANE_CHANGE_CFG) == []


def test_lane_change_needs_minimum_speed():
    """低速の切り返しは対象外。"""
    from near_miss.features import find_lane_changes

    t = np.arange(0.0, 40.0, 0.05)
    yaw = _s_shape(t, 15.0, 12.0, 4.0)
    assert find_lane_changes(t, np.full_like(t, 5.0), yaw, 20.0, LANE_CHANGE_CFG) == []


def test_lane_change_during_a_curve_is_still_found():
    """カーブ中の車線変更を、カーブの曲率と切り分けて拾えること。

    曲率の推定に移動平均を使うと、カーブの出入りで偏差に左右一組の振れが出て、
    カーブ 1 つにつき 2 件の誤検出になる。中央値フィルタはこれを起こさない。
    """
    from near_miss.features import find_lane_changes

    t = np.arange(0.0, 40.0, 0.05)
    v = np.full_like(t, 30.0)
    curve = np.where((t > 5.0) & (t < 35.0), 4.0, 0.0).astype(float)
    got = find_lane_changes(t, v, curve + _s_shape(t, 15.0, 3.2, 4.0), 20.0, LANE_CHANGE_CFG)
    assert len(got) == 1
    assert 2.0 <= abs(got[0].offset_m) <= 6.0


CUT_IN_CFG = {
    "cut_in": {
        "min_jump_m": 5.0,
        "persist_s": 0.8,
        "min_distance_drop_m": 10.0,
        "max_appear_distance_m": 40.0,
        "max_thw_after_s": 0.8,
        "min_speed_mps": 8.0,
        "window_s": 3.0,
    }
}


def _cut_in_inputs(dist, ids, ego=30.0):
    t = np.arange(0.0, 10.0, 0.05)
    d = np.asarray(dist, dtype=float)
    v = np.full_like(t, ego)
    thw = d / v
    target = np.full_like(t, ego - 1.0)
    return t, np.asarray(ids, dtype=float), d, thw, target, v


def test_cut_in_found_when_a_closer_vehicle_takes_over():
    """遠い先行車から近い車両へ、距離が跳んで入れ替わる。"""
    from near_miss.features import find_cut_ins

    t = np.arange(0.0, 10.0, 0.05)
    i = 100
    d = np.where(np.arange(len(t)) < i, 80.0, 20.0)
    ids = np.where(np.arange(len(t)) < i, 528.0, 531.0)
    args = _cut_in_inputs(d, ids)
    got = find_cut_ins(*args[:1], args[1], args[2], args[3], args[4], args[5], 20.0, CUT_IN_CFG)
    assert len(got) == 1
    assert got[0].i_switch == i
    assert got[0].distance_before_m == 80.0 and got[0].distance_after_m == 20.0


def test_same_vehicle_decelerating_is_not_a_cut_in():
    """同じ車両が減速して車間が詰まっただけなら割り込みではない。

    距離は連続的に縮み、トラック ID も変わらない。
    """
    from near_miss.features import find_cut_ins

    t = np.arange(0.0, 10.0, 0.05)
    d = np.clip(80.0 - 8.0 * t, 15.0, None)      # 連続的に接近
    ids = np.full(len(t), 528.0)
    args = _cut_in_inputs(d, ids)
    assert find_cut_ins(args[0], args[1], args[2], args[3], args[4], args[5], 20.0, CUT_IN_CFG) == []


def test_cut_in_needs_short_headway_afterwards():
    """入れ替わっても車間が空いたままなら拾わない。"""
    from near_miss.features import find_cut_ins

    t = np.arange(0.0, 10.0, 0.05)
    i = 100
    d = np.where(np.arange(len(t)) < i, 120.0, 60.0)   # 60m / 30(m/s) = THW 2.0s
    ids = np.where(np.arange(len(t)) < i, 528.0, 531.0)
    args = _cut_in_inputs(d, ids)
    assert find_cut_ins(args[0], args[1], args[2], args[3], args[4], args[5], 20.0, CUT_IN_CFG) == []


def test_cut_in_ignores_track_id_churn_without_distance_jump():
    """ID だけが入れ替わっても、距離が跳ばなければ割り込みではない。

    レーダはトラックの枠を使い回すので、同じ車両でも ID が頻繁に変わる
    (実測で 14 秒間に 28〜92 回)。ID の変化だけを条件にはできない。
    """
    from near_miss.features import find_cut_ins

    t = np.arange(0.0, 10.0, 0.05)
    d = np.full(len(t), 20.0)
    ids = np.where(np.arange(len(t)) % 2 == 0, 528.0, 531.0)   # 毎サンプル入れ替わる
    args = _cut_in_inputs(d, ids)
    assert find_cut_ins(args[0], args[1], args[2], args[3], args[4], args[5], 20.0, CUT_IN_CFG) == []


WEAVING_CFG = {
    "weaving": {
        "curvature_window_s": 10.0,
        "enter_dps": 1.0,
        "lobe_merge_gap_s": 0.3,
        "lobe_gap_s": 2.0,
        "min_reversals": 3,
        "min_lateral_accel_mps2": 1.0,
        "min_steer_rate_dps": 8.0,
        "min_duration_s": 2.0,
        "max_duration_s": 8.0,
        "max_net_heading_deg": 8.0,
        "min_speed_mps": 10.0,
    }
}


def _weave_inputs(t, yaw, v=25.0):
    ay = v * np.deg2rad(yaw)
    steer = np.gradient(yaw, t) * 3.0        # 舵角レートの代わり
    return np.full_like(t, v), yaw, ay, steer


def test_weaving_found_for_repeated_alternating_swings():
    """短時間に左右交互の振れが 3 回以上あれば蛇行。"""
    from near_miss.features import find_weaving

    t = np.arange(0.0, 30.0, 0.05)
    yaw = np.zeros_like(t)
    m = (t >= 10.0) & (t < 16.0)
    yaw[m] = 5.0 * np.sin(2 * np.pi * (t[m] - 10.0) / 4.0)    # 1.5 周期 = 3 振れ
    v, yaw, ay, steer = _weave_inputs(t, yaw)
    got = find_weaving(t, v, yaw, ay, steer, 20.0, WEAVING_CFG)
    assert len(got) == 1 and got[0][2] >= 3


def test_two_lane_changes_in_the_same_direction_are_not_weaving():
    """同じ向きへ続けて車線変更しただけなら蛇行にしない。

    以前の「舵角の符号反転回数」だけの定義はこれを蛇行として拾っていた。
    """
    from near_miss.features import find_weaving

    t = np.arange(0.0, 30.0, 0.05)
    yaw = np.zeros_like(t)
    for t0 in (10.0, 14.0):
        m = (t >= t0) & (t < t0 + 3.0)
        yaw[m] = 5.0 * np.sin(2 * np.pi * (t[m] - t0) / 3.0)   # 同じ向きの S 字を 2 回
    v, yaw, ay, steer = _weave_inputs(t, yaw)
    got = find_weaving(t, v, yaw, ay, steer, 20.0, WEAVING_CFG)
    assert all(abs(g[3]) <= WEAVING_CFG["weaving"]["max_net_heading_deg"] for g in got)


def test_weaving_needs_lateral_response():
    """舵角が動いても横加速度が伴わなければ蛇行としない (低速のふらつき)。"""
    from near_miss.features import find_weaving

    t = np.arange(0.0, 30.0, 0.05)
    yaw = np.zeros_like(t)
    m = (t >= 10.0) & (t < 16.0)
    yaw[m] = 5.0 * np.sin(2 * np.pi * (t[m] - 10.0) / 4.0)
    v, yaw, ay, steer = _weave_inputs(t, yaw, v=3.0)      # 低速なので横加速度が小さい
    assert find_weaving(t, np.full_like(t, 3.0), yaw, ay, steer, 20.0, WEAVING_CFG) == []


def test_net_heading_separates_turn_from_swerve():
    """左折は正味方位角が大きく、元へ戻る動きは小さい。"""
    from near_miss.features import rolling_net_heading

    t = np.arange(0.0, 30.0, 0.05)
    turn = np.where((t > 10.0) & (t < 18.0), 11.0, 0.0)       # 8 秒で約 88 deg
    swerve = np.zeros_like(t)
    m = (t >= 10.0) & (t < 14.0)
    swerve[m] = 5.0 * np.sin(2 * np.pi * (t[m] - 10.0) / 4.0)

    i = int(14.0 / 0.05)
    assert rolling_net_heading(t, turn, 3.0)[i] > 50.0
    assert rolling_net_heading(t, swerve, 3.0)[i] < 5.0


def test_bicycle_model_reproduces_steady_turn():
    """定常旋回で、舵角から期待されるヨーレートが出ること。"""
    from near_miss.features import bicycle_yaw_rate

    v = np.full(10, 25.0)
    sr, wb, kus = 16.75, 2.66, 0.00235
    steer = np.full(10, 20.0)
    got = bicycle_yaw_rate(v, steer, wb, sr, kus, steer_offset_deg=0.0)
    expected = np.rad2deg(25.0 * np.deg2rad(20.0) / (sr * (wb + kus * 25.0 ** 2)))
    assert np.allclose(got, expected)
    # 同じ舵角でも速度が上がるほどヨーレートは頭打ちになる (アンダーステア)
    fast = bicycle_yaw_rate(np.full(10, 35.0), steer, wb, sr, kus)
    assert fast[0] / got[0] < 35.0 / 25.0


def test_bicycle_model_not_computed_at_low_speed():
    from near_miss.features import bicycle_yaw_rate

    got = bicycle_yaw_rate(np.full(5, 1.0), np.full(5, 20.0), 2.66, 16.75, min_speed_mps=3.0)
    assert np.isnan(got).all()


def test_counter_steer_needs_opposite_signs_and_magnitude():
    from near_miss.features import counter_steer

    yaw = np.array([5.0, 5.0, 5.0, 1.0, -5.0])
    rate = np.array([-20.0, 20.0, -5.0, -20.0, -20.0])
    got = counter_steer(yaw, rate, min_yaw_dps=3.0, min_steer_rate_dps=15.0)
    assert list(got) == [1.0, 0.0, 0.0, 0.0, 0.0]
    # 1: 旋回中に逆向き / 2: 同じ向き / 3: 舵角レート不足 / 4: ヨー不足 / 5: 同じ向き


S_EVASION_CFG = {
    "s_evasion": {
        "curvature_window_s": 10.0,
        "enter_dps": 1.0,
        "lobe_merge_gap_s": 0.3,
        "lobe_gap_s": 1.5,
        "min_lobes": 3,
        "min_duration_s": 1.0,
        "max_duration_s": 5.0,
        "min_lateral_accel_mps2": 2.0,
        "min_excursion_m": 1.2,
        "max_return_m": 1.0,
        "min_speed_mps": 10.0,
    }
}


def _yaw_from_heading(t, t0, amp_rad, period_s):
    """方位角の軌跡を与えて、その微分をヨーレート [deg/s] として返す。

    回避は方位角が + と − の両方へ振れる (出て、戻る)。
    方位角が + のままなら横へ移り続けるので、それは車線変更にあたる。
    """
    psi = np.zeros_like(t)
    m = (t >= t0) & (t < t0 + period_s)
    psi[m] = amp_rad * np.sin(2 * np.pi * (t[m] - t0) / period_s)
    return np.rad2deg(np.gradient(psi, t))


def test_s_evasion_found_when_the_path_returns():
    """横へ出て元へ戻る動きを回避として拾う。"""
    from near_miss.features import find_s_evasions

    t = np.arange(0.0, 30.0, 0.05)
    v = np.full_like(t, 28.0)
    yaw = _yaw_from_heading(t, 12.0, 0.08, 2.5)
    ay = v * np.deg2rad(yaw)
    got = find_s_evasions(t, v, yaw, ay, 20.0, S_EVASION_CFG)
    assert len(got) == 1
    assert got[0][2] >= 1.2          # 途中でふくらむ
    assert abs(got[0][3]) <= 1.0     # 最後は元へ戻る


def test_lane_change_is_not_an_s_evasion():
    """車線変更は横へ移って留まるので回避にはしない。

    ヨーレートが 1 周期ぶん振れると方位角は + のまま戻るので、
    車体は横へ移った位置に留まる。回避との違いはここにある。
    """
    from near_miss.features import find_s_evasions

    t = np.arange(0.0, 30.0, 0.05)
    v = np.full_like(t, 28.0)
    yaw = np.zeros_like(t)
    m = (t >= 12.0) & (t < 16.0)
    yaw[m] = 4.0 * np.sin(2 * np.pi * (t[m] - 12.0) / 4.0)
    ay = v * np.deg2rad(yaw)
    assert find_s_evasions(t, v, yaw, ay, 20.0, S_EVASION_CFG) == []
