"""検出層とスコア付けの検証。"""

import numpy as np

from near_miss.detectors import detect_threshold_events


def _spec(**kw):
    base = {
        "label": "test",
        "feature": "x",
        "op": "lt",
        "threshold": -3.0,
        "min_duration_s": 0.3,
        "merge_gap_s": 1.0,
        "stage": 1,
        "weight": 1.0,
    }
    base.update(kw)
    return base


def test_short_excursion_is_rejected():
    t = np.arange(0.0, 10.0, 0.05)
    x = np.zeros_like(t)
    x[(t > 5.0) & (t < 5.1)] = -5.0          # 0.1 秒だけ
    assert detect_threshold_events(t, x, "hard_brake", _spec()) == []


def test_sustained_excursion_is_detected_with_peak():
    t = np.arange(0.0, 10.0, 0.05)
    x = np.zeros_like(t)
    x[(t >= 5.0) & (t <= 6.0)] = -5.0
    ev = detect_threshold_events(t, x, "hard_brake", _spec())
    assert len(ev) == 1
    assert ev[0].peak_value == -5.0
    assert np.isclose(ev[0].exceedance, 2.0)
    assert np.isclose(ev[0].duration_s, 1.0, atol=0.06)


def test_nearby_excursions_are_merged():
    t = np.arange(0.0, 10.0, 0.05)
    x = np.zeros_like(t)
    x[(t >= 5.0) & (t <= 5.4)] = -4.0
    x[(t >= 5.7) & (t <= 6.1)] = -4.0        # 0.3 秒の間隔
    assert len(detect_threshold_events(t, x, "hard_brake", _spec())) == 1


def test_nan_never_fires():
    t = np.arange(0.0, 10.0, 0.05)
    x = np.full_like(t, np.nan)
    assert detect_threshold_events(t, x, "low_ttc", _spec(op="lt", threshold=2.5)) == []


def test_abs_gt_keeps_signed_peak():
    t = np.arange(0.0, 10.0, 0.05)
    x = np.zeros_like(t)
    x[(t >= 2.0) & (t <= 3.0)] = -80.0
    ev = detect_threshold_events(t, x, "hard_steer", _spec(op="abs_gt", threshold=40.0, min_duration_s=0.15))
    assert len(ev) == 1 and ev[0].peak_value == -80.0


def test_rising_edge_detection():
    t = np.arange(0.0, 10.0, 0.05)
    x = np.zeros_like(t)
    x[(t >= 3.0) & (t <= 3.5)] = 1.0
    x[(t >= 7.0) & (t <= 7.2)] = 1.0
    ev = detect_threshold_events(t, x, "abs_active", _spec(op="rising", threshold=0.5, min_duration_s=0.0))
    assert len(ev) == 2


def test_gate_suppresses_events_outside_allowed_region():
    """gate は「その条件が成り立つ区間だけ判定する」足切り。

    低速域で舵角レートが大きく出るような、挙動の意味が変わる領域を除く。
    """
    import pandas as pd

    from near_miss.detectors import build_gate

    t = np.arange(0.0, 20.0, 0.05)
    x = np.zeros_like(t)
    x[(t >= 2.0) & (t <= 3.0)] = -10.0     # 低速域での大きな振れ
    x[(t >= 12.0) & (t <= 13.0)] = -10.0   # 巡航域での大きな振れ
    v = np.where(t < 10.0, 3.0, 25.0)
    df = pd.DataFrame({"t": t, "x": x, "v_mps": v})

    spec = _spec(op="abs_gt", threshold=5.0, min_duration_s=0.15,
                 gate=[{"feature": "v_mps", "op": "gt", "threshold": 15.0}])
    gate, label = build_gate(df, spec)
    ev = detect_threshold_events(t, x, "hard_steer", spec, gate, label)

    assert len(ev) == 1
    assert 12.0 <= ev[0].t_start <= 12.1     # 巡航域の 1 件だけが残る
    assert ev[0].gate == "v_mps gt 15.0"     # 判定根拠に gate が残る


def test_gate_with_missing_column_blocks_all_detection():
    """gate に必要な列が無いときは、gate を無視せず判定そのものを行わない。"""
    import pandas as pd

    from near_miss.detectors import build_gate

    t = np.arange(0.0, 10.0, 0.05)
    x = np.full_like(t, -10.0)
    df = pd.DataFrame({"t": t, "x": x})
    spec = _spec(op="abs_gt", threshold=5.0,
                 gate=[{"feature": "v_mps", "op": "gt", "threshold": 15.0}])
    gate, _ = build_gate(df, spec)
    assert not gate.any()
    assert detect_threshold_events(t, x, "hard_steer", spec, gate) == []


def _agreement_spec(**kw):
    spec = {
        "label": "急操舵",
        "op": "agreement",
        "feature": "ay",
        "tolerance_s": 0.3,
        "conditions": [
            {"feature": "steer", "op": "abs_gt", "threshold": 25.0},
            {"feature": "yaw", "op": "abs_gt", "threshold": 12.0},
            {"feature": "ay", "op": "abs_gt", "threshold": 2.0},
        ],
        "min_duration_s": 0.15,
        "merge_gap_s": 1.0,
        "stage": 2,
        "weight": 0.8,
    }
    spec.update(kw)
    return spec


def _agreement_frame(t, steer, yaw, ay):
    import pandas as pd

    return pd.DataFrame({"t": t, "steer": steer, "yaw": yaw, "ay": ay})


def test_dilate_widens_mask_both_directions():
    from near_miss.detectors import dilate

    m = np.zeros(11, dtype=bool)
    m[5] = True
    assert list(np.flatnonzero(dilate(m, 2))) == [3, 4, 5, 6, 7]
    assert list(np.flatnonzero(dilate(m, 0))) == [5]


def test_agreement_requires_every_signal():
    """1 系統だけが閾値を超えても発火しないこと。"""
    from near_miss.detectors import detect_agreement_events

    t = np.arange(0.0, 10.0, 0.05)
    win = (t >= 4.0) & (t <= 5.0)
    steer = np.where(win, 40.0, 0.0)
    yaw = np.where(win, 20.0, 0.0)
    ay = np.where(win, 3.0, 0.0)

    df = _agreement_frame(t, steer, yaw, ay)
    assert len(detect_agreement_events(t, df, "hard_steer", _agreement_spec())) == 1

    # 横加速度だけ立たない = 低速で舵だけ切った状況。発火しない
    df_low = _agreement_frame(t, steer, yaw, np.zeros_like(t))
    assert detect_agreement_events(t, df_low, "hard_steer", _agreement_spec()) == []

    # 舵角レートだけのノイズスパイク。他が裏付けないので発火しない
    spike = np.zeros_like(t)
    spike[100] = 400.0
    df_spike = _agreement_frame(t, spike, np.zeros_like(t), np.zeros_like(t))
    assert detect_agreement_events(t, df_spike, "hard_steer", _agreement_spec()) == []


def test_agreement_tolerates_time_offset_between_signals():
    """センサ間に時間ずれがあっても、許容窓の内側なら発火すること。"""
    from near_miss.detectors import detect_agreement_events

    t = np.arange(0.0, 10.0, 0.05)
    steer = np.where((t >= 4.0) & (t <= 5.0), 40.0, 0.0)
    yaw = np.where((t >= 4.2) & (t <= 5.2), 20.0, 0.0)     # 0.2 秒遅れ
    ay = np.where((t >= 4.25) & (t <= 5.25), 3.0, 0.0)     # 0.25 秒遅れ
    df = _agreement_frame(t, steer, yaw, ay)

    assert len(detect_agreement_events(t, df, "hard_steer", _agreement_spec())) == 1
    # 許容をなくすと、重なりが短くなり最短継続を満たさなくなる
    tight = _agreement_spec(tolerance_s=0.0, min_duration_s=0.9)
    assert detect_agreement_events(t, df, "hard_steer", tight) == []


def test_agreement_records_all_conditions_in_rule():
    from near_miss.detectors import detect_agreement_events

    t = np.arange(0.0, 10.0, 0.05)
    win = (t >= 4.0) & (t <= 5.0)
    df = _agreement_frame(t, np.where(win, 40.0, 0.0), np.where(win, 20.0, 0.0), np.where(win, 3.0, 0.0))
    ev = detect_agreement_events(t, df, "hard_steer", _agreement_spec())[0]
    assert "steer abs_gt 25.0" in ev.rule
    assert "yaw abs_gt 12.0" in ev.rule
    assert "ay abs_gt 2.0" in ev.rule
    assert ev.peak_value == 3.0        # 代表値は feature が指すもの


def _ev(kind, t0, t1):
    from near_miss.detectors import Event

    return Event(kind, kind, "x", "lt", 0.0, t0, t1, t1 - t0, 0.0, 0.0, 1, 1.0)


COOC_SPEC = {
    "label": "危険な車線変更の疑い",
    "op": "cooccurrence",
    "base": "lane_change_candidate",
    "window_s": 2.0,
    "min_count": 1,
    "require_any": ["low_ttc", "short_thw", "hard_brake"],
    "stage": 2,
    "weight": 2.0,
}


def test_cooccurrence_needs_a_danger_indicator():
    from near_miss.detectors import detect_cooccurrence_events

    lc = _ev("lane_change_candidate", 10.0, 13.0)
    assert detect_cooccurrence_events("risky_lane_change", COOC_SPEC, [lc]) == []

    # 車線変更の最中に車間逼迫が重なる
    got = detect_cooccurrence_events("risky_lane_change", COOC_SPEC, [lc, _ev("low_ttc", 11.0, 12.0)])
    assert len(got) == 1
    assert got[0].t_start == 10.0 and got[0].t_end == 13.0   # 区間は車線変更のもの
    assert "low_ttc" in got[0].rule


def test_cooccurrence_window_covers_before_and_after():
    """車線変更の前後 ±window_s の急ブレーキも共起とみなす。"""
    from near_miss.detectors import detect_cooccurrence_events

    lc = _ev("lane_change_candidate", 10.0, 13.0)
    before = detect_cooccurrence_events("risky_lane_change", COOC_SPEC, [lc, _ev("hard_brake", 8.5, 9.0)])
    after = detect_cooccurrence_events("risky_lane_change", COOC_SPEC, [lc, _ev("hard_brake", 14.0, 14.5)])
    far = detect_cooccurrence_events("risky_lane_change", COOC_SPEC, [lc, _ev("hard_brake", 20.0, 20.5)])
    assert len(before) == 1 and len(after) == 1 and far == []


def test_cooccurrence_counts_distinct_types_only():
    from near_miss.detectors import detect_cooccurrence_events

    lc = _ev("lane_change_candidate", 10.0, 13.0)
    spec = dict(COOC_SPEC, min_count=2)
    same = [lc, _ev("low_ttc", 11.0, 11.5), _ev("low_ttc", 12.0, 12.5)]
    assert detect_cooccurrence_events("risky_lane_change", spec, same) == []   # 同種は 1 と数える
    mixed = [lc, _ev("low_ttc", 11.0, 11.5), _ev("short_thw", 12.0, 12.5)]
    assert len(detect_cooccurrence_events("risky_lane_change", spec, mixed)) == 1


SEQ_SPEC = {
    "label": "回避後の制動",
    "op": "sequence",
    "first": ["s_evasion", "hard_steer"],
    "then": "hard_brake",
    "min_gap_s": 0.0,
    "max_gap_s": 3.0,
    "stage": 2,
    "weight": 2.0,
}


def test_sequence_requires_the_given_order():
    from near_miss.detectors import detect_sequence_events

    steer = _ev("hard_steer", 10.0, 11.0)
    brake = _ev("hard_brake", 12.0, 13.0)
    got = detect_sequence_events("brake_after_evasion", SEQ_SPEC, [steer, brake])
    assert len(got) == 1
    assert got[0].t_start == 10.0 and got[0].t_end == 13.0   # 一連の流れ全体
    assert np.isclose(got[0].peak_value, 1.0)                # 間隔

    # 逆順は拾わない
    early_brake = _ev("hard_brake", 5.0, 6.0)
    assert detect_sequence_events("brake_after_evasion", SEQ_SPEC, [steer, early_brake]) == []


def test_sequence_respects_the_time_window():
    from near_miss.detectors import detect_sequence_events

    steer = _ev("hard_steer", 10.0, 11.0)
    far = _ev("hard_brake", 20.0, 21.0)
    assert detect_sequence_events("brake_after_evasion", SEQ_SPEC, [steer, far]) == []
