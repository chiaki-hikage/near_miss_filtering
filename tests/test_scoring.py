"""候補統合の検証。"""

import numpy as np
import pandas as pd

from near_miss.detectors import Event
from near_miss.scoring import build_candidates, event_severity
from near_miss.signals import GriddedSignals

CFG = {
    "scoring": {
        "window_pad_s": 2.0,
        "merge_gap_s": 2.0,
        "cooccurrence_bonus": 0.5,
        "max_severity_per_event": 3.0,
    }
}


def _gs():
    t = np.arange(0.0, 60.0, 0.05)
    df = pd.DataFrame({"t": t, "v_mps": np.full_like(t, 25.0), "ttc_s": np.full_like(t, 1.5)})
    return GriddedSignals(df=df, rate_hz=20.0, segment_id="d/1", drive_id="d", vehicle="x", raw_can_loaded=False)


def _ev(kind, t0, t1, peak=-5.0, threshold=-3.0, weight=1.0):
    return Event(kind, kind, "ax_mps2", "lt", threshold, t0, t1, t1 - t0, peak,
                 threshold - peak, 1, weight)


def test_severity_is_one_at_threshold():
    e = _ev("hard_brake", 0.0, 1.0, peak=-3.0)
    assert np.isclose(event_severity(e, 3.0), 1.0)


def test_severity_is_capped():
    e = _ev("hard_brake", 0.0, 1.0, peak=-100.0)
    assert event_severity(e, 3.0) == 3.0


def test_overlapping_events_form_one_candidate_with_bonus():
    gs = _gs()
    events = [_ev("hard_brake", 10.0, 11.0), _ev("low_ttc", 10.5, 11.5)]
    cands = build_candidates(gs, events, CFG)
    assert len(cands) == 1
    assert cands[0].event_types == ["hard_brake", "low_ttc"]
    # 2 種類が重なったので共起加点が入る
    assert cands[0].severity > sum(event_severity(e, 3.0) for e in events) - 1e-9


def test_distant_events_stay_separate():
    gs = _gs()
    events = [_ev("hard_brake", 10.0, 11.0), _ev("hard_brake", 40.0, 41.0)]
    assert len(build_candidates(gs, events, CFG)) == 2


def test_same_type_counted_once():
    gs = _gs()
    events = [_ev("hard_brake", 10.0, 11.0), _ev("hard_brake", 11.5, 12.0)]
    cands = build_candidates(gs, events, CFG)
    assert len(cands) == 1 and cands[0].event_types == ["hard_brake"]
