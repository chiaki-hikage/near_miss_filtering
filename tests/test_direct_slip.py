"""スリップの直接証拠 (direct_slip.py) の単体試験。

合成信号で確かめるのは 3 点。
  1. 証拠が種類ごとに分かれて残ること (まとめて 1 本のフラグにしない)
  2. 等級が「確定証拠が何系統重なったか」だけで決まること
  3. **輪速の開きだけでは確定 (A) にならない**こと
"""

from __future__ import annotations

import numpy as np
import pytest

from near_miss.config import DEFAULT_DETECTION, load_yaml
from near_miss.direct_slip import (
    DIRECT,
    GRADE_CANDIDATE,
    GRADE_HIGH,
    GRADE_SINGLE,
    SUPPORT,
    evidence_masks,
    events_to_frame,
    find_direct_slip,
)
from near_miss.signals import GriddedSignals

RATE = 20.0


@pytest.fixture(scope="module")
def cfg():
    return load_yaml(DEFAULT_DETECTION)


def make_gs(n: int = 400, v: float = 15.0, **cols) -> GriddedSignals:
    """一様グリッドの合成走行。指定しない列は 0 で埋める。"""
    import pandas as pd

    data = {
        "t": np.arange(n) / RATE,
        "v_mps": np.full(n, float(v)),
        "ax_mps2": np.zeros(n),
        "ay_kin_mps2": np.zeros(n),
        "yaw_rate_dps": np.zeros(n),
        "steer_deg": np.zeros(n),
        "beta_model_deg": np.zeros(n),
        "slip_warn": np.zeros(n),
        "abs_active_flag": np.zeros(n),
        "vsc_active_flag": np.zeros(n),
        "tc_disabled": np.zeros(n),
        "ws_spread_mps": np.zeros(n),
    }
    for k, val in cols.items():
        data[k] = np.asarray(val, dtype=float) if np.ndim(val) else np.full(n, float(val))
    return _gs(pd.DataFrame(data))


def _gs(df) -> GriddedSignals:
    return GriddedSignals(df=df, rate_hz=RATE, segment_id="seg", drive_id="drive",
                          vehicle="toyota_rav4_tss2", raw_can_loaded=True)


def span(n: int, a: int, b: int, val: float) -> np.ndarray:
    x = np.zeros(n)
    x[a:b] = val
    return x


# ---------------------------------------------------------------------------
# 証拠の切り分け
# ---------------------------------------------------------------------------
def test_ランプチェックの値3は証拠にならない(cfg):
    """slip_warn = 3 はイグニッション ON の点灯試験。閾値 3.5 で外れる。"""
    gs = make_gs(v=0.0, slip_warn=span(400, 0, 44, 3.0))
    events, counts = find_direct_slip(gs, cfg)
    assert events == []
    assert counts.runs_by_evidence.get("slip_warn", 0) == 0


def test_実作動の値4は証拠になる(cfg):
    gs = make_gs(slip_warn=span(400, 100, 120, 4.0))
    events, counts = find_direct_slip(gs, cfg)
    assert len(events) == 1
    e = events[0]
    assert e.slip_warn is True and e.slip_warn_peak == 4.0
    assert e.abs_active is False and e.vsc_active is False
    assert e.n_direct == 1 and e.grade == GRADE_SINGLE
    assert e.evidence == "slip_warn"


def test_確定証拠が2系統で確定になる(cfg):
    gs = make_gs(slip_warn=span(400, 100, 120, 4.0),
                 abs_active_flag=span(400, 105, 125, 1.0))
    events, _ = find_direct_slip(gs, cfg)
    assert len(events) == 1
    e = events[0]
    assert e.n_direct == 2 and e.grade == GRADE_HIGH
    # 種類ごとに分かれて残っている
    assert e.slip_warn and e.abs_active and not e.vsc_active
    assert e.evidence == "slip_warn|abs_active"


def test_輪速の開きだけでは確定にしない(cfg):
    """裏付けは単独では候補どまり。ここが緩むと目視の負担が跳ね上がる。"""
    gs = make_gs(ws_spread_mps=span(400, 100, 140, 3.0))
    events, _ = find_direct_slip(gs, cfg)
    assert len(events) == 1
    e = events[0]
    assert e.wheel_spread is True
    assert e.n_direct == 0
    assert e.grade == GRADE_CANDIDATE


def test_輪速の開きは確定証拠と重なれば等級を上げない(cfg):
    """裏付けは列には残るが、等級は確定証拠の数だけで決まる。"""
    gs = make_gs(slip_warn=span(400, 100, 120, 4.0),
                 ws_spread_mps=span(400, 100, 140, 3.0))
    events, _ = find_direct_slip(gs, cfg)
    e = events[0]
    assert e.wheel_spread and e.slip_warn
    assert e.n_direct == 1 and e.grade == GRADE_SINGLE


# ---------------------------------------------------------------------------
# 輪速の開きの条件
# ---------------------------------------------------------------------------
def test_輪速の開きは持続時間が足りないと通らない(cfg):
    """0.3 秒 = 20 Hz で 6 サンプル。3 サンプルの尖りは雑音として落とす。"""
    gs = make_gs(ws_spread_mps=span(400, 100, 103, 5.0))
    events, _ = find_direct_slip(gs, cfg)
    assert events == []


def test_輪速の開きは低速では見ない(cfg):
    """停車付近は輪速そのものが不安定で、開きに意味が無い。"""
    gs = make_gs(v=0.5, ws_spread_mps=span(400, 100, 140, 5.0))
    events, _ = find_direct_slip(gs, cfg)
    assert events == []


# ---------------------------------------------------------------------------
# 参考情報 (tc_disabled)
# ---------------------------------------------------------------------------
def test_TC解除だけでは候補を作らない(cfg):
    """「TC が切られている」は「TC が作動した」ではない。"""
    gs = make_gs(tc_disabled=span(400, 100, 140, 1.0))
    events, _ = find_direct_slip(gs, cfg)
    assert events == []


def test_TC解除は候補行に添えられる(cfg):
    gs = make_gs(slip_warn=span(400, 100, 120, 4.0),
                 tc_disabled=span(400, 95, 130, 1.0))
    events, _ = find_direct_slip(gs, cfg)
    e = events[0]
    assert e.tc_disabled is True
    assert e.n_direct == 1          # 等級には効かない


# ---------------------------------------------------------------------------
# 信号の有無
# ---------------------------------------------------------------------------
def test_信号が無いことと立たないことを区別する(cfg):
    """comma2k19 の RAV4 には 0x226 も 0x3B7 も無い。
    「立たなかった」と読み違えると『滑りが無い』と誤解する。"""
    import pandas as pd

    n = 200
    df = pd.DataFrame({"t": np.arange(n) / RATE, "v_mps": np.full(n, 15.0)})
    gs = _gs(df)
    _, counts = find_direct_slip(gs, cfg)
    for name in (*DIRECT, *SUPPORT):
        assert counts.missing.get(name) == 1
        assert counts.present.get(name, 0) == 0


def test_信号はあるが立たない場合(cfg):
    gs = make_gs()
    _, counts = find_direct_slip(gs, cfg)
    for name in (*DIRECT, *SUPPORT):
        assert counts.present.get(name) == 1
        assert counts.missing.get(name, 0) == 0


# ---------------------------------------------------------------------------
# 集計と表
# ---------------------------------------------------------------------------
def test_組み合わせ別の内訳が取れる(cfg):
    gs = make_gs(slip_warn=span(400, 40, 60, 4.0),
                 abs_active_flag=span(400, 45, 65, 1.0))
    _, counts = find_direct_slip(gs, cfg)
    assert counts.by_combination == {"slip_warn|abs_active": 1}
    assert counts.by_grade == {GRADE_HIGH: 1}


def test_表は等級の強い順に並ぶ(cfg):
    gs = make_gs(ws_spread_mps=span(400, 40, 80, 3.0),
                 slip_warn=span(400, 200, 220, 4.0),
                 abs_active_flag=span(400, 205, 225, 1.0))
    events, _ = find_direct_slip(gs, cfg)
    assert len(events) == 2
    df = events_to_frame(events, "abc")
    assert list(df.grade) == [GRADE_HIGH, GRADE_CANDIDATE]
    assert (df.config_hash == "abc").all()


def test_離れた証拠は別の確認単位になる(cfg):
    gs = make_gs(n=800, slip_warn=span(800, 50, 60, 4.0) + span(800, 600, 610, 4.0))
    events, _ = find_direct_slip(gs, cfg)
    assert len(events) == 2


def test_マスクは種類ごとに独立(cfg):
    gs = make_gs(slip_warn=span(400, 100, 120, 4.0),
                 abs_active_flag=span(400, 300, 320, 1.0))
    masks, present = evidence_masks(gs.df, cfg, RATE)
    assert masks["slip_warn"][110] and not masks["slip_warn"][310]
    assert masks["abs_active"][310] and not masks["abs_active"][110]
    assert not masks["vsc_active"].any()
    assert all(present[k] for k in ("slip_warn", "abs_active", "vsc_active"))


# ---------------------------------------------------------------------------
# 既存の枝に触れていないこと
# ---------------------------------------------------------------------------
def test_beta枝の設定を変えていない(cfg):
    """direct_slip を足しても sideslip の設定は変わらない。"""
    s1 = cfg["sideslip"]["stage1"]
    assert s1["direct_flags"] == ["vsc_active_flag", "abs_active_flag"]
    assert cfg["sideslip"]["min_speed_mps"] == 10.0


def test_値7の継続点灯は証拠にならない(cfg):
    """slip_warn は 3 ビットの多値。4 が実作動で、7 は継続点灯 (故障系)。
    上限を課さないと 60 秒続く「滑り」を拾う。"""
    gs = make_gs(slip_warn=span(400, 100, 140, 7.0))
    events, counts = find_direct_slip(gs, cfg)
    assert events == []
    assert counts.runs_by_evidence.get("slip_warn", 0) == 0


def test_上限は設定が無ければ効かない(cfg):
    """max を書いていないフラグは従来どおり下限だけで判定する。"""
    import copy
    c = copy.deepcopy(cfg)
    c["direct_slip"]["flags"]["slip_warn"].pop("max", None)
    gs = make_gs(slip_warn=span(400, 100, 140, 7.0))
    events, _ = find_direct_slip(gs, c)
    assert len(events) == 1
