"""スリップの直接証拠 (direct_slip.py) の単体試験。

合成信号で確かめるのは 4 点。
  1. 証拠が種類ごとに分かれて残ること (まとめて 1 本のフラグにしない)
  2. 等級が「異なる物理メカニズムの裏付けが何系統あるか」で決まること
     制御が作動しただけでは上位に行かない
  3. ABS の裏付けは輪速の開きだけでなく、輪単体の瞬間減速度でも取れること
  4. 物理挙動だけでは確定にしないこと
"""

from __future__ import annotations

import numpy as np
import pytest

from near_miss.config import DEFAULT_DETECTION, load_yaml
from near_miss.direct_slip import (
    DIRECT,
    GRADE_ACTIVATION,
    GRADE_HIGH,
    GRADE_MULTI,
    GRADE_PHYSICS,
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
        "wheel_decel_mps2": np.zeros(n),
        "yaw_residual_sigma": np.zeros(n),
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
    assert e.n_direct == 1
    assert e.evidence == "slip_warn"


# ---------------------------------------------------------------------------
# 等級 — 作動だけでは上げない
# ---------------------------------------------------------------------------
def test_作動だけで物理挙動が無ければ上位にしない(cfg):
    """RAV4 では slip_warn と ABS が 90% 同時に立つ。数えるだけでは意味が無い。"""
    gs = make_gs(slip_warn=span(400, 100, 120, 4.0),
                 abs_active_flag=span(400, 105, 125, 1.0))
    events, _ = find_direct_slip(gs, cfg)
    e = events[0]
    assert e.n_direct == 2              # 作動は 2 系統あるが
    assert e.n_corroborated == 0        # 裏付けはどちらも取れていない
    assert e.grade == GRADE_ACTIVATION
    assert e.corroboration == ""


def test_裏付けの取れた作動が1系統なら_high(cfg):
    gs = make_gs(slip_warn=span(400, 100, 120, 4.0),
                 ws_spread_mps=span(400, 100, 140, 3.0))
    events, _ = find_direct_slip(gs, cfg)
    e = events[0]
    assert e.slip_warn and e.wheel_spread
    assert e.n_corroborated == 1 and e.grade == GRADE_HIGH
    assert e.corroboration == "slip_warn:wheel_spread"


def test_同じ系統の裏付けは何系統あっても_high(cfg):
    """ABS も slip_warn も裏付けは輪速由来。分かった事実は 1 つなので 1 系統。
    実測でも multi 50 件のうち 48 件がこの形だった。"""
    gs = make_gs(slip_warn=span(400, 100, 120, 4.0),
                 abs_active_flag=span(400, 105, 125, 1.0),
                 ws_spread_mps=span(400, 100, 140, 3.0),
                 wheel_decel_mps2=span(400, 108, 112, -12.0))
    events, _ = find_direct_slip(gs, cfg)
    e = events[0]
    assert e.n_direct == 2 and e.n_corroborated == 2   # 作動は 2 つとも裏が取れたが
    assert e.n_mechanisms == 1 and e.mechanisms == "輪速"
    assert e.grade == GRADE_HIGH


def test_異なる系統の裏付けが2つなら_multi(cfg):
    """ABS を輪速で、VSC を横方向で裏付ける。独立した 2 つの事実。"""
    gs = make_gs(abs_active_flag=span(400, 100, 125, 1.0),
                 vsc_active_flag=span(400, 105, 125, 1.0),
                 wheel_decel_mps2=span(400, 108, 112, -12.0),
                 ay_kin_mps2=span(400, 100, 125, 5.0))
    events, _ = find_direct_slip(gs, cfg)
    e = events[0]
    assert e.n_mechanisms == 2
    assert set(e.mechanisms.split("|")) == {"輪速", "横方向"}
    assert e.grade == GRADE_MULTI


def test_舵で説明できないヨーも横方向の系統(cfg):
    """諸元が当てはめ済みの車種でだけ出る量。横加速度と同じ系統に入れる。"""
    gs = make_gs(abs_active_flag=span(400, 100, 125, 1.0),
                 vsc_active_flag=span(400, 105, 125, 1.0),
                 wheel_decel_mps2=span(400, 108, 112, -12.0),
                 yaw_residual_sigma=span(400, 100, 125, 25.0))
    events, _ = find_direct_slip(gs, cfg)
    e = events[0]
    assert e.yaw_residual is True
    assert e.n_mechanisms == 2 and e.grade == GRADE_MULTI


# ---------------------------------------------------------------------------
# ABS の裏付け — 輪単体の瞬間減速度
# ---------------------------------------------------------------------------
def test_ABSは輪単体の深い減速で裏が取れる(cfg):
    """4 輪が同時にロックと解放を繰り返すと、輪と輪の差には出ない。
    実測でも ABS 事象 66 件のうち 35 件が輪速差 1.0 m/s 未満だった。"""
    gs = make_gs(abs_active_flag=span(400, 100, 120, 1.0),
                 wheel_decel_mps2=span(400, 105, 110, -12.0))
    events, _ = find_direct_slip(gs, cfg)
    e = events[0]
    assert e.wheel_spread is False       # 輪速の開きは無い
    assert e.wheel_decel is True
    assert e.wheel_decel_peak_mps2 == -12.0
    assert e.grade == GRADE_HIGH
    assert e.corroboration == "abs_active:wheel_decel"


def test_通常の制動程度の輪減速では裏付けにしない(cfg):
    """対照窓では制動中でも輪の減速は中央 -2.8 m/s²。閾値 -8.0 はその外側。"""
    gs = make_gs(abs_active_flag=span(400, 100, 120, 1.0),
                 wheel_decel_mps2=span(400, 105, 110, -3.0))
    events, _ = find_direct_slip(gs, cfg)
    e = events[0]
    assert e.wheel_decel is False
    assert e.grade == GRADE_ACTIVATION


def test_輪の減速は低速では見ない(cfg):
    gs = make_gs(v=0.5, abs_active_flag=span(400, 100, 120, 1.0),
                 wheel_decel_mps2=span(400, 105, 110, -12.0))
    events, _ = find_direct_slip(gs, cfg)
    assert events[0].wheel_decel is False


# ---------------------------------------------------------------------------
# VSC の裏付け — 横加速度
# ---------------------------------------------------------------------------
def test_VSCは横加速度で裏が取れる(cfg):
    gs = make_gs(vsc_active_flag=span(400, 100, 120, 1.0),
                 ay_kin_mps2=span(400, 100, 120, 5.0))
    events, _ = find_direct_slip(gs, cfg)
    e = events[0]
    assert e.lateral is True
    assert e.grade == GRADE_HIGH and e.corroboration == "vsc_active:lateral"


def test_VSCでも横加速度が小さければ裏付けにしない(cfg):
    gs = make_gs(vsc_active_flag=span(400, 100, 120, 1.0),
                 ay_kin_mps2=span(400, 100, 120, 2.0))
    events, _ = find_direct_slip(gs, cfg)
    assert events[0].grade == GRADE_ACTIVATION


def test_ABSの裏付けに横加速度は使わない(cfg):
    """直進制動で見かけの横 G が出る車両がある。対応表にない裏付けは数えない。"""
    gs = make_gs(abs_active_flag=span(400, 100, 120, 1.0),
                 ay_kin_mps2=span(400, 100, 120, 6.0))
    events, _ = find_direct_slip(gs, cfg)
    e = events[0]
    assert e.lateral is True            # 列には残るが
    assert e.grade == GRADE_ACTIVATION  # ABS の裏付けにはならない


# ---------------------------------------------------------------------------
# 物理挙動だけ
# ---------------------------------------------------------------------------
def test_輪速の開きだけでは確定にしない(cfg):
    """裏付けは単独では候補どまり。ここが緩むと目視の負担が跳ね上がる。"""
    gs = make_gs(ws_spread_mps=span(400, 100, 140, 3.0))
    events, _ = find_direct_slip(gs, cfg)
    assert len(events) == 1
    e = events[0]
    assert e.wheel_spread is True
    assert e.n_direct == 0 and e.n_corroborated == 0
    assert e.grade == GRADE_PHYSICS


def test_輪の減速だけでは確認単位を作らない(cfg):
    """閾値 -8 m/s² は対照窓の 2% が届く。単独で拾わせると通常走行が大量に並ぶ。
    輪の減速は「作動があったときに本物かを確かめる」ためだけに使う。"""
    gs = make_gs(wheel_decel_mps2=span(400, 100, 110, -15.0))
    events, _ = find_direct_slip(gs, cfg)
    assert events == []


def test_横加速度だけでも確認単位を作らない(cfg):
    """|ay| >= 3.9 は対照窓の p99。曲がっているだけの区間を拾わない。"""
    gs = make_gs(ay_kin_mps2=span(400, 100, 140, 5.0))
    events, _ = find_direct_slip(gs, cfg)
    assert events == []


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
    assert e.n_corroborated == 0          # 等級には効かない


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
    assert counts.by_grade == {GRADE_ACTIVATION: 1}


def test_表は等級の強い順に並ぶ(cfg):
    gs = make_gs(ws_spread_mps=span(400, 40, 80, 3.0),
                 slip_warn=span(400, 200, 220, 4.0),
                 vsc_active_flag=span(400, 205, 225, 1.0),
                 wheel_decel_mps2=span(400, 208, 212, -12.0),
                 ay_kin_mps2=span(400, 200, 225, 5.0))
    events, _ = find_direct_slip(gs, cfg)
    assert len(events) == 2
    df = events_to_frame(events, "abc")
    assert list(df.grade) == [GRADE_MULTI, GRADE_PHYSICS]
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
