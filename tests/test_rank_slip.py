"""スリップ事象の種別分け (scripts/detect/rank_slip.py) の試験。

確かめること
  1. 横方向は**積極的な条件**を満たしたときだけ。残りものを寄せない
  2. どの向きか言えないものは「未分類」として残る
  3. 直接の証拠 (VSC) と、確かめが要る理由 (|ay| など) を区別する
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "detect"


@pytest.fixture(scope="module")
def rs():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("rank_slip", SCRIPTS / "rank_slip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ev(**kw) -> pd.Series:
    """既定は「何も起きていない」事象。必要な量だけ上書きする。"""
    base = dict(slip_warn=False, abs_active=False, vsc_active=False, wheel_spread=False,
                ws_spread_peak_mps=0.2, ax_at_peak_mps2=0.0, ay_at_peak_mps2=0.5,
                yaw_rate_at_peak_dps=1.0, steer_at_peak_deg=5.0)
    base.update(kw)
    return pd.Series(base)


# ---------------------------------------------------------------------------
# 横方向
# ---------------------------------------------------------------------------
def test_VSC_は横方向の確証(rs):
    r = ev(slip_warn=True, vsc_active=True, ay_at_peak_mps2=2.0)
    assert rs.classify(r, 2.0) == rs.KIND_LATERAL
    assert rs.lateral_confidence(rs.lateral_reason(r)) == "確証"


def test_大きな横G_は横方向だが要検証(rs):
    """|ay| は v x ヨーなので、強く曲がっているだけかもしれない。"""
    r = ev(slip_warn=True, ay_at_peak_mps2=6.0, ax_at_peak_mps2=1.0)
    assert rs.classify(r, 2.0) == rs.KIND_LATERAL
    assert rs.lateral_confidence(rs.lateral_reason(r)) == "要検証"


def test_横Gが対照p99_9に届かなければ横方向にしない(rs):
    r = ev(slip_warn=True, ay_at_peak_mps2=4.5, ax_at_peak_mps2=0.2)
    assert rs.lateral_reason(r) == ""


def test_縦加速度の方が大きければ横方向にしない(rs):
    """急制動に伴う見かけの横 G を横滑りとして拾わないため。"""
    r = ev(abs_active=True, ay_at_peak_mps2=5.2, ax_at_peak_mps2=-8.0)
    assert rs.lateral_reason(r) == ""
    assert rs.classify(r, 2.0) == rs.KIND_LOCK


def test_逆操舵は横方向(rs):
    """舵とヨーの符号は普通そろう (4 車種で相関 0.94〜0.96)。"""
    r = ev(slip_warn=True, ay_at_peak_mps2=4.2, ax_at_peak_mps2=0.3,
           yaw_rate_at_peak_dps=-20.0, steer_at_peak_deg=120.0)
    assert rs.classify(r, 2.0) == rs.KIND_LATERAL
    assert "逆操舵" in rs.lateral_reason(r)


# ---------------------------------------------------------------------------
# 残りもの
# ---------------------------------------------------------------------------
def test_向きが言えないものは未分類にする(rs):
    """以前はこれを横方向に寄せていた。輪速差が閾値をわずかに下回る加速中の空転。"""
    r = ev(slip_warn=True, ws_spread_peak_mps=1.7, ax_at_peak_mps2=2.5,
           ay_at_peak_mps2=2.0)
    assert rs.classify(r, 2.0) == rs.KIND_UNCLASSIFIED
    assert rs.lateral_confidence(rs.lateral_reason(r)) == ""


def test_輪速が開いて加速側なら空転(rs):
    r = ev(slip_warn=True, wheel_spread=True, ws_spread_peak_mps=3.0,
           ax_at_peak_mps2=2.5, ay_at_peak_mps2=1.0)
    assert rs.classify(r, 2.0) == rs.KIND_SPIN


def test_輪速が開いて減速側ならロック(rs):
    r = ev(wheel_spread=True, ws_spread_peak_mps=3.0, ax_at_peak_mps2=-3.0)
    assert rs.classify(r, 2.0) == rs.KIND_LOCK


def test_ABS_はロック(rs):
    r = ev(slip_warn=True, abs_active=True, ax_at_peak_mps2=-2.0)
    assert rs.classify(r, 2.0) == rs.KIND_LOCK
