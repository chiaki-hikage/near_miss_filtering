"""フェードの検討で使う計算の試験 (docs/brake_fade_feasibility.md)。

- 模擬の性能低下: 抵抗の分は下げず、ブレーキの分だけを下げる
- 熱の模型: 一定の熱の入力で、温度が 外気 + a P / c0 に近づく。センサーの遅れで追いつきが遅れる
- NCST の温度の抜け (10001・10002・0) を欠損にする
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "brake"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import fade_sensitivity as FS  # noqa: E402
import ncst_thermal as NT  # noqa: E402


def test_degrade_keeps_resistance():
    E = pd.DataFrame(dict(v_steady=[10.0, 30.0], decel_brake_obs=[1.0, 2.0], decel_brake_pred=[1.0, 2.0]))
    coef = (0.1, 0.0004)
    r = np.array([0.1 + 0.0004 * 100, 0.1 + 0.0004 * 900])
    eta = FS.degrade(E, 0.2, coef)
    assert eta == pytest.approx((r + 0.8 * (np.array([1.0, 2.0]) - r)) / np.array([1.0, 2.0]))
    assert FS.degrade(E, 0.0, coef) == pytest.approx([1.0, 1.0])
    # 効きが 20% 下がっても、eta は 20% より小さくしか下がらない (抵抗の分が残る)
    assert (1 - eta > 0).all() and (1 - eta < 0.2).all()


def test_thermal_model_steady_state_and_sensor_lag():
    n = 40000
    df = pd.DataFrame(dict(psi=np.full(n, 11.0), v=np.full(n, 10.0), t_amb=np.full(n, 20.0), T=np.r_[20.0, np.full(n - 1, np.nan)]))
    a, c0, p_th = 1e-3, 1e-3, 1.0
    sim_fast = NT.simulate((a, c0, 0.0, p_th, 1.0), df, "P_pv")
    P = (11.0 - p_th) * 10.0
    assert sim_fast[-1] == pytest.approx(20.0 + a * P / c0, rel=1e-3)
    sim_slow = NT.simulate((a, c0, 0.0, p_th, 600.0), df, "P_pv")
    assert sim_slow[300] < sim_fast[300]           # センサーの遅れで追いつきが遅い
    assert sim_slow[-1] == pytest.approx(sim_fast[-1], rel=1e-2)


def test_power_threshold():
    df = pd.DataFrame(dict(psi=[0.5, 1.0, 5.0], v=[10.0, 10.0, np.nan], P_ke=[0.0, 0.0, 0.0]))
    assert NT.power(df, "P_pv", 1.0).tolist() == [0.0, 0.0, 0.0]   # 押し出し圧より下と、車速の抜けは 0
    assert NT.power(df, "P_p", 1.0).tolist() == pytest.approx([0.0, 0.0, 60.0])


DATA = REPO / "data/brake/ncst_hdv_brake"


@pytest.mark.skipif(not (DATA / "DATA33_CBDx3.CSV").exists(), reason="NCST のデータが無い")
def test_ncst_logger_dropouts_are_missing():
    df = NT.read_logger("DATA33_CBDx3.CSV", "chassis")
    assert df["T"].notna().any()
    assert (df["T"].dropna() > 0).all() and (df["T"].dropna() < 1000).all()


def test_pedal_lag_recovers_clock_offset():
    import ncst_scene_illustration as SI
    rng = np.random.default_rng(0)
    n = 3000
    t = np.arange(n, dtype=float)
    brake = np.zeros(n)
    acc = np.zeros(n)
    k = 0
    acc[:] = 60.0                           # ふだんはアクセルを踏み、ブレーキの前後だけ離す
    while k < n - 60:
        brake[k:k + 8] = 15.0
        acc[k - 4 if k >= 4 else 0:k + 12] = 0.0          # ブレーキの前後 4 s ずつ離す (対称)
        k += int(rng.integers(40, 120))
    B = pd.DataFrame(dict(t=t, psi=brake))
    H = pd.DataFrame(dict(t=t - 16.0, acc=acc))     # HEM の時計が 16 s 遅れている
    lag, overlap = SI.pedal_lag(B, H, search=40)
    assert lag == 16 and overlap == 0.0


def test_stopping_scenario_brake_part_scales():
    """停止距離の試算: 制動区間は 1/(1-δ) 倍、空走は δ によらない (ncst_fade_safety)。"""
    import ncst_fade_safety as FSF

    S = dict(speeds_kmh=[100], base_decel_mps2=[4.0], reaction_s=1.5, actuation_s=0.45, efficiency_drops=[0.2])
    T = FSF.stopping(S)
    b = T[T.delta == 0].iloc[0]
    d = T[T.delta == 0.2].iloc[0]
    v = 100 / 3.6
    assert abs(b.brake_m - v ** 2 / 8.0) < 1e-9
    assert abs(d.pre_m - b.pre_m) < 1e-9
    assert abs(d.brake_m / b.brake_m - 1.25) < 1e-9
    assert abs(d.increase_pct_brake - 25.0) < 1e-9


def test_dyno_effects_adjusted_recovers_hot_effect():
    """作った台上のデータで、車速の効果と高温の効果を回帰で分けられる (ncst_fade_safety)。"""
    import ncst_fade_safety as FSF

    rng = np.random.default_rng(0)
    rows = []
    for traj, T0, T1, v, n in [("CBD0", 15, 45, 5.0, 20), ("CBDx3", 60, 125, 5.5, 40), ("UDDS", 130, 165, 10.0, 20)]:
        for _ in range(n):
            T = rng.uniform(T0, T1)
            vs = v + rng.normal(0, 1.5)
            psi = rng.uniform(18, 28)
            eff = 200 * np.exp(-0.04 * vs) * (0.9 if T >= 130 else 1.0) * np.exp(rng.normal(0, 0.02))
            F0 = 100.0
            rows.append(dict(traj=traj, T=T, v_steady=vs, psi=psi, F0=F0, F=eff * (psi - 1.2) + F0, eff=eff))
    E = pd.DataFrame(rows)
    E["rel"] = E.eff / E.loc[E["T"] < 50, "eff"].median()
    C = dict(hot_C=130, cold_C=50, pushout_psi=1.2, max_f0_frac=0.4, min_psi=15, per_C=30, bootstrap=200, seed=1)
    res, extra = FSF.dyno_effects(E, C)
    r = {x["key"]: x for x in res}
    assert abs(r["adjusted"]["effect"] - (-0.10)) < 0.03          # 高温の効果 -10% を取り戻す
    assert r["raw"]["effect"] < r["adjusted"]["effect"] - 0.05    # 調整前は車速の分だけ大きく出る


def test_fade_risk_score_rules():
    """フェードリスク指標: 重みつき平均、熱が無いときの効率低下の除外、欠けた成分でスコアを出さない (fade_risk_index)。"""
    import fade_risk_index as FRI

    cfg = dict(components=dict(thermal=dict(weight=0.4), continuous=dict(weight=0.2), downhill=dict(weight=0.2),
                               degradation=dict(weight=0.2, thermal_gate=0.2)),
               rank=dict(high=60, medium=30))
    D = pd.DataFrame(dict(thermal=[1.0, 0.1, 0.5], continuous=[1.0, 0.0, 0.5], downhill=[1.0, 0.0, np.nan], degradation=[1.0, 1.0, 0.5]))
    S = FRI.score(D, cfg, ["thermal", "continuous", "downhill", "degradation"])
    assert abs(S.score[0] - 100) < 1e-9 and S["rank"][0] == "High"
    assert abs(S.score[1] - 100 * 0.4 * 0.1) < 1e-9          # 熱が 0.2 未満なので効率低下は 0
    assert np.isnan(S.score[2])                                 # 下り坂が欠けた時刻はスコアなし
    S3 = FRI.score(D.iloc[:1], cfg, ["thermal", "continuous", "downhill"])   # 測れない成分を除くと残りの重みで平均
    assert abs(S3.score[0] - 100) < 1e-9
    assert abs(sum(S3[f"pts_{c}"][0] for c in FRI.COMP) - S3.score[0]) < 1e-9


def test_forward_view_projection():
    """前方の CG の投影: 視線が下を向く (下り) と、遠くの水平な点 (水平線) は画面の上に出る。真正面の点は画面の中央。"""
    import ncst_forward_view as FV
    cam = np.array([0.0, 0.0, 0.0])
    P = np.array([[0.0, 1e6, 0.0], [0.0, 100.0, 0.0], [10.0, 100.0, 0.0]])   # 北へ遠く・北へ 100 m・その 10 m 右 (東)
    x, y, f = FV.project(P, cam, (0.0, 1.0), -0.05, 1.0)
    assert y[0] > 0 and abs(y[0] - np.tan(0.05)) < 1e-4
    x0, y0, _ = FV.project(P, cam, (0.0, 1.0), 0.0, 1.0)
    assert abs(x0[1]) < 1e-12 and abs(y0[1]) < 1e-12
    assert x0[2] > 0                                                          # 右の点は画面の右
