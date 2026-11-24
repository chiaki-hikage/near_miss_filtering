"""Udacity CH2_002 の制動と熱負荷の計算の試験 (docs/udacity_hmb_brake.md)。

- 2 段の当てはめ: 作った信号から kw, c0, c2, kb を取り戻せる
- 下りのエネルギーの分け方: つり合う信号なら残りが 0 に近い
- ピッチの符号: 前上がりが正
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "brake"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import udacity_hmb_heat as UH  # noqa: E402

G = 9.80665


def synthetic(n=4000, dt=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    v = 15 + 3 * np.sin(t / 20)
    theta = np.radians(-4 * np.clip(np.sin(t / 50), 0, None))
    Tw = rng.uniform(-500, 800, n)
    Tb = np.where(rng.random(n) < 0.3, rng.uniform(100, 800, n), 0.0)
    kw, kb, c0, c2 = 1.5e-3, 1.2e-3, 0.12, 3e-4
    a = kw * Tw - kb * Tb - c0 - c2 * v**2 - G * np.sin(theta)
    return pd.DataFrame(dict(t=t, v=v, theta=theta, Tw=Tw, Tb=Tb, a=a)), dict(kw=kw, kb=kb, c0=c0, c2=c2)


def test_two_stage_fit_recovers_coefficients():
    D, true = synthetic()
    on = (D.Tb > 30).to_numpy()
    co, pred, ok1, ok2, d = UH.fit_model(D, ~on, on, G)
    for k in true:
        assert abs(co[k] - true[k]) < 1e-6 * max(1, abs(true[k])) + 1e-9
    assert np.allclose(d[on], true["kb"] * D.Tb.to_numpy()[on], atol=1e-6)


def test_energy_budget_closes():
    dt = 0.05
    t = np.arange(0, 200, dt)
    v = np.full_like(t, 15.0)
    s = np.cumsum(v) * dt
    alt = 200 - 0.05 * s  # 一定の下り
    kw, kb, c0, c2 = 1.5e-3, 1.2e-3, 0.1, 3e-4
    # 一定速: 重力 = 抵抗 + ブレーキ + 駆動側の負のトルク
    grav = G * 0.05
    res = c0 + c2 * v**2
    Tb = np.full_like(t, (grav - res[0]) * 0.5 / kb)
    Tw = np.full_like(t, -(grav - res[0]) * 0.5 / kw)
    D = pd.DataFrame(dict(t=t, v=v, s=s, alt=alt, Tw=Tw, Tb=Tb, p_brake=kb * Tb * v))
    out = UH.energy_budget(D, dict(kw=kw, kb=kb, c0=c0, c2=c2), dt, G, t[-1] + 1)
    assert abs(out["residual_Jkg"]) < 0.01 * out["potential_Jkg"]
    assert abs(out["brake_share_of_potential"] - 0.5 * (grav - res[0]) / (G * 0.05 * np.cos(0))) < 0.02


def test_quat_pitch_nose_up_positive():
    a = np.radians(5)
    # y 軸 (左) まわりに -a 回すと、x (前) が上を向く (ROS の右手系)
    q = dict(x=0.0, y=np.sin(-a / 2), z=0.0, w=np.cos(-a / 2))
    assert abs(UH.quat_pitch(q["x"], q["y"], q["z"], q["w"]) - a) < 1e-9


def test_route_landmarks_and_align():
    """地図の下りの区間 2 本を見つけ、目印の時刻に合わせて距離を割り当てる (ncst_route_grade)。"""
    import ncst_route_grade as RG

    s = np.arange(0, 30000, 50.0)
    # 上り → 下り 1 (5〜13 km、-5%) → 平ら → 下り 2 (19〜25 km、-4%) → 平ら
    elev = np.interp(s, [0, 5000, 13000, 19000, 25000, 30000], [1200, 1300, 900, 900, 660, 660])
    R = pd.DataFrame(dict(s_m=s, elev_s=elev, lat=np.zeros_like(s), lon=np.zeros_like(s), radius_m=np.full_like(s, 1e4)))
    R["grade"] = np.gradient(elev, s)
    cfg = dict(align=dict(descent_phases_min=[[10.0, 16.0], [20.0, 26.0]], descent_grade_pct=-2.5, descent_min_km=2.0))
    i_top, runs = RG.landmarks(R, cfg)
    assert abs(R.s_m[i_top] - 5000) <= 100
    assert len(runs) == 2
    assert abs(R.s_m[runs[1][0]] - 19000) < 800 and abs(R.s_m[runs[1][1]] - 25000) < 800
    g = pd.DataFrame(dict(min=np.arange(5.0, 27.0, 1 / 60), v=np.nan))
    G, checks, (tk, sk) = RG.align(g, R, 5.0, runs, cfg["align"])
    assert np.all(np.diff(sk) >= 0)
    k = int(np.argmin(np.abs(G["min"] - 12)))
    assert G.grade.iat[k] < -0.04  # 下り 1 の途中
