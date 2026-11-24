"""制動の評価の計算 (データセットに依らない層)。

ブレーキによる減速度の作り方 (縦の力のつり合い、質量あたり):
    dv/dt = -d_brake - g sin(theta) - d_coast(v)
    d_brake = -dv/dt - g sin(theta) - d_coast(v)
  theta: 道路の勾配 (上りが正)。**車体のピッチではない。** global_pose の速度ベクトルの向きから出す
  d_coast: ブレーキを使わずアクセルを離したときの減速 (空気・転がり・エンジンブレーキ)。走行データで当てはめる

正常制動の模型: d_brake(t + lag) = f(L(t), v)   L は制動の値 (0 点を引いたもの)
利き (効率) の比: eta = 観測の d_brake / 模型の予測。1 が平常、フェードなら 1 より下がるはず
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def ecef_to_geodetic(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ECEF (m) -> 測地緯度・経度 (deg)。Bowring の式。上向きの方向を地心でなく測地で取るために使う
    (地心緯度との差は最大 0.19 deg で、勾配に直すと 0.03 m/s^2 の誤差になる)。"""
    xyz = np.atleast_2d(xyz)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    b = WGS84_A * np.sqrt(1 - WGS84_E2)
    ep2 = (WGS84_A**2 - b**2) / b**2
    p = np.hypot(x, y)
    th = np.arctan2(z * WGS84_A, p * b)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - WGS84_E2 * WGS84_A * np.cos(th) ** 3)
    lon = np.arctan2(y, x)
    return np.degrees(lat).squeeze(), np.degrees(lon).squeeze()


def enu_basis(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    e = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], -1)
    n = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], -1)
    u = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    return e, n, u


def grade_from_ecef(pos: np.ndarray, vel: np.ndarray, min_speed: float):
    """進む向きの勾配 theta (rad、上りが正)、緯度・経度、方位 (deg、北から時計回り)。
    theta = atan2(v_up, |v_horizontal|)。速度ベクトルの向きなので、車体のピッチ (前のめり) は入らない。"""
    lat, lon = ecef_to_geodetic(pos)
    e, n, u = enu_basis(lat, lon)
    ve, vn, vu = (vel * e).sum(-1), (vel * n).sum(-1), (vel * u).sum(-1)
    vh = np.hypot(ve, vn)
    th = np.arctan2(vu, vh)
    hd = np.degrees(np.arctan2(ve, vn)) % 360
    slow = vh < min_speed
    th[slow] = np.nan
    hd[slow] = np.nan
    return th, lat, lon, hd


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def design(cols: list[str], L: np.ndarray, v: np.ndarray) -> np.ndarray:
    m = {"1": np.ones_like(L), "L": L, "L2": L * L, "Lv": L * v, "v": v, "v2": v * v}
    return np.stack([m[str(c)] for c in cols], -1)


def lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def fit_coast(v: np.ndarray, resid: np.ndarray, form: list[str]) -> np.ndarray:
    """resid = -dv/dt - g sin(theta) (ブレーキもアクセルも使っていない時刻) を車速の多項式で当てはめる。"""
    X = design(form, np.zeros_like(v), v)
    return lstsq(X, resid)


def coast(v: np.ndarray, coef: np.ndarray, form: list[str]) -> np.ndarray:
    return design(form, np.zeros_like(v), v) @ coef


def shift_by_lag(x: pd.Series, groups: pd.Series, lag_steps: int) -> pd.Series:
    """x(t + lag) を t の行に置く (塊をまたがない)。"""
    return x.groupby(groups).shift(-lag_steps) if lag_steps else x


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))


def drive_folds(drives: np.ndarray, k: int, seed: int) -> dict[str, int]:
    u = np.array(sorted(set(drives)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    return {d: i % k for i, d in enumerate(u)}


# ---------------------------------------------------------------------------
# 制動の区切り
# ---------------------------------------------------------------------------
def find_events(D: pd.DataFrame, active: np.ndarray, L: np.ndarray, cfg: dict, l_p99: float, dt: float) -> pd.DataFrame:
    """走行中の 1 回の制動 = active が続く区間。定常 (値が峰の 50% 以上) の部分で利きを測る。"""
    ec = cfg["events"]
    rows = []
    act = pd.Series(active)
    run = (act != act.shift()).cumsum() + D["block"].to_numpy() * 10**7
    for _, idx in pd.Series(np.arange(len(D)))[active].groupby(run[active]).groups.items():
        i = np.asarray(idx)
        if len(i) * dt < ec["min_duration_s"]:
            continue
        if D.v.to_numpy()[i].min() < ec["min_speed_mps"]:
            continue
        li = L[i]
        if not np.isfinite(li).all():
            continue
        peak = np.nanmax(li)
        if peak < ec["min_peak_frac"] * l_p99:
            continue
        st = i[li >= ec["steady_frac_of_peak"] * peak]
        if len(st) * dt < ec["min_steady_s"]:
            continue
        rows.append(dict(i0=int(i[0]), i1=int(i[-1]), steady=st, peak=float(peak)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 熱の指標
# ---------------------------------------------------------------------------
def heat_indices(power: np.ndarray, block: np.ndarray, dt: float, taus: list[float], windows: list[float]) -> dict[str, np.ndarray]:
    """power: ブレーキで熱になった仕事率 (W/kg)。塊 (連続した時刻) の頭で 0 から始める (欠番の前は分からない)。
    H_tau: 1 次遅れ (時定数 tau) の積分 (J/kg)。E_w: 直前 w 秒の和 (J/kg)。"""
    out = {}
    p = np.nan_to_num(power)
    starts = np.r_[True, block[1:] != block[:-1]]
    for tau in taus:
        h = np.zeros_like(p)
        a = np.exp(-dt / tau)
        acc = 0.0
        for k in range(len(p)):
            if starts[k]:
                acc = 0.0
            acc = acc * a + p[k] * dt
            h[k] = acc
        out[f"H{int(tau)}"] = h
    for w in windows:
        n = int(round(w / dt))
        e = np.zeros_like(p)
        for b in np.unique(block):
            m = np.where(block == b)[0]
            cs = np.cumsum(p[m] * dt)
            e[m] = cs - np.r_[np.zeros(min(n, len(cs))), cs[:-n]] if len(cs) > n else cs
        out[f"E{int(w)}"] = e
    return out
