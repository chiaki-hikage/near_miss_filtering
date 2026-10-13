"""舗装状態に対応した路面入力モデル (2 次元路面、距離座標)。

路面の高さ z(s, y) は道路に固定した座標で定義する。
  s: 交差点の入口 (手前の横断歩道を前輪が通過する位置) からの距離 [m]、進行方向が正
  y: 内側車線の中心からの横位置 [m]、右が正

ISO 8608 の定義
  変位 PSD (片側)        Gd(n) = Gd(n0) (n / n0)^-w、n0 = 0.1 cycles/m、w = 2  [m^3]
  帯域 [n1, n2] の RMS   sqrt( ∫ Gd(n) dn )                                    [m]
路面高さの RMS は合成に使った帯域の RMS であり、帯域を変えれば変わる。
この区別のため、RMS は必ず帯域と組で報告する (band_rms)。

路面の成分
  粗さ (区間内)   Σ_j A_j cos(2π (n_j s + m_j y) + φ_j)
                  A_j = sqrt(2 Gd(n_j) Δn)。m_j は [-n_j, n_j] の一様乱数 (左右方向の波数)。
                  進行方向の PSD は Gd(n) のまま。左右に Δy 離れた 2 本の軌跡の相関は
                  sinc(2 n Δy) になり、長い波長ほどそろう (部分相関が形から決まる)
  共通成分        m_j = 0 の同じ合成を割合 β で混ぜる。左右で完全にそろう (横断方向の継ぎ目に相当)
  なだらかな高さ  区間 [s_a, s_b] にわたる 1−cos の盛り上がり、高さ H (左右共通)
区間の外は ISO A 相当の弱い粗さ (gd0_outside) を置く。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N0 = 0.1
ISO8608_GD0 = {"A": 16e-6, "B": 64e-6, "C": 256e-6, "D": 1024e-6, "E": 4096e-6}


def gd(n: np.ndarray, gd0: float, w: float = 2.0) -> np.ndarray:
    return gd0 * (n / N0) ** (-w)


def band_rms(gd0: float, n1: float, n2: float, w: float = 2.0) -> float:
    """ISO 8608 の PSD を [n1, n2] で積分した RMS [m] (解析解)。"""
    if w == 2.0:
        return float(np.sqrt(gd0 * N0 ** 2 * (1 / n1 - 1 / n2)))
    n = np.linspace(n1, n2, 20000)
    return float(np.sqrt(np.trapezoid(gd(n, gd0, w), n)))


def iso_class(gd0: float) -> str:
    for k, g in ISO8608_GD0.items():
        if gd0 < 2 * g:
            return k
    return "F 以上"


@dataclass
class Zone:
    s: tuple[float, float]           # 進行方向の範囲 [m]
    y: tuple[float, float]           # 横方向の範囲 [m]
    gd0: float                       # 区間内の粗さ
    taper_m: float = 0.5

    def window(self, s: np.ndarray, y: float) -> np.ndarray:
        def edge(x, a, b):
            u = np.minimum((x - a) / self.taper_m, (b - x) / self.taper_m)
            return np.where(u <= 0, 0.0, np.where(u >= 1, 1.0, 0.5 * (1 - np.cos(np.pi * np.clip(u, 0, 1)))))
        return edge(s, *self.s) * float(edge(np.array([y]), *self.y)[0])


@dataclass
class Roughness:
    """1 つの乱数の組 (=1 つの路面) 。全走行で同じものを使う。"""
    band: tuple[float, float]
    seed: int
    n_comp: int = 600
    w: float = 2.0
    n: np.ndarray = field(init=False)
    m: np.ndarray = field(init=False)
    phase: np.ndarray = field(init=False)
    phase_common: np.ndarray = field(init=False)
    dn: float = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.n = np.linspace(self.band[0], self.band[1], self.n_comp)
        self.dn = self.n[1] - self.n[0]
        self.m = rng.uniform(-1, 1, self.n_comp) * self.n
        self.phase = rng.uniform(0, 2 * np.pi, self.n_comp)
        self.phase_common = rng.uniform(0, 2 * np.pi, self.n_comp)

    def unit(self, s: np.ndarray, y: float, common: bool) -> np.ndarray:
        """Gd0 = 1 のときの高さ。実際の高さは sqrt(Gd0) 倍。"""
        a = np.sqrt(2 * gd(self.n, 1.0, self.w) * self.dn)
        if common:
            arg = 2 * np.pi * self.n[None, :] * s[:, None] + self.phase_common[None, :]
        else:
            arg = 2 * np.pi * (self.n[None, :] * s[:, None] + self.m[None, :] * y) + self.phase[None, :]
        return (a[None, :] * np.cos(arg)).sum(axis=1)


@dataclass
class PavementModel:
    """路面モデル。kind は A / B / C。"""
    kind: str
    zones: list[Zone]
    rough: Roughness
    gd0_outside: float
    common_fraction: float = 0.0        # B・C
    crown: tuple[float, float, float] | None = None   # C: (s_a, s_b, 高さ H)

    def height(self, s: np.ndarray, y: float) -> np.ndarray:
        beta = self.common_fraction if self.kind in ("B", "C") else 0.0
        iso = self.rough.unit(s, y, common=False)
        com = self.rough.unit(s, y, common=True) if beta > 0 else 0.0
        r = np.sqrt(1 - beta) * iso + np.sqrt(beta) * com
        inside = np.zeros_like(s)
        level = np.full_like(s, np.sqrt(self.gd0_outside))
        for z in self.zones:
            w = z.window(s, y)
            level = level * (1 - w) + np.sqrt(z.gd0) * w
            inside = np.maximum(inside, w)
        h = level * r
        if self.kind == "C" and self.crown is not None:
            sa, sb, H = self.crown
            u = (s - sa) / (sb - sa)
            h = h + np.where((u >= 0) & (u <= 1), H / 2 * (1 - np.cos(2 * np.pi * u)), 0.0)
        return h


def track_inputs(model, s_front: np.ndarray, y_lane: float, wheelbase: float, track: float,
                 envelope_m: float, ds: float = 0.005) -> np.ndarray:
    """各輪の路面入力 (時刻ごと、[FL, FR, RL, RR])。

    s_front は時刻ごとの前輪の位置 (車速の積分)。後輪は wheelbase だけ後ろ、
    左右輪は y_lane ∓ track/2 の軌跡を通る。タイヤの包み込みは接地長の移動平均で近似する。
    """
    s_grid = np.arange(s_front.min() - wheelbase - 2, s_front.max() + 2, ds)
    n_env = max(1, int(round(envelope_m / ds)))
    k = np.ones(n_env) / n_env
    prof = {}
    for side, y in (("L", y_lane - track / 2), ("R", y_lane + track / 2)):
        h = model(s_grid, y) if callable(model) else model.height(s_grid, y)
        prof[side] = np.convolve(np.r_[np.full(n_env, h[0]), h, np.full(n_env, h[-1])], k, mode="same")[n_env:-n_env]
    out = np.zeros((len(s_front), 4))
    for i, (axle, side) in enumerate((("F", "L"), ("F", "R"), ("R", "L"), ("R", "R"))):
        s = s_front - (wheelbase if axle == "R" else 0.0)
        out[:, i] = np.interp(s, s_grid, prof[side])
    return out
