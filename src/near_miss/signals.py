"""一様グリッドへの再サンプル (L1)。

CAN の受信時刻は間隔が大きく揺れる (実測で最小 0.18 ms / 中央値 11 ms)。
生の時刻のまま微分すると、輪速の 0.01 km/h 量子化が 100 m/s^2 規模の
見かけの加速度になる。特徴量を作る前に必ず一様グリッドへ載せる。

欠測は埋めない。受信間隔が閾値を超えた区間は NaN のままにして、
下流で「検出できなかった」ことが分かるようにする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .io.canonical import Channel, SegmentData


@dataclass
class GriddedSignals:
    """一様グリッド上に載せた信号一式。"""

    df: pd.DataFrame          # 列 "t" [s] と各チャネル
    rate_hz: float
    segment_id: str
    drive_id: str
    vehicle: str
    raw_can_loaded: bool
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def t(self) -> np.ndarray:
        return self.df["t"].to_numpy()

    @property
    def dt(self) -> float:
        return 1.0 / self.rate_hz


def build_grid(t_start: float, t_end: float, rate_hz: float, edge_trim_s: float) -> np.ndarray:
    """両端を切り落とした一様時間グリッドを作る。

    端を切るのは、後段の移動平均が端で不正な値を出すため。

    グリッド点は boot time の 1/rate_hz の倍数に固定する。区間の始まりに
    合わせて刻むと、信号を 1 本足して開始時刻が数十 ms 動いただけで
    全サンプルの位置がずれ、検出結果が変わってしまう。
    位相を固定しておけば、動くのは端の 1 点だけで済む。
    """
    step = 1.0 / rate_hz
    lo = t_start + edge_trim_s
    hi = t_end - edge_trim_s
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.empty(0)
    return np.arange(np.ceil(lo / step) * step, hi, step)


def resample_continuous(t: np.ndarray, v: np.ndarray, grid: np.ndarray, max_gap_s: float) -> np.ndarray:
    """連続量を線形内挿する。受信間隔が max_gap_s を超える区間は NaN。"""
    out = np.full(grid.shape, np.nan)
    if t.size < 2:
        return out
    finite = np.isfinite(v)
    if finite.sum() < 2:
        return out
    t, v = t[finite], v[finite]

    inside = (grid >= t[0]) & (grid <= t[-1])
    out[inside] = np.interp(grid[inside], t, v)

    # グリッド点を挟む 2 つの観測の間隔が空きすぎている箇所を落とす
    idx = np.searchsorted(t, grid, side="left")
    i1 = np.clip(idx, 0, t.size - 1)
    i0 = np.clip(idx - 1, 0, t.size - 1)
    out[(t[i1] - t[i0]) > max_gap_s] = np.nan
    return out


def resample_flag(t: np.ndarray, v: np.ndarray, grid: np.ndarray, max_gap_s: float) -> np.ndarray:
    """フラグは内挿せずゼロ次ホールドで載せる。"""
    out = np.full(grid.shape, np.nan)
    if t.size == 0:
        return out
    idx = np.searchsorted(t, grid, side="right") - 1
    valid = idx >= 0
    idx_c = np.clip(idx, 0, t.size - 1)
    held = (grid - t[idx_c]) <= max_gap_s
    ok = valid & held
    out[ok] = v[idx_c][ok]
    return out


def resample_occupancy(t: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """区間内に 1 件でも発生していれば 1 とする。送信フレームの有無などに使う。"""
    out = np.zeros(grid.shape)
    if t.size == 0 or grid.size == 0:
        return out
    step = grid[1] - grid[0] if grid.size > 1 else 1.0
    bins = np.floor((t - grid[0]) / step).astype(np.int64)
    bins = bins[(bins >= 0) & (bins < grid.size)]
    out[np.unique(bins)] = 1.0
    return out


def resample_channel(ch: Channel, grid: np.ndarray, max_gap_s: float) -> np.ndarray:
    if ch.kind == "flag":
        return resample_flag(ch.t, ch.v, grid, max_gap_s)
    if ch.kind == "occupancy":
        return resample_occupancy(ch.t, grid)
    return resample_continuous(ch.t, ch.v, grid, max_gap_s)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """NaN を伝播させない移動平均。窓内が全て NaN のところだけ NaN を返す。

    端は窓が欠けるため NaN にする。build_grid の edge_trim_s と合わせて使う。
    """
    if window <= 1:
        return x.astype(float).copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window)
    valid = np.isfinite(x).astype(float)
    filled = np.where(np.isfinite(x), x, 0.0)
    num = np.convolve(filled, kernel, mode="same")
    den = np.convolve(valid, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    half = window // 2
    out[:half] = np.nan
    out[-half:] = np.nan
    return out


def window_samples(seconds: float, rate_hz: float) -> int:
    return max(1, int(round(seconds * rate_hz)))


def to_grid(segment: SegmentData, cfg: dict[str, Any]) -> GriddedSignals:
    """SegmentData を一様グリッド上の DataFrame に変換する。"""
    rs = cfg["resample"]
    t0, t1 = segment.t_span
    grid = build_grid(t0, t1, rs["rate_hz"], rs["edge_trim_s"])

    data: dict[str, np.ndarray] = {"t": grid}
    coverage: dict[str, float] = {}
    for name, ch in segment.channels.items():
        col = resample_channel(ch, grid, rs["max_gap_s"])
        data[name] = col
        coverage[name] = float(np.isfinite(col).mean()) if grid.size else 0.0

    df = pd.DataFrame(data)
    return GriddedSignals(
        df=df,
        rate_hz=float(rs["rate_hz"]),
        segment_id=segment.ref.segment_id,
        drive_id=segment.ref.drive_id,
        vehicle=segment.vehicle,
        raw_can_loaded=segment.raw_can_loaded,
        meta={
            "coverage": coverage,
            "t_start": float(t0),
            "t_end": float(t1),
            "n_samples": int(grid.size),
            "loader_notes": list(segment.notes),
            "byte_order": segment.byte_order,
        },
    )
