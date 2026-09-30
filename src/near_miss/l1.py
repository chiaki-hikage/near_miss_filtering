"""L1 軽量フィルタの固定条件。

生 CAN の 2 秒窓の max / min だけで判定する。平滑化・微分・チャネル間の時刻同期・
派生特徴量は使わない。**閾値は車種ごとに調整しない。**

基本候補は `BASE` (高圧縮)。comma2k19 で選び、commaCarSegments の 4 車種
587 時間に当てて通過率 5.76% / 60 秒 segment 45.2% を確認した値をそのまま使う。
`ALTERNATIVES` は比較用に残してあるだけで、既定では使わない。

詳細と根拠は docs/l1_filter.md。
"""

from __future__ import annotations

from typing import Any

import numpy as np

WINDOW_S = 2.0

# ---- 基本候補 (固定) --------------------------------------------------------
# v >= 5 m/s かつ (ax <= -2.5 m/s^2 または |ay| >= 2.0 m/s^2)
BASE_NAME = "高圧縮"
BASE: dict[str, float] = {"ax": -2.5, "ay": 2.0, "v_min": 5.0}

# 比較用。採用しない。
ALTERNATIVES: dict[str, dict[str, float]] = {
    "標準": {"ax": -2.0, "ay": 2.0, "v_min": 5.0},
    "1段階厳": {"ax": -3.0, "ay": 2.5, "v_min": 5.0},
}

# ---- 無条件通過にする制御・安全系フラグ ------------------------------------
# 速度・加速度の条件に関係なく通す。TRC 単独の旋回加速も L1 では落とさない。
PASS_FLAGS: tuple[str, ...] = ("abs_active_flag", "vsc_active_flag", "precollision_active")
# slip_warn は多値。3 = 始動時ランプ試験、4 = 実作動、7 = 継続点灯 (車種で意味が違う)。
SLIP_WARN_RANGE: tuple[float, float] = (3.5, 4.5)
# 立ち方は数えるが通過条件には使わない。
WATCH_FLAGS: tuple[str, ...] = ("tc_disabled", "abs_fault")

# ---- 定常区間 (通過率ではなく「decode を省略できる走行時間率」) -------------
STEADY_WIN_S = 10.0
STEADY_DELTA = 1.0


def window_edges(t0: float, t1: float, window_s: float = WINDOW_S) -> np.ndarray | None:
    """セグメント先頭を起点にした 2 秒窓の境界。Spark の floor(t / 2) と同じ割り方。"""
    if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
        return None
    e = np.arange(t0, t1 + window_s, window_s)
    return e if e.size >= 2 else None


def window_minmax(t: np.ndarray, v: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """窓ごとの min / max。各チャネルを**自分のタイムスタンプだけ**で振り分ける。

    チャネル間の時刻結合 (as-of join) を避けるためにこの形にしてある。
    """
    n = len(edges) - 1
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    if t.size == 0:
        return lo, hi
    idx = np.searchsorted(edges, t, side="right") - 1
    ok = (idx >= 0) & (idx < n) & np.isfinite(v)
    if not ok.any():
        return lo, hi
    i, x = idx[ok], v[ok]
    o = np.argsort(i, kind="stable")
    i, x = i[o], x[o]
    starts = np.flatnonzero(np.r_[True, np.diff(i) != 0])
    for s, e in zip(starts, np.r_[starts[1:], i.size]):
        lo[i[s]] = x[s:e].min()
        hi[i[s]] = x[s:e].max()
    return lo, hi


def window_stats(channels: dict[str, Any], edges: np.ndarray) -> dict[str, np.ndarray]:
    """L1 が使う窓統計だけを作る。使うチャネルは 4 アドレスぶんしかない。"""
    n = len(edges) - 1

    def mm(name: str) -> tuple[np.ndarray, np.ndarray]:
        c = channels.get(name)
        if c is None:
            return np.full(n, np.nan), np.full(n, np.nan)
        return window_minmax(c.t, c.v, edges)

    ax_lo, _ = mm("accel_x")
    ay_lo, ay_hi = mm("accel_y")
    sp_lo, sp_hi = mm("speed_mps")

    flag = np.zeros(n, dtype=bool)
    for f in PASS_FLAGS:
        _, hi = mm(f)
        flag |= np.where(np.isfinite(hi), hi > 0.5, False)
    _, sw_hi = mm("slip_warn")
    flag |= np.where(np.isfinite(sw_hi),
                     (sw_hi > SLIP_WARN_RANGE[0]) & (sw_hi <= SLIP_WARN_RANGE[1]), False)

    return {"ax_min": ax_lo, "ay_absmax": np.fmax(np.abs(ay_lo), np.abs(ay_hi)),
            "speed_min": sp_lo, "speed_max": sp_hi, "flag": flag}


def window_pass(stats: dict[str, np.ndarray],
                cfg: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(運動量だけの通過, フラグ込みの通過)。

    フラグが立った窓は速度・加速度に関係なく通す。
    """
    c = BASE if cfg is None else cfg
    gate = np.where(np.isfinite(stats["speed_max"]), stats["speed_max"] >= c["v_min"], False)
    kin = gate & (np.where(np.isfinite(stats["ax_min"]), stats["ax_min"] <= c["ax"], False)
                  | np.where(np.isfinite(stats["ay_absmax"]), stats["ay_absmax"] >= c["ay"], False))
    return kin, kin | stats["flag"]


def steady_mask(stats: dict[str, np.ndarray], win_s: float = STEADY_WIN_S,
                delta: float = STEADY_DELTA, window_s: float = WINDOW_S) -> np.ndarray:
    """車速レンジが小さい定常区間。**通過率を下げる処理ではない。**

    後段の decode / イベント判定を省略できる走行時間を数えるためのもの。
    イベントトリガが立つ窓は呼び出し側で必ず残すこと。
    """
    n = stats["speed_max"].size
    k = max(1, int(round(win_s / window_s)))
    lo = np.pad(np.where(np.isfinite(stats["speed_min"]), stats["speed_min"], np.inf),
                k // 2, mode="edge")
    hi = np.pad(np.where(np.isfinite(stats["speed_max"]), stats["speed_max"], -np.inf),
                k // 2, mode="edge")
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        rng = hi[i:i + k].max() - lo[i:i + k].min()
        out[i] = np.isfinite(rng) and rng < delta
    return out
