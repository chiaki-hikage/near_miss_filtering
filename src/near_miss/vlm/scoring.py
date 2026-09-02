"""VLM の判定を人手ラベルと突き合わせる (Phase 1)。

Phase 1 は**探索的**であり、性能の確定はしない。指標は必ず件数を併記し、
信頼区間は参考値として扱う。positive は 8 件しかないので、点推定を信じない。

指標の置き方
------------
negative (24 件) は **clip / episode 単位**を主指標にする。
約 900 時刻あるが、同一クリップ内の時刻は強く相関しており、独立標本として
数えると実効的な標本数を 30 倍以上に過大評価する。信頼区間はクリップ単位の
ブートストラップでのみ出す。

positive (8 件) は集約値だけでなく**イベント別の個票**を残す。
n=8 で平均を語っても意味がないので、分布をそのまま見る。
時間評価の基準は人手の t_onset_human であって、CAN 由来の t_start ではない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 区間推定 (すべて参考値)
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二項比率の Wilson 信頼区間。n が小さいので正規近似は使わない。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def fmt_count(k: int, n: int, ci: bool = True) -> str:
    """「6/8 (75.0%) [参考 41-93%]」の形。**件数を必ず前に置く。**"""
    if n == 0:
        return "0/0 (-)"
    s = f"{k}/{n} ({k / n:.1%})"
    if ci:
        lo, hi = wilson(k, n)
        s += f" [参考 {lo:.0%}-{hi:.0%}]"
    return s


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> tuple[float, float]:
    """Cohen の κ と、その標準誤差 (参考値)。"""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    if pe >= 1.0:
        return float("nan"), float("nan")
    k = (po - pe) / (1 - pe)
    se = math.sqrt(po * (1 - po) / n) / (1 - pe)
    return k, se


def bootstrap_clip(values: Sequence[float], n_boot: int = 2000,
                   seed: int = 0) -> tuple[float, float]:
    """クリップ単位のブートストラップ信頼区間 (参考値)。

    時刻単位で取ると相関を無視して区間を過小評価する。必ずクリップを単位にする。
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


# ---------------------------------------------------------------------------
# 時系列 (モード B)
# ---------------------------------------------------------------------------
@dataclass
class Series:
    """1 クリップ分の判定の時系列。"""

    event_id: str
    risky: bool
    t: list[float] = field(default_factory=list)          # boot time
    state: list[str] = field(default_factory=list)
    partial: list[bool] = field(default_factory=list)

    def sort(self) -> "Series":
        order = np.argsort(self.t)
        self.t = [self.t[i] for i in order]
        self.state = [self.state[i] for i in order]
        self.partial = [self.partial[i] for i in order]
        return self

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def span_s(self) -> float:
        return (self.t[-1] - self.t[0]) if self.n > 1 else 0.0


def alarm_runs(series: Series, states: Iterable[str], debounce: int) -> list[tuple[int, int]]:
    """警報状態が debounce 回以上続いた区間の添字。

    1 時刻のちらつきを警報として数えない。数えると誤警報エピソード数が
    判定の揺れそのものになってしまう。
    """
    st = set(states)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < series.n:
        if series.state[i] in st:
            j = i
            while j + 1 < series.n and series.state[j + 1] in st:
                j += 1
            if j - i + 1 >= debounce:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def alarm_time(series: Series, states: Iterable[str], debounce: int) -> float | None:
    """最初の警報時刻。デバウンスを満たした**区間の先頭**を返す。"""
    runs = alarm_runs(series, states, debounce)
    return series.t[runs[0][0]] if runs else None


def duration_ratio(series: Series, states: Iterable[str]) -> float:
    """非 normal だった時刻の割合。デバウンスはかけない (時間の比なので)。"""
    if series.n == 0:
        return float("nan")
    st = set(states)
    return sum(1 for s in series.state if s in st) / series.n
