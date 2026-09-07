"""因果な CAN 文脈の組み立て。

評価時刻 t において「その時点で参照してよい CAN だけ」を表にする。
オンライン判定 (モード B) で未来を渡さないための要になる層。

なぜ guard が要るか
-------------------
features.py の特徴量はすべて中心合わせで計算されている。

    derivative()      中心差分          (x[i+1] - x[i-1]) / 2dt
    rolling()         center=True
    moving_average()  対称カーネル

したがって時刻 t の ax_mps2 は t+0.25 秒までのサンプルを見ている。積算すると

    v_mps          MA 0.25 s                 ±0.10 s
    steer_deg_s    MA 0.15 s                 ±0.05 s
    ax_mps2        v の中心差分 -> MA 0.25 s  ±0.25 s
    ay_kin_mps2    v x yaw -> MA 0.25 s      ±0.20 s
    jerk_mps3      微分 2 回                  ±0.30 s
    *_win_*        + rolling(center) 0.3 s   ±0.45 s

これをそのまま渡すと「未来を与えていない」という前提が静かに崩れる。

対処は 2 方式。

    guard  : CAN を t - guard_s までに切る。**既存の compute_features を
             一切変更しない**ので、抽出ロジックへの副作用がない。
             代償は CAN が guard_s だけ古くなること。
    causal : 後ろ向きフィルタで再計算する。遅れは無いが実装と検証が増える。

Phase 1 は guard で始める。評価ストライドが 0.5 秒なので遅れは 1 ステップ分に
収まる。**guard 幅は結果メタに必ず記録すること。** これが検出遅れの分解能の
下限を決めるため。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .windows import can_sample_times

# ---------------------------------------------------------------------------
# 列ごとの未来参照量
# ---------------------------------------------------------------------------
# features.py の各列が「時刻 t の値を出すのに t+x までのサンプルを見ている」量。
# **detection.yaml の smoothing から導く。** 直値で書くと、平滑化窓を変えたときに
# 静かにずれる。導出は features.py の実装そのものに対応する。
#
#   moving_average(win)  対称カーネル (偶数なら +1 して奇数化) -> 片側 win//2
#   derivative()         中心差分                              -> 片側 1 サンプル
#
# 実測 (20 Hz / 現行の smoothing) と一致することを確認済み:
#   yaw_rate_dps / ay_can_mps2 / brake_pressed  0.00 s   生値のまま
#   steer_deg_s                                 0.05 s
#   v_mps / steer_rate_dps / thw_s / ttc_s      0.10 s
#   ay_kin_mps2                                 0.20 s
#   ax_mps2                                     0.25 s
#   jerk_mps3                                   0.30 s


def _half_s(seconds: float, rate_hz: float) -> float:
    """対称移動平均の片側の広がり [秒]。signals.moving_average と同じ丸め方。"""
    from ..signals import window_samples   # signals.moving_average と同じ関数を使う

    w = window_samples(float(seconds), rate_hz)
    if w % 2 == 0:
        w += 1
    return (w // 2) / rate_hz


def column_lookahead_s(det_cfg: dict[str, Any], rate_hz: float) -> dict[str, float]:
    """列ごとの未来参照量 [秒] を smoothing 設定から導く。"""
    sm = det_cfg["smoothing"]
    d = 1.0 / rate_hz                       # 中心差分の片側
    v = _half_s(sm["speed_window_s"], rate_hz)
    acc = _half_s(sm["accel_window_s"], rate_hz)
    st = _half_s(sm["steer_window_s"], rate_hz)
    ax = v + d + acc
    return {
        # 生値。グリッドに載せただけで加工していない
        "yaw_rate_dps": 0.0, "ay_can_mps2": 0.0, "brake_pressed": 0.0,
        "steer_deg_s": st,
        "steer_rate_dps": st + d,
        "v_mps": v,
        "thw_s": v, "ttc_s": v,             # ego_v = v_mps 由来。レーダ側は因果
        "lead_distance_m": 0.0,
        "ay_kin_mps2": v + acc,
        "ax_mps2": ax,
        "jerk_mps3": ax + d,
    }



@dataclass
class ContextRender:
    """組み立てた CAN 文脈。

    text だけでなく max_source_t を返すのは、**因果性を文字列の目視ではなく
    値で検査できるようにする**ため。tests/test_vlm_causality.py が使う。
    """

    text: str
    max_source_t: float           # 実際に読んだグリッド時刻の最大。NaN なら何も読めていない
    n_rows: int
    missing: int = 0
    columns: list[str] = field(default_factory=list)


class CanContext(Protocol):
    """評価時刻 t で参照可能な CAN だけを返す。

    実装を差し替えても呼び出し側は変えない。mode / guard_s を属性として
    公開するのは、結果メタへそのまま書き出すため。
    """

    mode: str
    guard_s: float

    def at(self, t: float) -> ContextRender: ...


class GuardContext:
    """guard 方式。既存の中心合わせ特徴量をそのまま使い、t - guard_s で切る。

    df は 20 Hz グリッド (compute_features 済み) で、列 "t" が boot time。
    各サンプル時刻 tau に対して **tau 以下で最も新しいグリッド行**を採る。
    内挿しない。内挿すると tau をまたいだ 2 点を混ぜることになり、
    因果性の主張が弱くなる。
    """

    def __init__(self, df: pd.DataFrame, cfg: dict[str, Any]) -> None:
        if "t" not in df.columns:
            raise ValueError("グリッドに列 't' がありません")
        self.mode = "guard"
        self.guard_s = float(cfg["context"]["guard_s"])
        self._cfg = cfg
        self._t = df["t"].to_numpy(dtype=float)
        self._df = df
        # 設定に書かれた列のうち、実在するものだけを使う。
        # 欠けている列を黙って 0 で埋めない (元データの品質を隠さない)。
        self._spec = [tuple(c) for c in cfg["context"]["columns"]
                      if str(c[0]) in df.columns]
        self._missing_cols = [str(c[0]) for c in cfg["context"]["columns"]
                              if str(c[0]) not in df.columns]

    @property
    def missing_columns(self) -> list[str]:
        return list(self._missing_cols)

    def at(self, t: float) -> ContextRender:
        times = can_sample_times(t, self._cfg)
        # tau 以下で最も新しい行。searchsorted(right) - 1 がその位置。
        idx = np.searchsorted(self._t, np.asarray(times), side="right") - 1

        head = ["時刻"] + [f"{name}[{unit}]" for _c, name, unit, _d, _s in self._spec]
        lines = ["  ".join(head)]
        used: list[float] = []
        missing = 0

        for tau, i in zip(times, idx):
            rel = tau - t          # 評価時刻からの相対秒。常に負
            if i < 0:
                lines.append(f"{rel:+6.2f}" + "  -" * len(self._spec))
                missing += 1
                continue
            used.append(float(self._t[i]))
            cells = [f"{rel:+6.2f}"]
            for col, _name, _unit, digits, scale in self._spec:
                v = self._df[col].to_numpy(dtype=float)[i]
                cells.append("-" if not np.isfinite(v) else f"{v * float(scale):.{int(digits)}f}")
            lines.append("  ".join(cells))

        return ContextRender(
            text="\n".join(lines),
            max_source_t=max(used) if used else float("nan"),
            n_rows=len(times),
            missing=missing,
            columns=[c[0] for c in self._spec],
        )


    def span(self, t0: float, t1: float) -> ContextRender:
        """区間全体を表にする。**モード A (一括判定) 専用。**

        モード B では使わないこと。オンライン判定に区間全体を渡せば未来を
        与えることになる。モード A は区間全体を見るのが前提なので、
        極値の要約 (extremes) も併せて使える。
        """
        rate = float(self._cfg["input"]["can_rate_hz"])
        n = max(2, int(round((t1 - t0) * rate)) + 1)
        times = [t0 + i * (t1 - t0) / (n - 1) for i in range(n)]
        idx = np.searchsorted(self._t, np.asarray(times), side="right") - 1

        head = ["時刻"] + [f"{name}[{unit}]" for _c, name, unit, _d, _s in self._spec]
        lines = ["  ".join(head)]
        used: list[float] = []
        missing = 0
        for tau, i in zip(times, idx):
            rel = tau - t0
            if i < 0:
                lines.append(f"{rel:6.2f}" + "  -" * len(self._spec))
                missing += 1
                continue
            used.append(float(self._t[i]))
            cells = [f"{rel:6.2f}"]
            for col, _name, _unit, digits, scale in self._spec:
                v = self._df[col].to_numpy(dtype=float)[i]
                cells.append("-" if not np.isfinite(v) else f"{v * float(scale):.{int(digits)}f}")
            lines.append("  ".join(cells))
        return ContextRender(text="\n".join(lines),
                             max_source_t=max(used) if used else float("nan"),
                             n_rows=len(times), missing=missing,
                             columns=[c[0] for c in self._spec])

    def extremes(self, t0: float, t1: float) -> str:
        """区間の極値。**モード A 専用。** モード B に渡せば未来を与えることになる。"""
        m = (self._t >= t0) & (self._t <= t1)
        if not m.any():
            return ""
        out = []
        for col, name, unit, digits, scale in self._spec:
            v = self._df[col].to_numpy(dtype=float)[m] * float(scale)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            out.append(f"{name} 最小 {v.min():.{int(digits)}f} / 最大 {v.max():.{int(digits)}f} {unit}")
        return " / ".join(out)


class PerColumnGuardContext:
    """列ごとに必要最小限だけ手前で切る。

    一律 guard は、最も未来を見ている列 (ax_mps2 の 0.25 秒) に全列を合わせる。
    生値のまま使える yaw_rate / ay_can / brake まで 0.5 秒古くする必要はない。

    行の時刻は t まで刻み、**各セルは「その行の時刻より後を参照しない最新の値」**を
    採る。列 c の値は s + lookahead[c] <= tau を満たす最大の s から採るので、
    セル単位で因果性が保たれる。

    代償として、同じ行でも列によって値の鮮度が違う。実運用のセンサ遅延と同じ状況で、
    プロンプトには列ごとの遅れを明記する。
    """

    mode = "per_column"

    def __init__(self, df: pd.DataFrame, cfg: dict[str, Any],
                 det_cfg: dict[str, Any], rate_hz: float) -> None:
        if "t" not in df.columns:
            raise ValueError("グリッドに列 't' がありません")
        self._cfg = cfg
        self._df = df
        self._t = df["t"].to_numpy(dtype=float)
        self._look = column_lookahead_s(det_cfg, rate_hz)
        self._spec = [tuple(c) for c in cfg["context"]["columns"]
                      if str(c[0]) in df.columns]
        self._missing_cols = [str(c[0]) for c in cfg["context"]["columns"]
                              if str(c[0]) not in df.columns]
        # 表としての一律 guard は無いが、最も遅い列の遅れは記録しておく
        self.guard_s = max((self._look.get(c[0], 0.0) for c in self._spec), default=0.0)

    @property
    def missing_columns(self) -> list[str]:
        return list(self._missing_cols)

    @property
    def lookahead(self) -> dict[str, float]:
        return {c[0]: self._look.get(c[0], 0.0) for c in self._spec}

    def latency_note(self) -> str:
        """列ごとの遅れをプロンプトに添える一文。"""
        parts = [f"{name} {self._look.get(col, 0.0):.2f}"
                 for col, name, _u, _d, _s in self._spec
                 if self._look.get(col, 0.0) > 0]
        return ("各信号は処理の都合で遅れがあります (秒): " + " / ".join(parts)
                if parts else "")

    def at(self, t: float) -> ContextRender:
        # 一律 guard を引かず、行は t まで刻む
        cfg2 = dict(self._cfg)
        cfg2["context"] = dict(self._cfg["context"], guard_s=0.0)
        times = can_sample_times(t, cfg2)

        head = ["時刻"] + [f"{name}[{unit}]" for _c, name, unit, _d, _s in self._spec]
        lines = ["  ".join(head)]
        used: list[float] = []
        missing = 0
        for tau in times:
            cells = [f"{tau - t:+6.2f}"]
            row_missing = False
            for col, _name, _unit, digits, scale in self._spec:
                limit = tau - self._look.get(col, 0.0)
                i = int(np.searchsorted(self._t, limit, side="right")) - 1
                if i < 0:
                    cells.append("-")
                    row_missing = True
                    continue
                used.append(float(self._t[i]))
                v = self._df[col].to_numpy(dtype=float)[i]
                cells.append("-" if not np.isfinite(v) else f"{v * float(scale):.{int(digits)}f}")
            missing += int(row_missing)
            lines.append("  ".join(cells))

        return ContextRender(
            text="\n".join(lines),
            max_source_t=max(used) if used else float("nan"),
            n_rows=len(times), missing=missing,
            columns=[c[0] for c in self._spec],
        )


class CausalContext:
    """causal 方式 (未実装)。後ろ向きフィルタで特徴量を作り直す。

    guard を外して遅れをゼロにするための差し替え先。features.py には
    手を入れず、ここで後方移動平均と後退差分を実装する。

    Phase 1 では使わない。差し替え点を明示しておくために型だけ置く。
    """

    mode = "causal"

    def __init__(self, df: pd.DataFrame, cfg: dict[str, Any]) -> None:
        raise NotImplementedError(
            "causal 文脈は未実装です。configs/vlm.yaml の context.mode を guard にしてください"
        )


def make_context(df: pd.DataFrame, cfg: dict[str, Any],
                 det_cfg: dict[str, Any] | None = None,
                 rate_hz: float | None = None) -> CanContext:
    mode = str(cfg["context"]["mode"])
    if mode == "guard":
        return GuardContext(df, cfg)
    if mode == "per_column":
        if det_cfg is None or rate_hz is None:
            raise ValueError("per_column には detection 設定とグリッド周波数が要ります")
        return PerColumnGuardContext(df, cfg, det_cfg, rate_hz)
    if mode == "causal":
        return CausalContext(df, cfg)
    raise ValueError(f"未知の context.mode: {mode}")
