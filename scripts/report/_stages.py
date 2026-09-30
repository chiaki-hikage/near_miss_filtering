"""レポート系スクリプトが共通で使う、段の当て方。

`report_pipeline.py` / `report_final.py` / `report_chain.py` は答える問いが
違うので 1 本にまとめていない。ただし次の 3 つはどれも同じことをしているので、
ここに出して 1 か所で持つ。

  1. 候補・イベント CSV の読み込み
  2. L1 の当て方 (窓表の読み込みと棄却条件)
  3. 人手ラベルとの突き合わせ (区間の重なり)

**集計と出力は各スクリプトに残してある。** ここは前処理だけを持つ。

`scripts/l1/l1_l3_chain.py` からも 3 を使う。
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from l1_eval_window_size import Eval, prepare  # noqa: F401  (呼び出し側が使う)
from l1_search_by_target import build, cond_of

# 探索で決めた L1 の棄却条件。**ここを固定して通しで測る。**
# すべて窓内の min/max/range だけで作れる。
#
# これは **窓統計 (window_stats) を入力とするレポート用の L1** で、
# 生 CAN の 2 秒窓で判定する `near_miss.l1.BASE` とは別物。
# 同じ「L1」でも入力も閾値も違うので、混ぜて読まないこと。
L1 = {
    "brake_press_max": 0.2,     # 制動の強さ [MPa 等]
    "gas_range": 5.0,           # アクセル開度の変化 [%]
    "steer_range": 5.0,         # 舵角の変化 [deg]
    "steer_rate_absmax": 60.0,  # 舵角レート [deg/s]
    "ax_absmax": 1.6,           # 前後加速度 [m/s^2]
}


def load_candidates(path: Path) -> pd.DataFrame:
    """候補 CSV。セグメント番号が無い行は落とす (窓に割り当てられない)。"""
    ca = pd.read_csv(path, low_memory=False).dropna(subset=["segment"])
    ca["segment"] = ca.segment.astype(int)
    return ca


def load_events(path: Path) -> pd.DataFrame:
    """イベント CSV。窓への割り当てに要る `duration_s` をここで作る。"""
    ev = pd.read_csv(path, low_memory=False).dropna(subset=["segment"])
    ev["segment"] = ev.segment.astype(int)
    ev["duration_s"] = ev.t_end - ev.t_start
    return ev


def l1_windows(path: Path, win: float) -> pd.DataFrame:
    """窓統計を読み、指定の窓長の行だけにする。

    `build` は行ごとの列演算しかしないので、窓長で絞る前後どちらで呼んでも
    同じ結果になる。絞ってから呼ぶ形に揃えてある。
    """
    W = prepare(pd.read_csv(path, low_memory=False))
    return build(W[W.win_s == win].reset_index(drop=True))


def l1_condition(w: pd.DataFrame, spec: dict[str, float] | None = None):
    """棄却してよい窓。**安全系フラグが立った窓は必ず残す。**

    車両自身が危険と判断した瞬間で、こちらが閾値を決める余地がないため。
    """
    return cond_of(w, L1 if spec is None else spec) & ~(w.safety_flag > 0.5).to_numpy()


def match_labels(df: pd.DataFrame, lab: pd.DataFrame) -> tuple[list, list]:
    """人手ラベルを**区間の重なり**で突き合わせ、(risky, labeled) を返す。

    `t_start` は設定が変わると動くので、時刻の一致では突き合わせない。

    risky は 3 値。True / False / None (= その区間に人手ラベルが無い)。
    **None を False として数えない。** 未評価であって、ヒヤリハットでないと
    判定されたわけではない。
    """
    risky = []
    for x in df.itertuples():
        m = ((lab.drive_id == x.drive_id)
             & (lab.t_end >= x.t_start) & (lab.t_start <= x.t_end))
        risky.append(bool(lab[m].risky.any()) if m.any() else None)
    return risky, [v is not None for v in risky]
