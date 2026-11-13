"""ZOD の外乱候補の時刻の定義 (Phase 2 → Phase 3 → 映像で共通に使う)。docs/zod_event_time_fix.md

1 つの候補は、次の 3 つの時刻を別々に持つ。**「推定開始 + 継続時間」で区間を作らない。**

- 推定開始 t_rise
  ピークから時刻を戻り、|値| が onset_frac × |ピーク| を下回った直後の時刻。高い状態の始まりの推定。
  閾値を超える前から続く持続型では、閾値超過区間より前になる (1〜20 s 前のこともある)。
  OXTS の区間 (block) の先頭まで下回らなければ、そこで打ち切る (rise_censored = True。本当の始まりは OXTS より前で分からない)。
  旧来の列名 t_onset は、この値と同じ (互換のため残す)。
- 閾値超過区間 [t_exceed_start, t_exceed_end]
  検出した run (配列の位置 i0〜i1、i1 は含まない) の最初のサンプルの時刻から、最後のサンプルの時刻 + 1 サンプル間隔まで。
  exceed_duration_s = t_exceed_end − t_exceed_start (= 旧来の duration_s)。
  run の前後が判定の対象の外 (block の端、方向指示器・途切れの除外、OXTS の端) なら、その側を打ち切りとする
  (exceed_start_censored / exceed_end_censored)。超過がその外まで続いていた可能性がある。
- ピーク t_peak: run の中で |値| が最大のサンプルの時刻。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DT = 0.01        # Phase 2 の配列の間隔 (OXTS 100 Hz)


def interval_fields(t: np.ndarray, blk: np.ndarray, mask: np.ndarray, i0: int, i1: int, ion: int, ip: int, dt: float = DT,
                    rise_censored: bool | None = None) -> dict:
    """配列の位置から、候補の時刻と打ち切りの印を作る。i1 は run の終わりの次の位置 (含まない)。
    rise_censored: 推定開始を戻る処理が、半分を下回る前に区間の先頭に達したか (呼ぶ側で判断して渡す)。
    None なら ion が区間の先頭か先頭の次の位置のときに打ち切りとする。"""
    if not (0 <= i0 < i1 <= len(t)):
        raise ValueError(f"run の位置が不正: i0={i0}, i1={i1}, n={len(t)}")
    if not (i0 <= ip < i1):
        raise ValueError(f"ピークが run の外: ip={ip}, run=[{i0}, {i1})")
    if ion > ip:
        raise ValueError(f"推定開始がピークより後: ion={ion}, ip={ip}")
    bm = np.where(blk == blk[i0])[0]
    b0, b1 = int(bm[0]), int(bm[-1])                       # この block の最初と最後の位置
    if blk[i1 - 1] != blk[i0] or blk[ion] != blk[i0]:
        raise ValueError("run・推定開始が block をまたいでいる")
    start_reason, end_reason = [], []
    if i0 == b0:
        start_reason.append("OXTS の区間の先頭")
    elif not mask[i0 - 1]:
        start_reason.append("判定の対象の外 (区間の端・方向指示器・途切れ・速さ・曲がり)")
    if i1 - 1 == b1:
        end_reason.append("OXTS の区間の末尾")
    elif not mask[i1]:
        end_reason.append("判定の対象の外 (区間の端・方向指示器・途切れ・速さ・曲がり)")
    if rise_censored is None:
        rise_censored = ion <= b0 + 1
    return dict(t_rise=float(t[ion]), rise_censored=bool(rise_censored),
                t_exceed_start=float(t[i0]), t_exceed_end=float(t[i1 - 1] + dt), exceed_duration_s=round(float(t[i1 - 1] + dt - t[i0]), 3),
                exceed_start_censored=bool(start_reason), exceed_start_censor_reason=" / ".join(start_reason),
                exceed_end_censored=bool(end_reason), exceed_end_censor_reason=" / ".join(end_reason),
                t_peak=float(t[ip]), oxts_block_start=float(t[b0]), oxts_block_end=float(t[b1] + dt))


def event_interval(row) -> tuple[float, float]:
    """候補の閾値超過区間。新しい列が無い (旧い実行の) 行は受け付けない (推定開始 + 継続時間で代用しない)。"""
    for k in ("t_exceed_start", "t_exceed_end"):
        v = row[k] if k in row else None
        if v is None or (isinstance(v, float) and np.isnan(v)):
            raise KeyError(f"{k} が無い。Phase 2 を event_time に対応した版で計算し直す")
    a, b = float(row["t_exceed_start"]), float(row["t_exceed_end"])
    if b < a:
        raise ValueError(f"閾値超過区間の終わりが始まりより前: {a} > {b}")
    return a, b


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return float(max(0.0, min(a1, b1) - max(a0, b0)))


def relation_to_window(a0: float, a1: float, w0: float, w1: float) -> str:
    """区間 [a0, a1] と窓 [w0, w1] (カメラなど) の関係。"""
    if a1 <= w0:
        return "窓の前"
    if a0 >= w1:
        return "窓の後"
    if a0 < w0 and a1 > w1:
        return "窓の前から後まで"
    if a0 < w0:
        return "窓の前から続く"
    if a1 > w1:
        return "窓の後まで続く"
    return "窓の中"


def add_window_columns(df: pd.DataFrame, w0, w1, prefix: str) -> pd.DataFrame:
    """候補の表に、窓 (例: カメラ) との重なりと関係の列を足す。w0, w1 はスカラーか列と同じ長さの配列。"""
    df = df.copy()
    w0 = np.broadcast_to(np.asarray(w0, float), (len(df),))
    w1 = np.broadcast_to(np.asarray(w1, float), (len(df),))
    iv = [event_interval(r) for _, r in df.iterrows()]
    df[f"{prefix}_overlap_s"] = [round(overlap(a, b, x, y), 3) for (a, b), x, y in zip(iv, w0, w1)]
    df[f"{prefix}_relation"] = [relation_to_window(a, b, x, y) for (a, b), x, y in zip(iv, w0, w1)]
    return df
