#!/usr/bin/env python3
"""L1 の**データ削減率**と**イベント保持率**を、窓長を変えて測る。

I/O や展開の時間は見ない。見るのは 2 つだけ。

  削減率  : L1 が早期に落とせる**走行時間**の割合 (= 後段に渡さないデータ量)
  保持率  : 落とした結果、既存の候補イベントがどれだけ残るか

**時間で測る。** 窓の数で測ると窓長を変えたときに比較できない。

窓の判定は「その窓に入っている生値だけ」で行う。イベントは
`t_in_segment_s` と `duration_s` で窓に割り当て、**跨る窓はすべて必要**とする。
`--margin` 窓ぶん前後を残す (既定 1)。イベントは境界を跨ぐため。

  uv run python scripts/l1/eval_l1_windows.py \
      --windows out/window_stats_ccs_TOYOTA_RAV4_TSS2.csv \
      --candidates out/screen_rav4_tss2_2k/candidates.csv --sweep
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

KEY = ["drive_id", "segment"]

# 連言の探索格子。inf = その量では絞らない。
GRID = {
    "v_range":    [0.3, 0.5, 1, 2, 3, 5, np.inf],
    "steer_range": [1, 3, 5, 10, 20, np.inf],
    "ay_bound":   [0.2, 0.5, 1.0, 1.5, 2.0, np.inf],
    "ax_absmax":  [0.4, 0.8, 1.2, 1.6, 2.0, np.inf],
}


def prepare(win: pd.DataFrame) -> pd.DataFrame:
    w = win[win.get("status", "ok") == "ok"].copy()
    w["v_range"] = w.v_max - w.v_min
    w["steer_range"] = w.steer_max - w.steer_min
    # 横加速度の上界。ay_kin = v * yaw を超えない (定義から厳密)。
    w["ay_bound"] = w.v_max * np.deg2rad(w.yaw_absmax)
    return w


def event_windows(cand: pd.DataFrame, w: pd.DataFrame, win_s: float,
                  margin: int = 0) -> pd.DataFrame:
    """候補ごとに、必要な窓の番号の範囲を出す。

    候補は時間の幅を持ち、窓を跨ぐ。**跨る窓が 1 つでも落ちたら喪失**とする。

    **既定は重なる窓だけ (margin=0)。** 保持マスク側を margin 窓ぶん広げる処理と
    二重にかけると、候補の前後まで保持を要求することになり、
    実際には残っている候補まで喪失に数えてしまう。
    前後の文脈が要るのは平滑化のため (0.6 秒 + 0.25 秒) で、
    窓 1 つぶん (10 秒) は過剰。
    """
    c = cand.dropna(subset=["t_in_segment_s"]).copy()
    dur = c["duration_s"] if "duration_s" in c.columns else 0.0
    k0 = np.floor(c.t_in_segment_s / win_s).astype(int) - margin
    k1 = np.floor((c.t_in_segment_s + dur) / win_s).astype(int) + margin
    c["k0"], c["k1"] = k0.clip(lower=0), k1
    return c


class Eval:
    """1 つの窓長について、条件を何通りも試すための下ごしらえ。

    候補ごとに「必要な窓の行番号」を先に平坦化しておく。
    条件ごとに候補を回すと組合せ探索が終わらない。
    """

    def __init__(self, w: pd.DataFrame, cand: pd.DataFrame, win_s: float,
                 margin: int, event_margin: int = 0) -> None:
        self.w, self.win_s, self.margin = w, win_s, margin
        self.dur = w.dur_s.to_numpy()
        self.total = self.dur.sum()
        seg = (w.drive_id.astype(str) + "|" + w.segment.astype(str)).to_numpy()
        self.same = np.empty(len(w), bool)
        self.same[0] = False
        self.same[1:] = seg[1:] == seg[:-1]

        pos = {k: i for i, k in enumerate(zip(w.drive_id, w.segment, w.k))}
        ev = event_windows(cand, w, win_s, event_margin)
        flat, off, self.has = [], [0], []
        for d, sg, a, b in zip(ev.drive_id, ev.segment, ev.k0, ev.k1):
            # **存在する窓だけを要求する。** セグメントの外へはみ出した窓番号は
            # そもそも無いので「対象外」であって「落とされた」ではない。
            idx = [pos[(d, sg, k)] for k in range(a, b + 1) if (d, sg, k) in pos]
            flat.extend(idx)
            off.append(len(flat))
            self.has.append(bool(idx))
        self.flat = np.asarray(flat, dtype=np.int64)
        self.off = np.asarray(off, dtype=np.int64)
        self.has = np.asarray(self.has, dtype=bool)
        self.ev = ev

    def keep_mask(self, cond: np.ndarray) -> np.ndarray:
        """cond = 棄却してよい窓。margin ぶん保持を広げて返す。"""
        keep = ~cond
        for _ in range(self.margin):
            nxt = np.zeros_like(keep)
            nxt[1:] |= keep[:-1] & self.same[1:]
            nxt[:-1] |= keep[1:] & self.same[1:]
            keep = keep | nxt
        return keep

    def __call__(self, cond: np.ndarray) -> dict:
        keep = self.keep_mask(np.asarray(cond, dtype=bool))
        kept_h = self.dur[keep].sum()
        n = len(self.off) - 1
        alive = np.zeros(n, dtype=bool)
        if self.flat.size:
            k = keep[self.flat]
            starts = self.off[:-1]
            # 空の候補 (対象窓が存在しない) は reduceat が壊れるので除いて計算
            nonempty = self.off[1:] > starts
            red = np.minimum.reduceat(k, starts[nonempty]) if nonempty.any() \
                else np.zeros(0, dtype=bool)
            alive[nonempty] = red
        alive &= self.has
        return {"win_s": self.win_s,
                "削減率": 100 * (1 - kept_h / self.total),
                "残時間_h": kept_h / 3600,
                "候補": n, "保持": int(alive.sum()),
                "保持率": 100 * alive.mean() if n else np.nan,
                "_alive": alive, "_ev": self.ev}


def named(w: pd.DataFrame) -> dict[str, pd.Series]:
    """ご提示の条件と、探索で出た代表案。"""
    return {
        "Speed max<=3km/h": w.v_max <= 0.833,
        "無操作・定常 (3 条件)": (w.v_range < 0.75) & (w.steer_range < 5.0)
                                & (w.ay_bound < 3.0),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--margin", type=int, default=1, help="前後に残す窓の数")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--keep-flags", action="store_true",
                   help="安全系フラグが立った窓は無条件で残す")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    W = prepare(pd.read_csv(args.windows, low_memory=False))
    cand = pd.read_csv(args.candidates, low_memory=False).dropna(subset=["segment"])
    cand["segment"] = cand.segment.astype(int)
    lens = sorted(W.win_s.unique())
    print(f"窓長 {lens}  候補 {len(cand)} 件  margin={args.margin}"
          + ("  安全フラグ窓は常に残す" if args.keep_flags else ""))

    print(f"\n{'条件':<22}{'窓[s]':>7}{'削減率':>8}{'保持':>8}{'保持率':>8}")
    for name, fn in (("Speed max<=3km/h", lambda w: w.v_max <= 0.833),
                     ("無操作・定常 (3 条件)",
                      lambda w: (w.v_range < 0.75) & (w.steer_range < 5.0)
                      & (w.ay_bound < 3.0))):
        for L in lens:
            w = W[W.win_s == L].reset_index(drop=True)
            c = fn(w).fillna(False).to_numpy()
            if args.keep_flags:
                c &= ~(w.safety_flag > 0.5).to_numpy()
            r = Eval(w, cand, L, args.margin)(c)
            print(f"{name if L == lens[0] else '':<22}{L:>7g}{r['削減率']:>7.1f}%"
                  f"{r['保持']:>8}{r['保持率']:>7.1f}%")
        print()

    if args.sweep:
        _sweep(W, cand, lens, args)
    if args.labels:
        _labels(W, cand, lens, args)
    return 0


def _sweep(W, cand, lens, args) -> None:
    print("=== 保持率別の最大削減率 ===")
    cols = list(GRID)
    for L in lens:
        w = W[W.win_s == L].reset_index(drop=True)
        pre = {c: {t: (w[c] < t).fillna(False).to_numpy() for t in ts}
               for c, ts in GRID.items()}
        flag = (w.safety_flag > 0.5).to_numpy() if args.keep_flags else None
        E = Eval(w, cand, L, args.margin)
        rows = []
        for combo in itertools.product(*GRID.values()):
            m = pre[cols[0]][combo[0]]
            for c, t in zip(cols[1:], combo[1:]):
                m = m & pre[c][t]
            if flag is not None:
                m = m & ~flag
            if not m.any():
                continue
            r = E(m)
            rows.append((r["削減率"], r["保持率"]) + combo)
        R = pd.DataFrame(rows, columns=["削減率", "保持率"] + cols)
        f = lambda x: "-" if not np.isfinite(x) else f"{x:g}"  # noqa: E731
        print(f"\n  窓 {L:g} 秒")
        print(f"  {'保持率下限':>10}{'削減率':>8}   " + " / ".join(cols))
        for lim in (100.0, 99.0, 98.0, 95.0, 90.0, 80.0):
            s = R[R.保持率 >= lim].sort_values("削減率", ascending=False)
            if s.empty:
                continue
            r = s.iloc[0]
            print(f"  {lim:>9.0f}%{r['削減率']:>7.1f}%   "
                  + " / ".join(f(r[c]) for c in cols))


def _labels(W, cand, lens, args) -> None:
    lab = pd.read_csv(args.labels)
    if "risky" not in lab.columns:
        return
    print("\n=== 人手ラベルの保持 ===")
    lab = lab.dropna(subset=["t_in_segment_s"]).copy()
    lab["duration_s"] = lab.t_end - lab.t_start
    print(f"{'条件':<22}{'窓[s]':>7}{'削減率':>8}{'risky保持':>11}{'全32保持':>10}")
    for name, fn in (("無操作・定常 (3 条件)",
                      lambda w: (w.v_range < 0.75) & (w.steer_range < 5.0)
                      & (w.ay_bound < 3.0)),):
        for L in lens:
            w = W[W.win_s == L].reset_index(drop=True)
            c = fn(w).fillna(False).to_numpy()
            if args.keep_flags:
                c &= ~(w.safety_flag > 0.5).to_numpy()
            r = Eval(w, lab, L, args.margin)(c)
            alive, ev = r["_alive"], r["_ev"]
            rk = ev.risky.astype(bool).to_numpy()
            print(f"{name if L == lens[0] else '':<22}{L:>7g}{r['削減率']:>7.1f}%"
                  f"{int(alive[rk].sum()):>7}/{int(rk.sum()):<3}"
                  f"{int(alive.sum()):>7}/{len(alive):<3}")
    return


if __name__ == "__main__":
    raise SystemExit(main())
