#!/usr/bin/env python3
"""L1 (前段の粗い除去) の削減率と取りこぼしを実測する。

summarize_features.py が出した**生チャネルの要約統計**と、既存の
**candidates.csv** を突き合わせるだけ。推論も再計算もしない。

  削減率  : L1 が棄却するセグメントの割合
  取りこぼし: 棄却したセグメントに入っていた既存候補の数

**候補はセグメント境界を跨ぐ。** 連結ブロックが分断されると継ぎ目の結果が
変わるため、棄却は「連続して失格した run」に限り、前後 margin セグメントを
必ず残す (既定 1)。

  uv run python scripts/eval_l1.py \
      --stats out/feature_stats_ccs_TOYOTA_RAV4_TSS2.csv \
      --candidates out/screen_rav4_tss2_2k/candidates.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

KEY = ["drive_id", "segment"]


def load_raw(path: Path) -> pd.DataFrame:
    """生チャネルの統計を「セグメント 1 行」の横持ちに直す。"""
    d = pd.read_csv(path, low_memory=False)
    bad = d[d.status != "ok"]
    d = d[(d.status == "ok") & (d.kind == "raw")]
    segs = d[KEY].drop_duplicates().set_index(KEY).index
    W = pd.DataFrame(index=segs)

    def take(col: str, stat: str):
        s = d[d.column == col]
        return s.set_index(KEY)[stat] if stat in s.columns else pd.Series(dtype=float)

    # 速度は輪速平均を第一に、無ければ個々の輪速に落とす
    for name, cands, stat in (
        ("v_max", ("speed_mps", "ws_fl_mps", "ws_fl_can_mps"), "max"),
        ("v_min", ("speed_mps", "ws_fl_mps", "ws_fl_can_mps"), "min"),
        ("yaw_absmax", ("yaw_rate",), "absmax"),
        ("steer_range", ("steer_deg", "steer_can_deg"), "range"),
        ("ay_absmax", ("accel_y",), "absmax"),
        ("ax_absmax", ("accel_x",), "absmax"),
        ("gvc_absmax", ("gvc_mps2",), "absmax"),
        ("gas_max", ("gas_pedal_pct",), "max"),
        ("brake_on", ("brake_pressed",), "on_ratio"),
    ):
        col = pd.Series(np.nan, index=segs)
        for c in cands:
            v = take(c, stat)
            if len(v):
                col = col.fillna(v)
        W[name] = col
    W["v_range"] = W.v_max - W.v_min
    # 横加速度の上界。ay_kin = v * yaw なので v_max * max|yaw| を超えない。
    W["ay_bound"] = W.v_max * np.deg2rad(W.yaw_absmax)
    return W, bad


def load_candidates(path: Path) -> pd.DataFrame:
    c = pd.read_csv(path, low_memory=False)
    if "segment" not in c.columns:
        raise SystemExit("candidates.csv に segment 列がありません")
    return c.dropna(subset=["segment"]).astype({"segment": int})


class Protector:
    """棄却マスクを run 単位にし、前後 margin セグメントを残す。

    連結ブロックの分断で継ぎ目の結果が変わるのを避けるため。
    margin=0 なら素の棄却。

    **並び替えとドライブ境界は 1 度だけ求めて使い回す。**
    条件を何千通りも試すので、毎回 groupby すると探索が終わらない。
    """

    def __init__(self, index: pd.MultiIndex, margin: int) -> None:
        self.margin = int(margin)
        df = index.to_frame(index=False)
        df.columns = ["drive_id", "segment"]
        self.order = np.lexsort((df.segment.to_numpy(), df.drive_id.to_numpy()))
        d = df.drive_id.to_numpy()[self.order]
        # ドライブが変わる位置。ここを跨いで隣接扱いしない。
        self.same_prev = np.empty(len(d), dtype=bool)
        self.same_prev[0] = False
        self.same_prev[1:] = d[1:] == d[:-1]
        self.inv = np.argsort(self.order)

    def __call__(self, rej: pd.Series) -> pd.Series:
        m = rej.to_numpy(dtype=bool)[self.order]
        if self.margin <= 0:
            return pd.Series(m[self.inv], index=rej.index)
        keep = ~m
        grow = keep.copy()
        for _ in range(self.margin):
            nxt = np.zeros_like(grow)
            nxt[1:] |= grow[:-1] & self.same_prev[1:]     # 前から広げる
            nxt[:-1] |= grow[1:] & self.same_prev[1:]     # 後ろから広げる
            grow |= nxt
        return pd.Series((m & ~grow)[self.inv], index=rej.index)


def protect(rej: pd.Series, margin: int) -> pd.Series:
    return Protector(rej.index, margin)(rej)


def report(name: str, rej: pd.Series, W: pd.DataFrame, cand: pd.DataFrame,
           margin: int) -> dict:
    rej = rej.fillna(False)
    r2 = protect(rej, margin)
    idx = pd.MultiIndex.from_frame(cand[KEY])
    lost = idx.isin(r2[r2].index)
    n = len(W)
    row = {
        "条件": name,
        "棄却": int(rej.sum()), "棄却率": 100 * rej.mean(),
        f"余白{margin}後": int(r2.sum()), "実削減率": 100 * r2.mean(),
        "候補喪失": int(lost.sum()), "喪失率": 100 * lost.mean(),
    }
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None,
                   help="人手ラベル (risky 列) があれば、その喪失も出す")
    p.add_argument("--sweep", action="store_true",
                   help="連言の閾値を総当たりし、recall 別の最大削減率を出す")
    p.add_argument("--margin", type=int, default=1,
                   help="棄却 run の前後に残すセグメント数 (既定 1)")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    W, bad = load_raw(args.stats)
    cand = load_candidates(args.candidates)
    n = len(W)
    have = cand.set_index(KEY).index.isin(W.index)
    print(f"セグメント {n}  候補 {len(cand)} 件 "
          f"(統計に対応が付いたもの {int(have.sum())})")
    if len(bad):
        print(f"  読み出し異常 {len(bad)} 件: {bad.status.value_counts().to_dict()}")
    print(f"  余白 margin={args.margin}\n")

    rows = []
    # --- ご提示の条件 ---
    rows.append(report("Speed max <= 3 km/h", W.v_max <= 0.833, W, cand, args.margin))
    rows.append(report("Speed max <= 5 km/h", W.v_max <= 1.389, W, cand, args.margin))
    steady = (W.v_range < 0.75) & (W.steer_range < 5.0) & (W.ay_bound < 3.0)
    rows.append(report("無操作・定常 (3 条件)", steady, W, cand, args.margin))
    rows.append(report("上記いずれか", (W.v_max <= 0.833) | steady, W, cand, args.margin))

    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:5.1f}"))

    # --- 1 条件ずつの安全上限を実測する ---
    print("\n=== 各量の「候補を 1 件も落とさない」上限 (実測) ===")
    print("候補があったセグメントでの最小値/最大値から、安全側の閾値を出す。\n")
    cidx = pd.MultiIndex.from_frame(cand[KEY].drop_duplicates())
    has = W.index.isin(cidx)
    print(f"候補ありセグメント {int(has.sum())} / 候補なし {int((~has).sum())}\n")
    print(f"{'量':<14}{'候補ありの下限':>14}{'安全閾値':>10}{'それで棄却':>12}{'削減率':>8}")
    for col, direction in (("v_max", "lt"), ("v_range", "lt"), ("steer_range", "lt"),
                           ("ay_bound", "lt"), ("yaw_absmax", "lt"),
                           ("ax_absmax", "lt"), ("gvc_absmax", "lt")):
        v = W[col]
        with_c = v[has].dropna()
        if with_c.empty:
            continue
        lim = with_c.min()
        rej = (v < lim).fillna(False)
        r2 = protect(rej, args.margin)
        print(f"{col:<14}{lim:>14.3f}{lim:>10.3f}{int(r2.sum()):>12}{100 * r2.mean():>7.1f}%")

    if args.labels:
        _labels(args, W, cand)
    if args.sweep:
        _sweep(args, W, cand)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\n出力: {args.out}")
    return 0


# --- 総当たり ---------------------------------------------------------------
# 「静かな区間ほど候補が無い」を連言で書く。**閾値の上限は制約なし (無限大) も
# 含める。** ある量が絞り込みに寄与しないことも結果として示せるようにするため。
GRID = {
    "v_range":     [0.3, 0.5, 1, 2, 3, 4, 6, 8, np.inf],
    "steer_range": [0.7, 2, 5, 10, 15, 20, 30, np.inf],
    "ay_bound":    [0.16, 0.5, 0.8, 1.2, 1.6, 2.0, np.inf],
    "ax_absmax":   [0.4, 0.8, 1.0, 1.2, 1.6, 2.0, np.inf],
}


def _fmt(x: float) -> str:
    return "-" if not np.isfinite(x) else f"{x:g}"


def _sweep(args, W: pd.DataFrame, cand: pd.DataFrame) -> None:
    import itertools

    P = Protector(W.index, args.margin)
    cidx = pd.MultiIndex.from_frame(cand[KEY])
    n = len(cand)
    pre = {c: {t: (W[c] < t).fillna(False).to_numpy() for t in ts}
           for c, ts in GRID.items()}
    cols = list(GRID)
    rows = []
    for combo in itertools.product(*GRID.values()):
        mask = pre[cols[0]][combo[0]]
        for c, t in zip(cols[1:], combo[1:]):
            mask = mask & pre[c][t]
        if not mask.any():
            continue
        r2 = P(pd.Series(mask, index=W.index))
        lost = int(cidx.isin(r2[r2].index).sum())
        rows.append((100 * r2.mean(), 100 * (1 - lost / n), lost) + combo)
    R = pd.DataFrame(rows, columns=["削減率", "recall", "喪失"] + cols)

    print(f"\n=== recall 別の最大削減率 (連言 / margin={args.margin} / "
          f"{n} 候補が基準 / {len(R)} 通り) ===")
    print(f"{'recall下限':>9}{'削減率':>8}{'喪失':>6}   " + " / ".join(cols))
    for lim in (100.0, 99.5, 99.0, 98.0, 95.0, 90.0, 85.0, 80.0):
        s = R[R.recall >= lim].sort_values("削減率", ascending=False)
        if s.empty:
            continue
        r = s.iloc[0]
        print(f"{lim:>8.1f}%{r['削減率']:>7.1f}%{int(r['喪失']):>6}   "
              + " / ".join(_fmt(r[c]) for c in cols))


def _labels(args, W: pd.DataFrame, cand: pd.DataFrame) -> None:
    """人手ラベルの喪失を見る。**候補の喪失より、こちらが本命。**"""
    lab = pd.read_csv(args.labels)
    if "risky" not in lab.columns:
        print("\nlabels に risky 列がありません")
        return
    P = Protector(W.index, args.margin)
    lidx = pd.MultiIndex.from_frame(lab[KEY])
    ridx = pd.MultiIndex.from_frame(lab[lab.risky.astype(bool)][KEY])
    cidx = pd.MultiIndex.from_frame(cand[KEY])
    print(f"\n=== 人手ラベル {len(lab)} 件 (risky {int(lab.risky.sum())}) の喪失 ===")
    print(f"{'条件':<28}{'削減率':>8}{'候補喪失':>9}{'ラベル喪失':>11}{'risky喪失':>10}")
    for name, c in _named_conditions(W).items():
        r2 = P(c.fillna(False))
        rej = r2[r2].index
        print(f"{name:<28}{100 * r2.mean():>7.1f}%{int(cidx.isin(rej).sum()):>9}"
              f"{int(lidx.isin(rej).sum()):>11}{int(ridx.isin(rej).sum()):>10}")


def _named_conditions(W: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "Speed max<=3km/h": W.v_max <= 0.833,
        "無操作・定常 (3 条件)": (W.v_range < 0.75) & (W.steer_range < 5.0)
                                & (W.ay_bound < 3.0),
        "recall100 案": (W.v_range < 2) & (W.ay_bound < 0.8) & (W.ax_absmax < 2),
        "recall98 案": (W.ay_bound < 0.8) & (W.ax_absmax < 2),
        "recall95 案": (W.v_range < 6) & (W.steer_range < 15) & (W.ay_bound < 2)
                       & (W.ax_absmax < 2),
        "recall90 案": (W.v_range < 8) & (W.steer_range < 20) & (W.ax_absmax < 2),
    }


if __name__ == "__main__":
    raise SystemExit(main())
