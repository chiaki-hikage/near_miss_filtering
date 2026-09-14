#!/usr/bin/env python3
"""候補に高優先度フラグを付ける (L3 の後段)。**候補生成は変えない。**

`configs/priority.yaml` の規則を候補の集計値に当てるだけ。
判定の根拠 (どの規則で採用されたか) を列に残す。

  uv run python scripts/filter_priority.py out/chunk1/candidates.csv \
      --labels out/chunk1/label_features.csv
  uv run python scripts/filter_priority.py out/screen_rav4_tss2_2k/candidates.csv \
      --out out/priority_ccs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from near_miss.config import load_yaml

DEFAULT_CFG = Path("configs/priority.yaml")


def _cmp(v: pd.Series, op: str, th: float) -> pd.Series:
    """欠測は False。**「判定できない」を「条件を満たす」にしない。**"""
    x = pd.to_numeric(v, errors="coerce")
    if op == "le":
        m = x <= th
    elif op == "lt":
        m = x < th
    elif op == "ge":
        m = x >= th
    elif op == "gt":
        m = x > th
    elif op == "abs_ge":
        m = x.abs() >= th
    elif op == "abs_gt":
        m = x.abs() > th
    else:
        raise ValueError(f"未知の op: {op}")
    return m.fillna(False)


def _has_type(d: pd.DataFrame, types: list[str]) -> pd.Series:
    et = d.get("event_types", pd.Series("", index=d.index)).fillna("")
    m = pd.Series(False, index=d.index)
    for t in types:
        # 種別名は '|' 区切り。部分一致で別の種別を拾わないよう境界を見る。
        m |= et.apply(lambda s, t=t: t in str(s).split("|"))
    return m


def apply_rules(d: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series]:
    """(採用マスク, 採用理由) を返す。理由は '|' 区切りで複数入りうる。"""
    reason = pd.Series("", index=d.index)
    keep = pd.Series(False, index=d.index)

    always = pd.Series(False, index=d.index)
    for item in cfg.get("always", []):
        # 列名だけの書き方 (閾値 0.5) と、閾値つきの書き方の両方を受ける。
        # slip_warn_max は 3 がランプチェック / 4 が実作動なので閾値が要る。
        col = item if isinstance(item, str) else item["feature"]
        th = 0.5 if isinstance(item, str) else float(item["threshold"])
        if col in d.columns:
            always |= pd.to_numeric(d[col], errors="coerce").fillna(0) > th
    keep |= always
    reason = reason.where(~always, reason + "|safety")

    for rule in cfg.get("rules", []):
        m = _has_type(d, rule.get("require_types", []))
        for c in rule.get("all_of", []):
            m &= _cmp(d.get(c["feature"], pd.Series(np.nan, index=d.index)),
                      c["op"], float(c["threshold"]))
        anys = rule.get("any_of", [])
        if anys:
            a = pd.Series(False, index=d.index)
            for c in anys:
                a |= _cmp(d.get(c["feature"], pd.Series(np.nan, index=d.index)),
                          c["op"], float(c["threshold"]))
            m &= a
        keep |= m
        reason = reason.where(~m, reason + "|" + rule["name"])
    return keep, reason.str.lstrip("|")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("candidates", type=Path)
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--labels", type=Path, default=None,
                   help="risky 列を持つ表。精度と再現を出す")
    p.add_argument("--min-severity", type=float, default=0.0,
                   help="これ以下の候補は先に落とす (重み 0 のイベントだけの候補)")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    src = args.labels if args.labels else args.candidates
    d = pd.read_csv(src, low_memory=False)

    n0 = len(d)
    if args.min_severity > 0 and "severity" in d.columns:
        d = d[d.severity > args.min_severity].reset_index(drop=True)
    keep, reason = apply_rules(d, cfg)
    d["priority"] = keep
    d["priority_reason"] = reason

    print(f"{src}: {n0} 件"
          + (f" -> severity>{args.min_severity:g} で {len(d)} 件" if len(d) != n0 else ""))
    print(f"優先度採用: {int(keep.sum())} 件 ({100 * keep.mean():.0f}%)")

    print(f"\n{'規則':<26}{'採用':>6}" + ("{:>6}{:>8}".format("risky", "精度")
                                           if "risky" in d.columns else ""))
    names = ["safety"] + [r["name"] for r in cfg.get("rules", [])]
    for name in names:
        m = reason.str.split("|").apply(lambda s: name in s)
        if not m.any():
            print(f"{name:<26}{0:>6}")
            continue
        line = f"{name:<26}{int(m.sum()):>6}"
        if "risky" in d.columns:
            P = d.risky.astype(bool)
            line += f"{int(P[m].sum()):>6}{100 * P[m].mean():>7.0f}%"
        print(line)

    if "risky" in d.columns:
        P = d.risky.astype(bool)
        n_pos = int(P.sum())
        print(f"\n{'':<26}{'件数':>6}{'TP':>6}{'精度':>8}{'再現':>8}")
        print(f"{'絞り込み前':<26}{len(d):>6}{n_pos:>6}{100 * P.mean():>7.0f}%{100:>7.0f}%")
        print(f"{'優先度採用':<26}{int(keep.sum()):>6}{int(P[keep].sum()):>6}"
              f"{100 * P[keep].mean():>7.0f}%{100 * P[keep].sum() / n_pos:>7.0f}%")
        lost = d[P & ~keep]
        if len(lost):
            print(f"\n落とした risky {len(lost)} 件:")
            for _, x in lost.iterrows():
                print(f"  {str(x.drive_id).split('|')[-1]} seg{int(x.segment)} "
                      f"{str(x.get('event_types', ''))[:44]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(args.out, index=False)
        print(f"\n出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
