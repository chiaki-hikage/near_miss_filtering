#!/usr/bin/env python3
"""2 回のスクリーニング結果を突き合わせる。設定変更の影響を見るため。

イベント種別ごとの増減と、候補の増減を出す。
**同じ区間かどうかは時刻の重なりで見る。** 設定が変わると t_start が動くため。

  uv run python scripts/compare_runs.py out/screen_rav4_tss2_2k out/screen_rav4_tss2_2k_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd


def load(d: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ev = pd.read_csv(d / "events.csv", low_memory=False)
    ca = pd.read_csv(d / "candidates.csv", low_memory=False)
    seg = pd.read_csv(d / "segments.csv", low_memory=False)
    h = seg.duration_s.sum() / 3600 if "duration_s" in seg else float("nan")
    return ev, ca, {"hours": h, "n_seg": len(seg)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    e0, c0, m0 = load(args.before)
    e1, c1, m1 = load(args.after)
    print(f"{'':<26}{'改定前':>10}{'改定後':>10}{'差':>10}")
    print("-" * 56)
    print(f"{'セグメント':<26}{m0['n_seg']:>10}{m1['n_seg']:>10}"
          f"{m1['n_seg'] - m0['n_seg']:>+10}")
    print(f"{'走行時間 [h]':<26}{m0['hours']:>10.2f}{m1['hours']:>10.2f}"
          f"{m1['hours'] - m0['hours']:>+10.2f}")
    print(f"{'イベント':<26}{len(e0):>10}{len(e1):>10}{len(e1) - len(e0):>+10}")
    print(f"{'候補':<26}{len(c0):>10}{len(c1):>10}{len(c1) - len(c0):>+10}")
    print(f"{'候補 (severity>0)':<26}"
          f"{int((c0.severity > 0).sum()):>10}{int((c1.severity > 0).sum()):>10}"
          f"{int((c1.severity > 0).sum()) - int((c0.severity > 0).sum()):>+10}")

    a = e0.event_type.value_counts()
    b = e1.event_type.value_counts()
    T = pd.DataFrame({"改定前": a, "改定後": b}).fillna(0).astype(int)
    T["差"] = T.改定後 - T.改定前
    print(f"\n{'イベント種別':<26}{'改定前':>8}{'改定後':>8}{'差':>8}")
    print("-" * 52)
    for k, r in T.sort_values("差").iterrows():
        mark = "  <-- 変更対象" if k in ("hard_accel", "hard_steer",
                                        "lane_change_candidate") else ""
        print(f"{k:<26}{r.改定前:>8}{r.改定後:>8}{r.差:>+8}{mark}")
    print("-" * 52)
    print(f"{'合計':<26}{T.改定前.sum():>8}{T.改定後.sum():>8}"
          f"{T.差.sum():>+8}")

    print(f"\nseverity: 改定前 中央 {c0.severity.median():.2f} / "
          f"改定後 中央 {c1.severity.median():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
