#!/usr/bin/env python3
"""L3 -> severity -> L1 -> 優先度 を**同じ鎖**で通し、種別内訳を保存する。

2 つのデータセットを同じ条件で比べるための道具。
comma2k19 は人手ラベルがあるので精度も出す。

  uv run python scripts/report/report_chain.py \
      --candidates out/screen_rav4_tss2_2k_v2/candidates.csv \
      --windows out/window_stats_ccs_TOYOTA_RAV4_TSS2.csv \
      --out out/chain_ccs_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from l1_search_by_target import build, cond_of
from l1_eval_window_size import Eval, prepare
from filter_priority import apply_rules
from near_miss.config import load_yaml
from report_pipeline import L1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--windows", type=Path, default=None,
                   help="省略すると L1 を飛ばす")
    p.add_argument("--labels", type=Path, default=None,
                   help="risky 列を持つ表。区間の重なりで突き合わせて精度を出す")
    p.add_argument("--priority", type=Path, default=Path("configs/priority.yaml"))
    p.add_argument("--win", type=float, default=10.0)
    p.add_argument("--margin", type=int, default=1)
    p.add_argument("--clip-s", type=float, default=20.0)
    p.add_argument("--out", type=Path, default=None, help="出力の先頭 (拡張子なし)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ca = pd.read_csv(args.candidates, low_memory=False).dropna(subset=["segment"])
    ca["segment"] = ca.segment.astype(int)
    total_h = np.nan

    ca["L1"] = True
    if args.windows:
        W = build(prepare(pd.read_csv(args.windows, low_memory=False)))
        w = W[W.win_s == args.win].reset_index(drop=True)
        cond = cond_of(w, L1) & ~(w.safety_flag > 0.5).to_numpy()
        r = Eval(w, ca, args.win, args.margin)(cond)
        ca = r["_ev"].copy()
        ca["L1"] = r["_alive"]
        total_h = w.dur_s.sum() / 3600

    ca["sev_ok"] = ca.severity > 0
    pri, reason = apply_rules(ca, load_yaml(args.priority))
    ca["priority"] = pri
    ca["priority_reason"] = reason
    ca["final"] = ca.L1 & ca.sev_ok & ca.priority

    risky = None
    if args.labels:
        lab = pd.read_csv(args.labels)
        if "risky" in lab.columns:
            # 区間の重なりで突き合わせる。t_start は設定が変わると動くため。
            risky = []
            for _, x in ca.iterrows():
                m = ((lab.drive_id == x.drive_id)
                     & (lab.t_end >= x.t_start) & (lab.t_start <= x.t_end))
                risky.append(bool(lab[m].risky.any()) if m.any() else None)
            ca["risky"] = risky
            ca["labeled"] = [v is not None for v in risky]

    stages = [("L3 候補", pd.Series(True, index=ca.index)),
              ("  + severity>0", ca.sev_ok),
              ("  + L1", ca.sev_ok & ca.L1),
              ("  + 優先度 = 最終", ca.final)]
    print(f"{args.candidates}")
    print(f"{'段':<20}{'件数':>7}{'映像[h]':>9}{'全走行%':>9}"
          + ("{:>8}{:>6}{:>8}".format("ラベル", "risky", "精度")
             if risky is not None else ""))
    print("-" * (45 + (22 if risky is not None else 0)))
    for name, m in stages:
        n = int(m.sum())
        line = (f"{name:<20}{n:>7}{n * args.clip_s / 3600:>9.2f}"
                + (f"{100 * n * args.clip_s / 3600 / total_h:>8.2f}%"
                   if np.isfinite(total_h) else f"{'-':>9}"))
        if risky is not None:
            sub = ca[m & ca.labeled]
            nl, nr = len(sub), int(sub.risky.sum())
            line += f"{nl:>8}{nr:>6}{(100 * nr / nl if nl else 0):>7.0f}%"
        print(line)

    f = ca[ca.final]
    tt = f.event_types.fillna("").str.get_dummies("|")
    all_tt = ca.event_types.fillna("").str.get_dummies("|")
    print(f"\n最終 {len(f)} 候補の種別内訳 (1 候補に複数種別)")
    print(f"{'イベント種別':<24}{'L3全体':>8}{'最終':>7}{'残存率':>8}{'候補比':>8}")
    rows = []
    for k in tt.sum().sort_values(ascending=False).index:
        a = int(all_tt[k].sum()) if k in all_tt else 0
        b = int(tt[k].sum())
        print(f"{k:<24}{a:>8}{b:>7}{(100 * b / a if a else 0):>7.0f}%"
              f"{100 * b / max(len(f), 1):>7.1f}%")
        rows.append({"event_type": k, "L3": a, "final": b,
                     "retain_pct": round(100 * b / a, 1) if a else 0,
                     "share_pct": round(100 * b / max(len(f), 1), 1)})

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out.with_suffix(".breakdown.csv"), index=False)
        keep = [c for c in ("drive_id", "segment", "t_start", "t_end",
                            "t_in_segment_s", "severity", "n_events", "event_types",
                            "lead_distance_m_min", "thw_s_min", "ttc_s_min",
                            "ax_mps2_min", "L1", "sev_ok", "priority",
                            "priority_reason", "final", "risky", "labeled")
                if c in ca.columns]
        ca[keep].to_csv(args.out.with_suffix(".candidates.csv"), index=False)
        print(f"\n出力: {args.out.with_suffix('.breakdown.csv')}")
        print(f"      {args.out.with_suffix('.candidates.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
