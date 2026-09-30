#!/usr/bin/env python3
"""L1 -> L2 -> L3 -> 優先度フィルタ を通した最終結果の内訳。

**既存の events.csv / candidates.csv は L3 の素の出力**で、L1 も優先度条件も
通っていない。この script が両方を当てて、段ごとの内訳を出す。

    L1        窓ごとの操作入力による棄却 (安全系フラグは override)
    L3        既存の検出結果をそのまま使う
    優先度    候補の集計値による絞り込み (precision 40% 級)

  uv run python scripts/report/report_final.py \
      --windows out/window_stats_ccs_TOYOTA_RAV4_TSS2.csv \
      --events out/screen_rav4_tss2_2k/events.csv \
      --candidates out/screen_rav4_tss2_2k/candidates.csv \
      --out out/final_ccs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from _stages import L1, Eval, l1_condition, l1_windows, load_candidates, load_events
from search_priority import base

# 優先度条件。search_priority.py の探索で precision 40% 級だったもの。
# **severity の分位はデータセット内で決まる相対値。** 母集団が変わると閾値も動く。
PRIORITY = {
    "p40": ("安全系 ∨ (severity 上位50% ∧ 回避反応あり)",
            lambda d: d._safety | ((d._sev >= d._sev.median()) & d._react)),
    "p50": ("安全系 ∨ (車間 < 1.0 s ∧ severity 上位50%)",
            lambda d: d._safety | ((d._thw < 1.0) & (d._sev >= d._sev.median()))),
    "p67": ("安全系 ∨ (車間 < 0.8 s ∧ 回避反応あり)",
            lambda d: d._safety | ((d._thw < 0.8) & d._react)),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--win", type=float, default=10.0)
    p.add_argument("--margin", type=int, default=1)
    p.add_argument("--priority", choices=sorted(PRIORITY), default="p40")
    p.add_argument("--clip-s", type=float, default=20.0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    w = l1_windows(args.windows, args.win)
    cond = l1_condition(w)

    ca = load_candidates(args.candidates)
    ev = load_events(args.events)

    # --- L1 ---
    r_ca = Eval(w, ca, args.win, args.margin)(cond)
    r_ev = Eval(w, ev, args.win, args.margin)(cond)
    ca = r_ca["_ev"].copy()
    ca["L1"] = r_ca["_alive"]
    ev = r_ev["_ev"].copy()
    ev["L1"] = r_ev["_alive"]
    total_h = w.dur_s.sum() / 3600
    keep_h = r_ca["残時間_h"]

    # --- 優先度 (L1 を通った候補の中で判定する) ---
    desc, fn = PRIORITY[args.priority]
    b = base(ca[ca.L1].copy())
    sel = fn(b)
    ca["PRI"] = False
    ca.loc[b.index[sel], "PRI"] = True

    # イベントは「所属する候補」で決まる。区間の重なりで対応づける。
    ev["PRI"] = False
    keep = ca[ca.PRI]
    for (d, s), g in keep.groupby(["drive_id", "segment"]):
        m = (ev.drive_id == d) & (ev.segment == s) & ev.L1
        if not m.any():
            continue
        idx = ev.index[m]
        for i in idx:
            t0, t1 = ev.at[i, "t_start"], ev.at[i, "t_end"]
            if ((g.t_start <= t1) & (g.t_end >= t0)).any():
                ev.at[i, "PRI"] = True

    print("=" * 74)
    print(f"L1: {'  '.join(f'{k}<{v:g}' for k, v in L1.items())}")
    print(f"    + 安全系フラグ override / 窓 {args.win:g} s / 前後 {args.margin} 窓保持")
    print(f"優先度 ({args.priority}): {desc}")
    print("=" * 74)

    n_ev, n_ca = len(ev), len(ca)
    n_ev1, n_ca1 = int(ev.L1.sum()), int(ca.L1.sum())
    n_evp, n_cap = int(ev.PRI.sum()), int(ca.PRI.sum())
    print(f"\n{'段':<20}{'走行時間':>10}{'イベント':>10}{'候補':>8}{'映像%':>9}")
    print("-" * 58)
    print(f"{'L0 全走行':<20}{total_h:>8.2f} h{'-':>10}{'-':>8}{'100.00%':>9}")
    print(f"{'L1 通過':<20}{keep_h:>8.2f} h{n_ev1:>10}{n_ca1:>8}"
          f"{100 * keep_h / total_h:>8.2f}%")
    print(f"{'L3 (素)':<20}{'':>10}{n_ev:>10}{n_ca:>8}"
          f"{100 * n_ca * args.clip_s / 3600 / total_h:>8.2f}%")
    print(f"{'L1 + L3':<20}{'':>10}{n_ev1:>10}{n_ca1:>8}"
          f"{100 * n_ca1 * args.clip_s / 3600 / total_h:>8.2f}%")
    print(f"{'L1 + L3 + 優先度':<20}{'':>10}{n_evp:>10}{n_cap:>8}"
          f"{100 * n_cap * args.clip_s / 3600 / total_h:>8.2f}%")
    print("-" * 58)
    print(f"最終: 候補 {n_cap} 件 ({100 * n_cap / n_ca:.1f}% of L3) "
          f"= 映像 {n_cap * args.clip_s / 3600:.2f} 時間 "
          f"({n_cap / total_h:.2f} 件/時)")

    # --- 種別内訳 ---
    print(f"\n{'イベント種別':<24}{'L3素':>7}{'L1後':>7}{'優先度後':>9}"
          f"{'L1保持':>8}{'優先度率':>9}{'最終構成%':>10}")
    print("-" * 76)
    tot = ev.event_type.value_counts()
    for k in tot.index:
        m = ev.event_type == k
        a, b1, c1 = int(m.sum()), int((m & ev.L1).sum()), int((m & ev.PRI).sum())
        print(f"{k:<24}{a:>7}{b1:>7}{c1:>9}{100 * b1 / a:>7.0f}%"
              f"{(100 * c1 / b1 if b1 else 0):>8.0f}%"
              f"{(100 * c1 / n_evp if n_evp else 0):>9.1f}%")
    print("-" * 76)
    print(f"{'合計':<24}{n_ev:>7}{n_ev1:>7}{n_evp:>9}{100 * n_ev1 / n_ev:>7.0f}%"
          f"{(100 * n_evp / n_ev1 if n_ev1 else 0):>8.0f}%{'100.0%':>10}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cols = [c for c in ("drive_id", "segment", "t_start", "t_end",
                            "t_in_segment_s", "duration_s", "severity", "n_events",
                            "event_types", "thw_s_min", "ttc_s_min", "ax_mps2_min",
                            "L1", "PRI") if c in ca.columns]
        ca[cols].to_csv(args.out, index=False)
        ep = args.out.with_name(args.out.stem + "_events.csv")
        ev[[c for c in ("drive_id", "segment", "event_type", "t_start", "t_end",
                        "t_in_segment_s", "peak_value", "L1", "PRI")
            if c in ev.columns]].to_csv(ep, index=False)
        print(f"\n出力: {args.out}  (候補 / L1・PRI 列つき)")
        print(f"      {ep}  (イベント)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
