#!/usr/bin/env python3
"""L1 -> L2 -> L3 の通し集計。**判定ロジックは変えない。**

L1 は探索で決めた固定条件。L2 は特徴量生成だけで絞り込まない。
L3 は既存の検出結果 (events.csv / candidates.csv) をそのまま使い、
L1 を通ったかどうかで残存を数える。

出したいのは 2 つ。
    通しの残存率 (走行時間 / 候補 / イベント)
    最終的に映像を取りに行く割合と、その種別構成

  uv run python scripts/report/report_pipeline.py \
      --windows out/window_stats_ccs_TOYOTA_RAV4_TSS2.csv \
      --events   out/screen_rav4_tss2_2k/events.csv \
      --candidates out/screen_rav4_tss2_2k/candidates.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from _stages import L1, Eval, l1_condition, l1_windows, load_candidates, load_events

# 種別の並び。回避反応 / 前方接近 / 安全系 の順に見せる。
ORDER = [
    ("回避反応", ["hard_brake", "panic_brake", "panic_brake_pedal",
                  "panic_brake_with_lead", "hard_accel", "hard_steer",
                  "counter_steer", "s_evasion", "brake_and_steer",
                  "brake_after_evasion", "lateral_accel"]),
    ("挙動異常", ["weaving", "yaw_instability", "wheel_speed_anomaly",
                  "lane_change_candidate", "risky_lane_change"]),
    ("前方接近", ["cut_in_candidate", "short_thw", "low_ttc", "closing_fast"]),
    ("安全系", ["abs_active", "vsc_active", "aeb_active", "slip_warn",
                "tc_disabled"]),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--win", type=float, default=10.0)
    p.add_argument("--margin", type=int, default=1)
    p.add_argument("--clip-s", type=float, default=20.0,
                   help="1 候補あたり取得する映像の長さ [秒]")
    p.add_argument("--breakdown", action="store_true",
                   help="L1 の削減が何によるものかを分解する")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    w = l1_windows(args.windows, args.win)
    over = (w.safety_flag > 0.5).to_numpy()
    cond = l1_condition(w)

    ev = load_events(args.events)
    ca = load_candidates(args.candidates)

    total_h = w.dur_s.sum() / 3600
    E_ev = Eval(w, ev, args.win, args.margin)
    E_ca = Eval(w, ca, args.win, args.margin)
    r_ev, r_ca = E_ev(cond), E_ca(cond)
    keep_h = r_ca["残時間_h"]

    print("=" * 74)
    print("L1 条件 (窓 %g 秒 / 前後 %d 窓を保持 / 安全系フラグは override)"
          % (args.win, args.margin))
    for k, v in L1.items():
        print(f"    {k:<20} < {v:g}")
    print(f"    override 窓: {int(over.sum())} / {len(w)}")
    print("=" * 74)

    # --- 通しの残存 ---
    n_seg_all = w[["drive_id", "segment"]].drop_duplicates().shape[0]
    kept_w = E_ca.keep_mask(cond)
    n_seg_keep = (w[kept_w][["drive_id", "segment"]].drop_duplicates().shape[0])
    print(f"\n{'段':<26}{'量':>14}{'残存率':>10}")
    print("-" * 52)
    print(f"{'L0 全走行':<26}{total_h:>11.2f} h{'100.0%':>10}")
    print(f"{'L1 通過 (走行時間)':<26}{keep_h:>11.2f} h{100 * keep_h / total_h:>9.1f}%")
    print(f"{'   (セグメント数)':<26}{n_seg_keep:>13}{100 * n_seg_keep / n_seg_all:>9.1f}%")
    print(f"{'L2 特徴量生成':<26}{keep_h:>11.2f} h{100 * keep_h / total_h:>9.1f}%"
          "   ※絞り込まない")
    print(f"{'L3 イベント':<26}{r_ev['保持']:>13}{r_ev['保持率']:>9.1f}%"
          f"   (L1 前 {r_ev['候補']})")
    print(f"{'L3 候補 (映像対象)':<26}{r_ca['保持']:>13}{r_ca['保持率']:>9.1f}%"
          f"   (L1 前 {r_ca['候補']})")

    clip_h = r_ca["保持"] * args.clip_s / 3600
    print(f"\n映像を取りに行く量: 候補 {r_ca['保持']} 件 x {args.clip_s:g} 秒 "
          f"= {clip_h:.2f} 時間")
    print(f"  全走行に対する割合 : {100 * clip_h / total_h:.2f}%")
    print(f"  L1 通過に対する割合: {100 * clip_h / keep_h:.2f}%")
    print(f"  候補の発生率       : {r_ca['保持'] / keep_h:.2f} 件/時 "
          f"(L1 前 {r_ca['候補'] / total_h:.2f} 件/時)")

    # --- 種別ごと ---
    alive = r_ev["_alive"]
    e = r_ev["_ev"].copy()
    e["alive"] = alive
    n_all, n_keep = len(e), int(alive.sum())
    print(f"\n{'区分':<10}{'イベント種別':<24}{'L1前':>7}{'構成比':>8}"
          f"{'L1後':>7}{'構成比':>8}{'L1保持率':>10}{'件/時':>8}")
    print("-" * 82)
    seen = set()
    for group, names in ORDER:
        first = True
        for k in names:
            m = (e.event_type == k).to_numpy()
            if not m.any():
                continue
            seen.add(k)
            a, b = int(m.sum()), int(alive[m].sum())
            print(f"{group if first else '':<10}{k:<24}{a:>7}{100 * a / n_all:>7.1f}%"
                  f"{b:>7}{100 * b / max(n_keep, 1):>7.1f}%"
                  f"{100 * b / a:>9.1f}%{b / keep_h:>8.2f}")
            first = False
    rest = sorted(set(e.event_type.unique()) - seen)
    for k in rest:
        m = (e.event_type == k).to_numpy()
        a, b = int(m.sum()), int(alive[m].sum())
        print(f"{'その他':<10}{k:<24}{a:>7}{100 * a / n_all:>7.1f}%"
              f"{b:>7}{100 * b / max(n_keep, 1):>7.1f}%"
              f"{100 * b / a:>9.1f}%{b / keep_h:>8.2f}")
    print("-" * 82)
    print(f"{'合計':<34}{n_all:>7}{'100.0%':>8}{n_keep:>7}{'100.0%':>8}"
          f"{100 * n_keep / n_all:>9.1f}%{n_keep / keep_h:>8.2f}")

    # --- 映像対象 (候補) の構成 ---
    # イベントは 1 候補に複数入る。**映像を取る単位は候補**なので別に数える。
    ac, ec = r_ca["_alive"], r_ca["_ev"].copy()
    tt = ec.event_types.fillna("").str.get_dummies("|")
    print(f"\n映像対象 {int(ac.sum())} 候補の構成 (1 候補に複数種別が入る)")
    print(f"{'イベント種別':<24}{'L1前':>7}{'L1後':>7}{'保持率':>9}{'候補比':>9}")
    for k in tt.columns[tt.sum().sort_values(ascending=False).index.argsort()][:0]:
        pass
    for k in tt.sum().sort_values(ascending=False).index:
        m = tt[k].astype(bool).to_numpy()
        a, b = int(m.sum()), int(ac[m].sum())
        if not a:
            continue
        print(f"{k:<24}{a:>7}{b:>7}{100 * b / a:>8.1f}%"
              f"{100 * b / max(int(ac.sum()), 1):>8.1f}%")
    if "severity" in ec.columns:
        print(f"\n  severity: L1前 中央 {ec.severity.median():.2f} / "
              f"L1後 中央 {ec.loc[ac, 'severity'].median():.2f}")

    if args.breakdown:
        _breakdown(w, over, E_ca, args)

    if args.labels:
        _labels(args, w, cond)

    if args.out:
        e[["drive_id", "segment", "event_type", "t_in_segment_s", "alive"]].to_csv(
            args.out, index=False)
        print(f"\n出力: {args.out}")
    return 0


def _breakdown(w, over, E, args) -> None:
    """L1 の削減が何によるものかを分解する。

    L1 は連言なので、条件を 1 つ外すと棄却は**増える**。
    「その条件がどれだけ棄却を止めているか」は leave-one-out で測る。
    """
    dur = w.dur_s.to_numpy()
    total = dur.sum()

    def t_of(mask) -> float:
        return 100 * dur[mask].sum() / total

    print("\n" + "=" * 74)
    print("L1 削減の内訳 (走行時間ベース)")
    print("=" * 74)

    # 1. 各条件が単独で「静か」と言う割合
    print(f"\n{'条件':<26}{'単独で成立':>12}{'欠測':>8}")
    single = {}
    for k, t in L1.items():
        v = w[k]
        m = (v < t).fillna(False).to_numpy()
        single[k] = m
        print(f"{k + ' < ' + format(t, 'g'):<26}{t_of(m):>11.1f}%"
              f"{100 * v.isna().mean():>7.1f}%")

    # 2. 連言 -> override -> margin
    allc = np.ones(len(w), dtype=bool)
    for m in single.values():
        allc &= m
    known = np.zeros(len(w), dtype=bool)
    for k in L1:
        known |= w[k].notna().to_numpy()
    raw = allc & known
    after_over = raw & ~over
    final = E.keep_mask(after_over)
    # keep_mask は**保持**側を返す。棄却は its 否定。
    red = t_of(~final)
    print(f"\n{'段階':<34}{'棄却時間':>10}{'差分':>10}")
    print(f"{'5 条件すべて成立 (素の棄却)':<34}{t_of(raw):>9.1f}%{'':>10}")
    print(f"{'  - 安全系フラグ override':<34}{t_of(after_over):>9.1f}%"
          f"{t_of(after_over) - t_of(raw):>+9.1f}%")
    print(f"{'  - 前後 %d 窓を保持 (margin)' % args.margin:<34}{red:>9.1f}%"
          f"{red - t_of(after_over):>+9.1f}%")
    print(f"{'  = 最終的な削減率':<34}{red:>9.1f}%")

    # 3. leave-one-out: その条件を外すとどれだけ棄却が増えるか
    print(f"\n{'条件を 1 つ外したときの削減率':<34}{'削減率':>10}{'増分':>10}")
    for k in L1:
        m = np.ones(len(w), dtype=bool)
        kn = np.zeros(len(w), dtype=bool)
        for j in L1:
            if j == k:
                continue
            m &= single[j]
            kn |= w[j].notna().to_numpy()
        r2 = t_of(~E.keep_mask(m & kn & ~over))
        print(f"{'  ' + k + ' を外す':<34}{r2:>9.1f}%{r2 - red:>+9.1f}%")

    # 4. 残った窓は「どの条件が止めたか」
    kept = ~raw & known
    n_fail = np.zeros(len(w), dtype=int)
    for m in single.values():
        n_fail += (~m).astype(int)
    print(f"\n残した窓 (棄却しなかった) の理由 — 成立しなかった条件の数")
    for n in range(1, 6):
        m = kept & (n_fail == n)
        if m.any():
            print(f"  {n} 個が不成立 : {t_of(m):>5.1f}% の時間")
    print(f"\n1 個だけ不成立だった窓 (= その条件が単独で棄却を止めた)")
    only = kept & (n_fail == 1)
    for k, m in single.items():
        b = only & ~m
        if b.any():
            print(f"  {k:<22}{t_of(b):>6.1f}%")
    print(f"  安全系 override            {t_of(raw & over):>5.1f}%")


def _labels(args, w, cond) -> None:
    lab = pd.read_csv(args.labels).dropna(subset=["t_in_segment_s"]).copy()
    if "risky" not in lab.columns:
        return
    lab["duration_s"] = lab.t_end - lab.t_start
    r = Eval(w, lab, args.win, args.margin)(cond)
    a, e = r["_alive"], r["_ev"]
    rk = e.risky.astype(bool).to_numpy()
    print(f"\n人手ラベル: risky {int(a[rk].sum())}/{int(rk.sum())}  "
          f"非risky {int(a[~rk].sum())}/{int((~rk).sum())}")


if __name__ == "__main__":
    raise SystemExit(main())
