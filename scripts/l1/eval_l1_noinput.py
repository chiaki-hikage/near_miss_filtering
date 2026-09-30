#!/usr/bin/env python3
"""L1「無操作」を**運転者の操作入力**で定義して評価する。

車体の運動 (速度・加速度・ヨー) ではなく、**アクセル・ブレーキ・操舵の入力**
そのものを見る。狙いは「ヒヤリハットに対する自車の回避反応が一切ない区間」を
落とすこと。

    無操作 = 制動なし ∧ アクセル操作なし ∧ 操舵変化なし   (すべて窓内)

**欠測は「操作なし」と判定しない。** 判定できない窓は残す (fail open)。
信号が流れていないだけの区間を「無操作」として捨てると取りこぼしになる。

データセットで揃う信号が違う。無い列は条件から外し、その旨を出力に出す。

    comma2k19        gas_pedal_pct / brake_pressed / steer_deg
    commaCarSegments gas_released / brake_pressed / brake_mc_mpa
                     / steer_deg / steer_rate_can_dps / steer_torque_driver

  uv run python scripts/l1/eval_l1_noinput.py \
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

from eval_l1_windows import Eval, prepare  # 窓の下ごしらえと保持判定は共通

# 「操作あり」とみなす下限。既定は控えめ (= 落としにくい) 側に置く。
DEFAULT = {
    "brake_press": 0.05,   # 制動の強さ [MPa 等]。これ以下は踏んでいないとみなす
    "gas_range": 1.0,      # アクセル開度の変化 [%]
    "steer_range": 3.0,    # 舵角の変化 [deg]
    "steer_rate": 15.0,    # 舵角レートの絶対値 [deg/s]
    "steer_tq": 0.0,       # 操舵トルク。既定では使わない (0 = 無効)
}

GRID = {
    "gas_range":   [0.5, 1.0, 2.0, 5.0, np.inf],
    "steer_range": [1.0, 3.0, 5.0, 10.0, 20.0, np.inf],
    "steer_rate":  [5.0, 15.0, 30.0, 60.0, np.inf],
    "brake_press": [0.02, 0.05, 0.2, np.inf],
}


def no_input(w: pd.DataFrame, th: dict) -> tuple[pd.Series, list[str]]:
    """窓ごとの「操作入力が一切ない」マスクと、実際に使えた条件の名前。

    **各条件は「その信号が取れている窓」でしか課さない。** 取れていない窓は
    その条件を満たさない扱いにして、最終的に棄却しない側へ倒す。
    """
    used: list[str] = []
    ok = pd.Series(True, index=w.index)      # 全条件が「操作なし」を示すか
    known = pd.Series(False, index=w.index)  # 1 つでも判定できた信号があるか

    def apply(mask: pd.Series, valid: pd.Series, label: str) -> None:
        nonlocal ok, known
        if not valid.any():
            return
        used.append(label)
        ok &= mask.where(valid, False) | ~valid
        known |= valid

    # --- 制動 ---
    v = w.brake_on.notna()
    apply(w.brake_on <= 0.5, v, "brake_pressed")
    if th["brake_press"] < np.inf:
        v = w.brake_press_max.notna()
        apply(w.brake_press_max < th["brake_press"], v, "brake_pressure")

    # --- アクセル ---
    if th["gas_range"] < np.inf:
        gr = w.gas_max - w.gas_min
        apply(gr < th["gas_range"], gr.notna(), "gas_pedal_pct")
        # 開度が流れない個体は GAS_RELEASED の変化で代用する
        chg = (w.gas_rel_max - w.gas_rel_min).abs()
        apply(chg <= 0.5, chg.notna(), "gas_released")

    # --- 操舵 ---
    if th["steer_range"] < np.inf:
        sr = w.steer_max - w.steer_min
        apply(sr < th["steer_range"], sr.notna(), "steer_angle")
    if th["steer_rate"] < np.inf:
        v = w.steer_rate_absmax.notna()
        apply(w.steer_rate_absmax < th["steer_rate"], v, "steer_rate")
    if th["steer_tq"] > 0:
        v = w.steer_tq_absmax.notna()
        apply(w.steer_tq_absmax < th["steer_tq"], v, "steer_torque")

    # 1 つも判定できない窓は落とさない
    return (ok & known).fillna(False), used


def reaction(c: pd.DataFrame) -> pd.Series:
    """候補区間に**自車の回避反応**があったか。

    候補ごとの集計値から見る。制動・減速・操舵のどれかが立てば反応ありとする。
    """
    r = pd.Series(False, index=c.index)
    if "brake_pressed_max" in c:
        r |= c.brake_pressed_max.fillna(0) > 0.5
    if "ax_mps2_min" in c:
        r |= c.ax_mps2_min.fillna(0) < -1.5
    if "steer_rate_dps_absmax" in c:
        r |= c.steer_rate_dps_absmax.fillna(0) > 100.0
    return r


AVOID = ("hard_brake", "panic_brake", "panic_brake_pedal", "panic_brake_with_lead",
         "brake_and_steer", "brake_after_evasion", "s_evasion", "hard_steer",
         "counter_steer", "abs_active", "aeb_active", "vsc_active")
SITUATION = ("cut_in_candidate", "short_thw", "low_ttc", "closing_fast")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--win", type=float, default=10.0, help="窓長 [秒]")
    p.add_argument("--margin", type=int, default=1)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--no-keep-flags", action="store_true",
                   help="安全系フラグの窓も棄却対象にする (既定は無条件保持)")
    for k, v in DEFAULT.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    W = prepare(pd.read_csv(args.windows, low_memory=False))
    w = W[W.win_s == args.win].reset_index(drop=True)
    if w.empty:
        raise SystemExit(f"窓 {args.win} 秒のデータがありません: {sorted(W.win_s.unique())}")
    c = pd.read_csv(args.candidates, low_memory=False).dropna(subset=["segment"])
    c["segment"] = c.segment.astype(int)

    th = {k: getattr(args, k) for k in DEFAULT}
    cond, used = no_input(w, th)
    keep_flag = not args.no_keep_flags
    if keep_flag:
        cond &= ~(w.safety_flag > 0.5)

    print(f"窓 {args.win:g} 秒 / margin {args.margin} / 総走行 {w.dur_s.sum() / 3600:.2f} 時間")
    print(f"使えた操作信号: {', '.join(used) or '(なし)'}")
    print(f"閾値: " + "  ".join(f"{k}={v:g}" for k, v in th.items()))
    print("安全系フラグの窓は無条件保持" if keep_flag else "安全系フラグも棄却対象")

    ev = Eval(w, c, args.win, args.margin)
    r = ev(cond.to_numpy())
    alive, cand = r["_alive"], r["_ev"].copy()
    cand["alive"] = alive
    print(f"\n削減率 {r['削減率']:.1f}%  (残 {r['残時間_h']:.2f} 時間)"
          f"   全候補 {r['保持']}/{r['候補']} = {r['保持率']:.1f}%")

    # --- 種別ごとの保持 ---
    tt = cand.event_types.fillna("").str.get_dummies("|")
    print(f"\n{'イベント種別':<24}{'全体':>6}{'保持':>6}{'保持率':>8}")
    print("-- 回避反応そのもの " + "-" * 24)
    for k in AVOID:
        if k in tt and tt[k].sum():
            m = tt[k].astype(bool)
            print(f"  {k:<22}{int(m.sum()):>6}{int(alive[m].sum()):>6}"
                  f"{100 * alive[m].mean():>7.1f}%")
    print("-- 危険状況 (前方・車間) " + "-" * 19)
    for k in SITUATION:
        if k in tt and tt[k].sum():
            m = tt[k].astype(bool)
            print(f"  {k:<22}{int(m.sum()):>6}{int(alive[m].sum()):>6}"
                  f"{100 * alive[m].mean():>7.1f}%")

    # --- 危険状況を「回避反応の有無」で分ける ---
    rea = reaction(cand)
    print(f"\n=== 危険状況を回避反応の有無で分ける ===")
    print("反応あり = 制動 or 減速 < -1.5 m/s^2 or 舵角レート > 100 deg/s")
    print(f"\n{'種別':<22}{'反応':>6}{'件数':>6}{'保持':>6}{'保持率':>8}")
    for k in SITUATION:
        if k not in tt or not tt[k].sum():
            continue
        m = tt[k].astype(bool)
        for lab, sub in (("あり", m & rea), ("なし", m & ~rea)):
            if not sub.sum():
                continue
            print(f"  {k if lab == 'あり' else '':<20}{lab:>6}{int(sub.sum()):>6}"
                  f"{int(alive[sub].sum()):>6}{100 * alive[sub].mean():>7.1f}%")

    if args.labels:
        _labels(args, w, th, keep_flag)
    if args.sweep:
        _sweep(args, w, c, keep_flag)
    return 0


def _labels(args, w, th, keep_flag) -> None:
    lab = pd.read_csv(args.labels).dropna(subset=["t_in_segment_s"]).copy()
    if "risky" not in lab.columns:
        return
    lab["duration_s"] = lab.t_end - lab.t_start
    cond, _ = no_input(w, th)
    if keep_flag:
        cond &= ~(w.safety_flag > 0.5)
    r = Eval(w, lab, args.win, args.margin)(cond.to_numpy())
    a, ev = r["_alive"], r["_ev"]
    rk = ev.risky.astype(bool).to_numpy()
    print(f"\n=== 人手ラベル ===  削減率 {r['削減率']:.1f}%")
    print(f"  risky   {int(a[rk].sum())}/{int(rk.sum())}")
    print(f"  非risky {int(a[~rk].sum())}/{int((~rk).sum())}")
    lost = ev[~a]
    for _, x in lost.iterrows():
        print(f"    喪失: {x.drive_id.split('|')[-1]} seg{int(x.segment)} "
              f"risky={x.risky}  {str(x.get('event_types_at_labeling', ''))[:50]}")


def _sweep(args, w, c, keep_flag) -> None:
    print("\n=== 保持率別の最大削減率 (操作入力の閾値を総当たり) ===")
    ev = Eval(w, c, args.win, args.margin)
    flag = (w.safety_flag > 0.5).to_numpy() if keep_flag else None
    cols = list(GRID)
    rows = []
    for combo in itertools.product(*GRID.values()):
        th = dict(DEFAULT) | dict(zip(cols, combo))
        cond, _ = no_input(w, th)
        m = cond.to_numpy()
        if flag is not None:
            m = m & ~flag
        if not m.any():
            continue
        r = ev(m)
        rows.append((r["削減率"], r["保持率"]) + combo)
    R = pd.DataFrame(rows, columns=["削減率", "保持率"] + cols)
    f = lambda x: "-" if not np.isfinite(x) else f"{x:g}"  # noqa: E731
    print(f"{'保持率下限':>10}{'削減率':>8}   " + " / ".join(cols))
    for lim in (100.0, 99.0, 98.0, 95.0, 90.0, 85.0):
        s = R[R.保持率 >= lim].sort_values("削減率", ascending=False)
        if s.empty:
            continue
        x = s.iloc[0]
        print(f"{lim:>9.0f}%{x['削減率']:>7.1f}%   " + " / ".join(f(x[k]) for k in cols))


if __name__ == "__main__":
    raise SystemExit(main())
