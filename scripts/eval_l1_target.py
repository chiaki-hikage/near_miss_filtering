#!/usr/bin/env python3
"""ターゲット定義から逆算して L1 条件を探索する。

**L1 はイベント種別を判定しない。**「明らかに対象になりにくい時間」を
生 CAN の min/max/range だけで安く落とすのが目的。

主対象 (T1)
    一定以上の速度域で、**回避反応**または**安全系の作動**を伴う候補
        回避反応 = 制動 / アクセルオフ / 操舵
        安全系   = ABS / PCS(precollision) / VSC / slip_warn / TRC
前方接近系 (T2)
    cut_in / short_thw / low_ttc / closing_fast のうち速度条件を満たすもの。
    回避反応の有無で分けて併記する。

**安全系フラグが立った窓は override として必ず残す。** 条件探索の対象外。

  uv run python scripts/eval_l1_target.py \
      --windows out/window_stats_ccs_TOYOTA_RAV4_TSS2.csv \
      --candidates out/screen_rav4_tss2_2k/candidates.csv --search
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from eval_l1_windows import Eval, prepare

FORWARD = ("cut_in_candidate", "short_thw", "low_ttc", "closing_fast")
SAFETY_COLS = ("abs_active_flag_max", "vsc_active_flag_max", "slip_warn_max",
               "precollision_active_max")


def targets(c: pd.DataFrame, v_min: float, ax_th: float,
            sr_th: float) -> pd.DataFrame:
    """候補に主対象フラグを付ける。**候補側の集計値だけを使う** (L1 とは独立)。"""
    c = c.copy()
    v = c.get("v_mps_mean", pd.Series(np.nan, index=c.index))
    c["speed_ok"] = v.fillna(0) >= v_min

    r = pd.Series(False, index=c.index)
    if "brake_pressed_max" in c:
        r |= c.brake_pressed_max.fillna(0) > 0.5
    if "ax_mps2_min" in c:
        r |= c.ax_mps2_min.fillna(0) < ax_th
    if "gas_pedal_pct_max" in c:            # アクセルオフは開度の存在で近似
        r |= c.gas_pedal_pct_max.fillna(0) > 0
    if "steer_rate_dps_absmax" in c:
        r |= c.steer_rate_dps_absmax.fillna(0) > sr_th
    c["reaction"] = r

    s = pd.Series(False, index=c.index)
    for col in SAFETY_COLS:
        if col in c:
            s |= c[col].fillna(0) > 0.5
    c["safety"] = s

    c["T1"] = c.speed_ok & (c.reaction | c.safety)
    et = c.event_types.fillna("")
    c["forward"] = et.str.contains("|".join(FORWARD))
    c["T2"] = c.speed_ok & c.forward
    return c


# L1 に使える量。すべて窓内の min/max/range で作れる。
FEATURES = {
    "v_max":            ("lt", [3, 5, 8, 11, 14, np.inf]),      # 低速すぎる窓
    "v_range":          ("lt", [0.3, 0.5, 1, 2, 3, np.inf]),
    "ax_absmax":        ("lt", [0.4, 0.8, 1.2, 1.6, np.inf]),
    "ay_absmax":        ("lt", [0.4, 0.8, 1.2, 1.6, np.inf]),
    "yaw_absmax":       ("lt", [1, 2, 4, 8, np.inf]),
    "steer_range":      ("lt", [1, 3, 5, 10, 20, np.inf]),
    "steer_rate_absmax": ("lt", [5, 15, 30, 60, np.inf]),
    "brake_press_max":  ("lt", [0.02, 0.05, 0.2, np.inf]),
    "gas_range":        ("lt", [0.5, 1, 2, 5, np.inf]),
    "ws_spread_max":    ("lt", [0.1, 0.3, 0.5, 1.0, np.inf]),
    "gvc_absmax":       ("lt", [0.2, 0.5, 1.0, np.inf]),
}
# 探索で使う組合せ (全 11 量の総当たりは大きすぎるので代表を選ぶ)
COMBOS = [
    ("v_max",),
    ("brake_press_max", "gas_range", "steer_range"),
    ("brake_press_max", "gas_range", "steer_rate_absmax"),
    ("v_max", "brake_press_max", "gas_range", "steer_range"),
    ("ax_absmax", "ay_absmax", "steer_rate_absmax"),
    ("ax_absmax", "ay_absmax", "yaw_absmax", "steer_range"),
    ("brake_press_max", "gas_range", "steer_range", "ax_absmax"),
    ("brake_press_max", "gas_range", "steer_rate_absmax", "ax_absmax", "ay_absmax"),
    ("v_max", "brake_press_max", "gas_range", "steer_rate_absmax", "ax_absmax",
     "ay_absmax"),
    ("v_range", "steer_range", "ax_absmax"),          # 従来の車体運動ベース (比較用)
]


def build(w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    w["gas_range"] = w.gas_max - w.gas_min
    # 開度が流れない個体は GAS_RELEASED の変化で代用する
    chg = (w.gas_rel_max - w.gas_rel_min).abs()
    w["gas_range"] = w.gas_range.fillna(chg * 1e3)
    return w


def cond_of(w: pd.DataFrame, spec: dict[str, float]) -> np.ndarray:
    """棄却してよい窓。**判定できない (NaN) 量は「静か」と見なさない。**"""
    ok = np.ones(len(w), dtype=bool)
    known = np.zeros(len(w), dtype=bool)
    for k, t in spec.items():
        if not np.isfinite(t):
            continue
        v = w[k]
        valid = v.notna().to_numpy()
        ok &= np.where(valid, (v < t).fillna(False).to_numpy(), False) | ~valid
        known |= valid
    return ok & known


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--win", type=float, default=10.0)
    p.add_argument("--margin", type=int, default=1)
    p.add_argument("--v-min", type=float, default=8.33, help="対象の下限速度 [m/s]")
    p.add_argument("--ax-th", type=float, default=-1.5)
    p.add_argument("--sr-th", type=float, default=100.0)
    p.add_argument("--search", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    W = prepare(pd.read_csv(args.windows, low_memory=False))
    w = build(W[W.win_s == args.win].reset_index(drop=True))
    c = pd.read_csv(args.candidates, low_memory=False).dropna(subset=["segment"])
    c["segment"] = c.segment.astype(int)
    c = targets(c, args.v_min, args.ax_th, args.sr_th)

    over = (w.safety_flag > 0.5).to_numpy()
    print(f"窓 {args.win:g} 秒 / margin {args.margin} / 総走行 "
          f"{w.dur_s.sum() / 3600:.2f} 時間 / override 窓 {int(over.sum())}")
    print(f"対象速度 >= {args.v_min:.2f} m/s ({args.v_min * 3.6:.0f} km/h)")
    print(f"\n候補 {len(c)} 件の内訳")
    print(f"  速度条件を満たす        {int(c.speed_ok.sum())}")
    print(f"  回避反応あり            {int(c.reaction.sum())}")
    print(f"  安全系作動あり          {int(c.safety.sum())}")
    print(f"  **主対象 T1**           {int(c.T1.sum())}")
    print(f"  前方接近系 T2           {int(c.T2.sum())} "
          f"(うち反応あり {int((c.T2 & c.reaction).sum())})")

    ev = Eval(w, c, args.win, args.margin)
    avail = [k for k in FEATURES if w[k].notna().any()]
    print(f"\n使える量: {', '.join(avail)}")
    miss = [k for k in FEATURES if k not in avail]
    if miss:
        print(f"欠測で使えない量: {', '.join(miss)}")

    if args.search:
        _search(w, c, ev, over, avail, args)
    if args.labels:
        _labels(args, w, over)
    return 0


def _row(ev, c, over, cond) -> dict:
    r = ev(cond & ~over)
    a = r["_alive"]
    e = r["_ev"]
    d = {"削減率": r["削減率"], "全候補": 100 * a.mean()}
    for k in ("T1", "T2"):
        m = e[k].to_numpy()
        d[k] = 100 * a[m].mean() if m.any() else np.nan
    m = (e.T2 & e.reaction).to_numpy()
    d["T2反応あり"] = 100 * a[m].mean() if m.any() else np.nan
    m = (e.T2 & ~e.reaction).to_numpy()
    d["T2反応なし"] = 100 * a[m].mean() if m.any() else np.nan
    return d


def _search(w, c, ev, over, avail, args) -> None:
    print("\n=== 条件の探索 (T1 保持率を最優先) ===")
    best: list[dict] = []
    for combo in COMBOS:
        combo = tuple(k for k in combo if k in avail)
        if not combo:
            continue
        grids = [FEATURES[k][1] for k in combo]
        rows = []
        for vals in itertools.product(*grids):
            spec = dict(zip(combo, vals))
            cond = cond_of(w, spec)
            if not cond.any():
                continue
            d = _row(ev, c, over, cond)
            d["条件"] = "  ".join(f"{k}<{v:g}" for k, v in spec.items()
                                  if np.isfinite(v))
            rows.append(d)
        if not rows:
            continue
        R = pd.DataFrame(rows)
        for lim in (100.0, 99.0, 98.0):
            s = R[R.T1 >= lim].sort_values("削減率", ascending=False)
            if not s.empty:
                x = s.iloc[0].to_dict()
                x["T1下限"] = lim
                x["量"] = "+".join(combo)
                best.append(x)
                break
    if not best:
        print("  条件が見つかりません")
        return
    B = pd.DataFrame(best).sort_values("削減率", ascending=False)
    print(f"{'削減率':>7}{'T1':>7}{'T2':>7}{'T2反応有':>9}{'T2反応無':>9}"
          f"{'全候補':>7}  条件")
    for _, x in B.iterrows():
        f = lambda v: "  -  " if not np.isfinite(v) else f"{v:5.1f}"  # noqa: E731
        print(f"{x['削減率']:>6.1f}%{f(x['T1'])}%{f(x['T2'])}%{f(x['T2反応あり'])}%"
              f"{f(x['T2反応なし'])}%{f(x['全候補'])}%  {x['条件']}")


def _labels(args, w, over) -> None:
    lab = pd.read_csv(args.labels).dropna(subset=["t_in_segment_s"]).copy()
    if "risky" not in lab.columns:
        return
    lab["duration_s"] = lab.t_end - lab.t_start
    print("\n=== 人手ラベル (代表条件) ===")
    ev = Eval(w, lab, args.win, args.margin)
    for name, spec in (("操作入力のみ",
                        {"brake_press_max": 0.05, "gas_range": 1.0,
                         "steer_range": 3.0, "steer_rate_absmax": 15.0}),
                       ("操作+運動",
                        {"brake_press_max": 0.05, "gas_range": 1.0,
                         "steer_rate_absmax": 15.0, "ax_absmax": 0.8,
                         "ay_absmax": 0.8})):
        r = ev(cond_of(w, spec) & ~over)
        a, e = r["_alive"], r["_ev"]
        rk = e.risky.astype(bool).to_numpy()
        print(f"  {name:<12} 削減 {r['削減率']:5.1f}%  "
              f"risky {int(a[rk].sum())}/{int(rk.sum())}  "
              f"非risky {int(a[~rk].sum())}/{int((~rk).sum())}")


if __name__ == "__main__":
    raise SystemExit(main())
