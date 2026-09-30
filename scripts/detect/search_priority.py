#!/usr/bin/env python3
"""L3 の「高優先度ヒヤリハット候補」条件を探索する。

**単独閾値の強化ではなく組合せで絞る。** 使う材料は候補の集計値だけで、
新しい特徴量は作らない (L2 の出力をそのまま使う)。

    重複     n_events / n_types / 特定の組合せ
    強さ     severity
    安全系   ABS / PCS / VSC / slip_warn
    車間     thw_s_min / ttc_s_min / lead_distance_m_min
    回避反応 制動 / 減速 / 舵角レート

人手ラベル 32 件 (risky 8) で precision / recall を測る。
**positive 8 件は探索的な標本で、閾値の確定には足りない。**
件数を必ず併記し、割合だけで判断しない。

  uv run python scripts/detect/search_priority.py \
      --labels out/chunk1/label_features.csv \
      --candidates out/screen_rav4_tss2_2k/candidates.csv
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

SAFETY = ("abs_active_flag_max", "vsc_active_flag_max", "slip_warn_max",
          "precollision_active_max")


def col(d: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    return d[name] if name in d.columns else pd.Series(default, index=d.index)


def base(d: pd.DataFrame) -> pd.DataFrame:
    """条件で使う派生量。**候補の集計値からしか作らない。**"""
    d = d.copy()
    et = col(d, "event_types", "").fillna("")
    d["_n_types"] = col(d, "n_types", np.nan)
    if d["_n_types"].isna().all():
        d["_n_types"] = et.apply(lambda s: len(s.split("|")) if s else 0)
    d["_n_events"] = col(d, "n_events", 0).fillna(0)
    d["_sev"] = col(d, "severity", 0).fillna(0)
    d["_thw"] = col(d, "thw_s_min", np.inf).fillna(np.inf)
    d["_ttc"] = col(d, "ttc_s_min", np.inf).fillna(np.inf)
    d["_lead"] = col(d, "lead_distance_m_min", np.inf).fillna(np.inf)
    d["_ax"] = col(d, "ax_mps2_min", 0).fillna(0)
    d["_jerk"] = col(d, "jerk_mps3_min", 0).fillna(0)
    d["_sr"] = col(d, "steer_rate_dps_absmax", 0).fillna(0).abs()
    d["_ay"] = col(d, "ay_kin_mps2_absmax", 0).fillna(0).abs()
    d["_v"] = col(d, "v_mps_mean", 0).fillna(0)
    s = pd.Series(False, index=d.index)
    for c in SAFETY:
        s |= col(d, c, 0).fillna(0) > 0.5
    d["_safety"] = s
    d["_brake"] = col(d, "brake_pressed_max", 0).fillna(0) > 0.5
    # 回避反応: 制動 or 明確な減速 or 大きな操舵
    d["_react"] = d._brake | (d._ax < -1.5) | (d._sr > 100)
    d["_fwd"] = et.str.contains("cut_in_candidate|short_thw|low_ttc|closing_fast")
    return d


# 探索する条件の部品。名前 -> (説明, 判定関数)
PARTS: dict[str, tuple[str, object]] = {
    "thw08": ("車間時間 < 0.8 s", lambda d: d._thw < 0.8),
    "thw06": ("車間時間 < 0.6 s", lambda d: d._thw < 0.6),
    "thw10": ("車間時間 < 1.0 s", lambda d: d._thw < 1.0),
    "ttc30": ("衝突余裕 < 3.0 s", lambda d: d._ttc < 3.0),
    "ttc20": ("衝突余裕 < 2.0 s", lambda d: d._ttc < 2.0),
    "react": ("回避反応あり", lambda d: d._react),
    "brake": ("制動あり", lambda d: d._brake),
    "ax25": ("減速 < -2.5 m/s^2", lambda d: d._ax < -2.5),
    "ax15": ("減速 < -1.5 m/s^2", lambda d: d._ax < -1.5),
    "n2": ("イベント種別 2 以上", lambda d: d._n_types >= 2),
    "n3": ("イベント種別 3 以上", lambda d: d._n_types >= 3),
    "ev3": ("イベント数 3 以上", lambda d: d._n_events >= 3),
    "sev50": ("severity 上位50%", lambda d: d._sev >= d._sev.median()),
    "sev75": ("severity 上位25%", lambda d: d._sev >= d._sev.quantile(0.75)),
    "v10": ("平均速度 >= 10 m/s", lambda d: d._v >= 10),
    "fwd": ("前方接近系を含む", lambda d: d._fwd),
    "safety": ("安全系フラグ", lambda d: d._safety),
}


def stats(hit: pd.Series, pos: pd.Series) -> dict:
    n = int(hit.sum())
    tp = int((hit & pos).sum())
    return {"件数": n, "TP": tp, "FP": n - tp,
            "precision": 100 * tp / n if n else np.nan,
            "recall": 100 * tp / int(pos.sum())}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path,
                   default=Path("out/chunk1/label_features.csv"))
    p.add_argument("--candidates", type=Path, default=None,
                   help="採用率と映像時間を出す対象 (別データセットでもよい)")
    p.add_argument("--clip-s", type=float, default=20.0)
    p.add_argument("--hours", type=float, default=33.34,
                   help="candidates の元になった総走行時間")
    p.add_argument("--max-parts", type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    L = base(pd.read_csv(args.labels, low_memory=False))
    pos = L.risky.astype(bool)
    print(f"人手ラベル {len(L)} 件 (risky {int(pos.sum())} / "
          f"base rate {100 * pos.mean():.1f}%)")

    C = None
    if args.candidates:
        C = base(pd.read_csv(args.candidates, low_memory=False))
        print(f"適用先の候補 {len(C)} 件 / {args.hours:.2f} 時間")

    # --- 単独 ---
    print(f"\n=== 単独条件 ===")
    print(f"{'条件':<22}{'件数':>6}{'TP':>4}{'FP':>4}{'精度':>8}{'再現':>8}"
          + ("{:>9}{:>9}".format("採用率", "映像%") if C is not None else ""))
    rows = []
    for k, (desc, fn) in PARTS.items():
        s = stats(fn(L), pos)
        line = f"{desc:<22}{s['件数']:>6}{s['TP']:>4}{s['FP']:>4}" \
               f"{s['precision']:>7.0f}%{s['recall']:>7.0f}%"
        if C is not None:
            m = fn(C)
            line += f"{100 * m.mean():>8.1f}%{_video(m, args):>8.2f}%"
        print(line)
        rows.append((k, s))

    # --- 組合せ (連言) ---
    print(f"\n=== 連言 ({args.max_parts} 個まで / precision 順) ===")
    found = []
    keys = list(PARTS)
    for r in range(1, args.max_parts + 1):
        for combo in itertools.combinations(keys, r):
            m = pd.Series(True, index=L.index)
            for k in combo:
                m &= PARTS[k][1](L)
            s = stats(m, pos)
            if s["件数"] < 3 or s["TP"] < 2:      # 標本が小さすぎるものは出さない
                continue
            found.append((combo, s))
    _show(found, C, args, "precision")

    # --- 選言 (安全系 or 高精度条件) ---
    print(f"\n=== 「安全系 OR 高精度条件」の形 ===")
    found2 = []
    for r in (1, 2):
        for combo in itertools.combinations([k for k in keys if k != "safety"], r):
            m = PARTS["safety"][1](L)
            inner = pd.Series(True, index=L.index)
            for k in combo:
                inner &= PARTS[k][1](L)
            m = m | inner
            s = stats(m, pos)
            if s["件数"] < 3 or s["TP"] < 2:
                continue
            found2.append((("safety∨",) + combo, s))
    _show(found2, C, args, "precision")
    return 0


def _video(mask: pd.Series, args) -> float:
    return 100 * mask.sum() * args.clip_s / 3600 / args.hours


def _show(found, C, args, key: str) -> None:
    if not found:
        print("  該当なし")
        return
    seen = set()
    out = []
    for combo, s in sorted(found, key=lambda x: (-x[1][key], -x[1]["recall"])):
        sig = (s["件数"], s["TP"])
        if sig in seen:
            continue
        seen.add(sig)
        out.append((combo, s))
    print(f"{'条件':<44}{'件数':>5}{'TP':>4}{'FP':>4}{'精度':>7}{'再現':>7}"
          + ("{:>8}{:>8}".format("採用率", "映像%") if C is not None else ""))
    for combo, s in out[:18]:
        name = " ∧ ".join(PARTS[k][0] if k in PARTS else k for k in combo)
        line = f"{name[:43]:<44}{s['件数']:>5}{s['TP']:>4}{s['FP']:>4}" \
               f"{s['precision']:>6.0f}%{s['recall']:>6.0f}%"
        if C is not None:
            m = pd.Series(True, index=C.index)
            if combo and combo[0] == "safety∨":
                inner = pd.Series(True, index=C.index)
                for k in combo[1:]:
                    inner &= PARTS[k][1](C)
                m = PARTS["safety"][1](C) | inner
            else:
                for k in combo:
                    m &= PARTS[k][1](C)
            line += f"{100 * m.mean():>7.1f}%{_video(m, args):>7.2f}%"
        print(line)


if __name__ == "__main__":
    raise SystemExit(main())
