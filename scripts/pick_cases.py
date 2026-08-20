#!/usr/bin/env python3
"""スクリーニング結果から、確認したい類型ごとに代表候補を選ぶ。

候補は severity 順に並んでいるが、上位は件数の多い割り込み・車間で埋まる。
「どの組み合わせを見たいか」を先に決めて、その条件に当たるものだけを取り出す。

  python scripts/pick_cases.py out/screen_rav4_tss2_2k --top 3
  python scripts/pick_cases.py out/screen_rav4_tss2_2k --case abs_aeb --top 5
  python scripts/pick_cases.py out/screen_rav4_tss2_2k --plot        # プロットまで実行

--plot を付けると scripts/plot_segment.py を選んだ候補ぶん呼び出す。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401

from near_miss.io.comma_car_segments import DEFAULT_CACHE


def _types(s) -> set[str]:
    return set(str(s).split("|")) if isinstance(s, str) and s else set()


def _all(df: pd.DataFrame, *ev: str) -> pd.Series:
    return df.event_types.apply(lambda s: set(ev) <= _types(s))


def _any(df: pd.DataFrame, *ev: str) -> pd.Series:
    return df.event_types.apply(lambda s: bool(set(ev) & _types(s)))


# 見たい類型。名前 -> (説明, 条件, 表に添える列)
CASES: dict[str, tuple[str, callable, tuple[str, ...]]] = {
    "abs_aeb": (
        "純正 AEB / PCS の作動、または ABS を伴う車間逼迫",
        lambda d: _any(d, "aeb_active") | (_all(d, "abs_active") & _any(d, "low_ttc", "short_thw")),
        ("ax_mps2_min", "ttc_s_min", "thw_s_min", "brake_mc_mpa_max"),
    ),
    "yaw_counter": (
        "ヨー応答の乖離 + 逆操舵 (車両が舵に従っていない)",
        lambda d: _all(d, "yaw_instability", "counter_steer"),
        ("yaw_residual_dps_absmax", "steer_rate_dps_absmax", "ay_kin_mps2_absmax",
         "net_heading_win_deg_max"),
    ),
    "abs_wheel": (
        "ABS 作動 または 輪速の異常 (車輪ロック)",
        lambda d: _all(d, "abs_active") | _all(d, "wheel_speed_anomaly"),
        ("ax_mps2_min", "ws_spread_excess_mps_max", "brake_mc_mpa_max", "slip_warn_max"),
    ),
    "brake_steer": (
        "制動 + 操舵の回避",
        lambda d: _all(d, "hard_brake", "hard_steer") | _any(d, "brake_and_steer", "brake_after_evasion"),
        ("ax_mps2_min", "ay_kin_mps2_absmax", "steer_rate_dps_absmax", "net_heading_win_deg_max"),
    ),
    "ttc_panic": (
        "車間が詰まった状態での急制動",
        lambda d: (d.ttc_s_min < 3.0)
        & _any(d, "panic_brake", "panic_brake_pedal", "panic_brake_with_lead", "aeb_active", "hard_brake"),
        ("ttc_s_min", "thw_s_min", "ax_mps2_min", "lead_distance_m_min", "gas_pedal_pct_max"),
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("screen_dir", type=Path, help="screen_segments.py の出力ディレクトリ")
    p.add_argument("--case", action="append", choices=list(CASES), help="類型を絞る (既定は全部)")
    p.add_argument("--top", type=int, default=3, help="類型ごとの件数")
    p.add_argument("--plot", action="store_true", help="plot_segment.py を呼んで PNG を作る")
    p.add_argument("--data-root", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--platform", default="TOYOTA_RAV4_TSS2")
    p.add_argument("--panels", default="speed,accel_x,brake,pedal,lead,physics,wheel_spread,steer,op")
    p.add_argument("--context", type=int, default=1, help="前後この数のセグメントを繋いで描く")
    p.add_argument("--span", type=float, default=24.0, help="拡大図の幅 [s]")
    p.add_argument("--no-zoom", action="store_true", help="拡大図を作らない")
    p.add_argument("--out", type=Path, default=None, help="PNG の出力先 (既定は screen_dir/plots)")
    return p.parse_args()


def focus_point(events: pd.DataFrame, cand: pd.Series, rarity: dict[str, int]) -> tuple[int, float]:
    """拡大位置を (セグメント番号, セグメント内の秒) で返す。

    候補は前後 2 秒を付けて統合してあるので、長いものは 50 秒に達する。
    候補の中点では見たい事象が画面の外に出る。
    候補の中で「最も珍しいイベント」を中心にする。
    セグメント番号もそのイベントのものを使う。候補の t_start は余白を含むぶん
    1 つ前のセグメントに割り当たることがある。
    """
    fallback = (int(cand.segment), float(cand.t_in_segment_s + cand.duration_s / 2))
    if events.empty:
        return fallback
    m = ((events.drive_id == cand.drive_id)
         & (events.t_start < cand.t_end) & (cand.t_start < events.t_end))
    hit = events[m]
    if hit.empty:
        return fallback
    rare = hit.loc[hit.event_type.map(lambda e: rarity.get(e, 10**9)).idxmin()]
    return int(rare.segment), float(rare.t_in_segment_s + rare.duration_s / 2)


def main() -> int:
    args = parse_args()
    path = args.screen_dir / "ranked_candidates.csv"
    if not path.is_file():
        path = args.screen_dir / "candidates.csv"
    df = pd.read_csv(path)
    df["event_types"] = df["event_types"].fillna("")
    ev_path = args.screen_dir / "events.csv"
    events = pd.read_csv(ev_path) if ev_path.is_file() else pd.DataFrame()
    rarity = events.event_type.value_counts().to_dict() if not events.empty else {}

    picked: list[tuple[str, pd.Series]] = []
    for name in (args.case or list(CASES)):
        label, cond, cols = CASES[name]
        sub = df[cond(df)].sort_values("severity", ascending=False)
        print(f"\n{'=' * 92}")
        print(f"[{name}] {label}   該当 {len(sub)} 件")
        print("=" * 92)
        if sub.empty:
            print("  該当なし")
            continue
        show = ["drive_id", "segment", "t_in_segment_s", "duration_s", "severity", "event_types"]
        show += [c for c in cols if c in sub.columns]
        head = sub.head(args.top)
        print(head[show].round(2).to_string(index=False))
        picked += [(name, r) for _, r in head.iterrows()]

    if not args.plot:
        print("\n--plot を付けると、選んだ候補のプロットを作ります。")
        return 0

    out_dir = args.out or (args.screen_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 92}\nプロット {len(picked)} 件 -> {out_dir}\n{'=' * 92}")
    for name, r in picked:
        seg = f"{r.drive_id}/{int(r.segment)}"
        stem = f"{name}_{r.drive_id.split('|')[0][:8]}_seg{int(r.segment)}"
        base = [sys.executable, str(Path(__file__).parent / "plot_segment.py"),
                str(args.data_root), "--dataset", "comma_car_segments",
                "--platform", args.platform, "-s", seg,
                "--panels", args.panels, "--context", str(args.context)]
        print(f"\n--- {name}  {seg}  t={r.t_in_segment_s:.1f}s  severity={r.severity:.1f}")
        # 全体図と、候補の位置を中心にした拡大図の 2 枚
        runs = [(out_dir / f"{stem}.png", [])]
        if not args.no_zoom:
            fs, ft = focus_point(events, r, rarity)
            focus = f"{fs}@{ft:.2f}"
            runs.append((out_dir / f"{stem}_zoom.png",
                         ["--focus", focus, "--span", str(args.span)]))
        for out, extra in runs:
            res = subprocess.run(base + extra + ["--out", str(out)],
                                 capture_output=True, text=True)
            first = res.stdout.strip().splitlines()
            print("  " + (first[0] if first else res.stderr.strip()[-400:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
