#!/usr/bin/env python3
"""横滑り抽出の 2 枝の結果を突き合わせ、**種別ごとに**並べる。

screen_sideslip.py の出力 (candidates.csv / direct_slip.csv / overlap.csv) を読み、
「どの候補が何のスリップか」を分けて出す。走査を広げると候補が増えるので、
毎回同じ基準で並べ替えられるようにしておく。

種別の切り分けは**測れる量だけ**で行う。

  駆動輪の空転   輪速の開きが大きく、加速側 (ax > 0)
  車輪のロック   ABS が立っている、または輪速の開きが大きく減速側 (ax < 0)
  横方向の滑り   上のどちらでもなく、舵で説明できない横滑り (beta_excess) が残る

**横方向の滑りだけは間接推定に頼る。** 直接証拠が無いので、beta_excess と
ヨー残差で見るしかない。信号異常が上位に来やすいので、
ay の 2 系統が整合しているかを併せて出す (docs/sideslip_filter.md §9)。

  uv run python scripts/rank_slip.py out/sideslip_rav4_tss2_full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

KIND_SPIN = "駆動輪の空転"
KIND_LOCK = "車輪のロック"
KIND_LATERAL = "横方向の滑り"
KIND_UNKNOWN = "判別不能"


def classify(r: pd.Series, ws_min: float) -> str:
    ws = float(r.get("ws_spread_peak_mps", np.nan))
    ax = float(r.get("ax_at_peak_mps2", np.nan))
    if bool(r.get("abs_active", False)):
        return KIND_LOCK
    if np.isfinite(ws) and ws >= ws_min:
        if np.isfinite(ax) and ax < 0:
            return KIND_LOCK
        return KIND_SPIN
    if bool(r.get("slip_warn", False)):
        # TRC/VSC は駆動側でも横方向でも立つ。輪速で分からなければ横方向に寄せる。
        return KIND_LATERAL if not np.isfinite(ws) or ws < ws_min else KIND_SPIN
    return KIND_UNKNOWN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path, help="screen_sideslip.py の出力ディレクトリ")
    p.add_argument("--ws-min", type=float, default=2.0,
                   help="空転・ロックとみなす輪速の開き [m/s]")
    p.add_argument("--top", type=int, default=15)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    d_path = args.out_dir / "direct_slip.csv"
    b_path = args.out_dir / "candidates.csv"
    direct = pd.read_csv(d_path) if d_path.is_file() and d_path.stat().st_size > 1 else pd.DataFrame()
    beta = pd.read_csv(b_path) if b_path.is_file() and b_path.stat().st_size > 1 else pd.DataFrame()

    pd.set_option("display.width", 250)
    print(f"直接証拠 {len(direct)} 件 / beta 候補 {len(beta)} 件\n")

    if not direct.empty:
        direct = direct.copy()
        direct["kind"] = direct.apply(classify, axis=1, ws_min=args.ws_min)
        print("=" * 72)
        print("枝 2 (直接証拠) — 種別 × 等級")
        print("=" * 72)
        print(pd.crosstab(direct.kind, direct.grade, margins=True).to_string())
        cols = [c for c in ("grade", "kind", "evidence", "drive_id", "segment",
                            "t_in_segment_s", "duration_s", "slip_warn_peak",
                            "ws_spread_peak_mps", "v_max_mps", "ax_at_peak_mps2",
                            "ay_at_peak_mps2", "steer_at_peak_deg",
                            "beta_model_peak_deg") if c in direct.columns]
        for kind in (KIND_LATERAL, KIND_SPIN, KIND_LOCK, KIND_UNKNOWN):
            g = direct[direct.kind == kind]
            if g.empty:
                continue
            print(f"\n--- {kind} ({len(g)} 件) ---")
            print(g[cols].head(args.top).round(2).to_string(index=False))

    if not beta.empty:
        print("\n" + "=" * 72)
        print("枝 1 (beta) — 舵で説明できない量の大きい順")
        print("=" * 72)
        b = beta.copy()
        b["absx"] = b.beta_excess_peak_deg.abs()
        # ay の 2 系統が離れている区間は信号異常を疑う。並べ替えの目安として出す。
        cols = [c for c in ("grade", "confidence", "confidence_items", "drive_id",
                            "peak_segment", "peak_t_in_segment_s", "beta_peak_deg",
                            "beta_excess_peak_deg", "yaw_sigma_peak", "ay_peak_mps2",
                            "v_at_peak_mps", "duration_s", "corroboration")
                if c in b.columns]
        print(b.nlargest(args.top, "absx")[cols].round(2).to_string(index=False))
        print(f"\n等級別: {b.grade.value_counts().to_dict()}")
        print(f"|beta| 最大 {b.beta_peak_deg.abs().max():.2f} deg / "
              f"|beta_excess| 最大 {b.absx.max():.2f} deg")

    ov_path = args.out_dir / "overlap.csv"
    if ov_path.is_file() and ov_path.stat().st_size > 1:
        ov = pd.read_csv(ov_path)
        print("\n" + "=" * 72)
        print("2 枝の重なり")
        print("=" * 72)
        print(ov.groupby(["beta", "direct"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
