#!/usr/bin/env python3
"""横滑り抽出の 2 枝の結果を突き合わせ、**種別ごとに**並べる。

screen_sideslip.py の出力 (candidates.csv / direct_slip.csv / overlap.csv) を読み、
「どの候補が何のスリップか」を分けて出す。走査を広げると候補が増えるので、
毎回同じ基準で並べ替えられるようにしておく。

種別の切り分けは**測れる量だけ**で行う。

  横方向の不安定 (候補)  VSC が立った、または |ay| が対照 p99.9 を超える、
                         または逆操舵。**積極的な条件を満たしたものだけ**
  駆動輪の空転           輪速の開きが大きく、加速側 (ax > 0)
  車輪のロック           ABS が立っている、または輪速の開きが大きく減速側 (ax < 0)
  未分類のスリップ事象   作動はしたが、どの向きの滑りかを測れる量で言えないもの

**残りものを横方向に寄せない。** 「ABS でもなく輪速も開いていない」を横方向と
みなすと、輪速差が閾値をわずかに下回っただけの空転 (ax > 0 で加速中) が
横滑りとして並ぶ。横方向は積極的な条件を満たしたものだけに限る。

  uv run python scripts/detect/rank_slip.py out/sideslip_rav4_tss2_full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

KIND_SPIN = "駆動輪の空転"
KIND_LOCK = "車輪のロック"
KIND_LATERAL = "横方向の不安定 (候補)"
KIND_UNCLASSIFIED = "未分類のスリップ事象"

# 横方向とみなす閾値。対照窓 (走行中の無作為な時刻 10.3 万件、4 車種) の分位から取る。
#   |ay|  p95 2.4〜2.5 / p99 3.6〜4.0 / p99.9 4.9〜5.3 m/s^2
#   |yaw| p95 11〜14  / p99 26〜28  / p99.9 34〜38 deg/s
# yaw は低速の交差点旋回でも p99 が 27 deg/s まで出るので、yaw 単独では横方向と
# 言えない。ay を主に使い、yaw は逆操舵の判定にだけ添える。
AY_HIGH = 5.0     # 対照 p99.9。これだけで横方向の外乱とみなす
AY_MID = 3.9      # 対照 p99。逆操舵と組で使う
YAW_MIN = 10.0    # 逆操舵の判定に要る最低限のヨー
STEER_MIN = 30.0  # 同、舵角


def lateral_reason(r: pd.Series) -> str:
    """横方向と**積極的に**言える理由。無ければ空文字。

    残りもの (ABS でもなく輪速も開いていないもの) を横方向に寄せることはしない。
    寄せると、輪速差が閾値をわずかに下回っただけの空転が横滑りとして並んでしまう。
    """
    ay = abs(float(r.get("ay_at_peak_mps2", np.nan)))
    ax = abs(float(r.get("ax_at_peak_mps2", np.nan)))
    yaw = float(r.get("yaw_rate_at_peak_dps", np.nan))
    steer = float(r.get("steer_at_peak_deg", np.nan))
    if bool(r.get("vsc_active", False)):
        return "VSC 作動"                       # 横方向の制御介入そのもの
    if np.isfinite(ay) and ay >= AY_HIGH and (not np.isfinite(ax) or ay > ax):
        return f"|ay| {ay:.1f} >= {AY_HIGH} (対照 p99.9)"
    if (np.isfinite(yaw) and np.isfinite(steer) and np.isfinite(ay)
            and ay >= AY_MID and abs(yaw) >= YAW_MIN and abs(steer) >= STEER_MIN
            and yaw * steer < 0):
        # 舵とヨーの符号は同じになるのが普通 (4 車種で相関 0.94〜0.96)。
        # 逆符号は逆操舵か、舵に追従しないヨーが出ている状態。
        return f"逆操舵 (舵 {steer:.0f} deg / ヨー {yaw:.0f} deg/s) かつ |ay| {ay:.1f}"
    return ""


def lateral_confidence(reason: str) -> str:
    """横方向の理由の確からしさ。**VSC だけが直接の証拠**。

    |ay| は v x ヨーレートから出しているので、大きいことは「強く曲がっている」
    ことしか意味しない。実測 187 件では、この理由で拾った 9 件のうち多くが
    加速しながらの小回り (舵 500 deg 前後) だった。ヨー残差で確かめるまでは
    不安定と断定できない (scripts/validate/eval_lateral_indicators.py)。
    """
    return "確証" if reason.startswith("VSC") else ("要検証" if reason else "")


def classify(r: pd.Series, ws_min: float) -> str:
    ws = float(r.get("ws_spread_peak_mps", np.nan))
    ax = float(r.get("ax_at_peak_mps2", np.nan))
    if lateral_reason(r):
        return KIND_LATERAL
    if bool(r.get("abs_active", False)):
        return KIND_LOCK
    if np.isfinite(ws) and ws >= ws_min:
        if np.isfinite(ax) and ax < 0:
            return KIND_LOCK
        return KIND_SPIN
    # ここに落ちるのは「作動はしたが、どの向きの滑りかを測れる量で言えない」もの。
    # 横方向に寄せずに未分類として残す。
    return KIND_UNCLASSIFIED


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
        direct["lateral_reason"] = direct.apply(lateral_reason, axis=1)
        direct["lateral_confidence"] = direct.lateral_reason.map(lateral_confidence)
        print("=" * 72)
        print("枝 2 (直接証拠) — 種別 × 等級")
        print("=" * 72)
        print(pd.crosstab(direct.kind, direct.grade, margins=True).to_string())
        cols = [c for c in ("grade", "kind", "lateral_confidence", "lateral_reason", "evidence", "drive_id",
                            "segment", "t_in_segment_s", "duration_s", "slip_warn_peak",
                            "ws_spread_peak_mps", "v_max_mps", "ax_at_peak_mps2",
                            "ay_at_peak_mps2", "yaw_rate_at_peak_dps", "steer_at_peak_deg",
                            "beta_model_peak_deg") if c in direct.columns]
        for kind in (KIND_LATERAL, KIND_SPIN, KIND_LOCK, KIND_UNCLASSIFIED):
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
