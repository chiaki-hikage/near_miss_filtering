"""ISO 2631-1 を参考にした MTVV で、通常走行・振動イベント・異常振動候補の振動の強さを比べる図。

iso2631_events.py が出した値をそのまま使い、計算はやり直さない。
横軸は MTVV 合成 [m/s²]、背景は附属書 C.2.3 の快適性の目安 (範囲の重なりは上側を採る。iso2631.comfort_label と同じ)。
描くのは全帯域の値。上下の 8 Hz 以上を除いた値 (IMU 取付部の共振の影響を見る感度) は注記に中央値だけ載せる。

    uv run python scripts/vibration/plot_iso2631_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iso2631 import COMFORT
from near_miss.plotting import configure_japanese_font

REPO = _bootstrap.SRC.parent
CANDIDATE_COLS = ["strong_vertical", "strong_roll", "simultaneous_vertical_roll", "impulsive_response",
                  "road_related_evidence"]
# 色 (dataviz の既定パレット 1・2 番。通常走行は無彩色)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
NORMAL, EVENT, CAND = "#8f8e88", "#2a78d6", "#eb6834"
BAND = ("#fcfcfb", "#f1f0ec")
# 図での表記だけを変える (iso2631.COMFORT と既存の出力の表記はそのまま)
LABEL = {"かなり不快": "やや不快"}


def comfort_bands(xmax: float) -> list[tuple[float, float, str]]:
    """重なりを上側で切った区分 (下限, 上限, 表現)。"""
    lows = [lo for lo, _, _ in COMFORT]
    return [(lo, (lows[i + 1] if i + 1 < len(lows) else xmax), LABEL.get(name, name))
            for i, (lo, _, name) in enumerate(COMFORT)]


def strip(ax, y: float, v: np.ndarray, color: str, size: float, alpha: float, rng, ring: bool = False) -> None:
    jit = rng.uniform(-0.1, 0.3, len(v))
    ax.scatter(v, y + jit, s=size, color=color, alpha=alpha, linewidths=1.2 if ring else 0,
               edgecolors="white" if ring else "none", zorder=3)


def percentile_bar(ax, y: float, v: np.ndarray) -> None:
    """10〜90% の範囲を線、中央値を縦線で示す。"""
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    yb = y - 0.45
    ax.plot([p10, p90], [yb, yb], color=INK, lw=2, solid_capstyle="round", zorder=4)
    ax.plot([p50, p50], [yb - 0.1, yb + 0.1], color=INK, lw=2, zorder=4)
    ax.text(p90 + 0.03, yb, f"中央値 {p50:.2f}", ha="left", va="center", fontsize=9, color=INK)


def panel(ax, groups: list[tuple[str, np.ndarray, str, float, float, bool]], xmax: float, title: str,
          rng, show_band_labels: bool) -> None:
    for i, (lo, hi, name) in enumerate(comfort_bands(xmax)):
        ax.axvspan(lo, min(hi, xmax), color=BAND[i % 2], zorder=0, lw=0)
        if lo > 0:
            ax.axvline(lo, color=GRID, lw=0.8, zorder=1)
        if show_band_labels:
            ax.text((lo + min(hi, xmax)) / 2, len(groups) - 0.35, name, ha="center", va="bottom",
                    fontsize=9.5, color=INK2)
    for k, (label, v, color, size, alpha, ring) in enumerate(groups):
        y = len(groups) - 1 - k
        strip(ax, y, v, color, size, alpha, rng, ring)
        percentile_bar(ax, y, v)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups][::-1], fontsize=10.5, color=INK)
    ax.set_ylim(-0.7, len(groups) - 0.1)
    ax.set_xlim(0, xmax)
    if title:
        ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=22)
    ax.tick_params(axis="x", colors=INK2, labelsize=9.5)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=REPO / "out/vibration/near_miss")
    ap.add_argument("--out", type=Path, default=REPO / "out/vibration/near_miss/iso2631_comparison.png")
    ap.add_argument("--seed", type=int, default=0, help="点を縦にばらす乱数の種")
    args = ap.parse_args()
    configure_japanese_font()

    ev = pd.read_csv(args.dir / "iso2631_events.csv")
    nw = pd.read_csv(args.dir / "iso2631_normal_windows.csv")
    flags = pd.read_csv(args.dir / "near_miss_vibration_events.csv", usecols=["event_id", *CANDIDATE_COLS])
    ev = ev.merge(flags, on="event_id", how="left", validate="one_to_one")
    cand = ev[CANDIDATE_COLS].fillna(False).astype(bool).all(axis=1)

    xmax = 2.5
    col, lp8 = "mtvv_total", "mtvv_total_lp8"
    fig, ax = plt.subplots(figsize=(11, 4.6))
    rng = np.random.default_rng(args.seed)
    if True:
        groups = [
            (f"通常走行\n({len(nw):,} 窓)", nw[col].to_numpy(), NORMAL, 7, 0.35, False),
            (f"振動イベント\n(候補以外 {int((~cand).sum()):,} 件)", ev.loc[~cand, col].to_numpy(), EVENT, 7, 0.35, False),
            (f"異常振動候補\n({int(cand.sum())} 件)", ev.loc[cand, col].to_numpy(), CAND, 46, 0.95, True),
        ]
        over = int((np.concatenate([g[1] for g in groups]) > xmax).sum())
        assert over == 0, f"{col}: 横軸の範囲 {xmax} を超える値が {over} 件ある"
        panel(ax, groups, xmax, "", rng, show_band_labels=True)
    med = {k: (float(nw[k].median()), float(ev.loc[cand, k].median())) for k in (col, lp8)}
    ax.set_xlabel("MTVV 合成 [m/s²] (1 秒移動 RMS の最大値、座面の重み付け)", fontsize=10, color=INK2)
    fig.suptitle("抽出した異常振動候補は、通常走行より強い振動の側に分布する", x=0.06, ha="left", fontsize=14,
                 color=INK, fontweight="bold")
    fig.text(0.06, 0.005,
             "点は 1 窓・1 イベント (縦方向は重なりを避けるためにばらしただけで意味はない)。黒線は 10〜90% の範囲、縦線は中央値。"
             "\n背景は ISO 2631-1 附属書 C.2.3 の快適性の目安 (範囲の重なりは上側を採る)。"
             "通常走行は、ジャイロ 104 Hz のセグメント 120 本から 5 秒おきに取った窓 (5 m/s 以上)。"
             "\nIMU はフロントガラスの取付部にあり座面ではないため、絶対値は座面での評価と一致しない可能性がある。"
             f"\n上下の 8 Hz 以上 (ばね下・取付部の共振) を除いても、中央値は通常走行 {med[lp8][0]:.2f}・候補 {med[lp8][1]:.2f} m/s² で、"
             "候補が通常走行より強い側にあるという結論は変わらない。",
             fontsize=8.5, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.86, bottom=0.30)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="#fcfcfb")
    print(f"→ {args.out}")
    for k, (a, b) in med.items():
        print(k, "中央値 通常", round(a, 2), "候補", round(b, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
