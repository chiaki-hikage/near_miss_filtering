#!/usr/bin/env python3
"""jerk_feasibility.py の結果を図にする。

    uv run python scripts/validate/plot_jerk_feasibility.py --dir out/jerk_feasibility
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from near_miss.plotting import configure_japanese_font

COLORS = {"speed": "#C8551F", "accel_x": "#2F6DB5", "gvc": "#3B8F5A", "imu": "#8A5BB5", "current": "#555555"}
LABELS = {"speed": "車速を 2 回微分", "accel_x": "ACCEL_X を 1 回微分", "gvc": "GVC を 1 回微分",
          "imu": "IMU 前後を 1 回微分", "current": "現行 (20 Hz・移動平均 0.25 s)"}
DATASETS = {"comma2k19": "comma2k19 (RAV4 2017)", "ccs": "commaCarSegments (RAV4 TSS2)"}


def lag_corrected(pairs: dict, ref: str, fcs=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0), max_lag=30) -> pd.DataFrame:
    """制動前後の窓 (−1〜+2 s、10 ms 刻み) で、時間ずれを ±0.3 s 探して相関が最大になる値を求める。

    信号源ごとに遅れが違う (GVC は ACCEL_X より約 0.1 s 遅い) ので、ずれ 0 の相関では一致度を低く見積もる。
    x(t) と ref(t + lag) を比べる。lag が正なら ref のほうが遅れている。
    """
    rows = []
    for fc in fcs:
        b = pairs.get(f"{ref}@{fc}")
        for src in ("speed", "accel_x", "gvc", "imu"):
            a = pairs.get(f"{src}@{fc}")
            if src == ref or a is None or b is None:
                continue
            m = min(len(a), len(b))
            A, B = np.array(a[:m]), np.array(b[:m])
            best = None
            for lag in range(-max_lag, max_lag + 1):
                x, y = A[:, 40:260], B[:, 40 + lag:260 + lag]
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 1000:
                    continue
                c = float(np.corrcoef(x[ok], y[ok])[0, 1])
                if best is None or c > best[1]:
                    best = (lag * 10, c)
            if best:
                rows.append(dict(source=src, ref=ref, fc_hz=fc, lag_ms=best[0], corr=best[1]))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("out/jerk_feasibility"))
    args = ap.parse_args()
    matplotlib.use("Agg")
    configure_japanese_font()
    import matplotlib.pyplot as plt

    names = [n for n in DATASETS if (args.dir / f"{n}_summary.csv").exists()]

    # 1. ノイズ床・強い制動の立ち上がり・独立センサとの一致度 (カットオフ周波数ごと)
    fig, axs = plt.subplots(3, len(names), figsize=(6.2 * len(names), 11), squeeze=False)
    for c, n in enumerate(names):
        s = pd.read_csv(args.dir / f"{n}_summary.csv")
        cur = s[s.source == "current"].iloc[0]
        for src, g in s[s.source != "current"].groupby("source"):
            axs[0, c].plot(g.fc_hz, g.cruise_p99, "o-", color=COLORS[src], label=LABELS[src])
            axs[1, c].plot(g.fc_hz, -g.onset_hard_median, "o-", color=COLORS[src], label=LABELS[src])
        pairs = np.load(args.dir / f"{n}_pairs.npy", allow_pickle=True).item()
        lc = pd.concat([lag_corrected(pairs, "accel_x")] + ([lag_corrected(pairs, "imu")] if n == "comma2k19" else []))
        lc.to_csv(args.dir / f"{n}_lag_corrected_corr.csv", index=False)
        for (src, ref), g in lc.groupby(["source", "ref"]):
            if n == "comma2k19" and ref == "accel_x" and src == "imu":
                continue
            axs[2, c].plot(g.fc_hz, g["corr"], "o-", color=COLORS[src], ls="-" if ref == "accel_x" else "--",
                           label=f"{LABELS[src]} ↔ {ref}（ずれ {g.lag_ms.median():+.0f} ms を補正）")
        axs[0, c].axhline(cur.cruise_p99, color=COLORS["current"], ls="--", label=LABELS["current"])
        axs[1, c].axhline(-cur.onset_hard_median, color=COLORS["current"], ls="--", label=LABELS["current"])
        axs[0, c].set_yscale("log")
        axs[0, c].set_ylabel("巡航中の |jerk| の p99 [m/s³]\n(ノイズ床の上限)")
        axs[1, c].set_ylabel("強い制動 (2.5 m/s² 以上) の\n立ち上がり jerk の中央値 [m/s³]")
        axs[2, c].set_ylabel("制動前後の jerk 波形の相関\n(別の信号源との一致度、時間ずれ補正後)")
        axs[2, c].set_ylim(-0.1, 1.05)
        axs[0, c].set_title(DATASETS[n], fontsize=11)
        for r in range(3):
            axs[r, c].grid(alpha=0.3, which="both")
            axs[r, c].set_xscale("log")
            axs[r, c].set_xticks([0.5, 1, 2, 3, 5, 8])
            axs[r, c].set_xticklabels(["0.5", "1", "2", "3", "5", "8"])
            axs[r, c].legend(fontsize=8)
        axs[2, c].set_xlabel("低域通過のカットオフ周波数 [Hz]")
    fig.tight_layout()
    fig.savefig(args.dir / "jerk_by_bandwidth.png", dpi=110)
    plt.close(fig)

    # 2. 誤検出 (件/時) と強い制動の検出率
    fig, axs = plt.subplots(1, len(names), figsize=(6.2 * len(names), 5), squeeze=False)
    for c, n in enumerate(names):
        f = pd.read_csv(args.dir / f"{n}_false_alarm.csv")
        ax = axs[0, c]
        for (src, fc), g in f.groupby(["source", "fc"]):
            if fc not in (0.0, 1.0, 2.0, 3.0):
                continue
            ls = {0.0: "--", 1.0: ":", 2.0: "-", 3.0: "-."}[fc]
            ax.plot(g.fa_per_hour + 0.05, g.hit_hard, marker="o", ls=ls, color=COLORS[src],
                    label=f"{LABELS[src]}" + ("" if src == "current" else f" {fc:g} Hz"))
            for r in g.itertuples():
                if src in ("accel_x", "speed") and fc == 2.0:
                    ax.annotate(f"{r.thr:g}", (r.fa_per_hour + 0.05, r.hit_hard), fontsize=7, color=COLORS[src])
        ax.set_xscale("log")
        ax.set_xlabel("ブレーキ無しの走行中の誤検出 [件/時] (+0.05)")
        ax.set_ylabel("強い制動での検出率")
        ax.set_title(f"{DATASETS[n]}  数字はしきい値 [m/s³]", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.dir / "jerk_false_alarm.png", dpi=110)
    plt.close(fig)

    # 3. 強い制動の波形例 (信号源ごと、2 Hz と 5 Hz)
    for n in names:
        ev = pd.read_csv(args.dir / f"{n}_events.csv")
        pairs = np.load(args.dir / f"{n}_pairs.npy", allow_pickle=True).item()
        hard = ev.index[ev.peak_decel >= 2.5].tolist()
        hard = [i for i in hard if i < 400][:3]
        if not hard:
            continue
        fig, axs = plt.subplots(len(hard), 2, figsize=(12, 3.2 * len(hard)), squeeze=False, sharex=True)
        t = np.arange(-1.0, 2.0, 0.01)
        for r, i in enumerate(hard):
            for c, fc in enumerate((2.0, 5.0)):
                ax = axs[r, c]
                for src in ("speed", "accel_x", "gvc", "imu"):
                    key = f"{src}@{fc}"
                    if key in pairs and i < len(pairs[key]):
                        y = pairs[key][i]
                        if np.isfinite(y).any():
                            ax.plot(t[:len(y)], y, color=COLORS[src], lw=1.2, label=LABELS[src])
                ax.axvline(0, color="0.5", ls="--", lw=0.8)
                ax.grid(alpha=0.3)
                ax.set_ylabel("jerk [m/s³]")
                ax.set_title(f"{ev.segment[i][:28]}  最大減速 {ev.peak_decel[i]:.1f} m/s²  カットオフ {fc:g} Hz", fontsize=9)
        axs[0, 0].legend(fontsize=8)
        for c in range(2):
            axs[-1, c].set_xlabel("ブレーキを踏み始めてからの時間 [s]")
        fig.tight_layout()
        fig.savefig(args.dir / f"{n}_hard_brake_examples.png", dpi=110)
        plt.close(fig)
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
