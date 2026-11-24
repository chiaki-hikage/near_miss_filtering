"""NCST の台上試験で、実測のブレーキ温度とブレーキの利きの関係を図にする。

利き = 定常部の減速度 ÷ (エア圧 − 押し出し圧)  [(m/s^2)/psi]。ncst_thermal.py の chassis_brake_events.csv を使う。
相対値: 50 °C 未満の制動の利きの中央値を 1 とする。
温度: その制動の間の熱電対の読み (ライニングの中。報告書は表面より約 100 °C 低い可能性を指摘)。

高温の制動 (UDDS) は車速が高いので、車速をそろえた図 (初速 5〜10 m/s の制動だけ) も並べる。
区間ごとの中央値の幅は bootstrap の 90% 区間。

    uv run python scripts/brake/ncst_gain_vs_temp.py --events out/brake_fade/20261118_r3_ncst_thermal/chassis_brake_events.csv --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from near_miss.plotting import configure_japanese_font

OUT = Path("out/brake_fade")
BINS = [0, 50, 90, 130, 150, 170]
COLORS = {"CBD0": "#7f8c8d", "CBD": "#2e86c1", "UDDS": "#c0392b"}
LABELS = {"CBD0": "CBD の暖機 (台上)", "CBD": "CBD × 3 (台上)", "UDDS": "UDDS × 3 (台上)"}


def binned(E: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        g = E[(E["T"] >= lo) & (E["T"] < hi)].rel.to_numpy()
        if len(g) < 3:
            continue
        bs = [np.median(rng.choice(g, len(g))) for _ in range(2000)]
        rows.append(dict(lo=lo, hi=hi, n=len(g), T_mid=float(E[(E["T"] >= lo) & (E["T"] < hi)]["T"].median()), median=float(np.median(g)),
                         ci5=float(np.percentile(bs, 5)), ci95=float(np.percentile(bs, 95))))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    out = OUT / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261118)
    E = pd.read_csv(a.events)
    E = E[E.decel > 0.2].reset_index(drop=True)
    ref = E[E["T"] < 50].gain.median()
    E["rel"] = E.gain / ref
    sets = {"all": E, "v5_10": E[(E.v0 >= 5) & (E.v0 <= 10)]}
    S = dict(events=a.events.as_posix(), ref_gain_below_50C=float(ref), n=int(len(E)))
    configure_japanese_font()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    titles = {"all": "全部の制動", "v5_10": "初速 5〜10 m/s の制動だけ (車速をそろえた)"}
    for ax, (k, D) in zip(axs, sets.items()):
        B = binned(D, rng)
        for tr, g in D.groupby("traj"):
            ax.scatter(g["T"], g.rel, s=22, color=COLORS[tr], alpha=0.75, edgecolor="white", linewidth=0.5, label=f"{LABELS[tr]} ({len(g)} 回)")
        ax.errorbar(B.T_mid, B["median"], yerr=[B["median"] - B.ci5, B.ci95 - B["median"]], fmt="s-", color="#222", ms=6, capsize=3, lw=1.5,
                    label="温度区間ごとの中央値 (90% 区間)")
        for _, r in B.iterrows():
            ax.annotate(f"n={int(r.n)}", (r.T_mid, r.ci95), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8, color="#444")
        ax.axhline(1, color="#888", lw=0.8)
        ax.set_xlabel("ブレーキ温度 (熱電対の読み) [°C]")
        ax.set_title(f"{titles[k]}  n={len(D)}")
        ax.grid(alpha=0.3)
        rs = D["T"].rank().corr(D.rel.rank())
        hot = D[D["T"] >= 130].rel
        cool = D[D["T"] < 130].rel
        S[k] = dict(n=int(len(D)), spearman=round(float(rs), 3), median_rel_below_130=round(float(cool.median()), 3) if len(cool) else None,
                    median_rel_130_plus=round(float(hot.median()), 3) if len(hot) else None, n_130_plus=int(len(hot)),
                    bins=B.round(3).to_dict("records"), v0_median_by_traj=D.groupby("traj").v0.median().round(2).to_dict())
        # 130 °C 以上と未満の中央値の差の bootstrap (90% 区間)
        if len(hot) >= 3 and len(cool) >= 3:
            d = [np.median(rng.choice(hot.to_numpy(), len(hot))) - np.median(rng.choice(cool.to_numpy(), len(cool))) for _ in range(4000)]
            S[k]["diff_median_130plus_minus_below_ci90"] = [round(float(np.percentile(d, 5)), 3), round(float(np.percentile(d, 95)), 3)]
    axs[0].set_ylabel("ブレーキの利き (50 °C 未満の中央値 = 1)\n= 減速度 ÷ (エア圧 − 押し出し圧)")
    axs[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("NCST 重量車 (台上試験): ブレーキ温度と利き。実測の温度は表面より低い可能性がある。フェードの確認ではない", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "gain_vs_temp.png", dpi=130)
    plt.close(fig)
    (out / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    (out / "meta.json").write_text(json.dumps(dict(run_id=a.run_id, events=a.events.as_posix(), bins=BINS), ensure_ascii=False, indent=1))
    print(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
