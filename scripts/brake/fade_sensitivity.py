"""模擬の制動性能の低下で、正常制動の模型を基準にした検出の感度 (誤報率と検出力) を評価する。

**実車でフェードを検出したという主張ではない。** 実データの制動 (学習に使っていない走行での予測つき) に、
ブレーキの効きだけを人工的に delta だけ下げた値を作り、見分けられる割合を数える。

低下の作り方 (1 回の制動の定常部):
    観測の減速度 d = 抵抗 r(v) + ブレーキの分
    低下後 d' = r(v) + (1 - delta) (d - r(v))         抵抗 (b0 + b2 v^2、模型 M4) は下がらない
    eta' = d' / 予測
検出の規則 (3 通り):
    single      1 回の eta' が閾値より小さい
    mean_n      同じ走行の続いた n 回の log eta' の平均が閾値より小さい (n 回とも同じ delta で下げる)
    self_ref    同じ走行のそれまでの制動 (3 回以上、低下なし) の eta の中央値で割った比が閾値より小さい
閾値は誤報率 alpha から決める。**閾値は評価する分割 (走行) とは別の分割の、低下なしの制動から決める** (入れ子)。
そのうえで、評価する分割の低下なしの制動で、実際の誤報率も数える。

    uv run python scripts/brake/fade_sensitivity.py --poc out/brake_performance/20261118_r8_rav4 --run-id <新しい ID>
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
DELTAS = [0.0, 0.05, 0.10, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40, 0.50]
ALPHAS = [0.05, 0.01]


def degrade(E: pd.DataFrame, delta: float, coef) -> np.ndarray:
    r = coef[0] + coef[1] * E.v_steady.to_numpy() ** 2
    d = E.decel_brake_obs.to_numpy()
    d2 = r + (1 - delta) * (d - r)
    return d2 / E.decel_brake_pred.to_numpy()


def stat_single(E, eta):
    return np.log(eta)


def windows(E: pd.DataFrame, n: int) -> list[np.ndarray]:
    """同じ走行の、時刻順に続いた n 回の制動の組 (重ならない)。"""
    out = []
    for _, g in E.sort_values("t0").groupby("drive"):
        idx = g.index.to_numpy()
        for s in range(0, len(idx) - n + 1, n):
            out.append(idx[s:s + n])
    return out


def run(E: pd.DataFrame, coef, rule: str, n: int, alpha: float) -> list[dict]:
    rows = []
    folds = sorted(E.fold.unique())
    base_eta = E.eta.to_numpy()
    for delta in DELTAS:
        eta_d = degrade(E, delta, coef)
        det_total, n_total, fa_total, n_fa = 0, 0, 0, 0
        for f in folds:
            ref = E.fold != f
            tst = E.fold == f
            if rule == "single":
                null = np.log(base_eta[ref.to_numpy()])
                thr = np.quantile(null, alpha)
                x_null = np.log(base_eta[tst.to_numpy()])
                x_deg = np.log(eta_d[tst.to_numpy()])
            elif rule == "mean_n":
                wr = windows(E[ref], n)
                wt = windows(E[tst], n)
                if len(wr) < 10 or not wt:
                    continue
                null = np.array([np.mean(np.log(base_eta[w])) for w in wr])
                thr = np.quantile(null, alpha)
                x_null = np.array([np.mean(np.log(base_eta[w])) for w in wt])
                x_deg = np.array([np.mean(np.log(eta_d[w])) for w in wt])
            elif rule == "self_ref":
                def sr(sub, eta_now):
                    xs = []
                    for _, g in sub.sort_values("t0").groupby("drive"):
                        idx = g.index.to_numpy()
                        for k in range(3, len(idx)):
                            xs.append(np.log(eta_now[idx[k]]) - np.log(np.median(base_eta[idx[:k]])))
                    return np.array(xs)
                null = sr(E[ref], base_eta)
                if len(null) < 10:
                    continue
                thr = np.quantile(null, alpha)
                x_null = sr(E[tst], base_eta)
                x_deg = sr(E[tst], eta_d)
            else:
                raise KeyError(rule)
            fa_total += int((x_null < thr).sum())
            n_fa += len(x_null)
            det_total += int((x_deg < thr).sum())
            n_total += len(x_deg)
        rows.append(dict(rule=rule, n=n, alpha=alpha, delta=delta, n_tests=n_total, detect_rate=det_total / n_total if n_total else np.nan,
                         realized_false_alarm=fa_total / n_fa if n_fa else np.nan))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poc", type=Path, action="append", required=True, help="brake_performance_poc の出力 (複数可)")
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    out = OUT / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    allrows = []
    S = {}
    for poc in a.poc:
        Sm = json.load(open(poc / "summary.json"))
        coef = next(x["coef_mean"] for x in Sm["q4_model"]["by_form"] if x["form"] == "M4")
        E = pd.read_csv(poc / "events.csv")
        E = E[E.eta.notna() & (E.eta > 0)].reset_index(drop=True)
        name = poc.name
        rows = []
        for alpha in ALPHAS:
            rows += run(E, coef, "single", 1, alpha)
            for n in (3, 5):
                rows += run(E, coef, "mean_n", n, alpha)
            rows += run(E, coef, "self_ref", 1, alpha)
        R = pd.DataFrame(rows)
        R.insert(0, "source", name)
        allrows.append(R)
        # 前回の「18%」の意味: 1 - p5/中央値。低下後の eta の中央値が閾値にちょうど届く低下
        eta = E.eta.to_numpy()
        S[name] = dict(n_events=int(len(E)), n_drives=int(E.drive.nunique()),
                       previous_metric_1_minus_p5_over_median=round(float(1 - np.percentile(eta, 5) / np.median(eta)), 3),
                       resistance_coef_M4=[float(coef[0]), float(coef[1])],
                       note="previous_metric は閾値 (誤報 5%) の位置を表すだけで、その低下での検出力は約 50%。検出力は detect_rate を見る")
    R = pd.concat(allrows, ignore_index=True)
    R.to_csv(out / "power_curves.csv", index=False)
    # 検出力 80%・95% に要る低下 (線形内挿)
    need = []
    for (src, rule, n, alpha), g in R.groupby(["source", "rule", "n", "alpha"]):
        g = g.sort_values("delta")
        for target in (0.5, 0.8, 0.95):
            x = np.interp(target, g.detect_rate.to_numpy(), g.delta.to_numpy(), left=np.nan, right=np.nan) if g.detect_rate.is_monotonic_increasing else \
                next((d for d, p in zip(g.delta, g.detect_rate) if p >= target), np.nan)
            need.append(dict(source=src, rule=rule, n=n, alpha=alpha, target_power=target, delta_needed=x,
                             realized_fa=float(g[g.delta == 0].realized_false_alarm.iloc[0]), n_tests=int(g.n_tests.iloc[0])))
    N = pd.DataFrame(need)
    N.to_csv(out / "delta_needed.csv", index=False)
    S["delta_needed"] = N.round(3).to_dict("records")
    S["power_at_previous_18pct"] = R[(R.rule == "single") & (R.delta == 0.175)][["source", "alpha", "detect_rate", "realized_false_alarm"]].round(3).to_dict("records")
    (out / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    (out / "meta.json").write_text(json.dumps(dict(run_id=a.run_id, poc=[str(p) for p in a.poc], deltas=DELTAS, alphas=ALPHAS), ensure_ascii=False, indent=1))
    configure_japanese_font()
    srcs = R.source.unique()
    fig, axs = plt.subplots(1, len(srcs), figsize=(6 * len(srcs), 4.2), squeeze=False)
    styles = {("single", 1): ("#c0392b", "-"), ("mean_n", 3): ("#2e86c1", "-"), ("mean_n", 5): ("#1abc9c", "-"), ("self_ref", 1): ("#8e44ad", "--")}
    for ax, src in zip(axs[0], srcs):
        for (rule, n), (c, ls) in styles.items():
            g = R[(R.source == src) & (R.rule == rule) & (R.n == n) & (R.alpha == 0.05)]
            lab = {"single": "1 回", "mean_n": f"同じ走行の {n} 回の平均", "self_ref": "同じ走行のそれまでとの比"}[rule]
            ax.plot(g.delta * 100, g.detect_rate * 100, color=c, ls=ls, marker="o", ms=3, label=f"{lab} (誤報 {g.realized_false_alarm.iloc[0]*100:.1f}%)")
        ax.axhline(5, color="#888", lw=0.8)
        ax.set_xlabel("模擬の効きの低下 [%]")
        ax.set_ylabel("検出された割合 [%]")
        ax.set_title(f"{src} (閾値は誤報 5%、別の走行で決定)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "power_curves.png", dpi=120)
    plt.close(fig)
    print(json.dumps({k: v for k, v in S.items() if k != "delta_needed"}, ensure_ascii=False, indent=1, default=float))
    print(N[N.alpha == 0.05].pivot_table(index=["source", "rule", "n"], columns="target_power", values="delta_needed").round(3).to_string())
    print(N[N.alpha == 0.05].groupby(["source", "rule", "n"])[["realized_fa", "n_tests"]].first().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
