"""NCST のフェードの評価を、温度だけでなく制動性能の低下の可能性と安全上の影響が分かる形にまとめる。

1. 台上試験の高温 (130 ℃ 以上) の利きの差を、車速・圧力・試験条件を考えて分解する
   - 調整前の差 (50 ℃ 未満を基準)、信頼できない制動を除いた差、惰行の力を引かない定義の差
   - 回帰 log(利き) ~ 高温 + 50〜130 ℃ + 車速 + 圧力 (信頼できる制動)。高温の係数と、その 90% 区間 (bootstrap)
   - UDDS の中だけの温度の傾き (車速・圧力を調整)。高温 = UDDS なので、UDDS の中で温度が効いているかを見る
2. 停止距離の試算 (仮定のシナリオ): 利きの低下 delta がそのまま減速度の低下になったとみなす
   停止距離 = v (t_反応 + t_作動) + v^2 / (2 a0 (1 - delta))
3. 路上 (高速道路の 2 本の下り) の 1 回ごとの制動: 踏み始めの温度と、圧力の峰・長さ・車速の記録の有無
4. 評価できる指標と、フェードの確率の推定に足りないデータの表

    uv run python scripts/brake/ncst_fade_safety.py --run-id <新しい ID>
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
import yaml
from scipy.stats import spearmanr

from near_miss.plotting import configure_japanese_font

OUT = Path("out/brake_fade")
CFG = Path("configs/brake/ncst_fade_safety.yaml")
NAVY, BLUE, SKY, PALE, STEEL, RED, ORANGE = "#1F2A44", "#2F6FB0", "#3E9BD6", "#EAF2FB", "#8A9BB0", "#C0392B", "#E08A2E"


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def dyno_effects(E: pd.DataFrame, C: dict) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(C["seed"])
    nb = C["bootstrap"]
    E = E.copy()
    E["f0frac"] = E.F0 / E.F
    E["reliable"] = (E.f0frac < C["max_f0_frac"]) & (E.psi >= C["min_psi"])
    hot = E["T"] >= C["hot_C"]
    cold = E["T"] < C["cold_C"]
    E["eff_noF0"] = E.F / (E.psi - C["pushout_psi"])

    def ratio_ci(col, mask):
        h, c = E.loc[hot & mask, col].to_numpy(), E.loc[cold & mask, col].to_numpy()
        est = np.median(h) / np.median(c) - 1
        bs = [np.median(rng.choice(h, len(h))) / np.median(rng.choice(c, len(c))) - 1 for _ in range(nb)]
        return est, np.percentile(bs, [5, 95]), int(len(h)), int(len(c))

    rows = []
    allm = np.ones(len(E), bool)
    for key, lab, col, m in [("raw", "調整前 (全 89 回)", "eff", allm),
                             ("reliable", "信頼できない 3 回を除く", "eff", E.reliable.to_numpy()),
                             ("noF0", "惰行の力を引かない定義", "eff_noF0", E.reliable.to_numpy())]:
        est, ci, nh, nc = ratio_ci(col, m)
        rows.append(dict(key=key, label=lab, effect=est, ci5=ci[0], ci95=ci[1], n_hot=nh, n_ref=nc, basis="中央値の比 (50 ℃ 未満が基準)"))
    R = E[E.reliable].reset_index(drop=True)
    X = np.c_[np.ones(len(R)), (R["T"] >= C["hot_C"]), (R["T"] >= C["cold_C"]) & (R["T"] < C["hot_C"]), R.v_steady, R.psi].astype(float)
    y = np.log(R.eff.to_numpy())
    b = ols(X, y)
    bs = []
    for _ in range(nb):
        i = rng.integers(0, len(R), len(R))
        if X[i, 1].sum() < 3 or X[i, 2].sum() < 3:
            continue
        bs.append(ols(X[i], y[i]))
    bs = np.array(bs)
    rows.append(dict(key="adjusted", label="車速・圧力で調整 (回帰)", effect=np.exp(b[1]) - 1, ci5=np.exp(np.percentile(bs[:, 1], 5)) - 1,
                     ci95=np.exp(np.percentile(bs[:, 1], 95)) - 1, n_hot=int(X[:, 1].sum()), n_ref=int((R["T"] < C["cold_C"]).sum()),
                     basis="log(利き) の回帰の係数。基準は 50 ℃ 未満"))
    d = b[1] - b[2]
    rows.append(dict(key="adjusted_vs_warm", label="同上、50〜130 ℃ を基準", effect=np.exp(d) - 1, ci5=np.exp(np.percentile(bs[:, 1] - bs[:, 2], 5)) - 1,
                     ci95=np.exp(np.percentile(bs[:, 1] - bs[:, 2], 95)) - 1, n_hot=int(X[:, 1].sum()), n_ref=int(X[:, 2].sum()),
                     basis="高温と 50〜130 ℃ の係数の差"))
    U = R[R.traj == "UDDS"].reset_index(drop=True)
    XU = np.c_[np.ones(len(U)), U["T"], U.v_steady, U.psi]
    yU = np.log(U.eff.to_numpy())
    bu = ols(XU, yU)
    bsu = np.array([ols(XU[i], yU[i]) for i in (rng.integers(0, len(U), len(U)) for _ in range(nb))])
    k = C["per_C"]
    rows.append(dict(key="within_udds_T", label=f"UDDS の中の温度 (+{k} ℃ あたり)", effect=np.exp(bu[1] * k) - 1,
                     ci5=np.exp(np.percentile(bsu[:, 1], 5) * k) - 1, ci95=np.exp(np.percentile(bsu[:, 1], 95) * k) - 1,
                     n_hot=int(len(U)), n_ref=0, basis=f"UDDS ({U['T'].min():.0f}〜{U['T'].max():.0f} ℃) の回帰で、車速・圧力を調整"))
    extra = dict(
        excluded=E.loc[~E.reliable, ["traj", "T", "psi", "f0frac", "rel"]].round(3).to_dict("records"),
        regression=dict(coef=dict(zip(["const", "hot", "warm_50_130", "v_steady_mps", "psi"], np.round(b, 4).tolist())),
                        speed_effect_per_mps=float(np.exp(b[3]) - 1), n=int(len(R))),
        within_udds=dict(coef=dict(zip(["const", "T", "v_steady_mps", "psi"], np.round(bu, 4).tolist())), n=int(len(U)),
                         spearman_eff_T=float(spearmanr(U["T"], U.eff)[0]), spearman_eff_v=float(spearmanr(U.v_steady, U.eff)[0]),
                         spearman_T_v=float(spearmanr(U["T"], U.v_steady)[0])),
        v_steady_median=dict(cold=float(E.loc[cold, "v_steady"].median()), hot=float(E.loc[hot, "v_steady"].median())),
        psi_median=dict(cold=float(E.loc[cold, "psi"].median()), hot=float(E.loc[hot, "psi"].median())),
        hot_all_udds=bool((E.loc[hot, "traj"] == "UDDS").all()))
    return rows, extra


def stopping(S: dict) -> pd.DataFrame:
    rows = []
    for vk in S["speeds_kmh"]:
        v = vk / 3.6
        for a0 in S["base_decel_mps2"]:
            for dlt in [0.0] + list(S["efficiency_drops"]):
                pre = v * (S["reaction_s"] + S["actuation_s"])
                brk = v ** 2 / (2 * a0 * (1 - dlt))
                rows.append(dict(v_kmh=vk, a0=a0, delta=dlt, pre_m=pre, brake_m=brk, total_m=pre + brk))
    T = pd.DataFrame(rows)
    base = T[T.delta == 0].set_index(["v_kmh", "a0"]).total_m
    T["increase_m"] = T.total_m - T.set_index(["v_kmh", "a0"]).index.map(base)
    T["increase_pct_total"] = T.increase_m / T.set_index(["v_kmh", "a0"]).index.map(base) * 100
    T["increase_pct_brake"] = (1 / (1 - T.delta) - 1) * 100
    return T


def road_events(G: pd.DataFrame, C: dict) -> pd.DataFrame:
    on = (G.psi > C["brake_on_psi"]).to_numpy()
    idx = np.where(np.diff(np.r_[0, on.astype(int), 0]))[0].reshape(-1, 2)
    rows = []
    for i0, i1 in idx:
        g = G.iloc[i0:i1]
        t = g["min"].iloc[0]
        if not (C["window_min"][0] <= t <= C["window_min"][1]) or len(g) < 2:
            continue
        rows.append(dict(t_min=t, dur_s=len(g), psi_peak=g.psi.max(), psi_mean=g.psi.mean(),
                         T_start=G["T"].iloc[max(0, i0 - C["T_pre_s"]):i0 + 1].mean(), phase=g.phase.iloc[0],
                         grade_pct=g.grade.mean() * 100, speed_samples=int(g.v.notna().sum()),
                         descent=1 if t < 150 else 2))
    return pd.DataFrame(rows)


def fig_effects(rows, path):
    configure_japanese_font()
    D = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    y = np.arange(len(D))[::-1]
    col = {"raw": STEEL, "reliable": STEEL, "noF0": STEEL, "adjusted": BLUE, "adjusted_vs_warm": BLUE, "within_udds_T": NAVY}
    for yi, (_, r) in zip(y, D.iterrows()):
        c = col[r.key]
        ax.errorbar(r.effect * 100, yi, xerr=[[(r.effect - r.ci5) * 100], [(r.ci95 - r.effect) * 100]], fmt="o", color=c, ms=7, capsize=4, lw=1.8)
        ax.text(max(r.ci95 * 100, r.effect * 100) + 1.5, yi, f"{r.effect * 100:+.0f}%  [{r.ci5 * 100:+.0f}, {r.ci95 * 100:+.0f}]", va="center", fontsize=8.5, color=c)
    ax.set_yticks(y)
    ax.set_yticklabels(D.label, fontsize=9)
    ax.axvline(0, color="#555", lw=0.8)
    ax.axvspan(-100, 0, color="#FBE5D6", alpha=0.35, lw=0)
    ax.set_xlim(-45, 30)
    ax.set_xlabel("高温 (130 ℃ 以上) の利きの差 [%] (点 = 推定、横線 = 90% 区間)。左ほど効きが弱い", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.axhline(y[2] - 0.5, color=STEEL, lw=0.6, ls=":")
    ax.set_ylim(y[-1] - 0.6, y[0] + 0.9)
    ax.text(-44, y[0] + 0.45, "定義を変えた差 (車速は未調整)", fontsize=8, color=STEEL, va="bottom")
    ax.text(-44, y[3] + 0.45, "車速・圧力を考えた差", fontsize=8, color=BLUE, va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_stopping(T, S, path):
    configure_japanese_font()
    a0 = S["base_decel_mps2"][len(S["base_decel_mps2"]) // 2]
    D = T[T.a0 == a0]
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    speeds = S["speeds_kmh"]
    deltas = [0.0] + list(S["efficiency_drops"])
    w = 0.26
    cols = [STEEL, ORANGE, RED]
    for j, dlt in enumerate(deltas):
        for i, vk in enumerate(speeds):
            r = D[(D.v_kmh == vk) & (D.delta == dlt)].iloc[0]
            x = i + (j - 1) * w
            ax.bar(x, r.pre_m, w * 0.92, color="#D5DBDB", edgecolor="white")
            ax.bar(x, r.brake_m, w * 0.92, bottom=r.pre_m, color=cols[j], edgecolor="white",
                   label=(f"利き {'正常' if dlt == 0 else f'-{dlt * 100:.0f}%'}" if i == 0 else None))
            txt = f"{r.total_m:.0f} m" if dlt == 0 else f"{r.total_m:.0f} m\n(+{r.increase_m:.0f})"
            ax.text(x, r.total_m + 2, txt, ha="center", va="bottom", fontsize=7.5)
    ax.bar([np.nan], [np.nan], color="#D5DBDB", label=f"反応 {S['reaction_s']} s + 作動 {S['actuation_s']} s の間の空走")
    ax.set_xticks(range(len(speeds)))
    ax.set_xticklabels([f"{v} km/h" for v in speeds])
    ax.set_ylabel("停止距離 [m]")
    ax.set_ylim(0, D.total_m.max() * 1.22)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"仮定のシナリオ: 正常時の平均減速度 {a0} m/s²、利きの低下がそのまま減速度の低下になるとみなした場合 (NCST の実測ではない)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_road(Rd, dyno_hot_T, path, rs):
    configure_japanese_font()
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    ax.axvspan(*dyno_hot_T, color="#FBE5D6", alpha=0.6, lw=0)
    ax.text(np.mean(dyno_hot_T), 24.2, "台上で効きが低く出た温度域\n(車速の影響と未分離)", ha="center", fontsize=8, color=RED, va="top")
    for dsc, c, lab in [(1, BLUE, "下り 1 (140.8〜146.5 分)"), (2, NAVY, "下り 2 (151.5〜158 分)")]:
        d = Rd[Rd.descent == dsc]
        ax.scatter(d.T_start, d.psi_peak, s=12 + d.dur_s * 6, color=c, alpha=0.8, edgecolor="white", label=lab)
    sp = Rd[Rd.speed_samples >= 2]
    ax.scatter(sp.T_start, sp.psi_peak, s=120, facecolor="none", edgecolor=ORANGE, lw=1.5, label=f"車速の記録が 2 点以上ある制動 ({len(sp)} 回)")
    ax.set_xlabel("踏み始めのブレーキ温度 [℃] (実測)")
    ax.set_ylabel("ブレーキ圧の峰 [psi]")
    ax.set_ylim(0, 25)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.text(0.99, 0.03, f"点の大きさ = 踏んでいた長さ。順位相関 (温度と圧力の峰) {rs:+.2f}\n減速度は車速の記録がほぼ無く計算できない",
            transform=ax.transAxes, ha="right", fontsize=8, color="#444")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evidence_table() -> pd.DataFrame:
    rows = [
        ("評価できる", "ブレーキ温度の推移 (路上・台上)", "NCST の熱電対 (ライニングの表面から 2 mm)", "表面より約 100 ℃ 低い可能性 (報告書)。1 輪だけ"),
        ("評価できる", "温度と利きの関係 (台上)", "動力計の力 ÷ 圧力", "高温 = UDDS (速い) で車速と分離できない。130 ℃ 以上は 16 回"),
        ("評価できる", "熱負荷 (仕事の和・反復制動の回数)", "圧力・車速・勾配 (地図)", "路上は車速の記録が約 5%"),
        ("評価できる", "利きが下がった場合の停止距離", "仮定のシナリオ", "実測ではない。減速度の仮定に依存"),
        ("足りない", "路上の制動応答 (減速度 ÷ 圧力)", "連続した車速か前後加速度", "下りの制動 39 回のうち車速が取れるのは 4 回"),
        ("足りない", "フェードの温度域 (この車のライニング)", "同じライニングの高温の台上試験 (摩擦係数と温度)", "公開の規格は試験の手順だけで、温度の閾値を定めていない"),
        ("足りない", "フェードの発生確率", "多数の車両・経路の下り坂の温度と制動の記録", "NCST は 1 台・1 往復。発生の事例が 0 件"),
        ("足りない", "全輪の温度と荷重の分担", "全輪の熱電対、軸重", "測ったのは後輪 1 輪"),
    ]
    return pd.DataFrame(rows, columns=["区分", "指標", "必要なもの・使ったもの", "注意・不足"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text())
    out = OUT / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    E = pd.read_csv(cfg["inputs"]["dyno_events"])
    rows, extra = dyno_effects(E, cfg["dyno"])
    pd.DataFrame(rows).to_csv(out / "dyno_effects.csv", index=False)
    T = stopping(cfg["stopping"])
    T.round(2).to_csv(out / "stopping_scenario.csv", index=False)
    G = pd.read_csv(cfg["inputs"]["road_timeline"])
    Rd = road_events(G, cfg["road"])
    Rd.round(3).to_csv(out / "road_brake_events.csv", index=False)
    rs = float(spearmanr(Rd.T_start, Rd.psi_peak)[0])
    hot = E[E["T"] >= cfg["dyno"]["hot_C"]]
    ev = evidence_table()
    ev.to_csv(out / "evidence_table.csv", index=False)
    fig_effects(rows, out / "fig_dyno_effect_decomposition.png")
    fig_stopping(T, cfg["stopping"], out / "fig_stopping_scenario.png")
    fig_road(Rd, (float(hot["T"].min()), float(hot["T"].max())), out / "fig_road_pressure_vs_temp.png", rs)
    S = dict(dyno_effects={r["key"]: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows}, dyno_extra=extra,
             road=dict(n_events=int(len(Rd)), n_with_speed_ge2=int((Rd.speed_samples >= 2).sum()), spearman_psi_peak_vs_T=round(rs, 3),
                       spearman_dur_vs_T=round(float(spearmanr(Rd.T_start, Rd.dur_s)[0]), 3),
                       T_start_range=[round(float(Rd.T_start.min()), 1), round(float(Rd.T_start.max()), 1)],
                       psi_peak_range=[round(float(Rd.psi_peak.min()), 1), round(float(Rd.psi_peak.max()), 1)],
                       n_events_T_ge_hot=int((Rd.T_start >= cfg["dyno"]["hot_C"]).sum())),
             stopping_mid=T[T.a0 == cfg["stopping"]["base_decel_mps2"][1]].round(1).to_dict("records"))
    (out / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    (out / "meta.json").write_text(json.dumps(dict(run_id=a.run_id, config=CFG.as_posix(), cfg=cfg), ensure_ascii=False, indent=1))
    print(pd.DataFrame(rows).round(3).to_string())
    print(json.dumps({k: S[k] for k in ("road",)}, ensure_ascii=False, indent=1))
    print(T[T.a0 == 4.0].round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
