"""フェードリスク指標 (0〜100 と High / Medium / Low) を、NCST・Udacity・comma2k19 で同じ形に計算して並べる。

成分 (それぞれ 0〜1、設定 configs/brake/fade_risk_index.yaml):
  thermal      NCST: 実測のブレーキ温度。Udacity・comma2k19: 直前 300 s のブレーキの仕事 E300 (温度の代わり)
  continuous   直前 300 s のうちブレーキを踏んでいた時間の割合
  downhill     直前 300 s の平均の下り勾配 (NCST は地図からの推定。推定の範囲外は欠損)
  degradation  直前 300 s の制動の利き eta (平常のモデル比) の中央値の低下。NCST の路上は測れない (欠損)
スコア = 100 × (使える成分の重みつき平均)。フェードの確率ではない。

comma2k19 は 1 回ごとの制動 (events.csv) で計算する (時系列を持っていないため)。continuous は、同じ走行の
直前 300 s の制動イベントの長さの和 (イベントにならない弱い踏み込みは入らないので、少なめに出る)。

    uv run python scripts/brake/fade_risk_index.py --run-id <新しい ID>
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

from near_miss.plotting import configure_japanese_font

OUT = Path("out/brake_fade")
CFG = Path("configs/brake/fade_risk_index.yaml")
COMP = ["thermal", "continuous", "downhill", "degradation"]
COMP_JA = {"thermal": "熱の状態", "continuous": "連続制動", "downhill": "下り坂", "degradation": "効率低下の兆候"}
COMP_COL = {"thermal": "#C0392B", "continuous": "#E08A2E", "downhill": "#2F6FB0", "degradation": "#8E44AD"}
RANK_COL = {"High": "#C0392B", "Medium": "#E08A2E", "Low": "#8A9BB0"}


def scale(x, lo, hi):
    return np.clip((np.asarray(x, float) - lo) / (hi - lo), 0, 1)


def score(D: pd.DataFrame, cfg: dict, avail: list[str]) -> pd.DataFrame:
    C = cfg["components"]
    D = D.copy()
    if "degradation" in avail:  # 熱が入っていない時刻の利きのばらつきは数えない
        cold = (D.thermal < C["degradation"]["thermal_gate"]) & D.degradation.notna()
        D.loc[cold, "degradation"] = 0.0
    w = np.array([C[c]["weight"] if c in avail else 0.0 for c in COMP])
    M = np.column_stack([D[c].to_numpy(float) for c in COMP])
    complete = np.all(np.isfinite(M[:, w > 0]), axis=1)
    den = w.sum()
    D["score"] = np.where(complete, 100 * np.nansum(M * w, axis=1) / den, np.nan)
    for c in COMP:  # スコアへの寄与 (点)
        D[f"pts_{c}"] = np.where(complete & np.isfinite(D[c]), 100 * D[c] * (C[c]["weight"] if c in avail else 0) / den, 0.0)
    R = cfg["rank"]
    D["rank"] = np.where(D.score >= R["high"], "High", np.where(D.score >= R["medium"], "Medium", "Low"))
    return D


def trailing_mean(t, x, win):
    """時刻 t (s、昇順) で、直前 win 秒の x の平均 (x の欠損は除く)。"""
    s = pd.Series(x, index=pd.to_datetime(t, unit="s"))
    return s.rolling(f"{int(win)}s", min_periods=1).mean().to_numpy()


def ncst(cfg) -> pd.DataFrame:
    C, I = cfg["components"], cfg["inputs"]
    G = pd.read_csv(I["ncst_timeline"])
    t = G["min"].to_numpy() * 60
    on = (G.psi > 5).astype(float).to_numpy()
    D = pd.DataFrame(dict(dataset="NCST 高速道路", scene_key="FWY", t=t, minute=G["min"], T=G["T"], psi=G.psi, grade_pct=G.grade * 100, v_kmh=G.v))
    D["thermal"] = scale(G["T"], C["thermal"]["temperature_C"]["lo"], C["thermal"]["temperature_C"]["hi"])
    D["continuous"] = scale(trailing_mean(t, on, C["continuous"]["window_s"]), C["continuous"]["lo"], C["continuous"]["hi"])
    g = trailing_mean(t, -G.grade.to_numpy() * 100, C["downhill"]["window_s"])
    g[~np.isfinite(G.grade.to_numpy())] = np.nan   # 地図の推定の範囲外 (GPS が切れる前の分は地図と対応させていない)
    D["downhill"] = scale(g, C["downhill"]["lo_pct"], C["downhill"]["hi_pct"])
    D["degradation"] = np.nan                        # 路上の利きは測れない
    D["braking"] = on > 0
    w0, w1 = I["ncst_window_min"]
    return score(D[(D.minute >= w0) & (D.minute <= w1)].reset_index(drop=True), cfg, cfg["available"]["ncst"])


def eta_trailing(ev_t, ev_eta, t, win):
    out = np.full(len(t), np.nan)
    for k, tk in enumerate(t):
        m = (ev_t <= tk) & (ev_t > tk - win)
        if m.any():
            out[k] = np.median(ev_eta[m])
    return out


def udacity(cfg) -> pd.DataFrame:
    C, I = cfg["components"], cfg["inputs"]
    U = pd.read_csv(I["udacity_timeseries"])
    E = pd.read_csv(I["udacity_events"])
    base = E.loc[E.H300_start < I["udacity_cold_H300_Jkg"], "eta"].median()
    t = U.t.to_numpy()
    on = (U.Tb >= I["udacity_brake_on_Nm"]).astype(float).to_numpy()
    D = pd.DataFrame(dict(dataset="Udacity HMB_2", scene_key="HMB_2", t=t, minute=t / 60, v_kmh=U.v * 3.6, Tb=U.Tb, E300=U.E300, grade_pct=np.degrees(U.theta) * 0 + np.tan(U.theta) * 100))
    D["thermal"] = scale(U.E300, C["thermal"]["brake_work_E300_Jkg"]["lo"], C["thermal"]["brake_work_E300_Jkg"]["hi"])
    D["continuous"] = scale(trailing_mean(t, on, C["continuous"]["window_s"]), C["continuous"]["lo"], C["continuous"]["hi"])
    D["downhill"] = scale(trailing_mean(t, -np.tan(U.theta.to_numpy()) * 100, C["downhill"]["window_s"]), C["downhill"]["lo_pct"], C["downhill"]["hi_pct"])
    rel = eta_trailing((E.t0 + E.dur).to_numpy(), (E.eta / base).to_numpy(), t, C["degradation"]["window_s"])
    D["eta_rel"] = rel
    D["braking"] = on > 0
    D["degradation"] = np.nan_to_num(scale(1 - rel, C["degradation"]["lo_drop"], C["degradation"]["hi_drop"]), nan=0.0)  # 直前に評価できる制動が無ければ兆候なし
    return score(D, cfg, cfg["available"]["udacity"]), float(base)


def comma(cfg) -> pd.DataFrame:
    C, I = cfg["components"], cfg["inputs"]
    E = pd.read_csv(I["comma2k19_events"]).sort_values(["drive", "t0"]).reset_index(drop=True)
    rows = []
    for drv, g in E.groupby("drive"):
        te = (g.t0 + g.dur_s).to_numpy()
        for k, (_, r) in enumerate(g.iterrows()):
            m = (te <= te[k]) & (te > te[k] - C["continuous"]["window_s"])
            md = m & (te > te[k] - C["degradation"]["window_s"])
            rows.append(dict(dataset="comma2k19 RAV4", scene_key=drv, t=te[k], minute=te[k] / 60, v_kmh=r.v0 * 3.6, E300=r.E300_before,
                             grade_pct=np.tan(np.radians(r.grade_deg)) * 100, duty=g.dur_s.to_numpy()[m].sum() / C["continuous"]["window_s"],
                             eta_rel=float(np.median(g.eta.to_numpy()[md])), auto=bool(r.auto_overlap)))
    D = pd.DataFrame(rows)
    D["thermal"] = scale(D.E300, C["thermal"]["brake_work_E300_Jkg"]["lo"], C["thermal"]["brake_work_E300_Jkg"]["hi"])
    D["continuous"] = scale(D.duty, C["continuous"]["lo"], C["continuous"]["hi"])
    D["downhill"] = scale(-D.grade_pct, C["downhill"]["lo_pct"], C["downhill"]["hi_pct"])   # 制動中の勾配 (直前 300 s の平均ではない)
    D["degradation"] = scale(1 - D.eta_rel, C["degradation"]["lo_drop"], C["degradation"]["hi_drop"])
    D["braking"] = True
    return score(D, cfg, cfg["available"]["comma2k19"])


def top_scenes(D: pd.DataFrame, n: int, sep: float) -> pd.DataFrame:
    rows = []
    cand = D[D.braking].dropna(subset=["score"]).sort_values("score", ascending=False)   # 場面はブレーキを踏んでいる時刻から選ぶ
    for _, r in cand.iterrows():
        if any((x.scene_key == r.scene_key) and abs(x.t - r.t) < sep for x in rows):
            continue
        rows.append(r)
        if len(rows) >= n:
            break
    return pd.DataFrame(rows)


def fig_timeseries(N, U, cfg, path):
    configure_japanese_font()
    R = cfg["rank"]
    fig, axs = plt.subplots(2, 1, figsize=(9.0, 5.6), gridspec_kw=dict(hspace=0.55))
    for ax, D, x, xl, title in [(axs[0], N, N.minute, "走り始めからの時間 [分]", "NCST 大型トラック 高速道路 (実測温度あり)"),
                                (axs[1], U, U.t, "時刻 [s]", "Udacity HMB_2 乗用車 峠の下り (温度なし、ブレーキの仕事で代替)")]:
        ax.axhspan(R["high"], 100, color="#FADBD8", lw=0)
        ax.axhspan(R["medium"], R["high"], color="#FDEBD0", lw=0)
        bottom = np.zeros(len(D))
        for c in COMP:
            ax.fill_between(x, bottom, bottom + D[f"pts_{c}"], color=COMP_COL[c], alpha=0.75, lw=0, label=COMP_JA[c])
            bottom = bottom + D[f"pts_{c}"].to_numpy()
        ax.plot(x, D.score, color="#1F2A44", lw=1.2, label="スコア")
        ax.set_ylim(0, 100)
        ax.set_ylabel("リスクスコア", fontsize=8.5)
        ax.set_xlabel(xl, fontsize=8)
        ax.set_title(title, fontsize=9.5, loc="left", color="#1F2A44")
        ax.tick_params(labelsize=7.5)
        ax.text(x.iloc[0], R["high"] + 2, " High", fontsize=7.5, color="#C0392B")
        ax.text(x.iloc[0], R["medium"] + 2, " Medium", fontsize=7.5, color="#E08A2E")
        k = D.score[D.braking].idxmax()
        ax.set_xlim(x.min(), x.max())
        ax.annotate(f"最大 {D.score[k]:.0f} ({D['rank'][k]})", (x[k], D.score[k]), xytext=(10, 6), textcoords="offset points", fontsize=8, color="#1F2A44")
    axs[0].legend(fontsize=7, loc="upper left", ncol=5, bbox_to_anchor=(0, 1.32))
    axs[0].text(0.01, 0.55, "勾配 (地図からの推定) が無い時刻は\nスコアを出していない (GPS が切れる前・158 分以降)", transform=axs[0].transAxes, fontsize=7, color="#555")
    axs[0].text(0.99, 0.03, "効率低下の兆候: 路上では測れない (成分から除外)", transform=axs[0].transAxes, fontsize=7, color="#555", ha="right")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_summary(T, cfg, path):
    """データセットごとの上位の場面を、成分の寄与の積み上げで並べる。"""
    configure_japanese_font()
    R = cfg["rank"]
    T = T.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    y = np.arange(len(T))[::-1]
    for c in COMP:
        left = sum(T[f"pts_{k}"] for k in COMP[: COMP.index(c)])
        ax.barh(y, T[f"pts_{c}"], left=left, color=COMP_COL[c], height=0.62, label=COMP_JA[c], edgecolor="white")
    for yi, (_, r) in zip(y, T.iterrows()):
        ax.text(r.score + 1, yi, f"{r.score:.0f}  {r['rank']}", va="center", fontsize=8.5, color=RANK_COL[r["rank"]], fontweight="bold")
    ax.axvline(R["medium"], color="#E08A2E", lw=0.8, ls="--")
    ax.axvline(R["high"], color="#C0392B", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(T.label, fontsize=7.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("フェードリスクスコア (0〜100)。棒の内訳 = 成分ごとの寄与", fontsize=8.5)
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.01, 0.01, "熱の状態: NCST は実測のブレーキ温度、Udacity・comma2k19 は直前 300 s のブレーキの仕事 (FMVSS 121 のフェード試験のペースを 1)。"
             "\ncomma2k19 の下り坂は制動中の勾配 (直前 300 s の平均ではない)。データセット間の比較は目安", fontsize=7, color="#555")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def label(r) -> str:
    if r.dataset.startswith("NCST"):
        return f"NCST 高速道路 {r.minute:.1f} 分 (温度 {r['T']:.0f} ℃)"
    if r.dataset.startswith("Udacity"):
        return f"Udacity HMB_2 {r.t:.0f} s (E300 {r.E300:.0f} J/kg)"
    return f"comma2k19 {r.scene_key.split('|')[1][:10]} (E300 {r.E300:.0f} J/kg)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text())
    out = OUT / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    N = ncst(cfg)
    U, base = udacity(cfg)
    Cm = comma(cfg)
    N.round(4).to_csv(out / "ncst_fwy_score.csv", index=False)
    U.round(4).to_csv(out / "udacity_hmb2_score.csv", index=False)
    Cm.round(4).to_csv(out / "comma2k19_rav4_event_score.csv", index=False)
    TS = cfg["top_scenes"]
    tops = []
    for D in (N, U, Cm):
        t = top_scenes(D, TS["per_dataset"], TS["min_separation_s"])
        tops.append(pd.concat([t.iloc[:1], t.iloc[1:][t.score.iloc[1:] >= cfg["rank"]["medium"]]]))   # 2 件目以降は Medium 以上だけ
    T = pd.concat(tops, ignore_index=True)
    T["label"] = [label(r) for _, r in T.iterrows()]
    keep = ["dataset", "scene_key", "t", "minute", "label", "score", "rank"] + COMP + [f"pts_{c}" for c in COMP] + \
           [c for c in ("T", "psi", "E300", "grade_pct", "eta_rel", "v_kmh") if c in T]
    T[keep].round(3).to_csv(out / "top_scenes.csv", index=False)
    fig_timeseries(N, U, cfg, out / "fig_risk_timeseries.png")
    fig_summary(T, cfg, out / "fig_risk_top_scenes.png")
    S = {}
    for name, D in [("ncst_fwy", N), ("udacity_hmb2", U), ("comma2k19_rav4_events", Cm)]:
        S[name] = dict(n=int(len(D)), score_max=round(float(D.score.max()), 1), score_p90=round(float(D.score.quantile(0.9)), 1),
                       rank_counts=D["rank"].value_counts().to_dict(),
                       frac_time_high=round(float((D["rank"] == "High").mean()), 3), frac_time_medium=round(float((D["rank"] == "Medium").mean()), 3),
                       component_max={c: round(float(D[c].max()), 3) if D[c].notna().any() else None for c in COMP})
    S["udacity_eta_baseline"] = base
    S["ncst_high_window_min"] = [round(float(N.minute[N["rank"] == "High"].min()), 1), round(float(N.minute[N["rank"] == "High"].max()), 1)] if (N["rank"] == "High").any() else None
    S["top_scenes"] = T[["label", "score", "rank"]].round(1).to_dict("records")
    (out / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    (out / "meta.json").write_text(json.dumps(dict(run_id=a.run_id, config=CFG.as_posix(), cfg=cfg), ensure_ascii=False, indent=1))
    print(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
