"""NCST の台上試験で、ブレーキ温度と利きの関係を、動力計で測ったタイヤの力から描き直す。

前の版 (ncst_gain_vs_temp.py) の利き = 減速度 ÷ 圧力 は、ブレーキ以外の減速 (動力計の走行抵抗・エンジンの引きずり、
圧力 0 で 0.25〜0.54 m/s^2) を含み、弱く踏んだ制動ほど利きが高く出た。この版では:

  タイヤの力 F = -Total Force [lbf] (動力計のローラーとタイヤの間の力。ブレーキ中は正)
     動力計が模擬する走行抵抗 (PAU) と慣性はローラー側の力なので、F には入らない。入るのは駆動軸のブレーキとエンジンの引きずり
  ブレーキ以外の力 F0 = その制動の直前 10 s の惰行 (圧力 < 0.5 psi、F > 0) の F の中央値
     直前に惰行が無ければ、同じ走行の惰行の F を車速で直線に当てはめた値
  利き = (F の定常部の平均 - F0) ÷ (圧力の定常部の平均 - 押し出し圧)   [lbf/psi]
     押し出し圧はデータから決められない (圧力の範囲が 16〜26 psi と狭く、x 切片が -32 psi になった)。
     標準は熱の模型の当てはめの値 1.2 psi (ncst_thermal.py、台上で当てはめ)。0・5 psi と、F0 を引かない場合も並べて、結論が変わるかを見る
  圧力を測っている右後輪は、ローラーに載る駆動軸。温度も後輪のライニング

    uv run python scripts/brake/ncst_force_vs_temp.py --thermal out/brake_fade/20261118_r3_ncst_thermal --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from near_miss.plotting import configure_japanese_font

import ncst_thermal as NT

OUT = Path("out/brake_fade")
P_TH_STD = 1.2   # 押し出し圧 [psi]。ncst_thermal.py の台上での当てはめ (1.19)
BINS = [0, 50, 90, 130, 150, 170]
COLORS = {"CBD0": "#7f8c8d", "CBD1": "#5dade2", "CBDx3": "#2e86c1", "UDDS": "#c0392b"}
LABELS = {"CBD0": "CBD の暖機 1 (台上)", "CBD1": "CBD の暖機 2 (台上)", "CBDx3": "CBD × 3 (台上)", "UDDS": "UDDS × 3 (台上)"}


def read_dyno_force(name: str) -> pd.DataFrame:
    L = open(NT.DATA / name, encoding="latin1").read().splitlines()
    t0 = dt.datetime.strptime(next(x for x in L if x.startswith("Date/Time Tested")).split(": ", 1)[1].strip(), "%Y-%m-%d %H:%M:%S").timestamp()
    h = next(i for i, x in enumerate(L) if x.strip().startswith("Time,"))
    cols = [c.strip() for c in L[h].split(",")]
    rows = []
    for x in L[h + 3:]:
        p = [c.strip() for c in x.split(",")]
        try:
            rows.append([float(p[0]), float(p[cols.index("Speed")]), float(p[cols.index("Total Forc")])])
        except (ValueError, IndexError):
            continue
    a = np.array(rows)
    return pd.DataFrame(dict(t=t0 + a[:, 0], v=a[:, 1] * NT.MPH, F=-a[:, 2]))


def events_for(pair: str, lag: float) -> pd.DataFrame:
    lf, sf, kind = NT.PAIRS[pair]
    B = NT.read_logger(lf, kind)
    psi0 = float(np.nanmedian(B.psi[B.psi < np.nanpercentile(B.psi, 50)]))
    Dy = read_dyno_force(sf)
    t = B.t.to_numpy()
    F = np.interp(t, Dy.t.to_numpy() + lag, Dy.F.to_numpy(), left=np.nan, right=np.nan)
    v = np.interp(t, Dy.t.to_numpy() + lag, Dy.v.to_numpy(), left=np.nan, right=np.nan)
    psi = np.clip(B.psi.to_numpy() - psi0, 0, None)
    T = B["T"].interpolate(limit=5).to_numpy()
    coast = (psi < 0.5) & (F > 0) & (v > 2) & np.isfinite(F)
    cfit = np.polyfit(v[coast], F[coast], 1) if coast.sum() > 20 else np.array([0.0, np.nanmedian(F[coast]) if coast.any() else 0.0])
    on = (psi > 5) & (v > 2)
    run = np.cumsum(np.r_[True, on[1:] != on[:-1]])
    rows = []
    for n_ev, r in enumerate(np.unique(run[on])):
        i = np.where(run == r)[0]
        if len(i) < 3:
            continue
        ps = psi[i]
        st = i[ps >= 0.5 * ps.max()]
        Tt = np.nanmean(T[i])
        if len(st) < 2 or not np.isfinite(Tt) or not np.isfinite(F[st]).all():
            continue
        pre = np.arange(max(0, i[0] - 10), i[0])
        pre = pre[coast[pre]]
        F0 = float(np.median(F[pre])) if len(pre) >= 2 else float(np.polyval(cfit, np.mean(v[st])))
        rows.append(dict(traj=pair, n_in_traj=n_ev, t_min=(t[i[0]] - t[0]) / 60, v0=float(v[i[0]]), v_steady=float(np.mean(v[st])), psi=float(np.mean(psi[st])),
                         F=float(np.mean(F[st])), F0=float(F0), F0_from="pre_coast" if len(pre) >= 2 else "speed_fit", T=float(Tt)))
    return pd.DataFrame(rows), dict(coast_fit_lbf=[float(cfit[0]), float(cfit[1])], coast_samples=int(coast.sum()), psi0=psi0)


def binned(D: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (D["T"] >= lo) & (D["T"] < hi)
        g = D[m].rel.to_numpy()
        if len(g) < 3:
            continue
        bs = [np.median(rng.choice(g, len(g))) for _ in range(2000)]
        rows.append(dict(lo=lo, hi=hi, n=len(g), T_mid=float(D[m]["T"].median()), median=float(np.median(g)),
                         ci5=float(np.percentile(bs, 5)), ci95=float(np.percentile(bs, 95))))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thermal", type=Path, required=True, help="ncst_thermal.py の出力 (時計のずれを使う)")
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    out = OUT / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    pairs = json.load(open(a.thermal / "summary.json"))["pairs"]
    parts, info = [], {}
    for p in ("CBD0", "CBD1", "CBDx3", "UDDS"):
        E, inf = events_for(p, float(pairs[p]["clock_lag_used_s"]))
        parts.append(E)
        info[p] = inf
    E = pd.concat(parts, ignore_index=True)
    E["Fb"] = E.F - E.F0
    mid = E[E["T"].between(50, 130)]
    k = np.polyfit(mid.psi, mid.Fb, 1)
    p_fit = float(-k[1] / k[0])                       # 参考 (使わない)
    p_th = P_TH_STD
    E["eff"] = E.Fb / (E.psi - p_th)
    ref = E[E["T"] < 50].eff.median()
    E["rel"] = E.eff / ref
    variants = {}
    for pth in (0.0, P_TH_STD, 5.0):
        for base in (True, False):
            num = E.Fb if base else E.F
            eff = num / (E.psi - pth)
            rel = eff / eff[E["T"] < 50].median()
            variants[f"p_th={pth:g}, {'F0 を引く' if base else 'F0 を引かない'}"] = [float(rel[(E["T"] >= lo) & (E["T"] < hi)].median()) for lo, hi in zip(BINS[:-1], BINS[1:])]
    E.to_csv(out / "brake_events_force.csv", index=False)
    S = dict(method="利き = (タイヤの力 - 直前の惰行の力) ÷ (圧力 - 押し出し圧)。タイヤの力は動力計の Total Force の符号を反転",
             pushout_psi_used=p_th, pushout_psi_xintercept_unreliable=round(p_fit, 2), fit_Fb_vs_psi_50_130C=dict(slope_lbf_per_psi=round(float(k[0]), 1), intercept_lbf=round(float(k[1]), 1), n=int(len(mid))),
             sensitivity_bin_medians=dict(bins=[f"{lo}-{hi}" for lo, hi in zip(BINS[:-1], BINS[1:])], **{k2: [round(x, 3) for x in v2] for k2, v2 in variants.items()}),
             F0_median_by_traj=E.groupby("traj").F0.median().round(1).to_dict(),
             ref_eff_below_50C_lbf_per_psi=round(float(ref), 2), n=int(len(E)), F0_from=E.F0_from.value_counts().to_dict(), coast=info)
    rng = np.random.default_rng(20261118)
    configure_japanese_font()
    sets = {"all": E, "v5_10": E[(E.v0 >= 5) & (E.v0 <= 10)]}
    titles = {"all": "全部の制動", "v5_10": "初速 5〜10 m/s の制動だけ (車速をそろえた)"}
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.9), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    for ax, (key, D) in zip(axs[:2], sets.items()):
        B = binned(D, rng)
        for tr, g in D.groupby("traj"):
            ax.scatter(g["T"], g.rel, s=22, color=COLORS[tr], alpha=0.8, edgecolor="white", linewidth=0.5, label=f"{LABELS[tr]} ({len(g)} 回)")
        ax.errorbar(B.T_mid, B["median"], yerr=[B["median"] - B.ci5, B.ci95 - B["median"]], fmt="s-", color="#222", ms=6, capsize=3, lw=1.5,
                    label="温度区間ごとの中央値 (90% 区間)")
        for _, r in B.iterrows():
            ax.annotate(f"n={int(r.n)}", (r.T_mid, r.ci95), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8, color="#444")
        ax.axhline(1, color="#888", lw=0.8)
        ax.set_xlabel("ブレーキ温度 (熱電対の読み) [°C]")
        ax.set_title(f"{titles[key]}  n={len(D)}")
        ax.grid(alpha=0.3)
        hot, cool = D[D["T"] >= 130].rel, D[D["T"] < 130].rel
        S[key] = dict(n=int(len(D)), spearman_T=round(float(D["T"].rank().corr(D.rel.rank())), 3),
                      median_rel_below_130=round(float(cool.median()), 3), median_rel_130_plus=round(float(hot.median()), 3) if len(hot) else None,
                      n_130_plus=int(len(hot)), bins=B.round(3).to_dict("records"), v0_median_by_traj=D.groupby("traj").v0.median().round(2).to_dict(),
                      psi_median_by_traj=D.groupby("traj").psi.median().round(2).to_dict())
        if len(hot) >= 3 and len(cool) >= 3:
            d = [np.median(rng.choice(hot.to_numpy(), len(hot))) - np.median(rng.choice(cool.to_numpy(), len(cool))) for _ in range(4000)]
            S[key]["diff_median_130plus_minus_below_ci90"] = [round(float(np.percentile(d, 5)), 3), round(float(np.percentile(d, 95)), 3)]
    axs[0].set_ylabel("ブレーキの利き (50 °C 未満の中央値 = 1)\n= (タイヤの力 − 惰行の力) ÷ (圧力 − 押し出し圧)")
    axs[0].legend(fontsize=8, loc="lower left")
    # 3 枚目: 定義を変えたときの、温度区間ごとの中央値
    ax = axs[2]
    mids = [(lo + hi) / 2 for lo, hi in zip(BINS[:-1], BINS[1:])]
    sty = {True: "-", False: "--"}
    cols = {0.0: "#16a085", P_TH_STD: "#222222", 5.0: "#d35400"}
    for name, vals in variants.items():
        pth = float(name.split(",")[0].split("=")[1])
        ax.plot(mids, vals, sty["引く" in name and "引かない" not in name], color=cols[pth], marker="o", ms=4, label=name)
    ax.axhline(1, color="#888", lw=0.8)
    ax.set_xlabel("ブレーキ温度の区間の中央 [°C]")
    ax.set_title("定義を変えたときの区間ごとの中央値 (全部の制動)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    w = E[E.traj.isin(["CBD0", "CBD1"])]
    S["warmup_rel_vs_order_spearman"] = round(float(pd.Series(np.arange(len(w))).corr(w.rel.reset_index(drop=True).rank())), 3)
    fig.suptitle("NCST 重量車 (台上試験): ブレーキ温度とタイヤの力で測った利き。温度は熱電対の読み (表面より低い可能性)。フェードの確認ではない", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "force_eff_vs_temp.png", dpi=130)
    plt.close(fig)
    (out / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    (out / "meta.json").write_text(json.dumps(dict(run_id=a.run_id, thermal=a.thermal.as_posix(), bins=BINS), ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in S.items() if k not in ("all", "v5_10")}, ensure_ascii=False, indent=1, default=float))
    for key in ("all", "v5_10"):
        s = S[key]
        print(key, {k: s[k] for k in s if k != "bins"})
        print([(b["lo"], b["hi"], b["n"], round(b["median"], 3), round(b["ci5"], 3), round(b["ci95"], 3)) for b in s["bins"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
