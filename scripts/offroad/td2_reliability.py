"""TartanDrive 2.0 の PoC (td2_vibration_poc.py) の結果がどこまで確かかを調べる。

1. 照合の窓の感度: racepak の窓 [te + a, te + b] を変える。ずらした窓 (前・後ろに数秒) を「誤った対応」の対照にする。
   荒れた路面は数秒続くので、時刻が合っていなくてもストロークは大きくなる。対照との差が、時刻の対応から来る分である
2. 比べる相手 (無作為の時刻) の取り方: 全走行・速度をそろえる・同じ走行・同じ地点・同じ走行の前後 ±30 s (近くの時刻)。
   近くの時刻と比べて大きいなら「その場の路面の中でも突出した入力」、全走行とだけ比べて大きいなら「荒れた区間」
3. 独立性: 同じ走行の近いイベントをまとめた「エピソード」の数と、走行を単位にした bootstrap の信頼区間
4. 一過性か持続か: イベントの MTVV_total と、前後 ±30 s の中央値の比 (IMU だけで決まる)
5. 1 件ずつの遅れ: 強い上下の衝撃ごとに racepak の遅れを推定し、分布・速度との関係・走行の中の変化を見る

    uv run python scripts/offroad/td2_reliability.py --poc out/offroad/tartandrive2/20261117_r3 --run-id <新しい ID>
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
from scipy.signal import butter, find_peaks, sosfiltfilt

import td2_vibration_poc as P


def act_in(Sh: dict, te: float, w: tuple[float, float]) -> float:
    return P.max_in(Sh["g"], Sh["act"], te + w[0], te + w[1])


def pct_of(v: float, ref: np.ndarray) -> float:
    ref = np.sort(ref[np.isfinite(ref)])
    return 100 * np.searchsorted(ref, v, side="right") / len(ref) if len(ref) else np.nan


def impulse_lags(R: dict, Sh: dict, rc: dict) -> list[dict]:
    """強い上下の衝撃ごとに、包絡線 (4 Hz 以下に平滑した |上下加速度| と |ストローク速度|) の相互相関で遅れを推定する。"""
    fs, g = R["fs"], R["g"]
    az = P.bp(R["X"]["az"] - R["X"]["az"].mean(), *rc["az_band_hz"], fs)
    s = np.interp(g, Sh["g"], Sh["RL"] + Sh["RR"])
    sv = P.bp(np.gradient(s, 1 / fs), 0.5, 10, fs)
    sm = butter(2, rc["env_lp_hz"], fs=fs, output="sos")
    ea, es = sosfiltfilt(sm, np.abs(az)), sosfiltfilt(sm, np.abs(sv))
    mv = R["speed"] >= 1.0
    if mv.sum() < 30 * fs:
        return []
    x = np.where(mv & ~R["gap"], np.abs(az), 0)
    pk, _ = find_peaks(x, height=np.percentile(np.abs(az)[mv], rc["peak_percentile"]), distance=int(rc["min_separation_s"] * fs))
    L = np.arange(int(rc["lag_range_s"][0] * fs), int(rc["lag_range_s"][1] * fs) + 1)
    h = int(rc["half_window_s"] * fs)
    rows = []
    for i in pk:
        if i - h + L[0] < 0 or i + h + L[-1] >= len(g):
            continue
        c = np.array([np.corrcoef(ea[i - h:i + h], es[i - h + l:i + h + l])[0, 1] for l in L])
        j = int(np.nanargmax(c))
        other = c[np.abs(L - L[j]) > int(0.3 * fs)]
        rows.append(dict(run=R["name"], t=round(g[i] - g[0], 2), t_abs=g[i], speed=float(R["speed"][i]), az=float(az[i]),
                         lag_s=L[j] / fs, corr=float(c[j]), corr_second=float(np.nanmax(other)) if len(other) else np.nan))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offroad/td2_vibration_poc.yaml")
    ap.add_argument("--poc", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    rcfg = cfg["reliability"]
    out = Path("out/offroad/tartandrive2") / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    E = pd.read_csv(a.poc / "events.csv")
    N0 = pd.read_csv(a.poc / "null_samples.csv")
    runs = {}
    for d in sorted(Path(cfg["data_dir"]).iterdir()):
        R = P.load_run(d, cfg)
        if R is None:
            continue
        S = P.iso_series(R, cfg["lp8_hz"])
        Sh = P.shock_series(R, cfg)
        if Sh is not None:
            runs[R["name"]] = (R, S, Sh)
    site = lambda n: pd.Series([n]).str.replace(r"\d{4}(-\d{2}){5}", "", regex=True).str.strip("_")[0]
    N0["site"] = N0.run.map(site)
    wins = {k: tuple(v) for k, v in rcfg["windows"].items()}

    # ---- 1. 窓の感度 (全走行の無作為の時刻と比べる)
    rows = []
    for k, w in wins.items():
        ev = np.array([act_in(runs[r][2], t, w) for r, t in zip(E.run, E.t_abs)])
        nu = np.array([act_in(runs[r][2], t, w) for r, t in zip(N0.run, N0.t_abs)])
        p90 = np.nanpercentile(nu, 90)
        rows.append(dict(window=k, a=w[0], b=w[1], frac_events_ge_null_p90=float(np.mean(ev >= p90)),
                         n_ge=int(np.sum(ev >= p90)), auc=float(np.mean([np.mean(nu[np.isfinite(nu)] < x) for x in ev])),
                         events_median=float(np.nanmedian(ev)), null_median=float(np.nanmedian(nu))))
        E[f"act_{k}"] = ev
        N0[f"act_{k}"] = nu
    Wsens = pd.DataFrame(rows)

    # ---- 2. 比べる相手の取り方 (窓は base)
    wb = wins["base"]
    loc_rows = []
    for idx, e in E.iterrows():
        R, S, Sh = runs[e.run]
        g = R["g"]
        # 近くの時刻: 同じ走行の ±local_s の、イベントから exclude_s 以上離れた走行中の時刻 (0.5 s ごと)
        cand = np.arange(e.t_abs - rcfg["local_s"], e.t_abs + rcfg["local_s"] + 1e-9, 0.5)
        cand = cand[(np.abs(cand - e.t_abs) >= rcfg["local_exclude_s"]) & (cand > g[0] + 2) & (cand < g[-1] - 3)]
        sp = np.interp(cand, g, R["speed"])
        cand = cand[sp >= cfg["moving_min_speed_mps"]]
        loc_act = np.array([act_in(Sh, t, wb) for t in cand])
        loc_tot = np.array([P.max_in(g, S["tot"], t - 0.5, t + 0.5) for t in cand])
        same_run = N0[N0.run == e.run]["act_base"].to_numpy()
        same_site = N0[N0.site == site(e.run)]["act_base"].to_numpy()
        w_sp = max(1.0, 0.25 * e.speed)
        sp_m = N0[(N0.speed - e.speed).abs() <= w_sp]["act_base"].to_numpy()
        loc_rows.append(dict(
            event_id=e.event_id, run=e.run, site=site(e.run), t=e.t, speed=e.speed, mtvv_total=e.mtvv_total, dominant_axis=e.dominant_axis,
            act=e.act_base, pct_global=pct_of(e.act_base, N0.act_base.to_numpy()), pct_speed=pct_of(e.act_base, sp_m),
            pct_same_run=pct_of(e.act_base, same_run) if len(same_run) >= 20 else np.nan, n_same_run=len(same_run),
            pct_same_site=pct_of(e.act_base, same_site) if len(same_site) >= 20 else np.nan, n_same_site=len(same_site),
            pct_local=pct_of(e.act_base, loc_act) if len(loc_act) >= 20 else np.nan, n_local=len(loc_act),
            # 4. 一過性か持続か (IMU だけ): イベントの MTVV と近くの時刻の MTVV の中央値の比
            imu_contrast=float(e.mtvv_total / np.nanmedian(loc_tot)) if len(loc_tot) else np.nan,
            shock_contrast=float(e.act_base / np.nanmedian(loc_act)) if len(loc_act) else np.nan))
    Lc = pd.DataFrame(loc_rows)

    # ---- 3. 独立性: 同じ走行で episode_gap_s 以内のイベントを 1 エピソードにまとめる
    E = E.sort_values(["run", "t"])
    ep, last = [], {}
    k = 0
    for _, e in E.iterrows():
        if e.run in last and e.t - last[e.run][0] <= rcfg["episode_gap_s"]:
            ep.append(last[e.run][1])
        else:
            k += 1
            ep.append(k)
        last[e.run] = (e.t, ep[-1])
    E["episode"] = ep
    Lc = Lc.merge(E[["event_id", "episode"]], on="event_id")
    # 走行を単位にした bootstrap (全走行の無作為の時刻の p90 以上の割合、近くの時刻の p90 以上の割合)
    runs_ev = Lc.run.unique()
    bs = {"pct_global": [], "pct_local": [], "pct_same_run": []}
    for _ in range(rcfg["bootstrap"]):
        pick = rng.choice(runs_ev, len(runs_ev), replace=True)
        sub = pd.concat([Lc[Lc.run == r] for r in pick])
        for c in bs:
            v = sub[c].dropna()
            bs[c].append(np.mean(v >= 90) if len(v) else np.nan)
    ci = {c: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for c, v in bs.items()}

    # ---- 5. 1 件ずつの遅れ
    il = []
    for R, S, Sh in runs.values():
        il += impulse_lags(R, Sh, rcfg["impulse_lag"])
    IL = pd.DataFrame(il)
    good = IL[(IL["corr"] >= rcfg["impulse_lag"]["min_corr"]) & (IL["corr"] - IL.corr_second >= rcfg["impulse_lag"]["min_margin"])]

    summ = dict(
        events=len(E), episodes=int(E.episode.nunique()), runs_with_events=int(E.run.nunique()),
        window_sensitivity=Wsens.round(3).to_dict(orient="records"),
        comparison={c: dict(n=int(Lc[c].notna().sum()), median_pct=float(Lc[c].median()), frac_ge_p90=float((Lc[c].dropna() >= 90).mean()),
                            frac_ge_p50=float((Lc[c].dropna() >= 50).mean()))
                    for c in ("pct_global", "pct_speed", "pct_same_run", "pct_same_site", "pct_local")},
        bootstrap_by_run_frac_ge_p90_ci95=ci,
        imu_contrast=dict(p10=float(Lc.imu_contrast.quantile(0.1)), p50=float(Lc.imu_contrast.median()), p90=float(Lc.imu_contrast.quantile(0.9)),
                          n_ge_2=int((Lc.imu_contrast >= 2).sum())),
        corr_imu_contrast_vs_pct_local=float(Lc[["imu_contrast", "pct_local"]].corr("spearman").iloc[0, 1]),
        impulse_lag=dict(impulses=len(IL), good=len(good), p10=float(good.lag_s.quantile(0.1)), p25=float(good.lag_s.quantile(0.25)),
                         p50=float(good.lag_s.median()), p75=float(good.lag_s.quantile(0.75)), p90=float(good.lag_s.quantile(0.9)),
                         spearman_lag_vs_speed=float(good[["lag_s", "speed"]].corr("spearman").iloc[0, 1]),
                         frac_in_base_window=float(((good.lag_s >= wb[0]) & (good.lag_s <= wb[1])).mean())),
    )
    Wsens.round(4).to_csv(out / "window_sensitivity.csv", index=False)
    Lc.round(4).to_csv(out / "event_comparisons.csv", index=False)
    E.round(4).to_csv(out / "events_with_windows.csv", index=False)
    IL.round(4).to_csv(out / "impulse_lags.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    (out / "meta.json").write_text(json.dumps(dict(config=cfg, poc=str(a.poc), run_id=a.run_id), ensure_ascii=False, indent=2))

    plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].bar(Wsens.window, Wsens.frac_events_ge_null_p90, color=["#1f4e9c" if w[0] >= 0 and w[1] <= 3 else "#9aa5b1" for w in wins.values()])
    ax[0].axhline(0.1, color="#c2410c", ls="--", lw=1, label="無作為なら 10%")
    ax[0].set_ylabel("イベントの割合 (無作為の時刻の p90 以上)")
    ax[0].set_title("照合の窓 (灰色はずらした対照)", fontsize=10)
    ax[0].tick_params(axis="x", rotation=45, labelsize=8)
    ax[0].legend(fontsize=8)
    cols = ["pct_global", "pct_speed", "pct_same_site", "pct_same_run", "pct_local"]
    lab = ["全走行", "速度をそろえる", "同じ地点", "同じ走行", "前後 ±30 s"]
    ax[1].boxplot([Lc[c].dropna() for c in cols], labels=lab)
    ax[1].axhline(90, color="#c2410c", ls="--", lw=1)
    ax[1].set_ylabel("イベントのストローク速度の百分位")
    ax[1].set_title("比べる相手の取り方", fontsize=10)
    ax[1].tick_params(axis="x", labelsize=8)
    ax[2].scatter(good.speed, good.lag_s, s=10, color="#5b8bd6")
    ax[2].axhspan(wb[0], wb[1], color="#fde68a", alpha=0.4, label="照合の窓")
    ax[2].set_xlabel("速度 [m/s]")
    ax[2].set_ylabel("衝撃ごとの racepak の遅れ [s]")
    ax[2].set_title(f"衝撃ごとの遅れ (n={len(good)})", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
