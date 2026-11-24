"""racepak の時刻のずれを、KITTI 形式の信号だけで補正する (2 状態モデル) と、その補正で PoC の確かめをやり直す。

分かっていること (td2_racepak_timing.py、2026-11-17):
  racepak のヘッダの時刻 − bag の記録時刻 は、+1.00 s か 0 の 2 つの値を取り、走行の中で切り替わる。
  IMU は −0.02 s で一定。包絡線の相互相関の遅れは、+1 の所で約 1.3 s、0 の所で約 0.4 s だった (8 block すべて)
補正:
  1. 窓 (既定 20 s、5 s ごと) ごとに、遅れの候補 2 つの範囲 [lo, hi] と [lo + 1, hi + 1] で包絡線の相関の最大を比べ、
     大きい方を状態 (0 か 1) とする。差が小さい窓は前後の窓の多数決で埋める
  2. racepak の時刻から 状態 × 1.0 s を引く
  3. bag で測った状態 (timing.csv・blocks.csv) と照らし合わせる
  4. 補正した時刻で、衝撃ごとの遅れ (残り = 伝わるまでの遅れ + 物理の遅れ) を出し、速度との関係を見る
  5. 補正した時刻と狭い窓で、イベントのストローク速度を無作為の時刻・近くの時刻と比べ直す

    uv run python scripts/offroad/td2_sync_correct.py --poc out/offroad/tartandrive2/20261117_r3 --run-id <新しい ID>
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
from scipy.signal import butter, sosfiltfilt

import td2_reliability as REL
import td2_vibration_poc as P


def envelopes(R: dict, Sh: dict, sc: dict):
    fs, g = R["fs"], R["g"]
    sm = butter(2, sc["env_lp_hz"], fs=fs, output="sos")
    az = P.bp(R["X"]["az"] - R["X"]["az"].mean(), *sc["env_band_hz"], fs)
    s = np.interp(g, Sh["g"], Sh["RL"] + Sh["RR"])      # racepak の時刻 = g の値 (補正前)
    sv = P.bp(np.gradient(s, 1 / fs), *sc["env_band_hz"], fs)
    return sosfiltfilt(sm, np.abs(az)), sosfiltfilt(sm, np.abs(sv))


def states(R: dict, ea: np.ndarray, es: np.ndarray, sc: dict) -> pd.DataFrame:
    fs, g = R["fs"], R["g"]
    n, step = int(sc["win_s"] * fs), int(sc["step_s"] * fs)
    lo, hi = sc["cand_lag_s"]
    L0 = np.arange(int(lo * fs), int(hi * fs) + 1)
    L1 = L0 + int(fs)
    rows = []
    for i in range(0, len(g) - n - L1[-1], step):
        a = ea[i:i + n]
        c0 = max(np.corrcoef(a, es[i + l:i + l + n])[0, 1] for l in L0)
        c1 = max(np.corrcoef(a, es[i + l:i + l + n])[0, 1] for l in L1)
        rows.append(dict(t_center=g[i + n // 2], c0=c0, c1=c1, speed=float(np.median(R["speed"][i:i + n]))))
    S = pd.DataFrame(rows)
    if S.empty:
        return S
    S["state_raw"] = np.where(S.c1 - S.c0 >= sc["min_margin"], 1, np.where(S.c0 - S.c1 >= sc["min_margin"], 0, -1))
    S.loc[S.speed < 1.0, "state_raw"] = -1
    # 決まらない窓は、前後 k 窓の多数決 (それでも決まらなければ走行全体の多数)
    st = S.state_raw.to_numpy().copy()
    known = st >= 0
    overall = int(np.round(st[known].mean())) if known.any() else 1
    k = sc["fill_neighbors"]
    for j in np.flatnonzero(~known):
        nb = st[max(0, j - k):j + k + 1]
        nb = nb[nb >= 0]
        st[j] = int(np.round(nb.mean())) if len(nb) else overall
    S["state"] = st
    return S


def state_at(S: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    if S.empty:
        return np.ones(len(t), int)
    j = np.clip(np.searchsorted(S.t_center.to_numpy(), t), 0, len(S) - 1)
    j0 = np.clip(j - 1, 0, len(S) - 1)
    tc = S.t_center.to_numpy()
    jj = np.where(np.abs(tc[j0] - t) < np.abs(tc[j] - t), j0, j)
    return S.state.to_numpy()[jj]


def correct_times(t: np.ndarray, St: pd.DataFrame, latency: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """racepak の時刻 t (昇順) を補正し、(補正した時刻, 使うサンプルの印) を返す。

    状態が 0 → 1 に変わる所では、後ろのサンプルが 1 s 戻って前の 1 s と重なる。並べ替えて混ぜると、
    1 s 離れた 2 つの区間のサンプルが交互に並び、高い周波数の振れに見える (2026-11-17、A047 で見つけた)。
    そこで、それまでの最大の時刻より前に戻るサンプルは使わない (重なりの後ろの側を捨てる)。1 → 0 の所は 1 s の抜けになる。
    """
    tc = t - state_at(St, t) * 1.0 - latency
    prev_max = np.maximum.accumulate(np.r_[-np.inf, tc[:-1]])
    return tc, tc > prev_max


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offroad/td2_vibration_poc.yaml")
    ap.add_argument("--poc", type=Path, required=True)
    ap.add_argument("--bag-timing", nargs="*", default=["out/offroad/tartandrive2/20261117_r8_frames/timing.csv"])
    ap.add_argument("--bag-blocks", nargs="*", default=["out/offroad/tartandrive2/20261117_r9_racepak_timing"])
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    sc = cfg["sync_correct"]
    out = Path("out/offroad/tartandrive2") / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    E = pd.read_csv(a.poc / "events.csv")
    N0 = pd.read_csv(a.poc / "null_samples.csv")

    runs, allS, il = {}, [], []
    for d in sorted(Path(cfg["data_dir"]).iterdir()):
        R = P.load_run(d, cfg)
        if R is None:
            continue
        Sh = P.shock_series(R, cfg)
        if Sh is None:
            continue
        S = P.iso_series(R, cfg["lp8_hz"])
        ea, es = envelopes(R, Sh, sc)
        St = states(R, ea, es, sc)
        St.insert(0, "run", R["name"])
        allS.append(St)
        # racepak の時刻を補正した写し
        Shc = dict(Sh)
        tc, keep = correct_times(Sh["g"], St)
        for key in ("act", "RL", "RR"):
            Shc[key] = Sh[key][keep]
        Shc["g"] = tc[keep]
        runs[R["name"]] = (R, S, Sh, Shc, St)
        # 衝撃ごとの遅れ (補正後)
        il += [dict(r, corrected=True) for r in REL.impulse_lags(R, Shc, cfg["reliability"]["impulse_lag"])]
    ST = pd.concat(allS, ignore_index=True)
    IL = pd.DataFrame(il)
    ilc = cfg["reliability"]["impulse_lag"]
    good = IL[(IL["corr"] >= ilc["min_corr"]) & (IL["corr"] - IL.corr_second >= ilc["min_margin"])]

    # ---- bag で測った状態との照合
    chk = []
    for f in a.bag_timing:
        T = pd.read_csv(f)
        T = T[T.topic == "/shock_pos"]
        for (item, run), g in T.groupby(["item", "run"]):
            if run not in runs:
                continue
            hr = float(np.median(g.header_t - g.record_t))
            t = float(np.median(g.record_t))
            chk.append(dict(source=f, item=item, run=run, t=t, bag_state=int(round(hr)), est_state=int(state_at(runs[run][4], np.array([t]))[0])))
    for dd in a.bag_blocks:
        B = pd.read_csv(Path(dd) / "blocks.csv")
        M = pd.read_csv(Path(dd) / "messages.csv")
        run = json.loads((Path(dd) / "meta.json").read_text())["run"]
        M = M[M.topic == "/shock_pos"]
        for b, g in M.groupby("block_start_s"):
            hr = float(np.median(g.header_t - g.record_t))
            t = float(np.median(g.record_t))
            chk.append(dict(source=str(dd), item=f"block {b:.0f}", run=run, t=t, bag_state=int(round(hr)),
                            est_state=int(state_at(runs[run][4], np.array([t]))[0])))
    CK = pd.DataFrame(chk)

    # ---- 補正した時刻で、狭い窓の確かめをやり直す
    wins = {k: tuple(v) for k, v in sc["windows"].items()}
    res = []
    for k, w in wins.items():
        for label, idx in (("uncorrected", 2), ("corrected", 3)):
            ev = np.array([REL.act_in(runs[r][idx], t, w) for r, t in zip(E.run, E.t_abs)])
            nu = np.array([REL.act_in(runs[r][idx], t, w) for r, t in zip(N0.run, N0.t_abs)])
            p90 = np.nanpercentile(nu, 90)
            # 近くの時刻 (±30 s、5 s 以内を除く)
            loc = []
            for r, t, x in zip(E.run, E.t_abs, ev):
                R = runs[r][0]
                cand = np.arange(t - 30, t + 30.01, 0.5)
                cand = cand[(np.abs(cand - t) >= 5) & (cand > R["g"][0] + 2) & (cand < R["g"][-1] - 3)]
                cand = cand[np.interp(cand, R["g"], R["speed"]) >= cfg["moving_min_speed_mps"]]
                loc.append(REL.pct_of(x, np.array([REL.act_in(runs[r][idx], c, w) for c in cand])))
            loc = np.array(loc)
            res.append(dict(window=k, a=w[0], b=w[1], timing=label, frac_ge_null_p90=float(np.mean(ev >= p90)), n_ge=int(np.sum(ev >= p90)),
                            auc=float(np.mean([np.mean(nu[np.isfinite(nu)] < x) for x in ev])),
                            frac_ge_local_p90=float(np.nanmean(loc >= 90)), median_local_pct=float(np.nanmedian(loc))))
    RS = pd.DataFrame(res)

    X = np.c_[np.ones(len(good)), 1 / good.speed]
    m = good.speed >= 1.5
    coef = np.linalg.lstsq(X[m.to_numpy()], good.lag_s[m], rcond=None)[0] if m.sum() > 10 else [np.nan, np.nan]
    summ = dict(
        windows=len(ST), frac_state1=float((ST.state == 1).mean()), frac_undecided_raw=float((ST.state_raw == -1).mean()),
        runs_all_state1=int(ST.groupby("run").state.apply(lambda s: (s == 1).all()).sum()),
        runs_mixed=int(ST.groupby("run").state.apply(lambda s: s.nunique() > 1).sum()), runs=int(ST.run.nunique()),
        bag_check=dict(n=len(CK), agree=int((CK.bag_state == CK.est_state).sum())),
        impulse_lag_corrected=dict(good=len(good), p10=float(good.lag_s.quantile(0.1)), p25=float(good.lag_s.quantile(0.25)),
                                   p50=float(good.lag_s.median()), p75=float(good.lag_s.quantile(0.75)), p90=float(good.lag_s.quantile(0.9)),
                                   iqr=float(good.lag_s.quantile(0.75) - good.lag_s.quantile(0.25)),
                                   fit_lag_eq_a_plus_b_over_v=dict(a_s=float(coef[0]), b_m=float(coef[1])),
                                   by_speed={f"{lo}-{hi}": dict(n=int(((good.speed >= lo) & (good.speed < hi)).sum()),
                                                                 p50=float(good.lag_s[(good.speed >= lo) & (good.speed < hi)].median()))
                                             for lo, hi in ((1.5, 3), (3, 5), (5, 8), (8, 15))}),
        recheck=RS.round(3).to_dict(orient="records"),
    )
    ST.round(4).to_csv(out / "sync_states.csv", index=False)
    CK.to_csv(out / "bag_check.csv", index=False)
    IL.round(4).to_csv(out / "impulse_lags_corrected.csv", index=False)
    RS.round(4).to_csv(out / "recheck_windows.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    (out / "meta.json").write_text(json.dumps(dict(config=sc, poc=str(a.poc), run_id=a.run_id), ensure_ascii=False, indent=2))

    plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    old = pd.read_csv(a.poc.parent / cfg["sync_correct"]["compare_impulse_lags"]) if (a.poc.parent / cfg["sync_correct"]["compare_impulse_lags"]).exists() else None
    b = np.arange(-0.5, 3.01, 0.05)
    if old is not None:
        go = old[(old["corr"] >= ilc["min_corr"]) & (old["corr"] - old.corr_second >= ilc["min_margin"])]
        ax[0].hist(go.lag_s, bins=b, color="#9aa5b1", alpha=0.7, label=f"補正前 (n={len(go)})")
    ax[0].hist(good.lag_s, bins=b, color="#1f4e9c", alpha=0.8, label=f"補正後 (n={len(good)})")
    ax[0].set_xlabel("衝撃ごとの racepak の遅れ [s]")
    ax[0].set_ylabel("衝撃の数")
    ax[0].legend(fontsize=8)
    ax[1].scatter(1 / good.speed, good.lag_s, s=10, color="#1f4e9c")
    xs = np.linspace(1 / 15, 1 / 1.5, 50)
    ax[1].plot(xs, coef[0] + coef[1] * xs, color="#c2410c", label=f"遅れ = {coef[0]:.2f} s + {coef[1]:.2f} m / 速度")
    ax[1].set_xlabel("1 / 速度 [s/m]")
    ax[1].set_ylabel("補正後の遅れ [s]")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sync_correction.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
