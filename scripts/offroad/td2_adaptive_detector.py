"""地形に依らない検出の試し: 「その場の普段」との比で強い振動を取り出し、補正した racepak の時刻のストロークで確かめる。

固定の閾値 (PoC: 全走行の走行中の窓の p99) は、荒れた地点ほど多く拾う。ここでは
  コントラスト = MTVV_total (1 秒移動) / 前後 ±context_s の MTVV_total の中央値
が閾値以上のピークを取る (IMU だけ)。ストロークは確かめにだけ使う:
  racepak の時刻を td2_sync_correct.py の 2 状態で補正し、狭い窓 [te + a, te + b] のストローク速度を、
  同じ走行の前後 ±30 s の時刻 (5 s 以内を除く) と比べた百分位を出す。地点ごとにまとめる

    uv run python scripts/offroad/td2_adaptive_detector.py --sync out/offroad/tartandrive2/20261117_r10 --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

import td2_reliability as REL
import td2_sync_correct as SC
import td2_vibration_poc as P


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offroad/td2_vibration_poc.yaml")
    ap.add_argument("--sync", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    ac = cfg["adaptive"]
    out = Path("out/offroad/tartandrive2") / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    ST = pd.read_csv(a.sync / "sync_states.csv")
    site = lambda n: pd.Series([n]).str.replace(r"\d{4}(-\d{2}){5}", "", regex=True).str.strip("_")[0]
    w = tuple(ac["window"])
    rows, hours = [], {}
    rng = np.random.default_rng(2)
    for d in sorted(Path(cfg["data_dir"]).iterdir()):
        R = P.load_run(d, cfg)
        if R is None:
            continue
        Sh = P.shock_series(R, cfg)
        if Sh is None:
            continue
        S = P.iso_series(R, cfg["lp8_hz"])
        St = ST[ST.run == R["name"]]
        tc, keep = SC.correct_times(Sh["g"], St)          # 状態の継ぎ目の重なりは捨てる
        Shc = dict(g=tc[keep], act=Sh["act"][keep])
        g, fs = R["g"], R["fs"]
        mv = (R["speed"] >= cfg["moving_min_speed_mps"]) & ~R["gap"]
        hours.setdefault(site(R["name"]), 0.0)
        hours[site(R["name"])] += mv.sum() / fs / 3600
        tot = np.nan_to_num(S["tot"])
        # 前後の中央値は 1 Hz に間引いて取る (速さのため)
        k = int(fs)
        t1 = tot[::k]
        med = median_filter(np.where(mv[::k], t1, np.nan), size=int(2 * ac["context_s"]) + 1, mode="nearest")
        med = np.where(np.isfinite(med), med, np.nanmedian(t1[mv[::k]]) if mv.any() else np.nan)
        base = np.interp(np.arange(len(g)), np.arange(len(t1)) * k, med)
        con = tot / np.maximum(base, 1e-6)
        x = np.where(mv & (tot >= ac["min_mtvv"]), con, 0)
        pk, _ = find_peaks(x, height=ac["contrast_min"], distance=int(cfg["events"]["min_separation_s"] * fs))

        def local_pct(te: float) -> float:
            v = REL.act_in(Shc, te, w)
            cand = np.arange(te - 30, te + 30.01, 0.5)
            cand = cand[(np.abs(cand - te) >= 5) & (cand > g[0] + 2) & (cand < g[-1] - 3)]
            cand = cand[np.interp(cand, g, R["speed"]) >= cfg["moving_min_speed_mps"]]
            return REL.pct_of(v, np.array([REL.act_in(Shc, c, w) for c in cand])) if len(cand) >= 20 else np.nan

        for i in pk:
            rows.append(dict(run=R["name"], site=site(R["name"]), kind="adaptive", t=round(g[i] - g[0], 2), t_abs=g[i], speed=float(R["speed"][i]),
                             mtvv_total=float(tot[i]), contrast=float(con[i]), local_shock_pct=local_pct(g[i])))
        # 対照: 同じ走行の無作為の走行中の時刻 (検出と同じ数)
        cand = np.flatnonzero(mv)
        for i in rng.choice(cand, size=min(len(pk), len(cand)), replace=False) if len(pk) else []:
            rows.append(dict(run=R["name"], site=site(R["name"]), kind="random", t=round(g[i] - g[0], 2), t_abs=g[i], speed=float(R["speed"][i]),
                             mtvv_total=float(tot[i]), contrast=float(con[i]), local_shock_pct=local_pct(g[i])))
        print(R["name"][:40], len(pk), flush=True)
    D = pd.DataFrame(rows)
    A, Rn = D[D.kind == "adaptive"], D[D.kind == "random"]
    by = []
    for s, h in hours.items():
        a_ = A[A.site == s]
        r_ = Rn[Rn.site == s]
        by.append(dict(site=s, moving_h=round(h, 3), n=len(a_), per_h=round(len(a_) / h, 1) if h else np.nan,
                       frac_local_ge_p90=float((a_.local_shock_pct >= 90).mean()) if len(a_) else np.nan,
                       random_frac_local_ge_p90=float((r_.local_shock_pct >= 90).mean()) if len(r_) else np.nan))
    BY = pd.DataFrame(by).sort_values("moving_h", ascending=False)
    poc = pd.read_csv(Path(cfg.get("adaptive_compare_poc", "out/offroad/tartandrive2/20261117_r3")) / "events.csv")
    poc["site"] = poc.run.map(site)
    BY["poc_fixed_threshold_n"] = BY.site.map(poc.site.value_counts()).fillna(0).astype(int)
    summ = dict(params=ac, events=len(A), per_h=round(len(A) / sum(hours.values()), 1),
                frac_local_ge_p90=float((A.local_shock_pct >= 90).mean()), random_frac_local_ge_p90=float((Rn.local_shock_pct >= 90).mean()),
                median_local_pct=float(A.local_shock_pct.median()), random_median_local_pct=float(Rn.local_shock_pct.median()),
                sites_with_events=int((BY.n > 0).sum()), sites=len(BY),
                cv_per_h_adaptive=float(BY.per_h[BY.moving_h >= 0.05].std() / BY.per_h[BY.moving_h >= 0.05].mean()),
                cv_per_h_fixed=float((BY.poc_fixed_threshold_n / BY.moving_h)[BY.moving_h >= 0.05].std()
                                     / (BY.poc_fixed_threshold_n / BY.moving_h)[BY.moving_h >= 0.05].mean()))
    D.round(4).to_csv(out / "detections.csv", index=False)
    BY.to_csv(out / "by_site.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    (out / "meta.json").write_text(json.dumps(dict(config=ac, sync=str(a.sync), run_id=a.run_id), ensure_ascii=False, indent=2))
    print(BY.to_string())
    print(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
