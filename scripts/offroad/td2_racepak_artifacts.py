"""racepak の信号の、計測の誤りとみられる所を全走行で探し、候補 (td2_candidate_profile.py) との重なりを数える。

1. 左右の後輪のストロークが、高い周波数 (既定 5 Hz 以上) で鏡写しに振れる短い区間。
   (A047 で見つけた振れは、補正の継ぎ目で 2 つの区間を混ぜたことによるもので、td2_sync_correct.correct_times で直した。ここでは直した後に残るものを探す)
   0.5 s の窓ごとに、高域の RMS が左右とも走行の上位 1% 以上で、左右の高域の相関が −0.7 以下の窓を「鏡写しの振れ」とする
2. ブレーキとみられる信号の飛び値 (6,000 以上) が、候補の前後 [te − 2, te + 1] に出る割合と、無作為の時刻での割合

    uv run python scripts/offroad/td2_racepak_artifacts.py --profile out/offroad/tartandrive2/<id> --sync out/offroad/tartandrive2/20261117_r10 --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, sosfiltfilt

import td2_sync_correct as SC
import td2_vibration_poc as P


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offroad/td2_vibration_poc.yaml")
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--sync", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    ac = cfg["artifacts"]
    lat = cfg["profile"]["racepak_latency_s"]
    out = Path("out/offroad/tartandrive2") / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    F = pd.read_csv(a.profile / "candidates.csv")
    N = pd.read_csv(a.profile / "random_times.csv")
    ST = pd.read_csv(a.sync / "sync_states.csv")
    bursts, brk_rows = [], []
    for d in sorted(Path(cfg["data_dir"]).iterdir()):
        sh = P.load_topic(d, "shock_pos")
        ct = P.load_topic(d, "controls")
        if sh is None:
            continue
        t, x = sh
        tc, keep = SC.correct_times(t, ST[ST.run == d.name], lat)
        tc, x = tc[keep], x[keep]
        fs = 50.0
        g = np.arange(tc[0], tc[-1], 1 / fs)
        if len(g) < 200:
            continue
        hp = butter(2, ac["hp_hz"], btype="high", fs=fs, output="sos")
        L = sosfiltfilt(hp, np.interp(g, tc, x[:, 2]))
        R_ = sosfiltfilt(hp, np.interp(g, tc, x[:, 3]))
        n = int(ac["win_s"] * fs)
        m = len(g) // n
        Lw, Rw = L[:m * n].reshape(m, n), R_[:m * n].reshape(m, n)
        rl, rr = np.sqrt((Lw ** 2).mean(1)), np.sqrt((Rw ** 2).mean(1))
        c = np.array([np.corrcoef(Lw[i], Rw[i])[0, 1] for i in range(m)])
        th_l, th_r = np.percentile(rl, ac["rms_pct"]), np.percentile(rr, ac["rms_pct"])
        hit = (rl >= th_l) & (rr >= th_r) & (c <= ac["corr_max"])
        for i in np.flatnonzero(hit):
            bursts.append(dict(run=d.name, t_start=g[i * n], t_end=g[(i + 1) * n - 1], rms_RL=rl[i], rms_RR=rr[i], corr=c[i]))
        if ct is not None:
            t2, c2 = ct
            t2c, keep2 = SC.correct_times(t2, ST[ST.run == d.name], lat)
            t2c, c2 = t2c[keep2], c2[keep2]
            bad = c2[:, 1] > cfg["reaction"]["brake_invalid_above"]
            brk_rows.append(dict(run=d.name, frac_invalid=float(bad.mean())))
            for df, kind in ((F[F.run == d.name], "candidate"), (N[N.run == d.name], "random")):
                for te in df.t_abs:
                    w = (t2c >= te + ac["brake_window_s"][0]) & (t2c <= te + ac["brake_window_s"][1])
                    brk_rows.append(dict(run=d.name, kind=kind, t_abs=te, any_invalid=bool(bad[w].any()) if w.any() else np.nan))
    B = pd.DataFrame(bursts)
    BR = pd.DataFrame(brk_rows)

    def overlaps(df):
        res = []
        for r, te in zip(df.run, df.t_abs):
            b = B[B.run == r] if len(B) else B
            res.append(bool(((b.t_end >= te + ac["burst_window_s"][0]) & (b.t_start <= te + ac["burst_window_s"][1])).any()) if len(b) else False)
        return np.array(res)
    F["racepak_mirror_burst"] = overlaps(F)
    N["racepak_mirror_burst"] = overlaps(N)
    ev = BR[BR.kind == "candidate"].any_invalid
    rn = BR[BR.kind == "random"].any_invalid
    summ = dict(
        mirror_bursts=len(B), runs_with_bursts=int(B.run.nunique()) if len(B) else 0,
        candidates_with_burst=int(F.racepak_mirror_burst.sum()), candidate_ids_with_burst=F.cand_id[F.racepak_mirror_burst].tolist(),
        random_frac_with_burst=float(N.racepak_mirror_burst.mean()), candidates_frac_with_burst=float(F.racepak_mirror_burst.mean()),
        brake_invalid_overall_frac=float(BR.frac_invalid.dropna().mean()),
        brake_invalid_near_candidates=float(ev.mean()), brake_invalid_near_random=float(rn.mean()),
        flags_on_burst_candidates=F[F.racepak_mirror_burst][[c for c in F.columns if c.startswith("flag_")]].sum().to_dict(),
    )
    B.to_csv(out / "mirror_bursts.csv", index=False)
    BR.to_csv(out / "brake_invalid.csv", index=False)
    F[["cand_id", "run", "t", "racepak_mirror_burst"]].to_csv(out / "candidates_artifact_flags.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    (out / "meta.json").write_text(json.dumps(dict(config=ac, profile=str(a.profile), run_id=a.run_id), ensure_ascii=False, indent=2))
    print(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
