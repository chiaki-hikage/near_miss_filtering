"""手元の commaCarSegments (取得済み) の、10 分以上続く走行に熱負荷の高い制動があるかをふるい分ける。

commaCarSegments の rlog には CAN しか無く、GPS・高度が無い。勾配は Toyota の加速度センサ (重力を含む) と
輪速の微分の差から見当をつける: theta ≈ asin((ACCEL_X - dv/dt) / g)。制動中は車体の前のめりが混ざる
(RAV4 2017 で 1 m/s^2 あたり約 0.26 deg、docs/brake_performance_rav4.md 4 章) ので、ふるい分けにだけ使う。

熱の指標は brake_performance_poc と同じ考え: ブレーキを踏んでいる間 (BRAKE_PRESSED か brake_mc_mpa > 0.1) の
max(0, -dv/dt - g sin(theta) - 抵抗)·v の 300 s・900 s の和。抵抗は RAV4 2017 の模型 M4 の値 (0.107 + 0.000368 v^2) を
仮に使う (車種ごとには当てはめていない)。Prius はハイブリッドで回生ブレーキがあるので、摩擦ブレーキの熱は少なめになる。

    uv run python scripts/brake/ccs_heat_screen.py --run-id <新しい ID> [--min-run 10]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from near_miss.config import find_vehicle_config_for_platform, load_vehicle_configs
from near_miss.io import comma_car_segments as CCS

import brake_model as BM

OUT = Path("out/brake_fade")
PLATFORMS = ["TOYOTA_RAV4_TSS2", "TOYOTA_COROLLA_TSS2", "TOYOTA_HIGHLANDER_TSS2", "TOYOTA_PRIUS"]
DT = 0.05
G = 9.80665
RESIST = (0.107, 0.000368)


def grid(seg, t0: float | None):
    ch = seg.channels
    need = ["ws_fl_mps", "ws_fr_mps", "ws_rl_mps", "ws_rr_mps", "accel_x"]
    if any(k not in ch for k in need):
        return None
    t = ch["ws_fl_mps"].t
    tg = np.arange(t.min(), t.max(), DT)
    v = np.mean([np.interp(tg, ch[k].t, ch[k].v) for k in need[:4]], axis=0)
    w = int(0.5 / DT)
    vs = np.convolve(v, np.ones(w) / w, mode="same")
    a = np.gradient(vs, DT)
    ax = np.interp(tg, ch["accel_x"].t, ch["accel_x"].v)
    axs = np.convolve(ax, np.ones(w * 4) / (w * 4), mode="same")
    th = np.arcsin(np.clip((axs - np.convolve(a, np.ones(w * 4) / (w * 4), mode="same")) / G, -0.3, 0.3))
    pr = np.interp(tg, ch["brake_pressed"].t, ch["brake_pressed"].v) > 0.5 if "brake_pressed" in ch else np.zeros(len(tg), bool)
    mc = np.interp(tg, ch["brake_mc_mpa"].t, ch["brake_mc_mpa"].v) if "brake_mc_mpa" in ch else np.zeros(len(tg))
    df = pd.DataFrame(dict(t=tg, v=vs, a=a, theta=th, pressed=pr, mc=mc))
    df = df.iloc[w * 2:-w * 2]
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ccs", default="raw_data/comma_car_segments")
    ap.add_argument("--min-run", type=int, default=10)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    out = OUT / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    cfgs = load_vehicle_configs()
    rows = []
    for plat in PLATFORMS:
        vc = find_vehicle_config_for_platform(plat, cfgs)
        refs = CCS.find_segments(a.ccs, plat)
        by = collections.defaultdict(list)
        for r in refs:
            by[r.drive_id].append(r)
        for drive, rs in by.items():
            rs = sorted(rs, key=lambda r: r.index)
            runs, cur = [], [rs[0]]
            for x, y in zip(rs, rs[1:]):
                if y.index == x.index + 1:
                    cur.append(y)
                else:
                    runs.append(cur)
                    cur = [y]
            runs.append(cur)
            for run in runs:
                if len(run) < a.min_run:
                    continue
                parts = []
                for r in run:
                    try:
                        s = CCS.load_segment(r, vc, with_radar=False)
                    except Exception:  # noqa: BLE001
                        continue
                    g = grid(s, None)
                    if g is not None:
                        parts.append(g)
                if not parts:
                    continue
                D = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)
                blk = (D.t.diff().fillna(1e9) > 5).cumsum().to_numpy()
                on = D.pressed.to_numpy() | (D.mc.to_numpy() > 0.1)
                d = -D.a - G * np.sin(D.theta) - (RESIST[0] + RESIST[1] * D.v**2)
                p = np.where(on, np.clip(d, 0, None) * D.v, 0.0)
                H = BM.heat_indices(p, blk, DT, [300], [300, 900])
                drop = -np.cumsum((D.v * np.sin(D.theta) * DT).to_numpy())
                rows.append(dict(platform=plat, drive=drive, start_seg=run[0].index, n_min=len(run), hours=len(D) * DT / 3600,
                                 E300_max=float(H["E300"].max()), E900_max=float(H["E900"].max()), H300_max=float(H["H300"].max()),
                                 brake_frac=float(on.mean()), mc_max_mpa=float(D.mc.max()), v_max_kmh=float(D.v.max() * 3.6),
                                 est_net_descent_m=float(drop[-1] - drop[0]),
                                 est_max_drop_300s_m=float(np.max(drop[int(300 / DT):] - drop[:-int(300 / DT)])) if len(D) > 300 / DT else np.nan))
                print(plat, drive[-24:], run[0].index, len(run), round(rows[-1]["E300_max"]), flush=True)
    R = pd.DataFrame(rows).sort_values("E300_max", ascending=False)
    R.to_csv(out / "runs.csv", index=False)
    ref = 0.5 * (100 / 3.6) ** 2
    S = dict(n_runs=int(len(R)), hours=round(float(R.hours.sum()), 2), by_platform=R.groupby("platform").size().to_dict(),
             E300_max_over_runs=round(float(R.E300_max.max()), 1), E300_max_in_ref_stops=round(float(R.E300_max.max() / ref), 2),
             E900_max_over_runs=round(float(R.E900_max.max()), 1), E900_max_in_ref_stops=round(float(R.E900_max.max() / ref), 2),
             runs_E300_over_2_stops=int((R.E300_max > 2 * ref).sum()), top=R.head(8).round(2).to_dict("records"))
    (out / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    (out / "meta.json").write_text(json.dumps(dict(run_id=a.run_id, min_run=a.min_run, platforms=PLATFORMS, resist=RESIST), ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in S.items() if k != "top"}, ensure_ascii=False, indent=1))
    print(R.head(10).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
