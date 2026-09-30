#!/usr/bin/env python3
"""生チャネルの**窓ごと**の要約統計を出す。L1 の粒度を変えて評価するための素。

L1 をセグメント (60 秒) 単位で当てると、60 秒に 1 度でも動きがあれば
その 60 秒すべてが残る。**窓を短くすればどこまで削れるか**を測れるようにする。

補間も微分もしない。生チャネルの受信時刻のまま、窓に切って min/max を取る。
複数の窓長を 1 回の読み出しでまとめて出す。

  uv run python scripts/detect/window_stats.py --platform TOYOTA_RAV4_TSS2 -w 8
  uv run python scripts/detect/window_stats.py --comma2k19 raw_data/Chunk_1 -w 8
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from near_miss.config import (
    DEFAULT_VEHICLE_DIR,
    find_vehicle_config,
    load_vehicle_configs,
)

# 既定の窓長 [秒]。60 は現行のセグメント単位に相当する。
WINDOWS = (2.0, 5.0, 10.0, 20.0, 60.0)

# L1 に使う量と、その素になる生チャネルの候補 (先に見つかったものを使う)。
SPEED = ("speed_mps", "ws_fl_mps", "ws_fl_can_mps")
STEER = ("steer_deg", "steer_can_deg")
YAW = ("yaw_rate",)
AX = ("accel_x",)
AY = ("accel_y",)
GVC = ("gvc_mps2",)
# 輪速 4 本。窓ごとに「4 本の最大 - 4 本の最小」を取る。
# 瞬時のばらつきの**上界**であって同時刻の差ではないが、min/max だけで作れる。
WHEELS = (("ws_fl_mps", "ws_fl_can_mps"), ("ws_fr_mps", "ws_fr_can_mps"),
          ("ws_rl_mps", "ws_rl_can_mps"), ("ws_rr_mps", "ws_rr_can_mps"))

# --- 運転者の操作入力 -------------------------------------------------------
# 「無操作」を車体の運動ではなく**操作そのもの**で定義するために使う。
# データセットで揃っていないので、先に見つかったものを使い、
# 無い列は NaN のままにする (無いことを 0 と混同しない)。
GAS = ("gas_pedal_pct",)                 # アクセル開度 [%]
# アクセル開度が流れない個体がある (RAV4 TSS2 で有効なのは 39.8%)。
# GAS_RELEASED フラグなら 1,949/2,000 セグメントで取れるので、
# **窓内で値が変われば「アクセル操作があった」**として代用する。
GAS_REL = ("gas_released",)
BRAKE_P = ("brake_mc_mpa", "brake_pressure", "brake_position")   # 制動の強さ
BRAKE_ON = ("brake_pressed",)            # 制動の有無 (フラグ)
STEER_RATE = ("steer_rate_can_dps",)     # 舵角レート (CAN 直値)
STEER_TQ = ("steer_torque_driver",)      # 運転者の操舵トルク
FLAGS = ("abs_active_flag", "vsc_active_flag", "slip_warn", "precollision_active",
         "abs_fault", "tc_disabled", "decel_flag")

_S: dict = {}


def pick(ch: dict, names) -> object | None:
    for n in names:
        if n in ch and ch[n].t.size:
            return ch[n]
    return None


def win_stats(t: np.ndarray, v: np.ndarray, edges: np.ndarray, how: str) -> np.ndarray:
    """各窓の統計。**窓に標本が無ければ NaN。** 0 で埋めない。"""
    out = np.full(len(edges) - 1, np.nan)
    if t.size == 0:
        return out
    idx = np.searchsorted(edges, t, side="right") - 1
    ok = (idx >= 0) & (idx < len(out)) & np.isfinite(v)
    if not ok.any():
        return out
    i, x = idx[ok], v[ok]
    order = np.argsort(i, kind="stable")
    i, x = i[order], x[order]
    starts = np.searchsorted(i, np.arange(len(out)), side="left")
    ends = np.searchsorted(i, np.arange(len(out)), side="right")
    for k, (a, b) in enumerate(zip(starts, ends)):
        if b <= a:
            continue
        seg = x[a:b]
        if how == "max":
            out[k] = seg.max()
        elif how == "min":
            out[k] = seg.min()
        elif how == "absmax":
            out[k] = np.abs(seg).max()
        elif how == "any":
            out[k] = float((seg > 0.5).any())
        elif how == "all":
            out[k] = float((seg > 0.5).all())
    return out


def init(dataset: str, vdir: Path, platform: str, windows) -> None:
    _S["vehicles"] = load_vehicle_configs(vdir)
    _S["dataset"], _S["platform"], _S["windows"] = dataset, platform, windows
    if dataset == "ccs":
        from near_miss.io import comma_car_segments as m
    else:
        from near_miss.io import comma2k19 as m
    _S["mod"] = m


def run_drive(args) -> list[dict]:
    drive_id, refs = args
    mod = _S["mod"]
    if _S["dataset"] == "ccs":
        vehicle = next((v for v in _S["vehicles"] if _S["platform"] in v.platforms), None)
    else:
        vehicle = find_vehicle_config(refs[0].dongle_id, _S["vehicles"])
    if vehicle is None:
        return []

    out: list[dict] = []
    for ref in refs:
        try:
            sd = mod.load_segment(ref, vehicle, with_raw_can=True)
        except Exception as exc:
            out.append({"drive_id": drive_id, "segment": ref.index,
                        "status": f"load_failed:{type(exc).__name__}"})
            continue
        t0, t1 = sd.t_span
        if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
            out.append({"drive_id": drive_id, "segment": ref.index,
                        "status": "no_time_span"})
            continue
        ch = sd.channels
        sp, st, yw, ax = (pick(ch, SPEED), pick(ch, STEER), pick(ch, YAW), pick(ch, AX))
        for W in _S["windows"]:
            edges = np.arange(t0, t1 + W, W)
            if len(edges) < 2:
                continue
            n = len(edges) - 1
            d = {"win_s": W, "k": np.arange(n),
                 "t_rel": edges[:-1] - t0,
                 "dur_s": np.diff(edges).clip(max=t1 - edges[:-1])}
            d["v_max"] = win_stats(sp.t, sp.v, edges, "max") if sp else np.full(n, np.nan)
            d["v_min"] = win_stats(sp.t, sp.v, edges, "min") if sp else np.full(n, np.nan)
            d["steer_max"] = win_stats(st.t, st.v, edges, "max") if st else np.full(n, np.nan)
            d["steer_min"] = win_stats(st.t, st.v, edges, "min") if st else np.full(n, np.nan)
            d["yaw_absmax"] = win_stats(yw.t, yw.v, edges, "absmax") if yw else np.full(n, np.nan)
            d["ax_absmax"] = win_stats(ax.t, ax.v, edges, "absmax") if ax else np.full(n, np.nan)

            ay = pick(ch, AY)
            d["ay_absmax"] = win_stats(ay.t, ay.v, edges, "absmax") if ay else np.full(n, np.nan)
            gv = pick(ch, GVC)
            d["gvc_absmax"] = win_stats(gv.t, gv.v, edges, "absmax") if gv else np.full(n, np.nan)
            hi = np.full(n, -np.inf)
            lo = np.full(n, np.inf)
            for names in WHEELS:
                c_ = pick(ch, names)
                if c_ is None:
                    continue
                hi = np.fmax(hi, win_stats(c_.t, c_.v, edges, "max"))
                lo = np.fmin(lo, win_stats(c_.t, c_.v, edges, "min"))
            d["ws_spread_max"] = np.where(np.isfinite(hi) & np.isfinite(lo), hi - lo, np.nan)

            # --- 操作入力 ---
            gs_ = pick(ch, GAS)
            d["gas_max"] = win_stats(gs_.t, gs_.v, edges, "max") if gs_ else np.full(n, np.nan)
            d["gas_min"] = win_stats(gs_.t, gs_.v, edges, "min") if gs_ else np.full(n, np.nan)
            gr = pick(ch, GAS_REL)
            d["gas_rel_max"] = win_stats(gr.t, gr.v, edges, "max") if gr else np.full(n, np.nan)
            d["gas_rel_min"] = win_stats(gr.t, gr.v, edges, "min") if gr else np.full(n, np.nan)
            bp = pick(ch, BRAKE_P)
            d["brake_press_max"] = win_stats(bp.t, bp.v, edges, "max") if bp else np.full(n, np.nan)
            bo = pick(ch, BRAKE_ON)
            d["brake_on"] = win_stats(bo.t, bo.v, edges, "any") if bo else np.full(n, np.nan)
            sr = pick(ch, STEER_RATE)
            d["steer_rate_absmax"] = (win_stats(sr.t, sr.v, edges, "absmax")
                                      if sr else np.full(n, np.nan))
            tq = pick(ch, STEER_TQ)
            d["steer_tq_absmax"] = (win_stats(tq.t, tq.v, edges, "absmax")
                                    if tq else np.full(n, np.nan))
            flag = np.zeros(n)
            for f in FLAGS:
                if f in ch and ch[f].t.size:
                    z = win_stats(ch[f].t, ch[f].v, edges, "any")
                    flag = np.fmax(flag, np.nan_to_num(z))
            d["safety_flag"] = flag
            df = pd.DataFrame(d)
            df["drive_id"] = drive_id
            df["segment"] = ref.index
            df["status"] = "ok"
            out.extend(df.to_dict("records"))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--platform")
    g.add_argument("--comma2k19", type=Path)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("-w", "--workers", type=int, default=1)
    p.add_argument("--windows", type=float, nargs="+", default=list(WINDOWS))
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.platform:
        from near_miss.io import comma_car_segments as m
        refs = m.find_segments(platform=args.platform)
        dataset = "ccs"
        out = args.out or Path(f"out/window_stats_ccs_{args.platform}.csv")
    else:
        from near_miss.io import comma2k19 as m
        refs = m.find_segments(args.comma2k19)
        dataset = "c2k19"
        out = args.out or Path("out/window_stats_c2k19.csv")
    refs = sorted(refs, key=lambda r: (r.drive_id, r.index))
    if args.limit:
        refs = refs[: args.limit]
    by: dict[str, list] = defaultdict(list)
    for r in refs:
        by[r.drive_id].append(r)
    tasks = sorted(by.items())
    print(f"{len(refs)} セグメント / {len(tasks)} ドライブ  窓 {args.windows}  "
          f"worker {args.workers}")

    t0 = time.perf_counter()
    rows: list[dict] = []
    ini = (dataset, args.vehicles, args.platform or "", tuple(args.windows))
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers, initializer=init, initargs=ini) as ex:
            for i, part in enumerate(ex.map(run_drive, tasks), 1):
                rows.extend(part)
                if i % 100 == 0:
                    print(f"  {i}/{len(tasks)} ...", flush=True)
    else:
        init(*ini)
        for task in tasks:
            rows.extend(run_drive(task))

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    ok = df[df.status == "ok"] if "status" in df else df
    print(f"\n{time.perf_counter() - t0:.0f} 秒  {len(df):,} 行")
    if "win_s" in ok:
        print(ok.groupby("win_s").agg(窓数=("k", "size"),
                                      合計時間_h=("dur_s", lambda s: s.sum() / 3600)
                                      ).round(2).to_string())
    print(f"出力: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
