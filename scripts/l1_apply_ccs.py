#!/usr/bin/env python3
"""comma2k19 で固定した L1 の閾値を、そのまま commaCarSegments に当てる。

**車種ごとに閾値を調整しない。** 調整すると「その車種の分布に合わせただけ」に
なり、別の車種へ持っていけない。固定した値で通過率がどう動くかを見る。

  標準    v >= 5 m/s かつ (ax <= -2.0 m/s^2 または |ay| >= 2.0 m/s^2)
  高圧縮  v >= 5 m/s かつ (ax <= -2.5 m/s^2 または |ay| >= 2.0 m/s^2)

これに加えて、**制御・安全系フラグが立った窓は速度・加速度に関係なく通す**。
TRC 単独の旋回加速も L1 では落とさない (優先度は後段で決める)。

  無条件通過   abs_active_flag / vsc_active_flag / slip_warn (値 4) /
               precollision_active (PCS)
  数えるだけ   tc_disabled (「TC が切られている」であって作動ではない) /
               abs_fault (故障であって作動ではない)

判定量はすべて**生 CAN の 2 秒窓の max/min**。平滑化も微分も時刻同期もしない。

定常区間の除外は通過率を下げる処理ではない。「後段の CAN decode / イベント判定を
省略できる走行時間率」として別に数える (10 秒の車速レンジ < 1 m/s)。

  uv run python scripts/l1_apply_ccs.py --platform TOYOTA_RAV4_TSS2 ... -w 8 \\
      --out out/l1_ccs

出力
  summary.csv   車種 x 設定ごとの通過率と内訳
  events.csv    direct_slip の事象ごとの保持 (運動量だけ / フラグ込み)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from near_miss.config import (
    DEFAULT_VEHICLE_DIR,
    find_vehicle_config_for_platform,
    load_vehicle_configs,
)
from near_miss.io import comma_car_segments as ccs

# comma2k19 で決めた固定条件。ここは車種で変えない。
CONFIGS = {
    "標準": {"ax": -2.0, "ay": 2.0, "v_min": 5.0},
    "高圧縮": {"ax": -2.5, "ay": 2.0, "v_min": 5.0},
    "1段階厳": {"ax": -3.0, "ay": 2.5, "v_min": 5.0},
}

# 無条件通過にする制御・安全系フラグ
PASS_FLAGS = ("abs_active_flag", "vsc_active_flag", "precollision_active")
# slip_warn は多値。値 4 だけが実作動 (3 = 始動時ランプ試験、7 = 継続点灯)
SLIP_WARN_RANGE = (3.5, 4.5)
# 立ち方は数えるが通過条件には使わないもの
WATCH_FLAGS = ("tc_disabled", "abs_fault")

WINDOW_S = 2.0
STEADY_WIN_S = 10.0
STEADY_DELTA = 1.0

_S: dict = {}


def init(platform: str, vdir: Path, events: dict) -> None:
    _S["vehicle"] = find_vehicle_config_for_platform(platform, load_vehicle_configs(vdir))
    _S["platform"] = platform
    _S["events"] = events


def _win_stats(t: np.ndarray, v: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(edges) - 1
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    if t.size == 0:
        return lo, hi
    idx = np.searchsorted(edges, t, side="right") - 1
    ok = (idx >= 0) & (idx < n) & np.isfinite(v)
    if not ok.any():
        return lo, hi
    i, x = idx[ok], v[ok]
    o = np.argsort(i, kind="stable")
    i, x = i[o], x[o]
    starts = np.flatnonzero(np.r_[True, np.diff(i) != 0])
    for s, e in zip(starts, np.r_[starts[1:], i.size]):
        lo[i[s]] = x[s:e].min()
        hi[i[s]] = x[s:e].max()
    return lo, hi


def run_segment(ref) -> dict | None:
    veh = _S["vehicle"]
    try:
        sd = ccs.load_segment(ref, veh, with_raw_can=True, with_radar=False,
                              apply_calibration=False)
    except Exception:                                          # noqa: BLE001
        return None
    a, b = sd.t_span
    if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
        return None
    edges = np.arange(a, b + WINDOW_S, WINDOW_S)
    if edges.size < 2:
        return None
    n = len(edges) - 1
    ch = sd.channels

    def st(name):
        c = ch.get(name)
        return _win_stats(c.t, c.v, edges) if c is not None else (np.full(n, np.nan),) * 2

    ax_lo, _ = st("accel_x")
    ay_lo, ay_hi = st("accel_y")
    sp_lo, sp_hi = st("speed_mps")
    ay_abs = np.fmax(np.abs(ay_lo), np.abs(ay_hi))

    # 制御・安全系フラグ
    flag = np.zeros(n, dtype=bool)
    per_flag = {}
    for f in PASS_FLAGS:
        _, hi = st(f)
        m = np.where(np.isfinite(hi), hi > 0.5, False)
        per_flag[f] = int(m.sum())
        flag |= m
    _, sw_hi = st("slip_warn")
    sw = np.where(np.isfinite(sw_hi),
                  (sw_hi > SLIP_WARN_RANGE[0]) & (sw_hi <= SLIP_WARN_RANGE[1]), False)
    per_flag["slip_warn"] = int(sw.sum())
    flag |= sw
    for f in WATCH_FLAGS:
        _, hi = st(f)
        per_flag[f] = int(np.where(np.isfinite(hi), hi > 0.5, False).sum())

    gate = np.where(np.isfinite(sp_hi), sp_hi >= 5.0, False)

    out = {"platform": _S["platform"], "drive_id": ref.drive_id, "segment": ref.index,
           "n_windows": n, "seconds": float(b - a), "n_gate": int(gate.sum()),
           "n_flag": int(flag.sum()), **{f"flag_{k}": v for k, v in per_flag.items()}}

    # 定常区間 (10 秒の車速レンジ < 1 m/s)。窓 5 個ぶんを見る。
    k = max(1, int(round(STEADY_WIN_S / WINDOW_S)))
    steady = np.zeros(n, dtype=bool)
    if n >= 1:
        lo_pad = np.pad(np.where(np.isfinite(sp_lo), sp_lo, np.inf), k // 2, mode="edge")
        hi_pad = np.pad(np.where(np.isfinite(sp_hi), sp_hi, -np.inf), k // 2, mode="edge")
        for i in range(n):
            rng = hi_pad[i:i + k].max() - lo_pad[i:i + k].min()
            steady[i] = np.isfinite(rng) and rng < STEADY_DELTA
    out["n_steady"] = int(steady.sum())

    masks = {}
    for name, c in CONFIGS.items():
        kin = gate & (np.where(np.isfinite(ax_lo), ax_lo <= c["ax"], False)
                      | np.where(np.isfinite(ay_abs), ay_abs >= c["ay"], False))
        tot = kin | flag
        out[f"n_kin_{name}"] = int(kin.sum())
        out[f"n_total_{name}"] = int(tot.sum())
        out[f"n_steady_pass_{name}"] = int((tot & steady).sum())
        out[f"seg_pass_{name}"] = bool(tot.any())          # この 60 秒 segment を残すか
        masks[name] = (kin, tot)

    # direct_slip の事象の保持
    rows = []
    for ev in _S["events"].get((ref.drive_id, ref.index), []):
        t0 = a + float(ev["t_in_segment_s"])
        t1 = t0 + float(ev["duration_s"])
        sel = (edges[:-1] <= t1) & (edges[1:] >= t0)
        r = {"platform": _S["platform"], "drive_id": ref.drive_id, "segment": ref.index,
             "grade": ev["grade"], "evidence": ev["evidence"], "t_in_segment_s": ev["t_in_segment_s"]}
        for name, (kin, tot) in masks.items():
            r[f"kept_kin_{name}"] = bool((kin & sel).any())
            r[f"kept_total_{name}"] = bool((tot & sel).any())
        rows.append(r)
    out["events"] = rows
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--platform", action="append", required=True)
    p.add_argument("--direct-dir", default="out/sideslip_{platform}_v3",
                   help="direct_slip.csv の場所 ({platform} を置換)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("-w", "--workers", type=int, default=8)
    p.add_argument("--cache", type=Path, default=ccs.DEFAULT_CACHE)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=Path("out/l1_ccs"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows, all_events = [], []

    for platform in args.platform:
        d = Path(args.direct_dir.format(platform=platform)) / "direct_slip.csv"
        events: dict = defaultdict(list)
        if d.is_file() and d.stat().st_size > 1:
            for r in pd.read_csv(d).to_dict("records"):
                events[(r["drive_id"], int(r["segment"]))].append(r)
        refs = ccs.find_segments(args.cache, platform)
        if args.limit:
            refs = refs[: args.limit]
        print(f"{platform}: {len(refs)} セグメント / direct_slip {sum(len(v) for v in events.values())} 件",
              flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init,
                                 initargs=(platform, args.vehicles, dict(events))) as ex:
            for r in ex.map(run_segment, refs, chunksize=4):
                if r:
                    all_events.extend(r.pop("events"))
                    all_rows.append(r)

    df = pd.DataFrame(all_rows)
    ev = pd.DataFrame(all_events)
    df.to_csv(args.out / "segments.csv", index=False)
    ev.to_csv(args.out / "events.csv", index=False)

    pd.set_option("display.width", 240)
    rows = []
    for platform, g in df.groupby("platform"):
        nw = g.n_windows.sum()
        base = {"platform": platform, "hours": g.seconds.sum() / 3600, "n_windows": int(nw),
                "gate_rate": g.n_gate.sum() / nw, "flag_rate": g.n_flag.sum() / nw,
                "steady_rate": g.n_steady.sum() / nw}
        for f in (*PASS_FLAGS, "slip_warn", *WATCH_FLAGS):
            col = f"flag_{f}"
            if col in g:
                base[f"rate_{f}"] = g[col].sum() / nw
        base["n_segments"] = int(len(g))
        for name in CONFIGS:
            base[f"seg_rate_{name}"] = g[f"seg_pass_{name}"].mean()
            base[f"kin_{name}"] = g[f"n_kin_{name}"].sum() / nw
            base[f"total_{name}"] = g[f"n_total_{name}"].sum() / nw
            base[f"delta_{name}"] = base[f"total_{name}"] - base[f"kin_{name}"]
            base[f"pass_in_steady_{name}"] = g[f"n_steady_pass_{name}"].sum() / nw
        rows.append(base)
    s = pd.DataFrame(rows)
    s.to_csv(args.out / "summary.csv", index=False)

    print("\n■ 車種別 通過率 (窓 2 秒)")
    show = (["platform", "hours"]
            + [f"total_{n}" for n in CONFIGS] + [f"seg_rate_{n}" for n in CONFIGS]
            + ["flag_rate", "steady_rate"])
    t = s[show].copy()
    for c in show[2:]:
        t[c] = (t[c] * 100).round(2)
    t["hours"] = t.hours.round(1)
    print(t.to_string(index=False))

    print("\n■ 制御・安全系フラグの立った窓の割合 [%]")
    fcols = [c for c in s.columns if c.startswith("rate_")]
    t2 = s[["platform"] + fcols].copy()
    for c in fcols:
        t2[c] = (t2[c] * 100).round(4)
    print(t2.to_string(index=False))

    if not ev.empty:
        print("\n■ direct_slip の事象の保持率 (等級別)")
        for name in CONFIGS:
            g = ev.groupby("grade")[[f"kept_kin_{name}", f"kept_total_{name}"]].agg(["size", "sum"])
            out = pd.DataFrame({
                "件数": g[(f"kept_kin_{name}", "size")],
                "運動量のみ": g[(f"kept_kin_{name}", "sum")],
                "フラグ込み": g[(f"kept_total_{name}", "sum")]})
            out["運動量のみ %"] = (out["運動量のみ"] / out["件数"] * 100).round(0)
            out["フラグ込み %"] = (out["フラグ込み"] / out["件数"] * 100).round(0)
            print(f"\n  [{name}]")
            print(out.to_string())

    (args.out / "meta.json").write_text(json.dumps({
        "configs": CONFIGS, "window_s": WINDOW_S,
        "pass_flags": list(PASS_FLAGS) + ["slip_warn(=4)"],
        "watch_flags": list(WATCH_FLAGS),
        "steady": {"win_s": STEADY_WIN_S, "delta": STEADY_DELTA},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
