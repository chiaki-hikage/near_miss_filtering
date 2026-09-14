#!/usr/bin/env python3
"""スリップの**直接証拠**を生 CAN のフラグだけで数え上げる。

横滑りの 2 段フィルタ (screen_sideslip.py) はモデル推定の beta を使うため
`min_speed_mps` (既定 10 m/s) の適用下限があり、**低速の駆動輪空転や
TRC/VSC 介入をまるごと取り逃がす**。この script はモデルを一切通さず、
車両自身が「滑った」と判断した瞬間だけを拾う。

  slip_warn            滑り警告灯 (0x3B7 SLP_WL)。TRC/VSC の介入で立つ
  abs_active_flag      ABS 作動 (0x226 ABSACT)
  vsc_active_flag      VSC 作動 (0x226 VSCACT)
  tc_disabled          トラクションコントロール解除
  ws_spread            4 輪速の最大 - 最小。ロック・空転の直接の現れ

補間も微分もしない。受信時刻のままフラグの立ち上がり / 立ち下がりを取る。
`ws_spread` だけは 4 本を共通時刻に最近傍で寄せてから差を取る。

  uv run python scripts/scan_slip_flags.py --platform TOYOTA_RAV4_TSS2 -w 6
  uv run python scripts/scan_slip_flags.py --comma2k19 raw_data/Chunk_1 -w 6

出力 (--out 配下)
  runs.csv    フラグが立った区間。1 行 1 区間
  counts.json 全体の件数と時間
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from near_miss.config import (
    DEFAULT_VEHICLE_DIR,
    find_vehicle_config,
    load_vehicle_configs,
)

# 車両自身の判断。閾値を置く余地がないので、立ったら全部拾う。
DIRECT_FLAGS = ("slip_warn", "abs_active_flag", "vsc_active_flag", "tc_disabled")

SPEED = ("speed_mps", "ws_fl_mps", "ws_fl_can_mps")
WHEELS = (("ws_fl_mps", "ws_fl_can_mps"), ("ws_fr_mps", "ws_fr_can_mps"),
          ("ws_rl_mps", "ws_rl_can_mps"), ("ws_rr_mps", "ws_rr_can_mps"))

_S: dict = {}


def pick(ch: dict, names):
    for n in names:
        c = ch.get(n)
        if c is not None and getattr(c, "t", None) is not None and c.t.size:
            return c
    return None


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """True が連続する区間の [開始, 終了) を返す。"""
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


def at(t_src: np.ndarray, v_src: np.ndarray, t: float) -> float:
    """最近傍の値。補間はしない (フラグと生値を混ぜて内挿しない)。"""
    if t_src.size == 0:
        return float("nan")
    return float(v_src[int(np.abs(t_src - t).argmin())])


def init(dataset: str, vdir: Path, platform: str, ws_min: float) -> None:
    _S["vehicles"] = load_vehicle_configs(vdir)
    _S["dataset"], _S["platform"], _S["ws_min"] = dataset, platform, ws_min
    if dataset == "ccs":
        from near_miss.io import comma_car_segments as m
    else:
        from near_miss.io import comma2k19 as m
    _S["mod"] = m


def run_drive(args) -> tuple[list[dict], dict]:
    drive_id, refs = args
    mod = _S["mod"]
    if _S["dataset"] == "ccs":
        vehicle = next((v for v in _S["vehicles"] if _S["platform"] in v.platforms), None)
    else:
        vehicle = find_vehicle_config(refs[0].dongle_id, _S["vehicles"])
    if vehicle is None:
        return [], {}

    out: list[dict] = []
    tally: dict = defaultdict(float)
    for ref in refs:
        try:
            sd = mod.load_segment(ref, vehicle, with_raw_can=True)
        except Exception:
            tally["n_load_failed"] += 1
            continue
        t0, t1 = sd.t_span
        if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
            continue
        tally["n_segments"] += 1
        tally["seconds"] += t1 - t0
        ch = sd.channels
        sp = pick(ch, SPEED)
        sp_t = sp.t if sp else np.empty(0)
        sp_v = sp.v if sp else np.empty(0)

        found: list[tuple[str, float, float, float]] = []   # 種別, 開始, 終了, ピーク

        for f in DIRECT_FLAGS:
            c = ch.get(f)
            if c is None or not c.t.size:
                tally[f"missing:{f}"] += 1
                continue
            tally[f"present:{f}"] += 1
            for i, j in runs_of(c.v > 0.5):
                found.append((f, float(c.t[i]), float(c.t[j - 1]), float(c.v[i:j].max())))

        # 4 輪速の開き。共通時刻 (最初に見つかった輪速の時刻) に最近傍で寄せる。
        cs = [pick(ch, names) for names in WHEELS]
        if all(c is not None for c in cs):
            tally["present:ws_spread"] += 1
            base = cs[0].t
            stack = np.vstack([np.interp(base, c.t, c.v) for c in cs])
            spread = stack.max(axis=0) - stack.min(axis=0)
            for i, j in runs_of(spread > _S["ws_min"]):
                found.append(("ws_spread", float(base[i]), float(base[j - 1]),
                              float(spread[i:j].max())))

        for kind, ta, tb, peak in found:
            out.append({
                "drive_id": drive_id, "segment": ref.index,
                "kind": kind,
                "t_in_segment_s": round(ta - t0, 2),
                "duration_s": round(tb - ta, 2),
                "peak": round(peak, 3),
                "v_at_start_mps": round(at(sp_t, sp_v, ta), 2),
                "v_max_in_run_mps": round(
                    float(np.nanmax(sp_v[(sp_t >= ta) & (sp_t <= tb)]))
                    if sp_t.size and ((sp_t >= ta) & (sp_t <= tb)).any() else float("nan"), 2),
            })
            tally[f"runs:{kind}"] += 1
    return out, dict(tally)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--platform", help="commaCarSegments の車種キー")
    g.add_argument("--comma2k19", type=Path, help="comma2k19 のチャンク")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("-w", "--workers", type=int, default=1)
    p.add_argument("--ws-min", type=float, default=1.0,
                   help="輪速の開きをスリップとみなす下限 [m/s] (既定 1.0)")
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=Path("out/slip_flags"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.platform:
        from near_miss.io import comma_car_segments as ccs
        refs = ccs.find_segments(ccs.DEFAULT_CACHE, args.platform)
        dataset, label = "ccs", args.platform
    else:
        from near_miss.io import comma2k19 as c2k
        refs = c2k.find_segments(args.comma2k19)
        dataset, label = "c2k19", str(args.comma2k19)
    if args.limit:
        refs = refs[: args.limit]

    by_drive: dict[str, list] = defaultdict(list)
    for r in refs:
        by_drive[r.drive_id].append(r)
    tasks = sorted(by_drive.items())
    print(f"対象 : {label}  {len(refs)} セグメント / {len(tasks)} ドライブ")
    print(f"直接フラグ: {', '.join(DIRECT_FLAGS)}  ＋ 輪速の開き > {args.ws_min} m/s")

    t0 = time.perf_counter()
    rows: list[dict] = []
    tally: dict = defaultdict(float)
    ini = (dataset, args.vehicles, args.platform or "", args.ws_min)
    if args.workers and args.workers != 1:
        n = args.workers if args.workers > 0 else None
        with ProcessPoolExecutor(max_workers=n, initializer=init, initargs=ini) as ex:
            for r, t in ex.map(run_drive, tasks):
                rows.extend(r)
                for k, v in t.items():
                    tally[k] += v
    else:
        init(*ini)
        for task in tasks:
            r, t = run_drive(task)
            rows.extend(r)
            for k, v in t.items():
                tally[k] += v
    elapsed = time.perf_counter() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["kind", "duration_s"], ascending=[True, False])
    df.to_csv(args.out / "runs.csv", index=False)

    hours = tally.get("seconds", 0.0) / 3600.0
    print(f"\n走査 {int(tally.get('n_segments', 0))} セグメント / {hours:.2f} 時間"
          f"  ({elapsed / 60:.1f} 分)")
    print("\n信号の有無 (セグメント数)")
    for f in (*DIRECT_FLAGS, "ws_spread"):
        print(f"  {f:20s} あり {int(tally.get(f'present:{f}', 0)):5d}"
              f" / なし {int(tally.get(f'missing:{f}', 0)):5d}")
    print("\n立った区間")
    if df.empty:
        print("  なし")
    else:
        for kind, g in df.groupby("kind"):
            print(f"  {kind:20s} {len(g):5d} 区間  "
                  f"のべ {g.duration_s.sum():7.1f} 秒  "
                  f"最長 {g.duration_s.max():5.2f} 秒  "
                  f"速度中央値 {g.v_at_start_mps.median():5.1f} m/s")

    meta = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset, "label": label,
        "n_segments": int(tally.get("n_segments", 0)), "hours": round(hours, 3),
        "ws_min_mps": args.ws_min, "elapsed_min": round(elapsed / 60, 2),
        "counts": {k: (int(v) if float(v).is_integer() else round(v, 2))
                   for k, v in sorted(tally.items())},
    }
    (args.out / "counts.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
