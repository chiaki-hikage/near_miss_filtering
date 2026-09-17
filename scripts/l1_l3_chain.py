#!/usr/bin/env python3
"""L1 (基本候補 = 高圧縮) を L3 の出力に当て、種別ごとの保持率と最終候補数を出す。

段の並び
  L3          既存の events.csv / candidates.csv をそのまま使う (検出条件は変えない)
  L1          生 CAN の 2 秒窓 max/min。事象の区間に通過窓が 1 つでも重なれば残す
  merge 後    candidates.csv の候補 (L3 の時点で既に merge 済み)
  優先度      configs/priority.yaml の規則

L1 は `near_miss.l1.BASE` の固定条件。**車種ごとに調整しない。**
制御・安全系フラグが立った窓は速度・加速度に関係なく無条件通過。

  uv run python scripts/l1_l3_chain.py ccs --platform TOYOTA_RAV4_TSS2 \\
      --screen out/screen_{platform}_2k_v4 --out out/l1_l3/ccs
  uv run python scripts/l1_l3_chain.py c2k19 raw_data/Chunk_1 raw_data/Chunk_2 \\
      --screen out/chunk1_v3 --screen out/chunk2_v3 \\
      --labels out/chunk1/labels.csv --labels out/chunk2_final/labels.csv \\
      --out out/l1_l3/c2k19

出力
  events.csv      L3 事象ごとの L1 通過
  candidates.csv  候補ごとの L1 / 優先度 / 最終
  by_type.csv     種別ごとの L1 前後の件数と保持率
  stages.csv      段ごとの候補数
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

from filter_priority import apply_rules
from near_miss import l1 as L1
from near_miss.config import (
    DEFAULT_VEHICLE_DIR,
    find_vehicle_config,
    find_vehicle_config_for_platform,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.io import comma2k19, comma_car_segments as ccs

_S: dict = {}


def init_ccs(platform: str, vdir: Path) -> None:
    _S["vehicle"] = find_vehicle_config_for_platform(platform, load_vehicle_configs(vdir))


def init_c2k19(vdir: Path) -> None:
    _S["vehicles"] = load_vehicle_configs(vdir)


def _mask_of(sd) -> dict | None:
    a, b = sd.t_span
    edges = L1.window_edges(float(a), float(b))
    if edges is None:
        return None
    st = L1.window_stats(sd.channels, edges)
    kin, tot = L1.window_pass(st)
    return {"t0": float(a), "n": int(tot.size), "kin": kin, "tot": tot,
            "flag": st["flag"], "seconds": float(b - a)}


def run_ccs(ref) -> tuple | None:
    try:
        sd = ccs.load_segment(ref, _S["vehicle"], with_raw_can=True, with_radar=False,
                              apply_calibration=False)
    except Exception:                                          # noqa: BLE001
        return None
    m = _mask_of(sd)
    return ((ref.drive_id, ref.index), m) if m else None


def run_c2k19(ref) -> tuple | None:
    veh = find_vehicle_config(ref.dongle_id, _S["vehicles"])
    if veh is None:
        return None
    try:
        sd = comma2k19.load_segment(ref, veh, with_raw_can=True)
    except Exception:                                          # noqa: BLE001
        return None
    m = _mask_of(sd)
    return ((ref.drive_id, ref.index), m) if m else None


def overlaps(masks: dict, drive_id: str, segment, t_in: float, dur: float,
             key: str = "tot") -> bool | None:
    """事象の区間に L1 通過窓が 1 つでも重なるか。窓が無ければ None。"""
    if not np.isfinite(t_in) or segment is None or not np.isfinite(segment):
        return None
    m = masks.get((drive_id, int(segment)))
    if m is None:
        return None
    d = dur if np.isfinite(dur) else 0.0
    i0 = int(np.floor(t_in / L1.WINDOW_S))
    i1 = int(np.ceil((t_in + d) / L1.WINDOW_S))
    i0, i1 = max(0, min(i0, i1)), min(m["n"], max(i0, i1) + 1)
    return bool(m[key][i0:i1].any()) if i1 > i0 else False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["ccs", "c2k19"])
    p.add_argument("chunks", nargs="*", type=Path)
    p.add_argument("--platform", action="append", default=[])
    p.add_argument("--screen", action="append", required=True,
                   help="events.csv / candidates.csv / segments.csv のあるディレクトリ")
    p.add_argument("--labels", action="append", type=Path, default=[])
    p.add_argument("--priority", type=Path, default=Path("configs/priority.yaml"))
    p.add_argument("-w", "--workers", type=int, default=8)
    p.add_argument("--cache", type=Path, default=ccs.DEFAULT_CACHE)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_masks_ccs(args) -> tuple[dict, list[pd.DataFrame], list[pd.DataFrame]]:
    masks, evs, cas = {}, [], []
    for platform in args.platform:
        # --screen が 1 つなら {platform} を埋めるひな形、複数なら --platform と同じ並び
        raw = (args.screen[0] if len(args.screen) == 1
               else args.screen[args.platform.index(platform)])
        d = Path(raw.format(platform=platform))
        seg = pd.read_csv(d / "segments.csv")
        want = {(r.drive_id, int(r.segment)) for r in seg.itertuples()}
        refs = [r for r in ccs.find_segments(args.cache, platform)
                if (r.drive_id, r.index) in want]
        print(f"{platform}: L3 {len(seg)} セグメント / L1 を当てる {len(refs)} セグメント",
              flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_ccs,
                                 initargs=(platform, args.vehicles)) as ex:
            for r in ex.map(run_ccs, refs, chunksize=4):
                if r:
                    masks[r[0]] = r[1]
        e = pd.read_csv(d / "events.csv", low_memory=False)
        c = pd.read_csv(d / "candidates.csv", low_memory=False)
        e["platform"], c["platform"] = platform, platform
        evs.append(e)
        cas.append(c)
    return masks, evs, cas


def load_masks_c2k19(args) -> tuple[dict, list[pd.DataFrame], list[pd.DataFrame]]:
    refs = []
    for c in args.chunks:
        refs.extend(comma2k19.find_segments(c))
    print(f"comma2k19: {len(refs)} セグメントに L1 を当てます", flush=True)
    masks = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_c2k19,
                             initargs=(args.vehicles,)) as ex:
        for r in ex.map(run_c2k19, refs, chunksize=4):
            if r:
                masks[r[0]] = r[1]
    evs, cas = [], []
    for d in args.screen:
        p = Path(d)
        evs.append(pd.read_csv(p / "events.csv", low_memory=False))
        cas.append(pd.read_csv(p / "candidates.csv", low_memory=False))
    return masks, evs, cas


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "ccs":
        masks, evs, cas = load_masks_ccs(args)
    else:
        masks, evs, cas = load_masks_c2k19(args)

    ev = pd.concat(evs, ignore_index=True)
    ca = pd.concat(cas, ignore_index=True)
    if "platform" not in ca.columns:
        ca["platform"] = "comma2k19"
        ev["platform"] = "comma2k19"
    for d in (ev, ca):
        d["segment"] = pd.to_numeric(d.segment, errors="coerce")

    hours = sum(m["seconds"] for m in masks.values()) / 3600
    n_win = sum(m["n"] for m in masks.values())
    n_pass = sum(int(m["tot"].sum()) for m in masks.values())
    print(f"\nL1 を当てた範囲: {len(masks)} セグメント / {hours:.1f} 時間 / "
          f"窓 {n_win} 個 / 通過 {100*n_pass/max(n_win,1):.2f}%")

    ev["L1"] = [overlaps(masks, r.drive_id, r.segment, r.t_in_segment_s, r.duration_s)
                for r in ev.itertuples()]
    ev["L1_kin"] = [overlaps(masks, r.drive_id, r.segment, r.t_in_segment_s, r.duration_s, "kin")
                    for r in ev.itertuples()]
    ca["L1"] = [overlaps(masks, r.drive_id, r.segment, r.t_in_segment_s, r.duration_s)
                for r in ca.itertuples()]

    # L1 を当てられなかった事象 (セグメントが読めなかった等) は落とさず別に数える
    n_unk = int(ev.L1.isna().sum())
    if n_unk:
        print(f"L1 を判定できなかった事象 {n_unk} 件 (セグメントが読めなかった)")
    ev["L1"] = ev.L1.fillna(False).astype(bool)
    ev["L1_kin"] = ev.L1_kin.fillna(False).astype(bool)
    ca["L1"] = ca.L1.fillna(False).astype(bool)

    # --- 種別ごとの L1 前後 --------------------------------------------------
    by = ev.groupby("event_type").agg(L3=("L1", "size"), L1=("L1", "sum"),
                                      L1_kin=("L1_kin", "sum")).reset_index()
    by["retain_pct"] = (100 * by.L1 / by.L3).round(1)
    by["retain_kin_pct"] = (100 * by.L1_kin / by.L3).round(1)
    by = by.sort_values("L3", ascending=False)
    by.to_csv(args.out / "by_type.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n■ L3 イベント種別ごとの L1 前後 (事象単位)")
    t = by.rename(columns={"event_type": "種別", "L3": "L1前", "L1": "L1後",
                           "L1_kin": "運動量のみ", "retain_pct": "保持率%",
                           "retain_kin_pct": "運動量のみ%"})
    print(t.to_string(index=False))
    print(f"\n  合計 {len(ev)} -> {int(ev.L1.sum())} 件 "
          f"({100*ev.L1.mean():.1f}%)")

    # --- 候補 (merge 後) と優先度 --------------------------------------------
    ca["sev_ok"] = pd.to_numeric(ca.severity, errors="coerce").fillna(0) > 0
    pri, reason = apply_rules(ca, load_yaml(args.priority))
    ca["priority"] = pri
    ca["priority_reason"] = reason
    ca["final"] = ca.L1 & ca.sev_ok & ca.priority
    ca["final_noL1"] = ca.sev_ok & ca.priority

    # 人手ラベルとの突き合わせ (区間の重なり)
    risky = None
    if args.labels:
        lab = pd.concat([pd.read_csv(p) for p in args.labels], ignore_index=True)
        lab["risky"] = lab.risky.astype(bool)
        risky = []
        for x in ca.itertuples():
            m = ((lab.drive_id == x.drive_id) & (lab.t_end >= x.t_start)
                 & (lab.t_start <= x.t_end))
            risky.append(bool(lab[m].risky.any()) if m.any() else None)
        ca["risky"] = risky
        ca["labeled"] = [v is not None for v in risky]

    stages = [("L3 候補 (merge 後)", pd.Series(True, index=ca.index)),
              ("  + severity > 0", ca.sev_ok),
              ("  + 優先度 (L1 なし)", ca.final_noL1),
              ("  + L1 = 最終", ca.final)]
    rows = []
    print("\n■ 候補の段ごとの件数")
    head = f"{'段':<24}{'候補':>7}"
    if risky is not None:
        head += f"{'ラベル':>8}{'risky':>7}{'精度':>7}"
    print(head)
    print("-" * len(head.encode("utf-8")) if False else "-" * 56)
    for name, m in stages:
        line = f"{name:<24}{int(m.sum()):>7}"
        if risky is not None:
            sub = ca[m & ca.labeled]
            nl, nr = len(sub), int(sub.risky.sum())
            line += f"{nl:>8}{nr:>7}{(100*nr/nl if nl else 0):>6.0f}%"
        rows.append({"stage": name.strip(), "n": int(m.sum()),
                     "n_labeled": int((m & ca.labeled).sum()) if risky is not None else None,
                     "n_risky": int(ca[m & ca.labeled].risky.sum()) if risky is not None else None})
        print(line)
    pd.DataFrame(rows).to_csv(args.out / "stages.csv", index=False)

    if risky is not None:
        is_risky = ca.risky.map(lambda v: bool(v) if v is not None else False).astype(bool)
        lost = ca[ca.final_noL1 & ~ca.L1 & ca.labeled & is_risky]
        print(f"\n  L1 で落ちた human positive の最終候補: {len(lost)} 件")
        if len(lost):
            print(lost[["drive_id", "segment", "t_in_segment_s", "event_types"]]
                  .to_string(index=False))

    # 車種別
    if ca.platform.nunique() > 1:
        print("\n■ 車種別")
        g = ca.groupby("platform").agg(
            L3候補=("L1", "size"), L1通過=("L1", "sum"),
            最終_L1なし=("final_noL1", "sum"), 最終=("final", "sum"))
        g["L1保持率%"] = (100 * g.L1通過 / g.L3候補).round(1)
        g["最終保持率%"] = (100 * g.最終 / g.最終_L1なし.replace(0, np.nan)).round(1)
        print(g.to_string())
        g.to_csv(args.out / "by_platform.csv")

    ev.to_csv(args.out / "events.csv", index=False)
    keep = [c for c in ("platform", "drive_id", "segment", "t_start", "t_end",
                        "t_in_segment_s", "duration_s", "severity", "n_events",
                        "event_types", "lead_distance_m_min", "thw_s_min", "ttc_s_min",
                        "ax_mps2_min", "L1", "sev_ok", "priority", "priority_reason",
                        "final_noL1", "final", "risky", "labeled") if c in ca.columns]
    ca[keep].to_csv(args.out / "candidates.csv", index=False)

    (args.out / "meta.json").write_text(json.dumps({
        "dataset": args.dataset, "l1": {"name": L1.BASE_NAME, **L1.BASE},
        "window_s": L1.WINDOW_S, "pass_flags": list(L1.PASS_FLAGS) + ["slip_warn(=4)"],
        "hours": hours, "n_segments_l1": len(masks), "n_windows": n_win,
        "l1_pass_rate": n_pass / max(n_win, 1),
        "n_events": int(len(ev)), "n_events_l1": int(ev.L1.sum()),
        "n_candidates": int(len(ca)), "n_final_noL1": int(ca.final_noL1.sum()),
        "n_final": int(ca.final.sum()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
