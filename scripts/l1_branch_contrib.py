#!/usr/bin/env python3
"""L1 の**枝ごとの寄与**を測る。閾値は再最適化せず、採用中の丸い固定値を使う。

測るもの (枝ごと)
  単独通過率        その枝だけを使ったときの 2 秒窓の通過時間率
  単独削減率        1 - 単独通過率 (窓と 60 秒 segment の両方)
  L1 対象 risky     その枝だけで拾える人手 risky の件数 (comma2k19 のみ)
  他枝との重複率    その枝が通す窓のうち、他の枝も通す窓の割合
  独自追加通過率    最終 OR 条件に対する、その枝だけの寄与
                    採用枝   = 外したときの通過率の減少分
                    非採用枝 = 足したときの通過率の増分

ablation
  採用構成 (高圧縮) と全枝構成のそれぞれから枝を 1 本ずつ外し、
  通過率・segment 率・risky 保持数・等級別保持率がどう動くかを見る。

comma2k19 の車両 (RAV4 2016) には ABS / VSC / PCS / slip_warn のチャネルが無い。
制御フラグ枝は commaCarSegments でしか測れない。

  uv run python scripts/l1_branch_contrib.py c2k19 raw_data/Chunk_1 raw_data/Chunk_2 \\
      --labels out/chunk1/labels.csv --labels out/chunk2_final/labels.csv \\
      --out out/l1_branch/c2k19
  uv run python scripts/l1_branch_contrib.py ccs --platform TOYOTA_RAV4_TSS2 ... \\
      --out out/l1_branch/ccs

出力
  branches.csv  枝ごとの寄与
  ablation.csv  枝を 1 本ずつ外した結果
  grades.csv    等級別の保持率 (ccs のみ)
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
    find_vehicle_config,
    find_vehicle_config_for_platform,
    load_vehicle_configs,
)
from near_miss.io import comma2k19, comma_car_segments as ccs

WINDOW_S = 2.0

# 採用中の丸い固定値。ここは再最適化しない。
#   ax_decel / ay / v_min  高圧縮条件 (docs/l1_filter.md §3)
#   ax_accel / yaw / steer 採用していないが、比較用に §3 と同じ丸い値を当てる
THR = {
    "ax_decel": -2.5,   # ax_min <= これ
    "ax_accel": 3.0,    # ax_max >= これ
    "ay": 2.0,          # max|ay| >= これ
    "yaw": 15.0,        # max|yaw| >= これ [deg/s]
    "steer": 50.0,      # max|steer| >= これ [deg]
    "v_min": 5.0,       # speed_max >= これ
}

# OR 枝。gate (速度) は枝ではなく運動量枝に掛ける条件なので別扱い。
KIN_BRANCHES = ("ax_decel", "ax_accel", "ay", "yaw", "steer")
ALL_BRANCHES = KIN_BRANCHES + ("flags",)
# 採用中の最終 OR 条件 (高圧縮)
FINAL_KIN = ("ax_decel", "ay")

BIT = {b: 1 << i for i, b in enumerate(ALL_BRANCHES + ("gate",))}

PASS_FLAGS = ("abs_active_flag", "vsc_active_flag", "precollision_active")
SLIP_WARN_RANGE = (3.5, 4.5)

# 人手 risky を「自車反応あり (L1 対象)」とみなす基準
REACT_AX, REACT_AY, REACT_STEER_RANGE = 2.0, 2.0, 30.0

_S: dict = {}


def init_c2k19(vdir: Path, labels: dict) -> None:
    _S["vehicles"] = load_vehicle_configs(vdir)
    _S["labels"] = labels


def init_ccs(platform: str, vdir: Path, events: dict) -> None:
    _S["vehicle"] = find_vehicle_config_for_platform(platform, load_vehicle_configs(vdir))
    _S["platform"] = platform
    _S["events"] = events


def _win_minmax(t: np.ndarray, v: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """窓ごとの min / max。チャネルは自分のタイムスタンプだけで振り分ける。"""
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


def _codes(sd, edges: np.ndarray) -> tuple[np.ndarray, set[str]]:
    """窓ごとの枝の成否をビットで詰めた配列と、実在したチャネル名。"""
    n = len(edges) - 1
    ch = sd.channels
    seen = set()

    def st(name):
        c = ch.get(name)
        if c is None:
            return np.full(n, np.nan), np.full(n, np.nan)
        seen.add(name)
        return _win_minmax(c.t, c.v, edges)

    ax_lo, ax_hi = st("accel_x")
    ay_lo, ay_hi = st("accel_y")
    yw_lo, yw_hi = st("yaw_rate")
    sw_lo, sw_hi = st("steer_deg")
    sp_lo, sp_hi = st("speed_mps")

    def absmax(lo, hi):
        return np.fmax(np.abs(lo), np.abs(hi))

    def ge(x, th):
        return np.where(np.isfinite(x), x >= th, False)

    def le(x, th):
        return np.where(np.isfinite(x), x <= th, False)

    hit = {
        "ax_decel": le(ax_lo, THR["ax_decel"]),
        "ax_accel": ge(ax_hi, THR["ax_accel"]),
        "ay": ge(absmax(ay_lo, ay_hi), THR["ay"]),
        "yaw": ge(absmax(yw_lo, yw_hi), THR["yaw"]),
        "steer": ge(absmax(sw_lo, sw_hi), THR["steer"]),
        "gate": ge(sp_hi, THR["v_min"]),
    }

    flag = np.zeros(n, dtype=bool)
    for f in PASS_FLAGS:
        _, hi = st(f)
        flag |= np.where(np.isfinite(hi), hi > 0.5, False)
    _, s_hi = st("slip_warn")
    flag |= np.where(np.isfinite(s_hi),
                     (s_hi > SLIP_WARN_RANGE[0]) & (s_hi <= SLIP_WARN_RANGE[1]), False)
    hit["flags"] = flag

    code = np.zeros(n, dtype=np.uint8)
    for b, m in hit.items():
        code |= (m.astype(np.uint8) * BIT[b])
    return code, seen


def _edges(sd):
    a, b = sd.t_span
    if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
        return None, None, None
    e = np.arange(a, b + WINDOW_S, WINDOW_S)
    return (a, b, e) if e.size >= 2 else (None, None, None)


def run_c2k19(ref) -> dict | None:
    veh = find_vehicle_config(ref.dongle_id, _S["vehicles"])
    if veh is None:
        return None
    try:
        sd = comma2k19.load_segment(ref, veh, with_raw_can=True)
    except Exception:                                          # noqa: BLE001
        return None
    a, b, edges = _edges(sd)
    if edges is None:
        return None
    code, seen = _codes(sd, edges)
    out = {"key": (ref.drive_id, ref.index), "code": code, "seen": seen,
           "seconds": float(b - a), "labels": []}

    # ラベル区間の中の自車の動き。生サンプルの max/min/range だけ。
    for lab in _S["labels"].get((ref.drive_id, ref.index), []):
        t0, t1 = float(lab["t_start"]), float(lab["t_end"])
        m = {"ax_min": np.nan, "ax_max": np.nan, "ay_absmax": np.nan, "steer_range": np.nan}
        for name, keys in (("accel_x", ("ax_min", "ax_max")), ("accel_y", ("ay_absmax",)),
                           ("steer_deg", ("steer_range",))):
            c = sd.channels.get(name)
            if c is None:
                continue
            sel = (c.t >= t0) & (c.t <= t1) & np.isfinite(c.v)
            if not sel.any():
                continue
            x = c.v[sel]
            if name == "accel_x":
                m["ax_min"], m["ax_max"] = float(x.min()), float(x.max())
            elif name == "steer_deg":
                m["steer_range"] = float(x.max() - x.min())
            else:
                m[keys[0]] = float(np.abs(x).max())
        i0 = int(np.floor((t0 - a) / WINDOW_S))
        i1 = int(np.ceil((t1 - a) / WINDOW_S))
        out["labels"].append({**lab, **m, "i0": max(0, min(i0, i1)),
                              "i1": min(len(code), max(i0, i1) + 1)})
    return out


def run_ccs(ref) -> dict | None:
    try:
        sd = ccs.load_segment(ref, _S["vehicle"], with_raw_can=True, with_radar=False,
                              apply_calibration=False)
    except Exception:                                          # noqa: BLE001
        return None
    a, b, edges = _edges(sd)
    if edges is None:
        return None
    code, seen = _codes(sd, edges)
    out = {"key": (ref.drive_id, ref.index), "code": code, "seen": seen,
           "seconds": float(b - a), "platform": _S["platform"], "events": []}
    for ev in _S["events"].get((ref.drive_id, ref.index), []):
        t0 = a + float(ev["t_in_segment_s"])
        t1 = t0 + float(ev["duration_s"])
        i0 = int(np.floor((t0 - a) / WINDOW_S))
        i1 = int(np.ceil((t1 - a) / WINDOW_S))
        out["events"].append({"grade": ev["grade"], "evidence": ev["evidence"],
                              "i0": max(0, min(i0, i1)), "i1": min(len(code), max(i0, i1) + 1)})
    return out


# --- 集計 -------------------------------------------------------------------

class Table:
    """全走行の窓を 1 本に連結し、枝のマスクを取り出せるようにしたもの。"""

    def __init__(self, segs: list[dict]):
        self.code = np.concatenate([s["code"] for s in segs])
        self.n = self.code.size
        self.seconds = sum(s["seconds"] for s in segs)
        self.seg_id = np.concatenate([np.full(s["code"].size, i, dtype=np.int32)
                                      for i, s in enumerate(segs)])
        self.n_segments = len(segs)
        self.offset = np.cumsum([0] + [s["code"].size for s in segs])
        self.seen: set[str] = set()
        for s in segs:
            self.seen |= s["seen"]

    def mask(self, branch: str) -> np.ndarray:
        return (self.code & BIT[branch]) != 0

    def rate(self, m: np.ndarray) -> float:
        return float(m.sum()) / self.n

    def seg_rate(self, m: np.ndarray) -> float:
        if not m.any():
            return 0.0
        return float(np.unique(self.seg_id[m]).size) / self.n_segments

    def spans(self, segs: list[dict], key: str) -> list[tuple[int, int]]:
        out = []
        for i, s in enumerate(segs):
            for r in s.get(key, []):
                out.append((self.offset[i] + r["i0"], self.offset[i] + r["i1"]))
        return out


def kept(spans: list[tuple[int, int]], m: np.ndarray) -> np.ndarray:
    c = np.concatenate([[0], np.cumsum(m.astype(np.int64))])
    return np.array([(b > a) and (c[min(b, m.size)] - c[a]) > 0 for a, b in spans], dtype=bool)


def branch_mask(tb: Table, branch: str, gate: np.ndarray) -> np.ndarray:
    """枝の通過窓。運動量枝には速度ゲートを掛ける。フラグ枝は無条件。"""
    m = tb.mask(branch)
    return m if branch == "flags" else (m & gate)


def union(tb: Table, branches, gate: np.ndarray) -> np.ndarray:
    out = np.zeros(tb.n, dtype=bool)
    for b in branches:
        out |= branch_mask(tb, b, gate)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["c2k19", "ccs"])
    p.add_argument("chunks", nargs="*", type=Path, help="c2k19 のとき Chunk ディレクトリ")
    p.add_argument("--labels", action="append", type=Path, default=[])
    p.add_argument("--platform", action="append", default=[])
    p.add_argument("--direct-dir", default="out/sideslip_{platform}_v3")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("-w", "--workers", type=int, default=8)
    p.add_argument("--cache", type=Path, default=ccs.DEFAULT_CACHE)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_c2k19(args) -> tuple[list[dict], pd.DataFrame]:
    labels = pd.concat([pd.read_csv(p) for p in args.labels], ignore_index=True)
    labels["risky"] = labels.risky.astype(bool)
    by_seg: dict = defaultdict(list)
    for i, r in enumerate(labels.to_dict("records")):
        by_seg[(r["drive_id"], int(r["segment"]))].append({**r, "_row": i})
    refs = []
    for c in args.chunks:
        refs.extend(comma2k19.find_segments(c))
    if args.limit:
        refs = refs[: args.limit]
    print(f"comma2k19: {len(refs)} セグメント / ラベル {len(labels)} 件", flush=True)
    segs = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_c2k19,
                             initargs=(args.vehicles, dict(by_seg))) as ex:
        for s in ex.map(run_c2k19, refs, chunksize=4):
            if s:
                segs.append(s)
    return segs, labels


def load_ccs(args) -> list[dict]:
    segs = []
    for platform in args.platform:
        d = Path(args.direct_dir.format(platform=platform)) / "direct_slip.csv"
        events: dict = defaultdict(list)
        if d.is_file() and d.stat().st_size > 1:
            for r in pd.read_csv(d).to_dict("records"):
                events[(r["drive_id"], int(r["segment"]))].append(r)
        refs = ccs.find_segments(args.cache, platform)
        if args.limit:
            refs = refs[: args.limit]
        print(f"{platform}: {len(refs)} セグメント / direct_slip "
              f"{sum(len(v) for v in events.values())} 件", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_ccs,
                                 initargs=(platform, args.vehicles, dict(events))) as ex:
            for s in ex.map(run_ccs, refs, chunksize=4):
                if s:
                    segs.append(s)
    return segs


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    risky_spans: list[tuple[int, int]] = []
    ev_spans: list[tuple[int, int]] = []
    ev_grade: list[str] = []

    if args.dataset == "c2k19":
        segs, labels = load_c2k19(args)
        tb = Table(segs)
        rows = [{**r, **{k: r.get(k) for k in ()}} for s in segs for r in s["labels"]]
        lab = pd.DataFrame(rows)
        lab["l1_target"] = ((lab.ax_min <= -REACT_AX) | (lab.ax_max >= REACT_AX)
                            | (lab.ay_absmax >= REACT_AY)
                            | (lab.steer_range >= REACT_STEER_RANGE)).fillna(False)
        tgt = lab[lab.risky.astype(bool) & lab.l1_target]
        base = {i: tb.offset[i] for i in range(len(segs))}
        pos = {s["key"]: base[i] for i, s in enumerate(segs)}
        risky_spans = [(pos[(r.drive_id, int(r.segment))] + r.i0,
                        pos[(r.drive_id, int(r.segment))] + r.i1) for r in tgt.itertuples()]
        print(f"L1 対象 risky {len(risky_spans)} 件 / 人手 risky "
              f"{int(lab.risky.astype(bool).sum())} 件")
    else:
        segs = load_ccs(args)
        tb = Table(segs)
        ev_spans = tb.spans(segs, "events")
        ev_grade = [e["grade"] for s in segs for e in s["events"]]

    gate = tb.mask("gate")
    have = {b: True for b in ALL_BRANCHES}
    if "slip_warn" not in tb.seen and not any(f in tb.seen for f in PASS_FLAGS):
        have["flags"] = False
    for b, chn in (("yaw", "yaw_rate"), ("steer", "steer_deg")):
        have[b] = chn in tb.seen

    print(f"\n{tb.n_segments} セグメント / {tb.seconds/3600:.2f} 時間 / 窓 {tb.n} 個")
    missing = [b for b, v in have.items() if not v]
    if missing:
        print(f"チャネルが無く測れない枝: {', '.join(missing)}")

    avail = [b for b in ALL_BRANCHES if have[b]]
    final = union(tb, [b for b in FINAL_KIN], gate) | (
        branch_mask(tb, "flags", gate) if have["flags"] else False)
    r_final, s_final = tb.rate(final), tb.seg_rate(final)

    # --- 枝ごとの寄与 --------------------------------------------------------
    rows = []
    for b in avail:
        mb = branch_mask(tb, b, gate)
        raw = tb.mask(b)
        others = union(tb, [x for x in avail if x != b], gate)
        ov = float((mb & others).sum()) / max(int(mb.sum()), 1)
        in_final = b in FINAL_KIN or b == "flags"
        if in_final:
            wo = union(tb, [x for x in FINAL_KIN if x != b], gate)
            if b != "flags" and have["flags"]:
                wo = wo | branch_mask(tb, "flags", gate)
            marg = r_final - tb.rate(wo)
        else:
            marg = tb.rate(final | mb) - r_final
        rows.append({
            "branch": b, "threshold": THR.get(b, "ABS/VSC/PCS/slip_warn=4"),
            "in_final": in_final,
            "solo_rate_raw": tb.rate(raw), "solo_rate": tb.rate(mb),
            "solo_cut_win": 1 - tb.rate(mb),
            "solo_seg_rate": tb.seg_rate(mb), "solo_cut_seg": 1 - tb.seg_rate(mb),
            "overlap": ov, "exclusive_rate": tb.rate(mb) * (1 - ov),
            "marginal_to_final": marg,
            "risky_kept": int(kept(risky_spans, mb).sum()) if risky_spans else None,
        })
    # 速度ゲートの行 (枝ではなく条件)
    rows.append({
        "branch": "speed_gate", "threshold": THR["v_min"], "in_final": True,
        "solo_rate_raw": tb.rate(gate), "solo_rate": tb.rate(gate),
        "solo_cut_win": 1 - tb.rate(gate),
        "solo_seg_rate": tb.seg_rate(gate), "solo_cut_seg": 1 - tb.seg_rate(gate),
        "overlap": None, "exclusive_rate": None,
        "marginal_to_final": r_final - tb.rate(
            union(tb, FINAL_KIN, np.ones(tb.n, dtype=bool))
            | (branch_mask(tb, "flags", gate) if have["flags"] else False)),
        "risky_kept": int(kept(risky_spans, gate).sum()) if risky_spans else None,
    })
    bd = pd.DataFrame(rows)
    bd.to_csv(args.out / "branches.csv", index=False)

    pd.set_option("display.width", 260)
    print(f"\n■ 枝ごとの寄与 (最終 OR = 高圧縮: 通過 {r_final:.2%} / segment {s_final:.1%})")
    t = bd.copy()
    for c in ("solo_rate_raw", "solo_rate", "solo_cut_win", "solo_seg_rate", "solo_cut_seg",
              "overlap", "exclusive_rate", "marginal_to_final"):
        t[c] = t[c].map(lambda x: "—" if x is None or pd.isna(x) else f"{x*100:.2f}%")
    t["risky_kept"] = t.risky_kept.map(lambda x: "—" if pd.isna(x) else f"{int(x)}/{len(risky_spans)}")
    t.columns = ["枝", "閾値", "採用", "単独(gate無)", "単独通過", "単独削減(窓)",
                 "単独seg率", "単独削減(seg)", "他枝と重複", "独自分", "最終への寄与", "risky"]
    print(t.to_string(index=False))

    # --- ablation ------------------------------------------------------------
    ab_rows = []

    def record(setup: str, name: str, m: np.ndarray, ref_rate: float, ref_kept: int | None):
        r, sr = tb.rate(m), tb.seg_rate(m)
        k = int(kept(risky_spans, m).sum()) if risky_spans else None
        row = {"setup": setup, "removed": name, "rate": r, "seg_rate": sr,
               "d_rate": r - ref_rate, "risky_kept": k,
               "d_risky": None if k is None or ref_kept is None else k - ref_kept}
        if ev_spans:
            kv = kept(ev_spans, m)
            g = pd.Series(ev_grade)
            for grade in sorted(g.unique()):
                sel = (g == grade).to_numpy()
                row[f"keep_{grade}"] = float(kv[sel].mean())
        ab_rows.append(row)
        return row

    print("\n■ ablation — 採用構成 (高圧縮) から 1 本ずつ外す")
    fin_branches = list(FINAL_KIN) + (["flags"] if have["flags"] else [])
    base_kept = int(kept(risky_spans, final).sum()) if risky_spans else None
    record("採用構成", "(なし)", final, r_final, base_kept)
    for b in fin_branches:
        m = union(tb, [x for x in fin_branches if x != b], gate)
        record("採用構成", b, m, r_final, base_kept)
    m_nogate = union(tb, FINAL_KIN, np.ones(tb.n, dtype=bool)) | (
        branch_mask(tb, "flags", gate) if have["flags"] else False)
    record("採用構成", "speed_gate", m_nogate, r_final, base_kept)

    print("\n■ ablation — 全枝構成から 1 本ずつ外す")
    allm = union(tb, avail, gate)
    r_all = tb.rate(allm)
    all_kept = int(kept(risky_spans, allm).sum()) if risky_spans else None
    record("全枝構成", "(なし)", allm, r_all, all_kept)
    for b in avail:
        record("全枝構成", b, union(tb, [x for x in avail if x != b], gate), r_all, all_kept)
    record("全枝構成", "speed_gate",
           union(tb, avail, np.ones(tb.n, dtype=bool)), r_all, all_kept)

    ab = pd.DataFrame(ab_rows)
    ab.to_csv(args.out / "ablation.csv", index=False)
    for setup, g in ab.groupby("setup", sort=False):
        print(f"\n  [{setup}]")
        t = g.drop(columns="setup").copy()
        for c in ("rate", "seg_rate", "d_rate"):
            t[c] = t[c].map(lambda x: f"{x*100:+.2f}%" if c == "d_rate" else f"{x*100:.2f}%")
        for c in [c for c in t.columns if c.startswith("keep_")]:
            t[c] = t[c].map(lambda x: "—" if pd.isna(x) else f"{x*100:.0f}%")
        t["risky_kept"] = t.risky_kept.map(
            lambda x: "—" if pd.isna(x) else f"{int(x)}/{len(risky_spans)}")
        t["d_risky"] = t.d_risky.map(lambda x: "—" if pd.isna(x) else f"{int(x):+d}")
        print(t.to_string(index=False))

    if ev_spans:
        pd.DataFrame({"grade": ev_grade}).value_counts().to_frame("n").to_csv(
            args.out / "grades.csv")

    (args.out / "meta.json").write_text(json.dumps({
        "dataset": args.dataset, "thresholds": THR, "window_s": WINDOW_S,
        "hours": tb.seconds / 3600, "n_windows": int(tb.n), "n_segments": tb.n_segments,
        "final_rate": r_final, "final_seg_rate": s_final,
        "available_branches": avail, "missing_branches": missing,
        "n_l1_target_risky": len(risky_spans), "n_events": len(ev_spans),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
