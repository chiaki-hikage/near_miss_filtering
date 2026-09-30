#!/usr/bin/env python3
"""生 CAN だけの L1 足切りで、**枝ごとの通過率の配分**を探索する。

前提
  前方接近系 (cut_in / short_thw) は L1 の対象外とする。自車の運動しか見ない
  フィルタでは、自車が反応していない接近事象は原理的に拾えない。
  そこで人手 risky を次の 2 つに分け、L1 対象だけで保持率を測る。

    L1 対象     ラベル区間の中で自車の運動が動いている (制動・加速・横・操舵)
    L1 対象外   前方接近が主体で、自車の運動はほぼ動いていない

  分ける基準は**測れる量だけ**で決める (--react-* で変えられる)。所見の文面は
  判断の裏取りにだけ使う。

枝
  ax_decel / ax_accel / ay / yaw / steer の 5 本を独立した OR 条件として扱い、
  枝ごとの通過率の配分を変えて、総通過率を一定にしたまま保持数を最大化する。

2 つの時間の数え方を比べる
  window  固定長の窓に区切り、窓の中に閾値超えがあればその窓を通す
  event   閾値を超えた**生サンプルの前後 ±pad 秒**だけを通す (窓に区切らない)
  どちらも通過率 = 通した時間 / 全走行時間。閾値の意味は同じなので、
  差は「時間の切り出し方」だけになる。

  uv run python scripts/l1/l1_trigger_search.py raw_data/Chunk_1 raw_data/Chunk_2 \\
      --labels out/chunk1/labels.csv --labels out/chunk2_final/labels.csv \\
      --rates 0.01 0.02 0.03 0.05 --out out/l1_trigger

出力
  classification.csv  人手 risky の L1 対象 / 対象外 と、その判断に使った実測値
  search.csv          配分の探索結果 (枝ごとの閾値と保持数)
  compare.csv         window と event の比較
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from near_miss.config import DEFAULT_VEHICLE_DIR, find_vehicle_config, load_vehicle_configs
from near_miss.io import comma2k19

# 枝の定義。(チャネル, 向き)。向きは lo = 小さいほど異常。
BRANCHES: dict[str, tuple[str, str]] = {
    "ax_decel": ("accel_x", "lo"),
    "ax_accel": ("accel_x", "hi"),
    "ay": ("accel_y", "abs"),
    "yaw": ("yaw_rate", "abs"),
    "steer": ("steer_deg", "abs"),
}

_S: dict = {}


def init(vdir: Path) -> None:
    _S["vehicles"] = load_vehicle_configs(vdir)


def load_one(ref) -> dict | None:
    """1 セグメントぶんの生サンプルを、チャネルごとにそのまま返す。"""
    veh = find_vehicle_config(ref.dongle_id, _S["vehicles"])
    if veh is None:
        return None
    try:
        sd = comma2k19.load_segment(ref, veh, with_raw_can=True)
    except Exception:                                          # noqa: BLE001
        return None
    a, b = sd.t_span
    if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
        return None
    out = {"drive_id": ref.drive_id, "segment": ref.index, "t0": float(a), "t1": float(b)}
    for ch in {c for c, _ in BRANCHES.values()}:
        c = sd.channels.get(ch)
        if c is None:
            out[ch] = (np.array([]), np.array([]))
        else:
            ok = np.isfinite(c.v)
            out[ch] = (c.t[ok].astype(float), c.v[ok].astype(float))
    return out


def branch_values(seg: dict, branch: str) -> tuple[np.ndarray, np.ndarray]:
    """枝の判定量。生サンプルのまま (平滑化も微分もしない)。"""
    ch, direction = BRANCHES[branch]
    t, v = seg[ch]
    if direction == "lo":
        return t, -v          # 符号を反転して「大きいほど異常」に揃える
    if direction == "abs":
        return t, np.abs(v)
    return t, v


class Timeline:
    """全走行を 1 本の細かい時間格子に載せる。評価を numpy だけで回すため。

    ドライブの境目は pad のぶんだけ空ける (event 方式で隣のドライブへ
    はみ出さないようにする)。
    """

    def __init__(self, segs: list[dict], grid_dt: float, gap_s: float):
        self.grid_dt = grid_dt
        self.index: dict[tuple[str, int], tuple[int, float]] = {}
        order = sorted(segs, key=lambda s: (s["drive_id"], s["t0"]))
        pos = 0
        prev_drive = None
        for s in order:
            if prev_drive is not None and s["drive_id"] != prev_drive:
                pos += int(round(gap_s / grid_dt))
            n = int(round((s["t1"] - s["t0"]) / grid_dt))
            self.index[(s["drive_id"], s["segment"])] = (pos, s["t0"])
            pos += n
            prev_drive = s["drive_id"]
        self.n_bins = pos
        self.total_s = sum(s["t1"] - s["t0"] for s in segs)

    def bins_of(self, drive_id: str, segment: int, t: np.ndarray) -> np.ndarray:
        base, t0 = self.index[(drive_id, segment)]
        return base + np.floor((t - t0) / self.grid_dt).astype(np.int64)

    def span_of(self, drive_id: str, segment: int, t0: float, t1: float) -> tuple[int, int] | None:
        """ラベル区間 -> 格子の範囲。ラベルはセグメント番号を持っている。"""
        key = (drive_id, int(segment))
        if key not in self.index:
            return None
        base, st = self.index[key]
        a = base + int(np.floor((t0 - st) / self.grid_dt))
        b = base + int(np.ceil((t1 - st) / self.grid_dt))
        return (min(a, b), max(a, b))


def build_arrays(segs: list[dict], tl: Timeline) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """枝ごとに (格子番号, 値) を全走行ぶん連結する。"""
    out = {}
    for b in BRANCHES:
        bins, vals = [], []
        for s in segs:
            t, v = branch_values(s, b)
            if t.size == 0:
                continue
            i = tl.bins_of(s["drive_id"], s["segment"], t)
            ok = (i >= 0) & (i < tl.n_bins)
            bins.append(i[ok])
            vals.append(v[ok])
        out[b] = (np.concatenate(bins) if bins else np.array([], dtype=np.int64),
                  np.concatenate(vals) if vals else np.array([]))
    return out


def dilate(mask: np.ndarray, half: int) -> np.ndarray:
    """True の前後 half ビンを True にする (累積和で O(n))。"""
    if half <= 0:
        return mask
    c = np.concatenate([[0], np.cumsum(mask.astype(np.int32))])
    n = mask.size
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    return (c[hi] - c[lo]) > 0


def evaluate(arr: dict, tl: Timeline, thr: dict[str, float], mode: str,
             window_s: float, pad_s: float) -> np.ndarray:
    """通過した格子ビンのマスク。"""
    hit = np.zeros(tl.n_bins, dtype=bool)
    for b, th in thr.items():
        i, v = arr[b]
        if i.size == 0 or not np.isfinite(th):
            continue
        sel = v >= th
        if sel.any():
            hit[i[sel]] = True
    if mode == "window":
        per = max(1, int(round(window_s / tl.grid_dt)))
        n = tl.n_bins
        pad_n = (-n) % per
        blocks = np.concatenate([hit, np.zeros(pad_n, dtype=bool)]).reshape(-1, per)
        passed = blocks.any(axis=1)
        return np.repeat(passed, per)[:n]
    return dilate(hit, int(round(pad_s / tl.grid_dt)))


def kept_rate(mask: np.ndarray, tl: Timeline) -> float:
    return float(mask.sum()) * tl.grid_dt / tl.total_s


def thresholds_from_rates(pool: dict[str, np.ndarray], rates: dict[str, float]) -> dict[str, float]:
    """枝ごとの「サンプルの何割を超えさせるか」から閾値を作る。"""
    thr = {}
    for b, r in rates.items():
        x = pool[b]
        if x.size == 0 or r <= 0:
            thr[b] = np.inf
            continue
        thr[b] = float(np.quantile(x, 1.0 - min(r, 0.5)))
    return thr


def solve_scale(arr: dict, tl: Timeline, pool: dict[str, np.ndarray],
                alloc: dict[str, float], target: float, mode: str,
                window_s: float, pad_s: float) -> tuple[dict[str, float], float, np.ndarray]:
    """配分を保ったまま全体の厳しさを二分探索し、目標通過率に合わせる。"""
    lo, hi = 1e-7, 0.3
    best = None
    for _ in range(26):
        p = (lo + hi) / 2
        thr = thresholds_from_rates(pool, {b: a * p for b, a in alloc.items()})
        mask = evaluate(arr, tl, thr, mode, window_s, pad_s)
        rate = kept_rate(mask, tl)
        if best is None or abs(rate - target) < abs(best[1] - target):
            best = (thr, rate, mask)
        if rate > target:
            hi = p
        else:
            lo = p
    return best


def retained(spans: list, mask: np.ndarray) -> np.ndarray:
    c = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    out = np.zeros(len(spans), dtype=bool)
    for i, sp in enumerate(spans):
        if sp is None:
            continue
        a, b = max(0, sp[0]), min(mask.size, sp[1])
        out[i] = b > a and (c[b] - c[a]) > 0
    return out


def measure_reaction(segs: list[dict], labels: pd.DataFrame) -> pd.DataFrame:
    """ラベル区間の中の自車の動き。生サンプルの max/min/range だけ。"""
    rows = []
    for r in labels.itertuples():
        m = {"ax_min": np.nan, "ax_max": np.nan, "ay_absmax": np.nan,
             "yaw_absmax": np.nan, "steer_range": np.nan, "steer_absmax": np.nan}
        for s in segs:
            if s["drive_id"] != r.drive_id:
                continue
            for ch, keys in (("accel_x", ("ax_min", "ax_max")), ("accel_y", ("ay_absmax",)),
                             ("yaw_rate", ("yaw_absmax",)),
                             ("steer_deg", ("steer_range", "steer_absmax"))):
                t, v = s[ch]
                sel = (t >= r.t_start) & (t <= r.t_end)
                if not sel.any():
                    continue
                x = v[sel]
                if ch == "accel_x":
                    m["ax_min"] = np.fmin(m["ax_min"], x.min())
                    m["ax_max"] = np.fmax(m["ax_max"], x.max())
                elif ch == "steer_deg":
                    m["steer_range"] = np.fmax(m["steer_range"], x.max() - x.min())
                    m["steer_absmax"] = np.fmax(m["steer_absmax"], np.abs(x).max())
                else:
                    k = keys[0]
                    m[k] = np.fmax(m[k], np.abs(x).max())
        rows.append(m)
    return pd.DataFrame(rows, index=labels.index)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("chunks", nargs="+", type=Path)
    p.add_argument("--labels", action="append", type=Path, required=True)
    p.add_argument("--rates", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.05])
    p.add_argument("--window-s", type=float, default=2.0)
    p.add_argument("--grid-dt", type=float, default=0.05,
                   help="評価に使う時間格子の刻み [s]")
    p.add_argument("--pad-s", type=float, default=1.0,
                   help="event 方式で閾値超えサンプルの前後に取る秒数")
    p.add_argument("--n-search", type=int, default=400, help="配分の試行回数")
    p.add_argument("--seed", type=int, default=0)
    # 自車が反応したとみなす基準
    p.add_argument("--react-ax", type=float, default=2.0)
    p.add_argument("--react-ay", type=float, default=2.0)
    p.add_argument("--react-steer-range", type=float, default=30.0)
    p.add_argument("-w", "--workers", type=int, default=8)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=Path("out/l1_trigger"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    refs = []
    for c in args.chunks:
        refs.extend(comma2k19.find_segments(c))
    print(f"{len(refs)} セグメントを読みます")
    segs = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init,
                             initargs=(args.vehicles,)) as ex:
        for s in ex.map(load_one, refs, chunksize=4):
            if s:
                segs.append(s)
    tl = Timeline(segs, args.grid_dt, gap_s=max(4.0, 2 * args.pad_s))
    arr = build_arrays(segs, tl)
    print(f"{len(segs)} セグメント / {tl.total_s/3600:.2f} 時間 / 格子 {tl.n_bins} ビン\n")

    pool = {b: v for b, (_, v) in arr.items()}

    labels = pd.concat([pd.read_csv(p) for p in args.labels], ignore_index=True)
    labels["risky"] = labels.risky.astype(bool)
    labels = pd.concat([labels, measure_reaction(segs, labels)], axis=1)
    react = ((labels.ax_min <= -args.react_ax) | (labels.ax_max >= args.react_ax)
             | (labels.ay_absmax >= args.react_ay)
             | (labels.steer_range >= args.react_steer_range))
    labels["l1_target"] = react.fillna(False)

    args.out.mkdir(parents=True, exist_ok=True)
    risky = labels[labels.risky].reset_index(drop=True)
    cols = ["drive_id", "segment", "t_in_segment_s", "event_types_at_labeling", "note",
            "ax_min", "ax_max", "ay_absmax", "yaw_absmax", "steer_range", "l1_target"]
    risky[cols].to_csv(args.out / "classification.csv", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 34)
    print("■ 人手 risky 12 件の分類 (ラベル区間の中で自車の運動が動いたか)")
    print(f"  基準: |ax| >= {args.react_ax} / |ay| >= {args.react_ay} / "
          f"舵角の振れ >= {args.react_steer_range} deg")
    print(risky[cols].round(2).to_string(index=False))
    n_t = int(risky.l1_target.sum())
    print(f"\n  L1 対象 {n_t} 件 / L1 対象外 (前方接近主体) {len(risky) - n_t} 件")

    tgt = labels[labels.risky & labels.l1_target]
    out_spans = labels[labels.risky & ~labels.l1_target]
    spans_t = [tl.span_of(r.drive_id, r.segment, r.t_start, r.t_end) for r in tgt.itertuples()]
    spans_o = [tl.span_of(r.drive_id, r.segment, r.t_start, r.t_end)
               for r in out_spans.itertuples()]

    rng = np.random.default_rng(args.seed)
    rows, comp = [], []
    for target in args.rates:
        print("\n" + "=" * 72)
        print(f"目標通過率 {target:.0%}")
        print("=" * 72)
        for mode in ("window", "event"):
            even = {b: 1.0 / len(BRANCHES) for b in BRANCHES}
            thr, rate, mask = solve_scale(arr, tl, pool, even, target, mode,
                                          args.window_s, args.pad_s)
            base = int(retained(spans_t, mask).sum())
            base_o = int(retained(spans_o, mask).sum())
            print(f"  [{mode:6s}] 均等配分  通過 {rate:.2%}  L1 対象 {base}/{len(tgt)}"
                  f"   (対象外 {base_o}/{len(spans_o)})")
            comp.append({"target": target, "mode": mode, "alloc": "even", "rate": rate,
                         "kept": base, "n_target": len(tgt), "kept_nontarget": base_o})

            best = (base, even, thr, rate, mask)
            tries = 0
            at_best: list[dict] = []
            for _ in range(args.n_search):
                a = rng.dirichlet(np.ones(len(BRANCHES)) * 0.6)
                alloc = {b: float(x) for b, x in zip(BRANCHES, a)}
                th2, r2, m2 = solve_scale(arr, tl, pool, alloc, target, mode,
                                          args.window_s, args.pad_s)
                if abs(r2 - target) > target * 0.2:
                    continue
                tries += 1
                k = int(retained(spans_t, m2).sum())
                if k > best[0]:
                    best = (k, alloc, th2, r2, m2)
                    at_best = [alloc]
                elif k == best[0]:
                    at_best.append(alloc)
            best_o = int(retained(spans_o, best[4]).sum())
            print(f"  [{mode:6s}] 最良配分  通過 {best[3]:.2%}  L1 対象 {best[0]}/{len(tgt)}"
                  f"   (対象外 {best_o}/{len(spans_o)})  試行 {tries} / "
                  f"同点 {len(at_best)} 通り")
            if len(at_best) > 1:
                med = {b: float(np.median([a[b] for a in at_best])) for b in BRANCHES}
                tot = sum(med.values())
                print("            同点の配分の中央 " + " / ".join(
                    f"{b} {v/tot:.0%}" for b, v in med.items()))
            print("            配分 " + " / ".join(f"{b} {v:.0%}" for b, v in best[1].items()))
            print("            閾値 " + " / ".join(
                f"{b} {'<= -' if BRANCHES[b][1] == 'lo' else '>= '}{v:.2f}"
                for b, v in best[2].items()))
            rows.append({"target": target, "mode": mode, "rate": best[3], "kept": best[0],
                         "n_target": len(tgt), "kept_nontarget": best_o,
                         "n_alloc_at_best": len(at_best),
                         **{f"alloc_{b}": round(v, 4) for b, v in best[1].items()},
                         **{f"thr_{b}": round(v, 3) for b, v in best[2].items()}})
            comp.append({"target": target, "mode": mode, "alloc": "best", "rate": best[3],
                         "kept": best[0], "n_target": len(tgt), "kept_nontarget": best_o})

    pd.DataFrame(rows).to_csv(args.out / "search.csv", index=False)
    pd.DataFrame(comp).to_csv(args.out / "compare.csv", index=False)
    (args.out / "meta.json").write_text(json.dumps({
        "hours": tl.total_s / 3600, "n_segments": len(segs), "grid_dt": args.grid_dt,
        "window_s": args.window_s, "pad_s": args.pad_s, "n_search": args.n_search,
        "react": {"ax": args.react_ax, "ay": args.react_ay,
                  "steer_range": args.react_steer_range},
        "branches": {k: list(v) for k, v in BRANCHES.items()},
        "n_l1_target": int(n_t), "n_l1_nontarget": int(len(risky) - n_t),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
