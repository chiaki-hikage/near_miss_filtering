#!/usr/bin/env python3
"""L1 軽量フィルタの閾値を**丸い値の候補から**選ぶ。過適合を避けるため細かく詰めない。

正例は 9 件しかない。連続値で最適化すると、その 9 件にだけ合う閾値が出る。
そこで実務で使える丸い値 (ax -3.0 / ay 3.0 / yaw 15 / steer 50 など) だけを候補にし、
総当たりで「目標通過率の帯に入り、保持数が最大で、通過率が最も小さい」ものを選ぶ。
選んだ条件は leave-one-out で確かめ、9 件に張り付いたものは採らない。

枝 (OR)  ax_decel / ax_accel / ay / yaw / steer
  生サンプルが閾値を超えたら、その 2 秒窓を通す。平滑化も微分もしない。

速度だけで決める軽量除外 (枝とは別)
  low_speed  非走行・超低速を落とす。**通過側にも掛ける**
             (停車中の横 G やヨーは雑音で、拾っても意味がない)
  steady     一定時間ほぼ速度が変わらない区間を走査対象から外す。
             **イベントトリガが立つ窓は必ず残す**ので、通過率そのものは下がらない。
             下がるのは「そもそも評価しなくてよい時間」の量なので、
             通過率とは別に「走査対象の削減率」として出す。

  uv run python scripts/l1/l1_round_thresholds.py raw_data/Chunk_1 raw_data/Chunk_2 \\
      --labels out/chunk1/labels.csv --labels out/chunk2_final/labels.csv \\
      --out out/l1_round

出力
  grid.csv       候補ごとの通過率・保持数
  selected.csv   目標通過率ごとに選んだ条件と、速度・定常除外の効き
  by_event.csv   選んだ条件でのイベント種別ごとの保持率
  stability.csv  leave-one-event-out / leave-one-drive-out の結果
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from near_miss.config import DEFAULT_VEHICLE_DIR, find_vehicle_config, load_vehicle_configs
from near_miss.io import comma2k19

SCRIPTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("l1t", SCRIPTS / "l1_trigger_search.py")
l1t = importlib.util.module_from_spec(_spec)
sys.modules["l1t"] = l1t
_spec.loader.exec_module(l1t)

BRANCHES = l1t.BRANCHES

# 丸い閾値の候補。None は「その枝を使わない」。
CANDIDATES: dict[str, list[float | None]] = {
    "ax_decel": [None, 2.0, 2.5, 3.0, 3.5, 4.0],    # 減速側。符号を反転した値で比べる
    "ax_accel": [None, 2.5, 3.0, 3.5, 4.0],
    "ay": [None, 2.0, 2.5, 3.0, 3.5, 4.0],
    "yaw": [None, 10.0, 15.0, 20.0, 25.0, 30.0],
    "steer": [None, 30.0, 50.0, 100.0, 200.0, 300.0],
}

LOW_SPEED = [0.0, 2.0, 5.0]                                   # これ未満を落とす [m/s]
STEADY = [(10.0, 1.0), (20.0, 1.0), (20.0, 2.0), (30.0, 1.0)]  # (窓 [s], 速度変化の上限 [m/s])

_S: dict = {}


def init(vdir: Path) -> None:
    _S["vehicles"] = load_vehicle_configs(vdir)


def load_one(ref):
    """1 セグメントを 1 回だけ読み、枝のチャネルと車速をそのまま返す。"""
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
    for ch in {c for c, _ in BRANCHES.values()} | {"speed_mps"}:
        c = sd.channels.get(ch)
        if c is None:
            out[ch] = (np.array([]), np.array([]))
        else:
            ok = np.isfinite(c.v)
            out[ch] = (c.t[ok].astype(float), c.v[ok].astype(float))
    return out


def per_bin_speed(segs: list[dict], tl) -> np.ndarray:
    """1 ビンあたりの車速 (生サンプルの最大)。空のビンはセグメント内で前に倣う。"""
    v = np.full(tl.n_bins, -np.inf)
    for s in segs:
        t, x = s["speed_mps"]
        if t.size == 0:
            continue
        i = np.clip(tl.bins_of(s["drive_id"], s["segment"], t), 0, tl.n_bins - 1)
        np.maximum.at(v, i, x)
    v = np.where(np.isneginf(v), np.nan, v)
    for s in segs:
        base, _ = tl.index[(s["drive_id"], s["segment"])]
        n = int(round((s["t1"] - s["t0"]) / tl.grid_dt))
        seg = v[base:base + n]
        if seg.size == 0:
            continue
        ok = np.isfinite(seg)
        if not ok.any():
            continue
        idx = np.where(ok, np.arange(seg.size), 0)
        np.maximum.accumulate(idx, out=idx)
        v[base:base + n] = seg[idx]
    return v


def steady_mask(speed: np.ndarray, tl, win_s: float, delta: float) -> np.ndarray:
    """窓の中で車速の範囲が delta 未満なら定常。max/min だけで決める。"""
    n = max(1, int(round(win_s / tl.grid_dt)))
    x = np.nan_to_num(speed, nan=0.0)
    rng = maximum_filter1d(x, n, mode="nearest") - minimum_filter1d(x, n, mode="nearest")
    return np.isfinite(speed) & (rng < delta)


def branch_masks(arr: dict, tl) -> dict[str, dict[float, np.ndarray]]:
    out: dict[str, dict[float, np.ndarray]] = {}
    for b, cands in CANDIDATES.items():
        i, v = arr[b]
        out[b] = {}
        for th in cands:
            if th is None:
                continue
            m = np.zeros(tl.n_bins, dtype=bool)
            sel = v >= th
            if sel.any():
                m[i[sel]] = True
            out[b][th] = m
    return out


def to_windows(hit: np.ndarray, tl, window_s: float) -> np.ndarray:
    per = max(1, int(round(window_s / tl.grid_dt)))
    n = hit.size
    padn = (-n) % per
    blocks = np.concatenate([hit, np.zeros(padn, dtype=bool)]).reshape(-1, per)
    return np.repeat(blocks.any(axis=1), per)[:n]


def combo_hit(masks: dict, combo: dict, tl) -> np.ndarray:
    hit = np.zeros(tl.n_bins, dtype=bool)
    for b, th in combo.items():
        if th is not None:
            hit |= masks[b][th]
    return hit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("chunks", nargs="+", type=Path)
    p.add_argument("--labels", action="append", type=Path, required=True)
    p.add_argument("--rates", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.05])
    p.add_argument("--band", type=float, default=0.25, help="目標通過率の許容幅 (相対)")
    p.add_argument("--window-s", type=float, default=2.0)
    p.add_argument("--grid-dt", type=float, default=0.05)
    p.add_argument("--react-ax", type=float, default=2.0)
    p.add_argument("--react-ay", type=float, default=2.0)
    p.add_argument("--react-steer-range", type=float, default=30.0)
    p.add_argument("-w", "--workers", type=int, default=8)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=Path("out/l1_round"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
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
    tl = l1t.Timeline(segs, args.grid_dt, gap_s=4.0)
    arr = l1t.build_arrays(segs, tl)
    speed = per_bin_speed(segs, tl)
    masks = branch_masks(arr, tl)
    print(f"{len(segs)} セグメント / {tl.total_s/3600:.2f} 時間\n")

    labels = pd.concat([pd.read_csv(p) for p in args.labels], ignore_index=True)
    labels["risky"] = labels.risky.astype(bool)
    labels = pd.concat([labels, l1t.measure_reaction(segs, labels)], axis=1)
    react = ((labels.ax_min <= -args.react_ax) | (labels.ax_max >= args.react_ax)
             | (labels.ay_absmax >= args.react_ay)
             | (labels.steer_range >= args.react_steer_range))
    labels["l1_target"] = react.fillna(False)
    labels["span"] = [tl.span_of(r.drive_id, r.segment, r.t_start, r.t_end)
                      for r in labels.itertuples()]
    tgt = labels[labels.risky & labels.l1_target].reset_index(drop=True)
    print(f"L1 対象 risky {len(tgt)} 件 / 全ラベル {len(labels)} 件")

    # --- 総当たり -----------------------------------------------------------
    keys = list(CANDIDATES)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(CANDIDATES[k] for k in keys))
              if any(x is not None for x in v)]
    print(f"候補 {len(combos)} 通り x 低速閾値 {len(LOW_SPEED)} 通りを評価します\n")

    rows = []
    for combo in combos:
        hit = combo_hit(masks, combo, tl)
        win = to_windows(hit, tl, args.window_s)
        for vmin in LOW_SPEED:
            m = win & (speed >= vmin) if vmin > 0 else win
            rate = float(m.sum()) * tl.grid_dt / tl.total_s
            kept = l1t.retained(list(tgt.span), m)
            rows.append({**{k: combo[k] for k in keys}, "v_min": vmin,
                         "rate": rate, "kept": int(kept.sum()),
                         "n_branches": sum(combo[k] is not None for k in keys),
                         "kept_vec": kept.copy()})
    g = pd.DataFrame(rows)
    g.drop(columns="kept_vec").to_csv(args.out / "grid.csv", index=False)

    # --- 目標通過率ごとに選ぶ ------------------------------------------------
    def choose(gg: pd.DataFrame, target: float) -> pd.Series | None:
        band = gg[(gg.rate >= target * (1 - args.band)) & (gg.rate <= target * (1 + args.band))]
        if band.empty:
            return None
        best = band[band.kept == band.kept.max()]
        # 同点なら「通過率が小さい → 枝が少ない」の順に選ぶ。閾値は詰めない。
        return best.sort_values(["rate", "n_branches"]).iloc[0]

    pd.set_option("display.width", 240)
    sel_rows, ev_rows = [], []
    for target in args.rates:
        s = choose(g, target)
        print("\n" + "=" * 72)
        print(f"目標通過率 {target:.0%}")
        print("=" * 72)
        if s is None:
            print("  帯に入る候補がありません")
            continue
        combo = {k: (None if pd.isna(s[k]) else float(s[k])) for k in keys}
        hit = combo_hit(masks, combo, tl)
        win = to_windows(hit, tl, args.window_s)
        m = win & (speed >= s.v_min) if s.v_min > 0 else win
        rate = float(m.sum()) * tl.grid_dt / tl.total_s
        print("  採用: " + " / ".join(
            f"{k} {'<= -' if k == 'ax_decel' else '>= '}{combo[k]:.0f}"
            for k in keys if combo[k] is not None)
            + f" / 低速除外 v < {s.v_min:.0f} m/s")
        print(f"  通過 {rate:.2%}  L1 対象 risky {int(s.kept)}/{len(tgt)}")

        # 低速除外だけの効き
        rate_nolow = float(win.sum()) * tl.grid_dt / tl.total_s
        print(f"  低速除外なしなら {rate_nolow:.2%} → 除外で {1 - rate/max(rate_nolow,1e-9):.1%} 削減")

        # 定常除外 (走査対象の削減。トリガが立つ窓は必ず残す)
        for win_s, delta in STEADY:
            st = steady_mask(speed, tl, win_s, delta)
            keep_scan = ~st | win               # トリガが立つ窓は必ず残す
            scan_rate = float(keep_scan.sum()) * tl.grid_dt / tl.total_s
            lost = int(l1t.retained(list(tgt.span), m).sum()
                       - l1t.retained(list(tgt.span), m & keep_scan).sum())
            print(f"    定常除外 (窓 {win_s:.0f} 秒 / 速度変化 < {delta:.0f} m/s): "
                  f"走査対象 {scan_rate:.1%} (={1-scan_rate:.1%} 削減)  risky 損失 {lost}")
            sel_rows.append({"target": target, "rate": rate, "kept": int(s.kept),
                             "n_target": len(tgt), "v_min": s.v_min,
                             "rate_without_lowspeed": rate_nolow,
                             "steady_win_s": win_s, "steady_delta": delta,
                             "scan_rate": scan_rate, "risky_lost_by_steady": lost,
                             **{k: combo[k] for k in keys}})

        # イベント種別ごとの保持率
        kept_all = l1t.retained(list(labels.span), m)
        lab = labels.assign(kept=kept_all)
        ex = lab.assign(t=lab.event_types_at_labeling.str.split("|")).explode("t")
        by = ex.groupby("t").kept.agg(["size", "sum"])
        by.columns = ["ラベル数", "残存"]
        by["残存率"] = (by["残存"] / by["ラベル数"]).map(lambda x: f"{x:.0%}")
        byr = ex[ex.risky & ex.l1_target].groupby("t").kept.agg(["size", "sum"])
        by["L1対象risky"] = byr["size"].reindex(by.index).fillna(0).astype(int)
        by["同 残存"] = byr["sum"].reindex(by.index).fillna(0).astype(int)
        print(by.sort_values("ラベル数", ascending=False).to_string())
        for t, r in by.iterrows():
            ev_rows.append({"target": target, "event": t, "n": int(r["ラベル数"]),
                            "kept": int(r["残存"]), "n_risky_target": int(r["L1対象risky"]),
                            "kept_risky_target": int(r["同 残存"])})

    pd.DataFrame(sel_rows).to_csv(args.out / "selected.csv", index=False)
    pd.DataFrame(ev_rows).to_csv(args.out / "by_event.csv", index=False)

    # --- 安定性 --------------------------------------------------------------
    print("\n" + "=" * 72)
    print("leave-one-out (選び直した条件が、外した正例を拾えるか)")
    print("=" * 72)
    st_rows = []
    kept_mat = np.vstack(g.kept_vec.to_numpy())          # 候補 x 正例
    for target in args.rates:
        band = g[(g.rate >= target * (1 - args.band)) & (g.rate <= target * (1 + args.band))]
        if band.empty:
            continue
        bi = band.index.to_numpy()
        for kind in ("event", "drive"):
            groups = (list(range(len(tgt))) if kind == "event"
                      else sorted(tgt.drive_id.unique()))
            hold_ok, n_held, chosen = 0, 0, []
            for gkey in groups:
                if kind == "event":
                    held = [gkey]
                else:
                    held = [i for i in range(len(tgt)) if tgt.drive_id[i] == gkey]
                keep_idx = [i for i in range(len(tgt)) if i not in held]
                score = kept_mat[np.ix_(bi, keep_idx)].sum(axis=1)
                # 学習側の保持が最大 -> 通過率が小さい -> 枝が少ない
                order = np.lexsort((band.n_branches.to_numpy(), band.rate.to_numpy(), -score))
                pick = bi[order[0]]
                chosen.append(tuple(g.loc[pick, k] for k in keys) + (g.loc[pick, "v_min"],))
                hold_ok += int(kept_mat[pick, held].sum())
                n_held += len(held)
            uniq = sorted({tuple("—" if pd.isna(x) else x for x in c) for c in chosen},
                          key=lambda c: [str(x) for x in c])
            print(f"  {target:.0%} / {kind:5s}: 外した正例の保持 {hold_ok}/{n_held}  "
                  f"選ばれた条件 {len(uniq)} 通り")
            for c in uniq[:3]:
                print("        " + " / ".join(f"{k} {v}" for k, v in
                                              zip(keys + ["v_min"], c)))
            st_rows.append({"target": target, "kind": kind, "held_kept": hold_ok,
                            "n_held": n_held, "n_distinct_config": len(uniq),
                            "configs": [list(c) for c in uniq]})
    pd.DataFrame(st_rows).to_csv(args.out / "stability.csv", index=False)
    (args.out / "meta.json").write_text(json.dumps({
        "hours": tl.total_s / 3600, "window_s": args.window_s, "band": args.band,
        "candidates": {k: [None if v is None else v for v in c] for k, c in CANDIDATES.items()},
        "low_speed": LOW_SPEED, "steady": STEADY, "n_l1_target": int(len(tgt)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
