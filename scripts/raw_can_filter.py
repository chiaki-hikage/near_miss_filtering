#!/usr/bin/env python3
"""生 CAN だけで走行時間を粗く絞る軽量フィルタ。**通過率を狙って閾値を決める。**

目的は再現率ではなく、確認すべき時間を全走行時間の数 % まで落とすこと。
後段 (特徴量を作る重い処理や目視) に渡す前の足切りとして使う。

使うもの
  - 安全系フラグ (ABS / VSC / TRC / PCS / 滑り警告) が**あれば**それ
  - 生の ax / ay / yaw / 舵角 / 車速 / 輪速の、窓ごとの max / min / range

使わないもの
  平滑化・微分・チャネル間の時刻同期・派生特徴量。
  各チャネルは**自分のタイムスタンプだけ**で窓に振り分けて集計する。
  輪速の開きも、4 輪の同時刻値の差ではなく
  「窓内の 4 輪の最大値 - 4 輪の最小値」で出す (時刻合わせが要らない)。

閾値の決め方
  条件ごとに「分布のどの裾を切るか」を 1 つのつまみ q で共通に決め、
  窓が**どれか 1 つでも**条件に触れたら通過とする。q を二分探索して
  目標の通過率に合わせる。閾値は実データの分位なので、そのまま絶対値で出せる。

  uv run python scripts/raw_can_filter.py raw_data/Chunk_1 raw_data/Chunk_2 \\
      --labels out/chunk1/labels.csv --labels out/chunk2_final/labels.csv \\
      --window-s 10 --out out/raw_can_filter

出力
  windows.csv    1 行 1 窓。生の max/min/range
  thresholds.csv 目標通過率ごとの閾値と実際の通過率
  retention.csv  目標通過率ごとの、人手ラベルの残り方 (イベント種別ごと)
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

# 窓ごとに見る量と向き。
#   "hi" = 大きいほど異常 / "lo" = 小さいほど異常 (制動側)
CONDITIONS: dict[str, str] = {
    "ax_min": "lo",          # 強い減速
    "ax_max": "hi",          # 強い加速
    "ay_absmax": "hi",       # 横加速度
    "yaw_absmax": "hi",      # ヨーレート
    "steer_range": "hi",     # 窓内の舵角の振れ幅
    "steer_absmax": "hi",    # 舵角そのもの
    "ws_gap": "hi",          # 4 輪の最大 - 最小 (時刻合わせなし)
    "speed_range": "hi",     # 窓内の車速変化
}

# 安全系フラグ。立っていれば無条件で通す。車種・データセットにより無いことがある。
SAFETY_FLAGS = ("abs_active_flag", "vsc_active_flag", "slip_warn", "tc_disabled",
                "abs_fault", "aeb_active", "pcs_active")

# 立ち方は見るが**通過条件には使わない**フラグ。
#   decel_flag は滑りではなく減速度そのもので立つ (comma2k19 RAV4 の実測で
#   -3.0 m/s^2 より強ければ 100%、-1.0 より弱ければ 0%)。ax_min の言い換えに
#   近く、しかも窓の 5% で立つので、これを通過条件に入れると
#   通過率が 5% より下がらなくなる。
WATCH_FLAGS = ("decel_flag", "brake_pressed")

WHEELS = ("ws_fl_mps", "ws_fr_mps", "ws_rl_mps", "ws_rr_mps")

_S: dict = {}


def init(vdir: Path, window_s: float) -> None:
    _S["vehicles"] = load_vehicle_configs(vdir)
    _S["window_s"] = window_s


def _bucket_stats(t: np.ndarray, v: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """各窓での最小・最大。そのチャネル自身の時刻だけを使う (時刻同期をしない)。"""
    n = len(edges) - 1
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    if t.size == 0:
        return lo, hi
    idx = np.searchsorted(edges, t, side="right") - 1
    ok = (idx >= 0) & (idx < n) & np.isfinite(v)
    if not ok.any():
        return lo, hi
    i, x = idx[ok], v[ok]
    order = np.argsort(i, kind="stable")
    i, x = i[order], x[order]
    starts = np.flatnonzero(np.r_[True, np.diff(i) != 0])
    for s, e in zip(starts, np.r_[starts[1:], i.size]):
        lo[i[s]] = x[s:e].min()
        hi[i[s]] = x[s:e].max()
    return lo, hi


def run_segment(ref) -> pd.DataFrame | None:
    veh = find_vehicle_config(ref.dongle_id, _S["vehicles"])
    if veh is None:
        return None
    try:
        sd = comma2k19.load_segment(ref, veh, with_raw_can=True)
    except Exception:                                          # noqa: BLE001
        return None
    ch = sd.channels
    if not ch:
        return None
    a, b = sd.t_span
    if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
        return None
    w = float(_S["window_s"])
    edges = np.arange(a, b + w, w)
    if edges.size < 2:
        return None
    n = len(edges) - 1

    out = {"drive_id": [ref.drive_id] * n, "segment": [ref.index] * n,
           "t_start": edges[:-1], "t_end": edges[1:]}

    def stats(name):
        c = ch.get(name)
        return _bucket_stats(c.t, c.v, edges) if c is not None else (np.full(n, np.nan),) * 2

    ax_lo, ax_hi = stats("accel_x")
    ay_lo, ay_hi = stats("accel_y")
    yaw_lo, yaw_hi = stats("yaw_rate")
    st_lo, st_hi = stats("steer_deg")
    sp_lo, sp_hi = stats("speed_mps")
    out["ax_min"], out["ax_max"] = ax_lo, ax_hi
    out["ay_absmax"] = np.fmax(np.abs(ay_lo), np.abs(ay_hi))
    out["yaw_absmax"] = np.fmax(np.abs(yaw_lo), np.abs(yaw_hi))
    out["steer_range"] = st_hi - st_lo
    out["steer_absmax"] = np.fmax(np.abs(st_lo), np.abs(st_hi))
    out["speed_range"] = sp_hi - sp_lo
    out["speed_max"] = sp_hi
    out["speed_min"] = sp_lo

    # 4 輪の開き。同時刻の差ではなく、窓の中の 4 輪全体の最大 - 最小。
    wl, wh = [], []
    for name in WHEELS:
        lo, hi = stats(name)
        wl.append(lo)
        wh.append(hi)
    if wl:
        out["ws_gap"] = np.nanmax(np.vstack(wh), axis=0) - np.nanmin(np.vstack(wl), axis=0)
    else:
        out["ws_gap"] = np.full(n, np.nan)

    for f in (*SAFETY_FLAGS, *WATCH_FLAGS):
        c = ch.get(f)
        if c is None:
            out[f] = np.zeros(n)
            continue
        _, hi = _bucket_stats(c.t, c.v, edges)
        out[f] = np.where(np.isfinite(hi), hi, 0.0)

    return pd.DataFrame(out)


def pass_mask(df: pd.DataFrame, thr: dict[str, float], use_flags: bool) -> np.ndarray:
    m = np.zeros(len(df), dtype=bool)
    for name, direction in CONDITIONS.items():
        if name not in thr or name not in df:
            continue
        x = df[name].to_numpy()
        hit = x <= thr[name] if direction == "lo" else x >= thr[name]
        m |= np.where(np.isfinite(x), hit, False)
    if use_flags:
        for f in SAFETY_FLAGS:
            if f in df:
                m |= df[f].to_numpy() > 0.5
    return m


def thresholds_at(df: pd.DataFrame, q: float) -> dict[str, float]:
    """つまみ q (0〜1) から条件ごとの閾値を作る。q が 1 に近いほど厳しい。"""
    thr = {}
    for name, direction in CONDITIONS.items():
        x = df[name].dropna()
        if x.empty:
            continue
        thr[name] = float(x.quantile(1.0 - q) if direction == "lo" else x.quantile(q))
    return thr


def solve_for_rate(df: pd.DataFrame, target: float, use_flags: bool) -> tuple[float, dict, float]:
    """目標通過率に一番近い q を二分探索する。"""
    lo, hi = 0.5, 1.0 - 1e-6
    best = None
    for _ in range(60):
        q = (lo + hi) / 2
        thr = thresholds_at(df, q)
        rate = float(pass_mask(df, thr, use_flags).mean())
        if best is None or abs(rate - target) < abs(best[2] - target):
            best = (q, thr, rate)
        if rate > target:
            lo = q
        else:
            hi = q
    return best


def load_labels(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["source"] = p.name
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["risky"] = d.risky.astype(bool)
    return d


def retained(df: pd.DataFrame, m: np.ndarray, labels: pd.DataFrame) -> np.ndarray:
    """各ラベルについて、その区間と重なる通過窓があるか。"""
    keep = df[m]
    out = np.zeros(len(labels), dtype=bool)
    for i, r in enumerate(labels.itertuples()):
        same = keep[(keep.drive_id == r.drive_id)]
        if same.empty:
            continue
        out[i] = bool(((same.t_start <= r.t_end) & (r.t_start <= same.t_end)).any())
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("chunks", nargs="+", type=Path, help="comma2k19 のチャンク")
    p.add_argument("--labels", action="append", type=Path, default=None,
                   help="人手ラベルの CSV。複数指定できる")
    p.add_argument("--window-s", type=float, default=10.0)
    p.add_argument("--rates", type=float, nargs="+", default=[0.10, 0.05, 0.03, 0.02, 0.01])
    p.add_argument("--no-flags", action="store_true", help="安全系フラグを条件に使わない")
    p.add_argument("-w", "--workers", type=int, default=8)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=Path("out/raw_can_filter"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    refs = []
    for c in args.chunks:
        refs.extend(comma2k19.find_segments(c))
    if not refs:
        raise SystemExit(f"セグメントがありません: {args.chunks}")
    print(f"{len(refs)} セグメント / 窓 {args.window_s} 秒")

    frames = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init,
                             initargs=(args.vehicles, args.window_s)) as ex:
        for f in ex.map(run_segment, refs, chunksize=4):
            if f is not None and len(f):
                frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "windows.csv", index=False)

    hours = len(df) * args.window_s / 3600.0
    print(f"窓 {len(df)} 個 = {hours:.2f} 時間\n")

    pd.set_option("display.width", 220)
    have = {f: float((df[f] > 0.5).mean()) for f in SAFETY_FLAGS if f in df}
    watch = {f: float((df[f] > 0.5).mean()) for f in WATCH_FLAGS if f in df}
    print("■ 安全系フラグの有無 (立った窓の割合)")
    for f, v in have.items():
        print(f"  {f:20s}{v:8.3%}" + ("   ← 一度も立たない" if v == 0 else ""))
    print("■ 参考 (通過条件には使わない)")
    for f, v in watch.items():
        print(f"  {f:20s}{v:8.3%}")
    use_flags = not args.no_flags and any(v > 0 for v in have.values())
    if not use_flags:
        print("\n  通過条件に使える安全系フラグがありません。運動量だけで絞ります。")

    labels = load_labels(args.labels or [])
    if not labels.empty:
        print(f"\n人手ラベル {len(labels)} 件 (risky {int(labels.risky.sum())} 件)")

    thr_rows, ret_rows = [], []
    for target in args.rates:
        q, thr, rate = solve_for_rate(df, target, use_flags)
        m = pass_mask(df, thr, use_flags)
        row = {"target": target, "actual_rate": rate, "n_windows": int(m.sum()),
               "minutes": round(m.sum() * args.window_s / 60, 1), "q": q}
        row.update({f"thr_{k}": round(v, 3) for k, v in thr.items()})
        thr_rows.append(row)

        print("\n" + "=" * 72)
        print(f"目標 {target:.0%} → 実際 {rate:.2%}  ({int(m.sum())} 窓 = {row['minutes']} 分)")
        print("=" * 72)
        print("  閾値: " + " / ".join(
            f"{k} {'<=' if CONDITIONS[k] == 'lo' else '>='} {v:.2f}" for k, v in thr.items()))
        # どの条件が効いたか
        contrib = {}
        for k, d in CONDITIONS.items():
            if k not in thr:
                continue
            x = df[k].to_numpy()
            hit = np.where(np.isfinite(x), x <= thr[k] if d == "lo" else x >= thr[k], False)
            contrib[k] = int(hit.sum())
        print("  条件ごとの発火窓数: " + ", ".join(f"{k} {v}" for k, v in
                                                 sorted(contrib.items(), key=lambda x: -x[1])))

        if labels.empty:
            continue
        keep = retained(df, m, labels)
        lab = labels.assign(kept=keep)
        risky = lab[lab.risky]
        print(f"  人手 risky {int(risky.kept.sum())}/{len(risky)} 件 残存 "
              f"({risky.kept.mean():.0%})   非 risky {int(lab[~lab.risky].kept.sum())}/"
              f"{int((~lab.risky).sum())} 件")
        ex = lab.assign(t=lab.event_types_at_labeling.str.split("|")).explode("t")
        by = ex.groupby("t").kept.agg(["size", "sum"])
        by.columns = ["ラベル数", "残存"]
        by["残存率"] = (by["残存"] / by["ラベル数"]).map(lambda x: f"{x:.0%}")
        byr = ex[ex.risky].groupby("t").kept.agg(["size", "sum"])
        by["risky"] = byr["size"].reindex(by.index).fillna(0).astype(int)
        by["risky 残存"] = byr["sum"].reindex(by.index).fillna(0).astype(int)
        print(by.sort_values("ラベル数", ascending=False).to_string())
        for _, r in by.iterrows():
            pass
        ret_rows.append({"target": target, "actual_rate": rate,
                         "risky_total": int(len(risky)), "risky_kept": int(risky.kept.sum()),
                         "nonrisky_total": int((~lab.risky).sum()),
                         "nonrisky_kept": int(lab[~lab.risky].kept.sum()),
                         **{f"kept_{i}": int(v) for i, v in by["残存"].items()},
                         **{f"n_{i}": int(v) for i, v in by["ラベル数"].items()}})

    pd.DataFrame(thr_rows).to_csv(args.out / "thresholds.csv", index=False)
    if ret_rows:
        pd.DataFrame(ret_rows).to_csv(args.out / "retention.csv", index=False)
    (args.out / "meta.json").write_text(json.dumps({
        "chunks": [str(c) for c in args.chunks], "window_s": args.window_s,
        "n_windows": int(len(df)), "hours": hours,
        "safety_flags_present": have, "use_flags": use_flags,
        "conditions": CONDITIONS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
