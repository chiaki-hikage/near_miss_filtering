#!/usr/bin/env python3
"""0x226 / 0x3B7 などの借用したビット定義が、別車種でも同じ意味かを確かめる。

RAV4 TSS2 で解析した ABSACT / VSCACT (0x226) と SLIP_WARN (0x3B7) は opendbc に
無い。**アドレスが同じでもビットの意味が同じとは限らない**ので、件数を数える前に
次を車種ごとに並べて比べる。

  1. 値の分布           slip_warn はどの値を取るか。RAV4 では 0 / 3 / 4 / 7
  2. 値ごとの出方       車速・持続時間・セグメント先頭か・tc_disabled と同時か
                        RAV4 では 3 = 始動時のランプ試験、4 = 走行中の作動
  3. 物理量との時間的整合
                        作動区間の前後で輪速差・ax・ay・yaw・ブレーキ油圧が
                        **無作為な対照窓より大きいか**。意味が合っていれば
                        作動区間に偏って大きく出るはず
  4. 基本信号の整合     ay ≒ v·yaw、ax ≒ dv/dt、表示車速 ≒ 輪速平均
                        0x024 / 0x0AA / 0x0B4 の読み方が合っているか

補間はしない。窓の中の生サンプルの最大・最小を取る。車速との突き合わせだけは
最近傍のサンプルを使う。

  uv run python scripts/validate/validate_slip_flags.py \\
      --platform TOYOTA_RAV4_TSS2 --platform TOYOTA_COROLLA_TSS2 -w 8

出力 (--out/<車種>/ 配下)
  runs.csv      フラグが立った区間ごとの前後の物理量
  control.csv   無作為な対照窓の同じ物理量
  values.csv    フラグ値ごとのサンプル数 (VIN の製造者コード別)
  sanity.json   基本信号の整合 (回帰の傾きと相関)
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from near_miss.config import DEFAULT_VEHICLE_DIR, find_vehicle_config_for_platform, load_vehicle_configs

# 立ち方を調べるフラグ。slip_warn は多値なので値ごとに区間を分ける。
FLAGS = ("slip_warn", "abs_active_flag", "vsc_active_flag", "tc_disabled")
WHEELS = ("ws_fl_mps", "ws_fr_mps", "ws_rl_mps", "ws_rr_mps")

_S: dict = {}


def _nearest(t_src: np.ndarray, v_src: np.ndarray, t: np.ndarray) -> np.ndarray:
    """最近傍の値。補間はしない。"""
    if t_src.size == 0:
        return np.full(np.shape(t), np.nan)
    i = np.clip(np.searchsorted(t_src, t), 1, t_src.size - 1)
    left = t_src[i - 1]
    right = t_src[i]
    j = np.where(np.abs(t - left) <= np.abs(right - t), i - 1, i)
    return v_src[j]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size)
    return list(zip(starts, ends))


def _win(ch: dict, name: str, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    c = ch.get(name)
    if c is None or c.t.size == 0:
        return np.empty(0), np.empty(0)
    i, j = np.searchsorted(c.t, [a, b])
    return c.t[i:j], c.v[i:j]


def context(ch: dict, spread: tuple[np.ndarray, np.ndarray] | None,
            t0: float, t1: float, ctx: float) -> dict:
    """[t0-ctx, t1+ctx] の生サンプルから、前後の物理量をまとめる。"""
    a, b = t0 - ctx, t1 + ctx
    out: dict = {}

    def mx(name, f):
        _, v = _win(ch, name, a, b)
        v = v[np.isfinite(v)]
        return float(f(v)) if v.size else np.nan

    out["ax_min"] = mx("accel_x", np.min)
    out["ax_max"] = mx("accel_x", np.max)
    out["ay_absmax"] = mx("accel_y", lambda v: np.max(np.abs(v)))
    out["yaw_absmax"] = mx("yaw_rate", lambda v: np.max(np.abs(v)))
    out["steer_absmax"] = mx("steer_deg", lambda v: np.max(np.abs(v)))
    out["brake_mc_max"] = mx("brake_mc_mpa", np.max)
    out["brake_pressed_any"] = mx("brake_pressed", np.max)
    out["v_max"] = mx("speed_mps", np.max)
    for f in FLAGS:
        # 同じ窓に他のフラグが立っているか (値そのものの最大)
        out[f"{f}_max"] = mx(f, np.max)

    if spread is not None:
        ts, sp = spread
        i, j = np.searchsorted(ts, [a, b])
        seg = sp[i:j]
        if seg.size and np.isfinite(seg).any():
            k = int(np.nanargmax(seg))
            out["ws_spread_max"] = float(seg[k])
            out["ws_spread_lag_s"] = float(ts[i + k] - t0)
        else:
            out["ws_spread_max"] = np.nan
            out["ws_spread_lag_s"] = np.nan
    tx, vx = _win(ch, "accel_x", a, b)
    ok = np.isfinite(vx)
    out["ax_min_lag_s"] = float(tx[ok][np.argmin(vx[ok])] - t0) if ok.any() else np.nan
    return out


def init(platform: str, vdir: Path, ctx: float, n_ctrl: int, seed: int) -> None:
    from near_miss.io import comma_car_segments as ccs

    _S["mod"] = ccs
    _S["vehicle"] = find_vehicle_config_for_platform(platform, load_vehicle_configs(vdir))
    _S["ctx"], _S["n_ctrl"], _S["seed"] = ctx, n_ctrl, seed
    _S["platform"] = platform


def run_drive(args) -> dict:
    drive_id, refs = args
    ccs, veh, ctx = _S["mod"], _S["vehicle"], _S["ctx"]
    # hash() はプロセスごとに乱数化されるので使わない。再実行で同じ対照窓にする。
    rng = np.random.default_rng(zlib.crc32(f"{drive_id}|{_S['seed']}".encode()))
    runs: list[dict] = []
    ctrl: list[dict] = []
    values: dict = defaultdict(int)
    pairs = {"ay_vyaw": [], "ax_dvdt": [], "speed_ratio": [], "yaw_steer": []}
    tally: dict = defaultdict(float)

    for ref in refs:
        try:
            sd = ccs.load_segment(ref, veh, with_raw_can=True)
        except Exception:
            tally["load_failed"] += 1
            continue
        t_seg0, t_seg1 = sd.t_span
        if not (np.isfinite(t_seg0) and np.isfinite(t_seg1)):
            continue
        ch = sd.channels
        vin = str(sd.meta.get("car_params", {}).get("carVin", "") or "")
        wmi = vin[:3] if len(vin) >= 3 else "?"
        tally["segments"] += 1
        tally["seconds"] += t_seg1 - t_seg0

        spd = ch.get("speed_mps")
        if spd is None or spd.t.size == 0:
            continue

        # 4 輪速の開き。0x0AA の 1 フレームに 4 輪が載っているので時刻が揃う。
        spread = None
        ws = [ch.get(w) for w in WHEELS]
        if all(c is not None and c.t.size for c in ws) and len({c.t.size for c in ws}) == 1 \
                and all(np.array_equal(ws[0].t, c.t) for c in ws[1:]):
            stack = np.vstack([c.v for c in ws])
            spread = (ws[0].t, np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0))
            tally["spread_aligned"] += 1
        else:
            tally["spread_unaligned"] += 1

        # --- 値の分布 ---
        for f in FLAGS:
            c = ch.get(f)
            if c is None or c.t.size == 0:
                values[(wmi, f, "欠落")] += 1
                continue
            u, n = np.unique(c.v[np.isfinite(c.v)], return_counts=True)
            for uu, nn in zip(u, n):
                values[(wmi, f, float(uu))] += int(nn)

        # --- 区間ごと ---
        for f in FLAGS:
            c = ch.get(f)
            if c is None or c.t.size == 0:
                continue
            nz = np.unique(c.v[np.isfinite(c.v) & (c.v > 0)])
            for val in nz:
                for i, j in _runs(c.v == val):
                    t0, t1 = float(c.t[i]), float(c.t[j - 1])
                    r = {"drive_id": drive_id, "segment": ref.index, "vin_wmi": wmi,
                         "flag": f, "value": float(val),
                         "t_in_segment_s": round(t0 - t_seg0, 2),
                         "duration_s": round(t1 - t0, 3),
                         "n_samples": j - i,
                         "at_route_start": bool(ref.index == 0 and (t0 - t_seg0) < 1.0),
                         "v_at_start": float(_nearest(spd.t, spd.v, np.array([t0]))[0])}
                    r |= context(ch, spread, t0, t1, ctx)
                    runs.append(r)

        # --- 対照窓: 走行中 (v > 5 m/s) の無作為な時刻 ---
        moving = spd.t[(spd.v > 5.0) & (spd.t > t_seg0 + ctx) & (spd.t < t_seg1 - ctx)]
        if moving.size:
            for tc in rng.choice(moving, size=min(_S["n_ctrl"], moving.size), replace=False):
                r = {"drive_id": drive_id, "segment": ref.index, "vin_wmi": wmi,
                     "v_at_start": float(_nearest(spd.t, spd.v, np.array([tc]))[0])}
                r |= context(ch, spread, float(tc), float(tc), ctx)
                ctrl.append(r)

        # --- 基本信号の整合 (間引いて対を集める) ---
        ay, yaw = ch.get("accel_y"), ch.get("yaw_rate")
        if ay is not None and yaw is not None and ay.t.size and np.array_equal(ay.t, yaw.t):
            v = _nearest(spd.t, spd.v, ay.t)
            m = (v > 5.0) & np.isfinite(ay.v) & np.isfinite(yaw.v)
            idx = np.flatnonzero(m)[::10]
            pairs["ay_vyaw"].append(np.c_[v[idx] * np.deg2rad(yaw.v[idx]), ay.v[idx]])
        st = ch.get("steer_deg")
        if yaw is not None and st is not None and yaw.t.size and st.t.size:
            s_ = _nearest(st.t, st.v, yaw.t)
            v = _nearest(spd.t, spd.v, yaw.t)
            m = (v > 5.0) & np.isfinite(s_) & np.isfinite(yaw.v)
            idx = np.flatnonzero(m)[::10]
            pairs["yaw_steer"].append(np.c_[s_[idx], yaw.v[idx]])
        ax = ch.get("accel_x")
        if ax is not None and ax.t.size:
            # 1 秒ごとの平均の ax と、その 1 秒での車速の変化
            edges = np.arange(t_seg0, t_seg1, 1.0)
            if edges.size > 2:
                bi = np.searchsorted(ax.t, edges)
                vb = _nearest(spd.t, spd.v, edges)
                for k in range(edges.size - 1):
                    seg = ax.v[bi[k]:bi[k + 1]]
                    if seg.size and vb[k] > 3.0 and np.isfinite(seg).any():
                        pairs["ax_dvdt"].append(np.array([[vb[k + 1] - vb[k], np.nanmean(seg)]]))
        sc = ch.get("speed_can_mps")
        if sc is not None and sc.t.size:
            v = _nearest(spd.t, spd.v, sc.t)
            m = (v > 5.0) & np.isfinite(sc.v)
            idx = np.flatnonzero(m)[::20]
            pairs["speed_ratio"].append(np.c_[v[idx], sc.v[idx]])

    return {"runs": runs, "ctrl": ctrl, "values": dict(values), "tally": dict(tally),
            "pairs": {k: (np.vstack(v) if v else np.empty((0, 2))) for k, v in pairs.items()}}


def fit(p: np.ndarray) -> dict:
    """y = a x の傾き (原点通過) と相関。"""
    if p.shape[0] < 50:
        return {"n": int(p.shape[0])}
    x, y = p[:, 0], p[:, 1]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    return {"n": int(x.size), "slope": round(float(np.dot(x, y) / np.dot(x, x)), 4),
            "corr": round(float(np.corrcoef(x, y)[0, 1]), 4)}


def run_platform(platform: str, args) -> Path:
    from near_miss.io import comma_car_segments as ccs

    if find_vehicle_config_for_platform(platform, load_vehicle_configs(args.vehicles)) is None:
        raise SystemExit(f"車種設定がありません: {platform}")
    refs = ccs.find_segments(ccs.DEFAULT_CACHE, platform)
    if args.limit:
        refs = refs[: args.limit]
    by = defaultdict(list)
    for r in refs:
        by[r.drive_id].append(r)
    tasks = sorted(by.items())
    print(f"\n{platform}: {len(refs)} セグメント / {len(tasks)} ドライブ")

    t0 = time.perf_counter()
    runs, ctrl, values, tally = [], [], defaultdict(int), defaultdict(float)
    pairs = defaultdict(list)
    ini = (platform, args.vehicles, args.ctx, args.n_ctrl, args.seed)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init, initargs=ini) as ex:
        for k, out in enumerate(ex.map(run_drive, tasks, chunksize=4)):
            runs += out["runs"]
            ctrl += out["ctrl"]
            for kk, vv in out["values"].items():
                values[kk] += vv
            for kk, vv in out["tally"].items():
                tally[kk] += vv
            for kk, vv in out["pairs"].items():
                if vv.size:
                    pairs[kk].append(vv)
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(tasks)} ドライブ  {time.perf_counter() - t0:.0f} 秒", flush=True)

    d = args.out / platform
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runs).to_csv(d / "runs.csv", index=False)
    pd.DataFrame(ctrl).to_csv(d / "control.csv", index=False)
    pd.DataFrame([{"vin_wmi": w, "flag": f, "value": v, "samples": n}
                  for (w, f, v), n in values.items()]).to_csv(d / "values.csv", index=False)
    sanity = {k: fit(np.vstack(v)) for k, v in pairs.items() if v}
    meta = {"platform": platform, "segments": int(tally["segments"]),
            "hours": round(tally["seconds"] / 3600, 2), "tally": dict(tally),
            "sanity": sanity, "ctx_s": args.ctx, "elapsed_min": round((time.perf_counter() - t0) / 60, 1)}
    (d / "sanity.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {meta['hours']} 時間 / {meta['elapsed_min']} 分 -> {d}")
    return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--platform", action="append", required=True,
                   help="車種キー。複数指定で並べて比べる (先頭を基準にする)")
    p.add_argument("-w", "--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ctx", type=float, default=2.0, help="区間の前後に取る秒数")
    p.add_argument("--n-ctrl", type=int, default=3, help="1 セグメントあたりの対照窓の数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=Path("out/validate_slip_flags"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for plat in args.platform:
        run_platform(plat, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
