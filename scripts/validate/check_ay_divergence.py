#!/usr/bin/env python3
"""CAN の横 G (ACCEL_Y) が 車速 x ヨーレート と大きく食い違う急制動を拾い、
**既存の検出器がどれだけ巻き込まれているか**を数える。

直進で強く制動しているだけなのに ACCEL_Y が数 m/s^2 を示す車両がある。
A 等級 69 件を読み直したときに 7 件で見つかった (Highlander では
ax -8.0 m/s^2 のとき ay_can 6.08 / v x yaw 0.13)。ピッチングで前後方向の
加速度が横軸へ回り込んでいるとみられるが、原因は確定していない。

影響する可能性があるのは **ay_can_mps2 を条件に含む検出器だけ**。
設定を読んでそれを自動で調べ、次を数える。

  1. 乖離した急制動が何区間・何秒あるか (1 時間あたりに直す)
  2. 各検出器の発火区間のうち、その窓と重なるものが何件あるか
  3. ay_can を使う検出器では、重なった発火が ay_can の条件によって
     成立しているのか (ay_kin 側だけでは通らないのか) まで見る

  uv run python scripts/validate/check_ay_divergence.py \\
      --platform TOYOTA_RAV4_TSS2 --platform TOYOTA_HIGHLANDER_TSS2 \\
      --limit 1200 -w 8 --out out/ay_divergence

出力
  windows.csv   乖離した急制動の区間 (車種・ドライブ・時刻・大きさ)
  events.csv    重なった検出イベント
  summary.json  検出器ごとの件数と割合
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
    DEFAULT_DETECTION,
    DEFAULT_VEHICLE_DIR,
    find_vehicle_config_for_platform,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.detectors import detect_all
from near_miss.features import compute_features
from near_miss.io import comma_car_segments as ccs
from near_miss.signals import to_grid

_S: dict = {}


def init(platform: str, vdir: Path, cfg: dict, opts: dict) -> None:
    _S["vehicle"] = find_vehicle_config_for_platform(platform, load_vehicle_configs(vdir))
    _S["cfg"] = cfg
    _S["opts"] = opts
    _S["platform"] = platform


def ay_can_users(cfg: dict) -> dict[str, float]:
    """ay_can_mps2 を条件に使っている検出器と、その閾値。"""
    out: dict[str, float] = {}
    for name, spec in cfg["events"].items():
        for c in spec.get("conditions", []) or []:
            if c.get("feature") == "ay_can_mps2":
                out[name] = float(c["threshold"])
        if spec.get("feature") == "ay_can_mps2":
            out[name] = float(spec.get("threshold", 0.0))
    return out


def _runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    st = list(np.flatnonzero(d == 1) + 1)
    en = list(np.flatnonzero(d == -1))
    if mask[0]:
        st.insert(0, 0)
    if mask[-1]:
        en.append(mask.size - 1)
    return list(zip(st, en))


def run_segment(ref) -> dict | None:
    cfg, veh, o = _S["cfg"], _S["vehicle"], _S["opts"]
    try:
        sd = ccs.load_segment(ref, veh, with_raw_can=True, with_radar=False)
        gs = to_grid(sd, cfg)
        gs = compute_features(gs, cfg, radar=sd.radar, vehicle=veh)
    except Exception:                                          # noqa: BLE001
        return None
    df = gs.df
    if df.empty or "ay_can_mps2" not in df or "ay_kin_mps2" not in df or "ax_mps2" not in df:
        return None

    t = df["t"].to_numpy()
    ax = df["ax_mps2"].to_numpy()
    ay_can = df["ay_can_mps2"].to_numpy()
    ay_kin = df["ay_kin_mps2"].to_numpy()
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.05

    braking = np.where(np.isfinite(ax), ax <= o["brake_mps2"], False)
    gap = np.abs(ay_can) - np.abs(ay_kin)
    diverged = np.where(np.isfinite(gap), gap >= o["gap_mps2"], False)
    mask = braking & diverged

    windows = []
    for a, b in _runs_of(mask):
        if (b - a + 1) * dt < o["min_duration_s"]:
            continue
        sl = slice(a, b + 1)
        windows.append({
            "platform": _S["platform"], "drive_id": ref.drive_id, "segment": ref.index,
            "t_start": float(t[a]), "t_end": float(t[b]),
            "duration_s": round(float(t[b] - t[a]) + dt, 2),
            "ax_min": round(float(np.nanmin(ax[sl])), 2),
            "ay_can_absmax": round(float(np.nanmax(np.abs(ay_can[sl]))), 2),
            "ay_kin_absmax": round(float(np.nanmax(np.abs(ay_kin[sl]))), 2),
            "gap_max": round(float(np.nanmax(gap[sl])), 2),
            "v_at_start": round(float(df["v_mps"].to_numpy()[a]), 2) if "v_mps" in df else np.nan,
        })

    events = []
    users = _S["opts"]["ay_can_users"]
    for e in detect_all(gs, cfg):
        hit = [w for w in windows if e.t_start <= w["t_end"] and w["t_start"] <= e.t_end]
        row = {"platform": _S["platform"], "drive_id": ref.drive_id, "segment": ref.index,
               "event": e.event_type, "t_start": float(e.t_start), "t_end": float(e.t_end),
               "in_divergence": bool(hit)}
        if hit and e.event_type in users:
            # その発火の窓で ay_can が本当に条件を満たしているか、
            # 満たしているのは乖離のぶんなのかを見る。
            m = (t >= e.t_start) & (t <= e.t_end)
            row["ay_can_absmax"] = round(float(np.nanmax(np.abs(ay_can[m]))), 2) if m.any() else np.nan
            row["ay_kin_absmax"] = round(float(np.nanmax(np.abs(ay_kin[m]))), 2) if m.any() else np.nan
            row["ay_can_threshold"] = users[e.event_type]
        events.append(row)

    return {"windows": windows, "events": events,
            "hours": float(len(df)) * dt / 3600.0, "segments": 1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--platform", action="append", required=True)
    p.add_argument("--limit", type=int, default=None, help="車種ごとのセグメント数の上限")
    p.add_argument("-w", "--workers", type=int, default=8)
    p.add_argument("--brake-mps2", type=float, default=-3.0,
                   help="急制動とみなす縦加速度 [m/s^2]")
    p.add_argument("--gap-mps2", type=float, default=2.0,
                   help="乖離とみなす |ay_can| - |ay_kin| [m/s^2]")
    p.add_argument("--min-duration-s", type=float, default=0.1)
    p.add_argument("--cache", type=Path, default=ccs.DEFAULT_CACHE)
    p.add_argument("--config", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=Path("out/ay_divergence"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    users = ay_can_users(cfg)
    print(f"ay_can_mps2 を条件に使う検出器: {users or '(なし)'}\n")

    opts = {"brake_mps2": args.brake_mps2, "gap_mps2": args.gap_mps2,
            "min_duration_s": args.min_duration_s, "ay_can_users": users}

    all_win: list[dict] = []
    all_ev: list[dict] = []
    hours: dict[str, float] = defaultdict(float)
    segs: dict[str, int] = defaultdict(int)

    for platform in args.platform:
        refs = ccs.find_segments(args.cache, platform)
        if args.limit:
            refs = refs[: args.limit]
        print(f"{platform}: {len(refs)} セグメント", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init,
                                 initargs=(platform, args.vehicles, cfg, opts)) as ex:
            for r in ex.map(run_segment, refs, chunksize=4):
                if not r:
                    continue
                all_win.extend(r["windows"])
                all_ev.extend(r["events"])
                hours[platform] += r["hours"]
                segs[platform] += 1

    args.out.mkdir(parents=True, exist_ok=True)
    win = pd.DataFrame(all_win)
    ev = pd.DataFrame(all_ev)
    win.to_csv(args.out / "windows.csv", index=False)
    ev.to_csv(args.out / "events.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n■ 乖離した急制動 (ax <= %.1f かつ |ay_can| - |ay_kin| >= %.1f)"
          % (args.brake_mps2, args.gap_mps2))
    for p in args.platform:
        w = win[win.platform == p] if not win.empty else win
        h = hours[p]
        print(f"  {p:<24}{len(w):>5} 区間 / {h:6.1f} 時間 = {len(w)/h if h else 0:5.2f} 件/h"
              f"   {segs[p]} セグメント")
    if not win.empty:
        print("\n  大きさ: |ay_can| 中央 %.2f / |ay_kin| 中央 %.2f / 差 中央 %.2f / 差 最大 %.2f"
              % (win.ay_can_absmax.median(), win.ay_kin_absmax.median(),
                 win.gap_max.median(), win.gap_max.max()))

    print("\n■ 検出器ごとの巻き込まれ")
    summary: dict[str, dict] = {}
    if not ev.empty:
        g = ev.groupby("event").in_divergence.agg(["size", "sum"])
        g.columns = ["発火", "乖離窓と重なる"]
        g["割合"] = (g["乖離窓と重なる"] / g["発火"]).map(lambda x: f"{x:.1%}")
        g["ay_can を使う"] = ["はい" if i in users else "" for i in g.index]
        print(g.sort_values("乖離窓と重なる", ascending=False).to_string())
        summary = {str(k): {"fired": int(v["発火"]), "in_divergence": int(v["乖離窓と重なる"])}
                   for k, v in g.iterrows()}

        for name in users:
            sub = ev[(ev.event == name) & ev.in_divergence]
            if sub.empty:
                continue
            thr = users[name]
            # ay_can の条件を外しても残るか。ay_kin 側は別条件で担保されている。
            print(f"\n  {name}: 乖離窓と重なる発火 {len(sub)} 件")
            print(f"    そのうち |ay_can| が閾値 {thr} を超えているもの: "
                  f"{int((sub.ay_can_absmax > thr).sum())} 件")
            print(f"    同じ窓の |ay_kin| 中央 {sub.ay_kin_absmax.median():.2f} / "
                  f"|ay_can| 中央 {sub.ay_can_absmax.median():.2f}")

    (args.out / "summary.json").write_text(json.dumps({
        "criteria": {"brake_mps2": args.brake_mps2, "gap_mps2": args.gap_mps2,
                     "min_duration_s": args.min_duration_s},
        "ay_can_users": users,
        "hours": dict(hours), "segments": dict(segs),
        "n_windows": int(len(win)), "events": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
