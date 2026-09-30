#!/usr/bin/env python3
"""加速度センサの符号を車両 (dongle) ごとに判定して、符号表を書き出す。

同じ車種キーでも個体によって accel_x / accel_y の向きが逆になる
(src/near_miss/sign_calibration.py)。キャッシュにあるその車種の全セグメントを
**補正を掛けずに**読み、輪速の Δv と v × ヨーレートを基準に符号を決める。

  uv run python scripts/validate/calibrate_accel_sign.py --platform TOYOTA_HIGHLANDER_TSS2 -w 8

出力
  data/calibration/accel_sign/<車種>.csv   1 行 1 台。符号 (+1 / -1 / 0=不明) と相関
  標準出力                                  判定の内訳と、VIN 型式との対応

書き出した表を使うには、車種設定に次を足す:
  calibration:
    accel_sign: data/calibration/accel_sign/<車種>.csv
    on_missing: drop      # 表に無い車両は加速度を使わない
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from near_miss.config import (
    DEFAULT_DETECTION,
    DEFAULT_VEHICLE_DIR,
    find_vehicle_config_for_platform,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.sign_calibration import AXES, VehicleSignStats, decide, decide_from_corr, segment_pairs

_S: dict = {}


def init(platform: str, vdir: Path, cfg: dict) -> None:
    from near_miss.io import comma_car_segments as ccs

    _S["ccs"] = ccs
    _S["vehicle"] = find_vehicle_config_for_platform(platform, load_vehicle_configs(vdir))
    _S["cfg"] = cfg


def run_dongle(args) -> VehicleSignStats:
    dongle, refs = args
    ccs, veh, cfg = _S["ccs"], _S["vehicle"], _S["cfg"]
    st = VehicleSignStats(dongle_id=dongle)
    for ref in refs:
        try:
            # 素の値を読む。補正済みの値から符号を決めると循環する。
            sd = ccs.load_segment(ref, veh, with_raw_can=True, with_radar=False,
                                  apply_calibration=False)
        except Exception:
            continue
        vin = str(sd.meta.get("car_params", {}).get("carVin", "") or "")
        st.add(segment_pairs(sd.channels, cfg), vin[:11])
    return st


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--platform", required=True)
    p.add_argument("-w", "--workers", type=int, default=8)
    p.add_argument("--config", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=None,
                   help="既定 data/calibration/accel_sign/<車種>.csv")
    p.add_argument("--from-table", action="store_true",
                   help="データを読み直さず、既存の符号表の相関と対の数から判定だけやり直す "
                        "(判定の規則や閾値を変えたとき用)")
    return p.parse_args()


def main() -> int:
    from near_miss.io import comma_car_segments as ccs

    args = parse_args()
    cfg = load_yaml(args.config)["accel_sign"]
    if find_vehicle_config_for_platform(args.platform, load_vehicle_configs(args.vehicles)) is None:
        raise SystemExit(f"車種設定がありません: {args.platform}")

    out = args.out or Path("data/calibration/accel_sign") / f"{args.platform}.csv"
    if args.from_table:
        df = pd.read_csv(out, dtype={"dongle_id": str})
        for i, r in df.iterrows():
            res = decide_from_corr({a: r[f"{a}_corr"] for a in AXES},
                                   {a: int(r[f"{a}_pairs"]) for a in AXES}, cfg)
            for a, (sgn, status) in res.items():
                df.loc[i, f"{a}_sign"], df.loc[i, f"{a}_status"] = sgn, status
        refs = ccs.find_segments(ccs.DEFAULT_CACHE, args.platform)
        return _write_and_report(df, out, args, cfg, len(refs))

    refs = ccs.find_segments(ccs.DEFAULT_CACHE, args.platform)
    by = defaultdict(list)
    for r in refs:
        by[r.dongle_id].append(r)
    tasks = sorted(by.items())
    print(f"{args.platform}: {len(refs)} セグメント / {len(tasks)} 台")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init,
                             initargs=(args.platform, args.vehicles, cfg)) as ex:
        for st in ex.map(run_dongle, tasks, chunksize=2):
            rows.append(decide(st, cfg))

    df = pd.DataFrame(rows).sort_values("dongle_id").reset_index(drop=True)
    return _write_and_report(df, out, args, cfg, len(refs))


def _write_and_report(df: pd.DataFrame, out: Path, args, cfg: dict, n_refs: int) -> int:
    refs = range(n_refs)
    df = df.drop(columns=["vin_model"], errors="ignore")
    for a in AXES:
        df[f"{a}_sign"] = df[f"{a}_sign"].astype(int)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    (out.with_suffix(".meta.txt")).write_text(
        f"platform: {args.platform}\nsegments: {len(refs)}\nvehicles: {len(df)}\n"
        f"generated_at: {datetime.now(timezone.utc).isoformat()}\n"
        f"criteria: {cfg}\n", encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\n判定 (台数)")
    print(pd.crosstab(df.accel_x_status, df.accel_y_status, margins=True).to_string())
    mixed = df[(df.accel_x_sign * df.accel_y_sign) == -1]
    print(f"\nax と ay で符号が食い違う車両: {len(mixed)} 台")
    if len(mixed):
        print(mixed[["dongle_id", "vin", "accel_x_corr", "accel_y_corr"]].to_string(index=False))

    df["vin_model"] = df.vin.str.split("|").str[0].str[3:8]
    t = pd.crosstab(df.vin_model, df.accel_x_status)
    flipped_models = t[t.get("flipped", 0) > 0] if "flipped" in t else t.iloc[0:0]
    print(f"\n反転した車両の VIN 型式 (4〜8 桁):")
    print(flipped_models.to_string() if len(flipped_models) else "  なし")

    unknown = df[(df.accel_x_sign == 0) | (df.accel_y_sign == 0)]
    print(f"\n不明 (加速度を使わない) 車両: {len(unknown)} 台 / "
          f"{int(unknown.segments.sum())} セグメント "
          f"({unknown.segments.sum() / max(df.segments.sum(), 1):.1%})")
    print(f"\n出力: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
