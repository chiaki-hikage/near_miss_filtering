#!/usr/bin/env python3
"""ヒヤリハット候補の抽出を実行する。

使い方:
  # comma2k19
  python scripts/run_detection.py raw_data/Chunk_1 --out out/chunk1

  # commaCarSegments (先に scripts/fetch_car_segments.py で取得しておく)
  python scripts/run_detection.py raw_data/comma_car_segments \
      --dataset comma_car_segments --platform TOYOTA_RAV4_TSS2 --out out/rav4_tss2

  <データルート> は comma2k19 ではチャンクのディレクトリ、
  commaCarSegments ではキャッシュのディレクトリ。

出力 (--out 配下):
  candidates.csv  確認単位となる候補区間。severity の降順
  events.csv      個々の検出。判定根拠 (trigger_rule) つき
  segments.csv    走査したセグメントの一覧と処理状況
  run_meta.json   実行条件
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from near_miss.config import (
    DEFAULT_DETECTION,
    DEFAULT_VEHICLE_DIR,
    config_hash,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.pipeline import run_source
from near_miss.sources import car_segments_source, comma1m_source, comma2k19_source


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_root", type=Path, help="データセットの所在")
    p.add_argument("--dataset", default="comma2k19",
                   choices=("comma2k19", "comma_car_segments", "comma1M"))
    p.add_argument("--platform", default="TOYOTA_RAV4_TSS2", help="commaCarSegments の車種キー")
    p.add_argument("--out", type=Path, default=Path("out"), help="出力ディレクトリ")
    p.add_argument("--config", type=Path, default=DEFAULT_DETECTION, help="検出設定 YAML")
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR, help="車種設定ディレクトリ")
    p.add_argument("--dataset-config", type=Path, default=Path("configs/datasets/comma1m.yaml"),
                   help="データセット固有の設定 (comma1M のみ)")
    p.add_argument("--stage", type=int, default=2, choices=(1, 2), help="1 = processed_log のみ, 2 = raw_can も使う")
    p.add_argument("--limit-drives", type=int, default=None, help="先頭 N ドライブだけ処理する (動作確認用)")
    p.add_argument("--limit-segments", type=int, default=None, help="先頭 N セグメントだけ処理する")
    p.add_argument("--top", type=int, default=20, help="標準出力に出す候補の件数")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    detection_cfg = load_yaml(args.config)
    vehicle_configs = load_vehicle_configs(args.vehicles)
    cfg_hash = config_hash(detection_cfg, [v.raw for v in vehicle_configs])

    if args.dataset == "comma2k19":
        source = comma2k19_source(args.data_root, vehicle_configs)
    elif args.dataset == "comma1M":
        ds_cfg = load_yaml(args.dataset_config) if args.dataset_config.exists() else {}
        source = comma1m_source(args.data_root, vehicle_configs,
                                localizer_cfg=ds_cfg.get("localizer"))
    else:
        source = car_segments_source(args.data_root, args.platform, vehicle_configs)

    print(f"データセット : {source.name}" + (f"  ({args.platform})" if args.dataset != "comma2k19" else ""))
    print(f"データルート : {args.data_root}")
    print(f"セグメント数 : {len(source.refs)}")
    print(f"検出設定     : {args.config}  (config_hash={cfg_hash})")
    print(f"車種設定     : {[v.name for v in vehicle_configs]}")
    print(f"段階         : stage {args.stage}")
    print()

    tables = run_source(
        source,
        detection_cfg=detection_cfg,
        vehicle_configs=vehicle_configs,
        stage=args.stage,
        limit_drives=args.limit_drives,
        limit_segments=args.limit_segments,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = args.out / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {path}  {len(df)} 行")

    meta = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset": source.name,
        "data_root": str(args.data_root),
        "source_meta": source.meta,
        "config": str(args.config),
        "config_hash": cfg_hash,
        "stage": args.stage,
        "n_segments_scanned": int(tables["segments"]["n_segments"].sum()) if not tables["segments"].empty else 0,
        "n_events": len(tables["events"]),
        "n_candidates": len(tables["candidates"]),
    }
    (args.out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    cands = tables["candidates"]
    print()
    if cands.empty:
        print("候補は検出されませんでした。")
        return 0

    cols = [c for c in ("drive_id", "segment", "t_in_segment_s", "duration_s", "severity", "event_types",
                        "ttc_s_min", "thw_s_min", "ax_mps2_min", "v_mps_mean", "op_tx_mean", "op_engaged_mean")
            if c in cands.columns]
    print(f"上位 {min(args.top, len(cands))} 件:")
    with_pd = cands.head(args.top)[cols].copy()
    for c in with_pd.columns:
        if with_pd[c].dtype.kind == "f":
            with_pd[c] = with_pd[c].round(2)
    print(with_pd.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
