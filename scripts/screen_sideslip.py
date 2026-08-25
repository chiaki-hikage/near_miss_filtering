#!/usr/bin/env python3
"""横滑り候補の 2 段抽出を、キャッシュ済みの実データに通す。

    全データ  ->  1 次通過  ->  2 次通過  ->  最終候補

各段で何が残ったかを数えて出す。取得はしない (キャッシュにあるものだけを使う)。
取得は scripts/fetch_car_segments.py / screen_segments.py で先に行うこと。

  # commaCarSegments (キャッシュにあるもの全部)
  uv run python scripts/screen_sideslip.py --platform TOYOTA_RAV4_TSS2 --out out/sideslip_rav4_tss2

  # comma2k19
  uv run python scripts/screen_sideslip.py --comma2k19 raw_data/Chunk_1 --out out/sideslip_chunk1

出力
  candidates.csv   最終候補。beta の大きい順
  counts.json      各段の件数
  stage_samples.csv.gz  1 次を通ったサンプルの明細 (--dump-stage1 のとき)
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from near_miss.config import (
    DEFAULT_DETECTION,
    DEFAULT_VEHICLE_DIR,
    config_hash,
    find_vehicle_config,
    find_vehicle_config_for_platform,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.features import compute_features
from near_miss.io import comma2k19, comma_car_segments as ccs
from near_miss.io.canonical import concat_segments, group_by_drive
from near_miss.pipeline import _annotate_segment, split_contiguous
from near_miss.signals import to_grid
from near_miss.sideslip import (
    CHECKS,
    FilterCounts,
    candidates_to_frame,
    find_sideslip_candidates,
    stage1_mask,
)

# 1 次を通ったサンプルの明細に残す列。なぜ通り、なぜ落ちたかを後から追えるように。
DUMP_COLUMNS = (
    "t", "v_mps", "ax_mps2", "yaw_rate_dps", "steer_deg_s", "steer_rate_dps",
    "ay_can_mps2", "ay_kin_mps2", "beta_model_deg", "beta_sigma", "beta_noise_deg",
    "beta_expected_deg", "beta_excess_deg", "beta_rate_dps", "yaw_residual_sigma",
    "net_heading_win_deg", "ws_spread_excess_mps",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--platform", help="commaCarSegments の車種キー")
    g.add_argument("--comma2k19", type=Path, help="comma2k19 のチャンク")
    p.add_argument("--limit", type=int, default=None, help="使うセグメント数の上限")
    p.add_argument("--select", choices=("cache", "catalog"), default="cache",
                   help="cache = キャッシュにあるものを名前順に (既定) / "
                        "catalog = database.json から決まった順で選ぶ。"
                        "catalog は手元に何が入っているかに依存しないので、"
                        "別のマシンで同じ結果を出したいときに使う")
    p.add_argument("--per-route", type=int, default=10,
                   help="--select catalog のときの 1 ルートあたりの連続セグメント数")
    p.add_argument("--cache", type=Path, default=ccs.DEFAULT_CACHE)
    p.add_argument("--out", type=Path, default=Path("out/sideslip"))
    p.add_argument("--config", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--min-speed", type=float, default=None,
                   help="適用範囲の下限を設定から変えて試す (既定は configs/detection.yaml)")
    p.add_argument("--dump-stage1", action="store_true",
                   help="1 次を通ったサンプルの明細も書き出す")
    return p.parse_args()


def _percent(n: int, d: int) -> str:
    return f"{n / d:.4%}" if d else "-"


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.min_speed is not None:
        cfg["sideslip"]["min_speed_mps"] = float(args.min_speed)
    vehicles = load_vehicle_configs(args.vehicles)

    if args.platform:
        vehicle = find_vehicle_config_for_platform(args.platform, vehicles)
        if vehicle is None:
            raise SystemExit(f"車種設定がありません: {args.platform}")
        refs = ccs.find_segments(args.cache, args.platform)
        if args.select == "catalog":
            # database.json の並びから決まった順で選ぶ。手元のキャッシュに
            # 何が入っているかで結果が変わらないので、別のマシンとの突き合わせに使える。
            want = ccs.select_segments(args.platform, limit=args.limit,
                                       per_route=args.per_route, cache_dir=args.cache)
            paths = {ccs.local_path(n, args.cache) for n in want}
            refs = [r for r in refs if r.path in paths]
            if len(refs) < len(want):
                print(f"注意: 選ばれた {len(want)} 本のうち手元にあるのは {len(refs)} 本です。"
                      f"\n      先に scripts/fetch_car_segments.py {args.platform} "
                      f"--limit {args.limit} --per-route {args.per_route} を実行してください。")
        load = lambda ref: ccs.load_segment(ref, vehicle, with_raw_can=True)  # noqa: E731
        dataset, label, video_fps = ccs.DATASET, args.platform, None
    else:
        refs = comma2k19.find_segments(args.comma2k19)
        if not refs:
            raise SystemExit(f"セグメントがありません: {args.comma2k19}")
        vehicle = find_vehicle_config(refs[0].dongle_id, vehicles)
        if vehicle is None:
            raise SystemExit(f"車種設定がありません: dongle={refs[0].dongle_id}")
        load = lambda ref: comma2k19.load_segment(ref, vehicle, with_raw_can=True)  # noqa: E731
        dataset, label, video_fps = "comma2k19", str(args.comma2k19), 20.0

    if vehicle.sideslip_ay_coeff() is None or vehicle.center_to_rear_m() is None:
        raise SystemExit(f"{vehicle.name} は beta を出せません (geometry が未確定)")
    if vehicle.geometry_value("yaw_rate_noise_dps") is None:
        raise SystemExit(
            f"{vehicle.name} に yaw_rate_noise_dps がありません。"
            "先に scripts/calibrate_beta_noise.py を実行してください"
        )

    if args.limit is not None and args.select == "cache":
        refs = refs[: args.limit]
    if not refs:
        # 何も無いまま 0 件の表を出すと「流したのに何も無かった」と読めてしまう。
        # 新しい環境で最初に踏むのはここなので、置き場と取り方を出して止める。
        where = args.cache if args.platform else args.comma2k19
        raise SystemExit(
            f"セグメントが 1 つもありません: {where}\n"
            + (f"  取得: uv run python scripts/fetch_car_segments.py {args.platform} "
               f"--limit 30 --per-route 10\n" if args.platform else "")
            + "  置き場は docs/environment.md の「データの配置」を参照"
        )
    cfg_hash = config_hash(cfg, [v.raw for v in vehicles])
    s = cfg["sideslip"]

    print(f"データ   : {dataset} / {label}")
    print(f"車種設定 : {vehicle.name}")
    print(f"セグメント: {len(refs)}")
    print(f"設定     : 適用 v>={s['min_speed_mps']} m/s / "
          f"1次 |beta|>={s['stage1']['min_beta_deg']} deg かつ "
          f">={s['stage1']['min_beta_sigma']} sigma")
    print(f"config_hash: {cfg_hash}\n")

    total = FilterCounts()
    cand_frames: list[pd.DataFrame] = []
    dumps: list[pd.DataFrame] = []
    n_seg, n_block, t0 = 0, 0, time.perf_counter()

    for drive_id, drefs in group_by_drive(refs).items():
        for block in split_contiguous(drefs):
            segs, spans = [], []
            for ref in block:
                try:
                    sd = load(ref)
                except Exception as exc:
                    print(f"  読み出し失敗 {ref.segment_id}: {exc}")
                    continue
                a, b = sd.t_span
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue
                spans.append((ref.index, float(a), float(b)))
                segs.append(sd)
            if not segs:
                continue
            merged = concat_segments(segs)
            gs = to_grid(merged, cfg)
            if gs.df.empty:
                continue
            gs.meta["segment_spans"] = spans
            gs = compute_features(gs, cfg, radar=merged.radar, vehicle=vehicle)

            cands, counts = find_sideslip_candidates(gs, cfg)
            total.add(counts)
            n_seg += len(segs)
            n_block += 1

            if cands:
                f = candidates_to_frame(cands, cfg_hash)
                f = _annotate_segment(f, spans, video_fps)
                f.insert(0, "drive_id", drive_id)
                f.insert(0, "dataset", dataset)
                cand_frames.append(f)
            if args.dump_stage1:
                m1, _ = stage1_mask(gs.df, cfg)
                if m1.any():
                    cols = [c for c in DUMP_COLUMNS if c in gs.df.columns]
                    d = gs.df.loc[m1, cols].copy()
                    d.insert(0, "drive_id", drive_id)
                    dumps.append(d)
            if n_block % 50 == 0:
                print(f"  {n_seg} セグメント / {time.perf_counter() - t0:.0f} 秒 / "
                      f"候補 {sum(len(f) for f in cand_frames)} 件", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    cd = pd.concat(cand_frames, ignore_index=True) if cand_frames else pd.DataFrame()
    if not cd.empty:
        cd = cd.reindex(cd.beta_peak_deg.abs().sort_values(ascending=False).index).reset_index(drop=True)
    cd.to_csv(args.out / "candidates.csv", index=False)
    if dumps:
        with gzip.open(args.out / "stage1_samples.csv.gz", "wt", encoding="utf-8") as f:
            pd.concat(dumps, ignore_index=True).to_csv(f, index=False)

    meta = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset, "label": label, "vehicle": vehicle.name,
        "config_hash": cfg_hash, "n_segments": n_seg, "n_blocks": n_block,
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "counts": asdict(total),
        "sideslip_config": s,
    }
    (args.out / "counts.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---- 報告 ------------------------------------------------------------
    n = total.n_samples
    print("\n" + "=" * 72)
    print(f"処理 {n_seg} セグメント / {total.n_hours:.1f} 時間 / "
          f"{(time.perf_counter() - t0) / 60:.1f} 分")
    print("=" * 72)
    print(f"{'段階':<26}{'サンプル':>12}{'全体比':>11}{'区間':>8}")
    print(f"{'全データ':<24}{n:>12,}{'100.0000%':>11}{'':>8}")
    print(f"{'  beta が計算できた':<22}{total.n_beta_valid:>12,}"
          f"{_percent(total.n_beta_valid, n):>11}{'':>8}")
    print(f"{'  適用範囲 (v>=' + str(s['min_speed_mps']) + ')':<21}{total.n_in_range:>12,}"
          f"{_percent(total.n_in_range, n):>11}{'':>8}")
    print(f"{'1 次通過':<24}{total.n_stage1:>12,}{_percent(total.n_stage1, n):>11}"
          f"{total.n_runs_stage1:>8,}")
    for k, v in sorted(total.stage1_by_reason.items(), key=lambda x: -x[1]):
        print(f"{'    理由 ' + k:<24}{v:>12,}{_percent(v, n):>11}{'':>8}")
    print(f"{'2 次通過':<24}{total.n_stage2:>12,}{_percent(total.n_stage2, n):>11}"
          f"{total.n_runs_stage2:>8,}")
    labels = dict(CHECKS) | {"confidence": "信頼度不足"}
    for k, v in sorted(total.stage2_reject.items(), key=lambda x: -x[1]):
        print(f"{'    落ちた: ' + labels.get(k, k):<22}{'':>12}{'':>11}{v:>8,}")
    print(f"{'最終候補':<24}{'':>12}{'':>11}{total.n_candidates:>8,}")

    if not cd.empty:
        print(f"\n等級別:")
        print(cd.grade.value_counts().sort_index().to_string())
        print("\n信頼度の内訳 (立った項目):")
        print(cd.confidence_items.value_counts().to_string())
        cols = [c for c in ("grade", "confidence", "confidence_items", "drive_id", "segment",
                            "peak_t_in_segment_s", "beta_peak_deg", "beta_excess_peak_deg",
                            "beta_rate_peak_dps", "ay_peak_mps2", "v_at_peak_mps",
                            "duration_s", "corroboration") if c in cd.columns]
        print(f"\n上位 {min(20, len(cd))} 件:")
        print(cd[cols].head(20).round(2).to_string(index=False))
    else:
        print("\n最終候補はありませんでした。")
    print(f"\n出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
