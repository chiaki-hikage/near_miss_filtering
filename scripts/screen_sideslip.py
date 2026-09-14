#!/usr/bin/env python3
"""横滑りの抽出を、キャッシュ済みの実データに通す。**独立した 2 枝**を同時に流す。

  枝 1  beta の 2 段フィルタ (sideslip.py)
        全データ -> 1 次通過 -> 2 次通過 -> 最終候補
        モデル推定の beta を使う。適用下限 min_speed_mps がある。

  枝 2  直接証拠 (direct_slip.py)
        slip_warn / ABS / VSC / 輪速の開き を**種類ごとに分けて**拾う。
        モデルを通さないので速度の下限が要らない。
        等級は確定証拠が何系統重なったかで決まる
        (A_確定 2 系統以上 / B_単一 1 系統 / C_候補 輪速の開きだけ)。

2 枝は互いに影響しない。最後に両者の重なりを出す。

各段で何が残ったかを数えて出す。取得はしない (キャッシュにあるものだけを使う)。
取得は scripts/fetch_car_segments.py / screen_segments.py で先に行うこと。

  # commaCarSegments (キャッシュにあるもの全部)
  uv run python scripts/screen_sideslip.py --platform TOYOTA_RAV4_TSS2 --out out/sideslip_rav4_tss2

  # comma2k19
  uv run python scripts/screen_sideslip.py --comma2k19 raw_data/Chunk_1 --out out/sideslip_chunk1

出力
  candidates.csv   枝 1 の最終候補。beta の大きい順
  direct_slip.csv  枝 2 の候補。証拠ごとの有無を個別の列で持つ
  overlap.csv      2 枝の重なり (区間の重複で突き合わせたもの)
  counts.json      各段の件数と、証拠の組み合わせ別の内訳
  stage1_samples.csv.gz  枝 1 の 1 次を通ったサンプルの明細 (--dump-stage1 のとき)
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
from near_miss.parallel import (
    DATASET_CAR_SEGMENTS,
    DATASET_COMMA2K19,
    build_tasks,
    map_drives,
    resolve_workers,
)
from near_miss.direct_slip import (
    ALL_EVIDENCE,
    CONTEXT,
    DIRECT,
    DirectSlipCounts,
    events_to_frame,
)
from near_miss.sideslip import CHECKS, FilterCounts, candidates_to_frame

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
    p.add_argument("--workers", type=int, default=1,
                   help="ドライブ単位の並列数。1 で逐次 (既定)。0 以下で CPU 数。"
                        "判定は変わらない。docs/environment.md の「並列実行」を参照")
    p.add_argument("--min-speed", type=float, default=None,
                   help="適用範囲の下限を設定から変えて試す (既定は configs/detection.yaml)")
    p.add_argument("--dump-stage1", action="store_true",
                   help="1 次を通ったサンプルの明細も書き出す")
    return p.parse_args()


def _percent(n: int, d: int) -> str:
    return f"{n / d:.4%}" if d else "-"


def _overlap(beta: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    """2 枝の候補を、同じドライブ・同じセグメントの区間の重なりで突き合わせる。

    どちらの枝も t_start / t_end はブロック先頭からの相対秒なので、
    同じドライブの同じブロックであればそのまま比べられる。
    片方にしか出ないものも行として残す。**どちらが取り、どちらが逃したか**が
    分かる形にしておく。
    """
    cols = ["drive_id", "segment", "t_start", "t_end"]
    rows: list[dict] = []
    b = beta[[c for c in cols if c in beta.columns]].copy() if not beta.empty else pd.DataFrame()
    d = direct[[c for c in cols if c in direct.columns]].copy() if not direct.empty else pd.DataFrame()
    if not b.empty:
        b["idx_b"] = range(len(b))
    if not d.empty:
        d["idx_d"] = range(len(d))

    matched_b: set[int] = set()
    matched_d: set[int] = set()
    if not b.empty and not d.empty:
        for r in b.itertuples():
            same = d[(d.drive_id == r.drive_id) & (d.segment == r.segment)]
            for q in same.itertuples():
                if r.t_start <= q.t_end and q.t_start <= r.t_end:
                    matched_b.add(r.idx_b)
                    matched_d.add(q.idx_d)
                    rows.append({"drive_id": r.drive_id, "segment": r.segment,
                                 "t_start": min(r.t_start, q.t_start),
                                 "t_end": max(r.t_end, q.t_end),
                                 "beta": True, "direct": True, "both": True})
    if not b.empty:
        for r in b.itertuples():
            if r.idx_b not in matched_b:
                rows.append({"drive_id": r.drive_id, "segment": r.segment,
                             "t_start": r.t_start, "t_end": r.t_end,
                             "beta": True, "direct": False, "both": False})
    if not d.empty:
        for q in d.itertuples():
            if q.idx_d not in matched_d:
                rows.append({"drive_id": q.drive_id, "segment": q.segment,
                             "t_start": q.t_start, "t_end": q.t_end,
                             "beta": False, "direct": True, "both": False})
    if not rows:
        return pd.DataFrame(columns=["drive_id", "segment", "t_start", "t_end",
                                     "beta", "direct", "both"])
    return pd.DataFrame(rows).sort_values(["both", "drive_id", "segment", "t_start"],
                                          ascending=[False, True, True, True])


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
        dataset, label, video_fps = DATASET_CAR_SEGMENTS, args.platform, None
    else:
        refs = comma2k19.find_segments(args.comma2k19)
        if not refs:
            raise SystemExit(f"セグメントがありません: {args.comma2k19}")
        vehicle = find_vehicle_config(refs[0].dongle_id, vehicles)
        if vehicle is None:
            raise SystemExit(f"車種設定がありません: dongle={refs[0].dongle_id}")
        dataset, label, video_fps = DATASET_COMMA2K19, str(args.comma2k19), 20.0

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

    workers = resolve_workers(args.workers)

    print(f"データ   : {dataset} / {label}")
    print(f"車種設定 : {vehicle.name}")
    print(f"セグメント: {len(refs)}")
    print(f"設定     : 適用 v>={s['min_speed_mps']} m/s / "
          f"1次 |beta|>={s['stage1']['min_beta_deg']} deg かつ "
          f">={s['stage1']['min_beta_sigma']} sigma")
    print(f"並列     : ドライブ単位 worker {workers}"
          + ("  (逐次)" if workers == 1 else ""))
    print(f"config_hash: {cfg_hash}\n")

    total = FilterCounts()
    dtotal = DirectSlipCounts()
    cand_frames: list[pd.DataFrame] = []
    direct_frames: list[pd.DataFrame] = []
    dumps: list[pd.DataFrame] = []
    n_seg, n_block, t0 = 0, 0, time.perf_counter()

    # ドライブ単位に分けて流す。1 ドライブの中は連番を連結してから処理するので、
    # 60 秒境界を跨ぐ事象の扱いは逐次実行と変わらない。
    tasks = build_tasks(refs)
    outcomes = map_drives(
        tasks, cfg, vehicle, dataset,
        workers=workers,
        dump_stage1=args.dump_stage1,
        dump_columns=DUMP_COLUMNS,
    )
    # 結果は投入順に返ってくる。並べ替えないこと (worker 数で行の並びが変わる)。
    for out in outcomes:
        for msg in out.errors:
            print(f"  {msg}")
        total.add(out.counts)
        dtotal.add(out.direct_counts)
        for blk in out.blocks:
            n_seg += blk.n_segments
            n_block += 1
            if blk.candidates:
                f = candidates_to_frame(blk.candidates, cfg_hash)
                f = _annotate_segment(f, blk.spans, video_fps)
                f.insert(0, "drive_id", out.drive_id)
                f.insert(0, "dataset", dataset)
                cand_frames.append(f)
            if blk.direct:
                f = events_to_frame(blk.direct, cfg_hash)
                f = _annotate_segment(f, blk.spans, video_fps)
                f.insert(0, "drive_id", out.drive_id)
                f.insert(0, "dataset", dataset)
                direct_frames.append(f)
            if blk.dump is not None:
                dumps.append(blk.dump)
        if n_block % 50 == 0 and n_block:
            print(f"  {n_seg} セグメント / {time.perf_counter() - t0:.0f} 秒 / "
                  f"候補 {sum(len(f) for f in cand_frames)} 件", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    cd = pd.concat(cand_frames, ignore_index=True) if cand_frames else pd.DataFrame()
    if not cd.empty:
        cd = cd.reindex(cd.beta_peak_deg.abs().sort_values(ascending=False).index).reset_index(drop=True)
    cd.to_csv(args.out / "candidates.csv", index=False)

    dd = pd.concat(direct_frames, ignore_index=True) if direct_frames else pd.DataFrame()
    if not dd.empty:
        order = {"A_確定": 0, "B_単一": 1, "C_候補": 2}
        dd = (dd.assign(_o=dd.grade.map(order).fillna(9))
                .sort_values(["_o", "n_direct", "v_max_mps"],
                             ascending=[True, False, False])
                .drop(columns="_o").reset_index(drop=True))
    dd.to_csv(args.out / "direct_slip.csv", index=False)

    # 2 枝の重なり。同じドライブ・同じセグメントで区間が重なるものを対にする。
    ov = _overlap(cd, dd)
    ov.to_csv(args.out / "overlap.csv", index=False)

    if dumps:
        with gzip.open(args.out / "stage1_samples.csv.gz", "wt", encoding="utf-8") as f:
            pd.concat(dumps, ignore_index=True).to_csv(f, index=False)

    meta = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset, "label": label, "vehicle": vehicle.name,
        "config_hash": cfg_hash, "n_segments": n_seg, "n_blocks": n_block,
        "workers": workers,
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "counts": asdict(total),
        "direct_counts": asdict(dtotal),
        "overlap": {
            "beta_candidates": int(len(cd)),
            "direct_events": int(len(dd)),
            "both": int(ov.both.sum()) if not ov.empty else 0,
        },
        "sideslip_config": s,
        "direct_slip_config": cfg.get("direct_slip"),
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

    # ---- 枝 2: 直接証拠 ----------------------------------------------------
    print("\n" + "=" * 72)
    print("枝 2  スリップの直接証拠 (beta を通さない)")
    print("=" * 72)
    print(f"{'信号':<20}{'ある':>8}{'ない':>8}{'立った区間':>12}")
    for name in (*ALL_EVIDENCE, *CONTEXT):
        runs = dtotal.runs_by_evidence.get(name, "-")
        print(f"{name:<20}{dtotal.present.get(name, 0):>8,}"
              f"{dtotal.missing.get(name, 0):>8,}{runs:>12}")
    if dd.empty:
        if not dtotal.present:
            print("\n  この車種には該当する信号がありません "
                  "(comma2k19 の RAV4 には 0x226 / 0x3B7 が無い)。")
        else:
            print("\n  立った区間はありませんでした。")
    else:
        print(f"\n確認単位 {len(dd)} 件")
        print("\n等級別 (確定証拠が何系統重なったか)")
        for g in ("A_確定", "B_単一", "C_候補"):
            n_g = int((dd.grade == g).sum())
            note = {"A_確定": "確定証拠 2 系統以上",
                    "B_単一": "確定証拠 1 系統",
                    "C_候補": "輪速の開きのみ。目視の確認が要る"}[g]
            print(f"  {g:<10}{n_g:>5} 件   {note}")
        print("\n証拠の組み合わせ別")
        for k, n_k in sorted(dtotal.by_combination.items(), key=lambda x: -x[1]):
            print(f"  {k:<40}{n_k:>5} 件")
        cols = [c for c in ("grade", "n_direct", "evidence", "drive_id", "segment",
                            "t_in_segment_s", "duration_s", "slip_warn_peak",
                            "ws_spread_peak_mps", "v_max_mps", "ax_at_peak_mps2",
                            "ay_at_peak_mps2", "steer_at_peak_deg",
                            "beta_model_peak_deg", "tc_disabled") if c in dd.columns]
        print(f"\n上位 {min(20, len(dd))} 件:")
        print(dd[cols].head(20).round(2).to_string(index=False))

    # ---- 2 枝の重なり ------------------------------------------------------
    if not ov.empty:
        both = int(ov.both.sum())
        only_b = int((ov.beta & ~ov.direct).sum())
        only_d = int((~ov.beta & ov.direct).sum())
        print("\n" + "=" * 72)
        print("2 枝の重なり")
        print("=" * 72)
        print(f"  両方が取った        {both:>5} 件")
        print(f"  beta だけが取った    {only_b:>5} 件")
        print(f"  直接証拠だけが取った  {only_d:>5} 件")
        if only_d:
            print("\n  ※ 直接証拠だけの分は beta 側の適用下限 "
                  f"(v >= {s['min_speed_mps']} m/s) や"
                  "\n     2 段フィルタで落ちたもの。両枝を残す理由がここに出る。")

    print(f"\n出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
