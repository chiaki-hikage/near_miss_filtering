#!/usr/bin/env python3
"""VLM に投げるリクエストを組み立てる (Phase 1 段階 1)。

GPU は要らない。Mac でも EC2 でも同じものが出る。

    labels.csv (人手確認済み 32 件)
        -> 必要なセグメントを JPEG に展開 (フレームキャッシュ)
        -> 20 Hz グリッド + compute_features   ※既存の関数をそのまま使う
        -> モード A (一括) と モード B (オンライン) のリクエスト JSONL

**人手ラベル (risky) はリクエストに入れない。** 採点時に event_id で突き合わせる。
入れてしまうと prompt に混ざる事故が起きうる。

**候補区間の集約値もモード B には入れない。** 与えた瞬間に未来を知ることになる。
モード A は区間全体を見るのが前提なので極値の要約を付けてよい。

使い方:
  uv run python scripts/build_vlm_inputs.py --dry-run     # 何を作るか見るだけ
  uv run python scripts/build_vlm_inputs.py               # フレーム展開 + JSONL
  uv run python scripts/build_vlm_inputs.py --no-frames   # JSONL だけ作り直す
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from near_miss.config import (
    DEFAULT_DETECTION,
    DEFAULT_VEHICLE_DIR,
    config_hash,
    find_vehicle_config,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.features import compute_features
from near_miss.io import comma2k19
from near_miss.io.canonical import concat_segments
from near_miss.pipeline import split_contiguous
from near_miss.signals import to_grid
from near_miss.vlm import frames as fr
from near_miss.vlm.context import make_context
from near_miss.vlm.windows import (
    available_segments,
    episodes_from_labels,
    timeline_available,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, default=Path("out/chunk1/labels.csv"))
    p.add_argument("--data-root", type=Path, default=Path("raw_data/Chunk_1"))
    p.add_argument("--out", type=Path, default=Path("out/chunk1/vlm"))
    p.add_argument("--cache", type=Path, default=Path("out/chunk1/vlm/framecache"))
    p.add_argument("--config", type=Path, default=Path("configs/vlm.yaml"))
    p.add_argument("--detection", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--no-frames", action="store_true", help="フレーム展開を飛ばす")
    p.add_argument("--hwaccel", action="store_true", help="NVDEC を使う (EC2)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    det = load_yaml(args.detection)
    vehicles = load_vehicle_configs(args.vehicles)

    if not args.data_root.is_dir():
        raise SystemExit(
            f"データ置き場がありません: {args.data_root}\n"
            "  scripts/list_required_segments.py で要るセグメントを確認してください")
    labels = pd.read_csv(args.labels)
    eps = episodes_from_labels(labels, cfg)
    print(f"エピソード {len(eps)} 件 "
          f"(positive {sum(e.risky for e in eps)} / negative {sum(not e.risky for e in eps)})")

    # --- 1. 各エピソードの評価時刻と必要セグメント -----------------------
    plans = []
    n_skipped = 0
    avail_cache: dict[str, set[int]] = {}
    for ep in eps:
        avail = avail_cache.setdefault(
            ep.drive_id, available_segments(args.data_root, ep.drive_id))
        # 評価時刻そのものが手元にあれば採る。映像窓が満たない時刻は
        # 短い窓のまま渡し、その事実を記録する (min_video_frames)。
        times, tr = timeline_available(ep, cfg, avail)
        if not times:
            print(f"  {ep.event_id}: {ep.drive_id.split('|')[-1]} の "
                  f"seg{ep.segment} に映像が無いので飛ばします")
            n_skipped += 1
            continue
        segs = fr.needed_segments(ep, cfg, times) & avail
        plans.append({"ep": ep, "times": times, "trunc": tr,
                      "segments": sorted(segs), "avail": avail})

    if n_skipped:
        print(f"\n  ** {n_skipped}/{len(eps)} 件を飛ばしました。"
              "映像の配置を確認してください:\n"
              "     uv run python scripts/list_required_segments.py\n")
    if not plans:
        raise SystemExit(
            "処理できるエピソードがありません。\n"
            f"  {args.data_root}/<drive_id>/<番号>/video.hevc の形で置かれているか"
            "確認してください")
    n_seg = len({(p["ep"].drive_id, s) for p in plans for s in p["segments"]})
    n_pts = sum(len(p["times"]) for p in plans)
    print(f"  評価時刻 {n_pts} 点 / 要するセグメント {n_seg} 本 "
          f"(展開すると約 {n_seg * 20:.0f} MB)")
    cut = [p for p in plans if p["trunc"].any]
    for p in cut:
        t = p["trunc"]
        print(f"  {p['ep'].event_id}: seg{t.missing_segments} が無く "
              f"前 {t.pre_lost_s:.1f} 秒 / 後 {t.post_lost_s:.1f} 秒を切り詰め")

    if args.dry_run:
        print("\n--dry-run のためここまで。")
        return 0

    # --- 2. フレームキャッシュ -------------------------------------------
    if not args.no_frames:
        _build_cache(plans, cfg, args)

    # --- 3. 20 Hz グリッド ------------------------------------------------
    grids = _build_grids(plans, det, vehicles, args)

    # --- 4. リクエスト ----------------------------------------------------
    cfg_hash = config_hash(cfg, det)
    n = _write_requests(plans, grids, cfg, args, cfg_hash)
    print(f"\nconfig_hash {cfg_hash}")
    print(f"出力: {args.out / 'requests_mode_a.jsonl'} / "
          f"{args.out / 'requests_mode_b.jsonl'}  計 {n} 件")
    return 0


def _build_cache(plans, cfg, args) -> None:
    todo = sorted({(p["ep"].drive_id, s) for p in plans for s in p["segments"]})
    print(f"\nフレーム展開: {len(todo)} セグメント")
    t0 = time.perf_counter()
    for i, (drive_id, seg) in enumerate(todo, 1):
        out_dir = fr.segment_dir(args.cache, drive_id, seg)
        if out_dir.is_dir() and list(out_dir.glob("*.jpg")):
            continue
        video = args.data_root / drive_id / str(seg) / "video.hevc"
        if not video.is_file():
            print(f"  映像が無い: {video}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(fr.extract_cmd(video, out_dir, cfg, args.hwaccel),
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"ffmpeg 失敗 {video}: {r.stderr[:400]}")
        if i % 10 == 0:
            print(f"  {i}/{len(todo)} ...", flush=True)
    print(f"  完了 {time.perf_counter() - t0:.0f} 秒")


def _build_grids(plans, det, vehicles, args) -> dict[str, pd.DataFrame]:
    """ドライブごとに 20 Hz グリッドを作る。既存の関数をそのまま通す。

    セグメントは連番のかたまりごとに連結してからグリッドに載せる。
    欠番を跨いで連結すると存在しない区間を内挿してしまう。
    """
    print("\n20 Hz グリッド:")
    refs_all = comma2k19.find_segments(args.data_root)
    by_drive: dict[str, list] = {}
    for r in refs_all:
        by_drive.setdefault(r.drive_id, []).append(r)

    grids: dict[str, pd.DataFrame] = {}
    for drive_id in sorted({p["ep"].drive_id for p in plans}):
        want = {s for p in plans if p["ep"].drive_id == drive_id for s in p["segments"]}
        refs = [r for r in by_drive.get(drive_id, []) if r.index in want]
        if not refs:
            print(f"  {drive_id}: セグメントが見つかりません")
            continue
        vehicle = find_vehicle_config(refs[0].dongle_id, vehicles)
        parts = []
        for block in split_contiguous(refs):
            segs = []
            for ref in block:
                try:
                    segs.append(comma2k19.load_segment(ref, vehicle, with_raw_can=True))
                except Exception as exc:
                    print(f"  読み出し失敗 {ref.segment_id}: {exc}")
            if not segs:
                continue
            merged = concat_segments(segs)
            gs = to_grid(merged, det)
            if gs.df.empty:
                continue
            # レーダを渡さないと thw_s / ttc_s / lead_* が作られない。
            # 車間時間は割り込みの CAN 側の主信号なので、落とすと条件 A が
            # 不当に不利になり、C - A の差を過大評価する。
            parts.append(compute_features(gs, det, radar=merged.radar, vehicle=vehicle).df)
        if parts:
            df = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)
            grids[drive_id] = df
            print(f"  {drive_id.split('|')[-1]}: {len(df):,} 行 "
                  f"({df.t.iloc[0]:.1f} 〜 {df.t.iloc[-1]:.1f})")
    return grids


def _sample_frames(refs, limit: int):
    """フレームが多すぎる場合に等間隔で間引く。末尾は必ず残す。"""
    if len(refs) <= limit:
        return refs
    idx = np.linspace(0, len(refs) - 1, limit).round().astype(int)
    return [refs[i] for i in sorted(set(idx.tolist()))]


def _write_requests(plans, grids, cfg, args, cfg_hash: str) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    fa = (args.out / "requests_mode_a.jsonl").open("w", encoding="utf-8")
    fb = (args.out / "requests_mode_b.jsonl").open("w", encoding="utf-8")
    conds = cfg["conditions"]
    limit = int(cfg["clip_max_frames"])
    n = 0
    missing_frames = 0
    n_partial = n_dropped = 0
    miss_cols: set[str] = set()

    for p in plans:
        ep, times = p["ep"], p["times"]
        df = grids.get(ep.drive_id)
        if df is None:
            continue
        ctx = make_context(df, cfg)
        if getattr(ctx, "missing_columns", None):
            miss_cols.update(ctx.missing_columns)

        # --- モード A: 区間全体を 1 回 ---
        span_refs = [r for t in times
                     for r in fr.frames_available(ep, t, cfg, args.cache, p["avail"])[0]]
        seen, uniq = set(), []
        for r in span_refs:
            key = (r.segment, r.frame)
            if key not in seen:
                seen.add(key)
                uniq.append(r)
        uniq.sort(key=lambda r: r.t)
        clip_refs = _sample_frames(uniq, limit)
        missing_frames += len(fr.missing(clip_refs))

        for cond, spec in conds.items():
            rec = {
                "request_id": f"{ep.event_id}|A|{cond}",
                "event_id": ep.event_id, "mode": "clip", "condition": cond,
                "t_eval": times[-1],
                "frames": [str(r.path) for r in clip_refs] if spec["video"] else [],
                "frame_times": [round(r.t - times[0], 2) for r in clip_refs] if spec["video"] else [],
                # モード A は区間全体を見るのが前提。末尾 6 秒だけでは
                # 29 秒の候補の大半が見えない。極値の要約もここでは許される。
                "can_text": ctx.span(times[0], times[-1]).text if spec["can"] else "",
                "can_extremes": ctx.extremes(times[0], times[-1]) if spec["can"] else "",
                "span_s": round(times[-1] - times[0], 2),
                "hint": ep.event_types if spec["hint"] else "",
                "guard_s": ctx.guard_s, "context_mode": ctx.mode,
                "config_hash": cfg_hash,
            }
            fa.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

        # --- モード B: 評価時刻ごと。条件 C のみ ---
        n_full = int(round(float(cfg["input"]["window_video_s"])
                           * float(cfg["input"]["video_fps"])))
        min_frames = int(cfg["input"]["min_video_frames"])
        for t in times:
            refs, partial = fr.frames_available(ep, t, cfg, args.cache, p["avail"])
            if len(refs) < min_frames:
                n_dropped += 1
                continue
            if partial:
                n_partial += 1
            missing_frames += len(fr.missing(refs))
            r = ctx.at(t)
            rec = {
                "request_id": f"{ep.event_id}|B|{t:.2f}",
                "event_id": ep.event_id, "mode": "online", "condition": "C",
                "t_eval": round(t, 3), "t_rel": round(t - times[0], 2),
                "frames": [str(x.path) for x in refs],
                "frame_times": [round(x.t - t, 2) for x in refs],
                # 窓が満たなかったか。採点側で層別するために必ず載せる。
                "n_frames": len(refs),
                "hist_s": round(t - refs[0].t, 2) if refs else 0.0,
                "partial_window": partial,
                "can_text": r.text,
                "can_max_source_t": None if not np.isfinite(r.max_source_t) else round(r.max_source_t, 3),
                "hint": "",
                "guard_s": ctx.guard_s, "context_mode": ctx.mode,
                "config_hash": cfg_hash,
            }
            fb.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    fa.close()
    fb.close()
    if n_partial or n_dropped:
        print(f"\n  映像窓が満たなかった時刻: {n_partial} 点 "
              f"(短い窓のまま渡し partial_window に記録)")
        if n_dropped:
            print(f"  フレームが 1 枚も無く落とした時刻: {n_dropped} 点")
    if miss_cols:
        print(f"\n  ** 設定にあるがグリッドに無い列: {sorted(miss_cols)}")
        print("     モデルに渡らないので、条件 A の不利につながります")
    if missing_frames:
        print(f"\n  ** 参照先に無いフレームが {missing_frames} 枚あります "
              "(--no-frames で作り直すか、キャッシュを確認してください)")
    return n


if __name__ == "__main__":
    raise SystemExit(main())
