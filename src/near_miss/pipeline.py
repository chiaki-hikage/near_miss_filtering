"""抽出処理の組み立て。

stage 2 (既定) は全セグメントを 1 回で通す。raw_can の復号込みでも
セグメントあたり 10 ms 程度 (実測) で、1 チャンク 188 セグメントが数秒で終わる。

当初は 1 段目で粗く絞ってから 2 段目で raw_can を復号する構成にしていたが、
横方向の指標 (ay_kin / lat_jerk) が YAW_RATE 由来になったことで
1 段目では操舵系のイベントを判定できなくなり、絞り込みで取りこぼしが出る。
復号が十分に安いため、絞り込みをやめて 1 回で通す。

stage 1 は processed_log だけで回すモードとして残してある。
raw_can が無い場合や、縦方向と車間だけを見たい場合に使う。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import VehicleConfig, config_hash, find_vehicle_config
from .detectors import Event, detect_all
from .features import compute_features
from .io.canonical import SegmentRef, concat_segments, group_by_drive
from .sources import SegmentSource, comma2k19_source
from .scoring import Candidate, build_candidates, candidates_to_frame, events_to_frame
from .signals import GriddedSignals, to_grid

log = logging.getLogger(__name__)


@dataclass
class BlockResult:
    """連続したセグメントのかたまり 1 つ分の処理結果。"""

    gs: GriddedSignals
    events: list[Event]
    candidates: list[Candidate]
    refs: list[SegmentRef]
    segment_spans: list[tuple[int, float, float]] = field(default_factory=list)


def split_contiguous(refs: list[SegmentRef]) -> list[list[SegmentRef]]:
    """セグメント番号が連番になっているかたまりに分ける。

    欠番を跨いで連結すると、存在しない区間を内挿してしまう。
    """
    blocks: list[list[SegmentRef]] = []
    for r in sorted(refs, key=lambda x: x.index):
        if blocks and r.index == blocks[-1][-1].index + 1:
            blocks[-1].append(r)
        else:
            blocks.append([r])
    return blocks


def segment_of_time(spans: list[tuple[int, float, float]], t: float) -> int | None:
    for index, t0, t1 in spans:
        if t0 <= t <= t1:
            return index
    return None


def process_block(
    refs: list[SegmentRef],
    vehicle: VehicleConfig | None,
    cfg: dict[str, Any],
    with_raw_can: bool,
    max_stage: int,
    source: SegmentSource | None = None,
) -> BlockResult | None:
    """連続セグメントを連結して特徴量・イベント・候補まで通す。"""
    load = source.load if source is not None else (
        lambda ref, veh, raw: __import__(
            "near_miss.io.comma2k19", fromlist=["load_segment"]
        ).load_segment(ref, veh, with_raw_can=raw)
    )
    segments = []
    spans: list[tuple[int, float, float]] = []
    for ref in refs:
        try:
            s = load(ref, vehicle, with_raw_can)
        except Exception as exc:  # 1 セグメントの破損で全体を止めない
            log.warning("読み出し失敗 %s: %s", ref.segment_id, exc)
            continue
        t0, t1 = s.t_span
        if not np.isfinite(t0) or not np.isfinite(t1):
            log.warning("有効な時間範囲がありません %s", ref.segment_id)
            continue
        spans.append((ref.index, float(t0), float(t1)))
        segments.append(s)
    if not segments:
        return None

    merged = concat_segments(segments)
    gs = to_grid(merged, cfg)
    if gs.df.empty:
        return None
    gs.meta["segment_spans"] = spans
    gs.meta["segment_indices"] = [r.index for r in refs]

    gs = compute_features(gs, cfg, radar=merged.radar, vehicle=vehicle)
    events = detect_all(gs, cfg, max_stage=max_stage)
    candidates = build_candidates(gs, events, cfg)
    return BlockResult(gs=gs, events=events, candidates=candidates, refs=refs, segment_spans=spans)


# comma2k19 の映像は 20 Hz。動画確認でコマ送りできるようフレーム番号も出す。
# commaCarSegments には映像が無いので、そちらでは video_frame を出さない。
VIDEO_FPS = 20.0


def _annotate_segment(
    df: pd.DataFrame, spans: list[tuple[int, float, float]], video_fps: float | None = VIDEO_FPS
) -> pd.DataFrame:
    """どのセグメントの何秒目かを付ける。

    映像はセグメント単位のファイルなので、ブロック先頭からの相対時刻ではなく
    セグメント先頭からの相対時刻でないと頭出しできない。
    """
    if df.empty:
        return df
    df = df.copy()
    starts = {index: t0 for index, t0, _ in spans}
    seg = [segment_of_time(spans, t) for t in df["t_start"]]
    df["segment"] = seg
    df["t_in_segment_s"] = [
        round(t - starts[i], 2) if i is not None else np.nan for t, i in zip(df["t_start"], seg)
    ]
    if video_fps:
        df["video_frame"] = [
            int(round(v * video_fps)) if np.isfinite(v) else -1 for v in df["t_in_segment_s"]
        ]

    # 最も強いイベントの位置。候補の t_start は前後の余白を含むため、
    # 60 秒境界の直前だと 1 つ前のセグメントに割り当たることがある。
    # 頭出しにはこちらを使う。
    if "t_peak" in df.columns:
        pk = [segment_of_time(spans, t) for t in df["t_peak"]]
        df["peak_segment"] = pk
        df["peak_t_in_segment_s"] = [
            round(t - starts[i], 2) if i is not None else np.nan
            for t, i in zip(df["t_peak"], pk)
        ]
    return df


def run(
    data_root: str | Path,
    detection_cfg: dict[str, Any],
    vehicle_configs: list[VehicleConfig],
    stage: int = 2,
    limit_drives: int | None = None,
) -> dict[str, pd.DataFrame]:
    """comma2k19 のデータルートから候補抽出まで通す (従来どおりの入口)。"""
    return run_source(
        comma2k19_source(data_root, vehicle_configs),
        detection_cfg,
        vehicle_configs,
        stage=stage,
        limit_drives=limit_drives,
    )


def run_source(
    source: SegmentSource,
    detection_cfg: dict[str, Any],
    vehicle_configs: list[VehicleConfig],
    stage: int = 2,
    limit_drives: int | None = None,
    limit_segments: int | None = None,
) -> dict[str, pd.DataFrame]:
    """任意のデータセットから候補抽出まで通す。

    戻り値は "segments" / "events" / "candidates" の 3 つの表。
    セグメントはドライブごとにまとめ、連番のかたまり単位で連結して処理する。
    """
    cfg_hash = config_hash(detection_cfg, [v.raw for v in vehicle_configs])
    if stage < 2 and not source.supports_stage1:
        raise ValueError(f"{source.name} は stage 1 に対応していません (生 CAN が必須)")

    refs = source.refs if limit_segments is None else source.refs[:limit_segments]
    drives = group_by_drive(refs)
    if limit_drives is not None:
        drives = dict(list(drives.items())[:limit_drives])

    seg_rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []
    cand_frames: list[pd.DataFrame] = []

    with_raw_can = stage >= 2
    for drive_id, drive_refs in drives.items():
        vehicle = source.vehicle_for(drive_refs[0])
        if vehicle is None:
            log.warning("対象外の車両です dongle=%s drive=%s", drive_refs[0].dongle_id, drive_id)
            seg_rows.append(
                {"dataset": source.name, "drive_id": drive_id, "n_segments": len(drive_refs),
                 "status": "skipped:unknown_vehicle"}
            )
            continue

        for block in split_contiguous(drive_refs):
            res = process_block(
                block, vehicle, detection_cfg,
                with_raw_can=with_raw_can, max_stage=stage, source=source,
            )
            if res is None:
                seg_rows.append({"dataset": source.name, "drive_id": drive_id,
                                 "n_segments": len(block), "status": "skipped:empty"})
                continue

            event_frames.append(
                _annotate_segment(
                    events_to_frame(res.gs, res.events, cfg_hash), res.segment_spans, source.video_fps
                )
            )
            cand_frames.append(
                _annotate_segment(
                    candidates_to_frame(res.candidates, res.gs, cfg_hash), res.segment_spans, source.video_fps
                )
            )
            seg_rows.append(
                {
                    "dataset": source.name,
                    "drive_id": drive_id,
                    "segments": ",".join(str(r.index) for r in block),
                    "n_segments": len(block),
                    "duration_s": float(res.gs.t[-1] - res.gs.t[0]),
                    "raw_can_loaded": res.gs.raw_can_loaded,
                    "n_events": len(res.events),
                    "n_candidates": len(res.candidates),
                    "op_ratio": _engagement_ratio(res.gs.df),
                    "status": "ok",
                }
            )

    tables = _collect(seg_rows, event_frames, cand_frames)
    for df in tables.values():
        if not df.empty and "dataset" not in df.columns:
            df.insert(0, "dataset", source.name)
    return tables


def _engagement_ratio(df: pd.DataFrame) -> float:
    """openpilot が介入していた割合。

    commaCarSegments は panda が controlsAllowed を直接報告するのでそれを使う。
    comma2k19 には無いので、制御フレームの送信有無から推定した op_tx で代用する。
    """
    for col in ("op_engaged", "op_tx"):
        if col in df:
            return float(np.nanmean(df[col]))
    return float("nan")


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """空の表を除いて連結する。すべて空なら空の DataFrame を返す。"""
    non_empty = [f for f in frames if f is not None and not f.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def _collect(
    seg_rows: list[dict[str, Any]],
    event_frames: Iterable[pd.DataFrame],
    cand_frames: Iterable[pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    events = _concat(event_frames)
    cands = _concat(cand_frames)
    if not cands.empty:
        cands = cands.sort_values("severity", ascending=False).reset_index(drop=True)
    return {
        "segments": pd.DataFrame(seg_rows),
        "events": events,
        "candidates": cands,
    }
