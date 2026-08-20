"""データセットの差を 1 か所に閉じ込める層。

pipeline は「セグメントの一覧」と「1 セグメントを SegmentData にする関数」だけを
受け取り、どのデータセットかは知らない。データセットを足すときは、
ここに SegmentSource を返す関数を 1 つ書けばよい。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import (
    VehicleConfig,
    find_vehicle_config,
    find_vehicle_config_by_name,
    find_vehicle_config_for_platform,
)
from .io import comma1m, comma2k19, comma_car_segments
from .io.canonical import SegmentData, SegmentRef

log = logging.getLogger(__name__)


@dataclass
class SegmentSource:
    """セグメントの供給元。"""

    name: str
    refs: list[SegmentRef]
    load: Callable[[SegmentRef, VehicleConfig | None, bool], SegmentData]
    vehicle_for: Callable[[SegmentRef], VehicleConfig | None]
    # 映像があるデータセットだけコマ番号を出す。commaCarSegments には映像が無い。
    video_fps: float | None = None
    # stage 1 (processed_log だけで回す粗い走査) が使えるか
    supports_stage1: bool = True
    meta: dict = field(default_factory=dict)


def comma2k19_source(
    data_root: str | Path, vehicle_configs: list[VehicleConfig]
) -> SegmentSource:
    """comma2k19 のチャンクを供給元にする。"""
    refs = comma2k19.find_segments(data_root)
    if not refs:
        raise FileNotFoundError(f"セグメントが見つかりません: {data_root}")

    return SegmentSource(
        name="comma2k19",
        refs=refs,
        load=lambda ref, veh, raw: comma2k19.load_segment(ref, veh, with_raw_can=raw),
        vehicle_for=lambda ref: find_vehicle_config(ref.dongle_id, vehicle_configs),
        video_fps=20.0,
        supports_stage1=True,
        meta={"data_root": str(data_root)},
    )


def car_segments_source(
    cache_dir: str | Path,
    platform: str,
    vehicle_configs: list[VehicleConfig],
    names: list[str] | None = None,
) -> SegmentSource:
    """commaCarSegments を供給元にする。

    names を渡すと、その一覧のうちローカルにあるものだけを使う。
    渡さない場合はキャッシュにあるもの全部。取得は scripts/fetch_car_segments.py で行い、
    ここでは落としに行かない (走査中に黙って通信しないため)。
    """
    vehicle = find_vehicle_config_for_platform(platform, vehicle_configs)
    if vehicle is None:
        raise KeyError(f"車種設定がありません: {platform}")

    refs = comma_car_segments.find_segments(cache_dir, platform)
    if names is not None:
        wanted = {comma_car_segments.local_path(n, cache_dir) for n in names}
        refs = [r for r in refs if r.path in wanted]
    if not refs:
        raise FileNotFoundError(f"セグメントがありません: {cache_dir}")

    return SegmentSource(
        name="comma_car_segments",
        refs=refs,
        load=lambda ref, veh, raw: comma_car_segments.load_segment(ref, veh, with_raw_can=True),
        vehicle_for=lambda ref: vehicle,
        video_fps=None,          # 映像は配布されていない
        supports_stage1=False,   # processed_log が無いので生 CAN 必須
        meta={"cache_dir": str(cache_dir), "platform": platform},
    )


def comma1m_source(
    cache_dir: str | Path,
    vehicle_configs: list[VehicleConfig],
    names: list[str] | None = None,
    localizer_cfg: dict | None = None,
) -> SegmentSource:
    """comma1M を供給元にする。

    CAN が無いので、正規化チャネルは localizer の位置・速度から作った
    speed_mps と yaw_rate (course rate) の 2 本だけになる。
    舵角・ブレーキ・輪速・レーダに依存する検出は自動的に無効になる。
    """
    cfg = localizer_cfg or {}
    refs = comma1m.find_segments(cache_dir, names)
    if not refs:
        raise FileNotFoundError(f"localizer がありません: {cache_dir}")

    vehicle = find_vehicle_config_by_name("comma1m_localizer", vehicle_configs)

    def _load(ref, veh, raw):
        return comma1m.load_segment(
            ref, veh,
            smooth_window_s=float(cfg.get("course_smooth_window_s", 0.1)),
            min_speed_mps=float(cfg.get("course_min_speed_mps", 2.0)),
        )

    return SegmentSource(
        name="comma1M",
        refs=refs,
        load=_load,
        vehicle_for=lambda ref: vehicle,
        video_fps=20.0,          # fcamera は 1200 frame / 60 s
        supports_stage1=False,
        meta={"cache_dir": str(cache_dir)},
    )
