"""comma2k19 の読み出し (L0)。

セグメント構成:
    <chunk>/<dongle_id>|<日時>/<セグメント番号>/
        processed_log/CAN/{speed,steering_angle,wheel_speed,radar,raw_can}/{t,value}
        processed_log/IMU/... , GNSS/...   ← 今回は CAN のみを使うため読まない
        raw_log.bz2, video.hevc            ← 特徴抽出には使わない

`processed_log` の単位はデータセットの README に明記されている値をそのまま使う。
`raw_can` から取る信号は車種設定 (configs/vehicles/*.yaml) の定義に従う。

正規化後の型 (Channel / RadarTracks / SegmentData) は io/canonical.py にある。
このモジュールは comma2k19 のディレクトリ構造とファイル形式だけを知っている。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..config import VehicleConfig
from .can_decode import decode_channels, detect_byte_order, openpilot_tx_channel, payload_to_u64
from .canonical import (  # 後方互換のため再輸出する
    Channel,
    RadarTracks,
    RawCanFrames,
    SegmentData,
    SegmentRef,
    concat_segments,
    group_by_drive,
)

DATASET = "comma2k19"

# README 記載の単位。ここを推測で変えない。
_PROCESSED_CHANNELS = {
    "speed": ("speed_mps", "m/s", "continuous"),
    "steering_angle": ("steer_deg", "deg", "continuous"),
}
_WHEEL_SPEED_COLUMNS = ("ws_fl_mps", "ws_fr_mps", "ws_rl_mps", "ws_rr_mps")

_DRIVE_DIR_RE = re.compile(r"^(?P<dongle>[0-9a-f]{16})\|(?P<start>.+)$")


# ---------------------------------------------------------------------------
# セグメントの探索
# ---------------------------------------------------------------------------
def find_segments(root: str | Path) -> list[SegmentRef]:
    """`processed_log` を持つディレクトリをセグメントとみなして列挙する。"""
    root = Path(root)
    refs: list[SegmentRef] = []
    for processed in sorted(root.rglob("processed_log")):
        if not processed.is_dir():
            continue
        seg_dir = processed.parent
        drive_dir = seg_dir.parent
        m = _DRIVE_DIR_RE.match(drive_dir.name)
        if m is None:
            continue
        try:
            index = int(seg_dir.name)
        except ValueError:
            continue
        refs.append(
            SegmentRef(
                path=seg_dir,
                dongle_id=m.group("dongle"),
                drive_id=drive_dir.name,
                index=index,
                dataset=DATASET,
            )
        )
    return sorted(refs, key=lambda r: (r.drive_id, r.index))


# ---------------------------------------------------------------------------
# 読み出し
# ---------------------------------------------------------------------------
def _load_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=True)


def _load_processed_can(seg_dir: Path) -> tuple[dict[str, Channel], RadarTracks | None, list[str]]:
    base = seg_dir / "processed_log" / "CAN"
    channels: dict[str, Channel] = {}
    notes: list[str] = []

    for sub, (name, unit, kind) in _PROCESSED_CHANNELS.items():
        d = base / sub
        if not d.is_dir():
            notes.append(f"missing:{sub}")
            continue
        t = _load_array(d / "t").ravel().astype(float)
        v = _load_array(d / "value").astype(float).reshape(len(t), -1)[:, 0]
        channels[name] = Channel(t=t, v=v, unit=unit, kind=kind)

    d = base / "wheel_speed"
    if d.is_dir():
        t = _load_array(d / "t").ravel().astype(float)
        v = _load_array(d / "value").astype(float).reshape(len(t), -1)
        if v.shape[1] != len(_WHEEL_SPEED_COLUMNS):
            notes.append(f"wheel_speed:unexpected_shape:{v.shape}")
        else:
            for i, name in enumerate(_WHEEL_SPEED_COLUMNS):
                channels[name] = Channel(t=t, v=v[:, i], unit="m/s", kind="continuous")
    else:
        notes.append("missing:wheel_speed")

    radar = None
    d = base / "radar"
    if d.is_dir():
        t = _load_array(d / "t").ravel().astype(float)
        v = _load_array(d / "value").astype(float)
        if v.ndim == 2 and v.shape[1] >= 7:
            radar = RadarTracks(
                t=t,
                distance_m=v[:, 0],
                lateral_m=v[:, 1],
                vrel_mps=v[:, 2],
                track_id=v[:, 5].astype(np.int64),
                new_track=v[:, 6].astype(np.int64),
            )
        else:
            notes.append(f"radar:unexpected_shape:{v.shape}")
    else:
        notes.append("missing:radar")

    return channels, radar, notes


def _load_raw_can(seg_dir: Path, vehicle: VehicleConfig) -> tuple[dict, str, list[str]]:
    """raw_can を RawCanFrames まで正規化してから、共通の復号にかける。

    ビット定義を持つのは車種設定だけで、ここは comma2k19 の
    ファイル配置 (t / address / data / src) を知っているにすぎない。
    """
    base = seg_dir / "processed_log" / "CAN" / "raw_can"
    notes: list[str] = []
    if not base.is_dir():
        return {}, "big", ["missing:raw_can"]

    t = _load_array(base / "t").ravel().astype(float)
    address = _load_array(base / "address").ravel().astype(np.int64)
    data = _load_array(base / "data")
    src = _load_array(base / "src").ravel().astype(np.int64)

    byte_order = detect_byte_order(address, data)
    if byte_order != "big":
        notes.append(f"byte_order:{byte_order}")

    raw = RawCanFrames(
        t=t, address=address, payload_u64=payload_to_u64(data, byte_order), src=src
    )
    channels, decode_notes = decode_channels(raw, vehicle)
    notes.extend(decode_notes)

    # openpilot が制御フレームを送出していた時刻。人間の運転挙動と切り分けるために使う。
    tx = openpilot_tx_channel(raw, vehicle)
    if tx is not None:
        channels["op_tx"] = tx

    return channels, byte_order, notes


def load_segment(
    ref: SegmentRef,
    vehicle: VehicleConfig | None = None,
    with_raw_can: bool = False,
) -> SegmentData:
    """1 セグメントを読み出す。

    with_raw_can=False のときは processed_log だけを読む。
    全セグメントを走査する 1 段目はこちらで足りる。
    """
    channels, radar, notes = _load_processed_can(ref.path)
    seg = SegmentData(
        ref=ref,
        vehicle=vehicle.name if vehicle else "unknown",
        channels=channels,
        radar=radar,
        notes=notes,
    )
    if with_raw_can:
        if vehicle is None:
            seg.notes.append("raw_can:skipped:no_vehicle_config")
        else:
            raw_channels, byte_order, raw_notes = _load_raw_can(ref.path, vehicle)
            seg.channels.update(raw_channels)
            seg.byte_order = byte_order
            seg.notes.extend(raw_notes)
            seg.raw_can_loaded = bool(raw_channels)
    return seg
