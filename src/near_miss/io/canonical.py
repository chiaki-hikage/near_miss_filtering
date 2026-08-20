"""データセットに依らない正規化済みの入力表現。

データセット固有の読み出し (comma2k19 / commaCarSegments) は、
最終的にすべてこのモジュールの型を返す。下流の再サンプル・特徴量・検出は
ここから先しか見ないので、CAN ID やビット定義を知らずに済む。

    生 CAN (データセット固有)
        → RawCanFrames        バス・アドレス・ペイロードまで正規化
        → SegmentData         車種設定で復号した「正規化車両信号」
        → GriddedSignals      一様グリッド (signals.py)
        → 特徴量 → 検出

チャネル名の取り決め (canonical channel names):
    speed_mps      車速 [m/s]
    steer_deg      舵角 [deg]  左が正
    ws_fl/fr/rl/rr_mps  各輪速 [m/s]
    yaw_rate       ヨーレート [deg/s]  左旋回が正
    accel_x        前後加速度 [m/s^2]  進行方向が正
    accel_y        横加速度 [m/s^2]    左が正
    brake_pressed  ブレーキスイッチ [-]
    op_engaged     openpilot 介入中 [-]
レーダの横位置 lateral_m は「左が正」。features.path_lateral_offset が
ヨーレート由来の進路ずれ (左旋回で正) と直接比較するため、ここを揃える。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Channel:
    """ひとつの信号の生時系列。時刻はデバイスの boot time [s]。"""

    t: np.ndarray
    v: np.ndarray
    unit: str
    kind: str  # "continuous" | "flag" | "occupancy"


@dataclass
class RadarTracks:
    """レーダトラックの観測列。1 行が 1 トラックの 1 観測。

    distance_m  前方距離 [m]
    lateral_m   横位置 [m]  左が正
    vrel_mps    相対速度 [m/s]  負が接近
    track_id    トラックの識別子。割り込み検出で切替を見るために使う
    new_track   そのトラックが新規に立ったフレームで 1
    """

    t: np.ndarray
    distance_m: np.ndarray
    lateral_m: np.ndarray
    vrel_mps: np.ndarray
    track_id: np.ndarray
    new_track: np.ndarray


@dataclass
class RawCanFrames:
    """バス番号まで正規化した生 CAN フレーム列。

    src の意味はデータセットによって違う (comma2k19 の raw_can/src、
    openpilot rlog の can.src) が、いずれも
    「下位 7 bit = バス番号、0x80 = 自分が送信したフレーム」で共通だったため、
    そのまま 1 本の配列で持つ。判定は can_decode.frame_mask が行う。
    """

    t: np.ndarray            # boot time [s]
    address: np.ndarray      # int64
    payload_u64: np.ndarray  # uint64 (big-endian で詰めた 8 バイト)
    src: np.ndarray          # int64

    def __len__(self) -> int:
        return int(self.t.size)


# 解析の主系列。時間範囲はこれらの重なりで決める。
# 付随信号 (アクセル開度など) は車種や年式で有無が変わり、
# 数十 ms 単位で受信の始まりがずれる。それに合わせて解析窓を動かすと、
# 信号を 1 本足しただけで検出結果が変わってしまう。
PRIMARY_CHANNELS = ("speed_mps", "steer_deg", "yaw_rate")


@dataclass
class SegmentRef:
    """セグメントの所在。読み出す前の識別情報だけを持つ。"""

    path: Path
    dongle_id: str
    drive_id: str
    index: int
    dataset: str = "comma2k19"
    platform: str = ""       # commaCarSegments の車種キー。無ければ空

    @property
    def segment_id(self) -> str:
        return f"{self.drive_id}/{self.index}"


@dataclass
class SegmentData:
    """1 セグメント分の正規化済み時系列。"""

    ref: SegmentRef
    vehicle: str
    channels: dict[str, Channel] = field(default_factory=dict)
    radar: RadarTracks | None = None
    raw_can_loaded: bool = False
    byte_order: str = "big"
    notes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def t_span(self) -> tuple[float, float]:
        """主系列が共通して覆っている時間範囲。

        主系列が 1 本も無いときだけ、全チャネルの重なりに落とす。
        主系列の外にある付随信号は再サンプルで NaN になり、
        欠測として coverage に残る。
        """
        primary = [c for n, c in self.channels.items() if n in PRIMARY_CHANNELS and c.t.size]
        pool = primary or [c for c in self.channels.values() if c.t.size]
        if not pool:
            return (np.nan, np.nan)
        return (max(c.t[0] for c in pool), min(c.t[-1] for c in pool))


def group_by_drive(refs: list[SegmentRef]) -> dict[str, list[SegmentRef]]:
    """同一ドライブのセグメントを番号順にまとめる。

    60 秒で切られているためイベントが境界を跨ぐ。連結して扱えるようにしておく。
    """
    drives: dict[str, list[SegmentRef]] = {}
    for r in refs:
        drives.setdefault(r.drive_id, []).append(r)
    for v in drives.values():
        v.sort(key=lambda r: r.index)
    return drives


def concat_segments(segments: list[SegmentData]) -> SegmentData:
    """同一ドライブの連続セグメントを時間軸で連結する。

    60 秒境界で分断されたイベントを取り逃がさないために使う。
    時刻は同じ boot time 基準なので、そのまま並べればよい。
    """
    if not segments:
        raise ValueError("連結するセグメントがありません")
    if len(segments) == 1:
        return segments[0]

    head = segments[0]
    names: list[str] = []
    for s in segments:
        for n in s.channels:
            if n not in names:
                names.append(n)

    merged: dict[str, Channel] = {}
    for name in names:
        parts = [s.channels[name] for s in segments if name in s.channels]
        t = np.concatenate([p.t for p in parts])
        v = np.concatenate([p.v for p in parts])
        order = np.argsort(t, kind="stable")
        merged[name] = Channel(t=t[order], v=v[order], unit=parts[0].unit, kind=parts[0].kind)

    radar_parts = [s.radar for s in segments if s.radar is not None]
    radar = None
    if radar_parts:
        t = np.concatenate([r.t for r in radar_parts])
        order = np.argsort(t, kind="stable")
        radar = RadarTracks(
            t=t[order],
            distance_m=np.concatenate([r.distance_m for r in radar_parts])[order],
            lateral_m=np.concatenate([r.lateral_m for r in radar_parts])[order],
            vrel_mps=np.concatenate([r.vrel_mps for r in radar_parts])[order],
            track_id=np.concatenate([r.track_id for r in radar_parts])[order],
            new_track=np.concatenate([r.new_track for r in radar_parts])[order],
        )

    notes: list[str] = []
    for s in segments:
        notes.extend(f"{s.ref.index}:{n}" for n in s.notes)

    meta: dict[str, Any] = dict(head.meta)
    return SegmentData(
        ref=head.ref,
        vehicle=head.vehicle,
        channels=merged,
        radar=radar,
        raw_can_loaded=all(s.raw_can_loaded for s in segments),
        byte_order=head.byte_order,
        notes=notes,
        meta=meta,
    )
