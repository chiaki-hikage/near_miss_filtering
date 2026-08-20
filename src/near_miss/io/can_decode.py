"""生 CAN ペイロードからの信号復号。

`~/work/thermo/decode_comma2k19_brake.py` のビット抽出を移植した。
DBC の Motorola (@0, big-endian) 表記をそのまま扱う。

このモジュールは CAN ID / ビット位置を「知らない」。すべて車種設定
(configs/vehicles/*.yaml) から渡される。データセット固有の読み出しが
RawCanFrames まで正規化したあとは、comma2k19 でも commaCarSegments でも
同じ decode_channels を通る。
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import SignalSpec, VehicleConfig
from .canonical import Channel, RawCanFrames

log = logging.getLogger(__name__)

_WEIGHTS_BE = np.array([1 << (8 * (7 - i)) for i in range(8)], dtype=np.uint64)
_WEIGHTS_LE = np.array([1 << (8 * i) for i in range(8)], dtype=np.uint64)


def payload_to_u64(data: np.ndarray, byte_order: str = "big") -> np.ndarray:
    """8 バイトの CAN データ列を uint64 配列にまとめる。

    comma2k19 の `raw_can/data` は (N, 8) の uint8 だったり |S8 だったりするので、
    どちらでも受けられるようにバイト列へ落としてから組み立てる。
    """
    b = np.frombuffer(np.ascontiguousarray(data).tobytes(), dtype=np.uint8).reshape(-1, 8)
    w = _WEIGHTS_BE if byte_order == "big" else _WEIGHTS_LE
    return (b.astype(np.uint64) * w).sum(axis=1, dtype=np.uint64)


def extract_bits(payload_u64: np.ndarray, start_bit: int, length: int, signed: bool) -> np.ndarray:
    """DBC の Motorola 表記 start_bit から length ビットを取り出す。"""
    byte_idx, bit_in_byte = divmod(start_bit, 8)
    msb_pos = byte_idx * 8 + (7 - bit_in_byte)
    shift = 64 - msb_pos - length
    if shift < 0:
        raise ValueError(f"start_bit={start_bit} length={length} が 8 バイトを超えます")
    raw = ((payload_u64 >> np.uint64(shift)) & np.uint64((1 << length) - 1)).astype(np.int64)
    if signed and length < 64:
        sign_bit = 1 << (length - 1)
        raw = np.where(raw & sign_bit, raw - (1 << length), raw)
    return raw


# comma2k19 の raw_can/src は下位ビットがバス番号、0x80 が立つと送信フレーム。
SRC_TX_MASK = 0x80
SRC_BUS_MASK = 0x7F


def frame_mask(
    address: np.ndarray,
    src: np.ndarray | None,
    can_id: int,
    bus: int | None = None,
    include_tx: bool = False,
) -> np.ndarray:
    """復号対象のフレームを選ぶ。

    同じアドレスが複数のバスに、別の内容で流れていることがある。
    実測で 0x224 はバス 0 が BRAKE_MODULE、バス 1 は先頭バイトがカウンタの
    別メッセージだった。バスを指定せずに復号すると両者が混ざり、
    フラグがチャタリングする。

    送信フレーム (openpilot が出したもの) も既定では除く。車両側の状態を
    見たいのであって、こちらの指令値を見たいのではないため。
    """
    m = address == can_id
    if src is None:
        return m
    if not include_tx:
        m &= (src & SRC_TX_MASK) == 0
    if bus is not None:
        m &= (src & SRC_BUS_MASK) == bus
    return m


def decode_signal(
    t: np.ndarray,
    address: np.ndarray,
    payload_u64: np.ndarray,
    spec: SignalSpec,
    src: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """1 信号を物理値に復号して (時刻, 値) を返す。

    `spec.sign` は実測で判明した符号補正。docs/signals_rav4.md に根拠を残してある。
    `spec.bus` を指定すると、そのバスで受信したフレームだけを使う。
    """
    m = frame_mask(address, src, spec.can_id, spec.bus)
    if not m.any():
        return np.empty(0), np.empty(0)
    raw = extract_bits(payload_u64[m], spec.start_bit, spec.length, spec.signed)
    # factor / offset は DBC 記載のまま。符号補正と単位換算をまとめて掛ける。
    value = (raw * spec.factor + spec.offset) * spec.scale
    return t[m], value


def detect_byte_order(address: np.ndarray, data: np.ndarray, probe_can_id: int = 170) -> str:
    """輪速メッセージが妥当な速度域に落ちる方をバイト順として選ぶ。

    comma2k19 は big-endian のはずだが、無検査で決め打ちしない。
    判定できない場合は "big" を返し、呼び出し側が仮定であることを扱えるようにする。
    """
    m = address == probe_can_id
    if m.sum() < 10:
        return "big"

    scores: dict[str, float] = {}
    for order in ("big", "little"):
        p = payload_to_u64(data, order)[m]
        ws = np.stack(
            [extract_bits(p, sb, 15, False) * 0.01 - 67.67 for sb in (6, 22, 38, 54)],
            axis=1,
        )
        in_range = float(np.mean((ws > -1.0) & (ws < 200.0)))
        spread = float(np.nanmedian(np.ptp(ws, axis=1)))
        scores[order] = in_range - min(spread / 10.0, 1.0)
    return max(scores, key=scores.__getitem__)


# ---------------------------------------------------------------------------
# 正規化チャネルの組み立て
# ---------------------------------------------------------------------------
def decode_channels(
    raw: RawCanFrames, vehicle: VehicleConfig
) -> tuple[dict[str, Channel], list[str]]:
    """車種設定に載っている信号を復号して、正規化チャネル名の辞書にする。

    チャネル名は SignalSpec.channel (既定は信号名の小文字) で決まる。
    ここで CAN ID / ビット位置は消え、以降の層は名前と単位しか見ない。
    """
    channels: dict[str, Channel] = {}
    notes: list[str] = []
    for spec in vehicle.enabled_signals():
        st, sv = decode_signal(raw.t, raw.address, raw.payload_u64, spec, raw.src)
        if st.size == 0:
            notes.append(f"absent:{spec.name}")
            continue
        channels[spec.channel] = Channel(t=st, v=sv, unit=spec.unit, kind=spec.kind)

    derived, derived_notes = build_derived(channels, vehicle)
    channels.update(derived)
    notes.extend(derived_notes)
    return channels, notes


def build_derived(
    channels: dict[str, Channel], vehicle: VehicleConfig
) -> tuple[dict[str, Channel], list[str]]:
    """複数の生信号をまとめて 1 本のチャネルにする。

    sum:  足す。舵角のように粗い本体と端数が別信号に分かれている場合に使う
          (Toyota: STEER_ANGLE 1.5 deg 刻み + STEER_FRACTION 0.1 deg 刻み)。
    mean: 平均する。車速を 4 輪の輪速平均から作る場合に使う。
          comma2k19 の processed_log/CAN/speed が輪速平均そのものだったため
          (実測 rmse 0.0002 m/s)、生 CAN だけのデータセットでも同じ作り方にそろえる。

    同じメッセージ由来なら時刻は一致するので、先頭チャネルの時刻に合わせる。
    """
    out: dict[str, Channel] = {}
    notes: list[str] = []
    for name, spec in vehicle.derived:
        how = "mean" if "mean" in spec else "sum"
        parts = list(spec.get(how, []))
        if not parts:
            notes.append(f"derived:{name}:no_parts")
            continue
        missing = [p for p in parts if p not in channels]
        if missing:
            notes.append(f"derived:{name}:missing:{','.join(missing)}")
            continue
        base = channels[parts[0]]
        total = base.v.astype(float).copy()
        for p in parts[1:]:
            ch = channels[p]
            if ch.t.size == base.t.size and np.array_equal(ch.t, base.t):
                total = total + ch.v
            else:
                # 別メッセージ由来のときだけ内挿する。ずれを注記に残す。
                notes.append(f"derived:{name}:interp:{p}")
                total = total + np.interp(base.t, ch.t, ch.v)
        if how == "mean":
            total = total / len(parts)
        out[name] = Channel(t=base.t, v=total, unit=base.unit, kind=base.kind)
    return out, notes


def openpilot_tx_channel(raw: RawCanFrames, vehicle: VehicleConfig) -> Channel | None:
    """openpilot が制御フレームを送出していた時刻を occupancy チャネルにする。

    人間の運転挙動と切り分けるために使う。
    """
    if not vehicle.control_addresses:
        return None
    is_tx = (raw.src & vehicle.src_tx_flag) != 0
    is_ctrl = np.isin(raw.address, np.asarray(vehicle.control_addresses, dtype=np.int64))
    m = is_tx & is_ctrl
    return Channel(t=raw.t[m], v=np.ones(int(m.sum())), unit="-", kind="occupancy")
