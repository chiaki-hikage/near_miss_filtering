"""openpilot の rlog を読む (L0)。

commaCarSegments が配る `rlog.zst` は openpilot のログをそのまま zstd で
固めたもの。中身は capnp の Event が連結されている。

このデータセットの rlog は間引かれており、実測で入っているのは

    can          100 Hz  全バスの生 CAN フレーム
    pandaStates   10 Hz  controlsAllowed など panda の状態
    carParams      1 件  車種判定と車両諸元

の 3 種類だけだった。carState や radarState といった openpilot の
復号済み信号は入っていないので、車両信号はすべて生 CAN から作る。

capnp のスキーマ (cereal) は openpilot 本体に入っている。ここでは
`data/cereal/` に置いた .capnp を読む。無ければ取得方法を示して落とす。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..config import REPO_ROOT
from .canonical import Channel, RawCanFrames

log = logging.getLogger(__name__)

DEFAULT_SCHEMA_DIR = REPO_ROOT / "data" / "cereal"
SCHEMA_FILES = ("log.capnp", "car.capnp", "custom.capnp", "deprecated.capnp", "include/c++.capnp")

_SCHEMA_HELP = (
    "cereal の capnp スキーマがありません。次で取得してください:\n"
    "  python scripts/fetch_cereal_schema.py"
)


@lru_cache(maxsize=4)
def load_schema(schema_dir: str | None = None) -> Any:
    """log.capnp を読み込む。プロセス内で 1 回だけ行う。"""
    import capnp  # 遅延 import。commaCarSegments を使わない実行では不要

    capnp.remove_import_hook()
    d = Path(schema_dir) if schema_dir else DEFAULT_SCHEMA_DIR
    log_capnp = d / "log.capnp"
    if not log_capnp.is_file():
        raise FileNotFoundError(f"{log_capnp} がありません。\n{_SCHEMA_HELP}")
    return capnp.load(str(log_capnp), imports=[str(d)])


@dataclass
class RlogData:
    """rlog 1 本から取り出した内容。"""

    can: RawCanFrames
    car_params: dict[str, Any] = field(default_factory=dict)
    panda_channels: dict[str, Channel] = field(default_factory=dict)
    event_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _decompress(path: Path) -> bytes:
    import zstandard

    with open(path, "rb") as f:
        return zstandard.ZstdDecompressor().stream_reader(f).read()


# openpilot の logMonoTime は起動からの経過時間 [ns]。comma2k19 の boot time と
# 同じ性質なので、秒に直すだけで下流は共通に扱える。
NS = 1e-9


def read_rlog(path: str | Path, schema_dir: str | None = None) -> RlogData:
    """rlog.zst を読んで生 CAN と付随情報を返す。"""
    path = Path(path)
    schema = load_schema(schema_dir)
    raw_bytes = _decompress(path)

    t_list: list[float] = []
    addr_list: list[int] = []
    data_list: list[bytes] = []
    src_list: list[int] = []
    counts: dict[str, int] = {}
    notes: list[str] = []
    car_params: dict[str, Any] = {}
    panda_t: list[float] = []
    panda_allowed: list[float] = []

    for msg in schema.Event.read_multiple_bytes(raw_bytes):
        which = msg.which()
        counts[which] = counts.get(which, 0) + 1
        if which == "can":
            t = msg.logMonoTime * NS
            for f in msg.can:
                t_list.append(t)
                addr_list.append(f.address)
                data_list.append(bytes(f.dat))
                src_list.append(f.src)
        elif which == "pandaStates":
            for ps in msg.pandaStates:
                panda_t.append(msg.logMonoTime * NS)
                panda_allowed.append(1.0 if ps.controlsAllowed else 0.0)
                break  # panda は通常 1 台。先頭だけ見る
        elif which == "carParams":
            car_params = _car_params_dict(msg.carParams)

    if not t_list:
        notes.append("rlog:no_can")

    can = RawCanFrames(
        t=np.asarray(t_list, dtype=float),
        address=np.asarray(addr_list, dtype=np.int64),
        payload_u64=_payloads_to_u64(data_list),
        src=np.asarray(src_list, dtype=np.int64),
    )

    panda_channels: dict[str, Channel] = {}
    if panda_t:
        # openpilot が実際に制御していた区間。comma2k19 では送信フレームの有無から
        # 推定するしかなかったが、こちらは panda が直接報告している。
        panda_channels["op_engaged"] = Channel(
            t=np.asarray(panda_t, dtype=float),
            v=np.asarray(panda_allowed, dtype=float),
            unit="-",
            kind="flag",
        )

    return RlogData(
        can=can,
        car_params=car_params,
        panda_channels=panda_channels,
        event_counts=counts,
        notes=notes,
    )


def _payloads_to_u64(payloads: list[bytes]) -> np.ndarray:
    """可変長 (0〜8 バイト) の CAN データを 8 バイト右詰めの uint64 にする。

    DBC の Motorola 表記は先頭バイトを byte 0 として数えるので、
    8 バイトに満たないメッセージは後ろを 0 で埋める (左詰め)。
    """
    n = len(payloads)
    buf = np.zeros((n, 8), dtype=np.uint8)
    for i, b in enumerate(payloads):
        k = min(len(b), 8)
        if k:
            buf[i, :k] = np.frombuffer(b[:k], dtype=np.uint8)
    weights = np.array([1 << (8 * (7 - i)) for i in range(8)], dtype=np.uint64)
    return (buf.astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64)


_CAR_PARAM_FIELDS = (
    "carFingerprint",
    "steerRatio",
    "wheelbase",
    "mass",
    "centerToFront",
    "tireStiffnessFactor",
    "steerActuatorDelay",
    "minSteerSpeed",
    "radarUnavailable",
    "openpilotLongitudinalControl",
    "flags",
)


def _car_params_dict(cp: Any) -> dict[str, Any]:
    """必要な諸元だけ取り出す。スキーマの版差で欠ける項目は飛ばす。"""
    out: dict[str, Any] = {}
    for f in _CAR_PARAM_FIELDS:
        try:
            v = getattr(cp, f)
        except Exception:
            continue
        out[f] = v if isinstance(v, (int, float, bool, str)) else str(v)
    return out
