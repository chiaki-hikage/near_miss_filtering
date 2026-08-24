"""設定ファイルの読み出し。

閾値やパラメータはすべて YAML 側に置く。ここでは読み出しと、
どの設定で実行したかを後から特定するためのハッシュ計算だけを行う。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETECTION = REPO_ROOT / "configs" / "detection.yaml"
DEFAULT_VEHICLE_DIR = REPO_ROOT / "configs" / "vehicles"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(*configs: dict[str, Any], length: int = 10) -> str:
    """設定内容から短いハッシュを作る。出力行に添えて実行条件を追跡する。"""
    blob = json.dumps(configs, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


# DBC の値をそのまま設定に書き写せるようにするための単位換算。
# factor / offset を電卓で潰して書くと opendbc との突き合わせができなくなるため、
# 「DBC 記載の単位 (unit_in)」と「使いたい単位 (unit)」を両方書いて機械的に直す。
UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
    ("km/h", "m/s"): 1.0 / 3.6,
    ("m/s", "km/h"): 3.6,
    ("rad/s", "deg/s"): 180.0 / 3.141592653589793,
    ("deg/s", "rad/s"): 3.141592653589793 / 180.0,
    ("g", "m/s^2"): 9.80665,
}


def unit_scale(unit_in: str, unit_out: str) -> float:
    """unit_in から unit_out への換算係数。同一単位なら 1.0。"""
    if not unit_in or unit_in == unit_out:
        return 1.0
    try:
        return UNIT_CONVERSIONS[(unit_in, unit_out)]
    except KeyError as exc:
        raise ValueError(f"未定義の単位換算です: {unit_in} -> {unit_out}") from exc


@dataclass(frozen=True)
class SignalSpec:
    """生 CAN から 1 信号を取り出すための定義。"""

    name: str
    can_id: int
    start_bit: int
    length: int
    signed: bool
    factor: float
    offset: float
    sign: float
    unit: str
    kind: str          # "continuous" | "flag"
    validated: bool
    enabled: bool
    bus: int | None = None
    note: str = ""
    channel: str = ""      # 正規化チャネル名。既定は name.lower()
    unit_in: str = ""      # DBC 記載の単位。unit と違えば換算する

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "SignalSpec":
        unit = str(d.get("unit", ""))
        return cls(
            name=name,
            can_id=int(d["can_id"]),
            start_bit=int(d["start_bit"]),
            length=int(d["length"]),
            signed=bool(d.get("signed", False)),
            factor=float(d.get("factor", 1.0)),
            offset=float(d.get("offset", 0.0)),
            sign=float(d.get("sign", 1.0)),
            unit=unit,
            kind=str(d.get("kind", "continuous")),
            validated=bool(d.get("validated", False)),
            enabled=bool(d.get("enabled", True)),
            bus=None if d.get("bus") is None else int(d["bus"]),
            note=str(d.get("note", "")),
            channel=str(d.get("channel", "") or name.lower()),
            unit_in=str(d.get("unit_in", "") or unit),
        )

    @property
    def scale(self) -> float:
        """単位換算まで含めた最終的な倍率。"""
        return self.sign * unit_scale(self.unit_in, self.unit)


@dataclass(frozen=True)
class RadarSpec:
    """生 CAN からレーダトラックを組み立てるための定義。

    Toyota のレーダは 1 トラック 1 アドレスで、連番のアドレスに
    「距離・横位置・相対速度」の A 系列と「スコア」の B 系列を分けて流す。
    アドレスの並びとビット定義だけを設定に置き、組み立ては io/can_radar.py が行う。
    """

    bus: int
    track_first_id: int
    track_count: int
    signals: dict[str, dict[str, Any]]
    score_first_id: int | None = None
    score_signal: dict[str, Any] | None = None
    min_score: float = 0.0
    max_distance_m: float = 254.0
    lateral_sign: float = 1.0    # 左が正になるように掛ける係数
    enabled: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "RadarSpec | None":
        if not d:
            return None
        score = d.get("score") or {}
        return cls(
            bus=int(d["bus"]),
            track_first_id=int(d["track_first_id"]),
            track_count=int(d["track_count"]),
            signals=dict(d["signals"]),
            score_first_id=None if score.get("first_id") is None else int(score["first_id"]),
            score_signal=score.get("signal"),
            min_score=float(score.get("min_score", 0.0)),
            max_distance_m=float(d.get("max_distance_m", 254.0)),
            lateral_sign=float(d.get("lateral_sign", 1.0)),
            enabled=bool(d.get("enabled", True)),
        )


@dataclass(frozen=True)
class VehicleConfig:
    """車種固有の CAN 定義。"""

    name: str
    dongle_ids: tuple[str, ...]
    signals: tuple[SignalSpec, ...]
    geometry: dict[str, Any]
    src_tx_flag: int
    control_addresses: tuple[int, ...]
    raw: dict[str, Any]
    platforms: tuple[str, ...] = ()          # commaCarSegments の車種キー
    radar: "RadarSpec | None" = None
    derived: tuple[tuple[str, dict[str, Any]], ...] = ()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VehicleConfig":
        op = d.get("openpilot_tx", {}) or {}
        return cls(
            name=d["name"],
            dongle_ids=tuple(d.get("dongle_ids", [])),
            signals=tuple(
                SignalSpec.from_dict(k, v) for k, v in (d.get("signals") or {}).items()
            ),
            geometry=d.get("geometry", {}) or {},
            src_tx_flag=int(op.get("src_tx_flag", 128)),
            control_addresses=tuple(int(a) for a in op.get("control_addresses", [])),
            raw=d,
            platforms=tuple(d.get("platforms", []) or []),
            radar=RadarSpec.from_dict(d.get("radar")),
            derived=tuple((k, v) for k, v in (d.get("derived") or {}).items()),
        )

    def matches_platform(self, platform: str) -> bool:
        return platform in self.platforms

    @classmethod
    def load(cls, path: str | Path) -> "VehicleConfig":
        return cls.from_dict(load_yaml(path))

    def geometry_value(self, key: str, default: float | None = None) -> float | None:
        """車両諸元を取り出す。未検証で null のものは default を返す。"""
        v = self.geometry.get(key)
        return default if v is None else float(v)

    def center_to_rear_m(self) -> float | None:
        """重心から後軸までの距離 l_r。"""
        wb = self.geometry_value("wheelbase_m")
        l_f = self.geometry_value("center_to_front_m")
        if wb is None or l_f is None:
            return None
        return wb - l_f

    def sideslip_ay_coeff(self) -> float | None:
        """横滑り角の式の第 2 項の係数 k = m*l_f/(C_r*L) [s^2/m]。

        定常の線形単軌道モデルでは、重心位置の横滑り角が
            beta = l_r * yaw_rate / v  -  k * a_y
        で書ける。k は本来 質量・後輪コーナリング剛性・ホイールベースから決まるが、
        前後の剛性が等しい (C_f = C_r) と置くと安定係数 Kus と結び付いて
            Kus = m (l_r - l_f) / (L * C)     ->     k = l_f * Kus / (l_r - l_f)
        となり、**質量も剛性も知らなくても当てはめ済みの Kus から出せる**。
        KIT の実測諸元で照合したところ、公称の k = 0.005018 に対して
        この式は 0.005024 を返し、0.1% で一致した (docs/kit_msdm.md)。

        l_r - l_f は近い 2 つの数の差なので l_f の誤差に敏感である。ただし
        k が 2 倍ずれても beta の誤差の標準偏差は 0.94〜1.37 度に収まることを
        KIT のデータで確かめてある。
        """
        l_f = self.geometry_value("center_to_front_m")
        l_r = self.center_to_rear_m()
        kus = self.geometry_value("understeer_gradient")
        if l_f is None or l_r is None or kus is None:
            return None
        denom = l_r - l_f
        if abs(denom) < 1e-6:
            return None
        return float(l_f * kus / denom)

    def enabled_signals(self) -> tuple[SignalSpec, ...]:
        return tuple(s for s in self.signals if s.enabled)

    def matches_dongle(self, dongle_id: str) -> bool:
        return dongle_id in self.dongle_ids


def load_vehicle_configs(directory: str | Path = DEFAULT_VEHICLE_DIR) -> list[VehicleConfig]:
    return [VehicleConfig.load(p) for p in sorted(Path(directory).glob("*.yaml"))]


def find_vehicle_config(dongle_id: str, configs: list[VehicleConfig]) -> VehicleConfig | None:
    for c in configs:
        if c.matches_dongle(dongle_id):
            return c
    return None


def find_vehicle_config_for_platform(
    platform: str, configs: list[VehicleConfig]
) -> VehicleConfig | None:
    """commaCarSegments の車種キー (例 TOYOTA_RAV4_TSS2) から車種設定を引く。"""
    for c in configs:
        if c.matches_platform(platform):
            return c
    return None


def find_vehicle_config_by_name(name: str, configs: list[VehicleConfig]) -> VehicleConfig | None:
    for c in configs:
        if c.name == name:
            return c
    return None
