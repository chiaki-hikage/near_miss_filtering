"""KIT Multi-Surface Driving Maneuvers の読み出し。

出典: https://radar.kit.edu/radar/en/dataset/44a91t97pmnha1k9
      DOI 10.35097/44a91t97pmnha1k9  (CC BY-SA 4.0)
      T. Schulz, F. Snobar (KIT Institut fuer Fahrzeugsystemtechnik), 2026-04-30

このデータセットは走行データの抽出対象ではなく、**物差し**として使う。
横滑り角 β が Kistler Correvit S-Motion 光学式センサで 1000 Hz 実測されており、
乾燥アスファルト (mu=1.1) と敷石 (mu=0.7) の閉鎖路で横方向の限界まで走っている。
comma 系のデータでは β が推定器のノイズ床に埋もれて検証できなかったため、
単軌道モデルと横加速度の作り方をここで裏取りする。

含まれないもの: CAN の主要信号 (輪速は dynamic run に無い)、ESC/VSC、映像。

--- ファイル形式について ---
各信号は MATLAB の timeseries オブジェクト (MCOS) として保存されている。
scipy.io.loadmat は MCOS を不透明オブジェクトとしてしか返さないため、
数値は読めない。実体は同じファイルの __function_workspace__ の中に

    (時刻ベクトル, データ) の対が、struct のフィールド順に並ぶ

という形で入っている。ここではその並びを直接読み出す。対応付けが正しいことは
「対の前半が 0 から始まる単調増加である」ことで毎回検証する。
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "raw_data" / "kit_msdm" / "10.35097-44a91t97pmnha1k9" / "data" / "dataset"

RATE_HZ = 1000.0

# MAT-file の要素タイプ
MI_DOUBLE = 9
MI_MATRIX = 14

# 信号名 -> (単位, 説明)。データセットの readme と parameter.m による。
SIGNALS: dict[str, tuple[str, str]] = {
    "delta_stm_rad": ("rad", "タイヤ切れ角 (ステアリングホイール角ではない)"),
    "F_trl_N": ("N", "左タイロッド力"),
    "F_trr_N": ("N", "右タイロッド力"),
    "M_eps_Nm": ("N*m", "EPS モータトルク"),
    "a_x_ra_mps2": ("m/s^2", "後軸位置の前後加速度"),
    "a_y_ra_mps2": ("m/s^2", "後軸位置の横加速度"),
    "a_x_cor_mps2": ("m/s^2", "Correvit 位置の前後加速度"),
    "a_y_cor_mps2": ("m/s^2", "Correvit 位置の横加速度"),
    "w_z_cor_radps": ("rad/s", "ヨーレート"),
    "w_fl_radps": ("rad/s", "左前輪の回転速度"),
    "v_x_cor_mps": ("m/s", "Correvit 位置の車体前後速度"),
    "v_y_cor_mps": ("m/s", "Correvit 位置の車体横速度"),
    "v_cor_mps": ("m/s", "Correvit 位置の速度の大きさ"),
    # v_cog_mps / beta_cog_mps は「重心位置の速度と横滑り角」とされているが、
    # Correvit の v_x/v_y と公開されている並進ベクトルからは再現できなかった。
    #   - v_cog は |v_cor| より常に 1〜1.5 m/s 大きい (定常直進でも)
    #   - 11 本中 1 本 (dynmic_asphalt_b_cobble_1) で beta_cog の符号が
    #     Correvit 由来の beta と逆になる
    #   - beta_cog の単位は名前の _mps ではなく rad
    # 別系統 (RTK GNSS/INS か推定器の出力) と思われる。定義が確認できるまで使わない。
    # 真値として使うのは v_x_cor_mps / v_y_cor_mps (光学式センサの直接計測)。
    "v_cog_mps": ("m/s", "重心位置の速度とされる量。定義未確認のため使用しない"),
    "beta_cog_mps": ("rad", "重心位置の横滑り角とされる量。定義未確認のため使用しない"),
    "lat_ra_deg": ("deg", "緯度"),
    "long_ra_deg": ("deg", "経度"),
}


@dataclass
class VehicleParams:
    """parameter.m の内容。座標系は ISO 8855 (x 前・y 左・z 上)。"""

    raw: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def wheelbase_m(self) -> float:
        return self.raw["vehicle.l"]

    @property
    def track_width_m(self) -> float:
        return self.raw["vehicle.tw"]

    @property
    def mass_kg(self) -> float:
        return self.raw["vehicle.m"]

    @property
    def l_f(self) -> float:
        return self.raw["vehicle.l_f"]

    @property
    def l_r(self) -> float:
        return self.raw["vehicle.l_r"]

    @property
    def c_f(self) -> float:
        return self.raw["tire.C_f"]

    @property
    def c_r(self) -> float:
        return self.raw["tire.C_r"]

    def trvec(self, name: str) -> np.ndarray:
        return np.asarray(self.raw[f"tf.trvec_{name}"], dtype=float)

    def mu(self, surface: str) -> float | None:
        return self.raw.get(f"mu.{surface}")

    def understeer_gradient(self) -> float:
        """線形単軌道モデルの安定係数 [rad*s^2/m]。

        Kus = m/L * (l_r/C_f - l_f/C_r)
        parameter.m の値は C_f = C_r なので、前後重量配分だけで決まる。
        """
        m, L = self.mass_kg, self.wheelbase_m
        return m / L * (self.l_r / self.c_f - self.l_f / self.c_r)


def load_parameters(path: str | Path) -> VehicleParams:
    """parameter.m を読む。MATLAB を使わず、代入文だけを拾う。"""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out: dict[str, Any] = {}
    for m in re.finditer(r"params\.([\w.]+)\s*=\s*([^;]+);", text):
        key, val = m.group(1), m.group(2).strip()
        if val.startswith("["):
            nums = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", val)]
            out[key] = nums
        else:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    return VehicleParams(raw=out)


# --- MAT-file の subsystem からの読み出し -------------------------------

def _iter_double_arrays(buf: bytes, min_bytes: int) -> list[np.ndarray]:
    """MAT-file のバイト列を走査して、長い double 配列を出現順に集める。

    miMATRIX は入れ物なので中へ降り、それ以外は読み飛ばす。
    """
    out: list[np.ndarray] = []
    p, n = 0, len(buf)
    while p + 8 <= n:
        w0, w1 = struct.unpack("<II", buf[p : p + 8])
        small = (w0 >> 16) != 0
        tag = (w0 & 0xFFFF) if small else w0
        size = (w0 >> 16) if small else w1
        if tag == MI_DOUBLE and not small and size >= min_bytes:
            out.append(np.frombuffer(buf[p + 8 : p + 8 + size], dtype="<f8"))
        if tag == MI_MATRIX and not small:
            p += 8                      # 中身へ降りる
        elif small:
            p += 8
        else:
            p += 8 + (size + 7) // 8 * 8
    return out


@dataclass
class Run:
    """1 回の走行分の信号。時刻は run 先頭からの秒。"""

    name: str
    path: Path
    t: np.ndarray
    channels: dict[str, np.ndarray]
    surface: str
    kind: str

    def __contains__(self, key: str) -> bool:
        return key in self.channels

    def __getitem__(self, key: str) -> np.ndarray:
        return self.channels[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.channels.get(key, default)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])

    @property
    def rate_hz(self) -> float:
        return float(1.0 / np.median(np.diff(self.t)))


def _classify(name: str) -> tuple[str, str]:
    """ファイル名から走行の種類と路面を取り出す。

    命名は dynamic_driving / slow_driving / parking / standstill に
    路面 (asphalt_a, asphalt_b, cobble, concrete, plates) が続く。
    dynmic_asphalt_b_cobble_1 のように綴りが揺れているものが 1 件ある。
    """
    stem = name.lower()
    if stem.startswith(("dynamic_driving", "dynmic")):
        kind = "dynamic"
    elif stem.startswith("slow_driving"):
        kind = "slow"
    elif stem.startswith("parking"):
        kind = "parking"
    elif stem.startswith("standstill"):
        kind = "standstill"
    else:
        kind = "unknown"
    surfaces = [s for s in ("asphalt_a", "asphalt_b", "cobble", "concrete", "plates") if s in stem]
    return kind, "+".join(surfaces) if surfaces else "unknown"


def read_run(path: str | Path) -> Run:
    """1 つの .mat を読む。"""
    import scipy.io as sio

    path = Path(path)
    raw = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    names = list(raw["data"]._fieldnames)
    if "__function_workspace__" not in raw:
        raise ValueError(f"{path.name}: __function_workspace__ が無い")

    # 先頭 8 バイトはサブシステムの前置き
    arrays = _iter_double_arrays(raw["__function_workspace__"].tobytes()[8:], min_bytes=8000)
    if len(arrays) != 2 * len(names):
        raise ValueError(
            f"{path.name}: 配列 {len(arrays)} 本、信号 {len(names)} 個から期待される {2 * len(names)} 本と違う"
        )

    t_ref: np.ndarray | None = None
    channels: dict[str, np.ndarray] = {}
    for i, nm in enumerate(names):
        tv, dv = arrays[2 * i], arrays[2 * i + 1]
        if tv.size < 2 or tv[0] != 0.0 or not np.all(np.diff(tv) > 0):
            raise ValueError(f"{path.name}/{nm}: 時刻ベクトルが 0 始まりの単調増加でない")
        if tv.size != dv.size:
            raise ValueError(f"{path.name}/{nm}: 時刻 {tv.size} 点に対しデータ {dv.size} 点")
        if t_ref is None:
            t_ref = tv
        elif not np.array_equal(tv, t_ref):
            raise ValueError(f"{path.name}/{nm}: 他の信号と時刻が揃っていない")
        channels[nm] = dv

    kind, surface = _classify(path.stem)
    return Run(name=path.stem, path=path, t=t_ref, channels=channels, surface=surface, kind=kind)


def find_runs(root: str | Path = DEFAULT_ROOT, kind: str | None = None) -> list[Path]:
    paths = sorted(Path(root).glob("*.mat"))
    if kind is None:
        return paths
    return [p for p in paths if _classify(p.stem)[0] == kind]


# --- 幾何と横滑り角 -------------------------------------------------------

def translate_velocity(
    v_x: np.ndarray, v_y: np.ndarray, yaw_rate: np.ndarray, trvec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """剛体の別の点における速度へ移す。

    v_P = v_O + omega x r。ISO 8855 (x 前・y 左・z 上) で omega = (0, 0, r_z) なら
        v_Px = v_Ox - yaw_rate * r_y
        v_Py = v_Oy + yaw_rate * r_x
    """
    rx, ry = float(trvec[0]), float(trvec[1])
    return v_x - yaw_rate * ry, v_y + yaw_rate * rx


def sideslip_deg(
    run: Run, params: VehicleParams | None = None, at: str = "cor", min_speed_mps: float = 2.0
) -> np.ndarray:
    """横滑り角 beta [deg] を返す。左向きの横滑りが正 (ISO 8855)。

    at="cor" は Correvit センサ位置、"ra" は後軸中心、"cog" は重心。
    min_speed_mps 未満は NaN。beta は低速で定義できないので埋めない。

    既定を "cor" にしてあるのは、Correvit の v_x/v_y が唯一の直接計測だから。
    "ra"/"cog" は剛体変換を掛けるだけで、収録されている beta_cog_mps とは一致しない
    (SIGNALS の注記を参照)。
    """
    v_x, v_y = run["v_x_cor_mps"], run["v_y_cor_mps"]
    if at != "cor":
        if params is None:
            raise ValueError("cor 以外の位置には parameter.m が要ります")
        r = params.trvec("cor_ra")
        if at == "cog":
            r = np.asarray(r, dtype=float) + params.trvec("ra_cog")
        elif at != "ra":
            raise ValueError(f"未知の位置: {at}")
        v_x, v_y = translate_velocity(v_x, v_y, run["w_z_cor_radps"], r)
    beta = np.degrees(np.arctan2(v_y, v_x))
    beta[run["v_x_cor_mps"] < min_speed_mps] = np.nan
    return beta


# --- 正規化表現への変換 -------------------------------------------------
#
# 抽出パイプライン (signals -> features -> sideslip) を KIT の走行にも
# そのまま通せるようにする。横滑りフィルタが本物の横滑りを拾えるかどうかを、
# β の実測がある走行で確かめるため。
#
# チャネルの対応と、そう決めた理由:
#   speed_mps  v_x を重心へ移した値。単軌道モデルの v は重心の前後速度。
#   steer_deg  delta_stm はタイヤ切れ角そのもの。車種設定側の steer_ratio を
#              1.0 にしてあるので、そのまま deg にして渡す。
#   yaw_rate   w_z。ISO 8855 で左が正。正規化の取り決めと同じ向き。
#   accel_y    a_y_ra (後軸位置)。重心へ移すには角加速度の項が要るが、
#              ヨーレートの微分で雑音が 30 m/s^2 規模まで増える。
#              CAN 由来の β 推定を検証したときも a_y_ra と突き合わせている
#              (docs/kit_msdm.md) ので、ここでも同じものを使う。

def segment_data(
    run: "Run",
    params: "VehicleParams",
    vehicle_name: str = "kit_msdm",
    dataset: str = "kit_msdm",
) -> Any:
    """1 走行を正規化済みの SegmentData にする。"""
    from .canonical import Channel, SegmentData, SegmentRef

    t = run.t
    yaw = run["w_z_cor_radps"]
    r = np.asarray(params.trvec("cor_ra"), dtype=float) + params.trvec("ra_cog")
    v_x, _v_y = translate_velocity(run["v_x_cor_mps"], run["v_y_cor_mps"], yaw, r)

    channels = {
        "speed_mps": Channel(t, v_x, "m/s", "continuous"),
        "steer_deg": Channel(t, np.degrees(run["delta_stm_rad"]), "deg", "continuous"),
        "yaw_rate": Channel(t, np.degrees(yaw), "deg/s", "continuous"),
        "accel_x": Channel(t, run["a_x_ra_mps2"], "m/s^2", "continuous"),
        "accel_y": Channel(t, run["a_y_ra_mps2"], "m/s^2", "continuous"),
    }
    ref = SegmentRef(
        path=run.path, dongle_id=run.surface, drive_id=run.name, index=0, dataset=dataset
    )
    return SegmentData(
        ref=ref,
        vehicle=vehicle_name,
        channels=channels,
        raw_can_loaded=False,
        notes=[
            "speed_mps は Correvit の v_x を重心へ剛体変換した値",
            "steer_deg はタイヤ切れ角。車種設定の steer_ratio は 1.0",
            "accel_y は後軸位置の実測値。重心へは移していない",
        ],
        meta={"surface": run.surface, "kind": run.kind, "mu": params.mu(run.surface)},
    )


def measured_sideslip_on_grid(
    run: "Run", params: "VehicleParams", t_grid: np.ndarray, at: str = "cog"
) -> np.ndarray:
    """光学式センサによる実測 β を、指定した時刻へ最近傍で並べ直す。

    1000 Hz を 20 Hz グリッドへ落とすだけなので内挿はしない。
    実測値そのものを比較に使いたいので、平滑化もしない。
    """
    beta = sideslip_deg(run, params, at=at)
    idx = np.searchsorted(run.t, t_grid).clip(0, run.t.size - 1)
    return beta[idx]
