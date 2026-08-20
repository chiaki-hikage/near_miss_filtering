"""comma1M (HuggingFace: commaai/comma1M) へのアクセス。

このデータセットは 1 分のセグメント単位で、道路カメラ映像と
オフライン推定の自己位置 (localizer.safetensors) を持つ。
commaCarSegments と違って CAN は入っていないが、位置が入っているので
地域による絞り込みができる。

公式 Dataset Card に記載があるのは states の 2 つの区間だけ:
    latitude, longitude, _ = ecef2geodetic(*states[:, :3].T)
    speed = np.linalg.norm(states[:, 7:10], axis=1)
それ以外の列の意味は公開仕様に無いので使わない。

映像は 1 セグメント 75 MB / カメラある。位置だけ欲しい段階で
localizer 全体 (2.5 MB) を落とすのも高いので、safetensors のヘッダを
読んでから必要な行だけ HTTP Range で取る経路を用意してある。
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = REPO_ROOT / "raw_data" / "comma1M"

REPO_ID = "commaai/comma1M"
API_TREE = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main"
RESOLVE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"

# 公式 Dataset Card に載っている列だけを定数にしておく。
ECEF_COLS = slice(0, 3)
VELOCITY_COLS = slice(7, 10)

STATE_DTYPE = np.dtype("<f8")
STATE_ITEMSIZE = STATE_DTYPE.itemsize


def segment_url(segment_id: str, name: str = "localizer.safetensors") -> str:
    return f"{RESOLVE}/data/{segment_id}/{name}"


def list_segments(session: Any, limit: int | None = None) -> list[str]:
    """data/ 直下のセグメント ID をすべて列挙する。"""
    url = f"{API_TREE}/data?limit=1000"
    out: list[str] = []
    while url:
        r = session.get(url, timeout=60)
        r.raise_for_status()
        out += [e["path"].split("/", 1)[1] for e in r.json() if e["type"] == "directory"]
        if limit is not None and len(out) >= limit:
            return out[:limit]
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("link", ""))
        url = m.group(1) if m else None
    return out


# --- safetensors のヘッダと部分読み出し ----------------------------------

@dataclass
class SafeHeader:
    """safetensors のヘッダ。data_offsets はデータ領域先頭からの相対値。"""

    data_start: int
    tensors: dict[str, dict[str, Any]]

    def offset(self, name: str) -> tuple[int, int]:
        a, b = self.tensors[name]["data_offsets"]
        return self.data_start + a, self.data_start + b

    def shape(self, name: str) -> list[int]:
        return list(self.tensors[name]["shape"])


def parse_safeheader(head: bytes) -> SafeHeader:
    n = struct.unpack("<Q", head[:8])[0]
    if len(head) < 8 + n:
        raise ValueError(f"ヘッダが途中で切れている (必要 {8 + n} バイト, 取得 {len(head)})")
    hdr = json.loads(head[8 : 8 + n])
    hdr.pop("__metadata__", None)
    return SafeHeader(data_start=8 + n, tensors=hdr)


def _get_range(session: Any, url: str, start: int, length: int) -> tuple[bytes, str]:
    r = session.get(url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=60)
    r.raise_for_status()
    if r.status_code != 206:
        raise ValueError(f"Range 非対応 (status={r.status_code}) {url}")
    return r.content, r.url


def fetch_state_rows(
    session: Any, segment_id: str, rows: tuple[float, ...] = (0.0, 0.5, 1.0), n_cols: int = 10
) -> dict[str, Any]:
    """localizer から指定位置の states 行だけを Range で取り出す。

    rows は 0.0〜1.0 の相対位置。n_cols は行の先頭から何列読むか。
    公式に意味が分かっているのは 0..2 (ECEF) と 7..9 (速度) なので既定は 10 列。
    1 セグメントあたりの転送量は数百バイト。
    """
    url = segment_url(segment_id)
    head, resolved = _get_range(session, url, 0, 2048)
    hdr = parse_safeheader(head)
    n_rows, n_state = hdr.shape("states")
    base, _ = hdr.offset("states")
    stride = n_state * STATE_ITEMSIZE

    vals = {}
    for frac in rows:
        i = min(n_rows - 1, max(0, int(round(frac * (n_rows - 1)))))
        buf, _ = _get_range(session, resolved, base + i * stride, n_cols * STATE_ITEMSIZE)
        vals[frac] = np.frombuffer(buf, dtype=STATE_DTYPE)
    return {"segment_id": segment_id, "n_rows": int(n_rows), "n_state": int(n_state), "rows": vals}


# --- localizer 全体 -------------------------------------------------------

def local_path(segment_id: str, cache: Path = DEFAULT_CACHE, name: str = "localizer.safetensors") -> Path:
    suffix = ".safetensors" if name.endswith(".safetensors") else Path(name).suffix
    stem = segment_id if name == "localizer.safetensors" else f"{segment_id}.{Path(name).stem}"
    return cache / f"{stem}{suffix}"


def ensure_file(session: Any, segment_id: str, cache: Path = DEFAULT_CACHE,
                name: str = "localizer.safetensors") -> Path:
    """未取得ならダウンロードして、ローカルのパスを返す。"""
    path = local_path(segment_id, cache, name)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with session.get(segment_url(segment_id, name), stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(path)
    return path


@dataclass
class Localizer:
    """localizer.safetensors の中身のうち、仕様が公開されている部分。"""

    segment_id: str
    t: np.ndarray            # [s] 端末の boot 時刻。壁時計ではない
    ecef: np.ndarray         # (N, 3) [m]
    velocity: np.ndarray     # (N, 3) [m/s] 速度ベクトル
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.velocity, axis=1)

    @property
    def rate_hz(self) -> float:
        dt = np.median(np.diff(self.t))
        return float(1.0 / dt) if dt > 0 else float("nan")


def load_localizer(path: Path, segment_id: str | None = None) -> Localizer:
    from pymap3d import ecef2geodetic
    from safetensors.numpy import load_file

    d = load_file(str(path))
    states = d["states"]
    ecef = states[:, ECEF_COLS]
    lat, lon, alt = ecef2geodetic(*ecef.T)
    return Localizer(
        segment_id=segment_id or path.stem,
        t=d["t"],
        ecef=ecef,
        velocity=states[:, VELOCITY_COLS],
        lat=np.asarray(lat), lon=np.asarray(lon), alt=np.asarray(alt),
    )


def iter_localizers(paths: Iterator[Path]) -> Iterator[Localizer]:
    for p in paths:
        yield load_localizer(p)


# --- 自己運動から正規化信号へ -------------------------------------------
#
# comma1M には CAN が無い。使えるのは公式に意味が確認できる
# 位置 (ECEF) と速度ベクトルだけなので、正規化チャネルは 2 本しか作らない。
#
#   speed_mps  速度ベクトルのノルム。公式の例と同じ
#   yaw_rate   速度ベクトルの向きの変化率 [deg/s]、左回りが正
#
# yaw_rate について:
#   これは車両のヨーレートセンサの値ではなく、速度ベクトルの向き
#   (course over ground) の変化率である。両者は横滑り角のぶんだけ食い違う。
#   通常走行では差は小さいが、滑っている最中は一致しない。
#   このデータセットでは「進路がどう変わったか」しか分からないので、
#   スリップの有無を判定せず、進路の変化そのものを見る用途に限る。

def ecef_to_enu_velocity(vel_ecef: np.ndarray, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """ECEF の速度ベクトルを各点のローカル ENU へ回す。"""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    vx, vy, vz = vel_ecef[:, 0], vel_ecef[:, 1], vel_ecef[:, 2]
    east = -so * vx + co * vy
    north = -sl * co * vx - sl * so * vy + cl * vz
    up = cl * co * vx + cl * so * vy + sl * vz
    return np.column_stack([east, north, up])


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    k = np.ones(window) / window
    pad = window // 2
    return np.convolve(np.pad(x, pad, mode="edge"), k, mode="same")[pad : pad + x.size]


def course_rate_dps(
    t: np.ndarray, vel_enu: np.ndarray, smooth_window_s: float, min_speed_mps: float
) -> tuple[np.ndarray, np.ndarray]:
    """速度ベクトルの向きの変化率 [deg/s] を返す。左回りが正。

    向きは低速で定義できない。min_speed_mps 未満は NaN にして、
    後段に「取れなかった」ことを伝える (埋めない)。
    """
    ve, vn = vel_enu[:, 0], vel_enu[:, 1]
    ground = np.hypot(ve, vn)
    course = np.unwrap(np.arctan2(vn, ve))          # 反時計回りが正
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01
    win = max(1, int(round(smooth_window_s / dt)))
    course_s = _smooth(course, win)
    rate = np.rad2deg(np.gradient(course_s, t))
    rate[ground < min_speed_mps] = np.nan
    return rate, ground


def segment_data(
    loc: Localizer,
    ref: Any,
    smooth_window_s: float = 0.1,
    min_speed_mps: float = 2.0,
) -> Any:
    """Localizer を正規化済み SegmentData に変換する。"""
    from .canonical import Channel, SegmentData

    vel_enu = ecef_to_enu_velocity(loc.velocity, loc.lat, loc.lon)
    rate, _ = course_rate_dps(loc.t, vel_enu, smooth_window_s, min_speed_mps)

    channels = {
        "speed_mps": Channel(t=loc.t, v=loc.speed, unit="m/s", kind="continuous"),
        "yaw_rate": Channel(t=loc.t, v=rate, unit="deg/s", kind="continuous"),
        # 検出には使わないが、候補の確認・地図表示のために残す
        "lat_deg": Channel(t=loc.t, v=loc.lat, unit="deg", kind="continuous"),
        "lon_deg": Channel(t=loc.t, v=loc.lon, unit="deg", kind="continuous"),
        "alt_m": Channel(t=loc.t, v=loc.alt, unit="m", kind="continuous"),
    }
    return SegmentData(
        ref=ref,
        vehicle="comma1m_localizer",
        channels=channels,
        radar=None,
        raw_can_loaded=False,
        notes=[
            "yaw_rate は速度ベクトルの向きの変化率 (course rate)。車両のヨーレートではない",
            f"course は {min_speed_mps} m/s 未満で NaN",
        ],
        meta={
            "lat_start": float(loc.lat[0]), "lon_start": float(loc.lon[0]),
            "alt_start": float(loc.alt[0]),
            "rate_hz": loc.rate_hz,
        },
    )


def find_segments(cache_dir: str | Path = DEFAULT_CACHE, names: list[str] | None = None) -> list[Any]:
    """ローカルに localizer がある segment の一覧を返す。

    comma1M のセグメントは互いに独立で、ドライブへまとめる情報が無い。
    連結すると存在しない区間を内挿してしまうので、drive_id は segment_id 自身にして
    1 セグメント = 1 ブロックとして扱う。
    """
    from .canonical import SegmentRef

    cache = Path(cache_dir)
    wanted = set(names) if names is not None else None
    refs = []
    for p in sorted(cache.glob("*.safetensors")):
        sid = p.stem
        if wanted is not None and sid not in wanted:
            continue
        refs.append(SegmentRef(path=p, dongle_id="", drive_id=sid, index=0,
                               dataset="comma1M", platform=""))
    return refs


def load_segment(ref: Any, vehicle: Any = None, with_raw_can: bool = False,
                 smooth_window_s: float = 0.1, min_speed_mps: float = 2.0) -> Any:
    loc = load_localizer(ref.path, ref.drive_id)
    return segment_data(loc, ref, smooth_window_s=smooth_window_s, min_speed_mps=min_speed_mps)


# --- 映像の部分取得 -------------------------------------------------------
#
# fcamera.hevc は 1 セグメント 75 MB ある。候補の前後数秒を見るだけなら
# 全部を落とす必要はない。frame_info.safetensors (50 KB) に
#
#   <camera>/index  (1201, 2) uint32   列 0 = フレーム種別、列 1 = ファイル内バイト位置
#   <camera>/t      (1200,)   float64  各フレームの時刻 (localizer の t と同じ boot time)
#   <camera>/global_prefix (82,) uint8 VPS/SPS/PPS (3 バイト開始コード)
#
# が入っている。実測で確認したこと:
#   - index[-1, 1] = 74,965,933 はファイルサイズと一致する (最終行は番兵)
#   - 列 0 は 2 が鍵フレーム (1200 枚中 40 枚 = 30 フレームおき = 1.5 秒おき)、1 が予測フレーム
#   - ファイル先頭 83 バイトはパラメータセット。global_prefix と中身は同じで、
#     開始コードが 4 バイトか 3 バイトかだけが違う
# したがって「直前の鍵フレームから必要な範囲までのバイト列」を Range で取り、
# 先頭に global_prefix を付ければ、それだけで復号できる HEVC になる。

KEYFRAME_TYPE = 2


def load_frame_info(path: Path) -> dict[str, Any]:
    from safetensors.numpy import load_file

    return load_file(str(path))


def clip_frame_range(cam_t: np.ndarray, t_start: float, t_end: float) -> tuple[int, int]:
    """時刻の範囲を包むフレーム番号 [f0, f1] を返す。"""
    n = int(cam_t.size)
    f0 = int(np.clip(np.searchsorted(cam_t, t_start, side="right") - 1, 0, n - 1))
    f1 = int(np.clip(np.searchsorted(cam_t, t_end, side="left"), 0, n - 1))
    return f0, max(f1, f0)


def clip_byte_range(index: np.ndarray, f0: int, f1: int) -> tuple[int, int, int]:
    """[f0, f1] を復号するために必要なバイト範囲と、実際の開始フレームを返す。

    予測フレームの途中からは復号できないので、f0 以前の直近の鍵フレームまで戻る。
    """
    kinds = index[:-1, 0]
    keys = np.flatnonzero(kinds == KEYFRAME_TYPE)
    prior = keys[keys <= f0]
    k0 = int(prior[-1]) if prior.size else 0
    byte0 = int(index[k0, 1])
    byte1 = int(index[min(f1 + 1, index.shape[0] - 1), 1])
    return byte0, byte1, k0


def fetch_clip(
    session: Any,
    segment_id: str,
    t_start: float,
    t_end: float,
    out_path: Path,
    camera: str = "fcamera",
    cache: Path = DEFAULT_CACHE,
) -> dict[str, Any]:
    """候補の前後だけを切り出した HEVC を書き出す。

    戻り値には、切り出したフレーム範囲・先頭フレームの時刻・転送量を入れる。
    先頭は鍵フレームまで戻るので、指定した t_start より前から始まる。
    """
    info_path = ensure_file(session, segment_id, cache, name="frame_info.safetensors")
    info = load_frame_info(info_path)
    cam_t = np.asarray(info[f"{camera}/t"], dtype=float)
    index = np.asarray(info[f"{camera}/index"])
    prefix = bytes(np.asarray(info[f"{camera}/global_prefix"])[0].tobytes())

    f0, f1 = clip_frame_range(cam_t, t_start, t_end)
    byte0, byte1, k0 = clip_byte_range(index, f0, f1)

    url = segment_url(segment_id, f"{camera}.hevc")
    r = session.get(url, headers={"Range": f"bytes={byte0}-{byte1 - 1}"}, timeout=300)
    r.raise_for_status()
    if r.status_code != 206:
        raise ValueError(f"Range 非対応 (status={r.status_code}) {url}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(prefix + r.content)
    return {
        "segment_id": segment_id,
        "camera": camera,
        "path": str(out_path),
        "frame_start": k0,
        "frame_target": f0,
        "frame_end": f1,
        "n_frames": f1 - k0 + 1,
        "t_first": float(cam_t[k0]),
        "t_target": float(cam_t[f0]),
        "bytes": len(r.content),
        "fps": float(1.0 / np.median(np.diff(cam_t))),
    }
