"""ヒヤリハット候補の canonical event (API 連携用の共通形式) を作る部品。

解析系統ごとの CSV はそのまま残し、ここではそれを共通の形へ写すための
**種別に依存しない**処理だけを持つ。CSV ごとの対応は
scripts/export_near_miss_canonical.py に書く。
(near_miss.io.canonical は生データの読み込み層で、これとは別物。)

  stable_event_id   再実行しても同じイベントなら同じ ID
  quantize_sec      ID に使う秒の丸め (10 ms 単位、10 進で四捨五入)
  Record            値を入れるたびに「どこから来たか」(copied / derived / constant) を記録する
  comma2k19_paths   生データの配置規則からの相対パス
  video_frame       セグメント内の秒 → 映像のフレーム番号 (近似)
  validate_events   JSON Schema による検証

仕様は docs/near_miss_canonical_schema.md。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "1.0"
ID_PREFIX = "nm_"
ID_HEX_LEN = 16          # SHA-256 の先頭 16 桁 (64 bit)
ID_SEC_QUANTUM = "0.01"  # start_sec を 10 ms 単位に丸めて ID に使う

TIME_BASE = "segment_relative_s"

# セグメント先頭 (= 0 秒) の定義。系統で違うので、イベントごとに明示する
ORIGIN_CAN = "can_primary_overlap_start"   # CAN の主系列が共通して始まる時刻 (near_miss.io.canonical の t_span)
ORIGIN_IMU = "imu_accel_first_sample"      # IMU 加速度の最初のサンプル (scripts/vibration/find_roll_bumps.py)

PROV_COPIED = "copied"
PROV_DERIVED = "derived"
PROV_CONSTANT = "constant"


# ---------------------------------------------------------------------------
# event_id
# ---------------------------------------------------------------------------

def quantize_sec(x: float) -> str:
    """ID に使う秒の文字列。10 ms 単位に、10 進表現で四捨五入する。

    2 進の浮動小数をそのまま丸めると 46.205 が 46.20 になるような揺れが出るので、
    repr の 10 進文字列を Decimal にしてから ROUND_HALF_UP で丸める。
    -0.00 は 0.00 にそろえる。
    """
    if x is None or not math.isfinite(float(x)):
        raise ValueError(f"ID に使う秒が有限でない: {x!r}")
    q = Decimal(repr(float(x))).quantize(Decimal(ID_SEC_QUANTUM), rounding=ROUND_HALF_UP)
    if q == 0:
        q = Decimal(ID_SEC_QUANTUM) * 0
    return f"{q:.2f}"


def id_key(dataset: str, drive_id: str, segment: int, start_sec: float,
           event_type: str, detector: str) -> str:
    """ID の元になる正規化文字列。各要素は前後の空白を除き、'|' 以外の区切りを使わない。

    drive_id 自体が '<dongle>|<route>' と '|' を含むので、区切りは改行にする。
    """
    parts = [dataset, drive_id, str(int(segment)), quantize_sec(start_sec), event_type, detector]
    for p in parts:
        if not isinstance(p, str) or not p.strip() or "\n" in p:
            raise ValueError(f"ID の要素が不正: {parts!r}")
    return "\n".join(p.strip() for p in parts)


def stable_event_id(dataset: str, drive_id: str, segment: int, start_sec: float,
                    event_type: str, detector: str) -> str:
    """nm_ + SHA-256(id_key) の先頭 16 桁。"""
    key = id_key(dataset, drive_id, segment, start_sec, event_type, detector)
    return ID_PREFIX + hashlib.sha256(key.encode("utf-8")).hexdigest()[:ID_HEX_LEN]


# ---------------------------------------------------------------------------
# 値の変換
# ---------------------------------------------------------------------------

def clean(v: Any) -> Any:
    """pandas / numpy の値を JSON に出せる形にする。欠測 (NaN・空文字) は None。"""
    if v is None:
        return None
    try:
        import numpy as np
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            v = float(v)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, str):
        return v if v != "" else None
    return v


def split_pipe(v: Any) -> list[str] | None:
    """'a|b|c' → ['a', 'b', 'c']。欠測は None (空リストにしない)。"""
    v = clean(v)
    if v is None:
        return None
    return [s for s in str(v).split("|") if s != ""]


def kmh_to_mps(v: Any, ndigits: int = 3) -> float | None:
    v = clean(v)
    return None if v is None else round(float(v) / 3.6, ndigits)


def to_bool(v: Any) -> bool | None:
    """CSV の True / False (bool・文字列) を bool に。それ以外 (欠測を含む) は None。"""
    v = clean(v)
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v in ("True", "False", "true", "false"):
        return v in ("True", "true")
    return None


def utc_iso(v: Any) -> str | None:
    """'YYYY-MM-DD HH:MM:SS' (UTC) → 'YYYY-MM-DDTHH:MM:SSZ'。読めなければ None。"""
    v = clean(v)
    if v is None:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(str(v), fmt).replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    return None


def video_frame(sec: float | None, fps: float, n_frames: int) -> int | None:
    """セグメント内の秒 → フレーム番号 round(sec × fps)。セグメントの外なら None。

    既存の video_frame 列と同じ式。映像の先頭は IMU / CAN の先頭より 10〜75 ms 早いので
    ±2 フレーム程度の近似 (正確には global_pose/frame_times で引く)。
    """
    if sec is None or not math.isfinite(sec):
        return None
    f = int(round(sec * fps))
    return f if 0 <= f < n_frames else None


# ---------------------------------------------------------------------------
# 生データの配置
# ---------------------------------------------------------------------------

def comma2k19_paths(chunk: str | None, drive_id: str, segment: int) -> dict[str, str | None]:
    """comma2k19 のセグメントの中の、映像・CAN・IMU・GNSS の相対パス (dataset ルートから)。

    配置: <Chunk_N>/<dongle>|<route>/<segment>/{video.hevc, processed_log/CAN, processed_log/IMU, global_pose}
    チャンクが分からなければ全部 None。
    """
    if not chunk:
        return dict(video_uri=None, can_uri=None, imu_uri=None, gnss_uri=None)
    d = f"{chunk}/{drive_id}/{int(segment)}"
    return dict(video_uri=f"{d}/video.hevc", can_uri=f"{d}/processed_log/CAN",
                imu_uri=f"{d}/processed_log/IMU", gnss_uri=f"{d}/global_pose")


def ccs_paths(drive_id: str, segment: int) -> dict[str, str | None]:
    """commaCarSegments の rlog の相対パス。映像・IMU・GNSS はデータセットに無い。"""
    dongle, route = drive_id.split("|", 1)
    return dict(video_uri=None, can_uri=f"segments/{dongle}/{route}/{int(segment)}/rlog.zst",
                imu_uri=None, gnss_uri=None)


def chunk_index(raw_root: Path) -> dict[str, str]:
    """drive_id → チャンク名 (raw_root/Chunk_*/<drive_id>/ の配置から)。"""
    out: dict[str, str] = {}
    for chunk in sorted(Path(raw_root).glob("Chunk_*")):
        for d in chunk.iterdir():
            if d.is_dir() and "|" in d.name:
                if d.name in out and out[d.name] != chunk.name:
                    raise ValueError(f"{d.name} が複数のチャンクにある")
                out[d.name] = chunk.name
    return out


# ---------------------------------------------------------------------------
# 値の出どころを記録しながらイベントを組み立てる
# ---------------------------------------------------------------------------

@dataclass
class Record:
    """canonical event 1 件。値を入れるたびに出どころを記録する。

    path はドット区切り ("time.start_sec")。出どころは
      copied   元 CSV の列の値をそのまま (必要なら型・表記だけ変えて) 入れた
      derived  元 CSV の列から式で出した
      constant 設定・ディレクトリ名・配置規則など、行の外から入れた
    null を入れたものは missing として数える。
    """

    doc: dict[str, Any] = field(default_factory=dict)
    prov: dict[str, tuple[str, str]] = field(default_factory=dict)

    def _put(self, path: str, value: Any) -> None:
        node = self.doc
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    def copy(self, path: str, value: Any, src: str) -> Any:
        value = clean(value) if not isinstance(value, (list, dict)) else value
        self._put(path, value)
        if value is not None:
            self.prov[path] = (PROV_COPIED, src)
        return value

    def derive(self, path: str, value: Any, rule: str) -> Any:
        value = clean(value) if not isinstance(value, (list, dict)) else value
        self._put(path, value)
        if value is not None:
            self.prov[path] = (PROV_DERIVED, rule)
        return value

    def const(self, path: str, value: Any, src: str) -> Any:
        self._put(path, value)
        if value is not None:
            self.prov[path] = (PROV_CONSTANT, src)
        return value

    def null(self, path: str) -> None:
        self._put(path, None)

    def get(self, path: str) -> Any:
        node: Any = self.doc
        for k in path.split("."):
            node = node[k]
        return node


def null_leaves(doc: Any, prefix: str = "") -> list[str]:
    """値が null の葉のパス。attributes と "_" で始まる内部用のキーは対象外。"""
    out: list[str] = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            p = f"{prefix}.{k}" if prefix else k
            if p == "attributes" or k.startswith("_"):
                continue
            if v is None:
                out.append(p)
            else:
                out.extend(null_leaves(v, p))
    return out


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------

def load_schema(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_events(events: Iterable[dict], schema: dict) -> list[tuple[str, str]]:
    """全件を検証し、(event_id, エラー) の並びを返す。空なら全件が通った。"""
    from jsonschema import Draft202012Validator
    from jsonschema import FormatChecker

    v = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = []
    for e in events:
        for err in v.iter_errors(e):
            loc = "/".join(str(p) for p in err.absolute_path)
            errors.append((str(e.get("event_id")), f"{loc}: {err.message}"))
    return errors


def git_commit(repo: Path) -> str | None:
    """現在の HEAD (作業ツリーに変更があれば '-dirty' を付ける)。git が無ければ None。"""
    import subprocess

    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("-dirty" if dirty else "")
