"""commaCarSegments の読み出し (L0)。

https://huggingface.co/datasets/commaai/commaCarSegments

comma2k19 との違い:
    * 車種ごとに分類済み。`database.json` が 車種キー -> セグメント名の一覧
    * 1 セグメント 60 秒。中身は `rlog.zst` だけで、映像も processed_log も無い
    * したがって車速・舵角・輪速・レーダも全部 生 CAN から作る
    * 走行環境は限定されない (comma2k19 は CA-280 の高速に限られる)

セグメント名は "<dongle_id>/<route_id>/<segment番号>/s" の形。
実データの URL は
    {BASE}/segments/<dongle_id>/<route_id>/<segment番号>/rlog.zst

このモジュールが返すのは comma2k19 と同じ SegmentData なので、
再サンプル以降の層はどちらのデータセットか知らずに動く。
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT, VehicleConfig
from .can_decode import decode_channels, openpilot_tx_channel
from .can_radar import radar_from_can
from .canonical import SegmentData, SegmentRef
from .rlog import read_rlog

log = logging.getLogger(__name__)

DATASET = "comma_car_segments"
BASE_URL = "https://huggingface.co/datasets/commaai/commaCarSegments/resolve/main"
DEFAULT_CACHE = REPO_ROOT / "raw_data" / DATASET

_NAME_RE = re.compile(r"^(?P<dongle>[0-9a-f]{16})/(?P<route>[^/]+)/(?P<num>\d+)(/.*)?$")


@dataclass(frozen=True)
class SegmentName:
    """database.json のセグメント名を分解したもの。"""

    dongle_id: str
    route_id: str
    index: int

    @classmethod
    def parse(cls, name: str) -> "SegmentName":
        m = _NAME_RE.match(name.strip())
        if m is None:
            raise ValueError(f"セグメント名を解釈できません: {name!r}")
        return cls(m.group("dongle"), m.group("route"), int(m.group("num")))

    @property
    def drive_id(self) -> str:
        """ドライブの識別子。comma2k19 の '<dongle>|<日時>' と同じ形にそろえる。"""
        return f"{self.dongle_id}|{self.route_id}"

    @property
    def rel_path(self) -> Path:
        return Path(self.dongle_id) / self.route_id / str(self.index) / "rlog.zst"

    @property
    def url(self) -> str:
        return f"{BASE_URL}/segments/{self.dongle_id}/{self.route_id}/{self.index}/rlog.zst"


# ---------------------------------------------------------------------------
# セグメント一覧 (database.json)
# ---------------------------------------------------------------------------
def database_path(cache_dir: str | Path = DEFAULT_CACHE) -> Path:
    return Path(cache_dir) / "database.json"


def fetch_database(cache_dir: str | Path = DEFAULT_CACHE, force: bool = False) -> Path:
    """database.json を取得する (約 9 MB)。既にあれば何もしない。"""
    path = database_path(cache_dir)
    if path.is_file() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    _download(f"{BASE_URL}/database.json", path)
    return path


def load_database(cache_dir: str | Path = DEFAULT_CACHE) -> dict[str, list[str]]:
    path = database_path(cache_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} がありません。先に scripts/fetch_car_segments.py --list を実行してください"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def platform_segments(platform: str, cache_dir: str | Path = DEFAULT_CACHE) -> list[str]:
    """指定車種のセグメント名を、ルート順・セグメント番号順に並べて返す。"""
    db = load_database(cache_dir)
    if platform not in db:
        raise KeyError(f"車種キーがありません: {platform}")
    names = db[platform]
    return sorted(names, key=lambda n: (SegmentName.parse(n).drive_id, SegmentName.parse(n).index))


def select_segments(
    platform: str,
    limit: int | None = None,
    routes: int | None = None,
    per_route: int | None = None,
    cache_dir: str | Path = DEFAULT_CACHE,
) -> list[str]:
    """取得対象のセグメント名を選ぶ。

    routes / per_route を指定すると「同じルートの連続したセグメント」を選ぶ。
    60 秒境界を跨ぐイベントを連結して扱えるかを確かめるにはこちらが要る。
    """
    names = platform_segments(platform, cache_dir)
    if routes is None and per_route is None:
        return names[:limit] if limit else names

    by_route: dict[str, list[str]] = {}
    for n in names:
        by_route.setdefault(SegmentName.parse(n).drive_id, []).append(n)

    # 連番が長く続くルートを優先する。飛び飛びのルートは連結できない。
    ordered = sorted(by_route.items(), key=lambda kv: -_max_run_length(kv[1]))
    picked: list[str] = []
    for _, segs in ordered[: routes or len(ordered)]:
        run = _longest_run(segs)
        picked.extend(run[: per_route] if per_route else run)
        if limit and len(picked) >= limit:
            break
    return picked[:limit] if limit else picked


def _indices(names: list[str]) -> list[int]:
    return [SegmentName.parse(n).index for n in names]


def _longest_run(names: list[str]) -> list[str]:
    """セグメント番号が連番になっている最長のかたまりを返す。"""
    names = sorted(names, key=lambda n: SegmentName.parse(n).index)
    idx = _indices(names)
    best: list[str] = []
    cur: list[str] = []
    for i, n in enumerate(names):
        if cur and idx[i] != idx[i - 1] + 1:
            cur = []
        cur.append(n)
        if len(cur) > len(best):
            best = list(cur)
    return best


def _max_run_length(names: list[str]) -> int:
    return len(_longest_run(names))


# ---------------------------------------------------------------------------
# 取得とキャッシュ
# ---------------------------------------------------------------------------
def local_path(name: str | SegmentName, cache_dir: str | Path = DEFAULT_CACHE) -> Path:
    sn = name if isinstance(name, SegmentName) else SegmentName.parse(name)
    return Path(cache_dir) / "segments" / sn.rel_path


def _download(url: str, dest: Path, timeout: float = 120.0, retries: int = 3) -> Path:
    """一時ファイルへ落としてから置き換える。途中で落ちた残骸を残さない。"""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as exc:  # ネットワークの一時障害は数回まで許す
            last = exc
            log.debug("取得失敗 (%d/%d) %s: %s", attempt + 1, retries, url, exc)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"取得に失敗しました: {url}") from last


def ensure_segment(name: str, cache_dir: str | Path = DEFAULT_CACHE) -> Path:
    """ローカルに無ければ取得する。あるものは触らない。"""
    dest = local_path(name, cache_dir)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    return _download(SegmentName.parse(name).url, dest)


def ensure_segments(
    names: list[str],
    cache_dir: str | Path = DEFAULT_CACHE,
    workers: int = 4,
    on_done=None,
) -> dict[str, Path | Exception]:
    """複数セグメントをまとめて取得する。1 件の失敗で全体を止めない。"""
    out: dict[str, Path | Exception] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(ensure_segment, n, cache_dir): n for n in names}
        for fut in as_completed(futs):
            n = futs[fut]
            try:
                out[n] = fut.result()
            except Exception as exc:
                out[n] = exc
                log.warning("取得できませんでした %s: %s", n, exc)
            if on_done is not None:
                on_done(n, out[n])
    return out


# ---------------------------------------------------------------------------
# 読み出し
# ---------------------------------------------------------------------------
def segment_ref(name: str, platform: str = "", cache_dir: str | Path = DEFAULT_CACHE) -> SegmentRef:
    sn = SegmentName.parse(name)
    return SegmentRef(
        path=local_path(sn, cache_dir),
        dongle_id=sn.dongle_id,
        drive_id=sn.drive_id,
        index=sn.index,
        dataset=DATASET,
        platform=platform,
    )


def find_segments(
    cache_dir: str | Path = DEFAULT_CACHE, platform: str = ""
) -> list[SegmentRef]:
    """既にローカルにある rlog.zst をセグメントとして列挙する。"""
    root = Path(cache_dir) / "segments"
    refs: list[SegmentRef] = []
    for p in sorted(root.rglob("rlog.zst")):
        try:
            index = int(p.parent.name)
        except ValueError:
            continue
        route_dir = p.parent.parent
        dongle = route_dir.parent.name
        refs.append(
            SegmentRef(
                path=p,
                dongle_id=dongle,
                drive_id=f"{dongle}|{route_dir.name}",
                index=index,
                dataset=DATASET,
                platform=platform,
            )
        )
    return sorted(refs, key=lambda r: (r.drive_id, r.index))


def load_segment(
    ref: SegmentRef,
    vehicle: VehicleConfig | None = None,
    with_raw_can: bool = True,
    with_radar: bool = True,
) -> SegmentData:
    """rlog を読んで正規化済みの SegmentData にする。

    with_raw_can は comma2k19 側と引数を揃えるためにある。commaCarSegments には
    復号済みの信号が無いので、False にすると信号が 1 本も出ない。
    """
    seg = SegmentData(
        ref=ref, vehicle=vehicle.name if vehicle else "unknown", channels={}, notes=[]
    )
    if vehicle is None:
        seg.notes.append("skipped:no_vehicle_config")
        return seg
    if not with_raw_can:
        seg.notes.append("skipped:raw_can_disabled")
        return seg

    data = read_rlog(ref.path)
    seg.notes.extend(data.notes)
    seg.meta["event_counts"] = data.event_counts
    seg.meta["car_params"] = data.car_params

    # 車種設定と rlog の carParams が食い違ったら、黙って進めずに注記に残す。
    fingerprint = str(data.car_params.get("carFingerprint", ""))
    if fingerprint and vehicle.platforms and fingerprint not in vehicle.platforms:
        seg.notes.append(f"fingerprint_mismatch:{fingerprint}")

    channels, notes = decode_channels(data.can, vehicle)
    seg.channels.update(channels)
    seg.notes.extend(notes)
    seg.raw_can_loaded = bool(channels)

    tx = openpilot_tx_channel(data.can, vehicle)
    if tx is not None:
        seg.channels["op_tx"] = tx
    seg.channels.update(data.panda_channels)

    if with_radar and vehicle.radar is not None:
        radar, radar_notes = radar_from_can(data.can, vehicle.radar)
        seg.radar = radar
        seg.notes.extend(radar_notes)
    elif with_radar:
        seg.notes.append("radar:no_config")

    return seg
