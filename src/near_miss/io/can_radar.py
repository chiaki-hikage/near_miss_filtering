"""生 CAN からレーダトラックを組み立てる (L0)。

comma2k19 は processed_log にレーダトラックが展開済みで入っているが、
commaCarSegments は rlog の生 CAN しか無い。前方車との車間指標
(THW / TTC / カットイン) は現行の検出条件の中心にあるので、
生 CAN からも同じ RadarTracks を作れるようにする。

トラックの並びとビット位置は車種設定の `radar:` 節から渡される。
このモジュールは「1 トラック 1 アドレスで連番に並ぶ」という構造だけを知っている。
"""

from __future__ import annotations

import numpy as np

from ..config import RadarSpec
from .canonical import RadarTracks, RawCanFrames
from .can_decode import extract_bits, frame_mask


def _extract(payload: np.ndarray, d: dict) -> np.ndarray:
    """設定の 1 信号定義でビットを取り出して物理値にする。"""
    raw = extract_bits(
        payload, int(d["start_bit"]), int(d["length"]), bool(d.get("signed", False))
    )
    return raw * float(d.get("factor", 1.0)) + float(d.get("offset", 0.0))


def _hold_previous(src_t: np.ndarray, src_v: np.ndarray, at_t: np.ndarray) -> np.ndarray:
    """at_t の各時刻について、直近過去の src_v を返す。内挿はしない。

    スコアは離散値なので線形内挿すると意味の無い中間値になる。
    """
    out = np.full(at_t.shape, np.nan)
    if src_t.size == 0:
        return out
    idx = np.searchsorted(src_t, at_t, side="right") - 1
    ok = idx >= 0
    out[ok] = src_v[np.clip(idx, 0, src_t.size - 1)][ok]
    return out


def radar_from_can(raw: RawCanFrames, spec: RadarSpec) -> tuple[RadarTracks | None, list[str]]:
    """連番アドレスのトラックメッセージを 1 本の観測列にまとめる。

    有効判定は openpilot の radar_interface と同じ考え方にする:
    VALID が立っているか、スコアが閾値を超えていて距離が上限未満のものを残す。
    """
    notes: list[str] = []
    if not spec.enabled:
        return None, ["radar:disabled"]

    need = ("long_dist", "lat_dist", "rel_speed")
    missing = [k for k in need if k not in spec.signals]
    if missing:
        return None, [f"radar:missing_signal:{','.join(missing)}"]

    ts, dist, lat, vrel, tid, new = [], [], [], [], [], []
    n_track_msgs = 0
    for i in range(spec.track_count):
        addr = spec.track_first_id + i
        m = frame_mask(raw.address, raw.src, addr, spec.bus)
        if not m.any():
            continue
        n_track_msgs += 1
        t_i = raw.t[m]
        p_i = raw.payload_u64[m]

        d_i = _extract(p_i, spec.signals["long_dist"])
        y_i = _extract(p_i, spec.signals["lat_dist"]) * spec.lateral_sign
        v_i = _extract(p_i, spec.signals["rel_speed"])
        valid_i = (
            _extract(p_i, spec.signals["valid"]) > 0.5
            if "valid" in spec.signals
            else np.ones(t_i.shape, dtype=bool)
        )
        new_i = (
            _extract(p_i, spec.signals["new_track"])
            if "new_track" in spec.signals
            else np.zeros(t_i.shape)
        )

        # スコアは別アドレスの B 系列。同じ周期で流れるので直近値を当てる。
        if spec.score_first_id is not None and spec.score_signal is not None:
            ms = frame_mask(raw.address, raw.src, spec.score_first_id + i, spec.bus)
            score_i = (
                _hold_previous(raw.t[ms], _extract(raw.payload_u64[ms], spec.score_signal), t_i)
                if ms.any()
                else np.full(t_i.shape, np.nan)
            )
        else:
            score_i = np.full(t_i.shape, np.nan)

        scored = np.isfinite(score_i) & (score_i > spec.min_score)
        keep = (d_i < spec.max_distance_m) & (valid_i | scored)
        if not keep.any():
            continue

        ts.append(t_i[keep])
        dist.append(d_i[keep])
        lat.append(y_i[keep])
        vrel.append(v_i[keep])
        new.append(new_i[keep])
        # トラック ID はアドレスそのもの。comma2k19 の radar/value 列 5 と揃える。
        tid.append(np.full(int(keep.sum()), addr, dtype=np.int64))

    if n_track_msgs == 0:
        return None, [f"radar:no_frames:bus{spec.bus}:0x{spec.track_first_id:X}"]
    if not ts:
        notes.append("radar:no_valid_points")
        return RadarTracks(*(np.empty(0) for _ in range(6))), notes

    t = np.concatenate(ts)
    order = np.argsort(t, kind="stable")
    return (
        RadarTracks(
            t=t[order],
            distance_m=np.concatenate(dist)[order],
            lateral_m=np.concatenate(lat)[order],
            vrel_mps=np.concatenate(vrel)[order],
            track_id=np.concatenate(tid)[order],
            new_track=np.concatenate(new)[order].astype(np.int64),
        ),
        notes,
    )
