"""候補区間の統合とスコア付け (L4)。

個々のイベントは短く細切れになるため、前後に余裕を付けて統合し、
確認単位となる「候補区間」にまとめる。スコアは順位付けのための目安であって、
危険度の絶対尺度ではない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .detectors import Event
from .signals import GriddedSignals

# 候補区間ごとに残す文脈量。(列名, 集約方法)
_CONTEXT = (
    ("v_mps", "mean"),
    ("v_mps", "min"),
    ("ax_mps2", "min"),
    ("ax_mps2", "max"),
    ("jerk_mps3", "min"),
    ("jerk_win_min", "min"),
    ("jerk_win_mean", "min"),
    ("steer_rate_dps", "absmax"),
    ("ay_kin_mps2", "absmax"),
    ("lat_jerk_mps3", "absmax"),
    ("lat_jerk_win_absmax", "max"),
    ("ay_can_mps2", "absmax"),
    ("yaw_rate_dps", "absmax"),
    ("steer_reversals", "max"),
    ("weave_reversals", "max"),
    ("net_heading_win_deg", "max"),
    ("ws_spread_mps", "max"),
    ("ws_spread_smooth_mps", "max"),
    ("ws_spread_excess_mps", "max"),
    ("lc_offset_m", "absmax"),
    ("lc_heading_amp_deg", "max"),
    ("lead_distance_m", "min"),
    ("lead_vrel_mps", "min"),
    ("thw_s", "min"),
    ("cut_in_distance_drop_m", "max"),
    ("cut_in_thw_after_s", "min"),
    ("lead_target_speed_mps", "min"),
    ("ttc_s", "min"),
    ("yaw_residual_dps", "absmax"),
    ("beta_model_deg", "absmax"),
    ("counter_steer_active", "max"),
    ("s_evasion_excursion_m", "absmax"),
    ("thw_rate_s_per_s", "min"),
    ("abs_active_flag", "max"),
    ("vsc_active_flag", "max"),
    ("slip_warn", "max"),
    ("brake_mc_mpa", "max"),
    ("abs_fault", "max"),
    ("brake_pressed", "max"),
    ("decel_flag", "max"),
    # 以下は commaCarSegments 側でのみ復号できる。comma2k19 では列ごと存在しない。
    ("gvc_mps2", "min"),
    ("gas_pedal_pct", "max"),
    ("brake_position", "max"),
    ("steer_torque_driver", "absmax"),
    ("precollision_active", "max"),
    ("precollision_force_n", "min"),
    ("acc_braking", "max"),
    ("cruise_active", "mean"),
    ("op_tx", "mean"),
    ("op_engaged", "mean"),
)


@dataclass
class Candidate:
    """確認単位となる区間。"""

    drive_id: str
    segment_id: str
    t_start: float
    t_end: float
    severity: float
    event_types: list[str] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    context: dict[str, float] = field(default_factory=dict)


def event_severity(event: Event, max_severity: float) -> float:
    """閾値をどれだけ超えたかを、閾値の大きさで正規化した値。

    閾値ちょうどで 1.0。重み付けは設定側の weight で行う。
    """
    scale = abs(event.threshold) if event.threshold != 0 else 1.0
    return event.weight * float(min(1.0 + max(event.exceedance, 0.0) / scale, max_severity))


def _aggregate(x: np.ndarray, how: str) -> float:
    if not np.isfinite(x).any():
        return float("nan")
    if how == "mean":
        return float(np.nanmean(x))
    if how == "min":
        return float(np.nanmin(x))
    if how == "max":
        return float(np.nanmax(x))
    if how == "absmax":
        i = int(np.nanargmax(np.abs(x)))
        return float(x[i])
    raise ValueError(f"未知の集約方法: {how}")


def _context_for(gs: GriddedSignals, t_start: float, t_end: float) -> dict[str, float]:
    t = gs.t
    m = (t >= t_start) & (t <= t_end)
    out: dict[str, float] = {}
    if not m.any():
        return out
    for col, how in _CONTEXT:
        if col not in gs.df.columns:
            continue
        out[f"{col}_{how}"] = _aggregate(gs.df[col].to_numpy()[m], how)
    return out


def build_candidates(gs: GriddedSignals, events: list[Event], cfg: dict[str, Any]) -> list[Candidate]:
    """イベントを時間方向にまとめて候補区間を作る。"""
    sc = cfg["scoring"]
    if not events:
        return []

    pad = float(sc["window_pad_s"])
    gap = float(sc["merge_gap_s"])
    max_sev = float(sc["max_severity_per_event"])

    spans = sorted(((e.t_start - pad, e.t_end + pad, e) for e in events), key=lambda x: x[0])
    groups: list[tuple[float, float, list[Event]]] = []
    for s, e, ev in spans:
        if groups and s - groups[-1][1] <= gap:
            ps, pe, evs = groups[-1]
            groups[-1] = (ps, max(pe, e), evs + [ev])
        else:
            groups.append((s, e, [ev]))

    t_lo, t_hi = float(gs.t[0]), float(gs.t[-1])
    candidates: list[Candidate] = []
    for s, e, evs in groups:
        s, e = max(s, t_lo), min(e, t_hi)
        # 同じ種類のイベントが複数あっても、最も強い 1 件だけを足し合わせる
        by_type: dict[str, float] = {}
        for ev in evs:
            sev = event_severity(ev, max_sev)
            by_type[ev.event_type] = max(by_type.get(ev.event_type, 0.0), sev)
        severity = sum(by_type.values())
        severity += float(sc["cooccurrence_bonus"]) * max(len(by_type) - 1, 0)

        candidates.append(
            Candidate(
                drive_id=gs.drive_id,
                segment_id=gs.segment_id,
                t_start=s,
                t_end=e,
                severity=severity,
                event_types=sorted(by_type),
                events=sorted(evs, key=lambda x: x.t_start),
                context=_context_for(gs, s, e),
            )
        )
    return sorted(candidates, key=lambda c: -c.severity)


def events_to_frame(gs: GriddedSignals, events: list[Event], config_hash: str) -> pd.DataFrame:
    """イベント 1 件 = 1 行。判定根拠をそのまま列に残す。"""
    rows = []
    for e in events:
        row = e.as_dict()
        row.update(
            drive_id=gs.drive_id,
            segment_id=gs.segment_id,
            vehicle=gs.vehicle,
            t_rel_start=e.t_start - float(gs.t[0]),
            trigger_rule=e.rule or f"{e.feature} {e.op} {e.threshold}",
            config_hash=config_hash,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _peak_time(c: Candidate) -> float:
    """候補の中で最もスコアの高いイベントの中心時刻。

    候補の t_start は統合と前後の余白を含むため、そこを頭出しにすると
    見たい事象が画面の外に出ることがある (実測で最長 50 秒の候補があった)。
    """
    if not c.events:
        return 0.5 * (c.t_start + c.t_end)
    # 頭出しの位置を決めるだけなので、上限で頭打ちにせず素の強さで比べる。
    # 上限を掛けると、飽和した複数のイベントの区別がつかなくなる。
    best = max(c.events, key=lambda ev: event_severity(ev, float("inf")))
    return 0.5 * (best.t_start + best.t_end)


def candidates_to_frame(candidates: list[Candidate], gs: GriddedSignals, config_hash: str) -> pd.DataFrame:
    rows = []
    for c in candidates:
        row: dict[str, Any] = {
            "drive_id": c.drive_id,
            "segment_id": c.segment_id,
            "vehicle": gs.vehicle,
            "t_start": c.t_start,
            "t_end": c.t_end,
            "t_rel_start": c.t_start - float(gs.t[0]),
            "duration_s": c.t_end - c.t_start,
            "severity": c.severity,
            "n_events": len(c.events),
            "event_types": "|".join(c.event_types),
            # 候補は前後 2 秒を付けて統合してあるので、t_start は事象そのものの位置ではない。
            # 動画やプロットの頭出しには、最も強いイベントの中心を使う。
            "t_peak": _peak_time(c),
            "raw_can_loaded": gs.raw_can_loaded,
            "config_hash": config_hash,
        }
        row.update(c.context)
        rows.append(row)
    return pd.DataFrame(rows)
