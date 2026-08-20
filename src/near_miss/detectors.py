"""閾値イベントの検出 (L3)。

特徴量の列名と閾値は設定から渡される。この層は「どの列を見るか」を知らない。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .signals import GriddedSignals

VALID_OPS = ("lt", "gt", "abs_gt", "rising")
AGREEMENT_OP = "agreement"
COOCCURRENCE_OP = "cooccurrence"
SEQUENCE_OP = "sequence"
DEFERRED_OPS = (COOCCURRENCE_OP, SEQUENCE_OP)


@dataclass
class Event:
    """1 件の検出。なぜ拾われたかを再現できる情報を必ず持たせる。"""

    event_type: str
    label: str
    feature: str
    op: str
    threshold: float
    t_start: float
    t_end: float
    duration_s: float
    peak_value: float
    exceedance: float
    stage: int
    weight: float
    gate: str = ""
    rule: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_list(value: Any) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _condition(x: np.ndarray, op: str, threshold: float) -> np.ndarray:
    """NaN は常に False。欠測を「条件を満たさない」ではなく「判定不能」として扱い、
    発火させないことで、欠測由来の誤検出を防ぐ。"""
    with np.errstate(invalid="ignore"):
        if op == "lt":
            m = x < threshold
        elif op == "gt":
            m = x > threshold
        elif op == "abs_gt":
            m = np.abs(x) > threshold
        else:
            raise ValueError(f"未知の op: {op}")
    return np.where(np.isfinite(x), m, False)


def dilate(mask: np.ndarray, samples: int) -> np.ndarray:
    """マスクを前後 samples 分だけ広げる。

    別々のセンサが同じ挙動を捉える時刻には必ずずれがある (取得周期も遅れも違う)。
    厳密な同時刻を要求すると、実在する事象を取り逃がす。
    広げるのは各条件の側だけで、連言そのものは緩めない。
    1 センサだけの単発スパイクは、他のセンサが裏付けないので通らない。
    """
    if samples <= 0 or not mask.any():
        return mask
    kernel = np.ones(2 * samples + 1)
    return np.convolve(mask.astype(float), kernel, mode="same") > 0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True が連続する区間を [start, end] のインデックス対で返す。"""
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1))
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size - 1)
    return list(zip(starts, ends))


def _merge_runs(runs: list[tuple[int, int]], t: np.ndarray, merge_gap_s: float) -> list[tuple[int, int]]:
    """近接した区間をひとつにまとめる。閾値付近での細切れを防ぐ。"""
    if not runs:
        return []
    merged = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        if t[s] - t[pe] <= merge_gap_s:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def _exceedance(peak: float, op: str, threshold: float) -> float:
    if op == "lt":
        return float(threshold - peak)
    if op == "gt":
        return float(peak - threshold)
    if op == "abs_gt":
        return float(abs(peak) - threshold)
    return 1.0


def build_gate(df: Any, spec: dict[str, Any]) -> tuple[np.ndarray | None, str]:
    """イベント定義の gate を評価して、判定を許可するサンプルのマスクを返す。

    gate は「その条件が成り立っているときだけ判定する」ための足切り。
    低速域で舵角レートが大きく出るような、挙動の意味が変わる領域を除くのに使う。
    必要な列が無いときは gate を無視せず、判定そのものを行わない (全 False)。
    """
    conditions = spec.get("gate") or []
    if not conditions:
        return None, ""
    mask = np.ones(len(df), dtype=bool)
    labels = []
    for c in conditions:
        feature, op, thr = c["feature"], c["op"], float(c["threshold"])
        labels.append(f"{feature} {op} {thr}")
        if feature not in df.columns:
            return np.zeros(len(df), dtype=bool), " & ".join(labels)
        mask &= _condition(df[feature].to_numpy(), op, thr)
    return mask, " & ".join(labels)


def detect_threshold_events(
    t: np.ndarray,
    x: np.ndarray,
    event_type: str,
    spec: dict[str, Any],
    gate: np.ndarray | None = None,
    gate_label: str = "",
) -> list[Event]:
    """閾値と継続時間で区間を切り出す汎用検出器。"""
    op = spec["op"]
    threshold = float(spec["threshold"])

    if op == "rising":
        return _detect_rising(t, x, event_type, spec, gate, gate_label)
    if op not in VALID_OPS:
        raise ValueError(f"未知の op: {op}")

    cond = _condition(x, op, threshold)
    if gate is not None:
        cond &= gate
    runs = _merge_runs(_runs(cond), t, float(spec.get("merge_gap_s", 0.0)))
    events: list[Event] = []
    for s, e in runs:
        duration = float(t[e] - t[s])
        if duration < float(spec.get("min_duration_s", 0.0)):
            continue
        seg = x[s : e + 1]
        if op == "lt":
            peak = float(np.nanmin(seg))
        elif op == "gt":
            peak = float(np.nanmax(seg))
        else:
            peak = float(seg[np.nanargmax(np.abs(seg))])
        events.append(
            Event(
                event_type=event_type,
                label=spec.get("label", event_type),
                feature=spec["feature"],
                op=op,
                threshold=threshold,
                t_start=float(t[s]),
                t_end=float(t[e]),
                duration_s=duration,
                peak_value=peak,
                exceedance=_exceedance(peak, op, threshold),
                stage=int(spec.get("stage", 1)),
                weight=float(spec.get("weight", 1.0)),
                gate=gate_label,
                rule=f"{spec['feature']} {op} {threshold}",
            )
        )
    return events


def _detect_rising(
    t: np.ndarray,
    x: np.ndarray,
    event_type: str,
    spec: dict[str, Any],
    gate: np.ndarray | None = None,
    gate_label: str = "",
) -> list[Event]:
    """フラグの 0 → 1 立ち上がりを 1 件のイベントとする。"""
    threshold = float(spec["threshold"])
    high = np.where(np.isfinite(x), x > threshold, False)
    if gate is not None:
        high &= gate
    if high.size < 2:
        return []
    rise = np.flatnonzero((~high[:-1]) & high[1:]) + 1
    events = []
    for i in rise:
        end = i
        while end + 1 < high.size and high[end + 1]:
            end += 1
        events.append(
            Event(
                event_type=event_type,
                label=spec.get("label", event_type),
                feature=spec["feature"],
                op="rising",
                threshold=threshold,
                t_start=float(t[i]),
                t_end=float(t[end]),
                duration_s=float(t[end] - t[i]),
                peak_value=1.0,
                exceedance=1.0,
                stage=int(spec.get("stage", 1)),
                weight=float(spec.get("weight", 1.0)),
                gate=gate_label,
            )
        )
    return events


def detect_agreement_events(
    t: np.ndarray,
    df: Any,
    event_type: str,
    spec: dict[str, Any],
    gate: np.ndarray | None = None,
    gate_label: str = "",
) -> list[Event]:
    """複数の信号がそろって条件を満たす区間を切り出す。

    ひとつの信号の閾値超過は、ノイズなのか実際の挙動なのか区別できない。
    別系統の信号が同じ時間帯に裏付けているかどうかで判断する。

    各条件のマスクを tolerance_s だけ前後に広げてから連言を取る。
    代表値 (spec["feature"]) は条件のどれかを指し、スコア計算に使う。
    """
    conditions = spec.get("conditions") or []
    if not conditions:
        raise ValueError(f"{event_type}: op=agreement には conditions が要ります")

    dt = float(t[1] - t[0]) if t.size > 1 else 1.0
    tol = int(round(float(spec.get("tolerance_s", 0.0)) / dt))

    primary = spec.get("feature")
    primary_threshold, primary_op = None, None
    combined = np.ones(len(t), dtype=bool)
    labels = []
    for c in conditions:
        feature, op, thr = c["feature"], c["op"], float(c["threshold"])
        labels.append(f"{feature} {op} {thr}")
        if feature == primary:
            primary_threshold, primary_op = thr, op
        if feature not in df.columns:
            return []
        combined &= dilate(_condition(df[feature].to_numpy(), op, thr), tol)
    if primary is None or primary_threshold is None:
        raise ValueError(f"{event_type}: feature は conditions のいずれかを指す必要があります")
    if gate is not None:
        combined &= gate

    rule = " & ".join(labels) + f" (許容ずれ {spec.get('tolerance_s', 0.0)}s)"
    x = df[primary].to_numpy()
    events: list[Event] = []
    for s_i, e_i in _merge_runs(_runs(combined), t, float(spec.get("merge_gap_s", 0.0))):
        duration = float(t[e_i] - t[s_i])
        if duration < float(spec.get("min_duration_s", 0.0)):
            continue
        seg = x[s_i : e_i + 1]
        if not np.isfinite(seg).any():
            continue
        peak = float(seg[np.nanargmax(np.abs(seg))]) if primary_op == "abs_gt" else (
            float(np.nanmin(seg)) if primary_op == "lt" else float(np.nanmax(seg))
        )
        events.append(
            Event(
                event_type=event_type,
                label=spec.get("label", event_type),
                feature=primary,
                op=AGREEMENT_OP,
                threshold=primary_threshold,
                t_start=float(t[s_i]),
                t_end=float(t[e_i]),
                duration_s=duration,
                peak_value=peak,
                exceedance=_exceedance(peak, primary_op, primary_threshold),
                stage=int(spec.get("stage", 1)),
                weight=float(spec.get("weight", 1.0)),
                gate=gate_label,
                rule=rule,
            )
        )
    return events


def detect_cooccurrence_events(
    event_type: str,
    spec: dict[str, Any],
    existing: list[Event],
) -> list[Event]:
    """ある事象に危険側の指標が重なっているときだけ立てるイベント。

    それ自体は正常な操作でも、危険な状況と重なると意味が変わるものを扱う。
    車線変更がその例で、単独では日常的な操作でしかないが、車間が詰まっている
    ところで行われれば危険側に寄る。

    区間は基準イベント (base) のものをそのまま使う。確認するのはその操作だからで、
    重なった指標の側へ広げると、何を見ればよいのか分からなくなる。
    """
    base_type = spec["base"]
    wanted = list(spec.get("require_any") or [])
    window = float(spec.get("window_s", 0.0))
    min_count = int(spec.get("min_count", 1))

    events: list[Event] = []
    for b in (e for e in existing if e.event_type == base_type):
        lo, hi = b.t_start - window, b.t_end + window
        hits = sorted(
            {e.event_type for e in existing
             if e.event_type in wanted and not (e.t_end < lo or e.t_start > hi)}
        )
        if len(hits) < min_count:
            continue
        events.append(
            Event(
                event_type=event_type,
                label=spec.get("label", event_type),
                feature=base_type,
                op=COOCCURRENCE_OP,
                threshold=float(min_count),
                t_start=b.t_start,
                t_end=b.t_end,
                duration_s=b.duration_s,
                peak_value=float(len(hits)),
                exceedance=float(len(hits) - min_count),
                stage=int(spec.get("stage", 1)),
                weight=float(spec.get("weight", 1.0)),
                rule=f"{base_type} と {'/'.join(hits)} が ±{window}s 以内に共起",
            )
        )
    return events


def detect_sequence_events(
    event_type: str,
    spec: dict[str, Any],
    existing: list[Event],
) -> list[Event]:
    """先に起きる事象と、その後に続く事象の並びを 1 件のイベントにする。

    同時に起きたかどうかではなく、順序に意味がある組み合わせを扱う。
    回避してから制動する、制動してから回避する、では運転の中身が違う。

    区間は先の事象の開始から後の事象の終了まで。確認するのは一連の流れなので、
    片方だけを見ても何が起きたのか分からない。
    """
    first_types = [spec["first"]] if isinstance(spec["first"], str) else list(spec["first"])
    then_types = [spec["then"]] if isinstance(spec["then"], str) else list(spec["then"])
    lo = float(spec.get("min_gap_s", 0.0))
    hi = float(spec.get("max_gap_s", 0.0))

    firsts = [e for e in existing if e.event_type in first_types]
    thens = [e for e in existing if e.event_type in then_types]

    events: list[Event] = []
    used: set[int] = set()
    for a in sorted(firsts, key=lambda e: e.t_start):
        for k, b in enumerate(sorted(thens, key=lambda e: e.t_start)):
            if k in used:
                continue
            gap = b.t_start - a.t_end
            if gap < lo or gap > hi:
                continue
            used.add(k)
            events.append(
                Event(
                    event_type=event_type,
                    label=spec.get("label", event_type),
                    feature=a.event_type,
                    op=SEQUENCE_OP,
                    threshold=hi,
                    t_start=a.t_start,
                    t_end=b.t_end,
                    duration_s=float(b.t_end - a.t_start),
                    peak_value=float(gap),
                    exceedance=float(max(hi - gap, 0.0)),
                    stage=int(spec.get("stage", 1)),
                    weight=float(spec.get("weight", 1.0)),
                    rule=f"{a.event_type} の {gap:.2f}s 後に {b.event_type}",
                )
            )
            break
    return events


def detect_all(gs: GriddedSignals, cfg: dict[str, Any], max_stage: int = 2) -> list[Event]:
    """設定に並んだイベント定義をすべて適用する。

    必要な特徴量列が無い定義は黙って飛ばさず、呼び出し側が分かるよう
    skipped として meta に残す。
    """
    events: list[Event] = []
    skipped: list[str] = []
    deferred: list[tuple[str, dict[str, Any]]] = []
    for name, spec in cfg["events"].items():
        if not spec.get("enabled", True):
            continue
        if int(spec.get("stage", 1)) > max_stage:
            skipped.append(f"{name}:stage")
            continue
        if spec["op"] in DEFERRED_OPS:
            # 他のイベントが出そろってから判定する
            deferred.append((name, spec))
            continue
        feature = spec["feature"]
        needed = [feature] + [c["feature"] for c in (spec.get("conditions") or [])]
        missing = [f for f in needed if f not in gs.df.columns]
        if missing:
            skipped.append(f"{name}:missing_feature:{','.join(sorted(set(missing)))}")
            continue
        gate, gate_label = build_gate(gs.df, spec)
        if spec["op"] == AGREEMENT_OP:
            events.extend(detect_agreement_events(gs.t, gs.df, name, spec, gate, gate_label))
        else:
            events.extend(
                detect_threshold_events(
                    gs.t, gs.df[feature].to_numpy(), name, spec, gate, gate_label
                )
            )
    for name, spec in deferred:
        if spec["op"] == COOCCURRENCE_OP:
            referenced = [spec["base"]] + list(spec.get("require_any") or [])
        else:
            referenced = _as_list(spec["first"]) + _as_list(spec["then"])
        missing = [f for f in referenced if f not in cfg["events"]]
        if missing:
            skipped.append(f"{name}:unknown_event:{','.join(missing)}")
            continue
        if spec["op"] == COOCCURRENCE_OP:
            events.extend(detect_cooccurrence_events(name, spec, events))
        else:
            events.extend(detect_sequence_events(name, spec, events))

    gs.meta.setdefault("skipped_events", []).extend(skipped)
    return sorted(events, key=lambda e: e.t_start)
