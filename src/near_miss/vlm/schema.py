"""VLM の出力 schema。

モデル間で共通。差し替えてよいのは重みと chat template だけで、
schema と prompt は固定する。そうしないと比較が成立しない。

pydantic に依存しない。ここは GPU の要らない経路 (Mac 側) でも動かすため、
新しい依存を持ち込まない。返すのは素の JSON Schema の dict で、
vLLM の guided_json にそのまま渡せる。
"""

from __future__ import annotations

from typing import Any

# 判定の状態。オンライン判定 (モード B) の時系列の主指標になる。
STATES = ("normal", "caution", "hazard", "unknown")

# 危険の類型。既存の event 語彙に、**映像でしか分からない類型**を足してある。
#   cut_in を明示的に置くのが要点。人手ラベルの risky 8 件のうち 5 件が
#   隣車線からの割り込み・車線変更との相互作用で、CAN 側では short_thw と
#   してしか見えない。VLM が付加価値を出せるとすればここなので、
#   それを測れる形にしておく。
#
#   crossing (対向車・交差車両の横断) も同じ理由で分けてある。CAN 側では
#   hard_brake としか見えないが、原因は完全に映像側にある。P08 がこれ。
HAZARD_TYPES = (
    "cut_in",            # 隣車線からの割り込み
    "crossing",          # 対向車・交差車両が自車進路を横切る
    "lead_brake",        # 先行車の減速
    "hard_brake",
    "hard_steer",
    "weaving",
    "short_thw",
    "low_ttc",
    "lane_change",
    "pedestrian",
    "signal",            # 信号・標識に起因
    "obstacle",
    "other",
    "none",
)

# 判定の根拠がどちらにあるか。**これが PoC の中心指標**。
# 映像を足した意味があったかを直接測る。
EVIDENCE = ("video", "can", "both", "neither")


def _agent_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "what": {"type": "string"},
            "where": {"type": "string"},
            "motion": {"type": "string"},
            "first_seen_s": {"type": "number"},
        },
        "required": ["what", "where", "motion"],
        "additionalProperties": False,
    }


def build_schema(mode: str) -> dict[str, Any]:
    """判定 1 件分の JSON Schema を返す。

    mode: "clip"   一括判定 (モード A)。区間全体を 1 回で見る
          "online" オンライン判定 (モード B)。評価時刻ごとに 1 件

    online にだけ時刻と変化の記述を足す。change_from_previous は
    「直近フレーム間で何が新しいか」であって、**前回の判定は渡さない**
    (stateless を既定とするため)。因果性は保たれる。
    """
    if mode not in ("clip", "online"):
        raise ValueError(f"未知のモード: {mode}")

    props: dict[str, Any] = {
        "scene": {"type": "string"},
        "ego_behavior": {"type": "string"},
        "other_agents": {"type": "array", "items": _agent_schema()},
        "state": {"type": "string", "enum": list(STATES)},
        "hazard_type": {"type": "string", "enum": list(HAZARD_TYPES)},
        "risk_level": {"type": "integer", "minimum": 0, "maximum": 3},
        "risky": {"type": "boolean"},
        "difference_from_normal": {"type": "string"},
        "evidence": {"type": "string", "enum": list(EVIDENCE)},
        "evidence_detail": {"type": "string"},
        "insufficient_information": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }
    required = [
        "scene", "ego_behavior", "other_agents", "state", "hazard_type",
        "risk_level", "risky", "difference_from_normal", "evidence",
        "insufficient_information", "confidence",
    ]

    if mode == "online":
        props["t_eval_s"] = {"type": "number"}
        props["change_from_previous"] = {"type": "string"}
        props["expected_next"] = {"type": "string"}
        required += ["t_eval_s", "change_from_previous", "expected_next"]

    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def validate(obj: Any, mode: str) -> list[str]:
    """schema 違反を並べて返す。空なら適合。

    guided_json が効いていれば違反は出ないはずだが、**効いていることを
    確かめずに信用しない**。適合率は Phase 1 のゲート条件 1 になっている。
    """
    schema = build_schema(mode)
    errs: list[str] = []
    if not isinstance(obj, dict):
        return [f"dict ではない: {type(obj).__name__}"]

    for key in schema["required"]:
        if key not in obj:
            errs.append(f"必須項目がない: {key}")
    for key in obj:
        if key not in schema["properties"]:
            errs.append(f"未知の項目: {key}")

    for key, spec in schema["properties"].items():
        if key not in obj:
            continue
        v = obj[key]
        want = spec["type"]
        if want == "string" and not isinstance(v, str):
            errs.append(f"{key}: 文字列でない")
        elif want == "boolean" and not isinstance(v, bool):
            errs.append(f"{key}: 真偽値でない")
        elif want == "integer" and not (isinstance(v, int) and not isinstance(v, bool)):
            errs.append(f"{key}: 整数でない")
        elif want == "number" and not (isinstance(v, (int, float)) and not isinstance(v, bool)):
            errs.append(f"{key}: 数値でない")
        elif want == "array" and not isinstance(v, list):
            errs.append(f"{key}: 配列でない")

        if "enum" in spec and isinstance(v, str) and v not in spec["enum"]:
            errs.append(f"{key}: 列挙にない値 {v!r}")
        if "minimum" in spec and isinstance(v, (int, float)) and not isinstance(v, bool):
            if v < spec["minimum"]:
                errs.append(f"{key}: 下限 {spec['minimum']} 未満")
        if "maximum" in spec and isinstance(v, (int, float)) and not isinstance(v, bool):
            if v > spec["maximum"]:
                errs.append(f"{key}: 上限 {spec['maximum']} 超過")
    return errs
