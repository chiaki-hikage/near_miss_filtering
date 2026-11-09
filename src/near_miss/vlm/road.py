"""路面凹凸の確認タスク (振動イベントの当たり地点に凹凸が写っているか)。

既存のヒヤリハット判定 (schema.py / prompt.py / runner.py の clip・online) とは別のタスク。
**既存のモジュールは変更せず**、モデルの設定 (configs/vlm.yaml) と adapter・Runner・計測を
import して使うだけにしてある。推論の実行は road_runner.py、スクリプトは
scripts/vlm/build_road_inputs.py → run_road_review.py → score_road_review.py。

入力は、前輪が当たる地点 (または対照の地点) が前方 d m に見えるフレームを
数枚、路面の部分だけ切り出したもの。モデルには次のどちらも**伝えない**。

    - その地点で振動イベントが起きたか (対照の地点と同じ文面・同じ形の入力にする)
    - 振動の強さ・候補かどうか・物理シグナル・以前の目視判定

伝えると「凹凸がある前提」で探させることになり、何を見てもあると答える
モデルと区別できなくなる (docs/vlm_review_poc.md 11.2 で、検出率と誤警報率が
連動していたのと同じ問題)。採点側でだけ manifest と突き合わせる。

このモジュールは vllm を import しない。幾何・schema・prompt・echo はすべて
GPU の無い機械で動き、試験できる。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


TASK = "road"

VISIBILITY = ("good", "poor", "not_visible")
PRESENT = ("yes", "no", "uncertain")
FEATURE_TYPES = (
    "pothole",        # 穴・欠け
    "manhole",        # マンホール・鉄蓋
    "patch",          # 補修跡 (舗装の打ち替え・パッチ)
    "crack",          # ひび・亀裂
    "joint",          # 継ぎ目・伸縮装置 (橋の継手など)
    "step",           # 段差 (舗装の境界の高低差)
    "speed_bump",     # 減速帯
    "rail_crossing",  # 線路の横断
    "rough_surface",  # 一帯に荒れた路面
    "debris",         # 落下物
    "other",
    "none",
)
WHEEL_PATH = ("left", "right", "both", "between", "outside", "unknown", "none")
NEAR = ("yes", "no", "uncertain")


def build_schema() -> dict[str, Any]:
    """判定 1 件分の JSON Schema。vLLM の構造化出力にそのまま渡す。"""
    props: dict[str, Any] = {
        "visibility": {"type": "string", "enum": list(VISIBILITY)},
        "feature_present": {"type": "string", "enum": list(PRESENT)},
        "feature_type": {"type": "string", "enum": list(FEATURE_TYPES)},
        "wheel_path": {"type": "string", "enum": list(WHEEL_PATH)},
        "near_target": {"type": "string", "enum": list(NEAR)},
        "first_seen_image": {"type": "integer", "minimum": 0},
        "description": {"type": "string"},
        "evidence_detail": {"type": "string"},
        "insufficient_information": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }
    required = ["visibility", "feature_present", "feature_type", "wheel_path", "near_target",
                "first_seen_image", "description", "insufficient_information", "confidence"]
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def validate(obj: Any) -> list[str]:
    """schema 違反を並べて返す。空なら適合。

    schema.validate と同じ観点 (必須・未知の項目・型・列挙・範囲) で見る。
    schema.py はヒヤリハット判定のものなので、触らずにここで同じ検査を持つ。
    """
    schema = build_schema()
    if not isinstance(obj, dict):
        return [f"dict ではない: {type(obj).__name__}"]
    errs = [f"必須項目がない: {k}" for k in schema["required"] if k not in obj]
    errs += [f"未知の項目: {k}" for k in obj if k not in schema["properties"]]
    types = {"string": str, "boolean": bool, "array": list}
    for key, spec in schema["properties"].items():
        if key not in obj:
            continue
        v, want = obj[key], spec["type"]
        is_num = isinstance(v, (int, float)) and not isinstance(v, bool)
        if want in types and not isinstance(v, types[want]):
            errs.append(f"{key}: 型が {want} でない")
        elif want == "integer" and not (isinstance(v, int) and not isinstance(v, bool)):
            errs.append(f"{key}: 整数でない")
        elif want == "number" and not is_num:
            errs.append(f"{key}: 数値でない")
        if "enum" in spec and isinstance(v, str) and v not in spec["enum"]:
            errs.append(f"{key}: 列挙にない値 {v!r}")
        if is_num and "minimum" in spec and v < spec["minimum"]:
            errs.append(f"{key}: 下限 {spec['minimum']} 未満")
        if is_num and "maximum" in spec and v > spec["maximum"]:
            errs.append(f"{key}: 上限 {spec['maximum']} 超過")
    return errs


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """configs/vlm_road.yaml を読み、base_config (configs/vlm.yaml) に重ねる。

    モデル・復号・adapter の設定はヒヤリハット判定と共通にする (base 側)。
    タスクに固有なのは prompt・schema・入力の作り方だけ。
    """
    import yaml

    path = Path(path)
    road = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_path = Path(road["base_config"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    cfg = _merge(base, {k: v for k, v in road.items() if k != "base_config"})
    cfg["task"] = TASK
    return cfg


# ---------------------------------------------------------------------------
# 幾何 (平坦・直進・カメラは車両中心の仮定。locate_candidates.plot_frames と同じ)
# ---------------------------------------------------------------------------

def frame_time_for_distance(t_target: float, v_mps: float, ahead_m: float, behind_m: float) -> float:
    """目標地点がカメラから ahead_m 先に見える時刻。

    前輪が目標地点を通る時刻を t_target とすると、時刻 t のカメラから目標地点までの
    距離は v (t_target - t) + behind (カメラは前輪より behind_m 後ろ)。
    """
    if v_mps <= 0:
        raise ValueError("車速が 0 以下")
    return t_target - (ahead_m - behind_m) / v_mps


def distance_at(t_target: float, t_frame: float, v_mps: float, behind_m: float) -> float:
    """実際に選んだフレームでの、カメラから目標地点までの距離 [m]。"""
    return v_mps * (t_target - t_frame) + behind_m


def ground_px(d_m: float, y_lat_m: float, cam: dict[str, Any]) -> tuple[float, float]:
    """前方 d_m・横 y_lat_m (右が正) の路面の点の、元画像 (1164×874) での列・行。"""
    col = cam["center_col"] + cam["focal_px"] * y_lat_m / d_m
    row = cam["horizon_row"] + cam["K"] / d_m
    return col, row


def wheel_paths_px(cam: dict[str, Any], track_m: float, d_range: tuple[float, float],
                   n: int = 60) -> dict[str, list[tuple[float, float]]]:
    """左右の車輪の通り道を、元画像の座標の折れ線で返す。"""
    lo, hi = d_range
    ds = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    return {side: [ground_px(d, sgn * track_m / 2, cam) for d in ds]
            for side, sgn in (("left", -1.0), ("right", 1.0))}


# ---------------------------------------------------------------------------
# プロンプト
# ---------------------------------------------------------------------------

OVERLAY_TEXT = (
    "画像には次の印を描き込んであります。\n"
    "- 青の破線: 自車の左の車輪が通る線\n"
    "- 赤の破線: 自車の右の車輪が通る線\n"
    "- ○: 注目する地点の推定位置 (左右の車輪の位置にそれぞれ 1 つ)。"
    "進行方向には 1〜2 m ほどずれていることがあります\n"
)
RAW_TEXT = (
    "画像には印を描いていません。注目する地点は自車の進路上 (画像の下側中央付近) にあり、"
    "各画像での距離は下の一覧のとおりです。\n"
)


def frame_lines(req: dict[str, Any]) -> str:
    lines = []
    for i, f in enumerate(req.get("frame_info", []), 1):
        lines.append(f"- 画像 {i}: 注目する地点まで約 {f['d_m']:.0f} m")
    return "\n".join(lines)


def captions(req: dict[str, Any]) -> list[str]:
    """画像の直前に差し込む短い見出し。adapter が画像と交互に並べる。"""
    return [f"画像 {i} (注目する地点まで約 {f['d_m']:.0f} m)"
            for i, f in enumerate(req.get("frame_info", []), 1)]


def build_prompt(req: dict[str, Any], cfg: dict[str, Any]) -> str:
    tmpl = Path(cfg["prompts"]["road"]).read_text(encoding="utf-8")
    overlay = req.get("condition") == "O"
    out = tmpl.replace("{marker_block}", OVERLAY_TEXT if overlay else RAW_TEXT)
    out = out.replace("{frame_list}", frame_lines(req))
    left = [x for x in ("{marker_block}", "{frame_list}") if x in out]
    if left:
        raise ValueError(f"未置換のプレースホルダ: {left}")
    return out


def echo(req: dict[str, Any]) -> str:
    """schema に適合する固定応答。経路の確認だけに使う (推論ではない)。"""
    return json.dumps({
        "visibility": "good", "feature_present": "uncertain", "feature_type": "none",
        "wheel_path": "unknown", "near_target": "uncertain", "first_seen_image": 0,
        "description": "(echo)", "evidence_detail": "(echo)",
        "insufficient_information": False, "confidence": 0.5,
    }, ensure_ascii=False)


def make_request(target_id: str, condition: str, frames: list[str], frame_info: list[dict[str, float]],
                 config_hash: str) -> dict[str, Any]:
    """リクエスト 1 件。イベントでも対照でも**同じ項目だけ**を持つ。

    採点に使う情報 (イベントか対照か・候補か・以前の目視判定) は manifest.csv にだけ書く。
    """
    return {"request_id": f"{target_id}|{condition}", "event_id": target_id, "task": TASK, "mode": "road",
            "condition": condition, "frames": list(frames), "frame_info": list(frame_info),
            "n_frames": len(frames), "span_s": round(frame_info[0]["dt_s"] - frame_info[-1]["dt_s"], 3),
            "config_hash": config_hash}


# リクエストに入れてはいけない項目。採点用の情報が prompt に混ざる事故を防ぐ。
# tests/test_vlm_road.py が実物のリクエストに対して検査する。
FORBIDDEN_KEYS = ("kind", "is_candidate", "candidate", "near_miss_candidate_score", "A", "wheel_metric",
                  "video_road_feature", "physics_signals", "site_id", "label", "risky")
