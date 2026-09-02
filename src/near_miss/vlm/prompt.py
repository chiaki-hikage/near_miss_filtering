"""プロンプトの組み立て。

固定の指示文は configs/prompts/ に置き、条件ごとに変わる部分だけをここで作る。
**モデル間で共通**。差し替えてよいのは重みと chat template だけで、
prompt を変えると比較が成立しない。

条件 A (CAN のみ) に「映像を見て」と書いたり、条件 B (映像のみ) に
空の信号表を出したりすると、その条件だけ不当に不利になる。
入力に合わせて文面を変える。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODALITY = {
    (True, True): "前方カメラの映像と車両信号を見て、",
    (True, False): "前方カメラの映像を見て、",
    (False, True): "車両信号を見て、",
    (False, False): "与えられた情報から、",
}


def _can_block(req: dict[str, Any], cfg: dict[str, Any]) -> str:
    table = req.get("can_text") or ""
    if not table.strip():
        return ""
    if req["mode"] == "online":
        head = (f"\n## 車両信号\n\n評価時刻より {req.get('guard_s', 0)} 秒手前までの値です"
                "（信号処理の遅れのため）。\n時刻は評価時刻からの相対秒。"
                "速度は km/h、加速度は m/s^2。\n")
    else:
        head = ("\n## 車両信号\n\n区間の先頭からの相対秒。"
                "速度は km/h、加速度は m/s^2。\n")
    return f"{head}\n```\n{table}\n```\n"


def _summary_block(req: dict[str, Any]) -> str:
    """区間の極値。**モード A 専用。** モード B には決して付けない。"""
    if req["mode"] != "clip":
        return ""
    ex = req.get("can_extremes") or ""
    return f"\n**区間の極値**: {ex}\n" if ex.strip() else ""


def _hint_block(req: dict[str, Any]) -> str:
    """既存フィルタの検出結果。条件 D だけで使うアブレーション。"""
    hint = req.get("hint") or ""
    if not hint.strip():
        return ""
    return ("\n## 参考: 既存の検出器の出力\n\n"
            f"この区間は `{hint}` として検出されています。"
            "ただし検出器は誤検出もするので、鵜呑みにしないでください。\n")


def build(req: dict[str, Any], cfg: dict[str, Any]) -> str:
    kind = "clip" if req["mode"] == "clip" else "online"
    tmpl = Path(cfg["prompts"][kind]).read_text(encoding="utf-8")
    has_video = bool(req.get("frames"))
    has_can = bool((req.get("can_text") or "").strip())

    out = tmpl.replace("{modality_line}", MODALITY[(has_video, has_can)])
    out = out.replace("{can_block}", _can_block(req, cfg))
    out = out.replace("{summary_block}", _summary_block(req))
    out = out.replace("{hint_block}", _hint_block(req))
    if "{" in out and "}" in out:
        left = [x for x in ("{modality_line}", "{can_block}", "{summary_block}",
                            "{hint_block}", "{guard_s}", "{can_table}") if x in out]
        if left:
            raise ValueError(f"未置換のプレースホルダ: {left}")
    return out
