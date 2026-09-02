"""モデル差の吸収。

**共通化するもの** (adapter が触ってはいけない)
    入力生成 (フレーム抽出・CAN 要約・窓)、prompt、JSON schema、復号パラメータ、評価

**adapter が吸収するもの**
    chat template、メディアの渡し方、視覚トークン化

Qwen2.5-VL / Qwen3-VL / Cosmos-Reason1 は同一系統なので adapter は 1 つで足りる。
Cosmos-Reason1-7B は Qwen2.5-VL-7B ベースなので、**前処理も vLLM 経路も完全に
共通**になり、「運転特化 post-training の効果」を他の変数を固定したまま測れる。

Alpamayo 系はカメラ較正・自車軌跡など入力要件が異なり同一入力での比較が
成立しないため、別 adapter・別比較枠にする (この表には並べない)。

フレームは**画像の列ではなく動画として渡す**。Qwen 系は時間方向のマージと
絶対時刻の埋め込みを持っており、画像を並べるだけだとそれが効かない。
トークン数もおよそ倍になる。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QwenVLAdapter:
    """Qwen2.5-VL / Qwen3-VL / Cosmos-Reason1 で共通。"""

    model_id: str
    name: str = "qwen_vl"
    input_kind: str = "video"        # "video" か "images"
    _processor: Any = None

    def processor(self):
        """chat template を当てるためだけに使う。重みは読まない。"""
        if self._processor is None:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        return self._processor

    def chat_text(self, prompt: str, n_media: int) -> str:
        """chat template を当てた素のプロンプト文字列を返す。"""
        content: list[dict[str, Any]] = []
        if n_media:
            content.append({"type": self.input_kind})
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        return self.processor().apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    def multi_modal(self, frames: list[str]) -> dict[str, Any] | None:
        """フレーム列を vLLM の multi_modal_data に直す。"""
        if not frames:
            return None
        import numpy as np
        from PIL import Image

        arr = np.stack([np.asarray(Image.open(Path(p)).convert("RGB")) for p in frames])
        return {"video": arr} if self.input_kind == "video" else {"image": list(arr)}

    def limits(self) -> dict[str, int]:
        return {self.input_kind: 1 if self.input_kind == "video" else 32}


class EchoAdapter:
    """vLLM を使わない試験用。prompt の組み立てと入出力の経路だけ確かめる。

    GPU の無い機械 (Mac) で全経路を通せるようにするためにある。
    **推論ではない。**
    """

    name = "echo"
    input_kind = "images"

    def __init__(self, model_id: str = "echo") -> None:
        self.model_id = model_id

    def chat_text(self, prompt: str, n_media: int) -> str:
        return prompt

    def multi_modal(self, frames: list[str]) -> dict[str, Any] | None:
        return None

    def limits(self) -> dict[str, int]:
        return {}


def make_adapter(name: str, model_id: str, cfg: dict[str, Any]):
    if name == "qwen_vl":
        return QwenVLAdapter(model_id=model_id,
                             input_kind=str(cfg.get("media_kind", "video")))
    if name == "echo":
        return EchoAdapter(model_id)
    raise ValueError(f"未知の adapter: {name}")
