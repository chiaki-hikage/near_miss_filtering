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


# Qwen3-VL が要求する動画メタデータの項目名。
# vLLM の版で綴りが変わりうるので、直すならここ 1 箇所で済むようにしておく。
META_FPS = "fps"
META_TOTAL = "total_num_frames"
META_INDICES = "frames_indices"
META_BACKEND = "video_backend"
META_DO_SAMPLE = "do_sample_frames"


@dataclass
class QwenVLAdapter:
    """Qwen2.5-VL / Qwen3-VL / Cosmos-Reason1 で共通。

    Qwen3-VL は動画のメタデータ (fps 等) を要求する。無いと vLLM が
    `video metadata is required but not found in mm input` で落ちる。
    Qwen2.5-VL は素の配列で動くので、**モデル単位で切り替える**
    (configs/vlm.yaml の models.<key>.video_metadata)。
    """

    model_id: str
    name: str = "qwen_vl"
    input_kind: str = "video"        # "video" か "images"
    video_metadata: bool = False     # Qwen3-VL 系で true
    video_fps: float = 2.0           # 渡すフレームの実際の間隔 (input.video_fps)
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
        """フレーム列を vLLM の multi_modal_data に直す。

        video_metadata=True のときは (配列, メタデータ) の組で渡す。
        メタデータは**こちらが渡すフレーム列そのもの**を記述する。

            fps               = input.video_fps (2.0)。実際の間隔と一致する
            total_num_frames  = 渡す枚数
            frames_indices    = 0..n-1
            do_sample_frames  = False  こちらで間引き済みなので再サンプルさせない

        fps と indices から Qwen3-VL が各フレームの時刻を出す。2 fps・連番なので
        0.0 / 0.5 / 1.0 ... となり、build_vlm_inputs が実際に抜いた間隔と一致する。
        履歴が足りず枚数が減る時刻 (P08 の 7 点) でも間隔は 0.5 秒のままなので、
        この作り方で正しい。
        """
        if not frames:
            return None
        import numpy as np
        from PIL import Image

        arr = np.stack([np.asarray(Image.open(Path(p)).convert("RGB")) for p in frames])
        if self.input_kind != "video":
            return {"image": list(arr)}
        if not self.video_metadata:
            return {"video": arr}
        n = len(frames)
        meta = {
            META_FPS: float(self.video_fps),
            META_TOTAL: n,
            META_INDICES: list(range(n)),
            META_BACKEND: "opencv",
            META_DO_SAMPLE: False,
        }
        return {"video": (arr, meta)}

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


def make_adapter(name: str, model_id: str, cfg: dict[str, Any],
                 spec: dict[str, Any] | None = None):
    """spec は configs/vlm.yaml の models.<key>。モデル固有の差だけを見る。"""
    spec = spec or {}
    if name == "qwen_vl":
        return QwenVLAdapter(
            model_id=model_id,
            input_kind=str(cfg.get("media_kind", "video")),
            video_metadata=bool(spec.get("video_metadata", False)),
            video_fps=float(cfg["input"]["video_fps"]),
        )
    if name == "echo":
        return EchoAdapter(model_id)
    raise ValueError(f"未知の adapter: {name}")
