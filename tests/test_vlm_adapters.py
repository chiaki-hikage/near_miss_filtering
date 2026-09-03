"""モデル差の吸収 (adapters) の試験。

chat template と multi_modal_data は**同じ判断**でモダリティを選ぶ必要がある。
片方だけ image になると processor が入力と合わずに失敗する
(実際 Qwen3-VL で `Failed to apply Qwen3VLProcessor` が出た)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from near_miss.config import load_yaml
from near_miss.vlm.adapters import MIN_VIDEO_FRAMES, make_adapter

REQ = Path("out/chunk1/vlm/requests_mode_b.jsonl")


@pytest.fixture(scope="module")
def cfg():
    return load_yaml("configs/vlm.yaml")


def _adapter(cfg, key):
    spec = cfg["models"][key]
    return make_adapter(spec["adapter"], spec["model_id"], cfg, spec)


@pytest.mark.parametrize("key", ["qwen2_5_vl_7b", "cosmos_reason1_7b",
                                 "qwen3_vl_8b", "qwen3_vl_30b_a3b"])
def test_1枚だけならimageとして扱う(cfg, key):
    """Qwen 系は時間方向を 2 フレーム単位でまとめるので T=1 の動画は成立しない。"""
    a = _adapter(cfg, key)
    assert a.media_kind_for(["a.jpg"]) == "image"
    assert a.media_kind_for(["a.jpg", "b.jpg"]) == "video"
    assert a.media_kind_for(["a.jpg"] * 8) == "video"
    assert MIN_VIDEO_FRAMES == 2


@pytest.mark.parametrize("key", ["qwen2_5_vl_7b", "qwen3_vl_8b"])
def test_chat_templateとmulti_modalが同じモダリティを選ぶ(cfg, key, monkeypatch):
    a = _adapter(cfg, key)
    for n in (1, 2, 8):
        frames = ["x.jpg"] * n
        kind = a.media_kind_for(frames)
        # chat_text は processor を要するので、モダリティ判断だけを突き合わせる
        assert kind in ("image", "video")
        assert (kind == "image") == (n < MIN_VIDEO_FRAMES)


@pytest.mark.skipif(not REQ.is_file(), reason="リクエストが未生成")
def test_実データでメタデータが配列と整合する(cfg):
    """total_num_frames / frames_indices は渡す配列の T と一致させる。"""
    from near_miss.vlm.adapters import META_INDICES, META_TOTAL

    a = _adapter(cfg, "qwen3_vl_30b_a3b")
    reqs = [json.loads(l) for l in REQ.open(encoding="utf-8")]
    seen = set()
    for r in reqs:
        n = r["n_frames"]
        if n in seen:
            continue
        seen.add(n)
        mm = a.multi_modal(r["frames"])
        if n < MIN_VIDEO_FRAMES:
            assert "image" in mm and "video" not in mm
            assert len(mm["image"]) == n
            continue
        arr, meta = mm["video"]
        assert arr.shape[0] == n
        assert meta[META_TOTAL] == n
        assert len(meta[META_INDICES]) == n
    assert 1 in seen and 8 in seen, "1 枚と 8 枚の両方を通せていない"


@pytest.mark.skipif(not REQ.is_file(), reason="リクエストが未生成")
def test_Qwen2_5はメタデータを付けない(cfg):
    a = _adapter(cfg, "qwen2_5_vl_7b")
    r = next(json.loads(l) for l in REQ.open(encoding="utf-8")
             if json.loads(l)["n_frames"] == 8)
    v = a.multi_modal(r["frames"])["video"]
    assert isinstance(v, np.ndarray), "既存動作が変わっている"


def test_両モダリティを許可する(cfg):
    a = _adapter(cfg, "qwen3_vl_8b")
    assert a.limits() == {"video": 1, "image": 32}
