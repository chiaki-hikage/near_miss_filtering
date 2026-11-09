"""路面凹凸の確認タスク (near_miss.vlm.road / road_runner) の試験。GPU も vllm も要らない。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from near_miss.vlm import road
from near_miss.vlm.road_runner import RoadQwenAdapter, RoadRunner, make_road_adapter

REPO = Path(__file__).resolve().parents[1]
CFG_PATH = REPO / "configs/vlm_road.yaml"


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.chdir(REPO)              # base_config と prompt はリポジトリからの相対パス
    return road.load_config(CFG_PATH)


def _req(cond: str = "O", n: int = 3) -> dict:
    info = [{"d_m": d, "dt_s": d / 10} for d in (20.0, 11.0, 5.0)[:n]]
    return road.make_request("T0001", cond, [f"f{i}.jpg" for i in range(n)], info, "abc")


# --- schema ---------------------------------------------------------------

def test_echo_fits_schema():
    assert road.validate(json.loads(road.echo({}))) == []


@pytest.mark.parametrize("patch,needle", [
    ({"feature_present": "maybe"}, "列挙にない値"),
    ({"confidence": 1.5}, "上限"),
    ({"first_seen_image": 1.0}, "整数でない"),
    ({"extra": 1}, "未知の項目"),
])
def test_schema_rejects(patch, needle):
    obj = json.loads(road.echo({})) | patch
    assert any(needle in e for e in road.validate(obj))


def test_schema_requires_fields():
    obj = json.loads(road.echo({}))
    del obj["feature_present"]
    assert any("必須項目がない" in e for e in road.validate(obj))


# --- 幾何 -----------------------------------------------------------------

def test_frame_time_and_distance_are_inverse():
    t = road.frame_time_for_distance(100.0, 15.0, 12.0, 0.6)
    assert math.isclose(road.distance_at(100.0, t, 15.0, 0.6), 12.0)
    assert t < 100.0                     # 目標を通る前のフレーム


def test_ground_px_matches_locate_candidates_formula():
    cam = dict(focal_px=910.0, center_col=582, horizon_row=428.3, K=910.0)
    col, row = road.ground_px(10.0, 0.785, cam)
    # locate_candidates.plot_frames: IMG_CENTER_COL + f * y_lat / d, yh + K / d
    assert math.isclose(col, 582 + 910 * 0.785 / 10)
    assert math.isclose(row, 428.3 + 910 / 10)
    paths = road.wheel_paths_px(cam, 1.57, (3.5, 40.0), n=5)
    assert all(l[0] < r[0] for l, r in zip(paths["left"], paths["right"]))   # 左輪が画像の左


# --- 入力の盲検化 ---------------------------------------------------------

def test_request_has_no_scoring_fields():
    r = _req()
    assert not set(road.FORBIDDEN_KEYS) & set(r)
    assert r["task"] == road.TASK and r["request_id"] == "T0001|O"


def test_prompt_does_not_reveal_vibration(cfg):
    for cond in ("R", "O"):
        p = road.build_prompt(_req(cond), cfg)
        for word in ("振動", "イベント", "候補", "ヒヤリ", "対照"):
            assert word not in p, (cond, word)
        assert "画像 3: 注目する地点まで約 5 m" in p


def test_conditions_differ_only_in_marker_text(cfg):
    r, o = road.build_prompt(_req("R"), cfg), road.build_prompt(_req("O"), cfg)
    assert r.replace(road.RAW_TEXT, "") == o.replace(road.OVERLAY_TEXT, "")


# --- adapter と runner ----------------------------------------------------

class _FakeProcessor:
    def __init__(self):
        self.msgs = None

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        self.msgs = msgs
        return "chat"


def test_adapter_places_one_slot_per_image_with_captions():
    a = RoadQwenAdapter(model_id="x", input_kind="images")
    a._processor = _FakeProcessor()
    a.chat_text("指示", ["a.jpg", "b.jpg", "c.jpg"], ["見出し1", "見出し2", "見出し3"])
    content = a._processor.msgs[0]["content"]
    assert [c["type"] for c in content] == ["text", "image"] * 3 + ["text"]
    assert content[0]["text"] == "見出し1" and content[-1]["text"] == "指示"
    with pytest.raises(ValueError):
        a.chat_text("指示", ["a.jpg"], ["1", "2"])


def test_make_road_adapter_requires_images(cfg):
    spec = cfg["models"]["qwen3_vl_8b"]
    a = make_road_adapter(spec["adapter"], spec["model_id"], cfg, spec)
    assert isinstance(a, RoadQwenAdapter) and a.media_kind_for(["a", "b"]) == "image"
    with pytest.raises(ValueError):
        make_road_adapter(spec["adapter"], spec["model_id"], dict(cfg, media_kind="video"), spec)


def test_config_reuses_models_from_base(cfg):
    import yaml
    base = yaml.safe_load((REPO / "configs/vlm.yaml").read_text(encoding="utf-8"))
    assert cfg["models"] == base["models"] and cfg["decode"] == base["decode"]
    assert cfg["prompt_version"] == "road_v1" and cfg["task"] == road.TASK


def test_echo_runner_writes_valid_rows(cfg, tmp_path):
    out = tmp_path / "res.jsonl"
    runner = RoadRunner("echo", cfg, make_road_adapter("echo", "echo", cfg))
    assert runner.run([_req("R"), _req("O")], 0, out) == 2
    rows = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert [r["condition"] for r in rows] == ["R", "O"]
    assert all(r["schema_errors"] == [] and r["task"] == road.TASK for r in rows)


def test_runner_refuses_hazard_requests(cfg, tmp_path):
    runner = RoadRunner("echo", cfg, make_road_adapter("echo", "echo", cfg))
    with pytest.raises(ValueError):
        runner.run([{"request_id": "P01|C", "event_id": "P01", "mode": "clip"}], 0, tmp_path / "x.jsonl")
