"""canonical event (API 連携用の共通形式) と provenance の試験。

前半は部品 (ID・丸め・変換・スキーマ) の単体試験。後半は実データの出力
(out/near_miss_canonical/events.jsonl・provenance.jsonl) を元の CSV と突き合わせる。
出力や元の CSV が無い環境では後半を飛ばす。
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from near_miss.canonical_event import (
    Record, ccs_paths, clean, comma2k19_paths, id_key, kmh_to_mps, null_leaves, quantize_sec, split_pipe,
    stable_event_id, to_bool, utc_iso, validate_events, video_frame,
)

REPO = Path(__file__).resolve().parents[1]
SCHEMA = REPO / "schemas/near_miss_event.schema.json"
OUT = REPO / "out/near_miss_canonical/events.jsonl"
PROV = REPO / "out/near_miss_canonical/provenance.jsonl"
REF_IDS = REPO / "tests/data/near_miss_canonical_event_ids.json"

BASE = dict(dataset="comma2k19", drive_id="b0c9d2329ad1606b|2018-10-05--22-20-31", segment=16,
            start_sec=46.2, event_type="road_vibration", detector="road_vibration_locate")


# ---------------------------------------------------------------------------
# event_id
# ---------------------------------------------------------------------------

def test_same_input_same_id():
    assert stable_event_id(**BASE) == stable_event_id(**BASE)
    assert stable_event_id(**BASE).startswith("nm_") and len(stable_event_id(**BASE)) == 3 + 16


def test_event_type_changes_id():
    assert stable_event_id(**BASE) != stable_event_id(**{**BASE, "event_type": "slip"})


def test_detector_and_other_keys_change_id():
    i = stable_event_id(**BASE)
    for k, v in (("detector", "l3_rules"), ("dataset", "comma_car_segments"), ("segment", 17),
                 ("drive_id", "b0c9d2329ad1606b|2018-10-05--22-20-32"), ("start_sec", 46.21)):
        assert stable_event_id(**{**BASE, k: v}) != i, k


def test_start_sec_rounding_is_stable():
    # 10 ms 単位、10 進で四捨五入
    assert quantize_sec(46.205) == "46.21"          # 2 進の丸めでは 46.20 になる値
    assert quantize_sec(46.2049) == "46.20"
    assert quantize_sec(54.04) == quantize_sec(54.040000000001) == quantize_sec(54.0399999999) == "54.04"
    assert quantize_sec(0.0) == quantize_sec(-0.0) == quantize_sec(-0.004) == "0.00"
    assert quantize_sec(62.8) == "62.80"
    # 丸めの範囲内の揺れでは ID が変わらない
    assert stable_event_id(**BASE) == stable_event_id(**{**BASE, "start_sec": 46.2 + 1e-9})
    assert stable_event_id(**BASE) == stable_event_id(**{**BASE, "start_sec": 46.2049})


def test_id_key_rejects_bad_parts():
    with pytest.raises(ValueError):
        id_key("comma2k19", "", 1, 0.0, "slip", "x")
    with pytest.raises(ValueError):
        quantize_sec(float("nan"))


# ---------------------------------------------------------------------------
# 変換
# ---------------------------------------------------------------------------

def test_unit_conversion():
    assert kmh_to_mps(36.0) == 10.0
    assert kmh_to_mps(53.0) == round(53.0 / 3.6, 3) == 14.722
    assert kmh_to_mps(float("nan")) is None


def test_missing_values_are_not_filled():
    assert clean(float("nan")) is None and clean("") is None
    assert split_pipe(float("nan")) is None
    assert to_bool(float("nan")) is None and to_bool("maybe") is None
    assert utc_iso("not a time") is None
    r = Record()
    r.copy("speed_mps", float("nan"), "x.csv:v")
    assert r.doc["speed_mps"] is None
    assert "speed_mps" not in r.prov            # 欠測は copied として数えない
    assert null_leaves(r.doc) == ["speed_mps"]


def test_formats():
    assert split_pipe("a|b||c") == ["a", "b", "c"]
    assert to_bool(True) is True and to_bool("False") is False
    assert utc_iso("2018-10-06 05:37:18") == "2018-10-06T05:37:18Z"


def test_video_frame():
    assert video_frame(0.74, 20, 1200) == 15          # 既存の video_frame 列と同じ式
    assert video_frame(59.97, 20, 1200) == 1199
    assert video_frame(59.99, 20, 1200) is None       # 1200 はセグメントの外
    assert video_frame(-0.2, 20, 1200) is None
    assert video_frame(None, 20, 1200) is None


def test_source_paths():
    p = comma2k19_paths("Chunk_2", BASE["drive_id"], 16)
    assert p["video_uri"] == "Chunk_2/b0c9d2329ad1606b|2018-10-05--22-20-31/16/video.hevc"
    assert p["can_uri"].endswith("/16/processed_log/CAN")
    assert p["imu_uri"].endswith("/16/processed_log/IMU") and p["gnss_uri"].endswith("/16/global_pose")
    assert all(v is None for v in comma2k19_paths(None, BASE["drive_id"], 16).values())
    c = ccs_paths("615671030be02e4e|000004cd--18bee28ac5", 18)
    assert c["can_uri"] == "segments/615671030be02e4e/000004cd--18bee28ac5/18/rlog.zst"
    assert c["video_uri"] is None and c["imu_uri"] is None and c["gnss_uri"] is None


def test_reference_ids_unchanged():
    """修正の前後で ID の作り方が変わっていないこと (基準値は 2026-10-28 の出力から)。"""
    ref = json.loads(REF_IDS.read_text(encoding="utf-8"))["events"]
    assert len(ref) >= 10
    for r in ref:
        assert stable_event_id(r["dataset"], r["drive_id"], r["segment"], r["start_sec"], r["event_type"],
                               r["detector"]) == r["event_id"]


MINIMAL = {
    "schema_version": "1.0", "event_id": "nm_0123456789abcdef", "event_type": "crosswind", "detector": "crosswind_rules",
    "dataset": "other_dataset", "drive_id": "d1", "time": {"base": "segment_relative_s", "start_sec": 1.0, "end_sec": 2.0},
    "detection": {"method": "rule_threshold"}, "source": {}, "attributes": {},
}


def test_schema_accepts_minimal_event():
    """optional の項目がすべて無くても通る。未知の event_type・segment の無いデータセットも通る。"""
    schema = json.loads(SCHEMA.read_text())
    assert validate_events([MINIMAL], schema) == []
    assert validate_events([{**MINIMAL, "segment": "clip_07"}], schema) == []


def test_schema_rejects_bad_events():
    schema = json.loads(SCHEMA.read_text())
    for bad in ({k: v for k, v in MINIMAL.items() if k != "source"},               # 必須が無い
                {**MINIMAL, "origin": {"file": "x.csv"}},                          # provenance は本体に入れない
                {**MINIMAL, "location": {"latitude": None, "longitude": 1.0}},     # null は入れない
                {**MINIMAL, "score": {"value": 1.0}},                              # 名前の無いスコア
                {**MINIMAL, "event_id": "E0818"}):
        assert validate_events([bad], schema), bad


# ---------------------------------------------------------------------------
# 実データの出力
# ---------------------------------------------------------------------------

needs_output = pytest.mark.skipif(not (OUT.is_file() and PROV.is_file()), reason="out/near_miss_canonical が無い")
OPTIONAL = ["vehicle", "chunk", "segment", "location", "speed_mps", "score", "grade", "candidate_flag", "review"]


def jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def events():
    return jsonl(OUT)


@pytest.fixture(scope="module")
def provs():
    return {p["event_id"]: p for p in jsonl(PROV)}


def by(events, detector, dataset=None):
    return [e for e in events if e["detector"] == detector and (dataset is None or e["dataset"] == dataset)]


def walk(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)
    else:
        yield x


@needs_output
def test_schema_validation_all(events):
    schema = json.loads(SCHEMA.read_text())
    assert validate_events(events, schema) == []


@needs_output
def test_schema_validation_without_optional_fields(events):
    """各イベントから optional の項目を全部外しても通る。"""
    schema = json.loads(SCHEMA.read_text())
    stripped = [{k: v for k, v in e.items() if k not in OPTIONAL} for e in events]
    assert validate_events(stripped, schema) == []


@needs_output
def test_no_nulls_and_no_provenance_in_events(events):
    for e in events:
        assert "origin" not in e and not any(k.startswith("_") for k in e)
        assert all(v is not None for v in walk(e))
        assert "root" not in e["source"] and "review_media" not in e["source"]


@needs_output
def test_api_essentials_present(events):
    """origin を外しても、イベントの特定と元データへの参照に要るものは全件にある。"""
    for e in events:
        assert e["event_id"] and e["event_type"] and e["dataset"] and e["drive_id"]
        assert isinstance(e["time"]["start_sec"], float | int) and isinstance(e["time"]["end_sec"], float | int)
        assert e["source"].get("can_uri"), e["event_id"]
        if e["dataset"] == "comma2k19":
            assert {"video_uri", "imu_uri", "gnss_uri"} <= set(e["source"])


@needs_output
def test_ids_unique_and_recomputable(events):
    assert len({e["event_id"] for e in events}) == len(events)
    for e in events:
        assert e["event_id"] == stable_event_id(e["dataset"], e["drive_id"], e["segment"],
                                                e["time"]["start_sec"], e["event_type"], e["detector"])


@needs_output
def test_reference_ids_present_in_output(events):
    ids = {e["event_id"] for e in events}
    for r in json.loads(REF_IDS.read_text(encoding="utf-8"))["events"]:
        assert r["event_id"] in ids


@needs_output
def test_provenance_one_to_one(events, provs):
    assert len(provs) == len(events) == len(jsonl(PROV))
    assert set(provs) == {e["event_id"] for e in events}
    for p in provs.values():
        assert {"event_id", "origin", "derived_fields", "missing_fields"} <= set(p)
        assert p["origin"]["file"] and isinstance(p["origin"]["row"], int)


@needs_output
def test_provenance_traces_back_to_csv_row(events, provs):
    """provenance の file と row で元 CSV の行を引くと、イベントと同じ走行・区間・時刻になる。"""
    cache = {}
    for e in events:
        o = provs[e["event_id"]]["origin"]
        if o["file"] not in cache:
            cache[o["file"]] = pd.read_csv(REPO / o["file"])
        r = cache[o["file"]].iloc[o["row"]]
        drive = r["drive"] if "drive" in r.index else r["drive_id"]
        assert drive == e["drive_id"] and int(r["segment"]) == e["segment"]
        start = r["t_start_s"] if "t_start_s" in r.index else r["t_in_segment_s"]
        assert start == e["time"]["start_sec"]
        if "event_id" in r.index:
            assert r["event_id"] == o["source_ids"]["event_id"]


@needs_output
def test_missing_fields_are_really_absent(events, provs):
    for e in events:
        for path in provs[e["event_id"]]["missing_fields"]:
            node = e
            for k in path.split(".")[:-1]:
                node = node.get(k, {})
            assert path.split(".")[-1] not in node, (e["event_id"], path)


@needs_output
def test_l3_values_copied(events):
    src = pd.read_csv(REPO / "out/l1_l3/c2k19/events.csv")
    src = src[src.event_type.isin(["hard_brake", "hard_steer"])]
    got = by(events, "l3_rules", "comma2k19")
    assert len(got) == len(src)
    idx = {(e["drive_id"], e["segment"], e["time"]["start_sec"]): e for e in got}
    for _, r in src.iterrows():
        e = idx[(r.drive_id, int(r.segment), r.t_in_segment_s)]
        assert e["time"]["boot_time_start_s"] == r.t_start and e["time"]["boot_time_end_s"] == r.t_end
        assert e["time"]["end_sec"] == round(r.t_in_segment_s + (r.t_end - r.t_start), 2)
        assert e["source"]["video_frame_start"] == int(r.video_frame)
        assert e["detection"]["rule"] == r.trigger_rule and e["detection"]["config_hash"] == r.config_hash
        key = "ax_min_mps2" if r.event_type == "hard_brake" else "lateral_accel_mps2"
        assert e["attributes"][key] == r.peak_value
        # 元に無いものは埋めていない (項目ごと無い)
        assert "location" not in e and "speed_mps" not in e and "peak_sec" not in e["time"]


@needs_output
def test_l3_ccs_has_no_video(events):
    for e in by(events, "l3_rules", "comma_car_segments"):
        s = e["source"]
        assert set(s) == {"can_uri"} and "chunk" not in e
        assert s["can_uri"].startswith("comma_car_segments/segments/") and s["can_uri"].endswith("/rlog.zst")


@needs_output
def test_slip_direct_values_copied(events):
    src = pd.concat([pd.read_csv(p) for p in sorted(REPO.glob("out/sideslip_TOYOTA_*_v3/direct_slip.csv"))
                     if p.stat().st_size > 1])
    got = by(events, "slip_direct_evidence")
    assert len(got) == len(src)
    idx = {(e["drive_id"], e["segment"], e["time"]["start_sec"]): e for e in got}
    for _, r in src.iterrows():
        e = idx[(r.drive_id, int(r.segment), r.t_in_segment_s)]
        assert e["grade"] == r.grade
        assert e["attributes"]["wheel_speed_diff_peak_mps"] == r.ws_spread_peak_mps
        if pd.isna(r.v_at_peak_mps):
            assert "speed_mps" not in e
        else:
            assert e["speed_mps"] == r.v_at_peak_mps
        if pd.isna(r.beta_model_peak_deg):
            assert "sideslip_angle_model_deg" not in e["attributes"].get("at_peak", {})


@needs_output
def test_road_vibration_values_copied(events, provs):
    loc = pd.read_csv(REPO / "out/vibration/locate/candidates.csv")
    nm = pd.read_csv(REPO / "out/vibration/near_miss/near_miss_vibration_events.csv").set_index("event_id")
    cols = ["strong_vertical", "strong_roll", "simultaneous_vertical_roll", "impulsive_response",
            "road_related_evidence"]
    got = {provs[e["event_id"]]["origin"]["source_ids"]["event_id"]: e for e in by(events, "road_vibration_locate")}
    assert len(got) == len(loc)
    for _, r in loc.iterrows():
        e = got[r.event_id]
        assert (e["drive_id"], e["chunk"], e["segment"]) == (r.drive, r.chunk, int(r.segment))
        assert (e["time"]["start_sec"], e["time"]["end_sec"], e["time"]["peak_sec"]) == (r.t_start_s, r.t_end_s, r.t_peak_s)
        assert e["location"]["latitude"] == r.lat and e["location"]["longitude"] == r.lon   # 6 桁の locate 側
        assert e["speed_mps"] == round(r.speed_kmh / 3.6, 3)
        n = nm.loc[r.event_id]
        assert e["candidate_flag"] == bool(n[cols].all())
        assert e["score"]["value"] == n.near_miss_candidate_score
        assert e["attributes"]["vertical_accel_peak_mps2"] == n.heave_peak_abs
        assert e["review"]["video_road_feature"] == n.video_road_feature      # 元の表記のまま
        assert "boot_time_start_s" not in e["time"]                            # 元 CSV に無い
        assert "site" not in e["attributes"] and "near_miss_rank" not in e["attributes"]   # 実行依存の値は provenance へ
    assert sum(e["candidate_flag"] for e in got.values()) == 21


@needs_output
def test_review_labels_keep_original_text(events):
    lab = pd.concat([pd.read_csv(REPO / p) for p in ("out/chunk1/labels.csv", "out/chunk2_final/labels.csv")])
    verdicts = set(lab.verdict)
    for e in by(events, "l3_rules", "comma2k19"):
        if "verdict" in e.get("review", {}):
            assert e["review"]["verdict"] in verdicts


@needs_output
def test_source_uris_resolve_locally(events):
    """URI (データのルートからの相対パス) を手元の配置に読み替えると実在する。"""
    roots = {"comma2k19": REPO / "raw_data", "comma_car_segments": REPO / "raw_data/comma_car_segments"}
    if not roots["comma2k19"].is_dir():
        pytest.skip("raw_data が無い")
    for e in events:
        for k in ("video_uri", "can_uri", "imu_uri", "gnss_uri"):
            if k in e["source"]:
                root, _, rest = e["source"][k].partition("/")
                assert root == e["dataset"]
                assert (roots[root] / rest).exists(), e["source"][k]


@pytest.mark.slow
@needs_output
def test_rerun_gives_same_events(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts/export_near_miss_canonical.py"), "--out", str(tmp_path)],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8") == OUT.read_text(encoding="utf-8")
    assert (tmp_path / "provenance.jsonl").read_text(encoding="utf-8") == PROV.read_text(encoding="utf-8")
