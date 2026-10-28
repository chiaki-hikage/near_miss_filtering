#!/usr/bin/env python3
"""既存のヒヤリハット候補 CSV を canonical event (API 連携用の共通形式) に変換する。

  uv run --extra api python scripts/export_near_miss_canonical.py

既存の CSV はそのまま残し、読むだけにする。出力 (configs/near_miss_canonical.yaml の out_dir):

  events.jsonl            canonical event (API 利用者向け)。1 行 1 イベント。値が無い項目は省く
  sample_events.json      種別ごとに数件
  provenance.jsonl        変換の記録 (開発・ETL 向け)。event_id で events.jsonl と 1 対 1。
                          変換元の CSV と行・元 CSV の ID・導出した項目・欠けている項目・内部の値
  conversion_report.json  全体の集計 (入力・出どころ・欠損の件数・突き合わせの検査・検証結果)

対応する検出器 (detector) と event_type:

  l3_rules               hard_braking / hard_steering   out/l1_l3/{c2k19,ccs}/events.csv
  slip_direct_evidence   slip                           out/sideslip_TOYOTA_<車種>_v3/direct_slip.csv
  slip_sideslip_beta     slip                           out/sideslip_*/candidates.csv
  road_vibration_locate  road_vibration                 out/vibration/locate/candidates.csv ほか

**値を推測で埋めない。** 元 CSV に無い値は null にする (attributes では項目ごと省く)。
式で出した値 (終了時刻・フレーム番号・単位換算・候補フラグ) は変換規則を
docs/near_miss_canonical_schema.md に書き、conversion_report.json に derived として残す。

--stamp-git-commit は detection.algorithm_version に現在の git commit を入れる。
**既存の出力をこのコードで作り直した直後にだけ使うこと。** 既定では null
(既存の結果がどの commit で出たかは記録が無く、確定できないため)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yaml

from near_miss.canonical_event import (
    ORIGIN_CAN, ORIGIN_IMU, SCHEMA_VERSION, TIME_BASE, Record, ccs_paths, chunk_index, clean, comma2k19_paths,
    git_commit, kmh_to_mps, load_schema, null_leaves, split_pipe, stable_event_id, to_bool, utc_iso,
    validate_events, video_frame,
)

REPO = Path(__file__).resolve().parents[1]
INTERNAL = "_internal"   # provenance へ移す値の一時置き場
ORIGIN = "_origin"


def read(path: str) -> pd.DataFrame:
    return pd.read_csv(REPO / path)


def sha10(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:10]


def opt(rec: Record, path: str, value, src: str, conv=None) -> None:
    """attributes 用。値が無ければ項目を作らない (null を入れない)。"""
    v = clean(value) if not isinstance(value, (list, dict)) else value
    if v is None:
        return
    rec.copy(path, conv(v) if conv else v, src)


class Ctx:
    """変換全体で共有するもの。"""

    def __init__(self, cfg: dict, stamp_commit: bool):
        self.cfg = cfg
        self.raw_root = REPO / cfg["raw_root"]
        self.chunks = chunk_index(self.raw_root) if self.raw_root.is_dir() else {}
        self.fps = cfg["video"]["fps"]
        self.n_frames = cfg["video"]["frames_per_segment"]
        self.roots = cfg["source_roots"]
        self.uri_base = ""   # --uri-base (例 s3://<バケット>/near_miss/) を付けると単独で解決できる URI になる
        self.local_roots = {k: REPO / v for k, v in cfg["uri_local_roots"].items()}
        self.commit = git_commit(REPO) if stamp_commit else None
        self.dongle_vehicle, self.platform_vehicle = {}, {}
        for f in sorted((REPO / cfg["vehicles_dir"]).glob("*.yaml")):
            v = yaml.safe_load(f.read_text()) or {}
            for d in v.get("dongle_ids") or []:
                self.dongle_vehicle[d] = (v["name"], str(f.relative_to(REPO)))
            for p in v.get("platforms") or []:
                self.platform_vehicle[p] = (v["name"], str(f.relative_to(REPO)))
        self.checks: dict[str, dict] = defaultdict(dict)   # 検出器ごとの突き合わせの検査
        self.inputs: set[str] = set()

    def use(self, *paths: str) -> None:
        self.inputs.update(paths)


# ---------------------------------------------------------------------------
# 共通部分
# ---------------------------------------------------------------------------

def base_record(ctx: Ctx, event_type: str, detector: str, et_src: str) -> Record:
    rec = Record()
    rec.const("schema_version", SCHEMA_VERSION, "schema")
    rec.null("event_id")   # 最後に入れる (キーの並びを固定するため先に枠を作る)
    rec.const("event_type", event_type, et_src)
    rec.const("detector", detector, "config detectors.*.detector")
    for k in ("dataset", "vehicle", "drive_id", "chunk", "segment"):
        rec.null(k)
    return rec


def empty_blocks(rec: Record) -> None:
    """canonical の枠を必ず全部作る (無い値は null)。"""
    for k in ("base", "segment_origin", "start_sec", "end_sec", "peak_sec", "boot_time_start_s",
              "boot_time_end_s", "utc"):
        rec.null(f"time.{k}")
    for k in ("latitude", "longitude", "heading_deg", "gnss_acc_m"):
        rec.null(f"location.{k}")
    rec.null("speed_mps")
    rec.null("score.name")
    rec.null("score.value")
    rec.null("grade")
    rec.null("candidate_flag")
    for k in ("method", "rule", "config_hash", "algorithm_version", "created_at"):
        rec.null(f"detection.{k}")
    for k in ("video_uri", "can_uri", "imu_uri", "gnss_uri", "video_frame_start", "video_frame_end"):
        rec.null(f"source.{k}")
    rec.doc["review"] = {}
    rec.doc["attributes"] = {}
    rec.doc[INTERNAL] = {}   # provenance へ移す値 (API のレスポンスには出さない)
    rec.doc[ORIGIN] = {}


def put_source(rec: Record, ctx: Ctx, dataset: str, chunk: str | None, drive: str, seg: int) -> None:
    paths = comma2k19_paths(chunk, drive, seg) if dataset == "comma2k19" else ccs_paths(drive, seg)
    root = ctx.roots[dataset]
    rule = (f"配置規則 {ctx.uri_base}{root}/<chunk>/<drive_id>/<segment>/..." if dataset == "comma2k19"
            else f"配置規則 {ctx.uri_base}{root}/segments/<dongle>/<route>/<segment>/rlog.zst")
    for k, v in paths.items():
        if v is None:
            rec.null(f"source.{k}")
        else:
            rec.const(f"source.{k}", f"{ctx.uri_base}{root}/{v}", rule)


def put_time_boot(rec: Record, row, start_col: str, peak_col: str | None, src: str) -> None:
    """boot time の t_start / t_end (/ t_peak) と、セグメント内の開始秒を持つ CSV の時刻。

    end_sec  = round(t_in_segment_s + (t_end - t_start), 2)
    peak_sec = round(t_in_segment_s + (t_peak - t_start), 2)   (どちらも開始のセグメントの先頭から)
    """
    rec.const("time.base", TIME_BASE, "schema")
    rec.const("time.segment_origin", ORIGIN_CAN, "near_miss.pipeline._annotate_segment (CAN 主系列の t_span)")
    start = rec.copy("time.start_sec", row[start_col], f"{src}:{start_col}")
    rec.derive("time.end_sec", round(float(start + (row.t_end - row.t_start)), 2),
               "end_sec = round(t_in_segment_s + (t_end - t_start), 2)")
    if peak_col is not None and clean(row[peak_col]) is not None:
        rec.derive("time.peak_sec", round(float(start + (row[peak_col] - row.t_start)), 2),
                   f"peak_sec = round(t_in_segment_s + ({peak_col} - t_start), 2)")
    else:
        rec.null("time.peak_sec")
    rec.copy("time.boot_time_start_s", row.t_start, f"{src}:t_start")
    rec.copy("time.boot_time_end_s", row.t_end, f"{src}:t_end")
    rec.null("time.utc")


def put_frames(rec: Record, ctx: Ctx, dataset: str, row, src: str, frame_col: str | None) -> None:
    """comma2k19 のみ。開始は元 CSV の video_frame があればそれを、無ければ同じ式で出す。"""
    if dataset != "comma2k19" or rec.get("source.video_uri") is None:
        rec.null("source.video_frame_start")
        rec.null("source.video_frame_end")
        return
    v = clean(row[frame_col]) if frame_col and frame_col in row.index else None
    if v is not None and int(v) >= 0:
        rec.copy("source.video_frame_start", int(v), f"{src}:{frame_col}")
    else:
        rec.derive("source.video_frame_start", video_frame(rec.get("time.start_sec"), ctx.fps, ctx.n_frames),
                   "video_frame_start = round(start_sec * fps)")
    rec.derive("source.video_frame_end", video_frame(rec.get("time.end_sec"), ctx.fps, ctx.n_frames),
               "video_frame_end = round(end_sec * fps)。セグメントの外なら null")


def run_meta_index(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        f = REPO / p
        if f.is_file():
            d = json.loads(f.read_text())
            out.append(dict(file=p, run_at=d.get("run_at"), config_hash=d.get("config_hash"),
                            data_root=str(d.get("data_root", "")), platform=d.get("platform")))
    return out


def iso_time(s: str | None) -> str | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 既存の L3 ルール (急ブレーキ・急操舵)
# ---------------------------------------------------------------------------

# 候補 (区間) の列 → attributes.candidate_window のキー。元 CSV に列が無ければ省く
L3_WINDOW = {
    "hard_brake": {
        "ax_mps2_min": "ax_min_mps2", "jerk_mps3_min": "jerk_min_mps3", "jerk_win_min_min": "jerk_win_min_mps3",
        "jerk_win_mean_min": "jerk_win_mean_min_mps3", "brake_pressed_max": "brake_pressed_max",
        "gvc_mps2_min": "gvc_min_mps2", "gas_pedal_pct_max": "gas_pedal_max_pct", "decel_flag_max": "decel_flag_max",
        "brake_mc_mpa_max": "brake_mc_max_mpa", "brake_position_max": "brake_position_max",
        "abs_active_flag_max": "abs_active_max", "vsc_active_flag_max": "vsc_active_max",
        "precollision_active_max": "precollision_active_max", "precollision_force_n_min": "precollision_force_min_n",
        "acc_braking_max": "acc_braking_max",
    },
    "hard_steer": {
        "steer_rate_dps_absmax": "steering_rate_dps", "ay_kin_mps2_absmax": "lateral_accel_mps2",
        "ay_can_mps2_absmax": "lateral_accel_can_mps2", "yaw_rate_dps_absmax": "yaw_rate_dps",
        "lat_jerk_mps3_absmax": "lateral_jerk_mps3", "lat_jerk_win_absmax_max": "lateral_jerk_win_mps3",
        "steer_reversals_max": "steer_reversals_max", "steer_torque_driver_absmax": "steer_torque_driver",
    },
}
L3_WINDOW_COMMON = {"v_mps_mean": "speed_mean_mps", "v_mps_min": "speed_min_mps", "severity": "severity",
                    "n_events": "n_events"}
# イベント行の peak_value の意味 (feature が決まっているので名前を付けられる)
L3_PEAK = {"hard_brake": "ax_min_mps2", "hard_steer": "lateral_accel_mps2"}


def containing(df: pd.DataFrame, row, platform_col: str | None) -> pd.DataFrame:
    m = (df.drive_id == row.drive_id) & (df.t_start <= row.t_start) & (df.t_end >= row.t_end)
    if platform_col and platform_col in df.columns:
        m &= df[platform_col] == row[platform_col]
    return df[m]


def convert_l3(name: str, dc: dict, ctx: Ctx) -> list[Record]:
    ev_path = dc["events"]
    ev = read(ev_path)
    ev = ev[ev.event_type.isin(dc["event_types"])]
    det = pd.concat([read(p).assign(_src=p) for p in dc["candidates_detail"]], ignore_index=True)
    flags = read(dc["candidates_flags"])
    labels = (pd.concat([read(p).assign(_src=p) for p in dc["labels"]], ignore_index=True)
              if dc["labels"] else pd.DataFrame(columns=["drive_id", "t_start", "t_end"]))
    metas = run_meta_index(dc["run_meta"])
    ctx.use(ev_path, dc["candidates_flags"], *dc["candidates_detail"], *dc["labels"], *dc["run_meta"])
    has_dataset = "dataset" in ev.columns
    chk = Counter()

    recs = []
    for _, row in ev.iterrows():
        etype = dc["event_types"][row.event_type]
        rec = base_record(ctx, etype, dc["detector"], f"config detectors.{name}.event_types.{row.event_type}")
        empty_blocks(rec)
        if has_dataset:
            dataset = rec.copy("dataset", row.dataset, f"{ev_path}:dataset")
        else:
            dataset = rec.const("dataset", dc["dataset"], f"config detectors.{name}.dataset")
        rec.copy("vehicle", row.vehicle, f"{ev_path}:vehicle")
        drive = rec.copy("drive_id", row.drive_id, f"{ev_path}:drive_id")
        chunk = ctx.chunks.get(drive) if dataset == "comma2k19" else None
        if chunk:
            rec.const("chunk", chunk, f"{ctx.cfg['raw_root']}/Chunk_*/<drive_id>/ の配置")
        seg = int(rec.copy("segment", int(row.segment), f"{ev_path}:segment"))
        put_time_boot(rec, row, "t_in_segment_s", None, ev_path)

        cand = containing(det, row, "platform")
        flag = containing(flags, row, "platform")
        chk["candidate_detail_matches_" + str(min(len(cand), 2))] += 1
        chk["candidate_flags_matches_" + str(min(len(flag), 2))] += 1
        c = cand.iloc[0] if len(cand) == 1 else None
        f = flag.iloc[0] if len(flag) == 1 else None
        if c is not None:
            rec.const("score.name", "candidate_severity", "このイベントを含む候補の severity")
            rec.copy("score.value", c.severity, f"{c._src}:severity (イベントを含む候補)")
        if f is not None and "final" in f.index:
            rec.copy("candidate_flag", to_bool(f.final), f"{dc['candidates_flags']}:final (イベントを含む候補)")

        rec.const("detection.method", "rule_threshold", "config")
        rec.copy("detection.rule", row.trigger_rule, f"{ev_path}:trigger_rule")
        rec.copy("detection.config_hash", row.config_hash, f"{ev_path}:config_hash")
        for k in ("feature", "op", "threshold", "gate", "stage", "weight"):
            opt(rec, f"detection.criteria.{k}", row[k], f"{ev_path}:{k}")
        rec.const(f"{INTERNAL}.detection_reference", dc["reference"], "config")
        if ctx.commit:
            rec.const("detection.algorithm_version", ctx.commit, "--stamp-git-commit")
        meta = [m for m in metas if m["config_hash"] == row.config_hash
                and ((chunk and m["data_root"].endswith(chunk)) or (m["platform"] and m["platform"] == row.get("platform")))]
        if len(meta) == 1 and iso_time(meta[0]["run_at"]):
            rec.const("detection.created_at", iso_time(meta[0]["run_at"]), f"{meta[0]['file']}:run_at")
        chk["created_at_found_" + str(len(meta) == 1)] += 1

        put_source(rec, ctx, dataset, chunk, drive, seg)
        put_frames(rec, ctx, dataset, row, ev_path, "video_frame")

        lab = labels[(labels.drive_id == drive) & (labels.t_end >= row.t_start) & (labels.t_start <= row.t_end)]
        chk["labels_matches_" + str(min(len(lab), 2))] += 1
        if len(lab) == 1:
            lr = lab.iloc[0]
            rec.copy("review.risky", to_bool(lr.risky), f"{lr._src}:risky")
            rec.copy("review.verdict", lr.verdict, f"{lr._src}:verdict")
            opt(rec, "review.note", lr.note, f"{lr._src}:note")
            rec.const(f"{INTERNAL}.review_label_source", lr._src, "区間の重なりで対応 (io_spec.md 4.4)")
        elif len(lab) > 1:
            rec.copy("review.labels", [{k: v for k, v in dict(risky=to_bool(x.risky), verdict=clean(x.verdict),
                                                                note=clean(x.note)).items() if v is not None}
                                        for _, x in lab.iterrows()], "labels.csv の複数行 (区間の重なり)")
            rec.const(f"{INTERNAL}.review_label_source", sorted(set(lab._src)), "区間の重なりで対応")

        A = "attributes"
        rec.const(f"{A}.source_event_type", row.event_type, f"{ev_path}:event_type")
        opt(rec, f"{A}.label", row.label, f"{ev_path}:label")
        if "platform" in row.index and row.platform != dataset:
            opt(rec, f"{A}.platform", row.platform, f"{ev_path}:platform")
        if row.event_type in L3_PEAK:
            opt(rec, f"{A}.{L3_PEAK[row.event_type]}", row.peak_value, f"{ev_path}:peak_value")
        else:
            # 意味に合った名前を決めていない種別は、元の列名のまま判定量 (feature) と組で入れる
            opt(rec, f"{A}.peak_value", row.peak_value, f"{ev_path}:peak_value")
            opt(rec, f"{A}.peak_feature", row.feature, f"{ev_path}:feature")
        opt(rec, f"{A}.threshold", row.threshold, f"{ev_path}:threshold")
        opt(rec, f"{A}.exceedance", row.exceedance, f"{ev_path}:exceedance")
        opt(rec, f"{A}.duration_s", row.duration_s, f"{ev_path}:duration_s")
        opt(rec, f"{INTERNAL}.l1_pass", to_bool(row.L1), f"{ev_path}:L1")
        opt(rec, f"{INTERNAL}.l1_kinematics_pass", to_bool(row.L1_kin), f"{ev_path}:L1_kin")
        if c is not None:
            W = f"{A}.candidate_window"
            for col, key in {**L3_WINDOW_COMMON, **L3_WINDOW.get(row.event_type, {})}.items():
                if col in c.index:
                    opt(rec, f"{W}.{key}", c[col], f"{c._src}:{col}")
            opt(rec, f"{W}.event_types", split_pipe(c.event_types), f"{c._src}:event_types ('|' を配列に)")
            opt(rec, f"{INTERNAL}.candidate_boot_time_start_s", c.t_start, f"{c._src}:t_start")
            opt(rec, f"{INTERNAL}.candidate_boot_time_end_s", c.t_end, f"{c._src}:t_end")

        rec.doc[ORIGIN] = dict(file=ev_path, row=int(row.name), joined_files=sorted({*(cand._src if len(cand) else []),
                                                                    dc["candidates_flags"], *dc["labels"]}),
                                 source_ids=({"block_segment_id": clean(row.segment_id)}))
        recs.append(rec)
    ctx.checks[name].update(chk)
    return recs


# ---------------------------------------------------------------------------
# スリップ
# ---------------------------------------------------------------------------

def slip_dirs(dc: dict, ctx: Ctx):
    for d in dc["dirs"]:
        f = REPO / d / dc["file"]
        meta_f = REPO / d / "counts.json"
        if not f.is_file() or f.stat().st_size <= 1:
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        meta = json.loads(meta_f.read_text()) if meta_f.is_file() else {}
        ctx.use(f"{d}/{dc['file']}", f"{d}/counts.json")
        yield f"{d}/{dc['file']}", f"{d}/counts.json", df, meta


def slip_common(rec: Record, ctx: Ctx, row, src: str, meta_src: str, meta: dict, method: str, ref: str) -> tuple:
    dataset = rec.copy("dataset", row.dataset, f"{src}:dataset")
    if meta.get("vehicle"):
        rec.const("vehicle", meta["vehicle"], f"{meta_src}:vehicle")
    drive = rec.copy("drive_id", row.drive_id, f"{src}:drive_id")
    chunk = ctx.chunks.get(drive) if dataset == "comma2k19" else None
    if chunk:
        rec.const("chunk", chunk, f"{ctx.cfg['raw_root']}/Chunk_*/<drive_id>/ の配置")
    seg = int(rec.copy("segment", int(row.segment), f"{src}:segment"))
    rec.copy("speed_mps", row.v_at_peak_mps, f"{src}:v_at_peak_mps")
    rec.const("detection.method", method, "config")
    rec.copy("detection.config_hash", row.config_hash, f"{src}:config_hash")
    rec.const(f"{INTERNAL}.detection_reference", ref, "config")
    if ctx.commit:
        rec.const("detection.algorithm_version", ctx.commit, "--stamp-git-commit")
    if meta.get("config_hash") == row.config_hash and iso_time(meta.get("run_at")):
        rec.const("detection.created_at", iso_time(meta["run_at"]), f"{meta_src}:run_at")
    put_source(rec, ctx, dataset, chunk, drive, seg)
    return dataset, drive, seg


def convert_slip_direct(name: str, dc: dict, ctx: Ctx) -> list[Record]:
    recs = []
    for src, meta_src, df, meta in slip_dirs(dc, ctx):
        for _, row in df.iterrows():
            rec = base_record(ctx, dc["event_type"], dc["detector"], f"config detectors.{name}.event_type")
            empty_blocks(rec)
            put_time_boot(rec, row, "t_in_segment_s", None, src)
            dataset, drive, seg = slip_common(rec, ctx, row, src, meta_src, meta, "direct_evidence", dc["reference"])
            rec.copy("grade", row.grade, f"{src}:grade")
            rec.copy("detection.rule", row.evidence, f"{src}:evidence (立った証拠)")
            put_frames(rec, ctx, dataset, row, src, "video_frame")

            A = "attributes"
            if meta.get("label"):
                rec.const(f"{A}.platform", meta["label"], f"{meta_src}:label")
            opt(rec, f"{A}.duration_s", row.duration_s, f"{src}:duration_s")
            opt(rec, f"{A}.evidence", split_pipe(row.evidence), f"{src}:evidence ('|' を配列に)")
            opt(rec, f"{A}.corroboration", row.corroboration, f"{src}:corroboration")
            opt(rec, f"{A}.mechanisms", split_pipe(row.mechanisms), f"{src}:mechanisms ('|' を配列に)")
            for col in ("n_direct", "n_corroborated", "n_mechanisms"):
                opt(rec, f"{A}.{col}", row[col], f"{src}:{col}", int)
            for col in ("slip_warn", "abs_active", "vsc_active", "wheel_spread", "wheel_decel", "lateral",
                        "yaw_residual", "tc_disabled"):
                opt(rec, f"{A}.{col}", to_bool(row[col]), f"{src}:{col}")
            opt(rec, f"{A}.slip_warn_peak", row.slip_warn_peak, f"{src}:slip_warn_peak")
            opt(rec, f"{A}.wheel_speed_diff_peak_mps", row.ws_spread_peak_mps, f"{src}:ws_spread_peak_mps")
            opt(rec, f"{A}.wheel_decel_peak_mps2", row.wheel_decel_peak_mps2, f"{src}:wheel_decel_peak_mps2")
            opt(rec, f"{A}.speed_max_mps", row.v_max_mps, f"{src}:v_max_mps")
            for col, key in (("ax_at_peak_mps2", "ax_mps2"), ("ay_at_peak_mps2", "ay_mps2"),
                             ("yaw_rate_at_peak_dps", "yaw_rate_dps"), ("steer_at_peak_deg", "steering_angle_deg"),
                             ("beta_model_peak_deg", "sideslip_angle_model_deg")):
                opt(rec, f"{A}.at_peak.{key}", row[col], f"{src}:{col}")
            rec.doc[ORIGIN] = dict(file=src, row=int(row.name), joined_files=[meta_src], source_ids={})
            recs.append(rec)
    return recs


def convert_slip_beta(name: str, dc: dict, ctx: Ctx) -> list[Record]:
    recs, chk = [], Counter()
    for src, meta_src, df, meta in slip_dirs(dc, ctx):
        for _, row in df.iterrows():
            rec = base_record(ctx, dc["event_type"], dc["detector"], f"config detectors.{name}.event_type")
            empty_blocks(rec)
            put_time_boot(rec, row, "t_in_segment_s", "t_peak", src)
            dataset, drive, seg = slip_common(rec, ctx, row, src, meta_src, meta, "sideslip_beta_model",
                                              dc["reference"])
            # 検査: 山が開始と同じセグメントなら、元 CSV の peak_t_in_segment_s と 0.01 s 以内で合うはず
            if int(row.peak_segment) == seg:
                ok = abs(rec.get("time.peak_sec") - float(row.peak_t_in_segment_s)) <= 0.011
                chk["peak_sec_matches_csv_" + str(ok)] += 1
            rec.copy("grade", row.grade, f"{src}:grade")
            rec.const("score.name", "sideslip_confidence", "config")
            rec.copy("score.value", row.confidence, f"{src}:confidence")
            rec.copy("detection.rule", row.reasons, f"{src}:reasons (通過の理由)")
            put_frames(rec, ctx, dataset, row, src, "video_frame")

            A = "attributes"
            if meta.get("label"):
                rec.const(f"{A}.platform", meta["label"], f"{meta_src}:label")
            opt(rec, f"{A}.duration_s", row.duration_s, f"{src}:duration_s")
            for col, key in (("beta_peak_deg", "sideslip_angle_peak_deg"), ("beta_sigma_peak", "sideslip_angle_peak_sigma"),
                             ("beta_excess_peak_deg", "sideslip_angle_excess_peak_deg"),
                             ("yaw_sigma_peak", "yaw_residual_peak_sigma"), ("ay_peak_mps2", "ay_peak_mps2"),
                             ("beta_rate_peak_dps", "sideslip_angle_rate_peak_dps"),
                             ("ax_at_peak_mps2", "ax_at_peak_mps2")):
                opt(rec, f"{A}.{key}", row[col], f"{src}:{col}")
            opt(rec, f"{A}.n_runs", row.n_runs, f"{src}:n_runs", int)
            opt(rec, f"{A}.confidence_items", split_pipe(row.confidence_items), f"{src}:confidence_items ('|' を配列に)")
            opt(rec, f"{A}.reasons", split_pipe(row.reasons), f"{src}:reasons ('|' を配列に)")
            opt(rec, f"{A}.corroboration", split_pipe(row.corroboration), f"{src}:corroboration ('|' を配列に)")
            opt(rec, f"{A}.peak_segment", row.peak_segment, f"{src}:peak_segment", int)
            opt(rec, f"{A}.peak_sec_in_peak_segment", row.peak_t_in_segment_s, f"{src}:peak_t_in_segment_s")
            rec.doc[ORIGIN] = dict(file=src, row=int(row.name), joined_files=[meta_src], source_ids={})
            recs.append(rec)
    ctx.checks[name].update(chk)
    return recs


# ---------------------------------------------------------------------------
# 路面凹凸による異常振動
# ---------------------------------------------------------------------------

def physics_tables(paths: list[str], key: str, cols: list[str], ctx: Ctx, chk: Counter) -> dict:
    """複数の物理評価の表を key で 1 つにまとめる。両方にある行は cols の値が一致することを確かめる。"""
    out: dict = {}
    for p in paths:
        ctx.use(p)
        for _, r in read(p).iterrows():
            k = r[key]
            if k in out:
                same = all(str(clean(out[k][0][c])) == str(clean(r[c])) for c in cols if c in r.index)
                chk[f"physics_{key}_duplicate_consistent_{same}"] += 1
                continue
            out[k] = (r, p)
    return out


def convert_road_vibration(name: str, dc: dict, ctx: Ctx) -> list[Record]:
    loc = read(dc["locate"])
    nm = read(dc["near_miss"]).set_index("event_id")
    iso = read(dc["iso2631"]).set_index("event_id")
    sig = read(dc["physics_signals"]).set_index("site_id")
    ctx.use(dc["locate"], dc["near_miss"], dc["iso2631"], dc["physics_signals"], dc["locate_meta"],
            dc["near_miss_meta"], dc["physics_meta"])
    chk = Counter()
    pev = physics_tables(dc["physics_event"], "event_id", ["P1", "P2", "P3", "P4", "P5", "inferred_input"], ctx, chk)
    psite = physics_tables(dc["physics_site"], "site_id",
                           ["P1", "P2", "P3", "P4", "P5", "P6", "physical_consistency"], ctx, chk)

    lm = json.loads((REPO / dc["locate_meta"]).read_text())
    nmm = json.loads((REPO / dc["near_miss_meta"]).read_text())
    phm = json.loads((REPO / dc["physics_meta"]).read_text())
    # 振動イベントの検出条件。今の設定ファイルが出力時と同じときだけ値を書き込む
    base_cfg = REPO / yaml.safe_load((REPO / lm["config"]).read_text())["base_config"]
    same_cfg = sha10(REPO / lm["config"]) == lm.get("config_sha1") and sha10(base_cfg) == lm.get("base_config_sha1")
    chk["locate_config_unchanged_" + str(same_cfg)] += 1
    if same_cfg:
        a_min = yaml.safe_load(base_cfg.read_text())["longtail"]["a_min"]
        rule = dc["event_rule"].replace("longtail.a_min", f"{a_min}")
        rule_src = f"config detectors.{name}.event_rule ({base_cfg.relative_to(REPO)} の longtail.a_min = {a_min})"
    else:
        rule, rule_src = dc["event_rule"], f"config detectors.{name}.event_rule (設定が出力時と違うので値は入れない)"
    media = defaultdict(list)
    for d in dc["media_dirs"]:
        for f in sorted((REPO / d).glob("site*_E*")):
            eid = next((p for p in f.stem.split("_") if p.startswith("E") and p[1:].isdigit()), None)
            if eid:
                media[eid].append(str(f.relative_to(REPO)))
    cand_cols = dc["candidate_columns"]

    recs = []
    L, N, I = dc["locate"], dc["near_miss"], dc["iso2631"]
    for _, row in loc.iterrows():
        eid = row.event_id
        n = nm.loc[eid] if eid in nm.index else None
        same = (n is not None and n.drive == row.drive and int(n.segment) == int(row.segment)
                and abs(float(n.event_time_s) - float(row.event_time_s)) < 0.011)
        chk["near_miss_join_consistent_" + str(same)] += 1
        if not same:
            n = None
        rec = base_record(ctx, dc["event_type"], dc["detector"], f"config detectors.{name}.event_type")
        empty_blocks(rec)
        dataset = rec.copy("dataset", row.dataset, f"{L}:dataset")
        drive = rec.copy("drive_id", row.drive, f"{L}:drive")
        veh = ctx.dongle_vehicle.get(drive.split("|")[0])
        if veh:
            rec.const("vehicle", veh[0], f"{veh[1]}:dongle_ids")
        chunk = rec.copy("chunk", row.chunk, f"{L}:chunk")
        seg = int(rec.copy("segment", int(row.segment), f"{L}:segment"))

        rec.const("time.base", TIME_BASE, "schema")
        rec.const("time.segment_origin", ORIGIN_IMU, "scripts/vibration/find_roll_bumps.py load_seg (t0 = IMU 加速度の先頭)")
        rec.copy("time.start_sec", row.t_start_s, f"{L}:t_start_s")
        rec.copy("time.end_sec", row.t_end_s, f"{L}:t_end_s")
        rec.copy("time.peak_sec", row.t_peak_s, f"{L}:t_peak_s")
        rec.copy("time.utc", utc_iso(row.event_utc), f"{L}:event_utc (ISO 8601 に表記を変更。event_time_s の時刻)")

        rec.copy("location.latitude", row.lat, f"{L}:lat")
        rec.copy("location.longitude", row.lon, f"{L}:lon")
        rec.copy("location.heading_deg", row.heading_deg, f"{L}:heading_deg")
        rec.copy("location.gnss_acc_m", row.gnss_acc_m, f"{L}:gnss_acc_m")
        opt(rec, "attributes.position_error.along_m", row.pos_err_along_m, f"{L}:pos_err_along_m")
        opt(rec, "attributes.position_error.cross_m", row.pos_err_cross_m, f"{L}:pos_err_cross_m")
        rec.derive("speed_mps", kmh_to_mps(row.speed_kmh), "speed_mps = round(speed_kmh / 3.6, 3)")

        if n is not None:
            rec.const("score.name", "near_miss_candidate_score", "config")
            rec.copy("score.value", n.near_miss_candidate_score, f"{N}:near_miss_candidate_score")
            flags = [to_bool(n[c]) for c in cand_cols]
            if all(f is not None for f in flags):
                rec.derive("candidate_flag", all(flags), " AND ".join(cand_cols) + f" ({N})")

        rec.const("detection.method", "rule_threshold", "config")
        rec.const("detection.rule", rule, rule_src)
        rec.const("detection.config_hash", lm.get("config_sha1"), f"{dc['locate_meta']}:config_sha1")
        rec.const(f"{INTERNAL}.config_hashes", {"locate": lm.get("config_sha1"), "locate_base": lm.get("base_config_sha1"),
                                              "near_miss": nmm.get("config_sha1"), "physics": phm.get("config_sha1")},
                  "各 meta.json の config_sha1")
        rec.const("detection.criteria", {"candidate_rule": " AND ".join(cand_cols)}, f"config detectors.{name}")
        rec.const(f"{INTERNAL}.detection_reference", dc["reference"], "config")
        if ctx.commit:
            rec.const("detection.algorithm_version", ctx.commit, "--stamp-git-commit")

        put_source(rec, ctx, dataset, chunk, drive, seg)
        rec.derive("source.video_frame_start", video_frame(rec.get("time.start_sec"), ctx.fps, ctx.n_frames),
                   "video_frame_start = round(start_sec * fps)")
        rec.derive("source.video_frame_end", video_frame(rec.get("time.end_sec"), ctx.fps, ctx.n_frames),
                   "video_frame_end = round(end_sec * fps)。セグメントの外なら null")
        if media.get(eid):
            rec.const(f"{INTERNAL}.review_media", media[eid], "media/ のファイル名 (site<NNN>_<種類>_<event_id>)")

        if n is not None:
            rec.copy("review.video_road_feature", n.video_road_feature, f"{N}:video_road_feature")
            rec.const(f"{INTERNAL}.review_video_source", dc["review_source"], "rank_near_miss_vibration.py が位置で対応させた")

        A = "attributes"
        opt(rec, f"{A}.hit_sec", row.event_time_s, f"{L}:event_time_s (前輪が当たった時刻、segment の先頭から)")
        if n is not None:
            for col, key in (("heave_peak_abs", "vertical_accel_peak_mps2"), ("roll_peak_abs", "roll_rate_peak_dps"),
                             ("pitch_peak_abs", "pitch_rate_peak_dps"), ("heave_roll_dt_s", "heave_roll_dt_s"),
                             ("concentration", "concentration"), ("rise_s", "rise_s"), ("side_index", "side_index")):
                opt(rec, f"{A}.{key}", n[col], f"{N}:{col}")
            # 順位はその実行の中での並びなので API には出さない
            opt(rec, f"{INTERNAL}.A_rank", n.A_rank, f"{N}:A_rank")
            opt(rec, f"{INTERNAL}.near_miss_rank", n["rank"], f"{N}:rank")
            opt(rec, f"{A}.repeatability", n.repeatability, f"{N}:repeatability")
            for c in [*cand_cols, "vibration_event", "eleven_hz_only"]:
                opt(rec, f"{A}.conditions.{c}", to_bool(n[c]), f"{N}:{c}")
        for col, key in (("A", "A"), ("heave_ratio", "heave_ratio"), ("roll_metric", "roll_ratio"),
                         ("pitch_metric", "pitch_ratio"), ("roll_over_pitch", "roll_over_pitch"),
                         ("wheel_metric", "wheel_ratio_max"), ("wheel_lr_log2", "wheel_lr_log2"),
                         ("wheel_side", "wheel_side"), ("wheel_front_rear_lag_s", "wheel_front_rear_lag_s"),
                         ("wheel_lag_expected_s", "wheel_lag_expected_s"), ("heave_dom_hz", "heave_dom_hz"),
                         ("vibration_score", "vibration_score"), ("repeat_count", "site_event_count"),
                         ("gps_dt_s", "gps_dt_s"), ("ay_range_mps2", "ay_range_mps2")):
            opt(rec, f"{A}.{key}", row[col], f"{L}:{col}")
        for w in ("FL", "FR", "RL", "RR"):
            opt(rec, f"{A}.wheel_ratio.{w}", row[f"wheel_ratio_{w}"], f"{L}:wheel_ratio_{w}")
        for col, key in (("heave_road_ratio", "heave_3_8hz"), ("heave_wheelhop_ratio", "heave_8_15hz"),
                         ("roll_road_ratio", "roll_3_8hz"), ("roll_wheelhop_ratio", "roll_8_15hz")):
            opt(rec, f"{A}.band_ratio.{key}", row[col], f"{L}:{col}")
        for col in ("wheel_lag_match", "wheelhop_only", "road_band", "one_sided", "spike_like", "brake_any"):
            opt(rec, f"{A}.flags.{col}", to_bool(row[col]), f"{L}:{col}")
        if n is not None:
            opt(rec, f"{A}.feature_type", n.feature_type, f"{N}:feature_type")
            opt(rec, f"{INTERNAL}.site_rank_locate", n.site_rank_locate, f"{N}:site_rank_locate", int)

        site = int(row.site_id)
        P = f"{A}.physics"
        if site in sig.index:
            s = sig.loc[site]
            opt(rec, f"{P}.signal_count", s.physics_signals, f"{dc['physics_signals']}:physics_signals (地点単位)", int)
            for c in ("P2", "P4", "P5", "P6"):
                opt(rec, f"{P}.signals.{c}", s[c], f"{dc['physics_signals']}:{c} (地点単位)")
        if site in psite:
            s, sp = psite[site]
            opt(rec, f"{P}.consistency", s.physical_consistency, f"{sp}:physical_consistency (地点単位)")
            opt(rec, f"{P}.inferred_side", s.inferred_side, f"{sp}:inferred_side (地点単位)")
            for c in ("P1", "P2", "P3", "P4", "P5", "P6"):
                opt(rec, f"{P}.site.{c}", s[c], f"{sp}:{c}")
        if eid in pev:
            e, ep = pev[eid]
            same_ev = e.drive == row.drive and int(e.segment) == seg
            chk["physics_event_join_consistent_" + str(same_ev)] += 1
            if same_ev:
                opt(rec, f"{P}.event.inferred_input", e.inferred_input, f"{ep}:inferred_input")
                for c in ("P1", "P2", "P3", "P4", "P5"):
                    opt(rec, f"{P}.event.{c}", e[c], f"{ep}:{c}")

        if eid in iso.index:
            i = iso.loc[eid]
            S = f"{A}.iso2631"
            for col, key in (("mtvv_total", "mtvv_total_mps2"), ("mtvv_translational", "mtvv_translational_mps2"),
                             ("mtvv_x", "mtvv_x_mps2"), ("mtvv_y", "mtvv_y_mps2"), ("mtvv_z", "mtvv_z_mps2"),
                             ("mtvv_rx", "mtvv_rx_seat_mps2"), ("mtvv_ry", "mtvv_ry_seat_mps2"),
                             ("vdv_z", "vdv_z_mps175"), ("vdv_total", "vdv_total_mps175"), ("aw_rms_z", "aw_rms_z_mps2"),
                             ("crest_z", "crest_factor_z"), ("mtvv_over_aw_z", "mtvv_over_aw_z"),
                             ("dominant_axis", "dominant_axis"), ("comfort", "comfort")):
                opt(rec, f"{S}.{key}", i[col], f"{I}:{col}")
            for col, key in (("mtvv_z_lp8", "mtvv_z_mps2"), ("mtvv_total_lp8", "mtvv_total_mps2"),
                             ("comfort_lp8", "comfort")):
                opt(rec, f"{S}.below_8hz.{key}", i[col], f"{I}:{col}")
            for col, key in (("mtvv_total_normal_pct", "mtvv_total"), ("mtvv_z_normal_pct", "mtvv_z"),
                             ("vdv_z_normal_pct", "vdv_z"), ("mtvv_total_lp8_normal_pct", "mtvv_total_below_8hz")):
                opt(rec, f"{S}.normal_percentile.{key}", i[col], f"{I}:{col}")

        joined = [N, I, dc["physics_signals"], *dc["physics_event"], *dc["physics_site"]]
        rec.doc[ORIGIN] = dict(file=L, row=int(row.name), joined_files=joined,
                                 source_ids={"event_id": eid, "site_id": site})
        recs.append(rec)
    ctx.checks[name].update(chk)
    return recs


CONVERTERS = {"l3": convert_l3, "slip_direct": convert_slip_direct, "slip_beta": convert_slip_beta,
              "road_vibration": convert_road_vibration}


# ---------------------------------------------------------------------------
# 仕上げ
# ---------------------------------------------------------------------------

def assign_ids(recs: list[Record]) -> None:
    for r in recs:
        d = r.doc
        d["event_id"] = stable_event_id(d["dataset"], d["drive_id"], d["segment"], d["time"]["start_sec"],
                                        d["event_type"], d["detector"])
        r.prov["event_id"] = ("derived", "stable_event_id(dataset, drive_id, segment, start_sec, event_type, detector)")


KEEP_EMPTY = {"attributes"}   # 必須なので空でも残す


def prune(d: dict, top: bool = True) -> dict:
    """null と空の object を落とす (API のレスポンスを null だらけにしない)。"""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = prune(v, False)
            if not v and not (top and k in KEEP_EMPTY):
                continue
        elif v is None:
            continue
        out[k] = v
    return out


def split(rec: Record) -> tuple[dict, dict]:
    """Record → (canonical event, provenance)。内部の値と変換元は provenance へ移す。"""
    doc = {k: v for k, v in rec.doc.items() if k not in (INTERNAL, ORIGIN)}
    missing = [p for p in null_leaves(doc)]
    if not doc.get("review"):
        missing.append("review")
    event = prune(doc)
    fields, derived = {}, []
    for path, (kind, src) in sorted(rec.prov.items()):
        key = path.replace(f"{INTERNAL}.", "internal.", 1)
        fields[key] = {"kind": kind, "source": src}
        if kind == "derived" and not path.startswith(INTERNAL):
            derived.append({"path": path, "rule": src})
    prov = dict(event_id=event["event_id"], origin=rec.doc[ORIGIN], derived_fields=derived,
                missing_fields=sorted(missing), internal=prune(rec.doc[INTERNAL], False), fields=fields)
    return event, prov


def pick_samples(events: list[dict], n: int, grade_order: list[str]) -> list[dict]:
    """種別ごとに、(検出器, データセット) を順に回して n 件。

    各組の中は レビューあり → candidate_flag が true → スコアの高い順 → 等級 (grade_order の順) → event_id。
    """
    rank = {g: i for i, g in enumerate(grade_order)}
    out = []
    by_type = defaultdict(lambda: defaultdict(list))
    for e in events:
        by_type[e["event_type"]][(e["detector"], e["dataset"])].append(e)
    for et in sorted(by_type):
        groups = []
        for k in sorted(by_type[et]):
            g = sorted(by_type[et][k], key=lambda e: (not e.get("review"), e.get("candidate_flag") is not True,
                                                      -e.get("score", {}).get("value", -1e9),
                                                      rank.get(e.get("grade"), len(rank)), e["event_id"]))
            groups.append(g)
        picked, i = [], 0
        while len(picked) < n and any(i < len(g) for g in groups):
            for g in groups:
                if i < len(g) and len(picked) < n:
                    picked.append(g[i])
            i += 1
        out.extend(picked)
    return out


def report(recs_by_det: dict[str, list[Record]], events: list[dict], provs: list[dict], ctx: Ctx, errors: list,
           uri_check: dict, cfg_path: Path) -> dict:
    """変換全体の検証用の集計 (イベントごとの詳細は provenance.jsonl)。"""
    prov_by_id = {p["event_id"]: p for p in provs}
    dets = {}
    for name, recs in recs_by_det.items():
        fields = defaultdict(Counter)
        missing, attrs = Counter(), Counter()
        for r in recs:
            for path, (kind, src) in r.prov.items():
                if path.startswith("attributes."):
                    attrs[path[len("attributes."):]] += 1
                else:
                    fields[path.replace(f"{INTERNAL}.", "internal.", 1)][f"{kind}: {src}"] += 1
            missing.update(prov_by_id[r.doc["event_id"]]["missing_fields"])
        dets[name] = dict(
            n_events=len(recs),
            event_types=dict(Counter(r.doc["event_type"] for r in recs)),
            datasets=dict(Counter(r.doc["dataset"] for r in recs)),
            fields={p: dict(c) for p, c in sorted(fields.items())},
            missing={p: c for p, c in sorted(missing.items())},
            attributes_present={p: c for p, c in sorted(attrs.items())},
            checks=dict(ctx.checks.get(name, {})),
        )
    n = len(events)
    all_missing = Counter(m for p in provs for m in p["missing_fields"])
    inputs = []
    for p in sorted(ctx.inputs):
        f = REPO / p
        if f.is_file():
            inputs.append(dict(file=p, sha1=hashlib.sha1(f.read_bytes()).hexdigest(),
                               modified=datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")))
        else:
            inputs.append(dict(file=p, sha1=None, modified=None))
    return dict(
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        exporter="scripts/export_near_miss_canonical.py",
        exporter_commit=git_commit(REPO),
        config=str(cfg_path.relative_to(REPO)) if cfg_path.resolve().is_relative_to(REPO) else str(cfg_path),
        algorithm_version_stamped=ctx.commit,
        uri_base=ctx.uri_base,
        n_events=n,
        by_event_type=dict(Counter(e["event_type"] for e in events)),
        by_detector_dataset={f"{k[0]} / {k[1]}": v for k, v in
                             sorted(Counter((e["detector"], e["dataset"]) for e in events).items())},
        missing_rate={p: round(c / n, 4) for p, c in sorted(all_missing.items(), key=lambda x: -x[1])},
        event_id=dict(rule="nm_ + sha256('\\n'.join([dataset, drive_id, segment, start_sec(10 ms 単位に四捨五入), "
                           "event_type, detector]))[:16]",
                      unique=len({e["event_id"] for e in events}) == n),
        uri_check=uri_check,
        validation=dict(n_checked=n, n_errors=len(errors), errors=errors[:50]),
        inputs=inputs,
        detectors=dets,
    )


def local_path(uri: str, ctx: Ctx) -> Path | None:
    """URI (データのルートからの相対パス) → 手元のパス。--uri-base を付けたときは調べない。"""
    if ctx.uri_base:
        return None
    root, _, rest = uri.partition("/")
    base = ctx.local_roots.get(root)
    return base / rest if base is not None else None


def check_uris(events: list[dict], ctx: Ctx) -> dict:
    """生成した *_uri が手元の元データに実在するか (configs の uri_local_roots で読み替える)。"""
    res = defaultdict(Counter)
    for e in events:
        for k in ("video_uri", "can_uri", "imu_uri", "gnss_uri"):
            uri = e["source"].get(k)
            if uri is None:
                continue
            p = local_path(uri, ctx)
            state = "not_checked" if p is None else ("exists" if p.exists() else "not_found_locally")
            res[f"{e['dataset']}.{k}"][state] += 1
    return {k: dict(v) for k, v in sorted(res.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO / "configs/near_miss_canonical.yaml")
    ap.add_argument("--out", type=Path, help="出力先 (既定は config の out_dir)")
    ap.add_argument("--only", nargs="*", help="この検出器 (config の detectors のキー) だけ変換する")
    ap.add_argument("--uri-base", default="",
                    help="source.*_uri の前に付ける (例 s3://<バケット>/near_miss/)。既定は付けない (相対パス)")
    ap.add_argument("--stamp-git-commit", action="store_true",
                    help="detection.algorithm_version に現在の git commit を入れる (出力を作り直した直後のみ)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    ctx = Ctx(cfg, args.stamp_git_commit)
    ctx.uri_base = args.uri_base if not args.uri_base or args.uri_base.endswith("/") else args.uri_base + "/"
    out = args.out if args.out else REPO / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    recs_by_det: dict[str, list[Record]] = {}
    for name, dc in cfg["detectors"].items():
        if args.only and name not in args.only:
            continue
        recs = CONVERTERS[dc["kind"]](name, dc, ctx)
        assign_ids(recs)
        recs_by_det[name] = recs
        print(f"{name}: {len(recs)} 件")

    pairs = [split(r) for rs in recs_by_det.values() for r in rs]
    pairs.sort(key=lambda ep: (ep[0]["event_type"], ep[0]["dataset"], ep[0]["drive_id"], str(ep[0].get("segment")),
                               ep[0]["time"]["start_sec"], ep[0]["detector"]))
    events = [e for e, _ in pairs]
    provs = [p for _, p in pairs]
    ids = Counter(e["event_id"] for e in events)
    dup = [k for k, c in ids.items() if c > 1]
    if dup:
        print(f"event_id が重複: {dup[:5]}", file=sys.stderr)
        return 1

    schema = load_schema(REPO / cfg["schema"])
    errors = validate_events(events, schema)
    uri = check_uris(events, ctx)

    def jsonl(path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")

    jsonl(out / "events.jsonl", events)
    jsonl(out / "provenance.jsonl", provs)
    samples = pick_samples(events, cfg["sample"]["per_event_type"], cfg["sample"].get("grade_order", []))
    (out / "sample_events.json").write_text(json.dumps(
        {"schema": cfg["schema"], "schema_version": SCHEMA_VERSION,
         "note": "events.jsonl から種別ごとに数件を抜いたもの。各イベントは schemas/near_miss_event.schema.json に従う",
         "events": samples}, ensure_ascii=False, indent=2), encoding="utf-8")
    rep = report(recs_by_det, events, provs, ctx, errors, uri, args.config)
    (out / "conversion_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"計 {len(events)} 件 → {out}/events.jsonl (provenance.jsonl も同数)")
    print("種別:", rep["by_event_type"])
    print(f"schema 検証: {len(events) - len({e for e, _ in errors})} / {len(events)} 件が通過、エラー {len(errors)} 件")
    for k, v in uri.items():
        print(f"  {k}: {v}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
