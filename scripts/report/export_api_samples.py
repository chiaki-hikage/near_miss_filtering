#!/usr/bin/env python3
"""API 連携の検討用に、既存のヒヤリハット出力からサンプルを抜き出す。

  uv run python scripts/report/export_api_samples.py

出力 (configs/near_miss_api_samples.yaml の out_dir):

  *_sample.csv                 既存の CSV の行を**列も値もそのまま**抜いたもの (現状の形式)
  api_response_examples.json   API 化した場合のレスポンス例 (**提案**。現状の形式ではない)

**再計算はしない。** JSON の値は既存の CSV の値をそのまま写すか、
CSV の列から簡単な式で出したもの (`derived` に式を残す) だけにする。
出力に無い値は null にして `missing` に名前を挙げる。推測で埋めない。
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]


def read(path: str) -> pd.DataFrame:
    return pd.read_csv(REPO / path)


def write_raw(parts: list[tuple[str, pd.Index]], dst: Path) -> None:
    """抜いた行を、元の CSV の文字列のまま書く。

    pandas で書き直すと浮動小数の末尾の桁が変わることがあるので、
    元ファイルを csv で読み、read() の行番号 (RangeIndex) と同じ行をそのまま写す。
    parts は (元ファイル, 行番号) の並び。元ファイルが複数でも列は同じであること。
    """
    header, body = None, []
    for src, index in parts:
        with open(REPO / src, newline="", encoding="utf-8") as f:
            raw = list(csv.reader(f))
        if header is not None and raw[0] != header:
            raise ValueError(f"{src}: 列が他のファイルと違う")
        header = raw[0]
        seen = set()
        for i in index:
            if int(i) not in seen:
                seen.add(int(i))
                body.append(raw[int(i) + 1])
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(body)


def jsonable(v):
    """pandas / numpy の値を JSON に出せる形にする。欠測は None。"""
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if v is None or (isinstance(v, str) and v == ""):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def chunk_index(raw_root: Path) -> dict[str, str]:
    """comma2k19 の drive_id → チャンク名。出力に無いので生データの置き場所から引く。"""
    out = {}
    for chunk in sorted(raw_root.glob("Chunk_*")):
        for d in chunk.iterdir():
            if d.is_dir() and "|" in d.name:
                out[d.name] = chunk.name
    return out


def overlap(df: pd.DataFrame, drive: str, t0: float, t1: float) -> pd.DataFrame:
    """区間の重なりで突き合わせる (io_spec.md 4.4 と同じ式)。"""
    return df[(df.drive_id == drive) & (df.t_end >= t0) & (df.t_start <= t1)]


# ---------------------------------------------------------------------------
# 既存の L3 イベント (急ブレーキ・急操舵)
# ---------------------------------------------------------------------------

EXISTING_TYPE = {"hard_brake": "hard_braking", "hard_steer": "hard_steering"}


def existing_samples(cfg: dict, chunks: dict, out: Path) -> list[dict]:
    c = cfg["existing"]["comma2k19"]
    ev = read(c["events"])
    # 元ファイルと行番号を持たせておき、サンプルを書くときに元の文字列を写す
    cand = pd.concat([read(p).assign(_src=p, _row=lambda d: d.index) for p in c["candidates"]],
                     ignore_index=True)
    labels = pd.concat([read(p) for p in c["labels"]], ignore_index=True)

    rows = []
    for etype, drive, t0 in c["picks"]:
        m = ev[(ev.event_type == etype) & (ev.drive_id == drive) & np.isclose(ev.t_start, t0)]
        if len(m) != 1:
            raise SystemExit(f"{etype} {drive} {t0}: {len(m)} 行が一致 (1 行のはず)")
        rows.append(m.iloc[0])
    picked = pd.DataFrame(rows)
    for etype, g in picked.groupby("event_type", sort=False):
        write_raw([(c["events"], g.index)], out / f"{EXISTING_TYPE[etype]}_events_sample.csv")

    # 同じ区間の候補行 (区間内の集計値はこちらにしか無い)
    cand_rows = pd.concat([overlap(cand, e.drive_id, e.t_start, e.t_end) for _, e in picked.iterrows()])
    write_raw([(src, g._row) for src, g in cand_rows.groupby("_src", sort=False)],
              out / "existing_candidates_sample.csv")

    # commaCarSegments (映像・位置なし) の例
    cc = cfg["existing"]["comma_car_segments"]
    ccs = read(cc["events"])
    ccs_rows = []
    for etype, n in cc["head_per_type"].items():
        # ファイルの並び順で先頭から。値の大きい順に選ぶと外れ値 (始動直後の無効値など) を引きやすい
        ccs_rows.append(ccs[ccs.event_type == etype].head(n))
    ccs_pick = pd.concat(ccs_rows)
    write_raw([(cc["events"], ccs_pick.index)], out / "existing_events_ccs_sample.csv")

    vf = cfg["video_fps"]
    docs = []
    for _, e in pd.concat([picked, ccs_pick]).iterrows():
        is_c2k = e.get("dataset", "") == "comma2k19" or e.get("platform", "") == "comma2k19"
        dataset = "comma2k19" if is_c2k else "comma_car_segments"
        seg = int(e.segment)
        lab = overlap(labels, e.drive_id, e.t_start, e.t_end) if is_c2k else labels.iloc[0:0]
        lab = lab[lab.segment == seg] if len(lab) else lab
        cnd = overlap(cand, e.drive_id, e.t_start, e.t_end) if is_c2k else cand.iloc[0:0]
        cnd = cnd.iloc[0] if len(cnd) else None
        chunk = chunks.get(e.drive_id) if is_c2k else None
        seg_dir = f"{chunk}/{e.drive_id}/{seg}" if chunk else None
        dongle, route = e.drive_id.split("|")
        attrs = {
            "feature": e.feature, "op": e.op, "threshold": jsonable(e.threshold),
            "peak_value": jsonable(e.peak_value), "exceedance": jsonable(e.exceedance),
            "stage": jsonable(e.stage), "weight": jsonable(e.weight),
            "gate": jsonable(e.gate), "rule": e.rule,
        }
        if cnd is not None:
            keys = (["ax_mps2_min", "jerk_mps3_min", "jerk_win_min_min", "brake_pressed_max", "gvc_mps2_min"]
                    if e.event_type == "hard_brake" else
                    ["ay_kin_mps2_absmax", "ay_can_mps2_absmax", "steer_rate_dps_absmax", "yaw_rate_dps_absmax",
                     "lat_jerk_mps3_absmax"])
            attrs["candidate_window"] = {k: jsonable(cnd[k]) for k in keys}
        docs.append({
            "event_id": None,
            "event_type": EXISTING_TYPE[e.event_type],
            "source_event_type": e.event_type,
            "label": e.label,
            "dataset": dataset,
            "vehicle": e.vehicle,
            "drive_id": e.drive_id,
            "segment": seg,
            "chunk": chunk,
            "time": {
                "base": "segment_relative_s",
                "start_sec": jsonable(e.t_in_segment_s),
                "end_sec": round(float(e.t_in_segment_s + e.duration_s), 2),
                "peak_sec": None,
                "boot_time_start_s": jsonable(e.t_start),
                "boot_time_end_s": jsonable(e.t_end),
            },
            "location": None,
            "speed_mps": jsonable(cnd["v_mps_mean"]) if cnd is not None else None,
            "score": {"name": "candidate_severity", "value": jsonable(cnd["severity"])} if cnd is not None else None,
            "candidate_flag": None,
            "detection": {"method": "rule_threshold", "rule": e.trigger_rule, "config_hash": e.config_hash,
                          "algorithm_version": None, "created_at": None},
            "source": {
                "video_uri": f"{seg_dir}/video.hevc" if seg_dir else None,
                "video_frame_start": jsonable(e.video_frame) if is_c2k and "video_frame" in e else None,
                "can_uri": (f"{seg_dir}/processed_log/CAN/raw_can" if seg_dir else
                            None if is_c2k else f"{cfg['ccs_base_url']}/segments/{dongle}/{route}/{seg}/rlog.zst"),
                "gnss_uri": f"{seg_dir}/global_pose" if seg_dir else None,
            },
            "review": ({"risky": jsonable(lab.iloc[0].risky), "verdict": lab.iloc[0].verdict}
                       if len(lab) else None),
            "attributes": attrs,
            "derived": [
                "time.end_sec = t_in_segment_s + duration_s",
                *(["chunk / source.*_uri は raw_data/ の配置から引いた (出力の列には無い)"] if is_c2k else
                  ["source.can_uri は drive_id と segment から公開元の URL の規則で組み立てた"]),
                *(["speed_mps = 候補行の v_mps_mean (候補区間の平均。イベント時点の値ではない)",
                   "score = 候補行の severity (区間の重なりで突き合わせた候補)"] if cnd is not None else []),
            ],
            "missing": ["event_id", "time.peak_sec", "location", "candidate_flag",
                        "detection.algorithm_version", "detection.created_at"]
                       + ([] if is_c2k else ["video (データセットに無い)", "speed_mps", "score"]),
        })
    return docs


# ---------------------------------------------------------------------------
# スリップ
# ---------------------------------------------------------------------------

def slip_samples(cfg: dict, chunks: dict, out: Path) -> list[dict]:
    s = cfg["slip"]
    d = read(s["direct"])
    pick = pd.concat([d[d.grade == g].head(1) for g in s["direct_grades"]])
    write_raw([(s["direct"], pick.index)], out / "slip_direct_evidence_sample.csv")
    b = read(s["beta"]).head(s["beta_n"])
    write_raw([(s["beta"], b.index)], out / "slip_beta_comma2k19_sample.csv")

    docs = []
    for _, e in pick.iterrows():
        dongle, route = e.drive_id.split("|")
        seg = int(e.segment)
        docs.append({
            "event_id": None,
            "event_type": "slip",
            "source_branch": "direct_slip",
            "dataset": e.dataset,
            "vehicle": None,
            "drive_id": e.drive_id,
            "segment": seg,
            "chunk": None,
            "time": {"base": "segment_relative_s", "start_sec": jsonable(e.t_in_segment_s),
                     "end_sec": round(float(e.t_in_segment_s + e.duration_s), 2), "peak_sec": None,
                     "boot_time_start_s": jsonable(e.t_start), "boot_time_end_s": jsonable(e.t_end)},
            "location": None,
            "speed_mps": jsonable(e.v_at_peak_mps),
            "score": None,
            "grade": e.grade,
            "candidate_flag": None,
            "detection": {"method": "direct_evidence", "rule": None, "config_hash": e.config_hash,
                          "algorithm_version": None, "created_at": None},
            "source": {"video_uri": None, "video_frame_start": None,
                       "can_uri": f"{cfg['ccs_base_url']}/segments/{dongle}/{route}/{seg}/rlog.zst",
                       "gnss_uri": None},
            "review": None,
            "attributes": {k: jsonable(e[k]) for k in [
                "evidence", "corroboration", "mechanisms", "n_mechanisms", "slip_warn", "slip_warn_peak",
                "abs_active", "vsc_active", "wheel_spread", "ws_spread_peak_mps", "wheel_decel",
                "wheel_decel_peak_mps2", "lateral", "yaw_residual", "tc_disabled", "ax_at_peak_mps2",
                "ay_at_peak_mps2", "yaw_rate_at_peak_dps", "steer_at_peak_deg", "beta_model_peak_deg"]},
            "derived": ["time.end_sec = t_in_segment_s + duration_s",
                        "source.can_uri は drive_id と segment から公開元の URL の規則で組み立てた",
                        "speed_mps = v_at_peak_mps"],
            "missing": ["event_id", "vehicle (ファイル名の車種でしか分からない)", "time.peak_sec",
                        "location", "score (等級のみ)", "candidate_flag", "video (データセットに無い)",
                        "detection.algorithm_version", "detection.created_at"],
        })
    for _, e in b.iterrows():
        chunk = chunks.get(e.drive_id)
        seg_dir = f"{chunk}/{e.drive_id}/{int(e.segment)}" if chunk else None
        docs.append({
            "event_id": None,
            "event_type": "slip",
            "source_branch": "sideslip_beta",
            "dataset": e.dataset,
            "vehicle": None,
            "drive_id": e.drive_id,
            "segment": int(e.segment),
            "chunk": chunk,
            "time": {"base": "segment_relative_s", "start_sec": jsonable(e.t_in_segment_s),
                     "end_sec": round(float(e.t_in_segment_s + e.duration_s), 2),
                     "peak_sec": jsonable(e.peak_t_in_segment_s), "peak_segment": int(e.peak_segment),
                     "boot_time_start_s": jsonable(e.t_start), "boot_time_end_s": jsonable(e.t_end),
                     "boot_time_peak_s": jsonable(e.t_peak)},
            "location": None,
            "speed_mps": jsonable(e.v_at_peak_mps),
            "score": {"name": "confidence", "value": jsonable(e.confidence)},
            "grade": e.grade,
            "candidate_flag": None,
            "detection": {"method": "sideslip_beta", "rule": e.reasons, "config_hash": e.config_hash,
                          "algorithm_version": None, "created_at": None},
            "source": {"video_uri": f"{seg_dir}/video.hevc" if seg_dir else None,
                       "video_frame_start": jsonable(e.video_frame),
                       "can_uri": f"{seg_dir}/processed_log/CAN/raw_can" if seg_dir else None,
                       "gnss_uri": f"{seg_dir}/global_pose" if seg_dir else None},
            "review": None,
            "attributes": {k: jsonable(e[k]) for k in [
                "beta_peak_deg", "beta_sigma_peak", "beta_excess_peak_deg", "yaw_sigma_peak", "ay_peak_mps2",
                "beta_rate_peak_dps", "ax_at_peak_mps2", "n_runs", "confidence_items", "corroboration"]},
            "derived": ["time.end_sec = t_in_segment_s + duration_s",
                        "chunk / source.*_uri は raw_data/ の配置から引いた (出力の列には無い)",
                        "speed_mps = v_at_peak_mps"],
            "missing": ["event_id", "vehicle", "location", "candidate_flag",
                        "detection.algorithm_version", "detection.created_at"],
        })
    return docs


# ---------------------------------------------------------------------------
# 路面凹凸による異常振動
# ---------------------------------------------------------------------------

def vibration_samples(cfg: dict, out: Path) -> list[dict]:
    v = cfg["road_vibration"]
    ids = v["event_ids"]
    def pick(df: pd.DataFrame) -> pd.DataFrame:
        # 列の並びは元の CSV のまま、行だけ event_ids の順にする
        return df.iloc[[int(np.flatnonzero(df.event_id == i)[0]) for i in ids]]

    nm, loc, iso = pick(read(v["near_miss"])), pick(read(v["locate"])), pick(read(v["iso2631"]))
    pev = read(v["physics_event"])
    pev = pev[pev.event_id.isin(ids)]
    psite = read(v["physics_site"])
    psig = read(v["physics_signals"])
    sites = read(v["sites"])
    write_raw([(v["near_miss"], nm.index)], out / "road_vibration_near_miss_sample.csv")
    write_raw([(v["locate"], loc.index)], out / "road_vibration_locate_sample.csv")
    write_raw([(v["iso2631"], iso.index)], out / "road_vibration_iso2631_sample.csv")
    write_raw([(v["physics_event"], pev.index)], out / "road_vibration_physics_event_sample.csv")
    sid = sorted(set(nm.site_id))
    write_raw([(v["physics_site"], psite.index[psite.site_id.isin(sid)])],
              out / "road_vibration_physics_site_sample.csv")
    write_raw([(v["sites"], sites.index[sites.site_id.isin(sid)])], out / "road_vibration_site_sample.csv")

    sha_locate = json.loads((REPO / v["locate_meta"]).read_text()).get("config_sha1")
    sha_nm = json.loads((REPO / v["near_miss_meta"]).read_text()).get("config_sha1")
    media = REPO / v["media_dir"]
    docs = []
    for _, e in nm.iterrows():
        l = loc[loc.event_id == e.event_id].iloc[0]
        i = iso[iso.event_id == e.event_id].iloc[0]
        p = pev[pev.event_id == e.event_id]
        ps = psite[psite.site_id == e.site_id]
        sg = psig[psig.site_id == e.site_id]
        seg_dir = f"{l.chunk}/{e.drive}/{int(e.segment)}"
        clips = sorted(str(f.relative_to(REPO)) for f in media.glob(f"site{int(e.site_id):03d}_clip_{e.event_id}*"))
        frames = sorted(str(f.relative_to(REPO)) for f in media.glob(f"site{int(e.site_id):03d}_frames_{e.event_id}*"))
        cand = bool(e[v["candidate_columns"]].all())
        docs.append({
            "event_id": None,
            "event_type": "road_vibration",
            "label": e.label,
            "dataset": l.dataset,
            "vehicle": None,
            "drive_id": e.drive,
            "segment": int(e.segment),
            "chunk": l.chunk,
            "time": {"base": "segment_relative_s", "start_sec": jsonable(l.t_start_s),
                     "end_sec": jsonable(l.t_end_s), "peak_sec": jsonable(l.t_peak_s),
                     "hit_sec": jsonable(e.event_time_s), "utc": e.event_utc},
            # near_miss_vibration_events.csv の lat/lon は小数 3 桁に丸めてあるので、6 桁ある locate 側を使う
            "location": {"latitude": jsonable(l.lat), "longitude": jsonable(l.lon),
                         "heading_deg": jsonable(l.heading_deg), "gnss_acc_m": jsonable(l.gnss_acc_m),
                         "pos_err_along_m": jsonable(l.pos_err_along_m), "site_id": int(e.site_id)},
            "speed_mps": round(float(e.speed_kmh) / 3.6, 2),
            "score": {"name": "near_miss_candidate_score", "value": jsonable(e.near_miss_candidate_score)},
            "candidate_flag": cand,
            "detection": {"method": "rule_threshold", "rule": " & ".join(v["candidate_columns"]),
                          "config_hash": {"locate": sha_locate, "near_miss": sha_nm},
                          "algorithm_version": None, "created_at": None},
            "source": {"video_uri": f"{seg_dir}/video.hevc", "video_frame_start": None,
                       "can_uri": f"{seg_dir}/processed_log/CAN", "imu_uri": f"{seg_dir}/processed_log/IMU",
                       "gnss_uri": f"{seg_dir}/global_pose",
                       "review_clips": clips, "review_frames": frames},
            "review": {"video_road_feature": e.video_road_feature},
            "attributes": {
                "vertical_accel_peak_mps2": jsonable(e.heave_peak_abs),
                "roll_rate_peak_dps": jsonable(e.roll_peak_abs),
                "pitch_rate_peak_dps": jsonable(e.pitch_peak_abs),
                "heave_roll_dt_s": jsonable(e.heave_roll_dt_s),
                "concentration": jsonable(e.concentration),
                "rise_s": jsonable(e.rise_s),
                "A": jsonable(e.A),
                "vibration_score": jsonable(l.vibration_score),
                "wheel_metric": jsonable(e.wheel_metric),
                "wheel_lr_log2": jsonable(e.wheel_lr_log2),
                "side_index": jsonable(e.side_index),
                "conditions": {k: bool(e[k]) for k in v["candidate_columns"]},
                "repeatability": e.repeatability,
                "physics": {
                    "event_P1_P5": ({k: jsonable(p.iloc[0][k]) for k in ["P1", "P2", "P3", "P4", "P5"]}
                                    if len(p) else None),
                    "inferred_input": jsonable(p.iloc[0].inferred_input) if len(p) else None,
                    "site_physical_consistency": ps.iloc[0].physical_consistency if len(ps) else None,
                    "site_physics_signal_count": int(sg.iloc[0].physics_signals) if len(sg) else None,
                },
                "iso2631": {"mtvv_total_mps2": jsonable(i.mtvv_total), "mtvv_z_mps2": jsonable(i.mtvv_z),
                            "vdv_z_mps175": jsonable(i.vdv_z), "comfort": i.comfort,
                            "mtvv_total_normal_pct": jsonable(i.mtvv_total_normal_pct)},
            },
            "derived": [
                "candidate_flag = 5 条件の列がすべて true (出力に専用の列は無い)",
                "location は locate/candidates.csv の値 (near_miss 側は小数 3 桁に丸めてある)",
                "speed_mps = speed_kmh / 3.6",
                "source.*_uri は chunk・drive・segment から raw_data/ の配置の規則で組み立てた",
                "physics.site_* は地点単位の値 (イベント単位ではない)。物理評価は候補の判定には使っていない",
            ],
            "missing": ["event_id (現状の E 番号は実行ごとに振り直す連番)", "vehicle", "time.video_frame",
                        "detection.algorithm_version", "detection.created_at"],
        })
    return docs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO / "configs/near_miss_api_samples.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = REPO / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    chunks = chunk_index(REPO / cfg["raw_root"])

    events = existing_samples(cfg, chunks, out) + slip_samples(cfg, chunks, out) + vibration_samples(cfg, out)
    doc = {
        "_status": "PROPOSAL — API 化した場合のレスポンス例。現状のデータ形式ではない。"
                   "現状の形式は同じディレクトリの *_sample.csv を見ること",
        "_notes": [
            "値は out/ 以下の既存 CSV から写したもの。再計算はしていない",
            "null は現状の出力に無い値。各イベントの missing に名前を挙げた",
            "derived は CSV の列から簡単な式で出した値と、その式",
            "event_id は現状どのイベント種別にも安定した ID が無いため null にした",
            "time.base = segment_relative_s: セグメント (60 秒) 先頭からの秒。"
            "先頭の定義は系統で違う (既存・スリップは CAN 主系列の重なりの先頭、振動は IMU 加速度の先頭)",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/report/export_api_samples.py",
        "events": events,
    }
    (out / "api_response_examples.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    print(f"{len(events)} 件 → {out.relative_to(REPO)}")
    for f in sorted(out.iterdir()):
        print(" ", f.name)


if __name__ == "__main__":
    main()
