#!/usr/bin/env python3
"""L3 のイベント判定条件を設定から機械的に書き出す。

**手で書き写さない。** configs/detection.yaml をそのまま読んで表にする。
閾値を変えたときに一覧がずれないようにするため。

派生特徴量そのものの定義 (weave_active など) は別節にあるので、
その節の設定値も併記する。

  uv run python scripts/report/dump_event_rules.py            # 表形式
  uv run python scripts/report/dump_event_rules.py --markdown # Markdown 表
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from near_miss.config import DEFAULT_DETECTION, load_yaml

# op の意味。detectors.py の実装に対応する。
OPS = {
    "lt": "x <",
    "gt": "x >",
    "abs_gt": "|x| >",
    "rising": "x が上向きに跨ぐ:",
    "agreement": "複数条件の AND (同時刻。tolerance_s で前後に広げる)",
    "cooccurrence": "他イベントとの共起 (window_s 内に min_count 件以上)",
    "sequence": "first の後 min_gap_s〜max_gap_s に then が起きる",
}

# 判定量が別節で作られているもの。節名を引けるようにする。
DERIVED = {
    "weave_active": "weaving", "s_evasion_active": "s_evasion",
    "panic_brake_active": "panic_brake", "panic_brake_pedal_active": "panic_brake",
    "closing_fast_active": "closing", "cut_in_active": "cut_in",
    "lc_active": "lane_change", "ws_anomaly_active": "wheel_speed",
    "counter_steer_active": "physics", "yaw_residual_sigma": "physics",
    "thw_s": "lead", "ttc_s": "lead", "lead_distance_m": "lead",
}


def one(c: dict) -> str:
    """1 つの条件を「特徴量 演算 閾値」の形にする。

    OPS の雛形は 'x' を特徴量の位置に置いてある。**部分一致で置換しない**
    (abs_gt の '|x|' が '||' になる)。
    """
    tmpl = OPS.get(c["op"], c["op"])
    return tmpl.replace("x", c["feature"], 1) + f" {c['threshold']:g}"


def gate_text(spec: dict) -> str:
    g = spec.get("gate") or []
    return " かつ ".join(one(c) for c in g) if g else "-"


def cond_text(name: str, spec: dict) -> str:
    op = spec["op"]
    if op in ("lt", "gt", "abs_gt"):
        return one(spec)
    if op == "rising":
        return f"{spec['feature']} が {spec['threshold']:g} を上向きに跨ぐ"
    if op == "agreement":
        parts = [one(c) for c in spec.get("conditions", [])]
        tol = spec.get("tolerance_s")
        s = " AND ".join(parts)
        return s + (f"  (時間ずれ許容 ±{tol:g} s)" if tol else "")
    if op == "cooccurrence":
        req = " OR ".join(spec.get("require_any", []))
        return (f"{spec['base']} と同時に [{req}] が "
                f"{spec.get('window_s', 0):g} s 以内に {spec.get('min_count', 1)} 件以上")
    if op == "sequence":
        return (f"{spec['first']} の {spec.get('min_gap_s', 0):g}〜"
                f"{spec.get('max_gap_s', 0):g} s 後に {spec['then']}")
    return op


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detection", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--markdown", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.detection)
    ev, sc = cfg["events"], cfg["scoring"]

    rows = []
    for name, spec in ev.items():
        rows.append({
            "イベント": name,
            "ラベル": spec.get("label", ""),
            "判定量": spec.get("feature") or spec.get("base") or
                      spec.get("first", "") or "(複合)",
            "条件": cond_text(name, spec),
            "継続": (f"{spec['min_duration_s']:g} s"
                     if spec.get("min_duration_s") is not None else "-"),
            "結合": (f"{spec['merge_gap_s']:g} s"
                     if spec.get("merge_gap_s") is not None else "-"),
            "ゲート": gate_text(spec),
            "重み": spec.get("weight", 1.0),
            "段": spec.get("stage", 1),
            "派生節": DERIVED.get(spec.get("feature") or spec.get("base", ""), ""),
        })

    if args.markdown:
        cols = list(rows[0])
        print("| " + " | ".join(cols) + " |")
        print("|" + "|".join("---" for _ in cols) + "|")
        for r in rows:
            print("| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |")
    else:
        for r in rows:
            print(f"■ {r['イベント']}  ({r['ラベル']})")
            print(f"    判定量  : {r['判定量']}"
                  + (f"   ← {r['派生節']} 節で生成" if r["派生節"] else ""))
            print(f"    条件    : {r['条件']}")
            print(f"    継続時間: {r['継続']}    結合間隔: {r['結合']}")
            print(f"    ゲート  : {r['ゲート']}")
            print(f"    重み {r['重み']} / stage {r['段']}")
            print()

    print("=" * 70)
    print("候補化 (build_candidates) — 全イベント共通")
    print(f"    前後の余白 window_pad_s : {sc['window_pad_s']:g} s")
    print(f"    候補の結合 merge_gap_s  : {sc['merge_gap_s']:g} s")
    print(f"    1 イベントの severity 上限: {sc['max_severity_per_event']:g}")
    print(f"    共起ボーナス            : {sc['cooccurrence_bonus']:g}")
    print("\n前処理 (L2) — 全イベント共通")
    r, sm = cfg["resample"], cfg["smoothing"]
    print(f"    再サンプル {r['rate_hz']:g} Hz / 欠測とみなす受信間隔 "
          f"{r['max_gap_s']:g} s / 端の切り落とし {r['edge_trim_s']:g} s")
    print("    平滑化 (対称移動平均): " + " / ".join(f"{k} {v:g} s" for k, v in sm.items()))
    print("\n派生量の設定節")
    for sec in ("lead", "lane_change", "physics", "s_evasion", "panic_brake",
                "wheel_speed", "closing", "cut_in", "weaving", "turn"):
        if sec in cfg:
            print(f"  [{sec}] " + "  ".join(
                f"{k}={v}" for k, v in cfg[sec].items()
                if not isinstance(v, (dict, list))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
