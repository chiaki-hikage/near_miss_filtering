#!/usr/bin/env python3
"""路面凹凸の確認タスクの採点 (VLM 路面 手順 3)。GPU も vllm も要らない。

結果 JSONL を manifest.csv (イベントか対照か・候補か・以前の目視判定) と突き合わせる。
「凹凸あり」は feature_present = yes かつ near_target ≠ no とする。

**割合だけでなく件数を必ず出す。** 対照 (振動イベントの無い時刻) でも凹凸ありと
答える割合が高ければ、イベントで高くても識別できているとは言えない。
主に見るのは、昼間のイベントと昼間の対照の差 (lift = イベントの割合 / 対照の割合)。

  uv run python scripts/vlm/score_road_review.py
  uv run python scripts/vlm/score_road_review.py --results out/vibration/vlm_road/results_*_mode_road.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from near_miss.vlm import road


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=Path("out/vibration/vlm_road"))
    p.add_argument("--results", nargs="*", default=None, help="既定 <dir>/results_*_mode_road.jsonl")
    p.add_argument("--config", type=Path, default=Path("configs/vlm_road.yaml"))
    p.add_argument("--out", type=Path, default=None, help="目標ごとの表 (既定 <dir>/scored_road.csv)")
    return p.parse_args()


def load_results(paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            resp = r.get("response") or {}
            rows.append(dict(target_id=r["event_id"], model=r["model"], condition=r["condition"], rep=r["rep"],
                             schema_ok=not r["schema_errors"], **{k: resp.get(k) for k in road.build_schema()["properties"]}))
    return pd.DataFrame(rows)


def frac(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "0/0"


def video_category(s: object) -> str:
    """以前の目視判定 (Claude による静止画 2 枚の暫定判定) の区分。地点単位。"""
    if not isinstance(s, str) or s == "未確認":
        return "未確認"
    for key in ("一致", "候補", "不明瞭", "判別不能"):
        if s.startswith(key):
            return key
    return "その他"


def summarize(d: pd.DataFrame, title: str) -> list[str]:
    L = [f"\n== {title}"]
    L.append(f"schema 適合 {frac(int(d.schema_ok.sum()), len(d))}")
    groups = [("対照 (振動イベント無し)", d.kind == "control"),
              ("振動イベント (候補以外)", (d.kind == "event") & ~d.is_candidate),
              ("異常振動候補 (5 条件)", (d.kind == "event") & d.is_candidate)]
    L.append("区分 | 昼間: 凹凸あり | 昼間: 路面が見えない | 夜間: 凹凸あり | 夜間: 路面が見えない")
    day_rates = {}
    for name, m in groups:
        cells = []
        for is_day in (True, False):
            g = d[m & (d.daytime == is_day)]
            cells += [frac(int(g.positive.sum()), len(g)), frac(int((g.visibility == "not_visible").sum()), len(g))]
            if is_day:
                day_rates[name] = (g.positive.mean() if len(g) else float("nan"), len(g))
        L.append(f"{name} | " + " | ".join(cells))
    c = day_rates["対照 (振動イベント無し)"][0]
    for name in ("振動イベント (候補以外)", "異常振動候補 (5 条件)"):
        r = day_rates[name][0]
        L.append(f"lift (昼間) {name} / 対照 = {r / c:.2f}" if c and c > 0 else f"lift (昼間) {name}: 対照が 0 件")

    ev = d[(d.kind == "event") & (d.daytime == True)]  # noqa: E712
    if len(ev):
        L.append("\n以前の目視判定 (Claude・静止画 2 枚・暫定、地点単位) との対応 (昼間のイベント)")
        L.append("以前の判定 | 件数 | VLM 凹凸あり")
        for cat, g in ev.groupby(ev.video_road_feature.map(video_category)):
            L.append(f"{cat} | {len(g)} | {frac(int(g.positive.sum()), len(g))}")
        if "physics_signals" in ev:
            ph = ev[ev.physics_signals.notna()]
            if len(ph):
                L.append("\n物理シグナルの数 (地点単位、映像に依存しない 5 つ) ごと (昼間のイベント)")
                L.append("シグナル | 件数 | VLM 凹凸あり")
                bins = pd.cut(ph.physics_signals, [-1, 2, 3, 4, 5], labels=["0〜2", "3", "4", "5"])
                for b, g in ph.groupby(bins, observed=True):
                    L.append(f"{b} | {len(g)} | {frac(int(g.positive.sum()), len(g))}")
        side = ev[ev.positive & ev.wheel_path.isin(["left", "right"]) & (ev.wheel_lr_log2.abs() >= 1.0)]
        if len(side):
            want = side.wheel_lr_log2.map(lambda x: "left" if x > 0 else "right")
            L.append(f"\n左右の一致 (VLM が片側と答え、輪速の左右差が 2 倍以上のもの): "
                     f"{frac(int((side.wheel_path == want).sum()), len(side))}")
    return L


def main() -> int:
    args = parse_args()
    cfg = road.load_config(args.config)
    paths = args.results or sorted(glob.glob(str(args.dir / "results_*_mode_road.jsonl")))
    if not paths:
        raise SystemExit(f"結果がありません: {args.dir}/results_*_mode_road.jsonl")
    res = load_results(paths)
    man = pd.read_csv(args.dir / "manifest.csv")
    man["is_candidate"] = man.is_candidate.eq(True)     # 対照は空欄
    ph_path = Path(cfg["road"]["physics_sites"])
    if ph_path.is_file():
        ph = pd.read_csv(ph_path)[["site_id", "physics_signals"]]
        man = man.merge(ph, on="site_id", how="left")
    d = res.merge(man, on="target_id", how="left", validate="many_to_one")
    if d.kind.isna().any():
        raise SystemExit(f"manifest に無い target_id が {int(d.kind.isna().sum())} 件")
    d["positive"] = (d.feature_present == "yes") & (d.near_target != "no")

    L = [f"結果 {len(paths)} ファイル / {len(res)} 件 / 目標 {man.target_id.nunique()} 件"
         f" (イベント {int((man.kind == 'event').sum())}・対照 {int((man.kind == 'control').sum())})",
         "凹凸あり = feature_present が yes かつ near_target が no でない"]
    for (model, cond), g in d.groupby(["model", "condition"]):
        L += summarize(g, f"{model} / 条件 {cond} ({cfg['road']['conditions'].get(cond, {}).get('note', '')})")
    for model, g in d.groupby("model"):
        w = g.pivot_table(index="target_id", columns="condition", values="positive", aggfunc="first")
        if {"R", "O"} <= set(w.columns):
            both = int((w.R & w.O).sum())
            L.append(f"\n{model}: 条件 R と O の両方で凹凸あり {both} / R だけ {int((w.R & ~w.O).sum())}"
                     f" / O だけ {int((~w.R & w.O).sum())} / どちらも無し {int((~w.R & ~w.O).sum())}")
    print("\n".join(L))
    out = args.out or (args.dir / "scored_road.csv")
    d.to_csv(out, index=False)
    print(f"\n目標ごとの表 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
