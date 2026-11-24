"""人のレビューの結果 (閲覧ページから書き出した review_result.csv) を、伏せていた群・旗 (key.csv) と突き合わせる。

見ること:
  1. 段 C と段 A で、人が「姿勢・向きが乱れる」「予想外」とした割合 (群の識別力)
  2. 挙動の手がかりの数 (0〜3) ごとの、人の「姿勢・向きが乱れる」の割合と、順位の AUC
  3. 手がかりごとの当たり外れ (人の「姿勢・向きが乱れる」を正とみたときの適合率・再現率)
  4. 自動の状況 (旋回中・斜面) と、人の「旋回に伴う」「斜面に沿う」の対応
  5. 信号の品質の注意と、人の「偽の異常の疑い」の対応
「不明」は分母から外し、外した数を並べて出す (無理に分類しない)。件数が少ないので、割合には Wilson の 95% 区間を付ける。

    uv run python scripts/offroad/td2_review_evaluate.py --review out/offroad/tartandrive2/<review_set の ID> \
        --result <review_result.csv> --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [np.nan, np.nan]
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def rate(df: pd.DataFrame, col: str, pos: str) -> dict:
    known = df[col].notna() & (df[col] != "") & (df[col] != "不明")
    k = int((df.loc[known, col] == pos).sum())
    n = int(known.sum())
    return dict(positive=k, known=n, unknown=int((~known).sum()), rate=round(k / n, 3) if n else None, ci95=wilson(k, n))


def auc(score: pd.Series, y: pd.Series) -> float | None:
    s1, s0 = score[y], score[~y]
    if len(s1) == 0 or len(s0) == 0:
        return None
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in s1 for b in s0]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    out = Path("out/offroad/tartandrive2") / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    key = pd.read_csv(a.review / "key.csv")
    lst = pd.read_csv(a.review / "review_list.csv")
    res = pd.read_csv(a.result, encoding="utf-8-sig", dtype=str).fillna("")
    D = key.merge(lst, on="review_id").merge(res, on="review_id", how="left").fillna({c: "" for c in res.columns})
    done = D[D.vehicle_behavior != ""]
    beh = "姿勢・向きが乱れる"
    summ = dict(reviewed=len(done), of=len(D))
    summ["by_group"] = {g: dict(disturbed=rate(d, "vehicle_behavior", beh), unexpected=rate(d, "expected_or_unexpected", "予想外"),
                                safety=rate(d, "safety_relevant", "関わりそう"), cause_visible=rate(d, "cause_visible", "見える"),
                                artifact=rate(d, "artifact_suspected", "あり"))
                        for g, d in done.groupby("group")}
    known = done[(done.vehicle_behavior != "") & (done.vehicle_behavior != "不明")]
    y = known.vehicle_behavior == beh
    summ["by_n_attitude_flags"] = {int(k): rate(d, "vehicle_behavior", beh) for k, d in done.groupby("n_attitude_flags")}
    summ["auc_n_attitude_flags_vs_disturbed"] = auc(known.n_attitude_flags, y)
    summ["auc_mtvv_vs_disturbed"] = auc(known.mtvv_total, y)
    summ["auc_contrast_vs_disturbed"] = auc(known.contrast, y)
    per = {}
    for f in ("flag_yaw", "flag_roll", "flag_wheel_lr", "flag_stroke_extend", "flag_stroke_compress", "flag_brake", "flag_throttle"):
        fl = known[f].astype(str).str.lower() == "true"
        tp = int((fl & y).sum())
        per[f] = dict(flagged=int(fl.sum()), precision=round(tp / fl.sum(), 3) if fl.sum() else None, recall=round(tp / y.sum(), 3) if y.sum() else None)
    summ["per_flag_vs_disturbed"] = per
    summ["auto_context_vs_human"] = pd.crosstab(done.auto_context, done.vehicle_behavior).to_dict()
    done = done.assign(has_quality_note=done.quality_notes != "なし")
    summ["quality_note_vs_artifact"] = pd.crosstab(done.has_quality_note, done.artifact_suspected).to_dict()
    summ["unknown_counts"] = {c: int((done[c] == "不明").sum()) for c in ("cause_visible", "vehicle_behavior", "expected_or_unexpected", "artifact_suspected", "safety_relevant")}
    D.to_csv(out / "merged.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    print(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
