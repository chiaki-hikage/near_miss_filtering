"""物理モデルのシグナルがあると、前方映像で路面の凹凸が見える割合が上がるかを確かめる。

映像の左右を使う P1 (左右) と P3 (比) は、映像の判定と循環するので使わない。
映像に依存しない次の 5 つだけを「物理シグナル」として数える。

  P2     前輪 → 後輪の輪速の揺れの時間差が L/v と合う
  P4     3〜8 Hz の上下 (とロールかピッチ) の波形を、路面入力のモデルが再現する
  P5     実測とモデルの主周波数がどちらも 11〜12 Hz 付近
  P6     同じ地点の複数通過で、輪速の強い側と側の指標の符号がそろう
  入力型 物理モデルから見た入力が、右片輪・左片輪・左右同時のどれかに決まる

夜間で判別できない地点は除く。地点の数が少ないので、片側の Fisher の正確確率検定の p 値は目安である。

    uv run python scripts/vibration/eval_physics_vs_video.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd
from scipy.stats import fisher_exact

REPO = _bootstrap.SRC.parent
TYPED = ("右片輪入力らしい", "左片輪入力らしい", "左右同時入力らしい")


def load(dirs: list[Path], near_miss: Path, strong_score: float) -> pd.DataFrame:
    parts = [pd.read_csv(d / "site_physics_validation.csv") for d in dirs]
    d = parts[0]
    for x in parts[1:]:
        d = pd.concat([d, x[~x.site_id.isin(d.site_id)]], ignore_index=True)
    nm = pd.read_csv(near_miss).set_index("site_id")
    d = d[d.video_status != "判別不能"].copy()
    d["visible"] = d.video_status == "一致"
    d["visible_or_candidate"] = d.video_status.isin(["一致", "候補"])
    d["near_miss_score"] = d.site_id.map(nm.near_miss_candidate_score)
    d["strong"] = d.near_miss_score >= strong_score
    d["five_conditions"] = d.site_id.map(nm.site_events_all_five).fillna(0) > 0
    for k in ("P2", "P4", "P5", "P6"):
        d[f"{k}_true"] = d[k] == "true"
    d["input_typed"] = d.inferred_side.isin(TYPED)
    d["physics_signals"] = d[["P2_true", "P4_true", "P5_true", "P6_true", "input_typed"]].sum(axis=1)
    return d


def compare(d: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    a, b = d[mask], d[~mask]
    t = [[int(a.visible.sum()), int((~a.visible).sum())], [int(b.visible.sum()), int((~b.visible).sum())]]
    _, p = fisher_exact(t, alternative="greater")
    return dict(signal=name, with_n=len(a), with_visible=int(a.visible.sum()), with_rate=round(a.visible.mean(), 2),
                without_n=len(b), without_visible=int(b.visible.sum()), without_rate=round(b.visible.mean(), 2),
                p_one_sided=round(p, 3))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--physics-dirs", nargs="+", type=Path,
                    default=[REPO / "out/vibration/physics_validation_near_miss", REPO / "out/vibration/physics_validation"])
    ap.add_argument("--near-miss", type=Path, default=REPO / "out/vibration/near_miss/near_miss_vibration_sites.csv")
    ap.add_argument("--strong-score", type=float, default=57.0, help="異常振動ヒヤリ候補の上位 20 地点に相当する点数")
    ap.add_argument("--out", type=Path, default=REPO / "out/vibration/near_miss")
    args = ap.parse_args()
    d = load(args.physics_dirs, args.near_miss, args.strong_score)
    rows = [compare(d, d.strong, "振動が強い (上位 20 地点相当)"),
            compare(d, d.five_conditions, "5 条件を満たすイベントがある"),
            compare(d, d.P2_true, "P2 前→後輪の時間差"),
            compare(d, d.P4_true, "P4 3〜8 Hz の波形"),
            compare(d, d.P5_true, "P5 11〜12 Hz"),
            compare(d, d.P6_true, "P6 複数通過で同じ側"),
            compare(d, d.input_typed, "入力の型が決まる"),
            compare(d, d.physics_signals >= 4, "物理シグナル 4 つ以上"),
            compare(d, d.physics_signals >= 5, "物理シグナル 5 つ")]
    res = pd.DataFrame(rows)
    by_n = d.groupby("physics_signals").agg(sites=("visible", "size"), visible=("visible", "sum"),
                                            visible_or_candidate=("visible_or_candidate", "sum")).reset_index()
    res.to_csv(args.out / "physics_vs_video.csv", index=False)
    d[["site_id", "video_status", "video_label", "strong", "five_conditions", "P2", "P4", "P5", "P6", "inferred_side",
       "physics_signals", "near_miss_score"]].sort_values(["physics_signals", "site_id"], ascending=[False, True]).to_csv(
        args.out / "physics_vs_video_sites.csv", index=False)
    print(f"昼間の映像がある地点 {len(d)}、うち凹凸が見える {int(d.visible.sum())} ({d.visible.mean():.0%})")
    print(res.to_string(index=False))
    print(by_n.to_string(index=False))
    (args.out / "physics_vs_video.json").write_text(json.dumps(
        dict(sites=len(d), visible=int(d.visible.sum()), by_physics_signals=by_n.to_dict("records")), ensure_ascii=False, indent=2, default=int))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
