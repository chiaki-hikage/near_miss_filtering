#!/usr/bin/env python3
"""記入済みの確認シートを labels.csv に取り込む。

`make_review_list.py` が出した CSV の `判定` / `備考` 列を読み、
既存の `labels.csv` に**追記**する。

    判定 = ヒヤリハット / 通常運転 / 判断不能

**上書きしない。** 既に同じ区間のラベルがあれば飛ばして報告する。
突合は drive_id + 区間の重なりで行う (t_start の完全一致は設定が変わると崩れる)。

**判断不能は risky を立てず、labels.csv にも入れない。** 曖昧なものを
False として混ぜると、negative が汚れて精度の評価がずれる。件数だけ報告する。

  # 記入内容の確認だけ
  uv run python scripts/import_review_sheet.py out/chunk1/review_r6.csv --dry-run

  # 取り込み
  uv run python scripts/import_review_sheet.py out/chunk1/review_r6.csv --round 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

# 表記ゆれを吸収する。**risky の真偽はここだけで決める。**
VERDICT = {
    "ヒヤリハット": (True, "ヒヤリハット"),
    "リスク高": (True, "ヒヤリハット"),
    "リスク高い": (True, "ヒヤリハット"),
    "リスクあり": (True, "ヒヤリハット"),
    "危険": (True, "ヒヤリハット"),
    "通常運転": (False, "通常運転"),
    "通常": (False, "通常運転"),
    "問題なし": (False, "通常運転"),
    "若干リスクあり": (False, "通常運転"),
    "ややリスクあり": (False, "通常運転"),
    # 境界は risky を立てないが、verdict に残して後から見直せるようにする。
    # 「危なくない」と「判断できない」は別物なので、判断不能と混ぜない。
    "境界": (False, "境界"),
    "判断不能": (None, "判断不能"),
    "不明": (None, "判断不能"),
}

COLUMNS = ["round", "rank_at_labeling", "drive_id", "segment", "t_start", "t_end",
           "t_in_segment_s", "event_types_at_labeling", "verdict", "risky", "note"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sheet", type=Path, help="記入済みの review_*.csv")
    p.add_argument("--labels", type=Path, default=Path("out/chunk1/labels.csv"))
    p.add_argument("--round", type=int, default=None,
                   help="round 番号。省略すると labels.csv の最大 + 1")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sheet = pd.read_csv(args.sheet, low_memory=False)
    if "判定" not in sheet.columns:
        raise SystemExit(f"{args.sheet} に 判定 列がありません")

    filled = sheet[sheet["判定"].notna() & (sheet["判定"].astype(str).str.strip() != "")]
    print(f"{args.sheet}: {len(sheet)} 行中 {len(filled)} 件が記入済み")
    if filled.empty:
        print("  記入がありません。シートに 判定 を書いてから実行してください。")
        return 1

    unknown = sorted(set(filled["判定"].astype(str).str.strip()) - set(VERDICT))
    if unknown:
        raise SystemExit(
            f"判定の表記が解釈できません: {unknown}\n"
            f"  使える語: {', '.join(sorted(VERDICT))}")

    old = (pd.read_csv(args.labels) if args.labels.exists()
           else pd.DataFrame(columns=COLUMNS))
    rnd = args.round if args.round is not None else (
        int(old["round"].max()) + 1 if len(old) else 1)

    rows, skipped, undecided = [], [], 0
    for _, r in filled.iterrows():
        risky, verdict = VERDICT[str(r["判定"]).strip()]
        if risky is None:
            undecided += 1
            continue
        # 区間の重なりで既存ラベルを探す。**同じ場面を二重に入れない。**
        dup = old[(old.drive_id == r.drive_id)
                  & ~((old.t_end < r.t_start) | (old.t_start > r.t_end))]
        if len(dup):
            skipped.append((r.drive_id, r.segment, r.t_start))
            continue
        rows.append({
            "round": rnd, "rank_at_labeling": int(r.get("rank", 0)),
            "drive_id": r.drive_id, "segment": int(r.segment),
            "t_start": r.t_start, "t_end": r.t_end,
            "t_in_segment_s": r.get("t_in_segment_s"),
            "event_types_at_labeling": r.get("event_types", ""),
            "verdict": verdict, "risky": risky,
            "note": r.get("備考", "") if pd.notna(r.get("備考", "")) else "",
        })

    new = pd.DataFrame(rows, columns=COLUMNS)
    print(f"  取り込み {len(new)} 件 (risky {int(new.risky.sum()) if len(new) else 0})")
    if undecided:
        print(f"  判断不能 {undecided} 件は入れません "
              "(negative に混ぜると精度の評価がずれるため)")
    if skipped:
        print(f"  既存ラベルと重なるため飛ばした {len(skipped)} 件:")
        for d, s, t in skipped[:5]:
            print(f"    {d.split('|')[-1]} seg{int(s)} t={t:.1f}")
        if len(skipped) > 5:
            print(f"    ... 他 {len(skipped) - 5} 件")

    if args.dry_run:
        print("\n--dry-run のため書き込みません。")
        if len(new):
            print(new[["drive_id", "segment", "verdict", "risky"]]
                  .assign(drive_id=lambda d: d.drive_id.str.split("|").str[-1])
                  .to_string(index=False))
        return 0

    if not len(new):
        print("  追記するものがありません。")
        return 0
    out = pd.concat([old, new], ignore_index=True)
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.labels, index=False)
    print(f"\n{args.labels}: {len(old)} -> {len(out)} 件 "
          f"(risky {int(out.risky.sum())})")
    print(out.groupby("round").agg(件数=("risky", "size"),
                                   risky=("risky", "sum")).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
