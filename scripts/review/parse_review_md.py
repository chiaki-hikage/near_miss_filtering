#!/usr/bin/env python3
"""記入済みの確認シート (Markdown) を読み、対応する CSV の 判定/備考 を埋める。

`make_review_list.py` が出す .md は 1 候補 1 節で、節の末尾に

    | 判定 | 備考 |
    |---|---|
    | ◯ | ... |

の表がある。**節の見出しの番号 (rank) で CSV と突き合わせる。**
行の順番に頼ると、記入時に節を入れ替えられていたときに黙ってずれる。

  uv run python scripts/review/parse_review_md.py out/chunk1/review_r6.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

HEAD = re.compile(r"^##\s+(\d+)\.\s*(.*)$")
# 記号の表記ゆれを吸収する。全角/半角、丸の種類が揺れやすい。
MARK = {
    "◯": "ヒヤリハット", "○": "ヒヤリハット", "〇": "ヒヤリハット",
    "o": "ヒヤリハット", "O": "ヒヤリハット", "◎": "ヒヤリハット",
    "×": "通常運転", "✗": "通常運転", "x": "通常運転", "X": "通常運転",
    "✕": "通常運転", "✖": "通常運転",
    "△": "境界", "▲": "境界",
    "?": "判断不能", "？": "判断不能", "-": "", "—": "",
}


def parse(md: str) -> dict[int, tuple[str, str]]:
    """rank -> (判定, 備考)。表の行は見出しの後に現れる最初のものだけ採る。"""
    out: dict[int, tuple[str, str]] = {}
    rank = None
    seen_header = False
    for line in md.splitlines():
        m = HEAD.match(line)
        if m:
            rank, seen_header = int(m.group(1)), False
            continue
        if rank is None or not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells:
            continue
        if cells[0] in ("判定", "---", ":---"):
            seen_header = True
            continue
        if set(cells[0]) <= set("-: ") or not seen_header:
            continue
        if rank in out:            # 節に表が複数あっても最初だけ
            continue
        mark = cells[0]
        verdict = MARK.get(mark, mark)
        note = cells[1] if len(cells) > 1 else ""
        if verdict:
            out[rank] = (verdict, note)
        rank = None                # 1 節 1 判定
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("md", type=Path)
    p.add_argument("--csv", type=Path, default=None,
                   help="書き込む CSV (既定は同名の .csv)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    csv = args.csv or args.md.with_suffix(".csv")
    d = pd.read_csv(csv, low_memory=False)
    got = parse(args.md.read_text(encoding="utf-8"))
    print(f"{args.md}: {len(got)} 件の判定を読みました")

    miss = sorted(set(d["rank"]) - set(got))
    extra = sorted(set(got) - set(d["rank"]))
    if extra:
        raise SystemExit(f"CSV に無い rank が md にあります: {extra}")
    if miss:
        print(f"  未記入 {len(miss)} 件: rank {miss[:10]}"
              + (" ..." if len(miss) > 10 else ""))

    d["判定"] = d["rank"].map(lambda r: got.get(r, ("", ""))[0])
    d["備考"] = d["rank"].map(lambda r: got.get(r, ("", ""))[1])
    vc = d.loc[d["判定"] != "", "判定"].value_counts()
    print("  " + "  ".join(f"{k} {v}" for k, v in vc.items()))

    if args.dry_run:
        print("\n--dry-run のため書き込みません。")
        return 0
    d.to_csv(csv, index=False)
    print(f"書き込み: {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
