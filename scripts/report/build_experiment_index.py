#!/usr/bin/env python3
"""out/ の解析結果の索引 experiments/index.csv を作る・更新する。

out/ のファイルは動かさない。読むのはファイルの一覧・大きさ・更新日時と、
docs/・configs/・tests/・README.md の中での参照だけ。

  uv run python scripts/report/build_experiment_index.py            # 更新 (手で書いた列は残す)
  uv run python scripts/report/build_experiment_index.py --check    # 書き出さずに差分の件数だけ表示

索引の単位
  - out/ 直下のフォルダ 1 つ = 1 行 (out/crosswind/・out/vibration/ はテーマのまとまりなので 1 段下を 1 行)
  - out/ 直下のばらのファイルは、最初の "." より前の名前でまとめて 1 行 (例 out/chain_ccs_v2.*)

列は 2 種類ある。
  自動の列   スクリプトが毎回作り直す (AUTO_COLS)
  手で書く列 作成スクリプト・採用状況とその根拠 (MANUAL_COLS)。既存の index.csv から path で引き継ぎ、
             スクリプトは書き換えない。新しい行は空欄で足す。消えた行は status を「out に無い」にして残す
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "out"
INDEX = REPO / "experiments" / "index.csv"

# テーマのまとまりで、1 段下を索引の単位にするフォルダ
THEME_DIRS = ("crosswind", "vibration")
# 索引の作業で置いた目印のファイル。更新日の計算から外す
IGNORE_FILES = {"MOVED.md", "SCRIPTS_MOVED.md", ".DS_Store"}
# 参照を探す場所 (出力を読む・説明する側)
REF_DIRS = ("docs", "configs", "tests")
REF_FILES = ("README.md",)
REF_SUFFIX = (".md", ".yaml", ".py")

AUTO_COLS = ["path", "theme", "n_files", "size_mb", "mtime_first", "mtime_last",
             "has_readme", "meta_files", "newer_by_name", "referenced_by"]
MANUAL_COLS = ["producer_script", "producer_basis", "status", "status_basis", "note"]
COLS = AUTO_COLS + MANUAL_COLS

# 名前の先頭 → テーマ。上から順に最初に一致したもの
THEMES = [
    (r"^crosswind/", "横風"),
    (r"^vibration/", "路面振動"),
    (r"^(chunk\d|demo/|example\d)", "L3 検出 (comma2k19)"),
    (r"^screen_", "L3 スクリーニング (commaCarSegments)"),
    (r"^(l1_|raw_can_filter)", "L1 フィルタ"),
    (r"^(sideslip_|slip_flags_|validate_slip_flags|review_direct_slip|slip_plots|ay_divergence|review_lateral|kit_msdm|demo_sideslip)", "スリップ・横滑り"),
    (r"^near_miss_", "出力形式・API"),
    (r"^comma1m", "comma1M"),
    (r"^jerk_", "jerk"),
    (r"^(chain_|final_|window_stats|feature_stats|profile_|priority_|platform_survey|benchmark_)", "集計・報告"),
    (r"^\d{4}-\d{2}-\d{2}--", "個別セグメントの図"),
    (r"^_backup", "バックアップ"),
]


def units() -> dict[str, list[Path]]:
    u: dict[str, list[Path]] = {}
    for p in sorted(OUT.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_dir() and p.name in THEME_DIRS:
            for c in sorted(p.iterdir()):
                if not c.name.startswith("."):
                    u[f"out/{p.name}/{c.name}" + ("/" if c.is_dir() else "")] = [c]
        elif p.is_dir():
            u[f"out/{p.name}/"] = [p]
        else:
            u.setdefault(f"out/{p.name.split('.')[0]}.*", []).append(p)
    return u


def base_of(key: str) -> str:
    """索引の path から、参照を探すときの文字列 (末尾の / と .* を外したもの)。"""
    return key.rstrip("/").removesuffix(".*")


def version_split(name: str) -> tuple[str, int]:
    m = re.match(r"^(.*)_v(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, 1)


def newer_by_name(key: str, keys: list[str]) -> str:
    """名前だけで見た後続の版 (同じ名前で _vN の N が大きいもの)。中身の比較はしていない。"""
    stem, v = version_split(base_of(key))
    newer = []
    for k in keys:
        s, w = version_split(base_of(k))
        if s == stem and w > v:
            newer.append((w, k))
    return ";".join(k for _, k in sorted(newer))


def ref_texts() -> dict[str, str]:
    t = {}
    for d in REF_DIRS:
        for f in (REPO / d).rglob("*"):
            if f.suffix in REF_SUFFIX and "__pycache__" not in f.parts and f.is_file():
                t[str(f.relative_to(REPO))] = f.read_text(encoding="utf-8", errors="replace")
    for f in REF_FILES:
        t[f] = (REPO / f).read_text(encoding="utf-8", errors="replace")
    return t


def auto_row(key: str, paths: list[Path], keys: list[str], texts: dict[str, str]) -> dict:
    files = []
    for p in paths:
        files += [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    files = [f for f in files if f.name not in IGNORE_FILES]
    mt = [f.stat().st_mtime for f in files]
    day = lambda t: dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d")
    rel = key[len("out/"):]
    theme = next((name for pat, name in THEMES if re.search(pat, rel)), "")
    pat = re.compile(re.escape(base_of(key)) + r"(?![A-Za-z0-9_])")
    refs = sorted(k for k, t in texts.items() if pat.search(t) and k != "docs/project_structure.md")
    return {
        "path": key,
        "theme": theme,
        "n_files": len(files),
        "size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 2),
        "mtime_first": day(min(mt)) if mt else "",
        "mtime_last": day(max(mt)) if mt else "",
        "has_readme": any((p / "README.md").exists() for p in paths if p.is_dir()),
        "meta_files": ";".join(sorted(str(f.relative_to(REPO)) for p in paths if p.is_dir()
                                      for f in p.glob("*meta*.json"))),
        "newer_by_name": newer_by_name(key, keys),
        "referenced_by": ";".join(refs),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--merge-from", type=Path, default=None,
                    help="手で書く列を、この CSV から path で取り込む (空欄でない値だけ。既存の値より優先)")
    ap.add_argument("--check", action="store_true", help="書き出さない")
    args = ap.parse_args()

    old = {}
    if args.index.exists():
        with args.index.open(encoding="utf-8", newline="") as f:
            old = {r["path"]: r for r in csv.DictReader(f)}
    extra = {}
    if args.merge_from:
        with args.merge_from.open(encoding="utf-8", newline="") as f:
            extra = {r["path"]: r for r in csv.DictReader(f)}

    u = units()
    keys = list(u)
    texts = ref_texts()
    rows = []
    for k in keys:
        r = auto_row(k, u[k], keys, texts)
        for c in MANUAL_COLS:
            r[c] = (old.get(k, {}).get(c) or "")
            if extra.get(k, {}).get(c):
                r[c] = extra[k][c]
        rows.append(r)
    gone = [p for p in old if p not in u]
    for p in gone:     # out から消えた行は消さずに残す
        r = dict(old[p])
        r["status"] = "out に無い"
        rows.append(r)
    unknown = [k for k in extra if k not in u]

    n_new = sum(1 for k in keys if k not in old)
    print(f"{len(rows)} 行 (新規 {n_new}、out に無い {len(gone)})")
    if unknown:
        print("取り込み元にあって out に無い path:", ", ".join(unknown), file=sys.stderr)
    if args.check:
        return 0
    args.index.parent.mkdir(parents=True, exist_ok=True)
    with args.index.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"-> {args.index.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
