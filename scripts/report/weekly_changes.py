#!/usr/bin/env python3
"""週次報告の「前回報告からの変更点」を、Git の履歴と実験索引から抜き出す。

  uv run python scripts/report/weekly_changes.py reports/weekly/2026-W46/notes.md
      -> reports/weekly/2026-W46/changes.md

notes.md の先頭の YAML (front matter) の base_commit (前回報告の時点のコミット) から
HEAD までを比べる。出すもの:
  1. コミットの一覧
  2. 変更したファイルを区分ごとに (区分は configs/report_weekly.yaml の changes.areas)
  3. experiments/index.csv の変化 (行の増減と、手で書く列の変化)
  4. 期間中に更新された out/ の結果 (索引の mtime_last が期間内のもの)
  5. notes.md の「根拠」に書いたパスの確認 (あるか・git 管理か・索引の行か・最後のコミット)

ここでは何も判断しない。notes.md に書いた内容と食い違っていないかを人が確かめるための材料。
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "report_weekly.yaml"


# ---------------------------------------------------------------------------
# notes.md の読み込み
# ---------------------------------------------------------------------------

@dataclass
class Result:
    title: str
    lines: list[str] = field(default_factory=list)       # 「結果」などの本文
    evidence: list[str] = field(default_factory=list)    # 根拠のパス


@dataclass
class Notes:
    meta: dict
    sections: dict[str, list[str]]       # 見出し → 箇条書きの本文 (先頭の "- " を外したもの)
    tables: dict[str, list[list[str]]]   # 見出し → 表 (1 行目が見出し行)
    results: list[Result]


def split_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.index("\n---\n", 4)
    return yaml.safe_load(text[4:end]) or {}, text[end + 5:]


EVIDENCE_RE = re.compile(r"`([^`]+)`")


def parse_notes(text: str) -> Notes:
    meta, body = split_front_matter(text)
    sections: dict[str, list[str]] = {}
    tables: dict[str, list[list[str]]] = {}
    results: list[Result] = []
    h2 = None
    cur: Result | None = None
    for raw in body.splitlines():
        line = raw.rstrip()
        if line.startswith("<!--") or not line.strip():
            continue
        if line.startswith("## "):
            h2, cur = line[3:].strip(), None
            sections.setdefault(h2, [])
            continue
        if h2 is None:
            continue
        if line.startswith("### "):
            cur = Result(line[4:].strip())
            results.append(cur)
            continue
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                continue
            tables.setdefault(h2, []).append(cells)
            continue
        item = re.sub(r"^\s*[-*]\s+", "", line)
        if cur is not None:
            m = re.match(r"(根拠|evidence)\s*[:：]\s*(.*)", item)
            if m:
                cur.evidence += EVIDENCE_RE.findall(m.group(2))
            else:
                cur.lines.append(item)
        else:
            sections[h2].append(item)
    return Notes(meta, sections, tables, results)


# ---------------------------------------------------------------------------
# Git と索引
# ---------------------------------------------------------------------------

def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout


def commits(base: str, head: str) -> list[dict]:
    out = git("log", "--reverse", "--date=short", "--format=%h%x09%ad%x09%s", f"{base}..{head}")
    rows = []
    for ln in out.splitlines():
        h, d, s = ln.split("\t", 2)
        rows.append(dict(hash=h, date=d, subject=s))
    return rows


def changed_files(base: str, head: str) -> list[tuple[str, str]]:
    out = git("diff", "--name-status", "--find-renames", base, head)
    rows = []
    for ln in out.splitlines():
        parts = ln.split("\t")
        rows.append((parts[0][0], parts[-1]))   # (A/M/D/R, 新しいパス)
    return rows


def area_of(path: str, areas: list[list[str]]) -> str:
    return next(name for prefix, name in areas if path.startswith(prefix))


def read_index_at(rev: str | None, index_path: str) -> dict[str, dict]:
    """rev の時点の索引。rev が None なら作業ツリーのもの。無ければ空。"""
    try:
        text = (REPO / index_path).read_text(encoding="utf-8") if rev is None else git("show", f"{rev}:{index_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    return {r["path"]: r for r in csv.DictReader(io.StringIO(text))}


def index_changes(old: dict, new: dict, cols: list[str]) -> dict:
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = []
    for p in sorted(set(old) & set(new)):
        for c in cols:
            a, b = old[p].get(c, ""), new[p].get(c, "")
            if c in old[p] and a != b:      # 旧い索引に無い列は比べない (列の追加は変化として数えない)
                changed.append((p, c, a, b))
    new_cols = sorted({c for r in new.values() for c in r} - {c for r in old.values() for c in r}) if old else []
    return dict(added=added, removed=removed, changed=changed, new_cols=new_cols)


def tracked(path: str) -> bool:
    return bool(git("ls-files", "--", path).strip())


def last_commit(path: str) -> str:
    return git("log", "-1", "--date=short", "--format=%h %ad", "--", path).strip()


def evidence_status(path: str, index: dict[str, dict]) -> dict:
    p = path.rstrip("/")
    exists = (REPO / p).exists()
    def hit(k: str) -> bool:
        if k.endswith(".*"):                       # ばらのファイルのまとまり (out/chain_ccs_v2.*)
            return p.startswith(k[:-1])
        return p == k.rstrip("/") or p.startswith(k)   # フォルダ (out/vibration/locate/)
    row = next((k for k in index if hit(k)), None)
    return dict(path=path, exists=exists, tracked=tracked(p) if exists else False,
                index_row=row or "", last_commit=last_commit(p) if exists else "")


# ---------------------------------------------------------------------------
# まとめ
# ---------------------------------------------------------------------------

def collect(notes: Notes, cfg: dict, head: str = "HEAD") -> dict:
    base = str(notes.meta["base_commit"])
    ch = cfg["changes"]
    head_hash = git("rev-parse", "--short", head).strip()
    files = changed_files(base, head)
    by_area: dict[str, list[tuple[str, str]]] = {}
    for st, p in files:
        by_area.setdefault(area_of(p, ch["areas"]), []).append((st, p))
    old_idx = read_index_at(base, ch["index"])
    new_idx = read_index_at(None, ch["index"])
    since, until = str(notes.meta.get("period_start", "")), str(notes.meta.get("period_end", ""))
    updated = [r for r in new_idx.values() if since and until and since <= r.get("mtime_last", "") <= until]
    ev = [evidence_status(p, new_idx) for r in notes.results for p in r.evidence]
    return dict(base=base, head=head_hash, commits=commits(base, head), by_area=by_area,
                index=index_changes(old_idx, new_idx, ch["index_cols"]),
                updated_outputs=sorted(updated, key=lambda r: r["path"]), evidence=ev,
                dirty=bool(git("status", "--porcelain", "--untracked-files=no").strip()))


def to_markdown(c: dict, week: str) -> str:
    L = [f"# {week} 前回報告からの変更点 (自動抽出)", "",
         f"`scripts/report/weekly_changes.py` が作成。比較: `{c['base']}..{c['head']}`"
         + ("。**作業ツリーに未コミットの変更がある状態で作った**" if c["dirty"] else ""), ""]
    L += [f"## コミット ({len(c['commits'])} 件)", ""]
    L += [f"- `{x['hash']}` {x['date']} {x['subject']}" for x in c["commits"]] or ["- なし"]
    L += ["", "## 変更したファイル (区分ごと)", "", "| 区分 | 件数 | 追加 | 変更 | 削除 |", "|---|---|---|---|---|"]
    for a, rows in c["by_area"].items():
        n = lambda s: sum(1 for st, _ in rows if st == s)
        L.append(f"| {a} | {len(rows)} | {n('A')} | {n('M') + n('R')} | {n('D')} |")
    L += [""]
    for a, rows in c["by_area"].items():
        L += [f"<details><summary>{a}</summary>", ""] + [f"- {st} `{p}`" for st, p in rows] + ["", "</details>", ""]
    ix = c["index"]
    L += ["## 実験索引 (experiments/index.csv) の変化", "",
          f"- 追加された行: {len(ix['added'])}", f"- なくなった行: {len(ix['removed'])}",
          f"- 手で書く列が変わった箇所: {len(ix['changed'])}"]
    if ix["new_cols"]:
        L.append(f"- 新しくできた列: {', '.join(ix['new_cols'])} (前回の索引には無かったので、値の変化としては数えていない)")
    L += [f"  - `{p}` {col}: 「{a}」→「{b}」" for p, col, a, b in ix["changed"][:30]]
    L += ["", "## 期間中に更新された out/ の結果 (索引の mtime_last)", ""]
    L += [f"- `{r['path']}` ({r['mtime_last']}) {r.get('current_use', '')} / {r.get('verification', '')}" for r in c["updated_outputs"]] or ["- なし"]
    L += ["", "## 報告に書いた根拠の確認", "", "| パス | ある | git 管理 | 索引の行 | 最後のコミット |", "|---|---|---|---|---|"]
    yn = lambda b: "はい" if b else "**いいえ**"
    L += [f"| `{e['path']}` | {yn(e['exists'])} | {yn(e['tracked'])} | {e['index_row'] or '—'} | {e['last_commit'] or '—'} |" for e in c["evidence"]]
    L += ["", "git 管理でない根拠 (out/ など) は、索引の行で作成スクリプトと状態を追う。"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notes", type=Path)
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--out", type=Path, default=None, help="既定は notes.md と同じフォルダの changes.md")
    args = ap.parse_args()
    notes = parse_notes(args.notes.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    c = collect(notes, cfg, args.head)
    out = args.out or args.notes.parent / "changes.md"
    out.write_text(to_markdown(c, str(notes.meta.get("week", args.notes.parent.name))), encoding="utf-8")
    print(f"コミット {len(c['commits'])} 件、変更ファイル {sum(len(v) for v in c['by_area'].values())} 件、"
          f"根拠 {len(c['evidence'])} 件 (無いもの {sum(not e['exists'] for e in c['evidence'])}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
