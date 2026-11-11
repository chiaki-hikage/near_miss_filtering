"""週次報告の原稿の読み込み、変更点の抽出、PPT の生成。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/report"))
import weekly_changes as WC  # noqa: E402

CFG = yaml.safe_load((REPO / "configs/report_weekly.yaml").read_text(encoding="utf-8"))

NOTES = """---
week: 2026-W01
period_start: 2026-01-05
period_end: 2026-01-11
base_commit: abc1234
---

<!-- コメントは読まない -->

## 今週の要点

- 要点 1
- 要点 2

## 進捗

| テーマ | 状況 | 今週の変化 |
|---|---|---|
| A | 進行中 | 変化 1 |

## 主な実験結果

### 結果その 1

- 何かが分かった
- 根拠: `out/x/`, `configs/y.yaml`

### 結果その 2

- 別のこと

## 課題

- 課題 1

## 次週の予定

- (案) 予定 1
"""


def test_parse_notes_sections_tables_results():
    n = WC.parse_notes(NOTES)
    assert n.meta["week"] == "2026-W01" and n.meta["base_commit"] == "abc1234"
    assert n.sections["今週の要点"] == ["要点 1", "要点 2"]
    assert n.tables["進捗"] == [["テーマ", "状況", "今週の変化"], ["A", "進行中", "変化 1"]]
    assert [r.title for r in n.results] == ["結果その 1", "結果その 2"]
    assert n.results[0].lines == ["何かが分かった"]
    assert n.results[0].evidence == ["out/x/", "configs/y.yaml"]
    assert n.sections["課題"] == ["課題 1"] and n.sections["次週の予定"] == ["(案) 予定 1"]


def test_area_of_uses_first_matching_prefix():
    areas = CFG["changes"]["areas"]
    assert WC.area_of("scripts/report/x.py", areas) == "スクリプト"
    assert WC.area_of("pyproject.toml", areas) == "その他"


def test_index_changes_ignores_columns_missing_in_old_index():
    old = {"out/a/": {"path": "out/a/", "decision": "未判断"}, "out/b/": {"path": "out/b/"}}
    new = {"out/a/": {"path": "out/a/", "decision": "採用", "verification": "未確認"},
           "out/c/": {"path": "out/c/"}}
    c = WC.index_changes(old, new, ["decision", "verification"])
    assert c["added"] == ["out/c/"] and c["removed"] == ["out/b/"]
    assert c["changed"] == [("out/a/", "decision", "未判断", "採用")]
    assert c["new_cols"] == ["verification"]


def test_build_makes_three_to_five_slides():
    pytest.importorskip("pptx")
    import build_weekly_pptx as B
    n = WC.parse_notes(NOTES)
    changes = dict(base="abc1234", head="def5678", commits=[dict(hash="def5678", date="2026-01-06", subject="x")],
                   by_area={"文書": [("M", "docs/a.md")]},
                   index=dict(added=[], removed=[], changed=[], new_cols=[]),
                   updated_outputs=[], evidence=[], dirty=False)
    prs = B.build(n, changes, CFG, "reports/weekly/2026-W01/notes.md")
    assert 3 <= len(prs.slides) <= CFG["limits"]["max_slides"]
    texts = [sh.text_frame.text for s in prs.slides for sh in s.shapes if sh.has_text_frame]
    assert any("要点 1" in t for t in texts)
    assert any("configs/y.yaml" in t for t in texts)     # 根拠のパスがスライドに出る
