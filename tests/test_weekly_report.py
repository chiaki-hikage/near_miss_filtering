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


def _outline(img: Path, n: int = 5) -> dict:
    slides = []
    for k in range(n):
        if k % 2 == 0:
            v = {"table": [["テーマ", "状況"], ["A", "進行中"]]}
        else:
            v = {"image": str(img)}
        slides.append({"title": f"見出し {k}", "message": f"言いたいこと {k}", "visual": v,
                       "notes": ["読み方"], "evidence": ["configs/report_weekly.yaml"]})
    return {"week": "2026-W01", "period_start": "2026-01-05", "period_end": "2026-01-11", "slides": slides}


def _png(tmp_path: Path) -> Path:
    pytest.importorskip("pptx")
    from PIL import Image
    p = tmp_path / "fig.png"
    Image.new("RGB", (400, 200), "white").save(p)
    return p


def test_check_outline_enforces_rules(tmp_path):
    import build_weekly_pptx as B
    img = _png(tmp_path)
    assert B.check_outline(_outline(img), CFG)[0] == []
    err, _ = B.check_outline(_outline(img, n=4), CFG)                      # 枚数
    assert any("4 枚" in e for e in err)
    o = _outline(img)
    o["slides"][1]["visual"] = {"image": str(img), "table": [["a"]]}         # 図と表を両方
    o["slides"][2]["message"] = ""                                          # 主文なし
    o["slides"][3]["evidence"] = ["out/no_such_dir/"]                       # 根拠が無い
    err, _ = B.check_outline(o, CFG)
    assert any("2 枚目" in e and "ちょうど 1 つ" in e for e in err)
    assert any("3 枚目" in e and "主文" in e for e in err)
    assert any("4 枚目" in e and "根拠" in e for e in err)


def test_build_follows_outline(tmp_path):
    import build_weekly_pptx as B
    img = _png(tmp_path)
    prs = B.build(_outline(img, n=6), CFG, "reports/weekly/2026-W01/outline.yaml")
    assert len(prs.slides) == 6
    for k, s in enumerate(prs.slides):
        texts = [sh.text_frame.text for sh in s.shapes if sh.has_text_frame]
        assert f"見出し {k}" in texts and f"言いたいこと {k}" in texts       # タイトルと主文
        n_visual = sum(1 for sh in s.shapes if sh.has_table or sh.shape_type == 13)   # 13 = 図
        assert n_visual == 1                                                # 図か表がちょうど 1 つ
        assert any("configs/report_weekly.yaml" in t for t in texts)          # 根拠
