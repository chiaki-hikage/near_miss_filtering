#!/usr/bin/env python3
"""週次報告の原稿 (notes.md) から PPT を作る。PPT は git 管理外 (.gitignore)。

  uv run --extra report python scripts/report/build_weekly_pptx.py reports/weekly/2026-W46/notes.md
      -> reports/weekly/2026-W46/weekly_2026-W46.pptx と changes.md

スライド (3〜5 枚。原稿に無い節のスライドは作らない):
  1. 表紙と今週の要点           notes.md の「今週の要点」
  2. 進捗                       「進捗」の表 (テーマ | 状況 | 今週の変化)
  3. 主な実験結果               「主な実験結果」の ### ごと。根拠のパスを添える
  4. 課題と次週の予定           「課題」「次週の予定」
  5. 前回報告からの変更点       weekly_changes.py が Git の履歴と実験索引から抜き出したもの

原稿の書き方は reports/README.md。色・文字の大きさ・上限は configs/report_weekly.yaml。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import weekly_changes as WC  # noqa: E402


def rgb(h: str):
    from pptx.dml.color import RGBColor
    return RGBColor.from_string(h)


class Deck:
    def __init__(self, cfg: dict, week: str, source: str):
        from pptx import Presentation
        from pptx.util import Inches
        self.cfg, self.T, self.L = cfg, cfg["theme"], cfg["limits"]
        self.prs = Presentation(cfg["template"]) if cfg.get("template") else Presentation()
        self.prs.slide_width = Inches(cfg["slide"]["width_in"])
        self.prs.slide_height = Inches(cfg["slide"]["height_in"])
        self.W, self.H = cfg["slide"]["width_in"], cfg["slide"]["height_in"]
        self.week, self.source = week, source
        self.slides = []

    # --- 部品 -------------------------------------------------------------
    def new_slide(self, title: str):
        layout = self.prs.slide_layouts[6] if len(self.prs.slide_layouts) > 6 else self.prs.slide_layouts[-1]
        s = self.prs.slides.add_slide(layout)
        self.text(s, 0.6, 0.35, self.W - 1.2, 0.75, [title], size=self.T["size_title"], color=self.T["color_title"], bold=True)
        self.rect(s, 0.6, 1.12, self.W - 1.2, 0.04, self.T["color_accent"])
        self.slides.append(s)
        return s

    def rect(self, s, x, y, w, h, color):
        from pptx.enum.shapes import MSO_SHAPE
        from pptx.util import Inches
        r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        r.fill.solid(); r.fill.fore_color.rgb = rgb(color); r.line.fill.background()
        return r

    def text(self, s, x, y, w, h, lines, size=None, color=None, bold=False, bullet=False, font=None, spacing=4):
        from pptx.util import Inches, Pt
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.05)
        for i, ln in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(spacing)
            r = p.add_run()
            r.text = ("• " if bullet else "") + ln
            r.font.size = Pt(size or self.T["size_body"])
            r.font.bold = bold
            r.font.name = font or self.T["font"]
            r.font.color.rgb = rgb(color or self.T["color_text"])
        return tb

    def band(self, s, x, y, w, label, h=0.42, size=None):
        """見出しの帯。長い見出しは h を大きくして折り返す (切らない)。"""
        self.rect(s, x, y, w, h, self.T["color_accent"])
        self.text(s, x + 0.1, y + 0.03, w - 0.2, h - 0.06, [label], size=size or self.T["size_heading"],
                  color=self.T["color_band_text"], bold=True, spacing=0)

    def cap(self, items: list[str], n: int | None = None) -> list[str]:
        n = n or self.L["max_bullets"]
        return items if len(items) <= n else items[: n - 1] + [f"ほか {len(items) - n + 1} 件 (原稿を参照)"]

    def table(self, s, x, y, w, rows, col_w, size=None):
        from pptx.util import Inches, Pt
        nr, nc = len(rows), len(rows[0])
        t = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(0.4 * nr)).table
        for j, cw in enumerate(col_w):
            t.columns[j].width = Inches(cw)
        for i, row in enumerate(rows):
            for j in range(nc):
                cell = t.cell(i, j)
                cell.text = row[j] if j < len(row) else ""
                cell.fill.solid(); cell.fill.fore_color.rgb = rgb(self.T["color_table_head"] if i == 0 else "FFFFFF")
                for p in cell.text_frame.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(size or self.T["size_small"] + 1)
                        r.font.name = self.T["font"]
                        r.font.bold = i == 0
                        r.font.color.rgb = rgb(self.T["color_text"])
        return t

    def footers(self):
        n = len(self.slides)
        for i, s in enumerate(self.slides, 1):
            self.text(s, 0.6, self.H - 0.45, self.W - 1.2, 0.3,
                      [f"週次報告 {self.week}  |  {i} / {n}  |  原稿: {self.source}"],
                      size=self.T["size_small"] - 1, color=self.T["color_muted"])


def build(notes: WC.Notes, changes: dict, cfg: dict, source: str):
    m = notes.meta
    week = str(m.get("week"))
    d = Deck(cfg, week, source)
    T, L = d.T, d.L

    # 1. 表紙と要点
    s = d.new_slide(f"週次報告 {week}  {m.get('title', '')}".strip())
    d.text(s, 0.6, 1.3, d.W - 1.2, 0.4,
           [f"期間 {m.get('period_start')} 〜 {m.get('period_end')}  |  対象: {m.get('audience', '')}"],
           size=T["size_body"], color=T["color_muted"])
    d.band(s, 0.6, 1.9, d.W - 1.2, "今週の要点")
    d.text(s, 0.75, 2.45, d.W - 1.5, 3.6, d.cap(notes.sections.get("今週の要点", [])), size=T["size_body"] + 2, bullet=True, spacing=10)
    d.text(s, 0.6, d.H - 1.0, d.W - 1.2, 0.4,
           [f"比較したコミット {changes['base']}..{changes['head']} (コミット {len(changes['commits'])} 件)。変更点の一覧は最後のスライドと changes.md"],
           size=T["size_small"], color=T["color_muted"])

    # 2. 進捗
    rows = notes.tables.get("進捗")
    if rows:
        s = d.new_slide("進捗 (ロードマップのテーマごと)")
        body = rows[:1] + rows[1:][: L["max_bullets"] + 1]
        w = d.W - 1.2
        d.table(s, 0.6, 1.4, w, body, [w * 0.24, w * 0.14, w * 0.62])
        extra = notes.sections.get("進捗", [])
        if extra:
            d.text(s, 0.6, d.H - 1.5, w, 0.9, d.cap(extra, 3), size=T["size_small"] + 1, color=T["color_muted"], bullet=True)

    # 3. 主な実験結果
    res = notes.results[: L["max_results"]]
    if res:
        s = d.new_slide("主な実験結果")
        n = len(res); gap = 0.25; cw = (d.W - 1.2 - gap * (n - 1)) / n
        for k, r in enumerate(res):
            x = 0.6 + k * (cw + gap)
            d.band(s, x, 1.4, cw, r.title, h=0.8, size=T["size_body"])     # 2 行まで
            d.text(s, x + 0.05, 2.3, cw - 0.1, 2.95, d.cap(r.lines, 5), size=T["size_body"] - 1, bullet=True, spacing=6)
            ev = r.evidence[: L["max_evidence_per_result"]]
            if len(r.evidence) > len(ev):
                ev.append(f"ほか {len(r.evidence) - len(ev)} 件")
            d.text(s, x + 0.05, 5.6, cw - 0.1, 0.3, ["根拠"], size=T["size_small"], color=T["color_muted"], bold=True)
            d.text(s, x + 0.05, 5.9, cw - 0.1, 1.0, ev, size=T["size_small"] - 1, color=T["color_muted"], font=T["font_mono"], spacing=1)

    # 4. 課題と次週の予定
    issues, plans = notes.sections.get("課題", []), notes.sections.get("次週の予定", [])
    if issues or plans:
        s = d.new_slide("課題と次週の予定")
        cw = (d.W - 1.2 - 0.3) / 2
        for k, (lab, items) in enumerate((("課題", issues), ("次週の予定", plans))):
            x = 0.6 + k * (cw + 0.3)
            d.band(s, x, 1.4, cw, lab)
            d.text(s, x + 0.05, 1.95, cw - 0.1, 4.7, d.cap(items), size=T["size_body"], bullet=True, spacing=8)

    # 5. 前回報告からの変更点
    s = d.new_slide("前回報告からの変更点 (Git の履歴と実験索引から自動抽出)")
    lw = (d.W - 1.2) * 0.55
    d.band(s, 0.6, 1.4, lw, f"コミット {len(changes['commits'])} 件")
    cm = [f"{c['hash']}  {c['date']}  {c['subject']}" for c in changes["commits"]]
    d.text(s, 0.65, 1.95, lw - 0.1, 3.0, d.cap(cm, 8) or ["なし"], size=T["size_small"], font=T["font"], spacing=2)
    rx = 0.6 + lw + 0.3; rw = d.W - 0.6 - rx
    d.band(s, rx, 1.4, rw, "変更したファイル")
    area_rows = [["区分", "件数"]] + [[a, str(len(v))] for a, v in list(changes["by_area"].items())[: L["max_changed_files"]]]
    d.table(s, rx, 1.95, rw, area_rows, [rw * 0.7, rw * 0.3])
    ix = changes["index"]; ev = changes["evidence"]
    bad = [e["path"] for e in ev if not e["exists"]]
    untracked = [e["path"] for e in ev if e["exists"] and not e["tracked"] and not e["index_row"]]
    summary = [
        f"実験索引: 行の追加 {len(ix['added'])}・削除 {len(ix['removed'])}・手で書く列の変化 {len(ix['changed'])}"
        + (f" (新しい列 {len(ix['new_cols'])})" if ix["new_cols"] else ""),
        f"期間中に更新された out/ の結果: {len(changes['updated_outputs'])} 件",
        f"報告の根拠 {len(ev)} 件: 見つからない {len(bad)} 件、git 管理外で索引にも無い {len(untracked)} 件",
    ]
    if changes["dirty"]:
        summary.append("注意: 未コミットの変更がある状態で作成した")
    d.text(s, 0.6, 5.2, d.W - 1.2, 1.5, summary, size=T["size_small"] + 1, bullet=True, spacing=3)

    if len(d.slides) > L["max_slides"]:
        raise SystemExit(f"スライドが {len(d.slides)} 枚になった (上限 {L['max_slides']})")
    d.footers()
    return d.prs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notes", type=Path)
    ap.add_argument("--config", type=Path, default=WC.CONFIG)
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    notes = WC.parse_notes(args.notes.read_text(encoding="utf-8"))
    week = str(notes.meta.get("week", args.notes.parent.name))
    changes = WC.collect(notes, cfg, args.head)
    (args.notes.parent / "changes.md").write_text(WC.to_markdown(changes, week), encoding="utf-8")
    try:
        source = str(args.notes.resolve().relative_to(REPO))
    except ValueError:
        source = str(args.notes)
    prs = build(notes, changes, cfg, source)
    out = args.out or args.notes.parent / f"weekly_{week}.pptx"
    prs.save(out)
    print(f"{len(prs.slides)} 枚 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
