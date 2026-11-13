#!/usr/bin/env python3
"""合意した構成案 (outline.yaml) から週次報告の PPT を作る。PPT は git 管理外 (.gitignore)。

  uv run --extra report python scripts/report/build_weekly_pptx.py reports/weekly/2026-W46/outline.yaml --check
      構成案の検査だけをする (枚数・主文・図表・根拠のパス)。PPT は作らない
  uv run --extra report python scripts/report/build_weekly_pptx.py reports/weekly/2026-W46/outline.yaml
      -> reports/weekly/2026-W46/weekly_2026-W46.pptx と changes.md

**PPT を作る時期はユーザーが決める。構成案に合意を得てから作る** (手順は .claude/skills/weekly-slides/SKILL.md)。

1 枚の基本構成 (1 スライド 1 メッセージ):
  タイトル / 主文 (言いたいことを 1 文) / 図か表 1 つ + 注釈 / 根拠のパス

outline.yaml の形 (ひな形は reports/weekly/OUTLINE_TEMPLATE.yaml):
  week, title, period_start, period_end, notes (原稿 notes.md のパス。base_commit を読む)
  header (任意): 上のヘッダーの文。省略時は「週次報告 <week> | <期間>」
  slides:
    - title: タイトル
      message: 主文
      visual:                       # 次のどれか 1 つ
        image: shared/x.png         #   図 (リポジトリからの相対パス)
        table: [[見出し, ...], [...]]  #   表 (1 行目が見出し)
        table_csv: {path: ..., columns: [...], max_rows: 8}   # CSV から表
        changes: commits | areas    #   前回報告からの変更点 (weekly_changes.py)
      notes: [注釈, ...]            # 図表の読み方・条件・限界
      evidence: [パス, ...]         # 根拠 (存在を確かめる)
      layout: right | below         # 注釈を図表の右に置くか下に置くか (省略時は設定の既定)

設定の style (任意。無ければ従来の見た目): title_bar (タイトルの前の縦棒)、title_rule (タイトルの下の横棒)、
message_marker (主文の頭の記号。帯の代わり)、evidence_beside_notes (下に置くとき注釈の幅の割合、根拠を右に)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import weekly_changes as WC  # noqa: E402

VISUAL_KEYS = ("image", "table", "table_csv", "changes")


# ---------------------------------------------------------------------------
# 検査
# ---------------------------------------------------------------------------

def check_outline(o: dict, cfg: dict) -> tuple[list[str], list[str]]:
    """(誤り, 注意) を返す。誤りがあれば PPT は作らない。"""
    L = cfg["limits"]
    err, warn = [], []
    slides = o.get("slides") or []
    if not L["min_slides"] <= len(slides) <= L["max_slides"]:
        err.append(f"スライドが {len(slides)} 枚 (決まりは {L['min_slides']}〜{L['max_slides']} 枚)")
    for i, s in enumerate(slides, 1):
        tag = f"{i} 枚目"
        if not s.get("title"):
            err.append(f"{tag}: タイトルが無い")
        msg = s.get("message")
        if not isinstance(msg, str) or not msg.strip():
            err.append(f"{tag}: 主文 (message) が 1 つの文になっていない")
        elif len(msg) > L["max_message_chars"]:
            warn.append(f"{tag}: 主文が {len(msg)} 字 (目安 {L['max_message_chars']} 字以内)")
        v = s.get("visual") or {}
        keys = [k for k in VISUAL_KEYS if k in v]
        if len(keys) != 1:
            err.append(f"{tag}: 図か表をちょうど 1 つ置く (今は {keys or 'なし'})")
        elif keys[0] == "image" and not (REPO / v["image"]).exists():
            err.append(f"{tag}: 図が見つからない: {v['image']}")
        elif keys[0] == "table_csv" and not (REPO / v["table_csv"]["path"]).exists():
            err.append(f"{tag}: 表の CSV が見つからない: {v['table_csv']['path']}")
        elif keys[0] == "table" and len(v["table"]) - 1 > L["max_table_rows"]:
            warn.append(f"{tag}: 表が {len(v['table']) - 1} 行 (目安 {L['max_table_rows']} 行以内)")
        if len(s.get("notes") or []) > L["max_notes"]:
            warn.append(f"{tag}: 注釈が {len(s['notes'])} 個 (目安 {L['max_notes']} 個以内)")
        if not s.get("evidence"):
            warn.append(f"{tag}: 根拠のパスが無い")
        for p in s.get("evidence") or []:
            if not (REPO / p.rstrip("/")).exists():
                err.append(f"{tag}: 根拠が見つからない: {p}")
    return err, warn


# ---------------------------------------------------------------------------
# 描画
# ---------------------------------------------------------------------------

def rgb(h: str):
    from pptx.dml.color import RGBColor
    return RGBColor.from_string(h)


class Deck:
    def __init__(self, cfg: dict, header: str, source: str):
        from pptx import Presentation
        from pptx.util import Inches
        self.cfg, self.T, self.G = cfg, cfg["theme"], cfg["layout"]
        self.prs = Presentation(cfg["template"]) if cfg.get("template") else Presentation()
        self.prs.slide_width = Inches(cfg["slide"]["width_in"])
        self.prs.slide_height = Inches(cfg["slide"]["height_in"])
        self.W, self.H = cfg["slide"]["width_in"], cfg["slide"]["height_in"]
        self.header, self.source = header, source

    def text(self, s, x, y, w, h, lines, size, color=None, bold=False, font=None, bullet=False, spacing=4):
        from pptx.util import Inches, Pt
        tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.05)
        for i, ln in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(spacing)
            r = p.add_run()
            r.text = ("• " if bullet else "") + str(ln)
            r.font.size, r.font.bold = Pt(size), bold
            r.font.name = font or self.T["font"]
            r.font.color.rgb = rgb(color or self.T["color_text"])

    def rect(self, s, x, y, w, h, color):
        from pptx.enum.shapes import MSO_SHAPE
        from pptx.util import Inches
        r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        r.fill.solid(); r.fill.fore_color.rgb = rgb(color); r.line.fill.background()

    def image(self, s, path: Path, x, y, w, h):
        """枠に収まるよう縦横比を保って置き、枠の中央にそろえる。"""
        from PIL import Image
        from pptx.util import Inches
        with Image.open(path) as im:
            iw, ih = im.size
        k = min(w / iw, h / ih)
        pw, ph = iw * k, ih * k
        s.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))

    def table(self, s, rows, x, y, w, widths=None):
        from pptx.util import Inches, Pt
        nr, nc = len(rows), max(len(r) for r in rows)
        rh = self.G["table_row_h_in"]
        t = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(rh * nr)).table
        ws = widths or [1] * nc
        for j in range(nc):
            t.columns[j].width = Inches(w * ws[j] / sum(ws))
        for i, row in enumerate(rows):
            for j in range(nc):
                c = t.cell(i, j)
                c.text = str(row[j]) if j < len(row) else ""
                c.fill.solid(); c.fill.fore_color.rgb = rgb(self.T["color_table_head"] if i == 0 else "FFFFFF")
                for p in c.text_frame.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(self.T["size_table"]); r.font.name = self.T["font"]
                        r.font.bold = i == 0; r.font.color.rgb = rgb(self.T["color_text"])

    def slide(self, spec: dict, rows=None):
        T, G = self.T, self.G
        st = self.cfg.get("style") or {}       # 見た目の切り替え (無ければ従来の週次報告の見た目)
        layout = self.prs.slide_layouts[6] if len(self.prs.slide_layouts) > 6 else self.prs.slide_layouts[-1]
        s = self.prs.slides.add_slide(layout)
        m = G["margin_in"]; cw = self.W - 2 * m
        self.text(s, m, 0.2, cw, 0.3, [self.header], T["size_small"], T["color_muted"])
        if st.get("title_bar"):                # タイトルの前に太い縦棒
            bw = st.get("title_bar_w_in", 0.12)
            self.rect(s, m, 0.55, bw, 0.5, T["color_accent"])
            self.text(s, m + bw + 0.1, 0.45, cw - bw - 0.1, 0.7, [spec["title"]], T["size_title"], T["color_title"], bold=True)
        else:
            self.text(s, m, 0.45, cw, 0.7, [spec["title"]], T["size_title"], T["color_title"], bold=True)
        if st.get("title_rule", True):
            self.rect(s, m, 1.15, cw, 0.04, T["color_accent"])
        mh = G["message_h_in"]
        my = G.get("message_top_in", 1.35)
        if st.get("message_marker"):           # 主文を「■」の箇条書きの形で
            self.text(s, m, my - 0.02, cw, mh, [f"{st['message_marker']} {spec['message']}"], T["size_message"], T["color_title"], bold=True)
        else:                                   # 主文: 左に帯を付けて、1 枚で言いたいことを目立たせる
            self.rect(s, m, my, 0.08, mh, T["color_accent"])
            self.text(s, m + 0.2, my - 0.02, cw - 0.2, mh, [spec["message"]], T["size_message"], T["color_title"], bold=True)
        top, bottom = my + mh + G.get("message_gap_in", 0.2), self.H - G["footer_h_in"]
        notes, ev = spec.get("notes") or [], spec.get("evidence") or []
        if spec.get("layout", G["notes_position"]) == "right":
            vw = cw * G["visual_ratio"]
            vx, vy, vh = m, top, bottom - top
            nx, ny, nw, nh = m + vw + 0.3, top, cw - vw - 0.3, bottom - top
        else:
            nh = G["notes_below_h_in"]
            vx, vy, vw, vh = m, top, cw, bottom - top - nh - 0.1
            nx, ny, nw = m, bottom - nh, cw
        v = spec["visual"]
        if "image" in v:
            self.image(s, REPO / v["image"], vx, vy, vw, vh)
        else:
            self.table(s, rows, vx, vy, vw, v.get("widths"))
        split = st.get("evidence_beside_notes")    # 下に置くとき、根拠を注釈の右に並べる (注釈の幅の割合)
        if split and spec.get("layout", G["notes_position"]) != "right":
            if notes:
                self.text(s, nx, ny, nw * split, nh, notes, T["size_notes"], T["color_text"], bullet=True, spacing=4)
            if ev:
                self.text(s, nx + nw * split + 0.15, ny, nw * (1 - split) - 0.15, nh, ["根拠"] + list(ev), T["size_small"] - 1.5,
                          T["color_muted"], font=T["font_mono"], spacing=1)
            return s
        if notes:
            self.text(s, nx, ny, nw, nh * 0.7, notes, T["size_notes"], T["color_text"], bullet=True, spacing=6)
        if ev:
            self.text(s, nx, ny + nh * 0.72, nw, nh * 0.28, ["根拠"] + list(ev), T["size_small"] - 1,
                      T["color_muted"], font=T["font_mono"], spacing=1)
        return s

    def footers(self):
        n = len(self.prs.slides)
        for i, s in enumerate(self.prs.slides, 1):
            self.text(s, self.G["margin_in"], self.H - 0.4, self.W - 2 * self.G["margin_in"], 0.3,
                      [f"{i} / {n}  |  構成案: {self.source}"], self.T["size_small"] - 1, self.T["color_muted"])


def visual_rows(v: dict, changes: dict | None) -> list[list[str]] | None:
    """表の行 (1 行目が見出し)。図のときは None。"""
    if "table" in v:
        return [[str(c) for c in r] for r in v["table"]]
    if "table_csv" in v:
        t = v["table_csv"]
        with (REPO / t["path"]).open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        cols = t.get("columns") or list(rows[0])
        return [cols] + [[r.get(c, "") for c in cols] for r in rows[: t.get("max_rows", 8)]]
    if "changes" in v:
        if v["changes"] == "commits":
            return [["コミット", "日付", "内容"]] + [[c["hash"], c["date"], c["subject"]] for c in changes["commits"]]
        return [["区分", "件数"]] + [[a, str(len(x))] for a, x in changes["by_area"].items()]
    return None


def build(outline: dict, cfg: dict, source: str, changes: dict | None = None):
    header = outline.get("header") or f"週次報告 {outline.get('week')}  |  {outline.get('period_start')} 〜 {outline.get('period_end')}"
    d = Deck(cfg, header, source)
    for spec in outline["slides"]:
        d.slide(spec, visual_rows(spec["visual"], changes))
    d.footers()
    return d.prs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outline", type=Path)
    ap.add_argument("--config", type=Path, default=WC.CONFIG)
    ap.add_argument("--check", action="store_true", help="構成案を検査するだけ (PPT を作らない)")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    outline = yaml.safe_load(args.outline.read_text(encoding="utf-8"))
    err, warn = check_outline(outline, cfg)
    for w in warn:
        print("注意:", w)
    for e in err:
        print("誤り:", e, file=sys.stderr)
    if err:
        return 1
    if args.check:
        print(f"構成案の検査を通った ({len(outline['slides'])} 枚)")
        return 0
    changes = None
    if outline.get("notes"):
        notes = WC.parse_notes((REPO / outline["notes"]).read_text(encoding="utf-8"))
        changes = WC.collect(notes, cfg, args.head)
        (args.outline.parent / "changes.md").write_text(WC.to_markdown(changes, str(outline.get("week"))), encoding="utf-8")
    elif any("changes" in s["visual"] for s in outline["slides"]):
        print("誤り: changes の表を使うには outline の notes (base_commit を持つ原稿) が要る", file=sys.stderr)
        return 1
    try:
        source = str(args.outline.resolve().relative_to(REPO))
    except ValueError:
        source = str(args.outline)
    prs = build(outline, cfg, source, changes)
    out = args.out or args.outline.parent / f"weekly_{outline.get('week')}.pptx"
    prs.save(out)
    print(f"{len(prs.slides)} 枚 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
