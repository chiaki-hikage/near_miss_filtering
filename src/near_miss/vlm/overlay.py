"""レビュー用 MP4 の字幕描画。

モード B の判定を元映像の時刻に重ねて、**いつ判断が変わったか**を目で追えるようにする。
推論にも採点にも影響しない。読むためだけのもの。

日本語を描くので cv2.putText は使えない (ASCII しか出ない)。PIL の truetype で
描き、ffmpeg に生フレームを流して符号化する。

**因果性はここでも守る。** 時刻 t のフレームに重ねてよいのは
`t_eval <= t` の判定だけ。評価前のフレームに未来の判定を出すと、
「いつ気づいたか」を見るという目的が崩れる。
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# 状態ごとの色 (RGB)。信号と同じ並びにして、止めて見なくても分かるようにする。
STATE_COLOR: dict[str, tuple[int, int, int]] = {
    "normal": (90, 190, 110),
    "caution": (235, 190, 70),
    "hazard": (230, 85, 85),
    "unknown": (150, 150, 150),
}
PENDING_COLOR = (110, 110, 110)

# 日本語が描けるフォントの候補。matplotlib に依らずに探す
# (この機能のためだけに viz extra を要求しない)。
FONT_CANDIDATES = (
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-sans-cjk-fonts/NotoSansCJKjp-Regular.otf",
    "/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.otf.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # 日本語は出ないが最後の砦
)

FONT_HINT = (
    "日本語フォントが見つかりません。字幕が □ になります。\n"
    "  Amazon Linux 2023 : sudo dnf install -y google-noto-sans-cjk-jp-fonts\n"
    "  Debian/Ubuntu     : sudo apt-get install -y fonts-noto-cjk"
)


def find_font_path() -> str | None:
    for p in FONT_CANDIDATES:
        if Path(p).is_file():
            return p
    try:  # matplotlib があれば、そちらの探索結果も使う
        from matplotlib import font_manager
        from matplotlib.font_manager import FontProperties

        for name in ("Noto Sans CJK JP", "Hiragino Sans", "IPAexGothic", "Yu Gothic"):
            path = font_manager.findfont(FontProperties(family=name), fallback_to_default=False)
            if path and Path(path).is_file():
                return path
    except Exception:
        pass
    return None


def load_font(size: int):
    from PIL import ImageFont

    path = find_font_path()
    if path is None:
        return ImageFont.load_default(), False
    return ImageFont.truetype(path, size), True


def wrap(text: str, font, max_px: int, draw) -> list[str]:
    """指定幅に収まるよう折り返す。

    日本語は単語区切りが無いので**1 文字ずつ**詰める。空白で切ると
    長い和文が 1 行のままはみ出す。
    """
    if not text:
        return []
    lines: list[str] = []
    cur = ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        trial = cur + ch
        if draw.textlength(trial, font=font) <= max_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


@dataclass
class Judgment:
    """ある評価時刻の判定。字幕に出す分だけ持つ。"""

    t_eval: float
    state: str
    hazard_type: str
    risk_level: int
    confidence: float
    lines: list[tuple[str, str]]      # (見出し, 本文)


def judgment_at(js: Sequence[Judgment], t: float) -> Judgment | None:
    """時刻 t のフレームに重ねてよい判定。**t_eval <= t のうち最も新しいもの。**

    未来の判定は決して返さない。評価前のフレームでは None。
    """
    ts = [j.t_eval for j in js]
    i = bisect.bisect_right(ts, t + 1e-9) - 1
    return js[i] if i >= 0 else None


def draw_caption(img, j: Judgment | None, head: str, font, small,
                 max_lines: int = 6):
    """画面下部に半透明の帯を敷いて字幕を描く。

    帯の高さは行数に合わせる。固定にすると短い判定で無駄に隠れ、
    長い判定でははみ出す。
    """
    from PIL import Image, ImageDraw

    W, H = img.size
    # 下端の余白は上より厚くする。行の下に伸びる字 (ゃ・ょ・g) が
    # 画面の縁で切れて読みにくくなるため。
    pad, pad_bottom, gap = 14, 20, 4
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    body: list[str] = []
    for label, text in (j.lines if j else []):
        for k, ln in enumerate(wrap(f"{label}: {text}" if label else text,
                                    font, W - 2 * pad, d)):
            body.append(ln)
    if not j:
        body = ["(この時刻はまだ評価されていません)"]
    body = body[:max_lines]

    lh = font.size + gap
    band = pad + pad_bottom + (small.size + gap) + len(body) * lh
    top = max(0, H - band)
    d.rectangle([0, top, W, H], fill=(0, 0, 0, 185))

    color = STATE_COLOR.get(j.state, PENDING_COLOR) if j else PENDING_COLOR
    d.rectangle([0, top, W, top + 4], fill=(*color, 255))

    y = top + pad
    d.text((pad, y), head, font=small, fill=(*color, 255))
    y += small.size + gap
    for ln in body:
        d.text((pad, y), ln, font=font, fill=(240, 240, 240, 255))
        y += lh

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
