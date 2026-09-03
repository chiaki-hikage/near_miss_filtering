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

# 日本語が描けるフォントを探す。
#
# **パスが見つかっただけで信用しない。** DejaVuSans は Linux にほぼ必ずあるが
# 日本語グリフを持たないので、そのまま使うと字幕が全部 □ になる。
# 実際に「あ」「車」を描いてみて、.notdef と違う絵が出るかで判定する。
FONT_CANDIDATES = (
    # macOS
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Amazon Linux 2023 / RHEL 系 (google-noto-sans-cjk-jp-fonts)
    "/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.otf.ttc",
    "/usr/share/fonts/google-noto-sans-cjk-fonts/NotoSansCJKjp-Regular.otf",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    # Debian / Ubuntu (fonts-noto-cjk)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    # その他よくあるもの
    "/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf",
    "/usr/share/fonts/ipa-gothic/ipag.ttf",
)

# 上のどれも無い場合に走査するディレクトリと、日本語フォントらしい名前。
SEARCH_DIRS = (
    "/usr/share/fonts", "/usr/local/share/fonts",
    "~/.fonts", "~/.local/share/fonts",
    "/Library/Fonts", "/System/Library/Fonts",
)
SEARCH_HINTS = ("cjk", "notosansjp", "notoserifjp", "gothic", "mincho",
                "ipa", "takao", "vl-", "hiragino", "yugoth", "meiryo", "msgothic")

FONT_HINT = (
    "日本語フォントが見つかりません。字幕が □ になります。\n"
    "  Amazon Linux 2023 : sudo dnf install -y google-noto-sans-cjk-jp-fonts\n"
    "  Debian/Ubuntu     : sudo apt-get install -y fonts-noto-cjk\n"
    "  入れた後に scripts/make_review_video.py を流し直してください。\n"
    "  特定のファイルを使う場合は --font <path> で指定できます。"
)

# 判定に使う文字。私用領域は .notdef になるので、これと同じ絵なら
# その文字を持っていない。
_PROBE = ("あ", "車", "間")
_NOTDEF = "\ue000"


def _render(font, ch: str) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("L", (72, 72), 0)
    ImageDraw.Draw(im).text((4, 4), ch, font=font, fill=255)
    return im.tobytes()


def has_japanese(font) -> bool:
    """このフォントで日本語が描けるか。

    .notdef (私用領域) と同じ絵になる、あるいは何も描かれないなら描けない。
    パスや名前ではなく**実際に描いた結果**で判定する。
    """
    try:
        notdef = _render(font, _NOTDEF)
        blank = _render(font, " ")
    except Exception:
        return False
    for ch in _PROBE:
        try:
            g = _render(font, ch)
        except Exception:
            return False
        if g == notdef or g == blank:
            return False
    return True


def _try_open(path: str, size: int):
    """TTC は複数の書体を含む。日本語を持つ面が見つかるまで順に試す。"""
    from PIL import ImageFont

    for index in range(6):
        try:
            f = ImageFont.truetype(path, size, index=index)
        except Exception:
            break
        if has_japanese(f):
            return f
    return None


def _scan_dirs(size: int):
    seen: list[str] = []
    for d in SEARCH_DIRS:
        root = Path(d).expanduser()
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in (".ttf", ".ttc", ".otf", ".otc"):
                continue
            if not any(h in p.name.lower() for h in SEARCH_HINTS):
                continue
            seen.append(str(p))
            f = _try_open(str(p), size)
            if f is not None:
                return f, str(p)
            if len(seen) > 200:      # 走査が長引かないよう打ち切る
                return None, None
    return None, None


def load_font(size: int) -> tuple[Any, bool, str]:
    """日本語が描けるフォントを返す。

    戻り値は (フォント, 日本語が描けるか, 選んだファイル).
    描けるものが無ければ既定のビットマップフォントを返し、呼び出し側が
    警告を出せるようにする。**字幕が読めなくても処理は止めない。**
    """
    from PIL import ImageFont

    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            f = _try_open(path, size)
            if f is not None:
                return f, True, path

    f, path = _scan_dirs(size)
    if f is not None:
        return f, True, path

    return ImageFont.load_default(), False, "(既定のビットマップ)"


def load_font_at(path: str, size: int) -> tuple[Any, bool, str]:
    """明示されたファイルを使う (--font)。日本語が描けるかは検査する。"""
    from PIL import ImageFont

    f = _try_open(path, size)
    if f is not None:
        return f, True, path
    try:
        return ImageFont.truetype(path, size), False, path
    except Exception as exc:
        raise SystemExit(f"フォントを開けません: {path} ({exc})")


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
