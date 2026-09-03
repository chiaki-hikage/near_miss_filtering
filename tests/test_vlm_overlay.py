"""レビュー用 MP4 の字幕まわりの試験。

描画そのものは目で見るしかないが、**因果性と折り返し**は機械で確かめられる。
"""

from __future__ import annotations

import pytest

from near_miss.vlm.overlay import Judgment, STATE_COLOR, judgment_at, wrap


def _j(t, state="normal"):
    return Judgment(t_eval=t, state=state, hazard_type="none", risk_level=0,
                    confidence=0.5, lines=[("情景", "x")])


def test_評価前のフレームには判定を出さない():
    js = [_j(10.0), _j(10.5), _j(11.0)]
    assert judgment_at(js, 9.9) is None
    assert judgment_at(js, 10.0).t_eval == 10.0


def test_重ねるのは過去の判定だけ():
    js = [_j(10.0), _j(10.5, "hazard"), _j(11.0)]
    # 10.2 秒のフレームに 10.5 秒の判定を出してはいけない
    assert judgment_at(js, 10.2).t_eval == 10.0
    assert judgment_at(js, 10.5).t_eval == 10.5
    assert judgment_at(js, 10.7).state == "hazard"
    assert judgment_at(js, 99.0).t_eval == 11.0      # 最後の判定は保持する


class _Draw:
    """幅の計算だけを差し替えた偽物。1 文字 10 px とする。"""

    def textlength(self, s, font=None):
        return len(s) * 10


def test_日本語は文字単位で折り返す():
    text = "右レーンから車が前方近くに入ってきておりリスクが高い"
    lines = wrap(text, None, 100, _Draw())
    assert all(len(l) <= 10 for l in lines)
    assert "".join(lines) == text


def test_改行を保つ():
    lines = wrap("あい\nうえお", None, 100, _Draw())
    assert lines == ["あい", "うえお"]


def test_空文字は行を作らない():
    assert wrap("", None, 100, _Draw()) == []


def test_1文字が幅を超えても落とさない():
    lines = wrap("あいう", None, 5, _Draw())
    assert lines == ["あ", "い", "う"]


def test_状態の色が揃っている():
    assert set(STATE_COLOR) == {"normal", "caution", "hazard", "unknown"}


# --- フォントの字形検査 --------------------------------------------------
def test_日本語を持たないフォントを弾く():
    """DejaVuSans のような欧文専用フォントは Linux にほぼ必ずある。

    パスが見つかっただけで採用すると、字幕が全部 □ の動画ができる。
    実際に描いた結果で判定する。
    """
    from PIL import ImageFont
    from near_miss.vlm.overlay import has_japanese

    # 既定のビットマップフォントは日本語を持たない
    assert has_japanese(ImageFont.load_default()) is False


def test_探索したフォントは日本語を描ける():
    from near_miss.vlm.overlay import has_japanese, load_font

    font, ok, path = load_font(20)
    if not ok:
        pytest.skip(f"この環境に日本語フォントがありません ({path})")
    assert has_japanese(font) is True


@pytest.mark.parametrize("path", [
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
])
def test_欧文専用フォントは日本語なしと判定される(path):
    from pathlib import Path as _P
    from PIL import ImageFont
    from near_miss.vlm.overlay import has_japanese

    if not _P(path).is_file():
        pytest.skip(f"{path} がありません")
    assert has_japanese(ImageFont.truetype(path, 20)) is False
