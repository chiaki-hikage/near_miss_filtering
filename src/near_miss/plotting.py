"""図を描くときの OS 依存部分。

抽出処理 (signals / features / detectors / sideslip) はこのモジュールを
一切参照しない。**検出の結果は OS によって変わらない。**
ここで吸収するのは「日本語が描けるフォントの名前が OS ごとに違う」ことだけ。

macOS には Hiragino Sans が標準で入っているが Linux には無い。逆に Linux では
Noto Sans CJK JP や IPAexGothic が使われる。名前を決め打ちにすると、
片方の OS で軸ラベルが豆腐 (□□□) になる。
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# 上から順に探して、最初に見つかったものを使う。
# 同じ図を Mac と Linux で描いたときに字形は変わるが、値は変わらない。
FONT_CANDIDATES: tuple[str, ...] = (
    # macOS
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Arial Unicode MS",
    # Linux (Debian/Ubuntu は fonts-noto-cjk、Amazon Linux は google-noto-sans-cjk-jp-fonts)
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "IPAGothic",
    "TakaoPGothic",
    "VL PGothic",
    # Windows
    "Yu Gothic",
    "MS Gothic",
)

# どれも無かったときの案内。図は描けるが日本語が豆腐になる。
INSTALL_HINT = (
    "日本語フォントが見つかりません。図の日本語が □ になります。\n"
    "  Debian/Ubuntu : sudo apt-get install -y fonts-noto-cjk\n"
    "  Amazon Linux 2023 : sudo dnf install -y google-noto-sans-cjk-jp-fonts\n"
    "  入れた後に matplotlib のキャッシュを消す: rm -rf ~/.cache/matplotlib"
)


def available_japanese_fonts() -> list[str]:
    """この環境に入っている候補フォントを、候補の順で返す。"""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    return [name for name in FONT_CANDIDATES if name in installed]


def configure_japanese_font(verbose: bool = False) -> str | None:
    """日本語が描けるフォントを matplotlib に設定し、選んだ名前を返す。

    見つからなければ None を返して警告を出す。図の生成そのものは止めない。
    OS を跨いで同じスクリプトを動かせるようにするのが目的で、
    フォントが無いことを致命的な失敗にはしない。
    """
    import matplotlib.pyplot as plt

    found = available_japanese_fonts()
    # **この環境に実際に入っているものだけ**を並べる。入っていない名前を
    # 残すと matplotlib が描画のたびに findfont の警告を出す。
    # DejaVu Sans は matplotlib 同梱で必ずある (日本語は持たない)。
    plt.rcParams["font.family"] = [*found, "DejaVu Sans"]
    # 負号が豆腐になるのを防ぐ (日本語フォントに U+2212 が無いことがある)
    plt.rcParams["axes.unicode_minus"] = False
    if not found:
        log.warning(INSTALL_HINT)
        if verbose:
            print(INSTALL_HINT)
        return None
    if verbose:
        print(f"日本語フォント: {found[0]}")
    return found[0]


def use_headless_backend() -> None:
    """画面の無い環境 (EC2 など) でも描けるようにする。

    matplotlib は既定で GUI バックエンドを選ぼうとするため、
    DISPLAY の無い Linux では import の時点で失敗することがある。
    """
    import matplotlib

    matplotlib.use("Agg")


def setup(verbose: bool = False) -> str | None:
    """バックエンドとフォントをまとめて整える。図を描くスクリプトの入口で呼ぶ。"""
    use_headless_backend()
    return configure_japanese_font(verbose=verbose)


def describe() -> dict[str, Any]:
    """環境確認用。どのバックエンド・フォントになるかを返す。"""
    import matplotlib

    return {
        "backend": matplotlib.get_backend(),
        "japanese_fonts": available_japanese_fonts(),
        "matplotlib": matplotlib.__version__,
    }
