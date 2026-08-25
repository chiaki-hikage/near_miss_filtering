"""スクリプトから src/ 配下のパッケージを読めるようにする。

あわせて、**OS で違いが出る 2 点**をここで揃える。抽出処理そのものは
この違いの影響を受けないが、実行できるかどうかは変わってしまう。

1. 標準出力の文字コード
   このリポジトリのスクリプトは進捗も表も日本語で出す。macOS の Python は
   ロケールに関係なく UTF-8 を使うが、Linux では LANG が未設定だと ASCII に
   なり、最初の日本語を出した時点で UnicodeEncodeError で落ちる。
   EC2 の素の状態がまさにそれなので、ここで UTF-8 に固定する。

2. src/ の場所
   パッケージを入れずにスクリプトを動かせるようにする (uv sync で
   editable 導入もされるが、素の python でも動くようにしておく)。
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Python 3.7 以降は reconfigure が使える。既に UTF-8 なら実質何も起きない。
for _stream in (sys.stdout, sys.stderr):
    try:
        if (_stream.encoding or "").lower().replace("-", "_") != "utf_8":
            _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        # パイプ先が閉じている等。出力の文字コードは諦めて処理は続ける。
        pass
