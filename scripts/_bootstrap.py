"""スクリプトから src/ 配下のパッケージを読めるようにする。

あわせて、**OS で違いが出る 2 点**をここで揃える。抽出処理そのものは
この違いの影響を受けないが、実行できるかどうかは変わってしまう。

1. 標準出力の文字コード
   このリポジトリのスクリプトは進捗も表も日本語で出す。macOS の Python は
   ロケールに関係なく UTF-8 を使うが、Linux では LANG が未設定だと ASCII に
   なり、最初の日本語を出した時点で UnicodeEncodeError で落ちる。
   ここで UTF-8 に固定する。

2. src/ と他のスクリプトの場所
   パッケージを入れずにスクリプトを動かせるようにする (uv sync で
   editable 導入もされるが、素の python でも動くようにしておく)。
   あわせて scripts/ 配下の役割別ディレクトリを全て sys.path に入れ、
   ディレクトリを跨いだスクリプト間の import が通るようにする。
"""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
SRC = SCRIPTS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# スクリプトは役割別のディレクトリに分かれているが、互いを import するものが
# ある (report_* が eval_l1_* を読むなど)。実行中のスクリプトのディレクトリしか
# sys.path に入らないので、ここで全てのディレクトリを通しておく。
for _d in sorted(SCRIPTS.iterdir()):
    if _d.is_dir() and not _d.name.startswith((".", "__")):
        if str(_d) not in sys.path:
            sys.path.append(str(_d))

# Python 3.7 以降は reconfigure が使える。既に UTF-8 なら実質何も起きない。
for _stream in (sys.stdout, sys.stderr):
    try:
        if (_stream.encoding or "").lower().replace("-", "_") != "utf_8":
            _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        # パイプ先が閉じている等。出力の文字コードは諦めて処理は続ける。
        pass
