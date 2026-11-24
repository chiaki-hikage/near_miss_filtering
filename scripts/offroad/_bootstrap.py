"""役割別ディレクトリから scripts/_bootstrap.py を読み込むための中継。

実体は scripts/_bootstrap.py。ここはその場所を解決して実行するだけで、
各スクリプトの ``import _bootstrap`` をそのまま通すために置いてある。
"""

import importlib.util
import sys
from pathlib import Path

_REAL = Path(__file__).resolve().parents[1] / "_bootstrap.py"
_spec = importlib.util.spec_from_file_location("_scripts_bootstrap", _REAL)
_mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_scripts_bootstrap", _mod)
_spec.loader.exec_module(_mod)

SCRIPTS = _mod.SCRIPTS
SRC = _mod.SRC
