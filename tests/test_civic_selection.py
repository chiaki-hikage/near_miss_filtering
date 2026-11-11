"""scripts/crosswind/civic_selection/ (Civic の取得セグメントの選定) が、移動後も同じ結果を出すこと。

選定の入力 (data/comma2k19_civic/_selection/wind/*.csv) と既存の選定結果
(data/comma2k19_civic/selection.csv) は git 管理外なので、無い環境では skip する。
ネットワークには接続しない (collect.py は引数なしなら何も取りに行かない)。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts/crosswind/civic_selection"
WIND = REPO / "data/comma2k19_civic/_selection/wind"
SELECTION = REPO / "data/comma2k19_civic/selection.csv"

# スクリプトの置き場所に __pycache__ を作らない
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


@pytest.mark.skipif(not (WIND / "all_segments_wind.csv").exists() or not SELECTION.exists(),
                    reason="data/comma2k19_civic の選定の入力が無い")
def test_pick_reproduces_existing_selection(tmp_path):
    out = tmp_path / "selection.csv"
    subprocess.run([sys.executable, str(SCRIPTS / "pick_civic_segments.py"), str(WIND), str(out)],
                   cwd=REPO, env=ENV, check=True, capture_output=True)
    assert out.read_bytes() == SELECTION.read_bytes()


def test_collect_imports_remote_zip_from_its_own_directory(tmp_path):
    """collect.py は同じフォルダの remote_zip を import し、出力をカレントディレクトリに書く。

    引数なしで動かすと、import だけが通って何も書かない。カレントディレクトリを
    リポジトリの外にしても import が通ることを確かめる。
    """
    r = subprocess.run([sys.executable, str(SCRIPTS / "collect.py")],
                       cwd=tmp_path, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert list(tmp_path.iterdir()) == []
