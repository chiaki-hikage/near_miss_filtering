"""OS を跨いで動かすための取り決めを固定する試験。

macOS では通るのに Linux では落ちる、という種類の違いを対象にする。
横滑りの判定そのものは扱わない (それは tests/test_sideslip.py)。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from near_miss import plotting
from near_miss.config import REPO_ROOT
from near_miss.io import comma1m, comma_car_segments, kit_msdm

REPO = Path(__file__).resolve().parents[1]


def test_repo_root_is_the_repository():
    assert REPO_ROOT == REPO
    assert (REPO_ROOT / "configs" / "detection.yaml").is_file()


def test_default_data_paths_are_spelled_as_documented():
    """データ置き場の綴りを固定する。

    macOS はファイル名の大小を区別しないので、コードと実物がずれていても
    手元では動いてしまう。Linux では動かない。docs/environment.md §4 の
    ツリーと食い違ったらここで落とす。
    """
    assert comma_car_segments.DEFAULT_CACHE == REPO / "raw_data" / "comma_car_segments"
    # 末尾は大文字の M。comma1m ではない。
    assert comma1m.DEFAULT_CACHE == REPO / "raw_data" / "comma1M"
    assert kit_msdm.DEFAULT_ROOT == (
        REPO / "raw_data" / "kit_msdm" / "10.35097-44a91t97pmnha1k9" / "data" / "dataset"
    )


def test_no_absolute_home_paths_in_sources():
    """/Users/... や /home/... の直書きが紛れ込んでいないこと。"""
    bad = []
    for d in ("src", "scripts", "configs"):
        for p in (REPO / d).rglob("*"):
            if p.suffix not in (".py", ".yaml", ".sh") or "__pycache__" in p.parts:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            for marker in ("/Users/", "/home/"):
                if marker in text:
                    bad.append(f"{p.relative_to(REPO)}: {marker}")
    assert not bad, "絶対パスが直書きされています: " + ", ".join(bad)


def test_src_modules_do_not_require_matplotlib():
    """抽出処理は matplotlib 無しで動くこと。

    EC2 で `uv sync` だけ (--extra viz 無し) を実行した場合に、
    横滑りフィルタが動かなくなっていないかを見る。
    """
    for path in (REPO / "src" / "near_miss").rglob("*.py"):
        if path.name == "plotting.py":
            continue        # ここだけは matplotlib を使ってよい (関数の中で import する)
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import matplotlib", "from matplotlib")):
                assert line != stripped, (
                    f"{path.name} が matplotlib をトップレベルで import しています"
                )


def test_bootstrap_forces_utf8_stdout():
    """LANG が無い Linux でも日本語が出せること。

    子プロセスの環境から LANG/LC_ALL を外し、標準出力の文字コードが
    UTF-8 になっていることを確かめる。scripts/_bootstrap.py の役目。
    """
    code = (
        "import sys; sys.path.insert(0, 'scripts'); import _bootstrap;"
        "print(sys.stdout.encoding); print('日本語')"
    )
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp"}      # LANG も LC_ALL も渡さない
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.split()
    assert lines[0].lower().replace("-", "_") == "utf_8"
    assert "日本語" in r.stdout


def test_font_candidates_cover_mac_and_linux():
    """フォント候補に mac 側と Linux 側の両方が入っていること。"""
    names = plotting.FONT_CANDIDATES
    assert "Hiragino Sans" in names                       # macOS
    assert any(n.startswith("Noto Sans CJK") for n in names)   # Linux
    assert any(n.startswith("IPA") for n in names)             # Linux


def test_plotting_setup_is_headless_and_reports_font():
    """図の設定が画面なしで通ること。matplotlib が無ければ飛ばす。"""
    pytest.importorskip("matplotlib")
    import matplotlib

    chosen = plotting.setup()
    assert matplotlib.get_backend().lower() == "agg"
    info = plotting.describe()
    assert isinstance(info["japanese_fonts"], list)
    # フォントが 1 つも無い環境でも None を返すだけで、例外にはしない。
    assert chosen is None or chosen in plotting.FONT_CANDIDATES
