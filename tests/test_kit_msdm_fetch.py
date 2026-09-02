"""KIT MSDM の受け取り経路の試験。

閉鎖環境に持ち込むものなので、**中身の照合**と**tar の安全性**を機械で見る。
ネットワークには一切出ない。
"""

from __future__ import annotations

import ast
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from near_miss.io import kit_msdm as kit

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "fetch_kit_msdm.py"


def _load_script():
    """スクリプトを import せずに関数だけ取り出す (import 時に副作用を出さないため)。"""
    import importlib.util

    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("fetch_kit_msdm", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 配布物の素性
# ---------------------------------------------------------------------------
def test_config_pins_the_archive():
    cfg = kit.load_dataset_config()
    a = cfg["archive"]
    assert len(a["md5"]) == 32 and a["size_bytes"] > 0
    # 直リンクは書かない (版で変わるうえ、閉鎖環境では使わない)
    assert cfg.get("download_url", "") == ""
    assert cfg["bag"]["root"] and cfg["bag"]["manifest_entries"] > 0


def test_config_records_the_unlisted_readme():
    """manifest に載らない readme.txt を「想定内」として持っていること。

    ここが抜けると、正しい配布物なのに毎回警告が出る。
    """
    cfg = kit.load_dataset_config()
    assert "data/readme.txt" in cfg["bag"]["unlisted"]


# ---------------------------------------------------------------------------
# BagIt の照合
# ---------------------------------------------------------------------------
def _make_bag(root: Path, files: dict[str, bytes], unlisted: dict[str, bytes] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        lines.append(f"{kit.file_md5(p)}  {rel}")
    for rel, data in (unlisted or {}).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (root / "manifest-md5.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


CFG = {"bag": {"root": "bag", "manifest_entries": 2, "unlisted": ["data/readme.txt"]}}


def test_verify_bag_accepts_intact(tmp_path):
    root = _make_bag(tmp_path / "bag", {"data/dataset/a.mat": b"aa", "data/dataset/b.mat": b"bb"})
    rep = kit.verify_bag(root, CFG)
    assert rep.ok and rep.checked == 2 and not rep.notes


def test_verify_bag_detects_altered_file(tmp_path):
    root = _make_bag(tmp_path / "bag", {"data/dataset/a.mat": b"aa", "data/dataset/b.mat": b"bb"})
    (root / "data/dataset/a.mat").write_bytes(b"XX")  # 大きさは同じ
    rep = kit.verify_bag(root, CFG)
    assert not rep.ok and rep.mismatched == ["data/dataset/a.mat"]


def test_verify_bag_detects_missing_file(tmp_path):
    root = _make_bag(tmp_path / "bag", {"data/dataset/a.mat": b"aa", "data/dataset/b.mat": b"bb"})
    (root / "data/dataset/b.mat").unlink()
    rep = kit.verify_bag(root, CFG)
    assert not rep.ok and rep.missing == ["data/dataset/b.mat"]


def test_verify_bag_allows_the_expected_unlisted_file(tmp_path):
    root = _make_bag(
        tmp_path / "bag",
        {"data/dataset/a.mat": b"aa", "data/dataset/b.mat": b"bb"},
        unlisted={"data/readme.txt": b"readme"},
    )
    rep = kit.verify_bag(root, CFG)
    assert rep.ok and rep.unlisted == ["data/readme.txt"] and not rep.notes


def test_verify_bag_flags_unexpected_extra_file(tmp_path):
    root = _make_bag(
        tmp_path / "bag",
        {"data/dataset/a.mat": b"aa", "data/dataset/b.mat": b"bb"},
        unlisted={"data/dataset/surprise.mat": b"?"},
    )
    rep = kit.verify_bag(root, CFG)
    assert any("surprise" in n for n in rep.notes)


def test_verify_bag_without_manifest(tmp_path):
    root = tmp_path / "bag"
    root.mkdir()
    rep = kit.verify_bag(root, CFG)
    assert not rep.ok and "manifest-md5.txt" in rep.missing


def test_quick_skips_digests(tmp_path):
    root = _make_bag(tmp_path / "bag", {"data/dataset/a.mat": b"aa", "data/dataset/b.mat": b"bb"})
    (root / "data/dataset/a.mat").write_bytes(b"XX")
    rep = kit.verify_bag(root, CFG, quick=True)
    assert rep.ok and any("quick" in n for n in rep.notes)


# ---------------------------------------------------------------------------
# tar の安全性 — 展開先の外に書かせない
# ---------------------------------------------------------------------------
def _evil_tar(path: Path) -> Path:
    with tarfile.open(path, "w") as tf:
        for name in ("../../escape.txt", "/etc/absolute.txt", "ok/normal.txt"):
            data = b"x"
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        ln = tarfile.TarInfo("ok/link")
        ln.type = tarfile.SYMTYPE
        ln.linkname = "/etc/passwd"
        tf.addfile(ln)
    return path


def test_unsafe_members_lists_every_kind(tmp_path):
    mod = _load_script()
    with tarfile.open(_evil_tar(tmp_path / "evil.tar")) as tf:
        bad = mod.unsafe_members(tf)
    joined = " ".join(bad)
    assert "escape.txt" in joined and "absolute" in joined and "link" in joined
    assert "normal.txt" not in joined


def test_extract_refuses_unsafe_tar_and_writes_nothing(tmp_path):
    mod = _load_script()
    dest = tmp_path / "dest"
    with pytest.raises(SystemExit):
        mod.extract(_evil_tar(tmp_path / "evil.tar"), dest)
    assert not any(dest.rglob("*")), "1 件も書かれてはいけない"
    assert not (tmp_path / "escape.txt").exists()


def test_extract_accepts_a_normal_tar(tmp_path):
    mod = _load_script()
    src = tmp_path / "bag"
    _make_bag(src, {"data/dataset/a.mat": b"aa"})
    tar = tmp_path / "good.tar"
    with tarfile.open(tar, "w") as tf:
        tf.add(src, arcname="bag")
    root = mod.extract(tar, tmp_path / "dest")
    assert root == tmp_path / "dest" / "bag"
    assert (root / "data/dataset/a.mat").read_bytes() == b"aa"


# ---------------------------------------------------------------------------
# 配布物そのものの照合
# ---------------------------------------------------------------------------
def test_check_archive_detects_size_and_digest(tmp_path):
    mod = _load_script()
    f = tmp_path / "a.tar"
    f.write_bytes(b"hello")
    good = {"size_bytes": 5, "md5": kit.file_md5(f)}
    assert mod.check_archive(f, good, quick=False)[0]

    ok, problems = mod.check_archive(f, {"size_bytes": 5, "md5": "0" * 32}, quick=False)
    assert not ok and any("MD5" in p for p in problems)

    ok, problems = mod.check_archive(f, {"size_bytes": 99, "md5": good["md5"]}, quick=False)
    assert not ok and any("大きさ" in p for p in problems)


# ---------------------------------------------------------------------------
# 外に出す情報を絞っていること
# ---------------------------------------------------------------------------
def test_network_is_only_used_in_the_download_path():
    """`--tar` / `--verify-only` で通信の準備すら起きないこと。

    requests は download() の中でだけ import する。module 直下にあると、
    ネットワークを使わないはずの経路でも読み込まれてしまう。
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    top_level = {
        n.names[0].name
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom)) and isinstance(n, ast.Import)
    }
    assert "requests" not in top_level and "urllib" not in top_level

    inside = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import) and any(a.name == "requests" for a in sub.names):
                    inside.append(node.name)
    assert inside == ["download"]


def test_user_agent_carries_no_host_or_user_info():
    mod = _load_script()
    ua = mod.USER_AGENT
    import getpass
    import socket

    assert socket.gethostname() not in ua
    assert getpass.getuser() not in ua
    assert ua == "near-miss-filtering/0.1"


def test_script_sends_no_credentials():
    """認証まわりの語がスクリプトに出てこないこと (公開データなので不要)。"""
    text = SCRIPT.read_text(encoding="utf-8")
    for word in ("Authorization", "api_key", "apikey", "token=", "password"):
        assert word not in text


def test_help_runs_without_network_or_data():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO / "src"), "PYTHONUTF8": "1"},
    )
    assert r.returncode == 0
    assert "--tar" in r.stdout and "--verify-only" in r.stdout


# ---------------------------------------------------------------------------
# 手元の実データ (あるときだけ)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_real_bag_verifies():
    root = REPO / "raw_data" / "kit_msdm" / "10.35097-44a91t97pmnha1k9"
    if not root.is_dir():
        pytest.skip("KIT MSDM が手元にない")
    rep = kit.verify_bag(root)
    assert rep.ok, rep.summary()
    assert rep.checked == kit.load_dataset_config()["bag"]["manifest_entries"]
