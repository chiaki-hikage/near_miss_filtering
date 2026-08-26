"""S3 取り込み経路の試験。

ここで確かめるのは「置くまで」の話だけ。判定ロジックには触れない。
実際の AWS には接続しない (下の StubS3 で代用する)。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from near_miss.io import s3_sync as s3

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# S3 クライアントの代役
# ---------------------------------------------------------------------------
class StubS3:
    """list_objects_v2 / head_object / download_file だけを持つ最小の代役。"""

    def __init__(self, objects: dict[str, int]):
        self.objects = dict(objects)  # key -> size
        self.blobs: dict[str, bytes] = {}  # 往復させたいときだけ中身も持つ
        self.downloaded: list[str] = []
        self.uploaded: list[str] = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        outer = self

        class _P:
            def paginate(self, Bucket, Prefix):  # noqa: N803 (boto3 の綴りに合わせる)
                hits = [
                    {"Key": k, "Size": v}
                    for k, v in sorted(outer.objects.items())
                    if k.startswith(Prefix)
                ]
                # 改ページも通ることを確かめたいので 2 件ずつ返す
                for i in range(0, max(len(hits), 1), 2):
                    yield {"Contents": hits[i : i + 2]}

        return _P()

    def head_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": self.objects[Key]}

    def download_file(self, bucket, key, path):
        self.downloaded.append(key)
        Path(path).write_bytes(self.blobs.get(key, b"x" * self.objects[key]))

    def upload_file(self, path, bucket, key):
        data = Path(path).read_bytes()
        self.objects[key] = len(data)
        self.blobs[key] = data
        self.uploaded.append(key)


LOC = s3.Location("verif-bucket", "near_miss/")


# ---------------------------------------------------------------------------
# バケットの指定
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "uri,bucket,prefix",
    [
        ("s3://my-bucket", "my-bucket", ""),
        ("s3://my-bucket/", "my-bucket", ""),
        ("s3://my-bucket/near_miss", "my-bucket", "near_miss/"),
        ("s3://my-bucket/a/b/", "my-bucket", "a/b/"),
        ("my-bucket/near_miss", "my-bucket", "near_miss/"),
    ],
)
def test_location_parse(uri, bucket, prefix):
    loc = s3.Location.parse(uri)
    assert (loc.bucket, loc.prefix) == (bucket, prefix)


@pytest.mark.parametrize("bad", ["", "s3://", "s3://ab", "s3://UPPER/x", "http://my-bucket/x"])
def test_location_parse_rejects_bad(bad):
    with pytest.raises(ValueError):
        s3.Location.parse(bad)


def test_resolve_location_precedence(tmp_path, monkeypatch):
    cfg = tmp_path / "s3.yaml"
    cfg.write_text("uri: s3://from-file/p\n", encoding="utf-8")

    monkeypatch.delenv(s3.ENV_URI, raising=False)
    assert s3.resolve_location(config_path=cfg).bucket == "from-file"

    monkeypatch.setenv(s3.ENV_URI, "s3://from-env/p")
    assert s3.resolve_location(config_path=cfg).bucket == "from-env"

    # 引数が最優先
    assert s3.resolve_location("s3://from-cli/p", config_path=cfg).bucket == "from-cli"


def test_resolve_location_errors_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv(s3.ENV_URI, raising=False)
    cfg = tmp_path / "s3.yaml"
    cfg.write_text('uri: ""\n', encoding="utf-8")
    with pytest.raises(ValueError, match="指定されていません"):
        s3.resolve_location(config_path=cfg)


def test_shipped_config_has_no_credentials():
    """配布する設定ファイルに鍵が書かれていないこと。"""
    text = s3.CONFIG_PATH.read_text(encoding="utf-8")
    for name in ("AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "aws_secret_access_key"):
        assert name not in text
    # バケット名もハードコードしない (既定は空)
    import yaml

    cfg = yaml.safe_load(text) or {}
    assert not (cfg.get("uri") or "").strip()


# ---------------------------------------------------------------------------
# 取り込み先の決定
# ---------------------------------------------------------------------------
def test_dest_mapping_keeps_existing_layout(tmp_path):
    """S3 のキーが raw_data の既存の構成にそのまま落ちること。"""
    assert s3.dest_for(
        "segments/abc/route/3/rlog.zst", "comma_car_segments", tmp_path
    ) == tmp_path / "comma_car_segments" / "segments" / "abc" / "route" / "3" / "rlog.zst"

    assert s3.dest_for(
        "10.35097-44a91t97pmnha1k9/data/dataset/x.mat", "kit_msdm", tmp_path
    ) == tmp_path / "kit_msdm" / "10.35097-44a91t97pmnha1k9" / "data" / "dataset" / "x.mat"

    # comma2k19 のチャンクは raw_data 直下 (raw_data/Chunk_1/...)
    assert s3.dest_for("Chunk_1/d|t/5/processed_log/CAN/speed/t", ".", tmp_path) == (
        tmp_path / "Chunk_1" / "d|t" / "5" / "processed_log" / "CAN" / "speed" / "t"
    )


@pytest.mark.parametrize("key", ["../outside", "a/../../outside", "/../x"])
def test_dest_rejects_escape(tmp_path, key):
    with pytest.raises(ValueError):
        s3.dest_for(key, "comma_car_segments", tmp_path)


def test_dest_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        s3.dest_for("/", "comma_car_segments", tmp_path)


# ---------------------------------------------------------------------------
# 計画づくり
# ---------------------------------------------------------------------------
def test_plan_prefix_skips_present_files(tmp_path):
    client = StubS3(
        {
            "near_miss/kit_msdm/a.mat": 100,
            "near_miss/kit_msdm/sub/b.mat": 200,
            "near_miss/kit_msdm/sub/": 0,  # ディレクトリ標識
            "near_miss/other/c.mat": 300,  # 対象外
        }
    )
    plan = s3.plan_prefix(client, LOC, s3.DATASETS["kit-msdm"], raw_data=tmp_path)
    assert len(plan.items) == 2
    assert plan.total_bytes == 300
    assert len(plan.todo) == 2

    # 同じ大きさで既にあるものは取り直さない
    dest = tmp_path / "kit_msdm" / "a.mat"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"x" * 100)
    plan2 = s3.plan_prefix(client, LOC, s3.DATASETS["kit-msdm"], raw_data=tmp_path)
    assert len(plan2.todo) == 1 and len(plan2.have) == 1
    assert plan2.todo_bytes == 200

    # 大きさが違えば取り直す (途中で切れたファイルを拾う)
    dest.write_bytes(b"x" * 7)
    assert len(s3.plan_prefix(client, LOC, s3.DATASETS["kit-msdm"], raw_data=tmp_path).todo) == 2


def test_plan_prefix_scope_limits_to_chunk(tmp_path):
    client = StubS3(
        {
            "near_miss/comma2k19/Chunk_1/d/1/processed_log/CAN/speed/t": 10,
            "near_miss/comma2k19/Chunk_2/d/1/processed_log/CAN/speed/t": 20,
        }
    )
    plan = s3.plan_prefix(
        client, LOC, s3.DATASETS["comma2k19"], scope="Chunk_1", raw_data=tmp_path
    )
    assert [i.dest.relative_to(tmp_path).parts[0] for i in plan.items] == ["Chunk_1"]
    assert plan.total_bytes == 10


def _write_database(tmp_path, mapping):
    cache = tmp_path / "comma_car_segments"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "database.json").write_text(json.dumps(mapping), encoding="utf-8")
    return cache


def test_plan_car_segments_selects_only_requested(tmp_path):
    """車種 / route 単位の絞り込みが、そのままキーの選択になること。"""
    from near_miss.io import comma_car_segments as ccs

    dongle = "a" * 16
    names = [f"{dongle}/00000001--aaaa/{i}" for i in range(4)]
    other = [f"{dongle}/00000002--bbbb/{i}" for i in range(3)]
    _write_database(tmp_path, {"TOYOTA_RAV4_TSS2": names + other})

    objects = {"near_miss/comma_car_segments/database.json": 9}
    for n in names + other:
        objects[f"near_miss/comma_car_segments/segments/{n}/rlog.zst"] = 1_380_000
    client = StubS3(objects)

    picked = ccs.select_segments(
        "TOYOTA_RAV4_TSS2", routes=1, per_route=2, cache_dir=tmp_path / "comma_car_segments"
    )
    assert len(picked) == 2

    plan = s3.plan_car_segments(client, LOC, picked, raw_data=tmp_path)
    segs = [i for i in plan.items if i.key.endswith("rlog.zst")]
    assert len(segs) == 2, "選んだセグメントだけが対象になるはず"
    assert any(i.key.endswith("database.json") for i in plan.items)
    assert not plan.notes


def test_plan_car_segments_reports_missing_keys(tmp_path):
    dongle = "b" * 16
    names = [f"{dongle}/00000001--aaaa/{i}" for i in range(3)]
    client = StubS3(
        {
            "near_miss/comma_car_segments/database.json": 9,
            f"near_miss/comma_car_segments/segments/{names[0]}/rlog.zst": 100,
        }
    )
    plan = s3.plan_car_segments(client, LOC, names, raw_data=tmp_path)
    assert plan.notes and "S3 に無いセグメント" in plan.notes[0]
    assert len([i for i in plan.items if i.key.endswith("rlog.zst")]) == 1


# ---------------------------------------------------------------------------
# 取り込み
# ---------------------------------------------------------------------------
def test_download_writes_via_part_file(tmp_path):
    client = StubS3({"near_miss/kit_msdm/a.mat": 32})
    plan = s3.plan_prefix(client, LOC, s3.DATASETS["kit-msdm"], raw_data=tmp_path)
    s3.download(client, plan, workers=2)
    dest = tmp_path / "kit_msdm" / "a.mat"
    assert dest.is_file() and dest.stat().st_size == 32
    assert not list(tmp_path.rglob("*.part")), ".part の残骸を残さない"


def test_download_failure_leaves_no_partial(tmp_path):
    class Broken(StubS3):
        def download_file(self, bucket, key, path):
            Path(path).write_bytes(b"half")
            raise RuntimeError("接続が切れました")

    client = Broken({"near_miss/kit_msdm/a.mat": 32})
    plan = s3.plan_prefix(client, LOC, s3.DATASETS["kit-msdm"], raw_data=tmp_path)
    results = s3.download(client, plan)
    assert isinstance(list(results.values())[0], Exception)
    assert not (tmp_path / "kit_msdm" / "a.mat").exists()
    assert not list(tmp_path.rglob("*.part"))


def test_download_does_not_touch_present_files(tmp_path):
    client = StubS3({"near_miss/kit_msdm/a.mat": 4})
    dest = tmp_path / "kit_msdm" / "a.mat"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"keep")
    plan = s3.plan_prefix(client, LOC, s3.DATASETS["kit-msdm"], raw_data=tmp_path)
    s3.download(client, plan)
    assert client.downloaded == [], "既にあるものは取り直さない"
    assert dest.read_bytes() == b"keep"



# ---------------------------------------------------------------------------
# 送り込み (バケットを埋める側)
# ---------------------------------------------------------------------------
def _make_tree(root: Path) -> dict[str, bytes]:
    """raw_data の形をした小さな木を作る。戻り値は相対パス -> 中身。"""
    files = {
        "kit_msdm/10.35097-44a91t97pmnha1k9/data/dataset/parameter.m": b"param",
        "kit_msdm/10.35097-44a91t97pmnha1k9/data/dataset/run1.mat": b"matdata",
    }
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return files


def test_key_for_is_inverse_of_dest_for(tmp_path):
    """上げる場所と取りに行く場所がずれないこと。**これが崩れると往復できない。**"""
    for ds in s3.DATASETS.values():
        rel = "Chunk_1/a/b.txt" if ds.local_root == "." else "x/y/z.bin"
        dest = s3.dest_for(rel, ds.local_root, tmp_path)
        key = s3.key_for(dest, ds, LOC, tmp_path)
        assert key == LOC.key(ds.s3_subprefix + rel)
        # キーから戻すと同じ場所に来る
        back = s3.dest_for(key[len(LOC.key(ds.s3_subprefix)) :], ds.local_root, tmp_path)
        assert back == dest


def test_upload_then_download_round_trips(tmp_path):
    """上げてから別の場所へ取り込むと、木も中身もそのまま復元されること。"""
    src = tmp_path / "src"
    src.mkdir()
    files = _make_tree(src)
    client = StubS3({})

    up = s3.plan_upload(client, LOC, s3.DATASETS["kit-msdm"], raw_data=src)
    assert len(up.todo) == len(files)
    s3.upload(client, up)

    dst = tmp_path / "dst"
    down = s3.plan_prefix(client, LOC, s3.DATASETS["kit-msdm"], raw_data=dst)
    s3.download(client, down)

    got = {
        p.relative_to(dst).as_posix(): p.read_bytes() for p in dst.rglob("*") if p.is_file()
    }
    assert got == files


def test_upload_skips_os_junk_and_partials(tmp_path):
    src = tmp_path / "src"
    base = src / "kit_msdm" / "10.35097-44a91t97pmnha1k9" / "data" / "dataset"
    base.mkdir(parents=True)
    (base / "run1.mat").write_bytes(b"ok")
    (base / ".DS_Store").write_bytes(b"junk")
    (base / "run2.mat.part").write_bytes(b"half")

    plan = s3.plan_upload(StubS3({}), LOC, s3.DATASETS["kit-msdm"], raw_data=src)
    names = sorted(Path(i.key).name for i in plan.items)
    assert names == ["run1.mat"]


def test_upload_is_idempotent(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _make_tree(src)
    client = StubS3({})
    s3.upload(client, s3.plan_upload(client, LOC, s3.DATASETS["kit-msdm"], raw_data=src))
    n = len(client.uploaded)
    assert n == 2

    # 2 回目は何も上げない
    again = s3.plan_upload(client, LOC, s3.DATASETS["kit-msdm"], raw_data=src)
    assert again.todo == [] and len(again.have) == 2
    s3.upload(client, again)
    assert len(client.uploaded) == n


def test_upload_replaces_size_mismatch(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _make_tree(src)
    client = StubS3({})
    s3.upload(client, s3.plan_upload(client, LOC, s3.DATASETS["kit-msdm"], raw_data=src))

    # 手元のファイルが増えたら上げ直す
    f = src / "kit_msdm/10.35097-44a91t97pmnha1k9/data/dataset/run1.mat"
    f.write_bytes(b"matdata-longer")
    plan = s3.plan_upload(client, LOC, s3.DATASETS["kit-msdm"], raw_data=src)
    assert len(plan.todo) == 1 and plan.todo[0].dest == f


def test_upload_scope_limits_to_chunk(tmp_path):
    src = tmp_path / "src"
    for chunk in ("Chunk_1", "Chunk_2"):
        p = src / chunk / "drive" / "1" / "processed_log" / "CAN" / "speed" / "t"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"t")
    plan = s3.plan_upload(
        StubS3({}), LOC, s3.DATASETS["comma2k19"], scope="Chunk_1", raw_data=src
    )
    assert plan.items and all("comma2k19/Chunk_1/" in i.key for i in plan.items)


def test_upload_notes_missing_local_tree(tmp_path):
    plan = s3.plan_upload(StubS3({}), LOC, s3.DATASETS["kit-msdm"], raw_data=tmp_path)
    assert plan.items == [] and plan.notes


def test_upload_selection_matches_fetch_selection(tmp_path):
    """同じ引数なら、上げる範囲と取りに行く範囲が一致すること。"""
    from near_miss.io import comma_car_segments as ccs

    dongle = "c" * 16
    names = [f"{dongle}/0000000{r}--aaaa/{i}" for r in (1, 2) for i in range(5)]
    cache = _write_database(tmp_path, {"TOYOTA_RAV4_TSS2": names})
    for n in names:
        p = ccs.local_path(n, cache)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"z" * 100)

    picked = ccs.select_segments("TOYOTA_RAV4_TSS2", routes=1, per_route=3, cache_dir=cache)
    client = StubS3({})

    up = s3.plan_upload(
        client, LOC, s3.DATASETS["car-segments"], raw_data=tmp_path,
        sources=[cache / "database.json"] + [ccs.local_path(n, cache) for n in picked],
    )
    s3.upload(client, up)

    down = s3.plan_car_segments(client, LOC, picked, raw_data=tmp_path / "elsewhere")
    up_keys = {i.key for i in up.items}
    down_keys = {i.key for i in down.items}
    assert up_keys == down_keys, "上げた範囲と取りに行く範囲が食い違っている"
    assert not down.notes


def test_upload_cli_refuses_outside_ec2():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "upload_to_s3.py"), "kit-msdm", "--dry-run"],
        capture_output=True, text=True, cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO / "src"), "PYTHONUTF8": "1"},
    )
    if s3.is_ec2():
        pytest.skip("EC2 の上なのでこの確認はできない")
    assert r.returncode != 0
    assert "EC2" in (r.stderr + r.stdout)

# ---------------------------------------------------------------------------
# 認証情報
# ---------------------------------------------------------------------------
class _Creds:
    def __init__(self, method):
        self.method = method


class _Session:
    def __init__(self, method):
        self._m = method

    def get_credentials(self):
        return _Creds(self._m) if self._m else None


@pytest.mark.parametrize("method", sorted(s3.STATIC_CREDENTIAL_METHODS))
def test_static_credentials_rejected(method, monkeypatch):
    monkeypatch.setattr(s3, "assert_no_env_file_credentials", lambda: None)
    with pytest.raises(PermissionError, match="静的な認証情報"):
        s3.verify_credentials(_Session(method))
    # 明示的に許した場合だけ通る
    assert s3.verify_credentials(_Session(method), allow_any=True).is_static


@pytest.mark.parametrize("method", ["iam-role", "container-role", "sso"])
def test_role_credentials_accepted(method, monkeypatch):
    monkeypatch.setattr(s3, "assert_no_env_file_credentials", lambda: None)
    info = s3.verify_credentials(_Session(method))
    assert info.is_role and not info.is_static


def test_missing_credentials_message(monkeypatch):
    monkeypatch.setattr(s3, "assert_no_env_file_credentials", lambda: None)
    with pytest.raises(PermissionError, match="IAM Role"):
        s3.verify_credentials(_Session(None))


def test_repo_has_no_env_file_with_credentials():
    """リポジトリに鍵入りの .env が紛れ込んでいないこと。"""
    s3.assert_no_env_file_credentials()


# ---------------------------------------------------------------------------
# 解析側から切り離されていること
# ---------------------------------------------------------------------------
def test_analysis_modules_do_not_import_s3():
    """S3 は「置くまで」の経路。判定に関わるモジュールから触らない。"""
    targets = [
        "features.py", "detectors.py", "sideslip.py", "scoring.py",
        "signals.py", "pipeline.py", "parallel.py", "sources.py",
    ]
    for name in targets:
        text = (REPO / "src" / "near_miss" / name).read_text(encoding="utf-8")
        assert "s3_sync" not in text and "boto3" not in text, f"{name} が S3 を参照している"

    for name in ("screen_sideslip.py", "benchmark_workers.py", "validate_sideslip_filter.py"):
        text = (REPO / "scripts" / name).read_text(encoding="utf-8")
        assert "s3_sync" not in text and "boto3" not in text, f"{name} が S3 を参照している"


def test_boto3_is_optional():
    """boto3 が無くても解析側は動くこと (必須依存にしない)。"""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    deps = text.split("[project.optional-dependencies]")[0]
    assert "boto3" not in deps, "boto3 を必須依存に入れない"
    assert "boto3" in text.split("[project.optional-dependencies]")[1]


def test_cli_refuses_outside_ec2():
    """Mac では既定で止まること (S3 は EC2 運用に限る)。"""
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "fetch_from_s3.py"), "car-segments", "--dry-run"],
        capture_output=True, text=True, cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO / "src"), "PYTHONUTF8": "1"},
    )
    if s3.is_ec2():
        pytest.skip("EC2 の上なのでこの確認はできない")
    assert r.returncode != 0
    assert "EC2" in (r.stderr + r.stdout)


def test_cli_show_layout_needs_no_aws():
    """対応表の表示に AWS も boto3 も要らないこと。"""
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "fetch_from_s3.py"), "--show-layout"],
        capture_output=True, text=True, cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO / "src"), "PYTHONUTF8": "1"},
    )
    assert r.returncode == 0
    assert "raw_data/comma_car_segments/" in r.stdout
