"""S3 に置いた検証用データを raw_data/ へ取り込む (L0 の手前)。

**EC2 でのみ使う経路。** Mac ではこれまで通りローカルの `raw_data/` を直接読むので、
解析コードはこのモジュールを import しない。ここは「データを置くまで」の話で、
置いたあとの読み出し・判定は S3 を使ったかどうかを知らない。

方針:

* **S3 側のキー構成を `raw_data/` の下と 1:1 にする。** 対応は `DATASETS` の 1 か所だけ。
  こうすると取り込んだあとのパス解決が今までと同じになり、解析側に手を入れずに済む。
* **認証情報はコードにも設定ファイルにも置かない。** boto3 の既定の解決順に任せる。
  EC2 ではインスタンスプロファイル (IAM Role) が使われる。静的な鍵が使われていたら
  `verify_credentials()` が止める。
* **既にあるものは触らない。** 大きさが一致するファイルは飛ばす。書くときは `.part` へ
  落としてから置き換えるので、途中で切れた残骸が `raw_data/` に残らない。
* バケット名・プレフィックスはハードコードしない (`configs/datasets/s3.yaml` / 環境変数 /
  コマンドライン引数)。
"""

from __future__ import annotations

import logging
import os
import platform
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from ..config import REPO_ROOT

log = logging.getLogger(__name__)

RAW_DATA = REPO_ROOT / "raw_data"
CONFIG_PATH = REPO_ROOT / "configs" / "datasets" / "s3.yaml"
ENV_URI = "NEAR_MISS_S3_URI"

# 静的な鍵とみなす boto3 の解決元。EC2 ではこれらを使わせない。
STATIC_CREDENTIAL_METHODS = frozenset(
    {"env", "environment", "shared-credentials-file", "credentials-file", "config-file"}
)
# インスタンス / コンテナ / SSO 由来。鍵がディスクに残らないもの。
ROLE_CREDENTIAL_METHODS = frozenset(
    {"iam-role", "container-role", "assume-role", "assume-role-with-web-identity", "sso"}
)


# ---------------------------------------------------------------------------
# データセットごとの対応表
#
# ここが S3 側にデータを置く人との唯一の取り決め。ここ以外に対応関係を書かない。
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dataset:
    name: str  # コマンドラインでの名前
    s3_subprefix: str  # バケット側。ルートプレフィックスの下に付く
    local_root: str  # raw_data からの相対。"." は raw_data 直下
    note: str
    # 中身の場所を絞る引数 (comma2k19 のチャンクなど)。無ければ丸ごと。
    scope_hint: str = ""


DATASETS: dict[str, Dataset] = {
    "car-segments": Dataset(
        name="car-segments",
        s3_subprefix="comma_car_segments/",
        local_root="comma_car_segments",
        note="commaCarSegments。横滑りフィルタの主対象",
        scope_hint="--platform / --routes / --per-route / --limit で車種・route 単位に絞れる",
    ),
    "kit-msdm": Dataset(
        name="kit-msdm",
        s3_subprefix="kit_msdm/",
        local_root="kit_msdm",
        note="KIT MSDM。再現率の物差し (約 172 MB)",
    ),
    "comma2k19": Dataset(
        name="comma2k19",
        s3_subprefix="comma2k19/",
        local_root=".",  # チャンクが raw_data 直下に来る (raw_data/Chunk_1/...)
        note="comma2k19。チャンク 1 本で約 9.7 GB",
        scope_hint="--chunk Chunk_1 のようにチャンク単位で絞る",
    ),
}


def layout_table() -> str:
    """S3 側の置き方と取り込み先の対応を人が読める形で返す。"""
    lines = [
        "S3 のキー構成は raw_data/ の下と 1:1 に対応させる。",
        "",
        f"  {'名前':<14} {'S3 (ルートプレフィックスの下)':<34} 取り込み先",
    ]
    for ds in DATASETS.values():
        local = "raw_data/" if ds.local_root == "." else f"raw_data/{ds.local_root}/"
        lines.append(f"  {ds.name:<14} {ds.s3_subprefix:<34} {local}")
    lines += [
        "",
        "例 (ルートプレフィックスを s3://my-bucket/near_miss/ とした場合):",
        "",
        "  s3://my-bucket/near_miss/comma_car_segments/database.json",
        "      -> raw_data/comma_car_segments/database.json",
        "  s3://my-bucket/near_miss/comma_car_segments/segments/<dongle>/<route>/<n>/rlog.zst",
        "      -> raw_data/comma_car_segments/segments/<dongle>/<route>/<n>/rlog.zst",
        "  s3://my-bucket/near_miss/kit_msdm/10.35097-44a91t97pmnha1k9/data/dataset/*.mat",
        "      -> raw_data/kit_msdm/10.35097-44a91t97pmnha1k9/data/dataset/*.mat",
        "  s3://my-bucket/near_miss/comma2k19/Chunk_1/<drive>/<n>/processed_log/...",
        "      -> raw_data/Chunk_1/<drive>/<n>/processed_log/...",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# バケットの指定
# ---------------------------------------------------------------------------
_S3_URI_RE = re.compile(r"^s3://(?P<bucket>[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9])(?:/(?P<prefix>.*))?$")


@dataclass(frozen=True)
class Location:
    """バケットと、その中のルートプレフィックス。"""

    bucket: str
    prefix: str = ""  # 空か、末尾 "/" 付き

    @classmethod
    def parse(cls, uri: str) -> "Location":
        uri = uri.strip()
        if not uri.startswith("s3://"):
            uri = "s3://" + uri.lstrip("/")
        m = _S3_URI_RE.match(uri)
        if m is None:
            raise ValueError(f"S3 の URI として読めません: {uri!r} (例 s3://my-bucket/near_miss)")
        prefix = (m.group("prefix") or "").strip("/")
        return cls(m.group("bucket"), f"{prefix}/" if prefix else "")

    def key(self, rel: str) -> str:
        return f"{self.prefix}{rel.lstrip('/')}"

    def uri(self, rel: str = "") -> str:
        return f"s3://{self.bucket}/{self.key(rel)}"


def resolve_location(cli_value: str | None = None, config_path: Path | None = None) -> Location:
    """バケットの指定を決める。優先順は 引数 > 環境変数 > 設定ファイル。

    バケット名は秘密ではないので設定ファイルに書いてよい。**鍵は書かない。**
    """
    if cli_value:
        return Location.parse(cli_value)
    env = os.environ.get(ENV_URI, "").strip()
    if env:
        return Location.parse(env)

    path = config_path or CONFIG_PATH
    if path.is_file():
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        uri = str(cfg.get("uri") or "").strip()
        if uri:
            return Location.parse(uri)
        bucket = str(cfg.get("bucket") or "").strip()
        if bucket:
            prefix = str(cfg.get("prefix") or "").strip()
            return Location.parse(f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}")

    raise ValueError(
        "S3 バケットが指定されていません。次のいずれかで指定してください:\n"
        "  --bucket s3://my-bucket/near_miss\n"
        f"  export {ENV_URI}=s3://my-bucket/near_miss\n"
        f"  {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path} に uri: を書く"
    )


# ---------------------------------------------------------------------------
# 実行環境と認証情報の確認
# ---------------------------------------------------------------------------
def is_ec2() -> bool:
    """EC2 の上かどうか。**この判定はここにしか無い** (解析側には無い)。

    DMI だけを見る。ネットワーク (IMDS) は叩かない — 待ち時間が読めないため。
    """
    if platform.system() != "Linux":
        return False
    for path in ("/sys/devices/virtual/dmi/id/product_uuid", "/sys/hypervisor/uuid"):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if f.read(3).lower() == "ec2":
                    return True
        except OSError:
            continue
    try:
        with open("/sys/devices/virtual/dmi/id/sys_vendor", "r", encoding="utf-8") as f:
            return "amazon" in f.read().lower()
    except OSError:
        return False


def _repo_env_files() -> list[Path]:
    return [p for p in (REPO_ROOT / ".env", REPO_ROOT / ".env.local") if p.is_file()]


def assert_no_env_file_credentials() -> None:
    """リポジトリ内の .env に AWS の鍵が書かれていないことを確かめる。

    「認証情報はコードや .env に保存しない」を機械で担保するための確認。
    """
    bad = []
    for path in _repo_env_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in ("AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "AWS_SESSION_TOKEN"):
            if re.search(rf"^\s*(export\s+)?{name}\s*=", text, re.MULTILINE):
                bad.append(f"{path.name}: {name}")
    if bad:
        raise PermissionError(
            "リポジトリ内の .env に AWS の認証情報が書かれています。取り除いてください:\n  "
            + "\n  ".join(bad)
            + "\nEC2 では IAM Role (インスタンスプロファイル) を使います。鍵は要りません。"
        )


@dataclass(frozen=True)
class Credentials:
    method: str
    is_role: bool
    is_static: bool


def verify_credentials(session, allow_any: bool = False) -> Credentials:
    """boto3 が何から認証情報を取ったかを調べ、静的な鍵なら止める。"""
    assert_no_env_file_credentials()

    creds = session.get_credentials()
    if creds is None:
        raise PermissionError(
            "AWS の認証情報が見つかりません。\n"
            "  EC2: インスタンスに IAM Role (S3 読み取り) を付けてください。\n"
            "  それ以外: aws sso login などで一時的な認証情報を用意してください。"
        )
    method = getattr(creds, "method", "") or "unknown"
    info = Credentials(
        method=method,
        is_role=method in ROLE_CREDENTIAL_METHODS,
        is_static=method in STATIC_CREDENTIAL_METHODS,
    )
    if info.is_static and not allow_any:
        raise PermissionError(
            f"静的な認証情報 ({method}) が使われようとしています。\n"
            "EC2 では IAM Role を使ってください (鍵をインスタンスに置かない)。\n"
            "意図的に別の認証を使う場合のみ --allow-any-credentials を付けてください。"
        )
    return info


def make_client(region: str | None = None, profile: str | None = None, allow_any: bool = False):
    """S3 クライアントと、使われた認証情報の素性を返す。

    client はスレッド安全なので、この 1 個を並列ダウンロードで共有してよい
    (スレッド安全でないのは resource の方)。
    """
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - 依存が無いときだけ
        raise ImportError(
            "boto3 がありません。EC2 で S3 から取り込む場合のみ必要です:\n"
            "  uv sync --extra s3"
        ) from exc

    session = boto3.session.Session(profile_name=profile) if profile else boto3.session.Session()
    info = verify_credentials(session, allow_any=allow_any)
    return session.client("s3", region_name=region), info


# ---------------------------------------------------------------------------
# 取り込む対象を決める (ダウンロードはしない)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Item:
    """ローカル側と S3 側の対。取り込みにも送り込みにも同じ形を使う。"""

    key: str  # S3 側
    dest: Path  # ローカル側 (取り込みでは書き先、送り込みでは読み元)
    size: int
    present: bool  # 相手側に同じ大きさで既にある = 転送しない


DOWNLOAD = "download"
UPLOAD = "upload"


@dataclass
class Plan:
    dataset: str
    location: Location
    s3_prefix: str
    items: list[Item] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    direction: str = DOWNLOAD

    @property
    def todo(self) -> list[Item]:
        return [i for i in self.items if not i.present]

    @property
    def have(self) -> list[Item]:
        return [i for i in self.items if i.present]

    @property
    def todo_bytes(self) -> int:
        return sum(i.size for i in self.todo)

    @property
    def total_bytes(self) -> int:
        return sum(i.size for i in self.items)


def dest_for(rel_key: str, local_root: str, raw_data: Path = RAW_DATA) -> Path:
    """S3 のキー (プレフィックスを除いた相対部分) から取り込み先を決める。

    `..` などで `raw_data/` の外へ出るキーは弾く。S3 のキーは任意の文字列を
    取りうるので、ここで必ず確かめる。
    """
    rel = rel_key.strip("/")
    if not rel:
        raise ValueError("空のキーです")
    parts = [p for p in rel.split("/") if p]
    if any(p in ("..", ".") for p in parts):
        raise ValueError(f"'..' を含むキーは受け付けません: {rel_key!r}")
    base = (raw_data if local_root == "." else raw_data / local_root).resolve()
    dest = (base / Path(*parts)).resolve()
    # シンボリックリンク越しでも外へ出ないことを確かめる
    if not dest.is_relative_to(base):
        raise ValueError(f"取り込み先の外を指すキーです: {rel_key!r}")
    return dest


def _present(dest: Path, size: int) -> bool:
    """手元にあって大きさが一致すれば取り直さない (元データを上書きしない)。"""
    try:
        return dest.is_file() and dest.stat().st_size == size
    except OSError:
        return False


def iter_objects(client, location: Location, rel_prefix: str) -> Iterable[dict]:
    """プレフィックスの下のオブジェクトを列挙する。ディレクトリ標識は飛ばす。"""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=location.bucket, Prefix=location.key(rel_prefix)):
        for obj in page.get("Contents", []) or []:
            if obj["Key"].endswith("/"):
                continue
            yield obj


def plan_prefix(
    client,
    location: Location,
    dataset: Dataset,
    scope: str = "",
    raw_data: Path = RAW_DATA,
) -> Plan:
    """プレフィックスの下を丸ごと取り込む計画を作る (kit-msdm / comma2k19 用)。"""
    rel_prefix = dataset.s3_subprefix + (scope.strip("/") + "/" if scope.strip("/") else "")
    plan = Plan(dataset=dataset.name, location=location, s3_prefix=rel_prefix)
    for obj in iter_objects(client, location, rel_prefix):
        rel = obj["Key"][len(location.key(rel_prefix)) :]
        if not rel:
            continue
        dest = dest_for(f"{scope.strip('/')}/{rel}" if scope.strip("/") else rel,
                        dataset.local_root, raw_data)
        plan.items.append(Item(obj["Key"], dest, int(obj["Size"]), _present(dest, int(obj["Size"]))))
    if not plan.items:
        plan.notes.append(f"{location.uri(rel_prefix)} にオブジェクトがありません")
    return plan


def plan_car_segments(
    client,
    location: Location,
    names: list[str],
    raw_data: Path = RAW_DATA,
    with_database: bool = True,
) -> Plan:
    """commaCarSegments を **選んだセグメントだけ** 取り込む計画を作る。

    `names` は `select_segments()` が返すセグメント名 (車種・route 単位で選べる)。
    バケット全体を列挙すると数十万件になるので、route ごとのプレフィックスだけを見る。
    列挙の回数は route 数と同じ。
    """
    from .comma_car_segments import SegmentName

    ds = DATASETS["car-segments"]
    plan = Plan(dataset=ds.name, location=location, s3_prefix=ds.s3_subprefix)

    if with_database:
        rel = "database.json"
        key = location.key(ds.s3_subprefix + rel)
        try:
            head = client.head_object(Bucket=location.bucket, Key=key)
            dest = dest_for(rel, ds.local_root, raw_data)
            size = int(head["ContentLength"])
            plan.items.append(Item(key, dest, size, _present(dest, size)))
        except Exception as exc:  # 無ければ HuggingFace から取れるので致命的ではない
            plan.notes.append(f"database.json が S3 にありません ({exc.__class__.__name__})")

    wanted = {SegmentName.parse(n).rel_path.as_posix(): n for n in names}
    routes = sorted({p.rsplit("/", 2)[0] for p in wanted})  # <dongle>/<route>
    seen: set[str] = set()
    for route in routes:
        rel_prefix = f"{ds.s3_subprefix}segments/{route}/"
        for obj in iter_objects(client, location, rel_prefix):
            rel = obj["Key"][len(location.key(ds.s3_subprefix)) :]  # segments/<dongle>/...
            seg_rel = rel[len("segments/") :]
            if seg_rel not in wanted:
                continue
            seen.add(seg_rel)
            dest = dest_for(rel, ds.local_root, raw_data)
            size = int(obj["Size"])
            plan.items.append(Item(obj["Key"], dest, size, _present(dest, size)))

    missing = sorted(set(wanted) - seen)
    if missing:
        plan.notes.append(
            f"S3 に無いセグメントが {len(missing)} 件あります (例 {missing[0]})。"
            "バケットの中身が database.json より少ない可能性があります"
        )
    return plan


# ---------------------------------------------------------------------------
# 取り込み
# ---------------------------------------------------------------------------
def download_item(client, bucket: str, item: Item) -> Path:
    """1 件落とす。`.part` へ書いてから置き換えるので、途中の残骸を残さない。"""
    item.dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = item.dest.with_suffix(item.dest.suffix + ".part")
    try:
        client.download_file(bucket, item.key, str(tmp))
        tmp.replace(item.dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return item.dest


def download(
    client,
    plan: Plan,
    workers: int = 8,
    on_done: Callable[[Item, Path | Exception], None] | None = None,
) -> dict[str, Path | Exception]:
    """計画のうち手元に無いものを落とす。1 件の失敗で全体を止めない。"""
    out: dict[str, Path | Exception] = {}
    todo = plan.todo
    if not todo:
        return out
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(download_item, client, plan.location.bucket, i): i for i in todo}
        for fut in as_completed(futs):
            item = futs[fut]
            try:
                out[item.key] = fut.result()
            except Exception as exc:
                out[item.key] = exc
                log.warning("取り込めませんでした %s: %s", item.key, exc)
            if on_done is not None:
                on_done(item, out[item.key])
    return out


# ---------------------------------------------------------------------------
# 送り込み (バケットを埋める側)
#
# 取り込みと同じ対応表を逆向きに使う。対応が 1 か所にしか無いので、
# 上げた場所と取りに行く場所がずれない。
# ---------------------------------------------------------------------------
# バケットに上げないもの。OS が勝手に作るものと、転送途中の残骸。
SKIP_NAMES = frozenset({".DS_Store", "Thumbs.db", ".gitkeep"})
SKIP_SUFFIXES = (".part", ".tmp")


def local_base(dataset: Dataset, raw_data: Path = RAW_DATA) -> Path:
    """そのデータセットがローカルで根を張る場所。"""
    return raw_data if dataset.local_root == "." else raw_data / dataset.local_root


def key_for(local_file: Path, dataset: Dataset, location: Location, raw_data: Path = RAW_DATA) -> str:
    """ローカルのファイルから S3 のキーを決める。`dest_for` の逆。"""
    rel = local_file.resolve().relative_to(local_base(dataset, raw_data).resolve()).as_posix()
    return location.key(dataset.s3_subprefix + rel)


def _skip(path: Path) -> bool:
    return path.name in SKIP_NAMES or path.name.endswith(SKIP_SUFFIXES)


def plan_upload(
    client,
    location: Location,
    dataset: Dataset,
    scope: str = "",
    raw_data: Path = RAW_DATA,
    sources: Iterable[Path] | None = None,
) -> Plan:
    """ローカルにあるものをバケットへ上げる計画を作る。

    S3 側に同じ大きさで既に載っているものは飛ばす。何度流してもよい。
    `sources` を渡すと、その一覧だけを対象にする (車種 / route 単位の送り込み)。
    """
    base = local_base(dataset, raw_data)
    rel_prefix = dataset.s3_subprefix + (scope.strip("/") + "/" if scope.strip("/") else "")
    plan = Plan(dataset=dataset.name, location=location, s3_prefix=rel_prefix, direction=UPLOAD)

    if sources is None:
        scan_root = base / scope.strip("/") if scope.strip("/") else base
        if not scan_root.is_dir():
            plan.notes.append(f"{scan_root} がありません。先に手元へ用意してください")
            return plan
        files = sorted(f for f in scan_root.rglob("*") if f.is_file())
    else:
        files = sorted(Path(f) for f in sources)

    remote = {o["Key"]: int(o["Size"]) for o in iter_objects(client, location, rel_prefix)}

    missing_local = 0
    for f in files:
        if _skip(f):
            continue
        if not f.is_file():
            missing_local += 1
            continue
        key = key_for(f, dataset, location, raw_data)
        size = f.stat().st_size
        plan.items.append(Item(key, f, size, remote.get(key) == size))

    if missing_local:
        plan.notes.append(f"手元に無いファイルが {missing_local} 件あります (上げる前に取得が要ります)")
    if not plan.items and not plan.notes:
        plan.notes.append(f"{base} に上げるものがありません")
    return plan


def upload_item(client, bucket: str, item: Item) -> Path:
    client.upload_file(str(item.dest), bucket, item.key)
    return item.dest


def upload(
    client,
    plan: Plan,
    workers: int = 8,
    on_done: Callable[[Item, Path | Exception], None] | None = None,
) -> dict[str, Path | Exception]:
    """計画のうち S3 側に無いものを上げる。1 件の失敗で全体を止めない。"""
    out: dict[str, Path | Exception] = {}
    todo = plan.todo
    if not todo:
        return out
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(upload_item, client, plan.location.bucket, i): i for i in todo}
        for fut in as_completed(futs):
            item = futs[fut]
            try:
                out[item.key] = fut.result()
            except Exception as exc:
                out[item.key] = exc
                log.warning("上げられませんでした %s: %s", item.key, exc)
            if on_done is not None:
                on_done(item, out[item.key])
    return out


def human_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


# ---------------------------------------------------------------------------
# コマンドラインから使う共通部品
#
# 取り込み (fetch_from_s3.py) と送り込み (upload_to_s3.py) で同じ文言・同じ
# 歯止めにするため、ここに置く。
# ---------------------------------------------------------------------------
def guard_ec2(allow_non_ec2: bool, what: str) -> None:
    """EC2 の上でだけ動かす。それ以外では理由を添えて止める。"""
    if is_ec2() or allow_non_ec2:
        return
    raise PermissionError(
        f"EC2 の上ではありません。\n"
        f"{what}は EC2 での運用に限っています。\n"
        "Mac ではローカルの raw_data/ をそのまま使ってください。\n"
        "  疎通確認だけしたい場合: --allow-non-ec2 --dry-run"
    )


def format_plan(plan: Plan, raw_data: Path, list_files: bool = False) -> str:
    """計画を人が読める形にする。取り込みと送り込みで向きだけ変える。"""
    up = plan.direction == UPLOAD
    src = str(raw_data) if up else plan.location.uri(plan.s3_prefix)
    dst = plan.location.uri(plan.s3_prefix) if up else str(raw_data)
    done = "送信済み" if up else "取得済み"
    todo = "送信予定" if up else "取得予定"

    lines = [
        "",
        f"データセット : {plan.dataset}",
        f"取得元       : {src}",
        f"送り先       : {dst}" if up else f"取り込み先   : {dst}",
    ]
    if plan.dataset == "car-segments":
        segs = [i for i in plan.items if i.dest.name == "rlog.zst"]
        if segs:
            # .../segments/<dongle>/<route>/<n>/rlog.zst
            routes = {(i.dest.parents[2].name, i.dest.parents[1].name) for i in segs}
            lines.append(f"内訳         : {len(segs):,} セグメント / {len(routes):,} route")
    lines += [
        f"対象         : {len(plan.items):,} ファイル / {human_bytes(plan.total_bytes)}",
        f"{done}     : {len(plan.have):,} ファイル",
        f"{todo}     : {len(plan.todo):,} ファイル / {human_bytes(plan.todo_bytes)}",
    ]
    lines += [f"  注意       : {n}" for n in plan.notes]
    if list_files:
        lines.append("")
        lines += [
            f"  {'済' if i.present else '→'} {i.key}  ({human_bytes(i.size)})" for i in plan.items
        ]
    return "\n".join(lines)


def confirm_size(total_bytes: int, max_gb: float, yes: bool, verb: str = "取得") -> bool:
    """量が大きいときだけ確認を取る。非対話なら止める。"""
    gb = total_bytes / (1 << 30)
    if gb <= max_gb or yes:
        return True
    print()
    print(f"{verb}予定が {gb:.2f} GB で上限 {max_gb:.2f} GB を超えています。")
    try:
        return input("続けますか [y/N]: ").strip().lower() in ("y", "yes")
    except EOFError:
        print("非対話のため中止します。実行するなら -y か --max-gb を付けてください。")
        return False


def run_transfer(
    client,
    plan: Plan,
    workers: int,
    verb: str,
) -> int:
    """計画を実行して結果を出す。戻り値はそのままプロセスの終了コードに使える。"""
    todo = plan.todo
    print()
    done = {"n": 0}

    def progress(item: Item, result):
        done["n"] += 1
        state = "失敗" if isinstance(result, Exception) else "完了"
        print(f"  [{done['n']}/{len(todo)}] {state} {item.key}")

    fn = upload if plan.direction == UPLOAD else download
    results = fn(client, plan, workers=workers, on_done=progress)
    failed = [k for k, v in results.items() if isinstance(v, Exception)]
    print()
    print(f"{verb}完了 {len(todo) - len(failed)} / {len(todo)}  (失敗 {len(failed)})")
    if failed:
        print("失敗したキー (先頭 5 件):")
        for k in failed[:5]:
            print(f"  {k}: {results[k]}")
    return 1 if failed else 0
