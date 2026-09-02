#!/usr/bin/env python3
"""KIT MSDM の配布物を受け取り、中身を照合してから raw_data/ に展開する。

配布物は BagIt を tar 1 本にまとめたもの (171.7 MB)。
**どこから来たかではなく、中身で正しさを確かめる。** 閉鎖環境で取得経路が
限られる場合を想定しているため、次の 3 つの入口を用意してある。

  # 1) 手元にある tar を使う (**ネットワークを一切使わない**)
  python scripts/fetch_kit_msdm.py --tar /mnt/media/msdm.tar

  # 2) 外に出られる環境で取ってくる
  python scripts/fetch_kit_msdm.py --url <配布 URL>

  # 3) 既に展開してあるものを照合するだけ
  python scripts/fetch_kit_msdm.py --verify-only

閉鎖 EC2 では 1) を使い、tar は S3 経由で持ち込む (docs/environment.md §4.6)。

安全のために:
  * 取得前に**接続先ホストを表示して確認を取る** (-y で省略)
  * MD5 が configs/datasets/kit_msdm.yaml の値と違えば**展開せずに捨てる**
  * tar は絶対パス・`..`・シンボリックリンク・デバイスを含む要素を**拒否する**
  * 送信するのは固定の User-Agent だけ。ホスト名も利用者名も外に出さない
  * 認証情報は一切要らない (CC BY-SA 4.0 の公開データ)
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

import _bootstrap  # noqa: F401

from near_miss.io import kit_msdm as kit

# 外に出す情報を最小にする。既定の requests/urllib はバージョン等を載せるので置き換える。
USER_AGENT = "near-miss-filtering/0.1"

DEFAULT_DEST = kit.REPO_ROOT / "raw_data" / "kit_msdm"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--tar", type=Path, help="手元の配布物 (ネットワークを使わない)")
    src.add_argument("--url", help="配布 URL (外に出られる環境でのみ)")
    src.add_argument("--verify-only", action="store_true", help="展開済みのものを照合するだけ")

    p.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="展開先 (既定 raw_data/kit_msdm)")
    p.add_argument("--config", type=Path, default=kit.DATASET_CONFIG, help="配布物の素性")
    p.add_argument("--quick", action="store_true", help="MD5 を取らず、存在と大きさだけ見る")
    p.add_argument("--dry-run", action="store_true", help="何をするかだけ出す")
    p.add_argument("-y", "--yes", action="store_true", help="確認を飛ばす (非対話)")
    p.add_argument("--proxy", help="HTTPS プロキシ (未指定なら環境変数 HTTPS_PROXY)")
    p.add_argument("--timeout", type=float, default=120.0, help="通信の待ち時間 [s]")
    p.add_argument(
        "--remove-archive", action="store_true",
        help="展開後に tar を消す (既定は残す。再照合と S3 への送り込みに使うため)",
    )
    p.add_argument(
        "--allow-checksum-mismatch", action="store_true",
        help="MD5 が合わなくても展開する (**既定では止める**)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# tar の中身を検査してから展開する
# ---------------------------------------------------------------------------
def unsafe_members(tf: tarfile.TarFile) -> list[str]:
    """展開してはいけない要素を挙げる。

    tar は展開先の外へ書ける形式を許してしまう。閉鎖環境に持ち込むものほど
    ここを省かない。Python の版によって既定の filter が違うので自前で見る。
    """
    bad = []
    for m in tf.getmembers():
        name = m.name
        if name.startswith("/") or Path(name).is_absolute():
            bad.append(f"{name} (絶対パス)")
        elif any(part == ".." for part in Path(name).parts):
            bad.append(f"{name} ('..' を含む)")
        elif m.issym() or m.islnk():
            bad.append(f"{name} (リンク)")
        elif m.ischr() or m.isblk() or m.isfifo() or m.isdev():
            bad.append(f"{name} (デバイス)")
        elif not (m.isfile() or m.isdir()):
            bad.append(f"{name} (通常のファイルでもディレクトリでもない)")
    return bad


def extract(tar_path: Path, dest: Path) -> Path:
    """検査してから展開する。bag の根の場所を返す。"""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        bad = unsafe_members(tf)
        if bad:
            raise SystemExit(
                "tar に展開してはいけない要素があります。展開を中止しました:\n  "
                + "\n  ".join(bad[:10])
            )
        tops = {Path(m.name).parts[0] for m in tf.getmembers() if m.name not in (".", "")}
        tf.extractall(dest)  # 上で全要素を検査済み
    if len(tops) == 1:
        return dest / next(iter(tops))
    return dest


# ---------------------------------------------------------------------------
# 取得
# ---------------------------------------------------------------------------
def download(url: str, dest: Path, proxy: str | None, timeout: float) -> Path:
    """`.part` へ落としてから置き換える。落ちたら残骸を残さない。"""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    proxies = {"https": proxy, "http": proxy} if proxy else None
    got = 0
    try:
        with requests.get(
            url, stream=True, timeout=timeout, proxies=proxies,
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        ) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        print(f"\r  {got / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def check_archive(path: Path, spec: dict, quick: bool) -> tuple[bool, list[str]]:
    """配布物そのものを照合する。"""
    problems = []
    size = path.stat().st_size
    if spec.get("size_bytes") and size != spec["size_bytes"]:
        problems.append(f"大きさが違います: {size:,} != {spec['size_bytes']:,}")
    if quick:
        return not problems, problems + ["--quick のため MD5 は取っていません"]
    digest = kit.file_md5(path)
    if spec.get("md5") and digest != spec["md5"]:
        problems.append(f"MD5 が違います:\n    実測 {digest}\n    期待 {spec['md5']}")
    return not problems, problems


def main() -> int:
    args = parse_args()
    cfg = kit.load_dataset_config(args.config)
    spec = cfg.get("archive") or {}
    bag_root_name = (cfg.get("bag") or {}).get("root", "")
    bag_root = args.dest / bag_root_name

    print("=" * 68)
    print("KIT MSDM の受け取り")
    print("=" * 68)
    print(f"  DOI          : {cfg.get('doi', '?')}")
    print(f"  ライセンス   : {cfg.get('license', '?')}")
    print(f"  展開先       : {args.dest}")
    print(f"  期待 MD5     : {spec.get('md5', '?')}  ({spec.get('size_bytes', 0):,} バイト)")

    # --- 照合だけ ---
    if args.verify_only:
        if not bag_root.is_dir():
            raise SystemExit(f"{bag_root} がありません。先に取得・展開してください。")
        print("\n展開済みのものを照合します (ネットワークは使いません)")
        rep = kit.verify_bag(bag_root, cfg, quick=args.quick)
        print(rep.summary())
        print("\n" + ("中身は配布物どおりです。" if rep.ok else "**中身が配布物と違います。**"))
        return 0 if rep.ok else 1

    # --- 入口を決める ---
    archive = args.dest / spec.get("name", "msdm.tar")
    if args.tar:
        source = f"手元のファイル {args.tar}"
        network = False
        if not args.tar.is_file():
            raise SystemExit(f"{args.tar} がありません。")
    elif args.url:
        source = args.url
        network = True
    elif archive.is_file():
        source = f"既に落としてある {archive}"
        network = False
    else:
        raise SystemExit(
            "配布物の入手方法を指定してください。\n"
            f"  手元の tar を使う   : --tar <path>   (ネットワークを使いません)\n"
            f"  取ってくる          : --url <URL>\n"
            f"  URL は次のページから辿れます: {cfg.get('landing_page', '')}\n"
            "\n閉鎖環境での進め方は docs/environment.md §4.6 を見てください。"
        )

    print(f"  入手元       : {source}")
    if network:
        from urllib.parse import urlparse

        host = urlparse(args.url).hostname or "?"
        proxy = args.proxy or "(環境変数 HTTPS_PROXY に従う)"
        print(f"  接続先ホスト : {host}")
        print(f"  プロキシ     : {proxy}")
        print(f"  送信する情報 : User-Agent: {USER_AGENT} のみ (認証情報なし)")
    else:
        print("  ネットワーク : 使いません")

    if args.dry_run:
        print("\n--dry-run のため何もしません。")
        return 0

    if network and not args.yes:
        try:
            if input(f"\n{host} へ接続します。よろしいですか [y/N]: ").strip().lower() not in ("y", "yes"):
                print("中止しました。")
                return 1
        except EOFError:
            print("\n非対話です。接続してよければ -y を付けて再実行してください。")
            return 1

    # --- 配布物を用意する ---
    if args.tar:
        archive = args.tar
    elif network:
        print(f"\n取得中 ({spec.get('size_bytes', 0) / 1e6:.0f} MB)")
        download(args.url, archive, args.proxy, args.timeout)

    # --- 配布物を照合する ---
    print(f"\n配布物の照合 ({archive})")
    ok, problems = check_archive(archive, spec, args.quick)
    for p in problems:
        print(f"  {p}")
    if ok:
        print("  期待どおりです。")
    elif not args.allow_checksum_mismatch:
        if network:
            archive.unlink(missing_ok=True)
            print("  取得したファイルは削除しました。")
        raise SystemExit(
            "配布物が期待と違います。展開しませんでした。\n"
            "  取得経路 (プロキシ等) が中身を書き換えていないか、\n"
            "  配布側で版が変わっていないかを確かめてください。\n"
            "  意図的に進める場合のみ --allow-checksum-mismatch を付けてください。"
        )

    # --- 展開して中身を照合する ---
    print(f"\n展開 -> {args.dest}")
    root = extract(archive, args.dest)
    print(f"  bag の根     : {root}")

    print("\n中身の照合 (BagIt manifest-md5.txt)")
    rep = kit.verify_bag(root, cfg, quick=args.quick)
    print(rep.summary())

    if args.remove_archive and archive != args.tar:
        archive.unlink(missing_ok=True)
        print(f"\n配布物を削除しました: {archive}")

    ds = root / (cfg.get("bag") or {}).get("dataset_dir", "data/dataset")
    n_mat = len(list(ds.glob("*.mat"))) if ds.is_dir() else 0
    print(f"\n走行データ   : {ds}")
    print(f"  .mat        : {n_mat} 本 (期待 {(cfg.get('bag') or {}).get('mat_files', '?')})")
    print("\n" + ("受け取り完了。" if rep.ok else "**中身が配布物と違います。**"))
    if rep.ok:
        print("  確認: uv run python scripts/validate_sideslip_filter.py --kind dynamic --min-speed 3")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
