#!/usr/bin/env python3
"""実行環境が整っているかを確かめる。

Mac でも Linux (EC2) でも同じ結果になるべき項目だけを並べる。
横滑りフィルタを動かすのに足りないものがあれば 1 で終わる。

  uv run python scripts/check_env.py
  uv run python scripts/check_env.py --data     # データの置き場も見る
"""

from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from near_miss.config import REPO_ROOT

# (モジュール名, 用途, 無いと止まるか)
CORE = [
    ("numpy", "数値計算", True),
    ("pandas", "表", True),
    ("yaml", "設定の読み出し", True),
    ("scipy", "雑音の当てはめ (calibrate_beta_noise.py)", True),
    ("capnp", "rlog の読み出し (commaCarSegments)", True),
    ("zstandard", "rlog.zst の展開", True),
    ("requests", "セグメントの取得", True),
]
OPTIONAL = [
    ("matplotlib", "図 (--extra viz)", False),
    ("safetensors", "comma1M (--extra comma1m)", False),
    ("pymap3d", "comma1M (--extra comma1m)", False),
    ("reverse_geocoder", "comma1M (--extra comma1m)", False),
    ("PIL", "comma1M の天候判定 (--extra comma1m)", False),
    ("pyarrow", "comma2k19 demo split (--extra demo-dataset)", False),
    ("pytest", "試験 (--extra dev)", False),
    ("boto3", "S3 からの取り込み (--extra s3、EC2 のみ)", False),
]

# (相対パス, 説明, 何のために要るか)
DATA_DIRS = [
    ("raw_data/comma_car_segments/segments", "commaCarSegments の rlog", "横滑りフィルタの主対象"),
    ("raw_data/comma_car_segments/database.json", "セグメント一覧", "取得に要る"),
    ("raw_data/kit_msdm/10.35097-44a91t97pmnha1k9/data/dataset", "KIT MSDM", "再現率の確認 (物差し)"),
    ("raw_data/Chunk_1", "comma2k19 Chunk_1", "任意"),
    # clone に付いてこない。commaCarSegments / comma2k19 の rlog を読むのに要る
    ("data/cereal/log.capnp", "capnp スキーマ", "rlog の読み出し (fetch_cereal_schema.py)"),
]


def _version(mod) -> str:
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    return "?"


def check_imports(rows) -> list[str]:
    missing = []
    for name, why, required in rows:
        try:
            mod = importlib.import_module(name)
            print(f"  {'OK ':<4}{name:<18}{_version(mod):<12}{why}")
        except Exception as exc:
            mark = "欠落" if required else "任意"
            print(f"  {mark:<4}{name:<18}{'-':<12}{why}   ({type(exc).__name__})")
            if required:
                missing.append(name)
    return missing


def dir_size_mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def _report_s3() -> None:
    """S3 経路の設定だけを見る。**AWS には問い合わせない** (EC2 の上でだけ認証を確かめる)。"""
    from near_miss.io import s3_sync as s3

    on_ec2 = s3.is_ec2()
    print(f"  EC2          : {'はい' if on_ec2 else 'いいえ'}")
    try:
        loc = s3.resolve_location()
        print(f"  バケット     : {loc.uri()}")
    except ValueError:
        print("  バケット     : 未設定  (--bucket / NEAR_MISS_S3_URI / configs/datasets/s3.yaml)")
        if not on_ec2:
            return

    try:
        s3.assert_no_env_file_credentials()
    except PermissionError as exc:
        print(f"  警告         : {str(exc).splitlines()[0]}")
        return

    if not on_ec2:
        print("  認証         : 確認しない (EC2 の上ではないため)")
        return
    try:
        import boto3

        info = s3.verify_credentials(boto3.session.Session())
        print(f"  認証         : {info.method}" + ("  (IAM Role)" if info.is_role else ""))
    except ImportError:
        print("  認証         : boto3 が無い  (uv sync --extra s3)")
    except PermissionError as exc:
        print(f"  認証         : {str(exc).splitlines()[0]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="store_true", help="データの置き場も確認する")
    args = ap.parse_args()

    print("=" * 68)
    print("実行環境")
    print("=" * 68)
    print(f"  OS        : {platform.system()} {platform.release()}")
    print(f"  アーキ    : {platform.machine()}")
    print(f"  Python    : {platform.python_version()}  ({sys.executable})")
    print(f"  リポジトリ: {REPO_ROOT}")
    try:
        import os
        print(f"  CPU 数    : {os.cpu_count()}")
    except Exception:
        pass

    print("\n必須の依存")
    missing = check_imports(CORE)
    print("\n任意の依存")
    check_imports(OPTIONAL)

    print("\n外部コマンド (任意)")
    for cmd, why in (("ffmpeg", "comma1M の動画切り出し"), ("git", "取得"), ("uv", "環境構築")):
        path = shutil.which(cmd)
        print(f"  {'OK ' if path else '無し':<4}{cmd:<18}{(path or '-'):<40}{why}")

    # 図まわりは判定経路で OS が効く唯一の箇所なので、必ず何が選ばれるか出す。
    print("\n図の設定 (判定経路で OS 依存の唯一の箇所)")
    try:
        from near_miss import plotting

        info = plotting.describe()
        fonts = info["japanese_fonts"]
        print(f"  matplotlib   : {info['matplotlib']}")
        print(f"  日本語フォント: {fonts[0] if fonts else '無し'}"
              + (f"  (他に {', '.join(fonts[1:])})" if len(fonts) > 1 else ""))
        if not fonts:
            print("  " + plotting.INSTALL_HINT.replace("\n", "\n  "))
    except Exception as exc:
        print(f"  matplotlib が使えません: {exc}  (図を描かないなら問題ない)")

    print("\nS3 からの取り込み (EC2 のみ。Mac はローカルの raw_data/ を直接使う)")
    _report_s3()

    if args.data:
        print("\nデータの置き場")
        for rel, name, why in DATA_DIRS:
            p = REPO_ROOT / rel
            if not p.exists():
                print(f"  無し  {rel:<52}{name} — {why}")
                continue
            extra = ""
            if p.is_dir() and "comma_car_segments" in rel:
                extra = f"  rlog {len(list(p.rglob('rlog.zst')))} 本"
            elif p.is_dir() and "kit_msdm" in rel:
                extra = f"  .mat {len(list(p.glob('*.mat')))} 本"
            print(f"  OK    {rel:<52}{dir_size_mb(p):8.1f} MB{extra}")

    print("\n" + "=" * 68)
    if missing:
        print(f"必須の依存が足りません: {', '.join(missing)}")
        print("  uv sync --extra viz --extra dev   を実行してください")
        return 1
    print("必須の依存はそろっています。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
