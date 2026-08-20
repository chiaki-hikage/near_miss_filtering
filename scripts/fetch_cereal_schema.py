#!/usr/bin/env python3
"""rlog を読むための capnp スキーマ (cereal) を取得する。

commaCarSegments の rlog は openpilot のログ形式そのままなので、
読むには cereal の .capnp が要る。openpilot を丸ごと入れる必要は無いため、
必要な 5 ファイルだけを取得して data/cereal/ に置く。

使い方:
  python scripts/fetch_cereal_schema.py [--out data/cereal] [--force]
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401

from near_miss.io.rlog import DEFAULT_SCHEMA_DIR

# log.capnp は openpilot 本体、car.capnp は opendbc に分かれている。
SOURCES = {
    "log.capnp": "https://raw.githubusercontent.com/commaai/openpilot/master/openpilot/cereal/log.capnp",
    "custom.capnp": "https://raw.githubusercontent.com/commaai/openpilot/master/openpilot/cereal/custom.capnp",
    "deprecated.capnp": "https://raw.githubusercontent.com/commaai/openpilot/master/openpilot/cereal/deprecated.capnp",
    "include/c++.capnp": "https://raw.githubusercontent.com/commaai/openpilot/master/openpilot/cereal/include/c%2B%2B.capnp",
    "car.capnp": "https://raw.githubusercontent.com/commaai/opendbc/master/opendbc/car/car.capnp",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_SCHEMA_DIR)
    p.add_argument("--force", action="store_true", help="既にあるファイルも取り直す")
    args = p.parse_args()

    for rel, url in SOURCES.items():
        dest = args.out / rel
        if dest.is_file() and not args.force:
            print(f"  skip  {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            body = r.read()
        dest.write_bytes(body)
        print(f"  取得  {dest}  {len(body)} bytes")

    print()
    print("capnp スキーマの読み込みを確認します...")
    from near_miss.io.rlog import load_schema

    schema = load_schema(str(args.out))
    print(f"  Event スキーマを読めました ({schema.Event.schema.node.displayName})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
