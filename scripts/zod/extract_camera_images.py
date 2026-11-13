#!/usr/bin/env python3
"""手元の ZOD Sequences の前方カメラのアーカイブ (images_front_blur.tar.gz) から、指定したシーケンスの画像だけを取り出す。
ネットワークは使わない。アーカイブは先頭から 1 回だけ読む (tar.gz は途中から読めないので、全体を読む。140 GB で数十分の見込み)。

  uv run --offline python scripts/zod/extract_camera_images.py --archive raw_data/zod/downloads/sequences/<名前> \\
      --ids 000765 000675 000662 000804 000695 000801 --log out/zod/front_video_review/extract_log.json

- 取り出す先は raw_data/zod (アーカイブの中の場所 sequences/<id>/camera_front_blur/<file>.jpg のまま)。既にある同じ名前のファイルは上書きしない
- アーカイブの中の場所が想定 (sequences/<id>/camera_front_blur/...) と違えば、最初の 1,000 件を見て止まる
- 取り出した後、各シーケンスの枚数とファイル名を info.json の camera_frames と照らす。各ファイルの sha256 を記録する
- 取り出すパスに .. や絶対パスが入っていれば取り出さない
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--root", default="raw_data/zod")
    ap.add_argument("--log", required=True)
    ap.add_argument("--stop-when-done", action="store_true", help="全シーケンスの枚数がそろったら読むのをやめる (並び順が分からないので、既定では最後まで読む)")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    want = {i: set(Path(f["filepath"]).name for f in json.loads((root / "sequences" / i / "info.json").read_text())["camera_frames"]["front_blur"]) for i in a.ids}
    got = {i: set() for i in a.ids}
    shas = {}
    t0 = time.time()
    seen = 0
    layout_ok = False
    with tarfile.open(a.archive, mode="r|gz") as tf:
        for m in tf:
            seen += 1
            parts = Path(m.name).parts
            if len(parts) >= 3 and parts[0] == "sequences" and parts[2] == "camera_front_blur":
                layout_ok = True
            elif seen > 1000 and not layout_ok:
                print("アーカイブの中の場所が想定と違う:", m.name, file=sys.stderr)
                return 2
            if not m.isfile() or len(parts) != 4 or parts[0] != "sequences" or parts[2] != "camera_front_blur" or parts[1] not in got:
                if seen % 20000 == 0:
                    print(f"{seen} 件を読んだ ({time.time() - t0:.0f} s)、取り出した {sum(len(v) for v in got.values())} 枚", flush=True)
                continue
            if m.name.startswith("/") or ".." in parts:
                continue
            dest = root / m.name
            data = tf.extractfile(m).read()
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            shas[m.name] = hashlib.sha256(dest.read_bytes()).hexdigest()
            got[parts[1]].add(parts[3])
            if a.stop_when_done and all(want[i] <= got[i] for i in a.ids):
                print("全シーケンスがそろったので止める", flush=True)
                break
    res = {i: dict(expected=len(want[i]), extracted=len(got[i]), missing=sorted(want[i] - got[i])[:5], extra=sorted(got[i] - want[i])[:5],
                   complete=want[i] <= got[i]) for i in a.ids}
    log = dict(archive=str(a.archive), archive_bytes=Path(a.archive).stat().st_size, members_read=seen, elapsed_s=round(time.time() - t0, 1),
               finished=time.strftime("%Y-%m-%dT%H:%M:%S"), sequences=res, sha256=shas)
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    Path(a.log).write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != "sha256"}, ensure_ascii=False, indent=1))
    return 0 if all(r["complete"] for r in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
