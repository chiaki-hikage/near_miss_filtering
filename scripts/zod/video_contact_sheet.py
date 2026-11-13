#!/usr/bin/env python3
"""前方映像のレビュー用に、フレームを時刻のラベルつきで並べた一覧画像を作る。ネットワークは使わない。

  uv run --offline --extra viz python scripts/zod/video_contact_sheet.py --index out/zod/front_video_review/<run>/index.csv --out out/zod/front_video_review/<run>/sheets

シーケンスごとに 2 枚:
  <seq>_overview.jpg   カメラの開始から 2 s おき (0, 2, …, 18 s) の 10 フレーム
  <seq>_events.jpg     注目の候補の閾値超過区間の始まり・ピーク・終わりと、その前後 1 s のフレーム (カメラの区間の中だけ)
ラベルはカメラの開始からの秒と UTC。カメラと車両信号の時刻の対応は約 ±1 s までしか確かめていない。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def grid(items, out: Path, cols=5, w=760):
    font = ImageFont.truetype(FONT, 22)
    ims = []
    for path, label in items:
        im = Image.open(path)
        im.draft("RGB", (im.width // 4, im.height // 4))
        im = im.convert("RGB")
        h = int(w * im.height / im.width)
        im = im.resize((w, h))
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle([0, 0, w, 34], fill=(0, 0, 0, 160))
        d.text((8, 4), label, font=font, fill=(255, 255, 255))
        ims.append(im)
    rows = (len(ims) + cols - 1) // cols
    h = ims[0].height
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for k, im in enumerate(ims):
        sheet.paste(im, ((k % cols) * w, (k // cols) * h))
    sheet.save(out, quality=88)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="raw_data/zod")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    idx = pd.read_csv(a.index, dtype={"sequence_id": str})
    for sid, g in idx.groupby("sequence_id"):
        info = json.loads((Path(a.root) / "sequences" / sid / "info.json").read_text())
        cf = info["camera_frames"]["front_blur"]
        times = np.array([ts(f["time"]) for f in cf])
        paths = [Path(a.root) / f["filepath"] for f in cf]
        if not all(p.exists() for p in paths):
            continue
        c0 = times[0]

        def pick(t_rel):
            i = int(np.argmin(np.abs(times - (c0 + t_rel))))
            return paths[i], f"+{times[i] - c0:5.1f} s  {datetime.fromtimestamp(times[i], timezone.utc):%H:%M:%S.%f}"[:-4] + " UTC"
        grid([pick(t) for t in range(0, 20, 2)], out / f"{sid}_overview.jpg")
        items = []
        for _, e in g[g.is_focus == True].iterrows():  # noqa: E712
            a0, a1 = [float(x) for x in e.detect_rel_camera_s.split("..")]
            pk = float(e.peak_rel_camera_s)
            for name, t in [("超過の始まり −1 s", a0 - 1), ("超過の始まり", a0), ("ピーク", pk), ("超過の終わり", a1), ("超過の終わり +1 s", a1 + 1)]:
                if 0 <= t <= times[-1] - c0:
                    p, lab = pick(t)
                    items.append((p, f"{e.event_id} {name}  {lab}"))
        if items:
            grid(items, out / f"{sid}_events.jpg", cols=min(5, len(items)))
        print(sid, "ok", len(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
