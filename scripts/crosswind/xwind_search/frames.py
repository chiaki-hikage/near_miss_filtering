"""指定時刻の前面映像のフレームを切り出して並べる (確認用)。
入力 csv: label, root, drive, seg, t  (t は global_pose/frame_times と同じ時計)
各行について offsets 秒ずらしたフレームを横に並べ、行を縦に積んだ画像を出す。
使い方: frames.py <list.csv> <out.png> [offsets (カンマ区切り)] [幅]"""
import sys, subprocess, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
L, OUTP = Path(sys.argv[1]), Path(sys.argv[2])
OFF = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "-2,0,2").split(",")]
WID = int(sys.argv[4]) if len(sys.argv) > 4 else 560
font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 18)
rows = []
for _, r in pd.read_csv(L).iterrows():
    tiles = []
    for off in OFF:
        t = r.t + off; seg = int(r.seg); d = Path(r.root) / r.drive
        ft = np.load(d / str(seg) / "global_pose/frame_times").ravel()
        if t > ft[-1] and (d / str(seg + 1)).exists():
            seg += 1; ft = np.load(d / str(seg) / "global_pose/frame_times").ravel()
        if t < ft[0] and (d / str(seg - 1)).exists():
            seg -= 1; ft = np.load(d / str(seg) / "global_pose/frame_times").ravel()
        k = int(np.argmin(np.abs(ft - t)))
        with tempfile.TemporaryDirectory() as td:
            o = Path(td) / "f.png"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(d / str(seg) / "video.hevc"), "-vf",
                            f"select=eq(n\\,{k})", "-vframes", "1", str(o)], check=True)
            im = Image.open(o).convert("RGB")
        im = im.resize((WID, int(im.height * WID / im.width)))
        dr = ImageDraw.Draw(im); dr.rectangle([0, 0, WID, 26], fill=(0, 0, 0))
        dr.text((6, 3), f"{r.label}  {off:+.0f} s  (seg{seg} f{k})", fill=(255, 255, 255), font=font)
        tiles.append(im)
    row = Image.new("RGB", (WID * len(tiles), tiles[0].height))
    for i, im in enumerate(tiles): row.paste(im, (i * WID, 0))
    rows.append(row)
out = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
y = 0
for r in rows: out.paste(r, (0, y)); y += r.height
OUTP.parent.mkdir(parents=True, exist_ok=True); out.save(OUTP); print(OUTP, out.size)
