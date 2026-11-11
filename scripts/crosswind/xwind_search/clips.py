"""候補の山 ±12 s を mp4 に切り出す。左上に「山からの時刻」を焼き込む。
使い方: clips.py <out_dir>"""
import sys, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
O = Path(sys.argv[1]); O.mkdir(parents=True, exist_ok=True)
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
C = [("data/comma2k19_civic_video", "99c94dc769b5d96e|2018-07-10--14-21-51", 26, 180742.0, "civic_0710_seg26", "Civic 07-10 seg26 北向き 西風9.8m/s (左から)"),
     ("data/comma2k19_civic_video", "99c94dc769b5d96e|2018-07-10--14-21-51", 31, 181028.0, "civic_0710_seg31", "Civic 07-10 seg31 北向き 西風9.8m/s (左から)"),
     ("raw_data/Chunk_2", "b0c9d2329ad1606b|2018-11-11--13-08-44", 20, 268484.0, "rav4_1111_seg20", "RAV4 11-11 seg20 南東向き 北東風6.7m/s (左から)"),
     ("raw_data/Chunk_1", "b0c9d2329ad1606b|2018-07-27--06-03-57", 11, 9809.0, "rav4_0727_seg11", "RAV4 07-27 seg11 南東向き 西風6.2m/s (右から)")]
for root, drive, seg, tp, name, label in C:
    d = Path(root) / drive / str(seg)
    ft = np.load(d / "global_pose/frame_times").ravel()
    a = max(0.0, tp - 12 - ft[0]); b = min(a + 24, ft[-1] - ft[0]); off = ft[0] - tp
    # ffmpeg に drawtext が無いので、生フレームを読んで PIL で書き込み、もう一度 ffmpeg で符号化する
    W, H = 1164, 874; k0, k1 = int(round(a * 20)), int(round(b * 20))
    dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-r", "20", "-i", str(d / "video.hevc"), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "20", "-i", "-",
                            "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", str(O / f"{name}.mp4")], stdin=subprocess.PIPE)
    font = ImageFont.truetype(FONT, 26)
    for k in range(k1):
        buf = dec.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        if k < k0: continue
        im = Image.frombytes("RGB", (W, H), buf); dr = ImageDraw.Draw(im)
        rel = ft[min(k, len(ft) - 1)] - tp
        dr.rectangle([0, 0, W, 40], fill=(0, 0, 0))
        dr.text((10, 6), f"{label}   山からの時刻 {rel:+5.1f} s", fill=(255, 255, 255) if abs(rel) > 1 else (255, 90, 90), font=font)
        enc.stdin.write(im.tobytes())
    dec.kill(); enc.stdin.close(); enc.wait()
    print(O / f"{name}.mp4", f"セグメント内 {a:.1f}–{b:.1f} s, 山 = {tp - ft[0]:.1f} s")
