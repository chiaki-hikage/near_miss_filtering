"""フレームキャッシュ。

素朴に「評価時刻ごとに ffmpeg を呼ぶ」と、同じ動画を何十回もデコードする。
1 エピソードあたり評価時刻は 25〜75 点あるので、g7e.2xlarge の 8 vCPU では
ここが律速になる。**セグメントごとに一度だけデコードして JPEG にする。**

命名はセグメント内の絶対フレーム番号にする (f00374.jpg = フレーム 374)。
ffmpeg の -start_number で採番をずらすと 1 ずれても気づけないので、
オフセットを持ち込まない。1 セグメント 1200 枚・約 50 MB で、
Phase 1 の全エピソードでも 2 GB 程度にしかならない。

生の video.hevc はタイムスタンプを持たない (ffprobe が幅も高さも返さない)。
setpts で 20 Hz を打ち直さないと、フレーム番号と秒の対応が崩れる。
raw_data/hevc2mpeg.sh が同じことをしている。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .windows import Episode, video_frame_times


@dataclass(frozen=True)
class FrameRef:
    """1 枚のフレーム。どの時刻のどのフレームかを持ち回る。"""

    path: Path
    t: float          # boot time
    segment: int
    frame: int

    @property
    def exists(self) -> bool:
        return self.path.is_file()


def segment_dir(cache_root, drive_id: str, segment: int) -> Path:
    """キャッシュ上のセグメントの置き場。

    drive_id には '|' が入る (b0c9d2329ad1606b|2018-07-30--13-03-07)。
    そのままディレクトリ名にすると扱いにくいので、ドライブ名の側だけ使う。
    """
    return Path(cache_root) / drive_id.split("|")[-1] / f"{segment:03d}"


def frame_path(cache_root, drive_id: str, segment: int, frame: int) -> Path:
    return segment_dir(cache_root, drive_id, segment) / f"f{frame:05d}.jpg"


def extract_cmd(video: Path, out_dir: Path, cfg: dict[str, Any],
                hwaccel: bool = False) -> list[str]:
    """1 セグメントを丸ごと JPEG に展開する ffmpeg コマンドを組む。

    hwaccel=True で NVDEC を使う。g7e は 8 vCPU しかないので、EC2 では
    デコードを GPU へ逃がして CPU を空けたい。使えるビルドかは実行側で確かめる。
    """
    fps = float(cfg["video"]["fps"])
    long_edge = int(cfg["input"]["video_long_edge"])
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-framerate", f"{fps:g}", "-i", str(video)]
    # setpts で 20 Hz を打ち直してから縮小する。scale の -2 は偶数に丸める指定。
    cmd += ["-vf", f"setpts=N/({fps:g}*TB),scale={long_edge}:-2",
            "-fps_mode", "passthrough", "-q:v", "3",
            "-start_number", "0", str(out_dir / "f%05d.jpg")]
    return cmd


def frames_for(ep: Episode, t: float, cfg: dict[str, Any], cache_root) -> list[FrameRef]:
    """評価時刻 t に渡すフレームを、後ろ揃えで返す。

    **最後の 1 枚がちょうど t。** 未来のフレームは決して含まない
    (video_frame_times が保証し、tests/test_vlm_causality.py が検査する)。
    """
    out: list[FrameRef] = []
    for ft in video_frame_times(t, cfg):
        p = ep.to_segment(ft, cfg)
        out.append(FrameRef(
            path=frame_path(cache_root, ep.drive_id, p.segment, p.frame),
            t=ft, segment=p.segment, frame=p.frame,
        ))
    return out


def needed_segments(ep: Episode, cfg: dict[str, Any], times: list[float]) -> set[int]:
    """与えた評価時刻をすべて賄うのに要るセグメント。

    映像の窓は t から window_video_s 遡るので、評価区間の先頭より
    さらに前のセグメントが要ることがある。ここを取り違えると、
    先頭の数時刻だけフレームが欠ける。
    """
    segs: set[int] = set()
    for t in times:
        for fr in frames_for(ep, t, cfg, cache_root="."):
            segs.add(fr.segment)
    return segs


def missing(refs: list[FrameRef]) -> list[FrameRef]:
    return [r for r in refs if not r.exists]


def frames_available(
    ep: Episode, t: float, cfg: dict[str, Any], cache_root, available: set[int]
) -> tuple[list[FrameRef], bool]:
    """手元にあるセグメントに入るフレームだけを、後ろ揃えのまま返す。

    ドライブの先頭付近では 4 秒前のセグメントが無いことがある。そこで時刻ごと
    捨てると、危険が立ち上がるまさにその区間を評価できなくなる
    (実測: P08 は候補開始が seg1 の 0.60 秒で、seg0 が手元に無い)。

    落とすのは**古い側だけ**で、最後の 1 枚が t であることは変わらない。
    返り値の 2 つ目は「窓が満たなかったか」。記録して採点側で層別する。
    """
    full = frames_for(ep, t, cfg, cache_root)
    keep = [r for r in full if r.segment in available]
    return keep, len(keep) < len(full)
