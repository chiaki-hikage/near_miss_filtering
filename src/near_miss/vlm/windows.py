"""評価時刻の生成と、boot time <-> セグメント内時刻・フレーム番号の対応。

comma2k19 の映像はセグメント単位のファイルで 20 Hz 固定、1 セグメント 60 秒。
一方 labels.csv / candidates.csv の時刻はデバイスの boot time なので、
両者を突き合わせる層がここになる。

**因果性の要点**

  評価時刻 t に対して
    映像 : (t - window_video_s, t] の等間隔フレーム。**最後の 1 枚がちょうど t**
    CAN  : (t - window_can_s, t - guard_s] の等間隔サンプル

  映像は t まで見てよい。CAN を guard_s だけ手前で切るのは、features.py の
  特徴量が中心合わせ (中心差分・center=True の rolling・対称カーネルの移動平均)
  で計算されており、時刻 t の値が最大 +0.45 秒先のサンプルを含むため。
  詳細は configs/vlm.yaml の context 節。

**評価区間**

  候補長は実測で 4.5〜29.2 秒とばらつく (中央 9.3 秒)。t_peak 中心の固定窓では
  長い候補を覆えないので、候補区間そのものを基準にする。

      t_start - pre_s  〜  t_end + post_s  を stride_s 刻み

  人手確認済み 32 件でこれを取ると 1,183 時刻になる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SegPos:
    """boot time をセグメント内の位置に直したもの。"""

    segment: int
    t_seg: float      # セグメント内の秒 [0, 60)
    frame: int        # セグメント内のフレーム番号 [0, 1200)


@dataclass(frozen=True)
class Episode:
    """評価の単位。labels.csv の 1 行に対応する。

    Phase 1 では人手確認済みの 32 件だけを扱う。risky は人手判定であって
    フィルタの出力ではない。「フィルタで落ちた = negative」とは扱わない。
    """

    event_id: str
    drive_id: str
    segment: int
    t_start: float          # boot time
    t_end: float
    t_in_segment_s: float   # t_start に対応するセグメント内の秒
    risky: bool
    note: str = ""
    verdict: str = ""
    event_types: str = ""

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    def to_segment(self, t: float, cfg: dict[str, Any]) -> SegPos:
        """boot time をセグメント内の位置に直す。

        セグメント境界を跨ぐ場合は繰り上げ・繰り下げる。実測で、評価区間は
        後方 (次セグメント) へ 7 件、前方 (前セグメント) へ 4 件はみ出す。
        フレームキャッシュは前後 1 本ずつを含めて作る必要がある。
        """
        seg_len = float(cfg["video"]["segment_len_s"])
        fps = float(cfg["video"]["fps"])
        n_frames = int(cfg["video"]["frames_per_segment"])

        off = self.t_in_segment_s + (t - self.t_start)
        shift = math.floor(off / seg_len)
        t_seg = off - shift * seg_len
        frame = int(round(t_seg * fps))
        # 境界での丸め上がり (59.98 秒 -> 1200) は次セグメントの先頭に送る。
        if frame >= n_frames:
            shift += 1
            t_seg -= seg_len
            frame = 0
        return SegPos(segment=self.segment + shift, t_seg=t_seg, frame=frame)

    def timeline(self, cfg: dict[str, Any]) -> list[float]:
        """オンライン判定 (モード B) の評価時刻を boot time で返す。"""
        tl = cfg["timeline"]
        stride = float(tl["stride_s"])
        lo = self.t_start - float(tl["pre_s"])
        hi = self.t_end + float(tl["post_s"])
        n = int(round((hi - lo) / stride)) + 1
        return [lo + i * stride for i in range(n)]

    def clip_span(self, cfg: dict[str, Any]) -> tuple[float, float]:
        """一括判定 (モード A) が見る区間。評価区間の全体と一致させる。

        モード A と B で見ている範囲を変えると、両者の差がモードの差なのか
        入力範囲の差なのか分からなくなる。同じ範囲にそろえる。
        """
        tl = cfg["timeline"]
        return (self.t_start - float(tl["pre_s"]), self.t_end + float(tl["post_s"]))


def video_frame_times(t: float, cfg: dict[str, Any]) -> list[float]:
    """評価時刻 t に対して渡す映像フレームの時刻 (boot time)。

    **最後の 1 枚がちょうど t** になるように後ろ揃えで並べる。
    先頭を揃えると t の直前が欠け、「今まさに起きていること」を見落とす。
    """
    ip = cfg["input"]
    fps = float(ip["video_fps"])
    n = int(round(float(ip["window_video_s"]) * fps))
    if n < 1:
        raise ValueError("window_video_s x video_fps が 1 未満です")
    return [t - (n - 1 - i) / fps for i in range(n)]


def can_sample_times(t: float, cfg: dict[str, Any]) -> list[float]:
    """評価時刻 t に対して渡す CAN サンプルの時刻 (boot time)。

    最後の 1 行が t - guard_s。guard 幅は結果メタに必ず記録する
    (検出遅れの分解能の下限を決めるため)。
    """
    ip = cfg["input"]
    guard = float(cfg["context"]["guard_s"])
    rate = float(ip["can_rate_hz"])
    n = int(round(float(ip["window_can_s"]) * rate))
    if n < 1:
        raise ValueError("window_can_s x can_rate_hz が 1 未満です")
    end = t - guard
    return [end - (n - 1 - i) / rate for i in range(n)]


def episodes_from_labels(df, cfg: dict[str, Any]) -> list[Episode]:
    """labels.csv (人手確認済み) を Episode に直す。

    event_id は drive_id / segment / t_start の順で通し番号を振る。
    positive は P01.., negative は N01.. とし、**個票を残せるようにする**。
    positive 8 件は集約値だけでなくイベント別の結果を保存する必要がある。
    """
    rows = df.sort_values(["risky", "drive_id", "segment", "t_start"],
                          ascending=[False, True, True, True])
    out: list[Episode] = []
    n_pos = n_neg = 0
    for _, r in rows.iterrows():
        risky = bool(r["risky"]) if not isinstance(r["risky"], str) else r["risky"] == "True"
        if risky:
            n_pos += 1
            eid = f"P{n_pos:02d}"
        else:
            n_neg += 1
            eid = f"N{n_neg:02d}"
        out.append(Episode(
            event_id=eid,
            drive_id=str(r["drive_id"]),
            segment=int(r["segment"]),
            t_start=float(r["t_start"]),
            t_end=float(r["t_end"]),
            t_in_segment_s=float(r["t_in_segment_s"]),
            risky=risky,
            note="" if _isna(r.get("note")) else str(r.get("note")),
            verdict="" if _isna(r.get("verdict")) else str(r.get("verdict")),
            event_types="" if _isna(r.get("event_types_at_labeling"))
                        else str(r.get("event_types_at_labeling")),
        ))
    return out


def _isna(v: Any) -> bool:
    try:
        return bool(np.asarray(v != v).all())
    except Exception:
        return v is None


@dataclass(frozen=True)
class Truncation:
    """評価区間を実在するセグメントへ切り詰めた記録。

    黙って縮めない。切り詰めた秒数を残さないと、pre-onset の誤警報率や
    検出遅れを「測れなかった」のか「出なかった」のか区別できなくなる。
    """

    pre_lost_s: float = 0.0
    post_lost_s: float = 0.0
    missing_segments: tuple[int, ...] = ()

    @property
    def any(self) -> bool:
        return bool(self.pre_lost_s or self.post_lost_s or self.missing_segments)


def available_segments(data_root, drive_id: str) -> set[int]:
    """手元にある (映像を持つ) セグメント番号。"""
    from pathlib import Path

    base = Path(data_root) / drive_id
    if not base.is_dir():
        return set()
    out = set()
    for p in base.iterdir():
        if p.is_dir() and p.name.isdigit() and (p / "video.hevc").is_file():
            out.add(int(p.name))
    return out


def timeline_available(
    ep: Episode,
    cfg: dict[str, Any],
    available: set[int],
    lookback_s: float = 0.0,
) -> tuple[list[float], Truncation]:
    """評価時刻のうち、映像が手元にあるものだけを返す。

    候補の開始時刻を含む**連続したかたまり**を採る。途中のセグメントが欠けて
    いる場合に穴の向こう側を拾うと、時間的に不連続な系列になってしまうため。

    lookback_s を渡すと、**その時刻から遡る映像窓が丸ごと揃っている**ことまで
    求める。評価時刻そのものが手元にあっても、4 秒前が欠けていれば入力を
    組めない。フレーム数が時刻によって変わると入力の分布が静かに変わり、
    モデル間の比較が成立しなくなるので、揃わない時刻は最初から外す。

    窓は 4 秒で 60 秒のセグメント境界を 2 つ跨げないため、両端を見れば足りる。
    """
    full = ep.timeline(cfg)
    if not full:
        return [], Truncation()

    def _ok(t: float) -> bool:
        if ep.to_segment(t, cfg).segment not in available:
            return False
        if lookback_s and ep.to_segment(t - lookback_s, cfg).segment not in available:
            return False
        return True

    ok = [_ok(t) for t in full]
    anchor = min(range(len(full)), key=lambda i: abs(full[i] - ep.t_start))
    if not ok[anchor]:
        missing = tuple(sorted({ep.to_segment(t, cfg).segment for t in full}
                               | {ep.to_segment(t - lookback_s, cfg).segment for t in full}
                               - available))
        return [], Truncation(pre_lost_s=full[anchor] - full[0],
                              post_lost_s=full[-1] - full[anchor],
                              missing_segments=missing)

    lo = anchor
    while lo - 1 >= 0 and ok[lo - 1]:
        lo -= 1
    hi = anchor
    while hi + 1 < len(full) and ok[hi + 1]:
        hi += 1

    missing = tuple(sorted(({ep.to_segment(t, cfg).segment
                             for t, o in zip(full, ok) if not o}
                            | {ep.to_segment(t - lookback_s, cfg).segment
                               for t, o in zip(full, ok) if not o})
                           - available))
    return full[lo:hi + 1], Truncation(
        pre_lost_s=round(full[lo] - full[0], 3),
        post_lost_s=round(full[-1] - full[hi], 3),
        missing_segments=missing,
    )


def from_segment(segment: int, t_seg: float, ep: Episode, cfg: dict[str, Any]) -> float:
    """セグメント内の秒を boot time に戻す。to_segment の逆。

    人手の注釈は動画を見て付けるのでセグメント内秒になる。判定は boot time で
    行うので、ここで一度だけ変換する。両方を持ち回らない。
    """
    seg_len = float(cfg["video"]["segment_len_s"])
    off = (segment - ep.segment) * seg_len + t_seg
    return ep.t_start + (off - ep.t_in_segment_s)
