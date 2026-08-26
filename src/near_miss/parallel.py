"""ドライブ単位の並列実行。

**判定は一切変えない。** 1 ドライブ分の処理 (読み出し → グリッド → 特徴量 →
横滑りフィルタ) はもともと他のドライブと何も共有していないので、
そのまとまりをそのままプロセスへ配るだけにしてある。

なぜセグメント単位ではなくドライブ単位か
----------------------------------------
セグメントは 60 秒で切られていて、イベントは境界を跨ぐ。連番のセグメントは
連結してから 20 Hz グリッドに載せる必要がある (pipeline.split_contiguous)。
セグメント単位で配るとこの連結が壊れ、境界付近の候補が変わってしまう。
ドライブ単位なら連結の単位がプロセス内に収まるので、結果は逐次実行と同じになる。

commaCarSegments の RAV4 TSS2 2,000 セグメントでは
    ドライブ 877 / 連続ブロック 877 / 1 ドライブあたり 1〜10 セグメント (中央 2)
なので、粒度としても細かすぎず粗すぎない。

結果を同じにするために守っていること
------------------------------------
* 1 ドライブ分の計算そのものは逐次でも並列でも同じ関数を通る
* 戻ってきた結果を **投入順に並べ直してから** 連結する。
  候補の最終的な並べ替えは |beta| の降順だが、pandas の既定の整列は
  安定ではないので、同じ値が並んだときの順が入力順で決まる。
  順序を保たないと worker 数で行の並びが変わりうる。
* 集計 (FilterCounts) は足し算だけなので順序に依存しない

OS 依存を避けるために守っていること
----------------------------------
* 起動方式を **spawn に固定**する。macOS の既定は spawn、Linux の既定は fork で、
  そのままだと OS で挙動が変わる。fork は親のメモリを引き継ぐぶん速いが、
  スレッドを持ったまま fork した場合の振る舞いが環境依存になる。
  ここでは「どちらでも同じ」を優先する。
* worker へ渡すものは設定・車種定義・ファイルの場所だけにする。
  ラムダやファイルハンドルなど、pickle できないものを渡さない。
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .config import VehicleConfig
from .features import compute_features
from .io.canonical import SegmentRef, concat_segments
from .pipeline import split_contiguous
from .sideslip import FilterCounts, SideslipCandidate, find_sideslip_candidates, stage1_mask
from .signals import to_grid

log = logging.getLogger(__name__)

# データセットの識別子。worker には文字列で渡す (読み出し関数は pickle できない)。
DATASET_CAR_SEGMENTS = "comma_car_segments"
DATASET_COMMA2K19 = "comma2k19"

# プロセス並列とライブラリ内スレッドが二重に増えないようにする。
# 抽出処理に行列積は無いので効きは小さいはずだが、環境によっては
# numpy が既定で複数スレッドを立てるため、念のため 1 に寄せる。
# 子プロセスは spawn で起きるので、**親で設定してから** pool を作る必要がある
# (initializer は numpy の import より後に走るので間に合わない)。
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass
class DriveTask:
    """1 ドライブ分の入力。pickle して worker へ渡す。"""

    order: int                  # 投入順。結果を並べ直すのに使う
    drive_id: str
    refs: list[SegmentRef]


@dataclass
class BlockOutcome:
    """連続ブロック 1 つ分の結果。"""

    spans: list[tuple[int, float, float]]
    candidates: list[SideslipCandidate]
    dump: Any = None            # pd.DataFrame | None (--dump-stage1 のときだけ)
    n_segments: int = 0


@dataclass
class DriveOutcome:
    """1 ドライブ分の結果。DataFrame にするのは親側でやる。"""

    order: int
    drive_id: str
    blocks: list[BlockOutcome] = field(default_factory=list)
    counts: FilterCounts = field(default_factory=FilterCounts)
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def n_segments(self) -> int:
        return sum(b.n_segments for b in self.blocks)


# worker ごとの状態。initializer が入れる。
_STATE: dict[str, Any] = {}


def _loader(dataset: str) -> Callable[[SegmentRef, VehicleConfig], Any]:
    """データセット名から読み出し関数を作る。worker の中で解決する。"""
    if dataset == DATASET_CAR_SEGMENTS:
        from .io import comma_car_segments as ccs

        return lambda ref, veh: ccs.load_segment(ref, veh, with_raw_can=True)
    if dataset == DATASET_COMMA2K19:
        from .io import comma2k19

        return lambda ref, veh: comma2k19.load_segment(ref, veh, with_raw_can=True)
    raise ValueError(f"未知のデータセットです: {dataset}")


def init_worker(
    cfg: dict[str, Any],
    vehicle: VehicleConfig,
    dataset: str,
    dump_stage1: bool,
    dump_columns: tuple[str, ...],
) -> None:
    """worker の初期化。タスクごとに設定を送り直さずに済むよう、一度だけ入れる。"""
    _STATE.clear()
    _STATE.update(
        cfg=cfg,
        vehicle=vehicle,
        load=_loader(dataset),
        dump_stage1=dump_stage1,
        dump_columns=dump_columns,
    )


def run_drive(task: DriveTask) -> DriveOutcome:
    """1 ドライブ分を処理する。逐次でも並列でもこの関数を通る。

    例外は握って errors に積む。1 ドライブの破損で全体を止めない。
    """
    import pandas as pd

    cfg = _STATE["cfg"]
    vehicle = _STATE["vehicle"]
    load = _STATE["load"]
    out = DriveOutcome(order=task.order, drive_id=task.drive_id)
    t0 = time.perf_counter()

    for block in split_contiguous(task.refs):
        segs: list[Any] = []
        spans: list[tuple[int, float, float]] = []
        for ref in block:
            try:
                sd = load(ref, vehicle)
            except Exception as exc:
                out.errors.append(f"読み出し失敗 {ref.segment_id}: {exc}")
                continue
            a, b = sd.t_span
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            spans.append((ref.index, float(a), float(b)))
            segs.append(sd)
        if not segs:
            continue

        merged = concat_segments(segs)
        gs = to_grid(merged, cfg)
        if gs.df.empty:
            continue
        gs.meta["segment_spans"] = spans
        gs = compute_features(gs, cfg, radar=merged.radar, vehicle=vehicle)

        cands, counts = find_sideslip_candidates(gs, cfg)
        out.counts.add(counts)

        dump = None
        if _STATE["dump_stage1"]:
            m1, _ = stage1_mask(gs.df, cfg)
            if m1.any():
                cols = [c for c in _STATE["dump_columns"] if c in gs.df.columns]
                dump = gs.df.loc[m1, cols].copy()
                dump.insert(0, "drive_id", task.drive_id)
        out.blocks.append(
            BlockOutcome(spans=spans, candidates=cands, dump=dump, n_segments=len(segs))
        )

    out.elapsed_s = time.perf_counter() - t0
    return out


def build_tasks(refs: list[SegmentRef]) -> list[DriveTask]:
    """セグメント一覧をドライブ単位のタスクに分ける。並びは決まった順。"""
    from .io.canonical import group_by_drive

    return [
        DriveTask(order=i, drive_id=drive_id, refs=drive_refs)
        for i, (drive_id, drive_refs) in enumerate(group_by_drive(refs).items())
    ]


def map_drives(
    tasks: list[DriveTask],
    cfg: dict[str, Any],
    vehicle: VehicleConfig,
    dataset: str,
    workers: int = 1,
    dump_stage1: bool = False,
    dump_columns: tuple[str, ...] = (),
    chunksize: int = 4,
) -> Iterator[DriveOutcome]:
    """タスクを処理して、**投入順のまま**結果を返す。

    workers <= 1 なら同じプロセスで回す。プロセスを起こさないぶん速く、
    並列版とまったく同じ関数を通るので結果も同じになる。
    """
    if workers <= 1:
        init_worker(cfg, vehicle, dataset, dump_stage1, dump_columns)
        for task in tasks:
            yield run_drive(task)
        return

    # 子プロセスは spawn で起きるので、環境変数はここで入れておく。
    for name in THREAD_ENV:
        os.environ.setdefault(name, "1")

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    ctx = mp.get_context("spawn")      # macOS / Linux で同じ挙動にする
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(cfg, vehicle, dataset, dump_stage1, dump_columns),
    ) as ex:
        # map は投入順で返す。結果の並びが worker 数で変わらない。
        yield from ex.map(run_drive, tasks, chunksize=chunksize)


def resolve_workers(requested: int | None) -> int:
    """--workers の指定を実際の数にする。0 や None なら CPU 数に合わせる。"""
    if requested is None or requested <= 0:
        return max(1, (os.cpu_count() or 1))
    return int(requested)
