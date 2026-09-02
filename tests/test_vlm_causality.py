"""VLM 後段レイヤの因果性の試験。

オンライン判定 (モード B) は「評価時刻 t より後の情報を一切与えない」ことを
前提にしている。この前提は気をつけるだけでは守れないので、機械的に検査する。

検査するのは 4 点。

  1. 映像フレームの時刻が t を超えないこと (最後の 1 枚がちょうど t)
  2. CAN の参照元グリッド時刻が t - guard_s を超えないこと
  3. 候補区間全体の集約値が文脈に混ざらないこと
  4. **漏れカナリア**: t より後のグリッド行を壊しても、組み上がる文脈が
     一字一句変わらないこと

4 が本命。1〜3 は書き忘れを拾うが、4 は経路全体を通した保証になる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from near_miss.config import load_yaml
from near_miss.vlm.context import GuardContext, make_context
from near_miss.vlm.windows import Episode, can_sample_times, video_frame_times

VLM_CONFIG = "configs/vlm.yaml"
RATE = 20.0


@pytest.fixture(scope="module")
def cfg():
    return load_yaml(VLM_CONFIG)


def make_grid(n: int = 2000, t0: float = 1000.0) -> pd.DataFrame:
    """20 Hz グリッドを合成する。値は時刻から決まる形にしておく。

    こうしておくと、行が差し替わったかどうかが値を見れば分かる。
    """
    t = t0 + np.arange(n) / RATE
    return pd.DataFrame({
        "t": t,
        "v_mps": 20.0 + 2.0 * np.sin(t),
        "ax_mps2": np.cos(t),
        "ay_kin_mps2": 0.5 * np.sin(2 * t),
        "yaw_rate_dps": np.sin(t / 2),
        "steer_deg_s": 3.0 * np.cos(t / 3),
        "brake_pressed": (np.sin(t) > 0.9).astype(float),
        "thw_s": 2.0 + np.sin(t),
        "ttc_s": 10.0 + np.cos(t),
    })


# --- 1. 映像 -------------------------------------------------------------
def test_映像フレームは評価時刻を超えない(cfg):
    t = 1234.5
    times = video_frame_times(t, cfg)
    assert len(times) == 8                      # 4.0 s x 2 fps
    assert max(times) == pytest.approx(t)       # 最後の 1 枚がちょうど t
    assert all(x <= t + 1e-12 for x in times)
    assert times == sorted(times)
    gaps = np.diff(times)
    assert np.allclose(gaps, 1.0 / float(cfg["input"]["video_fps"]))


# --- 2. CAN --------------------------------------------------------------
def test_CANサンプルはguard手前で切れる(cfg):
    t = 1234.5
    guard = float(cfg["context"]["guard_s"])
    times = can_sample_times(t, cfg)
    assert len(times) == 12                     # 6.0 s x 2 Hz
    assert max(times) == pytest.approx(t - guard)
    assert all(x <= t - guard + 1e-12 for x in times)


def test_参照したグリッド時刻がguardを超えない(cfg):
    df = make_grid()
    ctx = GuardContext(df, cfg)
    guard = float(cfg["context"]["guard_s"])
    for t in (1010.0, 1023.37, 1050.5, 1080.0):
        r = ctx.at(t)
        assert r.max_source_t <= t - guard + 1e-9, f"t={t} で未来を読んでいる"
        assert r.n_rows == 12


def test_グリッドの手前では欠測として扱い埋めない(cfg):
    df = make_grid(t0=1000.0)
    ctx = GuardContext(df, cfg)
    r = ctx.at(1002.0)          # 窓の一部がグリッド開始より前
    assert r.missing > 0
    assert "-" in r.text


# --- 3. 集約値の混入 -----------------------------------------------------
def test_禁止列は文脈に現れない(cfg):
    df = make_grid()
    # 候補区間全体を見て算出された値。混ざれば未来を知ることになる。
    df["severity"] = 4.2
    df["ax_mps2_min"] = -6.2
    df["ttc_s_min"] = 1.9
    df["event_types"] = "hard_brake"
    df["grade"] = "A"
    ctx = GuardContext(df, cfg)
    r = ctx.at(1050.0)

    for col in cfg["context"]["forbidden_columns"]:
        assert col not in r.columns
    for col in r.columns:
        for suf in cfg["context"]["forbidden_suffixes"]:
            assert not col.endswith(suf), f"{col} は集約列の疑い"
    assert "4.2" not in r.text and "hard_brake" not in r.text


# --- 4. 漏れカナリア (本命) ---------------------------------------------
@pytest.mark.parametrize("t", [1020.0, 1037.25, 1066.5])
def test_漏れカナリア_未来を壊しても文脈は不変(cfg, t):
    """t より後のグリッド行を破壊しても、組み上がる文脈が変わらないこと。

    変わるなら、経路のどこかが未来を読んでいる。
    """
    df = make_grid()
    before = GuardContext(df, cfg).at(t)

    rng = np.random.default_rng(0)
    broken = df.copy()
    future = broken["t"].to_numpy() > t - float(cfg["context"]["guard_s"])
    assert future.sum() > 0
    for col in broken.columns:
        if col == "t":
            continue
        broken.loc[future, col] = rng.normal(size=int(future.sum())) * 1e6

    after = GuardContext(broken, cfg).at(t)
    assert after.text == before.text, "未来の行を壊したら文脈が変わった = 未来を読んでいる"
    assert after.max_source_t == pytest.approx(before.max_source_t)


def test_make_contextはguardを返しcausalは未実装(cfg):
    df = make_grid()
    ctx = make_context(df, cfg)
    assert ctx.mode == "guard"
    assert ctx.guard_s == float(cfg["context"]["guard_s"])

    c2 = dict(cfg)
    c2["context"] = dict(cfg["context"], mode="causal")
    with pytest.raises(NotImplementedError):
        make_context(df, c2)


# --- セグメント対応 ------------------------------------------------------
def _episode(**kw) -> Episode:
    base = dict(event_id="P01", drive_id="d", segment=10, t_start=13023.91,
                t_end=13038.06, t_in_segment_s=45.95, risky=True)
    base.update(kw)
    return Episode(**base)


def test_セグメント境界を後方へ跨ぐ(cfg):
    """実データの例: seg10 の 45.95 秒から 14.15 秒続く候補は seg11 へ抜ける。"""
    ep = _episode()
    p0 = ep.to_segment(ep.t_start, cfg)
    assert (p0.segment, round(p0.t_seg, 2), p0.frame) == (10, 45.95, 919)

    p1 = ep.to_segment(ep.t_end, cfg)
    assert p1.segment == 11, "次セグメントへ繰り上がっていない"
    assert p1.t_seg == pytest.approx(0.10, abs=1e-6)
    assert p1.frame == 2


def test_セグメント境界を前方へ跨ぐ(cfg):
    """実データの例: seg1 の 0.60 秒から始まる候補は、6 秒前に遡ると seg0 に入る。"""
    ep = _episode(segment=1, t_start=78643.12, t_end=78650.37, t_in_segment_s=0.60)
    t = ep.t_start - float(cfg["timeline"]["pre_s"])
    p = ep.to_segment(t, cfg)
    assert p.segment == 0, "前セグメントへ繰り下がっていない"
    assert p.t_seg == pytest.approx(54.60, abs=1e-6)
    assert p.frame == 1092


def test_境界での丸め上がりは次セグメントの先頭になる(cfg):
    ep = _episode(segment=3, t_start=100.0, t_end=101.0, t_in_segment_s=59.99)
    p = ep.to_segment(100.0, cfg)       # 59.99 s -> frame 1199.8 -> 1200 に丸まる
    assert (p.segment, p.frame) == (4, 0)


def test_評価時刻の並び(cfg):
    ep = _episode(t_start=100.0, t_end=110.0)
    tl = ep.timeline(cfg)
    assert tl[0] == pytest.approx(94.0)          # t_start - 6
    assert tl[-1] == pytest.approx(112.0)        # t_end + 2
    assert len(tl) == 37                         # 18 s / 0.5 + 1
    assert np.allclose(np.diff(tl), 0.5)


def test_一括判定の区間はオンラインの評価区間と一致する(cfg):
    """モード A と B で見る範囲を変えると、差がモードの差か範囲の差か分からなくなる。"""
    ep = _episode(t_start=100.0, t_end=110.0)
    lo, hi = ep.clip_span(cfg)
    tl = ep.timeline(cfg)
    assert lo == pytest.approx(tl[0])
    assert hi == pytest.approx(tl[-1])


# --- 手元にないセグメントへの対処 ---------------------------------------
def test_評価区間を実在セグメントへ切り詰める(cfg):
    """実データの例: P08 は 6 秒前へ遡ると seg0 に入るが、Chunk_1 は seg1 から始まる。"""
    from near_miss.vlm.windows import timeline_available

    ep = _episode(segment=1, t_start=78643.12, t_end=78650.37, t_in_segment_s=0.60)
    full = ep.timeline(cfg)
    times, tr = timeline_available(ep, cfg, available={1, 2})

    assert len(times) < len(full)
    assert tr.pre_lost_s == pytest.approx(5.5, abs=0.5), "前側の欠落が記録されていない"
    assert tr.post_lost_s == 0.0
    assert 0 in tr.missing_segments
    # 残った時刻はすべて手元にあるセグメントに入る
    assert all(ep.to_segment(t, cfg).segment in {1, 2} for t in times)


def test_欠落が無ければ切り詰めない(cfg):
    from near_miss.vlm.windows import timeline_available

    ep = _episode(segment=20, t_start=11113.83, t_end=11118.83, t_in_segment_s=18.70)
    times, tr = timeline_available(ep, cfg, available={19, 20, 21})
    assert times == ep.timeline(cfg)
    assert not tr.any


def test_候補の開始そのものが欠けていれば空を返す(cfg):
    from near_miss.vlm.windows import timeline_available

    ep = _episode(segment=5, t_start=100.0, t_end=105.0, t_in_segment_s=10.0)
    times, tr = timeline_available(ep, cfg, available={9})
    assert times == []
    assert tr.any


# --- フレーム選択 --------------------------------------------------------
def test_選ぶフレームは後ろ揃えで最後が評価時刻(cfg):
    from near_miss.vlm.frames import frames_for

    ep = _episode(segment=20, t_start=11113.83, t_end=11118.83, t_in_segment_s=18.70)
    t = ep.t_start
    refs = frames_for(ep, t, cfg, cache_root="cache")

    assert len(refs) == 8
    assert refs[-1].t == pytest.approx(t)
    assert all(r.t <= t + 1e-12 for r in refs)
    # 候補開始 18.70 秒 = フレーム 374。3.5 秒前は 15.20 秒 = フレーム 304。
    assert (refs[-1].segment, refs[-1].frame) == (20, 374)
    assert (refs[0].segment, refs[0].frame) == (20, 304)
    assert refs[0].path.name == "f00304.jpg"


def test_映像窓は評価区間より前のセグメントを要求しうる(cfg):
    """映像は t から 4 秒遡るので、評価区間の先頭よりさらに前が要る。"""
    from near_miss.vlm.frames import needed_segments

    ep = _episode(segment=11, t_start=13042.76, t_end=13062.51, t_in_segment_s=4.78)
    segs = needed_segments(ep, cfg, ep.timeline(cfg))
    assert 10 in segs and 11 in segs


def test_映像窓が揃わない時刻は評価対象から外す(cfg):
    """P08 の実例: seg1 の 0.60 秒から始まる候補は、4 秒遡ると seg0 に届く。

    評価時刻そのものが seg1 にあってもフレームが揃わないので、入力を組めない。
    フレーム数が時刻によって変わると入力の分布が変わり、比較が成立しなくなる。
    """
    from near_miss.vlm.windows import timeline_available

    ep = _episode(segment=1, t_start=78643.12, t_end=78650.37, t_in_segment_s=0.60)
    look = float(cfg["input"]["window_video_s"])

    without, _ = timeline_available(ep, cfg, available={1, 2})
    with_look, tr = timeline_available(ep, cfg, available={1, 2}, lookback_s=look)

    assert len(with_look) < len(without)
    # 候補開始そのものが賄えないので、このエピソードは丸ごと落ちる
    assert with_look == []
    assert 0 in tr.missing_segments


def test_lookbackが要らなければ結果は変わらない(cfg):
    from near_miss.vlm.windows import timeline_available

    ep = _episode(segment=20, t_start=11113.83, t_end=11118.83, t_in_segment_s=18.70)
    a, _ = timeline_available(ep, cfg, available={19, 20, 21})
    b, tr = timeline_available(ep, cfg, available={19, 20, 21}, lookback_s=4.0)
    assert a == b and not tr.any


def test_履歴が足りない時刻は短い窓のまま渡す(cfg):
    """P08 の実例: 候補開始が seg1 の 0.60 秒で seg0 が手元に無い。

    時刻ごと捨てると危険が立ち上がるまさにその区間が消えるので、
    古い側のフレームだけ落として短い窓のまま渡す。
    **最後の 1 枚が評価時刻であることは変わらない。**
    """
    from near_miss.vlm.frames import frames_available

    ep = _episode(segment=1, t_start=78643.12, t_end=78650.37, t_in_segment_s=0.60)
    t = ep.t_start          # seg1 の 0.60 秒
    refs, partial = frames_available(ep, t, cfg, "cache", available={1, 2})

    assert partial is True
    assert 0 < len(refs) < 8
    assert all(r.segment in {1, 2} for r in refs)
    assert refs[-1].t == pytest.approx(t), "末尾は評価時刻のままであること"
    assert all(r.t <= t + 1e-12 for r in refs)


def test_履歴が足りていれば部分窓にならない(cfg):
    from near_miss.vlm.frames import frames_available

    ep = _episode(segment=20, t_start=11113.83, t_end=11118.83, t_in_segment_s=18.70)
    refs, partial = frames_available(ep, ep.t_start, cfg, "cache", available={19, 20, 21})
    assert partial is False and len(refs) == 8


def test_span_と_extremes_はモードAのみで使う(cfg):
    """区間全体と極値はモード A 専用。モード B に渡せば未来を与えることになる。"""
    df = make_grid()
    ctx = GuardContext(df, cfg)
    r = ctx.span(1010.0, 1040.0)
    assert r.n_rows == 61                       # 30 s x 2 Hz + 1
    assert r.max_source_t <= 1040.0 + 1e-9
    ex = ctx.extremes(1010.0, 1040.0)
    assert "最小" in ex and "最大" in ex
    # at() (モード B 用) は区間全体を返さない
    assert ctx.at(1040.0).n_rows == 12
