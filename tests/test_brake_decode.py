"""制動の信号の復号と、制動の評価の計算の試験 (docs/brake_performance_rav4.md)。

- RAV4 2017 の 0x224 byte4〜5 (BRAKE_LEVEL) を車種設定のとおりに復号できること
- opendbc の BRAKE_PRESSURE の位置 (start 43, 12 bit) では 256 の倍数になること (無効にしてある理由)
- Civic の信号 (configs/brake/brake_performance.yaml) の復号
- global_pose の速度からの勾配、熱の指標
- 実データ (raw_data があるときだけ): BRAKE_LEVEL の振る舞い
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "brake"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import brake_data as BD  # noqa: E402
import brake_model as BM  # noqa: E402
from near_miss.config import VehicleConfig  # noqa: E402
from near_miss.io.can_decode import decode_signal, payload_to_u64  # noqa: E402

RAV4 = VehicleConfig.load(REPO / "configs/vehicles/toyota_rav4.yaml")
CFG = yaml.safe_load(open(REPO / "configs/brake/brake_performance.yaml"))


def _spec(vc, name):
    return next(s for s in vc.signals if s.name == name)


def _frames(rows, addr, src):
    data = np.array([bytes(r) for r in rows], dtype="S8")
    n = len(rows)
    return np.arange(n, dtype=float), np.full(n, addr), np.full(n, src), payload_to_u64(data)


def test_rav4_brake_level_position():
    # byte0 bit5 = 踏んでいる、byte4 bit0 + byte5 = 0x12C = 300
    t, a, s, p = _frames([[0x20, 0, 0, 0, 0x01, 0x2C, 0, 0x08], [0, 0, 0, 0, 0, 0, 0, 0x08], [0x20, 0, 0, 0, 0, 0x4F, 0, 0x08]], 548, 0)
    _, lev = decode_signal(t, a, p, _spec(RAV4, "BRAKE_LEVEL"), s)
    _, prs = decode_signal(t, a, p, _spec(RAV4, "BRAKE_PRESSED"), s)
    assert lev.tolist() == [300, 0, 79]
    assert prs.tolist() == [1, 0, 1]
    spec = _spec(RAV4, "BRAKE_LEVEL")
    assert spec.enabled and spec.channel == "brake_level_raw"
    # 既存の検出 (features.py) が自動で使う名前と重ならない
    assert spec.channel not in ("brake_pressure", "brake_position")


def test_rav4_dbc_brake_pressure_position_gives_multiples_of_256():
    t, a, s, p = _frames([[0x20, 0, 0, 0, 0x01, 0x2C, 0, 0x08], [0x20, 0, 0, 0, 0, 0x4F, 0, 0x08]], 548, 0)
    _, v = decode_signal(t, a, p, _spec(RAV4, "BRAKE_PRESSURE"), s)
    # byte5 の下位 4 bit と byte6 (常に 0) を読むので 256 の倍数になる
    assert v.tolist() == [0xC * 256, 0xF * 256]
    assert not _spec(RAV4, "BRAKE_PRESSURE").enabled


def test_rav4_brake_level_ignores_other_bus():
    # 0x224 はバス 1 に別のメッセージ (先頭がカウンタ) が流れる。バス 0 だけを読む
    t, a, s, p = _frames([[0x20, 0, 0, 0, 0x01, 0x2C, 0, 0x08]], 548, 1)
    tt, v = decode_signal(t, a, p, _spec(RAV4, "BRAKE_LEVEL"), s)
    assert v.size == 0


def test_civic_signals_from_poc_config():
    sigs = BD.car_signals(CFG["cars"]["civic"])
    # USER_BRAKE: 7|16 big endian、0.015625、-1.609375。生の値 103 で 0
    t, a, s, p = _frames([[0x00, 0x67, 0x80, 0, 0, 0, 0, 0], [0x00, 0xA7, 0x00, 0, 0, 0, 0, 0]], 420, 0)
    _, ub = BD.decode(t, a, s, p, sigs["USER_BRAKE"])
    assert ub.tolist() == pytest.approx([0.0, 1.0])
    _, cb = BD.decode(t, a, s, p, sigs["COMPUTER_BRAKING"])     # 0x1A4 bit 23 = byte2 の bit7
    assert cb.tolist() == [1, 0]
    # BRAKE_PRESSURE1: 7|10 = byte0 の 8 bit + byte1 の上位 2 bit
    t, a, s, p = _frames([[0xCA, 0x80, 0, 0, 0, 0, 0, 0]], 487, 0)
    _, bp = BD.decode(t, a, s, p, sigs["BRAKE_PRESSURE1"])
    assert bp[0] == pytest.approx(((0xCA << 2) | 2) * 0.015625 - 103)
    # 送信フレーム (openpilot) は車両側の信号として読まない
    t, a, s, p = _frames([[0x00, 0xA7, 0x00, 0, 0, 0, 0, 0]], 420, 128)
    _, ub = BD.decode(t, a, s, p, sigs["USER_BRAKE"])
    assert ub.size == 0


def test_rav4_acc_command_reads_only_tx_frames():
    sigs = BD.car_signals(CFG["cars"]["rav4"])
    raw = (-1500) & 0xFFFF                          # -1.5 m/s^2
    row = [raw >> 8, raw & 0xFF, 0, 0, 0, 0, 0, 0]
    t, a, s, p = _frames([row, row], 835, 128)
    _, v = BD.decode(t, a, s, p, sigs["ACC_ACCEL_CMD"])
    assert v.tolist() == pytest.approx([-1.5, -1.5])
    t, a, s, p = _frames([row], 835, 0)              # 受信フレームは読まない
    _, v = BD.decode(t, a, s, p, sigs["ACC_ACCEL_CMD"])
    assert v.size == 0


def _llh_to_ecef(lat, lon, h=0.0):
    a, e2 = 6378137.0, 6.69437999014e-3
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    N = a / np.sqrt(1 - e2 * np.sin(la) ** 2)
    return np.array([(N + h) * np.cos(la) * np.cos(lo), (N + h) * np.cos(la) * np.sin(lo), (N * (1 - e2) + h) * np.sin(la)])


def test_grade_from_ecef_uses_geodetic_up():
    lat, lon = 37.4, -122.2
    pos = _llh_to_ecef(lat, lon, 100.0)[None, :]
    la, lo = BM.ecef_to_geodetic(pos)
    assert float(la) == pytest.approx(lat, abs=1e-6) and float(lo) == pytest.approx(lon, abs=1e-6)
    e, n, u = BM.enu_basis(np.array([lat]), np.array([lon]))
    vel = (20.0 * n + 1.0 * u)                         # 北へ 20 m/s、上へ 1 m/s
    th, _, _, hd = BM.grade_from_ecef(pos, vel, 3.0)
    assert th[0] == pytest.approx(np.arctan2(1.0, 20.0), abs=1e-9)
    assert hd[0] == pytest.approx(0.0, abs=1e-6)
    th, _, _, _ = BM.grade_from_ecef(pos, 1.0 * n, 3.0)  # 遅すぎると向きが定まらないので欠損
    assert np.isnan(th[0])


def test_heat_indices_constant_power():
    dt = 0.05
    n = int(600 / dt)
    p = np.full(n, 2.0)
    H = BM.heat_indices(p, np.zeros(n, int), dt, [300], [60, 900])
    assert H["E60"][-1] == pytest.approx(120.0, rel=1e-3)
    assert H["E900"][-1] == pytest.approx(1200.0, rel=1e-3)        # 窓より短いので全部
    assert H["H300"][-1] == pytest.approx(2.0 * 300 * (1 - np.exp(-2)), rel=2e-3)
    # 塊の頭で 0 から始める
    blk = np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)]
    H = BM.heat_indices(p, blk, dt, [300], [900])
    assert H["E900"][-1] == pytest.approx(600.0, rel=1e-3)


SEG = REPO / "raw_data/Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/3"


@pytest.mark.skipif(not SEG.is_dir(), reason="comma2k19 Chunk_1 が無い")
def test_rav4_brake_level_behaviour_on_real_segment():
    cfg = dict(CFG)
    df = BD.load_segment(SEG, cfg["cars"]["rav4"], BD.car_signals(cfg["cars"]["rav4"]), cfg)
    df = df[~df.edge]
    pressed = df.pressed > 0.5
    moving = df.v > 3
    # 踏んでいないときはほぼ 0 (Chunk 1〜2 全体では 0〜5)
    assert (df.level[~pressed & moving] <= 5).mean() > 0.99
    # 踏んでいる間は減速度と強く相関する
    m = pressed & moving
    assert m.sum() > 100
    assert np.corrcoef(df.level[m], -df.a[m])[0, 1] > 0.7
    # 値は 9 bit に収まる
    assert df.level.max() < 512
