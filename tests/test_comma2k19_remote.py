"""near_miss.io.comma2k19_remote の試験。ネットワークは使わず、メモリ上の zip で確かめる。"""

from __future__ import annotations

import io
import zipfile

import pytest

from near_miss.io import comma2k19_remote as rz

DRIVE = "99c94dc769b5d96e|2018-06-09--19-18-40"


class _MemFile:
    """HttpFile の代わり。read_range だけを持つ。"""

    def __init__(self, data: bytes):
        self.data, self.size, self.nreq = data, len(data), 0

    def read_range(self, start: int, end: int) -> bytes:
        self.nreq += 1
        return self.data[start:end]


def _zip() -> tuple[bytes, dict[str, bytes]]:
    files = {
        "raw_log.bz2": b"x" * 5000,
        "video.hevc": bytes(range(256)) * 400,
        "processed_log/CAN/speed/t": b"\x01\x02" * 300,
        "processed_log/CAN/speed/value": b"\x03" * 600,
        "processed_log/CAN/radar/value": b"r" * 900,
        "processed_log/IMU/gyro/value": bytes(range(200)) * 3,
        "global_pose/frame_times": b"\x00" * 128,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, (rel, data) in enumerate(files.items()):
            kind = zipfile.ZIP_STORED if i % 2 else zipfile.ZIP_DEFLATED
            z.writestr(f"Chunk_4/{DRIVE}/35/{rel}", data, compress_type=kind)
        z.writestr(f"Chunk_4/{DRIVE}/36/processed_log/CAN/speed/t", b"other")
    return buf.getvalue(), files


def test_members_follow_include_and_segment():
    data, _ = _zip()
    z = zipfile.ZipFile(io.BytesIO(data))
    rels = [m.rel for m in rz.segment_members(z, 4, DRIVE, 35)]
    assert rels == ["processed_log/CAN/speed/t", "processed_log/CAN/speed/value",
                    "processed_log/IMU/gyro/value", "global_pose/frame_times"]   # radar・映像は取らない


def test_fetch_segment_reads_one_span_and_checks_crc(tmp_path):
    data, files = _zip()
    z = zipfile.ZipFile(io.BytesIO(data))
    members = rz.segment_members(z, 4, DRIVE, 35)
    f = _MemFile(data)
    st = rz.fetch_segment(f, members, tmp_path)
    assert st["mode"] == "span" and f.nreq == 1
    for m in members:
        assert (tmp_path / m.rel).read_bytes() == files[m.rel]
    # 2 回目は取得済みなので読まない
    assert rz.fetch_segment(f, members, tmp_path)["mode"] == "skip" and f.nreq == 1


def test_fetch_segment_falls_back_to_per_file(tmp_path):
    data, files = _zip()
    z = zipfile.ZipFile(io.BytesIO(data))
    members = rz.segment_members(z, 4, DRIVE, 35, include=("raw_log.bz2", "global_pose/*"))
    f = _MemFile(data)
    st = rz.fetch_segment(f, members, tmp_path, max_span=1000)
    assert st["mode"] == "per_file" and f.nreq == 2
    assert (tmp_path / "raw_log.bz2").read_bytes() == files["raw_log.bz2"]


def test_crc_mismatch_is_rejected():
    data, _ = _zip()
    z = zipfile.ZipFile(io.BytesIO(data))
    m = rz.segment_members(z, 4, DRIVE, 35)[1]          # STORED のファイル
    bad = bytearray(data)
    start = m.info.header_offset + 30 + len(m.info.filename.encode())
    bad[start] ^= 0xFF
    with pytest.raises(ValueError, match="CRC"):
        rz.extract_from_buffer(bytes(bad), 0, m.info)
