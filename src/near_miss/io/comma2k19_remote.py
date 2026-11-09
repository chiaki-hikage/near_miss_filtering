"""comma2k19 の Chunk_N.zip (Hugging Face) から、要るファイルだけを HTTP の範囲指定で取る。

Chunk_N.zip は 1 個 9 GB 前後で、大半は video.hevc と raw_log.bz2。
CAN・IMU・global_pose だけなら 1 セグメントあたり圧縮後 1.5 MB 程度で済む。

    1. zip の末尾の目次 (central directory, 約 2 MB) だけを読む
    2. 1 セグメントの中で要るファイルは zip の中で続けて並んでいるので、
       その範囲を 1 回の範囲指定でまとめて読む
    3. 各ファイルの local header を解いて展開し、目次の CRC と照合する

書き出す構成は Chunk_N.zip を展開したときと同じ
(<out>/<dongle>|<route>/<seg>/processed_log/...) なので、io.comma2k19 でそのまま読める。
"""

from __future__ import annotations

import fnmatch
import io
import struct
import time
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

CHUNK_URL = "https://huggingface.co/datasets/commaai/comma2k19/resolve/main/raw_data/Chunk_{n}.zip"

# 既定で取るファイル (セグメントのディレクトリからの相対パス)
DEFAULT_INCLUDE = (
    "processed_log/CAN/speed/*",
    "processed_log/CAN/steering_angle/*",
    "processed_log/CAN/wheel_speed/*",
    "processed_log/CAN/raw_can/*",
    "processed_log/IMU/gyro/*",
    "processed_log/IMU/gyro_bias/*",
    "processed_log/IMU/gyro_uncalibrated/*",
    "processed_log/IMU/accelerometer/*",
    "global_pose/*",
)

LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")   # 30 バイト
LOCAL_SIG = b"PK\x03\x04"


class HttpFile(io.RawIOBase):
    """HTTP の範囲指定で読む、シーク可能なファイル。読んだバイト数と要求回数を数える。"""

    def __init__(self, url: str, retries: int = 8, timeout: float = 120.0):
        self.retries, self.timeout = retries, timeout
        req = urllib.request.Request(url, method="HEAD")
        with self._open(req) as r:
            self.url = r.geturl()                      # リダイレクト先 (CDN) を覚えておく
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.nbytes = 0
        self.nreq = 0

    def _open(self, req):
        for k in range(self.retries):
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except Exception:
                if k == self.retries - 1:
                    raise
                time.sleep(3 * (k + 1))

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read_range(self, start: int, end: int) -> bytes:
        """[start, end) を読む。"""
        if end <= start:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        for k in range(self.retries):
            try:
                with self._open(req) as r:
                    data = r.read()
                if len(data) != end - start:
                    raise IOError(f"範囲 {start}-{end} の応答が {len(data)} バイト")
                break
            except Exception:
                if k == self.retries - 1:
                    raise
                time.sleep(3 * (k + 1))
        self.nbytes += len(data)
        self.nreq += 1
        return data

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        data = self.read_range(self.pos, min(self.pos + n, self.size))
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def open_chunk(n: int) -> tuple[HttpFile, zipfile.ZipFile]:
    f = HttpFile(CHUNK_URL.format(n=n))
    return f, zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16))


@dataclass(frozen=True)
class Member:
    rel: str                 # セグメントのディレクトリからの相対パス
    info: zipfile.ZipInfo


def segment_members(z: zipfile.ZipFile, chunk: int, drive: str, seg: int | str,
                    include: tuple[str, ...] = DEFAULT_INCLUDE) -> list[Member]:
    """目次から、そのセグメントの要るファイルを zip の中の順に返す。"""
    prefix = f"Chunk_{chunk}/{drive}/{seg}/"
    out = []
    for i in z.infolist():
        if not i.filename.startswith(prefix) or i.is_dir():
            continue
        rel = i.filename[len(prefix):]
        if any(fnmatch.fnmatchcase(rel, p) for p in include):
            out.append(Member(rel, i))
    return sorted(out, key=lambda m: m.info.header_offset)


def extract_from_buffer(buf: bytes, base: int, info: zipfile.ZipInfo) -> bytes:
    """zip の base バイト目から始まる buf の中から、info のファイルを取り出して CRC を照合する。"""
    off = info.header_offset - base
    head = LOCAL_HEADER.unpack_from(buf, off)
    if head[0] != LOCAL_SIG:
        raise ValueError(f"local header の署名が違う: {info.filename}")
    name_len, extra_len = head[9], head[10]
    start = off + LOCAL_HEADER.size + name_len + extra_len
    raw = buf[start:start + info.compress_size]
    if len(raw) != info.compress_size:
        raise ValueError(f"読んだ範囲が足りない: {info.filename}")
    if info.compress_type == zipfile.ZIP_STORED:
        data = raw
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(raw, -15)
    else:
        raise ValueError(f"未対応の圧縮形式 {info.compress_type}: {info.filename}")
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise ValueError(f"CRC か大きさが目次と合わない: {info.filename}")
    return data


def span_of(members: list[Member], slack: int = 1024) -> tuple[int, int]:
    """要るファイルをすべて含む範囲 [start, end)。local header の可変長部分の分だけ余分に取る。"""
    start = min(m.info.header_offset for m in members)
    end = max(m.info.header_offset + LOCAL_HEADER.size + len(m.info.filename.encode()) + slack
              + m.info.compress_size for m in members)
    return start, end


def fetch_segment(f: HttpFile, members: list[Member], dest: Path, max_span: int = 16 << 20) -> dict:
    """1 セグメント分を dest に書く。既に同じ大きさのファイルがあれば取り直さない。

    f.read_range は位置を持たないので、複数のスレッドで 1 つの f を共有してよい。

    範囲が max_span を超えるとき (要るファイルの間に映像などが挟まるとき) は、ファイルごとに読む。
    """
    todo = [m for m in members
            if not ((dest / m.rel).is_file() and (dest / m.rel).stat().st_size == m.info.file_size)]
    stat = {"n_files": len(members), "n_fetched": len(todo), "bytes_read": 0, "mode": "skip"}
    if not todo:
        return stat
    start, end = span_of(todo)
    end = min(end, f.size)
    groups = [todo] if end - start <= max_span else [[m] for m in todo]
    stat["mode"] = "span" if len(groups) == 1 else "per_file"
    for g in groups:
        s, e = span_of(g)
        e = min(e, f.size)
        buf = f.read_range(s, e)
        stat["bytes_read"] += len(buf)
        for m in g:
            data = extract_from_buffer(buf, s, m.info)
            path = dest / m.rel
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(path)
    return stat
