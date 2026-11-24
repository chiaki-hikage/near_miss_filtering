"""ROS1 bag (v2.0) を HTTP の範囲指定で一部だけ読む (標準ライブラリだけ、依存を足さない)。

TartanDrive 2.0 の rosbag は 1 ファイル約 32 GB。末尾の索引 (約 2 MB) で、話題ごとの接続と、チャンクの位置・時刻の範囲を読み、
必要な時刻のチャンク (1 つ約 4 MB) だけを取る。圧縮なしのチャンクだけに対応する (確かめた bag は compression=none)。

メッセージの復号は、使う型だけ書いている:
- sensor_msgs/Imu: header, orientation(4), cov(9), angular_velocity(3), cov(9), linear_acceleration(3), cov(9)
- racepak の rp_shock_sensors / rp_wheel_encoders / rp_controls: header は無く float32 が並ぶ (castacks/tartan_drive の physics_atv_racepak/msg)
  ただし、TartanDrive 2.0 の bag の中の型の定義 (接続の message_definition) を見て確かめてから使う
"""

from __future__ import annotations

import struct
import urllib.request
from dataclasses import dataclass, field

OP_MSG, OP_BAG, OP_IDX, OP_CHUNK, OP_CHUNK_INFO, OP_CONN = 0x02, 0x03, 0x04, 0x05, 0x06, 0x07


CACHE_DIR = None   # Path を入れると、取った範囲をファイルに残し、同じ範囲は取り直さない


def _cache_path(url: str, start: int, end_incl: int):
    import hashlib
    from pathlib import Path
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    return Path(CACHE_DIR) / h / f"{start}-{end_incl}.bin"


def http_range(url: str, start: int, end_incl: int, timeout: float = 120) -> bytes:
    if CACHE_DIR is not None:
        cp = _cache_path(url, start, end_incl)
        if cp.exists():
            return cp.read_bytes()
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end_incl}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status != 206:
            raise RuntimeError(f"範囲指定が効かない (status {r.status})")
        b = r.read()
    if CACHE_DIR is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.with_suffix(".part").write_bytes(b)
        cp.with_suffix(".part").rename(cp)
    return b


def cached(url: str, start: int, end_incl: int) -> bool:
    return CACHE_DIR is not None and _cache_path(url, start, end_incl).exists()


def http_size(url: str, timeout: float = 60) -> int:
    if CACHE_DIR is not None:
        from pathlib import Path
        cp = _cache_path(url, 0, 0).with_name("size.txt")
        if cp.exists():
            return int(cp.read_text())
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout) as r:
        n = int(r.headers["Content-Length"])
    if CACHE_DIR is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(str(n))
    return n


def _fields(h: bytes) -> dict[str, bytes]:
    f, q = {}, 0
    while q < len(h):
        n = struct.unpack_from("<I", h, q)[0]
        k, v = h[q + 4:q + 4 + n].split(b"=", 1)
        f[k.decode()] = v
        q += 4 + n
    return f


def iter_records(b: bytes, p: int = 0):
    """(header の dict, data の bytes) を順に返す。"""
    while p + 4 <= len(b):
        hl = struct.unpack_from("<I", b, p)[0]
        h = _fields(b[p + 4:p + 4 + hl])
        dl = struct.unpack_from("<I", b, p + 4 + hl)[0]
        d0 = p + 8 + hl
        yield h, b[d0:d0 + dl]
        p = d0 + dl


def _time(v: bytes) -> float:
    s, ns = struct.unpack("<II", v)
    return s + ns * 1e-9


@dataclass
class BagIndex:
    url: str
    size: int
    conns: dict[int, dict] = field(default_factory=dict)       # id -> {topic, type, definition}
    chunks: list[dict] = field(default_factory=list)            # {pos, start, end, counts{conn: n}}
    index_bytes: int = 0                                        # 索引を読むのに取った量

    def topic_ids(self, topic: str) -> list[int]:
        return [i for i, c in self.conns.items() if c["topic"] == topic]


def read_index(url: str) -> BagIndex:
    size = http_size(url)
    head = http_range(url, 0, 4095 + 13)
    assert head[:13] == b"#ROSBAG V2.0\n", head[:13]
    h, _ = next(iter_records(head, 13))
    index_pos = struct.unpack("<Q", h["index_pos"])[0]
    tail = http_range(url, index_pos, size - 1)
    bi = BagIndex(url=url, size=size, index_bytes=len(head) + len(tail))
    for h, d in iter_records(tail):
        op = h["op"][0]
        if op == OP_CONN:
            cid = struct.unpack("<I", h["conn"])[0]
            cf = _fields(d)
            bi.conns[cid] = dict(topic=h["topic"].decode(), type=cf.get("type", b"").decode(),
                                 definition=cf.get("message_definition", b"").decode(errors="replace"))
        elif op == OP_CHUNK_INFO:
            n = struct.unpack("<I", h["count"])[0]
            counts = {struct.unpack_from("<I", d, 8 * i)[0]: struct.unpack_from("<I", d, 8 * i + 4)[0] for i in range(n)}
            bi.chunks.append(dict(pos=struct.unpack("<Q", h["chunk_pos"])[0], start=_time(h["start_time"]),
                                  end=_time(h["end_time"]), counts=counts))
    bi.chunks.sort(key=lambda c: c["pos"])
    return bi


def read_chunk_messages(bi: BagIndex, chunk: dict, conn_ids: set[int]) -> list[tuple[int, float, bytes]]:
    """1 つのチャンクを取り、指定の接続のメッセージ (conn, 記録時刻, 本体) を返す。"""
    nxt = [c["pos"] for c in bi.chunks if c["pos"] > chunk["pos"]]
    end = (min(nxt) if nxt else bi.size) - 1
    b = http_range(bi.url, chunk["pos"], end)
    h, data = next(iter_records(b))
    assert h["op"][0] == OP_CHUNK
    if h["compression"] != b"none":
        raise NotImplementedError(f"圧縮 {h['compression']!r} には対応していない")
    out = []
    for mh, md in iter_records(data):
        if mh["op"][0] != OP_MSG:
            continue
        cid = struct.unpack("<I", mh["conn"])[0]
        if cid in conn_ids:
            out.append((cid, _time(mh["time"]), md))
    return out


def decode_header(md: bytes) -> tuple[float, int]:
    """std_msgs/Header を読み、(stamp, header の後の位置) を返す。"""
    _seq, s, ns, n = struct.unpack_from("<IIII", md, 0)
    return s + ns * 1e-9, 16 + n


def decode_imu(md: bytes) -> tuple[float, tuple, tuple]:
    stamp, p = decode_header(md)
    p += 4 * 8 + 9 * 8                       # orientation + cov
    w = struct.unpack_from("<3d", md, p)
    p += 3 * 8 + 9 * 8
    a = struct.unpack_from("<3d", md, p)
    return stamp, w, a


def decode_floats(md: bytes, n: int, has_header: bool) -> tuple[float | None, tuple]:
    if has_header:
        stamp, p = decode_header(md)
    else:
        stamp, p = None, 0
    return stamp, struct.unpack_from(f"<{n}f", md, p)
