"""HTTP の範囲指定で zip の目次と一部のファイルだけを読む。"""
import io, sys, time, urllib.request, zipfile

class HttpFile(io.RawIOBase):
    def __init__(self, url):
        for k in range(6):
            try:
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=60) as r:
                    self.url = r.geturl(); self.size = int(r.headers["Content-Length"])
                break
            except Exception:
                if k == 5: raise
                time.sleep(5 * (k + 1))
        self.pos = 0; self.nbytes = 0; self.nreq = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for k in range(8):
            try:
                with urllib.request.urlopen(req, timeout=120) as r: data = r.read()
                break
            except Exception:
                if k == 7: raise
                time.sleep(3 * (k + 1))
        self.pos += len(data); self.nbytes += len(data); self.nreq += 1
        return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

def open_chunk(n):
    f = HttpFile(f"https://huggingface.co/datasets/commaai/comma2k19/resolve/main/raw_data/Chunk_{n}.zip")
    return f, zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16))
