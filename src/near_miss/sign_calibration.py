"""加速度センサ (accel_x / accel_y) の符号を車両ごとに判定し、補正する。

なぜ車両ごとか
--------------
同じ車種キー・同じ DBC でも、加速度センサの向きが個体によって逆になる。
commaCarSegments の実測 (scripts/calibrate_accel_sign.py):

    RAV4 TSS2        298 台中   0 台が逆
    Corolla TSS2     253 台中   2 台が逆 (VIN が JTH = Lexus。同じ車種キーに入っている)
    Highlander TSS2   33 台中  21 台が逆 (VIN 型式 *BRCH がすべて逆、*ZR*H / KDR*H は正)

ax と ay は必ず一緒に反転し、片方だけ逆の車両は無かった。ヨーレートと
ブレーキスイッチは全台正しい向きだった。「ハイブリッドなら逆」のような
単純な規則にはならない (Corolla のハイブリッド 20 台はすべて正)。したがって
車種単位では決められず、**車両 (dongle) ごとに実データから決める**。

どう決めるか
------------
加速度センサと**独立な系統**を基準にして相関を取る。推測では決めない。

    accel_x  と  輪速から出した 1 秒ごとの Δv        (前後)
    accel_y  と  v × ヨーレート                         (横)

輪速とヨーレートは別のセンサで、ヨーは舵角との対応でも向きを確かめてある。
相関が十分に強く負なら反転、正ならそのまま、弱ければ「不明」とする。
**不明の車両は加速度チャネルを使わない。** 誤った符号で横滑り角を出すより、
出さないほうがよい。

補正の適用
----------
車種設定の calibration.accel_sign に符号表 (CSV) を書いておくと、
読み出し直後に apply_accel_signs() がチャネルに符号を掛ける。
表に無い車両の扱いは calibration.on_missing で決める (既定 drop = 使わない)。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

AXES = ("accel_x", "accel_y")

STATUS_OK = "ok"
STATUS_FLIPPED = "flipped"
STATUS_UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# 判定に使う対を集める
# ---------------------------------------------------------------------------
def _nearest(t_src: np.ndarray, v_src: np.ndarray, t: np.ndarray) -> np.ndarray:
    """最近傍の値。補間はしない。"""
    if t_src.size == 0:
        return np.full(np.shape(t), np.nan)
    i = np.clip(np.searchsorted(t_src, t), 1, t_src.size - 1)
    j = np.where(np.abs(t - t_src[i - 1]) <= np.abs(t_src[i] - t), i - 1, i)
    return v_src[j]


def segment_pairs(channels: dict[str, Any], cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """1 セグメントから (基準, センサ) の対を集める。

    accel_x : 1 秒ごとの Δv [m/s] と、その 1 秒の accel_x 平均
    accel_y : v × ヨーレート [m/s^2] と、その時刻の accel_y (間引き)

    どちらも生の受信時刻のまま扱い、車速だけ最近傍で合わせる。
    """
    out = {"accel_x": np.empty((0, 2)), "accel_y": np.empty((0, 2))}
    spd = channels.get(cfg["speed_channel"])
    if spd is None or spd.t.size < 2:
        return out

    ax = channels.get("accel_x")
    if ax is not None and ax.t.size:
        bin_s = float(cfg["ax_bin_s"])
        edges = np.arange(spd.t[0], spd.t[-1], bin_s)
        if edges.size > 2:
            vb = _nearest(spd.t, spd.v, edges)
            bi = np.searchsorted(ax.t, edges)
            rows = []
            for k in range(edges.size - 1):
                seg = ax.v[bi[k]:bi[k + 1]]
                seg = seg[np.isfinite(seg)]
                if seg.size and vb[k] > float(cfg["ax_min_speed_mps"]):
                    rows.append((vb[k + 1] - vb[k], float(seg.mean())))
            if rows:
                out["accel_x"] = np.asarray(rows, dtype=float)

    ay, yaw = channels.get("accel_y"), channels.get("yaw_rate")
    if ay is not None and yaw is not None and ay.t.size and yaw.t.size:
        y = _nearest(yaw.t, yaw.v, ay.t)
        v = _nearest(spd.t, spd.v, ay.t)
        m = (v > float(cfg["ay_min_speed_mps"])) & np.isfinite(ay.v) & np.isfinite(y)
        idx = np.flatnonzero(m)[:: max(1, int(cfg["ay_subsample"]))]
        if idx.size:
            out["accel_y"] = np.c_[v[idx] * np.deg2rad(y[idx]), ay.v[idx]]
    return out


@dataclass
class VehicleSignStats:
    """1 台ぶんの対を貯める。セグメントをまたいで足していく。"""

    dongle_id: str
    segments: int = 0
    vins: set[str] = field(default_factory=set)
    pairs: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {a: [] for a in AXES})

    def add(self, pairs: dict[str, np.ndarray], vin: str = "") -> None:
        self.segments += 1
        if vin:
            self.vins.add(vin)
        for a in AXES:
            if pairs.get(a) is not None and pairs[a].size:
                self.pairs[a].append(pairs[a])

    def merge(self, other: "VehicleSignStats") -> None:
        self.segments += other.segments
        self.vins |= other.vins
        for a in AXES:
            self.pairs[a].extend(other.pairs[a])


def _corr(p: list[np.ndarray]) -> tuple[float, int]:
    if not p:
        return float("nan"), 0
    x = np.vstack(p)
    ok = np.isfinite(x).all(axis=1)
    x = x[ok]
    if x.shape[0] < 3 or np.std(x[:, 0]) == 0 or np.std(x[:, 1]) == 0:
        return float("nan"), int(x.shape[0])
    return float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]), int(x.shape[0])


def decide_from_corr(corr: dict[str, float], pairs: dict[str, int],
                     cfg: dict[str, Any]) -> dict[str, tuple[int, str]]:
    """軸ごとの (相関, 対の数) から符号を決める。**確信が持てなければ不明にする。**

    1 段目  |r| >= min_abs_corr かつ対が足りていれば、その軸単独で決める。
    2 段目  単独では決まらなかった軸でも、
              * もう一方の軸が 1 段目で決まっていて
              * この軸も |r| >= min_abs_corr_corroborated で**同じ符号**を示し
              * 対の数が足りている
            なら決める。

    2 段目を置く理由: Highlander TSS2 では前後が r = -0.7〜-0.8 ではっきり逆なのに、
    横が r = -0.43〜-0.49 (1 台あたり 4 万点) で 0.5 に届かず、31 台 / 10% が
    不明になっていた。弱い側にも独立に同じ向きの証拠を求めるので、
    「一緒に反転するはず」という経験則だけで埋めることにはならない。
    符号が食い違う場合は 2 段目でも決めない。
    """
    need = {"accel_x": int(cfg["min_pairs_ax"]), "accel_y": int(cfg["min_pairs_ay"])}
    strong = float(cfg["min_abs_corr"])
    weak = float(cfg.get("min_abs_corr_corroborated", strong))

    def ok(a: str, th: float) -> bool:
        r = corr[a]
        return pairs[a] >= need[a] and np.isfinite(r) and abs(r) >= th

    out: dict[str, tuple[int, str]] = {}
    first = {a: (int(np.sign(corr[a])) if ok(a, strong) else 0) for a in AXES}
    for a in AXES:
        sgn = first[a]
        if sgn == 0:
            other = [b for b in AXES if b != a][0]
            if first[other] != 0 and ok(a, weak) and int(np.sign(corr[a])) == first[other]:
                sgn = first[other]
        status = STATUS_UNKNOWN if sgn == 0 else (STATUS_FLIPPED if sgn < 0 else STATUS_OK)
        out[a] = (sgn, status)
    return out


def decide(stats: VehicleSignStats, cfg: dict[str, Any]) -> dict[str, Any]:
    """1 台ぶんの対から符号を決める。判定の規則は decide_from_corr()。"""
    row: dict[str, Any] = {"dongle_id": stats.dongle_id, "segments": stats.segments,
                           "vin": "|".join(sorted(stats.vins))}
    corr, pairs = {}, {}
    for a in AXES:
        r, n = _corr(stats.pairs[a])
        corr[a], pairs[a] = r, n
        row[f"{a}_corr"] = round(r, 4) if np.isfinite(r) else float("nan")
        row[f"{a}_pairs"] = n
    for a, (sgn, status) in decide_from_corr(corr, pairs, cfg).items():
        row[f"{a}_sign"], row[f"{a}_status"] = sgn, status
    return row


# ---------------------------------------------------------------------------
# 補正の適用
# ---------------------------------------------------------------------------
def table_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=16)
def load_sign_table(path: str) -> dict[str, dict[str, int]]:
    """符号表を読む。dongle -> {"accel_x": ±1 or 0, "accel_y": ±1 or 0}。"""
    import pandas as pd

    df = pd.read_csv(path, dtype={"dongle_id": str})
    out: dict[str, dict[str, int]] = {}
    for r in df.itertuples(index=False):
        out[r.dongle_id] = {a: int(getattr(r, f"{a}_sign")) for a in AXES}
    return out


def apply_accel_signs(seg: Any, vehicle: Any, repo_root: Path) -> None:
    """読み出した直後のセグメントに、車両ごとの符号を掛ける。

    表で 0 (不明) の軸、および表に無い車両 (on_missing: drop のとき) は
    チャネルを取り除く。**NaN にせず取り除く**のは、後段が「列が無い」ことで
    その特徴量を作らないようにするため (features.py は列の有無で分岐する)。
    何をしたかは seg.meta["accel_sign"] と seg.notes に残す。
    """
    cal = (vehicle.raw.get("calibration") or {}) if vehicle is not None else {}
    path = cal.get("accel_sign")
    if not path:
        return
    p = Path(path)
    if not p.is_absolute():
        p = repo_root / p
    if not p.is_file():
        raise FileNotFoundError(
            f"加速度の符号表がありません: {p}\n"
            f"  先に scripts/calibrate_accel_sign.py --platform <車種> を実行してください")

    dongle = seg.ref.dongle_id
    table = load_sign_table(str(p))
    signs = table.get(dongle)
    info: dict[str, Any] = {"table": str(path), "dongle_id": dongle}
    if signs is None:
        policy = str(cal.get("on_missing", "drop"))
        info["status"] = f"missing:{policy}"
        if policy == "drop":
            for a in AXES:
                seg.channels.pop(a, None)
            seg.notes.append("accel_sign:missing_vehicle:dropped")
        else:
            seg.notes.append("accel_sign:missing_vehicle:kept_uncorrected")
        seg.meta["accel_sign"] = info
        return

    for a in AXES:
        s = signs[a]
        info[a] = s
        ch = seg.channels.get(a)
        if ch is None:
            continue
        if s == 0:
            seg.channels.pop(a)
            seg.notes.append(f"accel_sign:{a}:unknown:dropped")
        elif s == -1:
            seg.channels[a] = replace(ch, v=-ch.v)
            seg.notes.append(f"accel_sign:{a}:flipped")
    info["status"] = "applied"
    seg.meta["accel_sign"] = info
