"""ISO 2631-1:1997 の周波数重み付けと、衝撃を含む振動の評価量。

重み付けは附属書 A の伝達関数 (帯域制限 Hh・Hl、加速度-速度遷移 Ht、上方ステップ Hs) を
周波数領域でそのまま掛ける。双一次変換の周波数のゆがみが無いので、表の値をそのまま再現できる。
帯域制限の低域通過 (f2 = 100 Hz) は、IMU のナイキスト周波数 (50 Hz) より上なので省く。

    Wk  上下 (座面 z)            Wd  前後・左右 (座面 x, y)       We  回転 (ロール・ピッチ・ヨー)

評価量 (附属書 6 章)
    aw     重み付き加速度の RMS
    MTVV   1 秒の移動 RMS (線形積分) の最大値 (maximum transient vibration value)
    VDV    (∫ aw⁴ dt)^(1/4)  [m/s^1.75] (vibration dose value)
    crest  ピーク / RMS (9 を超えると RMS だけでは衝撃を過小評価する)

快適性の目安は附属書 C.2.3 (公共交通の乗客。範囲が重なっているのは原文どおり)。
"""

from __future__ import annotations

import numpy as np

# 附属書 A 表 A.2 (f は Hz、Q は無次元。None はその段を使わない)
WEIGHTINGS = {
    "Wk": dict(f1=0.4, f3=12.5, f4=12.5, Q4=0.63, f5=2.37, Q5=0.91, f6=3.35, Q6=0.91),
    "Wd": dict(f1=0.4, f3=2.0, f4=2.0, Q4=0.63, f5=None, Q5=None, f6=None, Q6=None),
    "We": dict(f1=0.4, f3=1.0, f4=1.0, Q4=0.63, f5=None, Q5=None, f6=None, Q6=None),
}

# 座面での快適性の評価の軸の係数 (8.2.3)。回転は [m/rad]
K_SEAT = dict(x=1.0, y=1.0, z=1.0, rx=0.63, ry=0.40, rz=0.20)

# 附属書 C.2.3 の目安 (下限, 上限, 表現)
COMFORT = (
    (0.0, 0.315, "不快でない"),
    (0.315, 0.63, "少し不快"),
    (0.5, 1.0, "かなり不快"),
    (0.8, 1.6, "不快"),
    (1.25, 2.5, "非常に不快"),
    (2.0, np.inf, "極めて不快"),
)


def response(f: np.ndarray, name: str) -> np.ndarray:
    """重み付け name の周波数応答 H(j2πf) (複素数)。f = 0 は 0 にする (高域通過)。"""
    w = WEIGHTINGS[name]
    p = 2j * np.pi * np.asarray(f, float)
    w1, w3, w4 = (2 * np.pi * w[k] for k in ("f1", "f3", "f4"))
    with np.errstate(divide="ignore", invalid="ignore"):
        hh = p ** 2 / (p ** 2 + np.sqrt(2) * w1 * p + w1 ** 2)
        ht = (1 + p / w3) / (1 + p / (w["Q4"] * w4) + (p / w4) ** 2)
        h = hh * ht
        if w["f5"] is not None:
            w5, w6 = 2 * np.pi * w["f5"], 2 * np.pi * w["f6"]
            hs = (1 + p / (w["Q5"] * w5) + (p / w5) ** 2) / (1 + p / (w["Q6"] * w6) + (p / w6) ** 2) * (w5 / w6) ** 2
            h = h * hs
    return np.where(np.asarray(f) == 0, 0, h)


def weight(x: np.ndarray, fs: float, name: str, pad_s: float = 5.0) -> np.ndarray:
    """時系列 x に重み付けを掛ける。両端にゼロを足して FFT の循環の影響を避ける。"""
    x = np.asarray(x, float) - np.mean(x)
    pad = int(pad_s * fs)
    n = len(x) + 2 * pad
    nfft = 1 << int(np.ceil(np.log2(n)))
    X = np.fft.rfft(np.r_[np.zeros(pad), x, np.zeros(nfft - len(x) - pad)])
    y = np.fft.irfft(X * response(np.fft.rfftfreq(nfft, 1 / fs), name), nfft)
    return y[pad:pad + len(x)]


def running_rms(aw: np.ndarray, fs: float, tau_s: float = 1.0) -> np.ndarray:
    """1 秒の移動 RMS (線形積分、過去 tau 秒)。先頭 tau 秒は窓が足りないので NaN。"""
    n = int(round(tau_s * fs))
    c = np.cumsum(np.r_[0.0, aw ** 2])
    out = np.full(len(aw), np.nan)
    out[n - 1:] = np.sqrt((c[n:] - c[:-n]) / n)
    return out


def vdv(aw: np.ndarray, fs: float) -> float:
    return float((np.sum(aw ** 4) / fs) ** 0.25)


def comfort_label(a: float) -> str:
    """附属書 C.2.3 の目安。範囲が重なるところは、上側の表現を採る (安全側)。"""
    label = COMFORT[0][2]
    for lo, hi, name in COMFORT:
        if a >= lo:
            label = name
    return label
