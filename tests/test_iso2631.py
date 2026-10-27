import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "vibration"))

import iso2631 as ISO  # noqa: E402

# ISO 2631-1:1997 表 3 の値 (×1000)。31.5 Hz 以上は帯域制限の低域通過 (100 Hz) を省いた分だけずれる
WK = {0.5: 418, 1.0: 482, 2.0: 531, 4.0: 967, 5.0: 1039, 6.3: 1054, 8.0: 1036, 10.0: 988, 12.5: 902, 16.0: 768, 20.0: 636}
WD = {0.5: 853, 1.0: 1011, 2.0: 890, 4.0: 512, 8.0: 253, 16.0: 125}


@pytest.mark.parametrize("name,table", [("Wk", WK), ("Wd", WD)])
def test_weighting_matches_iso_table(name, table):
    f = np.array(list(table))
    got = np.abs(ISO.response(f, name)) * 1000
    assert np.allclose(got, list(table.values()), rtol=0.01)


def test_we_is_wd_shifted_by_one_octave():
    # We と Wd は遷移の周波数が 1 Hz と 2 Hz の違いだけ (高域通過は共通で、2 Hz 以上ではほぼ効かない)
    f = np.array([2.0, 4.0, 8.0])
    assert np.allclose(np.abs(ISO.response(f, "We")), np.abs(ISO.response(2 * f, "Wd")), rtol=0.01)


def test_weight_of_sine_scales_by_table_value():
    fs = 100.0
    t = np.arange(0, 20, 1 / fs)
    x = np.sin(2 * np.pi * 5.0 * t)
    y = ISO.weight(x, fs, "Wk")
    mid = slice(int(5 * fs), int(15 * fs))
    assert np.sqrt(np.mean(y[mid] ** 2)) == pytest.approx(1.039 / np.sqrt(2), rel=0.01)


def test_running_rms_and_vdv_of_sine():
    fs = 100.0
    t = np.arange(0, 10, 1 / fs)
    a = 2.0 * np.sin(2 * np.pi * 4.0 * t)
    rr = ISO.running_rms(a, fs)
    assert np.isnan(rr[0]) and np.nanmax(rr) == pytest.approx(2 / np.sqrt(2), rel=0.01)
    # 正弦波の 4 乗の平均は 3/8 A⁴
    assert ISO.vdv(a, fs) == pytest.approx((3 / 8 * 2 ** 4 * 10) ** 0.25, rel=0.01)


@pytest.mark.parametrize("a,label", [(0.2, "不快でない"), (0.4, "少し不快"), (0.7, "かなり不快"), (1.0, "不快"),
                                     (1.5, "非常に不快"), (2.2, "極めて不快")])
def test_comfort_label_takes_upper_category_on_overlap(a, label):
    assert ISO.comfort_label(a) == label
