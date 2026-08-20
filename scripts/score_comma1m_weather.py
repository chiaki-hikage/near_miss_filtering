#!/usr/bin/env python3
"""thumbnail から天候・路面の見た目を数値化する。

雪かどうかを画像だけで断定はしない。ここで出すのは「雪・悪天候の疑いが強い順」に
並べるための指標で、最終確認は目視で行う。

指標 (すべて 0..1 に正規化した HSV から算出):
  sat_mean        全体の彩度平均。雪・曇天は低い、晴天は高い
  colorfulness    Hasler-Susstrunk の色鮮やかさ
  sky_blue        空領域 (上 30%) の青み B-R。晴天で大きい
  road_val        路面領域 (下 35% の中央 50%) の明度平均
  road_sat        同じく彩度平均
  road_white      路面領域で「白い」画素の割合 (S<0.15 かつ V>0.55)
  side_white      路肩領域 (縦 40-85%、左右それぞれ外側 22%) の白画素割合
  val_mean        全体の明度。夜間の判別に使う
  contrast        明度の標準偏差。霧・雪・逆光で下がる

snow_score  = side_white x (色鮮やかさの低さ)
  路面だけを見ると明るいコンクリート舗装を雪と取り違える (実測: road_white 上位 24 件は
  すべて乾いた舗装だった)。積雪は路肩・法面まで白くなり、画面全体の色味が失われるので、
  路肩の白さと色鮮やかさの低さを掛ける。
gloom_score = 曇天・降雨・薄暮の疑い (彩度と青みとコントラストの低さ)
  実測では彩度が低く明るい画像の上位が、雨天の飛沫・濡れた路面・曇天でよく揃った。

使い方:
  python scripts/score_comma1m_weather.py --out out/comma1m/weather.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from PIL import Image

SKY_ROWS = 0.30
ROAD_ROWS = 0.35
ROAD_COLS = 0.50
WHITE_S = 0.15
WHITE_V = 0.55
# 路肩帯。積雪は路面より先に路肩・法面へ残る
SIDE_ROWS = (0.40, 0.85)
SIDE_COLS = 0.22
SIDE_WHITE_S = 0.18
SIDE_WHITE_V = 0.60


def image_features(path: Path) -> dict:
    im = Image.open(path).convert("RGB")
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    h, w, _ = rgb.shape
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    sat, val = hsv[:, :, 1], hsv[:, :, 2]

    sky = slice(0, int(h * SKY_ROWS))
    road_r = slice(int(h * (1 - ROAD_ROWS)), h)
    c0 = int(w * (1 - ROAD_COLS) / 2)
    road_c = slice(c0, w - c0)

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))

    road_s = sat[road_r, road_c]
    road_v = val[road_r, road_c]

    r0, r1 = int(h * SIDE_ROWS[0]), int(h * SIDE_ROWS[1])
    cw = int(w * SIDE_COLS)
    side_white = (sat[r0:r1] < SIDE_WHITE_S) & (val[r0:r1] > SIDE_WHITE_V)
    side_white = np.concatenate([side_white[:, :cw].ravel(), side_white[:, w - cw:].ravel()])

    return {
        "sat_mean": float(sat.mean()),
        "val_mean": float(val.mean()),
        "contrast": float(val.std()),
        "colorfulness": colorfulness,
        "sky_blue": float((b[sky, :] - r[sky, :]).mean()),
        "road_val": float(road_v.mean()),
        "road_sat": float(road_s.mean()),
        "road_white": float(((road_s < WHITE_S) & (road_v > WHITE_V)).mean()),
        "side_white": float(side_white.mean()),
    }


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """順位づけ用の合成指標。閾値ではなく順位で使う前提の量。"""
    df = df.copy()
    # 雪: 路肩が白く、かつ画面の色味が失われている
    df["snow_score"] = df["side_white"] * np.clip(1.0 - df["colorfulness"] / 0.12, 0, 1)
    # 曇天・降雨: 彩度も青みもコントラストも低い。ただし夜間は別扱い
    df["gloom_score"] = (
        np.clip(1.0 - df["colorfulness"] / 0.20, 0, 1)
        + np.clip(1.0 - df["sky_blue"] / 0.10, 0, 1)
        + np.clip(1.0 - df["contrast"] / 0.20, 0, 1)
    ) / 3.0
    # 雨天・濡れ路面・曇天: 明るいのに彩度が無い状態。
    # 夜間 (単に暗い) と区別するため、明度が一定以上あることを条件に入れる。
    df["wet_score"] = (
        np.clip(1.0 - df["sat_mean"] / 0.15, 0, 1)
        * np.clip((df["val_mean"] - 0.28) / 0.15, 0, 1)
    )
    # 好天の対照群を取るための指標
    df["clear_score"] = (
        np.clip(df["colorfulness"] / 0.15, 0, 1)
        * np.clip(df["sky_blue"] / 0.25, 0, 1)
        * np.clip(df["contrast"] / 0.20, 0, 1)
    )
    df["is_dark"] = df["val_mean"] < 0.25
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--thumbs", type=Path, default=Path("raw_data/comma1M/thumbnails"))
    p.add_argument("--regions", type=Path, default=Path("out/comma1m/regions.csv"))
    p.add_argument("--out", type=Path, default=Path("out/comma1m/weather.csv"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.thumbs.glob("*.jpg"))
    print(f"thumbnail {len(paths)} 件")
    rows = []
    for i, p in enumerate(paths, 1):
        f = image_features(p)
        f["segment_id"] = p.stem.replace(".thumbnail", "")
        rows.append(f)
        if i % 1000 == 0:
            print(f"  {i}/{len(paths)}")
    df = add_scores(pd.DataFrame(rows))

    if args.regions.exists():
        reg = pd.read_csv(args.regions)
        keep = ["segment_id", "country", "admin1", "admin2", "place", "cold_tier",
                "cold_reason", "alt_max_m", "lat_mid", "lon_mid", "duration_s"]
        df = df.merge(reg[[c for c in keep if c in reg.columns]], on="segment_id", how="left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    cols = ["sat_mean", "val_mean", "contrast", "colorfulness", "sky_blue",
            "road_val", "road_sat", "road_white", "side_white", "snow_score", "gloom_score", "wet_score", "clear_score"]
    print("\n--- 分布 ---")
    print(df[cols].describe(percentiles=[0.05, 0.5, 0.95, 0.99]).round(3).to_string())
    print(f"\n夜間らしい (val_mean<0.25): {int(df['is_dark'].sum())} 件")
    print(f"\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
