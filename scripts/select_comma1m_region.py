#!/usr/bin/env python3
"""comma1M の位置一覧から地域を逆引きし、寒冷地・降雪可能地域を選ぶ (段階 B)。

scripts/fetch_comma1m_positions.py が作った positions.csv を入力にする。
逆引きはオフライン (reverse_geocoder) なので通信しない。

判定条件は configs/datasets/comma1m.yaml の cold_region に置いてある。
行政区分・標高・緯度のどれか 1 つでも当たれば候補にする (再現率優先)。

注意: localizer に録画日時は入っていない。ここで分かるのは「積雪しうる地域か」
であって「積雪期に走ったか」ではない。実際の路面状態は thumbnail で確認する。

使い方:
  python scripts/select_comma1m_region.py --positions out/comma1m/positions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from near_miss.config import load_yaml

DEFAULT_CFG = Path("configs/datasets/comma1m.yaml")


def reverse_geocode(df: pd.DataFrame, lat_col: str = "lat_mid", lon_col: str = "lon_mid") -> pd.DataFrame:
    import reverse_geocoder as rg

    hits = rg.search([(float(a), float(b)) for a, b in zip(df[lat_col], df[lon_col])])
    out = df.copy()
    out["country"] = [h["cc"] for h in hits]
    out["admin1"] = [h["admin1"] for h in hits]
    out["admin2"] = [h["admin2"] for h in hits]
    out["place"] = [h["name"] for h in hits]
    return out


def classify_cold(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """寒冷度を heavy / moderate / low の 3 段階に分ける。判定根拠も残す。"""
    c = cfg["cold_region"]
    heavy_a1 = set(c["heavy_snow_admin1"])
    mod_a1 = set(c["moderate_snow_admin1"])
    heavy_cc = set(c["heavy_snow_country"])
    mod_cc = set(c["moderate_snow_country"])
    alt = c["altitude_m"]
    latc = c["latitude_deg"]

    alt_m = df[["alt_start", "alt_mid", "alt_end"]].max(axis=1)
    abslat = df["lat_mid"].abs()

    tiers, reasons = [], []
    for a1, cc, a, la in zip(df["admin1"], df["country"], alt_m, abslat):
        hit = []
        if cc in heavy_cc:
            hit.append(("heavy", f"country={cc}"))
        if a1 in heavy_a1:
            hit.append(("heavy", f"admin1={a1}"))
        if a >= alt["heavy"]:
            hit.append(("heavy", f"alt={a:.0f}m"))
        if la >= latc["heavy"]:
            hit.append(("heavy", f"lat={la:.1f}"))
        if cc in mod_cc:
            hit.append(("moderate", f"country={cc}"))
        if a1 in mod_a1:
            hit.append(("moderate", f"admin1={a1}"))
        if a >= alt["moderate"]:
            hit.append(("moderate", f"alt={a:.0f}m"))
        if la >= latc["moderate"]:
            hit.append(("moderate", f"lat={la:.1f}"))

        if any(t == "heavy" for t, _ in hit):
            tiers.append("heavy")
            reasons.append("+".join(r for t, r in hit if t == "heavy"))
        elif hit:
            tiers.append("moderate")
            reasons.append("+".join(r for _, r in hit))
        else:
            tiers.append("low")
            reasons.append("")
    out = df.copy()
    out["alt_max_m"] = alt_m
    out["cold_tier"] = tiers
    out["cold_reason"] = reasons
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--positions", type=Path, default=Path("out/comma1m/positions.csv"))
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--out", type=Path, default=Path("out/comma1m/regions.csv"))
    p.add_argument("--top", type=int, default=25, help="標準出力に出す地域の件数")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    df = pd.read_csv(args.positions)
    ok = df[df["error"].fillna("") == ""].copy()
    print(f"位置を取得できたセグメント: {len(ok)} / {len(df)}")

    ok = classify_cold(reverse_geocode(ok), cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok.to_csv(args.out, index=False)

    print("\n--- 国別 ---")
    print(ok["country"].value_counts().to_string())

    print(f"\n--- 一次行政区分 上位 {args.top} ---")
    a1 = ok.groupby(["country", "admin1"]).size().sort_values(ascending=False)
    print(a1.head(args.top).to_string())

    print("\n--- 寒冷度 ---")
    print(ok["cold_tier"].value_counts().to_string())
    print(f"\n走行時間 [h]:")
    print((ok.groupby("cold_tier")["duration_s"].sum() / 3600).round(1).to_string())
    print(f"\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
