"""USGS 3DEP の 1 秒角の DEM のタイル (GeoTIFF) から、NCST の経路の周りを切り出して .npy に保存する。

tifffile は依存関係に入れていないので、その場だけ使う (pyproject.toml は変えない):
    uv run --with tifffile --with imagecodecs python -I scripts/brake/ncst_dem_crop.py \
        --tif data/brake/ncst_dem/USGS_1_n35w118.tif --out data/brake/ncst_dem/crop.npz --bbox 34.02 -117.68 34.50 -117.22
出力 (npz): elev (float32、行 = 北から南、列 = 西から東)、lat (行ごと)、lon (列ごと)、元の値の情報。
地理の情報は GeoTIFF の ModelTiepoint・ModelPixelScale から読む (画素の中心か角かは GTRasterTypeGeoKey で決める)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("S", "W", "N", "E"))
    a = ap.parse_args()
    with tifffile.TiffFile(a.tif) as tf:
        pg = tf.pages[0]
        tie = pg.tags["ModelTiepointTag"].value
        scale = pg.tags["ModelPixelScaleTag"].value
        geo = tf.geotiff_metadata or {}
        nodata = pg.tags["GDAL_NODATA"].value if "GDAL_NODATA" in pg.tags else None
        Z = pg.asarray()
    i0, j0, _, lon0, lat0, _ = tie[:6]
    dx, dy = scale[0], scale[1]
    raster_type = str(geo.get("GTRasterTypeGeoKey", ""))
    off = 0.5 if "Area" in raster_type or raster_type in ("1", "") else 0.0   # PixelIsArea: 角の座標 → 中心へ半画素
    lon = lon0 + (np.arange(Z.shape[1]) - j0 + off) * dx
    lat = lat0 - (np.arange(Z.shape[0]) - i0 + off) * dy
    S, W, N, E = a.bbox
    r = (lat >= S) & (lat <= N)
    c = (lon >= W) & (lon <= E)
    sub = Z[np.ix_(r, c)].astype(np.float32)
    if nodata is not None:
        sub[sub == float(nodata)] = np.nan
    info = dict(shape_full=list(Z.shape), shape_crop=list(sub.shape), pixel_deg=[float(dx), float(dy)], raster_type=raster_type,
                nodata=None if nodata is None else str(nodata), elev_range_m=[float(np.nanmin(sub)), float(np.nanmax(sub))],
                nan_frac=float(np.isnan(sub).mean()), bbox=a.bbox, geokeys={k: str(v) for k, v in geo.items() if "GeoKey" in k})
    np.savez_compressed(a.out, elev=sub, lat=lat[r], lon=lon[c], info=json.dumps(info))
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
