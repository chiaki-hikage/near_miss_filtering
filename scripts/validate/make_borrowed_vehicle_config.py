#!/usr/bin/env python3
"""検証済みの車種設定から、同じ系列の別車種の設定を「未検証」として書き出す。

opendbc で同じ DBC を使う車種は、アドレスもビット位置も同じである可能性が高い。
だが同じとは限らないし、opendbc に無い信号 (Toyota の 0x226 / 0x3B7) は
なおさら確かめる必要がある。そこで

  * signals 節は元の設定をそのまま写す
  * **validated はすべて false に落とす**
  * geometry は rlog の carParams から取れる値だけを入れ、当てはめが要る量は null
    (元の車種の当てはめ値を流用しない。steer_ratio は carParams と 30% 違った)
  * レーダの DBC が違う車種は radar 節を外す (--no-radar)

を機械的に行う。書き出した後は scripts/validate/validate_slip_flags.py で確かめること。

  uv run python scripts/validate/make_borrowed_vehicle_config.py \\
      --source configs/vehicles/toyota_rav4_tss2.yaml \\
      --platform TOYOTA_HIGHLANDER_TSS2 --name toyota_highlander_tss2
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from near_miss.io import comma_car_segments as ccs
from near_miss.io import rlog as R


def car_params(platform: str, n: int) -> tuple[dict, Counter]:
    """キャッシュ済みの先頭 n セグメントから carParams と VIN 製造者コードを集める。"""
    sch = R.load_schema()
    params: Counter = Counter()
    wmi: Counter = Counter()
    for ref in ccs.find_segments(ccs.DEFAULT_CACHE, platform)[:n]:
        for ev in sch.Event.read_multiple_bytes(R._decompress(ref.path)):
            if ev.which() != "carParams":
                continue
            cp = ev.carParams
            params[(round(cp.wheelbase, 4), round(cp.centerToFront, 4),
                    round(cp.steerRatio, 3), round(cp.mass, 1))] += 1
            vin = str(cp.carVin)
            wmi[vin[:3] if len(vin) >= 3 else "?"] += 1
            break
    if not params:
        raise SystemExit(f"{platform} のセグメントがキャッシュにありません。先に取得してください")
    (wb, c2f, sr, mass), _ = params.most_common(1)[0]
    if len(params) > 1:
        print(f"注意: carParams が {len(params)} 通りあります。最多のものを使います: {params}")
    return {"wheelbase": wb, "centerToFront": c2f, "steerRatio": sr, "mass": mass}, wmi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="写し元の車種設定")
    p.add_argument("--platform", required=True, help="commaCarSegments の車種キー")
    p.add_argument("--name", required=True, help="新しい設定の name")
    p.add_argument("--out", type=Path, default=None,
                   help="書き出し先 (既定 configs/vehicles/<name>.yaml)")
    p.add_argument("--no-radar", action="store_true",
                   help="radar 節を外す。レーダの DBC が写し元と違う車種で使う")
    p.add_argument("--sample", type=int, default=40, help="carParams を読むセグメント数")
    p.add_argument("--force", action="store_true", help="既存のファイルを上書きする")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out or Path("configs/vehicles") / f"{args.name}.yaml"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} は既にあります (--force で上書き)")

    cp, wmi = car_params(args.platform, args.sample)
    src = args.source.read_text(encoding="utf-8")
    src_name = re.search(r"^name:\s*(\S+)", src, re.M).group(1)

    body = src[src.index("\nsignals:\n") + 1:]
    n_true = body.count("validated: true")
    body = body.replace("validated: true", "validated: false")
    if args.no_radar and "\nradar:\n" in body:
        # radar 節は次のトップレベルキーまで。
        m = re.search(r"\nradar:\n(?:[ ].*\n|\n)*", body)
        body = body[: m.start() + 1] + "# radar 節は外した: レーダの DBC が写し元と違う\n" + body[m.end():]

    wmi_txt = " / ".join(f"{k} x{v}" for k, v in wmi.most_common(8))
    head = f"""# {args.platform} (commaCarSegments)
#
# **信号定義は {args.source.name} の写しで、この車種では未検証。**
# scripts/validate/make_borrowed_vehicle_config.py で機械的に作った。
#
# アドレスが同じでもビットの意味が同じとは限らない。特に 0x226 (ABSACT / VSCACT) と
# 0x3B7 (SLIP_WARN) は opendbc に無く、{src_name} の実データから解析したもの。
# validated はすべて false に落としてある (元は true が {n_true} 件)。
# scripts/validate/validate_slip_flags.py の結果で 1 つずつ戻すこと。
#
# 収録車両の VIN 製造者コード (先頭 {args.sample} セグメント): {wmi_txt}

name: {args.name}
description: "{args.platform} (commaCarSegments) — 信号定義は {src_name} から借用・未検証"
dongle_ids: []
platforms:
  - {args.platform}

# carParams の値だけを入れる。当てはめが要る量は null にして、
# 物理モデルに依存する判定 (beta / ヨー残差) を無効のままにする。
geometry:
  wheelbase_m: {cp['wheelbase']}           # carParams.wheelbase
  steer_ratio: null           # 未当てはめ (carParams.steerRatio = {cp['steerRatio']} は制御用の調律値)
  understeer_gradient: null
  steer_offset_deg: null
  track_width_m: null
  yaw_residual_std_dps: null
  center_to_front_m: {cp['centerToFront']}   # carParams.centerToFront
  yaw_rate_noise_dps: null
  accel_y_noise_mps2: null


"""
    out.write_text(head + body, encoding="utf-8")
    print(f"書き出し: {out}")
    print(f"  carParams: {cp}")
    print(f"  VIN 製造者コード: {dict(wmi.most_common())}")
    print(f"  validated true -> false: {n_true} 件" + ("  / radar 節を外した" if args.no_radar else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
