#!/usr/bin/env python3
"""commaCarSegments の車種ごとに、スリップ検出に使える信号があるかを調べる。

**ダウンロードしなくても分かる部分**と、**1 セグメント落とさないと分からない部分**を
分けて出す。

  段 1  opendbc の DBC 定義から「復号できるはずの信号」を引く (取得不要)
  段 2  実際に 1 セグメント落として、その信号を載せたメッセージが
        **バスに流れているか**を確かめる (--verify)

段 2 が要るのは、DBC に定義があってもその個体が送っていないことがあるため。
comma2k19 の RAV4 2017 では ABS 作動を載せた 0x226 が DBC にはあったが
車両が送っておらず、FABS (故障フラグ) しか取れなかった。

調べる区分:
  直接作動   ABS / VSC-ESC / TRC-TCS / 滑り警告。**作動と故障を分ける**
  4 輪速     4 隅すべて揃うか (ロック・空転の判定に要る)
  横運動     舵角 / ヨーレート / 横加速度 (横滑りの判定に要る)

  # 車種 -> DBC の対応を書き出す (opendbc は Python 3.11 以上を要求するので別に走らせる)
  cd <opendbc> && uv run --python 3.12 --with pycapnp --with numpy python -c \
    "import sys,json; sys.path.insert(0,'.'); from opendbc.car.values import PLATFORMS; \
     json.dump({n:{str(getattr(k,'value',k)):v for k,v in (p.config.dbc_dict or {}).items()} \
                for n,p in PLATFORMS.items()}, open('platform_dbc.json','w'))"

  # Toyota 系と 50 時間以上 (既定)
  uv run python scripts/survey_platforms.py --opendbc <opendbc のパス> \
      --mapping <opendbc>/platform_dbc.json

  # 実機バスの確認つき (1 車種あたり数セグメント落とす。1 本 1.4 MB)
  uv run python scripts/survey_platforms.py --opendbc <path> \
      --mapping <path>/platform_dbc.json --verify --verify-n 3

出力 (--out 配下)
  platforms.csv   1 行 1 車種。区分ごとの可否と、見つかった信号名
  signals.csv     1 行 1 信号。どのメッセージのどのバスにあるか
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

README_URL = ("https://huggingface.co/datasets/commaai/commaCarSegments"
              "/resolve/main/README.md")

# --- DBC の字句解析 --------------------------------------------------------
# BO_ <id> <名前>: <長さ> <送信元>
_BO = re.compile(r"^BO_\s+(\d+)\s+([A-Za-z0-9_]+)\s*:\s*(\d+)")
# SG_ <名前> : <ビット定義>
_SG = re.compile(r"^\s+SG_\s+([A-Za-z0-9_]+)\s*:")

# --- 区分の判定 ---------------------------------------------------------------
# 信号名は車種ごとにばらばらなので、語幹で候補を拾ってから性質で分ける。
#
# **性質の区別が肝心。** 同じ "ABS" でも
#   ABSACT          作動した        → 使える
#   FABS            故障している    → 使えない (comma2k19 の RAV4 で踏んだ)
#   VSC_DISABLED    切られている    → 使えない
# を混ぜると、故障灯しか無い車種が「作動信号あり」に見える。

# 故障・妥当性・検査用。作動ではない。
_FAULT = re.compile(
    r"fault|malfunc|error|fail|invalid|checksum|crc|counter|alive|"
    r"sigsta|_q$|_stat_|avail|^f[a-z]*(abs|vsc|vdc)|sum$|sm$", re.I)
# 設定・要求。運転者や他 ECU が設定した状態で、介入ではない。
_SETTING = re.compile(
    r"disab|_off$|off_?road|button|request|_req($|_)|_cmd($|_)|mode|"
    r"_max$|_min$|menu|limit", re.I)
# 計測値。ECU の名前が付いているだけで、値そのもの。
_VALUE = re.compile(r"val$|value$|_p?res|torque|speed|spd|accel|angle|rate|press", re.I)
# 「作動した」と読める語。ここに当たったものだけを作動として数える。
_ACTIVE = re.compile(r"activ|(^|_)act($|_|[0-9])|engag|interven|operat|"
                     r"warn|(^|_)wl($|_)|indicator|lamp", re.I)

SYSTEMS = {
    # 区分 -> 信号名にこれが含まれること
    "abs": re.compile(r"(^|_)abs($|_|[a-z])|absact", re.I),
    "esc": re.compile(r"vsc|(^|_)esp($|_|i)|(^|_)esc($|_)|(^|_)vdc($|_)|stabilit", re.I),
    "trc": re.compile(r"(^|_)tcs($|_)|(^|_)trc($|_)|traction", re.I),
    "slip": re.compile(r"slip|(^|_)slp($|_)", re.I),
}

# 4 輪速。**メッセージ名も見る。** Subaru は BO_ Wheel_Speeds の中に
# SG_ FL / FR / RL / RR とだけ書いてあり、信号名だけでは車輪と分からない。
_WHEELISH = re.compile(r"wheel|whl", re.I)
_SPEEDISH = re.compile(r"spd|speed", re.I)
CORNERS = (
    ("FL", re.compile(r"(^|_)fl($|_|[a-z])|front_?left|left_?front|(^|_)lf($|_)", re.I)),
    ("FR", re.compile(r"(^|_)fr($|_|[a-z])|front_?right|right_?front|(^|_)rf($|_)", re.I)),
    ("RL", re.compile(r"(^|_)rl($|_|[a-z])|rear_?left|left_?rear|(^|_)lr($|_)", re.I)),
    ("RR", re.compile(r"(^|_)rr($|_|[a-z])|rear_?right|right_?rear", re.I)),
)

LATERAL = {
    "steer_angle": re.compile(r"(steer\w*angle|angle\w*steer|steering_?angle|sas\w*angle"
                              r"|internalsas|steeringwheelangle)", re.I),
    "yaw_rate": re.compile(r"yaw", re.I),
    "lat_accel": re.compile(r"lat\w*acc|acc\w*lat|accel_?y|(^|_)ay($|_)", re.I),
}

# データセットの車種名と opendbc の車種名がずれているもの。
ALIASES = {"RIVIAN_R1_GEN1": "RIVIAN_R1"}

# **opendbc に無いが、こちらで解析して使えることが分かっている信号。**
# Toyota TSS2 の ABS/VSC 作動と滑り警告灯は opendbc の DBC に入っていない
# (docs/comma_car_segments.md)。DBC だけで比べると Toyota 系が不当に低く出る。
#
# **車種名の接頭辞で当てるが、それだけでは採用しない。** ここで足すのは
# 「そのアドレスを見ろ」という指示だけで、実際にバスへ流れているかは
# --verify の実機確認が決める。定義があって送っていない例を既に踏んでいる
# (comma2k19 の RAV4 2017 は 0x226 を送っていなかった)。
KNOWN_EXTRA = {
    ("TOYOTA", "LEXUS"): {
        "abs": [("ABSACT", 0x226)],
        "esc": [("VSCACT", 0x226)],
        "slip": [("SLIP_WARN / SLP_WL", 0x3B7)],
    },
}


def known_extra(platform: str) -> dict[str, list[tuple[str, int]]]:
    for prefixes, table in KNOWN_EXTRA.items():
        if platform.startswith(prefixes):
            return table
    return {}


def parse_dbc(path: Path) -> list[dict]:
    """DBC から (メッセージ, 信号) の一覧を取る。capnp も cantools も要らない。"""
    out: list[dict] = []
    msg_id, msg_name = None, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _BO.match(line)
        if m:
            msg_id, msg_name = int(m.group(1)), m.group(2)
            continue
        m = _SG.match(line)
        if m and msg_id is not None:
            out.append({"msg_id": msg_id, "msg_name": msg_name, "signal": m.group(1)})
    return out


def classify(sig: str) -> tuple[str | None, str]:
    """信号名を区分と性質に分ける。性質は 作動 / 故障 / 設定 / 値 のいずれか。

    **作動と言い切れるものだけを作動にする。** 迷ったら作動にしない。
    ここが緩むと「故障灯しか無い車種」が有望に見えてしまう。
    """
    if _FAULT.search(sig):
        kind = "故障"
    elif _SETTING.search(sig):
        kind = "設定"
    elif _ACTIVE.search(sig):
        kind = "作動"
    elif _VALUE.search(sig):
        kind = "値"
    else:
        kind = "不明"
    for name, pat in SYSTEMS.items():
        if pat.search(sig):
            return name, kind
    return None, kind


def wheel_corners(rows: list[dict]) -> dict[str, str]:
    """4 隅の輪速が揃っているかを見る。揃った隅 -> 信号名。

    「車輪の速度」らしさは **メッセージ名と信号名を合わせて**判定する。
    Subaru は BO_ Wheel_Speeds の中が SG_ FL / FR / RL / RR なので、
    信号名だけでは分からない。
    """
    found: dict[str, str] = {}
    for r in rows:
        sig, msg = r["signal"], r["msg_name"]
        both = f"{msg}_{sig}"
        if not (_WHEELISH.search(both) and _SPEEDISH.search(both)):
            continue
        if _FAULT.search(sig) or re.search(r"dir|pulse|moving", sig, re.I):
            continue
        for corner, pat in CORNERS:
            if corner not in found and pat.search(sig):
                found[corner] = sig
    return found


def survey(platform: str, dbcs: dict[str, str],
           dbc_dir: Path) -> tuple[dict, list[dict], dict[str, set[int]]]:
    rows: list[dict] = []
    for bus, name in dbcs.items():
        path = dbc_dir / f"{name}.dbc"
        if not path.is_file():
            continue
        for r in parse_dbc(path):
            r |= {"platform": platform, "bus": bus, "dbc": name}
            sysname, kind = classify(r["signal"])
            r |= {"system": sysname or "", "kind": kind}
            rows.append(r)

    sigs = [r["signal"] for r in rows]
    res: dict = {"platform": platform,
                 "dbc": "|".join(sorted({r["dbc"] for r in rows}))}

    # 直接作動。**作動するものだけ**を数える。故障灯も設定も数えない。
    extra = known_extra(platform)
    for sysname in SYSTEMS:
        hit = sorted({r["signal"] for r in rows
                      if r["system"] == sysname and r["kind"] == "作動"})
        add = extra.get(sysname, [])
        res[f"{sysname}_n"] = len(hit) + len(add)
        res[f"{sysname}_sig"] = "|".join(
            (hit + [f"{nm} (0x{a:03X}) ※DBC外" for nm, a in add])[:4])
        # 故障・設定しか無い場合が分かるようにしておく
        res[f"{sysname}_other"] = "|".join(sorted(
            {f"{r['signal']}({r['kind']})" for r in rows
             if r["system"] == sysname and r["kind"] in ("故障", "設定")})[:3])
    res["direct_n"] = sum(res[f"{s}_n"] > 0 for s in SYSTEMS)
    res["dbc_gap"] = bool(extra)

    corners = wheel_corners(rows)
    res["wheel_corners"] = len(corners)
    res["wheel_sig"] = "|".join(corners[c] for c in ("FL", "FR", "RL", "RR") if c in corners)

    for name, pat in LATERAL.items():
        hit = sorted({s for s in sigs if pat.search(s) and not _FAULT.search(s)
                      and not _SETTING.search(s)})
        res[name] = len(hit) > 0
        res[f"{name}_sig"] = hit[0] if hit else ""
    res["lateral_n"] = sum(bool(res[k]) for k in LATERAL)

    # 区分ごとに「その信号を載せているメッセージ ID」。実機確認で使う。
    need: dict[str, set[int]] = {}
    for sysname in SYSTEMS:
        need[sysname] = {r["msg_id"] for r in rows
                         if r["system"] == sysname and r["kind"] == "作動"}
        # DBC 外で分かっているアドレスも確認対象に入れる。
        need[sysname] |= {a for _nm, a in extra.get(sysname, [])}
    need["wheel"] = {r["msg_id"] for r in rows if r["signal"] in set(corners.values())}
    for name, pat in LATERAL.items():
        need[name] = {r["msg_id"] for r in rows
                      if pat.search(r["signal"]) and not _FAULT.search(r["signal"])
                      and not _SETTING.search(r["signal"])}
    return res, rows, need


def score(r: pd.Series) -> float:  # noqa: C901
    """スリップ探索の有望度。**直接作動信号を最優先**する。

    直接作動は車両自身の判断で、こちらが閾値を決める余地がない。
    輪速と横運動はそれを裏付ける・種別を分けるために要る。
    収録時間は出現数に効くので対数で入れる。
    """
    import math

    def have(key: str, n_key: str) -> int:
        """DBC に定義があり、かつ実機確認で否定されていないこと。

        実機確認をしていない (--verify 無し) 場合は DBC の定義だけで見る。
        確認して**流れていなかった**ものは 0 にする。
        """
        if not r.get(n_key, 0):
            return 0
        v = r.get(f"bus_{key}")
        return 0 if v is False else 1

    s = 0.0
    s += 3.0 * have("slip", "slip_n")     # 滑り警告。TSS2 で最も効いた
    s += 3.0 * have("esc", "esc_n")       # VSC/ESC 作動。横滑りの直接証拠
    s += 2.0 * have("abs", "abs_n")       # ABS 作動
    s += 1.5 * have("trc", "trc_n")       # TRC/TCS
    wheel_ok = 0 if r.get("bus_wheel") is False else 1
    s += 2.0 * wheel_ok * (r.get("wheel_corners", 0) / 4.0)
    lat = sum(1 for k in ("steer_angle", "yaw_rate", "lat_accel")
              if r.get(k) and r.get(f"bus_{k}") is not False)
    s += 1.5 * (lat / 3.0)
    s += 1.5 * math.log10(max(r.get("hours", 1.0), 1.0)) / math.log10(200.0)
    return round(s, 2)


def load_hours(cache: Path) -> dict[str, float]:
    """README の一覧表から車種ごとの収録時間を取る。取得は 1 回だけ。"""
    if not cache.is_file():
        import urllib.request
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(README_URL) as r:
            cache.write_bytes(r.read())
    out: dict[str, float] = {}
    pat = re.compile(r"^\|\s*([A-Z0-9_]+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|")
    for line in cache.read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1)] = float(m.group(3))
    return out


class _Spec:
    """--mapping から読んだときに PLATFORMS と同じ形で扱うための薄い入れ物。"""

    def __init__(self, dbc: dict[str, str]) -> None:
        self.config = self
        self.dbc_dict = dbc


# 実機確認で必ず見るアドレス。**DBC に無いがこちらで解析したもの**を含む。
#   0x226  ABSACT / VSCACT / ブレーキ油圧   (Toyota。opendbc の DBC に無い)
#   0x3B7  SLIP_WARN (SLP_WL) / TC_DISABLED (同上)
#   0x024  YAW_RATE / ACCEL_Y
WATCH_ADDRESSES = {
    0x226: "ABS/VSC 作動 + ブレーキ油圧 (Toyota)",
    0x3B7: "滑り警告灯 + TC 解除 (Toyota)",
    0x024: "ヨーレート + 横加速度 (Toyota)",
    0x0AA: "4 輪速 (Toyota)",
}


def bus_addresses(cache: Path, names: list[str]) -> dict[int, set[int]]:
    """キャッシュ済みセグメントを読んで、バスごとに流れているアドレスを集める。"""
    from near_miss.io import rlog as R

    sch = R.load_schema()
    out: dict[int, set[int]] = defaultdict(set)
    for name in names:
        path = Path(name)
        if not path.is_file():
            continue
        raw = R._decompress(path)
        for ev in sch.Event.read_multiple_bytes(raw):
            if ev.which() != "can":
                continue
            for c in ev.can:
                out[int(c.src)].add(int(c.address))
    return dict(out)


def verify(platform: str, seg_names: list[str], cache: Path, n: int,
           need: dict[str, set[int]] | None = None) -> dict:
    """数セグメント取得して、実際にバスへ流れているアドレスを確かめる。

    `need` は区分ごとに「その信号を載せているメッセージ ID」。DBC に定義が
    あっても車両が送っていないことがあるので、**区分ごとに実機で確かめる**。
    comma2k19 の RAV4 2017 では 0x226 が DBC にあって車両は送っていなかった。
    """
    from near_miss.io import comma_car_segments as ccs

    want = seg_names[:n]
    got = ccs.ensure_segments(want, cache_dir=cache)
    bad = [f"{k}: {v}" for k, v in got.items() if isinstance(v, Exception)]
    paths = [str(v) for v in got.values() if not isinstance(v, Exception)]
    if not paths:
        return {"verify_error": bad[0] if bad else "取得できませんでした"}
    buses = bus_addresses(cache, paths)
    allad = set().union(*buses.values()) if buses else set()
    res = {"verify_n": len(paths), "n_addresses": len(allad),
           "buses": "|".join(str(b) for b in sorted(buses))}
    for addr in WATCH_ADDRESSES:
        res[f"addr_{addr:03X}"] = addr in allad
    for key, ids in (need or {}).items():
        res[f"bus_{key}"] = bool(ids & allad) if ids else False
    return res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--opendbc", type=Path, required=True,
                   help="opendbc のソースディレクトリ (dbc/ と car/ を含む)")
    p.add_argument("--mapping", type=Path, default=None,
                   help="車種 -> DBC の対応 JSON。opendbc は Python 3.11 以上を要求するので、"
                        "3.10 の環境ではこれを先に別インタプリタで書き出して渡す "
                        "(docstring の手順を参照)")
    p.add_argument("--brands", default="TOYOTA,LEXUS",
                   help="この接頭辞の車種は時間によらず含める (空で無効)")
    p.add_argument("--min-hours", type=float, default=50.0,
                   help="この時間以上の車種を含める")
    p.add_argument("--all", action="store_true", help="全車種を対象にする")
    p.add_argument("--readme-cache", type=Path,
                   default=Path("raw_data/comma_car_segments/README.md"))
    p.add_argument("--out", type=Path, default=Path("out/platform_survey"))
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--verify", action="store_true",
                   help="車種ごとに数セグメント落として、実際にバスへ流れている "
                        "アドレスを確かめる")
    p.add_argument("--verify-n", type=int, default=3,
                   help="--verify のとき 1 車種あたり読むセグメント数 (既定 3)")
    p.add_argument("--cache", type=Path,
                   default=Path("raw_data/comma_car_segments"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.opendbc
    if not (root / "opendbc" / "dbc").is_dir():
        raise SystemExit(f"opendbc が見つかりません: {root}\n"
                         "  取得: git clone https://github.com/commaai/opendbc")
    if args.mapping and args.mapping.is_file():
        PLATFORMS = {k: _Spec(v) for k, v in
                     json.loads(args.mapping.read_text(encoding="utf-8")).items()}
        print(f"車種 -> DBC: {args.mapping} から {len(PLATFORMS)} 件")
    else:
        sys.path.insert(0, str(root))
        try:
            from opendbc.car.values import PLATFORMS  # type: ignore
        except Exception as exc:
            raise SystemExit(
                f"opendbc を読めません: {exc}\n"
                "  opendbc は Python 3.11 以上を要求します。この環境が 3.10 なら、\n"
                "  対応表を別インタプリタで書き出して --mapping で渡してください "
                "(docstring の手順)")

    dbc_dir = root / "opendbc" / "dbc"
    if not list(dbc_dir.glob("*_generated.dbc")):
        raise SystemExit(f"生成済み DBC がありません。先に次を実行してください:\n"
                         f"  cd {dbc_dir / 'generator'} && python generator.py")

    hours = load_hours(args.readme_cache)
    brands = tuple(b.strip() for b in args.brands.split(",") if b.strip())

    targets = []
    for ds_name, h in hours.items():
        key = ALIASES.get(ds_name, ds_name)
        if key not in PLATFORMS:
            continue          # opendbc に定義が無い車種
        if args.all or h >= args.min_hours or (brands and ds_name.startswith(brands)):
            targets.append((ds_name, key))
    print(f"対象 {len(targets)} 車種 / データセット掲載 {len(hours)} 車種")
    print(f"  条件: {'全部' if args.all else f'{brands} または {args.min_hours} 時間以上'}\n")

    res_rows, sig_rows = [], []
    needs: dict[str, dict[str, set[int]]] = {}
    for ds_name, key in sorted(targets):
        p = PLATFORMS[key]
        dbcs = {str(getattr(k, "value", k)): v for k, v in (p.config.dbc_dict or {}).items()}
        r, rows, need = survey(ds_name, dbcs, dbc_dir)
        r["hours"] = hours[ds_name]
        r["opendbc_name"] = key
        res_rows.append(r)
        sig_rows.extend(rows)
        needs[ds_name] = need

    if args.verify:
        from near_miss.io import comma_car_segments as ccs
        db = ccs.load_database(args.cache) if hasattr(ccs, "load_database") else None
        if db is None:
            db = json.loads((args.cache / "database.json").read_text(encoding="utf-8"))
        print(f"実機確認: 1 車種あたり {args.verify_n} セグメント "
              f"(約 {args.verify_n * 1.4:.1f} MB x {len(res_rows)} 車種)\n")
        for r in res_rows:
            names = db.get(r["platform"], [])
            if not names:
                r["verify_error"] = "database.json に該当なし"
                continue
            r |= verify(r["platform"], names, args.cache, args.verify_n,
                        needs.get(r["platform"]))
            cat = "".join("○" if r.get(f"bus_{k}") else "×"
                          for k in ("abs", "esc", "trc", "slip", "wheel",
                                    "steer_angle", "yaw_rate", "lat_accel"))
            toy = "".join("○" if r.get(f"addr_{a:03X}") else "×"
                          for a in WATCH_ADDRESSES)
            print(f"  {r['platform']:32s} {r.get('n_addresses', 0):4d}種  "
                  f"区分 {cat}  Toyota枠 {toy}  {r.get('verify_error','')}")

    df = pd.DataFrame(res_rows)
    df["score"] = df.apply(score, axis=1)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "platforms.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(args.out / "signals.csv", index=False)

    show = ["platform", "hours", "score", "abs_n", "esc_n", "trc_n", "slip_n",
            "wheel_corners", "steer_angle", "yaw_rate", "lat_accel"]
    disp = df[show].head(args.top).copy()
    disp.columns = ["車種", "時間", "点", "ABS", "ESC", "TRC", "滑り",
                    "輪速", "舵角", "ヨー", "横G"]
    for c in ("舵角", "ヨー", "横G"):
        disp[c] = disp[c].map({True: "○", False: "×"})
    print(disp.to_string(index=False))
    print(f"\n出力: {args.out}")
    print("  ※ ここまでは DBC 定義。実際にバスへ流れているかは別問題です "
          "(docs/comma_car_segments.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
