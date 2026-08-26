#!/usr/bin/env python3
"""検証用データを指定した S3 バケットへ送り込む (**EC2 専用**)。

バケットを埋める側の作業。取りに行く側は scripts/fetch_from_s3.py。
**Mac から持っていく前提は取らない。** 公開元 (HuggingFace など) から EC2 上へ
取ってから、そのまま同じインスタンスでバケットへ上げられる。

  # 何をどこへ上げるかの取り決めを見る (fetch 側と同じ表)
  python scripts/upload_to_s3.py --show-layout

  # commaCarSegments を公開元から取りつつバケットへ (量の確認だけ)
  python scripts/upload_to_s3.py car-segments --platform TOYOTA_RAV4_TSS2 \
      --limit 2000 --per-route 10 --fetch --dry-run

  # 手元 (raw_data/) にあるものをそのまま上げる
  python scripts/upload_to_s3.py kit-msdm
  python scripts/upload_to_s3.py comma2k19 --chunk Chunk_1 --dry-run

キーの決め方は fetch 側と同じ対応表 (s3_sync.DATASETS) を逆向きに使う。
対応が 1 か所にしか無いので、上げた場所と取りに行く場所がずれない。

書き込み権限が要るので、取り込み用とは **別の IAM Role** を使うこと
(取り込み側は読み取り専用のままにしておく)。docs/environment.md §4.6 を見ること。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from near_miss.io import s3_sync as s3

# これを超える送信は -y が無いと実行しない。
DEFAULT_MAX_GB = 5.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dataset", nargs="?", choices=sorted(s3.DATASETS) + ["all"], help="上げるデータセット")
    p.add_argument("--show-layout", action="store_true", help="S3 と raw_data の対応を出して終わる")
    p.add_argument("--bucket", help="s3://<bucket>/<prefix> (未指定なら環境変数 / 設定ファイル)")
    p.add_argument("--raw-data", type=Path, default=s3.RAW_DATA, help="読み元 (既定 raw_data/)")
    p.add_argument("--region", help="バケットのリージョン (通常は不要)")
    p.add_argument("--profile", help="AWS プロファイル (EC2 では使わない)")

    g = p.add_argument_group("commaCarSegments の絞り込み")
    g.add_argument("--platform", help="車種キー (例 TOYOTA_RAV4_TSS2)")
    g.add_argument("--limit", type=int, help="セグメント数の上限")
    g.add_argument("--routes", type=int, help="使う route 数")
    g.add_argument("--per-route", type=int, help="1 route あたりの連続セグメント数")
    g.add_argument(
        "--fetch",
        action="store_true",
        help="手元に無いセグメントを公開元から取ってから上げる (Mac を経由しないため)",
    )
    g.add_argument("--fetch-workers", type=int, default=4, help="公開元からの並列取得数")

    g2 = p.add_argument_group("comma2k19 の絞り込み")
    g2.add_argument("--chunk", default="", help="チャンク名 (例 Chunk_1)。既定は全部")

    p.add_argument("--dry-run", action="store_true", help="対象と量だけ出す。上げない")
    p.add_argument("--workers", type=int, default=8, help="並列アップロード数")
    p.add_argument("--max-gb", type=float, default=DEFAULT_MAX_GB, help="確認なしで上げる上限")
    p.add_argument("-y", "--yes", action="store_true", help="確認を飛ばす (非対話)")
    p.add_argument("--list-files", action="store_true", help="対象ファイルを 1 件ずつ出す")
    p.add_argument("--allow-non-ec2", action="store_true", help="EC2 以外でも動かす")
    p.add_argument(
        "--allow-any-credentials",
        action="store_true",
        help="IAM Role 以外の認証も許す (既定は静的な鍵を拒否)",
    )
    return p.parse_args()


def _car_segment_sources(args: argparse.Namespace) -> list[Path]:
    """上げる commaCarSegments のファイル一覧を決める。

    --platform を付けたときだけ選び方が効く。`fetch_from_s3.py` と同じ
    `select_segments()` を使うので、上げる側と取る側で同じ範囲になる。
    """
    from near_miss.io import comma_car_segments as ccs

    cache = args.raw_data / s3.DATASETS["car-segments"].local_root
    db = cache / "database.json"
    if not db.is_file():
        if not args.fetch:
            raise SystemExit(
                f"{db} がありません。\n"
                "  公開元から取る場合   : uv run python scripts/fetch_car_segments.py --list\n"
                "  この場で取る場合     : --fetch を付ける"
            )
        print("database.json を公開元から取得します (約 9 MB)")
        ccs.fetch_database(cache)

    try:
        names = ccs.select_segments(
            args.platform,
            limit=args.limit,
            routes=args.routes,
            per_route=args.per_route,
            cache_dir=cache,
        )
    except KeyError as exc:
        raise SystemExit(f"{exc}\n車種キーの一覧: uv run python scripts/fetch_car_segments.py --list")

    paths = [ccs.local_path(n, cache) for n in names]
    missing = [n for n, p in zip(names, paths) if not p.is_file()]

    if missing and args.fetch:
        if args.dry_run:
            print(f"手元に無い {len(missing)} 本は公開元から取ります (--dry-run のため今は取りません)")
        else:
            print(f"公開元から {len(missing)} 本を取得します (約 {len(missing) * 1.38:.0f} MB)")
            done = {"n": 0}

            def progress(name, result):
                done["n"] += 1
                state = "失敗" if isinstance(result, Exception) else "完了"
                print(f"  [{done['n']}/{len(missing)}] {state} {name}")

            ccs.ensure_segments(missing, cache, workers=args.fetch_workers, on_done=progress)
    elif missing:
        print(f"手元に無いセグメントが {len(missing)} 本あります (--fetch で公開元から取れます)")

    return [db] + [p for p in paths if p.is_file()]


def build_plan(client, location: s3.Location, name: str, args: argparse.Namespace) -> s3.Plan:
    ds = s3.DATASETS[name]
    if name == "car-segments" and args.platform:
        sources = _car_segment_sources(args)
        return s3.plan_upload(client, location, ds, raw_data=args.raw_data, sources=sources)

    scope = args.chunk if name == "comma2k19" else ""
    plan = s3.plan_upload(client, location, ds, scope=scope, raw_data=args.raw_data)
    if name == "car-segments":
        plan.notes.append(
            "--platform を付けていないので raw_data にあるもの全部が対象です。"
            "車種 / route 単位に絞るには --platform / --routes / --per-route を使ってください"
        )
    return plan


def main() -> int:
    args = parse_args()

    if args.show_layout:
        print(s3.layout_table())
        return 0
    if not args.dataset:
        raise SystemExit(
            "データセットを指定してください: " + " / ".join(sorted(s3.DATASETS)) + " / all\n"
            "対応は --show-layout で確認できます。"
        )

    try:
        s3.guard_ec2(args.allow_non_ec2, "S3 への送り込み")
        location = s3.resolve_location(args.bucket)
        client, creds = s3.make_client(
            region=args.region, profile=args.profile, allow_any=args.allow_any_credentials
        )
    except (PermissionError, ImportError, ValueError) as exc:
        raise SystemExit(str(exc))

    print(f"バケット     : {location.uri()}")
    print(f"認証         : {creds.method}" + ("  (IAM Role)" if creds.is_role else ""))
    print(f"EC2          : {'はい' if s3.is_ec2() else 'いいえ (--allow-non-ec2)'}")
    print("向き         : raw_data/ -> S3  (**バケットへ書き込みます**)")

    names = sorted(s3.DATASETS) if args.dataset == "all" else [args.dataset]
    plans = []
    for name in names:
        try:
            plans.append(build_plan(client, location, name, args))
        except SystemExit:
            raise
        except Exception as exc:
            raise SystemExit(f"{name} の対象を決められません: {exc}")

    for plan in plans:
        print(s3.format_plan(plan, args.raw_data, args.list_files))

    total_todo = sum(len(p.todo) for p in plans)
    total_bytes = sum(p.todo_bytes for p in plans)
    if len(plans) > 1:
        print()
        print(f"合計送信予定 : {total_todo:,} ファイル / {s3.human_bytes(total_bytes)}")

    if args.dry_run:
        print("\n--dry-run のため上げません。")
        return 0
    if total_todo == 0:
        print("\nすべて送信済みです。")
        return 0

    if not s3.confirm_size(total_bytes, args.max_gb, args.yes, "送信"):
        return 1

    rc = 0
    for plan in plans:
        if plan.todo:
            rc |= s3.run_transfer(client, plan, args.workers, "送り込み")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
