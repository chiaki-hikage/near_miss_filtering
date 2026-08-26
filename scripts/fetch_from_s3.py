#!/usr/bin/env python3
"""指定した S3 バケットから検証用データを raw_data/ へ取り込む (**EC2 専用**)。

Mac ではこれまで通りローカルの raw_data/ を直接使う。この経路は通らない。
解析側 (screen_sideslip.py など) は S3 を一切知らない — ここは「置くまで」の話。

  # 何をどこへ取り込むかの取り決めを見る
  python scripts/fetch_from_s3.py --show-layout

  # 量を見積もるだけ。ダウンロードはしない
  python scripts/fetch_from_s3.py car-segments --platform TOYOTA_RAV4_TSS2 \
      --routes 3 --per-route 10 --dry-run

  # 実際に取り込む (車種 / route 単位)
  python scripts/fetch_from_s3.py car-segments --platform TOYOTA_RAV4_TSS2 --limit 200

  # KIT MSDM を丸ごと (約 172 MB)
  python scripts/fetch_from_s3.py kit-msdm

  # comma2k19 をチャンク単位で (1 本 約 9.7 GB)
  python scripts/fetch_from_s3.py comma2k19 --chunk Chunk_1 --dry-run

バケットの指定は --bucket > 環境変数 NEAR_MISS_S3_URI > configs/datasets/s3.yaml。
認証は boto3 の既定の解決順に任せる。EC2 では IAM Role が使われる。
**静的な鍵 (環境変数 / ~/.aws/credentials / .env) が使われようとしたら止まる。**
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from near_miss.io import s3_sync as s3

# これを超える取り込みは -y (--yes) が無いと実行しない。
# 事故で数十 GB を落とさないための歯止め。
DEFAULT_MAX_GB = 5.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "dataset",
        nargs="?",
        choices=sorted(s3.DATASETS) + ["all"],
        help="取り込むデータセット",
    )
    p.add_argument("--show-layout", action="store_true", help="S3 と raw_data の対応を出して終わる")
    p.add_argument("--bucket", help="s3://<bucket>/<prefix> (未指定なら環境変数 / 設定ファイル)")
    p.add_argument("--raw-data", type=Path, default=s3.RAW_DATA, help="取り込み先 (既定 raw_data/)")
    p.add_argument("--region", help="バケットのリージョン (通常は不要)")
    p.add_argument("--profile", help="AWS プロファイル (EC2 では使わない)")

    g = p.add_argument_group("commaCarSegments の絞り込み")
    g.add_argument("--platform", help="車種キー (例 TOYOTA_RAV4_TSS2)")
    g.add_argument("--limit", type=int, help="セグメント数の上限")
    g.add_argument("--routes", type=int, help="使う route 数")
    g.add_argument("--per-route", type=int, help="1 route あたりの連続セグメント数")

    g2 = p.add_argument_group("comma2k19 の絞り込み")
    g2.add_argument("--chunk", default="", help="チャンク名 (例 Chunk_1)。既定は全部")

    p.add_argument("--dry-run", action="store_true", help="対象と量だけ出す。取り込まない")
    p.add_argument("--workers", type=int, default=8, help="並列ダウンロード数")
    p.add_argument("--max-gb", type=float, default=DEFAULT_MAX_GB, help="確認なしで取り込む上限")
    p.add_argument("-y", "--yes", action="store_true", help="確認を飛ばす (非対話)")
    p.add_argument("--list-files", action="store_true", help="対象ファイルを 1 件ずつ出す")
    p.add_argument(
        "--allow-non-ec2",
        action="store_true",
        help="EC2 以外でも動かす (Mac はローカルの raw_data/ を使う運用なので通常は不要)",
    )
    p.add_argument(
        "--allow-any-credentials",
        action="store_true",
        help="IAM Role 以外の認証も許す (既定は静的な鍵を拒否)",
    )
    return p.parse_args()


def build_plan(client, location: s3.Location, name: str, args: argparse.Namespace) -> s3.Plan:
    ds = s3.DATASETS[name]
    if name != "car-segments":
        scope = args.chunk if name == "comma2k19" else ""
        return s3.plan_prefix(client, location, ds, scope=scope, raw_data=args.raw_data)

    if not args.platform:
        # 車種を指定しない場合は丸ごと。数十万件になりうるので警告を残す。
        plan = s3.plan_prefix(client, location, ds, raw_data=args.raw_data)
        plan.notes.append(
            "--platform を付けていないので車種を問わず全件が対象です。"
            "車種 / route 単位に絞るには --platform / --routes / --per-route を使ってください"
        )
        return plan

    names = _select_names(args, location, client)
    return s3.plan_car_segments(client, location, names, raw_data=args.raw_data)


def _select_names(args: argparse.Namespace, location: s3.Location, client) -> list[str]:
    """database.json を見て、取り込むセグメント名を決める。

    database.json はまず S3 から取る (バケットの中身と食い違わないようにするため)。
    無ければ手元のものを使う。
    """
    from near_miss.io import comma_car_segments as ccs

    cache = args.raw_data / s3.DATASETS["car-segments"].local_root
    db_path = cache / "database.json"
    if not db_path.is_file():
        key = location.key(s3.DATASETS["car-segments"].s3_subprefix + "database.json")
        print(f"database.json を取得 : {location.uri('comma_car_segments/database.json')}")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = db_path.with_suffix(".json.part")
        try:
            client.download_file(location.bucket, key, str(tmp))
            tmp.replace(db_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"database.json を S3 から取得できません ({location.uri('comma_car_segments/database.json')})。\n"
                "バケットの中身を確認するか、先に "
                "`uv run python scripts/fetch_car_segments.py --list` で取得してください。"
            )

    try:
        return ccs.select_segments(
            args.platform,
            limit=args.limit,
            routes=args.routes,
            per_route=args.per_route,
            cache_dir=cache,
        )
    except KeyError as exc:
        raise SystemExit(
            f"{exc}\n車種キーの一覧: uv run python scripts/fetch_car_segments.py --list"
        )


def main() -> int:
    args = parse_args()

    if args.show_layout:
        print(s3.layout_table())
        return 0
    if not args.dataset:
        raise SystemExit(
            "データセットを指定してください: "
            + " / ".join(sorted(s3.DATASETS))
            + " / all\n対応は --show-layout で確認できます。"
        )

    try:
        s3.guard_ec2(args.allow_non_ec2, "S3 からの取り込み")
    except PermissionError as exc:
        raise SystemExit(str(exc))
    try:
        location = s3.resolve_location(args.bucket)
    except ValueError as exc:
        raise SystemExit(str(exc))

    try:
        client, creds = s3.make_client(
            region=args.region, profile=args.profile, allow_any=args.allow_any_credentials
        )
    except (PermissionError, ImportError) as exc:
        raise SystemExit(str(exc))

    print(f"バケット     : {location.uri()}")
    print(f"認証         : {creds.method}" + ("  (IAM Role)" if creds.is_role else ""))
    print(f"EC2          : {'はい' if s3.is_ec2() else 'いいえ (--allow-non-ec2)'}")

    names = sorted(s3.DATASETS) if args.dataset == "all" else [args.dataset]
    plans = []
    for name in names:
        try:
            plans.append(build_plan(client, location, name, args))
        except Exception as exc:
            raise SystemExit(f"{name} の対象を決められません: {exc}")

    for plan in plans:
        print(s3.format_plan(plan, args.raw_data, args.list_files))

    total_todo = sum(len(p.todo) for p in plans)
    total_bytes = sum(p.todo_bytes for p in plans)
    if len(plans) > 1:
        print()
        print(f"合計取得予定 : {total_todo:,} ファイル / {s3.human_bytes(total_bytes)}")

    if args.dry_run:
        print("\n--dry-run のため取り込みません。")
        return 0
    if total_todo == 0:
        print("\nすべて取得済みです。")
        return 0

    if not s3.confirm_size(total_bytes, args.max_gb, args.yes, "取得"):
        return 1

    rc = 0
    for plan in plans:
        if plan.todo:
            rc |= s3.run_transfer(client, plan, args.workers, "取り込み")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
