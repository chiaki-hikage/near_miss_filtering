#!/usr/bin/env bash
#
# 横滑り 2 段フィルタを 1 コマンドで動かす。
#
#   ./scripts/demo_sideslip.sh            # 既定 30 セグメント (約 30 分の走行)
#   ./scripts/demo_sideslip.sh -n 100     # セグメント数を変える
#   ./scripts/demo_sideslip.sh -y         # 取得の確認を飛ばす (CI / 非対話)
#   ./scripts/demo_sideslip.sh --no-fetch # 手元にあるぶんだけで動かす
#   ./scripts/demo_sideslip.sh --from-s3  # EC2: 取得元を指定の S3 バケットにする
#
# 全データ (2,000 セグメント / 33 時間 / 約 12 分) は流さない。
# それをやるときは docs/environment.md の「全件を流す」を見ること。
#
# Mac (Darwin/arm64) でも EC2 (Linux/aarch64) でも同じ手順で動く。
set -euo pipefail

PLATFORM="TOYOTA_RAV4_TSS2"
N=30
ASSUME_YES=0
FETCH=1
FROM_S3=0
OUT="out/demo_sideslip"
MB_PER_SEGMENT=1.38          # 実測平均 (scripts/fetch_car_segments.py と同じ値)

while [ $# -gt 0 ]; do
  case "$1" in
    -n) N="$2"; shift 2 ;;
    -y|--yes) ASSUME_YES=1; shift ;;
    --no-fetch) FETCH=0; shift ;;
    --from-s3) FROM_S3=1; shift ;;
    --out) OUT="$2"; shift 2 ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "不明な引数: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."
REPO="$(pwd)"
CACHE="$REPO/raw_data/comma_car_segments"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "0. 環境"
echo "  $(uname -s) $(uname -m)  /  $REPO"
command -v uv >/dev/null 2>&1 || {
  echo "uv がありません。次で入れてください:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}
echo "  uv $(uv --version | awk '{print $2}')"

step "1. 依存の同期 (uv sync)"
if [ "$FROM_S3" -eq 1 ]; then
  uv sync --extra viz --extra dev --extra s3      # boto3 は S3 から取るときだけ
else
  uv sync --extra viz --extra dev
fi

step "2. 環境の確認"
uv run python scripts/check_env.py --data

step "3. 単体試験 (データ不要)"
uv run python -m pytest tests/test_sideslip.py -q

step "4. データの用意"
HAVE=0
if [ -d "$CACHE/segments" ]; then
  HAVE=$(find "$CACHE/segments" -name 'rlog.zst' | wc -l | tr -d ' ')
fi
echo "  手元の rlog: $HAVE 本 (要求 $N 本)"
if [ "$FROM_S3" -eq 0 ] && [ "$(uname -s)" = "Linux" ] && [ -n "${NEAR_MISS_S3_URI:-}" ]; then
  echo "  (NEAR_MISS_S3_URI が設定されています。S3 から取るなら --from-s3)"
fi
if [ "$HAVE" -lt "$N" ]; then
  if [ "$FETCH" -eq 0 ]; then
    echo "  --no-fetch が指定されています。手元の $HAVE 本で続けます。"
    N="$HAVE"
  else
    NEED=$((N - HAVE))
    SIZE=$(awk "BEGIN{printf \"%.0f\", $NEED * $MB_PER_SEGMENT}")
    echo "  取得が要ります: 約 ${SIZE} MB (rlog ${NEED} 本)"
    [ -f "$CACHE/database.json" ] || echo "  ＋ セグメント一覧 database.json 約 9 MB"
    if [ "$ASSUME_YES" -ne 1 ]; then
      if [ ! -t 0 ]; then
        echo "  非対話で実行されています。取得してよければ -y を付けて再実行してください。" >&2
        exit 1
      fi
      printf "  取得しますか? [y/N] "
      read -r ans
      case "$ans" in [yY]*) ;; *) echo "  中止しました。"; exit 1 ;; esac
    fi
    if [ "$FROM_S3" -eq 1 ]; then
      # EC2: 指定の S3 バケットから取り込む。認証は IAM Role。
      uv run python scripts/fetch_from_s3.py car-segments \
        --platform "$PLATFORM" --limit "$N" --per-route 10 -y
    else
      uv run python scripts/fetch_car_segments.py "$PLATFORM" --limit "$N" --per-route 10
    fi
  fi
fi
if [ "$N" -lt 1 ]; then
  echo "処理できるセグメントがありません。" >&2
  exit 1
fi

step "5. 横滑り 2 段フィルタ ($N セグメント)"
# --select catalog: 手元のキャッシュに何が入っていても同じ N 本を選ぶ。
# 別のマシンで数字を突き合わせられるようにするため。
uv run python scripts/screen_sideslip.py --platform "$PLATFORM" \
  --limit "$N" --select catalog --per-route 10 --out "$OUT"

step "6. 出力の確認"
uv run python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
meta = json.loads((out / "counts.json").read_text(encoding="utf-8"))
c = meta["counts"]
chain = [("全データ", c["n_samples"]), ("適用範囲", c["n_in_range"]),
         ("1 次通過", c["n_stage1"]), ("2 次通過", c["n_stage2"])]
assert all(a[1] >= b[1] for a, b in zip(chain, chain[1:])), "各段の件数が単調に減っていません"
print(f"  {meta['dataset']} / {meta['label']} / {meta['vehicle']}")
print(f"  config_hash {meta['config_hash']}  {meta['n_segments']} セグメント "
      f"{c['n_hours']:.2f} 時間  {meta['elapsed_min']:.1f} 分")
print("  " + "  ->  ".join(f"{k} {v:,}" for k, v in chain) + f"  ->  候補 {c['n_candidates']}")
for name in ("candidates.csv", "counts.json"):
    p = out / name
    assert p.is_file(), f"{p} がありません"
    print(f"  OK  {p}  ({p.stat().st_size:,} バイト)")
PY

step "7. 本物の横滑りを拾えるかの確認 (KIT MSDM)"
KIT="$REPO/raw_data/kit_msdm/10.35097-44a91t97pmnha1k9/data/dataset"
if [ -d "$KIT" ] && [ -n "$(ls -A "$KIT"/*.mat 2>/dev/null)" ]; then
  uv run python scripts/validate_sideslip_filter.py --kind dynamic --min-speed 3 \
    | tail -n 6
else
  echo "  KIT MSDM が置かれていないので飛ばします。"
  echo "  置き場: raw_data/kit_msdm/  (docs/environment.md の「データの配置」)"
fi

printf '\n\033[1m完了\033[0m  出力: %s\n' "$OUT"
