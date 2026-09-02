# 実行環境 — Mac と AWS EC2 (ARM/Linux) で同じコードを動かす

横滑り 2 段フィルタ ([sideslip_filter.md](sideslip_filter.md)) を、
手元の Mac と AWS EC2 の両方で同じ手順・同じ結果で流すための手引き。

| | 検証状況 |
|---|---|
| macOS 15 / Apple M4 Pro (arm64) | **実測済み** (この文書の数値はすべてこの環境) |
| AWS EC2 c7g.4xlarge / Linux (aarch64) | **未実行**。依存の解決可否までは確認済み (§2) |

> EC2 では**まだ動かしていない**。この文書は「動かすための手順」であって
> 「動いた記録」ではない。実際に流したら §8.2 の突き合わせを行い、
> 結果をこの表に書き足すこと。
>
> S3 とのやりとり (§4.4 / §4.5) も**実際の AWS には接続していない**。
> キーの対応・往復・認証の拒否条件は偽のクライアントで確認済み
> (`tests/test_s3_sync.py` 48 件) だが、実バケットでの疎通は未確認。
> 最初は必ず `--dry-run` で件数と容量を確かめること。

---

## 1. OS に依存するのはどこか

**判定・抽出の処理は OS に依存しない。** 依存するのは次の 4 点だけで、
いずれも 1 か所に閉じ込めてある。

| 依存するもの | 何が違うか | どこで吸収しているか |
|---|---|---|
| 日本語フォント | mac は Hiragino Sans、Linux は Noto Sans CJK JP など | [`src/near_miss/plotting.py`](../src/near_miss/plotting.py) |
| matplotlib のバックエンド | 画面の無い Linux では `Agg` が要る | 同上 (`plotting.setup()`) |
| 標準出力の文字コード | Linux は `LANG` 未設定だと ASCII になり、日本語を出した時点で落ちる | [`scripts/_bootstrap.py`](../scripts/_bootstrap.py) |
| データの取得元 | EC2 は指定の S3 バケット、Mac はローカルの `raw_data/` | [`src/near_miss/io/s3_sync.py`](../src/near_miss/io/s3_sync.py) (§4.4) |

上 3 つは判定の経路にある。4 つ目 (S3) は**判定の経路には無い**。
データを `raw_data/` に置くまでの話で、置いたあとの読み出し・判定は
S3 を使ったかどうかを知らない。

`src/near_miss/` の他のモジュール (`signals` / `features` / `detectors` /
`sideslip` / `scoring` / `parallel` / `sources`) は `plotting.py` も
`s3_sync.py` も **import していない** (`tests/test_s3_sync.py` で機械的に確認)。
横滑りの判定ロジックには一切手を入れていない。

外部コマンドへの依存は `ffmpeg` だけで、これは comma1M の動画切り出し
(`review_comma1m_clips.py`) にしか使わない。横滑りフィルタには要らない。

### 依存していないと確認したもの

* **絶対パス** — `src` / `scripts` / `configs` に `/Users/...` の直書きは無い。
  基準は `config.REPO_ROOT` (`__file__` からの相対) だけ。
* **ファイル名の大小** — macOS は大小を区別しないので、コード上の綴りと
  実際のディレクトリ名がずれていても Mac では動いてしまう。
  下記 3 つは実際に一致していることを確認済み。とくに `comma1M` は**末尾が大文字**。

      raw_data/comma_car_segments
      raw_data/comma1M
      raw_data/kit_msdm/10.35097-44a91t97pmnha1k9/data/dataset

* **並列処理** — `--workers` でドライブ単位に並列化できる (§7)。
  起動方式を `spawn` に固定してあるので、mac (既定 spawn) と
  Linux (既定 fork) で挙動が変わらない。結果は worker 数によらず同じ。

---

## 2. 依存パッケージ

`pyproject.toml` と `uv.lock` で固定してある。`uv.lock` は
**プラットフォーム非依存の解決結果**なので、Mac と Linux で同じバージョンが入る。

コンパイルが要る依存 (pycapnp / scipy / numpy / pandas / zstandard) は
**すべて linux aarch64 の wheel が `uv.lock` に載っている**ことを確認済み。
EC2 側に C/C++ のツールチェインは要らない。

| 区分 | 中身 | いつ要るか |
|---|---|---|
| 必須 | numpy, pandas, PyYAML, scipy, pycapnp, zstandard, requests | 横滑りフィルタ |
| `--extra viz` | matplotlib | 図を描くとき |
| `--extra dev` | pytest | 試験 |
| `--extra comma1m` | safetensors, pymap3d, reverse_geocoder, Pillow | comma1M を扱うとき |
| `--extra demo-dataset` | pyarrow | comma2k19 の demo split を取るとき |
| `--extra s3` | boto3 | **EC2 で S3 とデータをやりとりするとき** (§4.4 / §4.5)。Mac では要らない |

KIT MSDM の受け取りと照合 (§4.2) に追加の依存は要らない (標準ライブラリだけ)。

Python は `.python-version` で **3.10** に固定してある。
uv が無ければ自分で取ってくるので、OS 側の Python は使わない。

---

## 3. EC2 での構築

### 3.1 インスタンス

| | |
|---|---|
| 種別 | c7g.4xlarge (Graviton3 / arm64 / 16 vCPU / 32 GiB) |
| OS | Ubuntu 24.04 LTS (arm64) または Amazon Linux 2023 (aarch64) |
| ディスク | gp3 **50 GB 以上**を推奨 (内訳は §4) |
| IAM Role | 検証用バケットへの **読み取り専用** (§4.4)。バケットを埋めるときだけ書き込み可のロール (§4.5) |

**アクセスキーはインスタンスに置かない。** インスタンスプロファイル (IAM Role) を
付けておけば boto3 が自動で拾う。`scripts/fetch_from_s3.py` は静的な鍵が
使われようとしたら止まる (§4.4)。

メモリは 32 GiB あれば十分すぎる。Mac で 200 セグメントを流したときの
最大常駐は **382 MB** で、セグメント数を増やしても大きくは伸びない
(処理はルート単位で、持ち越すのは候補の表だけ)。

### 3.2 OS 側の準備

```bash
# Ubuntu 24.04
sudo apt-get update
sudo apt-get install -y git curl fonts-noto-cjk      # フォントは図を描く場合のみ

# Amazon Linux 2023
sudo dnf install -y git tar
sudo dnf install -y google-noto-sans-cjk-jp-fonts    # 図を描く場合のみ
```

`ffmpeg` は横滑りフィルタには不要。comma1M の動画を扱うときだけ入れる。

### 3.3 uv を入れる

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"      # または新しいシェルを開く
uv --version
```

### 3.4 clone

```bash
git clone <リポジトリの URL> near_miss_filtering
cd near_miss_filtering
```

> 現在の `origin` は GitLab ではなく GitHub
> (`https://github.com/chiaki-hikage/near_miss_filtering.git`) になっている。
> GitLab に置き換えるなら `git remote set-url origin <GitLab の URL>`。
> 手順自体はどちらでも変わらない。

### 3.5 依存を入れる

```bash
uv sync --extra viz --extra dev

# データを S3 とやりとりする場合は boto3 も入れる (§4.4 / §4.5)
uv sync --extra viz --extra dev --extra s3
```

`.venv/` が作られ、`uv.lock` のとおりに入る。**`uv.lock` は編集しないこと。**
Mac と EC2 で同じものが入っていることが、結果を突き合わせる前提になる。

### 3.6 確認

```bash
uv run python scripts/check_env.py --data
```

必須の依存が欠けていれば 1 で終わる。日本語フォントが無ければここで警告が出る
(図を描かないなら無視してよい)。

---

## 4. データの配置

`raw_data/` は `.gitignore` に入っている。**clone しても中身は付いてこない。**
下の場所に置く。パスの綴りはコード側の既定値と一致していること (§1)。

```
near_miss_filtering/
└── raw_data/
    ├── comma_car_segments/            横滑りフィルタの主対象
    │   ├── database.json                  約 9 MB
    │   └── segments/<dongle>/<route>/<n>/rlog.zst
    │                                      1 本あたり約 1.38 MB
    ├── kit_msdm/                      再現率の物差し (任意だが強く推奨)
    │   └── 10.35097-44a91t97pmnha1k9/data/dataset/*.mat
    │                                      41 本 / 約 172 MB
    ├── comma1M/                       任意 (寒冷地スクリーニング)
    └── Chunk_1/                       任意 (comma2k19。約 9.7 GB)
```

**この構成は取得元によらず同じ。** 解析側 (`screen_sideslip.py` など) は
データがどこから来たかを知らない。

| 環境 | 取得元 | 手段 |
|---|---|---|
| Mac | 公開元 (HuggingFace / RADAR4KIT) | `fetch_car_segments.py` など。**S3 は使わない** |
| EC2 | 指定の S3 バケット | `fetch_from_s3.py` (§4.4) |

    公開元 --(§4.1〜4.3)--> raw_data/     ← Mac はここで終わり
    公開元 --(§4.5 --fetch)--> S3 バケット --(§4.4)--> raw_data/   ← EC2

§4.1〜4.3 が公開元から取る場合。§4.4 が EC2 で S3 から取り込む場合、
§4.5 がそのバケットを埋める場合。**Mac から EC2 へデータを持っていく経路は用意しない。**

### 4.1 commaCarSegments (公開元から)

取得スクリプトが `raw_data/comma_car_segments/` へ落とす。
**必ず `--dry-run` で量を確かめてから実行すること。**

```bash
# 車種の一覧 (database.json 約 9 MB を落とすだけ)
uv run python scripts/fetch_car_segments.py --list

# 量の見積り
uv run python scripts/fetch_car_segments.py TOYOTA_RAV4_TSS2 --limit 30 --per-route 10 --dry-run

# 取得 (30 本 ≒ 41 MB)
uv run python scripts/fetch_car_segments.py TOYOTA_RAV4_TSS2 --limit 30 --per-route 10
```

| 本数 | 走行時間 | 容量の目安 |
|---:|---:|---:|
| 30 | 0.5 h | 41 MB |
| 200 | 3.3 h | 276 MB |
| 2,000 | 33.1 h | **2.8 GB** |

### 4.2 KIT MSDM (強く推奨)

フィルタが**本物の横滑りを拾えるか**を確かめる唯一のデータ
([kit_msdm.md](kit_msdm.md))。RADAR4KIT が配布している。

| | |
|---|---|
| DOI | 10.35097/44a91t97pmnha1k9 |
| 配布ページ | https://radar.kit.edu/radar/en/dataset/44a91t97pmnha1k9 |
| ライセンス | **CC BY-SA 4.0** (商用可・継承)。認証も申請も要らない |
| 配布物 | BagIt を tar 1 本にまとめたもの `msdm.tar` **171,741,696 バイト** |
| MD5 | `d7eda9478c28a88b60074ee8ab2b0286` |

`scripts/fetch_kit_msdm.py` が受け取りから照合までを行う。
**どこから来たかではなく中身で確かめる**ので、経路が限られる環境でも使える。

```bash
# 手元にある tar を使う (ネットワークを一切使わない)
uv run python scripts/fetch_kit_msdm.py --tar /mnt/media/msdm.tar

# 外に出られる環境で取ってくる (接続先を表示して確認を取る)
uv run python scripts/fetch_kit_msdm.py --url <配布 URL>

# 既に置いてあるものを照合するだけ
uv run python scripts/fetch_kit_msdm.py --verify-only
```

照合は 2 段。**どちらかが合わなければ展開しない。**

1. 配布物の大きさと MD5 が `configs/datasets/kit_msdm.yaml` の値と一致するか
2. 展開後、BagIt の `manifest-md5.txt` に載る **44 件すべて**の MD5

展開後は `raw_data/kit_msdm/10.35097-44a91t97pmnha1k9/data/dataset/` に
`*.mat` 41 本と `parameter.m` が並ぶ。

> **配布 URL はこの repo に持っていない。** 最初の取得が手作業で、コマンドが
> 記録として残っていないため。`--url` には配布ページから辿ったものを渡すこと。
> 閉鎖環境では §4.6 のとおり `--tar` を使うので URL は要らない。

### 4.3 comma2k19 (公開元から / 任意)

信号の突き合わせ (`check_signal_parity.py`) と、抽出の再確認に使う。
チャンク 1 本で約 9.7 GB。展開して `raw_data/Chunk_1/` に置く。
`screen_sideslip.py --comma2k19 raw_data/Chunk_1` のように場所を渡す。

---

### 4.4 EC2: 指定の S3 バケットから取り込む

EC2 では公開元ではなく、**あらかじめ用意した S3 バケット**から `raw_data/` へ
取り込む。対象は commaCarSegments / KIT MSDM / comma2k19 の 3 つ。

```bash
uv run python scripts/fetch_from_s3.py --show-layout
```

> **Mac ではこの経路は使わない。** EC2 の上でないと既定で止まる
> (疎通確認だけしたいときは `--allow-non-ec2 --dry-run`)。

#### バケット側の置き方

**S3 のキー構成を `raw_data/` の下と 1:1 にする。** これが唯一の取り決めで、
対応表は [`s3_sync.DATASETS`](../src/near_miss/io/s3_sync.py) の 1 か所にしかない。
こうしておくと取り込んだあとのパスが従来と同じになり、解析側に手を入れずに済む。

| 名前 | S3 (ルートプレフィックスの下) | 取り込み先 |
|---|---|---|
| `car-segments` | `comma_car_segments/` | `raw_data/comma_car_segments/` |
| `kit-msdm` | `kit_msdm/` | `raw_data/kit_msdm/` |
| `comma2k19` | `comma2k19/` | `raw_data/` (チャンクが直下に来る) |

ルートプレフィックスを `s3://<バケット>/near_miss/` とした場合の例:

```
s3://<バケット>/near_miss/comma_car_segments/database.json
    -> raw_data/comma_car_segments/database.json
s3://<バケット>/near_miss/comma_car_segments/segments/<dongle>/<route>/<n>/rlog.zst
    -> raw_data/comma_car_segments/segments/<dongle>/<route>/<n>/rlog.zst
s3://<バケット>/near_miss/kit_msdm/10.35097-44a91t97pmnha1k9/data/dataset/*.mat
    -> raw_data/kit_msdm/10.35097-44a91t97pmnha1k9/data/dataset/*.mat
s3://<バケット>/near_miss/comma2k19/Chunk_1/<drive>/<n>/processed_log/...
    -> raw_data/Chunk_1/<drive>/<n>/processed_log/...
```

バケットにデータを入れる側の手順は §4.5。**Mac から持っていく前提は取らない。**

#### バケットの指定

優先順は **引数 > 環境変数 > 設定ファイル**。

```bash
uv run python scripts/fetch_from_s3.py car-segments --bucket s3://<バケット>/near_miss
export NEAR_MISS_S3_URI=s3://<バケット>/near_miss
```

恒久的に決めるなら [`configs/datasets/s3.yaml`](../configs/datasets/s3.yaml) に
`uri:` を書く。**このファイルに書いてよいのはバケット名とプレフィックスだけ。**

#### 認証 — IAM Role のみ

インスタンスプロファイル (IAM Role) を付けておけば boto3 が自動で拾う。
**鍵はコードにも設定ファイルにも `.env` にも置かない。** 次の 2 つで機械的に担保している。

* 解決元が静的な鍵 (環境変数 / `~/.aws/credentials` / `.env`) だと**その場で止まる**
* リポジトリ内の `.env` / `.env.local` に `AWS_SECRET_ACCESS_KEY` 等があれば止まる

必要な権限は読み取りだけ。書き込み・削除は一切しない。

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::<バケット>",
      "Condition": {"StringLike": {"s3:prefix": ["near_miss/*"]}} },
    { "Effect": "Allow", "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::<バケット>/near_miss/*" }
  ]
}
```

同じ AZ / リージョンなら **S3 ゲートウェイ VPC エンドポイント**を作っておくと
NAT の転送料がかからない。2.8 GB を落とすだけでも効く。

#### 使い方

**必ず `--dry-run` で量を確かめてから実行すること。**

```bash
# 車種 / route 単位で絞って見積り
uv run python scripts/fetch_from_s3.py car-segments \
    --platform TOYOTA_RAV4_TSS2 --routes 3 --per-route 10 --dry-run

# 取り込む
uv run python scripts/fetch_from_s3.py car-segments \
    --platform TOYOTA_RAV4_TSS2 --limit 2000 --per-route 10

# KIT MSDM を丸ごと (約 172 MB)
uv run python scripts/fetch_from_s3.py kit-msdm

# comma2k19 はチャンク単位 (1 本 約 9.7 GB)
uv run python scripts/fetch_from_s3.py comma2k19 --chunk Chunk_1 --dry-run

# 3 つまとめて
uv run python scripts/fetch_from_s3.py all --chunk Chunk_1 --dry-run
```

`--dry-run` の出力例:

```
バケット     : s3://<バケット>/near_miss/
認証         : iam-role  (IAM Role)
EC2          : はい

データセット : car-segments
取得元       : s3://<バケット>/near_miss/comma_car_segments/
取り込み先   : /home/ec2-user/near_miss_filtering/raw_data
内訳         : 30 セグメント / 3 route
対象         : 31 ファイル / 50.0 MB
取得済み     : 0 ファイル
取得予定     : 31 ファイル / 50.0 MB
```

| 引数 | 効き |
|---|---|
| `--platform` | 車種で絞る。`database.json` を見て決める |
| `--routes` / `--per-route` | route 単位。**連番のセグメントを選ぶ**ので 60 秒境界の連結が保てる |
| `--limit` | セグメント数の上限 |
| `--chunk` | comma2k19 のチャンク |
| `--dry-run` | 対象と量だけ出す。取り込まない |
| `--list-files` | 対象を 1 件ずつ出す |
| `--workers` | 並列ダウンロード数 (既定 8) |
| `--max-gb` | この量を超えると確認を求める (既定 5 GB)。`-y` で飛ばす |

絞り込みの引数は `fetch_car_segments.py` と同じ意味・同じ選び方
(`select_segments()` を共有している)。同じ引数なら公開元から取っても
S3 から取っても**同じセグメントが選ばれる**。

#### 動作上の約束

* **元データを上書きしない。** 大きさが一致するファイルは飛ばす。
  途中で切れたファイル (大きさが違う) だけ取り直す
* **途中の残骸を残さない。** `.part` へ書いてから置き換える。失敗したら消す
* **バケット全体は列挙しない。** commaCarSegments は選んだ route の
  プレフィックスだけを見る (要求回数 = route 数)
* **`..` を含むキーは受け付けない。** 取り込み先が `raw_data/` の外に出ない
* 何度でも流してよい。落ちた続きから再開できる

#### 1 コマンドのデモを S3 経由で

```bash
./scripts/demo_sideslip.sh --from-s3
```

`uv sync` に `--extra s3` が付き、取得だけが S3 経由になる。
**抽出と出力は `--from-s3` の有無で変わらない。**

---

### 4.5 EC2: バケットにデータを入れる

バケットを埋める側の作業。**一度やれば以後は §4.4 だけで済む。**
`scripts/upload_to_s3.py` が使う対応表は §4.4 と同じ (`s3_sync.DATASETS`) で、
それを逆向きに使う。対応が 1 か所にしか無いので、**上げた場所と取りに行く場所が
ずれない**（`tests/test_s3_sync.py` で往復を確認している）。

```bash
uv run python scripts/upload_to_s3.py --show-layout
```

#### commaCarSegments — 公開元から取ってそのまま上げる

`--fetch` を付けると、手元に無いセグメントを公開元 (HuggingFace) から取ってから
上げる。**別のマシンにデータを用意しておく必要がない。**

```bash
# まず量を確かめる
uv run python scripts/upload_to_s3.py car-segments \
    --platform TOYOTA_RAV4_TSS2 --limit 2000 --per-route 10 --fetch --dry-run

# 実行 (公開元から 2.8 GB を取り、そのままバケットへ)
uv run python scripts/upload_to_s3.py car-segments \
    --platform TOYOTA_RAV4_TSS2 --limit 2000 --per-route 10 --fetch --max-gb 4
```

絞り込みの引数は §4.4 および `fetch_car_segments.py` と同じ意味。
**同じ引数なら、上げる範囲と取りに行く範囲が一致する**（同じ `select_segments()`
を使っている。`test_upload_selection_matches_fetch_selection` で確認)。

#### KIT MSDM / comma2k19 — 手元にあるものを上げる

この 2 つは公開元からの自動取得を用意していない (§4.2 / §4.3 の手順で
インスタンス上へ置く)。置いたあとで:

```bash
uv run python scripts/upload_to_s3.py kit-msdm --dry-run
uv run python scripts/upload_to_s3.py kit-msdm

uv run python scripts/upload_to_s3.py comma2k19 --chunk Chunk_1 --dry-run
uv run python scripts/upload_to_s3.py comma2k19 --chunk Chunk_1 -y --max-gb 10
```

#### 書き込み用の IAM Role は分ける

**取り込み用のロールは読み取り専用のままにしておくこと。** 埋める作業は
書き込みができる別のロール (別インスタンス、または一時的に付け替え) で行う。

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::<バケット>",
      "Condition": {"StringLike": {"s3:prefix": ["near_miss/*"]}} },
    { "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::<バケット>/near_miss/*" }
  ]
}
```

`s3:DeleteObject` は入れない。このスクリプトは**消さない**。

#### 動作上の約束

* **S3 側に同じ大きさで載っているものは上げ直さない。** 何度流してもよい
* **大きさが違えば上げ直す。** 途中で切れたものを直せる
* `.DS_Store` / `Thumbs.db` / `*.part` / `*.tmp` は上げない
* 削除は一切しない。バケットから消すのは手作業 (`aws s3 rm`)
* `--max-gb` (既定 5 GB) を超えると確認を求める。`-y` で飛ばす

---

### 4.6 閉鎖環境 (外に出られない EC2) への持ち込み

外向きの通信が塞がっている、あるいは許可制のところへ持ち込む場合の手順。
**内部の情報を外へ出さずに済む形**にしてある。

#### 考え方 — 取得と持ち込みを分ける

```
[外に出られる作業機]                       [閉鎖 EC2]
  公開元から取得
  MD5 と manifest で照合   --(S3)-->   S3 から取り込み
  S3 バケットへ送り込み                  再度 manifest で照合
```

* **閉鎖 EC2 は外に出ない。** 取りに行く先は S3 だけ。
  S3 ゲートウェイ VPC エンドポイントを作れば、**通信は AWS の網から出ない**
  (インターネットゲートウェイも NAT も経由しない)
* **外へ出す情報は「公開データを 1 本くれ」という要求だけ。** 送るのは固定の
  User-Agent のみで、認証情報もホスト名も利用者名も載せない
  (`test_user_agent_carries_no_host_or_user_info` で確認)
* **持ち込んだものは中身で照合する。** 経路 (プロキシ、媒体、S3) を
  信用しなくてよい

#### 手順

**1. 外に出られる作業機で取得し、照合する**

```bash
uv run python scripts/fetch_kit_msdm.py --url <配布 URL>
```

接続前に、接続先ホスト・プロキシ・送信する情報を表示して確認を取る
(`-y` で省略)。MD5 が合わなければ**展開せず、落としたファイルも消す**。

媒体で渡された tar があるなら、この機すら要らない:

```bash
uv run python scripts/fetch_kit_msdm.py --tar /mnt/media/msdm.tar
```

**2. S3 バケットへ送り込む**

```bash
uv run python scripts/upload_to_s3.py kit-msdm --dry-run
uv run python scripts/upload_to_s3.py kit-msdm
```

**3. 閉鎖 EC2 で取り込み、もう一度照合する**

```bash
uv run python scripts/fetch_from_s3.py kit-msdm
uv run python scripts/fetch_kit_msdm.py --verify-only
```

`--verify-only` は BagIt の 44 件すべての MD5 を取り直す (172 MB / 約 0.3 秒)。
ここが通れば、**配布元が出したものと 1 バイトも違わない**。

#### tar のまま持ち込む場合

`upload_to_s3.py` は展開後のファイルを上げるが、tar 1 本のまま置いてもよい。
その場合は閉鎖 EC2 側で:

```bash
aws s3 cp s3://<バケット>/near_miss/kit_msdm/msdm.tar raw_data/kit_msdm/
uv run python scripts/fetch_kit_msdm.py --tar raw_data/kit_msdm/msdm.tar
```

tar を展開する前に**中身を検査する**。絶対パス・`..`・シンボリックリンク・
デバイスファイルを含む要素があれば、**1 件も書かずに中止する**
(`test_extract_refuses_unsafe_tar_and_writes_nothing`)。
MD5 の照合を `--allow-checksum-mismatch` で飛ばしても、この検査は残る。

#### プロキシ経由で直接取る場合

社内プロキシの許可リストに `radar.kit.edu` を入れられるなら、閉鎖 EC2 から
直接取ってもよい。

```bash
export HTTPS_PROXY=http://proxy.internal:3128
uv run python scripts/fetch_kit_msdm.py --url <配布 URL>
# または
uv run python scripts/fetch_kit_msdm.py --url <配布 URL> --proxy http://proxy.internal:3128
```

**プロキシが TLS を終端していても MD5 の照合で気付ける。** 中身が書き換わって
いれば展開せずに止まる。

#### 外に出る通信の一覧

このプロジェクトで外部に出る可能性があるのは次だけ。閉鎖 EC2 では
**どれも起きない** (S3 だけを見るため)。

| 相手 | 何のため | 使うもの |
|---|---|---|
| `radar.kit.edu` | KIT MSDM の配布物 | `fetch_kit_msdm.py --url` |
| `huggingface.co` | commaCarSegments / comma2k19 | `fetch_car_segments.py`, `upload_to_s3.py --fetch` |
| `raw.githubusercontent.com` | rlog を読む capnp スキーマ (5 ファイル / 120 KB) | `fetch_cereal_schema.py` |
| `pypi.org` | 依存パッケージ | `uv sync` |

**capnp スキーマは clone に付いてこない。** `data/` が `.gitignore` に入っている
ため、clone しただけでは `data/cereal/` は空で、`rlog.zst` を 1 本も読めない。
commaCarSegments を扱うなら、**データと同じように持ち込む必要がある**。

```bash
# 外に出られる作業機で (120 KB)
uv run python scripts/fetch_cereal_schema.py

# 閉鎖 EC2 へは媒体か S3 で運ぶ
aws s3 cp --recursive data/cereal s3://<バケット>/near_miss/cereal      # 作業機
aws s3 cp --recursive s3://<バケット>/near_miss/cereal data/cereal      # 閉鎖 EC2
```

KIT MSDM だけを使うなら capnp は要らない (MAT ファイルを直接読むため)。

`uv sync` の依存は、閉鎖環境では社内ミラーを見るようにする:

```bash
export UV_INDEX_URL=https://pypi.internal/simple
uv sync --extra viz --extra dev --extra s3
```

持ち込みが足りているかは次で分かる:

```bash
uv run python scripts/check_env.py --data
```

---

## 5. 1 コマンドで動かす

```bash
./scripts/demo_sideslip.sh
```

これだけで次を順に行う。所要 **30 秒程度** (取得を除く)。

1. `uv sync`
2. `scripts/check_env.py --data` で環境とデータを確認
3. `pytest tests/test_sideslip.py` (データ不要の単体試験 15 件)
4. データが足りなければ**容量を表示して確認を取ってから**取得
5. `screen_sideslip.py` を 30 セグメントで実行
6. 出力の存在と、各段の件数が単調に減っていることを確認
7. KIT MSDM があれば再現率も測る

主な引数:

```bash
./scripts/demo_sideslip.sh -n 100        # セグメント数
./scripts/demo_sideslip.sh -y            # 取得の確認を飛ばす (非対話)
./scripts/demo_sideslip.sh --no-fetch    # 手元にあるぶんだけで動かす
./scripts/demo_sideslip.sh --from-s3     # EC2: 取得元を S3 にする (§4.4)
```

`--from-s3` が変えるのは **4 の取得元だけ**。5 以降の抽出・出力は同じ。

セグメントは `--select catalog` で選んでいる。**手元のキャッシュに何が
入っていても同じ 30 本が選ばれる**ので、Mac と EC2 で数字を突き合わせられる。

---

## 6. 個別に流す

```bash
# 横滑りの抽出 (commaCarSegments)
uv run python scripts/screen_sideslip.py --platform TOYOTA_RAV4_TSS2 \
    --limit 200 --out out/sideslip_rav4_tss2

# 決まった順で選ぶ (別マシンとの突き合わせ用)
uv run python scripts/screen_sideslip.py --platform TOYOTA_RAV4_TSS2 \
    --limit 30 --select catalog --out out/demo_sideslip

# 再現率の確認 (KIT MSDM)
uv run python scripts/validate_sideslip_filter.py --kind dynamic --min-speed 3

# beta の雑音の再当てはめ (車種を増やしたとき)
uv run python scripts/calibrate_beta_noise.py --platform TOYOTA_RAV4_TSS2 --limit 300

# 図 (matplotlib が要る。--extra viz)
uv run python scripts/plot_kit_run.py dynamic_driving_cobble_1 --out out/kit_msdm

# EC2: データを S3 から取り込む (§4.4)。まず --dry-run で量を見る
uv run python scripts/fetch_from_s3.py car-segments \
    --platform TOYOTA_RAV4_TSS2 --limit 2000 --per-route 10 --dry-run
```

### 全件を流す

```bash
# 逐次 (既定)
uv run python scripts/screen_sideslip.py --platform TOYOTA_RAV4_TSS2 \
    --out out/sideslip_rav4_tss2 --dump-stage1

# ドライブ単位の並列。結果は逐次とバイト単位で同じ (§7)
uv run python scripts/screen_sideslip.py --platform TOYOTA_RAV4_TSS2 \
    --workers 8 --out out/sideslip_rav4_tss2 --dump-stage1
```

| | Mac M4 Pro (12 コア) |
|---|---|
| 2,000 セグメント / 33.1 時間 / worker=1 | **11.9 分** |
| 同 worker=8 | **1.8 分** |
| 最大常駐メモリ (200 セグメント / worker=1) | 382 MB |

Graviton3 の単スレッド性能は M4 Pro より低いので、EC2 では 1 worker あたりは
これより遅いと見込まれる (**未実測**)。ただし c7g.4xlarge は 16 vCPU あるので、
worker を増やせば実時間では追いつく余地がある。

---

## 7. ドライブ単位の並列実行

`--workers N` でドライブ単位に並列化できる。**判定は一切変わらない。**

### なぜドライブ単位か

セグメントは 60 秒で切られていて、事象は境界を跨ぐ。連番のセグメントは
連結してから 20 Hz グリッドに載せる必要がある。**セグメント単位で配ると
この連結が壊れ、境界付近の候補が変わってしまう。**
ドライブ単位なら連結の単位がプロセスの中に収まるので、結果が変わらない。

RAV4 TSS2 の 2,000 セグメントは

    ドライブ 877 / 連続ブロック 877 / 1 ドライブあたり 1〜10 セグメント (中央 2)

で、877 個のタスクに分かれる。最大のタスクでも 10 セグメントなので、
8 並列でも終盤に 1 つだけ残って待つ、という詰まり方をしない。

1 ドライブ分の処理 (読み出し → グリッド → 特徴量 → 横滑りフィルタ) は
もともと他のドライブと何も共有していない。持ち回る状態が無いので、
そのまとまりをプロセスへ配るだけで済んでいる。

### 実測 (Mac M4 Pro / 12 コア / 2,000 セグメント)

    uv run python scripts/benchmark_workers.py --platform TOYOTA_RAV4_TSS2 \
        -w 1 2 4 8 --dump-stage1

| worker | 所要 [秒] | 速度比 | 並列効率 | 候補 | 結果 |
|---:|---:|---:|---:|---:|---|
| 1 | 713.7 | 1.00x | 100% | 41 | 基準 |
| 2 | 366.7 | 1.95x | 97% | 41 | **一致** |
| 4 | 187.3 | 3.81x | 95% | 41 | **一致** |
| 8 | 109.6 | 6.51x | 81% | 41 | **一致** |

worker=8 で **6.5 倍**。効率が 8 並列で落ちるのは、12 コアのうち性能コアが
限られていることと、プロセス起動 (spawn) のぶん。

### 結果が同じであることの確認

`scripts/benchmark_workers.py` が毎回、次の 3 つを worker=1 の結果と
**ハッシュで**突き合わせる。1 つでも違えば終了コード 1 で落ちる。

| 突き合わせるもの | 中身 |
|---|---|
| `candidates.csv` | 最終候補。ファイルのバイト列そのもの |
| `stage1_samples.csv.gz` | 1 次通過サンプル 2,502 行。**展開してから**比較 (gzip は書いた時刻が入るのでバイト比較できない) |
| `counts.json` | 各段の件数。実行ごとに変わる項目 (`run_at` / `elapsed_min` / `workers`) を除く |

上の 4 回すべてで一致した。さらに、**並列化を入れる前に保存してあった
逐次実行の `candidates.csv` とも一致**している。

同じ結果になるように、実装で守っていることは 3 つ。

1. 1 ドライブ分の計算は、逐次でも並列でも**同じ関数** (`parallel.run_drive`) を通る
2. 結果を**投入順に並べ直してから**連結する。候補の最終的な並べ替えは
   `|beta|` の降順だが、pandas の既定の整列は安定ではないので、
   同じ値が並んだときの順は入力順で決まる。順序を保たないと worker 数で行が入れ替わりうる
3. 集計 (`FilterCounts`) は足し算だけなので順序に依存しない

### OS 依存を避けるために

* **起動方式を `spawn` に固定**してある。macOS の既定は spawn、Linux の既定は fork で、
  既定に任せると同じコードでも OS で挙動が変わる。
  `tests/test_portability.py` が `parallel.py` に OS 分岐が無いことを確認している。
* worker へ渡すのは設定・車種定義・ファイルの場所だけ。
  ラムダやファイルハンドルのような pickle できないものは渡さない。
* `OMP_NUM_THREADS` 等を 1 に寄せてから pool を作る。プロセス並列と
  ライブラリ内スレッドが二重に増えないようにするため
  (抽出処理に行列積は無いので効きは小さいはずだが、環境差を減らす)。

### 使いどころ

* `--workers 0` で CPU 数に合わせる
* 既定は `1` (逐次)。**既存の動作を変えないため**、明示的に指定したときだけ並列になる
* 取得 (`fetch_car_segments.py`) の並列度は別物 (`--workers` はダウンロードのスレッド数)

---

## 8. 出力の確認

### 8.1 出るもの

| ファイル | 中身 |
|---|---|
| `<out>/candidates.csv` | 最終候補。`beta` の大きい順。等級・信頼度・通過理由・裏付け |
| `<out>/counts.json` | 各段の件数と、そのとき使った `sideslip` 設定一式 |
| `<out>/stage1_samples.csv.gz` | 1 次を通ったサンプルの明細 (`--dump-stage1` のとき) |

### 8.2 Mac と突き合わせる

`--select catalog` で 30 セグメントを流したときの **Mac での実測値**。
EC2 でも同じ値が出れば、環境の移植は成功している。

```
config_hash : 4d8905fa3b
時間        : 0.499 h (30 セグメント)
```

| 段階 | 値 |
|---|---:|
| 全データ | 35,925 |
| beta が計算できた | 32,763 |
| 適用範囲 (v>=10) | 30,891 |
| 1 次通過 | 285 (区間 10) |
| 　理由 lat_dynamics | 99 |
| 2 次通過 | 241 (区間 2) |
| 　落ちた: 持続時間 | 7 |
| 　落ちた: 整合 | 4 |
| **最終候補** | **2** |

候補 2 件 (どちらも `C_弱い候補` / 信頼度 2.0 / `unexplained|lateral_force`):

| ドライブ | セグメント | 時刻 [s] | `beta` [deg] | v [m/s] |
|---|---:|---:|---:|---:|
| `ba88d5bd99ef8188|00000002--bcf716e22f` | 2 | 7.50 | 0.58 | 26.89 |
| `ba88d5bd99ef8188|00000002--bcf716e22f` | 7 | 11.85 | −0.16 | 21.32 |

**まず `config_hash` を見ること。** ここが違えば設定が違うので、
件数を比べても意味がない。

`config_hash` が同じで件数が違う場合、疑う順:

1. 選ばれたセグメントが違う (`database.json` の版が違う)。
   選定は `select_segments` の決まった順なので、同じ `database.json` なら同じ 30 本になる。
   確認: `--select catalog` を使っているか、`counts.json` の `n_segments` が 30 か。
2. 依存のバージョンが違う (`uv sync` を使わず手で入れた等)。
   確認: `uv run python scripts/check_env.py` の版数を Mac と見比べる。
3. rlog が途中までしか落ちていない。
   確認: `find raw_data/comma_car_segments/segments -name rlog.zst | wc -l`

### 8.3 KIT MSDM での再現率

```
実測 |β| >= 5.0 deg を含む走行 : 13 / 13
候補を 1 件以上出した走行       : 13 / 13
正解時間を覆った割合           : 100.0%
```

ここが 100% から下がったら、環境ではなく**判定の設定が変わっている**。

---

## 9. よくある詰まり

| 症状 | 原因と対処 |
|---|---|
| `UnicodeEncodeError` が出る | `LANG` が未設定。`scripts/_bootstrap.py` が UTF-8 に直すので、通常は出ないはず。素の `python` で `src/` を直接叩いた場合は `export PYTHONUTF8=1` |
| 図の日本語が □ になる | 日本語フォントが無い。§3.2 で入れて `rm -rf ~/.cache/matplotlib` |
| `matplotlib` が import できない | `uv sync --extra viz` |
| `capnp` が無いと言われる | `uv sync` (必須依存に入っている)。手で pip を使っていないか確認 |
| `車種設定がありません` | `--platform` の綴り。`fetch_car_segments.py --list` で確認 |
| `yaw_rate_noise_dps がありません` | 車種設定に beta の雑音が入っていない。`calibrate_beta_noise.py` を先に実行 |
| `uv run` が `VIRTUAL_ENV` の警告を出す | 別の venv が有効になっている。`deactivate` するか無視してよい |
| ディスクが足りない | §4 の内訳を見て、要らないデータセットを置かない |
| `EC2 の上ではありません` | S3 からの取り込みは EC2 限定。Mac はローカルの `raw_data/` を使う。疎通確認だけなら `--allow-non-ec2 --dry-run` |
| `S3 バケットが指定されていません` | `--bucket` / `NEAR_MISS_S3_URI` / `configs/datasets/s3.yaml` のどれかで指定する (§4.4) |
| `静的な認証情報 (env) が使われようとしています` | 鍵がインスタンスに置かれている。取り除いて IAM Role を付ける。意図的な場合のみ `--allow-any-credentials` |
| `AWS の認証情報が見つかりません` | インスタンスに IAM Role が付いていない (§4.4) |
| `boto3 がありません` | `uv sync --extra s3` |
| `S3 に無いセグメントが N 件あります` | バケットの中身が `database.json` より少ない。`--list-files` で対象を確認し、上げ直す |
| `AccessDenied` が出る | IAM ポリシーの `s3:prefix` 条件とバケット側のプレフィックスがずれている (§4.4) |
| S3 からの取り込みが遅い / 転送料が高い | S3 ゲートウェイ VPC エンドポイントを作る (§4.4)。`--workers` も上げられる |
| 上げるときに `AccessDenied` | 取り込み用の読み取り専用ロールのままになっている。書き込み可のロールに替える (§4.5) |
| `database.json がありません` (上げる側) | `--fetch` を付けるか、先に `fetch_car_segments.py --list` |
| 上げたのに `fetch_from_s3.py` が見つけられない | プレフィックスがずれている。両方で同じ `--bucket` / `NEAR_MISS_S3_URI` を使っているか確認。`--show-layout` で対応を見る |

---

## 10. 変更したときに崩れていないか

```bash
uv run python -m pytest -q                                    # 193 件
uv run python scripts/validate_sideslip_filter.py --kind dynamic --min-speed 3
./scripts/demo_sideslip.sh -n 30 --no-fetch
```

この 3 つが通れば、環境と判定の両方が保たれている。

S3 経路やデータの受け取りを触ったときは追加で:

```bash
uv run python -m pytest tests/test_s3_sync.py -q               # 48 件、AWS には繋がない
uv run python -m pytest tests/test_kit_msdm_fetch.py -q        # 18 件、外に出ない
uv run python scripts/fetch_from_s3.py --show-layout           # 対応表が壊れていないか
uv run python scripts/fetch_kit_msdm.py --verify-only          # 手元の KIT MSDM を再照合
```

`tests/test_s3_sync.py` は次を機械的に見ている。

* S3 のキーが `raw_data/` の既存の構成にそのまま落ちること
* **上げる場所と取りに行く場所が一致すること** (`key_for` が `dest_for` の逆であること、
  上げてから別の場所へ取り込むと木も中身も復元されること)
* 判定に関わるモジュール (`features` / `detectors` / `sideslip` / `scoring` /
  `signals` / `pipeline` / `parallel` / `sources`) と `screen_sideslip.py` が
  `s3_sync` も `boto3` も参照していないこと
* `boto3` が必須依存になっていないこと
* 静的な鍵を拒み、`.env` の鍵を見つけたら止まること
* 既にあるファイルを取り直さない / 上げ直さないこと、失敗時に `.part` を残さないこと
* `.DS_Store` や `*.part` をバケットへ上げないこと

`tests/test_kit_msdm_fetch.py` は次を見ている。

* 配布物の MD5 / 大きさの不一致を捕まえること
* BagIt の 44 件から 1 件でも書き換われば気付くこと
* 絶対パス・`..`・シンボリックリンク・デバイスを含む tar を**1 件も書かずに**拒むこと
* `requests` を `download()` の中でしか import しないこと
  (`--tar` / `--verify-only` で通信の準備すら起きない)
* User-Agent にホスト名も利用者名も載せないこと、認証情報を送らないこと
