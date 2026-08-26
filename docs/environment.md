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

---

## 1. OS に依存するのはどこか

**判定・抽出の処理は OS に依存しない。** 依存するのは次の 3 点だけで、
いずれも 1 か所に閉じ込めてある。

| 依存するもの | 何が違うか | どこで吸収しているか |
|---|---|---|
| 日本語フォント | mac は Hiragino Sans、Linux は Noto Sans CJK JP など | [`src/near_miss/plotting.py`](../src/near_miss/plotting.py) |
| matplotlib のバックエンド | 画面の無い Linux では `Agg` が要る | 同上 (`plotting.setup()`) |
| 標準出力の文字コード | Linux は `LANG` 未設定だと ASCII になり、日本語を出した時点で落ちる | [`scripts/_bootstrap.py`](../scripts/_bootstrap.py) |

`src/near_miss/` の他のモジュール (`signals` / `features` / `detectors` /
`sideslip` / `scoring`) は `plotting.py` を **import していない**。
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

### 4.1 commaCarSegments (必須)

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

### 4.2 KIT MSDM (任意、強く推奨)

フィルタが**本物の横滑りを拾えるか**を確かめる唯一のデータ
([kit_msdm.md](kit_msdm.md))。RADAR4KIT から取得する。

* DOI 10.35097/44a91t97pmnha1k9 / CC BY-SA 4.0
* https://radar.kit.edu/radar/en/dataset/44a91t97pmnha1k9
* 展開後 `raw_data/kit_msdm/10.35097-44a91t97pmnha1k9/data/dataset/` に
  `*.mat` と `parameter.m` が並ぶ形にする

### 4.3 Mac からそのまま持っていく場合

再取得せずに済ませたいなら rsync でよい。**大小の区別に注意**
(Mac 側の綴りがそのまま移るので問題は起きないが、手で作り直さないこと)。

```bash
rsync -av --progress \
  raw_data/comma_car_segments raw_data/kit_msdm \
  ec2-user@<host>:~/near_miss_filtering/raw_data/
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
```

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

---

## 10. 変更したときに崩れていないか

```bash
uv run python -m pytest -q                                    # 115 件
uv run python scripts/validate_sideslip_filter.py --kind dynamic --min-speed 3
./scripts/demo_sideslip.sh -n 30 --no-fetch
```

この 3 つが通れば、環境と判定の両方が保たれている。
