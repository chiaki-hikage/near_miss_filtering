# ZOD のダウンロード手順 (ユーザーが行う)

**この手順はユーザー自身が、Claude Code とは別の端末で行う。** Claude Code は Dropbox の共有リンクを読み取らず、ダウンロードも実行しない。仕様と調べた結果は `docs/zod.md`。

この文書の CLI の動き (`zod` 0.8.0) は、公式のソース (`zod/cli/download.py`, `zod/cli/verify.py`) と、作業用の一時環境での `--help`・対話式の入力の確認 (リンクは入れず、ダミーの文字列を入れて確認を断った) で確かめた範囲だけを書く (2026-11-12)。

## 守ること

- 共有リンクは、**公式の CLI の対話式の入力とブラウザのアドレス欄にだけ**入れる。コマンドの引数 (`--url=...`)、ファイル、環境変数、Claude Code への入力には入れない
- **macOS の「ターミナル」(Terminal.app) など、VS Code と Claude Code の外の端末**を使う。VS Code の統合ターミナルは使わない
- 端末の出力を記録しない (`script`、`| tee`、tmux のログ、画面の共有をしない)
- CLI は、入力した URL を**画面にそのまま表示する** (入力の文字は伏せ字にならず、確認の前の設定の一覧にも `url:` の行が出る)。終わったら画面とスクロールバックを消す (手順 6)
- `-y` / `--no-confirm` は使わない。毎回、設定の一覧を見てから `y` を押す
- 最初は小さい構成 (車両データ + OXTS + infos) だけを取る

Claude Code に伝えてよいもの: ファイル名と大きさの一覧、`Would download ...` の行、`zod verify` の結果、`du` の結果。
**伝えないもの: 設定の一覧の部分 (`url:` の行がある)。エラーの文面は、中に `dropbox.com` の文字が無いかを確かめてから。**

---

## 手順 1. CLI を入れる (プロジェクトとは別の環境)

動作を確かめた版の組み合わせに固定して入れる。

```bash
uv tool install --python 3.10 "zod[cli]==0.8.0" \
    --with "dataclass-wizard==0.39.1" --with "typer==0.15.4"
zod --help
```

| パッケージ | 版 | 確かめたこと (2026-11-12、作業用の一時環境) |
|---|---|---|
| `zod` | 0.8.0 (固定) | `import zod`、`ZodSequences`・`ZodDrives`・CLI の読み込み、`zod download --help`・`zod verify --help`、対話式の入力と設定の一覧の表示 (確認は断った) |
| `dataclass-wizard` | 0.39.1 | 上と同じ。**版を固定しないと 1.0.0 が入り、`import zod` が落ちる** (`JSONSerializable` が無い) |
| `typer` | 0.15.4 | 上と同じ (click 8.1.8 が入る) |
| ほか | `dropbox` 12.1.0、`h5py` 3.16.0、`numpy` 2.2.6 | 依存として自動で入った版 |

- 実際のダウンロード (Dropbox との通信) は Claude Code では試していない
- `uv tool` は `~/.local/share/uv/tools/zod/` に専用の環境を作る。リポジトリの `pyproject.toml` と `uv.lock` は変わらない
- 後で消すときは `uv tool uninstall zod`

## 手順 2. 保存先を確かめる (Git と Dropbox の同期)

保存先は既存の comma・NVIDIA のデータと分けて、`raw_data/zod/` にする (`docs/zod.md` §5)。取る前に、次の 2 つを確かめる。

**(1) Git の管理外か**

```bash
cd ~/work/near_miss_filtering
git check-ignore -v raw_data/zod raw_data/zod/sequences/000000/dummy
git ls-files raw_data | wc -l
```

- 2 行とも `.gitignore:...:raw_data/` と表示されれば管理外。何も表示されない行があれば、取らずに止める
- `git ls-files` の結果が 0 であること (raw_data/ の下に Git 管理のファイルが無い)
- 2026-11-12 に Claude Code で確かめたときは、2 行とも `.gitignore:25:raw_data/`、0 件だった

**(2) Dropbox などの同期フォルダの中に無いか**

ZOD のデータ (数 GB〜数百 GB) が Dropbox に同期されると、容量を使い、ほかの機械にも複製される。

```bash
cd ~/work/near_miss_filtering && pwd -P
```

- 表示された実際の場所が `~/Dropbox`、`~/Library/CloudStorage/` (Dropbox・Google Drive など) の下でないこと
- 2026-11-12 に Claude Code で確かめたときは `/Users/hikage/work/near_miss_filtering` で、同期フォルダの外だった。リポジトリや `raw_data/` をシンボリックリンクで同期フォルダに移した場合は、もう一度確かめる
- Dropbox の共有リンクのフォルダを、**Dropbox のアプリで自分の Dropbox に追加しない** (追加すると同期の対象になりうる)。取得は CLI だけで行う

## 手順 3. ブラウザで、ファイルの名前と大きさを見る

CLI は事前に大きさを表示しないので、ブラウザで見る。

1. 受け取った共有リンクをブラウザで開く
2. `sequences` と `drives` のフォルダを開き、ファイルの名前と大きさを書き写す (名前に `vehicle_data`、`oxts`、`infos`、`images`、`blur`、`lidar`、`radar`、`annotations`、`mini` を含むもの)
3. Claude Code に渡してよいのは、この**名前と大きさの表だけ**

ブラウザの履歴にはリンクが残る。気になるなら、終わったあとに履歴から消す。

## 手順 4. 取るものを確かめる (dry run)

```bash
zod download --subset sequences --version full \
    --output-dir ~/work/near_miss_filtering/raw_data/zod \
    --no-images --no-blur --no-lidar --no-radar --no-annotations \
    --dry-run
```

- `--no-blur` は明示しておく。公式の実装では、`blur` での除外は名前に `images` を含むファイルにだけ当たるので、`--no-images` を付けていれば結果は同じ。付けておけば、名前の付け方が想定と違っても、ぼかし画像のアーカイブを取らない側に倒れる

1. 「What is the dropbox url ...?」と聞かれたら、共有リンクを貼り付けて Enter
2. 設定の一覧が出る。次の値になっているか確かめる

   | 項目 | 値 |
   |---|---|
   | `output_dir` | `.../near_miss_filtering/raw_data/zod` |
   | `dry_run` | True |
   | `version` | Version.FULL |
   | `images` / `blur` / `lidar` / `radar` / `annotations` | False |
   | `oxts` / `vehicle_data` / `infos` | True |

3. 「Download with the above settings? [y/N]」に `y`
4. `Would download /sequences/<ファイル名> to ...` の行が並ぶ。手順 3 の表と照らし、取るファイルと大きさの合計を出す。**この時点ではまだ何も取っていない**
5. 既に取り終えたファイルがあると、そのファイルは `Would download` ではなく `... already exists. Skipping download and extraction.` と出る

## 手順 5. 取る (構成 A: 車両データ + OXTS + infos)

手順 4 のコマンドから `--dry-run` を外す。

```bash
zod download --subset sequences --version full \
    --output-dir ~/work/near_miss_filtering/raw_data/zod \
    --no-images --no-blur --no-lidar --no-radar --no-annotations
```

- URL を入れ、設定の一覧で `dry_run: False` とほかの値 (手順 4 の表) を確かめてから `y`
- アーカイブは `raw_data/zod/downloads/sequences/<ファイル名>_<hash の先頭 8 桁>` に落ち、`raw_data/zod/` に展開される
- 最後の「Download complete.」の前まで、次の文面が出ていないか見る。**どれが出ても CLI はほかのファイルを続け、最後に「Download complete.」と表示する**
  - `Error downloading <名前>: ... Please retry` (そのファイルの取得に失敗)
  - `Error extracting <名前>: ... Please retry` (展開の途中で例外)
  - `Error extracting <アーカイブ> with tar.` (tar が 0 以外で終了。例外にならない)
  - `Error! File ... already exists and is larger than expected.`

Drives も同じ形で取る (`--subset drives` に変える)。

### 中断・失敗のあとの扱い (公式の実装で確かめた範囲)

再実行すると、CLI は Dropbox の一覧から各ファイルの大きさを取り直し、`downloads/` の下のファイルと比べて、次のように扱う。

| `downloads/` の下のファイル | CLI の動き | 注意 |
|---|---|---|
| **大きさが一覧と同じ** (取り終えている) | 取得を飛ばす。**展開も飛ばす** | 比べるのは大きさだけで、中身 (hash) は確かめない。展開の途中で止まっていた場合は、展開されないまま残る → `--extract-already-downloaded` を付けて再実行する |
| **一覧より小さい** (途中まで) | 今の大きさの位置から続きを要求し (HTTP の `Range`)、ファイルの後ろに書き足す | サーバーが続きから返したかを CLI は確かめない。最後に大きさが一覧と合わなければ `Error downloading ...` になる。続きから取れることは、実際には試していない |
| **一覧より大きい** | `Error! ... larger than expected` を出し、そのファイルは失敗になる | そのファイルを消してから再実行する |
| 無い | 最初から取る | |

- 1 回の実行の中で、Dropbox が接続を切ったとき (`ChunkedEncodingError`) は、CLI が同じファイルの続きを自動で要求し直す。それ以外の例外では、そのファイルをあきらめて次へ進む
- Dropbox の側でファイルが更新されて hash が変わると、ファイル名の末尾 8 桁が変わるので、古いファイルは使われず、最初から取り直しになる
- 一覧の取得そのものが失敗したときは、Dropbox のエラーが表示される。CLI の案内では、原因は URL の誤り、帯域の上限 (翌日に再実行)、通信の不調のどれか
- `Ctrl-C` で止めた場合の動きは、実装からは決められない (並列の取得で、例外の扱いが書かれていない)。止めたあとは、上の表に従って再実行し、手順 7 で確かめる

## 手順 6. 画面を消す

ダウンロードが終わったら (dry run の後も)、URL が表示された画面を消す。

```bash
clear
```

そのあと、Terminal.app ならメニューの「編集」→「スクロールバックを消去」(⌘K)。

シェルの履歴には URL が残らない (URL は対話式の入力で入れたため)。念のため `history | grep -i dropbox` で何も出ないことを確かめる。

## 手順 7. 確かめる

```bash
zod verify --dataset-root ~/work/near_miss_filtering/raw_data/zod --subset sequences --version full
```

`zod verify` の仕様 (`zod/cli/verify.py` で確認):

- `trainval-sequences-full.json` (infos) から、全シーケンスの、あるはずのファイルの一覧を作る。infos が無いと、確かめる前に `FileNotFoundError` で止まる
- 確かめるのは **ファイルがあるかどうかだけ**。大きさ・中身・壊れていないかは見ない
- 表は `GENERAL` (`calibration`, `ego_motion`, `metadata`, `oxts`、車両データの場所が infos にあれば `vehicle_data`)、`ANNOTATIONS`、`LIDAR`、`CAMERA` に分かれ、項目ごとに「あった割合 (%)」と「あるはずの数」を出す。**レーダーは確かめない**
- **取っていないストリームも、infos に載っていれば 0% として表に出る。** 全部 100% のときだけ「All files downloaded succesfully!」、全部 0% のときは「No files were found」、それ以外は黄色で「Found xx.x% of the dataset. This is could be intentional」と出る。xx.x は項目ごとの割合の平均
- そのため、構成 A では最後の行は黄色の「Found ...%」になるのが正しい。**判断は表の項目ごとに行う**

構成 A で見るところ:

| 項目 | 期待 |
|---|---|
| `GENERAL` の `oxts`, `vehicle_data` | 100% |
| `GENERAL` の `calibration`, `ego_motion`, `metadata` | 100% (infos のアーカイブに入っている想定。100% でなければ、どのアーカイブに入っているかを手順 3 の表で調べる) |
| `CAMERA`, `LIDAR`, `ANNOTATIONS` | 0% (取っていないので正しい) |

この結果は Claude Code に渡してよい (URL は含まれない)。

### 容量を確かめる

```bash
du -sh ~/work/near_miss_filtering/raw_data/zod/downloads ~/work/near_miss_filtering/raw_data/zod
df -h ~/work/near_miss_filtering
```

- 既定 (`--rm` 無し) では、アーカイブ (`downloads/`) と展開したデータの両方が残る
- アーカイブを消すかどうかは、展開と `zod verify` が済んだあとに、`du` で実際の大きさを見て決める。消すと、展開をやり直すにはもう一度取得が要る
- `--rm` は確かめる前にアーカイブを消すので、使わない

## 手順 8. Claude Code に伝える

「ZOD の Sequences (と Drives) の構成 A を `raw_data/zod/` に取った」と、手順 3 の表、手順 7 の `zod verify` と `du` の結果を伝える。Claude Code は `scripts/zod/inspect_signals.py` で信号の中身を調べる (ネットワークは使わない)。

前方カメラ (構成 B) は、信号を調べて候補を絞ってから、取り方を相談して決める。Sequences 全部のカメラは、mini の実測からの掛け算で展開後 約 110〜120 GB (推定、`docs/zod.md` §4)。構成 A は約 20 GB (推定)。
