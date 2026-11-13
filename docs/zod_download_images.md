# ZOD Sequences の前方カメラ画像の取得 (ユーザーが行う)

2026-11-13 (同日に改訂: アーカイブの名前と大きさを Academic Torrents の公開の一覧で確かめた)。横風候補の映像を確かめるため (`out/zod/front_video_review/README.md`)。

**取得はユーザー自身が、Claude Code とは別の端末で行う。** 守ることは `docs/zod_download.md` の「守ること」と同じ。

- 共有リンクは、公式の CLI の対話式の入力とブラウザのアドレス欄にだけ入れる
- 端末は Terminal.app などの VS Code の外のもの。出力を記録しない
- `-y` は使わない。終わったら画面とスクロールバックを消す

## 1. 取るもの

公式の配布 (https://zod.zenseact.com/download/) は 2 通り: Dropbox の共有フォルダ (申請で得たリンク、公式の CLI で取る) と Academic Torrents。どちらも**シーケンスを選んで取る方法は無い**。

Academic Torrents の Sequences の一覧 (https://academictorrents.com/details/95ece9c22470c4f7df51a82bcc3407cc038419be、2026-11-13 に確かめた):

| ファイル | 大きさ | 手元 |
|---|---|---|
| **`images_front_blur.tar.gz`** | **140.76 GB** | **無い (今回取る)** |
| `vehicle_data.tar.gz` | 9.60 GB | ある (同じ大きさ) |
| `oxts.tar.gz` | 1.64 GB | ある |
| `infos.tar.gz` | 126.90 MB | ある |
| `sequences_mini.tar.gz` | 1.43 GB | ある (`mini.tar.gz`) |
| `lidar_velodyne.tar.gz` | 919.66 GB | 取らない |
| `annotations.tar.gz` | 13.99 MB | 取らない |

**前方画像は 1 つのアーカイブ (140.76 GB) にまとまっている**

- dnat 版は Sequences には無い
- 前の見積もり (110〜120 GB) は mini からの推定で、小さかった
- 利用条件は CC BY-SA 4.0 (Torrent のページの記載)

**途中まで取るのでは足りない見込み**

- tar.gz は先頭から順にしか読めない
- 手元の `oxts.tar.gz` の中は、シーケンスの順が ID の順でない (001280, 001146, 000385, … の順)
- 画像のアーカイブも同じ作り方なら、必要な 6 シーケンスがどこにあるか分からないので、全体が要る

**必要なディスク**

- アーカイブ 140.76 GB (展開しない)
- 取り出す 6 シーケンスは約 0.5 GB
- 空きは 379 GB (2026-11-13)。展開すると、さらに約 140 GB が要るので、展開しない

## 1.1 Dropbox の分け方 (2026-11-13、ユーザーがブラウザで確かめた)

Dropbox の `sequences` フォルダでは、前方画像は 3 つに分かれている: `images_blur_000000_000490.tar.gz`、`images_blur_000491_000981.tar.gz`、`images_blur_000982_001472.tar.gz` (Torrent は 1 つ)。

- 名前のとおり ID の範囲で分かれているなら、今回の 6 シーケンス (000662, 000675, 000695, 000765, 000801, 000804) は**すべて `images_blur_000491_000981.tar.gz`** に入る (1 つあたり約 47 GB の見込み。正確な大きさはブラウザの表示)
- 公式の CLI は名前の語で除外するので、3 つのうち 1 つだけを選べない (3 つとも `images` と `blur` を含む)
- **その 1 ファイルだけを、ブラウザで共有リンクを開いてダウンロードする** (方法 C)。保存先は `raw_data/zod/downloads/sequences/`。展開しない

## 2. 方法 A: 公式の CLI (Dropbox、これまでと同じ。勧める)

### 手順 A-1. ブラウザで名前と大きさを見る

共有リンクをブラウザで開き、`sequences` フォルダの、名前に `images` を含むファイルの名前と大きさを確かめる。上の表と同じ (140.76 GB 前後の 1 ファイル) かを見る。Claude Code に渡してよいのは名前と大きさだけ。

### 手順 A-2. dry run

```bash
zod download --subset sequences --version full \
    --output-dir ~/work/near_miss_filtering/raw_data/zod \
    --no-extract \
    --no-lidar --no-radar --no-annotations --no-oxts --no-vehicle-data --no-infos \
    --dry-run
```

| オプション | 意味 |
|---|---|
| `--no-extract` | **展開しない**。必要なシーケンスだけを、取得の後に Claude Code が手元のファイルから取り出す |
| 画像のオプションを付けない | `--images`・`--blur` は既定で有効 |
| `--no-oxts` など | 取得済みのものを除く |

- 表示される `Would download ...` が、`images` を含む 1 ファイルだけであることを確かめる
- ほかのファイルが入っていたら、止めて Claude Code に伝える

### 手順 A-3. 取得

- `--dry-run` を外して実行する。設定の一覧で `dry_run: False`、`extract: False` を確かめてから `y`
- 置き場所は `raw_data/zod/downloads/sequences/<名前>_<番号>`
- 途中で止まっても、同じコマンドで続きから取れる (`docs/zod_download.md` 手順 5)
- Dropbox の帯域の制限 (CLI の文面「zod bandwidth limit」) で止まったら、翌日に同じコマンドで続ける

## 3. 方法 B: Academic Torrents (共有リンクが要らない)

- Torrent のファイルは公開。Dropbox のリンクを扱わない
- 注意点
  - **取得中は、ほかの利用者へのアップロードも行う** (BitTorrent の仕組み)
  - 取得元 (シード) の数は、ページからは分からなかった。遅い・止まる可能性がある
  - クライアントがこの Mac に無い (`aria2c` も `transmission` も無い)。入れるのはプロジェクトの外 (Homebrew など) で、`pyproject.toml` は変わらない
- 例 (aria2): まずファイルの番号を確かめ、画像の 1 ファイルだけを選ぶ

```bash
brew install aria2
cd ~/work/near_miss_filtering/raw_data/zod
aria2c --dir=torrent_meta "https://academictorrents.com/download/95ece9c22470c4f7df51a82bcc3407cc038419be.torrent"
aria2c --show-files torrent_meta/*.torrent          # images_front_blur.tar.gz の番号 (idx) を見る
aria2c --select-file=<idx> --seed-time=0 --dir=downloads/torrent torrent_meta/*.torrent
```

- `--seed-time=0`: 取り終えたらアップロードをやめる
- 置き場所は `raw_data/zod/downloads/torrent/sequences/images_front_blur.tar.gz` になる見込み

## 4. 取得の後に Claude Code に伝えるもの

```bash
ls -la ~/work/near_miss_filtering/raw_data/zod/downloads/sequences/ ~/work/near_miss_filtering/raw_data/zod/downloads/torrent/sequences/ 2>/dev/null
df -h ~/work/near_miss_filtering
```

この出力を伝える (リンクを含まない)。Claude Code は手元のアーカイブだけで次を行う。

1. 必要なシーケンスの画像を取り出す: `scripts/zod/extract_camera_images.py`
   - アーカイブを最初から最後まで 1 回読む。数十分の見込み
   - mini のアーカイブで動作を確かめた (201 枚、中身が一致)
   - 枚数を `info.json` と照らし、sha256 を記録する
2. 動画を作る: `scripts/zod/video_review.py --config configs/zod/front_video_review.yaml`
3. 動画を見て、道路の形・橋・遮るもの・車線の中の動き・周りの車をまとめる

アーカイブを残すか消すかは、取り出しの後にユーザーが決める。残せば、ほかのシーケンスもネットワークなしで取り出せる。
