# Zenseact Open Dataset (ZOD) の構成と、横風の解析に使う準備

2026-11-12 に調べた内容。**データはまだ取得していない。** ダウンロードの手順 (ユーザーが行う) は `docs/zod_download.md`。

調べたもの:
- 公式サイト https://zod.zenseact.com/ (トップ、download、sequences、drives)
- GitHub https://github.com/zenseact/zod の README
- 公式 SDK `zod` 0.8.0 (PyPI。MIT)。**wheel を作業用の一時フォルダに落として、ソースを読んだ**。プロジェクトには入れていない
- 論文 arXiv:2305.02008 (Alibeigi ほか、ICCV 2023)

「確認済み」は、上の資料かソースに書いてあることを指す。データで確かめたことは、まだ無い。

---

## 1. 全体

| 項目 | 内容 | 出典 |
|---|---|---|
| 利用条件 | CC BY-SA 4.0 (商用・非商用とも可)。利用の申請が要る (承認済み) | 公式サイト |
| 地域・期間 | 欧州 14 か国、2 年間 | 公式サイト・論文 |
| 配布 | Dropbox の共有フォルダ (申請者ごとにリンクが届く)。Academic Torrents でも配布とあるが、CLI の対応は未完成 | download ページ |
| SDK | `zod` 0.8.0、Python ≥ 3.9。CLI は `zod[cli]` (追加で `dropbox`、`typer`) | PyPI |

### 3 つの subset

| subset | 量 | 中身 | 車両データ | 注釈 |
|---|---|---|---|---|
| Frames | 100k 枚 (55.6 h 相当)、14 か国 | 画像 1 枚 + 前後のライダー・OXTS | **無い** (CLI の `--vehicle-data` は Sequences/Drives 用) | 物体・車線・標識・自車道路・路面状態 |
| Sequences | 1,473 本 × 20 s (8.2 h)、6 か国 | 全センサ 20 s | **ある** | 中央の 1 フレームだけ |
| Drives | 29 本、平均 3 分 (1〜5 分、計 1.5 h 超)、ヨーテボリとパリ | 全センサ数分 | **ある** | 無い |

- 論文によると、Drives は直線・カーブ・バンク・合流・ランプなどの道路の形と、市街地・郊外・高速道路を含むように選ばれている
- Frames のメタデータには天候 (`scraped_weather`)・道路の種別 (`road_type`) がある。Sequences のメタデータは開始・終了の時刻、国、車両、緯度経度だけ (SDK `data_classes/metadata.py`)

### mini と full

- `--version mini` は、Dropbox のフォルダの中で名前に `mini` を含むファイルだけを取る。**mini では `--no-lidar` などの絞り込みが全部無視される** (SDK `cli/download.py` の `_filter_entry`、CLI の表示「other settings are ignored for mini」)。mini の中身と容量は公開資料に無い
- `--version full` では、名前に `mini` を含むファイルを除き、絞り込みを当てる
- 読み込み側は `trainval-<subset>-<version>.json` で区別する (SDK `constants.py`)

## 2. センサと信号

| センサ | 周期 | 形式 | 出典 |
|---|---|---|---|
| 前方カメラ 1 台 | 10 Hz (公式サイトは 10.1 Hz) | 3848×2168、120° の魚眼。JPG (Sequences/Drives はぼかしの匿名化だけ) | 公式サイト・論文 |
| OXTS RT3000 (GNSS/IMU) | 100 Hz | HDF5。位置精度 0.01 m (RTK) | 同上 |
| 車両の操作信号 | 100 Hz | HDF5 (`vehicle_data`) | 同上 |
| 車両の IMU (量産品) | 50 Hz | 同上 | 同上 |
| 車両の衛星測位 | 1 Hz | 同上 | 同上 |
| ライダー (VLS128 + VLP16×2) | 約 10 Hz | `.npy` | 同上 |
| レーダー | 60 ms ごと | Sequences/Drives に 2025-01-27 に追加 | 公式サイト |

### 横風の解析に使える信号 (SDK のソースで確認。データでは未確認)

| 量 | 置き場所 (HDF5 の dataset) | 単位・説明 (SDK のコメントか dataset の名前) |
|---|---|---|
| 操舵角 | `vehicle_data`: `ego_vehicle_controls/steering_wheel_angle/angle/radians/value` | rad、「反時計回りが正」 |
| 操舵角速度 | `.../steering_wheel_angle/angle_rate/radians_per_second/value` | rad/s |
| **操舵トルク** | `ego_vehicle_controls/steer_wheel_torque/torque/newton_meters/value` | N·m。横風に当て舵で耐えるときの手がかりになりうる |
| 方向指示器 | `ego_vehicle_controls/turn_indicator_status/state` | 0 消灯、1 左、2 右。車線変更を除くのに使える |
| アクセル・ブレーキ | `acceleration_pedal/ratio/unitless/value` (0〜100)、`brake_pedal_pressed/...` | |
| 車両の横・縦の速度 | `ego_vehicle_data/lat_vel_data/...`、`lon_vel_data/...` | m/s |
| 車両の横・縦の加速度 | `ego_vehicle_data/lat_acc_data/...`、`lon_acc_data/...` | m/s² |
| 車両のロール・ピッチ角速度 | `ego_vehicle_data/roll_rate_data/...`、`pitch_rate_data/...` | rad/s |
| 車体の高さ・ピッチ | `ego_vehicle_data/body_height/...`、`body_pitch/...` | m、rad |
| 衛星測位 | `satellite/latposn/nanodegrees/value` など | ナノ度、速さ m/s、方位 deg、1 Hz |
| 車両データの時刻 | 各グループの `timestamp/nanoseconds/value` | epoch の ns |
| OXTS の速度 | `velForward`, `velLateral`, `velDown` | **横速度がある** → 横滑り角を直接求められる見込み |
| OXTS の加速度・角速度 | `accelerationX/Y/Z`, `angularRateX/Y/Z` | **ヨーレートは `angularRateZ`** |
| OXTS の位置・姿勢 | `posLat`, `posLon`, `poses` (N×4×4、最初の位置に対する相対) | |
| OXTS の時刻 | `timestamp` + `leapSeconds` | GPS 時刻の秒。SDK は 1980-01-06 起点 + `timestamp` + `leapSeconds` で UTC にする |
| カメラの時刻 | infos の各フレームの `time` | datetime |
| 校正 | `calibration` (json) | カメラ・ライダー・レーダーの車体座標での取付位置。論文の図では OXTS は後車軸の中心 |

注意 (データで確かめる前に決めないこと):
- **車両データの SDK のクラス (`EgoVehicleData`) にヨーレートが無い。** 論文は「量産品の IMU 50 Hz」と書くが、SDK が読むのはロール・ピッチの角速度と横・縦の速度・加速度だけ。**mini のデータで、HDF5 にもヨーレートが無いことを確かめた** (`docs/zod_signals.md` §2.1)
- **mini のデータで確かめたこと (2026-11-12)**: 符号・単位・周期・時刻の対応は `docs/zod_signals.md`。車両データは 20 s ではなく走行全体 (55〜98 分) が入っている。OXTS は 50 s (シーケンスの前 5 s から後 25 s)。OXTS は y 右・z 下でヨーレートは deg/s、車両データの横は左が正
- OXTS の軸の向き (`angularRateZ` の符号、`velDown` の扱い) と車両データの符号は、SDK の読み込み (`velocities = [velForward, velLateral, -velDown]`) と論文の図から推定はできるが、**データで確かめるまで決めない**
- 量産 IMU の 50 Hz と操作信号の 100 Hz が同じ時刻の列を使っているかは不明 (SDK は `ego_vehicle_data` と `ego_vehicle_controls` で別々の `timestamp` を読む)

## 3. 公式 CLI (`zod download`) の調査

`zod` 0.8.0 の `zod/cli/download.py` を読み、作業用の一時環境で `--help` と入力の流れを動かして確かめた (リンクは入れていない。ダミーの文字列を入れて確認を「n」で断り、通信が起きない範囲だけ)。

| 調べたこと | 結果 |
|---|---|
| 1. インストール | 公式は `pip install "zod[cli]"`。**`zod==0.8.0` に固定し、`dataclass-wizard==0.39.1`・`typer==0.15.4` と組み合わせる** (2026-11-12 に作業用の一時環境で、import・`--help`・対話式の入力まで確認)。版を固定しないと `dataclass-wizard` 1.0.0 が入り、`import zod` が落ちる (`JSONSerializable` が無い)。入れ方は `docs/zod_download.md` 手順 1 (`uv tool install`、プロジェクトの依存は変えない) |
| 2. 対話式の入力 | `--url`, `--output-dir`, `--subset`, `--version` を省くと、その順に聞かれる。続いて設定の一覧が表示され、「Download with the above settings? [y/N]」で確かめる (既定は N) |
| 3. Frames / Sequences / Drives | §1。Dropbox の中のフォルダは `single_frames`, `sequences`, `drives` |
| 4. mini と full | §1。mini では絞り込みが無視される |
| 5. 特定のシーケンス・ファイルだけ | **CLI にはシーケンスの ID を選ぶ機能が無い。** 選べるのは Dropbox のフォルダ内のファイル (アーカイブ) 単位で、名前に含まれる語 (`images`, `lidar`, `oxts`, `vehicle_data`, `radar`, `annotations`, `infos`, `blur`, `dnat`) での除外だけ。アーカイブが何個に分かれているかは、フォルダの一覧を見るまで分からない |
| 6. 車両データ・OXTS・前方カメラだけ | `--no-lidar --no-radar --no-annotations` (+ カメラも要らないなら `--no-images --no-blur`。`blur` の除外は名前に `images` を含むファイルにだけ当たるので `--no-images` だけでも結果は同じだが、明示する)。`infos` (各シーケンスのファイルの場所・メタデータ・校正) は SDK で読むのに要るので残す。カメラは前方 1 台しかない |
| 7. LiDAR・レーダーを除く | `--no-lidar`, `--no-radar` |
| 8. 事前に総容量を確かめる | **CLI ではできない。** `--dry-run` は「Would download <ファイル名> to <保存先>」を並べるだけで、大きさを出さない (中ではファイルの大きさを持っているが表示しない)。大きさは Dropbox の共有フォルダをブラウザで開けば見える (`docs/zod_download.md` 手順 2) |
| 9. 失敗したときの再開 | 再実行すると、`downloads/` の下のファイルの大きさを Dropbox の一覧の大きさと比べる。**同じなら取得と展開を飛ばす** (中身の hash は確かめない。展開の途中で止まっていたら `--extract-already-downloaded` が要る)。**小さければ、その位置から続きを要求して書き足す** (HTTP の `Range`。サーバーが続きから返したかは確かめず、最後に大きさが合わなければエラー)。大きければエラー (消して再実行)。続きから取れることは実際には試していない。1 回の実行の中では、接続が切れたとき (`ChunkedEncodingError`) だけ自動で続きを要求し直す。**1 つのファイルで失敗しても、ほかのファイルを続け、最後に「Download complete.」と表示する**。詳しくは `docs/zod_download.md` 手順 5 |
| 10. URL がどこに残るか | 下の表 |
| 11. `zod verify` | infos から、あるはずのファイルの一覧を作り、**あるかどうかだけ**を確かめる (大きさ・中身は見ない。レーダーは対象外)。取っていないストリームも infos に載っていれば 0% として出る。全部 100% のときだけ成功の表示で、部分取得では「Found xx.x% ... could be intentional」(項目ごとの割合の平均) になる。判断は項目ごとに行う。infos が無いと止まる (`docs/zod_download.md` 手順 7) |

### Dropbox の URL がどこに残るか (ソースで確認)

| 場所 | 残るか |
|---|---|
| シェルの履歴 | `--url=...` で渡すと残る。**対話式の入力で入れれば、履歴に残るのは `zod download ...` の行だけ** |
| 画面 (端末のスクロールバック) | **残る。** 対話式の入力は伏せ字にならない (typer の prompt で `hide_input` が無い)。さらに、確認の前に表示する設定の一覧に `url: <入れた URL>` が出る (ダミーの文字列で確認した) |
| CLI が書くファイル | 書かない。設定・キャッシュ・ログのファイルを作る処理はソースに無い。保存先に作るのは `downloads/<subset>/<ファイル名>_<hash 8 桁>` と展開したデータだけ |
| エラーの表示 | 一覧の取得に失敗したとき、Dropbox のエラーをそのまま表示する。中に URL が入るかはソースからは決められない |
| その他 | CLI には Zenseact のアプリの鍵が埋め込まれていて、利用者の Dropbox のログインは要らない |

### `-y` / `--no-confirm` について

公式の例では使われているが、確認を省くので**使わない**。`--dry-run` でも確認は出る。

## 4. 容量の見積もり

**ファイルの大きさは公開されていない。** Zenseact の Hugging Face (`Zenseact/ZOD`) は外からは見えず (404)、Academic Torrents の検索はブラウザの確認画面があり、取りにいかなかった。旧 SDK (`zenseact-github/development_kit`) には「全体で約 3.4 TB」とあるが、今の版 (レーダーの追加後) に合うかは不明。

下は **推定** (2026-11-12 に mini の 2 シーケンスの実測から掛け算したもの。アーカイブの圧縮は考えていない)。確かな値は、ユーザーが Dropbox の一覧で見る (`docs/zod_download.md` 手順 3)。

| 構成 | 取得の指定 (Sequences full) | 推定の大きさ (Sequences) | 推定の大きさ (Drives) | 推定の根拠 |
|---|---|---|---|---|
| A. 車両データ + OXTS | `--no-images --no-blur --no-lidar --no-radar --no-annotations` (infos は残る) | **約 20 GB** (OXTS 約 12.5 GB + 車両データ 7〜12 GB + json 約 0.3 GB) | 不明 (車両データは走行の長さによる) | **mini の実測** (1 シーケンスあたり OXTS 8.5 MB、車両データ 5.0〜8.3 MB、json 0.2 MB) × 1,473。展開後の大きさ |
| B. A + 前方カメラ | `--no-lidar --no-radar --no-annotations` | A + **約 110〜120 GB** | 不明 | mini の実測 (201 枚で 76〜83 MB) × 1,473。展開後の大きさ |
| C. B + 注釈 | `--no-lidar --no-radar` | B + 小 (注釈は 1 シーケンス 1 フレーム) | B と同じ (注釈が無い) | Sequences の注釈は中央の 1 フレームだけ |

- **必要なディスク**: 既定では、アーカイブ (`downloads/`) と展開したものが両方残るので、アーカイブの合計より多く要る。展開したものがアーカイブに比べてどれだけになるかは分からない (圧縮の有無も不明)。アーカイブを消すかは、展開と `zod verify` のあとに `du` で実際の大きさを見て決める (`--rm` は確かめる前に消すので使わない)
- この Mac の空きは 414 GB (2026-11-12、mini の取得前)。B を Sequences 全部で取ると、展開後で約 130〜140 GB (推定) に、アーカイブの分が加わる。mini では、アーカイブ (`mini.tar.gz`) 1.43 GB に対して展開後 2.0 GB だった (ライダーを含む)
- **2026-11-12 の訂正**: 最初の版の見積もり (A は数 GB、B のカメラは 300〜600 GB) は、周期と列の数からの概算で、mini の実測と大きく違った (OXTS は 1 シーケンス 8.5 MB で、50 s 分・73 個の dataset (精度・取付けなどの構造体を含む)、車両データは走行全体が入っている、JPG は 1 枚 0.4 MB)。上の表は実測に基づくが、2 シーケンスからの掛け算なので幅がある
- ライダーは mini で 1 シーケンス 0.92〜0.97 GB (1,473 本で約 1.4 TB、推定)
- **構成 A の容量は `docs/zod_full_plan.md` §2 で見直した** (展開後 18〜37 GB、作業中の最大 24〜74 GB。車両データが走行全体を含み、走行の長さに比例するため幅がある)
- **取得の単位**: アーカイブ単位。シーケンスを選んで取ることは CLI ではできない。カメラのアーカイブがシーケンスごと・まとまりごとに分かれていれば、`--dry-run` の一覧から一部だけを選ぶ方法を別に考える (公式の CLI にはその機能が無い)
- C の注釈 (中央の 1 フレームの車線など) は、時間の変化を見る横風の解析にはほとんど役に立たない。最初は取らない

### 勧める順番

1. Sequences full の **A** (車両データ + OXTS + infos)。小さく、1,473 本 × 20 s 全部の操舵・ヨーレート・横加速度・位置・UTC 時刻がそろう
2. 同じく Drives full の **A** (29 本、高速道路を含む長い区間)
3. 信号を確かめ (`scripts/zod/inspect_signals.py`)、横風の候補を絞ってから、カメラ (B) の取り方を決める

## 5. ダウンロード後の配置

既存の comma・NVIDIA のデータとは分ける。CLI の保存先 (`--output-dir`) を `raw_data/zod/` にし、CLI が作る構成のまま置く (`docs/project_structure.md` §3 の 2: 公開元から丸ごと取ったものは `raw_data/`)。`raw_data/` は git の管理外。

```
raw_data/zod/                         ← zod download の --output-dir
├── downloads/sequences/...           アーカイブ (消すかは展開・検証の後に容量を見て決める)
├── downloads/drives/...
├── trainval-sequences-full.json      infos (SDK はここから各ファイルの場所を読む)
├── trainval-drives-full.json
├── sequences/<6 桁の ID>/...         展開したデータ (中の構成は取得後に確かめる)
└── drives/<ID>/...
scripts/zod/                          ZOD 専用のコード (git 管理)
out/zod/<解析名>/                     ZOD の解析の出力 (git 管理外)
docs/zod.md, docs/zod_download.md    この文書と手順
```

## 6. ダウンロード後の解析の準備

### 環境

プロジェクトの環境 (`pyproject.toml`, `uv.lock`) は変えない。

- ダウンロード用: `uv tool install` で、プロジェクトと別の環境に CLI を入れる (`docs/zod_download.md`)
- 解析用: `uv run --no-project` で、その場限りの環境を作る

```bash
uv run --no-project --python 3.10 --with "zod==0.8.0" --with "dataclass-wizard==0.39.1" --with pandas \
    python scripts/zod/inspect_signals.py --root raw_data/zod --subset sequences --version full --n 20
```

### 最初に確かめること (横風の判定の前)

`scripts/zod/inspect_signals.py` (取得後の初回実行はまだ。合成の HDF5 で時刻と周期の計算だけ確かめた) が `out/zod/signal_inventory/<subset>_<version>/` に出す。

| 確かめること | 出力 |
|---|---|
| HDF5 の全 dataset の名前・形・型・属性 (単位が名前に入っている)・NaN の数。SDK が読まない列 (車両の IMU のヨーレートなど) があるか | `fields.csv` |
| ストリームごとの周期・時刻の範囲・抜け・逆行。OXTS (100 Hz)、車両データの各グループ (100 / 50 / 1 Hz)、カメラ (10 Hz) | `timing.csv`, `summary.json` |
| 各ストリームの始まりの、OXTS に対するずれ | `timing.csv` の `t0_minus_oxts_s` |
| 走行の時刻 (UTC)・国・車両・位置 | `metadata.csv` |

そのあとに、データで確かめる (この順):
1. 符号と軸: 曲がる区間で、操舵角・OXTS の `angularRateZ`・横加速度・横速度の符号の関係 (comma2k19 の符号の較正と同じ考え方。ただしコードは共通化しない)
2. 時刻の対応: 操舵角とヨーレートの遅れ、OXTS と車両データの時刻のずれ。カメラは、画面の横の流れとヨーレートの相互相関 (PhysicalAI-AV で使った方法、`out/physical_ai_av/video_review/README.md`)
3. 品質: 欠損・飽和・RTK の状態 (OXTS の衛星の情報の列があれば)
4. 横風の判定に入る前の材料: 高速・直進で、操舵トルクと横速度 (横滑り角) の分布

comma の解析のコードとの共通化・作り直しは、今回はしない。
