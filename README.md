# near_miss_filtering

走行データからヒヤリハット候補となる区間を抽出する。
初期検証は comma2k19 (Toyota RAV4, CAN のみ) で行い、
同じ検出処理を commaCarSegments (Toyota RAV4 TSS2) へ広げている。

## 現在の対象範囲

| データセット | 車種 | 規模 | 状態 |
|---|---|---|---|
| comma2k19 | Toyota RAV4 2017 (dongle `b0c9d2329ad1606b`, Chunk 1-2) | Chunk_1 で 188 セグメント (3.1 時間) | 目視判定 32 件つきで較正中 |
| commaCarSegments | Toyota RAV4 TSS2 (`TOYOTA_RAV4_TSS2`) | 11,387 セグメント。うち 196 (3.3 時間) をスクリーニング済 | 信号の整合性を確認済。高緊急度イベントは 0 件 |

使う信号は CAN のみ。IMU / GNSS / 映像は特徴抽出に使わない。
どちらのデータセットにも UDS / OBD-II の診断通信は含まれない。

Honda Civic (comma2k19 Chunk 3-10) は DBC の当て直しと符号の再検証が済むまで対象外。
`configs/vehicles/` に該当する dongle / 車種キーが無いものは
`skipped:unknown_vehicle` として記録される。

## 構成

```
configs/
  detection.yaml                 閾値・窓長・スコア設定。しきい値はすべてここ
  vehicles/toyota_rav4.yaml      RAV4 2017 (comma2k19) の CAN 定義と符号補正
  vehicles/toyota_rav4_tss2.yaml RAV4 TSS2 (commaCarSegments) の CAN 定義とレーダ定義
src/near_miss/
  io/canonical.py          L0  データセットに依らない入力表現 (ここから下が共通)
  io/comma2k19.py          L0  comma2k19 の読み出し
  io/comma_car_segments.py L0  commaCarSegments の一覧・取得・読み出し
  io/rlog.py               L0  openpilot rlog (capnp + zstd) の読み出し
  io/can_decode.py         L0  DBC (Motorola) ビット抽出。CAN ID は持たない
  io/can_radar.py          L0  生 CAN からレーダトラックを組み立てる
  sources.py                   データセットの差をここ 1 か所に閉じ込める
  signals.py               L1  一様グリッドへの再サンプルと欠測の扱い
  features.py              L2  特徴量。ここから下は車種にもデータセットにも依存しない
  detectors.py             L3  閾値と継続時間によるイベント検出
  scoring.py               L4  候補区間の統合とスコア付け
  pipeline.py                  供給元を受け取って候補抽出まで通す
scripts/
  fetch_demo_dataset.py    comma2k19 の demo split を取得して展開する
  fetch_cereal_schema.py   rlog を読むための capnp スキーマを取得する
  fetch_car_segments.py    commaCarSegments の一覧表示とセグメント取得
  check_signal_parity.py   2 つのデータセットの信号・単位・周期の整合を確認する
  screen_segments.py       数百〜数千セグメントのスクリーニング (取得と処理を重ねる)
  run_detection.py         抽出の実行 (--dataset で供給元を選ぶ)
  make_review_list.py      候補を動画確認できる形に並べ直す
  pick_cases.py            類型ごとに代表候補を選び、プロットまで通す
  plot_segment.py          1 セグメントの時系列を描く (--dataset / --context / --focus)
  inspect_segment.py       1 セグメントの中間結果と閾値までの余裕を見る
  validate_signals.py      車種設定の符号・前提を実データで検証する
  screen_sideslip.py       横滑り候補の 2 段抽出を実データに通す
  calibrate_beta_noise.py  横滑り角に載るセンサ雑音を直進区間から実測する
  validate_sideslip_filter.py  横滑りフィルタの再現率を KIT MSDM で測る
  check_env.py             実行環境が整っているかを確かめる
  demo_sideslip.sh         横滑りフィルタを 1 コマンドで動かす
docs/
  environment.md         Mac / AWS EC2 (ARM Linux) での構築・実行・出力確認
  comma2k19_data.md      データセットの構成、単位・周期、取り扱い上の注意
  comma_car_segments.md  commaCarSegments の構成と、層の分け方・整合性の確認結果
  near_miss_filters.md   各ヒヤリハット候補の検出条件 (暫定)
  near_miss_filters_summary.md  イベント定義の一覧 (読むための要約)
  sideslip_filter.md     横滑り候補の 2 段抽出 — 設計、閾値の根拠、実データでの結果
  kit_msdm.md            横滑り物理の物差し (β の実測がある閉鎖路データ)
  datasets.md            データセットの比較表
  signals_rav4.md        信号ごとの検証結果、単軌道モデルの諸元、使わない信号の理由
```

### 層の分け方

車種依存の CAN ID / ビット定義を検出側に持ち込まないため、次のように分けてある。

```
生 CAN → RawCanFrames → 正規化車両信号 (SegmentData) → 一様グリッド → 特徴量 → 検出
         ^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
         データセット固有  車種設定 (configs/vehicles/*.yaml) が決める
```

`features.py` 以降はデータセットも車種も知らない。
データセットを足すときは `sources.py` に `SegmentSource` を返す関数を 1 つ書く。

## データの取得

comma2k19 は Hugging Face の [`commaai/comma2k19`](https://huggingface.co/datasets/commaai/comma2k19)
に 2 通りの形で置かれている。

| 置き場所 | 容量 | 中身 | 用途 |
|---|---|---|---|
| `data/demo-*.parquet` | 228 MB | 64 セグメント。映像なし。**raw_can まで入る** | 動作確認・閾値の較正 |
| `raw_data/Chunk_1.zip` | 8.7 GB | RAV4 の 1 チャンク。映像込みの完全版 | 本番の走査 |
| `raw_data/Chunk_2.zip` | 9.1 GB | RAV4 の残り | 同上 |

Chunk_3 以降は Honda Civic なので現在の対象外。
[academic torrents](http://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb)
にも同じものがあるが、Hugging Face のほうが再開と部分取得が楽。

### まず小さく試す (推奨)

demo split を取得して、Chunk_N.zip と同じディレクトリ構成に展開する。

```bash
uv run python scripts/fetch_demo_dataset.py --out data/comma2k19_demo
```

64 セグメント (64 分、RAV4、21 ドライブ) が 382 MB で展開される。
映像と raw_log.bz2 は含まれないが、CAN のみを使う現在の範囲では困らない。
目視確認用に preview.png は入る。

### チャンクを取得する

```bash
# huggingface-cli を一時的に用意して使う (環境を汚さない)
uv tool run --from huggingface_hub huggingface-cli \
    download commaai/comma2k19 raw_data/Chunk_1.zip \
    --repo-type dataset --local-dir data/

unzip data/raw_data/Chunk_1.zip -d data/
```

ディレクトリ名に `|` が入るため、exFAT / NTFS では展開に失敗する。
その場合は comma2k19 リポジトリの `utils/unzip_msft_fs.py` を使う。

映像が容量の 7 割を占める (1 セグメントあたり video.hevc 36 MB に対し
processed_log は 6.8 MB)。CAN しか使わないので、展開後に video.hevc を
消せば 10 GB が 1.5 GB 程度になる。

## commaCarSegments を使う

車種キーで分類された約 40 万セグメントの CAN ログ。映像と復号済み信号は無く、
`rlog.zst` (openpilot のログ) だけが入っている。詳細は
[docs/comma_car_segments.md](docs/comma_car_segments.md)。

1 セグメントあたり平均 **1.38 MB** (実測)。取得前に必ず `--dry-run` で量を確かめる。

| セグメント数 | 概算 |
|---|---|
| 15 | 21 MB |
| 200 | 0.28 GB |
| 2,000 | 2.75 GB |
| 11,387 (RAV4 TSS2 全件) | 15.7 GB |

```bash
# capnp スキーマを取る (最初の一度だけ)
uv run python scripts/fetch_cereal_schema.py

# 車種の一覧とセグメント数を見る
uv run python scripts/fetch_car_segments.py --list

# 取得量を確かめてから取る (3 ルート × 連続 5 セグメント)
uv run python scripts/fetch_car_segments.py TOYOTA_RAV4_TSS2 --routes 3 --per-route 5 --dry-run
uv run python scripts/fetch_car_segments.py TOYOTA_RAV4_TSS2 --routes 3 --per-route 5

# comma2k19 との信号・単位・周期の整合性を確認する
uv run python scripts/check_signal_parity.py --platform TOYOTA_RAV4_TSS2

# 検出を回す
uv run python scripts/run_detection.py raw_data/comma_car_segments \
    --dataset comma_car_segments --platform TOYOTA_RAV4_TSS2 --out out/rav4_tss2

# 数百〜数千セグメントをスクリーニングする
uv run python scripts/screen_segments.py TOYOTA_RAV4_TSS2 --limit 500 --dry-run
uv run python scripts/screen_segments.py TOYOTA_RAV4_TSS2 --limit 500 --out out/screen \
    --discard-cache --resume

# 類型ごとに代表候補を選び、時系列プロットまで作る
uv run python scripts/pick_cases.py out/screen --top 3
uv run python scripts/pick_cases.py out/screen --top 2 --plot --span 20
```

`pick_cases.py` は候補を 5 つの類型に振り分ける。severity 順に並べると
件数の多い割り込み・車間で上位が埋まるため、先に「どの組み合わせを見たいか」で絞る。

| 類型 | 条件 |
|---|---|
| `abs_aeb` | 純正 AEB / PCS の作動、または ABS を伴う車間逼迫 |
| `yaw_counter` | ヨー応答の乖離 + 逆操舵 |
| `abs_wheel` | ABS 作動 または 輪速の異常 |
| `brake_steer` | 制動 + 操舵の回避 |
| `ttc_panic` | TTC < 3 秒での急制動 |

`--plot` は候補ごとに全体図と拡大図の 2 枚を作る。拡大の中心は
**候補の中で最も珍しいイベント**に合わせる。候補は前後 2 秒の余白を付けて
統合してあり、長いものは 50 秒に達するため、中点では事象が画面の外に出る。

## 使い方

環境の作り方 (Mac / AWS EC2 の両方) は [docs/environment.md](docs/environment.md)。

```bash
# 依存を入れる。Python も uv が用意する (.python-version で 3.10 に固定)
uv sync --extra viz --extra dev

# 環境が整っているかを確認する
uv run python scripts/check_env.py --data

# 横滑り 2 段フィルタを 1 コマンドで動かす (30 セグメント / 約 30 秒)
./scripts/demo_sideslip.sh

# 抽出を実行する (チャンクのディレクトリを指定)
uv run python scripts/run_detection.py raw_data/Chunk_1 --out out/chunk1

# 動画確認用のシートを作る
uv run python scripts/make_review_list.py out/chunk1 raw_data/Chunk_1 --top 30

# 1 セグメントの中身を見る (クリップ名は review.md の表記をそのまま貼れる)
uv run python scripts/plot_segment.py raw_data/Chunk_1 -s "b0c9d2329ad1606b|2018-08-17--14-55-39/1"
uv run python scripts/plot_segment.py --list-panels

# 1 セグメントを詳しく見る。候補が 0 件のときの切り分けに使う
uv run python scripts/inspect_segment.py /path/to/Chunk_1 --index 40 --dump-csv out/ts.csv

# 車種設定の前提が崩れていないか確かめる (新しいチャンクを扱う前に必ず)
uv run python scripts/validate_signals.py /path/to/Chunk_1 --max-segments 5

uv run python -m pytest -q
```

### 出力

- `candidates.csv` — 確認単位の区間。`severity` 降順。前後の文脈量 (最小 TTC、最小車間時間、最小縦加速度、ABS 作動、openpilot 送信割合など) を同じ行に持つ
- `events.csv` — 個々の検出。`trigger_rule` にどの条件で拾ったかがそのまま入る
- `segments.csv` — 走査したセグメントと処理状況
- `run_meta.json` — 実行条件と `config_hash`
- `labels.csv` — 目視判定の記録 (手作業の結果なので上書きしない)
- `review.csv` / `review.md` — 動画確認用。セグメント内の相対秒とフレーム番号、映像の場所つき

`config_hash` は検出設定と車種設定から作る。閾値を変えて回した結果は
ハッシュが変わるので、混ざっても後から区別できる。

## 設計上の決めごと

**縦加速度の主系列は車速の微分にしてある。** CAN の `ACCEL_X` はスケールファクタの
妥当性が確認できていないため、`ax_can_mps2` として並べて持ち、差分を
`ax_residual_mps2` に残すだけにしている。

**横加速度の主系列は `ay_kin = 車速 × ヨーレート`。** `YAW_RATE` のスケールは
`global_pose` を基準に検証済み (回帰係数 0.988)。`ACCEL_Y` は路面カントの寄与と
スケール誤差を分離できないため主系列にしない。

**車両の応答は物理モデルと突き合わせる。** 線形単軌道モデルで舵角から
ヨーレートを予測し (実測で SR = 16.75、Kus = 0.00235、R² = 0.973)、
残差を標準偏差で割った σ 単位で異常を見る。Chunk_1 の 3.1 時間では
残差の標準偏差が 0.341 deg/s で、車両は一貫して線形領域にあった。

**単独の閾値超過より、並びと組み合わせ。** 回避してから制動する、
割り込みに制動が伴う、といった順序と共起を `sequence` / `cooccurrence` で
記述する。回避系は Chunk_1 で 0 件だが、**件数を作るために閾値を緩めない**。

**危険の中身は「車間が短いこと」より「急に詰まったこと」。** 目視判定で
リスクとされた車間逼迫はすべて他車の割り込みだった。先行車の距離が
1 サンプルで跳ぶことを主判定にして `cut_in_candidate` を立てる。
トラック ID の変化は補助にとどめる (レーダが枠を使い回すため、同じ車両でも
ID が頻繁に入れ替わる)。

**正常な操作は、危険側と重なったときだけ格上げする。** 車線変更それ自体は
日常的な操作なので `lane_change_candidate` として低い重みで拾い、車間逼迫や
急制動が前後 2 秒以内に重なったときだけ `risky_lane_change` を立てる。

**急操舵は 1 つの信号で判定しない。** 舵角センサ・ヨーレートセンサ・横加速度計の
3 系統がそろって立つことを条件にし、センサ間の時間ずれを 0.3 秒まで許容する。
1 系統だけのノイズや欠損由来のスパイクは、他が裏付けないので通らない。
躍度 (jerk) は微分 2 回でノイズが支配的になるため、単独のイベント判定には使わず、
0.3 秒窓で均した補助特徴量として候補行に添える。

**先行車は「動いているもの」だけから選ぶ。** 路側の静止物は相対速度が −自車速に
なるため TTC が恒常的に小さく出て、しかも近いので実在の先行車より優先される。
Chunk_1 では `low_ttc` の検出 25 件すべてがこれだった。先行車の絶対速度で
足切りし、あわせて自車のヨーレートから求めた曲率で自車レーンの帯を曲げる。

**CAN はバスを指定して復号する。** 同じアドレスが複数のバスに別の内容で流れて
いることがある。実測で `0x224` はバス 0 が BRAKE_MODULE、バス 1 は別メッセージだった。
送信フレーム (`src` の 0x80) も常に除外する。

**生の受信時刻で微分しない。** CAN の受信間隔は最小 0.18 ms まで詰まる。
輪速の分解能 0.01 km/h と組み合わさると、見かけの加速度が 100 m/s^2 規模まで
振れる。必ず一様グリッド (既定 20 Hz) へ載せてから微分する。

**欠測は埋めない。** 受信間隔が `max_gap_s` を超えた区間は NaN のままにする。
閾値判定は NaN で発火しないので、「条件を満たさなかった」ではなく
「判定できなかった」として下流に伝わる。

**openpilot の介入区間を捨てない。** `raw_can/src` の 0x80 ビットが立った
制御フレーム (`STEERING_LKA` / `ACC_CONTROL`) の有無を `op_tx` として持ち、
候補行に `op_tx_mean` として付ける。人間の運転挙動ではない区間を
後から選り分けられるようにするためで、抽出時には除外しない。

**符号は推測で決めない。** 車種設定の `sign` はすべて実測で確かめた値で、
根拠は `docs/signals_rav4.md` にある。`scripts/validate_signals.py` で
いつでも再確認できる。

## 未了

- 閾値の較正は途中。32 件の目視判定 (`out/chunk1/labels.csv`) で上位 10 位の
  適合率は 71%。**再現率は測れていない** (下位帯の網羅的な確認をしていない)
- `cut_in_candidate` 33 件のうち目視で確かめたのは 4 件だけ
- `lateral_accel` が旋回除外で 0 件になった。イベントとして残すか判断が要る
- 回避系の 4 定義は Chunk_1 で 0 件。他のデータで動作確認が要る
- アクセル開度 (`GAS_PEDAL`) は両データセットで復号できたが、`panic_brake` はまだ縦加速度で代用したまま
- **ESC / TCS の介入フラグは DBC に存在しなかった** (あるのは無効化状態のみ)。スリップ判定は組めない
- `BRAKE_PRESSURE` (0x224 bit 43) はビット位置が誤っており無効化してある
- `ACCEL_Y` のスケールが未確定 (路面カントの寄与と分離できない)
- comma2k19 では `op_tx` が全区間で 1 に張り付き、openpilot の作動判別ができていない
  (commaCarSegments では `pandaStates.controlsAllowed` から直接取れる)
- `abs_active` / `aeb_active` / `wheel_speed_anomaly` / `s_evasion` / `brake_and_steer` /
  `brake_after_evasion` は、comma2k19 3.1 時間 + TSS2 3.3 時間で 1 件も出ていない。**定義が未検証**
- アクセル開度 (`0x2C1`) の復号率は TSS2 で 34.7%。`panic_brake_pedal` は 3 台に 1 台でしか使えない
- TSS2 の `TURN_SIGNALS` は左右の対応が確定できず無効のまま (1 Hz で分解能も足りない)
- TSS2 の単軌道モデルは 15 分 / 3 車両ぶんの当てはめで暫定。台数を増やす際は再当てはめが要る
- 曲線区間での先行車選択が横位置しきい値だけの割り切り
- 車線変更は運動の形からの候補に留まる (車線マーカの正解が無いため断定できない)
