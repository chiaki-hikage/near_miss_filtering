# near_miss_filtering

走行データ (CAN + 映像) からヒヤリハット候補となる区間を抽出する。

大量の走行ログから確認すべき区間を数 % まで絞り込み、最後に映像 AI (VLM) で
確認する。特徴抽出に使うのは CAN のみで、映像は VLM 段と目視確認にだけ使う。

---

## ドキュメント

| 文書 | 内容 |
|---|---|
| [docs/pipeline_overview.md](docs/pipeline_overview.md) | 全体処理フローと L1 / L2 / L3 の役割 |
| [docs/io_spec.md](docs/io_spec.md) | 入出力仕様。ファイル形式、列、単位、時刻、欠損、ID |
| [docs/event_detection_spec.md](docs/event_detection_spec.md) | イベント判定仕様。分類、特徴量、判定条件、閾値 |
| [docs/l1_filter.md](docs/l1_filter.md) | L1 軽量フィルタの設計と根拠 |
| [docs/sideslip_filter.md](docs/sideslip_filter.md) | 横滑り候補の抽出、スリップの等級 |
| [docs/environment.md](docs/environment.md) | 実行環境の構築 |
| [docs/comma2k19_data.md](docs/comma2k19_data.md) | comma2k19 の構成、単位・周期、注意点 |
| [docs/comma_car_segments.md](docs/comma_car_segments.md) | commaCarSegments の構成と整合性の確認結果 |
| [docs/signals_rav4.md](docs/signals_rav4.md) | 信号ごとの検証結果、単軌道モデルの諸元 |
| [docs/near_miss_filters.md](docs/near_miss_filters.md) | 各イベントの検出条件 (詳細) |
| [docs/vlm_review_poc.md](docs/vlm_review_poc.md) | VLM 判定レイヤの設計 |
| [docs/datasets.md](docs/datasets.md) | データセットの比較 |
| [docs/kit_msdm.md](docs/kit_msdm.md) | 横滑り検証用の閉鎖路データ |

---

## 処理の流れ

```
入力 (rlog.zst / raw_can)
  → CAN 前処理     車種設定による復号と符号補正
  → L1 軽量フィルタ 生 CAN の 2 秒窓 max/min だけで走行時間を絞る (通過率 5.8%)
  → L2 特徴量生成   20 Hz グリッドへ再サンプル → 平滑化 → 特徴量 約 50 列
  → L3 イベント判定 24 種の閾値・共起・系列ルール
  → 候補統合        時間方向にまとめ、severity と優先度規則で絞る
  → 動画切り出し    候補の時刻からフレームを JPEG に展開
  → VLM 判定        構造化出力で risky / hazard_type / evidence を得る
```

L1 と L2 と L3 は目的が違う。

| | L1 | L2 | L3 |
|---|---|---|---|
| 目的 | 走行時間を捨てる | 判定できる量を作る | 事象を特定する |
| 平滑化・時刻同期 | しない | する | (L2 の結果を使う) |
| 落とすもの | 走行時間の 94% | 落とさない | 事象でない時間 |

詳しくは [docs/pipeline_overview.md](docs/pipeline_overview.md)。

---

## 対象範囲

| データセット | 車種 | 規模 |
|---|---|---|
| comma2k19 | Toyota RAV4 2017 (Chunk 1–2) | 382 セグメント / 6.4 時間。人手ラベル 100 件 (risky 12) |
| commaCarSegments | Toyota RAV4 TSS2 / Corolla TSS2 / Prius / Highlander TSS2 | 35,226 セグメント / 587 時間。うち 8,000 (133 時間) を L3 までスクリーニング済 |
| KIT MSDM | — | 横滑りフィルタの検証用 (β の実測がある閉鎖路データ) |

Honda Civic (comma2k19 Chunk 3–10) は DBC の当て直しと符号の再検証が済むまで
対象外。`configs/vehicles/` に該当する dongle / 車種キーが無いものは
`skipped:unknown_vehicle` として記録される。

どちらのデータセットにも UDS / OBD-II の診断通信は含まれない。

---

## ディレクトリ構成

```
configs/
  detection.yaml          閾値・窓長・スコア設定。しきい値はすべてここに置く
  priority.yaml           高優先度候補の条件 (L3 の後段)
  vlm.yaml                VLM 判定レイヤの設定
  vehicles/*.yaml         車種ごとの CAN 定義・符号補正・車両諸元

src/near_miss/
  io/canonical.py         L0  データセットに依らない入力表現
  io/comma2k19.py         L0  comma2k19 の読み出し
  io/comma_car_segments.py L0 commaCarSegments の一覧・取得・読み出し
  io/rlog.py              L0  openpilot rlog (capnp + zstd) の読み出し
  io/can_decode.py        L0  DBC (Motorola) ビット抽出
  io/can_radar.py         L0  生 CAN からレーダトラックを組み立てる
  io/s3_sync.py           L0  S3 経由でのデータ取り込み (IAM Role を使う)
  sources.py                  データセットの差をここ 1 か所に閉じ込める
  l1.py                   L1  軽量フィルタの固定条件
  signals.py              L2  一様グリッドへの再サンプルと欠測の扱い
  features.py             L2  特徴量。ここから下は車種にもデータセットにも依存しない
  detectors.py            L3  閾値・共起・系列によるイベント検出
  direct_slip.py          L3  スリップの直接証拠 (モデルを通さない枝)
  sideslip.py             L3  横滑り角 β による 2 段フィルタ
  scoring.py              L4  候補区間の統合とスコア付け
  pipeline.py                 供給元を受け取って候補抽出まで通す
  vlm/                    L5  映像 AI 判定レイヤ (windows / frames / prompt /
                              schema / adapters / runner / scoring)

scripts/                      実行用スクリプト (下記)
docs/                         仕様と検証結果
tests/                        単体試験
```

### 層の分け方

車種依存の CAN ID / ビット定義を検出側に持ち込まない。

```
生 CAN → RawCanFrames → 正規化車両信号 (SegmentData) → 一様グリッド → 特徴量 → 検出
         ^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
         データセット固有  車種設定 (configs/vehicles/*.yaml) が決める
```

`features.py` 以降はデータセットも車種も知らない。データセットを足すときは
`sources.py` に `SegmentSource` を返す関数を 1 つ書く。

---

## 実行方法

### 準備

```bash
# 依存を入れる。Python も uv が用意する (.python-version で 3.10 に固定)
uv sync --extra viz --extra dev

# 環境が整っているかを確認する
uv run python scripts/validate/check_env.py --data

# capnp スキーマを取る (commaCarSegments を使う場合、最初の一度だけ)
uv run python scripts/fetch/fetch_cereal_schema.py
```

環境構築の詳細は [docs/environment.md](docs/environment.md)。

### データの取得

**取得前に必ず `--dry-run` で量を確かめる。** commaCarSegments は 1 セグメント
平均 1.38 MB (RAV4 TSS2 全 11,387 件で 15.7 GB)。

```bash
# 車種の一覧とセグメント数
uv run python scripts/fetch/fetch_car_segments.py --list

# 取得量の確認 → 取得
uv run python scripts/fetch/fetch_car_segments.py TOYOTA_RAV4_TSS2 --routes 3 --per-route 5 --dry-run
uv run python scripts/fetch/fetch_car_segments.py TOYOTA_RAV4_TSS2 --routes 3 --per-route 5

# comma2k19 を小さく試す (64 セグメント / 382 MB、映像なし)
uv run python scripts/fetch/fetch_demo_dataset.py --out data/comma2k19_demo
```

S3 経由で取り込む場合は `fetch_from_s3.py` / `upload_to_s3.py`。
認証は IAM Role を使い、**鍵はリポジトリにもインスタンスにも置かない**。

### 抽出

```bash
# L3 まで (comma2k19)
uv run python scripts/detect/run_detection.py raw_data/Chunk_1 --out out/chunk1

# L3 まで (commaCarSegments を数千セグメント。取得と処理を重ねて流す)
uv run python scripts/detect/screen_segments.py TOYOTA_RAV4_TSS2 --limit 2000 --dry-run
uv run python scripts/detect/screen_segments.py TOYOTA_RAV4_TSS2 --limit 2000 --out out/screen --resume

# L1 を当てて段ごとの残存を見る
uv run python scripts/l1/l1_l3_chain.py ccs --platform TOYOTA_RAV4_TSS2 \
    --screen "out/screen_{platform}_2k_v4" --out out/l1_l3/ccs

# 優先度で絞る
uv run python scripts/detect/filter_priority.py out/chunk1/candidates.csv --out out/priority.csv

# 横滑り 2 段フィルタを 1 コマンドで動かす (30 セグメント / 約 30 秒)
./scripts/demo_sideslip.sh
```

### VLM 判定

```bash
uv run python scripts/vlm/build_vlm_inputs.py ...   # GPU 不要。リクエスト JSONL を作る
uv run python scripts/vlm/run_vlm_review.py ...     # GPU が要る (--backend echo なら不要)
uv run python scripts/vlm/score_vlm_review.py ...   # GPU 不要。人手ラベルと突き合わせる
```

### 確認・検証

```bash
# 動画確認用のシートを作る
uv run python scripts/review/make_review_list.py out/chunk1 raw_data/Chunk_1 --top 30

# 1 セグメントの時系列を見る
uv run python scripts/review/plot_segment.py raw_data/Chunk_1 -s "b0c9d2329ad1606b|2018-08-17--14-55-39/1"
uv run python scripts/review/plot_segment.py --list-panels

# 候補が 0 件のときの切り分け
uv run python scripts/review/inspect_segment.py raw_data/Chunk_1 --index 40 --dump-csv out/ts.csv

# 車種設定の前提が崩れていないか確かめる (新しいデータを扱う前に必ず)
uv run python scripts/validate/validate_signals.py raw_data/Chunk_1 --max-segments 5

# イベント判定条件の一覧を設定から書き出す
uv run python scripts/report/dump_event_rules.py --markdown

# 試験
uv run pytest -q
```

### 主なスクリプト

| 分類 | スクリプト |
|---|---|
| 取得 | `fetch_car_segments` / `fetch_demo_dataset` / `fetch_cereal_schema` / `fetch_from_s3` / `upload_to_s3` / `fetch_kit_msdm` |
| 抽出 | `run_detection` / `screen_segments` / `screen_sideslip` / `filter_priority` |
| L1 | `l1_apply_ccs` / `l1_l3_chain` / `l1_branch_contrib` / `l1_round_thresholds` / `l1_trigger_search` / `raw_can_filter` |
| 集計 | `report_chain` / `report_final` / `report_pipeline` / `dump_event_rules` / `summarize_features` |
| VLM | `build_vlm_inputs` / `run_vlm_review` / `score_vlm_review` / `make_review_video` / `list_required_segments` |
| 確認 | `plot_segment` / `plot_slip_presentation` / `inspect_segment` / `make_review_list` / `pick_cases` |
| 検証 | `validate_signals` / `validate_slip_flags` / `validate_sideslip_filter` / `check_signal_parity` / `calibrate_accel_sign` / `check_env` |

---

## 出力

| ファイル | 内容 |
|---|---|
| `candidates.csv` | 確認単位の区間。`severity` 降順。区間内の特徴量の代表値を同じ行に持つ |
| `events.csv` | 個々の検出。`trigger_rule` にどの条件で拾ったかがそのまま入る |
| `segments.csv` | 走査したセグメントと処理状況 |
| `run_meta.json` | 実行条件と `config_hash` |
| `labels.csv` | 目視判定の記録。手作業の結果なので上書きしない |
| `review.csv` / `review.md` | 動画確認用。セグメント内の相対秒とフレーム番号つき |

列の意味・単位・時刻の扱いは [docs/io_spec.md](docs/io_spec.md)。

`config_hash` は検出設定と車種設定から作る。閾値を変えて回した結果は
ハッシュが変わるので、混ざっても後から区別できる。

---

## 実装上の決めごと

### データの扱い

生の受信時刻で微分しない。CAN の受信間隔は最小 0.18 ms まで詰まる。
輪速の分解能 0.01 km/h と組み合わさると、見かけの加速度が 100 m/s² 規模まで
振れる。必ず一様グリッド (既定 20 Hz) へ載せてから微分する。

欠測は埋めない。受信間隔が `max_gap_s` を超えた区間は NaN のままにする。
閾値判定は NaN で発火しないので、「条件を満たさなかった」ではなく
「判定できなかった」として下流に伝わる。

同じ CAN ID が複数のバスに別内容で流れる。車種設定では必ずバス番号を指定する。
実測で `0x224` はバス 0 が BRAKE_MODULE、バス 1 は別メッセージだった。
送信フレーム (`src` の 0x80) も常に除外する。

符号は推測で決めない。車種設定の `sign` はすべて実測で確かめた値で、根拠は
[docs/signals_rav4.md](docs/signals_rav4.md) にある。`validate_signals.py` で
いつでも再確認できる。

加速度センサの符号は個体ごとに違う。同じ車種キーでも `accel_x` / `accel_y` の
向きが逆の車両がある (Highlander TSS2 は 33 台中 21 台)。`ax` と `ay` は必ず一緒に
反転する。判定は `calibrate_accel_sign.py` が行う。

openpilot の介入区間を捨てない。制御フレームの有無を `op_tx` として持ち、
候補行に `op_tx_mean` として付ける。人間の運転挙動でない区間を後から選り分け
られるようにするためで、抽出時には除外しない。

### 判定の設計

閾値をコードに埋め込まない。すべて `configs/` に置く。

縦加速度の主系列は車速の微分。CAN の `ACCEL_X` はスケールの妥当性が確認
できていないため、`ax_can_mps2` として並べて持ち、差分を `ax_residual_mps2` に残す。

横加速度の主系列は `ay_kin = 車速 × ヨーレート`。`YAW_RATE` のスケールは
`global_pose` を基準に検証済み (回帰係数 0.988)。`ACCEL_Y` は路面カントの寄与と
スケール誤差を分離できないため主系列にしない。

車両の応答は物理モデルと突き合わせる。線形単軌道モデルで舵角からヨーレートを
予測し、残差を標準偏差で割った σ 単位で異常を見る。

単独の閾値超過より、並びと組み合わせ。回避してから制動する、割り込みに制動が
伴う、といった順序と共起を `sequence` / `cooccurrence` で記述する。
件数を作るために閾値を緩めない。

危険の中身は「車間が短いこと」より「急に詰まったこと」。目視判定でリスクと
された車間逼迫はすべて他車の割り込みだった。先行車の距離が 1 サンプルで跳ぶことを
主判定にする。

正常な操作は、危険側と重なったときだけ格上げする。車線変更それ自体は日常的な
操作なので低い重み (0) で拾い、車間逼迫や急制動が前後 2 秒以内に重なったときだけ
`risky_lane_change` を立てる。

急操舵は 1 つの信号で判定しない。舵角センサ・ヨーレートセンサ・横加速度計の
3 系統がそろって立つことを条件にし、センサ間の時間ずれを 0.3 秒まで許容する。

安全系フラグが立った区間は L1 で無条件に通す。車両自身が危険と判断した区間なので、
こちらが閾値を決める余地がない。通過率への影響は +0.01 ポイント以下と小さく、
外すと低速・短時間の ABS 作動が落ちる。

---

## 未了・確認できていないこと

### 較正

- 閾値の較正は途中。人手判定 32 件で上位 10 位の適合率は 71%。
  再現率は測れていない (下位帯の網羅的な確認をしていない)
- `cut_in_candidate` 33 件のうち目視で確かめたのは 4 件
- L1 の正例は 9 件しかなく、閾値の細かい値は leave-one-out で不安定
  (3% で held-out 5〜6/9)。枝の構成 (ax 減速 + ay) だけが安定していた

### 検出条件

- `s_evasion` / `brake_and_steer` / `brake_after_evasion` は件数が極端に少なく、
  定義が未検証
- 回避系の定義は comma2k19 Chunk_1 で 0 件。他のデータでの動作確認が要る
- 曲線区間での先行車選択が横位置しきい値だけの割り切り
- 車線変更は運動の形からの候補に留まる (車線マーカの正解が無いため断定できない)

### 信号

- `BRAKE_PRESSURE` (0x224 bit 43) はビット位置が誤っており無効化してある
- `ACCEL_Y` のスケールが未確定 (路面カントの寄与と分離できない)
- comma2k19 では `op_tx` が全区間で 1 に張り付き、openpilot の作動判別ができていない
  (commaCarSegments では `pandaStates.controlsAllowed` から直接取れる)
- アクセル開度 (`0x2C1`) の復号率は TSS2 で 34.7%。`panic_brake_pedal` は
  3 台に 1 台でしか使えない
- TSS2 の `TURN_SIGNALS` は左右の対応が確定できず無効のまま (1 Hz で分解能も足りない)
- TSS2 の単軌道モデルは 15 分 / 3 車両ぶんの当てはめ。台数を増やす際は再当てはめが要る
- `yaw_residual_sigma` は車両諸元が要るため、RAV4 と RAV4 TSS2 でしか計算されない

### パイプライン

- 候補 ID / イベント ID が L3 に無い。区間の重なりで突き合わせている
- 候補全件を VLM に流す経路が無い。現在の VLM 入力は人手確認済み 32 件のみ
- VLM 判定を候補表へ書き戻す経路が無い。結果は JSONL に留まる
- commaCarSegments には映像が無い。映像 AI を当てられるのは現在 comma2k19 のみ
- VLM の性能は確定していない。positive が 8 件しかなく、指標は参考値
