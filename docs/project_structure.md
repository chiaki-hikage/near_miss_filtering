# プロジェクトの構成とファイル配置ルール

リポジトリの構成、各フォルダの役割、ファイルの置き場所の決まりをまとめる。2026-11-11 に調べて書き、同じ日の整理 (§6) を反映した。

- 「ルール」の節 (§3) には、`.gitignore`、`CLAUDE.md`、`docs/environment.md`、既存のコード、2026-11-11 に合意した整理の内容のように、決まっていることだけを書いた。
- 決まりから外れている箇所は「現状のずれ」(§4)、直し方の案と合意の状況は「整理の提案」(§5)、実施した内容は「変更の記録」(§6) に分けた。

---

## 1. 全体像

```
near_miss_filtering/
├── src/near_miss/   共通ライブラリ (検出処理の本体)              git 管理
├── scripts/         実行用スクリプト (役割別のサブフォルダ)       git 管理
├── configs/         閾値・車種定義・データセット設定・プロンプト   git 管理
├── schemas/         出力の JSON Schema                           git 管理
├── ref/             外部から受け取った参照表                      git 管理
├── tests/           単体試験 (pytest)。tests/data/ に基準値        git 管理
├── docs/            仕様・検証結果・データセットの説明            git 管理
├── experiments/     out/ の解析結果の索引 (index.csv)             git 管理
├── shared/          他の人に渡した資料 (図・動画・スライド)        git 管理
├── raw_data/        公開元から取得した元データ        71 GB      git 管理外
├── data/            選んで取った元データ・補助データ   3.0 GB     git 管理外
├── out/             解析の出力 (再生成できるもの)      2.6 GB     git 管理外 (人手ラベルだけ例外)
├── pyproject.toml / uv.lock   依存の正
├── requirements.txt           uv export で自動生成 (直接編集しない)
├── .python-version            Python 3.10 に固定
├── README.md                  入口。処理の流れと実行方法
└── CLAUDE.md                  プロジェクトの方針
```

容量は 2026-11-11 の `du -sh` の値。

---

## 2. 各フォルダの役割

### src/near_miss/ — 共通ライブラリ

検出処理の本体。層ごとにファイルを分けている (README の「層の分け方」)。

| 層 | ファイル | 役割 |
|---|---|---|
| L0 入力 | `io/*.py` | データセットごとの読み出し (comma2k19、commaCarSegments、comma1M、KIT MSDM、rlog、S3)、CAN の復号 |
| | `sources.py` | データセットの差をここ 1 か所に閉じ込める |
| L1 | `l1.py` | 生 CAN だけを使う軽量フィルタ |
| L2 | `signals.py`, `features.py`, `sign_calibration.py` | 一様グリッドへの再サンプル、特徴量、符号の補正 |
| L3 | `detectors.py`, `direct_slip.py`, `sideslip.py` | イベント検出 |
| L4 | `scoring.py` | 候補区間の統合とスコア |
| L5 | `vlm/` | 映像 AI (VLM) による判定 |
| 共通 | `pipeline.py`, `config.py`, `parallel.py`, `plotting.py`, `canonical_event.py` | 全体の通し、設定の読み込み、並列化、描画、出力形式 |

`features.py` から下は、データセットも車種も知らない作りにしている。

### scripts/ — 実行用スクリプト

`uv run python scripts/<役割>/<名前>.py` で、リポジトリの直下から実行する。各サブフォルダの `_bootstrap.py` が `src/` と `scripts/` の各サブフォルダ (1 段目) を `sys.path` に入れるので、サブフォルダをまたいだ import ができる。

| フォルダ | 役割 | 例 |
|---|---|---|
| `fetch/` | データの取得 (公開元、S3) | `fetch_car_segments.py`, `fetch_comma2k19_remote.py` |
| `l1/` | L1 フィルタの適用と評価 | `l1_l3_chain.py` |
| `detect/` | L3 までの抽出、優先度での絞り込み | `run_detection.py`, `screen_segments.py` |
| `validate/` | 信号の意味・符号・前提の検証 | `validate_signals.py`, `check_env.py` |
| `review/` | 人が確認するための一覧・図・切り出し | `make_review_list.py`, `plot_segment.py` |
| `report/` | 集計・段ごとの比較・API 用サンプルの出力・結果の索引 | `report_final.py`, `export_api_samples.py`, `build_experiment_index.py` |
| `vlm/` | VLM 判定の入力作成・実行・採点 | `build_vlm_inputs.py`, `run_vlm_review.py` |
| `vibration/` | 路面振動の解析 (フルカーモデル、ISO 2631、路面の特定) | `locate_candidates.py`, `fullcar.py` |
| `crosswind/` | 横風の解析。解析ごとのサブフォルダ (`xwind_search/`, `civic_v2/`, `c1m_search/` など 7 つ) | `xwind_search/rank.py` |

- `scripts/vibration/` の `fullcar.py`、`iso2631.py` などは、ほかのスクリプトと `tests/` から import される部品でもある。
- `scripts/crosswind/` のスクリプトは、各解析で実際に使ったものを内容を変えずに移したもの (§6)。解析ごとに同じ名前で中身の違うファイル (`windows.py`, `fit.py` など) があるので、サブフォルダを分けている。直下に `.py` は置かない (`sys.path` に入るため、ほかのスクリプトと名前がぶつかる)。詳しくは `scripts/crosswind/README.md`。

### configs/ — 設定

閾値や窓長はコードに直接書かず、ここに置く (CLAUDE.md)。

| 置き場所 | 内容 |
|---|---|
| `detection.yaml`, `priority.yaml` | L3 の閾値・窓長・スコア、高優先度の条件 |
| `vehicles/*.yaml` | 車種ごとの CAN 定義・符号補正・車両諸元 |
| `datasets/*.yaml` | データセットごとの取得・読み出しの設定 (comma1M、KIT MSDM、S3) |
| `vlm.yaml`, `vlm_road.yaml`, `prompts/*.md` | VLM 判定の設定とプロンプト |
| `vibration*.yaml` | 路面振動の各解析の設定 |
| `near_miss_canonical.yaml`, `near_miss_api_samples.yaml` | 統一形式の出力と API 用サンプルの設定。`near_miss_canonical.yaml` には、統一形式の元として使う `out/` の結果が列挙されている |
| `crosswind/honda_civic.yaml` | 横風の解析用に当てはめた Civic の諸元。`vehicles/` に入れると config_hash が変わるため、外に置いている |

### schemas/, ref/

- `schemas/near_miss_event.schema.json`: ヒヤリハットイベントの出力形式。`tests/test_near_miss_canonical.py` が使う。
- `ref/label_list.csv`: 外部から受け取ったラベル一覧。対応表は `docs/label_list_mapping.md`。

### tests/

pytest の単体試験。ファイル名は `test_<対象>.py`。試験の基準値は `tests/data/` に置き、git で管理する (2026-11-11 から。§6)。

### docs/

31 本ある。主なまとまりは次のとおり。

| まとまり | 文書 |
|---|---|
| 全体と仕様 | `pipeline_overview.md`, `io_spec.md`, `event_detection_spec.md`, `environment.md`, `project_structure.md` (この文書) |
| フィルタの設計 | `l1_filter.md`, `sideslip_filter.md`, `near_miss_filters.md`, `near_miss_filters_summary.md` |
| データセットと信号 | `datasets.md`, `comma2k19_data.md`, `comma_car_segments.md`, `comma1M.md`, `kit_msdm.md`, `signals_rav4.md` |
| 出力形式と API | `near_miss_canonical_schema.md`, `near_miss_api_data_spec.md`, `near_miss_api_response_draft.md`, `near_miss_data_inventory.md`, `label_list_mapping.md` |
| VLM | `vlm_review_poc.md`, `vlm_road_review.md` |
| 路面振動 | `vibration_*.md` (8 本) |
| 個別の検証 | `jerk_feasibility.md`, `crosswind_civic_feasibility.md` |

### experiments/ — 解析結果の索引

`out/` は git の管理外なので、どの結果をどのスクリプトで作り、今どれを使っているかを `experiments/index.csv` に記録して git で持つ。`out/` のファイルは動かさず、索引を作るだけにしている。

- 1 行 = `out/` 直下のフォルダ 1 つ (`out/crosswind/`・`out/vibration/` は 1 段下)、またはばらのファイルの名前のまとまり 1 つ。初版は 111 行
- 自動の列 (ファイル数・容量・更新日・README の有無・参照元・名前上の後続の版) は `scripts/report/build_experiment_index.py` が作る
- 手で書く列 (作成スクリプト・採用状況と、それぞれの根拠) は、スクリプトが書き換えない
- 列の意味と `status` の値は `experiments/README.md`

### shared/ — 他の人に渡した資料

発表や他チームへの提供に使った図・動画・スライド・PDF。git で管理している (202 MB)。

- `provided_to_api/`: API 側に渡した仕様・Schema・サンプル JSON
- `vibration/`: 路面振動の発表資料の図

### raw_data/ — 公開元から取得した元データ

公開元や S3 から、**丸ごと (チャンク・データセット単位で) 取得したもの**を、取得したときの構成のまま置く。git 管理外なので、clone しても中身は付いてこない (`docs/environment.md` §4)。

| フォルダ | 内容 |
|---|---|
| `Chunk_1/`, `Chunk_2/` | comma2k19 の RAV4 (Chunk_N.zip を展開したもの) |
| `comma_car_segments/` | commaCarSegments (`database.json` と `segments/`) |
| `comma1M/` | comma1M の自己位置推定の出力 (safetensors) |
| `kit_msdm/` | KIT MSDM の閉鎖路データ |

### data/ — 選んで取った元データと補助データ

git 管理外。次の 2 種類を置く。

- **選んで取った元データ**: 目的に合わせて一部だけを取得したもの。選んだ理由と取得の記録 (`selection.csv`、`fetch_manifest.csv` など) を一緒に置く
- **補助データ**: 元データを読む・解釈するための外部のスキーマ、較正の結果、取得用のキャッシュ

| フォルダ | 種類 | 内容 |
|---|---|---|
| `comma2k19_civic/`, `comma2k19_civic_video/` | 選んで取った元データ | comma2k19 Civic から選んだセグメントと映像 (`selection.csv` に選んだ理由) |
| `comma2k19_demo/` | 選んで取った元データ | comma2k19 の小さなデモ (64 セグメント) |
| `cereal/` | 補助データ | openpilot の capnp スキーマ (`fetch_cereal_schema.py` が取得) |
| `calibration/accel_sign/` | 補助データ | 車種ごとの加速度の符号の較正結果 |
| `_hf_cache/` | 補助データ | Hugging Face から取得したときのキャッシュ |

同じ comma2k19 でも、チャンクを丸ごと取った RAV4 は `raw_data/`、選んで取った Civic とデモは `data/` に置く。2026-11-11 時点の配置はこの定義に合っている。

### out/ — 解析の出力

スクリプトの出力先。git 管理外で、作り直せることが前提。ただし、作り直せない人手ラベルだけは git で持つ。

- `out/chunk1/labels.csv`
- `out/chunk1/vlm/labels_onset.csv`, `labels_onset.md`

`out/_backup/` には、整理の作業のときに取った写しを置く (2026-11-11 の横風スクリプトの写し)。

---

## 3. ファイル配置ルール (決まっていること)

### データ

1. 元データは上書きしない (CLAUDE.md「元データを上書きしない」)。
2. 公開元から丸ごと取得した元データは `raw_data/` に、取得したときの構成のまま置く。
3. 目的に合わせて一部だけを取得した元データと、補助データ (スキーマ・較正・キャッシュ) は `data/` に置く。選んで取ったものには、選んだ理由と取得の記録を一緒に置く。
4. 加工した結果は `out/` に出す。元データの横には書き出さない。
5. `raw_data/`, `data/`, `out/` は git に入れない。取得の手順は `scripts/fetch/` と `docs/environment.md` に残す。`.gitignore` ではリポジトリ直下に限って除外する (`/data/` のように先頭に `/` を付ける)。
6. 認証情報 (`.env`, `.aws/`) は置かない。S3 は IAM Role で使う。

### コードと設定

7. 何度も使う処理は `src/near_miss/` に関数として置く。データセットに固有の処理は `io/` と `sources.py` に閉じ込める。
8. 実行用のスクリプトは `scripts/<役割>/` に置き、`uv run` で実行する。**`out/` の下にスクリプトを置かない** (`out/` は git の管理外なので、手順が残らない)。
9. 1 つの解析のためのスクリプトがまとまってあるときは、`scripts/<テーマ>/<解析名>/` に置き、出力先 `out/<テーマ>/<解析名>/` と名前をそろえる (例: `scripts/crosswind/xwind_search/`)。`scripts/<テーマ>/` の直下には `.py` を置かない。
10. 閾値・窓長・車種の諸元は `configs/` に置き、コードに直接書かない。
11. 車種の定義は `configs/vehicles/<メーカー>_<車種>.yaml`。解析用に当てはめた別の諸元は、`config_hash` を変えないように `vehicles/` の外に置く (例: `configs/crosswind/`)。
12. 依存は `pyproject.toml` と `uv.lock` で管理し、`uv add` / `uv remove` で変える。`requirements.txt` は `uv export` で作る。
13. 試験の基準値は `tests/data/` に置き、git で管理する。

### 出力

14〜16 は文書に明記された決まりではなく、これまでの解析で守ってきた慣行である。

14. 出力先は `--out` で指定する。解析ごとに `out/<テーマ>/<解析名>/` のフォルダを作る (例: `out/crosswind/xwind_search/`, `out/vibration/locate/`)。
15. 既存の結果を変えずにやり直すときは、新しいフォルダを作る (`_v2`, `_v3` など)。古いフォルダは上書きしない。
16. 結論を書いた解析フォルダには `README.md` を置き、方法・結果・限界を残す。
17. 再生成できない人手ラベルは、`.gitignore` に例外を足して git で持つ。
18. `out/` に結果を足したら `scripts/report/build_experiment_index.py` を流し、`experiments/index.csv` の作成スクリプトと採用状況を根拠と一緒に書く。根拠が無いときは「未確認」のままにする。

### 移動と記録

19. ファイルを動かすときは、先に写し (`out/_backup/`) と sha256 を取り、移動の前後で内容が同じことを確かめる。
20. 過去の記録 (README・文書) に書かれた旧パスは書き換えない。旧パスと新パスの対応表を残し (例: `scripts/crosswind/path_map.csv`)、旧位置には移動先を示すファイル (`MOVED.md`) を置く。

### 文書と共有資料

21. 仕様と検証結果は `docs/` に置く。
22. 他の人に渡した資料は `shared/` に置く。

---

## 4. 現状のずれ

調べて分かった、ルールから外れている箇所と、対応の状況。

| # | 箇所 | 状況 | 影響 | 対応 |
|---|---|---|---|---|
| 1 | `out/crosswind/*/scripts/` | 横風の解析スクリプト 35 本が `out/` の下にあった | `out/` は git 管理外なので、このスクリプトは git に入っていなかった | **対応済み** (2026-11-11)。`scripts/crosswind/` へ移した |
| 2 | `tests/data/` | `.gitignore` の `data/` が先頭の `/` なしで書かれていたため、`tests/data/` も除外されていた | `tests/test_near_miss_canonical.py` が読む基準値が git に入っていなかった | **対応済み** (2026-11-11)。`/data/` に直した |
| 3 | `out/` 直下 | 99 項目あり、解析フォルダとばらの CSV・PNG が混ざっている。`_v2`〜`_v4` のどれが採用されたかの索引が無かった | どの結果を使えばよいか、フォルダ名だけでは分からない | **索引だけ対応済み** (2026-11-11)。`experiments/index.csv`。ファイルは動かしていない。未確認の行が 41 ある |
| 4 | `raw_data/` 直下 | 映像ファイル (`video.hevc`, `video_org.hevc`, `*_seg_video.mp4`)、`mpeg/`、変換スクリプト `hevc2mpeg.sh` がある | 元データと作業中のファイル・スクリプトが混ざっている | 承認待ち (提案 4) |
| 5 | `data/` と `raw_data/` の境目 | comma2k19 の Civic とデモは `data/`、RAV4 は `raw_data/` にある | どちらに置くかの基準が書かれていなかった | **対応済み** (2026-11-11)。役割を §2・§3 に書いた。今の配置は定義に合っているので、ファイルは動かしていない |
| 6 | `configs/` | 横風は `crosswind/` のサブフォルダ、路面振動は直下に `vibration_*.yaml` が 6 本 | 同じ種類の設定で置き方がそろっていない | 承認待ち (提案 6) |
| 7 | `scripts/` 直下 | `export_near_miss_canonical.py` と `demo_sideslip.sh` が役割別のフォルダに入っていない | 小さい問題 | 未提案 |
| 8 | `shared/` | 動画・pptx・PDF などのバイナリを git で持っている (202 MB) | リポジトリが重くなる。更新のたびに履歴が増える | 承認待ち (提案 7) |
| 9 | `data/comma2k19_civic/_selection/` | セグメントを選んだスクリプト (`pick_civic_segments.py`) と、風のデータを集めたスクリプト (`wind/collect.py`, `wind/remote_zip.py`) が `data/` の下にある (2026-11-11 に見つけた) | ずれ 1 と同じで、git に入っていない | 未提案。今回の整理の対象外 |

---

## 5. 整理の提案と合意の状況

2026-11-11 の合意。

| # | 提案 | 状況 |
|---|---|---|
| 1 | 横風のスクリプトを `scripts/crosswind/` へ移す (ずれ 1) | **実施済み**。旧パスを書いた README と文書は書き換えず、対応表を残した |
| 2 | `.gitignore` の `data/` を `/data/` に直し、`tests/data/` を git の対象にする (ずれ 2) | **実施済み** |
| 3 | 解析結果の索引を作る (ずれ 3)。当初案の `out/INDEX.md` (git 管理外) ではなく、git で管理する `experiments/index.csv` にする。`out/` のファイルは動かさない | **実施済み** (修正した案で) |
| 4 | `raw_data/` 直下の作業ファイルを動かす (ずれ 4) | **当面は実施しない**。参照関係・実行環境への影響・移動の利点を整理してから判断する |
| 5 | `raw_data/` と `data/` の役割を明文化する (ずれ 5)。ファイルは動かさない | **実施済み** (§2・§3) |
| 6 | 路面振動の設定を `configs/vibration/` にまとめる (ずれ 6) | **当面は実施しない**。提案 4 と同じ |
| 7 | `shared/` の大きなバイナリを git の外に出す (Git LFS か共有ドライブ) (ずれ 8) | **当面は実施しない**。提案 4 と同じ |

当初案 3 にあった「ばらのファイルをテーマごとのフォルダ (`out/chain/` など) にまとめる」は、今回は実施していない。索引ができたので、動かすかどうかは改めて判断する。

---

## 6. 変更の記録

### 2026-11-11 提案 1・2・3・5 の実施

**提案 1: 横風のスクリプトの移動**

- 対象: `out/crosswind/` の下の `.py` 35 本 (`*/scripts/` の下に 28 本、解析フォルダの直下に 7 本)
- 移動先: `scripts/crosswind/<解析名>/` (例: `out/crosswind/civic_v2/scripts/fit.py` → `scripts/crosswind/civic_v2/fit.py`)
- 保全: 移す前に `out/_backup/crosswind_scripts_2026-11-11.tar.gz` (35 本) と sha256 の一覧を取った。移した後の sha256 は 35 本とも一致した。**スクリプトの内容は変えていない**
- パスの影響: どのスクリプトも、入力のパスをリポジトリの直下からの相対パスで書いている (`__file__` を使うものは無い)。リポジトリの直下から実行すれば、置き場所が変わってもパスの解決は変わらない。ほかのスクリプトや `src/`・`tests/` から、移したスクリプトへの参照は無かった
- import の影響: `scripts/_bootstrap.py` は `scripts/crosswind/` を `sys.path` に入れるが、直下に `.py` を置いていないので、`fit.py` などがほかのスクリプトの import とぶつかることはない (`fit`・`windows`・`score` などの名前で import 先が変わらないことを確認した)
- 旧パスの記録: `out/crosswind/*/README.md` と `docs/crosswind_civic_feasibility.md` §11 の旧パスは書き換えていない。`docs/crosswind_civic_feasibility.md` §11 には移動の注記だけを足した。対応表は `scripts/crosswind/path_map.csv` (旧パス・新パス・sha256)。旧位置に `MOVED.md` / `SCRIPTS_MOVED.md` を置いた

**提案 2: `.gitignore`**

- `data/` → `/data/`。新たに git の対象になったのは `tests/data/near_miss_canonical_event_ids.json` (3 KB、event_id の基準値) だけ。直下の `data/` と `out/` は引き続き除外されている

**提案 3: 索引**

- `experiments/index.csv` (111 行)、`experiments/README.md`、`scripts/report/build_experiment_index.py` を足した
- 作成スクリプトは、`--out` の既定値・文書のコマンド・対応表から決めた。出力ファイル名が一致しただけのもの (58 行) は「推定:」と書いた
- 採用状況は、`configs/near_miss_canonical.yaml` の入力と `docs/near_miss_data_inventory.md` の区分から決めた。根拠が無いもの (41 行) は「未確認」とした
- `out/` で変えたのは、上の写しと `MOVED.md` を足したことだけ (今日更新されたファイルを一覧して確かめた)

**提案 5: 役割の明文化**

- §2 の `raw_data/`・`data/` と §3 のルール 2・3 に書いた。ファイルは動かしていない

**確認の結果**

- 試験: 変更の前・後とも `uv run --extra dev pytest` で 442 passed / 2 skipped。`tests/test_near_miss_canonical.py` と `tests/test_portability.py` (絶対パスの検査を含む) も通った
- 実行結果: 横風のスクリプト 5 本 (`xwind_search` の rank → score、site_slope、pooled_perm、`c1m_search` の rank_passages_comp) を、移動前の位置と移動後の位置からそれぞれ流した。既存の結果を上書きしないよう、出力先は作業用の写しにした。出力の CSV・JSON 10 本は、移動の前後で一致し、`out/` にある既存の結果ともバイト単位で一致した。ログで違ったのは、警告に出るスクリプトのパスだけだった
- 移動したスクリプトはすべて `py_compile` を通った
