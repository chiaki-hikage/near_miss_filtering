# NVIDIA PhysicalAI-Autonomous-Vehicles の構成と、横風の検出に使えるか

2026-11-11 に確かめた内容。データセットは Hugging Face の `nvidia/PhysicalAI-Autonomous-Vehicles` (リビジョン `2ac2e9c056e0`、2026-11-06 更新)。

- 利用条件: NVIDIA Autonomous Vehicle Dataset License Agreement (`LICENSE.pdf`)。README は「自動運転に関する用途に限り、商用・非商用とも可」としている。派生物の配布や個人の特定は禁止 (§4)。利用者は Hugging Face で同意済み
- 認証: Hugging Face に保存した名前付きトークン `physical_ai_av` を `huggingface_hub.utils.get_stored_tokens()` で取り出して使う。**トークンの文字列を表示・保存しない**
- 取得したもの (`data/physical_ai_av/`、git 管理外): メタデータ類 55.9 MB と、チャンク 0 (100 クリップ) の Ego Motion・車両寸法・取付位置 38 MB。**それ以外は取得していない**

## 1. 全体

| 項目 | 内容 |
|---|---|
| 量 | 306,151 クリップ × 20 秒 (約 1,700 時間)、25 か国。全体で約 133 TB (大半はカメラ 32 TB とライダー 100 TB) |
| 単位 | 100 クリップずつのチャンク (3,146 個)。`clip_index.parquet` が `clip_id` → チャンク番号 |
| 車両 | `platform_class`: hyperion_8.1 が 218,588、hyperion_8 が 87,563 クリップ |
| SDK | `physical_ai_av` 0.2.3 (PyPI、2026-10-23)、GitHub `NVlabs/physical_ai_av` (MIT)。**Python 3.11 以上が必要**で、このプロジェクト (3.10) には入れていない。中身は読んで確かめた。データは parquet なので SDK 無しでも読める (`uv run --with pyarrow`) |
| 食い違い | README は `splits/challenging_scenarios.parquet` の読み込みに `physical_ai_av >= 0.2.4` が要るとするが、PyPI の最新は 0.2.3 |

`features.csv` が定義する特徴量は 36 個: `egomotion`, `egomotion.offline`, `obstacle.offline`, 校正 6 種 (`camera_intrinsics(.offline)`, `lidar_intrinsics.offline`, `sensor_extrinsics(.offline)`, `vehicle_dimensions`)、カメラ 7、ライダー 1、レーダー 19。**CAN・IMU・舵角・GNSS の特徴量は定義されていない。**

`metadata/feature_presence.parquet` (クリップ × 36 特徴量の有無): `egomotion` 100%、`egomotion.offline`・`obstacle.offline`・ライダー 97.4%、前方広角カメラ 100%、レーダーのいずれか 52.5%。

## 2. Ego Motion (`labels/egomotion/`)

| 項目 | 内容 | 確かめ方 |
|---|---|---|
| 列 | `timestamp`, `qx`, `qy`, `qz`, `qw`, `x`, `y`, `z`, `vx`, `vy`, `vz`, `ax`, `ay`, `az`, `curvature` | ファイル、SDK `egomotion.py`、wiki |
| 時刻 | int64、マイクロ秒。**クリップの最初からの相対時刻** (例 −0.2〜33.1 s)。20 秒のクリップの前後を含む。wiki は「絶対時刻」と書くが、データは相対 | ファイル |
| 周期 | **20 秒のクリップの本体 (t = −0.2〜20 s) だけが 100 Hz** (間隔の最大 18 ms)。20 s より後 (最大 120 s 続く) は平均 約 7 Hz に間引かれている。1 クリップ全体の平均は 17〜100 Hz (中央値 25 Hz)。2026-11-11 の最初の版の「約 100 Hz、一部 25 Hz」は、間隔の中央値だけを見た誤り | ファイル (`out/physical_ai_av/egomotion_check/README.md`) |
| 座標系 | anchor frame: クリップの最初の時刻の rig (車両) 座標系。最初のヨーを 0 にし、ピッチ・ロールは重力に対して推定 | README、wiki |
| 軸 | x 前・y 左・z 上 | データ (最初の速度の向き、左カメラの取付位置 y = +0.94 m、カメラの高さ z > 0) |
| 原点 | rig の原点。**車体のどこかは文書に文字の記載がない** | — |
| 位置・姿勢 | `x, y, z` [m]、四元数 `qx, qy, qz, qw` (スカラーが最後) | SDK |
| 速度 | `vx, vy, vz` [m/s]。**anchor frame (局所座標) の値で、車体座標ではない** | 位置の微分と 0.003 m/s で一致 (中央値) |
| 加速度 | `ax, ay, az` [m/s²]。**anchor frame の値**。速度の微分と 0.04 m/s² で一致し、生の IMU ではなく推定の出力とみられる | ファイル |
| 曲率 | `curvature` [1/m]。ヨーレート ÷ 速度と相関 0.999 | ファイル |
| 量 | 全体 128.6 GB、1 チャンク約 41 MB (1 クリップ約 0.4 MB) | ファイル一覧 |

`labels/egomotion.offline/`: 10 Hz、0〜20 s、位置と姿勢だけ (速度・加速度・曲率なし)。全体 4.9 GB、1 チャンク 1.6 MB。97.4% のクリップ。

ヨーレートと横加速度は列として無い。四元数から作る (ヨー角の微分) か、加速度を姿勢で車体座標に回して作る。どちらも推定値から作った量で、独立な観測ではない。

チャンク 0 の高速 (> 15 m/s)・ほぼ直進 (|r| < 1 deg/s) の区間 (34 クリップ) で、車体座標の横滑り角 β の標準偏差は 0.044 deg (1〜99%: −0.14〜+0.08 deg)。推定の段階で横滑りが抑えられている可能性がある (comma1M の自己位置推定と同じ傾向。`docs/comma1M.md`)。

追加の検証 (100 Hz の区間だけを使い直したもの) は `out/physical_ai_av/egomotion_check/README.md`。横滑り角はヨーレートから幾何で決まる分 (x·r/v) だけでほぼ説明でき、タイヤの横すべりの分が見えない。ヨーレートと横加速度は 0.2〜3 Hz で実際のセンサと同程度の変動を持つ。

## 3. そのほかの情報

| データ | 内容 |
|---|---|
| `metadata/data_collection.parquet` | `country`, `month`, `hour_of_day`, `platform_class`, `radar_config`。**緯度経度・日付は無い** |
| `calibration/vehicle_dimensions` | `length`, `width`, `height`, `rear_axle_to_bbox_center`, `wheelbase`, `track_width` [m] (チャンク 0 でホイールベース 2.85〜3.17 m) |
| `reasoning/ood_reasoning.parquet` | 1,740 クリップ。`feature` (カメラ名)、`event_cluster` (9 分類: 工事区間 856、歩行者 380、特殊な車両の挙動 260、自転車など 79、複雑な交差点 50、その他 39、緊急車両・事故現場 30、路上の落下物 23、動物 23)、`events` (JSON: 開始フレーム・時刻 [μs]・説明文 `coc`)、`split`。**風・天候の分類は無く、説明文に風の語 (wind・gust など) も無い** |
| `splits/challenging_scenarios.parquet` | 1,156 クリップ、`t0` (注目する時刻のリスト、計 1,434 個、μs)。**事象の種類のラベルは無い** |
| `splits/alpamayo_aligned.parquet` | train / test / alpasim |

## 4. 横風の検出に使えるか

| 必要なもの | このデータセット |
|---|---|
| 横方向の運動 | Ego Motion から横滑り角・ヨーレート・横加速度を作れる。ただし推定の出力で、横滑りが小さく抑えられている可能性がある |
| 運転操作と外乱の区別 | **できない**。舵角・操舵トルク・CAN が無い |
| 強風の時刻の特定 | **できない**。位置と日付が無く、気象データと照合できない (国・月・時の粒度だけ) |
| 映像での確認 | できる。7 カメラ (前方広角 1 クリップ約 13 MB) |
| 規模 | 約 1,700 時間で、comma1M (約 70 時間) の 24 倍 |
