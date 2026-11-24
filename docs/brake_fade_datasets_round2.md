# ブレーキフェードの候補抽出: 追加の公開データの調査 (2026-11-19)

[brake_fade_datasets.md](brake_fade_datasets.md) (comma2k19・ZOD・NCST・GR Cup) の続き。目的は、フェードが起きたデータに限らず、熱負荷の高い実走行の場面を 1 件以上取り出し、前方映像・制動操作・車速・減速度・熱負荷を並べて確かめること。

最低限の条件: 連続した反復制動の履歴、車速、制動要求を表す信号、減速度。前方映像があれば優先する。
今回は何もダウンロードしていない。確かめたのは公開ページ、GitHub の API、PDR の CSV の 400 kB の一部 (範囲指定の取得) だけ。

## 1. 区別
- **実データを取得できる**: 誰でも取得でき、利用条件が書いてある。
- **条件が不明**: 取得はできるが、企業内研究で使ってよいかが書いていない。取得しない。
- **取得できない**: 論文だけ、ソフトだけ、申請が要る、非商用に限る。

## 2. 比較 (優先順)

| 順 | 候補 | 映像 | 制動要求 | 車速 | 減速度 | 勾配・高度 | 熱負荷 | 利用条件 | 取得 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Udacity CH2_002 (HMB_1〜6、2016-11-18、Lincoln MKZ) | 前方 3 台 | ブレーキのペダルとトルク (dbw_mkz の BrakeReport・BrakeInfoReport。CH2_002 の HMB_1 で確認、HMB_2 は未確認。5.1 節) | 輪速 | IMU と BrakeInfoReport の対地加速度 | GPS 高度 | 中。HMB_2 は「峠越えで高度が急に上下する」2 車線道路 (791 s) | MIT (datasets/LICENSE.md) | トレント 4.4 GB。直リンクなし |
| 2 | Corvette C8 の PDR から取り出した CSV ([a7v7/gm_pdr_analyzer](https://github.com/a7v7/gm_pdr_analyzer) の pdr_telemetry.csv) | **なし** (CSV だけ) | `Brake Pos (%)` 連続値 (1/255 刻み、約 19 Hz) | 車速と 4 輪の輪速 | 縦加速度 (約 48 Hz) | GPS 高度 | 高。Portland International Raceway の周回 (20 分以上) | MIT の文面はソフトウェアが対象で、データに及ぶかは確認できない (5.2 節)。保留 | 31 MB |
| 3 | A2D2 (Audi、バイエルンの市街地 3 系列) | 前方ほか 6 台 | ブレーキ圧 (バス信号) | あり | あり | ピッチ角 (バス信号) | 低 (市街地) | CC BY-ND 4.0 (商用可) | バス信号は系列ごとの tar。映像は大きい |
| 4 | NCST (取得済み) | なし | エア圧 | あり | 車速の微分 | 使える高度なし | 高。路上で 159 °C | CC0 | 済み |
| - | CSU の J1939 (前回の候補) | なし | 前回の調査どおり | - | - | - | - | - | 320 MB。相談待ち |

取得しないもの:

| 候補 | 理由 |
|---|---|
| PDR の元の MP4 (映像と信号が入ったもの) | 公開され、利用条件の付いたものは見つからない。フォーラムの個人間の共有 (CorvetteForum「Sharing PDR files」) は条件が不明。YouTube などに上げた動画は再エンコードで信号のトラックが消える |
| RACECAR (Indy Autonomous Challenge) | 制動の信号がない (CAN はレーダーだけ)。CC BY-NC 4.0 |
| nuScenes の CAN 拡張 | 制動の信号はあるが、非商用 (商用は別契約) |
| GR Cup (Hack the Track) | 前回の調査どおり、規約がハッカソンに限る |
| F1 (OpenF1・FastF1) | ブレーキは 0/100 の 2 値。映像なし。元データの利用条件が不明 |
| MoTeC i2・AiM のサンプルログ | ソフトに同梱。i2 Pro は「職業的でない利用は無料」とされ、企業内の利用は条件が不明 |
| RaceCapture の Podium | 共有された走行をファイルで取り出す方法と条件が見つからない |
| Wyoming CV Pilot の BSM (I-80 の大型車) | 制限付きの公開 (Restricted Access) |
| Caltrans CA21-3232 (大型車のローターの温度) | 報告書だけ。データは公開されていない |
| 制動試験機のデータ (FIVE@ECL の鳴きの試験、Zenodo の熱電対格子) | 台上の試験片。走行の場面ではない |

## 3. PDR について確かめたこと
- **信号の置き場所**: 映像と同じ MP4 の中の時刻付きメタデータのトラック (`ctbx` の `marl`、GM の Marlin 形式)。ExifTool 12.82 (2024-04-05) 以降が `-ee` で読める。Python の移植 (gm_marlin.py、MIT) もある。
- **制動の信号**: `BrakePos` だけ。ブレーキ圧とブレーキ温度の信号はない。ほかに `ABSActive`、縦・横・上下の加速度、4 輪の輪速、`Speed`、アクセル、GPS (緯度・経度・高度) がある。
- **連続値か**: 連続値。公開 CSV の 604〜619 s の一部で `Brake Pos (%)` は 0〜14.5% の 15 段階、刻みは 0.392% (1/255)。値が変わったときだけ記録される差分形式なので、記録の間隔は一定でない (平均約 19 Hz)。単位は CSV の列名が % としている。ペダルの位置で、油圧ではない。
- **入手**: 映像付きの元の MP4 を公開しているところは見つからなかった。gm_pdr_analyzer の Makefile は `PDR_0034.mp4` を前提にしているが、MP4 はリポジトリにない。

## 4. 最短の方法の提案
1. **映像あり (第 1 候補): Udacity CH2_002 の HMB_2**。峠越えの下りの制動を、前方映像・ペダル・ブレーキトルク・輪速・IMU・GPS 高度で並べられる。必要なこと:
   - 4.4 GB の取得の判断 (トレントだけ)。
   - ROS1 bag を読む道具。依存は増やさず、`uv run --no-project --with rosbags` を一時的に使う (前回の pypdf と同じ形)。これも判断が要る。
   - 最初に `rosbag info` 相当で CH2_002 の話題名を確かめる。制動の話題がなければ、ここで止める。
   - 流れは comma2k19 と同じ: 勾配の補正、熱の指標 (E300・H_τ)、制動の回数、動画。
   - 限界: 乗用車の 13 分の峠道なので、熱負荷はサーキットや大型車より低いとみられる。ブレーキ温度はない。
2. **熱負荷が最も高い (映像なし): PDR の CSV (31 MB)**。サーキットの周回の反復制動から、熱の指標が最大の数周を取り出す。映像の代わりに GPS の走行線を描く。取得の前に、データの利用条件を作者に確かめるか、社内で判断する。
3. **自前の計測**: 社内に PDR 付きの Corvette (C7・C8) か Camaro があれば、サーキット走行の MP4 1 本で映像・ペダル・加速度・輪速がそろう。ExifTool で読める。ブレーキ圧と温度は別の計測が要る。

公開データだけでは、前方映像とブレーキ温度 (または油圧) と高い熱負荷が同時にそろう候補は、今回も見つからなかった。

## 出典
- PDR: [Engadget の紹介](https://www.engadget.com/2014/01/05/chevrolet-corvette-performance-recorder/)、[CorvetteForum「PDR data decoded」](https://www.corvetteforum.com/forums/c8-tech-performance/4955873-pdr-data-decoded.html)、[CorvetteForum「Sharing PDR files」](https://www.corvetteforum.com/forums/c7-z06-discussion/3917487-sharing-pdr-files.html)、[ExifTool GM tags](https://exiftool.org/TagNames/GM.html)、[ExifTool の版の履歴](https://exiftool.org/history.html)、[a7v7/gm_pdr_analyzer](https://github.com/a7v7/gm_pdr_analyzer)、[Corvette Blogger (2026 年型の表示の変更)](https://www.corvetteblogger.com/2025/10/31/gm-changed-how-telemetry-data-is-displayed-on-your-2026-corvettes-performance-data-recorder-videos/)
- Udacity: [datasets/README.md](https://github.com/udacity/self-driving-car/tree/master/datasets)、[CH2 の README](https://github.com/udacity/self-driving-car/tree/master/datasets/CH2)、[datasets/LICENSE.md](https://github.com/udacity/self-driving-car/blob/master/datasets/LICENSE.md)、CH3_002 の話題名の例 [slevin48/self-driving](https://www.github.com/slevin48/self-driving)
- A2D2: [論文 arXiv:2004.06320](https://arxiv.org/abs/2004.06320)、[a2d2.audi](https://a2d2.audi)
- RACECAR: [Hugging Face の README](https://huggingface.co/datasets/madhurbehl/RACECAR_DATA)、[AWS Open Data](https://registry.opendata.aws/racecar-dataset/)
- nuScenes の CAN 拡張: [nuscenes-devkit](https://github.com/peiyunh/nuscenes-devkit)、[Foxglove の解説](https://foxglove.dev/blog/visualizing-nuscenes-with-foxglove-studio)
- F1: [OpenF1](https://freeapihub.com/apis/openf1-api)
- RaceCapture: [Autosport Labs のブログ](https://autosportlabs.com/blog)
- MoTeC: [HP Academy のフォーラム](https://www.hpacademy.com/es/foro/analisis-de-datos-de-deportes-de-motor-profesionales-1/show/example-logged-data-and-software-tools-download-2/)
- Wyoming CV Pilot: [AmeriGEOSS の目録](https://data.amerigeoss.org/dataset/wyoming-cv-pilot-basic-safety-message-bsm-restricted-access)
- 大型車: [Caltrans CA21-3232](https://dot.ca.gov/-/media/dot-media/programs/research-innovation-system-information/documents/final-reports/ca21-3232-final-report-a11y.pdf)、NCST [Zenodo](https://zenodo.org/records/5839097)
- 制動試験機: [FIVE@ECL (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6205071)、[Zenodo の熱電対格子](https://zenodo.org/records/21371063)

## 5. 追加の確認 (2026-11-19 午後)

### 5.1 Udacity CH2_002
- **取得の方法**: トレントだけ。
  - 公式の README は CH2_002 の直接ダウンロードを「None」としている。
  - Academic Torrents の登録はファイル 1 個 (`Ch2_002.tar.gz`、4,716,005,956 バイト) で、HTTP の配布元はない。
  - GitHub の .torrent を読むと、トラッカーは opentrackr の 1 本で、HTTP の web seed (`url-list`) はない。
  - 第三者の写し (Gitee の Udacity_Dataset) は HMB_3 の白黒縮小画像と舵角だけで、制動の信号はない。
- **中身**: CH2_002 の HMB_1 (221 s、2016-11-17) の `rosbag info` の出力が、第三者のリポジトリ ([jakobsvenningsson/DD2424-steering-wheel-prediction-project](https://github.com/jakobsvenningsson/DD2424-steering-wheel-prediction-project) の `baseline network/HMB_1/HMB_1.yaml`) にある。そこから確かめた話題と平均の頻度 (件数 ÷ 221 s):

| 必要なもの | 話題 | 型 | 頻度 |
|---|---|---|---|
| ブレーキペダル | `/vehicle/brake_report` | dbw_mkz_msgs/BrakeReport | 約 50 Hz |
| ブレーキトルク・対地加速度・ABS | `/vehicle/brake_info_report` | dbw_mkz_msgs/BrakeInfoReport | 約 50 Hz |
| 車速 | `/vehicle/wheel_speed_report` | dbw_mkz_msgs/WheelSpeedReport | 約 100 Hz |
| IMU | `/vehicle/imu/data_raw`、`/imu/data` | sensor_msgs/Imu | 約 100 Hz、約 400 Hz |
| 前方カメラ | `/center_camera/image_color/compressed` (左右も) | sensor_msgs/CompressedImage | 約 20 Hz |
| 位置・高度 | `/fix`、`/vehicle/gps/fix`、`/pressure` (気圧) | NavSatFix、FluidPressure | 約 400 Hz、1 Hz、約 50 Hz |

- **信号の意味**: 公開された dbw_mkz_msgs の定義 (Autoware_Toolbox の写し) では、次のとおり。
  - BrakeReport: `pedal_input` などは「単位なし、0.15〜0.50」、`torque_*` は Nm。
  - BrakeInfoReport: `brake_torque_request`・`brake_torque_actual` が Nm、`accel_over_ground` が m/s^2、`abs_active` がある。
  - 2016 年の bag の型の版 (md5) と同じ定義かは確かめていない。bag には定義そのものが入っているので、読めば分かる。
- **確かめられなかったこと**: 峠越えの HMB_2 の話題の一覧は見つからなかった。HMB_1 と同じ日・同じ車・同じ記録の設定とみられるが、確認はしていない。
- **判断**: 必要な信号はそろう見込みが高い。ただし取得の方法がトレントだけなので、ユーザーの方針どおり保留する。

### 5.2 PDR の CSV (a7v7/gm_pdr_analyzer)
- **LICENSE**: 根元に 1 個 (MIT、Copyright (c) 2025 Anthony Verbeck)。許諾の対象は「this software and associated documentation files (the "Software")」。
- **データへの言及**: README と CLAUDE.md にも、データの利用条件の記述はない。`pdr_telemetry.csv` は、作者が最初のコミット (addfff8、2025-10-10) で入れている。
- **判断**: MIT がこのデータファイルにも及ぶとは確認できない。ソフトウェアの許諾の文面で、データを名指ししていないため。取得しない。作者に確かめる (問い合わせはユーザーが行う) まで保留する。

### 5.3 その他
利用条件が明確で、反復制動・車速・制動要求・減速度がそろう新しい候補は見つからなかった。
- F1 系: Kaggle の F1 は CC BY-SA 3.0 だが、2 Hz でブレーキは圧ではない。OpenF1 は非商用が前提。
- Colorado State 大学の J1939: 前回のとおり、利用条件の記載がない。

いま利用条件が明確で手元にある高熱負荷の実走行は、NCST の高速道路の 140〜158 分 (映像なし、[brake_fade_feasibility.md](brake_fade_feasibility.md) 2.5 節) だけである。

## 6. 取得後 (2026-11-19)
Udacity CH2_002 を公式の .torrent で取得し、MD5 が一致した。HMB_2 に制動・IMU・高度・前方映像がそろうことを確かめた。峠の下りの熱負荷は 100 km/h の停止 0.9 回分で、高くない。詳細は [udacity_hmb_brake.md](udacity_hmb_brake.md)。
