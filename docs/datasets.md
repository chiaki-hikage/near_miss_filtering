# データセット比較

本プロジェクトで使う走行データセットの一覧と比較。
**新しいデータセットを足すときは、各表に列を 1 つ増やし、§5 にカードを 1 枚足す。**
埋める項目と手順は §6 にまとめてある。

個別の詳細は各データセットの文書を参照。

- comma2k19 → [comma2k19_data.md](comma2k19_data.md)
- commaCarSegments → [comma_car_segments.md](comma_car_segments.md)
- comma1M → [comma1M.md](comma1M.md)
- KIT Multi-Surface Driving Maneuvers → [kit_msdm.md](kit_msdm.md)

## 記号

| 記号 | 意味 |
|---|---|
| ○ | そのまま使える |
| △ | 使えるが制約がある。制約は表の下に書く |
| × | 無い、または取り出せないことを確認した |
| **?** | **調べていない。埋める前に実測すること** |

数値は原則として自分で測った値を載せる。データセットの README や API の
公称値をそのまま載せた場合は「公称」と書く。推定値は「推定」と書く。

---

## 1. 規模と入手性

| 項目 | comma2k19 | commaCarSegments | comma1M | KIT MSDM |
|---|---|---|---|---|
| 配布元 | comma.ai / HuggingFace | HF `commaai/commaCarSegments` | HF `commaai/comma1M` | RADAR4KIT |
| セグメント数 | 2,019 (公称) | **188,883** (`database.json` 実測) | 4,216 (tree API 全走査) | 41 走行 |
| 合計時間 | 33 h (公称) | **3,148 h** (60 s x 件数) | 70.4 h | **42.3 分** (限界走行は 14.5 分) |
| 1 セグメント長 | 60 s | 60 s | 60 s | 可変 17.8〜183.2 s |
| まとまりの単位 | ドライブ (`dongle\|日時`) | route 101,471 本 | **公開情報の範囲では無し** | 走行 1 本 |
| 車種 | 2 (RAV4 / Civic) | **230 車種キー** | 不明 | 1 (Hyundai IONIQ 5 後輪駆動) |
| 走行環境 | CA-280 の高速のみ | 限定なし | 限定なし (§4 地理分布) | **KIT 閉鎖走行エリア。横方向の限界まで** |
| 配布総量 | 約 100 GB (公称) | 約 260 GB (推定) | 550 GB (API `usedStorage`) | **171.7 MB** |

- commaCarSegments の総量は 1 件平均 1.38 MB (12 件のヘッダ実測) x 188,883 からの推定
- 上位車種: TOYOTA_RAV4_TSS2 11,387 / TOYOTA_COROLLA_TSS2 8,559 /
  TOYOTA_PRIUS 8,310 / CHEVROLET_BOLT_EUV 7,662 / TOYOTA_HIGHLANDER_TSS2 6,971
- 実際に手元で処理した量: comma2k19 Chunk_1 188 件 (3.11 h) /
  commaCarSegments RAV4 TSS2 2,000 件 (33.3 h) / comma1M 834 件 (13.6 h) /
  KIT MSDM 全 41 走行 (42.3 分)
- KIT MSDM は 1000 Hz。他の 3 つは 20〜100 Hz

---

## 2. 使える信号

| 信号 | comma2k19 | commaCarSegments | comma1M | KIT MSDM |
|---|---|---|---|---|
| 車速 | ○ `processed_log` | ○ 生 CAN | △ 速度ベクトルのノルム | ○ 光学式 |
| 舵角 | ○ | ○ | × | △ **タイヤ切れ角**でホイール角ではない |
| 輪速 (4 輪) | ○ | ○ | × | △ 前左のみ。**限界走行には無い** |
| ヨーレート | ○ CAN | ○ CAN | △ **進路変化率** (course rate) | ○ |
| 縦横加速度 | ○ CAN + IMU | ○ CAN | △ 車速の微分と `v x r` | ○ 後軸位置 |
| ブレーキ圧 | × ビット位置が不確かで無効化 | ○ 0x226 `BRAKE_MC` | × | × |
| アクセル開度 | ○ | ○ | × | × |
| ABS 作動 | × 0x226 が存在せず故障フラグのみ | ○ `ABSACT` | × | × |
| VSC / TCS 作動 | ? | ○ 復号可 (作動は §3b) | × | × 作動状態の記載も無い |
| 純正 AEB (PCS) | ? | ○ 0x283 | × | × |
| レーダ | ○ `processed_log` | ○ 生 CAN (バス 1) | × | × |
| IMU (加速度計 / ジャイロ) | ○ 110 Hz / 20 Hz | × | △ localizer に融合済みで分離できない | △ 加速度とヨーレートのみ |
| GNSS 測位解 | ○ 5 Hz。lat/lon/alt + UTC | × | △ 同上 (ECEF として出る) | ○ RTK の lat/lon |
| GNSS 生観測 (疑似距離) | ○ `raw_gnss_{ublox,qcom}` | × | × | × |
| openpilot 介入 | △ 送信フレームからの推定 | ○ `pandaStates` (**CAN ではない**) | × | — |
| 車両諸元 (WB / SR) | ○ 実測当てはめ | ○ 実測当てはめ。`carParams` にも値があるが不一致 | × 車種不明 | **○ ヨー慣性・コーナリング剛性まで公開** |
| 絶対位置 | ○ `global_pose` 20 Hz。ただし全件 CA-280 で地域の絞り込みには使えない | × | ○ localizer (ECEF) | ○ ただし 80 x 90 m の範囲 |
| **録画日時** | ○ `frame_gps_times` に絶対 UTC | × | × | × |
| 映像 | ○ 20 Hz 全編 | × | ○ 20 Hz。**部分取得可** | × |
| **横滑り角 β** | × 推定のみ (ノイズ床 5°) | × | × 推定のみ (ノイズ床 5°) | **○ 光学式で実測。最大 19.9°** |

制約と未確認の内訳:

- comma2k19 の ABS は 0x320 の**故障**フラグ。作動フラグを載せた 0x226 が
  2017 年式には無いことを確認済み ([signals_rav4.md](signals_rav4.md))
- comma2k19 の VSC / AEB は **調べていない**。0x226 が無い以上 VSC も期待できないが、
  0x283 (PCS) の有無は確認していない
- comma1M の「ヨーレート」は速度ベクトルの向きの変化率で、車両のヨーレートではない。
  横滑り角のぶんだけ食い違うので、**スリップの判定には使わない**。
  2 m/s 未満では定義できず NaN になる

### commaCarSegments は CAN だけではない

rlog に入っているメッセージは 3 種類だけで、由来が分かれる。
ローカルの **2,000 件すべてで種別の集合が同一**であることを確認した。

| 由来 | 該当する信号 |
|---|---|
| `can` 100 Hz | 車速・舵角・輪速・ヨーレート・加速度・ブレーキ圧・アクセル開度・ABS/VSC/TCS・AEB・**レーダ** |
| `pandaStates` 10 Hz | **openpilot 介入** (`controlsAllowed`) |
| `carParams` 1 件 | **車種判定と車両諸元** |

`carParams` の中身の例 (`TOYOTA_RAV4_TSS2`):

```
carFingerprint: TOYOTA_RAV4_TSS2
steerRatio: 14.30      wheelbase: 2.6899      mass: 1762.1
centerToFront: 1.1835  tireStiffnessFactor: 0.7933
steerActuatorDelay: 0.12   radarUnavailable: False
```

**`steerRatio` は使っていない。** `carParams` は 14.30、こちらの設定は実測当てはめの
18.569 で食い違う。openpilot の `steerRatio` は横方向制御の調整値で、
`δsw = SR·L·(r/v) + SR·Kus·(v·r) + δoffset` を当てはめた値とは意味が違う。
ホイールベースは 2.6899 対 2.69 で一致する。
現状 `carParams` は fingerprint の照合と `meta` への保存にしか使っていない。

### IMU / GNSS があるのは comma2k19 だけ

| データセット | IMU | GNSS |
|---|---|---|
| comma2k19 | `processed_log/IMU/{accelerometer, gyro, gyro_bias, gyro_uncalibrated}` | `processed_log/GNSS/{live,raw}_gnss_{ublox,qcom}` + `global_pose` |
| commaCarSegments | 無し (2,000 件で確認) | 無し (同上) |
| comma1M | localizer の入力として使われているが、単独では取り出せない | 同左。出るのは融合後の ECEF だけ |

comma2k19 の実測 (Chunk_1 の 1 セグメント):

- `IMU/accelerometer` 6,614 点 / 60 s = 110 Hz、`IMU/gyro` 1,202 点 = 20 Hz
- `GNSS/live_gnss_ublox` 5 Hz、列は `[緯度, 経度, 標高, unix 時刻 ms, ?, ?]`
  (列 0-3 は `global_pose` と GPS 時刻で裏取り済み。列 4-5 は未確認)
- `GNSS/raw_gnss_ublox` は (12,887, 10) の衛星ごとの生観測
- `global_pose/frame_positions` は ECEF、20 Hz。緯度経度に直すと
  37.649071, -122.452035 → 37.635601, -122.439387 (CA-280 沿い)
- `global_pose/frame_gps_times` は `[GPS 週, 週内秒]`。週 2011 + 454455.3 s は
  UTC 2018-07-27 06:13:57 で、`live_gnss` の unix 時刻とも、
  ドライブ名 `2018-07-27--06-03-57` + セグメント 10 (= 10 分) とも一致する。
  **ドライブ名は現地時刻ではなく UTC**

---

## 3a. 検出器が動くか (入力の有無で決まる)

`configs/detection.yaml` の 23 種類。入力の無い検出器は例外を出さず静かに 0 件になる。

| 検出器 | 要る入力 | comma2k19 | commaCarSegments | comma1M |
|---|---|---|---|---|
| `hard_brake` / `hard_accel` / `panic_brake` | 車速 | ○ | ○ | ○ |
| `lateral_accel` | 車速 + ヨーレート | ○ | ○ | △ |
| `lane_change_candidate` | 車速 + ヨーレート | ○ | ○ | △ |
| `s_evasion` | 車速 + ヨーレート | ○ | ○ | △ |
| `risky_lane_change` | 上記のいずれか | ○ | ○ | △ |
| `hard_steer` | 舵角レート + 横加速度 | ○ | ○ | × |
| `weaving` | 舵角レート | ○ | ○ | × |
| `counter_steer` | 舵角レート + 諸元 | ○ | ○ | × |
| `brake_and_steer` / `brake_after_evasion` | `hard_steer` か `s_evasion` | ○ | ○ | △ |
| `short_thw` / `low_ttc` / `closing_fast` / `cut_in_candidate` | レーダ | ○ | ○ | × |
| `panic_brake_with_lead` | レーダ | ○ | ○ | × |
| `yaw_instability` | 舵角 + 諸元 | ○ | ○ | × |
| `wheel_speed_anomaly` | 輪速 + トレッド幅 | ○ | ○ | × |
| `panic_brake_pedal` | アクセル開度 | ○ | ○ | × | × |
| `abs_active` | ABS 作動フラグ | × | ○ | × |
| `vsc_active` | VSC 作動フラグ | ? | ○ | × |
| `aeb_active` | PCS フラグ | ? | ○ | × |

comma1M の △ は、ヨーレートが進路変化率であることによる。閾値は comma2k19 の
CAN 由来ヨーレート向けに決めてあるので、**そのままでは合わない**。

KIT MSDM は列に入れていない。42 分の閉鎖路走行で、候補の抽出対象ではなく
**物差し**として使うため ([kit_msdm.md](kit_msdm.md))。

## 3b. 実測で観測された件数

同じ `configs/detection.yaml` での実行結果。走査した時間が違うので、
件数の直接比較ではなく「その事象が実データに出るか」の確認として読む。
comma2k19 は Chunk_1 だけなので母数が小さい。

| 検出器 | comma2k19 (3.11 h) | commaCarSegments (33.3 h) | comma1M (13.6 h) |
|---|---:|---:|---:|
| `hard_brake` | 11 | 216 | 280 |
| `hard_accel` | 8 | 78 | 248 |
| `lane_change_candidate` | 17 | 186 | 88 |
| `cut_in_candidate` | 33 | 136 | — |
| `short_thw` | 17 | 139 | — |
| `counter_steer` | 8 | 100 | — |
| `hard_steer` | 6 | 41 | — |
| `low_ttc` | 0 | 30 | — |
| `risky_lane_change` | 4 | 27 | 9 |
| `yaw_instability` | 0 | 22 | — |
| `abs_active` | — (0x226 が無い) | 11 | — |
| `panic_brake_pedal` | 2 | 10 | — |
| `panic_brake` | 1 | 8 | 39 |
| `weaving` | 2 | 7 | — |
| `wheel_speed_anomaly` | 0 | 6 | — |
| `lateral_accel` | 0 | 4 | 9 |
| `closing_fast` | 2 | 4 | — |
| `aeb_active` | ? | 2 | — |
| `brake_after_evasion` | 0 | 2 | 0 |
| `brake_and_steer` | 0 | 1 | 0 |
| `panic_brake_with_lead` | 0 | 1 | — |
| `s_evasion` | 0 | 0 | 0 |
| `vsc_active` | ? | 0 | — |

`—` は入力が無く動かないもの、`?` は未確認。
`s_evasion` は 3 データセット・計 50 h でまだ 1 件も出ていない。条件が厳しすぎる可能性がある。

---

## 4. 取得コスト

1 セグメントあたりの実測値。

| 内訳 | comma2k19 | commaCarSegments | comma1M | KIT MSDM |
|---|---|---|---|---|
| 最小構成 | 5.1 MB (`processed_log/CAN`) | **1.38 MB** (`rlog.zst`) | 2.54 MB (`localizer`) | **171.7 MB で全部** |
| 位置だけ | — | — | **2.3 KB** (Range 読み) | — |
| 静止画 | 0.49 MB (`preview.png`) | 無し | **13 KB** (`thumbnail.jpg`) | 無し |
| 映像 (全編) | 36 MB | 無し | 75 MB x カメラ数 | 無し |
| 映像 (前後 7 秒) | ? | — | **4.6〜10.5 MB** | — |
| HTTP Range | ? | ? | ○ 単一 206。複数 Range は 416 | 不要 (全体で 171.7 MB) |

comma1M は safetensors / HEVC の一部だけを取れる。位置は全 4,216 件で 9.6 MB、
映像は候補の前後だけで 1 件 5〜10 MB。comma2k19 と commaCarSegments で
同じことができるかは調べていない。

---

## 5. データセットごとの要点

### comma2k19

- **役割**: 基準。閾値と符号・スケールはすべてここで決めた
- **強み**: 復号済み信号・レーダ・映像・IMU・絶対姿勢がそろう。**録画日時がある唯一のデータセット**
- **弱み**: 走行環境が CA-280 の高速に固定。33 h と小さい。ABS の作動フラグが無い
- **向く用途**: 検出条件の設計、目視による正誤の確認、他データセットの照合基準
- **注意**: セグメント番号は連番とは限らない。欠番を跨いで連結しない

### commaCarSegments

- **役割**: 量を集める本命。高緊急度イベントの実在確認
- **強み**: 3,148 h。230 車種。**ABS / VSC / AEB の作動フラグを直接取れる**。
  openpilot の介入も `controlsAllowed` で直接分かる
- **弱み**: **映像が無い**ので目視確認ができない。位置も日時も無いので地域・季節で絞れない。
  復号済み信号が一切無く、すべて生 CAN から作る
- **向く用途**: 大量スクリーニング、車種横断の比較、車両フラグを起点にした逆引き
- **注意**: 同じ車種キーでも車両ごとに受信周期が違う。DBC が同じでも符号が違う実例がある
  (TSS2 の `ACCEL_X` / `ACCEL_Y` は 2017 年式と逆)

### comma1M

- **役割**: 位置で地域を絞る唯一の手段
- **強み**: セグメントごとの緯度経度。thumbnail 13 KB と映像の部分取得で、確認が安い
- **弱み**: **CAN が無い**ので正規化信号が 2 本だけ。母数 70.4 h。
  **録画日時が無い**ので季節が分からない。車種不明で諸元が使えない
- **向く用途**: 地域・天候で層別した挙動の比較、候補の映像確認
- **注意**: `yaw_rate` は進路変化率。スリップの判定に使わない。
  積雪の見た目判定は精度 3 割なので、映像で確認するまで積雪とみなさない

### KIT Multi-Surface Driving Maneuvers

- **役割**: 横滑り物理の**物差し**。候補の抽出対象ではない
- **強み**: 横滑り角 β を光学式センサで **1000 Hz 実測**。最大 19.9°。
  ヨー慣性・コーナリング剛性・センサ取り付け位置まで公開。μ が 2 水準 (1.1 / 0.7)。
  171.7 MB と小さく、CC BY-SA 4.0 で商用可
- **弱み**: **42 分**しかない (限界走行は 14.5 分)。1 台・乾燥・50 km/h 以下。
  **限界走行に輪速が無い**。ESC/VSC フラグも映像も無い
- **向く用途**: 特徴量の較正と検証。単軌道モデルの適用範囲の確定
- **注意**: MAT ファイルが MATLAB の timeseries (MCOS) で、**scipy では読めない**。
  β は計測点で大きく変わる (Correvit 位置と重心で符号すら合わない)。
  IONIQ 5 はオーバーステア傾向 (Kus < 0)、RAV4 は逆

---

## 6. データセットを追加するときの手順

### 6.1 調べて書くこと (この文書に足す)

§1〜§4 の各表に列を 1 つ足し、§5 にカードを 1 枚足す。
**分からないところは埋めずに `?` を置く。** 推測で埋めない。カードの雛形:

```markdown
### <名前>

- **役割**: 何のために使うか
- **強み**: 他に無いもの
- **弱み**: 無いもの、信用できないもの
- **向く用途**:
- **注意**: 実測で分かった落とし穴
```

埋める前に実測で確かめる項目:

1. セグメント数・合計時間 (一覧 API を全ページ辿る。公称値を信じない)
2. 1 セグメントあたりの実容量 (数件のヘッダで測る)
3. 信号の有無と**単位・符号**。DBC や README の記載だけで決めない
4. 受信周期 (中央値と分散。車両差があるか)
5. 位置・録画日時の有無
6. 部分取得ができるか (HTTP Range に 206 で応じるか)

### 6.2 コード側でやること

層は `生データ → 正規化信号 → 特徴量 → 検出` に分かれており、
データセット固有の処理は最初の層だけに閉じる。触るのは次の 4 か所。

| 場所 | やること |
|---|---|
| `src/near_miss/io/<name>.py` | 取得と読み出し。`SegmentRef` と `SegmentData` を返す |
| `src/near_miss/sources.py` | `<name>_source()` を 1 つ足す |
| `configs/vehicles/*.yaml` | 車種が分かるなら信号定義。分からないなら空の設定 |
| `configs/datasets/<name>.yaml` | データセット固有の前処理・絞り込みの設定 |

`features.py` より下は触らない。**触る必要が出たら層の切り方を間違えている。**

正規化チャネル名の取り決めは `src/near_miss/io/canonical.py` の冒頭にある。
無い信号は「無い」まま渡す。埋めない。

### 6.3 確認すること

- `scripts/run_detection.py --dataset <name>` が通る
- 既存データセットの検出結果が変わらない (comma2k19 Chunk_1 が基準)
- `tests/test_<name>.py` に、単位・符号・欠測の扱いの確認を書く
- 取得スクリプトに `--dry-run` を付け、**落とす前に転送量を出す**
