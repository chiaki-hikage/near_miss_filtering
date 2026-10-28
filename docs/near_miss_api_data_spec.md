# ヒヤリハット候補イベント データ仕様 (API 連携用)

## 1. 概要

走行データ (comma2k19・commaCarSegments) から、急ブレーキ・急操舵・スリップ・路面凹凸による異常振動などのヒヤリハット候補を検出している。

解析ごとに出力の形式が異なるので、API 連携ではすべてのイベントを共通の canonical event 形式 (JSON、1 イベント 1 件) にそろえた。

イベント種別に依存しない共通情報と、種別ごとの固有情報 (`attributes`) を分けており、強い横風などの新しいイベントも、共通部分を変えずに追加できる。

---

## 2. データの基本構造

| 区分 | 項目 | 内容 |
|---|---|---|
| 共通情報 | `schema_version`・`event_id`・`event_type`・`detector` | 形式の版、イベントの ID、種別、検出ロジック |
| | `dataset`・`vehicle`・`drive_id`・`chunk`・`segment` | どのデータセット・車両・走行・区間か |
| | `time`・`location`・`speed_mps` | いつ・どこで・どの速度で |
| | `score`・`grade`・`candidate_flag` | どの程度か (スコア・等級・候補かどうか) |
| 判定・参照情報 | `detection` | 何を根拠に検出したか (方法・規則・設定) |
| | `source` | 元の映像・CAN・IMU・GNSS の場所 |
| | `review` | 人手・映像での確認結果 (ある場合) |
| イベント固有情報 | `attributes` | event_type ごとの固有値 |

共通項目はイベント種別に依存しない情報、`attributes` は event_type ごとの固有情報。
API 側で種別を問わず扱う項目 (検索・一覧・元データへの導線など) は共通項目だけで足りるようにしている。

### 必須・optional

| 必須 | optional (値があるときだけ出す) |
|---|---|
| `schema_version`・`event_id`・`event_type`・`detector`・`dataset`・`drive_id`・`time` (`base`・`start_sec`・`end_sec`)・`detection` (`method`)・`source`・`attributes` | `vehicle`・`chunk`・`segment`・`location`・`speed_mps`・`score`・`grade`・`candidate_flag`・`review`、および `time`・`detection`・`source` の中のその他の項目 |

値が無い項目は null を入れずに省略。たとえば commaCarSegments には映像と位置情報が無いので、`location` は出ず、`source` は `can_uri` だけになる。

`score` は種別ごとに意味が違うため、必ず `{ "name": …, "value": … }` の組で持つ。

---

## 3. JSON 例 (road_vibration)

```json
{
  "schema_version": "1.0",
  "event_id": "nm_92aab011233b0217",
  "event_type": "road_vibration",
  "detector": "road_vibration_locate",
  "dataset": "comma2k19",
  "vehicle": "toyota_rav4",
  "drive_id": "b0c9d2329ad1606b|2018-10-05--22-20-31",
  "chunk": "Chunk_2",
  "segment": 16,
  "time": {
    "base": "segment_relative_s",
    "start_sec": 46.2,
    "end_sec": 46.72,
    "peak_sec": 46.45,
    "utc": "2018-10-06T05:37:18Z"
  },
  "location": { "latitude": 37.726728, "longitude": -122.471957, "heading_deg": 2.5 },
  "speed_mps": 14.722,
  "score": { "name": "near_miss_candidate_score", "value": 73.6 },
  "candidate_flag": true,
  "detection": {
    "method": "rule_threshold",
    "rule": "A = sqrt(heave_ratio * max(roll_ratio, pitch_ratio)) >= 3.0",
    "config_hash": "ce06a1d3af"
  },
  "source": {
    "video_uri": "comma2k19/Chunk_2/b0c9d2329ad1606b|2018-10-05--22-20-31/16/video.hevc",
    "can_uri": "comma2k19/Chunk_2/b0c9d2329ad1606b|2018-10-05--22-20-31/16/processed_log/CAN",
    "imu_uri": "comma2k19/Chunk_2/b0c9d2329ad1606b|2018-10-05--22-20-31/16/processed_log/IMU",
    "gnss_uri": "comma2k19/Chunk_2/b0c9d2329ad1606b|2018-10-05--22-20-31/16/global_pose",
    "video_frame_start": 924,
    "video_frame_end": 934
  },
  "attributes": {
    "vertical_accel_peak_mps2": 4.155,
    "roll_rate_peak_dps": 9.349,
    "concentration": 0.758,
    "repeatability": "2 ドライブ以上でイベント",
    "physics": { "consistency": "Strong physical consistency" },
    "iso2631": { "mtvv_total_mps2": 1.8137, "comfort": "非常に不快" }
  }
}
```

---

## 4. event_type ごとの attributes

代表的な項目で、データセットによって出ない項目もある。

### hard_braking (急ブレーキ)

| 項目 | 内容 |
|---|---|
| `ax_min_mps2` | イベント区間の前後加速度の最小 [m/s²] |
| `threshold`・`exceedance` | 判定の閾値と超過量 |
| `candidate_window.jerk_min_mps3` | 前後方向の躍度 [m/s³] |
| `candidate_window.brake_pressed_max` | ブレーキの踏み込み |
| `candidate_window.abs_active_max`・`brake_mc_max_mpa` | ABS 作動・ブレーキ圧 (commaCarSegments のみ) |

### hard_steering (急操舵)

| 項目 | 内容 |
|---|---|
| `lateral_accel_mps2` | イベント区間の横加速度 [m/s²] |
| `candidate_window.steering_rate_dps` | 舵角速度 [deg/s] |
| `candidate_window.yaw_rate_dps` | ヨーレート [deg/s] |
| `candidate_window.lateral_jerk_mps3` | 横方向の躍度 [m/s³] |

`candidate_window` は、イベントを含む前後数秒の確認区間での値(イベント時点の値ではない)。

### slip (スリップ)

検出ロジックが 2 つあり、`detector` で区別します。

| 項目 | 内容 |
|---|---|
| `grade` (共通項目) | 等級。裏付けの強さを表す (`multi_evidence`・`high_evidence` など) |
| `evidence`・`mechanisms` | 立った証拠と、その物理的な系統 (輪速・横方向) |
| `abs_active`・`vsc_active`・`slip_warn` | 車両の制御 (ABS・VSC・滑り警告) の作動 |
| `wheel_speed_diff_peak_mps` | 4 輪の輪速差 [m/s] |
| `wheel_decel_peak_mps2` | 輪単体の減速度 [m/s²] |
| `yaw_residual` | 舵から期待されるヨーレートとのずれ |
| `sideslip_angle_peak_deg` 等 | 横滑り角 [deg] (横滑り角を使う検出ロジックのみ) |

### road_vibration (路面凹凸による異常振動)

| 項目 | 内容 |
|---|---|
| `vertical_accel_peak_mps2` | 上下加速度の最大 [m/s²] |
| `roll_rate_peak_dps` | ロール角速度の最大 [deg/s] |
| `concentration`・`rise_s` | 振動が短時間に集中した度合い・立ち上がりの時間 |
| `repeatability` | 同じ地点を通るたびに再現するか |
| `physics` | 車両の物理モデルとの整合 (評価した一部の地点のみ) |
| `iso2631` | ISO 2631-1 を参考にした人体振動の指標 (MTVV・VDV・快適性の目安) |

---

## 5. イベントの特定と元データ参照

### event_id

- `event_id` は検出結果単位の stable ID 。同じ event_type・detector の同じ検出結果であれば、再出力しても同じ ID になる。
- 同じ実世界の出来事でも、`hard_braking` と `slip` のように別の event_type (または別の検出ロジック) で検出されたものは、別の `event_id` になる
- 現時点では、複数の検出結果を同じ出来事として束ねる ID はない。必要になれば上位の ID として追加できる構成

### time

- 開始 (`start_sec`)・終了 (`end_sec`)・ピーク (`peak_sec`、ある場合) を持ちます
- 現状はセグメント (60 秒単位の区間) の先頭からの秒 (`time.base = "segment_relative_s"`)。60 秒を超える値や負の値は、隣のセグメントにまたがることを示す
- UTC (`utc`) や、元ログの時刻 (`boot_time_start_s` など) を持つイベントもある。元データの中で同じ時刻を正確に引くときに使用
- `time.segment_origin` は、解析ごとのセグメント先頭の定義の違い (数十 ms) を示す補助情報。秒単位で扱う場合は気にしなくてよい

### source

`source` の `video_uri`・`can_uri`・`imu_uri`・`gnss_uri` から、元の映像・CAN・IMU・GNSS のファイルに戻れる。API でイベントを取得したあと、そのイベントの映像や CAN をそのまま確認できる設計。

- 映像は `video_frame_start`・`video_frame_end` で該当区間を頭出しできる (誤差は ±2 フレーム程度)
- URI は現在、データ置き場のルートからの相対パス (例 `comma2k19/Chunk_2/…/video.hevc`) 。保存先 (S3 など) が決まれば、`s3://…` のように単独で解決できる形で出力できる

---

## 6. 拡張方法・注意事項

- 新しい event_type は、トップレベルの schema を変えずに追加可能
- 新しい event_type の固有値は `attributes` に入れる
- 必要になれば、event_type ごとの `attributes` の schema を別に定義できる
- 値が存在しない optional 項目は省略
- この canonical schema は API でやり取りする形式の定義であり、DB の物理スキーマを指定するものではない

今後、強い横風などのイベントを追加する場合も、共通項目はそのまま利用し、event_type と attributes を追加する想定。

---

### 参考: 提供ファイル

```text
near_miss_event.schema.json      共通形式の JSON Schema
near_miss_api_data_spec.md       この文書
samples/
  hard_braking.json
  hard_steering.json
  road_vibration.json
  slip.json
```
