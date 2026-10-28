# API 実装担当者への回答案 (ヒヤリハット候補データの現状と連携の方針)

根拠:

- 現状の棚卸し: [near_miss_data_inventory.md](near_miss_data_inventory.md) (2026-10-28 時点で `out/` を実際に開いて確かめたもの)
- 連携用の共通形式: [near_miss_canonical_schema.md](near_miss_canonical_schema.md)

構成: 1 章が回答の本文 (そのまま送れる形)、2 章が渡すファイル、3 章が JSON の例、4 章が送る前に決めること。

---

## 1. 回答

> ヒヤリハット候補データの現状と、API 連携に向けた方針をお答えします。
>
> **1. 現状**
> DB はありません。解析を実行するたびに、イベント種別 (急ブレーキ・急操舵などの既存ルール、スリップ、路面凹凸による異常振動) ごとに CSV を出力しています。
> 種別ごとに列名・時刻の基準・ID の付け方がそろっていません。
>
> **2. API 連携の方針**
> 解析の CSV はそのまま残し、そこから **canonical event (共通形式の JSON)** に正規化してお渡しします。変換処理とスキーマは用意済みです。
>
> | 方針 | 内容 |
> |---|---|
> | 共通項目 | `event_id`・`event_type`・`detector`・`dataset`・`drive_id`・`time` を全イベントに持たせます。`segment`・位置・車速・スコアなどは、値があるときだけ出します (null は入れません) |
> | 固有値 | 種別ごとの値は `attributes` に入れます。種別が増えても共通部分は変わりません |
> | stable event_id | 同じ入力から作り直せば同じ ID になります。「検出結果単位」の ID で、同じ出来事に急ブレーキとスリップが付けば別の ID です |
> | 時刻 | セグメント (60 秒) の先頭からの秒 (`start_sec`・`end_sec`・`peak_sec`) にそろえます。UTC と元ログの時刻 (boot time) も、あるものは持たせます |
> | 元データ | `source` の `video_uri`・`can_uri`・`imu_uri`・`gnss_uri` と映像のフレーム番号から、元の映像・CAN などに戻れます |
> | 判定の根拠 | `detection` に、方法・規則・設定のハッシュを持たせます |
> | 変換の記録 | 「どの CSV のどの行から作ったか」などの変換の記録 (provenance) はこちらで管理し、API のレスポンスとは分けます |
>
> 形式は JSON Schema (`near_miss_event.schema.json`) で定義していて、今回の 2,621 件はすべて検証を通っています。
>
> **3. 対応している種別と件数**
>
> | event_type | 検出器 | データセット | 件数 |
> |---|---|---|---|
> | hard_braking | 既存ルール | comma2k19 / commaCarSegments | 18 / 1,015 |
> | hard_steering | 既存ルール | comma2k19 / commaCarSegments | 15 / 138 |
> | slip | 直接証拠 (ABS・VSC・輪速など) | commaCarSegments | 187 |
> | slip | 横滑り角 | comma2k19 / commaCarSegments | 13 / 198 |
> | road_vibration | 路面凹凸による異常振動 | comma2k19 | 1,037 (うち候補 21) |
>
> 既存ルールの他の種別 (蛇行・車線変更・車間など) も、同じ形式で出せることを確かめています。今回の出力には入れていません。
>
> **4. 一部のイベントにしか無い項目**
> 元データの制約によるもので、推測で埋めてはいません。
>
> - 位置 (緯度経度)・UTC: 異常振動だけ。commaCarSegments には元データにも位置がありません
> - 映像・IMU・GNSS: comma2k19 だけ。commaCarSegments のイベントは `source` が `can_uri` だけです
> - ピーク時刻: 既存ルールとスリップ直接証拠には無いので省いています
> - コードの版 (git commit): 既存の結果では確定できないので省いています
>
> **5. 元データへの戻り方**
> `source.*_uri` は、データのルートからの相対パス (例 `comma2k19/Chunk_2/<drive_id>/16/video.hevc`) です。S3 の配置と同じ形なので、`s3://<バケット>/near_miss/` を前に付ければ届きます。
> バケットが決まれば、`s3://…` の形で出力することもできます。
> comma2k19 では `video_frame_start`・`video_frame_end` で映像を頭出しできます (誤差は ±2 フレーム程度)。
>
> **6. お渡しするもの**
> 設計の検討には、次の 2 つで足りるはずです。元の CSV は不要です。
>
> 1. スキーマ `schemas/near_miss_event.schema.json` と仕様書 `docs/near_miss_canonical_schema.md`
> 2. サンプル `out/near_miss_canonical/sample_events.json` (種別ごとに 3 件、計 12 件)
>
> 必要なら、全件 `events.jsonl` (2,621 件、約 5.5 MB) もお渡しします。
>
> **7. 今後**
> API 側のスキーマは、この共通形式をたたき台にして相談させてください。
> 横風など新しいイベントは、`event_type` を足し、固有の値を `attributes` に入れる形で、共通部分を変えずに追加できます。
> 種別ごとの `attributes` の形は、必要になれば別のスキーマに分けて定義できます。

---

## 2. 渡すファイル

| 優先 | ファイル | 中身 | 対象 |
|---|---|---|---|
| 1 | `schemas/near_miss_event.schema.json` | 共通形式の JSON Schema (2020-12) | API 担当者 |
| 1 | `docs/near_miss_canonical_schema.md` | 共通項目・ID・時刻・URI の規則・種別ごとの attributes・欠損 | API 担当者 |
| 1 | `out/near_miss_canonical/sample_events.json` | 種別ごとに 3 件 (計 12 件) | API 担当者 |
| 2 | `out/near_miss_canonical/events.jsonl` | 全 2,621 件 (1 行 1 件) | API 担当者 (必要なら) |
| — | `out/near_miss_canonical/provenance.jsonl`・`conversion_report.json` | 変換元の CSV と行・導出・欠損の記録 | こちらで管理 (渡さない) |
| — | `out/near_miss_api_samples/*_sample.csv`・`docs/near_miss_data_inventory.md` | 現状の CSV と棚卸し | 聞かれたときの補足 |

`out/` は git の管理外なので、渡すときは別に送る必要がある。

---

## 3. JSON の例 (canonical event)

`out/near_miss_canonical/sample_events.json` の 1 件。`attributes` は一部だけを載せた。

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
    "base": "segment_relative_s", "segment_origin": "imu_accel_first_sample",
    "start_sec": 46.2, "end_sec": 46.72, "peak_sec": 46.45, "utc": "2018-10-06T05:37:18Z"
  },
  "location": { "latitude": 37.726728, "longitude": -122.471957, "heading_deg": 2.5, "gnss_acc_m": 0.41 },
  "speed_mps": 14.722,
  "score": { "name": "near_miss_candidate_score", "value": 73.6 },
  "candidate_flag": true,
  "detection": {
    "method": "rule_threshold",
    "rule": "A = sqrt(heave_ratio * max(roll_ratio, pitch_ratio)) >= 3.0",
    "criteria": { "candidate_rule": "strong_vertical AND strong_roll AND simultaneous_vertical_roll AND impulsive_response AND road_related_evidence" },
    "config_hash": "ce06a1d3af"
  },
  "source": {
    "video_uri": "comma2k19/Chunk_2/b0c9d2329ad1606b|2018-10-05--22-20-31/16/video.hevc",
    "can_uri": "comma2k19/Chunk_2/b0c9d2329ad1606b|2018-10-05--22-20-31/16/processed_log/CAN",
    "imu_uri": "comma2k19/Chunk_2/b0c9d2329ad1606b|2018-10-05--22-20-31/16/processed_log/IMU",
    "gnss_uri": "comma2k19/Chunk_2/b0c9d2329ad1606b|2018-10-05--22-20-31/16/global_pose",
    "video_frame_start": 924, "video_frame_end": 934
  },
  "review": { "video_road_feature": "一致: マンホール・補修跡 (右輪の通り道)" },
  "attributes": {
    "vertical_accel_peak_mps2": 4.155,
    "roll_rate_peak_dps": 9.349,
    "concentration": 0.758,
    "rise_s": 0.27,
    "repeatability": "2 ドライブ以上でイベント",
    "physics": { "signal_count": 5, "consistency": "Strong physical consistency" },
    "iso2631": { "mtvv_total_mps2": 1.8137, "vdv_z_mps175": 2.3093, "comfort": "非常に不快" }
  }
}
```

- いつ: `time` (セグメント内の秒と UTC)
- どこで: `location`
- 何が・どの程度: `event_type`・`score`・`attributes`
- 何を根拠に: `detection`
- 元データはどこか: `source`

commaCarSegments のイベントは `location`・`chunk` が無く、`source` は `{"can_uri": "comma_car_segments/segments/<dongle>/<route>/<segment>/rlog.zst"}` だけになる。

新しい種別 (横風など) も同じ枠に入る。

```json
{
  "schema_version": "1.0",
  "event_id": "nm_…",
  "event_type": "crosswind",
  "detector": "crosswind_rules",
  "...": "共通項目は他の種別と同じ",
  "attributes": { "（横風の判定に使う値。未定）": "..." }
}
```

---

## 4. 送る前に決めること (こちら側)

- 渡し方: `out/` は git の管理外。スキーマ・仕様書・`sample_events.json` を zip で送るか、共有ストレージに置くか
- URI の形: S3 のバケットが決まれば `--uri-base s3://<バケット>/near_miss/` で単独で解決できる URI にして出し直す
- 人手ラベルの保管先: `out/chunk2_final/labels.csv` と `out/vibration/locate/review_video.csv` は git に入っておらず、消えると作り直せない
- ID の同一性: event_id は検出結果単位。再解析で開始時刻が 10 ms 単位で変わると別の ID になる。API 側で「同じ実世界の事象」を束ねる必要があるなら、上位の physical event ID を別に相談する (今の形式を変えずに足せる)
- 未変換の既存ルール (車線変更・車間など) も渡すか
