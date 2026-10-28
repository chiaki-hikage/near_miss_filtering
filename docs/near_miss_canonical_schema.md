# ヒヤリハット候補の canonical event schema

解析系統ごとに別々の CSV を、API 連携用の共通形式 (canonical event) に変換する仕様。
出力は **API 利用者向けの canonical event** と、**開発・ETL 検証向けの provenance** の 2 つに分ける。

| もの | 場所 |
|---|---|
| スキーマ (JSON Schema 2020-12) | `schemas/near_miss_event.schema.json` (`schema_version` = `1.0`) |
| 変換スクリプト | `scripts/export_near_miss_canonical.py` (設定 `configs/near_miss_canonical.yaml`) |
| 種別に依存しない部品 | `src/near_miss/canonical_event.py` (ID・丸め・パス・出どころの記録・検証) |
| 出力 (API 向け) | `out/near_miss_canonical/events.jsonl`・`sample_events.json` |
| 出力 (開発・ETL 向け) | `out/near_miss_canonical/provenance.jsonl`・`conversion_report.json` |
| 試験 | `tests/test_near_miss_canonical.py` |

```bash
uv run --extra api python scripts/export_near_miss_canonical.py
uv run --extra api python scripts/export_near_miss_canonical.py --uri-base s3://<バケット>/near_miss/   # 単独で解決できる URI にする
uv run --extra api --extra dev python -m pytest tests/test_near_miss_canonical.py
```

現状のデータ (どのファイルに何があるか) は [near_miss_data_inventory.md](near_miss_data_inventory.md) を見ること。

---

## 1. 目的と 2 つの出力

既存の CSV は系統ごとに列名・時刻の基準・ID の付け方が違い、解析者でないと元の映像・CAN にたどり着けない。
**既存の CSV はそのまま残し**、そこから次の 2 つを機械的に作る。

```
既存 CSV (各解析の元出力、形式は変えない)
    │  scripts/export_near_miss_canonical.py (読むだけ)
    ├──▶ canonical event   events.jsonl      API 利用者向け
    │      event_id・event_type・dataset・drive_id・segment・time・location・score
    │      detection (何を根拠に) / source (元データはどこか) / review / attributes (種別ごとの値)
    │
    └──▶ provenance        provenance.jsonl  開発・ETL 検証向け (event_id で 1 対 1 に対応)
           変換元の CSV と行・元 CSV の ID (E0818 など)・導出した項目・欠けている項目・内部の値
```

| | canonical event | provenance |
|---|---|---|
| 読む人 | API 利用者・業務 | 開発者・ETL の検証 |
| 目的 | イベントを特定し、いつ・どこで・何が・どの程度・何を根拠に起きたか、元データはどこかを知る | どの CSV のどの行から作ったか、どの値を計算で出したかを追い、再現する |
| 中身 | 共通項目・`detection`・`source`・`review`・`attributes` | `origin` (変換元)・`derived_fields`・`missing_fields`・`internal`・`fields` |
| 値が無いとき | 項目ごと省く (null を入れない) | `missing_fields` に名前を挙げる |
| 形式の保証 | JSON Schema で検証する | 形式は変換スクリプトが決める (API の契約には含めない) |

**推測で値を埋めない。** 元 CSV に無い値は canonical event では省き、provenance の `missing_fields` に記録する。

---

## 2. canonical event の項目

### 2.1 必須項目

| キー | 型 | 意味 |
|---|---|---|
| `schema_version` | string | `"1.0"` |
| `event_id` | string | 検出結果単位の stable ID (3 章) |
| `event_type` | string | `hard_braking`・`hard_steering`・`slip`・`road_vibration` など。snake_case で増やせる (列挙はしない) |
| `detector` | string | 検出ロジックの名前。`l3_rules`・`slip_direct_evidence`・`slip_sideslip_beta`・`road_vibration_locate` |
| `dataset` | string | `comma2k19`・`comma_car_segments` |
| `drive_id` | string | 走行の識別子。comma 系では `<dongle_id>\|<route>` |
| `time` | object | 4 章。`base`・`start_sec`・`end_sec` は必須 |
| `detection` | object | 7 章。`method` は必須 |
| `source` | object | 5 章。中身は元データにあるものだけ (commaCarSegments は `can_uri` だけ) |
| `attributes` | object | 6 章。固有値が無い種別では `{}` |

`detection`・`source`・`attributes` は、中身が少なくても**必ずある object** にした。
どのイベントでも `event.source.can_uri` や `event.attributes.xxx` の形で同じ書き方で読めるようにするため (存在の確認が 1 段で済む)。

### 2.2 optional 項目 (値があるときだけ出す)

| キー | 型 | 意味 |
|---|---|---|
| `vehicle` | string | 車種 (`configs/vehicles/*.yaml` の `name`) |
| `chunk` | string | データセットの分割単位 (comma2k19 の `Chunk_N`)。commaCarSegments には無い |
| `segment` | integer / string | 元データの時系列区間の識別子。comma 系では 60 秒セグメントの番号。区間の概念が無いデータセットでは省く |
| `location` | object | `latitude`・`longitude` (この 2 つは揃って出す)・`heading_deg`・`gnss_acc_m` |
| `speed_mps` | number | 車速 [m/s]。どの時点の車速かは種別で違う (6 章) |
| `score` | object | `{name, value}`。種別ごとに意味が違うので、必ず名前と組 (7 章) |
| `grade` | string | 元データに明示の等級がある場合のみ |
| `candidate_flag` | boolean | 元データの明示値か、文書化した規則で出せる場合のみ |
| `review` | object | 人手・映像の判定。元の表記のまま |

**null は入れない。** スキーマは null を受け付けない (`"type": "number"` などで、null を許していない)。

### 2.3 例

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
  "attributes": { "vertical_accel_peak_mps2": 4.155, "roll_rate_peak_dps": 9.349, "concentration": 0.758, "...": "..." }
}
```

---

## 3. event_id

```
key       = "\n".join([dataset, drive_id, str(segment), quantize_sec(start_sec), event_type, detector])
event_id  = "nm_" + sha256(key.encode("utf-8")).hexdigest()[:16]
```

| 要素 | 内容 |
|---|---|
| `dataset`・`drive_id`・`segment`・`event_type`・`detector` | canonical の値そのもの |
| `start_sec` | `time.start_sec`。**10 ms 単位に、10 進表現で四捨五入** (`ROUND_HALF_UP`)。`repr(float)` を `Decimal` にしてから丸めるので、`46.205` は `46.21` (2 進の丸めの `46.20` にはならない)。`-0.00` は `0.00` |
| 区切り | 改行 (`drive_id` が `\|` を含むため) |
| 長さ | SHA-256 の先頭 16 桁 (64 bit)。書き出し時に重複を検査して止める (現在 2,621 件で重複なし) |

**本 schema の `event_id` は「検出結果単位」の ID であり、複数の event_type / detector が同一の実世界事象を指す場合に、それらを束ねる physical event ID ではない。**
同じ急制動に `hard_braking` と `slip` が付けば 2 つの ID になる。スリップの 2 つの検出器 (直接証拠・横滑り角) が同じ区間を拾った場合も別の ID になる (`detector` を ID の要素に入れたのはこのため)。

- 同じ CSV から作り直せば同じ ID になる (試験で、書き出しを 2 回して出力が一字一句同じことを確かめている)
- 再解析で開始時刻が 10 ms 単位で変わったら別の ID になる
- 元 CSV の ID (`E0818` など、実行ごとに振り直す連番) は provenance の `origin.source_ids` にある
- 将来、実世界の事象で束ねる必要が出たら、上位の概念を足せる。今の event をその下の「検出イベント」として残せば、共通 schema は変えずに済む

```
physical_event_id   (将来。今は無い)
  ├─ detection event nm_…  (hard_braking / l3_rules)
  ├─ detection event nm_…  (slip / slip_direct_evidence)
  └─ detection event nm_…  (hard_steering / l3_rules)
```

---

## 4. 時刻 (time)

| キー | 必須 | 意味 |
|---|---|---|
| `base` | ○ | `segment_relative_s` = `segment` の先頭からの秒 |
| `start_sec` / `end_sec` | ○ | 開始・終了。60 秒を超えたら次のセグメント、負なら前のセグメントにまたがる |
| `peak_sec` | | ピーク。元データに無い種別では省く |
| `utc` | | 代表時刻の UTC (ISO 8601) |
| `boot_time_start_s` / `boot_time_end_s` | | 元ログのタイムスタンプ (comma 系では端末の boot time [s]) |
| `segment_origin` | | segment の 0 秒をどの系列で決めたか (下記) |

`boot_time_*` を API に残した理由: comma 系の元ファイル (CAN・IMU・GNSS) の時刻はすべて boot time なので、元データの中で同じ時刻を正確に引くのに要る。

### segment_origin (v1 では互換のために残す)

セグメントの 0 秒の定義が解析系統で違う。

| `segment_origin` | 定義 | 系統 |
|---|---|---|
| `can_primary_overlap_start` | CAN の主系列が共通して始まる時刻 (`near_miss.pipeline._annotate_segment`) | `l3_rules`・`slip_*` |
| `imu_accel_first_sample` | IMU 加速度の最初のサンプル (`scripts/vibration/find_roll_bumps.py` の `load_seg`) | `road_vibration_locate` |

どちらも映像の先頭との差は 10〜75 ms (20 セグメントで実測)。秒単位の利用ではこの差を気にしなくてよい。

**本来は変換時に共通の基準 (例えば映像の先頭) へそろえ、`segment_origin` は無くしたい。** v1 でそうしなかった理由:

- 異常振動の出力には boot time が無く、そろえるには元データ (`processed_log/IMU`・`global_pose/frame_times`) を読み直す必要がある
- 基準を変えると `start_sec` が変わり、event_id も変わる

そのため v1 では値を変えずに基準の名前だけを明示している。将来の版で吸収する内部差異として扱う。

### 変換規則

| 検出器 | `start_sec` | `end_sec` | `peak_sec` | `utc` |
|---|---|---|---|---|
| `l3_rules`・`slip_direct_evidence` | `t_in_segment_s` | `round(t_in_segment_s + (t_end − t_start), 2)` | (無し) | (無し) |
| `slip_sideslip_beta` | 同上 | 同上 | `round(t_in_segment_s + (t_peak − t_start), 2)` | (無し) |
| `road_vibration_locate` | `t_start_s` | `t_end_s` | `t_peak_s` | `event_utc` (ISO 8601 に表記を変更) |

- `t_start`・`t_end`・`t_peak` は boot time (元 CSV の列名のまま `boot_time_*` に入れる)
- 異常振動の `segment` は前輪が当たった時刻のセグメントなので、`start_sec` が負のことがある (最小 −0.63 s)。`utc` もその時刻のもの
- 横滑り角の `peak_sec` は、山が開始と同じセグメントにある 208 件で元 CSV の `peak_t_in_segment_s` と 0.01 s 以内で一致することを書き出し時に確かめている

---

## 5. 元データ参照 (source)

| キー | 内容 |
|---|---|
| `video_uri`・`can_uri`・`imu_uri`・`gnss_uri` | 元のファイル。データセットに無いものは省く |
| `video_frame_start`・`video_frame_end` | 前方映像のフレーム番号 (comma2k19 のみ) |

### URI の形

URI は **データのルートからの相対パス**で、先頭がデータセット名になる。S3 の配置 (`docs/environment.md`) と同じ形にしてある。

| データセット | 映像 | CAN | IMU | GNSS |
|---|---|---|---|---|
| comma2k19 | `comma2k19/<chunk>/<drive_id>/<segment>/video.hevc` | `…/processed_log/CAN` | `…/processed_log/IMU` | `…/global_pose` |
| commaCarSegments | (無い) | `comma_car_segments/segments/<dongle>/<route>/<segment>/rlog.zst` | (無い) | (無い) |

解決のしかた:

| 置き場所 | `comma2k19/…` | `comma_car_segments/…` |
|---|---|---|
| S3 | `s3://<バケット>/near_miss/comma2k19/…` | `s3://<バケット>/near_miss/comma_car_segments/…` |
| 手元 (Mac) | `raw_data/…` (チャンクが `raw_data/` の直下) | `raw_data/comma_car_segments/…` |
| 公開元 | comma2k19 の配布物 | `https://huggingface.co/datasets/commaai/commaCarSegments/resolve/main/…` |

- **相対パスは、今のデータ配置に依存する内部表現である。** API で外に出すなら、`--uri-base s3://<バケット>/near_miss/` を付けて書き出し、それだけで解決できる URI (`s3://…`) にするのが望ましい。バケットが決まっていないので、既定では付けていない
- 前の案にあった `source.root` は、URI の先頭にデータセット名を含めたので無くした
- パスはすべて配置規則から導出した値で、元 CSV にパスの列は無い (provenance では `constant` として記録)
- comma2k19 の `chunk` は元 CSV に無い (異常振動を除く)。書き出し時に `raw_data/Chunk_*/<drive_id>/` の配置から引く
- 書き出し時に、生成したパスが手元に実在するかを確かめている (今回は全件が実在。`conversion_report.json` の `uri_check`)

### フレーム番号

```
video_frame_start = 元 CSV の video_frame があればその値。無ければ round(start_sec × 20)
video_frame_end   = round(end_sec × 20)
0〜1199 (セグメントの外) に入らなければ省く
```

セグメントの 0 秒と映像の先頭が 10〜75 ms ずれるので、**±2 フレーム程度の近似**。正確なフレームは `global_pose/frame_times` (boot time) で引ける。

---

## 6. attributes (event_type ごとの固有値)

**自由な構造の object** で、event_type ごとに中身が違う。トップレベルの schema を変えずに、新しい種別の値を足せる。
元 CSV にある値だけを入れ、欠測の値は項目ごと省く。`'a|b'` の形の列は配列にする。

v1 では attributes の中身を schema で縛っていない。**将来は event_type ごとの attributes schema に分けられる**:

```
schemas/
  near_miss_event.schema.json      共通部分 (今のもの)
  attributes/
    hard_braking.schema.json
    slip.schema.json
    road_vibration.schema.json
    crosswind.schema.json          (将来)
```

### 6.1 hard_braking / hard_steering (`l3_rules`)

| キー | 元の列 | 備考 |
|---|---|---|
| `source_event_type`・`label` | `event_type`・`label` | L3 の種別名と日本語名 |
| `platform` | `platform` | commaCarSegments のみ (車種キー) |
| `ax_min_mps2` (急ブレーキ) | `peak_value` | イベント区間の前後加速度の最小 [m/s²] |
| `lateral_accel_mps2` (急操舵) | `peak_value` | イベント区間の横加速度 (`v × yaw_rate`) の極値 (符号付き) [m/s²] |
| `peak_value`・`peak_feature` (上記以外の L3 種別) | `peak_value`・`feature` | 名前を決めていない種別は、判定量の名前と組で入れる |
| `threshold`・`exceedance`・`duration_s` | 同名 | 閾値・超過量・区間の長さ |
| `candidate_window.*` | 候補の CSV | **イベントを含む候補 (前後 2 s を足して統合した区間) の値**。イベントそのものの値ではない |

`candidate_window` の中身 (列が元に無い・欠測なら省く):

| キー | 元の列 | 種別 |
|---|---|---|
| `speed_mean_mps`・`speed_min_mps`・`severity`・`n_events`・`event_types` | `v_mps_mean`・`v_mps_min`・`severity`・`n_events`・`event_types` | 共通 |
| `ax_min_mps2`・`jerk_min_mps3`・`jerk_win_min_mps3`・`jerk_win_mean_min_mps3`・`brake_pressed_max`・`gvc_min_mps2`・`gas_pedal_max_pct`・`decel_flag_max` | `ax_mps2_min`・`jerk_mps3_min`・`jerk_win_min_min`・`jerk_win_mean_min`・`brake_pressed_max`・`gvc_mps2_min`・`gas_pedal_pct_max`・`decel_flag_max` | 急ブレーキ |
| `brake_mc_max_mpa`・`brake_position_max`・`abs_active_max`・`vsc_active_max`・`precollision_active_max`・`precollision_force_min_n`・`acc_braking_max` | `brake_mc_mpa_max` ほか | 急ブレーキ (commaCarSegments のみ) |
| `steering_rate_dps`・`lateral_accel_mps2`・`lateral_accel_can_mps2`・`yaw_rate_dps`・`lateral_jerk_mps3`・`lateral_jerk_win_mps3`・`steer_reversals_max` | `steer_rate_dps_absmax`・`ay_kin_mps2_absmax`・`ay_can_mps2_absmax`・`yaw_rate_dps_absmax`・`lat_jerk_mps3_absmax`・`lat_jerk_win_absmax_max`・`steer_reversals_max` | 急操舵。`_absmax` は絶対値が最大の時点の**符号付き**の値 |
| `steer_torque_driver` | `steer_torque_driver_absmax` | 急操舵 (commaCarSegments のみ) |

### 6.2 slip (`slip_direct_evidence`)

| キー | 元の列 |
|---|---|
| `platform` | `counts.json` の `label` |
| `evidence`・`mechanisms` (配列)・`corroboration` | 同名 |
| `n_direct`・`n_corroborated`・`n_mechanisms` | 同名 |
| `slip_warn`・`abs_active`・`vsc_active`・`wheel_spread`・`wheel_decel`・`lateral`・`yaw_residual`・`tc_disabled` (bool) | 同名 |
| `slip_warn_peak`・`wheel_speed_diff_peak_mps`・`wheel_decel_peak_mps2`・`speed_max_mps` | `slip_warn_peak`・`ws_spread_peak_mps`・`wheel_decel_peak_mps2`・`v_max_mps` |
| `at_peak.{ax_mps2, ay_mps2, yaw_rate_dps, steering_angle_deg, sideslip_angle_model_deg}` | `*_at_peak_*`・`beta_model_peak_deg` |

`friction_demand`・`physics_consistency` は元 CSV に無いので入れていない。

### 6.3 slip (`slip_sideslip_beta`)

| キー | 元の列 |
|---|---|
| `platform` | `counts.json` の `label` |
| `sideslip_angle_peak_deg`・`sideslip_angle_peak_sigma`・`sideslip_angle_excess_peak_deg`・`sideslip_angle_rate_peak_dps` | `beta_peak_deg`・`beta_sigma_peak`・`beta_excess_peak_deg`・`beta_rate_peak_dps` |
| `yaw_residual_peak_sigma`・`ay_peak_mps2`・`ax_at_peak_mps2`・`n_runs` | `yaw_sigma_peak` ほか |
| `confidence_items`・`reasons`・`corroboration` (配列) | 同名 |
| `peak_segment`・`peak_sec_in_peak_segment` | `peak_segment`・`peak_t_in_segment_s` |

### 6.4 road_vibration (`road_vibration_locate`)

| キー | 元の列 (ファイル) |
|---|---|
| `hit_sec` | `event_time_s` (locate)。前輪が凹凸に当たった推定時刻 |
| `position_error.{along_m, cross_m}` | `pos_err_along_m`・`pos_err_cross_m` (locate)。位置の誤差の目安 |
| `vertical_accel_peak_mps2`・`roll_rate_peak_dps`・`pitch_rate_peak_dps` | `heave_peak_abs`・`roll_peak_abs`・`pitch_peak_abs` (near_miss) |
| `heave_roll_dt_s`・`concentration`・`rise_s`・`side_index`・`repeatability`・`feature_type` | 同名 (near_miss) |
| `conditions.*` (bool) | 5 条件 + `vibration_event`・`eleven_hz_only` (near_miss) |
| `A`・`heave_ratio`・`roll_ratio`・`pitch_ratio`・`roll_over_pitch`・`wheel_ratio_max`・`wheel_ratio.{FL,FR,RL,RR}`・`wheel_lr_log2`・`wheel_side`・`wheel_front_rear_lag_s`・`wheel_lag_expected_s`・`heave_dom_hz`・`vibration_score`・`site_event_count`・`gps_dt_s`・`ay_range_mps2` | locate |
| `band_ratio.{heave_3_8hz, heave_8_15hz, roll_3_8hz, roll_8_15hz}`・`flags.*` | locate |
| `physics.signal_count`・`physics.signals.{P2,P4,P5,P6}` | `physics_vs_video_sites.csv` (**地点単位**、138 件) |
| `physics.consistency`・`physics.inferred_side`・`physics.site.{P1〜P6}` | `site_physics_validation.csv` (**地点単位**、152 件) |
| `physics.event.{P1〜P5, inferred_input}` | `event_physics_validation.csv` (**イベント単位**、152 件) |
| `iso2631.*` | `iso2631_events.csv` (単位を名前に付けた) |

物理評価は候補の判定には使っておらず、評価した一部の地点・イベントにしか無い。P の値は元の文字列 (`true`・`false`・`uncertain`・`—`) のまま。
地点番号 (`site_id`) と実行内の順位 (`A_rank` など) は、実行ごとに変わるので provenance に移した。

---

## 7. score・grade・candidate_flag・detection・review

| 検出器 | `score` | `grade` | `candidate_flag` |
|---|---|---|---|
| `l3_rules` | `candidate_severity` = イベントを含む候補の `severity` | (無し) | イベントを含む候補の `final` (`L1 AND sev_ok AND priority`) |
| `slip_direct_evidence` | (無し。数値の点が無い) | `grade` (`multi_evidence` など) | (無し。明示の列が無い) |
| `slip_sideslip_beta` | `sideslip_confidence` = `confidence` | `grade` | (無し) |
| `road_vibration_locate` | `near_miss_candidate_score` | (無し) | **導出**: 5 条件の AND (21 件が true) |

「イベントを含む候補」は、`drive_id` (と `platform`) が同じで、候補の区間がイベントの区間を含むもの。全 1,186 件でちょうど 1 つに決まる。

| `detection` | `l3_rules` | `slip_direct_evidence` | `slip_sideslip_beta` | `road_vibration_locate` |
|---|---|---|---|---|
| `method` | `rule_threshold` | `direct_evidence` | `sideslip_beta_model` | `rule_threshold` |
| `rule` | `trigger_rule` | `evidence` (立った証拠) | `reasons` (通過の理由) | `A = sqrt(...) >= 3.0` (設定ファイルが出力時と同じことを確かめて値を入れた) |
| `criteria` | `feature`・`op`・`threshold`・`gate`・`stage`・`weight` | — | — | `candidate_rule` |
| `config_hash` | `config_hash` | `config_hash` | `config_hash` | locate の `config_sha1` |
| `created_at` | `run_meta.json` / `screen_meta.json` の `run_at` | `counts.json` の `run_at` | 同左 | (無し) |
| `algorithm_version` | (無し) | (無し) | (無し) | (無し) |

`algorithm_version` は、既存の出力がどの git commit で作られたか確定できないので入れていない。
出力を作り直した直後に限り、`--stamp-git-commit` で現在の commit を入れられる。

`review` は、L3 (comma2k19) では `labels.csv` の `risky`・`verdict`・`note` (33 件中 23 件)、異常振動では `video_road_feature` が入る。
`verdict` の表記ゆれ (`リスク高` と `リスク高い` など) は書き換えずに元の値を入れている。正規化した値を足すときは別のキーにする。

---

## 8. provenance (開発・ETL 検証向け)

`provenance.jsonl` は、canonical event と**同じ順・同じ件数**で、`event_id` で 1 対 1 に対応する。API のレスポンスには含めない。

| キー | 内容 |
|---|---|
| `event_id` | canonical event と同じ |
| `origin.file`・`origin.row` | 変換元の CSV と、そのデータ行の番号 (0 始まり、`pandas.read_csv` の行番号) |
| `origin.joined_files` | 突き合わせた CSV (候補・ラベル・物理評価など) |
| `origin.source_ids` | 元 CSV の ID (`event_id`: `E0818`、`site_id`: 7、L3 のブロックの `block_segment_id` など。実行ごとに変わる) |
| `derived_fields` | 計算で出した項目と式 (`time.end_sec`・`source.video_frame_*`・`speed_mps`・`candidate_flag`・`event_id`) |
| `missing_fields` | canonical event で省いた項目 (元データに無いもの) |
| `internal` | API に出さない内部の値 (下表) |
| `fields` | canonical event の全項目の出どころ (`copied` 元の列 / `derived` 式 / `constant` 設定・配置規則) |

前の案で canonical event にあり、provenance (`internal` など) へ移したもの:

| もの | 理由 |
|---|---|
| `origin` (変換元の CSV・行・元 ID) | 変換処理の追跡用 |
| `source.root` | URI にデータセット名を含めたので不要 (削除) |
| `source.review_media` (`out/` の確認用動画) | 解析者の手元のパスで、API 利用者は開けない |
| `detection.reference` (docs のパス)・`detection.config_hashes` | 解析内部の文書・設定 |
| `review.label_source`・`review.video_review_source` | ラベルのファイルのパス |
| `attributes.l1_pass`・`l1_kinematics_pass` | 解析パイプラインの段のフラグ |
| `attributes.candidate_window.boot_time_*` | 候補区間の内部の時刻 |
| `attributes.site.*`・`A_rank`・`near_miss_rank` | 実行ごとに変わる番号・順位 |
| `location.pos_err_*` | 異常振動に固有の推定なので `attributes.position_error` へ |

`conversion_report.json` は全体の集計 (入力ファイルの SHA-1、検出器ごとの出どころと欠損の件数、突き合わせの検査、スキーマ検証、URI の実在)。

---

## 9. 欠損 (今回の 2,621 件で省いた割合)

| 項目 | 割合 | 理由 |
|---|---|---|
| `detection.algorithm_version` | 100% | 既存の出力の commit は確定できない |
| `grade` | 85% | 等級を持つのはスリップだけ |
| `location`・`time.utc` | 60% | 異常振動以外は持たない。commaCarSegments は元データにも無い |
| `review` | 60% | ラベル・映像判定があるのは L3 (comma2k19) の一部と異常振動だけ |
| `source.video_uri`・`imu_uri`・`gnss_uri`・`chunk` | 59% | commaCarSegments に映像・IMU・GNSS・チャンクは無い |
| `source.video_frame_*` | 59% | 同上と、セグメントの外 (異常振動の 8 / 14 件など) |
| `time.peak_sec` | 52% | L3 とスリップ直接証拠の元 CSV にピーク時刻が無い |
| `speed_mps` | 45% | L3 のイベント行に車速が無い (候補の平均は `attributes.candidate_window.speed_mean_mps`) |
| `time.boot_time_*`・`detection.created_at` | 40% | 異常振動の CSV・実行記録に無い |
| `candidate_flag` | 15% | スリップは明示の列が無い |
| `score` | 7% | スリップ直接証拠は数値の点が無い |

元データから再計算すれば埋められるもの (今回はしていない): comma2k19 の L3・スリップの位置・UTC・イベント時点の車速、L3 とスリップ直接証拠のピーク時刻、正確なフレーム番号。

---

## 10. 新しい event_type (横風など) を足す手順

共通の schema は変えずに足せる (`event_type` は列挙していない)。

1. 検出スクリプトの出力 CSV に、最低限 `dataset`・`drive_id`・区間の識別子 (`segment` など)・開始秒・終了秒 (か長さ)・`config_hash` を持たせる。ピーク・車速・位置があれば列にしておく
2. `scripts/export_near_miss_canonical.py` に変換関数 `convert_<kind>(name, dc, ctx)` を足し、`CONVERTERS` に登録する。
   `base_record`・`empty_blocks`・`put_time_boot`・`put_source`・`put_frames` を使い、値は `rec.copy` / `rec.derive` / `rec.const` / `opt` で入れる
   (出どころが provenance に自動で残る)。API に出さない値は `_internal.` の下に入れると provenance へ移る
3. `configs/near_miss_canonical.yaml` の `detectors` に `kind`・`detector` (例 `crosswind_rules`)・`event_type` (例 `crosswind`)・入力のパスを書く
4. 固有の値は `attributes` に入れ、この文書の 6 章に表を足す (必要なら `schemas/attributes/crosswind.schema.json` を作る)
5. `tests/test_near_miss_canonical.py` に、元 CSV の主要値が写っていることの試験を足す
6. 書き出して、`conversion_report.json` の `checks`・`validation` を確かめる

共通項目の意味を変えるとき (キーを足す・必須にするなど) は、`schema_version` を上げて schema を改版する。
