# 入出力仕様

各段の入力・出力の形を実装から書き起こしたもの。処理の意味は
[pipeline_overview.md](pipeline_overview.md)、閾値は
[event_detection_spec.md](event_detection_spec.md) を見ること。

**実装に無いものは書いていない。** 決まっていない箇所は「未定義」と明記する。

---

## 1. 共通の約束

### 1.1 タイムスタンプ

| 列 | 基準 | 用途 |
|---|---|---|
| `t_start` / `t_end` | **ルート先頭からの絶対秒** | 区間の突き合わせ |
| `t_in_segment_s` | **そのセグメント先頭からの相対秒** | 映像の頭出し |
| `t_rel_start` | ブロック先頭からの相対秒 | 連結処理の内部用 |
| `video_frame` | `t_in_segment_s × 20` の整数 | comma2k19 のコマ送り |

**`t_start` と `t_in_segment_s` を混同しない。** 映像はセグメント単位のファイル
なので、頭出しには `t_in_segment_s` が要る。区間の重なり判定には `t_start` を使う。

`video_frame` は comma2k19 でのみ出る。commaCarSegments には映像が無い。

セグメントは 60 秒。候補は前後に余白 2 秒を付けて統合するため、**60 秒境界を
またぐことがある**。またいだ場合、`segment` は候補の `t_start` が属するセグメント
になる。最も強いイベントの位置は別に `peak_segment` / `peak_t_in_segment_s` が出る。

### 1.2 欠測の扱い

* 受信間隔が `resample.max_gap_s` (0.5 s) を超えた箇所は**欠測**として扱い、
  補完しない
* 再サンプル後の両端 `edge_trim_s` (0.6 s) は平滑化の影響を受けるため捨てる
* 判定量が欠測の窓・サンプルは**条件を満たさない**とみなす。
  「判定できない」を「条件を満たす」にしない
* CSV 上は空欄 (pandas の `NaN`)

### 1.3 単位

| 量 | 単位 |
|---|---|
| 速度 | m/s (`*_mps`) |
| 加速度 | m/s² (`*_mps2`) |
| 躍度 | m/s³ (`*_mps3`) |
| 角速度 | deg/s (`*_dps`) |
| 角度 | deg |
| 時間 | s |
| 距離 | m |
| ブレーキ圧 | MPa (`brake_mc_mpa`) |
| アクセル開度 | % (`gas_pedal_pct`) |

**列名に単位の接尾辞が付く。** 接尾辞が無い列 (`slip_warn` / `abs_active_flag` など)
は無次元のフラグまたは多値の状態。

### 1.4 config_hash

`configs/detection.yaml` と車種設定から計算した 10 桁の十六進値。
**閾値を変えた実行は `config_hash` が変わる**ので、出力を混ぜても区別できる。
`events.csv` / `candidates.csv` / `run_meta.json` に入る。

---

## 2. 入力データ

### 2.1 comma2k19

| | |
|---|---|
| 形式 | ディレクトリ構成。`raw_can` (復号前の CAN)、`video.hevc`、`processed_log` |
| 車種 | Toyota RAV4 2017 (dongle `b0c9d2329ad1606b`) |
| 必須 | `raw_can`。IMU / GNSS は特徴抽出に使わない |
| 単位・周期 | [comma2k19_data.md](comma2k19_data.md) |

### 2.2 commaCarSegments

| | |
|---|---|
| 形式 | `rlog.zst` (openpilot ログ、capnp + zstd)。1 セグメント平均 1.38 MB |
| 車種 | 車種キーで分類。現在の対象は Toyota 4 車種 |
| 必須 | `rlog.zst` と `database.json` (セグメント一覧) |
| 映像 | **無い** |

**キャッシュは車種を問わず 1 つのディレクトリに溜まる。** `database.json` の
一覧と突き合わせないと他の車種が混ざる。別の車種の rlog を誤った CAN 定義で
読むことになるため、`find_segments` には必ず `platform` を渡す。

### 2.3 車種設定 (`configs/vehicles/*.yaml`)

| 節 | 内容 |
|---|---|
| `dongle_ids` / `platform` | どのデータに当てるか |
| `signals` | CAN ID・開始ビット・長さ・符号・係数・オフセット・バス |
| `derived` | 復号済み信号から作る合成信号 (`steer_deg` など) |
| `geometry` | ホイールベース、ステアリングギヤ比、アンダーステア勾配、トレッド幅 |
| `calibration.accel_sign` | 個体ごとの加速度センサの符号 |

**`geometry` が無い車種では `yaw_residual_sigma` と `beta_model_deg` が計算されない。**
現在この 2 つが揃っているのは RAV4 (comma2k19) と RAV4 TSS2 のみ。

---

## 3. L1 の出力

### 3.1 `scripts/l1/l1_apply_ccs.py` → `out/<name>/`

| ファイル | 内容 |
|---|---|
| `segments.csv` | セグメントごとの窓数・通過窓数・フラグの立った窓数・定常窓数 |
| `events.csv` | `direct_slip` の事象ごとの保持 (運動量のみ / フラグ込み) |
| `summary.csv` | 車種 × 設定ごとの通過率 |
| `meta.json` | 使った閾値・窓長・無条件通過フラグの一覧 |

主な列 (`segments.csv`)

| 列 | 意味 |
|---|---|
| `n_windows` | そのセグメントの 2 秒窓の数 |
| `n_gate` | 速度ゲート (`v ≥ 5`) を通った窓数 |
| `n_flag` | 安全系フラグが立った窓数 |
| `n_kin_<設定>` | 運動量の条件だけで通った窓数 |
| `n_total_<設定>` | フラグ込みで通った窓数 |
| `seg_pass_<設定>` | この 60 秒 segment に通過窓が 1 つ以上あるか |
| `n_steady` | 定常区間 (10 秒の車速レンジ < 1 m/s) の窓数 |

**`n_steady` は通過率を下げる処理ではない。** 「後段の decode / イベント判定を
省略できる走行時間」を別指標として数えたもの。

### 3.2 `scripts/l1/l1_l3_chain.py` → `out/<name>/`

| ファイル | 内容 |
|---|---|
| `events.csv` | L3 事象ごとの L1 通過 (`L1` / `L1_kin`) |
| `candidates.csv` | 候補ごとの `L1` / `sev_ok` / `priority` / `final_noL1` / `final` |
| `by_type.csv` | 種別ごとの L1 前後の件数と保持率 (事象単位) |
| `by_type_final.csv` | 最終候補の種別構成 (候補単位) |
| `stages.csv` | 段ごとの候補数 |
| `by_platform.csv` | 車種別の保持率 (車種が複数のときのみ) |

**`by_type.csv` は事象単位、`by_type_final.csv` は候補単位。** 1 候補に複数種別が
付くので、`by_type_final.csv` の合計は候補数と一致しない。

---

## 4. L3 の出力

### 4.1 `events.csv` — イベント 1 件 = 1 行

| 列 | 意味 |
|---|---|
| `dataset` | `comma2k19` / `comma_car_segments` |
| `event_type` | イベント名 (`hard_brake` など) |
| `label` | 日本語の表示名 |
| `feature` | 判定に使った列名 |
| `op` / `threshold` | 判定の向きと閾値 |
| `t_start` / `t_end` / `duration_s` | 区間 (ルート絶対秒) |
| `peak_value` | 区間内の判定量の極値 |
| `exceedance` | 閾値からの超過量 |
| `stage` | 1 = CAN の基本信号のみ / 2 = レーダ・安全系フラグを使う |
| `weight` | severity の重み |
| `gate` | 適用されたゲートの説明 |
| `rule` / `trigger_rule` | 判定式の文字列 |
| `drive_id` / `segment_id` / `segment` | どこの区間か |
| `vehicle` | 使った車種設定の名前 |
| `t_in_segment_s` / `video_frame` | 映像の頭出し用 |
| `config_hash` | 実行時の設定 |

**判定根拠 (`feature` / `op` / `threshold` / `rule`) を行に残す。** 後から
「なぜこれが出たか」を追えるようにするため。

### 4.2 `candidates.csv` — 候補 1 件 = 1 行

| 列 | 意味 |
|---|---|
| `drive_id` / `segment_id` / `segment` | どこの区間か |
| `t_start` / `t_end` / `duration_s` | 統合後の区間 (前後 2 秒の余白を含む) |
| `t_peak` / `peak_segment` / `peak_t_in_segment_s` | 最も強いイベントの位置 |
| `severity` | 統合スコア |
| `n_events` / `event_types` | 含まれるイベント数と種別 (`\|` 区切り) |
| `raw_can_loaded` | 生 CAN を読めたか |
| `config_hash` | 実行時の設定 |
| 集計列 (約 50 列) | 候補区間内の特徴量の代表値 |

集計列の接尾辞が集計方法を示す。

| 接尾辞 | 意味 |
|---|---|
| `_min` / `_max` | 区間内の最小・最大 |
| `_absmax` | 絶対値の最大 |
| `_mean` | 区間内の平均 |

例: `ax_mps2_min` / `ay_kin_mps2_absmax` / `thw_s_min` / `abs_active_flag_max`

### 4.3 `segments.csv` / `run_meta.json`

`segments.csv` はセグメントごとの処理結果 (`status`、イベント数、候補数、
各フラグが存在したか・立った最大値)。`run_meta.json` は実行時刻・データ根・
`config_hash`・段・走査セグメント数・イベント数・候補数。

### 4.4 ID の考え方

**L3 の段階では候補 ID もイベント ID も振っていない。**
区間は `(drive_id, segment, t_start)` の組で一意に決まるものとして扱う。
突き合わせは**区間の重なり**で行う (`t_end >= x.t_start AND t_start <= x.t_end`)。

**明示的な ID が付くのは VLM 段だけ** (次節の `event_id`)。

---

## 5. 優先度フィルタの出力

`scripts/detect/filter_priority.py` が候補表に 2 列を足す。

| 列 | 意味 |
|---|---|
| `priority` | 採用フラグ (bool) |
| `priority_reason` | 採用した規則名を `\|` で連結。`safety` は `always` 節による採用 |

**候補生成は変えない。** 既存の列はそのまま残る。

`scripts/report/report_final.py` / `report_chain.py` はさらに段ごとのフラグを足す。

| 列 | 意味 |
|---|---|
| `sev_ok` | `severity > 0` |
| `L1` | L1 を通ったか |
| `final` | `L1 AND sev_ok AND priority` |
| `final_noL1` | `sev_ok AND priority` (L1 を当てない場合) |
| `risky` / `labeled` | 人手ラベルとの突き合わせ結果 (ラベルがある場合のみ) |

`risky` は 3 値。`True` / `False` / 空欄 (= その区間に人手ラベルが無い)。
**空欄を `False` として数えない。** 未評価であってヒヤリハットでないという意味ではない。

---

## 6. VLM 段の入出力

### 6.1 `event_id` — VLM 段で唯一の明示的な ID

`labels.csv` を `risky` 降順 → `drive_id` → `segment` → `t_start` の順に並べ、
**positive に `P01`, `P02`, …、negative に `N01`, `N02`, … を振る**。

**この ID は `labels.csv` の内容と並び順に依存する。** ラベルを足すと振り直しに
なるため、過去の結果と突き合わせるときは同じ `labels.csv` を使うこと。

### 6.2 `request_id` — リクエスト 1 件の ID

| モード | 形 | 例 |
|---|---|---|
| A (clip) | `<event_id>\|A\|<condition>` | `P01\|A\|A` |
| B (online) | `<event_id>\|B\|<t_eval:.2f>` | `P01\|B\|11120.83` |

**`request_id` は再開に使う。** 途中で落ちても、書き終えた `request_id` を
飛ばして続きから流せる。

### 6.3 リクエスト JSONL (`requests_mode_a.jsonl` / `requests_mode_b.jsonl`)

| キー | 意味 |
|---|---|
| `request_id` / `event_id` | 上記 |
| `mode` | `clip` / `online` |
| `condition` | 条件名 (入力の組み合わせを変えた比較用) |
| `t_eval` | 評価時刻 (ルート絶対秒) |
| `frames` | 画像ファイルのパスの並び |
| `frame_times` | 各フレームの時刻 |
| `can_text` | CAN の要約テーブル (テキスト) |
| `can_extremes` | 区間内の極値の要約 (テキスト) |
| `span_s` | 見せる区間の長さ |
| `hint` | 追加の指示 (条件によって変える) |
| `guard_s` / `context_mode` | モード B で未来を見せないための余裕 |
| `config_hash` | 実行時の設定 |

**人手ラベル (`risky`) は入らない。** 採点時に `event_id` で突き合わせる。

### 6.4 結果 JSONL (`results_*.jsonl`)

リクエストの `request_id` / `event_id` に、VLM の構造化出力が付く。
スキーマは `src/near_miss/vlm/schema.py` が定義する。

| キー | 型 | 意味 |
|---|---|---|
| `scene` | string | 場面の記述 |
| `ego_behavior` | string | 自車の挙動 |
| `other_agents` | array | 他の交通参加者 (`what` / `where` / `motion` / `first_seen_s`) |
| `state` | enum | 状態 |
| `hazard_type` | enum | 危険の種類 |
| `risk_level` | integer 0–3 | 危険度 |
| `risky` | boolean | ヒヤリハットか |
| `difference_from_normal` | string | 通常走行との違い |
| `evidence` | enum | **判定の根拠が映像と CAN のどちらにあるか。中心指標** |
| `evidence_detail` | string | 根拠の詳細 |
| `insufficient_information` | boolean | 判断に足りないか |
| `confidence` | number 0–1 | 確信度 |

モード B ではさらに `t_eval_s` / `change_from_previous` / `expected_next` が付く。

**`additionalProperties: False`。** スキーマ外のキーは受け付けない。

### 6.5 CAN 候補と動画候補の対応

**対応関係は `event_id` 1 本で持っている。**

```
labels.csv の 1 行
    -> Episode (event_id = P01 / N01 ...)
         |- drive_id / segment / t_start / t_end / t_in_segment_s   (CAN 側の区間)
         |- フレームキャッシュ上の JPEG の並び                       (映像側)
         `- can_text / can_extremes                                 (CAN の要約)
```

**候補 (`candidates.csv`) から直接 `event_id` は引けない。** 現在の VLM 段の入力は
`labels.csv` (人手確認済み 32 件) であって `candidates.csv` ではない。
候補全件を VLM に流す経路は**未実装**。

### 6.6 採点結果

`scripts/vlm/score_vlm_review.py` が結果 JSONL と `labels.csv` を `event_id` で
突き合わせ、指標と digest を出す。

* negative (24 件) は **clip / episode 単位**を主指標にする
* positive は 8 件しかないため、集約値だけでなく**イベント別の個票**を残す
* 指標には必ず件数を併記する

---

## 7. 未定義・要確認

* **候補 ID / イベント ID が L3 に無い。** 区間の重なりで突き合わせている。
  出力を跨いで安定した ID が要るなら新たに決める必要がある
* **候補全件を VLM に流す経路が無い。** 現在の入力は人手確認済み 32 件のみ
* **VLM 判定を候補表へ書き戻す経路が無い。** 結果は JSONL に留まり、
  `candidates.csv` には反映されない
* **`risky` の空欄 (未評価) の扱いが指標側で統一されていない。**
  `l1_l3_chain.py` は分母から除くが、他の集計での扱いは要確認
* **commaCarSegments には映像が無いため、VLM 段に渡せない。**
  映像 AI を当てられるのは現在 comma2k19 のみ
