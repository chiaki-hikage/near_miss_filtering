# ヒヤリハット関連データの棚卸し (API 連携の検討用)

2026-10-28 時点で、リポジトリの `out/` にある出力を実際に開いて確かめた結果。
列名・件数・値は実ファイルから取った。**ファイルに無いものは「無い」と書き、推測では埋めていない。**

- 対象: 既存のヒヤリハット候補 (急ブレーキ・急操舵など)、スリップ、路面凹凸による異常振動
- API 担当者への回答案と JSON の例 (提案) は [near_miss_api_response_draft.md](near_miss_api_response_draft.md)
- サンプルは `out/near_miss_api_samples/` (`scripts/report/export_api_samples.py` で再生成できる)
- API 連携用の共通形式 (canonical event) は [near_miss_canonical_schema.md](near_miss_canonical_schema.md)。この文書の 4〜5 章の不足のうち、ID・時刻の基準・元データのパス・候補フラグはそちらで埋めた

---

## 0. 要点

| 問い | 現状 |
|---|---|
| DB はあるか | **無い。** SQLite・DuckDB・RDB への書き込みは、コードのどこにも無い |
| 何で持っているか | **解析ごとの CSV。** 実行条件は JSON (`run_meta.json`・`counts.json`・`meta.json`)。JSONL は VLM のリクエストだけ。Parquet は入力 (デモデータのダウンロードキャッシュ) だけ |
| イベントごとに別ファイルか | **系統ごとに別ファイル・別スキーマ。** 既存イベント (L3)・スリップ・異常振動は列名も時刻の基準も違う |
| 横断的に結合できるか | **できない。** 共通の `event_id` は無い。突き合わせは `(drive_id, segment)` と区間の重なりで行っている |
| 元データに戻れるか | comma2k19 は `drive_id`・`segment`・セグメント内の秒から映像と CAN に戻れる。**ただしチャンク名を持つのは異常振動だけ。** commaCarSegments は rlog の URL に戻れるが、映像と位置はデータセットに無い |
| 共有されているか | **`out/` は git の管理外**で、解析者の手元にしか無い。人手のラベルのうち git に入っているのは `out/chunk1/labels.csv` だけ |

---

## 1. 現状のデータ保持方式

### 1.1 形式

| 形式 | 実在するもの | 用途 |
|---|---|---|
| DB | なし | — |
| CSV | `out/` 以下に約 330 本 | イベント・候補・地点・通過・人手ラベル・集計のすべて |
| JSON | 約 80 本 | 実行条件と件数 (`run_meta.json`・`counts.json`・`meta.json`・`*_meta.json`) |
| JSONL | 2 本 (`out/chunk1/vlm/requests_mode_{a,b}.jsonl`) | VLM へのリクエスト。結果の JSONL (`results_*.jsonl`) は `out/` に無い |
| Parquet | 3 本 (`data/_hf_cache/demo-*.parquet`) | **入力のキャッシュ** (`scripts/fetch/fetch_demo_dataset.py`)。出力には使っていない |
| CSV.gz・npy | `stage1_samples.csv.gz` (スリップ 1 次の明細)・`out/jerk_feasibility/*.npy` | 中間結果。イベントの一覧ではない |
| Markdown | `review*.md`・`top20.md` など | 人が見るための一覧。映像・図へのリンク |
| PNG・MP4 | `out/vibration/near_miss/media/` に 89 本 (動画 33・フレーム 33・波形 23) など | 異常振動の確認用。既存イベント・スリップには切り出した動画が無い |

### 1.2 ディレクトリと版

- 1 回の実行 = 1 ディレクトリ (`--out` で指定)。**版はディレクトリ名の接尾辞** (`_v2`・`_v3`・`_full`) で区別しているだけで、最新版を指す仕組みは無い
- 古い版も消さずに残っている (例: `out/chunk1`・`chunk1_v2`・`chunk1_v3`)。この文書では、更新日時・件数・docs の記述から最新と判断したものを「正」とした (2 章)
- 設定の違いは行の `config_hash` (既存イベント・スリップ) か `meta.json` の `config_sha1` (異常振動) で区別できる。**コードの版 (git commit) はどこにも記録していない**

### 1.3 横断的な結合

| 結合 | 現状 |
|---|---|
| 系統をまたぐ ID | 無い |
| 既存イベント ↔ 候補 | ID 無し。`drive_id` が同じで区間が重なるものを対応させる (`t_end >= x.t_start AND t_start <= x.t_end`、[io_spec.md](io_spec.md) 4.4) |
| 候補 ↔ 人手ラベル (`labels.csv`) | 同上 (区間の重なり) |
| VLM 段 | `event_id` (`P01`・`N01`…) は `labels.csv` の並び順で振る。ラベルを足すと振り直しになる |
| 異常振動の中 | `event_id` (`E0001`…) と `site_id` で全ファイルを結合できる。ただし**どちらも実行ごとに振り直す連番** (`locate_candidates.py:884`) |
| 異常振動 ↔ 映像の目視 (`review_video.csv`) | `site_id` ではなく**緯度・経度・方位で**対応させている (地点番号が実行で変わるため) |
| スリップ ↔ 既存イベント | ID 無し。L3 にも `abs_active`・`slip_warn_active` などのスリップ系の event_type があるが、`direct_slip.csv` とは別の処理 |

### 1.4 元データとの対応

| 情報 | 既存イベント (comma2k19) | 既存イベント (commaCarSegments) | スリップ (直接証拠) | スリップ (beta, comma2k19) | 異常振動 |
|---|---|---|---|---|---|
| データセット | `dataset` 列 | **無し** (`platform` 列に車種キー) | `dataset` | `dataset` | `dataset` (locate のみ) |
| チャンク | **無し** | 該当なし | 該当なし | **無し** | `chunk` (locate のみ) |
| ドライブ | `drive_id` | `drive_id` | `drive_id` | `drive_id` | `drive` |
| セグメント | `segment` | `segment` | `segment` | `segment` | `segment` |
| 開始 | `t_start` (boot time)・`t_in_segment_s` | 同左 | 同左 | 同左 | `t_start_s` (セグメント内) |
| 終了 | `t_end` (boot time) のみ | 同左 | 同左 | 同左 | `t_end_s` |
| ピーク | **無し** (候補には `peak_t_in_segment_s`) | 同左 | **無し** | `t_peak`・`peak_t_in_segment_s` | `t_peak_s`・`event_time_s` (前輪が当たった時刻) |
| 映像のフレーム | `video_frame` (開始) | 映像なし | 映像なし | `video_frame` (開始) | **無し** |
| 映像ファイル | **無し** (`review.csv` の `video` 列だけ) | 映像なし | 映像なし | **無し** | **無し** (確認用の `media/` はファイル名の規則だけ) |
| CAN ファイル | **無し** | **無し** (URL は規則で組み立てられる) | **無し** | **無し** | **無し** |
| 位置 | **無し** | データセットに無い | データセットに無い | **無し** | `lat`・`lon`・`heading_deg` |
| UTC 時刻 | 無し | 無し | 無し | 無し | `event_utc` |

### 1.5 時刻の基準 (重要)

出力には 2 種類の時刻がある。

| 列 | 基準 | 備考 |
|---|---|---|
| `t_start`・`t_end`・`t_peak` (既存・スリップ) | **端末の boot time [s]** ([canonical.py:37](../src/near_miss/io/canonical.py)) | ルート先頭からの秒では**ない**。[io_spec.md](io_spec.md) 1.1 は「ルート先頭からの絶対秒」と書いているが、実装と合わない |
| `t_in_segment_s`・`peak_t_in_segment_s` (既存・スリップ) | セグメント先頭からの秒 | 先頭 = CAN の主系列が共通して始まる時刻 ([canonical.py:120](../src/near_miss/io/canonical.py) `t_span`) |
| `event_time_s`・`t_start_s`・`t_end_s`・`t_peak_s` (異常振動) | セグメント先頭からの秒 | 先頭 = **IMU 加速度の最初のサンプル** ([find_roll_bumps.py:107](../scripts/vibration/find_roll_bumps.py)) |
| `t_rel_start` (既存) | 連結した処理ブロックの先頭からの秒 | 内部用 |

- comma2k19 の映像の先頭は、IMU・CAN の先頭より 10〜75 ms 早い (20 セグメントで実測)。`video_frame = round(t_in_segment_s × 20)` は ±2 フレーム程度ずれうる。正確なフレームは `global_pose/frame_times` (boot time) で引ける
- **候補はセグメントの境界をまたぐことがある。** `segment` は開始が属するセグメントで、終了側のセグメントは列に無い (例: スリップ beta の 1 件は `t_in_segment_s` 44.57 + 18.25 s で 62.8 s)

---

## 2. イベント種別ごとのファイル一覧

**太字**が、この文書で最新と判断したもの。

### 2.1 既存のヒヤリハット候補 (急ブレーキ・急操舵など)

L3 のイベント種別は comma2k19 で 12 種類、commaCarSegments で 22 種類 (`out/l1_l3/*/events.csv` の `event_type`)。
急ブレーキ (`hard_brake`)・急操舵 (`hard_steer`) はその一部で、ファイルは種別で分かれていない。

| event_type | ファイル | 粒度 | 行数 | 主な内容 | 主キー候補 |
|---|---|---|---|---|---|
| hard_brake・hard_steer ほか全種別 | **`out/l1_l3/c2k19/events.csv`** (2026-09-17) | event | 200 | comma2k19 Chunk_1・2 の全イベント。判定根拠 (`feature`・`op`・`threshold`・`rule`) と L1 の通過 | (`drive_id`, `segment`, `event_type`, `t_start`) |
| 同上 | **`out/l1_l3/c2k19/candidates.csv`** | candidate | 141 | 候補 (前後 2 s を足して統合した区間) と段ごとのフラグ (`L1`・`priority`・`final`)・人手ラベルの照合 (`risky`) | (`drive_id`, `segment`, `t_start`) |
| 同上 | **`out/chunk1_v3/`・`out/chunk2_v3/`** の `events.csv`・`candidates.csv`・`segments.csv`・`run_meta.json` (2026-09-14) | event / candidate / drive | 103+97 / 70+71 / — | 上の 2 本の元。`candidates.csv` には区間内の特徴量の代表値が約 45 列ある (l1_l3 側には無い) | 同上 |
| 同上 | **`out/l1_l3/ccs/events.csv`・`candidates.csv`** (2026-09-17) | event / candidate | 3,519 / 2,558 | commaCarSegments の Toyota 4 車種 (各 2,000 セグメント)。映像・位置なし | 同上 (+ `platform`) |
| 同上 | **`out/screen_TOYOTA_<車種>_2k_v4/`** の `events.csv`・`candidates.csv`・`ranked_candidates.csv` ほか | event / candidate | 車種ごと | 上の元。`candidates.csv` に特徴量の代表値 (ABS・VSC・ブレーキ圧などを含む約 50 列) | 同上 |
| 人手ラベル | **`out/chunk1/labels.csv`** (git 管理)・`out/chunk2_final/labels.csv` (git 管理外) | candidate | 82 / 18 | 目視判定 (`verdict`・`risky`・`note`) | (`drive_id`, `segment`, `t_start`) |
| 確認用 | `out/chunk1/review*.csv`・`.md`、`out/chunk2_final/review_chunk2.*` | candidate | — | 映像のパス (`video`・`preview`) と頭出しの秒・フレーム | 同上 |
| VLM | `out/chunk1/vlm/requests_mode_{a,b}.jsonl`・`labels_onset.csv` | event | — | VLM のリクエスト (`event_id` = P01/N01…) | `event_id` (labels の並びに依存) |
| (comma1M) | `out/comma1m/screen/events.csv`・`candidates.csv` (2026-08-20) | event / candidate | 673 / 507 | CAN の無い comma1M を自己位置推定だけで走査したもの。候補に地域・天候・`lat_mid`・`lon_mid` | 同上 |
| (旧版) | `out/chunk1`・`chunk1_v2`・`chunk2_final`・`chunk2_v2`、`out/final_*.csv`、`out/chain_*.csv`、`out/screen_rav4_tss2*` | — | — | 古い設定の実行。API の元にはしない | — |

### 2.2 スリップ

| event_type | ファイル | 粒度 | 行数 | 主な内容 | 主キー候補 |
|---|---|---|---|---|---|
| slip (直接証拠) | **`out/sideslip_TOYOTA_{RAV4_TSS2,COROLLA_TSS2,HIGHLANDER_TSS2,PRIUS}_v3/direct_slip.csv`** (2026-09-16、等級の改訂後) | 確認単位 (区間) | 65 / 27 / 59 / 36 (計 187) | 制御作動 (slip_warn・ABS・VSC) と物理挙動 (輪速・横方向) の証拠を個別の列で持つ。等級 `grade` | (`drive_id`, `segment`, `t_start`) |
| slip (beta) | 同じディレクトリの `candidates.csv` | 確認単位 | 198 / 0 / 0 / 0 | 車両モデルの横滑り角 beta による候補 (諸元がある RAV4 TSS2 だけ) | 同上 |
| 同上 | `overlap.csv`・`counts.json` | 区間 / 実行 | — | 2 つの枝の重なり / 各段の件数と設定一式 | — |
| slip (beta, comma2k19) | **`out/sideslip_chunk1_v2/candidates.csv`・`out/sideslip_chunk2_v2/candidates.csv`** | 確認単位 | 6 / 7 | 映像のある comma2k19 での beta 候補。全件 `C_弱い候補`。直接証拠は 0 件 (`direct_slip.csv` は空) | 同上 |
| 作動フラグの生の区間 | `out/slip_flags_*/runs.csv` | フラグの連続区間 | 〜12,854 | フラグごとの立ち上がり (`kind`・`peak`) | (`drive_id`, `segment`, `kind`, `t_in_segment_s`) |
| 自動チェック | `out/review_direct_slip/reviewed.csv` | 確認単位 | 69 | 旧等級 (`A_確定` など) の 69 件に対する**スクリプトの**整合チェック (`verdict` は人手ではない) | — |
| L3 のスリップ系 | `out/l1_l3/ccs/events.csv` の `abs_active`・`slip_warn_active`・`wheel_speed_anomaly`・`yaw_instability` | event | 25 / 42 / 6 / 22 | 既存の L3 ルールでの検出。`direct_slip.csv` とは独立 | 2.1 と同じ |
| (旧版) | `out/sideslip_*_full`・`_v2`、`out/sideslip_rav4_tss2*`、`out/validate_slip_flags*` | — | — | 等級改訂前・検証用 | — |

`scripts/detect/rank_slip.py` は種別 (車輪のロック・駆動輪の空転・横方向・未分類) を**標準出力に出すだけ**で、ファイルには残していない。

### 2.3 路面凹凸による異常振動

対象は comma2k19 Chunk_1・2 (RAV4) だけ。すべて 2026-10-27 の実行。

| event_type | ファイル | 粒度 | 行数 | 主な内容 | 主キー候補 |
|---|---|---|---|---|---|
| road_vibration | **`out/vibration/locate/candidates.csv`** | event | 1,037 | 全振動イベント。チャンク・時刻・UTC・位置 (小数 6 桁)・誤差の目安・帯域別の比・輪速 | `event_id` |
| 同上 | **`out/vibration/near_miss/near_miss_vibration_events.csv`** | event | 1,037 | 異常振動ヒヤリハット候補の 5 条件 (bool)・点数・根拠の値・映像の判定。位置は**小数 3 桁に丸めてある** | `event_id` |
| 同上 | **`out/vibration/near_miss/iso2631_events.csv`** | event | 1,037 | ISO 2631-1 の評価量 (MTVV・VDV・快適性の目安・普段の走行での分位) | `event_id` |
| 同上 | `out/vibration/near_miss/near_miss_vibration_top20.csv` | event | 20 | 上の上位 20 件 | `event_id` |
| 同上 | **`out/vibration/locate/sites.csv`** | site | 498 | 同じ地点のまとまり。再現性・印 A〜E・Google Maps / Street View のリンク | `site_id` |
| 同上 | `out/vibration/near_miss/near_miss_vibration_sites.csv`・`_top20_sites.csv` | site | 498 / 20 | 地点ごとの代表イベント (最高点) と候補の数 | `site_id` |
| 同上 | **`out/vibration/locate/site_passes.csv`** | pass (地点 × 通過) | 4,459 | 地点を通った全走行の指標 (イベントにならなかった通過も含む) | (`site_id`, `drive_id`, `segment`, `t_closest_s`) |
| 同上 | `out/vibration/physics_validation_near_miss/event_physics_validation.csv`・`site_physics_validation.csv` | event / site | 76 / 23 | 上位 20 地点と候補 10 地点の物理的整合性 (P1〜P6、実測とモデル) | `event_id` / `site_id` |
| 同上 | `out/vibration/physics_validation/` の同名 2 本・`all_events_side_index.csv` | event / site / event | — / — / 1,037 | 最初に評価した 7 地点と、全イベントの側の指標 | `event_id` / `site_id` |
| 同上 | `out/vibration/near_miss/physics_vs_video_sites.csv`・`physics_vs_video.csv` | site / 集計 | 31 / 9 | 物理シグナルの数と映像の判定の対応 | `site_id` |
| 人手 (目視) | **`out/vibration/locate/review_video.csv`** (git 管理外) | site (位置で対応) | 38 | 映像・Street View の目視 (Claude の暫定判定を含む) | (`lat`, `lon`, `heading_deg`) |
| 確認用 | `out/vibration/near_miss/top20.md`・`media/`、`out/vibration/locate/top20.md`・`media/` | site | — | 動画 (前後 3 s)・波形・フレームへのリンク | — |
| 実行条件 | `out/vibration/locate/meta.json`・`near_miss/meta.json`・`iso2631_meta.json` | 実行 | — | 設定のハッシュ・件数・しきい値。**実行時刻は無い** | — |
| (旧版・別目的) | `out/vibration/longtail/`・`search_roll/`・`pavement/`・`fullcar*/`・`final/` | — | — | 前段の PoC とモデルの計算。イベント一覧としては locate に置き換わった | — |

### 2.4 今後追加するイベント (横風など)

横風などのイベントの出力は**まだ無い**。3 系統とも「区間 (開始・終了) + 判定根拠 + 固有の値」という同じ形をしているので、
共通項目 (5 章) に `event_type` を 1 つ足し、固有の値を `attributes` に入れる形にすれば同じ枠に載る。

---

## 3. カラム定義

「共通」= イベント種別によらず持てそうな項目、「固有」= その種別だけの項目。「必須候補」は API の共通項目として必須にしたい候補 (◎ 必須候補、○ あると良い、— 不要)。

### 3.1 依頼の共通項目が既にあるか

| 項目 | 既存イベント (`l1_l3/*/events.csv`) | 既存の候補 (`chunk*_v3/candidates.csv`) | スリップ直接証拠 (`direct_slip.csv`) | スリップ beta (`candidates.csv`) | 異常振動 (`locate/candidates.csv`・`near_miss_vibration_events.csv`) |
|---|---|---|---|---|---|
| event_id | 存在しない | 存在しない | 存在しない | 存在しない | `event_id` (E0001…、実行ごとの連番) |
| event_type | `event_type` (+ 日本語の `label`) | `event_types` (`\|` 区切りで複数) | 存在しない (`grade` のみ) | 存在しない (`grade` のみ) | 存在しない (`label` は全行同じ文字列) |
| dataset_name | `dataset` (c2k19)。ccs は存在しない (`platform` に車種キー) | `dataset` (c2k19) / `platform` (ccs) | `dataset` | `dataset` | `dataset` (locate のみ) |
| drive_id | `drive_id` (`<dongle>\|<route>`) | `drive_id` | `drive_id` | `drive_id` | `drive` |
| route_id | 存在しない (`drive_id` に含まれる) | 同左 | 同左 | 同左 | 同左 |
| clip_id / segment_id | `segment` (番号)。`segment_id` は**ブロック先頭のセグメント**で、イベントのセグメントではない | `segment`・`peak_segment` | `segment` | `segment`・`peak_segment` | `segment` |
| event_start_time | `t_start` (boot time)・`t_in_segment_s` | `t_start`・`t_in_segment_s` | `t_start`・`t_in_segment_s` | `t_start`・`t_in_segment_s` | `t_start_s` (locate) |
| event_end_time | `t_end` (boot time) のみ。セグメント内の秒は存在しない | 同左 | 同左 | 同左 | `t_end_s` (locate) |
| peak_time | 存在しない | `t_peak`・`peak_t_in_segment_s` (区間内で最も強いイベント) | 存在しない | `t_peak`・`peak_t_in_segment_s` | `t_peak_s` (locate)・`event_time_s` |
| frame | `video_frame` (c2k19 のみ、開始) | `video_frame` (開始) | 存在しない | `video_frame` | 存在しない |
| timestamp (UTC) | 存在しない | 存在しない | 存在しない | 存在しない | `event_utc` |
| latitude / longitude | 存在しない | 存在しない | 存在しない | 存在しない | `lat`・`lon` |
| vehicle_speed | 存在しない | `v_mps_mean`・`v_mps_min` [m/s] | `v_at_peak_mps`・`v_max_mps` [m/s] | `v_at_peak_mps` [m/s] | `speed_kmh` [**km/h**] |
| score | `peak_value`・`exceedance`・`weight` | `severity` | 存在しない | `confidence` | `near_miss_candidate_score`・`vibration_score`・`A` |
| severity | 存在しない (`weight` は種別の重み) | `severity` | 存在しない (`grade` が等級) | 存在しない (`grade`) | 存在しない |
| candidate_flag | `L1`・`L1_kin` | l1_l3 側に `sev_ok`・`priority`・`final` | 存在しない (全行が候補) | 存在しない (全行が候補) | 存在しない (5 条件の bool 列の積で決まる) |
| detection_method | `feature`・`op`・`threshold`・`rule`・`trigger_rule`・`gate`・`stage` | 存在しない | `evidence`・`corroboration`・`mechanisms` | `reasons`・`confidence_items` | 5 条件の bool 列 (式は docs にしか無い) |
| video_path / clip_path | 存在しない (`review.csv` の `video` 列のみ) | 存在しない | 存在しない | 存在しない | 存在しない (`media/` はファイル名の規則のみ) |
| algorithm_version | `config_hash` (設定のハッシュ) | `config_hash` | `config_hash` | `config_hash` | 存在しない (`meta.json` の `config_sha1`) |
| created_at | 存在しない (`run_meta.json` の `run_at`) | 同左 | 存在しない (`counts.json` の `run_at`) | 同左 | 存在しない (meta にも無い) |

### 3.2 既存イベント `out/l1_l3/c2k19/events.csv` (ccs も同じ。ccs は `dataset`・`video_frame`・`platform` の代わりに先頭が `platform`)

単位の規則は [io_spec.md](io_spec.md) 1.3 (列名の接尾辞 `_mps`・`_mps2`・`_dps` など)。

| column | 意味 | 単位 | 共通 / 固有 | 必須候補 |
|---|---|---|---|---|
| `dataset` | `comma2k19` / `comma_car_segments` (ccs のファイルには無い) | — | 共通 | ◎ |
| `event_type` | イベント名 (`hard_brake`・`hard_steer` など) | — | 共通 | ◎ |
| `label` | 日本語の表示名 | — | 共通 | ○ |
| `feature` | 判定に使った列名 (`hard_brake` は `ax_mps2`、`hard_steer` は `ay_kin_mps2`) | — | 共通 (判定根拠) | ○ |
| `op` / `threshold` | 判定の向き (`lt`・`gt`・`abs_gt`・`agreement`・`cooccurrence` など) と閾値 | feature の単位 | 共通 (判定根拠) | ○ |
| `t_start` / `t_end` | 区間 | s (boot time) | 共通 | ◎ (基準を明示して) |
| `duration_s` | 区間の長さ | s | 共通 | ○ |
| `peak_value` | 区間内の判定量の極値 (`hard_brake` は最小の前後加速度、`hard_steer` は符号付きの横加速度) | feature の単位 | 固有 | ○ |
| `exceedance` | 閾値からの超過量 | feature の単位 | 固有 | ○ |
| `stage` | 1 = CAN の基本信号のみ / 2 = レーダ・安全系フラグも使う | — | 共通 | — |
| `weight` | severity に足す重み | — | 共通 | — |
| `gate` | 適用したゲート (例 `v_mps gt 5.0`) | — | 共通 (判定根拠) | — |
| `rule` / `trigger_rule` | 判定式の文字列 (`hard_steer` は `steer_rate_dps abs_gt 25.0 & ay_kin_mps2 abs_gt 2.0 & ay_can_mps2 abs_gt 1.4`) | — | 共通 (判定根拠) | ○ |
| `drive_id` | `<dongle_id>\|<route>` | — | 共通 | ◎ |
| `segment_id` | `<drive_id>/<n>`。**連結ブロックの先頭のセグメント**で、イベントのあるセグメントではない | — | 共通 | — (誤解を招く) |
| `vehicle` | 車種設定の名前 (`toyota_rav4` など) | — | 共通 | ○ |
| `t_rel_start` | ブロック先頭からの秒 | s | 内部用 | — |
| `config_hash` | 設定のハッシュ (10 桁) | — | 共通 | ◎ (版として) |
| `segment` | 開始が属するセグメント番号 | — | 共通 | ◎ |
| `t_in_segment_s` | セグメント先頭からの開始秒 | s | 共通 | ◎ |
| `video_frame` | `round(t_in_segment_s × 20)` (c2k19 のみ) | frame | 共通 | ○ |
| `platform` | `comma2k19` または車種キー (`TOYOTA_RAV4_TSS2` など) | — | 共通 | ○ |
| `L1` / `L1_kin` | 軽量フィルタ L1 を通ったか (フラグ込み / 運動量のみ) | bool | 共通 | — |

### 3.3 既存の候補 `out/chunk1_v3/candidates.csv` (区間の集計値。主な列)

候補は複数のイベントを前後 2 s の余白で統合した区間。集計列の接尾辞は `_min`・`_max`・`_mean`・`_absmax`。
**`_absmax` は絶対値が最大になった時点の符号付きの値** ([scoring.py:107](../src/near_miss/scoring.py))。

| column | 意味 | 単位 | 共通 / 固有 | 必須候補 |
|---|---|---|---|---|
| `t_start` / `t_end` / `duration_s` | 統合後の区間 (余白込み) | s (boot time) | 共通 | ◎ |
| `t_peak`・`peak_segment`・`peak_t_in_segment_s` | 区間内で最も強いイベントの位置 | s | 共通 | ◎ |
| `severity` | 統合スコア | — | 共通 | ◎ |
| `n_events` / `event_types` | 含むイベント数と種別 (`\|` 区切り) | — / — | 共通 | ○ |
| `raw_can_loaded` | 生 CAN を読めたか | bool | 共通 | — |
| `v_mps_mean` / `v_mps_min` | 区間の平均・最小車速 | m/s | 共通 | ○ |
| `ax_mps2_min` / `ax_mps2_max` | 前後加速度の最小・最大 | m/s² | 固有 (急ブレーキ・急加速) | — |
| `jerk_mps3_min`・`jerk_win_min_min`・`jerk_win_mean_min` | 前後躍度 (瞬時・窓) の最小 | m/s³ | 固有 (急ブレーキ) | — |
| `brake_pressed_max`・`gvc_mps2_min`・`gas_pedal_pct_max` | ブレーキ踏み込み・車両側の前後 G・アクセル開度 | — / m/s² / % | 固有 (急ブレーキ) | — |
| `brake_mc_mpa_max`・`abs_active_flag_max`・`vsc_active_flag_max`・`slip_warn_max`・`precollision_active_max` | ブレーキ圧・安全系フラグ (**ccs の候補のみ**) | MPa / — | 固有 | — |
| `steer_rate_dps_absmax` | 舵角速度 | deg/s | 固有 (急操舵) | — |
| `ay_kin_mps2_absmax`・`ay_can_mps2_absmax` | 横加速度 (車速×ヨーレート / CAN の横 G) | m/s² | 固有 (急操舵) | — |
| `yaw_rate_dps_absmax` | ヨーレート | deg/s | 固有 (急操舵) | — |
| `lat_jerk_mps3_absmax`・`lat_jerk_win_absmax_max` | 横躍度 | m/s³ | 固有 (急操舵) | — |
| `steer_reversals_max`・`weave_reversals_max`・`net_heading_win_deg_max` | 操舵の反転回数・蛇行の反転回数・窓内の進路変化 | 回 / 回 / deg | 固有 (蛇行) | — |
| `lc_offset_m_absmax`・`lc_heading_amp_deg_max` | 車線変更の横移動・方位の振れ | m / deg | 固有 (車線変更) | — |
| `lead_distance_m_min`・`lead_vrel_mps_min`・`thw_s_min`・`ttc_s_min` | 先行車距離・相対速度・車間時間・TTC | m / m/s / s / s | 固有 (接近) | — |
| `ws_spread_mps_max` ほか | 4 輪の輪速の開き | m/s | 固有 (輪速) | — |
| `yaw_residual_dps_absmax`・`beta_model_deg_absmax`・`counter_steer_active_max` | ヨー残差・モデルの横滑り角・逆操舵 | deg/s / deg / — | 固有 (横方向) | — |
| `segment`・`t_in_segment_s`・`video_frame` | 開始のセグメントと頭出し | — / s / frame | 共通 | ◎ |

**急操舵の「舵角そのもの」は候補にも無い** (舵角速度・横加速度・ヨーレートはある)。舵角は スリップの `steer_at_peak_deg` だけ。

`out/l1_l3/*/candidates.csv` は上の列のうち `t_start`〜`ax_mps2_min` の一部だけを持ち、代わりに `L1`・`sev_ok`・`priority`・`priority_reason`・`final_noL1`・`final`・(c2k19 のみ) `risky`・`labeled` を持つ。

### 3.4 人手ラベル `out/chunk1/labels.csv`・`out/chunk2_final/labels.csv`

| column | 意味 | 単位 | 共通 / 固有 | 必須候補 |
|---|---|---|---|---|
| `round` / `rank_at_labeling` | 判定した回と、その時点の順位 | — | — | — |
| `drive_id`・`segment`・`t_start`・`t_end`・`t_in_segment_s` | 判定した区間 | s | 共通 | ◎ (対応づけ用) |
| `event_types_at_labeling` | 判定した時点の種別 | — | — | — |
| `verdict` | 判定。**表記がそろっていない** (100 行で 8 種類: `通常運転` 80・`境界` 8・`ヒヤリハット` 4・`若干リスクあり` 3・`リスク高` 2・`リスク高い`・`リスクあり`・`ややリスクあり` 各 1) | — | 共通 (レビュー) | ○ (`risky` の方が使いやすい) |
| `risky` | ヒヤリハットか (True / False) | bool | 共通 (レビュー) | ○ |
| `note` | 映像で見た状況の記述 | — | 共通 (レビュー) | ○ |

### 3.5 スリップ直接証拠 `direct_slip.csv`

| column | 意味 | 単位 | 共通 / 固有 | 必須候補 |
|---|---|---|---|---|
| `dataset`・`drive_id`・`segment`・`t_in_segment_s` | どこの区間か | — / s | 共通 | ◎ |
| `t_start` / `t_end` / `duration_s` | 確認単位の区間 | s (boot time) | 共通 | ◎ |
| `grade` | 等級 (`multi_evidence` = 異なる系統の裏付けが 2 つ以上、`high_evidence` = 1 系統、`activation_only` = 作動のみ、`physics_only` = 挙動のみ) | — | 固有 (共通の severity の候補) | ◎ |
| `n_direct` / `n_corroborated` / `n_mechanisms` | 作動の数 / 裏付けのある作動の数 / 裏付けの系統数 | 個 | 固有 | — |
| `evidence` | 立った証拠 (`\|` 区切り) | — | 固有 | ○ |
| `corroboration` | 作動ごとの裏付け (`作動:証拠+証拠\|…`) | — | 固有 | — |
| `mechanisms` | 系統 (`輪速`・`横方向`) | — | 固有 | — |
| `slip_warn`・`slip_warn_peak` | 滑り警告 (TRC/VSC 介入) が立ったか・最大値 | bool / 状態値 | 固有 | — |
| `abs_active` / `vsc_active` | ABS / VSC の作動 | bool | 固有 | — |
| `wheel_spread`・`ws_spread_peak_mps` | 4 輪の輪速の開き (≥ 2.0 m/s が 0.3 s 以上) とその最大 | bool / m/s | 固有 | — |
| `wheel_decel`・`wheel_decel_peak_mps2` | 輪単体の瞬間減速度 (≤ −8 m/s²) とその最小 | bool / m/s² | 固有 | — |
| `lateral` / `yaw_residual` | 横加速度 ≥ 3.9 m/s² / ヨー残差 ≥ 16σ | bool | 固有 | — |
| `tc_disabled` | TC が切られていた (参考) | bool | 固有 | — |
| `v_at_peak_mps`・`v_max_mps` | 証拠の山での車速・区間の最大車速 | m/s | 共通 | ○ |
| `ax_at_peak_mps2`・`ay_at_peak_mps2`・`yaw_rate_at_peak_dps`・`steer_at_peak_deg`・`beta_model_peak_deg` | 山での前後・横加速度・ヨーレート・舵角・モデルの横滑り角 | m/s² / m/s² / deg/s / deg / deg | 固有 | — |
| `config_hash` | 設定のハッシュ | — | 共通 | ◎ |

閾値は [sideslip_filter.md](sideslip_filter.md) 9.2。**車種の列は無い** (ディレクトリ名で分かるだけ)。山の時刻の列も無い。

### 3.6 スリップ beta `candidates.csv` (`sideslip_*`)

| column | 意味 | 単位 | 共通 / 固有 | 必須候補 |
|---|---|---|---|---|
| `t_start`・`t_end`・`duration_s`・`t_peak` | 区間と beta の山 | s (boot time) | 共通 | ◎ |
| `beta_peak_deg` | 横滑り角 beta の山 | deg | 固有 | ○ |
| `beta_sigma_peak` | beta をモデル誤差 σ で割った値 | σ | 固有 | — |
| `beta_excess_peak_deg` | 舵で説明できない beta (ヨー残差と横加速度残差の重み付き和) | deg | 固有 | — |
| `yaw_sigma_peak` | ヨー残差 / σ | σ | 固有 | — |
| `ay_peak_mps2`・`beta_rate_peak_dps`・`ax_at_peak_mps2` | 横加速度・beta の変化率・前後加速度 | m/s² / deg/s / m/s² | 固有 | — |
| `v_at_peak_mps` | 山での車速 | m/s | 共通 | ○ |
| `n_runs` | まとめた連続区間の数 | 個 | 固有 | — |
| `confidence`・`confidence_items` | 確信度の点と、その内訳 | — | 固有 | ○ |
| `grade` | `A_大横滑り` (\|beta\| ≥ 5°)・`B_要確認` (3〜5°)・`C_弱い候補` (< 3°) | — | 固有 | ◎ |
| `reasons`・`corroboration` | 通過の理由・同じ区間に立った裏付け | — | 固有 | — |
| `segment`・`t_in_segment_s`・`video_frame` (c2k19)・`peak_segment`・`peak_t_in_segment_s` | 頭出し | — / s / frame | 共通 | ◎ |

### 3.7 異常振動 `out/vibration/locate/candidates.csv`

時刻はすべてセグメント内の秒 (先頭 = IMU 加速度の最初のサンプル)。比 (`*_ratio`・`*_metric`) は同じ車速帯の普段の値 (中央値) で割ったもの。

| column | 意味 | 単位 | 共通 / 固有 | 必須候補 |
|---|---|---|---|---|
| `event_id` | イベント番号 (実行ごとの連番) | — | 共通 | ◎ (安定化が必要) |
| `dataset` / `chunk` | `comma2k19` / `Chunk_1`・`Chunk_2` | — | 共通 | ◎ |
| `drive` / `segment` | ドライブ / セグメント番号 | — | 共通 | ◎ |
| `event_time_s` | 前輪が凹凸に当たった時刻の推定 | s | 共通 (peak 相当) | ◎ |
| `t_peak_s` / `t_start_s` / `t_end_s` | 振動の山・始まり・終わり | s | 共通 | ◎ |
| `event_utc` | `event_time_s` の UTC | 日時 | 共通 | ○ |
| `lat` / `lon` / `heading_deg` | 前輪の位置 (小数 6 桁) と方位 | deg | 共通 | ◎ |
| `speed_kmh` | 車速 | km/h | 共通 | ◎ |
| `vertical_accel_peak_mps2` | 上下加速度の最大 | m/s² | 固有 | ○ |
| `heave_ratio` | 上下の振れの普段比 | 倍 | 固有 | — |
| `roll_rate_peak_dps` / `roll_metric` | ロール角速度の最大 / 普段比 | deg/s / 倍 | 固有 | ○ |
| `pitch_rate_peak_dps` / `pitch_metric` | ピッチ角速度の最大 / 普段比 | deg/s / 倍 | 固有 | — |
| `A` | 振動イベントの強さ = sqrt(上下比 × max(ロール比, ピッチ比))。3 以上でイベント | — | 固有 | ○ |
| `roll_over_pitch` | ロール比 / ピッチ比 | — | 固有 | — |
| `wheel_ratio_FL`〜`RR` / `wheel_metric` | 輪速の揺れの普段比 (4 輪 / 最大) | 倍 | 固有 | — |
| `wheel_lr_log2` / `wheel_side` | log2(左/右) と強い側 | — | 固有 | — |
| `wheel_front_rear_lag_s` / `wheel_lag_expected_s` / `wheel_lag_match` | 前→後輪の時間差・L/v・一致したか | s / s / bool | 固有 | — |
| `heave_road_ratio`・`heave_wheelhop_ratio`・`roll_road_ratio`・`roll_wheelhop_ratio` | 3〜8 Hz / 8〜15 Hz の普段比 | 倍 | 固有 | — |
| `heave_dom_hz` | 上下の主周波数 | Hz | 固有 | — |
| `wheelhop_only` / `road_band` / `one_sided` | 8〜15 Hz だけ高い / 3〜8 Hz も高い / 片側入力らしい | bool | 固有 | — |
| `frequency_features` | 帯域の要約 (文字列) | — | 固有 | — |
| `vibration_score` | A・輪速・3〜8 Hz の上下比の分位の平均 × 100 | 0〜100 | 固有 | — |
| `site_id` / `repeat_count` | 地点番号 (実行ごとの連番) / 地点のイベント数 | — / 件 | 固有 | ○ |
| `gps_dt_s` / `gnss_acc_m` | 最寄りの GPS 点との時間差 / GNSS の自己申告精度 | s / m | 共通 (位置の質) | ○ |
| `pos_err_along_m` / `pos_err_cross_m` | 位置の誤差の目安 (進行方向 / 横) | m | 共通 (位置の質) | ○ |
| `spike_like` / `brake_any` / `ay_range_mps2` | 飛び値らしい / 前後 1 s にブレーキ / 横加速度の幅 | bool / bool / m/s² | 固有 | — |

### 3.8 異常振動 `out/vibration/near_miss/near_miss_vibration_events.csv`

| column | 意味 | 単位 | 共通 / 固有 | 必須候補 |
|---|---|---|---|---|
| `rank` | 点数の順位 | — | — | — |
| `event_id` | locate と同じ | — | 共通 | ◎ |
| `label` | `路面凹凸により短時間に強い上下・ロール応答が発生した異常振動ヒヤリハット候補` (**1,037 行すべて同じ**。候補かどうかは表さない) | — | — | — |
| `drive`・`segment`・`event_time_s`・`event_utc`・`speed_kmh`・`site_id` | locate と同じ | — | 共通 | ◎ |
| `lat` / `lon` | 位置 (**小数 3 桁に丸めてある** ≒ 100 m) | deg | 共通 | — (locate 側を使う) |
| `site_rank_locate` / `feature_type` | 位置特定での地点の順位 / `点状`・`区間` | — | 固有 | — |
| `vibration_event` | 既存の振動イベント (A ≥ 3)。全件 true | bool | 固有 | — |
| `strong_vertical` | 上下の最大が全イベントの上位 25% (≥ 2.81 m/s²) | bool | 固有 | — |
| `strong_roll` | ロール角速度の最大が上位 25% (≥ 3.94 deg/s) | bool | 固有 | — |
| `simultaneous_vertical_roll` | 上下とロールの山の時刻差 ≤ 0.25 s | bool | 固有 | — |
| `impulsive_response` | 集中度 ≥ 0.5 かつ立ち上がり ≤ 0.3 s | bool | 固有 | — |
| `road_related_evidence` | 輪速の揺れ ≥ 普段の 3 倍、11 Hz だけでない、飛び値でない | bool | 固有 | — |
| **上の 5 列がすべて true** | **異常振動ヒヤリハット候補 (21 件)。専用の列は無い** | bool | 共通 (candidate_flag) | ◎ |
| `repeatability` | `再現 (偶然なら 1% 未満)`・`2 ドライブ以上でイベント`・`1 回のみ` | — | 固有 | ○ |
| `video_road_feature` | 映像の目視 (`未確認`・`一致: …`・`候補: …`・`不明瞭`・`判別不能: 夜間`) | — | 共通 (レビュー) | ○ |
| `near_miss_candidate_score` | 100 × sqrt(上下の分位 × ロールの分位) × 集中度 × 減点 | 0〜100 | 共通 (score) | ◎ |
| `heave_peak_abs` / `roll_peak_abs` / `pitch_peak_abs` | 上下 / ロール / ピッチの最大の絶対値 | m/s² / deg/s / deg/s | 固有 | ○ |
| `heave_roll_dt_s` | ロールの山 − 上下の山 | s | 固有 | — |
| `concentration` | 前後 1.5 s のエネルギーのうち前後 0.5 s の割合 (上下・ロールの小さい方) | 0〜1 | 固有 | — |
| `rise_s` | 包絡線が最大の 25% から最大に達するまで | s | 固有 | — |
| `A` / `A_rank` | locate と同じ / その順位 | — | 固有 | — |
| `heave_ratio`・`roll_metric`・`wheel_metric`・`wheel_lr_log2` | locate と同じ | 倍 | 固有 | — |
| `side_index` | 3〜8 Hz のロール角速度と上下速度の相関 (負 = 右片輪、正 = 左片輪) | −1〜1 | 固有 | — |
| `eleven_hz_only` / `spike_like` / `brake_any` | 11 Hz の固有応答だけ / 飛び値らしい / ブレーキ | bool | 固有 | — |
| `google_maps` | 位置の Google Maps のリンク (3 桁の位置) | URL | 共通 | — |

### 3.9 異常振動 `iso2631_events.csv` (ISO 2631-1)

| column | 意味 | 単位 | 必須候補 |
|---|---|---|---|
| `mtvv_x`・`_y`・`_z`・`_rx`・`_ry` | 軸ごとの MTVV (周波数重み付けした加速度の 1 s 移動 RMS の最大) | m/s² (回転軸は座面係数を掛けた値) | — |
| `mtvv_total` / `mtvv_translational` | 5 軸 / 並進 3 軸の合成 | m/s² | ○ |
| `vdv_z` / `vdv_total` | 振動暴露値 | m/s^1.75 | ○ |
| `aw_rms_z` / `crest_z` / `mtvv_over_aw_z` | 重み付き RMS / クレストファクタ / MTVV と RMS の比 | m/s² / — / — | — |
| `dominant_axis` | 最も大きい軸 | — | — |
| `comfort` | 附属書 C の快適性の目安 (`少し不快`〜`極めて不快`) | — | ○ |
| `mtvv_z_lp8`・`mtvv_total_lp8`・`comfort_lp8` | 8 Hz 以上を除いた感度分析 | m/s² | — |
| `*_normal_pct` | 普段の走行の窓 (1,094 窓) の中での分位 | % | ○ |

### 3.10 異常振動の地点・通過・物理評価 (主な列)

| ファイル | 主な列 | 備考 |
|---|---|---|
| `locate/sites.csv` | `site_id`・`lat`・`lon`・`heading_deg`・`n_events`・`n_pass`・`repeat_A_k`/`_n`/`_surprise` (再現した数 / 通過数 / −log10 p)・`flag_A`〜`flag_E`・`feature_type`・`best_event_id`・`google_maps`・`street_view` | 63 列 |
| `locate/site_passes.csv` | `site_id`・`drive_id`・`segment`・`t_closest_s` (地点に最も近づいたセグメント内の秒)・`speed_kmh`・`same_lane`・`event_id` (イベントがあれば)・`A_max`・`repeat_hit_A` | イベントにならなかった通過も入る |
| `physics_validation_near_miss/event_physics_validation.csv` | `site_id`・`event_id`・`P1`〜`P5` (`true`/`false`/`uncertain`)・`inferred_input`・`meas_*` (実測)・`model_*` (モデル) | 86 列 |
| `physics_validation_near_miss/site_physics_validation.csv` | `site_id`・`P1`〜`P6`・`physical_consistency` (`Strong physical consistency`・`Partial …`・`Not explained`)・`inferred_side`・`representative_event` | 58 列 |
| `near_miss/physics_vs_video_sites.csv` | `site_id`・`P2`・`P4`・`P5`・`P6`・`inferred_side`・`physics_signals` (映像に依存しない 5 つのうち成り立つ数) | 31 地点 (昼間の映像があるもの) |
| `locate/review_video.csv` | `lat`・`lon`・`heading_deg`・`site_id_at_review`・`d_video`・`d_streetview`・`note`・`reviewer` | **位置で地点に対応させる** |

---

## 4. 元データに戻るための情報

### 4.1 判定 — 解析者以外が API のレスポンスから元の映像・CAN を開けるか

| 情報 | 既存 (c2k19) | 既存 (ccs) | スリップ直接 (ccs) | スリップ beta (c2k19) | 異常振動 (c2k19) |
|---|---|---|---|---|---|
| dataset | ○ | △ `platform` から分かる | ○ | ○ | ○ |
| drive | ○ | ○ | ○ | ○ | ○ |
| clip / segment | ○ | ○ | ○ | ○ | ○ |
| 開始時刻 (セグメント内) | ○ | ○ | ○ | ○ | ○ |
| 終了時刻 (セグメント内) | △ `t_in_segment_s + duration_s` で出せる。境界をまたぐと出ない | △ 同左 | △ 同左 | △ 同左 | ○ |
| ピーク時刻 | × (候補の値で代用) | × | × | ○ | ○ |
| 動画ファイル | △ チャンクが分からない | — (映像なし) | — (映像なし) | △ チャンクが分からない | △ チャンクはあるがパスの列は無い |
| CAN ファイル | △ 同上 | △ URL を規則で組み立てられる | △ 同左 | △ 同上 | △ 同上 |
| GPS 位置 | × (元データにはある) | — (データセットに無い) | — | × (元データにはある) | ○ |
| フレーム番号 | △ `video_frame` は開始のみ、±2 フレームの誤差 | — | — | △ 同左 | × |

○ = 出力の列だけで分かる、△ = 規則や計算で出せるが列は無い、× = 出力に無い、— = データセットに無い

**結論: 現在の CSV だけでは、解析者でない人が確実に元データへ戻るのは難しい。** 理由は次の 4 つ。

1. **チャンク名が無い** (異常振動以外)。comma2k19 の生データは `Chunk_N/<drive_id>/<segment>/` の下にあり、`drive_id` だけでは置き場所が決まらない。今回のサンプルでは手元の `raw_data/` を走査して引いた
2. **ファイルのパスや URI の列が無い。** パスは「データセットの配置の規則 + drive + segment」で組み立てるしかない。規則はコード (`src/near_miss/io/`) と docs にしか書かれていない
3. **時刻の基準が系統で違い、docs と実装も食い違う** (1.5 節)。`t_start` を「ルート先頭からの秒」と読むと、まったく別の場所を見ることになる
4. **出力と人手ラベルが共有されていない。** `out/` は git の管理外で、どこにも上げていない。S3 への同期の仕組みはある (`scripts/fetch/upload_to_s3.py`、配置は `raw_data/` と 1:1) が、**対象は元データだけ**で出力は含まない。バケットも未設定 (`configs/datasets/s3.yaml` の `uri: ""`)

### 4.2 不足している項目の一覧

| # | 不足 | 対象 | 影響 | 現状のデータから足せるか |
|---|---|---|---|---|
| 1 | 安定した `event_id` | 全系統 | 再実行・ラベル追加で番号が変わり、外部から参照できない | 足せる (内容から決まる ID にする。例: `dataset`・`drive_id`・`segment`・開始の秒・`event_type` から作る) |
| 2 | `event_type` の列 | スリップ・異常振動 | 種別で絞り込めない | 足せる (定数) |
| 3 | `dataset` の列 | ccs の既存イベント | — | 足せる (定数) |
| 4 | `chunk` | 既存・スリップ (c2k19) | 元データの場所が決まらない | 足せる (`SegmentRef.path` に既にある。異常振動はそこから出している) |
| 5 | 元データの URI (映像・CAN・GNSS) | 全系統 | 規則を知らないと開けない | 足せる (配置の規則で組み立てる) |
| 6 | 時刻の基準の明示と統一 | 全系統 | 取り違え | 足せる (列名か meta に基準を書く。io_spec の記述を直す) |
| 7 | セグメント内の終了時刻・終了側のセグメント | 既存・スリップ | 境界をまたぐ区間の終わりが分からない | 足せる (`t_end` と各セグメントの先頭の時刻から) |
| 8 | ピーク時刻 | 既存イベント・スリップ直接証拠 | 動画の見どころが分からない | 再計算が必要 (検出時に極値の時刻を残す) |
| 9 | 位置 (lat / lon) | 既存・スリップ (c2k19) | 地図で見られない | 再計算が必要 (comma2k19 は `global_pose` がある。commaCarSegments は元データに位置が無い) |
| 10 | 映像のフレーム番号 (正確なもの) | 全系統 | ±2 フレームのずれ / 異常振動は無い | 足せる (`global_pose/frame_times` で引く) |
| 11 | 候補かどうかの列 | 異常振動 | 5 条件の積を利用者が計算する必要がある | 足せる |
| 12 | 車速の単位の統一 | 異常振動 (km/h) と他 (m/s) | 取り違え | 足せる |
| 13 | 位置の精度 | `near_miss_vibration_events.csv` (3 桁) | 100 m 程度ずれる | 足せる (locate 側の 6 桁を使う) |
| 14 | アルゴリズムの版 (git commit)・作成日時 | 全系統 | どのコードで出したか追えない | 次の実行から足せる (過去の出力については分からない) |
| 15 | 車種の列 | スリップ (ディレクトリ名のみ)・異常振動 | 車種で絞り込めない | 足せる |
| 16 | 人手ラベルと ID の対応 | 既存 (区間の重なり)・異常振動 (位置の近さ) | 再実行で対応が外れうる | 1 を決めれば足せる |
| 17 | 確認用の切り出し動画 | 既存・スリップ | 元の 60 秒の映像から探す必要がある | 足せる (異常振動と同じ方法で切り出す) |
| 18 | 出力の共有先 | 全系統 | 解析者の手元にしか無い | 運用で決める (S3 の仕組みは元データ用のみ) |
| 19 | 人手ラベルのバックアップ | `chunk2_final/labels.csv`・`review_video.csv` | git 管理外で、消えると作り直せない | 運用で決める (git に入れるか、別に保存する) |
| 20 | 人手ラベルの値の統一 | `labels.csv` の `verdict` (8 種類の表記) | 集計・絞り込みがしにくい | 足せる (`risky` は bool でそろっている) |

---

## 5. 共通項目とイベント固有項目

### 5.1 共通項目の候補 (現状の列との対応)

| 共通項目 (案) | 既存 | スリップ | 異常振動 | 現状 |
|---|---|---|---|---|
| `event_id` | — | — | `event_id` (不安定) | **新設が必要** |
| `event_type` | `event_type` | — | — | 異常振動・スリップは定数で足す |
| `dataset` | `dataset` / `platform` | `dataset` | `dataset` | ccs の既存は足す |
| `vehicle` | `vehicle` | — | — | スリップ・異常振動は足す |
| `drive_id` | `drive_id` | `drive_id` | `drive` | 列名をそろえる |
| `chunk` | — | — | `chunk` | comma2k19 は足す |
| `segment` | `segment` | `segment` | `segment` | ある |
| `start_sec` (セグメント内) | `t_in_segment_s` | `t_in_segment_s` | `t_start_s` | 基準の定義をそろえる |
| `end_sec` | 計算 | 計算 | `t_end_s` | 既存・スリップは足す |
| `peak_sec` | 候補のみ | beta のみ | `t_peak_s` | 既存・スリップ直接は再計算 |
| `latitude` / `longitude` | — | — | `lat` / `lon` | comma2k19 は再計算で足せる |
| `speed_mps` | 候補の `v_mps_mean` | `v_at_peak_mps` | `speed_kmh` / 3.6 | 定義 (いつの車速か) をそろえる |
| `score` | 候補の `severity` | beta の `confidence` | `near_miss_candidate_score` | 種別ごとに意味が違う。名前を併記する |
| `severity` / `grade` | — | `grade` | — | 段階をどう共通化するかは要相談 |
| `candidate_flag` | `final` (l1_l3) | 全行が候補 | 5 条件の積 | 定義をそろえる |
| `detection_method` / `rule` | `rule`・`trigger_rule` | `evidence`・`reasons` | (docs のみ) | ある程度ある |
| `source_uri` / `video_uri` | — | — | — | **新設が必要** |
| `algorithm_version` | `config_hash` | `config_hash` | (meta の `config_sha1`) | git commit も足す |
| `created_at` | (`run_meta.json`) | (`counts.json`) | — | 行か meta に足す |

### 5.2 イベント固有項目 (実在する列のみ)

**急ブレーキ** (`hard_brake`)
- イベント: `peak_value` (= 前後加速度の最小 [m/s²])・`exceedance`・`threshold` (−2.5 m/s²)
- 候補の区間: `ax_mps2_min`・`jerk_mps3_min`・`jerk_win_min_min`・`jerk_win_mean_min`・`brake_pressed_max`・`gvc_mps2_min`・`gas_pedal_pct_max`。ccs のみ `brake_mc_mpa_max`・`abs_active_flag_max`・`precollision_active_max`・`precollision_force_n_min`・`acc_braking_max`
- 依頼の例のうち `min_ax` は `ax_mps2_min`、`max_jerk` は `jerk_mps3_min` (減速側なので最小)、`brake_status` は `brake_pressed_max` に当たる。名前どおりの列は無い

**急操舵** (`hard_steer`)
- イベント: `peak_value` (= 横加速度 `ay_kin_mps2` [m/s²])・`rule` (舵角速度 > 25 deg/s かつ横加速度 > 2.0 かつ CAN の横 G > 1.4)
- 候補の区間: `steer_rate_dps_absmax`・`ay_kin_mps2_absmax`・`ay_can_mps2_absmax`・`yaw_rate_dps_absmax`・`lat_jerk_mps3_absmax`・`steer_reversals_max`
- **舵角 (`steering_angle`) の列は無い**

**スリップ**
- 直接証拠: `grade`・`evidence`・`mechanisms`・`n_mechanisms`・`slip_warn`・`abs_active`・`vsc_active`・`wheel_spread`・`ws_spread_peak_mps`・`wheel_decel`・`wheel_decel_peak_mps2`・`lateral`・`yaw_residual`・`tc_disabled`・`ax_at_peak_mps2`・`ay_at_peak_mps2`・`yaw_rate_at_peak_dps`・`steer_at_peak_deg`・`beta_model_peak_deg`
- beta: `beta_peak_deg`・`beta_sigma_peak`・`beta_excess_peak_deg`・`yaw_sigma_peak`・`confidence`・`grade`
- 依頼の例のうち TRC は `slip_warn` (TRC/VSC の介入) に含まれ、単独の列は無い。`friction_demand` は**存在しない**。`physics_consistency` は列名としては無く、`grade` (裏付けの系統数) が近い

**異常振動**
- `heave_peak_abs` (vertical_accel_max)・`roll_peak_abs` (roll_rate_max)・`pitch_peak_abs`・`near_miss_candidate_score`・`vibration_score`・`A`
- `impulsive_response`・`concentration`・`rise_s`・`heave_roll_dt_s`
- 物理: `physics_signals` (地点単位、31 地点のみ)・`physical_consistency` (地点単位、23 地点のみ)・`P1`〜`P5` (イベント単位、76 件のみ)・`side_index` (全件)
- ISO 2631-1: `mtvv_total`・`mtvv_z`・`vdv_z`・`comfort`・`mtvv_total_normal_pct`
- 再現性: `repeatability` (イベント)・`repeat_A_k`/`_n`/`_surprise` (地点)
- 物理評価は**上位・候補の一部の地点にしか無い**。全 1,037 件にあるのは `side_index` だけ

---

## 6. サンプルデータ

`out/near_miss_api_samples/` に置いた。CSV は**元ファイルの行を文字列のまま写したもの** (列・値・桁を変えていない。全行が元ファイルの行と一致することを確認した)。

| ファイル | 元 | 行 | 選び方 |
|---|---|---|---|
| `hard_braking_events_sample.csv` | `out/l1_l3/c2k19/events.csv` | 3 | 減速の大きい 2 件 (人手ラベル「境界」「リスク高」) と、ラベル「通常運転」の 1 件 |
| `hard_steering_events_sample.csv` | 同上 | 3 | 人手ラベル「リスク高」の 1 件と、人手ラベル「通常運転」の 2 件 |
| `existing_candidates_sample.csv` | `out/chunk1_v3/candidates.csv`・`out/chunk2_v3/candidates.csv` | 6 | 上の 6 件と区間が重なる候補 |
| `existing_events_ccs_sample.csv` | `out/l1_l3/ccs/events.csv` | 2 | `hard_brake`・`hard_steer` それぞれファイルの先頭の 1 件 |
| `slip_direct_evidence_sample.csv` | `out/sideslip_TOYOTA_RAV4_TSS2_v3/direct_slip.csv` | 3 | 等級 `multi`・`high`・`activation_only` の先頭 |
| `slip_beta_comma2k19_sample.csv` | `out/sideslip_chunk1_v2/candidates.csv` | 2 | 先頭 2 件 |
| `road_vibration_near_miss_sample.csv` | `out/vibration/near_miss/near_miss_vibration_events.csv` | 3 | E0818・E0474 (地点 7 のマンホール)・E0870 (地点 10 の継ぎ目) |
| `road_vibration_locate_sample.csv` | `out/vibration/locate/candidates.csv` | 3 | 同上 |
| `road_vibration_iso2631_sample.csv` | `out/vibration/near_miss/iso2631_events.csv` | 3 | 同上 |
| `road_vibration_physics_event_sample.csv`・`_physics_site_sample.csv`・`_site_sample.csv` | `physics_validation_near_miss/`・`locate/sites.csv` | 3 / 2 / 2 | 同上の地点 |
| `api_response_examples.json` | 上のすべて | 16 件 | **API 化した場合の例 (提案)**。現状の形式ではない |

選び方は `configs/near_miss_api_samples.yaml` にある。生成: `uv run python scripts/report/export_api_samples.py`。

### 6.1 現在の CSV そのままの形式 (抜粋)

急ブレーキ (`hard_braking_events_sample.csv`、全 27 列のうち先頭から)

```csv
dataset,event_type,label,feature,op,threshold,t_start,t_end,duration_s,peak_value,exceedance,stage,weight,gate,rule,drive_id,segment_id,vehicle,t_rel_start,trigger_rule,config_hash,segment,t_in_segment_s,video_frame,platform,L1,L1_kin
comma2k19,hard_brake,急ブレーキ,ax_mps2,lt,-2.5,267990.3499999549,267992.34999995446,1.9999999995343387,-4.51326362668406,2.0132636266840596,1,1.0,,ax_mps2 lt -2.5,b0c9d2329ad1606b|2018-11-11--13-08-44,b0c9d2329ad1606b|2018-11-11--13-08-44/9,toyota_rav4,193.6999999549007,ax_mps2 lt -2.5,231eee1fe6,12,14.32,286,comma2k19,True,True
comma2k19,hard_brake,急ブレーキ,ax_mps2,lt,-2.5,78643.25000000001,78643.95000000006,0.7000000000407454,-4.348145698130188,1.848145698130188,1,1.0,,ax_mps2 lt -2.5,b0c9d2329ad1606b|2018-08-17--14-55-39,b0c9d2329ad1606b|2018-08-17--14-55-39/1,toyota_rav4,0.1000000000058207,ax_mps2 lt -2.5,231eee1fe6,1,0.74,15,comma2k19,True,True
```

急操舵 (`hard_steering_events_sample.csv`)

```csv
dataset,event_type,label,feature,op,threshold,t_start,t_end,duration_s,peak_value,exceedance,stage,weight,gate,rule,drive_id,segment_id,vehicle,t_rel_start,trigger_rule,config_hash,segment,t_in_segment_s,video_frame,platform,L1,L1_kin
comma2k19,hard_steer,急操舵,ay_kin_mps2,agreement,2.0,13053.849999995413,13054.249999995409,0.3999999999941792,-2.4280018350653143,0.4280018350653143,2,0.3,net_heading_win_deg lt 15.0 & v_mps gt 5.0,steer_rate_dps abs_gt 25.0 & ay_kin_mps2 abs_gt 2.0 & ay_can_mps2 abs_gt 1.4 (許容ずれ 0.3s),b0c9d2329ad1606b|2018-07-30--13-44-30,b0c9d2329ad1606b|2018-07-30--13-44-30/6,toyota_rav4,315.2499999954125,steer_rate_dps abs_gt 25.0 & ay_kin_mps2 abs_gt 2.0 & ay_can_mps2 abs_gt 1.4 (許容ずれ 0.3s),231eee1fe6,11,15.9,318,comma2k19,True,True
```

スリップ直接証拠 (`slip_direct_evidence_sample.csv`)

```csv
dataset,drive_id,t_start,t_end,duration_s,grade,n_direct,n_corroborated,n_mechanisms,evidence,corroboration,mechanisms,slip_warn,slip_warn_peak,abs_active,vsc_active,wheel_spread,ws_spread_peak_mps,wheel_decel,wheel_decel_peak_mps2,lateral,yaw_residual,tc_disabled,v_at_peak_mps,v_max_mps,ax_at_peak_mps2,ay_at_peak_mps2,yaw_rate_at_peak_dps,steer_at_peak_deg,beta_model_peak_deg,config_hash,segment,t_in_segment_s
comma_car_segments,615671030be02e4e|000004cd--18bee28ac5,37455.85000000004,37460.550000000316,4.7,multi_evidence,2,2,2,slip_warn|vsc_active|wheel_spread|wheel_decel|lateral|yaw_residual,slip_warn:wheel_spread+wheel_decel+lateral+yaw_residual|vsc_active:lateral+yaw_residual,輪速|横方向,True,4.0,False,True,True,3.748,True,-19.21,True,True,False,11.08,12.95,-0.5,4.712,43.1,356.3,9.901,9bb78e6b6e,18,1.33
comma_car_segments,19180698b750ab29|000003f2--f077f06feb,326.2000000000125,329.95000000001335,3.75,high_evidence,2,2,1,slip_warn|abs_active|wheel_decel,slip_warn:wheel_decel|abs_active:wheel_decel,輪速,True,4.0,True,False,False,1.569,True,-17.27,False,False,False,19.41,21.89,-2.081,-1.452,-4.95,-13.9,-0.64,9bb78e6b6e,4,55.63
```

異常振動 (`road_vibration_near_miss_sample.csv`、全 39 列のうち先頭 28 列)

```csv
rank,event_id,label,drive,segment,event_time_s,event_utc,lat,lon,speed_kmh,site_id,site_rank_locate,feature_type,vibration_event,strong_vertical,strong_roll,simultaneous_vertical_roll,impulsive_response,road_related_evidence,repeatability,video_road_feature,near_miss_candidate_score,heave_peak_abs,roll_peak_abs,pitch_peak_abs,heave_roll_dt_s,concentration,rise_s
4,E0818,路面凹凸により短時間に強い上下・ロール応答が発生した異常振動ヒヤリハット候補,b0c9d2329ad1606b|2018-10-05--22-20-31,16,46.29,2018-10-06 05:37:18,37.727,-122.472,53.0,7,69,点状,True,True,True,True,True,True,2 ドライブ以上でイベント,一致: マンホール・補修跡 (右輪の通り道),73.6,4.155,9.349,5.793,-0.15,0.758,0.27
5,E0870,路面凹凸により短時間に強い上下・ロール応答が発生した異常振動ヒヤリハット候補,b0c9d2329ad1606b|2018-10-09--15-48-37,12,13.02,2018-10-09 23:00:51,37.595,-122.418,117.6,10,13,区間,True,True,True,True,True,True,再現 (偶然なら 1% 未満),一致: 横断方向の継ぎ目・亀裂 (両輪),72.9,4.423,4.657,5.656,-0.13,0.802,0.3
```

同じイベントの locate 側 (`road_vibration_locate_sample.csv`、先頭 14 列)

```csv
event_id,dataset,chunk,drive,segment,event_time_s,t_peak_s,t_start_s,t_end_s,event_utc,lat,lon,heading_deg,speed_kmh
E0818,comma2k19,Chunk_2,b0c9d2329ad1606b|2018-10-05--22-20-31,16,46.29,46.45,46.2,46.72,2018-10-06 05:37:18,37.726728,-122.471957,2.5,53.0
E0870,comma2k19,Chunk_2,b0c9d2329ad1606b|2018-10-09--15-48-37,12,13.02,13.21,12.91,13.67,2018-10-09 23:00:51,37.595364,-122.418405,322.1,117.6
```

### 6.2 API 化した場合の JSON の例 (提案)

**`api_response_examples.json` は最初の試案で、canonical event schema (`schemas/near_miss_event.schema.json`、`out/near_miss_canonical/`) に置き換わった。** 以下は経緯として残す。

[near_miss_api_response_draft.md](near_miss_api_response_draft.md) の 3 章と `out/near_miss_api_samples/api_response_examples.json` を見ること。
**現状のデータ形式ではない。** 値は上の CSV から写したもので、無い値は null にしてある。

---

## 7. 確かめたこと・確かめていないこと

- 確かめた: DB・Parquet 出力の有無 (コードの grep)、各ファイルの列・行数・更新日時、時刻の基準 (実装)、映像の先頭と IMU・CAN の先頭の差 (20 セグメントで 10〜75 ms)、git の管理対象
- 確かめていない: commaCarSegments の急ブレーキ 1,015 件のうち 1 件 (RAV4 TSS2、`63e74ebb84173067|0000009c--84e520ce41` seg 2) が −11.4 m/s² と 1 G を超える。始動直後の無効値 (ルート先頭) ではなく、原因は調べていない
- 確かめていない: VLM の結果 JSONL (`results_*.jsonl`) は `out/` に無く、どこにあるかは分からない。EC2 側の出力は見ていない
- docs との食い違い: [io_spec.md](io_spec.md) 1.1 の「`t_start` はルート先頭からの絶対秒」は実装と合わない (boot time)。4.2 の `_absmax` は「絶対値の最大」とあるが、実際は符号付きの値。どちらもこの文書では直していない
