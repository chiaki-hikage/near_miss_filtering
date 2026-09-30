# イベント判定仕様

L3 が検出する 24 種のイベントの分類・使用する特徴量・判定条件・閾値。

**この表は `configs/detection.yaml` から機械的に生成したものを貼っている。**
手で書き写していないので、閾値を変えても表がずれない。再生成は次のとおり。

```bash
uv run python scripts/report/dump_event_rules.py --markdown
```

処理の位置づけは [pipeline_overview.md](pipeline_overview.md)、
出力の列は [io_spec.md](io_spec.md) を見ること。

---

## 1. 分類

イベントを 5 つに分けて扱う。**分類は検出条件ではなく、後段での読み方の区分**で、
`configs/` には入っていない。1 つのイベントが複数の分類に入ることはない。

| 分類 | イベント | 主に使う量 |
|---|---|---|
| **縦方向** | `hard_brake` / `hard_accel` / `panic_brake` / `panic_brake_pedal` / `panic_brake_with_lead` | `ax_mps2`、ブレーキ・アクセル |
| **横方向** | `hard_steer` / `lateral_accel` / `weaving` / `yaw_instability` / `counter_steer` / `lane_change_candidate` / `risky_lane_change` | `ay_kin_mps2`、`yaw_rate_dps`、舵角 |
| **前方接近** | `short_thw` / `low_ttc` / `cut_in_candidate` / `closing_fast` | レーダ由来の `thw_s` / `ttc_s` / `lead_distance_m` |
| **複合回避** | `s_evasion` / `brake_and_steer` / `brake_after_evasion` | 縦と横の同時成立・前後関係 |
| **安全 / スリップ** | `abs_active` / `vsc_active` / `slip_warn_active` / `aeb_active` / `wheel_speed_anomaly` | 安全系フラグ、輪速 |

---

## 2. 判定方式

| 方式 (`op`) | 意味 |
|---|---|
| `lt` / `gt` / `abs_gt` | 判定量が閾値を `min_duration_s` だけ継続して超える |
| `rising` | フラグが閾値 (0.5) を**上向きに跨ぐ**瞬間 |
| `agreement` | 複数条件の AND。`tolerance_s` だけ時間ずれを許容する |
| `cooccurrence` | 他イベントとの共起 (`window_s` 内に `min_count` 件以上) |
| `sequence` | `first` の後 `min_gap_s`〜`max_gap_s` に `then` が起きる |

**ゲート**は「その条件下では判定しない」区間を外す。閾値を緩めるのではなく、
判定の対象から外す。

**段 (stage)** は 1 = CAN の基本信号だけで決まるもの、2 = レーダまたは安全系
フラグを使うもの。段 1 だけで動かすこともできる。

**重み (weight)** は候補統合時の severity に効く。1 イベントの severity には
上限 3 がある。

---

## 3. イベント一覧

| イベント | ラベル | 判定量 | 条件 | 継続 | 結合 | ゲート | 重み | 段 | 派生節 |
|---|---|---|---|---|---|---|---|---|---|
| hard_brake | 急ブレーキ | ax_mps2 | ax_mps2 < -2.5 | 0.3 s | 1 s | - | 1.0 | 1 |  |
| hard_accel | 急加速 | ax_mps2 | ax_mps2 > 2.5 | 0.3 s | 1 s | v_mps > 8 | 0.4 | 1 |  |
| hard_steer | 急操舵 | ay_kin_mps2 | \|steer_rate_dps\| > 25 AND \|ay_kin_mps2\| > 2 AND \|ay_can_mps2\| > 1.4  (時間ずれ許容 ±0.3 s) | 0.15 s | 1 s | net_heading_win_deg < 15 かつ v_mps > 5 | 0.3 | 2 |  |
| lateral_accel | 横加速度過大 | ay_kin_mps2 | \|ay_kin_mps2\| > 3 | 0.2 s | 1 s | net_heading_win_deg < 15 かつ v_mps > 5 | 0.3 | 2 |  |
| weaving | 蛇行 | weave_active | weave_active > 0.5 | 0.5 s | 2 s | - | 0.5 | 2 | weaving |
| short_thw | 車間時間過小 | thw_s | thw_s < 0.6 | 0.5 s | 1 s | - | 1.0 | 1 | lead |
| low_ttc | 衝突余裕時間過小 | ttc_s | ttc_s < 2 | 0.3 s | 1 s | - | 1.5 | 1 | lead |
| yaw_instability | ヨー応答の乖離 | yaw_residual_sigma | \|yaw_residual_sigma\| > 6 | 0.3 s | 1 s | v_mps > 10 かつ net_heading_win_deg < 15 | 1.5 | 2 | physics |
| s_evasion | S字回避 | s_evasion_active | s_evasion_active > 0.5 | 0.3 s | 1 s | - | 2.0 | 2 | s_evasion |
| panic_brake | 急なアクセルOFFからの制動 | panic_brake_active | panic_brake_active > 0.5 | 0.2 s | 1 s | - | 0.4 | 1 | panic_brake |
| panic_brake_with_lead | 前方要因の急制動 | panic_brake | panic_brake と同時に [short_thw OR cut_in_candidate OR closing_fast OR low_ttc] が 2 s 以内に 1 件以上 | - | - | - | 2.0 | 2 |  |
| closing_fast | 急接近 | closing_fast_active | closing_fast_active > 0.5 | 0.5 s | 1 s | v_mps > 15 | 1.2 | 2 | closing |
| brake_and_steer | 制動＋操舵の回避 | hard_brake | hard_brake と同時に [hard_steer OR s_evasion] が 1 s 以内に 1 件以上 | - | - | - | 2.0 | 2 |  |
| brake_after_evasion | 回避後の制動 | ['s_evasion', 'hard_steer'] | ['s_evasion', 'hard_steer'] の 0〜3 s 後に hard_brake | - | - | - | 2.0 | 2 |  |
| cut_in_candidate | 割り込み | cut_in_active | cut_in_active > 0.5 | 0.3 s | 1 s | - | 1.5 | 2 | cut_in |
| lane_change_candidate | 車線変更の候補 | lc_active | lc_active > 0.5 | 0.5 s | 0.5 s | - | 0.0 | 2 | lane_change |
| risky_lane_change | 危険な車線変更の疑い | lane_change_candidate | lane_change_candidate と同時に [short_thw OR cut_in_candidate OR closing_fast OR low_ttc OR hard_steer OR lateral_accel OR hard_brake OR abs_active OR vsc_active OR aeb_active OR wheel_speed_anomaly] が 2 s 以内に 1 件以上 | - | - | - | 2.0 | 2 |  |
| abs_active | ABS 作動 | abs_active_flag | abs_active_flag が 0.5 を上向きに跨ぐ | 0 s | 1 s | - | 2.0 | 2 |  |
| vsc_active | VSC 作動 | vsc_active_flag | vsc_active_flag が 0.5 を上向きに跨ぐ | 0 s | 1 s | - | 2.5 | 2 |  |
| slip_warn_active | 滑り警告灯 (TRC/VSC 介入) | slip_warn | slip_warn > 3.5 | 0 s | 1 s | - | 2.0 | 2 |  |
| aeb_active | 純正 AEB 作動 | precollision_active | precollision_active が 0.5 を上向きに跨ぐ | 0 s | 1 s | - | 2.5 | 2 |  |
| panic_brake_pedal | アクセルOFFからの急制動 (ペダル判定) | panic_brake_pedal_active | panic_brake_pedal_active > 0.5 | 0.2 s | 1 s | - | 1.0 | 2 | panic_brake |
| counter_steer | 逆操舵 | counter_steer_active | counter_steer_active > 0.5 | 0.2 s | 1 s | v_mps > 10 かつ net_heading_win_deg < 15 | 0.3 | 2 | physics |
| wheel_speed_anomaly | 輪速の異常 | ws_anomaly_active | ws_anomaly_active > 0.5 | 0.2 s | 0.3 s | - | 1.5 | 2 | wheel_speed |

---

## 4. 全イベント共通の設定

**候補化 (`build_candidates`)**

| 項目 | 値 |
|---|---|
| 前後の余白 `window_pad_s` | 2 s |
| 候補の結合 `merge_gap_s` | 2 s |
| 1 イベントの severity 上限 | 3 |
| 共起ボーナス | 0.5 |

**前処理 (L2)**

| 項目 | 値 |
|---|---|
| 再サンプル | 20 Hz |
| 欠測とみなす受信間隔 | 0.5 s |
| 端の切り落とし | 0.6 s |
| 平滑化 (対称移動平均) | 車速 0.25 s / 舵角 0.15 s / 縦加速度 0.25 s / 輪速 0.3 s |

---

## 5. 派生特徴量の閾値

上の表で判定量が `*_active` などになっているものは、別節で作られる派生量。
その節の設定値を載せる。

**`[lead]`** — `lane_half_width_m=1.8` / `min_distance_m=1.0` / `max_distance_m=150.0` / `max_hold_s=0.5` / `min_ego_speed_mps=5.0` / `min_target_speed_mps=2.0` / `min_target_speed_ratio=0.2` / `curvature_compensation=True` / `curvature_min_speed_mps=3.0`

**`[lane_change]`** — `curvature_window_s=10.0` / `enter_dps=0.6` / `lobe_merge_gap_s=0.5` / `pair_max_gap_s=3.0` / `min_lobe_heading_deg=1.5` / `max_net_heading_deg=2.5` / `min_duration_s=1.0` / `max_duration_s=8.0` / `min_offset_m=2.0` / `max_offset_m=6.0` / `min_speed_mps=10.0`

**`[physics]`** — `min_speed_mps=3.0` / `counter_steer_min_yaw_dps=3.0` / `counter_steer_min_rate_dps=15.0`

**`[s_evasion]`** — `curvature_window_s=10.0` / `enter_dps=1.0` / `lobe_merge_gap_s=0.3` / `lobe_gap_s=1.5` / `min_lobes=3` / `min_duration_s=1.0` / `max_duration_s=5.0` / `min_lateral_accel_mps2=2.0` / `min_excursion_m=1.2` / `max_return_m=1.0` / `min_speed_mps=10.0`

**`[panic_brake]`** — `lookback_s=1.5` / `accel_threshold=0.3` / `brake_threshold=-3.0`

**`[wheel_speed]`** — `min_excess_mps=0.4` / `min_speed_mps=5.0` / `tolerance_s=0.5` / `corroborate_ax_mps2=1.5` / `corroborate_yaw_sigma=3.0`

**`[closing]`** — `max_thw_rate=-0.4` / `max_thw_s=1.5`

**`[cut_in]`** — `min_jump_m=5.0` / `persist_s=0.8` / `min_distance_drop_m=10.0` / `max_appear_distance_m=40.0` / `max_thw_after_s=0.8` / `min_speed_mps=8.0` / `window_s=3.0`

**`[weaving]`** — `curvature_window_s=10.0` / `enter_dps=1.0` / `lobe_merge_gap_s=0.3` / `lobe_gap_s=2.0` / `min_reversals=3` / `min_lateral_accel_mps2=1.0` / `min_steer_rate_dps=8.0` / `min_duration_s=2.0` / `max_duration_s=8.0` / `max_net_heading_deg=8.0` / `min_speed_mps=10.0` / `detrend_window_s=5.0` / `amplitude_deg=3.0` / `count_window_s=10.0`

**`[turn]`** — `half_window_s=3.0` / `max_net_heading_deg=15.0`

---

## 6. 補足・注意点

### 6.1 `slip_warn` は多値。値ごとに意味が違う

| 値 | 意味 |
|---|---|
| 3 | **始動時のランプ試験。** 走行中の滑りではない |
| 4 | **TRC / VSC の実作動** |
| 7 | 継続点灯。**車種で意味が違う** (Highlander では始動時のランプ試験を含む) |

`slip_warn_active` の判定は `> 3.5` なので値 4 と 7 が入る。
L1 の無条件通過は**値 4 のみ** (`3.5 < x ≤ 4.5`)。
`configs/priority.yaml` の `always` も閾値 3.5 を明示している。
**列名だけ書くと閾値 0.5 になり、ランプ試験を拾う。**

### 6.2 `tc_disabled` と `abs_fault` は「作動」ではない

`tc_disabled` は「TC が切られている状態」、`abs_fault` は「故障」。
どちらも介入の記録ではないので、**通過条件にも優先度の `always` にも使わない**。
数えるだけ。

### 6.3 横加速度は 2 系統ある

* `ay_can_mps2` — CAN の横 G センサ
* `ay_kin_mps2` — `車速 × ヨーレート` の運動学値

**直進制動中に CAN 側が見かけの横 G を出すことがある** (Highlander で 2.90 件/h)。
`hard_steer` は両方に条件を課している (`|ay_kin| > 2 AND |ay_can| > 1.4`)。
摩擦要求のような合成量を出すときは `ay_kin_mps2` を優先する。

### 6.4 ゲート `net_heading_win_deg < 15` の意味

`hard_steer` / `lateral_accel` / `yaw_instability` / `counter_steer` に付く。
**回避操作は元の向きへ戻るが、カーブは戻らない。** 大きさ (`|ay|` や舵角レート)
だけではカーブと回避を分離できないため、正味の方位変化で切る。

### 6.5 `yaw_residual_sigma` は車種を選ぶ

ステアリングギヤ比・アンダーステア勾配・ヨー残差の標準偏差が要る。
**諸元が揃っている車種でしか計算されない** (現在は RAV4 と RAV4 TSS2)。
他の車種では `yaw_instability` が出ない。

### 6.6 `lane_change_candidate` の重みは 0

車線変更そのものは危険ではない。**重み 0 で severity に効かせず**、
`risky_lane_change` (他のイベントとの共起) で初めて危険とみなす。

### 6.7 ABS の実体は輪速差ではなく単輪の減速深さ

4 輪が同時にロックと解放を繰り返すため、輪と輪の差 (`ws_spread_mps`) には
出にくい。**ABS 事象の 73% が単輪で −8 m/s² 以下に落ちる** (対照窓では 2%)。
`wheel_decel_mps2` (4 輪のうち最も深い瞬間減速度) がこれを捉える。

---

## 7. スリップの直接証拠 (`direct_slip`)

上の 24 種とは別に、**車両自身が「滑った」と判断した瞬間**だけを拾う枝がある
(`src/near_miss/direct_slip.py`)。モデル推定を一切通さない。

### 7.1 証拠の種類

| 区分 | 証拠 | 意味 |
|---|---|---|
| **制御作動 (DIRECT)** | `slip_warn` / `abs_active` / `vsc_active` | 車両自身の判断。こちらが閾値を決める余地がない |
| **物理挙動 (SUPPORT)** | `wheel_spread` / `wheel_decel` / `lateral` / `yaw_residual` | 人が閾値を決めるので、**単独では確定にしない** |
| **参考 (CONTEXT)** | `tc_disabled` | 証拠に数えない。候補行に添えるだけ |

**確認単位を作れるのは `TRIGGER` = 制御作動 3 種 + `wheel_spread` のみ。**
`wheel_decel` / `lateral` / `yaw_residual` は閾値が対照窓の p98〜p99 にあり、
単独で事象を作らせると通常走行が大量に並ぶため、裏付けとしてのみ数える。

### 7.2 等級

**作動の数では決めない。裏付けが何系統あるかで決める。**

| 等級 | 条件 |
|---|---|
| `multi_evidence` | 異なる物理メカニズムの裏付けが **2 系統以上** |
| `high_evidence` | 裏付けの取れた作動があるが、系統は 1 つ |
| `activation_only` | 作動はしたが、対応する物理挙動が確認できない |
| `physics_only` | 作動なし。物理挙動だけ |

系統は 2 つ (`configs/detection.yaml` の `mechanisms`)。

| 系統 | 含む証拠 |
|---|---|
| 輪速 | `wheel_spread` / `wheel_decel` |
| 横方向 | `lateral` / `yaw_residual` |

どの作動に何が対応するかは `corroboration` が決める。

| 作動 | 裏付けになる物理挙動 |
|---|---|
| `abs_active` | `wheel_decel` (主) / `wheel_spread` |
| `slip_warn` | `wheel_spread` / `wheel_decel` / `lateral` / `yaw_residual` |
| `vsc_active` | `lateral` / `yaw_residual` |

**なぜ作動の数で決めないか。** RAV4 では `slip_warn` と `abs_active` が 90%
同時に立つ。同じ `wheel_decel` で 2 つの作動を裏付けても、分かった事実は
「輪が 1 本深く落ちた」という 1 つだけなので 1 系統と数える。

**`counter_steer_active` は裏付けに使わない。** 対照窓の 15% で立つため、
証拠としての分離能力が無い。
