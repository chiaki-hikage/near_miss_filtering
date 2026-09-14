# `ref/label_list.csv`（利用可 37 項目）でのヒヤリハット抽出

> **意味はすべて推定。** 単位・符号・符号化・`_TypeA/B/C` の違いは未確認。
> 実データで検証するまでこの解釈を前提に実装しない。

`利用可` に ◯ の付いた 37 項目だけを対象にする。全 254 項目のうち 37 項目。

---

## 0. 何が使えて、何が使えないか

### 使える動的信号は 100 ms の 9 本だけ

| 項目 | 推定される意味 |
|---|---|
| `Speed` | 車速 |
| `AccelerationFB_TypeA` | 前後加速度 |
| `AccelerationLR` / `AccelerationLR_TypeA` | 横加速度（2 系統） |
| `AngularVelocity` | ヨーレート |
| `ABSControl` | ABS 作動 |
| `VSCControl` | VSC（横滑り防止）作動 |
| `TRCControl` | TRC（トラクションコントロール）作動 |
| `PCSAlarmBuzzer_TypeA` | PCS 警報ブザー |

### 使えないもの（設計に直接効く）

| 失われたもの | 何ができなくなるか |
|---|---|
| **舵角（`SteeringAngle` 全系統）** | 舵角レート、反転回数、単軌道モデル、横滑り角、逆操舵 |
| **制動（`BrakePedal` / `BrakePressure`）** | 制動の有無・強さ。パニック制動の踏み替え判定 |
| **アクセル（`AccelPedalAngle`）** | アクセル OFF の検出 |
| **輪速 4 本** | 輪速ばらつき（ロック・スリップ） |
| **車間距離・レーダ** | THW、TTC、割り込み、急接近 |
| `PCSBrakeAssitState` | PCS の作動段階。**警報ブザーしか無い** |
| `ShiftGear` / `ParkingBrake` | 駐車・後退の除外 |
| `TurnLampSwitchStatus` | 意図的な車線変更の切り分け |
| `EstimateMu` / ワイパー / ヘッドランプ | 路面μ・天候・昼夜による重み付け |
| `AccelerationFB`（無印） | **`_TypeA` のみ利用可。**無印との違いは不明 |
| `WheelSpeedFL/FR/RL/RR` | 輪速のばらつき（ロック・スリップ） |

**運転者の操作入力が 1 つも残っていない。** 残るのは車両の運動（速度・加速度・
ヨー）と、車両自身の判断（安全系フラグ）だけ。

### 新たに使えるもの

| 項目 | 周期 | 推定される意味 |
|---|---:|---|
| `LDWBuzzer_TypeA/B/C` | 1 s | 車線逸脱警報ブザー |
| `LDWIndicator_TypeA/E` | 1 s | 車線逸脱警報表示 |
| `LDWIndicator_TypeD` | 20 s | 同上（別系統） |
| `LKASteeringSupport_TypeA` | 1 s | LKA 操舵支援の作動 |
| `LKASteeringSupport_TypeB` | — | 同上（周期不明） |
| `RadarCruiseControl` | 20 s | ACC の作動状態 |
| GPS `Latitude` / `Longitude` | 1 s | 位置 |
| MM `LinkID` / `LinkAdditionalID` / `AreaCode` | 1 s | 道路リンク・地域 |
| `TMNBuzzer` / `TMNSetState` | 20 s | **意味不明。** 何らかの警報。要確認 |

---

## 1. L1 — 車体運動ベースに戻さざるを得ない

操作入力が無いので、以前測って「劣る」と分かった**車体運動ベース**しか選べない。

| 使う項目 | 条件（窓内の生値のみ） |
|---|---|
| `Speed` | `max − min < Δv` → 加減速なし |
| `AccelerationFB_TypeA` | `\|max\| < 閾値` → 縦方向の動きなし |
| `AccelerationLR` | `\|max\| < 閾値` → 横方向の動きなし |
| `AngularVelocity` | `\|max\| < 閾値` → 旋回なし |

**override（必ず残す）**
```
ABSControl ∨ VSCControl ∨ TRCControl ∨ PCSAlarmBuzzer_TypeA
∨ LDWBuzzer_* ∨ LDWIndicator_*
```

**加速度が生値で来るので上界の見積もりが直接できる。** 現行のように
`v_max × max|yaw|` で `ay_kin` の上界を作る必要がなく、`AccelerationLR` の
`|max|` をそのまま使える。**その分だけ現行より正確に切れる。**

commaCarSegments での実測では、車体運動ベースの L1 は保持率 95% で削減 17%
（操作入力ベースは 20%）。**大きくは変わらない。**

---

## 2. L2 — 微分が 1 段減り、失われる特徴量が多い

### 作れるもの

| 特徴量 | 作り方 | 現行との違い |
|---|---|---|
| `ax` | `AccelerationFB_TypeA` をそのまま | **微分不要。** 現行は速度の中心差分 |
| `ay` | `AccelerationLR` をそのまま | 同上 |
| `yaw_rate` | `AngularVelocity` をそのまま | 同じ |
| `ay_kin` | `Speed × AngularVelocity` | 同じ。`ay` との整合検査に使える |
| `jerk` | `ax` の微分 | **10 Hz で 1 階微分。現行の 2 階から 1 段減る** |
| 方位変化 | `AngularVelocity` の積分 | 同じ |
| 横移動量 | `Speed × AngularVelocity` の 2 重積分 | 同じ |

### 作れないもの

`steer_rate` / `steer_reversals` / `steer_detrended` / `yaw_expected` /
`yaw_residual` / `beta_model` / `beta_excess` / `ws_spread` /
`thw_s` / `ttc_s` / `lead_distance_m` / `lead_vrel` / `cut_in_*` /
`panic_brake_*`

**単軌道モデルが成立しない。** 舵角が無いので「舵から期待されるヨーレート」が
出せず、ヨー残差も横滑り角も作れない。**横滑り 2 段フィルタは移植不可。**

### 周期の問題

100 ms（10 Hz）なので Nyquist は 5 Hz。**`jerk` は 1 階微分でも荒くなる。**
現行の平滑化窓（速度 0.25 s = 5 サンプル、加速度 0.25 s）はそのまま使えるが、
中心差分の刻みが 0.05 s → 0.1 s になる。閾値の再較正が要る。

---

## 3. L3 — 抽出できるイベント

### そのまま作れる（車両の判断）

| イベント | 使う項目 | 備考 |
|---|---|---|
| `abs_active` | `ABSControl` | 立ち上がりで検出。現行と同じ |
| `vsc_active` | `VSCControl` | 同上 |
| **`trc_active`** | `TRCControl` | **現行に無い。駆動輪の空転** |
| `pcs_alarm` | `PCSAlarmBuzzer_TypeA` | **警報のみ。**制動作動は取れない |
| **`lane_departure`** | `LDWBuzzer_*` / `LDWIndicator_*` | **現行に無い。車線逸脱を事実として取れる** |
| **`lka_intervention`** | `LKASteeringSupport_TypeA` | LKA が操舵を補正した |

**これらが新しい骨格になる。** 閾値を自分で決める余地がなく、
車両メーカーの判断そのものなので、精度の議論が恣意的にならない。

### 作れる（閾値判定）

| イベント | 使う項目 | 現行からの変更 |
|---|---|---|
| `hard_brake` | `AccelerationFB_TypeA < −2.5` | 生値なので**むしろ素直** |
| `hard_accel` | `AccelerationFB_TypeA > 2.5` | 速度ゲート `Speed > 8` で停止発進を除く |
| `lateral_accel` | `\|AccelerationLR\| > 3.0` | 同じ |
| `lane_change_candidate` | `AngularVelocity` のローブ対 + `Speed` | 舵角を使わない実装なので**そのまま動く** |
| `s_evasion` | `AngularVelocity` のローブ ≥ 3 + `AccelerationLR` | 同上 |

### 劣化する

| イベント | 現行 | この構成での代替 |
|---|---|---|
| `hard_steer` | 舵角レート ∧ `ay_kin` ∧ `ay_can` の連言 | **`\|AccelerationLR\|` ∧ `\|AngularVelocity\|` の 2 条件に減る。**舵角レートが抜けるぶん分離が落ちる |
| `weaving` | 舵角の反転 + 横加速度 + 舵角レート | **ヨーの反転のみ。**舵角の細かい修正操作が見えない |

### 作れない

`short_thw` / `low_ttc` / `cut_in_candidate` / `closing_fast` /
`panic_brake` / `panic_brake_pedal` / `brake_and_steer` /
`brake_after_evasion` / `counter_steer` / `wheel_speed_anomaly` /
`yaw_instability` / `risky_lane_change`（共起相手が減るため実質不可）

---

## 4. 人手ラベルで検証済みの risky を拾えるか

comma2k19 の人手ラベル 82 件、risky 10 件で照合する。

| risky の内訳 | 件数 | この構成で拾えるか |
|---|---:|---|
| `cut_in` / `short_thw` 系（車間が主因） | **6** | **× 直接は不可。**PCS 警報が鳴っていれば拾える |
| `hard_brake`（ax −3.30 / −4.35） | 2 | **○** `AccelerationFB_TypeA` で拾える |
| `cut_in` + `hard_brake`（ax −2.70） | 1 | **○** 制動側で拾える |
| `hard_steer`（\|ay\| 2.6〜2.7） | 1 | **△** 横加速度の閾値 3.0 では届かない。2.5 に下げれば拾える |

**確実に拾えるのは 10 件中 3 件、閾値を調整して 4 件。** 残る 6 件は
車間距離に依存しており、**PCS 警報が鳴ったかどうかに賭けることになる。**

参考として commaCarSegments 33.3 時間での安全系の発生率:
PCS 2 件（0.06/h）、ABS 11 件（0.33/h）、VSC 0 件。**きわめて稀。**
LDW の発生率は不明で、これが実用的な頻度で出るかが鍵になる。

---

## 5. 位置情報の使い道

GPS と MM が 1 s で取れるので、**現行データには無かった層別ができる。**

| 用途 | 使い方 |
|---|---|
| 道路種別の層別 | `LinkID` / `AreaCode` で高速・一般道・市街地を分ける |
| 交差点の同定 | `LinkID` の切り替わり。交差点内のイベントを別扱いに |
| 「密集状態」の推定 | 市街地リンク内で `Speed` が低いまま `AccelerationFB_TypeA` の符号反転が頻発する区間 |
| 同一地点の反復 | 同じ `LinkID` で繰り返し起きるイベントは道路構造要因の可能性 |

**車間距離が無いので「密集」は直接測れない。** 上記は間接的な推定にとどまる。

---

## 6. 実現できる構成の見立て

```
L1  車体運動ベース（Speed / AccelerationFB_TypeA / AccelerationLR / AngularVelocity）
    + 安全系・LDW の override

L2  加速度とヨーは生値をそのまま。jerk は 1 階微分
    単軌道モデル・横滑り角・車間指標は作らない

L3  骨格 = 車両の判断
      ABS / VSC / TRC / PCS警報 / LDW / LKA介入
    補助 = 閾値判定
      hard_brake / hard_accel / lateral_accel / lane_change / s_evasion
    劣化版
      hard_steer（横加速度 ∧ ヨー）/ weaving（ヨーの反転のみ）

優先度  安全系・LDW が立ったものを最優先
        次に AccelerationFB_TypeA の裾（強い制動）
        位置情報で層別（高速 / 一般道 / 交差点）
```

**性格が変わる。** 現行は「車間距離と自車挙動から危険を推定する」設計だったが、
この構成では**「車両の安全装置が反応したか」が主軸**になる。
精度優先の方針とは相性が良い一方、装置が作動しなかったヒヤリハット
（運転者が早めに対処した場合）は原理的に拾えない。

---

## 7. 着手前に確認すべきこと

1. **`LDWBuzzer` / `LDWIndicator` の発生頻度。** これが実用的な頻度で出るかが
   この構成の成否を分ける。稀すぎれば骨格にならない。
2. **`AccelerationFB_TypeA` と `AccelerationLR` / `_TypeA` の違い。**
   前後は `_TypeA` しか無く、横は 2 系統ある。単位（m/s² か G か）と符号。
3. **`TMNBuzzer` / `TMNSetState` の意味。** 警報系なら骨格に加えられる。
4. **`PCSAlarmBuzzer_TypeA`（100 ms）と `_TypeB`（20 s）の違い。**
5. **`サンプリング周期ms` は送信周期。** 実際の受信間隔を実データで測る。
6. **前方情報が無い前提で、どこまで許容するか。** 検証済みの risky 10 件のうち
   6 件が車間依存だった。**この 6 割を諦めるか、別の入力（映像・レーダ）を
   足すかの判断が要る。**
