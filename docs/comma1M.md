# comma1M — 位置による地域の絞り込みと悪天候下の候補抽出
> データセット間の比較は [datasets.md](datasets.md) にまとめてある。

commaCarSegments には位置情報が無く、地域による絞り込みができなかった
(`docs/comma_car_segments.md`)。comma1M はセグメントごとに自己位置を持つので、
寒冷地・降雪可能地域を選ぶことができる。ここではその調査結果と実装をまとめる。

## 1. データセットの規模

`https://huggingface.co/api/datasets/commaai/comma1M/tree/main/data` を
すべて辿って数えた実測値 (2026-08-20 時点)。

| 項目 | 値 |
|---|---|
| セグメント数 | **4,216** |
| 合計時間 | 70.4 h (1 セグメント約 60 s) |
| リポジトリ使用量 | 550 GB |

名前に反して現時点で公開されているのは 4,216 件しかない。
commaCarSegments (188,883 件 / 3,148 h) の 2.2% にあたる。

### セグメント 1 件の中身と大きさ

| ファイル | 大きさ | 内容 |
|---|---|---|
| `fcamera.hevc` | 約 75 MB | 前方カメラ 1200 frame (1928x1208) |
| `ecamera.hevc` | 約 75 MB | 広角カメラ (comma three 以降のみ) |
| `localizer.safetensors` | **2.54 MB** | オフライン自己位置推定 |
| `frame_info.safetensors` | 約 50 KB | 各カメラのフレーム時刻と索引 |
| `thumbnail.jpg` | **13 KB** | 1 枚の静止画 (482x302) |

CAN は入っていない。したがって ABS / VSC / 舵角 / 輪速 / レーダは使えない。

## 2. localizer で公式に意味が確認できる列

Dataset Card の使用例に載っているのは次の 2 つだけ。

```python
latitude, longitude, _ = ecef2geodetic(*states[:, :3].T)
speed = np.linalg.norm(states[:, 7:10], axis=1)
```

| キー | 形 | 内容 |
|---|---|---|
| `states` | (N, 43) f64 | 100 Hz。列 0-2 = ECEF [m]、列 7-9 = 速度ベクトル [m/s] |
| `t` | (N,) | 端末の boot time [s]。**壁時計ではない** |
| `frame_states` | (1200, 43) | 映像フレーム時刻での同じ状態 |
| `frame_t` | (1200,) | 同上の時刻 |
| `rpy`, `wide_from_device_euler` | (3,) | 取り付け姿勢 |

残り 33 列の意味は公開仕様に無いので使わない。

**録画日時は入っていない。** `t` は boot time で、`frame_info` にも壁時計は無い。
このため「積雪しうる地域か」は言えるが「積雪期に走ったか」は言えない。
実際の路面状態は thumbnail の目視に頼るしかない。

### 検算

ECEF の差分から求めた移動距離と、速度の時間積分の比が 15 件すべてで 1.000 だった。
位置と速度が互いに整合していることの確認になる。

## 3. 段階を分けた取得

映像を落とさずに済ませるため、必要な情報から順に取る。

| 段階 | 対象 | 転送量 | スクリプト |
|---|---|---|---|
| A. 位置 | 4,216 件 | **9.6 MB** | `fetch_comma1m_positions.py` |
| B. 地域の逆引き | 通信なし | 0 | `select_comma1m_region.py` |
| C. 見た目 (天候) | 4,216 件 | **57 MB** | `fetch_comma1m_thumbnails.py` |
| D. 自己運動 | 選んだ 831 件 | **2.11 GB** | `fetch_comma1m_localizers.py` |
| E. 映像 | 候補の前後 7 秒だけ 23 件 | **197 MB** | `review_comma1m_clips.py` |

段階 A は localizer 全体 (2.54 MB) を落とさない。safetensors のヘッダを
HTTP Range で読んで `states` の位置を求め、必要な 3 行だけを取り出す。
1 件あたり 2.3 KB で済む。全件を落とすと 10.7 GB になるので 1,100 分の 1。

HF の CDN は単一 Range (206) に対応するが、複数 Range (multipart/byteranges) は
416 を返す。1 セグメントにつきヘッダ 1 回 + 行 3 回の計 4 リクエストになる。

## 4. 地理分布 (4,216 件全数)

逆引きは `reverse_geocoder` (オフライン、cities1000)。

| 国 | 件数 | | 国 | 件数 |
|---|---:|---|---|---:|
| US | 3,619 | | AE | 16 |
| CA | 182 | | NL / FR / BE / PL | 各 15 |
| AU | 73 | | AT | 14 |
| GB | 69 | | TW | 11 |
| CH | 24 | | SE | 11 |
| DE / JP | 各 17 | | NO | 10 |

上位の一次行政区分: California 931 / Texas 259 / Ohio 206 / Florida 166 /
Washington 151 / Ontario 126 / Virginia 126 / Illinois 120 / Colorado 104。

**California だけで全体の 22%** を占め、その標高中央値は 24 m
(900 m 以上は 931 件中 14 件)。州名だけで寒冷地を判定すると誤る。

## 5. 寒冷度の判定

`configs/datasets/comma1m.yaml` の `cold_region`。行政区分・標高・緯度の
いずれか 1 つでも当たれば候補にする (再現率優先)。

- 州の全域が多雪な 22 州 (Minnesota, Michigan, New York, Colorado, Utah など) → heavy
- 冬に積雪はあるが多くない 10 州 (Ohio, Illinois, Indiana, Virginia など) → moderate
- California / Washington / Oregon / Arizona / Nevada / New Mexico は
  州内の気候差が大きすぎるので一覧に入れず、標高 (heavy 1,600 m / moderate 900 m) で拾う
- 緯度は保険。海洋性気候を過大評価するので heavy 50 度 / moderate 45 度と高めに置く

| 寒冷度 | 件数 | 時間 | うち夜間 |
|---|---:|---:|---:|
| heavy | 1,125 | 18.8 h | 336 |
| moderate | 1,067 | 17.8 h | 240 |
| low | 2,024 | 33.8 h | 473 |

heavy の内訳上位: Ontario 126 / Colorado 104 / New York 99 / Pennsylvania 78 /
Utah 76 / Massachusetts 69 / Minnesota 65 / Iowa 61。

## 6. thumbnail からの天候の判定

`scripts/score_comma1m_weather.py`。HSV と色鮮やかさから指標を作る。

### 失敗した最初の案

路面領域 (画面下 35% の中央) の白画素割合を雪の指標にしたところ、
**上位 24 件すべてが乾いた明るいコンクリート舗装だった**。California の
高速道路と Florida の幹線道路が並ぶ結果になり、雪は 1 件も入らなかった。

### 採用した指標

積雪は路面より先に路肩・法面に残り、画面全体の色味が失われる。

```
snow_score = side_white x clip(1 - colorfulness / 0.12, 0, 1)
```

- `side_white`: 縦 40-85%、左右それぞれ外側 22% の帯で S<0.18 かつ V>0.60 の割合
- `colorfulness`: Hasler-Susstrunk

雨天・濡れ路面・曇天は別の指標で拾える。

```
wet_score = clip(1 - sat_mean / 0.15) x clip((val_mean - 0.28) / 0.15)
```

「明るいのに彩度が無い」状態を見る。上位には雨の飛沫、濡れた路面の反射、
どんよりした曇天がよく揃った。夜間 (`val_mean < 0.25`、1,049 件) は
暗さと雪の区別がつかないので除く。

### 地理と見た目は併用しないと効かない

`snow_score >= 0.05` を全体に掛けると 71 件が残るが、その内訳は
California 15 / Ohio 7 / Utah 5 / Texas 4 / Florida 3 で、南部の明るい舗装が
混じる。寒冷度 heavy+moderate に限ると 38 件になる。

地理だけでは季節が分からず、見た目だけでは舗装と雪を取り違える。両方が要る。

### 精度は 3 割。小さく並べて数えてはいけない

38 件を 170x106 の一覧で目視したときは「26〜30 件が積雪」と数えたが、
360x225 で見直すと **積雪と確認できるのは 12 件 (32%)** だった。
明るいコンクリート舗装と白い砂利路肩は、小さく表示すると積雪と区別がつかない。
実際、`snow_score` 最上位の 4 件 (Indiana 0.381 / Ohio 0.218 /
Washington 0.185 / Wisconsin 0.166) は映像を見るとすべて夏の乾いた舗装だった。

内訳 (38 件):

| 判定 | 件数 |
|---|---:|
| 積雪あり | 12 |
| 判断がつかない | 4 |
| 濡れているが雪ではない | 4 |
| 乾いている (誤り) | 18 |

**目視確認は必ず原寸に近い大きさで行うこと。** 小さい一覧は候補を並べる用途に留める。

雨・濡れ路面の側は精度が高い。`wet_score` 上位 12 件を同じ大きさで見たところ
**11 件が実際に濡れた路面または降雨**だった (Nevada / Illinois x3 /
California x4 / Ohio / England / Idaho)。雪より雨のほうが見分けやすい。

## 7. 自己運動から作る正規化信号

CAN が無いので、正規化チャネルは 2 本だけになる。

| チャネル | 由来 |
|---|---|
| `speed_mps` | 速度ベクトルのノルム |
| `yaw_rate` | 速度ベクトルの向きの変化率 [deg/s]、左回りが正 |

`yaw_rate` は**車両のヨーレートセンサではなく、進路 (course over ground) の
変化率**である。両者は横滑り角のぶんだけ食い違い、滑っている最中は一致しない。
localizer 自体が車両運動拘束を仮定して融合した推定でもあるので、
**この値からスリップの有無を判定しない**。見るのは進路と速度の変化だけ。

向きは低速で定義できないため、対地速度 2 m/s 未満は NaN にして埋めない。
方位を微分する前に 0.1 s (100 Hz で 11 点) で均す。

ECEF → ENU の回転は `pymap3d.ecef2enuv` と 3.6e-15 まで一致することを確認した。

これで既存の `features.py` / `detectors.py` がそのまま動く。舵角・ブレーキ圧・
輪速・レーダ・車両諸元を要する検出は、入力が無いので自動的に無効になる。

| 動く | 動かない (入力が無い) |
|---|---|
| hard_brake / hard_accel / panic_brake | hard_steer (舵角レートが要る) |
| lateral_accel / lane_change / risky_lane_change | weaving / counter_steer |
| s_evasion / brake_and_steer / brake_after_evasion | short_thw / low_ttc / cut_in (レーダ) |
| | yaw_instability / wheel_speed_anomaly / ABS / VSC / AEB |

## 8. 天候グループ別のスクリーニング結果

`scripts/screen_comma1m.py`。snow 300 件 / wet 234 件 / control 300 件、
計 13.6 h。閾値は comma2k19 で決めたまま (`configs/detection.yaml`)。

### イベント件数 [件/h]

| イベント | snow | wet | control |
|---|---:|---:|---:|
| hard_brake | 17.3 | 23.5 | 21.4 |
| hard_accel | 19.0 | 19.1 | 16.7 |
| lane_change_candidate | 6.9 | 7.8 | 4.9 |
| risky_lane_change | 0.8 | 0.8 | 0.4 |
| panic_brake | 1.6 | 3.7 | 3.5 |
| lateral_accel | 0.8 | 0.0 | 1.0 |
| s_evasion | 0 | 0 | 0 |

急制動は雪で**減る**。最高速度も snow 36.1 m/s に対し control 41.0 / wet 46.0 m/s と
低い。運転そのものが慎重になるためと考えられる。
一方、車線変更系は雪・雨で 1.4〜1.6 倍に増える。

### 特徴量の分位点

| 特徴量 | 群 | p99.9 | p99.99 | 最大 (絶対値) |
|---|---|---:|---:|---:|
| `lat_jerk_mps3` | snow | **7.67** | **13.49** | **31.0** |
| | wet | 6.49 | 10.65 | 19.4 |
| | control | 6.58 | 11.47 | 18.2 |
| `ay_kin_mps2` | snow | 4.36 | **7.25** | 8.24 |
| | control | 3.99 | 5.17 | 8.02 |
| `ax_mps2` (減速側) | snow | -5.28 | -6.11 | -10.04 |
| | control | -5.91 | -6.04 | -9.20 |

**横方向の裾だけが雪で明確に重い。** 縦方向は差が無いか、むしろ雪で穏やか。
悪天候下の異常挙動を探すなら、減速の強さではなく横運動の急峻さを見るべき、
ということになる。

### s_evasion が 1 件も出ない

`min_lobes: 3` / `min_lateral_accel_mps2: 2.0` / `min_excursion_m: 1.2` /
`min_speed_mps: 10.0` の連言で、13.6 h では 0 件だった。
comma2k19 の CAN 由来ヨーレート向けに決めた条件なので、
course rate に対しては改めて分布から決め直す必要がある。

## 9. 候補の映像確認 (段階 E)

### 部分取得の仕組み

`frame_info.safetensors` (50 KB) に次が入っている。実測で確認した内容。

| キー | 内容 |
|---|---|
| `<cam>/index` (1201, 2) u32 | 列 0 = フレーム種別、列 1 = ファイル内バイト位置 |
| `<cam>/t` (1200,) | 各フレームの時刻。**localizer の t と同じ boot time** |
| `<cam>/global_prefix` (82,) u8 | VPS/SPS/PPS (3 バイト開始コード) |

- `index[-1, 1]` はファイルサイズと一致する (最終行は番兵)
- 列 0 は 2 が鍵フレームで 1200 枚中 40 枚。30 フレーム (1.5 秒) おき
- ファイル先頭 83 バイトもパラメータセット。`global_prefix` と中身は同じで、
  開始コードが 4 バイトか 3 バイトかだけが違う

したがって「対象フレーム直前の鍵フレームから必要な範囲まで」を Range で取り、
先頭に `global_prefix` を付ければ、それだけで復号できる HEVC になる。
`scripts/review_comma1m_clips.py` がこれを行う。

前後 3.5 秒 (実測 93〜169 フレーム) で **1 件あたり 4.6〜10.5 MB**。
23 件で 197 MB。全編を落とすと同じ 23 件で 1,725 MB になる。

### 確認した 23 件で分かったこと

**1. 実際に雪の路面で起きている候補がある。**
Pennsylvania Elverson (`c581ef0e`) は路肩に雪が積もった圧雪・凍結の山道で、
対向車とすれ違う場面に急制動 (ax -3.95 m/s^2) と横運動 (|ay_kin| 4.17 m/s^2) が
18.5 m/s (67 km/h) で重なっていた。狙っていた事象そのもの。

**2. `snow_score` の誤りは映像で確実に落ちる。**
確認した A 群 8 件のうち、実際に雪だったのは Pennsylvania の 2 件だけ。
Indiana Beech Grove / Vancouver / Indiana Zionsville / Wisconsin River Hills は
すべて夏の乾いた舗装だった。thumbnail の判定を最終判断にしてはいけない。

**3. 横躍度のピークは、ほとんどが道路形状である。**
雪確認セグメントの `lat_jerk_mps3` 上位を見たところ、
Virginia Norfolk の 13.4 m/s^3 はランプの進入、
Utah South Jordan の -11.8 m/s^3 は交差点の左折だった。
横躍度だけを条件にすると通常の旋回で埋まる。
`lateral_accel` と同じ `net_heading_win_deg < 15` の門を必ず併用する。

**4. 「制動 + 横運動」も、門が無いと一時停止からの右左折で埋まる。**
Illinois Coal City (`d3360033`) の ax -6.6 / ay 3.99 / lat_jerk 7.55 は
一時停止の交差点で止まって曲がっただけだった。
既存の `brake_and_steer` が `hard_steer` / `s_evasion` を要求し、
どちらも旋回の門を持っていることが効いている。

**5. `wet_score` の上位は本当に濡れているが、群の下位は違う。**
上位 12 件は 11 件が濡れ路面。一方、群 (上位 300 件、`wet_score >= 0.30`) の
severity 上位だった La Jolla (0.43) と Ottawa (0.39) は、
映像では乾いた路面の薄暮・曇天だった。`wet_score >= 0.5` 程度まで
絞らないと天候の群として使えない。

## 10. 分かっている限界

1. **母数が小さい。** 4,216 件 / 70.4 h。積雪が確認できるのは数十件。
   長い裾のイベントを数で集める用途には足りない。
2. **録画日時が無い。** 季節が分からないので、寒冷地の絞り込みは
   「積雪しうる地域」までしか行かない。
3. **CAN が無い。** ABS / VSC / AEB / 舵角 / 輪速が使えないため、
   commaCarSegments で使っていた高緊急度の判定材料がそのまま失われる。
4. **車種が分からない。** 車両諸元が無いので自転車モデルの残差が出せない。
5. **thumbnail は 1 枚。** 60 秒のうち 1 瞬しか見ていないので、
   途中で降り出した雨などは取りこぼす。
6. **積雪の判定精度が低い。** `snow_score >= 0.05` + 寒冷地で 3 割。
   映像で確認するまで積雪とみなせない。雨・濡れ路面は 9 割で、こちらは使える。
