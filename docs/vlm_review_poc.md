# 候補イベントの VLM 確認 — Phase 1 探索的 PoC

抽出済みのヒヤリハット候補を、映像と CAN をあわせてローカル VLM に見せ、
**何が起きているかを説明・分類できるか**を確かめる。

    既存の抽出          →  VLM review layer (後段)
    candidates.csv          映像 + CAN  →  判定 (JSON)
    labels.csv (人手)       一括判定 / オンライン判定

**既存の抽出ロジックは一切変更しない。** `detectors.py` / `scoring.py` /
`sideslip.py` / `features.py` には手を入れず、`src/near_miss/vlm/` として
独立した後段に置く。設定は `configs/vlm.yaml`。

追加学習・LoRA・fine-tuning は行わない。**既存の pretrained/instruct モデルを
そのまま使った場合の zero-shot 能力**を比較する。

---

## 1. 答えたい問い

| # | 問い | 主に答える測定 | Phase 1 で得られる確度 |
|---|---|---|---|
| 1 | 人間と概ね同じように危険/非危険を判断できるか | 条件 C の `risky` 一致、κ、説明文の 3 段階評価 | **兆候まで**。positive 8 件では確定不可 |
| 2 | 映像に CAN を加えると判断が改善するか | **C − B**（主）と C − A。同一 32 件の対応比較 | **比較的よく分かる** |
| 3 | 危険認識は時間とともにどう立ち上がるか | positive 8 件の `delta_onset` 個票、`risk_level(t)` の軌跡 | **定性的に十分** |
| 4 | 平常時に誤警報を出し続けないか | negative 24 クリップの clip/episode 単位指標 | **Phase 1 で最も確度が高い** |
| 5 | 規模・運転特化 post-training で判断がどう変わるか | 同一 32 件での 4 モデル対応比較 | **設計上いちばんきれい**。大きな差のみ検出 |

問い 5 について。**Qwen2.5-VL-7B と Cosmos-Reason1-7B は同一ベース・同一
パラメータ数**なので、両者の差は事後学習だけに帰着する。Qwen3-VL-8B が世代差、
Qwen3-VL-30B-A3B が規模差を担う。変数が 1 つずつ分離された比較になっている。

---

## 2. Phase 1 の位置づけ — 何を確定しないか

**探索的 PoC である。性能は確定しない。**

positive 8 件・negative 24 件では、点推定に意味のある精度が出ない。

| 指標 | 仮の結果 | 95% 信頼区間 |
|---|---|---|
| 再現率 (positive 8 件) | 6/8 = 75% | 約 **[41%, 93%]** |
| 誤警報 (negative 24 件) | 3/24 = 12.5% | 約 [4%, 31%] |
| Cohen の κ | 0.5 | 標準誤差 約 0.17 |

したがって指標は**必ず件数を併記**し（`6/8 (75.0%)`）、信頼区間は「参考」と
明記して結論の根拠にしない。

n=32 でも決着することは 4 つある。

1. **harness が動くか** — schema 適合率・パース失敗率・因果性は件数に依存しない
2. **決定的な失敗があるか** — 全件同一判定、`evidence` が常に `can` 等
3. **モデル間の相対比較** — 同一 32 件に対する対応のある比較
4. **失敗の質** — 32 件の説明文を読む。定量指標より情報量が多い

---

## 3. データ — 人手確認済みの 32 件のみ

`out/chunk1/labels.csv`（comma2k19 Chunk_1 の候補 71 件のうちラベル済み）。

| | 件数 |
|---|---|
| positive (`risky=True`) | **8** |
| negative (`risky=False`) | **24** |

**主評価ラベルは `risky` の bool。** `verdict` は 5 表記に割れているので
（リスク高 / リスク高い / リスクあり / ややリスクあり / 若干リスクあり）
Phase 1 では使わない。

### 3.1 「フィルタで落ちた = negative」とは扱わない

フィルタが取りこぼすものを拾えるかを試す層の評価に、フィルタの出力を正解として
持ち込めば、**フィルタの死角がそのまま評価の死角として埋め込まれる**。
negative として使うのは人手確認済みのものだけに限る。

Stage1 通過・閾値直下などの hard negative は Phase 2 で**人手ラベルを付けてから**
使う。

### 3.2 オンセット時刻（人手付与）

オンライン評価の基準。`out/chunk1/vlm/labels_onset.csv`。

| 列 | 内容 |
|---|---|
| `t_onset_seg_s` | 予兆が見て取れる最初の瞬間 |
| `t_apparent_seg_s` | 明らかに危険と分かる瞬間 |
| `onset_cue` | 根拠の類型 |

**基準は `t_onset_human` であって CAN 由来の `t_start` ではない。**
`t_start` は検出器が閾値を超えた時刻であって、人が危険を認識できる時刻ではない。

実測（8 件）:

```
onset_cue    cut_in 5 / crossing 1 / lead_brake 1 / other 1
```

**人手のオンセットが CAN の検出より先だったもの: 4/8 件**

| event | 先行 | cue |
|---|---:|---|
| P05 | **+3.16 秒** | cut_in |
| P04 | +0.87 秒 | other |
| P08 | +0.60 秒 | crossing |
| P07 | +0.51 秒 | cut_in |

**人間が CAN より先に気づける事例が実在する**ことが確認できた。これにより
「VLM が先行検知できるか」という問いに意味が生じる。逆に P02（−5.05 秒）
P03（−8.22 秒）は `short_thw` が持続条件なのでフィルタが先に鳴っている。

### 3.3 データ側で分かった制約

**(a) 隣接する 2 件は別事象らしい。** P02（seg10）と P03（seg11）は候補として
4.7 秒差で隣接するが、**人手オンセットは 22.0 秒離れている**。独立事象として
扱ってよいと見られる（`same_episode_as` で追跡）。

**(b) 手元にないセグメントへ評価区間がはみ出す。** Chunk_1 はドライブが
途中から始まる（例: `2018-08-17--14-55-39` は seg1 から）。

| event | 欠落 | 影響 |
|---|---|---|
| P06 | seg39 | 後 1.0 秒 |
| P08 | seg0 | **前 5.5 秒** |
| N23 | seg14 | 後 1.0 秒 |

黙って縮めず `pre_lost_s` / `post_lost_s` / `missing_segments` に記録する。

**(c) P08 は映像窓が満たない時刻がある。** 候補開始が seg1 の 0.60 秒で
seg0 が無いため、4 秒遡る映像窓が揃わない。**時刻ごと捨てると危険が
立ち上がるまさにその区間が消える**ので、短い窓のまま渡し
`n_frames` / `hist_s` / `partial_window` に記録する。

```
P08 の立ち上がり
  t_rel=0.00  frames=1  partial=True
  t_rel=1.50  frames=4  partial=True
  t_rel=3.50  frames=8  partial=False   ← 以降は満杯
```

部分窓は全 1,190 時刻中 **7 点のみ**（すべて P08）。他は例外なく 8 枚。

P08 のオンセット（seg1 0.00 秒）は評価区間の開始（0.10 秒）の 0.10 秒手前なので、
検出遅れには **+0.10 秒の床**が生じる。評価ストライド 0.5 秒より小さいので
実用上は測れる。`latency_floor_s` に記録。

---

## 4. 方法 — 2 つの評価モード

```
モード A: 一括判定 (clip)
    入力 = 候補区間全体の映像 + CAN + 区間の極値
    出力 = 判定 1 件
    用途 = 人手判定との直接比較

モード B: オンライン判定 (causal streaming)   ★主眼
    評価時刻 t を  t_start - 6 秒  〜  t_end + 2 秒  で 0.5 秒刻み
    各 t で  映像 = (t-4, t] の 8 フレーム   ※末尾がちょうど t
             CAN  = (t-6, t-guard] の 12 行
    未来の映像・CAN・候補区間の集約値は一切与えない
    用途 = 判断がいつ・どう変わるかの観察
```

候補長は 4.5〜29.2 秒とばらつく（中央 9.3 秒）ので、`t_peak` 中心の固定窓は
使えない。候補区間そのものを基準にする。**モード A が見る範囲はモード B の
評価区間と一致させる**（範囲の差とモードの差を混同しないため）。

### 4.1 stateless を既定とする

各時刻を独立した推論にする。理由は 3 つ。

- 前の判定を渡すと経路依存になり、「時刻 t のデータから何を読み取れるか」を
  測れなくなる。**誤りが伝播し、一度 hazard と言うと戻らなくなる**
- 独立なら完全に並列化でき、vLLM の continuous batching に載る
- 順序効果が無く再現性が保てる

`change_from_previous` は「直近フレーム間で何が新しいか」であって、
前回の出力は渡さない。因果性は保たれる。

### 4.2 条件

| 条件 | 入力 | 位置づけ |
|---|---|---|
| A | CAN のみ | 主比較 |
| B | 映像のみ | 主比較 |
| **C** | 映像 + CAN | **主比較（本命）** |
| D | C + `event_types` | **アブレーション。独立性能とは別表** |

報告する差分は **C − B（CAN 追加の効果）** と **C − A（映像追加の効果）** の両方。

D は既存フィルタの結果をヒントとして与える条件で、復唱の有無を見るためだけに
使う。性能比較の表には含めない。

プロンプトは条件ごとに文面を変える。条件 A に「映像を見て」と書いたり、
条件 B に空の信号表を出したりすると、**その条件だけが不当に不利になり
C−A / C−B の差を歪める**。

---

## 5. 因果性の担保

オンライン判定の前提「評価時刻より後の情報を一切与えない」は、
気をつけるだけでは守れない。機械的に検査する。

### 5.1 中心合わせ特徴量による未来漏れ

`features.py` の特徴量は**すべて中心合わせ**で計算されている。

```python
def derivative(x, dt):          # 中心差分
    out[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
def rolling(x, window, how):    # center=True
def moving_average(x, window):  # 対称カーネル
```

時刻 t の値が参照する未来の幅:

| 列 | 経路 | 未来漏れ |
|---|---|---|
| `v_mps` | MA 0.25 s | ±0.10 s |
| `steer_deg_s` | MA 0.15 s | ±0.05 s |
| `ax_mps2` | v の中心差分 → MA 0.25 s | **±0.25 s** |
| `ay_kin_mps2` | v×yaw → MA 0.25 s | ±0.20 s |
| `jerk_mps3` | 微分 2 回 | ±0.30 s |
| `*_win_*` | + rolling(center) 0.3 s | **±0.45 s** |

**対処は 2 方式**（`configs/vlm.yaml` の `context.mode`）。

| 方式 | 内容 | 状態 |
|---|---|---|
| **`guard`** | CAN を `t - guard_s` までに切る。既存の `compute_features` を変更しない | **既定・実装済み** |
| `causal` | 後ろ向きフィルタで再計算。遅れなし | 差し替え点のみ用意（未実装） |

`guard_s: 0.5`。評価ストライドが 0.5 秒なので遅れは 1 ステップ分。
**guard 幅は結果メタに必ず記録する**（検出遅れの分解能の下限を決めるため）。

### 5.2 候補区間の集約値を渡さない

`severity` / `event_types` / `grade` / `*_min` / `*_absmax` は候補区間全体を見て
算出された値であり、モード B に渡した瞬間にモデルは未来を知る。
`GuardContext.span()` と `.extremes()` は**モード A 専用**。

### 5.3 試験

`tests/test_vlm_causality.py`（24 件）。

1. 映像フレームの時刻が t を超えない（末尾がちょうど t）
2. CAN の参照元グリッド時刻が `t - guard_s` を超えない
3. 禁止列が文脈に現れない
4. **漏れカナリア** — t より後のグリッド行を乱数で破壊しても、組み上がる文脈が
   一字一句変わらないこと

4 が本命。1〜3 は書き忘れを拾うが、4 は経路全体を通した保証になる。

生成済みリクエスト **1,318 件の実物**に対しても同じ検査を行い、違反 0 件を確認
（`scripts/build_vlm_inputs.py` の出力）。

---

## 6. 映像の扱い

comma2k19 の映像は **1164×874 / 20 fps / 60 秒** のセグメント単位。

**`video.hevc` は生の Annex-B ストリームでタイムスタンプを持たない**
（`ffprobe` が幅も高さも返さない）。`setpts` で 20 Hz を打ち直さないと
フレーム番号と秒の対応が崩れる。`raw_data/hevc2mpeg.sh` が同じことをしている。

### 6.1 フレームキャッシュ

評価時刻ごとに ffmpeg を呼ぶと同じ動画を何十回もデコードする。
g7e.2xlarge は 8 vCPU しかないのでここが律速になる。
**セグメントごとに一度だけデコードして JPEG にする。**

命名はセグメント内の絶対フレーム番号（`f00374.jpg` = フレーム 374）。
`-start_number` で採番をずらすと 1 ずれても気づけないので、オフセットを
持ち込まない。

実測: **37 セグメント / 1.0 GB / 38 秒**（1 セグメント 1200 枚・約 20 MB）。

### 6.2 フレーム番号の検証

`f00374.jpg` と、同じ動画の 18.70 秒から取り出した画像の平均絶対差を、
前後 10 フレームと比較した。

```
f00370  2.252    f00373  1.249
f00371  2.195    f00374  0.283   ← 最小
f00372  1.596    f00375  1.365
```

明確な最小値があり、off-by-one が無いことを確認。

なお実測でセグメントが 1201 フレーム（60.05 秒）のものがある
（`2018-08-03--10-35-16/12`）。境界の丸め上がりを次セグメントへ送るためだけに
`frames_per_segment` を使うので、1 フレームの差は判定に影響しない。

---

## 7. 指標

### 7.1 モード A

```
条件 C  再現率 7/8 (87.5%) [参考 53%-98%]   誤警報 1/24 (4.2%)   κ +0.833 [参考 SE 0.114]

CAN 追加の効果 (C - B): 再現率 +4 件 / 誤警報 -4 件 / κ +0.666
映像追加の効果 (C - A): 再現率 +3 件 / 誤警報 -5 件 / κ +0.602
```

あわせて `evidence` の内訳、schema 違反件数、反復間の一致を出す。

**temperature=0 が既定なので、3 反復は再現性確認であって自己一致率ではない。**
自己一致率を主指標にするのは temperature>0 のアームだけ。

### 7.2 モード B / negative（24 クリップ・7.6 分・917 時刻）

**主指標は clip / episode 単位。** 同一クリップ内の時刻は強く相関しており、
917 時刻を独立標本として扱うと実効標本数を 30 倍以上に過大評価する。

| 指標 | 定義 |
|---|---|
| **全時間 normal を維持** | 全時刻が `normal`。**デバウンス無し**（strict clean） |
| **誤警報エピソード数** | `normal` 以外が **2 時刻以上連続**した極大区間の数 |
| 同・毎分 | エピソード総数 / 7.6 分 |
| **誤警報時間比** | クリップごとの（非 normal 時刻 / 全時刻）の**中央値と最大値** |
| 補助 | 時刻単位の誤警報率。CI はクリップ単位ブートストラップのみ |

分母の粗さを明記する。negative は 7.6 分しかないので**毎分エピソード数は
0.13 刻みでしかない**。生の件数を主表記とする。

時間比は平均でなく中央値と最大値。平均だと 1 件の暴走が埋もれる。

strict clean とエピソード基準の差は大きい。時刻あたり 4% のちらつきがあると:

```
全時間 normal を維持     :  6/24
エピソードが無いクリップ : 22/24
  うち単発のちらつきのみ : 16 件
```

### 7.3 モード B / positive（8 件）

**イベント別の個票を残す。** n=8 で平均を語らない。

```
event_id  onset_cue  detected  delta_onset_s  delta_apparent_s  latency_floor_s  filter_lag_s  n_partial
```

- `t_alarm` = `state ∈ {caution, hazard}` が 2 時刻連続した最初の時刻
- **`delta_onset_s = t_alarm − t_onset_human`（主指標）**。負なら人間より早い
- `latency_floor_s` — オンセットが評価区間の手前にある場合の下限
- `n_partial` — 部分窓だった時刻数（P08 のみ 7）

---

## 8. ゲート条件

**探索的な最低条件であって性能目標ではない。** 満たせば Phase 2 へ進む、
という判断にのみ使う。**数値を目標として最適化しない。**

| # | 条件 | 満たさない場合 |
|---|---|---|
| 1 | schema 適合率 > 95%、因果性試験全通過 | 実装の問題。修正して再実行 |
| 2 | 明白な退化がない（全件同一判定、`evidence` が常に `can` 等） | **打ち切り** |
| 3 | temperature=0 で 3 反復が一致 | 非決定性の原因を潰す |
| 4 | **誤警報エピソード無しが 12/24 以上** かつ 時間比中央値 < 20% | 常時警報型 |
| 5 | 映像由来の類型（`cut_in` / `crossing`）6 件のうち **2 件以上**で `t_alarm ≤ t_onset_human` または `t_apparent_human ± 1.0 秒`以内 | 先行/同時検知の兆候なし |
| 6 | 4 モデルのうち少なくとも 1 つが 1〜5 を満たす | **打ち切り** |

ゲート 4 は `n_episodes == 0` を基準にする。運用上、単発 0.5 秒のちらつきは
警報として成立しないため。strict clean と時間比は併記する。

**ゲート 4 単独では不十分。** 「常に normal」と答える退化モデルは誤警報ゲートを
難なく通る（スタブで実証済み: エピソード無し 24/24 で通過、ゲート 2 と 5 が捕捉）。

ゲート 5 は「兆候があるか」を見るもの。n=6 で厳密な合否は問えない。
満たさなくても、`delta_onset` の分布が一貫して正なのか散らばっているのかで
Phase 2 の設計判断は変わる。**個票を残す理由がここにある。**

---

## 9. モデル

実行環境は **g7e.2xlarge**（NVIDIA RTX PRO 6000 Blackwell **96 GB** /
8 vCPU / 64 GB RAM / 1,900 GB NVMe / $3.363・時）。

| モデル | 規模 | bf16 重み | 96 GB | 優先度 |
|---|---|---|---|---|
| **Qwen2.5-VL-7B-Instruct** | 8.3B | 約 17 GB | ◎ | ① 基準 |
| **Cosmos-Reason1-7B** | 7B | 約 15 GB | ◎ | ① 運転特化 post-training の効果 |
| **Qwen3-VL-8B** | 8B | 約 17 GB | ◎ | ① 世代差 |
| **Qwen3-VL-30B-A3B** | 30B (活性 3B) | 約 60 GB | ○ | ② 規模の効果 |
| Qwen2.5-VL-32B / 72B-AWQ | 33B / 72B | 66 / 40 GB | ○ | ③ 予備 |

**律速はモデルではなく入力の作り方。** ①の 3 モデルで harness を確定させ、
失敗事例の質を見てから②へ進む。3 モデルとも同じ失敗をするなら規模を上げても
直らない。

**Alpamayo 系は別枠。** カメラ較正・自車軌跡など入力要件が異なり、同一入力での
比較が成立しない。別 adapter・別比較表とし、①②と同じ表に並べない。

### 9.1 何を共通化するか

| 共通化する（モデル間で不変） | adapter が吸収する |
|---|---|
| 入力生成、prompt、JSON schema、復号パラメータ、評価コード | chat template、メディアの渡し方、視覚トークン化 |

**視覚トークン数ではなくピクセル予算を固定する。** パッチサイズもマージ率も
モデルごとに違うので、トークン数を揃えると入力画像そのものが変わる。
同じ画像を渡し、生じたトークン数をモデルごとに記録・併記する。

フレームは**画像の列ではなく動画として渡す**。Qwen 系は時間方向のマージと
絶対時刻の埋め込みを持っており、画像を並べるだけではそれが効かず、
トークン数もおよそ倍になる。

---

## 10. 実装

```
src/near_miss/vlm/
    schema.py     出力 JSON Schema と検証 (新規依存なし)
    windows.py    評価時刻、boot time <-> セグメント内秒・フレーム、切り詰め
    context.py    因果 CAN 文脈 (GuardContext / CausalContext は差し替え点)
    frames.py     フレームキャッシュ
    prompt.py     条件対応のプロンプト組み立て
    adapters.py   モデル差の吸収 (QwenVLAdapter / EchoAdapter)
    runner.py     vLLM 実行。**唯一 vllm に依存する**
    scoring.py    指標の素 (Wilson CI / κ / クリップ単位ブートストラップ)
configs/vlm.yaml
configs/prompts/vlm_v1_clip.md, vlm_v1_online.md
scripts/make_onset_template.py   段階 0: onset 付与の様式
scripts/import_onset_sheet.py    段階 0: 記入の取り込みと検証
scripts/build_vlm_inputs.py      段階 1: フレーム展開 + リクエスト生成
scripts/run_vlm_review.py        段階 2: 推論
scripts/score_vlm_review.py      段階 3: 採点
tests/test_vlm_causality.py      因果性の試験 24 件
```

`runner.py` 以外は `vllm` を import しない。**前処理・採点・試験は Mac で動く。**

`vllm` は `pyproject.toml` で管理しない。torch の CUDA ビルドを引き連れており、
**GPU 世代（Blackwell）と driver に合うものをその機械の上で選ぶ**必要がある。
Mac から解決した lock を持ち込むと合わないビルドに固定される。

### 10.1 手順

```bash
# 段階 0 (Mac): onset 付与の様式を作る
uv run python scripts/make_onset_template.py
sh out/chunk1/vlm/convert_segments.sh      # 該当セグメントを mp4 に変換
#   -> out/chunk1/vlm/labels_onset.md を人手で記入
uv run python scripts/import_onset_sheet.py

# 段階 1 (Mac / EC2): 入力を作る
uv run python scripts/build_vlm_inputs.py            # Mac
uv run python scripts/build_vlm_inputs.py --hwaccel  # EC2 (NVDEC で CPU を空ける)

# 経路の確認 (GPU 不要)
uv run python scripts/run_vlm_review.py --model echo --mode a --limit 4

# 段階 2 (EC2)
uv pip install vllm          # その機械で解決する
uv run python scripts/run_vlm_review.py --model qwen2_5_vl_7b   --mode a
uv run python scripts/run_vlm_review.py --model qwen2_5_vl_7b   --mode b
uv run python scripts/run_vlm_review.py --model cosmos_reason1_7b --mode b
uv run python scripts/run_vlm_review.py --model qwen3_vl_8b     --mode b
uv run python scripts/run_vlm_review.py --model qwen3_vl_30b_a3b --mode b

# 段階 3 (Mac)
uv run python scripts/score_vlm_review.py --results out/chunk1/vlm/results_*.jsonl
```

途中で落ちても、書き終えた `request_id` は飛ばして続きから流せる。

### 10.2 規模

```
エピソード 32 件 (positive 8 / negative 24)。すべて評価可能
モード A   128 件 (32 エピソード × 4 条件) × 3 反復 = 384
モード B 1,190 件 (評価時刻ごと)          × 1 反復 = 1,190
計 1,574 件 / モデル      4 モデルで 6,296 件
```

GPU 1〜2 時間 / $4〜7 程度の見込み（**未実測**）。

---

## 11. Phase 2（Phase 1 のゲートを満たした場合のみ）

1. 未ラベル 39 件に人手で `risky` を付ける（positive が増える可能性）
2. hard negative を人手ラベルする
   - Stage1 通過・Final 非採用（横滑りフィルタは棄却理由を記録している）
   - 閾値直下（`detection.yaml` を 0.8 倍した複製との差集合）
   - ランダム通常走行
3. **人手確認済みのものだけを negative として評価に使う**

Phase 1 の失敗様式が分かってから、どの区間を優先的にラベルするかを決める。

---

## 12. わかっていないこと

* **実機で未検証。** GPU が無いため、次は確かめられていない。
  - `vllm.LLM` / `SamplingParams` / `GuidedDecodingParams` の API 互換（版差がある）
  - `guided_json` が実際に効いているか（`schema_errors` が 0 でなければ効いていない）
  - `AutoProcessor.apply_chat_template` が `type: "video"` を受けるか
    （受けなければ `media_kind: images` へ切り替え）
  - VRAM と `max_model_len` の実際の必要量
* **`guard` 方式の妥当性。** CAN が 0.5 秒古いことが判定に効くかは実測でしか
  分からない。`guard_s` を 0.5 / 1.0 で振って感度を見る必要がある。
* **positive が実質 7〜8 件。** P02/P03 が同一事象なら 7 件。映像にしか根拠が
  無い類型（cut_in 5 + crossing 1）は 6 件で、**問い 2 と 3 を検証できるのは
  この 6 件だけ**。
* **説明文の質は自動では測れない。** 幻覚（映像に無い対象への言及）の計数は
  人手による。Phase 1 では 32 件 × 最良モデル 1 つを盲検で採点する想定。
* **人手側の天井が未測定。** 既存 32 件のうち 10 件を間隔を空けて再判定し、
  人間同士（あるいは同一人物の再現）の κ を出さないと、VLM の κ を評価する
  基準が無い。
* **`causal` 文脈が未実装。** guard の遅れを外す差し替え先として型だけ用意して
  ある。
