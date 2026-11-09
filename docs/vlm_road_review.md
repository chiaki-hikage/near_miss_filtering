# 振動イベントの路面凹凸を VLM で確認する

振動イベント (前輪の当たり時刻) の地点に、凹凸 (穴・マンホール・補修跡・継ぎ目・段差など) が
前方カメラの映像に写っているかを、ローカル VLM に判定させる。

これまでの映像の確認は、Claude が静止画 2 枚を見て行った暫定の判定だけだった
(`out/vibration/locate/review_video.csv`)。この仕組みで、同じ確認を全イベントに対して、
決まった手順で、複数のモデルで行えるようにする。

**既存のヒヤリハット判定 (`docs/vlm_review_poc.md`) のコード・設定・結果には手を入れていない。**
モデルの一覧と復号の設定 (`configs/vlm.yaml`)、adapter・Runner・計測は import して使うだけで、
このタスクのファイルはすべて新しく足したもの。

## 1. 設計

| 項目 | 内容 |
|---|---|
| 目標 | 振動イベント 1,037 件 (`near_miss_vibration_events.csv`) と、対照 300 件 |
| 対照 | 振動イベントのあるセグメントの中で、同じドライブのどのイベントからも 5 秒以上離れた時刻 (車速 5 m/s 以上) |
| 画像 | 目標地点がカメラから 20・15・11・8・5 m 先に見えるフレームを 1 枚ずつ (計 5 枚) |
| 切り出し | 元画像 1164×874 の列 250〜1100・行 420〜720 (以前の目視と同じ範囲)。縮小しない |
| 条件 R | 印なし。注目する地点までの距離だけを文で伝える |
| 条件 O | 左右の車輪の通り道 (青・赤の破線) と推定位置 (○) を描き込む |
| 出力 | `feature_present` (yes/no/uncertain)・`feature_type`・`wheel_path`・`near_target`・`visibility` など (`src/near_miss/vlm/road.py`) |

### 1.1 モデルに伝えないもの (盲検化)

モデルには、**その地点で振動イベントが起きたかどうかを伝えない**。
振動の強さ・候補かどうか・物理シグナル・以前の目視判定も入れない。

- 対照の地点にも、同じ文面・同じ形の入力を渡す
- 目標の ID は `T0001…` の通し番号で、イベントと対照を混ぜて乱数で並べてある
- 採点に使う情報は `manifest.csv` にだけ書き、リクエストには入れない (`tests/test_vlm_road.py` で検査)
- プロンプトに「振動」「イベント」「候補」などの語が入らないことも試験で確かめている

理由: ヒヤリハット判定の最初の結果 (`docs/vlm_review_poc.md` 11.2) では、検出率と誤警報率が
連動しており、モデルが識別ではなく「鳴りやすさ」で並んでいた。「凹凸がある前提」で探させると
同じことが起きる。**対照でも凹凸ありと答える割合を測り、イベントと比べる (lift)** ことで、
識別できているかを確かめる。

### 1.2 フレームの選び方

前輪が目標地点を通る時刻を t、車速を v、カメラが前輪より 0.6 m 後ろにあるとすると、
カメラから目標地点までの距離が d になる時刻は t − (d − 0.6) / v。
その時刻に最も近いフレームを `global_pose/frame_times` (各フレームの boot time) から選ぶ。
ずれが 0.05 秒を超えるフレームは使わない。

印の位置は `locate_candidates.plot_frames` と同じ式で描く
(平坦・直進・カメラは車両中心の仮定。地平線の行と K は地点 19 での当てはめ結果)。

### 1.3 画像の渡し方

フレームの間隔は車速で変わるので、動画 (一定の fps) ではなく**画像の列**として渡す。
各画像の直前に「画像 k (注目する地点まで約 d m)」という見出しを挟む。
既存の `QwenVLAdapter.chat_text` は画像の枠を 1 つしか置かないので、
枚数分の枠を置く `RoadQwenAdapter` を `road_runner.py` に足した (既存の adapter は変えていない)。

## 2. 手順

### 2.1 入力を作る (Mac か EC2。GPU は要らない)

```bash
uv run python scripts/vlm/build_road_inputs.py --dry-run     # 件数とセグメント数を見るだけ
uv run python scripts/vlm/build_road_inputs.py               # 全振動イベント + 対照 300
uv run python scripts/vlm/build_road_inputs.py --targets candidates --control-n 40   # 候補 21 件だけで試す
```

出力は `out/vibration/vlm_road/`:

| ファイル | 内容 |
|---|---|
| `requests_mode_road.jsonl` | リクエスト (目標 × 条件 R・O) |
| `manifest.csv` | 採点用。イベントか対照か・候補か・地点・以前の目視判定・輪速の左右差・昼夜。**モデルには渡さない** |
| `frames/` | 切り出した画像 (印なし) |
| `overlay/` | 印を描いた画像 |
| `needed_segments.txt` | デコードに使ったセグメント |
| `build_meta.json` | 件数・設定のハッシュ |

全件の dry-run では、デコードするのは 275 セグメント・6,681 枚、
一部の距離のフレームが欠ける目標が 2 件だった。

**Mac で入力を作って EC2 に運ぶのが楽。** 画像は 1 枚 30 KB 前後なので、
`out/vibration/vlm_road/` は全件でも 0.4 GB 程度で済む (候補 21 件・対照 4 件の試作での実測から見積もり)。
EC2 で作る場合は、`needed_segments.txt` のセグメントの `video.hevc`・`global_pose/`・
`processed_log/IMU/accelerometer/`・`processed_log/CAN/speed/` が要る (1 セグメント約 44 MB、計 12 GB 程度)。
`--hwaccel` で NVDEC を使う。

### 2.2 推論 (EC2)

既存の VLM 評価と同じ機械 (g7e.2xlarge、RTX PRO 6000 96 GB) を想定。vllm はその機械で入れる。

```bash
uv pip install vllm                                                  # その機械で解決する
uv run python scripts/vlm/run_road_review.py --model echo --limit 8  # 経路の確認
uv run python scripts/vlm/run_road_review.py --model qwen2_5_vl_7b
uv run python scripts/vlm/run_road_review.py --model cosmos_reason1_7b
uv run python scripts/vlm/run_road_review.py --model qwen3_vl_8b
uv run python scripts/vlm/run_road_review.py --model qwen3_vl_30b_a3b
```

- モデルのキーは `configs/vlm.yaml` の `models` と同じ
- 途中で落ちても、書き終えた分は飛ばして続きから流せる
- 時間と資源は既定で計測し、`perf_<model>_mode_road.json` に書く (既存の計測と同じ形式)
- 条件を絞るときは `--conditions O`

規模の目安: 目標 1,337 件 × 2 条件 = 2,674 件 / モデル。
1 件は画像 5 枚 (850×300) と指示文。処理時間は**未実測**。

### 2.3 採点 (Mac か EC2)

```bash
uv run python scripts/vlm/score_road_review.py
```

モデル × 条件ごとに、件数を必ず併記して次を出す。

| 出力 | 見るところ |
|---|---|
| 区分 (対照 / 振動イベント / 異常振動候補) × 昼夜の「凹凸あり」の割合 | 主に昼間で比べる。夜間は路面が見えないことが多い |
| lift (昼間) = イベントの割合 / 対照の割合 | 1 に近ければ、地点によらず同じ割合で「あり」と言っているだけ |
| 以前の目視判定 (一致 / 候補 / 不明瞭 / 判別不能) との対応 | 以前の判定は Claude の暫定のもので、正解ではない |
| 物理シグナルの数ごとの割合 | `physics_vs_video_sites.csv` の 31 地点に入るイベントだけ |
| 左右の一致 | VLM が片側と答え、輪速の左右差が 2 倍以上のもの |
| 条件 R と O の対応 | 印があると「あり」が増えるか (印に引きずられていないか) |

目標ごとの表は `scored_road.csv`。「凹凸あり」は `feature_present = yes` かつ `near_target ≠ no`。

## 3. 注意

- **正解ラベルは無い。** 以前の目視判定は Claude による静止画 2 枚の暫定判定で、地点単位。
  対照で「あり」と答える割合が、いちばん信頼できる比較の基準になる
- **対照は振動イベントのあるセグメントから選んでいる。** 映像の追加取得が要らない代わりに、
  凹凸の多い道路に偏っている可能性がある。そのぶん、対照の「あり」の割合は高めに出うる
- 印の位置は平坦・直進・カメラ位置の仮定に基づく。進行方向に 1〜2 m ずれることがあると、プロンプトにも書いてある
- 昼夜は `event_utc` の現地時刻 (UTC−7) の 7〜19 時を昼とした。対照の UTC は、同じドライブの
  イベントから boot time との差を取って求めている (秒単位)
- 条件 O は、モデルに注目する位置を教えるぶん有利だが、印そのものを凹凸と取り違えるおそれもある。
  R と O を並べて比べる

## 4. ファイル

```
configs/vlm_road.yaml                 タスクの設定 (base_config: configs/vlm.yaml)
configs/prompts/vlm_road_v1.md        指示文
src/near_miss/vlm/road.py             schema・検証・幾何・プロンプト・echo (vllm 不要)
src/near_miss/vlm/road_runner.py      RoadQwenAdapter・RoadRunner (既存の adapter・Runner を継承)
scripts/vlm/build_road_inputs.py      手順 1: 入力を作る
scripts/vlm/run_road_review.py        手順 2: 推論
scripts/vlm/score_road_review.py      手順 3: 採点
tests/test_vlm_road.py                試験 16 件
```
