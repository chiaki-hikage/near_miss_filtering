# 作業の引き継ぎ (claude-progress.md)

セッションをまたいで作業を続けるためのメモ。**セッションの終わりに更新する**。長期の状況は `docs/roadmap.md`、報告は `reports/`。
ここには「次に開いたときにすぐ要ること」だけを書き、終わった項目は消す (履歴は git に残る)。

最終更新: 2026-11-18

## 今の状態

- ブランチ: `main` (2026-11-11 に `chore/project-structure` を早送りで統合。リモートには push していない)
- 作業ツリー: このファイルの更新時点で未コミットの変更なし (コミット後に確認すること)
- 試験: `uv run --extra dev pytest` は 466 件が通り、`test_weekly_report.py::test_build_follows_outline` の 1 件が失敗 (2026-11-18 時点。ブレーキの作業とは関係しない、未調査)

## 進行中

| 項目 | 状態 | 次の一手 | 参照 |
|---|---|---|---|
| 横風の検証に使う公開データセット | 候補を調査済み (BDD100K、ZOD、MAN TruckScenes など)。調査結果は文書にしていない | ユーザーの判断を待つ (下の「判断待ち」) | `docs/roadmap.md` テーマ F |
| 実験索引の確認 | 確認優先度 A 5・B 9・C 27。正式な採否は全件「未判断」 | A から確かめる。採否の決め方をユーザーと決める | `experiments/README.md` |
| 異常振動の追加データ (TartanDrive 2.0) | 人のレビューの準備まで完了。2026-11-18 に racepak の throttle と brake の名前と中身が逆とわかり、設定 `reaction.controls_cols` で直して作り直した。現行は 20261118_r6 (特徴づけ)・r7 (対象 42 件)・r8 (閲覧ページ)・r9 (長め 5 件)・r11 (V18 の車体のイラスト)。**未コミット** | ユーザーがレビューして `review_result.csv` を書き出す → `td2_review_evaluate.py --review .../20261118_r7`。5 件の車体のイラストの動画は 20261118_r13 (現行) | `docs/offroad_tartandrive2_review.md` |
| ブレーキフェード (制動性能の評価の PoC 段階 1) | 2026-11-18 完了。RAV4 の 0x224 byte4〜5 を `BRAKE_LEVEL` (チャネル `brake_level_raw`、単位不明) として車種設定に追加し、`tests/test_brake_decode.py` を足した。global_pose の勾配で補正した正常制動の模型で、学習に使っていない走行の R² 0.93、利きの比のばらつき 11% (RAV4)。熱負荷の蓄積はデータに無い。現行の結果は `out/brake_performance/20261118_r8〜r11`。**未コミット** | ユーザーの判断待ち (熱負荷の高いデータの入手、GR Cup の許可など) | `docs/brake_performance_rav4.md`、`docs/brake_fade_datasets.md` |
| ブレーキフェード (検出と検証の実現可能性) | 2026-11-18。NCST (CC0、17 MB) を取得 (`data/brake/ncst_hdv_brake/`)。圧力×車速の熱の模型が、台上で当てはめて路上の温度を 11〜21 °C の誤差で追えた。車速の減りだけの熱の指標は重量車で成り立たない。模擬の低下で、RAV4 の 1 回の制動の検出力 80% に約 43% の低下が要る (前回の「18%」は閾値の位置で、検出力は 23%。訂正済み)。公開データに熱負荷の高い実測+ブレーキ圧+温度のものは無い。結果は `out/brake_fade/20261118_r3〜r5`。**未コミット** | ユーザーの判断待ち: GR Cup の問い合わせ、自前の計測、CSU の J1939 (320 MB) の確認 | `docs/brake_fade_feasibility.md` |
| ブレーキフェード (追加の公開データの調査) | 2026-11-19。PDR・モータースポーツのロガー・大型車・試験データを調べた。映像付きで取得できる第 1 候補は Udacity CH2_002 (MIT、4.4 GB、トレント。峠越えの HMB_2)。PDR は元の MP4 の公開がなく、MIT のリポジトリにある C8 のサーキットの CSV (31 MB、Brake Pos % 連続値、映像なし) が熱負荷の最も高い候補。何もダウンロードしていない。**未コミット** | ユーザーの判断待ち: CH2_002 4.4 GB と rosbags の一時利用、PDR の CSV の利用条件 | `docs/brake_fade_datasets_round2.md` |
| ブレーキフェード (Udacity CH2_002 の峠の下り) | 2026-11-19。公式 .torrent を aria2 (DHT/PEX/LPD 切、完了後すぐ停止、アップロード 0 B) で取得、MD5 一致。`data/udacity_ch2/` (git 管理外)。HMB_2 に制動・IMU・高度・前方映像がそろう。峠の下り 201 m でブレーキに入ったのは位置エネルギーの 18%、E300 の最大 352 J/kg (100 km/h 停止 0.9 回分、comma2k19 RAV4 最大の 0.4 倍)。利きは熱とともに下がらない (フェードの判定はしない)。代表の動画 `out/brake_fade/20261119_r18_udacity_hmb2_scene/scene.mp4`。rosbags は `uv run --with` の一時利用。`tests/test_udacity_hmb.py`。**未コミット** | ユーザーの判断待ち: 熱負荷の高い映像付きデータは公開では未入手 (PDR の CSV の許諾、自前の計測) | `docs/udacity_hmb_brake.md` |
| ブレーキフェード (NCST の坂を地図で推定) | 2026-11-19。HEM の GPS が 136.85 分で切れた後を I-15 南行き (Cajon Pass) と推定。OpenStreetMap (Overpass、ODbL) の道と USGS 3DEP (EPQS、公有) の標高 766 点を `data/brake/ncst_route/` に取得 (取得の記録あり。OSRM の公開サーバーは非商用限りなので不使用)。峠 1,298 m から 2 本の下り (−4.6%・−4.1%、計 651 m) が、エンジンの信号で判定した下りの局面 2 本と合う。下り 2 の後半は車速の記録と対応が合わない (位置の不確かさを文書に記載)。IMU・舵角は無く、ハンドル操作は推定できない。現行 `out/brake_fade/20261119_r22_ncst_route_scene/`。**未コミット** | なし | `docs/brake_fade_feasibility.md` 2.6 節 |
| 週次報告 (ブレーキフェード、2026-W47) | 2026-11-19。4 枚の PPT を作った (`reports/weekly/2026-W47/brake_fade_2026-W47.pptx`、git 管理外)。構成案 `brake_fade_outline.yaml`、設定 `brake_fade_report_config.yaml` (min_slides 4)、図 `brake_fade_fig/` (`scripts/report/brake_fade_slides_figs.py`)。Quick Look で見た目を確認済み。**未コミット** | なし | `reports/weekly/2026-W47/` |
| ブレーキフェード (安全上の影響の評価) | 2026-11-19。台上の高温の利きの差は調整前 −22% → 車速・圧力で調整 −9% (90% 区間 −18〜+2%)、50〜130 ℃ 基準で −11%。調整前の約半分は車速で説明できる。停止距離は仮定のシナリオで 100 km/h から +10 m (δ 9%)・+24 m (δ 20%)。路上の 159 ℃ 付近は減速度を出せず (車速の記録 39 回中 4 回)、性能低下の根拠も反証も無い。FMVSS 121 は熱電対の測り方は近いが温度閾値は無い。現行 `out/brake_fade/20261119_r24_ncst_fade_safety/`。**未コミット** | 週次スライドへの反映 (未依頼) | `docs/brake_fade_safety.md` |
| ブレーキフェード (フェードリスク指標) | 2026-11-19。熱の状態 (NCST は実測温度、乗用車は直前 300 s のブレーキの仕事)・連続制動・下り坂・効率低下の兆候の 4 成分で 0〜100 と High/Medium/Low。High は NCST の 2 本の下り (145.8〜158 分、最大 79) だけ。Udacity は最大 41、comma2k19 は最大 58 (Medium)。確率ではなく、0 と 1 の点の多くは暫定。現行 `out/brake_fade/20261119_r31_fade_risk_index/`。**未コミット** | 週次スライドへの反映 (未依頼) | `docs/brake_fade_risk_index.md` |
| ブレーキフェード (リスク指標の動画・スライドへの反映) | 2026-11-19。動画は NCST `20261119_r34_ncst_route_scene_risk`、Udacity `20261119_r32_udacity_hmb2_scene_risk` (旧版 r22・r18 は残した)。週次スライド 3・4 枚目の図にスコアの段、注釈に 1 行ずつ。PPT を作り直し Quick Look で確認。**未コミット** | なし | `docs/brake_fade_risk_index.md` 4 章 |
| ブレーキフェード (Udacity の短い動画) | 2026-11-24。`20261124_r35_udacity_hmb2_short_descent` (443〜463 s、20 s、等速、勾配のイラストなし)。`udacity_hmb_heat.py` に `--window` `--speedup` `--no-slope` を足した (既定の動作は変えていない)。**未コミット** | なし | `out/brake_fade/README.md` |
| TartanDrive 2.0 の週次スライド (2 枚) | 2026-11-24。`reports/weekly/2026-W48/td2_2026-W48.pptx` (構成案 `td2_outline.yaml`、図は `scripts/report/td2_slides_figs.py`、新しい取得なし)。Quick Look で確認。**未コミット** | なし | `reports/weekly/2026-W48/` |
| 週次報告 | 2026-W46 の原稿 (`notes.md`) と変更点 (`changes.md`) はある。構成案 (`outline.yaml`) は未作成。`reports/weekly/2026-W46/weekly_2026-W46.pptx` は旧い作り (原稿の節から 5 枚) で作ったもので、今の決まりに合っていない | **ユーザーが時期を指定するまで PPT は作らない。** 頼まれたら skill `weekly-slides` の手順で構成案を示し、合意を得てから作る | `reports/weekly/2026-W46/`、`.claude/skills/weekly-slides/SKILL.md` |

## 判断待ち (ユーザー)

- ブレーキフェード (2026-11-19): Udacity CH2_002 は取得・解析済み (熱負荷は低い)。PDR の CSV は MIT がデータに及ぶか確認できず、作者への問い合わせ (ユーザーが行う) を待つ。展開した bag (約 6.2 GB、`data/udacity_ch2/extracted/`) を残すか消すか
- ブレーキフェード: 実証のデータの確保の方法 (GR Cup の許可の問い合わせ・自前の計測・論文の著者への問い合わせ・CSU の J1939 320 MB の取得) を選ぶNCST (17 MB) を取得してよいか。GR Cup の利用の許可を TRD に問い合わせるか (問い合わせはユーザーが行う)
- TartanDrive 2.0: 上位 5 件を 10 Hz・長めにする動画の範囲 (−8〜+4 s で約 540 MB / −6〜+3 s で約 380 MB) と 5 件の組み合わせ (V32 は IMU の抜けの近く、V01 は旋回中で振動小)。PVS は CC BY-NC-ND の確認まで保留。代わりの候補 A2D2 (CC BY-ND 4.0) を調べるか
- BDD100K の利用: 非商用ライセンスの範囲に入るか、3.9 GB を取得してよいか。公式サイトは https の証明書が期限切れで、ログインが要る。**AI からは取得しない** (ユーザーがログインして得たリンクを渡す)
- 構成の整理の提案 4・6・7: 保留 (`docs/project_structure.md` §5)
- `docs/crosswind_wind_data.md` の付録のコードをスクリプトにするか

## 守ること (このプロジェクトで決めたこと)

- 既存の結果・CSV・コードは、合意なしに変えない。追加の解析は別のフォルダに出す
- データの取得の前に、公式の配布元と利用条件を確かめる。無理にアクセスしない
- 信号の意味や単位を推測で決めない。推定は「推定:」と書く
- ファイルを動かすときは、写しと sha256 → 移動 → 一致の確認 → 全試験 (`docs/project_structure.md` §3 の 19・20)
- コミットはユーザーが頼んだときだけ。目的ごとに分ける
- 週次スライド: 作る時期はユーザーが指定する。構成案 (5〜7 枚、タイトル・主文 1 つ・図か表 + 注釈、1 スライド 1 メッセージ) に合意してから作る

## セッションの終わりに

1. `git status` で、意図しない変更が無いか確かめる
2. このファイルの「今の状態」「進行中」「判断待ち」を直す
3. `out/` に結果を足したら `uv run python scripts/report/build_experiment_index.py` を流す
