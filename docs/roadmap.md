# ロードマップ

C&A (FY26 2H) の目標のうち、このリポジトリが受け持つテーマの状況をまとめる。目標の本文は `shared/C&A_FY26_2H.txt`。
中身は書かず、状況と参照先だけを置く。詳しい結果は参照先の文書と `experiments/index.csv` を見る。

- 更新: 週次報告 (`reports/weekly/`) を書くときに、状況の列と「最終更新」を直す
- 状況の値: 未着手 / 調査中 / 進行中 / 結論あり (区切り) / 保留
- マイルストーンの期限は、決まっていないものは「未設定」と書く (推測で日付を入れない)

最終更新: 2026-11-13 (2026-W46)

## 1. C&A の目標との対応

| C&A の項目 | 内容 (目標の本文から要約) | このリポジトリのテーマ |
|---|---|---|
| 1-(1) ヒヤリハットシーン抽出方式の構築・検証 | CAN 信号・物理モデルによる候補の絞り込みと、VLM による危険性判定を組み合わせた抽出方式を構築し、有効性を検証する | A・B・C |
| 1-(2) 物理モデルを活用したロングテールデータの拡充 | スリップで作った物理モデルの使い方を、段差・くぼみの異常振動、突風による走行不安定、ブレーキフェードなどに広げる | D・E・F・G |
| 2 生成 AI 活用による開発・業務生産性向上 | AI を使った開発手法の実践・整理、効果と課題、部内展開 | H (`reports/ca/`) |
| 3 フィジカル AI への物理学的アプローチ | 先行研究調査、物理モデルと AI の検証、対外発信 | このリポジトリの対象外 |

## 2. テーマごとの状況

| # | テーマ | C&A | 状況 | 直近の状況 (日付) | 次のマイルストーン (期限) | 参照先 |
|---|---|---|---|---|---|---|
| A | CAN による候補抽出 (L1〜L3・優先度) | 1-(1) | 進行中 | comma2k19 と commaCarSegments 4 車種 (各 2,000 セグメント) を L3 まで処理済み。統一形式の入力になっている (2026-09-17) | 抽出結果の検証の記録をそろえる (未設定) | `README.md`、`docs/pipeline_overview.md`、`experiments/index.csv` の `out/l1_l3/` |
| B | VLM による危険性判定 | 1-(1) | 調査中 | PoC Phase 1 の結果あり (`shared/digest.txt`、2026-09-03)。路面の VLM 判定の設定とスクリプトを追加 (コミット af0c3a9) | 未設定 | `docs/vlm_review_poc.md`、`docs/vlm_road_review.md` |
| C | 統一形式・API 連携 | 1-(1) | 進行中 | 仕様・Schema・サンプルを API 側へ提供 (2026-10-28、`shared/provided_to_api/`) | 未設定 | `docs/near_miss_canonical_schema.md`、`docs/near_miss_api_data_spec.md` |
| D | スリップ | 1-(2) (2Q からの継続) | 結論あり | 等級 (multi / high / activation / physics) を改訂した v3 が統一形式の入力 (2026-09-16) | 未設定 | `docs/sideslip_filter.md` |
| E | 段差・くぼみによる異常振動 | 1-(2) | 進行中 | 振動イベント 1,037 件、地点 498。物理整合性の評価と報告用の図 (2026-10-27〜11-09) | 未設定 | `docs/vibration_near_miss_candidates.md`、`docs/vibration_physics_validation.md`、`docs/vibration_poc_final_summary.md` |
| F | 突風・横風による走行不安定 | 1-(2) | 進行中 | ZOD を採用し、全 177 走行でヨー残差の候補 14 件を気象観測 (DWD・SMHI) と照合。「支持」3 件 (うち 2 件は同じ連続事象)・反証 6・不明 5。横風が原因と確定した事象は無い。前方映像で 6 件を確認 (2026-11-12〜13) | 他の要因 (路面の勾配・操舵のゼロ点・周辺車両) との識別 (未設定) | `docs/zod_full_screening.md`、`docs/zod_full_weather.md`、`out/zod/front_video_review/review.md`、`reports/weekly/2026-W46/crosswind_outline.yaml` |
| G | ブレーキフェード | 1-(2) | 未着手 | 説明用の図だけがある (`shared/fade_illust.jpeg`)。解析はリポジトリに無い | 未設定 | — |
| H | AI 駆動開発の事例・ノウハウ | 2 | 進行中 | 事例の記録を始めた (2026-11-11) | 未設定 | `reports/ca/` |
| — | 実験・報告の管理 | (共通) | 進行中 | 構成文書、実験索引、週次報告の仕組みを整備し、main に統合 (2026-11-11)。週次スライドの作り方を skill にした | 週次スライドの skill を使いながら直す (未設定) | `docs/project_structure.md`、`experiments/README.md`、`reports/README.md`、`.claude/skills/weekly-slides/SKILL.md` |

## 3. 保留・判断待ち

| 項目 | 状態 | 参照先 |
|---|---|---|
| 構成の整理の提案 4・6・7 (`raw_data/` の作業ファイル、路面振動の設定の場所、`shared/` のバイナリ) | 保留 | `docs/project_structure.md` §5 |
| BDD100K の利用 (非商用ライセンス・登録が必要) | 利用可否の判断待ち | `claude-progress.md` |
| 実験の正式な採否 | 記録が 0 件。決め方が未定 | `experiments/README.md` |
