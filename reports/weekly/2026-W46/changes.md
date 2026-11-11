# 2026-W46 前回報告からの変更点 (自動抽出)

`scripts/report/weekly_changes.py` が作成。比較: `e50c8b9..3eedfa4`

## コミット (8 件)

- `af0c3a9` 2026-11-09 add analysis of crosswind_civic
- `94c1e24` 2026-11-09 add figures
- `1f2b926` 2026-11-11 横風スクリプトを scripts/crosswind/ へ移し、実験索引を追加
- `99e116c` 2026-11-11 実験索引の利用状況・文書上の区分・検証・採否を別の列に分ける
- `82a06ec` 2026-11-11 Civic の選定スクリプトを scripts/crosswind/civic_selection/ へ移す
- `bb59e83` 2026-11-11 横風の検討に使った風のデータと突き合わせの処理を記録する
- `857dd03` 2026-11-11 進捗管理と週次報告の仕組みを追加する
- `3eedfa4` 2026-11-11 CLAUDE.md に進捗管理・報告の参照先とセッションの手順を追記する

## 変更したファイル (区分ごと)

| 区分 | 件数 | 追加 | 変更 | 削除 |
|---|---|---|---|---|
| その他 | 5 | 1 | 4 | 0 |
| 設定 | 4 | 4 | 0 | 0 |
| 文書 | 5 | 5 | 0 | 0 |
| 実験索引 | 2 | 2 | 0 | 0 |
| 報告 | 5 | 5 | 0 | 0 |
| スクリプト | 49 | 49 | 0 | 0 |
| 共有資料 | 6 | 6 | 0 | 0 |
| コード (src) | 3 | 3 | 0 | 0 |
| 試験 | 5 | 5 | 0 | 0 |

<details><summary>その他</summary>

- M `.gitignore`
- M `CLAUDE.md`
- A `claude-progress.md`
- M `pyproject.toml`
- M `uv.lock`

</details>

<details><summary>設定</summary>

- A `configs/crosswind/honda_civic.yaml`
- A `configs/prompts/vlm_road_v1.md`
- A `configs/report_weekly.yaml`
- A `configs/vlm_road.yaml`

</details>

<details><summary>文書</summary>

- A `docs/crosswind_civic_feasibility.md`
- A `docs/crosswind_wind_data.md`
- A `docs/project_structure.md`
- A `docs/roadmap.md`
- A `docs/vlm_road_review.md`

</details>

<details><summary>実験索引</summary>

- A `experiments/README.md`
- A `experiments/index.csv`

</details>

<details><summary>報告</summary>

- A `reports/README.md`
- A `reports/ca/README.md`
- A `reports/ca/cases.md`
- A `reports/weekly/2026-W46/notes.md`
- A `reports/weekly/TEMPLATE.md`

</details>

<details><summary>スクリプト</summary>

- A `scripts/crosswind/README.md`
- A `scripts/crosswind/c1m_search/exposed_scan.py`
- A `scripts/crosswind/c1m_search/osm_exposure.py`
- A `scripts/crosswind/c1m_search/plot_seg.py`
- A `scripts/crosswind/c1m_search/rank_passages.py`
- A `scripts/crosswind/c1m_search/rank_passages_comp.py`
- A `scripts/crosswind/c1m_search/seg_signals.py`
- A `scripts/crosswind/c1m_search/verify_cols.py`
- A `scripts/crosswind/civic_bicycle_fit/build_grid.py`
- A `scripts/crosswind/civic_bicycle_fit/fit.py`
- A `scripts/crosswind/civic_crosswind_feasibility/analyze.py`
- A `scripts/crosswind/civic_crosswind_feasibility/windows.py`
- A `scripts/crosswind/civic_selection/collect.py`
- A `scripts/crosswind/civic_selection/pick_civic_segments.py`
- A `scripts/crosswind/civic_selection/remote_zip.py`
- A `scripts/crosswind/civic_sideslip/beta.py`
- A `scripts/crosswind/civic_sideslip/beta_wind.py`
- A `scripts/crosswind/civic_signal_check/check_signals.py`
- A `scripts/crosswind/civic_v2/analyze.py`
- A `scripts/crosswind/civic_v2/beta.py`
- A `scripts/crosswind/civic_v2/beta_wind.py`
- A `scripts/crosswind/civic_v2/build_grid.py`
- A `scripts/crosswind/civic_v2/fit.py`
- A `scripts/crosswind/civic_v2/gust_events.py`
- A `scripts/crosswind/civic_v2/plot_events.py`
- A `scripts/crosswind/civic_v2/windows.py`
- A `scripts/crosswind/path_map.csv`
- A `scripts/crosswind/xwind_search/clips.py`
- A `scripts/crosswind/xwind_search/features_scan.py`
- A `scripts/crosswind/xwind_search/frames.py`
- A `scripts/crosswind/xwind_search/grid_common.py`
- A `scripts/crosswind/xwind_search/join_wind.py`
- A `scripts/crosswind/xwind_search/plot_cand.py`
- A `scripts/crosswind/xwind_search/pooled_perm.py`
- A `scripts/crosswind/xwind_search/positions.py`
- A `scripts/crosswind/xwind_search/rank.py`
- A `scripts/crosswind/xwind_search/reversal.py`
- A `scripts/crosswind/xwind_search/reversal_wres.py`
- A `scripts/crosswind/xwind_search/score.py`
- A `scripts/crosswind/xwind_search/site_slope.py`
- A `scripts/fetch/fetch_comma2k19_remote.py`
- A `scripts/report/build_experiment_index.py`
- A `scripts/report/build_weekly_pptx.py`
- A `scripts/report/weekly_changes.py`
- A `scripts/vibration/plot_iso2631_comparison.py`
- A `scripts/vibration/plot_p2_p4_compare.py`
- A `scripts/vlm/build_road_inputs.py`
- A `scripts/vlm/run_road_review.py`
- A `scripts/vlm/score_road_review.py`

</details>

<details><summary>共有資料</summary>

- A `shared/vibration/iso2631_comparison.png`
- A `shared/vibration/site007_E0474_p2_p4.png`
- A `shared/vibration/site007_frames_E0474_day.png`
- A `shared/vibration/site024_frames_E0249_day.png`
- A `shared/vibration/site046_frames_E1010_day.png`
- A `shared/vibration/site344_frames.png`

</details>

<details><summary>コード (src)</summary>

- A `src/near_miss/io/comma2k19_remote.py`
- A `src/near_miss/vlm/road.py`
- A `src/near_miss/vlm/road_runner.py`

</details>

<details><summary>試験</summary>

- A `tests/data/near_miss_canonical_event_ids.json`
- A `tests/test_civic_selection.py`
- A `tests/test_comma2k19_remote.py`
- A `tests/test_vlm_road.py`
- A `tests/test_weekly_report.py`

</details>

## 実験索引 (experiments/index.csv) の変化

- 追加された行: 111
- なくなった行: 0
- 手で書く列が変わった箇所: 0

## 期間中に更新された out/ の結果 (索引の mtime_last)

- `out/_backup/` (2026-11-11) 対象外 / 対象外
- `out/crosswind/c1m_search/` (2026-11-10) configs/ からの参照なし / 候補を映像で確認
- `out/crosswind/xwind_search/` (2026-11-10) configs/ からの参照なし / 候補を映像で確認
- `out/vibration/near_miss/` (2026-11-09) 統一形式・API サンプルの入力 / 一部評価あり (物理シグナルと映像判定の対応)

## 報告に書いた根拠の確認

| パス | ある | git 管理 | 索引の行 | 最後のコミット |
|---|---|---|---|---|
| `out/crosswind/c1m_search/README.md` | はい | **いいえ** | out/crosswind/c1m_search/ | — |
| `out/crosswind/c1m_search/exposed_scan.csv` | はい | **いいえ** | out/crosswind/c1m_search/ | — |
| `scripts/crosswind/c1m_search/exposed_scan.py` | はい | はい | — | 1f2b926 2026-11-11 |
| `out/comma1m/` | はい | **いいえ** | out/comma1m/ | — |
| `out/crosswind/xwind_search/README.md` | はい | **いいえ** | out/crosswind/xwind_search/ | — |
| `out/crosswind/xwind_search/pooled_perm.json` | はい | **いいえ** | out/crosswind/xwind_search/ | — |
| `scripts/crosswind/xwind_search/pooled_perm.py` | はい | はい | — | 1f2b926 2026-11-11 |
| `configs/crosswind/honda_civic.yaml` | はい | はい | — | af0c3a9 2026-11-09 |
| `experiments/index.csv` | はい | はい | — | 857dd03 2026-11-11 |
| `experiments/README.md` | はい | はい | — | 99e116c 2026-11-11 |
| `docs/project_structure.md` | はい | はい | — | 857dd03 2026-11-11 |

git 管理でない根拠 (out/ など) は、索引の行で作成スクリプトと状態を追う。
