# experiments/ — 解析結果の索引

`out/` は git の管理外なので、どの結果がどのスクリプトで作られ、今どう使われているかを git で持つ索引をここに置く。
`out/` のファイルは動かしていない。索引は `out/` を読むだけで作る。

- `index.csv`: 索引の本体 (1 行 = 1 つの結果)
- 作り直し: `uv run python scripts/report/build_experiment_index.py`

## 行の単位

- `out/` 直下のフォルダ 1 つで 1 行。`out/crosswind/` と `out/vibration/` はテーマのまとまりなので、1 段下のフォルダで 1 行
- `out/` 直下のばらのファイルは、最初の `.` より前の名前でまとめて 1 行 (例 `out/chain_ccs_v2.*` は `.breakdown.csv` と `.candidates.csv` の 2 本)

## 区別していること

次の 4 つは別の事柄なので、列を分けている。**ある列の値から、ほかの列の値を決めない。**

| 事柄 | 列 | 例 |
|---|---|---|
| 現行処理での利用 | `current_use` | 設定ファイルが入力に使っている。**それだけでは、品質を確かめたことにも、正式に採用したことにもならない** |
| 文書上の区分 | `doc_label` | `docs/near_miss_data_inventory.md` の書き手が「正」(最新と判断) や「旧版」とした。書き手の判断であって、正式な採否ではない |
| 検証の記録 | `verification` | 人手ラベル・目視・自動検査・外部レビューの記録があるか |
| 正式な採否 | `decision` | 採用・不採用を決めた記録。2026-11-11 時点で記録は 1 件も無いので、すべて「未判断」 |

## 列

### 自動の列 (スクリプトが毎回作り直す)

| 列 | 内容 |
|---|---|
| `path` | 結果の場所 |
| `theme` | 名前から決めたテーマ (対応はスクリプトの `THEMES`) |
| `n_files`, `size_mb` | ファイル数と合計の大きさ |
| `mtime_first`, `mtime_last` | 中のファイルの更新日時の最古と最新。**作成日ではない** (コピーや上書きで変わる)。移動先を示す `MOVED.md` などは除いた |
| `has_readme` | フォルダに `README.md` があるか |
| `meta_files` | 実行条件を書いた `*meta*.json` |
| `newer_by_name` | 名前だけで見た後続の版 (`_vN` の N が大きいもの)。中身は比べていない |
| `referenced_by` | `docs/`・`configs/`・`tests/`・`README.md` のうち、この path を書いているファイル |
| `config_refs` | `referenced_by` のうち `configs/` のもの。入力として読む場合と、コメントに出典として書いた場合の両方を含む |
| `shared_copies` | `shared/` にある、中身 (sha256) が同じファイル。報告資料に使った結果を見つけるため |
| `attention` | 列どうしの食い違いの指摘 (下の「注意が要る行」)。手で書いた列の値は変えず、指摘だけを出す |

### 手で書く列 (スクリプトは書き換えない)

| 列 | 値 |
|---|---|
| `producer_script`, `producer_basis` | 作ったスクリプトと根拠。`--out の既定値 (:行)`・文書のコマンド・`scripts/crosswind/path_map.csv` は確かな根拠。**「推定:」で始まるものは出力ファイル名が一致しただけ**で、実行の記録は無い。推定は推定のまま残し、確かめたときだけ書き換える |
| `current_use`, `current_use_basis` | 統一形式・API サンプルの入力 / 統一形式・API サンプルの入力 (labels.csv のみ) / 現行の出力 / ほかの解析の入力 / 設定値の出典 (コメントのみ) / configs/ からの参照なし / 対象外。「configs/ からの参照なし」は、ほかのスクリプトが読んでいないことまでは確かめていない |
| `doc_label`, `doc_label_basis` | 正 / 旧版 / 旧版・別目的 (以上は `docs/near_miss_data_inventory.md`) / 作り直し前 (`docs/crosswind_civic_feasibility.md`) / 記録 (README に結論) / 記載のみ (文書に出てくるが区分なし) / 記載なし |
| `verification`, `verification_basis` | 見つかった記録の内容。見つからなければ「未確認」 |
| `decision`, `decision_basis` | 採用 / 不採用 / 未判断。決めたら、誰がいつ何を根拠に決めたかを `decision_basis` に書く |
| `check_priority` | 確認の優先度と理由。A: 今の解析・報告資料が使う / B: 設計文書が根拠として引用 / C: 参照なし |
| `note` | 補足 |

新しい行は空欄で足され、`out/` から消えた行は `current_use` が「out に無い」になって残る。

## 2026-11-11 の状況 (111 行)

- `current_use`: 統一形式・API サンプルの入力 17、同 (labels.csv のみ) 2、現行の出力 2、ほかの解析の入力 2、設定値の出典 3、configs/ からの参照なし 84、対象外 1
- `doc_label`: 正 15、旧版 33、旧版・別目的 7、作り直し前 3、記録 7、記載のみ 17、記載なし 28、対象外 1
- `verification`: 記録あり 10、未確認 100、対象外 1
- `decision`: 未判断 110、対象外 1
- `check_priority`: A 5、B 9、C 27 (旧版の索引で「未確認」だった 41 件に付けた)
- `producer_basis` が「推定:」のもの: 58

## 注意が要る行 (`attention` 列)

`doc_label` の「旧版」「旧版・別目的」「作り直し前」は、**文書の書き手がイベント一覧などの用途で古いと判断したもの**で、フォルダの中身がどこにも使われていないという意味ではない。スクリプトは次の食い違いを `attention` に書き出す。

| 指摘 | 2026-11-11 の件数 | 該当 |
|---|---|---|
| 文書上は旧版だが、現行処理が使っている | 3 | `out/vibration/pavement/` (`configs/vibration_locate.yaml:7-8` と `configs/vlm_road.yaml:50` がカメラの当てはめ結果と通過時刻を読む)、`out/chunk1/`・`out/chunk2_final/` (人手ラベル `labels.csv` だけを読む) |
| 文書上は旧版だが、報告資料に使われている | 2 | `out/vibration/final/` (図 5 本)、`out/screen_rav4_tss2_2k/` (図 1 本) |
| 文書上は旧版だが、設定値の出典として書かれている | 2 | `out/vibration/longtail/` (`configs/vibration_pavement.yaml:5`)、`out/crosswind/civic_bicycle_fit/` (`configs/crosswind/honda_civic.yaml:8`) |
| 現行処理が使っているが、検証の記録が見つかっていない | 16 | `out/chunk*_v3/`、`out/screen_TOYOTA_*_2k_v4/`、`out/sideslip_*_v3/` など |
| configs/ が参照しているのに current_use が未記入 | 0 | |

これらの行は、旧版のフォルダでも消したり動かしたりしない。整理するときは、使われているファイルだけを現行の場所へ移すか、`doc_label` を見直すかを、根拠をそろえて判断する。

## 手で書く列を直すとき

`index.csv` を直接編集してから、スクリプトを流す。手で書いた値は残り、自動の列だけが新しくなる。
まとめて書き込むときは、`path` と手で書く列を持つ CSV を `--merge-from` で渡す。**空欄でない値はそのまま上書きされる**ので、`note` などに追記するときは元の値も含めて渡す。
