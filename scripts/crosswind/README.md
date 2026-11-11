# scripts/crosswind/ — 横風の解析スクリプト

横風の影響の検討 (comma2k19 Civic・RAV4、comma1M) で使ったスクリプト。
2026-11-11 に `out/crosswind/` の下 (35 本) と `data/comma2k19_civic/_selection/` の下 (3 本) から移した。**内容は変えていない** (移動の前後で sha256 が一致することを確認した)。

## 置き方

解析ごとにフォルダを分け、出力先の `out/crosswind/<解析名>/` と同じ名前にしている。
解析ごとに同じ名前で中身の違うファイルがあるので (`windows.py`, `fit.py` など)、フォルダはまとめない。

| フォルダ | 出力先 | 説明 |
|---|---|---|
| `civic_signal_check/` | `out/crosswind/civic_signal_check/` | Civic の信号の確認 |
| `civic_bicycle_fit/` | `out/crosswind/civic_bicycle_fit/` | 単軌道モデルの当てはめ |
| `civic_crosswind_feasibility/` | `out/crosswind/civic_crosswind_feasibility/` | 直進窓での横風との関係 |
| `civic_sideslip/` | `out/crosswind/civic_sideslip/` | 横滑り角 β と横風 |
| `civic_v2/` | `out/crosswind/civic_v2/`, `civic_gust_events*/` | 全精度で作り直した版と、突風イベントの抽出 |
| `xwind_search/` | `out/crosswind/xwind_search/` | Civic + RAV4 の横風候補の探索 |
| `c1m_search/` | `out/crosswind/c1m_search/` | comma1M の横風候補の探索 |
| `civic_selection/` | `data/comma2k19_civic/selection.csv`、`data/comma2k19_civic/_selection/wind/remote_chunk_N.csv` | Civic の取得セグメントの選定 (下の「Civic の選定」) |

手順と引数は、各出力先の `README.md` と `docs/crosswind_civic_feasibility.md` §11 に書いてある。

## 実行のしかた

**リポジトリの直下で実行する。** 入力のパスはどれもリポジトリ直下からの相対パスで書いてある (`raw_data/...`, `configs/...`, `src`)。多くは出力先のフォルダを第 1 引数で受け取る。

```bash
uv run python scripts/crosswind/xwind_search/rank.py out/crosswind/xwind_search
```

既存の結果を上書きしたくないときは、出力先を別のフォルダにする。

### Civic の選定 (`civic_selection/`)

`out/` ではなく `data/comma2k19_civic/` に書く。3 本の関係と実行のしかた:

| スクリプト | 入力 | 出力 | 注意 |
|---|---|---|---|
| `remote_zip.py` | Hugging Face の `commaai/comma2k19` の `Chunk_N.zip` (HTTP の範囲指定) | — | `collect.py` が import する部品。同じフォルダに置く |
| `collect.py` | 引数にチャンク番号 | **カレントディレクトリ**に `remote_chunk_N.csv` | ネットワークに接続する。引数なしなら何もしない |
| `pick_civic_segments.py` | 第 1 引数のフォルダの `all_segments_wind.csv` と `remote_chunk_3〜10.csv` | 第 2 引数のファイル | ネットワークは使わない |

```bash
# 選定 (既存の selection.csv と同じものができることを tests/test_civic_selection.py で確かめている)
uv run python scripts/crosswind/civic_selection/pick_civic_segments.py data/comma2k19_civic/_selection/wind <出力.csv>

# 時刻と位置の収集 (取得になる。既存の remote_chunk_N.csv を上書きしないよう、別のフォルダで実行する)
cd <作業フォルダ> && uv run --project <リポジトリ> python <リポジトリ>/scripts/crosswind/civic_selection/collect.py 3 4 5
```

`all_segments_wind.csv` を作った処理と、SFO の気象データを取得した処理はスクリプトとして残っていない。分かっていることは `docs/crosswind_wind_data.md`。

## 旧パスとの対応

- `path_map.csv`: 旧パス・新パス・sha256 の対応表 (38 本)
- 移動前の写し: `out/_backup/crosswind_scripts_2026-11-11.tar.gz` と `out/_backup/civic_selection_scripts_2026-11-11.tar.gz` (sha256 の一覧は同じ場所の `.sha256`)
- 旧パスを書いた README と文書は書き換えていない。旧位置には、移動先を示す `MOVED.md` / `SCRIPTS_MOVED.md` を置いた

この直下には `.py` を置かない。`scripts/_bootstrap.py` が `scripts/crosswind/` を `sys.path` に入れるため、直下に `fit.py` などを置くと、ほかのスクリプトの import と名前がぶつかるおそれがある。
