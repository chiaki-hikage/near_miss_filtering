# reports/ — 報告資料の作成・管理

プロジェクト関係者・管理者への報告資料と、その原稿を置く。

```
reports/
├── README.md                 この文書
├── weekly/
│   ├── TEMPLATE.md           週次報告の原稿のひな形
│   └── YYYY-Www/             ISO 週 (例 2026-W46)。1 週 1 フォルダ
│       ├── notes.md          原稿 (人が書く)                       git 管理
│       ├── changes.md        前回報告からの変更点 (自動で作る)      git 管理
│       └── weekly_YYYY-Www.pptx  PPT (自動で作る)                  git 管理外
└── ca/                       AI 駆動開発の事例・ノウハウ・課題 (C&A 目標 2)
```

- 原稿 (Markdown) と生成スクリプトは git で管理する。**PPT は当面 git 管理外** (`.gitignore` の `reports/weekly/*/*.pptx`)。PPT はいつでも原稿から作り直せる
- 生成スクリプト: `scripts/report/weekly_changes.py` (変更点の抽出)、`scripts/report/build_weekly_pptx.py` (PPT)
- 体裁の設定: `configs/report_weekly.yaml`。社内テンプレートが見つからなかったので、暫定テンプレート (スクリプトの中で組み立てる) を使っている。テンプレートが手に入ったら `template:` にパスを書く

## 週次報告の作り方

1. `reports/weekly/TEMPLATE.md` を `reports/weekly/<YYYY-Www>/notes.md` に写す
2. 先頭の YAML を埋める
   - `base_commit`: **前回の報告の `changes.md` に書かれた比較の終点** (`..` の右側)。初回は、期間が始まる前の最後のコミット
   - `period_start`・`period_end`: ISO 週の月曜〜日曜
3. 本文を書く (下の「原稿の書き方」)
4. 変更点を抜き出して、原稿と食い違っていないか確かめる
   ```bash
   uv run python scripts/report/weekly_changes.py reports/weekly/<YYYY-Www>/notes.md
   ```
5. PPT を作る (`changes.md` も作り直す)
   ```bash
   uv run --extra report python scripts/report/build_weekly_pptx.py reports/weekly/<YYYY-Www>/notes.md
   ```
6. `notes.md` と `changes.md` をコミットする。`changes.md` の比較の終点が、次の報告の `base_commit` になる
7. `docs/roadmap.md` の状況の列と「最終更新」を直す

**比較の終点は、PPT を作った時点の HEAD** になる。原稿を書き終えてコミットしてから作ると、報告の内容と比較の範囲がそろう。未コミットの変更があると `changes.md` と PPT に注意書きが出る。

## PPT の構成 (3〜5 枚)

| 枚 | 内容 | 原稿の節 |
|---|---|---|
| 1 | 表紙と今週の要点 | `## 今週の要点` |
| 2 | 進捗 (ロードマップのテーマごと) | `## 進捗` の表 (テーマ \| 状況 \| 今週の変化) |
| 3 | 主な実験結果 (最大 3 件。根拠のパスを添える) | `## 主な実験結果` の `### 見出し` ごと |
| 4 | 課題と次週の予定 | `## 課題`、`## 次週の予定` |
| 5 | 前回報告からの変更点 (自動) | — (`weekly_changes.py` が作る) |

原稿に無い節のスライドは作らない。上限 (項目数・件数) は `configs/report_weekly.yaml` の `limits`。超えた分は「ほか N 件」にまとめる。

## 原稿の書き方

- **相手**: プロジェクト関係者・管理者。結論と、それが何を意味するかを先に書く。解析の細部は参照先の文書に任せる
- **根拠を必ず書く**: 実験結果には `- 根拠:` の行を付け、パスをバッククォートで囲む。`weekly_changes.py` がそのパスについて、あるか・git 管理か・実験索引の行か・最後のコミットを確かめて `changes.md` に書く
  - `out/` の結果は git 管理外なので、`experiments/index.csv` に行があるもの (作成スクリプト・利用状況・検証の記録を追える) を挙げる
  - 設定 (`configs/...`)、スクリプト (`scripts/...`)、文書 (`docs/...`) も、結果を左右したものは挙げる
- **確かさを書き分ける**: 「検出できなかった」「候補はあるが根拠は弱い」「検証の記録が無い」を区別する。実験索引の `verification`・`decision` を超えたこと (検証済み・採用済み) は書かない
- **予定は案として書く**: 決まっていない予定には「(案)」を付ける

## ca/ (AI 駆動開発の事例)

`reports/ca/README.md` を参照。週次報告の作業中に気づいた事例は、その週のうちに `reports/ca/cases.md` に足す。
