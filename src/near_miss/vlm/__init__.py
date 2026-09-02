"""候補イベントを VLM で確認する後段レイヤ (Phase 1)。

既存の抽出ロジックには一切依存させない。入力は candidates.csv / labels.csv と
映像・20 Hz グリッドで、出力は判定の JSONL。

    labels.csv (人手確認済み 32 件)
        -> windows.py   評価時刻の生成 (モード A: 一括 / モード B: オンライン)
        -> frames.py    フレームキャッシュから窓内の画像を選ぶ
        -> context.py   因果な CAN 文脈を組む (guard / causal)
        -> schema.py    出力 JSON schema
        -> runner.py    vLLM 呼び出し (GPU が要る。EC2 側でのみ import される)

runner.py 以外は vllm を import しない。Mac で前処理と採点を続けられるようにする。
"""
