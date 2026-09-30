"""候補イベントを VLM で確認する後段レイヤ。

既存の抽出ロジックには一切依存させない。入力は candidates.csv / labels.csv と
映像・20 Hz グリッドで、出力は判定の JSONL。

    labels.csv (人手確認済み 32 件)
        -> windows.py   評価時刻の生成 (モード A: 一括 / モード B: オンライン)
        -> frames.py    フレームキャッシュから窓内の画像を選ぶ
        -> context.py   因果な CAN 文脈を組む (guard / causal)
        -> schema.py    出力 JSON schema
        -> runner.py    vLLM 呼び出し (GPU が要る。推論機でのみ import される)

**runner.py 以外は vllm を import しない。** 入力生成と採点を GPU の無い機械でも
動かせるようにするため。
"""
