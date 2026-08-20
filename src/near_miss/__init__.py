"""走行データからヒヤリハット候補区間を抽出する。

層構成:
    io.comma2k19  L0 データセット固有の読み出し
    signals       L1 一様グリッドへの再サンプルと品質フラグ
    features      L2 汎用の特徴量算出
    detectors     L3 汎用の閾値イベント検出
    scoring       L4 候補区間の統合とスコア付け

L2 以降はデータセットにも車種にも依存しない。
"""

__all__ = ["config", "signals", "features", "detectors", "scoring", "pipeline"]
