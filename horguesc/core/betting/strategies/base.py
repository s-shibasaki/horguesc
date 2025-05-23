import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

class BaseStrategy:
    """馬券購入戦略の基底クラス"""
    
    # 馬券種類の定数
    TANSHO = 'tansho'          # 単勝
    FUKUSHO = 'fukusho'        # 複勝
    WAKUREN = 'wakuren'        # 枠連
    UMAREN = 'umaren'          # 馬連
    WIDE = 'wide'              # ワイド
    UMATAN = 'umatan'          # 馬単
    SANRENPUKU = 'sanrenpuku'  # 三連複
    SANRENTAN = 'sanrentan'    # 三連単
    
    ALL_BET_TYPES = [TANSHO, FUKUSHO, WAKUREN, UMAREN, WIDE, UMATAN, SANRENPUKU, SANRENTAN]
    
    # 馬券の最小購入単位（日本の馬券は100円単位）
    BET_UNIT = 100.0
    
    def __init__(self, config=None):
        """初期化
        
        Args:
            config: 設定情報を含むオブジェクト
        """
        self.config = config
    
    def calculate_bet_amounts(self, 
                    model_outputs: Dict[str, torch.Tensor], 
                    odds_data: Dict[str, torch.Tensor],
                    initial_capital: float = 300000.0,
                    **kwargs) -> Dict[str, torch.Tensor]:
        """
        モデル出力とオッズデータに基づいて馬券購入金額を計算する
        
        Args:
            model_outputs: モデルの出力（各馬券種の予測確率を含む辞書）
                キー命名規則: "{馬券種}_probabilities"
                例: "sanrentan_probabilities", "umatan_probabilities", "tansho_probabilities"
                
                各テンソルの形状:
                - tansho_probabilities: [バッチサイズ, 馬数]
                - fukusho_probabilities: [バッチサイズ, 馬数]
                - wakuren_probabilities: [バッチサイズ, 枠連組合せ数]
                - umaren_probabilities: [バッチサイズ, 馬連組合せ数]
                - wide_probabilities: [バッチサイズ, ワイド組合せ数]
                - umatan_probabilities: [バッチサイズ, 馬単組合せ数]
                - sanrenpuku_probabilities: [バッチサイズ, 三連複組合せ数]
                - sanrentan_probabilities: [バッチサイズ, 三連単組合せ数]
                
            odds_data: オッズデータ（各馬券種のオッズを含む辞書）
                キー命名規則: "odds_{馬券種}"
                例: "odds_sanrentan", "odds_umatan", "odds_tansho"
                
                各テンソルの形状:
                - odds_tansho: [バッチサイズ, 馬数]
                - odds_fukusho: [バッチサイズ, 馬数]
                - odds_wakuren: [バッチサイズ, 枠連組合せ数]
                - odds_umaren: [バッチサイズ, 馬連組合せ数]
                - odds_wide: [バッチサイズ, ワイド組合せ数]
                - odds_umatan: [バッチサイズ, 馬単組合せ数]
                - odds_sanrenpuku: [バッチサイズ, 三連複組合せ数]
                - odds_sanrentan: [バッチサイズ, 三連単組合せ数]
                
                注意: オッズデータには無効な組み合わせや発売されていない馬券の場合に
                NaN (Not a Number) 値が含まれることがあります。戦略を実装する際は、
                torch.isnan() を使用して NaN をチェックし、適切に処理してください。
                
            initial_capital: 初期資金 (円単位)
            **kwargs: 追加のパラメータ
            
        Returns:
            dict: 馬券種ごとの購入金額のテンソルを含む辞書
                キーは self.ALL_BET_TYPES に含まれる馬券種名と一致する
                各テンソルの形状は [バッチサイズ, 組み合わせ数]
                金額は円単位でBET_UNIT (100円) の倍数に丸められる
        """
        # サブクラスで実装するメソッドを呼び出して購入金額を計算
        bet_amounts = self._calculate_bet_amounts_impl(model_outputs, odds_data, initial_capital, **kwargs)

        # バリデーション
        if not isinstance(bet_amounts, dict):
            raise TypeError("bet_amounts must be a dictionary")
        
        # 各馬券種類の購入金額を検証して調整
        for bet_type, amounts in bet_amounts.items():
            if bet_type not in self.ALL_BET_TYPES:
                raise ValueError(f"Unknown bet type: {bet_type}")
            
            if not isinstance(amounts, torch.Tensor):
                raise TypeError(f"Bet amounts for {bet_type} must be a torch.Tensor")
            
            if amounts.ndim != 2:
                raise ValueError(f"Bet amounts tensor for {bet_type} must have shape [batch_size, combinations]")
            
            # 負の値がないことを確認
            if torch.any(amounts < 0):
                logger.warning(f"Negative bet amounts found for {bet_type}. Setting to 0.")
                bet_amounts[bet_type] = torch.clamp(amounts, min=0)
            
            # 馬券の最小購入単位 (100円) に丸める
            # - 100円未満の端数は切り捨て
            bet_amounts[bet_type] = torch.floor(bet_amounts[bet_type] / self.BET_UNIT) * self.BET_UNIT
                
        return bet_amounts
        
    def _calculate_bet_amounts_impl(self, 
                      model_outputs: Dict[str, torch.Tensor],
                      odds_data: Dict[str, torch.Tensor],
                      initial_capital: float = 300000.0,
                      **kwargs) -> Dict[str, torch.Tensor]:
        """
        モデル出力とオッズデータに基づいて馬券購入金額を計算する内部実装
        
        サブクラスでこのメソッドをオーバーライドして独自の購入戦略を実装する
        
        注意事項:
        1. 馬券種類は横断的に考慮して、全体として最適な購入金額を決定すること
        2. 購入金額はBET_UNIT (100円) の倍数に丸められる（この丸め処理はcalculate_bet_amountsで実行される）
        3. モデルの予測精度には限界があるため、慎重な資金配分が求められる
        4. 各レースは独立して処理すること（レース間での情報の漏洩や依存関係を作らないこと）
        5. オッズデータには NaN 値が含まれる場合があります。これらは無効な組み合わせや
           発売されていない馬券を表します。NaN 値に対する賭け金は常にゼロにしてください。
           torch.isnan() を使用して NaN 値をチェックし、torch.nan_to_num() や
           マスキング処理で適切に対応してください。
        6. 馬券の購入量に注意してください。特に三連単などの組み合わせ数が多い馬券種では、
           一つのレースで数千通りの組み合わせがあります。少額を多数購入すると合計金額が過大になります。
           各レースでの合計購入金額に上限を設定し、購入件数を制限するか、高確率/高期待値の
           組み合わせのみに集中して購入するようにしてください。
        7. 各馬券種類で購入金額の配分を考慮してください。資金全体に対する割合（例:5%）を
           複数の馬券種で独立に使うと、合計が過大になる可能性があります。
        8. 各レースで使用する資金を初期資金の一定割合内に抑えることで、長期的なシミュレーションの安定性が向上します。
           レースごとの購入上限額を設定することを推奨します。
        
        Args:
            model_outputs: モデルの出力（各馬券種の予測確率を含む辞書）
                キー命名規則: "{馬券種}_probabilities" (例: "sanrentan_probabilities")
            odds_data: オッズデータ（各馬券種のオッズを含む辞書）
                キー命名規則: "odds_{馬券種}" (例: "odds_sanrentan")
                ※オッズデータには NaN 値が含まれる場合があります
            initial_capital: 初期資金 (円単位)
            **kwargs: 追加のパラメータ
    
        Returns:
            dict: 馬券種ごとの購入金額のテンソルを含む辞書
                キーは馬券種名 (例: "sanrentan", "tansho")
                各テンソルの形状は [バッチサイズ, 組み合わせ数]
                金額は円単位
        """
        raise NotImplementedError("サブクラスで実装する必要があります")
