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
    
    def __init__(self, config=None):
        """初期化
        
        Args:
            config: 設定情報を含むオブジェクト
        """
        self.config = config
    
    def calculate_bet_proportions(self, 
                    model_outputs: Dict[str, torch.Tensor], 
                    odds_data: Dict[str, torch.Tensor], 
                    **kwargs) -> Dict[str, torch.Tensor]:
        """
        モデル出力とオッズデータに基づいて資金に対する馬券購入比率を計算する
        
        注意事項:
        1. 馬券種類は個別に判断せず、全ての馬券種類を横断的に考慮して比率を決定すること
        2. 全ての馬券種類・組み合わせの比率の合計は MAX_TOTAL_BET_PROPORTION 以下でなければならない
        (合計が MAX_TOTAL_BET_PROPORTION の場合は全資金の一定割合を投資することになる)
        3. モデルの予測精度には限界があるため、慎重な資金配分が求められる
        
        Args:
            model_outputs: モデルの出力（各馬券種の予測確率を含む辞書）
            odds_data: オッズデータ（各馬券種のオッズを含む辞書）
            **kwargs: 追加のパラメータ
            
        Returns:
            dict: 馬券種ごとの購入比率のテンソルを含む辞書
                各テンソルの形状は [バッチサイズ, 組み合わせ数]
        """
        # サブクラスで実装するメソッドを呼び出して比率を計算
        bet_proportions = self._calculate_bet_proportions_impl(model_outputs, odds_data, **kwargs)

        # 必要に応じてバリデーション
                
        return bet_proportions
        
    def _calculate_bet_proportions_impl(self, 
                               model_outputs: Dict[str, torch.Tensor],
                               odds_data: Dict[str, torch.Tensor],
                               **kwargs) -> Dict[str, torch.Tensor]:
        """
        モデル出力とオッズデータに基づいて資金に対する馬券購入比率を計算する内部実装
        
        サブクラスでこのメソッドをオーバーライドして独自の購入戦略を実装する
        
        Args:
            model_outputs: モデルの出力（各馬券種の予測確率を含む辞書）
            odds_data: オッズデータ（各馬券種のオッズを含む辞書）
            **kwargs: 追加のパラメータ
            
        Returns:
            dict: 馬券種ごとの購入比率のテンソルを含む辞書
                各テンソルの形状は [バッチサイズ, 組み合わせ数]
        """
        raise NotImplementedError("サブクラスで実装する必要があります")
