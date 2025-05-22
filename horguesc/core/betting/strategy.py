import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

class BettingStrategy:
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
        
        # 設定から戦略パラメータを読み込む
        self._load_parameters()
    
    def _load_parameters(self):
        """設定から戦略パラメータを読み込む"""
        if self.config:
            # 最大賭け金（レース毎）
            self.max_bet_per_race = self.config.getfloat(
                'betting.strategy', 'max_bet_per_race', fallback=10000.0)
            
            # 最小賭け金（組み合わせ毎）
            self.min_bet_per_combo = self.config.getfloat(
                'betting.strategy', 'min_bet_per_combo', fallback=100.0)
            
            # 許容される馬券種
            bet_types_str = self.config.get(
                'betting.strategy', 'allowed_bet_types', 
                fallback=','.join(self.ALL_BET_TYPES))
            self.allowed_bet_types = [bt.strip() for bt in bet_types_str.split(',')]
        else:
            # デフォルト値
            self.max_bet_per_race = 10000.0  # 1レース最大1万円
            self.min_bet_per_combo = 100.0   # 1組み合わせ最小100円
            self.allowed_bet_types = self.ALL_BET_TYPES
        
        logger.debug(f"戦略パラメータを設定: 最大賭け金={self.max_bet_per_race}, "
                   f"最小賭け金={self.min_bet_per_combo}, "
                   f"許容馬券種={self.allowed_bet_types}")
    
    def calculate_bets(self, 
                       model_outputs: Dict[str, torch.Tensor], 
                       odds_data: Dict[str, torch.Tensor], 
                       **kwargs) -> Dict[str, torch.Tensor]:
        """モデル出力とオッズデータに基づいて馬券購入量を計算する
        
        Args:
            model_outputs: モデルの出力（各馬券種の予測確率を含む辞書）
            odds_data: オッズデータ（各馬券種のオッズを含む辞書）
            **kwargs: 追加のパラメータ
            
        Returns:
            dict: 馬券種ごとの購入金額のテンソルを含む辞書
        """
        raise NotImplementedError("サブクラスで実装する必要があります")
