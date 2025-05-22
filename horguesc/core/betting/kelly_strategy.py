import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union
from horguesc.core.betting.strategy import BettingStrategy

logger = logging.getLogger(__name__)

class KellyStrategy(BettingStrategy):
    """ケリー基準に基づいた馬券購入戦略
    
    ケリー基準を応用して最適な賭け金を決定する戦略
    """
    
    def __init__(self, config=None):
        """初期化
        
        Args:
            config: 設定情報を含むオブジェクト
        """
        super().__init__(config)
        
        # 追加設定の読み込み
        if config:
            # ケリー係数（フルケリーに対する割合、リスク調整用）
            self.kelly_fraction = config.getfloat(
                'betting.kelly', 'kelly_fraction', fallback=0.25)
            
            # 予測確率の最小閾値
            self.min_probability = config.getfloat(
                'betting.kelly', 'min_probability', fallback=0.001)
            
            # オッズの最大値（異常値除去）
            self.max_odds = config.getfloat(
                'betting.kelly', 'max_odds', fallback=1000.0)
            
            # 排他的オプション: 馬券種ごとに別々の予算を使うかどうか
            self.separate_budgets = config.getboolean(
                'betting.kelly', 'separate_budgets', fallback=False)
        else:
            # デフォルト値
            self.kelly_fraction = 0.25  # 四分の一ケリー（リスク軽減）
            self.min_probability = 0.001
            self.max_odds = 1000.0
            self.separate_budgets = False
            
        logger.info(f"ケリー戦略を初期化: 係数={self.kelly_fraction}, "
                   f"最小確率={self.min_probability}, "
                   f"最大オッズ={self.max_odds}, "
                   f"分離予算={self.separate_budgets}")
    
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
        device = list(model_outputs.values())[0].device
        batch_size = list(model_outputs.values())[0].shape[0]
        
        # 結果の辞書を初期化
        bet_amounts = {}
        
        # 馬券種ごとの予算を設定
        if self.separate_budgets:
            # 対象の馬券種の数に応じて予算を分配
            bet_type_count = sum(1 for bt in self.allowed_bet_types 
                               if f'{bt}_probabilities' in model_outputs 
                               and f'odds_{bt}' in odds_data)
            budget_per_type = self.max_bet_per_race / max(1, bet_type_count)
        else:
            # 共通予算
            budget_per_type = self.max_bet_per_race
        
        # 各馬券種のケリー基準賭け金を計算
        for bet_type in self.allowed_bet_types:
            # 確率とオッズのデータが存在するか確認
            prob_key = f'{bet_type}_probabilities'
            odds_key = f'odds_{bet_type}'
            
            if prob_key in model_outputs and odds_key in odds_data:
                # 予測確率とオッズを取得
                probs = model_outputs[prob_key]
                odds = odds_data[odds_key]
                
                # オッズの前処理: NaN → 0, 最大値制限
                odds = torch.nan_to_num(odds, nan=0.0)
                odds = torch.clamp(odds, max=self.max_odds)
                
                # ケリー基準の計算: f* = p - (1-p)/(b-1)
                # p: 確率, b: オッズ (払戻率), f*: 賭け金の割合
                # 分母がゼロにならないよう調整
                adjusted_odds = torch.maximum(odds - 1.0, torch.tensor(0.01, device=device))
                kelly_fractions = probs - (1.0 - probs) / adjusted_odds
                
                # 分数ケリーを適用（リスク軽減）
                kelly_fractions = kelly_fractions * self.kelly_fraction
                
                # 負の値を0に（賭けない）
                kelly_fractions = torch.maximum(kelly_fractions, torch.tensor(0.0, device=device))
                
                # 最小確率未満のエントリは0に
                kelly_fractions = torch.where(
                    probs >= self.min_probability,
                    kelly_fractions,
                    torch.tensor(0.0, device=device)
                )
                
                # レースごとに処理
                race_bets = torch.zeros_like(kelly_fractions)
                
                for i in range(batch_size):
                    race_kelly = kelly_fractions[i]
                    
                    # ケリー値が正の馬券がある場合のみ処理
                    if torch.any(race_kelly > 0):
                        # 全ての賭け金合計が予算を超える場合は比例縮小
                        total_fraction = race_kelly.sum()
                        if total_fraction > 1.0:
                            race_kelly = race_kelly / total_fraction
                        
                        # 予算に基づいて賭け金を計算
                        race_bets[i] = race_kelly * budget_per_type
                
                # 最小賭け金以下の場合は切り捨て
                race_bets = torch.where(
                    race_bets >= self.min_bet_per_combo,
                    race_bets,
                    torch.tensor(0.0, device=device)
                )
                
                # 結果に追加
                bet_amounts[bet_type] = race_bets
            else:
                logger.debug(f"{bet_type}の確率またはオッズデータがありません")
        
        return bet_amounts