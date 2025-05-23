import torch
import logging
from typing import Dict, Any, Optional
import numpy as np
from horguesc.core.betting.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class KellyBasedStrategy(BaseStrategy):
    """ケリー基準と期待値に基づく馬券購入戦略"""
    
    def __init__(self, config=None):
        """初期化"""
        super().__init__(config)
        
        # 設定からパラメータを取得（デフォルト値付き）
        if config:
            self.betting_factor = config.getfloat('betting.strategy.kelly', 'betting_factor', fallback=0.1)
            self.ev_threshold = config.getfloat('betting.strategy.kelly', 'ev_threshold', fallback=1.05)
            self.limit_factor = config.getfloat('betting.strategy.kelly', 'limit_factor', fallback=0.01)
        else:
            # デフォルトパラメータ
            self.betting_factor = 0.1  # 係数b - 確率に対する賭け金の割合
            self.ev_threshold = 1.05   # 閾値e - 期待値の最小値
            self.limit_factor = 0.01   # 係数l - レース毎の最大賭け金の初期資金に対する割合

        logger.info(f"KellyBasedStrategy initialized: betting_factor={self.betting_factor}, "
                    f"ev_threshold={self.ev_threshold}, limit_factor={self.limit_factor}")

    def _calculate_bet_amounts_impl(self, 
                      model_outputs: Dict[str, torch.Tensor],
                      odds_data: Dict[str, torch.Tensor],
                      initial_capital: float = 300000.0,
                      **kwargs) -> Dict[str, torch.Tensor]:
        """
        モデル出力とオッズデータに基づいて馬券購入金額を計算する内部実装
        
        Args:
            model_outputs: モデルの出力（各馬券種の予測確率を含む辞書）
            odds_data: オッズデータ（各馬券種のオッズを含む辞書）
            initial_capital: 初期資金 (円単位)
            
        Returns:
            dict: 馬券種ごとの購入金額のテンソルを含む辞書
        """
        # 1. 全ての馬券種のオッズと確率を収集する準備
        all_probs = []
        all_odds = []
        bet_type_indices = {}  # 馬券種ごとの開始・終了インデックスを記録
        current_idx = 0
        
        for bet_type in self.ALL_BET_TYPES:
            prob_key = f"{bet_type}_probabilities"
            odds_key = f"odds_{bet_type}"
            
            if prob_key in model_outputs and odds_key in odds_data:
                probs = model_outputs[prob_key]  # [race_count, n_combinations]
                odds = odds_data[odds_key]       # [race_count, n_combinations]
                
                # インデックスの範囲を記録
                start_idx = current_idx
                n_combinations = probs.shape[1]
                end_idx = start_idx + n_combinations
                bet_type_indices[bet_type] = (start_idx, end_idx)
                current_idx = end_idx
                
                # リストに追加
                all_probs.append(probs)
                all_odds.append(odds)
        
        # 2. 馬券種がない場合は空辞書を返す
        if not all_probs:
            logger.warning("有効な馬券データがありません。")
            return {}
        
        # 3. 全ての馬券種を結合して大きなテンソルを作る
        # [race_count, total_combinations]
        flat_probs = torch.cat(all_probs, dim=1)
        flat_odds = torch.cat(all_odds, dim=1)
        
        # 4. NaNを0に置き換え
        flat_probs = torch.nan_to_num(flat_probs, nan=0.0)
        flat_odds = torch.nan_to_num(flat_odds, nan=0.0)
        
        # 5. 期待値テンソルを計算: EV = probability * odds
        expected_values = flat_probs * flat_odds
        
        # 6. 購入金額テンソルを計算: amount = probability * betting_factor * initial_capital
        bet_amounts = flat_probs * self.betting_factor * initial_capital
        
        # 7. 期待値が閾値未満の投票をゼロにする
        bet_amounts = torch.where(expected_values >= self.ev_threshold, bet_amounts, torch.zeros_like(bet_amounts))
        
        # 8. レース毎の合計金額を計算
        race_totals = torch.sum(bet_amounts, dim=1)  # [race_count]
        
        # 9. 上限を超えるレースの調整係数を計算
        limit_amount = initial_capital * self.limit_factor  # レースごとの上限額
        scaling_factors = torch.ones(bet_amounts.shape[0], device=bet_amounts.device)
        
        # 上限を超えるレースは比例配分で調整
        mask = race_totals > limit_amount
        scaling_factors[mask] = limit_amount / race_totals[mask]
        
        # 10. 調整係数を使って購入金額を調整
        bet_amounts = bet_amounts * scaling_factors.unsqueeze(1)
        
        # 11. 結果を馬券種ごとに分割して辞書に格納
        result = {}
        for bet_type, (start_idx, end_idx) in bet_type_indices.items():
            result[bet_type] = bet_amounts[:, start_idx:end_idx]
            
        return result