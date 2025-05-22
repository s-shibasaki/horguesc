import torch
import logging
from typing import Dict, Any
from horguesc.core.betting.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class OddsDiscrepancyStrategy(BaseStrategy):
    """オッズと予測確率の乖離に基づく馬券購入戦略
    
    市場確率（オッズの逆数）と予測確率の乖離が大きい場合に賭ける戦略。
    特に予測確率が市場確率よりも高い場合に購入する。
    """
    
    def __init__(self, config=None):
        """初期化
        
        Args:
            config: 設定情報を含むオブジェクト
        """
        super().__init__(config)
        
        # 投資比率（各レースで投入する資金の割合）
        self.base_ratio = config.getfloat(
            'betting.strategies.odds_discrepancy', 'base_ratio', fallback=0.0003)
        
        # 最低乖離（この値以上の乖離がある場合のみ購入）
        self.min_discrepancy = config.getfloat(
            'betting.strategies.odds_discrepancy', 'min_discrepancy', fallback=0.05)
        
        # 最低確率閾値（この確率以上の場合のみ乖離計算を行う）
        self.min_probability = config.getfloat(
            'betting.strategies.odds_discrepancy', 'min_probability', fallback=0.001)
        
        # 乖離最大値（異常値を抑制）
        self.max_discrepancy = config.getfloat(
            'betting.strategies.odds_discrepancy', 'max_discrepancy', fallback=0.5)
        
        # 最大合計比率（レース当たりの合計投資比率の上限）
        self.max_total_bet_proportion = config.getfloat(
            'betting.strategies.odds_discrepancy', 'max_total_bet_proportion', fallback=0.05)
        
        logger.debug(f"OddsDiscrepancyStrategy initialized with base_ratio={self.base_ratio}, "
                    f"min_discrepancy={self.min_discrepancy}, min_probability={self.min_probability}, "
                    f"max_total_bet_proportion={self.max_total_bet_proportion}")
    
    def _calculate_bet_proportions_impl(self, 
                               model_outputs: Dict[str, torch.Tensor],
                               odds_data: Dict[str, torch.Tensor],
                               **kwargs) -> Dict[str, torch.Tensor]:
        """
        モデル出力とオッズデータに基づいて馬券購入比率を計算する

        Args:
            model_outputs: モデルの出力（各馬券種の予測確率を含む辞書）
            odds_data: オッズデータ（各馬券種のオッズを含む辞書）
            
        Returns:
            dict: 馬券種ごとの購入比率のテンソルを含む辞書
        """
        bet_proportions = {}
        
        # 馬券種ごとに処理
        for bet_type in self.ALL_BET_TYPES:
            probs_key = f'{bet_type}_probabilities'
            odds_key = f'odds_{bet_type}'
            
            if probs_key in model_outputs and odds_key in odds_data:
                probabilities = model_outputs[probs_key]  # [race_count, n_combinations]
                odds = odds_data[odds_key]  # [race_count, n_combinations]
                
                # 確率閾値以上の組み合わせのみ処理
                valid_prob_mask = probabilities > self.min_probability
                
                # 市場確率を計算 (オッズの逆数)
                market_probabilities = torch.zeros_like(probabilities)
                market_probabilities[valid_prob_mask] = 1.0 / odds[valid_prob_mask]
                
                # 正規化係数を計算（市場確率の合計が1を超える分を調整）
                normalization_factor = torch.sum(market_probabilities, dim=1, keepdim=True)
                # ゼロ除算を防止
                normalization_factor = torch.where(
                    normalization_factor > 0,
                    normalization_factor,
                    torch.ones_like(normalization_factor)
                )
                # 市場確率を正規化
                normalized_market_probs = market_probabilities / normalization_factor
                
                # 乖離を計算（予測確率 - 市場確率）
                discrepancy = probabilities - normalized_market_probs
                
                # 予測確率が市場確率よりも高い場合のみ検討
                positive_discrepancy = torch.clamp(discrepancy, min=0)
                
                # 最小乖離以上の場合のみ購入対象とする
                proportions = torch.zeros_like(probabilities)
                profitable_mask = positive_discrepancy >= self.min_discrepancy
                
                if profitable_mask.any():
                    # 乖離を正規化（最大値で制限）
                    capped_discrepancy = torch.clamp(positive_discrepancy, 0, self.max_discrepancy)
                    
                    # 乖離に比例した購入比率を計算（基本比率をベース）
                    # 乖離が大きいほど多く購入する
                    proportions[profitable_mask] = self.base_ratio * (capped_discrepancy[profitable_mask] / self.min_discrepancy)
                
                bet_proportions[bet_type] = proportions

        # 全馬券種横断で、レース毎の合計が max_total_bet_proportion を超えないように調整
        race_count = next(iter(model_outputs.values())).shape[0]
        
        # レース毎の全馬券種の合計比率を計算
        total_proportions_per_race = torch.zeros(race_count, device=next(iter(model_outputs.values())).device)
        
        for bet_type, proportions in bet_proportions.items():
            # 各レースにおける馬券種ごとの合計比率を加算
            total_proportions_per_race += proportions.sum(dim=1)
        
        # 最大比率を超えるレースを特定
        excess_mask = total_proportions_per_race > self.max_total_bet_proportion
        
        # 最大比率を超えるレースがある場合は調整
        if excess_mask.any():
            # 調整係数を計算 (最大比率/合計比率)
            adjustment_factors = torch.ones_like(total_proportions_per_race)
            adjustment_factors[excess_mask] = self.max_total_bet_proportion / total_proportions_per_race[excess_mask]
            
            # 各馬券種の比率を調整
            for bet_type in bet_proportions:
                # 各レースの調整係数を適用
                # [race_count, 1] * [race_count, n_combinations] -> [race_count, n_combinations]
                bet_proportions[bet_type] *= adjustment_factors.unsqueeze(1)
        
            logger.debug(f"Adjusted bet proportions for {excess_mask.sum().item()}/{race_count} races "
                        f"that exceeded max total proportion of {self.max_total_bet_proportion}")
        
        return bet_proportions