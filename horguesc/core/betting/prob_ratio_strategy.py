import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union
from horguesc.core.betting.strategy import BettingStrategy

logger = logging.getLogger(__name__)

class ProbabilityRatioStrategy(BettingStrategy):
    """確率比に基づいた馬券購入戦略
    
    モデルの予測確率とオッズから計算される市場確率の比に基づいて投資する戦略
    """
    
    def __init__(self, config=None):
        """初期化
        
        Args:
            config: 設定情報を含むオブジェクト
        """
        super().__init__(config)
        
        # 追加設定の読み込み
        if config:
            # 最小確率比閾値
            self.min_ratio = config.getfloat(
                'betting.prob_ratio', 'min_ratio', fallback=1.1)
            
            # 予測確率の最小閾値
            self.min_probability = config.getfloat(
                'betting.prob_ratio', 'min_probability', fallback=0.001)
            
            # オッズの最大値（異常値除去）
            self.max_odds = config.getfloat(
                'betting.prob_ratio', 'max_odds', fallback=1000.0)
            
            # オッズの最小値（ゼロ除算防止）
            self.min_odds = config.getfloat(
                'betting.prob_ratio', 'min_odds', fallback=1.01)
            
            # 配分タイプ (proportional, square, log)
            self.allocation_type = config.get(
                'betting.prob_ratio', 'allocation_type', fallback='square')
        else:
            # デフォルト値
            self.min_ratio = 1.1
            self.min_probability = 0.001
            self.max_odds = 1000.0
            self.min_odds = 1.01
            self.allocation_type = 'square'
            
        logger.info(f"確率比戦略を初期化: 最小比率={self.min_ratio}, "
                   f"最小確率={self.min_probability}, "
                   f"最大オッズ={self.max_odds}, "
                   f"最小オッズ={self.min_odds}, "
                   f"配分タイプ={self.allocation_type}")
    
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
        
        # 各馬券種の確率比を計算
        for bet_type in self.allowed_bet_types:
            # 確率とオッズのデータが存在するか確認
            prob_key = f'{bet_type}_probabilities'
            odds_key = f'odds_{bet_type}'
            
            if prob_key in model_outputs and odds_key in odds_data:
                # 予測確率とオッズを取得
                probs = model_outputs[prob_key]
                odds = odds_data[odds_key]
                
                # オッズの前処理: NaN → 0, 最大/最小制限
                odds = torch.nan_to_num(odds, nan=0.0)
                odds = torch.clamp(odds, min=self.min_odds, max=self.max_odds)
                
                # Debug information
                logger.debug(f"Ratio Strategy - {bet_type}: " 
                            f"Odds shape: {odds.shape}, min: {odds.min().item():.2f}, max: {odds.max().item():.2f}, "
                            f"Probs shape: {probs.shape}, min: {probs.min().item():.6f}, max: {probs.max().item():.6f}")
                
                # 市場確率を計算（控除率を考慮せず単純に1/オッズ）
                # 本来は控除率も考慮すべきだが、簡略化のため省略
                market_probs = 1.0 / odds
                
                # モデル確率と市場確率の比を計算
                # ゼロ除算を防ぐためにepsilonを追加
                epsilon = 1e-10
                probability_ratio = probs / (market_probs + epsilon)
                
                # 確率が最小閾値未満のエントリは比率を0にする
                probability_ratio = torch.where(
                    probs >= self.min_probability,
                    probability_ratio,
                    torch.tensor(0.0, device=device)
                )
                
                # 閾値以上の比率を持つエントリのみを対象にする
                good_value = torch.where(
                    probability_ratio >= self.min_ratio,
                    probability_ratio,
                    torch.tensor(0.0, device=device)
                )
                
                # ゼロでない比率があるかチェック
                if torch.any(good_value > 0):
                    # レースごとに処理
                    race_bets = torch.zeros_like(good_value)
                    
                    for i in range(batch_size):
                        race_values = good_value[i]
                        
                        # 閾値以上の比率を持つ馬券がある場合のみ処理
                        if torch.any(race_values > 0):
                            # 配分方法に応じて賭け金を決定
                            if self.allocation_type == 'proportional':
                                # 比率に比例して配分
                                weights = race_values
                            elif self.allocation_type == 'square':
                                # 比率の二乗に比例して配分（差を強調）
                                weights = torch.pow(race_values, 2)
                            elif self.allocation_type == 'log':
                                # 比率の対数に比例して配分
                                # 比率が1以下の場合は0になるよう調整
                                adjusted_ratio = torch.maximum(race_values - 1.0, torch.tensor(0.0, device=device))
                                weights = torch.log1p(adjusted_ratio)  # log(1+x)
                            else:
                                # デフォルトは比率に比例
                                weights = race_values
                            
                            # 重みの合計を計算
                            total_weight = weights.sum()
                            
                            # 合計が0より大きい場合のみ配分
                            if total_weight > 0:
                                race_bets[i] = weights / total_weight * self.max_bet_per_race
                    
                    # 最小賭け金以下の場合は切り捨て
                    race_bets = torch.where(
                        race_bets >= self.min_bet_per_combo,
                        race_bets,
                        torch.tensor(0.0, device=device)
                    )
                    
                    # 結果に追加
                    bet_amounts[bet_type] = race_bets
                else:
                    # 閾値以上の比率がない場合はゼロを設定
                    bet_amounts[bet_type] = torch.zeros_like(good_value)
            else:
                logger.debug(f"{bet_type}の確率またはオッズデータがありません")
        
        return bet_amounts