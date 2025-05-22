import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union
from horguesc.core.betting.strategy import BettingStrategy

logger = logging.getLogger(__name__)

class ExpectedValueStrategy(BettingStrategy):
    """期待値に基づいた馬券購入戦略
    
    予測確率とオッズから期待値を計算し、プラスの期待値を持つ馬券に投資する戦略
    """
    
    def __init__(self, config=None):
        """初期化
        
        Args:
            config: 設定情報を含むオブジェクト
        """
        super().__init__(config)
        
        # 追加設定の読み込み
        if config:
            # プラス期待値の閾値
            self.ev_threshold = config.getfloat(
                'betting.ev_strategy', 'ev_threshold', fallback=0.0)
            
            # 予測確率の最小閾値
            self.min_probability = config.getfloat(
                'betting.ev_strategy', 'min_probability', fallback=0.0001)
            
            # 配分タイプ (proportional or equal)
            self.allocation_type = config.get(
                'betting.ev_strategy', 'allocation_type', fallback='proportional')
            
            # 期待値の累乗係数（期待値の差を強調）
            self.ev_power = config.getfloat(
                'betting.ev_strategy', 'ev_power', fallback=1.0)
        else:
            # デフォルト値
            self.ev_threshold = 0.0
            self.min_probability = 0.0001
            self.allocation_type = 'proportional'
            self.ev_power = 1.0
            
        logger.debug(f"期待値戦略を初期化: 閾値={self.ev_threshold}, "
                   f"最小確率={self.min_probability}, "
                   f"配分タイプ={self.allocation_type}, "
                   f"期待値累乗係数={self.ev_power}")
    
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
        
        # 各馬券種の期待値と投資額を計算
        for bet_type in self.allowed_bet_types:
            # 確率とオッズのデータが存在するか確認
            prob_key = f'{bet_type}_probabilities'
            odds_key = f'odds_{bet_type}'
            
            if prob_key in model_outputs and odds_key in odds_data:
                # 予測確率とオッズを取得
                probs = model_outputs[prob_key]
                odds = odds_data[odds_key]
                
                # デバッグ情報
                logger.debug(f"EV Strategy - {bet_type}: " 
                            f"Odds shape: {odds.shape}, min: {odds.min().item():.2f}, max: {odds.max().item():.2f}, "
                            f"Probs shape: {probs.shape}, min: {probs.min().item():.6f}, max: {probs.max().item():.6f}")
                
                # NaNを0に置き換え
                odds = torch.nan_to_num(odds, nan=0.0)
                
                # 期待値を計算: (オッズ * 確率 - 1)
                # オッズは払戻率（賭け金を含む）なので、利益だけを取るために1を引く
                expected_values = odds * probs - 1.0
                
                # 期待値のデバッグ
                if torch.any(expected_values > 0):
                    positive_count = torch.sum(expected_values > 0).item()
                    pos_ev_max = torch.max(expected_values).item()
                    logger.debug(f"Found {positive_count} positive EVs, max: {pos_ev_max:.4f}")
                else:
                    logger.debug(f"No positive expected values found for {bet_type}")
                
                # 最小確率未満のエントリは期待値を0にする
                expected_values = torch.where(
                    probs >= self.min_probability,
                    expected_values,
                    torch.tensor(0.0, device=device)
                )
                
                # 閾値以上の期待値を持つエントリのみを対象にする
                positive_ev = torch.where(
                    expected_values > self.ev_threshold,
                    expected_values,
                    torch.tensor(0.0, device=device)
                )
                
                # ゼロでない期待値があるかチェック
                if torch.any(positive_ev > 0):
                    # レースごとに処理
                    race_bets = torch.zeros_like(positive_ev)
                    
                    for i in range(batch_size):
                        race_ev = positive_ev[i]
                        
                        # プラスの期待値を持つ馬券がある場合のみ処理
                        if torch.any(race_ev > 0):
                            # 配分方法に応じて賭け金を決定
                            if self.allocation_type == 'equal':
                                # 均等配分: プラス期待値の馬券に均等に配分
                                positive_mask = race_ev > 0
                                num_positive = positive_mask.sum().item()
                                if num_positive > 0:
                                    race_bets[i, positive_mask] = self.max_bet_per_race / num_positive
                            else:
                                # 比例配分: 期待値に比例して配分（デフォルト）
                                # 期待値の累乗を使用して差を強調
                                weighted_ev = torch.pow(race_ev, self.ev_power)
                                total_weighted_ev = weighted_ev.sum()
                                if total_weighted_ev > 0:
                                    race_bets[i] = weighted_ev / total_weighted_ev * self.max_bet_per_race
                    
                    # 最小賭け金以下の場合は切り捨て
                    race_bets = torch.where(
                        race_bets >= self.min_bet_per_combo,
                        race_bets,
                        torch.tensor(0.0, device=device)
                    )
                    
                    # 結果に追加
                    bet_amounts[bet_type] = race_bets
                else:
                    # プラス期待値がない場合はゼロを設定
                    bet_amounts[bet_type] = torch.zeros_like(positive_ev)
            else:
                logger.debug(f"{bet_type}の確率またはオッズデータがありません")
        
        return bet_amounts