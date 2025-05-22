import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from horguesc.core.betting.strategy import BettingStrategy

logger = logging.getLogger(__name__)

class ExpectedValueStrategy(BettingStrategy):
    """馬券の期待値に基づいて購入比率を決定する戦略
    
    この戦略は各馬券の期待値を計算し、正の期待値を持つ組合せに資金を配分します。
    配分は期待値の大きさに比例して行われます。
    """
    
    def __init__(self, config=None):
        """初期化
        
        Args:
            config: 設定情報を含むオブジェクト
        """
        super().__init__(config)
        
        # 設定からパラメータを取得
        self.min_ev_threshold = config.getfloat('betting.expected_value', 'min_ev_threshold', fallback=0.1)
        self.max_allocation_per_race = config.getfloat('betting.expected_value', 'max_allocation_per_race', fallback=0.2)
        self.max_allocation_per_bet = config.getfloat('betting.expected_value', 'max_allocation_per_bet', fallback=0.05)
        
        # ベットタイプの優先順位 (デフォルトでは三連単を最優先)
        self.bet_type_priority = {
            self.SANRENTAN: 1,
            self.SANRENPUKU: 2,
            self.UMATAN: 3,
            self.UMAREN: 4,
            self.WIDE: 5,
            self.TANSHO: 6,
            self.FUKUSHO: 7,
            self.WAKUREN: 8
        }
        
        logger.debug(f"ExpectedValueStrategy initialized with threshold={self.min_ev_threshold}")
    
    def calculate_bet_proportions(self, 
                       model_outputs: Dict[str, torch.Tensor], 
                       odds_data: Dict[str, torch.Tensor], 
                       **kwargs) -> Dict[str, torch.Tensor]:
        """モデル出力とオッズデータに基づいて資金に対する馬券購入比率を計算する
        
        Args:
            model_outputs: モデルの出力（各馬券種の予測確率を含む辞書）
            odds_data: オッズデータ（各馬券種のオッズを含む辞書）
            **kwargs: 追加のパラメータ
            
        Returns:
            dict: 馬券種ごとの購入比率のテンソルを含む辞書
        """
        bet_proportions = {}
        device = next(iter(model_outputs.values())).device
        
        # 各ベットタイプの期待値を計算
        expected_values_by_type = {}
        
        # 各ベットタイプごとに処理
        for bet_type in self.ALL_BET_TYPES:
            # このベットタイプの確率とオッズを取得（存在する場合）
            prob_key = f'{bet_type}_probabilities'
            odds_key = f'odds_{bet_type}'
            
            if prob_key in model_outputs and odds_key in odds_data:
                # 確率とオッズを取得
                probabilities = model_outputs[prob_key]
                odds = odds_data[odds_key].to(device)
                
                # NaNを0に置き換え
                odds = torch.nan_to_num(odds, nan=0.0)
                
                # 期待値を計算: EV = probability * odds - 1
                # オッズは100円当たりの配当なので、1を引く
                expected_values = probabilities * odds - 1.0
                
                # 最小期待値以下のものは0に設定
                expected_values = torch.where(
                    expected_values > self.min_ev_threshold,
                    expected_values,
                    torch.zeros_like(expected_values)
                )
                
                # 正の期待値を持つベットの合計を計算
                positive_ev_sum = expected_values.sum(dim=1, keepdim=True)
                
                # ゼロ除算を避ける
                positive_ev_sum = torch.where(
                    positive_ev_sum > 0,
                    positive_ev_sum,
                    torch.ones_like(positive_ev_sum)
                )
                
                # 期待値に基づいて比率を計算
                bet_proportions[bet_type] = expected_values / positive_ev_sum
                
                # 1レース当たりの最大配分を制限
                race_allocation = torch.clamp(
                    bet_proportions[bet_type].sum(dim=1, keepdim=True),
                    max=self.max_allocation_per_race
                )
                
                # 各ベットの最大配分を制限
                bet_proportions[bet_type] = torch.clamp(
                    bet_proportions[bet_type],
                    max=self.max_allocation_per_bet
                )
                
                # 期待値情報を保存
                expected_values_by_type[bet_type] = expected_values
                
                logger.debug(f"{bet_type} - Calculated EV: min={expected_values.min().item():.4f}, max={expected_values.max().item():.4f}, "
                            f"positive_count={torch.sum(expected_values > 0).item()}, "
                            f"total_allocation={bet_proportions[bet_type].sum().item():.4f}")
            else:
                logger.debug(f"{bet_type} - 確率またはオッズデータがないためスキップ")
        
        # ベットタイプの優先度に基づいて調整
        # 複数のベットタイプが同じ組み合わせに賭けないようにする
        self._adjust_by_priority(bet_proportions, expected_values_by_type)
        
        return bet_proportions
    
    def _adjust_by_priority(self, bet_proportions: Dict[str, torch.Tensor], 
                           expected_values: Dict[str, torch.Tensor]) -> None:
        """ベットタイプの優先度に基づいて比率を調整する
        
        Args:
            bet_proportions: 各ベットタイプの購入比率
            expected_values: 各ベットタイプの期待値
        """
        # ベットタイプを優先度でソート
        sorted_types = sorted(
            [bet_type for bet_type in bet_proportions.keys()],
            key=lambda x: self.bet_type_priority.get(x, 999)
        )
        
        # 最も優先度の高いタイプから順に処理
        for i, high_priority_type in enumerate(sorted_types):
            if high_priority_type not in bet_proportions:
                continue
                
            high_priority_props = bet_proportions[high_priority_type]
            
            # より低い優先度のタイプを調整
            for low_priority_type in sorted_types[i+1:]:
                if low_priority_type not in bet_proportions:
                    continue
                
                # 特定のベットタイプ間の相関を処理
                if self._should_adjust(high_priority_type, low_priority_type):
                    # 重複する組み合わせを特定して調整
                    self._remove_overlapping_bets(
                        high_priority_type, low_priority_type,
                        bet_proportions, expected_values
                    )
    
    def _should_adjust(self, high_priority_type: str, low_priority_type: str) -> bool:
        """2つのベットタイプ間で調整が必要かどうかを判断
        
        Args:
            high_priority_type: 高優先度のベットタイプ
            low_priority_type: 低優先度のベットタイプ
            
        Returns:
            bool: 調整が必要な場合True
        """
        # 例: 三連単と三連複、馬単と馬連は重複するので調整が必要
        related_pairs = [
            {self.SANRENTAN, self.SANRENPUKU},
            {self.UMATAN, self.UMAREN},
            {self.TANSHO, self.FUKUSHO}
        ]
        
        return any({high_priority_type, low_priority_type} == pair for pair in related_pairs)
    
    def _remove_overlapping_bets(self, high_type: str, low_type: str,
                               bet_proportions: Dict[str, torch.Tensor],
                               expected_values: Dict[str, torch.Tensor]) -> None:
        """重複するベットを調整する
        
        高優先度のベットタイプで既に購入する組み合わせは、
        低優先度のベットタイプでは購入しない（または配分を下げる）
        
        Args:
            high_type: 高優先度のベットタイプ
            low_type: 低優先度のベットタイプ
            bet_proportions: 各ベットタイプの購入比率
            expected_values: 各ベットタイプの期待値
        """
        # 単純化のため、馬連と馬単の関係を例に実装
        if {high_type, low_type} == {self.UMATAN, self.UMAREN}:
            # 馬単と馬連の関係では、馬連で購入する組み合わせは馬単でカバーされている可能性がある
            # ここでは単純に低優先度の比率を下げる実装とする
            bet_proportions[low_type] *= 0.5
            logger.debug(f"Adjusted {low_type} proportions due to overlap with {high_type}")
            
        elif {high_type, low_type} == {self.SANRENTAN, self.SANRENPUKU}:
            # 三連単と三連複の関係
            bet_proportions[low_type] *= 0.3
            logger.debug(f"Adjusted {low_type} proportions due to overlap with {high_type}")
            
        elif {high_type, low_type} == {self.TANSHO, self.FUKUSHO}:
            # 単勝と複勝の関係
            bet_proportions[low_type] *= 0.7
            logger.debug(f"Adjusted {low_type} proportions due to overlap with {high_type}")