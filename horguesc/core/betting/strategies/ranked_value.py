import torch
import logging
from typing import Dict, Any, Optional
import numpy as np
from horguesc.core.betting.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class RankedValueStrategy(BaseStrategy):
    """期待値と確率ランキングに基づく馬券購入戦略"""
    
    DEFAULT_MAX_BET_COUNTS = {
        BaseStrategy.TANSHO: 5,       # 単勝
        BaseStrategy.FUKUSHO: 5,      # 複勝
        BaseStrategy.WAKUREN: 5,      # 枠連
        BaseStrategy.UMAREN: 10,       # 馬連
        BaseStrategy.WIDE: 10,         # ワイド
        BaseStrategy.UMATAN: 10,       # 馬単
        BaseStrategy.SANRENPUKU: 10,   # 三連複
        BaseStrategy.SANRENTAN: 20,    # 三連単
    }

    def __init__(self, config=None):
        """初期化"""
        super().__init__(config)

        
        # 設定からパラメータを取得（デフォルト値付き）
        self.ev_threshold = config.getfloat('betting.strategy.ranked_value', 'ev_threshold', fallback=1.5)
        self.bet_unit = config.getint('betting.strategy.ranked_value', 'bet_unit', fallback=100)
        self.max_bet_count = {}
        for bet_type in self.ALL_BET_TYPES:
            self.max_bet_count[bet_type] = config.getint(
                'betting.strategy.ranked_value', f'max_bet_count_{bet_type}', fallback=self.DEFAULT_MAX_BET_COUNTS[bet_type]
            )

        logger.info(f"RankedValueStrategy initialized: ev_threshold={self.ev_threshold}, "
                    f"bet_unit={self.bet_unit}")

    def _calculate_bet_amounts_impl(self, 
                  model_outputs: Dict[str, torch.Tensor],
                  odds_data: Dict[str, torch.Tensor],
                  initial_capital: float = 300000.0,
                  **kwargs) -> Dict[str, torch.Tensor]:
        """
        モデル出力とオッズデータに基づいて馬券購入金額を計算する内部実装
        馬券種ごとに個別に処理する
        
        Args:
            model_outputs: モデルの出力（各馬券種の予測確率を含む辞書）
            odds_data: オッズデータ（各馬券種のオッズを含む辞書）
            initial_capital: 初期資金 (円単位)
            
        Returns:
            dict: 馬券種ごとの購入金額のテンソルを含む辞書
        """
        # 結果を格納する辞書
        result = {}
        
        # 各馬券種ごとに独立して処理
        for bet_type in self.ALL_BET_TYPES:
            prob_key = f"{bet_type}_probabilities"
            odds_key = f"odds_{bet_type}"
            
            # 馬券種のデータが揃っている場合のみ処理
            if prob_key in model_outputs and odds_key in odds_data:
                # 確率とオッズを取得
                probs = model_outputs[prob_key]  # [race_count, n_combinations]
                odds = odds_data[odds_key]       # [race_count, n_combinations]
                
                # NaNを0に置き換え
                probs = torch.nan_to_num(probs, nan=0.0)
                odds = torch.nan_to_num(odds, nan=0.0)
                
                # Debug logging
                logger.debug(f"Bet type: {bet_type}, Probs shape: {probs.shape}, Odds shape: {odds.shape}")
                
                # 期待値テンソルを計算: EV = probability * odds
                expected_values = probs * odds
                
                # 期待値が閾値以上の場所をマスクで示す
                ev_mask = expected_values >= self.ev_threshold
                
                # ev_maskがFalseの場所の確率を0にする
                filtered_probs = probs.clone()
                filtered_probs[~ev_mask] = 0.0
                
                # 各レースの確率を降順にソート
                sorted_indices = torch.argsort(filtered_probs, descending=True, dim=1)
                
                # オッズを確率と同じ順序に並べ替え
                sorted_ev = torch.gather(expected_values, 1, sorted_indices)
                
                # 確率のランク (1から始まる) を作成
                ranks = torch.arange(1, sorted_ev.shape[1] + 1, device=sorted_ev.device).unsqueeze(0)
                
                # 期待値が確率のランクを上回るかどうかを表すマスクを作成
                ev_exceeds_rank_mask = sorted_ev.cummin(dim=1).values > self.ev_threshold * ranks
                
                # 最初の連続したTrueの値のみを保持するマスクを作成
                first_valid_bets_mask = ((ev_exceeds_rank_mask.cumsum(dim=1) == ranks) & (ranks <= self.max_bet_count[bet_type])).float()

                # ソートされたインデックスを使って元の位置への逆マッピングを作成
                inverse_indices = torch.argsort(sorted_indices, dim=1)
                
                # 逆インデックスを使って元の順序に戻し、これが true の部分の馬券を購入する
                bet = torch.gather(first_valid_bets_mask, 1, inverse_indices)  # [レース数, n_combinations]

                result[bet_type] = bet * self.bet_unit  # [レース数, n_combinations] 
    
        # 有効な馬券種がない場合は空の辞書を返す
        if not result:
            logger.warning("有効な馬券データがありません。")
        
        return result