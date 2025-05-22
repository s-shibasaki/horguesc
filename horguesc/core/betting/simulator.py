import torch
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Type
import matplotlib.pyplot as plt
from horguesc.core.betting.strategy import BettingStrategy
from horguesc.core.betting.ev_strategy import ExpectedValueStrategy
from horguesc.core.betting.prob_ratio_strategy import ProbabilityRatioStrategy
from horguesc.core.betting.kelly_strategy import KellyStrategy

logger = logging.getLogger(__name__)

class BettingSimulator:
    """馬券購入戦略のシミュレーションを行うクラス"""
    
    def __init__(self, config=None):
        """初期化
        
        Args:
            config: 設定情報を含むオブジェクト
        """
        self.config = config
        
        # 戦略のインスタンスを保持する辞書
        self.strategies = {}
        
        # 共通のメモリにあるモデル出力とオッズデータを保持
        self.common_data = {}
        
        logger.info("BettingSimulator初期化完了")
    
    def register_strategy(self, name: str, strategy: BettingStrategy) -> None:
        """戦略を登録する
        
        Args:
            name: 戦略の名前
            strategy: 戦略のインスタンス
        """
        self.strategies[name] = strategy
        logger.info(f"戦略「{name}」を登録しました")
    
    def add_default_strategies(self) -> None:
        """Add default strategies with relaxed thresholds for debugging"""
        # Create strategies with relaxed thresholds
        ev_strategy = ExpectedValueStrategy(self.config)
        self.register_strategy('expected_value', ev_strategy)
        
        ratio_strategy = ProbabilityRatioStrategy(self.config)
        self.register_strategy('probability_ratio', ratio_strategy)
        
        kelly_strategy = KellyStrategy(self.config)
        self.register_strategy('kelly', kelly_strategy)
        
        logger.info("Added default strategies with relaxed thresholds for debugging")
    
    def set_common_data(self, model_outputs: Dict[str, torch.Tensor],
                       odds_data: Dict[str, torch.Tensor],
                       race_results: Dict[str, torch.Tensor],
                       race_ids: List[str] = None) -> None:
        """共通データを設定する（メモリ効率のため）
        
        Args:
            model_outputs: モデルの出力
            odds_data: オッズデータ
            race_results: レース結果
            race_ids: レースID
        """
        self.common_data = {
            'model_outputs': model_outputs,
            'odds_data': odds_data,
            'race_results': race_results,
            'race_ids': race_ids
        }
        logger.debug(f"共通データを設定: {len(race_ids or [])}レース")  # INFO → DEBUG
    
    def simulate(self, strategy_names: Optional[List[str]] = None, 
                **kwargs) -> Dict[str, Dict[str, Any]]:
        """指定された戦略でシミュレーションを実行する
        
        Args:
            strategy_names: シミュレーションする戦略名のリスト（Noneの場合は全戦略）
            **kwargs: 追加のパラメータ
            
        Returns:
            dict: 戦略名をキーとするシミュレーション結果の辞書
        """
        if not self.common_data:
            logger.error("共通データが設定されていません")
            return {}
        
        # 指定がなければ全戦略を使用
        if strategy_names is None:
            strategy_names = list(self.strategies.keys())
        
        # シミュレーション結果を格納する辞書
        results = {}
        
        # 指定された戦略ごとにシミュレーション
        for name in strategy_names:
            if name in self.strategies:
                logger.info(f"戦略「{name}」でシミュレーションを開始")
                
                # 共通データを取得
                model_outputs = self.common_data['model_outputs']
                odds_data = self.common_data['odds_data']
                race_results = self.common_data['race_results']
                race_ids = self.common_data.get('race_ids')
                
                # シミュレーション実行
                strategy_result = self.strategies[name].simulate(
                    model_outputs, odds_data, race_results, race_ids, **kwargs)
                
                # 結果を保存
                results[name] = strategy_result
                
                logger.info(f"戦略「{name}」のシミュレーション完了: "
                           f"ROI={strategy_result['overall_roi']:.2%}, "
                           f"利益={strategy_result['total_profit']:,.0f}円")
            else:
                logger.warning(f"戦略「{name}」は登録されていません")
        
        return results
    
    def to_dataframe(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """シミュレーション結果をDataFrameに変換する
        
        Args:
            results: simulate()の戻り値
            
        Returns:
            pd.DataFrame: 結果の概要DataFrame
        """
        # 各戦略の結果を横に並べたDataFrameを作成
        summary_data = []
        
        for strategy_name, result in results.items():
            summary_data.append({
                'strategy': strategy_name,
                'total_bet': result['total_bet'],
                'total_return': result['total_return'],
                'total_profit': result['total_profit'],
                'roi': result['overall_roi'],
                'hit_races': result['hit_races'],
                'race_count': result['race_count'],
                'hit_rate': result['hit_rate']
            })
        
        return pd.DataFrame(summary_data)
    
    def visualize_results(self, results: Dict[str, Dict[str, Any]], 
                         figsize=(14, 10)) -> None:
        """シミュレーション結果を視覚化する
        
        Args:
            results: simulate()の戻り値
            figsize: グラフのサイズ
        """
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # 戦略名のリスト
        strategy_names = list(results.keys())
        
        # 1. ROIのグラフ
        roi_values = [results[name]['overall_roi'] * 100 for name in strategy_names]
        axs[0, 0].bar(strategy_names, roi_values)
        axs[0, 0].set_title('ROI (%)')
        axs[0, 0].set_ylabel('収益率 (%)')
        axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. 総利益のグラフ
        profit_values = [results[name]['total_profit'] for name in strategy_names]
        axs[0, 1].bar(strategy_names, profit_values)
        axs[0, 1].set_title('Total Profit')
        axs[0, 1].set_ylabel('利益 (円)')
        axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. 当たり率のグラフ
        hit_rates = [results[name]['hit_rate'] * 100 for name in strategy_names]
        axs[1, 0].bar(strategy_names, hit_rates)
        axs[1, 0].set_title('Hit Rate (%)')
        axs[1, 0].set_ylabel('当たり率 (%)')
        axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 4. 賭け金と払戻金のグラフ
        bets = [results[name]['total_bet'] for name in strategy_names]
        returns = [results[name]['total_return'] for name in strategy_names]
        
        width = 0.35
        x = np.arange(len(strategy_names))
        axs[1, 1].bar(x - width/2, bets, width, label='Total Bet')
        axs[1, 1].bar(x + width/2, returns, width, label='Total Return')
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(strategy_names)
        axs[1, 1].set_title('Bets vs. Returns')
        axs[1, 1].set_ylabel('金額 (円)')
        axs[1, 1].legend()
        axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()