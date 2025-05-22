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
        
        logger.debug("BettingSimulator初期化完了")

    def register_strategy(self, name: str, strategy: BettingStrategy) -> None:
        """戦略を登録する
        
        Args:
            name: 戦略の名前
            strategy: 戦略のインスタンス
        """
        self.strategies[name] = strategy
        logger.debug(f"戦略「{name}」を登録しました")
    
    def add_default_strategies(self) -> None:
        """Add default strategies with relaxed thresholds for debugging"""
        # Create strategies with relaxed thresholds
        ev_strategy = ExpectedValueStrategy(self.config)
        self.register_strategy('expected_value', ev_strategy)
        
        ratio_strategy = ProbabilityRatioStrategy(self.config)
        self.register_strategy('probability_ratio', ratio_strategy)
        
        kelly_strategy = KellyStrategy(self.config)
        self.register_strategy('kelly', kelly_strategy)

        logger.debug("Added default strategies with relaxed thresholds for debugging")

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
                logger.debug(f"戦略「{name}」でシミュレーションを開始")
                
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
                           f"利益={strategy_result['total_profit']:,.0f}円, "
                           f"賭け金={strategy_result['total_bet']:,.0f}円, "
                           f"的中率={strategy_result['hit_rate']:.2%} "
                           f"({strategy_result['hit_races']}/{strategy_result['race_count']}), "
                           f"回収率={(strategy_result['total_return']/strategy_result['total_bet']):.2f}倍")
            else:
                logger.warning(f"戦略「{name}」は登録されていません")
        
        return results
    
    def save_capital_trend(self, results: Dict[str, Dict[str, Any]], 
                          output_path: str = 'capital_trend.png',
                          figsize=(12, 8),
                          initial_capital: float = 100000) -> None:
        """Visualize and save the capital trend over time for each strategy.
        
        Args:
            results: The simulation results from simulate()
            output_path: Path where the image will be saved
            figsize: Size of the figure (width, height)
            initial_capital: Starting capital amount
        """
        # Check if detailed race results are available
        has_race_details = all('race_results' in strategy_result and 
                              strategy_result['race_results'] is not None 
                              for strategy_result in results.values())
        
        if not has_race_details:
            logger.error("Cannot create capital trend: detailed race results are not available")
            return
        
        plt.figure(figsize=figsize)
        
        # Plot for each strategy
        for strategy_name, result in results.items():
            race_results = result['race_results']
            
            # Sort race results by race_id if available to ensure chronological order
            if all('race_id' in race for race in race_results):
                race_results = sorted(race_results, key=lambda x: x['race_id'])
            
            # Calculate cumulative capital over time
            capital = [initial_capital]
            for race in race_results:
                profit = race.get('profit', 0)
                capital.append(capital[-1] + profit)
            
            # Plot timeline (excluding initial capital point for x-axis alignment)
            x_values = range(len(race_results) + 1)
            plt.plot(x_values, capital, label=strategy_name, linewidth=2, marker='o', markersize=4)
        
        # Add reference line for initial capital
        plt.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7, 
                   label='Initial capital')
        
        # Add chart elements
        plt.title('Capital Trend Over Time', fontsize=16)
        plt.xlabel('Race Number', fontsize=12)
        plt.ylabel('Capital (JPY)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        # Format y-axis as currency
        plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        
        # Add annotations for final capital
        for strategy_name, result in results.items():
            race_results = result['race_results']
            final_capital = initial_capital + result['total_profit']
            plt.annotate(f'{final_capital:,.0f}', 
                        xy=(len(race_results), final_capital),
                        xytext=(10, 0), 
                        textcoords='offset points',
                        fontsize=10,
                        va='center')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Capital trend visualization saved to {output_path}")
        plt.close()