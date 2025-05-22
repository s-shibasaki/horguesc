import torch
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Type
import matplotlib.pyplot as plt
from horguesc.core.betting.strategy import BettingStrategy
import os
from datetime import datetime

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
        
        # シミュレーション時の初期資金（デフォルト値）
        self.initial_capital = config.getfloat(
            'betting.simulator', 'initial_capital', fallback=100000.0)
        
        # 馬券の最小購入単位（日本の馬券は100円単位）
        self.bet_unit = config.getfloat(
            'betting.simulator', 'bet_unit', fallback=100.0)
        
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
        """デフォルトの戦略を登録する"""
        # Import the strategy
        from horguesc.core.betting.strategies.expected_value import ExpectedValueStrategy
        
        # Create and register the expected value strategy
        ev_strategy = ExpectedValueStrategy(self.config)
        self.register_strategy('expected_value', ev_strategy)
        
        logger.debug("デフォルト戦略を登録しました")

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
        logger.debug(f"共通データを設定: {len(race_ids or [])}レース")
    
    def simulate(self, strategy_names: Optional[List[str]] = None, 
                initial_capital: Optional[float] = None,
                capital_trend_path: Optional[str] = None,
                **kwargs) -> Dict[str, Dict[str, Any]]:
        """指定された戦略でシミュレーションを実行する
    
        Args:
            strategy_names: シミュレーションする戦略名のリスト（Noneの場合は全戦略）
            initial_capital: シミュレーションの初期資金（指定がない場合はデフォルト値を使用）
            capital_trend_path: 資金推移のグラフを保存するパス（Noneの場合は保存しない）
            **kwargs: 追加のパラメータ
        
        Returns:
            dict: 戦略名をキーとするシミュレーション結果の辞書
        """
        if not self.common_data:
            logger.error("共通データが設定されていません")
            return {}
        
        # 初期資金を設定（引数で上書き可能）
        if initial_capital is None:
            initial_capital = self.initial_capital
            
        # 指定がなければ全戦略を使用
        if strategy_names is None:
            strategy_names = list(self.strategies.keys())
        
        # シミュレーション結果を格納する辞書
        results = {}
        
        # 共通データを取得
        model_outputs = self.common_data['model_outputs']
        odds_data = self.common_data['odds_data']
        race_results = self.common_data['race_results']
        race_ids = self.common_data.get('race_ids', None)
        
        # NaN値を含むオッズデータを前処理（NaN値を0に変換）
        cleaned_odds_data = {}
        for key, odds in odds_data.items():
            if 'odds_' in key:
                # NaNを0に変換
                cleaned_odds_data[key] = torch.nan_to_num(odds, nan=0.0)
            else:
                cleaned_odds_data[key] = odds
        
        # 処理済みオッズデータを使用
        odds_data = cleaned_odds_data
        
        # レース数を取得
        batch_size = next(iter(model_outputs.values())).shape[0]
        device = next(iter(model_outputs.values())).device
        
        # 指定された戦略ごとにシミュレーション
        for name in strategy_names:
            if name not in self.strategies:
                logger.warning(f"戦略「{name}」は登録されていません")
                continue
                
            logger.debug(f"戦略「{name}」でシミュレーションを開始")
            strategy = self.strategies[name]
            
            # 購入比率を計算
            bet_proportions = strategy.calculate_bet_proportions(model_outputs, odds_data, **kwargs)
            
            # 集計用の変数を初期化
            capital_history = [torch.full((batch_size,), initial_capital, device=device)]
            total_bet_amount = torch.zeros(batch_size, device=device)
            total_return_amount = torch.zeros(batch_size, device=device)
            hit_races = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            # 馬券種別ごとの集計データ
            bet_type_stats = {bet_type: {
                'total_bet': torch.zeros(batch_size, device=device),
                'total_return': torch.zeros(batch_size, device=device),
                'total_tickets': torch.zeros(batch_size, device=device),
                'hit_tickets': torch.zeros(batch_size, device=device)
            } for bet_type in bet_proportions.keys()}
            
            # レース毎の詳細結果
            race_details = []
            
            # レースごとにループ処理
            for race_idx in range(batch_size):
                current_capital = capital_history[-1][race_idx].item()
                race_id = race_ids[race_idx] if race_ids else f"race_{race_idx}"
                
                # このレースの購入情報
                race_bets = {}
                race_bet_amount = 0
                race_return_amount = 0
                race_hit = False
                
                # 馬券種ごとの処理
                for bet_type, proportions in bet_proportions.items():
                    # このレースのこの馬券種の購入比率
                    race_proportion = proportions[race_idx]
                    
                    # 購入金額を計算（100円単位で切り捨て）
                    raw_amounts = current_capital * race_proportion
                    unit_amounts = (raw_amounts / self.bet_unit).floor() * self.bet_unit
                    
                    # 実際の購入金額を格納
                    race_bets[bet_type] = unit_amounts
                    
                    # このレースのこの馬券種の合計購入金額
                    bet_amount = unit_amounts.sum().item()
                    race_bet_amount += bet_amount
                    
                    # 購入枚数をカウント
                    tickets_count = (unit_amounts > 0).sum().item()
                    bet_type_stats[bet_type]['total_tickets'][race_idx] += tickets_count
                    
                    # 的中計算
                    odds_key = f'odds_{bet_type}'
                    target_key = f'target_{bet_type}'
                    
                    if odds_key in odds_data and target_key in race_results:
                        odds = odds_data[odds_key][race_idx]
                        target = race_results[target_key][race_idx]
                        
                        # 的中計算（的中情報の形状による分岐）
                        if isinstance(target, torch.Tensor) and target.dim() == 0:
                            # 単一の的中（単勝、馬単など）
                            if target >= 0 and target < len(odds):  # 有効な的中
                                return_amount = unit_amounts[target].item() * odds[target].item()
                                hit = unit_amounts[target].item() > 0  # 購入していれば的中
                                
                                # 的中した馬券数をカウント
                                if hit:
                                    bet_type_stats[bet_type]['hit_tickets'][race_idx] += 1
                            else:
                                return_amount = 0
                                hit = False
                        else:
                            # 複数的中の可能性（複勝、ワイドなど）
                            hit_mask = target.bool() if isinstance(target, torch.Tensor) else torch.zeros_like(unit_amounts, dtype=torch.bool)
                            
                            # NaN値を考慮した的中計算
                            hit_odds = torch.where(hit_mask, odds, torch.zeros_like(odds))
                            return_amount = (unit_amounts * hit_odds).sum().item()
                            hit = (hit_mask & (unit_amounts > 0)).any().item()
                            
                            # 的中した馬券数をカウント
                            if hit:
                                bet_type_stats[bet_type]['hit_tickets'][race_idx] += (hit_mask & (unit_amounts > 0)).sum().item()
                        
                        # 集計に追加
                        race_return_amount += return_amount
                        race_hit |= hit
                        
                        # 馬券種別の統計を更新
                        bet_type_stats[bet_type]['total_bet'][race_idx] += bet_amount
                        bet_type_stats[bet_type]['total_return'][race_idx] += return_amount
                        
                    else:
                        logger.warning(f"レース {race_id} の {bet_type} のオッズまたは的中情報がありません")
                
                # 資金を更新
                updated_capital = current_capital - race_bet_amount + race_return_amount
                new_capital = torch.clone(capital_history[-1])
                new_capital[race_idx] = updated_capital
                capital_history.append(new_capital)
                
                # レース全体の統計を更新
                total_bet_amount[race_idx] = race_bet_amount
                total_return_amount[race_idx] = race_return_amount
                hit_races[race_idx] = race_hit
                
                # レース詳細を記録
                race_detail = {
                    'race_id': race_id,
                    'bet': race_bet_amount,
                    'return': race_return_amount,
                    'profit': race_return_amount - race_bet_amount,
                    'roi': (race_return_amount / race_bet_amount - 1) if race_bet_amount > 0 else 0,
                    'hit': race_hit,
                    'capital_before': current_capital,
                    'capital_after': updated_capital
                }
                
                # 馬券種別の詳細も追加
                for bet_type in bet_proportions.keys():
                    race_detail[f'{bet_type}_bet'] = bet_type_stats[bet_type]['total_bet'][race_idx].item()
                    race_detail[f'{bet_type}_return'] = bet_type_stats[bet_type]['total_return'][race_idx].item()
                    race_detail[f'{bet_type}_hit'] = (bet_type_stats[bet_type]['hit_tickets'][race_idx] > 0).item()
                
                race_details.append(race_detail)
            
            # 全体の集計結果を計算
            total_bet = total_bet_amount.sum().item()
            total_return = total_return_amount.sum().item()
            profit = total_return - total_bet
            
            # 0除算を回避
            overall_roi = (total_return / total_bet - 1) if total_bet > 0 else 0
            hit_count = hit_races.sum().item()
            hit_rate = hit_count / batch_size if batch_size > 0 else 0
            
            # 馬券種別の統計で0除算を回避
            bet_type_details = {}
            for bet_type, stats in bet_type_stats.items():
                bet_sum = stats['total_bet'].sum().item()
                return_sum = stats['total_return'].sum().item()
                tickets_sum = stats['total_tickets'].sum().item()
                
                bet_type_details[bet_type] = {
                    'bet': bet_sum,
                    'return': return_sum,
                    'profit': return_sum - bet_sum,
                    'roi': (return_sum / bet_sum - 1) if bet_sum > 0 else 0,
                    'tickets': tickets_sum,
                    'hit_tickets': stats['hit_tickets'].sum().item(),
                    'hit_rate': (stats['hit_tickets'].sum() / tickets_sum).item() if tickets_sum > 0 else 0
                }
            
            # 結果をまとめる
            strategy_result = {
                'total_bet': total_bet,
                'total_return': total_return,
                'total_profit': profit,
                'overall_roi': overall_roi,
                'hit_races': hit_count,
                'race_count': batch_size,
                'hit_rate': hit_rate,
                'bet_details': bet_type_details,
                'race_results': race_details,
                'capital_history': [cap.cpu().numpy() for cap in capital_history],
                'initial_capital': initial_capital,
                'final_capital': capital_history[-1].mean().item()
            }
            
            # 結果を保存
            results[name] = strategy_result
            
            # NaNを回避するためのフォーマット
            return_rate = strategy_result['total_return']/strategy_result['total_bet'] if strategy_result['total_bet'] > 0 else 0
            
            logger.info(f"戦略「{name}」のシミュレーション完了: "
                      f"ROI={strategy_result['overall_roi']:.2%}, "
                      f"利益={strategy_result['total_profit']:,.0f}円, "
                      f"賭け金={strategy_result['total_bet']:,.0f}円, "
                      f"的中率={strategy_result['hit_rate']:.2%} "
                      f"({strategy_result['hit_races']}/{strategy_result['race_count']}), "
                      f"回収率={return_rate:.2f}倍")
    
        # 資金推移のグラフを保存（パスが指定されている場合）
        if capital_trend_path and results:
            try:
                self.save_capital_trend(results, output_path=capital_trend_path, initial_capital=initial_capital)
                logger.info(f"資金推移グラフを保存しました: {capital_trend_path}")
            except Exception as e:
                logger.error(f"資金推移グラフの保存に失敗しました: {e}")

        return results
    
    def save_capital_trend(self, simulation_results: Dict[str, Dict[str, Any]], 
                         output_path: str, initial_capital: float = None) -> None:
        """Save a graph of capital trends for different strategies

        Args:
            simulation_results: Dictionary with strategy names as keys and simulation results as values
            output_path: Path where to save the graph
            initial_capital: Initial capital value (uses the value from simulation results if None)
        """
        plt.figure(figsize=(12, 8))
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Keep track of the final capital for each strategy for the legend
        final_capitals = {}
        
        for strategy_name, result in simulation_results.items():
            capital_history = result['capital_history']
            
            # Convert capital history to average if it's per race
            if isinstance(capital_history[0], np.ndarray):
                capital_trend = [np.mean(cap) for cap in capital_history]
            else:
                capital_trend = capital_history
            
            # Plot the capital trend for this strategy
            x_values = range(len(capital_trend))
            plt.plot(x_values, capital_trend, label=f"{strategy_name}", linewidth=2)
            
            # Record final capital for legend
            final_capital = capital_trend[-1]
            final_capitals[strategy_name] = final_capital
            
        # Add a horizontal line for the initial capital
        if initial_capital is None:
            # Use the initial capital from the first strategy
            initial_capital = next(iter(simulation_results.values()))['initial_capital']
        
        plt.axhline(y=initial_capital, color='gray', linestyle='--', linewidth=1, label=None)
        
        # Add annotations to the graph
        plt.title("Capital Trends by Strategy", fontsize=16)
        plt.xlabel("Race Number", fontsize=14)
        plt.ylabel("Capital (JPY)", fontsize=14)
        
        # Create legend with final capital values
        legend_labels = [f"{name} (Final: ¥{final_capitals[name]:,.0f})" for name in simulation_results.keys()]
        plt.legend(legend_labels, loc='best', fontsize=12)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.figtext(0.02, 0.02, f"Generated: {timestamp}", fontsize=8)
        
        # Format y-axis with commas for thousands
        plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.debug(f"Capital trend graph saved to {output_path}")

