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
        logger.debug(f"共通データを設定: {len(race_ids or [])}レース")
    
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
        
        # 共通データを取得
        model_outputs = self.common_data['model_outputs']
        odds_data = self.common_data['odds_data']
        race_results = self.common_data['race_results']
        race_ids = self.common_data.get('race_ids')
        
        # 指定された戦略ごとにシミュレーション
        for name in strategy_names:
            if name in self.strategies:
                logger.debug(f"戦略「{name}」でシミュレーションを開始")
                
                # 購入金額を計算
                bets = self.strategies[name].calculate_bets(model_outputs, odds_data, **kwargs)
                
                # デバッグ情報をログに出力
                total_bet_amount = sum(torch.sum(bet).item() for bet in bets.values())
                logger.debug(f"総賭け金額: {total_bet_amount:,.0f} 円")
                
                # 賭けがあるレースの数をログに出力
                bet_races = sum(1 for bet in bets.values() if torch.any(bet > 0))
                logger.debug(f"賭けがあるレース: {bet_races}/{len(next(iter(bets.values())))} ({bet_races/len(next(iter(bets.values())))*100:.1f}%)")
                
                # 結果を計算
                strategy_result = self._calculate_results(
                    bets, odds_data, race_results, race_ids)
                
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
    
    def _calculate_results(self, 
        bets: Dict[str, torch.Tensor],
        odds_data: Dict[str, torch.Tensor], 
        race_results: Dict[str, torch.Tensor],
        race_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """馬券の購入結果を計算する
        
        Args:
            bets: 馬券種ごとの購入金額
            odds_data: オッズデータ
            race_results: レース結果（的中情報）
            race_ids: レースID（オプション）
            
        Returns:
            dict: シミュレーション結果の詳細
        """
        device = list(bets.values())[0].device
        batch_size = list(bets.values())[0].shape[0]
        
        # Debug outputs to identify data issues
        logger.debug(f"_calculate_results - Device: {device}, Batch size: {batch_size}")  # INFO → DEBUG
        logger.debug(f"Available bet types: {list(bets.keys())}")  # INFO → DEBUG
        logger.debug(f"Available odds data: {list(odds_data.keys())}")  # INFO → DEBUG
        logger.debug(f"Available race results: {list(race_results.keys())}")  # INFO → DEBUG
        
        # 集計用の変数を初期化
        total_bets = torch.zeros(batch_size, device=device)
        total_returns = torch.zeros(batch_size, device=device)
        hit_races = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Process odds data to replace NaN values with zeros
        processed_odds_data = {}
        for key, odds in odds_data.items():
            # Replace NaN with 0 and convert to the correct device
            processed_odds = torch.nan_to_num(odds, nan=0.0).to(device)
            processed_odds_data[key] = processed_odds
            
            # Log preprocessing results
            logger.debug(f"Processed {key}: min={processed_odds.min().item():.1f}, max={processed_odds.max().item():.1f}, nan_count={torch.isnan(odds).sum().item()}")  # INFO → DEBUG
    
        # 各馬券種ごとに集計
        bet_details = {}
        for bet_type in bets.keys():  # Only iterate over bet types we actually have bets for
            odds_key = f'odds_{bet_type}'
            if odds_key not in processed_odds_data:
                logger.warning(f"{bet_type}のオッズデータがありません。スキップします。")
                continue
                
            # 購入金額とオッズを取得
            bet_amounts = bets[bet_type]
            odds = processed_odds_data[odds_key]
            
            # デバッグ: 購入金額とオッズの状態を確認
            logger.debug(f"{bet_type} - Bet amounts shape: {bet_amounts.shape}, sum: {bet_amounts.sum().item()}, " 
                    f"min: {bet_amounts.min().item()}, max: {bet_amounts.max().item()}")  # INFO → DEBUG
            logger.debug(f"{bet_type} - Odds shape: {odds.shape}, " 
                    f"min: {odds.min().item()}, max: {odds.max().item()}")  # INFO → DEBUG
        
            # 的中情報を取得
            target_key = f'target_{bet_type}'
            if target_key not in race_results:
                logger.warning(f"{bet_type}の的中情報を取得できませんでした。スキップします。")
                continue
            
            target = race_results[target_key].to(device)
            logger.debug(f"Processing {bet_type} - Target shape: {target.shape}, Target min: {target.min().item()}, Target max: {target.max().item()}")  # INFO → DEBUG
            
            # 的中情報の形状に基づいて処理を分岐
            if target.dim() == 1:
                # 1次元の場合（単勝、馬単、馬連、枠連、三連複、三連単など）
                # インデックスに基づいて的中馬券の払戻金を計算
                indices = torch.arange(batch_size, device=device)
                # 有効な的中のみを処理（-1などの無効値をフィルタリング）
                valid_mask = target >= 0
                # レース毎に的中した馬券の払戻金を計算
                returns = torch.zeros(batch_size, device=device)
                if valid_mask.any():
                    valid_indices = indices[valid_mask]
                    valid_targets = target[valid_mask]
                    
                    # デバッグ: インデックス範囲を確認
                    logger.debug(f"{bet_type} - Valid indices count: {len(valid_indices)}")  # INFO → DEBUG
                    logger.debug(f"{bet_type} - Valid targets min: {valid_targets.min().item()}, max: {valid_targets.max().item()}")  # INFO → DEBUG
                    logger.debug(f"{bet_type} - Bet amounts shape for valid indices: {bet_amounts[valid_mask].shape}")  # INFO → DEBUG
                    logger.debug(f"{bet_type} - Odds shape for valid indices: {odds[valid_mask].shape}")  # INFO → DEBUG
                    
                    # インデックス範囲のチェックを追加
                    max_target = valid_targets.max().item()
                    max_allowed = odds[valid_mask].shape[1] - 1
                    if max_target > max_allowed:
                        logger.error(f"{bet_type} - Invalid target index detected: max={max_target}, allowed={max_allowed}")
                        # 範囲外のインデックスを修正
                        valid_targets = torch.clamp(valid_targets, 0, max_allowed)
                    
                    try:
                        # 範囲チェック後に計算を試みる
                        valid_returns = (bet_amounts[valid_mask] * odds[valid_mask])[torch.arange(len(valid_indices), device=device), valid_targets]
                        returns[valid_mask] = valid_returns
                        
                        # デバッグ: 計算された払戻を確認
                        logger.debug(f"{bet_type} - Returns calculated: sum={returns.sum().item()}, " 
                                f"min={returns[returns > 0].min().item() if returns.gt(0).any() else 0}, " 
                                f"max={returns.max().item()}")  # INFO → DEBUG
                        
                    except IndexError as e:
                        logger.error(f"{bet_type} - IndexError during return calculation: {e}")
                        # インデックスエラーが発生した場合、計算をスキップ
                        continue
                
                # レースごとの合計を集計
                race_bet = bet_amounts.sum(dim=1)
                race_return = returns
                
                # 的中フラグを計算
                hits = valid_mask & (returns > 0)
                
            else:
                # 多次元の場合（複勝、ワイドなど、複数の的中がある場合）
                # バイナリマスク形式なので、的中した組み合わせの払戻金を合計
                hits = target.bool()
                
                # デバッグ: 形状の一致を確認
                logger.debug(f"{bet_type} - Hit mask shape: {hits.shape}, hits count: {hits.sum().item()}")  # INFO → DEBUG
                
                try:
                    returns = bet_amounts * odds * hits
                    
                    # デバッグ: 計算された払戻を確認
                    logger.debug(f"{bet_type} - Returns shape: {returns.shape}, sum: {returns.sum().item()}")  # INFO → DEBUG
                    
                    # レースごとの合計を集計
                    race_bet = bet_amounts.sum(dim=1)
                    race_return = returns.sum(dim=1)
                    
                    # デバッグ: レース毎の払戻を確認
                    logger.debug(f"{bet_type} - Race bets: sum={race_bet.sum().item()}")  # INFO → DEBUG
                    logger.debug(f"{bet_type} - Race returns: sum={race_return.sum().item()}")  # INFO → DEBUG
                    
                    # 的中フラグを計算（いずれかが的中していればTrue）
                    hits = hits.any(dim=1)
                
                except RuntimeError as e:
                    logger.error(f"{bet_type} - RuntimeError during calculation: {e}")
                    # エラーが発生したら次へ
                    continue
        
        # 全体の集計に加算
        total_bets += race_bet
        total_returns += race_return
        hit_races |= hits
        
        # 馬券種ごとの詳細を保存
        bet_details[bet_type] = {
            'bet': race_bet,
            'return': race_return,
            'hit': hits,
            'profit': race_return - race_bet,
            'roi': torch.where(race_bet > 0, (total_returns / total_bets) - 1.0, torch.tensor(0.0, device=device))
        }
    
        # 全体の集計結果を計算
        profit = total_returns - total_bets
        
        # デバッグ: 最終的な集計結果を確認
        logger.debug(f"Final totals - Total bets: {total_bets.sum().item()}, Total returns: {total_returns.sum().item()}")  # INFO → DEBUG
        
        # ROI計算時のゼロ除算を避ける
        if total_bets.sum() > 0:
            overall_roi = (total_returns.sum() / total_bets.sum() - 1.0).item()
            logger.debug(f"Overall ROI calculated: {overall_roi:.4f}")  # INFO → DEBUG
        else:
            overall_roi = 0.0
            logger.warning("Zero total bets, setting ROI to 0")
        
        roi = torch.where(total_bets > 0, (total_returns / total_bets) - 1.0, torch.tensor(0.0, device=device))
        
        # レースIDがある場合は結果と紐付ける
        race_level_results = None
        if race_ids is not None:
            race_level_results = []
            for i in range(batch_size):
                race_detail = {
                    'race_id': race_ids[i],
                    'bet': total_bets[i].item(),
                    'return': total_returns[i].item(),
                    'profit': profit[i].item(),
                    'roi': roi[i].item(),
                    'hit': hit_races[i].item()
                }
                for bet_type, detail in bet_details.items():
                    race_detail[f'{bet_type}_bet'] = detail['bet'][i].item()
                    race_detail[f'{bet_type}_return'] = detail['return'][i].item()
                    race_detail[f'{bet_type}_hit'] = detail['hit'][i].item()
                race_level_results.append(race_detail)
        
        # 全体の結果をまとめる
        results = {
            'total_bet': total_bets.sum().item(),
            'total_return': total_returns.sum().item(),
            'total_profit': profit.sum().item(),
            'overall_roi': overall_roi,
            'hit_races': hit_races.sum().item(),
            'race_count': batch_size,
            'hit_rate': (hit_races.sum() / batch_size).item(),
            'bet_details': bet_details,
            'race_results': race_level_results
        }
        
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
        has_race_details = all('race_results' in strategy_result for strategy_result in results.values())
        
        if not has_race_details:
            logger.error("Cannot create capital trend: detailed race results are not available")
            return
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.figure(figsize=figsize)
        
        # Plot for each strategy
        for strategy_name, result in results.items():
            if 'race_results' not in result or result['race_results'] is None:
                logger.warning(f"Strategy {strategy_name} has no race results, skipping in visualization")
                continue
                
            race_results = result['race_results']
            
            # Sort race results by race_id if available to ensure chronological order
            if isinstance(race_results, list) and all(isinstance(race, dict) and 'race_id' in race for race in race_results):
                race_results = sorted(race_results, key=lambda x: x['race_id'])
            
            # Calculate cumulative capital over time
            capital = [initial_capital]
            race_profits = []
            
            # Extract profit data from race results
            if isinstance(race_results, list) and all(isinstance(race, dict) for race in race_results):
                # List of dictionaries format
                for race in race_results:
                    profit = race.get('profit', 0)
                    race_profits.append(profit)
            else:
                # Extract profit from the overall result
                total_profit = result.get('total_profit', 0)
                race_count = result.get('race_count', 1) 
                # Distribute total profit evenly (simple approximation if detailed data is missing)
                race_profits = [total_profit / race_count] * race_count
            
            # Calculate cumulative capital
            for profit in race_profits:
                capital.append(capital[-1] + profit)
            
            # Plot timeline (excluding initial capital point for x-axis alignment)
            x_values = range(len(capital))
            plt.plot(x_values, capital, label=strategy_name, linewidth=2, marker='o', markersize=4)
        
        # Add reference line for initial capital
        plt.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7, 
                   label='Initial capital')
        
        # Add timestamp to the title
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plt.title(f'Capital Trend Over Time ({current_time})', fontsize=16)
        plt.xlabel('Race Number', fontsize=12)
        plt.ylabel('Capital (JPY)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        # Format y-axis as currency
        plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        
        # Add annotations for final capital
        for strategy_name, result in results.items():
            if 'race_results' not in result or result['race_results'] is None:
                continue
                
            race_results = result['race_results']
            final_capital = initial_capital + result['total_profit']
            
            # Calculate number of races for x-coordinate
            race_count = 0
            if isinstance(race_results, list):
                race_count = len(race_results)
            elif 'race_count' in result:
                race_count = result['race_count']
                
            plt.annotate(f'{final_capital:,.0f}', 
                        xy=(race_count, final_capital),
                        xytext=(10, 0), 
                        textcoords='offset points',
                        fontsize=10,
                        va='center')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Capital trend visualization saved to {output_path}")
        plt.close()