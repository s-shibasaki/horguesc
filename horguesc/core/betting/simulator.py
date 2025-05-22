import torch
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Type
import matplotlib.pyplot as plt
from horguesc.core.betting.strategies.base import BaseStrategy
import os
from datetime import datetime
import random

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
            'betting.simulator', 'initial_capital', fallback=300000.0)
        
        # 馬券の最小購入単位（日本の馬券は100円単位）
        self.bet_unit = config.getfloat(
            'betting.simulator', 'bet_unit', fallback=100.0)
        
        # 馬券購入明細に含めるレース数の上限（デフォルト10件）
        self.max_bet_details_races = config.getint(
            'betting.simulator', 'max_bet_details_races', fallback=10)
        
        logger.debug("BettingSimulator初期化完了")

    def register_strategy(self, name: str, strategy: BaseStrategy) -> None:
        """戦略を登録する
        
        Args:
            name: 戦略の名前
            strategy: 戦略のインスタンス
        """
        self.strategies[name] = strategy
        logger.debug(f"戦略「{name}」を登録しました")

    def set_common_data(self, model_outputs: Dict[str, torch.Tensor], 
                       odds_data: Dict[str, torch.Tensor],
                       race_results: Dict[str, torch.Tensor],
                       race_ids: Optional[List[str]] = None,
                       **kwargs) -> None:
        """共通データをセットする
        
        Args:
            model_outputs: モデルの出力（各馬券種の予測確率を含む辞書）
            odds_data: オッズデータ（各馬券種のオッズを含む辞書）
            race_results: レース結果（各馬券種の的中情報を含む辞書）
            race_ids: レースID（レース識別子のリスト）
            **kwargs: 追加のデータ
        """
        self.common_data = {
            'model_outputs': model_outputs,
            'odds_data': odds_data,
            'race_results': race_results,
            'race_ids': race_ids,
            **kwargs
        }
        
        # レース数を確認
        race_count = next(iter(model_outputs.values())).shape[0]
        logger.info(f"共通データをセットしました: {race_count}レース")

    def add_default_strategies(self) -> None:
        """デフォルトの戦略を登録する"""
        try:
            # 戦略クラスをインポート
            from horguesc.core.betting.strategies.expected_value import ExpectedValueStrategy
            from horguesc.core.betting.strategies.odds_discrepancy import OddsDiscrepancyStrategy
            
            # 基本戦略インスタンスを作成して登録
            self.register_strategy("Expected Value Strategy", ExpectedValueStrategy(self.config))
            self.register_strategy("Odds Discrepancy Strategy", OddsDiscrepancyStrategy(self.config))

            logger.info("デフォルトの戦略を登録しました")
        except ImportError as e:
            logger.warning(f"デフォルト戦略の登録に失敗しました: {e}")
            raise
        except Exception as e:
            logger.error(f"戦略登録中にエラーが発生しました: {e}", exc_info=True)
            raise

    def simulate(self, capital_trend_path: Optional[str] = None, 
                betting_details_path: Optional[str] = None) -> Dict[str, Any]:
        """各戦略のシミュレーションを実行する
        
        Args:
            capital_trend_path: 資金推移グラフを保存するパス
            betting_details_path: 馬券購入明細を保存するパス
            
        Returns:
            dict: 戦略ごとのシミュレーション結果
        """
        if not self.strategies:
            logger.warning("登録された戦略がありません")
            return {}
            
        if not self.common_data:
            logger.warning("共通データがセットされていません")
            return {}
        
        # 共通データを取得
        model_outputs = self.common_data['model_outputs']
        odds_data = self.common_data['odds_data']
        race_results = self.common_data['race_results']
        race_ids = self.common_data.get('race_ids')
        
        # レース数を取得
        race_count = next(iter(model_outputs.values())).shape[0]
        
        # 馬券明細対象のレース選択（設定値に基づく件数制限）
        if betting_details_path and self.max_bet_details_races > 0 and race_count > self.max_bet_details_races:
            # 明細を出力するレースのインデックスをランダムに選択
            detail_race_indices = set(random.sample(range(race_count), min(self.max_bet_details_races, race_count)))
            logger.info(f"馬券購入明細は全{race_count}レース中{len(detail_race_indices)}レースのみ出力します")
        else:
            # すべてのレースの明細を出力
            detail_race_indices = set(range(race_count))
        
        # シミュレーション結果を保持する辞書
        results = {}
        
        # 全戦略の資金推移を記録するデータフレーム
        all_capital_trends = pd.DataFrame(index=range(race_count + 1))
        all_capital_trends.iloc[0] = self.initial_capital  # 初期資金
        
        # 全戦略の馬券購入明細を保持するデータフレームのリスト
        all_betting_details = []
        
        # デバイスを取得
        device = next(iter(model_outputs.values())).device
        
        # 各戦略についてシミュレーションを実行
        for strategy_name, strategy in self.strategies.items():
            logger.info(f"戦略「{strategy_name}」のシミュレーションを開始")
            
            # 馬券購入比率を計算
            bet_proportions = strategy.calculate_bet_proportions(model_outputs, odds_data)
            
            # 初期資金をテンソルとして準備 [race_count+1]
            capital_tensor = torch.full((race_count+1,), self.initial_capital, device=device)
            
            # 集計用のテンソルを初期化
            total_bets_per_race = torch.zeros(race_count, device=device)
            total_hits_per_race = torch.zeros(race_count, device=device)
            total_bet_amount_per_race = torch.zeros(race_count, device=device)
            total_return_amount_per_race = torch.zeros(race_count, device=device)
            
            # 馬券購入明細を保持するリスト
            bet_details = []
            
            # 各馬券種について処理
            for bet_type in BaseStrategy.ALL_BET_TYPES:
                if bet_type in bet_proportions:
                    # この馬券種の購入比率を取得
                    proportions = bet_proportions[bet_type]  # [race_count, n_combinations]
                    odds_key = f'odds_{bet_type}'
                    results_key = f'target_{bet_type}'
                    
                    if odds_key in odds_data and results_key in race_results:
                        type_odds = odds_data[odds_key]  # [race_count, n_combinations]
                        type_result = race_results[results_key]  # [race_count] or [race_count, n_combinations]
                        
                        # 購入金額を計算 (資金 × 購入比率を bet_unit 単位に切り捨て)
                        # [race_count, 1] * [race_count, n_combinations] -> [race_count, n_combinations]
                        capital_expanded = capital_tensor[:-1].unsqueeze(1)
                        raw_amounts = capital_expanded * proportions
                        
                        # bet_unit単位に切り捨て
                        amounts = (raw_amounts // self.bet_unit) * self.bet_unit
                        
                        # 有効な購入（金額 > 0）のマスク
                        valid_bets = amounts > 0
                        
                        # インデックス形式の的中情報か、2次元の的中情報かを判断
                        if type_result.dim() == 1:  # インデックス形式 (単勝、三連単など)
                            # [race_count, 1]
                            result_indices = type_result.unsqueeze(1)
                            
                            # 各レースで的中したベットのマスクを作成
                            # [race_count, n_combinations]
                            indices_range = torch.arange(proportions.size(1), device=device)
                            hit_mask = (indices_range.unsqueeze(0) == result_indices) & (result_indices >= 0)
                            
                            # 的中した購入のみを抽出
                            # [race_count, n_combinations]
                            valid_hits = hit_mask & valid_bets
                            
                            # 的中払戻金を計算
                            # [race_count, n_combinations]
                            returns = torch.zeros_like(amounts)
                            returns[valid_hits] = amounts[valid_hits] * type_odds[valid_hits]
                            
                        else:  # 2次元的中情報 (複勝、ワイドなど)
                            # 的中マスク [race_count, n_combinations]
                            hit_mask = type_result == 1
                            
                            # 的中した購入のみを抽出
                            valid_hits = hit_mask & valid_bets
                            
                            # 的中払戻金を計算
                            returns = torch.zeros_like(amounts)
                            returns[valid_hits] = amounts[valid_hits] * type_odds[valid_hits]
                        
                        # レースごとの集計
                        race_bet_counts = torch.sum(valid_bets.float(), dim=1)
                        race_hit_counts = torch.sum(valid_hits.float(), dim=1)
                        race_bet_amounts = torch.sum(amounts * valid_bets.float(), dim=1)
                        race_return_amounts = torch.sum(returns, dim=1)
                        
                        # 全体の集計に加算
                        total_bets_per_race += race_bet_counts
                        total_hits_per_race += race_hit_counts
                        total_bet_amount_per_race += race_bet_amounts
                        total_return_amount_per_race += race_return_amounts
                        
                        # 馬券購入明細を作成（対象レースのみ）
                        if betting_details_path:
                            for race_idx in range(race_count):
                                # 明細対象外のレースはスキップ
                                if race_idx not in detail_race_indices:
                                    continue
                                
                                race_id = race_ids[race_idx] if race_ids else f"Race_{race_idx}"
                                
                                for bet_idx in range(proportions.size(1)):
                                    if valid_bets[race_idx, bet_idx]:
                                        amount = amounts[race_idx, bet_idx].item()
                                        hit = False
                                        if type_result.dim() == 1:
                                            # インデックス形式
                                            hit = (type_result[race_idx].item() == bet_idx)
                                        else:
                                            # 2次元形式
                                            if bet_idx < type_result.size(1):
                                                hit = type_result[race_idx, bet_idx].item() == 1
                                                
                                        bet_details.append({
                                            'race_id': race_id,
                                            'race_idx': race_idx,
                                            'bet_type': bet_type,
                                            'bet_idx': bet_idx,
                                            'amount': amount,
                                            'odds': type_odds[race_idx, bet_idx].item(),
                                            'hit': hit,
                                            'return': amount * type_odds[race_idx, bet_idx].item() if hit else 0.0,
                                            'capital_before': capital_tensor[race_idx].item(),
                                        })
            
            # レースごとの資金推移を計算 (差分で計算)
            # 次のレースの資金 = 今のレースの資金 - 賭け金 + 払戻金
            capital_tensor[1:] = capital_tensor[0] - torch.cumsum(total_bet_amount_per_race, dim=0) + torch.cumsum(total_return_amount_per_race, dim=0)
            
            # 集計結果
            total_bet_amount = total_bet_amount_per_race.sum().item()
            total_return_amount = total_return_amount_per_race.sum().item()
            total_bets = total_bets_per_race.sum().item()
            total_hits = total_hits_per_race.sum().item()
            
            # 戦略の結果を記録
            results[strategy_name] = {
                'capital_trend': capital_tensor.tolist(),
                'final_capital': capital_tensor[-1].item(),
                'total_profit': capital_tensor[-1].item() - self.initial_capital,
                'total_bet': total_bet_amount,
                'total_return': total_return_amount,
                'overall_roi': (total_return_amount / total_bet_amount) - 1.0 if total_bet_amount > 0 else 0.0,
                'bet_count': total_bets,
                'hit_count': total_hits,
                'hit_rate': total_hits / total_bets if total_bets > 0 else 0.0,
                'bet_details': bet_details
            }
            
            # 資金推移を全体のデータフレームに追加
            all_capital_trends[strategy_name] = capital_tensor.cpu().numpy()
            
            # 馬券購入明細を全体のリストに追加
            if bet_details:
                strategy_details = pd.DataFrame(bet_details)
                strategy_details['strategy'] = strategy_name
                all_betting_details.append(strategy_details)
            
            logger.info(f"戦略「{strategy_name}」のシミュレーション完了: "
                        f"最終資金={capital_tensor[-1].item():,.0f}円, ROI={(results[strategy_name]['overall_roi']*100):.2f}%, "
                        f"的中率={results[strategy_name]['hit_rate']*100:.2f}%, "
                        f"明細出力レース数={len(detail_race_indices)}/{race_count}")
        
        # 資金推移グラフを保存
        if capital_trend_path:
            self._save_capital_trend_plot(all_capital_trends, capital_trend_path)
        
        # 馬券購入明細を保存
        if betting_details_path and all_betting_details:
            # 全戦略の馬券購入明細を結合
            combined_details = pd.concat(all_betting_details, ignore_index=True)
            
            # 拡張子からフォーマットを判断
            ext = os.path.splitext(betting_details_path)[1].lower()
            if ext == '.xlsx':
                combined_details.to_excel(betting_details_path, index=False)
            else:
                combined_details.to_csv(betting_details_path, index=False)
                
            logger.info(f"馬券購入明細を保存しました: {betting_details_path} ({len(combined_details)}件)")
        
        return results

    def _save_capital_trend_plot(self, capital_trends: pd.DataFrame, save_path: str) -> None:
        """資金推移グラフを保存する
        
        Args:
            capital_trends: 資金推移データフレーム
            save_path: 保存先のパス
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # レース数
            race_count = len(capital_trends) - 1
            
            # 各戦略の資金推移をプロット
            for strategy in capital_trends.columns:
                plt.plot(capital_trends.index, capital_trends[strategy], label=strategy)
            
            # グラフの設定
            plt.title('Capital Trend by Strategy')
            plt.xlabel('Race Number')
            plt.ylabel('Capital (JPY)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # 初期資金の水平線を追加
            plt.axhline(y=self.initial_capital, color='r', linestyle='-', alpha=0.3)
            
            # 指数表記を無効化
            plt.ticklabel_format(style='plain', axis='y')
            
            # Y軸のティックラベルを通常の数値形式でフォーマット
            from matplotlib.ticker import FuncFormatter
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
            
            # グラフを保存
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"資金推移グラフを保存しました: {save_path}")
        except Exception as e:
            logger.error(f"資金推移グラフの保存中にエラーが発生しました: {e}")

