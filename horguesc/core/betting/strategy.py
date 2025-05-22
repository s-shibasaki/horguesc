import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

class BettingStrategy:
    """馬券購入戦略の基底クラス"""
    
    # 馬券種類の定数
    TANSHO = 'tansho'          # 単勝
    FUKUSHO = 'fukusho'        # 複勝
    WAKUREN = 'wakuren'        # 枠連
    UMAREN = 'umaren'          # 馬連
    WIDE = 'wide'              # ワイド
    UMATAN = 'umatan'          # 馬単
    SANRENPUKU = 'sanrenpuku'  # 三連複
    SANRENTAN = 'sanrentan'    # 三連単
    
    ALL_BET_TYPES = [TANSHO, FUKUSHO, WAKUREN, UMAREN, WIDE, UMATAN, SANRENPUKU, SANRENTAN]
    
    def __init__(self, config=None):
        """初期化
        
        Args:
            config: 設定情報を含むオブジェクト
        """
        self.config = config
        
        # 設定から戦略パラメータを読み込む
        self._load_parameters()
    
    def _load_parameters(self):
        """設定から戦略パラメータを読み込む"""
        if self.config:
            # 最大賭け金（レース毎）
            self.max_bet_per_race = self.config.getfloat(
                'betting.strategy', 'max_bet_per_race', fallback=10000.0)
            
            # 最小賭け金（組み合わせ毎）
            self.min_bet_per_combo = self.config.getfloat(
                'betting.strategy', 'min_bet_per_combo', fallback=100.0)
            
            # 許容される馬券種
            bet_types_str = self.config.get(
                'betting.strategy', 'allowed_bet_types', 
                fallback=','.join(self.ALL_BET_TYPES))
            self.allowed_bet_types = [bt.strip() for bt in bet_types_str.split(',')]
        else:
            # デフォルト値
            self.max_bet_per_race = 10000.0  # 1レース最大1万円
            self.min_bet_per_combo = 100.0   # 1組み合わせ最小100円
            self.allowed_bet_types = self.ALL_BET_TYPES
        
        logger.debug(f"戦略パラメータを設定: 最大賭け金={self.max_bet_per_race}, "
                   f"最小賭け金={self.min_bet_per_combo}, "
                   f"許容馬券種={self.allowed_bet_types}")
    
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
        raise NotImplementedError("サブクラスで実装する必要があります")
    
    def simulate(self, 
                 model_outputs: Dict[str, torch.Tensor],
                 odds_data: Dict[str, torch.Tensor], 
                 race_results: Dict[str, torch.Tensor],
                 race_ids: List[str] = None,
                 **kwargs) -> Dict[str, Any]:
        """戦略をシミュレーションして結果を返す
        
        Args:
            model_outputs: モデルの出力（各馬券種の予測確率を含む辞書）
            odds_data: オッズデータ（各馬券種のオッズを含む辞書）
            race_results: レース結果（各馬券種の的中情報を含む辞書）
            race_ids: シミュレーション対象のレースID（オプション）
            **kwargs: 追加のパラメータ
            
        Returns:
            dict: シミュレーション結果を含む辞書
        """
        # 購入金額を計算
        bets = self.calculate_bets(model_outputs, odds_data, **kwargs)
        
        # デバッグ情報をログに出力
        total_bet_amount = sum(torch.sum(bet).item() for bet in bets.values())
        logger.debug(f"総賭け金額: {total_bet_amount:,.0f} 円")  # INFO → DEBUG
        
        # 賭けがあるレースの数をログに出力
        bet_races = sum(1 for bet in bets.values() if torch.any(bet > 0))
        logger.debug(f"賭けがあるレース: {bet_races}/{len(next(iter(bets.values())))} ({bet_races/len(next(iter(bets.values())))*100:.1f}%)")  # INFO → DEBUG
        
        # オッズと確率の統計情報をログに出力
        for bet_type in bets.keys():
            odds_key = f'odds_{bet_type}'
            prob_key = f'{bet_type}_probabilities'
            
            if odds_key in odds_data and prob_key in model_outputs:
                odds = odds_data[odds_key]
                probs = model_outputs[prob_key]
                
                valid_odds = odds[odds > 0]
                valid_probs = probs[probs > 0]
                
                if len(valid_odds) > 0 and len(valid_probs) > 0:
                    logger.debug(f"{bet_type} - オッズ: 最小={valid_odds.min().item():.1f}, 最大={valid_odds.max().item():.1f}, " 
                               f"確率: 最小={valid_probs.min().item():.6f}, 最大={valid_probs.max().item():.6f}")  # INFO → DEBUG
    
        # 結果を計算
        results = self._calculate_results(bets, odds_data, race_results, race_ids)
        
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

    def to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """シミュレーション結果をDataFrameに変換する
        
        Args:
            results: simulate()の戻り値
            
        Returns:
            pd.DataFrame: 結果のDataFrame
        """
        if 'race_results' in results and results['race_results']:
            return pd.DataFrame(results['race_results'])
        else:
            # レースごとの結果がない場合は全体の結果だけのDataFrameを作成
            return pd.DataFrame([{
                'total_bet': results['total_bet'],
                'total_return': results['total_return'],
                'total_profit': results['total_profit'],
                'overall_roi': results['overall_roi'],
                'hit_races': results['hit_races'],
                'race_count': results['race_count'],
                'hit_rate': results['hit_rate']
            }])