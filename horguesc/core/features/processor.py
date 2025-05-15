import logging
import numpy as np
import torch
from collections import defaultdict
import pandas as pd

logger = logging.getLogger(__name__)

class FeatureProcessor:
    """特徴量の前処理を一元管理するクラス"""
    
    # 1. 初期化
    def __init__(self, config):
        """
        Args:
            config: アプリケーションの設定
        """
        self.config = config 
        
        # 特徴量のグループ情報を取得
        self.feature_groups = config.feature_groups
        
        # グループごとのエンコーダと正規化パラメータを追跡
        self.numerical_parameters = {}
        self.categorical_encoders = {}
        
        # 数値特徴量のグループごとの値
        self.group_numerical_values = defaultdict(list)
        # カテゴリカル特徴量のグループごとの観測値
        self.group_observed_values = defaultdict(set)
    
    # 2. 公開API（ライフサイクル順）
    def collect_values_for_fitting(self, data_source):
        """データソースから特徴量の値を収集します
    
        Args:
            data_source: numpy配列を含む辞書
    
        Raises:
            TypeError: 特徴量の値がnumpy配列でない場合
        """
        for feature_name, values in data_source.items():
            if not isinstance(values, np.ndarray):
                raise TypeError(f"特徴量 {feature_name} はnumpy配列(np.ndarray)である必要があります。"
                               f"提供された型: {type(values)}")
            
            # 多次元配列をフラット化する
            flat_values = values.flatten() if values.ndim > 1 else values
                
            if feature_name in self.config.categorical_features:
                # この特徴量が属するグループを取得
                group_name = self.feature_groups['categorical'][feature_name]
                self.group_observed_values[group_name].update(flat_values)
                
            # 数値特徴量の値も収集
            elif feature_name in self.config.numerical_features:
                # この特徴量が属するグループを取得
                group_name = self.feature_groups['numerical'][feature_name]
                
                # NaNではない値だけを収集
                valid_values = flat_values[~np.isnan(flat_values)]
                self.group_numerical_values[group_name].extend(valid_values)
                
                # ログに記録
                invalid_count = len(flat_values) - len(valid_values)
                if invalid_count > 0:
                    logger.debug(f"特徴量 {feature_name} に {invalid_count} 個のNaN値が含まれています")

    def fit(self, clear_after_fit=True):
        """カテゴリカル特徴量のエンコーダと数値特徴量の正規化パラメータをフィットします。
        
        Args:
            clear_after_fit: フィット後に収集された値をクリアするかどうか
        """
        logger.info("特徴量のエンコーダと正規化パラメータをフィットしています...")
        
        # 内部メソッドを呼び出し
        self._fit_categorical_encoders()
        self._fit_numerical_parameters()
        
        # フィット後のデータクリア（オプション）
        if clear_after_fit:
            self._clear_collected_values()
            
        logger.info("特徴量処理パラメータのフィットが完了しました")
        
        return self

    def encode_categorical_feature(self, name, values):
        """カテゴリカル特徴量をエンコーディングします。
        
        Args:
            name: 特徴量の名前
            values: エンコーディングする値のnumpy配列、多次元配列も可
            
        Returns:
            torch.Tensor: エンコーディングされたインデックス（入力と同じ形状）
        """
        # 入力型をチェック
        if not isinstance(values, np.ndarray):
            raise TypeError(f"特徴量 {name} の値はnumpy配列(np.ndarray)である必要があります。"
                        f"提供された型: {type(values)}")
        
        # 元の形状を保存
        original_shape = values.shape
        values = values.flatten()
        
        # この特徴量がどのグループに属しているか確認
        group_name = self.feature_groups['categorical'][name]
        
        # グループのエンコーダが存在することを確認
        if group_name not in self.categorical_encoders:
            raise ValueError(f"グループ {group_name} のエンコーダが存在しません")
        
        encoder = self.categorical_encoders[group_name]
        
        # 値をインデックスに変換（不明な値は0にマッピング）
        indices = [encoder.get(val, 0) for val in values]
        
        # 多次元だった場合は元の形状に戻す
        if len(original_shape) > 1:
            return torch.tensor(indices, dtype=torch.long).reshape(original_shape)
        else:
            return torch.tensor(indices, dtype=torch.long)

    def normalize_numerical_feature(self, name, values):
        """数値特徴量を正規化します。
        
        Args:
            name: 特徴量の名前
            values: 正規化する値のnumpy配列、多次元配列も可
            
        Returns:
            torch.Tensor: 正規化された値（入力と同じ形状）
        """
        # 入力型をチェック
        if not isinstance(values, np.ndarray):
            raise TypeError(f"特徴量 {name} の値はnumpy配列(np.ndarray)である必要があります。"
                        f"提供された型: {type(values)}")

        # 元の形状を保存
        original_shape = values.shape
        if values.ndim > 1:
            values = values.flatten()

        # 欠損値をnanに置換
        values = np.where(pd.isna(values), np.nan, values)

        # この特徴量がどのグループに属しているか確認
        group_name = self.feature_groups['numerical'][name]
        
        # グループの正規化パラメータが存在することを確認
        if group_name not in self.numerical_parameters:
            raise ValueError(f"グループ {group_name} の正規化パラメータが存在しません")
        
        params = self.numerical_parameters[group_name]
        mean_val = params.get('mean_value', 0.0)
        std_val = params.get('std_value', 1.0)
        
        # 標準化を実行（すべての特徴量で統一方法）
        if std_val > 0:
            normalized = (values - mean_val) / std_val
        else:
            normalized = values - mean_val  # 少なくとも平均を引く

        # 多次元だった場合は元の形状に戻す
        if len(original_shape) > 1:
            normalized = normalized.reshape(original_shape)

        # NaNはそのまま保持
        return torch.tensor(normalized, dtype=torch.float32)
    
    # 3. 内部実装メソッド
    def _fit_categorical_encoders(self):
        """カテゴリ値エンコーダをフィットします。（内部メソッド）"""
        logger.info("カテゴリカル特徴量のグループエンコーダをフィットしています...")
        
        # グループごとのエンコーダをフィット
        for group_name, values in self.group_observed_values.items():
            logger.info(f"グループ {group_name} のカテゴリカルエンコーダをフィットします（{len(values)} 個の一意な値）")
            values = sorted(values)
            
            # 値からインデックスへのマッピングを作成（0はNone/未知の値用に予約）
            value_to_index = {val: idx + 1 for idx, val in enumerate(values)}
            self.categorical_encoders[group_name] = value_to_index

    def _fit_numerical_parameters(self):
        """数値特徴量の正規化パラメータをフィットします。（内部メソッド）"""
        logger.info("数値特徴量のグループ正規化パラメータをフィットしています...")
        
        # グループごとのパラメータをフィット
        for group_name, values in self.group_numerical_values.items():
            if not values:
                logger.warning(f"グループ {group_name} の有効な値が見つかりません。デフォルトパラメータを使用します。")
                # デフォルトパラメータで初期化
                self.numerical_parameters[group_name] = {
                    'mean_value': 0.0,
                    'std_value': 1.0
                }
                continue
            
            # 値を配列に変換
            values_array = np.array(values, dtype=np.float32)
            
            # 標準化パラメータを計算
            mean_value = float(np.mean(values_array))
            std_value = float(np.std(values_array)) or 1.0  # 標準偏差が0の場合は1.0を使う
            
            self.numerical_parameters[group_name] = {
                'mean_value': mean_value,
                'std_value': std_value
            }
            
            logger.info(f"グループ {group_name} の正規化パラメータを計算しました: "
                    f"{self.numerical_parameters[group_name]} (有効値数: {len(values_array)}/{len(values_array)})")
    
    def _clear_collected_values(self):
        """フィット後に収集された値をクリアしてメモリを節約します。（内部メソッド）"""
        self.group_observed_values = defaultdict(set)
        self.group_numerical_values = defaultdict(list)
        logger.debug("収集された特徴量の値をクリアしました")
        
    # 4. 永続化/シリアライゼーション 
    def save_state(self, file_path):
        """処理器の状態（エンコーダと正規化パラメータ）を保存します。
        
        Args:
            file_path: 保存先のファイルパス
        """
        processor_state = {
            'categorical_encoders': self.categorical_encoders,
            'numerical_parameters': self.numerical_parameters,
            'feature_groups': self.feature_groups,
            'version': '0.1'
        }
        
        try:
            torch.save(processor_state, file_path)
            logger.info(f"処理器の状態を {file_path} に保存しました")
        except Exception as e:
            logger.error(f"処理器の状態の保存に失敗しました: {e}")
            raise

    def load_state(self, file_path):
        """保存された処理器の状態を読み込みます。
        
        Args:
            file_path: 読み込むファイルパス
        """
        try:
            processor_state = torch.load(file_path)
            
            # バージョン確認
            version = processor_state.get('version', '0.0')
            logger.debug(f"読み込んだ処理器の状態バージョン: {version}")
            
            self.categorical_encoders = processor_state['categorical_encoders']
            self.numerical_parameters = processor_state['numerical_parameters']
            
            # グループ情報も読み込み
            if version >= '0.1':
                self.feature_groups = processor_state.get('feature_groups', {'categorical': {}, 'numerical': {}})
            else:
                # 互換性のために空のマッピングを作成
                logger.warning("古いバージョンのstate fileからの読み込み: グループ情報がありません")

            logger.info(f"処理器の状態を {file_path} から読み込みました")
            
            # 特徴量数をログに出力
            logger.info(f"読み込まれたカテゴリカルグループ: {len(self.categorical_encoders)}個")
            logger.info(f"読み込まれた数値グループ: {len(self.numerical_parameters)}個")
            
        except Exception as e:
            error_msg = f"処理器の状態の読み込みに失敗しました: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_group_cardinalities(self):
        """カテゴリカルグループのカーディナリティを取得します。
        
        Returns:
            dict: グループ名をキー、カーディナリティを値とする辞書
        """
        cardinalities = {}
        for group_name, encoder in self.categorical_encoders.items():
            if encoder:
                cardinalities[group_name] = max(encoder.values())
        return cardinalities