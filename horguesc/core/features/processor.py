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
        
        self.categorical_encoders = {}
        self.numerical_parameters = {}
        
        # カテゴリカル特徴量の見た値を記録
        self.observed_values = defaultdict(set)
        # 数値特徴量の値を全て記録 (統計計算用)
        self.numerical_values = defaultdict(list)
        
    # 2. 公開API（ライフサイクル順）
    def collect_values_for_fitting(self, data_source):
        """データソースから特徴量の値を収集します（フィットはせず、値の収集のみ）
        
        Args:
            data_source: 生データを含む辞書
        """
        for feature_name, values in data_source.items():
            if feature_name in self.config.feature_config.categorical_features:
                # None値を除外して追加
                self.observed_values[feature_name].update(
                    v for v in values if v is not None
                )
            # 数値特徴量の値も収集
            elif feature_name in self.config.feature_config.numerical_features:
                # 統計計算用に全ての値をリストにも保存
                self.numerical_values[feature_name].extend(
                    v for v in values if v is not None
                )
        
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
        """カテゴリカル特徴量をエンコードします。
        
        Args:
            name: 特徴量の名前
            values: エンコーディングする値のリストまたはnumpy配列
            
        Returns:
            torch.Tensor: エンコーディングされたインデックス
        """
        # エンコーダが存在することを確認
        if name not in self.categorical_encoders:
            raise ValueError(f"特徴量 {name} のエンコーダが存在しません。fit_categorical_encoders を先に実行してください。")
        
        encoder = self.categorical_encoders[name]
        
        if isinstance(values, np.ndarray):
            values = values.tolist()
        
        # 値をインデックスに変換（不明な値は0にマッピング）
        indices = [encoder.get(val, 0) for val in values]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def normalize_numerical_feature(self, name, values):
        """数値特徴量を正規化します。
        
        Args:
            name: 特徴量の名前
            values: 正規化する値のリストまたはnumpy配列
            
        Returns:
            torch.Tensor: 正規化された値
        """
        # numpy配列に変換
        if not isinstance(values, np.ndarray):
            values = np.array(values, dtype=np.float32)
        
        # 欠損値をnanに置換
        values = np.where(pd.isna(values), np.nan, values)
        
        # 正規化パラメータを取得（存在しない場合はエラー）
        if name not in self.numerical_parameters:
            raise ValueError(f"特徴量 {name} の正規化パラメータが存在しません。fit_numerical_parameters を先に実行してください。")
        
        params = self.numerical_parameters.get(name, {})
        normalization = params.get('normalization', 'none')
        
        # 正規化方法に基づいて処理
        if normalization == 'minmax':
            min_val = params.get('min_value')
            max_val = params.get('max_value')
            range_val = max_val - min_val
            if range_val > 0:
                normalized = (values - min_val) / range_val
            else:
                normalized = np.zeros_like(values)
        
        elif normalization == 'standard':
            mean_val = params.get('mean_value')
            std_val = params.get('std_value')
            if std_val > 0:
                normalized = (values - mean_val) / std_val
            else:
                logger.warning(f"特徴量 {name} の標準偏差が0です。値をそのまま使用します。")
                normalized = values - mean_val  # 少なくとも平均を引く
        
        else:  # 'none'または他の不明な正規化
            normalized = values

        # NaNはそのまま保持
        return torch.tensor(normalized, dtype=torch.float32)
    
    # 3. 内部実装メソッド
    def _fit_categorical_encoders(self):
        """カテゴリ値エンコーダをフィットします。（内部メソッド）"""
        logger.info("カテゴリカル特徴量のエンコーダをフィットしています...")
        
        # 各カテゴリカル特徴量のエンコーダを作成 (値はすでに収集済み)
        for feature_name, feature_info in self.config.feature_config.categorical_features.items():
            values = sorted(self.observed_values.get(feature_name, set()))
            
            # 設定されたカーディナリティを確認
            cardinality = feature_info['cardinality']
            if len(values) >= cardinality:
                logger.warning(
                    f"特徴量 {feature_name} の実際の値の数 ({len(values)}) が "
                    f"設定されたカーディナリティ ({cardinality}) 以上です。"
                )
            
            # 値からインデックスへのマッピングを作成
            # 0はNone/未知の値用に予約、1から有効な値を割り当て
            value_to_index = {val: idx + 1 for idx, val in enumerate(values)}
            
            self.categorical_encoders[feature_name] = value_to_index
            logger.info(f"特徴量 {feature_name} のエンコーダを作成しました（{len(values)} 個の一意な値）")
    
    def _fit_numerical_parameters(self):
        """数値特徴量の正規化パラメータをフィットします。（内部メソッド）"""
        logger.info("数値特徴量の正規化パラメータをフィットしています...")
        
        # 各数値特徴量について正規化パラメータを計算
        for feature_name, feature_info in self.config.feature_config.numerical_features.items():
            normalization = feature_info.get('normalization', 'none')
            
            # collect_values_for_fitting で収集された値を使用
            if feature_name not in self.numerical_values or not self.numerical_values[feature_name]:
                logger.warning(f"特徴量 {feature_name} の有効な値が見つかりません。デフォルトパラメータを使用します。")
                # 正規化方法も一緒に保存
                self.numerical_parameters[feature_name] = {
                    'normalization': normalization
                }
                
                # 正規化方法に応じて必要なパラメータだけを追加
                if normalization == 'minmax':
                    self.numerical_parameters[feature_name].update({
                        'min_value': 0.0,
                        'max_value': 1.0
                    })
                elif normalization == 'standard':
                    self.numerical_parameters[feature_name].update({
                        'mean_value': 0.0,
                        'std_value': 1.0
                    })
                
                continue
                
            values_array = np.array(self.numerical_values[feature_name], dtype=np.float32)
            
            # 基本情報として正規化方法を常に保存
            self.numerical_parameters[feature_name] = {
                'normalization': normalization
            }
            
            # 正規化方法に応じて必要なパラメータだけを計算・保存
            if normalization == 'minmax':
                self.numerical_parameters[feature_name].update({
                    'min_value': float(np.min(values_array)),
                    'max_value': float(np.max(values_array))
                })
            elif normalization == 'standard':
                self.numerical_parameters[feature_name].update({
                    'mean_value': float(np.mean(values_array)),
                    'std_value': float(np.std(values_array)) or 1.0  # 標準偏差が0の場合は1.0を使う
                })
            
            if normalization != 'none':
                logger.info(f"特徴量 {feature_name} の正規化パラメータを計算しました: "
                          f"{self.numerical_parameters[feature_name]}")
    
    def _clear_collected_values(self):
        """フィット後に収集された値をクリアしてメモリを節約します。（内部メソッド）"""
        self.observed_values = defaultdict(set)
        self.numerical_values = defaultdict(list)
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
            'version': '1.0'  # バージョン管理のため
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
            
        Raises:
            RuntimeError: ファイルの読み込みに失敗した場合
        """
        try:
            processor_state = torch.load(file_path)
            
            # バージョン確認（将来の互換性のため）
            version = processor_state.get('version', '0.0')
            logger.debug(f"読み込んだ処理器の状態バージョン: {version}")
            
            self.categorical_encoders = processor_state['categorical_encoders']
            self.numerical_parameters = processor_state['numerical_parameters']

            logger.info(f"処理器の状態を {file_path} から読み込みました")
            
            # カテゴリカル特徴量数と数値特徴量数をログに出力
            logger.info(f"読み込まれたカテゴリカル特徴量: {len(self.categorical_encoders)}個")
            logger.info(f"読み込まれた数値特徴量: {len(self.numerical_parameters)}個")
            
        except Exception as e:
            error_msg = f"処理器の状態の読み込みに失敗しました: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

