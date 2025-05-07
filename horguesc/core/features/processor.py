import logging
import numpy as np
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)

class FeatureProcessor:
    """特徴量の前処理を一元管理するクラス"""
    
    def __init__(self, config, feature_config):
        """
        Args:
            config: アプリケーションのメイン設定
            feature_config: 特徴量固有の設定
        """
        self.config = config  # メイン設定も保持
        self.feature_config = feature_config
        
        # カテゴリカル特徴量のエンコーディング情報を保存
        self.categorical_encoders = {}
        
        # 各特徴量の見た値を記録
        self.observed_values = defaultdict(set)
        
        # トレーニングモード（新しい値を記録するかどうか）
        self.training_mode = True
    
    def set_training_mode(self, mode=True):
        """トレーニングモードを設定します。
        
        Args:
            mode: Trueの場合、新しい値を記録します。Falseの場合は既存の値のみを使用します。
        """
        self.training_mode = mode
    
    def fit_categorical_encoders(self, data_sources):
        """複数のデータソースからカテゴリ値を収集し、エンコーダをフィットします。
        
        Args:
            data_sources: 各データセットのデータソースのリスト
        """
        logger.info("カテゴリカル特徴量のエンコーダをフィットしています...")
        
        # 各データソースから値を収集
        for source in data_sources:
            for feature_name, values in source.items():
                if feature_name in self.feature_config.categorical_features:
                    # None値を除外して追加
                    self.observed_values[feature_name].update(
                        v for v in values if v is not None
                    )
        
        # 各カテゴリカル特徴量のエンコーダを作成
        for feature_name, feature_info in self.feature_config.categorical_features.items():
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
    
    def encode_categorical_feature(self, name, values):
        """カテゴリカル特徴量をエンコードします。
        
        Args:
            name: 特徴量の名前
            values: エンコードする値のリストまたはnumpy配列
            
        Returns:
            torch.Tensor: エンコードされたインデックス
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
        feature_info = self.feature_config.get_feature_info(name)
        normalization = feature_info.get('normalization', 'none')
        
        # numpy配列に変換
        if not isinstance(values, np.ndarray):
            values = np.array(values, dtype=np.float32)
        
        # 欠損値をnanに置換
        values = np.where(values == None, np.nan, values)
        
        # 正規化方法に基づいて処理
        if normalization == 'minmax':
            min_val = feature_info.get('min_value', np.nanmin(values))
            max_val = feature_info.get('max_value', np.nanmax(values))
            range_val = max_val - min_val
            if range_val > 0:
                normalized = (values - min_val) / range_val
            else:
                normalized = np.zeros_like(values)
        
        elif normalization == 'standard':
            mean_val = feature_info.get('mean_value', np.nanmean(values))
            std_val = feature_info.get('std_value', np.nanstd(values))
            if std_val > 0:
                normalized = (values - mean_val) / std_val
            else:
                normalized = np.zeros_like(values)
        
        else:  # 'none'または他の不明な正規化
            normalized = values

        # NaNはそのまま保持
        return torch.tensor(normalized, dtype=torch.float32)
    
    def save_encoders(self, file_path):
        """エンコーダの状態を保存します。
        
        Args:
            file_path: 保存先のファイルパス
        """
        encoder_state = {
            'categorical_encoders': self.categorical_encoders,
            'observed_values': {k: list(v) for k, v in self.observed_values.items()}
        }
        torch.save(encoder_state, file_path)
        logger.info(f"エンコーダの状態を {file_path} に保存しました")
    
    def load_encoders(self, file_path):
        """保存されたエンコーダの状態を読み込みます。
        
        Args:
            file_path: 読み込むファイルパス
        """
        encoder_state = torch.load(file_path)
        self.categorical_encoders = encoder_state['categorical_encoders']
        self.observed_values = defaultdict(set)
        for k, v in encoder_state['observed_values'].items():
            self.observed_values[k] = set(v)
        logger.info(f"エンコーダの状態を {file_path} から読み込みました")