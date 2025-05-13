import abc
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class BaseDataset(abc.ABC):
    """全てのタスク固有データセットが継承する基底データセットクラス。"""
    
    # データセットのモード定義
    MODE_TRAIN = 'train'  # 訓練用（シャッフル有り）
    MODE_EVAL = 'eval'    # 評価用（シャッフル無し）
    
    def __init__(self, 
                 config: Any, 
                 mode: str = 'train',
                 batch_size: int = 32,
                 start_date: Optional[Union[str, datetime]] = None, 
                 end_date: Optional[Union[str, datetime]] = None,
                 random_seed: Optional[int] = None):
        """初期化
        
        Args:
            config: アプリケーション設定
            mode: データセットモード ('train' または 'eval')
            batch_size: バッチサイズ
            start_date: データの開始日 (日付文字列またはdatetime)
            end_date: データの終了日 (日付文字列またはdatetime)
            random_seed: シャッフル用の乱数シード
        """
        self.config = config
        
        # データセットモード
        valid_modes = [self.MODE_TRAIN, self.MODE_EVAL]
        if mode not in valid_modes:
            raise ValueError(f"モードは {', '.join(valid_modes)} のいずれかである必要があります")
        self.mode = mode
        
        # 日付範囲の設定
        self.start_date = self._parse_date(start_date)
        self.end_date = self._parse_date(end_date)
        
        # 乱数生成器の初期化
        self._rng = np.random.RandomState(random_seed)
        
        # データ状態
        self.raw_data = None
        self.processed_data = None
        self._data_collected = False
        self._data_processed = False
        self._batch_indices = None
        self._next_batch_index = 0
        self.batch_size = batch_size
        
        # 初期化時の情報をログに記録
        logger.info(f"{self.get_name()} データセット - モード: {self.mode}")
    
    # 公開API - データアクセスと処理のフロー
    
    def fetch_data(self) -> 'BaseDataset':
        """データを取得します。"""
        if self.raw_data is None:
            self._fetch_data()
            
            if self.raw_data is None:
                raise ValueError(f"{self.get_name()} データセットの _fetch_data() メソッドが raw_data を設定していません")
        
        return self
    
    def collect_features(self, feature_processor: Any) -> 'BaseDataset':
        """特徴量プロセッサーに生データの特徴量を収集させます。"""
        if self.raw_data is None:
            raise ValueError(f"{self.get_name()} データが取得されていません。先に fetch_data() を呼び出してください")
        
        if not self._data_collected:
            logger.info(f"{self.get_name()} データセットから特徴量の値を収集中...")
            feature_processor.collect_values_for_fitting(self.raw_data)
            self._data_collected = True
            logger.info(f"{self.get_name()} データセットからの特徴量値収集が完了しました")
        
        return self
    
    def process_features(self, feature_processor: Any) -> 'BaseDataset':
        """特徴量プロセッサーを使って特徴量を処理します。"""
        if self.raw_data is None:
            raise ValueError(f"{self.get_name()} データが取得されていません。先に fetch_data() を呼び出してください")
        
        if not self._data_processed:
            logger.info(f"{self.get_name()} データセットの特徴量を処理中...")
            self.processed_data = self._process_features(self.raw_data, feature_processor)
            self._data_processed = True
            self._init_batch_indices()  # バッチインデックスの初期化を追加
            logger.info(f"{self.get_name()} データセットの特徴量処理が完了しました")
        
        return self
    
    def get_all_data(self) -> Dict[str, np.ndarray]:
        """全てのデータを一度に取得します。"""
        return self._get_all_data()
    
    def get_batch(self) -> Dict[str, np.ndarray]:
        """固定サイズのバッチを取得します。
        
        Returns:
            バッチデータを含む辞書
            
        Note:
            最後のバッチに到達した後、訓練モードではデータが自動的にシャッフルされます。
        """
        # シーケンシャルなバッチ取得
        batch_data = self._get_batch_at_index(self.batch_size, self._next_batch_index)
        self._next_batch_index += 1
        
        # 最後のバッチかどうかを判定（端数も考慮）
        data_size = self._get_data_size()
        total_batches = (data_size + self.batch_size - 1) // self.batch_size  # 切り上げ除算
        
        if self._next_batch_index >= total_batches:
            # 最後のバッチだった場合、次回のためにインデックスを初期化
            if self.mode == self.MODE_TRAIN:
                logger.debug(f"{self.get_name()} データセット: 最後のバッチに到達したため再シャッフルします")
            self._init_batch_indices()  # インデックス初期化（シャッフルも含む）
        
        return batch_data
    
    # プロテクティッドメソッド - サブクラスでオーバーライド可能
    
    def _parse_date(self, date_value: Optional[Union[str, datetime]]) -> Optional[datetime]:
        """日付文字列またはdatetime型の値を処理して、datetime型に変換します。"""
        if date_value is None:
            return None
            
        if isinstance(date_value, datetime):
            return date_value
            
        return datetime.strptime(date_value, '%Y-%m-%d')
    
    def _process_features(self, raw_data: Dict[str, Any], feature_processor: Any) -> Dict[str, np.ndarray]:
        """特徴量を処理します。サブクラスでカスタム処理が必要な場合はオーバーライドします。"""
        processed = {}
        
        # 設定から特徴量情報を取得
        numerical_features = self.config.get_numerical_feature_names()
        categorical_features = self.config.get_categorical_feature_names()
        
        # 数値特徴量の処理
        for feature_name in numerical_features:
            if feature_name in raw_data:
                processed[feature_name] = feature_processor.normalize_numerical_feature(
                    feature_name, raw_data[feature_name]
                )
        
        # カテゴリカル特徴量の処理
        for feature_name in categorical_features:
            if feature_name in raw_data:
                processed[feature_name] = feature_processor.encode_categorical_feature(
                    feature_name, raw_data[feature_name]
                )
        
        # 処理された特徴量の数をログに記録
        numerical_count = len([f for f in numerical_features if f in raw_data])
        categorical_count = len([f for f in categorical_features if f in raw_data])
        logger.info(f"処理された特徴量: 数値 {numerical_count}個, カテゴリ {categorical_count}個")
        
        return processed
    
    def _init_batch_indices(self) -> None:
        """バッチ処理用のインデックス配列を初期化します。"""
        data_size = self._get_data_size()
        self._batch_indices = np.arange(data_size)
        self._next_batch_index = 0
        
        # 訓練モードの場合はシャッフル
        if self.mode == self.MODE_TRAIN:
            self._shuffle_indices()
    
    def _shuffle_indices(self) -> None:
        """インデックス配列をシャッフルします。"""
        self._rng.shuffle(self._batch_indices)
    
    def _get_data_size(self) -> int:
        """データサイズを取得します。サブクラスでオーバーライド可能。"""
        # デフォルトの実装 - 最初に見つかった特徴量の長さを使用
        if self.processed_data and len(self.processed_data) > 0:
            first_feature = next(iter(self.processed_data.values()))
            return len(first_feature)
        return 0
    
    def _get_all_data(self) -> Dict[str, np.ndarray]:
        """全てのデータを返します。サブクラスでオーバーライド可能。"""
        return self.processed_data
    
    def _get_batch_at_index(self, batch_size: int, batch_index: int) -> Dict[str, np.ndarray]:
        """特定のインデックスのバッチを取得します。サブクラスでオーバーライド可能。"""
        data_size = self._get_data_size()
        start_idx = batch_index * batch_size
        end_idx = min(start_idx + batch_size, data_size)
        
        indices = self._batch_indices[start_idx:end_idx]
        return self._get_items_at_indices(indices)
    
    def _get_items_at_indices(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """指定されたインデックスのアイテムを取得します。サブクラスでオーバーライド可能。"""
        batch = {}
        for feature_name, feature_data in self.processed_data.items():
            batch[feature_name] = feature_data[indices]
        return batch
    
    # 抽象メソッド - サブクラスで必ず実装する必要あり
    
    @abc.abstractmethod
    def _fetch_data(self) -> None:
        """データを取得し、self.raw_dataに格納します。サブクラスで実装する必要があります。"""
        pass
    
    @abc.abstractmethod
    def get_name(self) -> str:
        """データセットの名前を取得します。"""
        pass