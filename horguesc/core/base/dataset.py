import abc
import logging
from horguesc.core.features.processor import FeatureProcessor

logger = logging.getLogger(__name__)

class BaseDataset(abc.ABC):
    """全てのタスク固有データセットが継承する基底データセットクラス。"""
    
    def __init__(self, config, feature_processor=None):
        self.config = config
        
        # 特徴量プロセッサを引数として受け取るか、新しく作成
        if feature_processor is None:
            self.feature_processor = FeatureProcessor(config)
        else:
            self.feature_processor = feature_processor
        
        # トレーニング/バリデーションデータ
        self.train_data = None
        self.val_data = None
        
        # このデータセットが担当する特徴量のリスト
        self.numerical_features = []
        self.categorical_features = []
    
    def set_features(self, numerical_features, categorical_features):
        """このデータセットが担当する特徴量を設定します。
        
        Args:
            numerical_features: 数値特徴量名のリスト
            categorical_features: カテゴリ特徴量名のリスト
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        logger.info(f"{self.get_name()} データセットが担当する特徴量: "
                   f"数値特徴量 {len(numerical_features)}個, "
                   f"カテゴリ特徴量 {len(categorical_features)}個")
    
    def process_features(self, raw_data):
        """生データから特徴量を処理します。
        
        Args:
            raw_data: データベースから取得した生データ
            
        Returns:
            dict: 処理済み特徴量を含む辞書
        """
        processed = {}
        
        # 数値特徴量の処理
        for feature_name in self.numerical_features:
            if feature_name in raw_data:
                values = raw_data[feature_name]
                processed[feature_name] = self.feature_processor.normalize_numerical_feature(
                    feature_name, values
                )
        
        # カテゴリカル特徴量の処理
        for feature_name in self.categorical_features:
            if feature_name in raw_data:
                values = raw_data[feature_name]
                processed[feature_name] = self.feature_processor.encode_categorical_feature(
                    feature_name, values
                )
        
        return processed
    
    def get_raw_data(self):
        """データベースから生データを取得します。
        デフォルトでは空のデータを返します。
        サブクラスでオーバーライドして実装する必要があります。
        
        Returns:
            dict: 特徴量名をキー、生データ値のリストを値とする辞書
        """
        return {}
    
    def prepare_data(self):
        """データの準備とエンコーダのフィッティングを行います。"""
        # 生データを取得
        raw_data = self.get_raw_data()
        
        # フィッティングとエンコーディング
        # トレーニングモードの場合はエンコーダをフィット
        if self.feature_processor.training_mode:
            self.feature_processor.fit_categorical_encoders([raw_data])
        
        # 特徴量を処理
        processed_data = self.process_features(raw_data)
        
        # トレーニング/検証データに分割
        self._split_data(processed_data)
    
    def _split_data(self, processed_data):
        """処理済みデータをトレーニングデータと検証データに分割します。
        デフォルトではすべてトレーニングデータとして扱います。
        
        Args:
            processed_data: 処理済みの特徴量データ
        """
        # サブクラスでオーバーライドして分割ロジックを実装
        self.train_data = processed_data
        self.val_data = processed_data
    
    @abc.abstractmethod
    def get_next_batch(self, batch_size):
        """トレーニング用のデータの次のバッチを取得します。
        
        Args:
            batch_size: 取得するバッチのサイズ
            
        Returns:
            dict: タスクの入力とターゲットを含む辞書
        """
        pass
    
    @abc.abstractmethod
    def get_validation_data(self):
        """評価用のバリデーションデータを取得します。
        
        Returns:
            dict: バリデーション入力とターゲットを含む辞書
        """
        pass
    
    @abc.abstractmethod
    def get_name(self):
        """タスクの名前を取得します。
        
        Returns:
            str: タスク名
        """
        pass