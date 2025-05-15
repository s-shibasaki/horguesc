import abc
import logging
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class BaseDataset(abc.ABC):
    """全てのタスク固有データセットが継承する基底データセットクラス。
    
    このクラスはデータの取得から前処理、バッチ処理までのパイプラインを提供します。
    サブクラスでは主に _fetch_data メソッドを実装する必要があります。
    
    特徴:
    - ターゲット変数: 'target' または 'target_' で始まる名前のデータは自動的にバッチに含まれます
    - 補助データ: 特徴量やターゲット以外のデータも自動的にバッチに含まれます
    - データ型処理: テンソル/NumPy配列はインデックスでスライス、リストは要素抽出、その他はコピー
    
    使用例:
    ```python
    class MyDataset(BaseDataset):
        def _fetch_data(self, *args, **kwargs):
            # データ取得処理...
            self.raw_data = {
                # 特徴量 (必ずnumpy配列で、ini定義と名前が一致すること)
                'numerical_feature': np.array(...),
                'categorical_feature': np.array(...),
                
                # ターゲット (自動的にバッチに含まれる)
                'target': np.array(...),
                'target_auxiliary': np.array(...),
                
                # 補助データ (自動的にバッチに含まれる)
                'race_id': [...],  # リストでもOK
                'horse_names': [...],
            }
    ```
    """
    
    # データセットのモード定義
    MODE_TRAIN = 'train'  # 訓練用（シャッフル有り）
    MODE_EVAL = 'eval'    # 評価用（シャッフル無し）
    
    def __init__(self, 
                 config: Any, 
                 mode: str = 'train',
                 batch_size: int = 32,
                 start_date: Optional[Union[str, datetime]] = None, 
                 end_date: Optional[Union[str, datetime]] = None,
                 random_seed: Optional[int] = None,
                 *args,
                 **kwargs):
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
        
        # 追加の引数を保存
        self.args = args
        self.kwargs = kwargs
        
        # 初期化時の情報をログに記録
        logger.info(f"{self.__class__.__name__} データセット - モード: {self.mode}")
    
    # 公開API - データアクセスと処理のフロー
    
    def fetch_data(self, *args, **kwargs) -> 'BaseDataset':
        """データを取得します。
        
        Args:
            *args: データ取得時に渡す任意の位置引数
            **kwargs: データ取得時に渡す任意のキーワード引数
        """
        if self.raw_data is None:
            self._fetch_data(*args, **kwargs)
            
            if self.raw_data is None:
                logger.warning(f"{self.__class__.__name__} データ取得結果が空でした")
                self.raw_data = {}
            else:
                # データ取得後に数値特徴量を float32 型に変換
                self._convert_numerical_features_to_float()
    
        return self
    
    def collect_features(self, feature_processor: Any) -> 'BaseDataset':
        """特徴量プロセッサーに生データの特徴量を収集させます。"""
        if self.raw_data is None:
            raise ValueError(f"{self.get_name()} データが取得されていません。先に fetch_data() を呼び出してください")
        
        if not self._data_collected:
            logger.info(f"{self.__class__.__name__} データセットから特徴量の値を収集中...")
            feature_processor.collect_values_for_fitting(self.raw_data)
            self._data_collected = True
            logger.info(f"{self.__class__.__name__} データセットからの特徴量値収集が完了しました")
        
        return self
    
    def process_features(self, feature_processor: Any) -> 'BaseDataset':
        """特徴量プロセッサーを使って特徴量を処理します。"""
        if self.raw_data is None:
            raise ValueError(f"{self.get_name()} データが取得されていません。先に fetch_data() を呼び出してください")
        
        if not self._data_processed:
            logger.info(f"{self.__class__.__name__} データセットの特徴量を処理中...")
            self.processed_data = self._process_features(self.raw_data, feature_processor)
            self._data_processed = True
            self._init_batch_indices()  # バッチインデックスの初期化を追加
            logger.info(f"{self.__class__.__name__} データセットの特徴量処理が完了しました")
        
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
                logger.debug(f"{self.__class__.__name__} データセット: 最後のバッチに到達したため再シャッフルします")
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
        """特徴量を処理します。サブクラスでカスタム処理が必要な場合はオーバーライドします。
        
        このメソッドでは以下が自動的に処理されます:
        1. 設定ファイルで定義された数値特徴量とカテゴリカル特徴量のエンコード
        2. ターゲット変数の自動転送 (target または target_ で始まるキー)
        3. 補助データの自動転送 (AUXILIARY_KEYS に指定されたキー)
        
        オーバーライド時のヒント:
        - 特別な処理が必要な場合は、super()._process_features()の結果に追加処理を行うことを推奨
        
        Args:
            raw_data: 生データの辞書
            feature_processor: 特徴量処理オブジェクト
            
        Returns:
            処理済みデータの辞書
        """
        processed = {}
        
        # 設定から特徴量情報を取得
        numerical_features = self.config.numerical_features
        categorical_features = self.config.categorical_features
        
        # 処理済みキーのトラッキング
        processed_keys = set()
        
        # 数値特徴量の処理
        for feature_name in numerical_features:
            if feature_name in raw_data:
                processed[feature_name] = feature_processor.normalize_numerical_feature(
                    feature_name, raw_data[feature_name]
                )
                processed_keys.add(feature_name)
    
        # カテゴリカル特徴量の処理
        for feature_name in categorical_features:
            if feature_name in raw_data:
                processed[feature_name] = feature_processor.encode_categorical_feature(
                    feature_name, raw_data[feature_name]
                )
                processed_keys.add(feature_name)
                
        # ターゲット変数をそのままコピー (target または target_で始まるキー)
        for key, value in raw_data.items():
            if key == 'target' or key.startswith('target_'):
                if isinstance(value, np.ndarray):
                    processed[key] = torch.tensor(value)
                else:
                    # NumPy配列でない場合はそのまま保存
                    processed[key] = value
                processed_keys.add(key)
            
        # 自動的に処理されていない残りのデータを含める
        # AUXILIARY_KEYS を明示的に定義する必要がなくなる
        for key, value in raw_data.items():
            if key not in processed_keys:
                processed[key] = value
            
        # 処理結果をログに記録
        numerical_count = len([f for f in numerical_features if f in raw_data])
        categorical_count = len([f for f in categorical_features if f in raw_data])
        targets_count = len([k for k in processed.keys() if k == 'target' or k.startswith('target_')])
        auxiliary_count = len([k for k in processed.keys() if k not in processed_keys])
        
        logger.info(f"処理された特徴量: 数値 {numerical_count}個, カテゴリ {categorical_count}個, "
                    f"ターゲット {targets_count}個, 補助データ {auxiliary_count}個")
        
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
    
    def _get_items_at_indices(self, indices: np.ndarray) -> Dict[str, Any]:
        """指定されたインデックスのアイテムを取得します。サブクラスでオーバーライド可能。
        
        このメソッドは、データタイプに応じて以下のように処理します:
        1. Tensor/NumPy配列: インデックスでスライスされます (data[indices])
        2. リスト: インデックスに対応する要素が抽出されます ([data[i] for i in indices])
        3. その他のデータ型: そのままコピーされます
        
        Args:
            indices: 取得するデータのインデックス配列
            
        Returns:
            指定インデックスのデータを含む辞書
        """
        batch = {}
        for feature_name, feature_data in self.processed_data.items():
            if isinstance(feature_data, (torch.Tensor, np.ndarray)):
                # テンソルやNumPy配列の場合はインデックスでスライス
                batch[feature_name] = feature_data[indices]
            elif isinstance(feature_data, list) and len(feature_data) > 0:
                # リストの場合はインデックスでスライス
                batch[feature_name] = [feature_data[i] for i in indices]
            else:
                # その他の場合はそのままコピー
                batch[feature_name] = feature_data
        return batch
    
    # 抽象メソッド - サブクラスで必ず実装する必要あり
    
    @abc.abstractmethod
    def _fetch_data(self, *args, **kwargs) -> None:
        """データを取得し、self.raw_dataに格納します。
        
        実装時の必須ルール:
        1. 名前と形状の一致要件:
           - モデル入力特徴量は、horguesc.ini で定義した特徴量名と完全に一致させる
           - horguesc.ini の [features] セクションに定義されていない特徴量はエンコード対象外となる
           - エンコードされる特徴量（numerical_features と categorical_features に含まれるもの）は
             すべて同じ形状である必要がある（FeatureEncoder がバッチ処理を行うため）
        
        2. ターゲットの扱い:
           - ターゲットカラムは 'target' という名前、または 'target_' で始まる名前で格納する
             (例: 'target', 'target_win', 'target_place' など)
           - 'target' や 'target_' で始まるカラムは自動的に処理され、MultitaskTrainer によって使用される
        
        3. 補助データの扱い:
           - 特徴量（numerical_features/categorical_features）やターゲット（target/target_*）以外の
             情報も必要に応じて self.raw_data に含めることができる
           - これらの補助データはモデルトレーニングには直接使用されないが、後処理や分析に利用可能
           - 例: race_id, timestamp, metadata などの補足情報
           - 補助データはNumPy配列である必要はなく、リストや辞書など任意の形式で保持可能
        
        4. データ形式:
           - self.raw_data に辞書形式でデータを格納する
           - モデルの入力となる特徴量とターゲットのデータは NumPy 配列 (np.ndarray) として格納する
           - 1次元配列だけでなく、多次元配列 (2D, 3D等) も必要に応じて使用可能
        
        5. 欠損値の取り扱い:
           - 数値特徴量: None や非数値データは欠損値として処理され、np.nan に変換される
           - カテゴリカル特徴量: None は欠損値として特別に処理される
    
        データ構造の例:
        # 例1: すべての特徴量が1次元配列の場合
        self.raw_data = {
            'feature_a': np.array([1, 2, 3, ...]),           # 数値特徴量 (1次元)
            'feature_b': np.array([5, 8, 2, ...]),           # 数値特徴量 (1次元)
            'feature_c': np.array(['A', 'B', 'A', ...]),     # カテゴリカル特徴量 (1次元)
            'target': np.array([0, 1, 1, ...]),              # 基本的なターゲット変数
            'target_place': np.array([1, 3, 2, ...]),        # 追加のターゲット変数
            'race_id': ['race_123', 'race_123', 'race_123', ...]  # 補助データ (リスト形式も可)
        }
        
        # 例2: すべての特徴量が2次元配列の場合
        self.raw_data = {
            'feature_x': np.array([[1, 2], [3, 4], ...]),    # 数値特徴量 (2次元)
            'feature_y': np.array([[5, 6], [7, 8], ...]),    # 数値特徴量 (2次元)
            'target': np.array([0, 1, 1, ...]),              # ターゲット変数
            'metadata': {...}                                # その他の補助情報
        }
        
        Args:
            *args: データ取得時に渡される任意の位置引数
            **kwargs: データ取得時に渡される任意のキーワード引数
        """
        # サブクラスで実装
        pass
    
    def _convert_numerical_features_to_float(self) -> None:
        """数値特徴量を float32 型に変換します。None や非数値は np.nan に変換します."""
        if not hasattr(self.config, 'numerical_features') or not self.raw_data:
            return
            
        for feature_name in self.config.numerical_features:
            if feature_name in self.raw_data:
                original = self.raw_data[feature_name]
                
                # 変換処理
                values = []
                for value in original.flatten() if original.ndim > 1 else original:
                    try:
                        # Noneまたは変換できない値はnp.nanにする
                        if value is None:
                            values.append(np.nan)
                        else:
                            values.append(float(value))
                    except (ValueError, TypeError):
                        values.append(np.nan)
                
                # 元の形状を維持して配列を作り直す
                converted = np.array(values, dtype=np.float32)
                if original.ndim > 1:
                    converted = converted.reshape(original.shape)
                    
                # 変換後の配列に置き換え
                self.raw_data[feature_name] = converted
                
                logger.debug(f"数値特徴量 '{feature_name}' を float32 型に自動変換しました")