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
    
    データセットのモード:
    - train: 訓練用（過去データ、結果ありでシャッフル有り）
    - eval: 評価用（過去データ、結果ありでシャッフル無し）
    - inference: 推論用（未知データ、結果なしでシャッフル無し）
    """
    
    # データセットのモード定義
    MODE_TRAIN = 'train'        # 訓練用（過去データ、結果あり、シャッフル有り）
    MODE_EVAL = 'eval'          # 評価用（過去データ、結果あり、シャッフル無し）
    MODE_INFERENCE = 'inference'  # 推論用（未知データ、結果なし、シャッフル無し）
    
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
            mode: データセットモード ('train', 'eval', または 'inference')
            batch_size: バッチサイズ
            start_date: データの開始日 (日付文字列またはdatetime)
            end_date: データの終了日 (日付文字列またはdatetime)
            random_seed: シャッフル用の乱数シード
        """
        self.config = config
        
        # データセットモード
        valid_modes = [self.MODE_TRAIN, self.MODE_EVAL, self.MODE_INFERENCE]
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
        
        # データ拡張のパラメータを設定から読み込む
        self.num_noise_scale = config.getfloat('augmentation', 'num_noise_scale', fallback=0.05)
        self.num_nan_prob = config.getfloat('augmentation', 'num_nan_prob', fallback=0.02)
        self.cat_null_prob = config.getfloat('augmentation', 'cat_null_prob', fallback=0.02)
        
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
            raise ValueError(f"{self.__class__.__name__} データが取得されていません。先に fetch_data() を呼び出してください")
        
        if not self._data_collected:
            logger.info(f"{self.__class__.__name__} データセットから特徴量の値を収集中...")
            feature_processor.collect_values_for_fitting(self.raw_data)
            self._data_collected = True
            logger.info(f"{self.__class__.__name__} データセットからの特徴量値収集が完了しました")
        
        return self
    
    def process_features(self, feature_processor: Any) -> 'BaseDataset':
        """特徴量プロセッサーを使って特徴量を処理します。"""
        if self.raw_data is None:
            raise ValueError(f"{self.__class__.__name__} データが取得されていません。先に fetch_data() を呼び出してください")
        
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
    
    def get_batch(self) -> tuple[Dict[str, np.ndarray], bool]:
        """固定サイズのバッチを取得します。
        
        Returns:
            tuple: (バッチデータを含む辞書, 最後のバッチかどうかを示すブール値)
        """
        # シーケンシャルなバッチ取得
        batch_data = self._get_batch_at_index(self.batch_size, self._next_batch_index)
        self._next_batch_index += 1
        
        # 最後のバッチかどうかを判定（端数も考慮）
        data_size = self._get_data_size()
        total_batches = (data_size + self.batch_size - 1) // self.batch_size  # 切り上げ除算
        
        is_last_batch = self._next_batch_index >= total_batches
        
        if is_last_batch:
            # 最後のバッチだった場合の処理
            if self.mode == self.MODE_TRAIN:
                # 訓練モードでは再シャッフル
                logger.debug(f"{self.__class__.__name__} データセット: 最後のバッチに到達したため再シャッフルします")
                self._init_batch_indices()
            elif self.mode == self.MODE_EVAL or self.mode == self.MODE_INFERENCE:
                # 評価・推論モードではインデックスをリセットするだけ（シャッフルなし）
                logger.debug(f"{self.__class__.__name__} データセット: 最後のバッチに到達したためインデックスをリセットします")
                self._init_batch_indices(shuffle=False)

        # トレーニングモードの場合、バッチデータに拡張を適用
        if self.mode == self.MODE_TRAIN:
            batch_data = self._apply_data_augmentation(batch_data)

        return batch_data, is_last_batch
    
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
        4. 生データの保持 (raw_ プレフィックス付きで追加)
        
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
        
        # 処理カウンター
        numerical_processed = []
        categorical_processed = []
        target_copied = []
        auxiliary_copied = []
        raw_added = []
        
        logger.info(f"特徴量データ処理の開始: {len(raw_data)}個のキーを処理します")
        
        # 数値特徴量の処理
        for feature_name in numerical_features:
            if feature_name in raw_data:
                logger.debug(f"数値特徴量 '{feature_name}' の正規化処理を実行します")
                processed[feature_name] = feature_processor.normalize_numerical_feature(
                    feature_name, raw_data[feature_name]
                )
                processed_keys.add(feature_name)
                numerical_processed.append(feature_name)
                logger.debug(f"数値特徴量 '{feature_name}' の正規化が完了しました: "
                        f"形状={processed[feature_name].shape}, 型={processed[feature_name].dtype}")
        
                # 生データも保持する
                raw_key = f"raw_{feature_name}"
                # Object型の配列はテンソルに変換せずそのまま保持
                if isinstance(raw_data[feature_name], np.ndarray) and raw_data[feature_name].dtype == np.dtype('O'):
                    processed[raw_key] = raw_data[feature_name]
                else:
                    try:
                        # 変換可能な場合のみテンソルに変換
                        processed[raw_key] = torch.tensor(raw_data[feature_name]) if isinstance(raw_data[feature_name], np.ndarray) else raw_data[feature_name]
                    except TypeError:
                        # 変換できない場合は元の配列をそのまま保持
                        processed[raw_key] = raw_data[feature_name]
                        logger.debug(f"数値特徴量 '{feature_name}' の生データをテンソルに変換できないため、元の形式で保存します")
                
                processed_keys.add(raw_key)
                raw_added.append(raw_key)

        # カテゴリカル特徴量の処理
        for feature_name in categorical_features:
            if feature_name in raw_data:
                logger.debug(f"カテゴリカル特徴量 '{feature_name}' のエンコード処理を実行します")
                processed[feature_name] = feature_processor.encode_categorical_feature(
                    feature_name, raw_data[feature_name]
                )
                processed_keys.add(feature_name)
                categorical_processed.append(feature_name)
                logger.debug(f"カテゴリカル特徴量 '{feature_name}' のエンコードが完了しました: "
                        f"形状={processed[feature_name].shape}, 型={processed[feature_name].dtype}")
        
                # 生データも保持する
                raw_key = f"raw_{feature_name}"
                # Object型の配列はテンソルに変換せずそのまま保持
                if isinstance(raw_data[feature_name], np.ndarray) and raw_data[feature_name].dtype == np.dtype('O'):
                    processed[raw_key] = raw_data[feature_name]
                else:
                    try:
                        # 変換可能な場合のみテンソルに変換
                        processed[raw_key] = torch.tensor(raw_data[feature_name]) if isinstance(raw_data[feature_name], np.ndarray) else raw_data[feature_name]
                    except TypeError:
                        # 変換できない場合は元の配列をそのまま保持
                        processed[raw_key] = raw_data[feature_name]
                        logger.debug(f"カテゴリカル特徴量 '{feature_name}' の生データをテンソルに変換できないため、元の形式で保存します")
                
                processed_keys.add(raw_key)
                raw_added.append(raw_key)
    
        # ターゲット変数をそのままコピー (target または target_で始まるキー)
        for key, value in raw_data.items():
            if key == 'target' or key.startswith('target_'):
                if isinstance(value, np.ndarray):
                    logger.debug(f"ターゲット変数 '{key}' をテンソルに変換します (原型: {value.dtype}, 形状: {value.shape})")
                    if value.dtype == np.dtype('O'):
                        # Object型の場合はテンソル変換をスキップ
                        processed[key] = value
                        logger.debug(f"ターゲット変数 '{key}' はnumpy.object_型のため、NumPy配列のまま保持します")
                    else:
                        try:
                            processed[key] = torch.tensor(value)
                            logger.debug(f"ターゲット変数 '{key}' の変換が完了しました: "
                                    f"形状={processed[key].shape}, 型={processed[key].dtype}")
                        except TypeError:
                            # 変換できない場合はNumPy配列のまま
                            processed[key] = value
                            logger.debug(f"ターゲット変数 '{key}' はテンソルに変換できないため、NumPy配列のまま保持します")
                else:
                    # NumPy配列でない場合はそのまま保存
                    logger.debug(f"ターゲット変数 '{key}' はNumPy配列でないため、そのままコピーします (型: {type(value).__name__})")
                    processed[key] = value
                processed_keys.add(key)
                target_copied.append(key)
        
        # 自動的に処理されていない残りのデータを含める
        # AUXILIARY_KEYS を明示的に定義する必要がなくなる
        for key, value in raw_data.items():
            if key not in processed_keys:
                # NumPy配列はテンソルに変換（対応可能な型のみ）
                if isinstance(value, np.ndarray):
                    # NumPy object配列はテンソルに変換せずそのまま保持
                    if value.dtype == np.dtype('O'):
                        logger.debug(f"補助データ '{key}' はnumpy.object_型のため、NumPy配列のまま保持します")
                        processed[key] = value
                    else:
                        try:
                            logger.debug(f"補助データ '{key}' をテンソルに変換します (原型: {value.dtype}, 形状: {value.shape})")
                            processed[key] = torch.tensor(value)
                            logger.debug(f"補助データ '{key}' の変換が完了しました: "
                                    f"形状={processed[key].shape}, 型={processed[key].dtype}")
                        except TypeError:
                            # 変換できない場合はNumPy配列のまま保持
                            logger.debug(f"補助データ '{key}' はテンソルに変換できないため、配列のまま保持します")
                            processed[key] = value
                else:
                    # その他のデータ型はそのままコピー
                    logger.debug(f"補助データ '{key}' はNumPy配列でないため、そのままコピーします (型: {type(value).__name__})")
                    processed[key] = value
                auxiliary_copied.append(key)
        
        # 処理結果の詳細をログに記録
        logger.info(f"処理された特徴量: 数値 {len(numerical_processed)}個, カテゴリ {len(categorical_processed)}個, "
                    f"ターゲット {len(target_copied)}個, 補助データ {len(auxiliary_copied)}個, 生データ {len(raw_added)}個")
        
        # 処理した特徴量の詳細をログに出力
        if numerical_processed:
            logger.info(f"処理された数値特徴量: {', '.join(numerical_processed)}")
        
        if categorical_processed:
            logger.info(f"処理されたカテゴリカル特徴量: {', '.join(categorical_processed)}")
        
        if target_copied:
            logger.info(f"コピー/変換されたターゲット変数: {', '.join(target_copied)}")
        
        if auxiliary_copied:
            logger.info(f"コピー/変換された補助データ: {', '.join(auxiliary_copied)}")
            
        if raw_added:
            logger.info(f"追加された生データ: {', '.join(raw_added)}")
        
        # メモリ使用状況の概要を出力
        total_memory = 0
        for key, value in processed.items():
            if isinstance(value, torch.Tensor):
                size_bytes = value.element_size() * value.nelement()
                total_memory += size_bytes
                if size_bytes > 1000000:  # 1MB以上のテンソルの場合
                    logger.debug(f"大きなテンソル '{key}': {size_bytes/1048576:.2f} MB")
        
        logger.info(f"処理後のデータの総メモリ使用量(テンソルのみ): {total_memory/1048576:.2f} MB")
        
        return processed
    
    def _apply_data_augmentation(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """データ拡張処理を適用します。トレーニングモードでのみ使用されます。
        
        Args:
            batch_data: バッチデータ
            
        Returns:
            拡張されたバッチデータ
        """
        # コピーして元のデータを変更しないようにする
        augmented_data = batch_data.copy()
        
        # 数値特徴量にノイズを追加
        for feature_name in self.config.numerical_features:
            if feature_name in augmented_data and isinstance(augmented_data[feature_name], torch.Tensor):
                tensor = augmented_data[feature_name]
                
                # NaNの場所を記録
                mask = ~torch.isnan(tensor)
                
                # 非NaN値にのみノイズを追加
                if mask.any():
                    # 特徴量はすでに正規化されているため、標準偏差は常に1.0
                    noise_scale = self.num_noise_scale
                    
                    # ノイズテンソルを生成
                    noise = torch.randn_like(tensor) * noise_scale
                    
                    # マスクを適用してNaNにはノイズを加えない
                    noise = noise.masked_fill(~mask, 0.0)
                    
                    # ノイズを追加
                    augmented_data[feature_name] = tensor + noise
                    
                    logger.debug(f"数値特徴量 '{feature_name}' にスケール {noise_scale:.4f} のノイズを追加しました")
                    
                    # ランダムにNaNに設定
                    # num_nan_probの確率で各要素をNaNに設定
                    nan_prob = self.num_nan_prob
                    nan_mask = torch.rand_like(tensor, dtype=torch.float) < nan_prob
                    
                    # 既にNaNになっている場所はマスクから除外
                    nan_mask = nan_mask & mask
                    
                    if nan_mask.any():
                        # マスクが適用される要素数を記録
                        num_nanned = nan_mask.sum().item()
                        
                        # NaNに設定
                        result = tensor.clone()
                        result[nan_mask] = float('nan')
                        augmented_data[feature_name] = result
                        
                        logger.debug(f"数値特徴量 '{feature_name}' の {num_nanned} 要素をランダムにNaNに設定しました")

        # カテゴリカル特徴量をランダムにNULLに設定
        for feature_name in self.config.categorical_features:
            if feature_name in augmented_data and isinstance(augmented_data[feature_name], torch.Tensor):
                tensor = augmented_data[feature_name]
                
                # ランダムマスクを作成（cat_null_probの確率で各要素をTrue）
                null_mask = torch.rand_like(tensor, dtype=torch.float) < self.cat_null_prob
                
                # 既に0（NULL）になっている場所はマスクから除外
                null_mask = null_mask & (tensor != 0)
                
                if null_mask.any():
                    # マスクが適用される要素数を記録
                    num_nulled = null_mask.sum().item()
                    
                    # 0（NULL）に設定
                    augmented_data[feature_name] = tensor.masked_fill(null_mask, 0)
                    
                    logger.debug(f"カテゴリカル特徴量 '{feature_name}' の {num_nulled} 要素をランダムにNULLに設定しました")

        return augmented_data
    
    def _init_batch_indices(self, shuffle=None) -> None:
        """バッチ処理用のインデックス配列を初期化します。
        
        Args:
            shuffle: シャッフルするかどうか。Noneの場合はモードに基づいて判定
        """
        data_size = self._get_data_size()
        self._batch_indices = np.arange(data_size)
        self._next_batch_index = 0
        
        # シャッフル判定
        if shuffle is None:
            # モードに基づいてシャッフル判定
            shuffle = (self.mode == self.MODE_TRAIN)
        
        if shuffle:
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
        # トレーニングモードの場合、データ拡張を適用
        if self.mode == self.MODE_TRAIN:
            return self._apply_data_augmentation(self.processed_data)
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