import torch
import torch.nn as nn
import logging
import math

logger = logging.getLogger(__name__)

class FeatureEncoder(nn.Module):
    """特徴量エンコーダークラス。
    
    数値特徴量とカテゴリカル特徴量を処理し、共通の表現に変換します。
    エンコードされる特徴量（numerical_featuresとcategorical_featuresに含まれるもの）は
    同じ形状であることを前提とします。
    
    入力データ仕様:
    - 数値特徴量: NaN値を含むことが許容されます。内部でこれらのNaN値は検出され、
      埋め込み時に0にマスクされます。出力にNaN値が伝播しないよう処理されます。
    - カテゴリカル特徴量: 0はパディングや不明値を表します。0以外の正の整数値が
      カテゴリのインデックスとして使用されます。
    """
    
    def __init__(self, config, group_cardinalities=None):
        """
        Args:
            config: アプリケーション設定
            group_cardinalities: グループ名とカーディナリティのマッピング辞書
        """
        super().__init__()
        self.config = config
        self.group_cardinalities = group_cardinalities or {}
        
        # 特徴量のグループ情報を取得
        self.feature_groups = config.feature_groups
        
        # スキップする特徴量のリストを取得
        self.skip_embedding_features = set(config.skip_embedding_features) if hasattr(config, 'skip_embedding_features') else set()
        
        # 実際に処理する特徴量を取得（スキップするものを除外）
        self.processed_numerical_features = [f for f in config.numerical_features if f not in self.skip_embedding_features]
        self.processed_categorical_features = [f for f in config.categorical_features if f not in self.skip_embedding_features]
        
        logger.info(f"スキップする特徴量: {len(self.skip_embedding_features)}個")
        
        # デフォルトの埋め込み次元
        self.default_numerical_dim = config.getint('model', 'numerical_embedding_dim', fallback=8)
        
        # 埋め込み層を一元管理 (グループごとに作成)
        self.numerical_embeddings = nn.ModuleDict()
        self.categorical_embeddings = nn.ModuleDict()
        
        # 埋め込み次元の記録
        self.num_embedding_dims = {}
        self.cat_embedding_dims = {}
        
        # 数値特徴量グループの埋め込み層を初期化
        processed_numerical_groups = set()
        for feature_name in self.processed_numerical_features:
            group_name = self.feature_groups['numerical'][feature_name]
            if group_name not in processed_numerical_groups:
                processed_numerical_groups.add(group_name)
                embedding_dim = self.default_numerical_dim
                
                # ゼロ初期化した線形層を作成
                linear = nn.Linear(1, embedding_dim)
                with torch.no_grad():
                    linear.weight.fill_(0.0)
                    linear.bias.fill_(0.0)
                
                self.numerical_embeddings[group_name] = nn.Sequential(
                    linear,
                    nn.ReLU()
                )
                # グループの埋め込み次元を記録
                self.num_embedding_dims[group_name] = embedding_dim
                
                logger.debug(f"数値グループ '{group_name}' の埋め込み層をゼロで初期化: dim={embedding_dim}, "
                            f"特徴量={len([f for f in self.processed_numerical_features if self.feature_groups['numerical'][f] == group_name])}個")
        
        # カテゴリカル特徴量グループの埋め込み層を初期化
        processed_categorical_groups = set()
        for feature_name in self.processed_categorical_features:
            group_name = self.feature_groups['categorical'][feature_name]
            if group_name not in processed_categorical_groups:
                processed_categorical_groups.add(group_name)
                try:
                    # カーディナリティの取得を試みる
                    cardinality = self.group_cardinalities[group_name]
                except KeyError:
                    # カーディナリティが指定されていない場合、デフォルト値を使用して警告
                    cardinality = 100  # デフォルト値
                    logger.warning(f"カテゴリグループ '{group_name}' のカーディナリティが指定されていません。"
                                  f"デフォルト値 ({cardinality}) を使用します。")
                
                # Add to _suggest_embedding_dim
                if cardinality > 10000:
                    # Use hashing trick or dimensionality reduction for extremely high cardinality
                    logger.warning(f"Very high cardinality ({cardinality}) detected for {group_name}")
                
                # カーディナリティに基づいて埋め込み次元を決定
                embedding_dim = self._suggest_embedding_dim(cardinality)
                
                # ゼロで初期化した埋め込み層を作成
                embedding = nn.Embedding(
                    num_embeddings=cardinality + 1,  # +1 はパディング/不明な値用
                    embedding_dim=embedding_dim,
                    padding_idx=0
                )
                with torch.no_grad():
                    embedding.weight.fill_(0.0)
                
                self.categorical_embeddings[group_name] = embedding
                
                # グループの埋め込み次元を記録
                self.cat_embedding_dims[group_name] = embedding_dim
                
                logger.debug(f"カテゴリグループ '{group_name}' の埋め込み層をゼロで初期化: cardinality={cardinality}, "
                           f"dim={embedding_dim}, 特徴量={len([f for f in self.processed_categorical_features if self.feature_groups['categorical'][f] == group_name])}個")
        
        # 出力次元の計算
        self.numerical_total_dim = sum(self.num_embedding_dims[self.feature_groups['numerical'][feature]] 
                                      for feature in self.processed_numerical_features)
        self.categorical_total_dim = sum(self.cat_embedding_dims[self.feature_groups['categorical'][feature]] 
                                      for feature in self.processed_categorical_features)
        self.output_dim = self.numerical_total_dim + self.categorical_total_dim
        
        # 特徴量ごとの埋め込み位置マッピングを作成
        self.feature_to_position = self._create_feature_position_mapping()
        
        logger.info(f"FeatureEncoder 初期化完了: 出力次元={self.output_dim} "
                   f"(数値特徴量={self.numerical_total_dim}, カテゴリ埋め込み={self.categorical_total_dim})")
        logger.info(f"処理される特徴量: 数値={len(self.processed_numerical_features)}個, カテゴリ={len(self.processed_categorical_features)}個")
        if self.skip_embedding_features:
            logger.info(f"スキップされる特徴量: {', '.join(self.skip_embedding_features)}")
    
    def _create_feature_position_mapping(self):
        """各特徴量の埋め込み結果が最終出力テンソルのどの位置に配置されるかのマッピングを作成します。
        
        Returns:
            dict: 特徴量名→(開始位置, 終了位置)のマッピング
        """
        mapping = {}
        current_position = 0
        
        # 数値特徴量の位置を決定
        for feature_name in self.processed_numerical_features:
            group_name = self.feature_groups['numerical'][feature_name]
            embed_dim = self.num_embedding_dims[group_name]
            
            mapping[feature_name] = (current_position, current_position + embed_dim)
            current_position += embed_dim
    
        # カテゴリカル特徴量の位置を決定
        for feature_name in self.processed_categorical_features:
            group_name = self.feature_groups['categorical'][feature_name]
            embed_dim = self.cat_embedding_dims[group_name]
            
            mapping[feature_name] = (current_position, current_position + embed_dim)
            current_position += embed_dim
        
        return mapping
    
    def _suggest_embedding_dim(self, cardinality):
        """カーディナリティに基づいて適切な埋め込み次元を提案します。
        
        一般的なルールとして、カーディナリティの4乗根に基づいて埋め込み次元を決定します。
        また、2の倍数に丸めて効率的な実装を可能にします。
        
        Args:
            cardinality: カテゴリの数
            
        Returns:
            int: 提案される埋め込み次元
        """
        # カーディナリティの4乗根を計算（経験則に基づく）
        suggested_dim = int(math.pow(cardinality, 0.25) * 2.0)
        
        # 2の倍数に切り上げ
        if suggested_dim % 2 != 0:
            suggested_dim += 1
            
        # 最小/最大の制限
        suggested_dim = max(2, min(50, suggested_dim))
        
        return suggested_dim
        
    def forward(self, features):
        """特徴量を埋め込み表現に変換します。
        
        エンコードされる特徴量は同じ形状である必要があります。
        最後の次元に特徴量埋め込みを追加します。
        skip_embedding_featuresに指定された特徴量は無視されます。
        
        例：入力が (batch_size,) の場合、出力は (batch_size, embedding_dim)
        例：入力が (batch_size, seq_len) の場合、出力は (batch_size, seq_len, embedding_dim)
        
        データの仕様:
        - 数値特徴量: NaN値を含むことができます。NaN値は検出され、0に置き換えられます。
          その後、NaN値に対応する位置の埋め込みベクトルが0にマスクされます。
        - カテゴリカル特徴量: 0は特別な値で、パディングや不明値を表します。
          0以外の整数値（1からcardinality）がカテゴリのインデックスとして使用されます。
        
        Args:
            features: {'feature_name': tensor, ...}の形式の特徴量辞書
            
        Returns:
            torch.Tensor: 結合された埋め込み表現
        """
        # 入力形状情報を抽出
        # スキップリストを除外して実際に処理する特徴量のみを対象とする
        encoder_features = set(self.processed_numerical_features + self.processed_categorical_features)
        valid_features = [f for f in encoder_features if f in features]
        
        if not valid_features:
            logger.warning("有効な特徴量が入力に含まれていません")
            # デフォルトのテンソルを返す
            device = next(self.parameters()).device
            return torch.zeros(1, self.output_dim, device=device)
        
        # Add validation that all features have compatible shapes
        shapes = {f: features[f].shape for f in features if f in valid_features}
        if len(set(shapes.values())) > 1:
            raise ValueError(f"Features have inconsistent shapes: {shapes}")
        
        # 形状情報を最初の有効な特徴量から取得
        first_feature = features[valid_features[0]]
        input_shape = first_feature.shape

        # 結果を格納するゼロテンソルを作成
        device = first_feature.device
        output_shape = list(input_shape) + [self.output_dim]
        combined = torch.zeros(output_shape, device=device)
        
        # 各特徴量の埋め込み計算と結果テンソルへの配置
        for feature_name in valid_features:
            # スキップリストに含まれる特徴量は処理しない
            if feature_name in self.skip_embedding_features:
                continue
                
            value = features[feature_name]
            
            if feature_name in self.processed_categorical_features:
                # この特徴量のグループを特定
                group_name = self.feature_groups['categorical'][feature_name]
                
                # カテゴリ特徴量の埋め込み
                original_shape = value.shape
                
                # 形状を平坦化して埋め込み
                flat_values = value.view(-1)
                flat_embeddings = self.categorical_embeddings[group_name](flat_values)
                
                # 元の形状に戻す + 埋め込み次元
                new_shape = list(original_shape) + [flat_embeddings.size(-1)]
                embedding = flat_embeddings.view(new_shape)
                
                # 結果テンソルの適切な位置に配置
                start_pos, end_pos = self.feature_to_position[feature_name]
                combined[..., start_pos:end_pos] = embedding
                
            elif feature_name in self.processed_numerical_features:
                # この特徴量のグループを特定
                group_name = self.feature_groups['numerical'][feature_name]
                
                # 数値特徴量の埋め込み
                original_shape = value.shape
                
                # NaN値を0に置き換えてマスクを作成（NaNの位置を記録）
                mask = ~torch.isnan(value)
                clean_value = torch.where(mask, value, torch.zeros_like(value))
                
                # 最後の次元に1を追加
                clean_value_unsqueezed = clean_value.unsqueeze(-1)
                
                # 形状を平坦化して埋め込み層に渡す
                flat_shape = (-1, 1)  # (total_elements, 1)
                flat_values = clean_value_unsqueezed.reshape(flat_shape)
                
                # 埋め込み
                flat_embeddings = self.numerical_embeddings[group_name](flat_values)
                
                # 元の形状に戻す
                embed_dim = flat_embeddings.size(-1)
                new_shape = list(original_shape) + [embed_dim]
                embedding = flat_embeddings.reshape(new_shape)
                
                # NaN値があった位置を0にマスク（マスクを拡張して埋め込み次元にも適用）
                if not mask.all():
                    # マスクを埋め込み次元に合わせて拡張
                    expanded_mask = mask.unsqueeze(-1).expand_as(embedding)
                    embedding = embedding * expanded_mask
                
                # 結果テンソルの適切な位置に配置
                start_pos, end_pos = self.feature_to_position[feature_name]
                combined[..., start_pos:end_pos] = embedding
        
        return combined
    
    def get_feature_embedding(self, feature_name, value):
        """Extract embedding for a single feature (useful for testing)
        
        Args:
            feature_name: 特徴量の名前
            value: 埋め込む値のテンソル
            
        Returns:
            torch.Tensor: 指定特徴量の埋め込み
        """
        # スキップする特徴量の場合はNoneを返す
        if feature_name in self.skip_embedding_features:
            logger.warning(f"特徴量 '{feature_name}' はスキップリストに含まれているため埋め込みを取得できません")
            return None
            
        if feature_name in self.processed_categorical_features:
            group_name = self.feature_groups['categorical'][feature_name]
            return self.categorical_embeddings[group_name](value)
        elif feature_name in self.processed_numerical_features:
            group_name = self.feature_groups['numerical'][feature_name]
            # NaN処理を適用
            mask = ~torch.isnan(value)
            clean_value = torch.where(mask, value, torch.zeros_like(value))
            embedding = self.numerical_embeddings[group_name](clean_value.unsqueeze(-1))
            
            # NaN値があった位置を0にマスク
            if not mask.all():
                expanded_mask = mask.unsqueeze(-1).expand_as(embedding)
                embedding = embedding * expanded_mask
            return embedding
        else:
            raise ValueError(f"Unknown feature: {feature_name}")