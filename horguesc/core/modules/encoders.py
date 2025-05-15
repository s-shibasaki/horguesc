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
        
        # デフォルトの埋め込み次元
        self.default_numerical_dim = config.getint('model', 'numerical_embedding_dim', fallback=8)
        
        # 埋め込み層を一元管理 (グループごとに作成)
        self.numerical_embeddings = nn.ModuleDict()
        self.categorical_embeddings = nn.ModuleDict()
        
        # 埋め込み次元の記録
        self.num_embedding_dims = {}
        self.cat_embedding_dims = {}
        
        # 数値特徴量グループの埋め込み層を初期化
        for group_name in config.group_features['numerical']:
            embedding_dim = self.default_numerical_dim
            
            self.numerical_embeddings[group_name] = nn.Sequential(
                nn.Linear(1, embedding_dim),
                nn.ReLU()
            )
            # グループの埋め込み次元を記録
            self.num_embedding_dims[group_name] = embedding_dim
            
            logger.debug(f"数値グループ '{group_name}' の埋め込み層を作成: dim={embedding_dim}, "
                        f"特徴量={len(config.group_features['numerical'][group_name])}個")
        
        # カテゴリカル特徴量グループの埋め込み層を初期化
        for group_name in config.group_features['categorical']:
            try:
                # カーディナリティの取得を試みる
                cardinality = self.group_cardinalities[group_name]
            except KeyError:
                # カーディナリティが指定されていない場合、デフォルト値を使用して警告
                cardinality = 100  # デフォルト値
                logger.warning(f"カテゴリグループ '{group_name}' のカーディナリティが指定されていません。"
                              f"デフォルト値 ({cardinality}) を使用します。")
            
            # カーディナリティに基づいて埋め込み次元を決定
            embedding_dim = self._suggest_embedding_dim(cardinality)
            
            self.categorical_embeddings[group_name] = nn.Embedding(
                num_embeddings=cardinality + 1,  # +1 はパディング/不明な値用
                embedding_dim=embedding_dim,
                padding_idx=0
            )
            
            # グループの埋め込み次元を記録
            self.cat_embedding_dims[group_name] = embedding_dim
            
            logger.debug(f"カテゴリグループ '{group_name}' の埋め込み層を作成: cardinality={cardinality}, "
                       f"dim={embedding_dim}, 特徴量={len(config.group_features['categorical'][group_name])}個")
        
        # 出力次元の計算
        self.numerical_total_dim = sum(self.num_embedding_dims[self.feature_groups['numerical'][feature]] 
                                      for feature in self.config.numerical_features)
        self.embedding_total_dim = sum(self.cat_embedding_dims[self.feature_groups['categorical'][feature]] 
                                      for feature in self.config.categorical_features)
        self.output_dim = self.numerical_total_dim + self.embedding_total_dim
        
        logger.info(f"FeatureEncoder 初期化完了: 出力次元={self.output_dim} "
                   f"(数値特徴量={self.numerical_total_dim}, カテゴリ埋め込み={self.embedding_total_dim})")
    
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
        
        エンコードされる特徴量（numerical_featuresとcategorical_featuresに含まれるもの）は
        同じ形状である必要があります。
        最後の次元に特徴量埋め込みを追加します。
        
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
        # 各特徴量の埋め込み結果を格納するリスト
        embeddings = []
        batch_size = None
        input_shape = None
        
        # エンコードする特徴量の形状を確認
        encoder_features = []
        for feature_name in features:
            if feature_name in self.config.numerical_features or feature_name in self.config.categorical_features:
                encoder_features.append(feature_name)
                if input_shape is None:
                    input_shape = features[feature_name].shape
                    batch_size = input_shape[0]
        
        # エンコードする特徴量の形状が一致するか確認
        for feature_name in encoder_features:
            if features[feature_name].shape != input_shape:
                raise ValueError(f"エンコードされる特徴量は同じ形状である必要があります。"
                               f"'{feature_name}'の形状 {features[feature_name].shape} が"
                               f"他のエンコード特徴量の形状 {input_shape} と異なります。")
        
        # 各特徴量の埋め込みを計算
        for feature_name in encoder_features:
            value = features[feature_name]
            
            if feature_name in self.config.categorical_features:
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
                embeddings.append(embedding)
                
            elif feature_name in self.config.numerical_features:
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
                
                embeddings.append(embedding)
        
        # すべての埋め込みを最後の次元で結合
        if embeddings:
            combined = torch.cat(embeddings, dim=-1)
            return combined
        
        # 特徴量がない場合は0テンソルを返す
        device = next(self.parameters()).device
        if input_shape:
            # 入力と同じ形状 + 出力次元
            output_shape = list(input_shape) + [self.output_dim]
            return torch.zeros(*output_shape, device=device)
        else:
            # デフォルトの形状
            return torch.zeros(batch_size or 1, self.output_dim, device=device)