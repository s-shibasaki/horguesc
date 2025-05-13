import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class FeatureEncoder(nn.Module):
    """特徴量エンコーダークラス。
    
    数値特徴量とカテゴリカル特徴量を処理し、共通の表現に変換します。
    複数次元の入力にも対応します。
    """
    
    def __init__(self, config):
        """
        Args:
            config: アプリケーション設定
        """
        super().__init__()
        self.config = config
        
        # 数値特徴量の埋め込み層
        self.numerical_feature_names = self.config.get_numerical_feature_names()
        self.numerical_embeddings = nn.ModuleDict()
        self.num_embedding_dims = {}
        for name, info in self.config.numerical_features.items():
            embedding_dim = info['embedding_dim']
            self.num_embedding_dims[name] = embedding_dim
            
            self.numerical_embeddings[name] = nn.Sequential(
                nn.Linear(1, embedding_dim),
                nn.ReLU()  # 非線形性を追加
            )
            logger.debug(f"数値特徴量 {name} の埋め込み層を作成: dim={embedding_dim}")
        
        # カテゴリカル特徴量の埋め込み層と情報を保存
        self.categorical_feature_names = self.config.get_categorical_feature_names()
        self.embedding_layers = nn.ModuleDict()
        self.cat_embedding_dims = {}
        for name, info in self.config.categorical_features.items():
            cardinality = info['cardinality']
            embedding_dim = info['embedding_dim']
            self.cat_embedding_dims[name] = embedding_dim
            
            # カーディナリティ+1で初期化（0はNone/未知の値用）
            self.embedding_layers[name] = nn.Embedding(
                num_embeddings=cardinality + 1,
                embedding_dim=embedding_dim,
                padding_idx=0
            )
            logger.debug(f"特徴量 {name} の埋め込み層を作成: cardinality={cardinality}, dim={embedding_dim}")
        
        # 出力次元の計算 - 保存した情報を使用
        self.numerical_total_dim = sum(self.num_embedding_dims.values())
        self.embedding_total_dim = sum(self.cat_embedding_dims.values())
        self.output_dim = self.numerical_total_dim + self.embedding_total_dim
        
        logger.info(f"FeatureEncoder 初期化完了: 出力次元={self.output_dim} "
                   f"(数値特徴量={self.numerical_total_dim}, カテゴリ埋め込み={self.embedding_total_dim})")
    
    def forward(self, features):
        """特徴量エンコーディングの順伝播。
        
        Args:
            features: 特徴量の辞書
                - 数値特徴量: (batch_size, [additional_dim_1, ..., additional_dim_n]) のテンソル（NaNを含む可能性あり）
                - カテゴリカル特徴量: (batch_size, [additional_dim_1, ..., additional_dim_n]) のインデックスのテンソル
        
        Returns:
            torch.Tensor: エンコードされた特徴量 (batch_size, [additional_dim_1, ..., additional_dim_n,] output_dim)
        """
        # バッチサイズとデバイスの取得
        first_tensor = next(iter(features.values()))
        batch_size = first_tensor.shape[0]
        device = first_tensor.device
        
        # 入力の追加次元を抽出
        additional_dims = []
        if first_tensor.dim() > 1:
            additional_dims = list(first_tensor.shape[1:])
        
        # 出力テンソルのサイズを計算
        output_shape = [batch_size] + additional_dims + [self.output_dim]
        additional_shape = [batch_size] + additional_dims
        
        # 数値特徴量の処理
        numerical_embeddings = []
        for name in self.numerical_feature_names:
            if name in features and name in self.numerical_embeddings:
                values = features[name]
                original_shape = values.shape
                
                # NaNを検出してマスクを作成
                mask = ~torch.isnan(values)
                
                # NaNを0に置換
                values = torch.nan_to_num(values, nan=0.0)
                
                # テンソルをフラット化して処理
                flat_values = values.reshape(-1, 1)
                flat_mask = mask.reshape(-1, 1)
                
                # Linear埋め込み層を適用
                embedded = self.numerical_embeddings[name](flat_values)
                
                # マスクを拡張して埋め込み次元に合わせる
                expanded_mask = flat_mask.expand_as(embedded)
                
                # マスクを適用（欠損値の埋め込みを0に）
                masked_embedding = embedded * expanded_mask.float()
                
                # 元の形状に戻して埋め込み次元を新たな次元として追加
                # (batch_size, [additional_dims]) -> (batch_size, [additional_dims], embedding_dim)
                embedding_dim = self.num_embedding_dims[name]
                reshaped_embedding = masked_embedding.reshape(*original_shape, embedding_dim)
                numerical_embeddings.append(reshaped_embedding)
        
        # 数値特徴量の埋め込みがない場合は0テンソルを使用
        if not numerical_embeddings:
            numerical_encoded = torch.zeros((*additional_shape, 0), device=device)
        else:
            numerical_encoded = torch.cat(numerical_embeddings, dim=-1)
        
        # カテゴリカル特徴量の埋め込み
        categorical_embeddings = []
        for name in self.categorical_feature_names:
            if name in features and name in self.embedding_layers:
                # インデックスを取得
                indices = features[name]
                original_shape = indices.shape
                
                # テンソルをフラット化して処理
                flat_indices = indices.reshape(-1)
                
                # 埋め込み適用
                embedding = self.embedding_layers[name](flat_indices)
                
                # 元の形状に戻して埋め込み次元を新たな次元として追加
                # (batch_size, [additional_dims]) -> (batch_size, [additional_dims], embedding_dim)
                embedding_dim = self.cat_embedding_dims[name]
                reshaped_embedding = embedding.reshape(*original_shape, embedding_dim)
                categorical_embeddings.append(reshaped_embedding)
        
        # カテゴリカル埋め込みがない場合は0テンソルを使用
        if not categorical_embeddings:
            categorical_encoded = torch.zeros((*additional_shape, 0), device=device)
        else:
            categorical_encoded = torch.cat(categorical_embeddings, dim=-1)
        
        # 数値特徴量とカテゴリカル特徴量の埋め込みを連結
        encoded_features = torch.cat([numerical_encoded, categorical_encoded], dim=-1)
        
        return encoded_features