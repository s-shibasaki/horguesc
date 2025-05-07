import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class FeatureEncoder(nn.Module):
    """特徴量エンコーダークラス。
    
    数値特徴量とカテゴリカル特徴量を処理し、共通の表現に変換します。
    """
    
    def __init__(self, config, feature_processor):
        """
        Args:
            config: アプリケーション設定
            feature_processor: 特徴量処理のインスタンス
        """
        super().__init__()
        self.config = config
        
        # 特徴量プロセッサ
        self.feature_processor = feature_processor
        
        # 特徴量設定はプロセッサから取得
        self.feature_config = feature_processor.feature_config
        
        # 数値特徴量の埋め込み次元（設定から取得またはデフォルト値）
        self.numerical_embedding_dim = self.config.getint(
            'embeddings', 'numerical_embedding_dim', fallback=8
        )
        
        # 数値特徴量の埋め込み層
        self.numerical_embeddings = nn.ModuleDict()
        for name in self.feature_config.get_numerical_feature_names():
            self.numerical_embeddings[name] = nn.Sequential(
                nn.Linear(1, self.numerical_embedding_dim),
                nn.ReLU()  # 非線形性を追加
            )
        
        # カテゴリカル特徴量の埋め込み層
        self.embedding_layers = nn.ModuleDict()
        
        # カテゴリカル特徴量ごとに埋め込み層を作成
        for name, info in self.feature_config.categorical_features.items():
            cardinality = info['cardinality']
            embedding_dim = info.get('embedding_dim', self.numerical_embedding_dim)  # 数値特徴量と同じ次元を使用
            
            # カーディナリティ+1で初期化（0はNone/未知の値用）
            self.embedding_layers[name] = nn.Embedding(
                num_embeddings=cardinality + 1,
                embedding_dim=embedding_dim,
                padding_idx=0
            )
            logger.debug(f"特徴量 {name} の埋め込み層を作成: cardinality={cardinality}, dim={embedding_dim}")
        
        # 出力次元の計算
        self.numerical_total_dim = len(self.feature_config.numerical_features) * self.numerical_embedding_dim
        self.embedding_total_dim = sum(
            info.get('embedding_dim', self.numerical_embedding_dim) 
            for info in self.feature_config.categorical_features.values()
        )
        self.output_dim = self.numerical_total_dim + self.embedding_total_dim
        
        logger.info(f"FeatureEncoder 初期化完了: 出力次元={self.output_dim} "
                   f"(数値特徴量={self.numerical_total_dim}, カテゴリ埋め込み={self.embedding_total_dim})")
    
    def forward(self, features):
        """特徴量エンコーディングの順伝播。
        
        Args:
            features: 特徴量の辞書
                - 数値特徴量: バッチ×特徴数のテンソル（NaNを含む可能性あり）
                - カテゴリカル特徴量: インデックスのテンソル
        
        Returns:
            torch.Tensor: エンコードされた特徴量
        """
        # バッチサイズとデバイスの取得
        first_tensor = next(iter(features.values()))
        batch_size = first_tensor.shape[0]
        device = first_tensor.device
        
        # 数値特徴量の処理
        numerical_embeddings = []
        
        for name in self.feature_config.get_numerical_feature_names():
            if name in features and name in self.numerical_embeddings:
                values = features[name]
                
                # NaNを検出してマスクを作成
                mask = ~torch.isnan(values)
                
                # NaNを0に置換
                values = torch.nan_to_num(values, nan=0.0)
                
                # バッチ×1 のテンソルを確保
                if values.dim() == 1:
                    values = values.unsqueeze(1)
                    mask = mask.unsqueeze(1)
                
                # Linear埋め込み層を適用
                embedded = self.numerical_embeddings[name](values)
                
                # マスクを拡張して埋め込み次元に合わせる
                expanded_mask = mask.expand_as(embedded)
                
                # マスクを適用（欠損値の埋め込みを0に）
                masked_embedding = embedded * expanded_mask.float()
                
                numerical_embeddings.append(masked_embedding)
        
        # 数値特徴量の埋め込みがない場合は0テンソルを使用
        if not numerical_embeddings:
            numerical_encoded = torch.zeros((batch_size, 0), device=device)
        else:
            numerical_encoded = torch.cat(numerical_embeddings, dim=1)
        
        # カテゴリカル特徴量の埋め込み
        categorical_embeddings = []
        for name in self.feature_config.get_categorical_feature_names():
            if name in features and name in self.embedding_layers:
                # インデックスを取得
                indices = features[name]
                if indices.dim() > 1:
                    indices = indices.squeeze(1)
                
                # 埋め込み適用
                embedding = self.embedding_layers[name](indices)
                categorical_embeddings.append(embedding)
        
        # カテゴリカル埋め込みがない場合は0テンソルを使用
        if not categorical_embeddings:
            categorical_encoded = torch.zeros((batch_size, 0), device=device)
        else:
            categorical_encoded = torch.cat(categorical_embeddings, dim=1)
        
        # 数値特徴量とカテゴリカル特徴量の埋め込みを連結
        encoded_features = torch.cat([numerical_encoded, categorical_encoded], dim=1)
        
        return encoded_features