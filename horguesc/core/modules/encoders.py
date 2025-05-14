import torch
import torch.nn as nn
import logging
import math

logger = logging.getLogger(__name__)

class FeatureEncoder(nn.Module):
    """特徴量エンコーダークラス。
    
    数値特徴量とカテゴリカル特徴量を処理し、共通の表現に変換します。
    複数次元の入力にも対応します。
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
            
            # グループ内の特徴量も同じ埋め込み次元を持つ
            for feature_name in config.group_features['numerical'][group_name]:
                self.num_embedding_dims[feature_name] = embedding_dim
                
            logger.debug(f"数値グループ '{group_name}' の埋め込み層を作成: dim={embedding_dim}, "
                        f"特徴量={len(config.group_features['numerical'][group_name])}個")
        
        # カテゴリカル特徴量グループの埋め込み層を初期化
        for group_name in config.group_features['categorical']:
            # カーディナリティの取得
            cardinality = self._get_group_cardinality(group_name)
            
            # カーディナリティに基づいて埋め込み次元を決定
            embedding_dim = self._suggest_embedding_dim(cardinality)
            
            self.categorical_embeddings[group_name] = nn.Embedding(
                num_embeddings=cardinality + 1,  # +1 はパディング/不明な値用
                embedding_dim=embedding_dim,
                padding_idx=0
            )
            
            # グループの埋め込み次元を記録
            self.cat_embedding_dims[group_name] = embedding_dim
            
            # グループ内の特徴量も同じ埋め込み次元を持つ
            for feature_name in config.group_features['categorical'][group_name]:
                self.cat_embedding_dims[feature_name] = embedding_dim
            
            logger.debug(f"カテゴリグループ '{group_name}' の埋め込み層を作成: cardinality={cardinality}, "
                       f"dim={embedding_dim}, 特徴量={len(config.group_features['categorical'][group_name])}個")
        
        # 出力次元の計算
        self.numerical_total_dim = sum(self.num_embedding_dims[feature] for feature in self.config.numerical_features)
        self.embedding_total_dim = sum(self.cat_embedding_dims[feature] for feature in self.config.categorical_features)
        self.output_dim = self.numerical_total_dim + self.embedding_total_dim
        
        logger.info(f"FeatureEncoder 初期化完了: 出力次元={self.output_dim} "
                   f"(数値特徴量={self.numerical_total_dim}, カテゴリ埋め込み={self.embedding_total_dim})")
    
    def _get_group_cardinality(self, group_name):
        """グループのカーディナリティを取得します。
        
        Args:
            group_name: グループの名前
            
        Returns:
            int: カーディナリティ
        """
        # 渡されたカーディナリティ情報から取得
        if self.group_cardinalities and group_name in self.group_cardinalities:
            cardinality = self.group_cardinalities[group_name]
            logger.debug(f"グループ {group_name} のカーディナリティをマッピングから取得: {cardinality}")
            return cardinality
        
        # 取得できない場合はデフォルト値を使用
        cardinality = 10  # デフォルト値
        logger.debug(f"グループ {group_name} のカーディナリティにデフォルト値を使用: {cardinality}")
        return cardinality
    
    def _suggest_embedding_dim(self, cardinality):
        """カーディナリティに基づいて適切な埋め込み次元を提案します。
        
        一般的なルールとして、カーディナリティの4乗根に基づいて埋め込み次元を決定します。
        また、2の倍数に丸めて効率的な実装を可能にします。
        
        Args:
            cardinality: カテゴリの数
            
        Returns:
            int: 提案される埋め込み次元
        """
        if cardinality <= 1:
            return 2  # 最小埋め込み次元
            
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
        
        Args:
            features: {'feature_name': tensor, ...}の形式の特徴量辞書
            
        Returns:
            torch.Tensor: 結合された埋め込み表現
        """
        # 各カテゴリと数値の埋め込み結果を格納するリスト
        embeddings = []
        
        # 各特徴量の埋め込みを計算
        for feature_name, value in features.items():
            if feature_name in self.config.categorical_features:
                # この特徴量のグループを特定
                group_name = self.feature_groups['categorical'][feature_name]
                embedding = self.categorical_embeddings[group_name](value)
                embeddings.append(embedding)
                
            elif feature_name in self.config.numerical_features:
                # この特徴量のグループを特定
                group_name = self.feature_groups['numerical'][feature_name]
                # (batch_size, ...) -> (batch_size, ..., 1) に変換して埋め込み
                value_unsqueezed = value.unsqueeze(-1)
                embedding = self.numerical_embeddings[group_name](value_unsqueezed)
                embeddings.append(embedding)
        
        # すべての埋め込みを結合
        if embeddings:
            # 各埋め込みを平坦化して結合
            flattened_embeddings = [emb.view(emb.size(0), -1) for emb in embeddings]
            combined = torch.cat(flattened_embeddings, dim=1)
            return combined
        
        # 特徴量がない場合は0テンソルを返す
        return torch.zeros(1, self.output_dim, device=next(self.parameters()).device)