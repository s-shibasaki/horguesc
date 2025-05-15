import torch
import torch.nn as nn
import torch.nn.functional as F
from horguesc.core.base.model import BaseModel
import logging
import itertools

logger = logging.getLogger(__name__)

class TrifectaModel(BaseModel):
    """
    3連単予測モデル。
    各レースの出走馬から1〜3着の順番を予測する。
    """
    
    def __init__(self, config, encoder):
        """
        Args:
            config: 設定オブジェクト
            encoder: 特徴量エンコーダー
        """
        super().__init__(config, encoder)
        
        # 設定からハイパーパラメータを取得
        self.embedding_dim = self.encoder.output_dim if self.encoder else 64
        self.hidden_dim = config.getint('model.trifecta', 'hidden_dim', fallback=128)
        self.dropout_rate = config.getfloat('model.trifecta', 'dropout_rate', fallback=0.2)
        
        # 各馬の特徴を処理するネットワーク
        self.horse_network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # 馬同士の組み合わせを考慮する注意機構
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # 3連単の組み合わせを予測する出力層
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),  # 3頭分の特徴を結合
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 1)  # 各組み合わせの確率を出力
        )
    
    def get_name(self):
        """モデルの名前を取得"""
        return "TrifectaModel"
    
    def forward(self, inputs):
        """
        順伝播処理
        
        Args:
            inputs: 入力データ辞書
            
        Returns:
            dict: {
                'logits': 各3連単組み合わせのロジット (batch_size, num_permutations)
                'permutations': 各組み合わせの馬番リスト [(1,2,3), (1,2,4), ...]
            }
        """
        # 特徴量エンコーダーで特徴量を埋め込み表現に変換
        # inputs から umaban を取り出しておく（後で使用）
        umaban = inputs.get('umaban')
        
        # エンコーダーで特徴量を埋め込み表現に変換
        embedded = self.encoder(inputs)  # (batch_size, max_horses, embed_dim)
        
        batch_size, max_horses, _ = embedded.shape
        
        # 各馬の特徴を処理
        horse_features = self.horse_network(embedded)  # (batch_size, max_horses, hidden_dim)
        
        # 注意機構で馬同士の関係性を学習
        attended_features, _ = self.attention(
            horse_features, horse_features, horse_features,
            key_padding_mask=(umaban == 0)  # パディング（umaban=0）部分をマスク
        )
        
        # 最終的な各馬の表現を取得（残差接続で元の特徴と結合）
        final_features = horse_features + attended_features  # (batch_size, max_horses, hidden_dim)
        
        # 欠損値の処理（パディングされた部分の特徴量をゼロにする）
        mask = (umaban != 0).unsqueeze(-1).float()  # (batch_size, max_horses, 1)
        final_features = final_features * mask  # (batch_size, max_horses, hidden_dim)
        
        # 3連単の順列を生成
        permutations = list(itertools.permutations(range(1, max_horses + 1), 3))
        
        # バッチ内の各レースについて全順列の確率を計算
        all_logits = []
        
        # 各バッチでループ
        for i in range(batch_size):
            race_features = final_features[i]  # (max_horses, hidden_dim)
            race_logits = []
            
            # 各順列でループ
            for perm in permutations:
                # 順列に含まれる馬のインデックス（0-indexed）
                # umaban（馬番）は1から始まるため-1でインデックスに変換
                idx1, idx2, idx3 = perm[0] - 1, perm[1] - 1, perm[2] - 1
                
                # それぞれの馬の特徴を取得
                horse1_feat = race_features[idx1]  # (hidden_dim)
                horse2_feat = race_features[idx2]  # (hidden_dim)
                horse3_feat = race_features[idx3]  # (hidden_dim)
                
                # 3頭の特徴を結合
                combined = torch.cat([horse1_feat, horse2_feat, horse3_feat], dim=0)  # (hidden_dim * 3)
                
                # この順列の確率を計算
                logit = self.output_layer(combined)
                race_logits.append(logit.squeeze())
            
            # 全順列の確率をテンソルにまとめる
            race_logits = torch.stack(race_logits)  # (num_permutations)
            all_logits.append(race_logits)
        
        # バッチ全体のロジットをテンソルにまとめる
        logits = torch.stack(all_logits)  # (batch_size, num_permutations)
        
        return {
            'logits': logits,
            'permutations': permutations
        }
    
    def compute_loss(self, outputs, targets):
        """
        3連単予測の損失を計算
        
        Args:
            outputs: forward()の出力
            targets: ターゲットデータ（'target_trifecta'を含む）
            
        Returns:
            損失値
        """
        logits = outputs['logits']  # (batch_size, num_permutations)
        target_indices = targets['target_trifecta']  # (batch_size)
        
        # クロスエントロピー損失を計算
        loss = F.cross_entropy(logits, target_indices)
        return loss
    
    def compute_metrics(self, outputs, targets):
        """
        評価用メトリクスを計算
        
        Args:
            outputs: forward()の出力
            targets: ターゲットデータ
            
        Returns:
            dict: メトリクス
        """
        logits = outputs['logits']  # (batch_size, num_permutations)
        target_indices = targets['target_trifecta']  # (batch_size)
        
        # 予測した最も確率の高い順列のインデックス
        _, predicted_indices = torch.max(logits, dim=1)
        
        # 正解率を計算
        correct = (predicted_indices == target_indices).float()
        accuracy = correct.mean().item()
        
        return {
            'accuracy': accuracy
        }