import torch
import pytest
from unittest.mock import MagicMock
from horguesc.core.modules.encoders import FeatureEncoder

class TestFeatureEncoder:
    @pytest.fixture
    def mock_config(self):
        """テスト用の設定オブジェクトを作成"""
        config = MagicMock()
        
        # 特徴量リスト
        config.numerical_features = ['age', 'income']
        config.categorical_features = ['gender', 'country']
        
        # グループ情報
        config.feature_groups = {
            'numerical': {
                'age': 'numeric_basic',
                'income': 'numeric_financial'
            },
            'categorical': {
                'gender': 'cat_personal',
                'country': 'cat_location'
            }
        }
        
        # グループ内の特徴量リスト
        config.group_features = {
            'numerical': {
                'numeric_basic': ['age'],
                'numeric_financial': ['income']
            },
            'categorical': {
                'cat_personal': ['gender'],
                'cat_location': ['country']
            }
        }
        
        # デフォルト設定
        config.getint.return_value = 8  # numerical_embedding_dim のデフォルト値
        
        return config
    
    @pytest.fixture
    def group_cardinalities(self):
        """テスト用のカーディナリティ辞書を作成"""
        return {
            'cat_personal': 3,
            'cat_location': 10
        }
    
    def test_encoder_initialization(self, mock_config, group_cardinalities):
        """エンコーダの初期化が正しく行われるかテスト"""
        encoder = FeatureEncoder(mock_config, group_cardinalities)
        
        # 数値特徴量グループの埋め込み層
        assert 'numeric_basic' in encoder.numerical_embeddings
        assert 'numeric_financial' in encoder.numerical_embeddings
        assert encoder.num_embedding_dims['numeric_basic'] == 8
        assert encoder.num_embedding_dims['numeric_financial'] == 8
        
        # カテゴリカル特徴量グループの埋め込み層
        assert 'cat_personal' in encoder.categorical_embeddings
        assert 'cat_location' in encoder.categorical_embeddings
        assert encoder.cat_embedding_dims['cat_personal'] > 0
        assert encoder.cat_embedding_dims['cat_location'] > 0
        
        # 出力次元
        assert encoder.numerical_total_dim == 16  # 8 + 8
        assert encoder.categorical_total_dim > 0
        assert encoder.output_dim == encoder.numerical_total_dim + encoder.categorical_total_dim
    
    def test_forward_1d_input(self, mock_config, group_cardinalities):
        """1次元入力での順伝播をテスト"""
        encoder = FeatureEncoder(mock_config, group_cardinalities)
        batch_size = 2
        
        # テスト用の入力特徴量を作成 (batch_size,) の形状
        features = {
            "age": torch.tensor([25.0, 30.0]),
            "income": torch.tensor([50000.0, 60000.0]),
            "gender": torch.tensor([1, 2]),
            "country": torch.tensor([5, 7]),
            "target": torch.tensor([0, 1])  # エンコードされない追加データ
        }
        
        # 順伝播
        output = encoder(features)
        
        # 出力形状の確認 - 入力が1次元なので出力は2次元 (batch_size, embedding_dim)
        assert output.dim() == 2
        assert output.shape[0] == batch_size
        assert output.shape[1] == encoder.output_dim
    
    def test_forward_2d_input(self, mock_config, group_cardinalities):
        """2次元入力での順伝播をテスト"""
        encoder = FeatureEncoder(mock_config, group_cardinalities)
        batch_size = 2
        seq_len = 3
        
        # テスト用の入力特徴量を作成 (batch_size, seq_len) の形状
        features = {
            "age": torch.randn(batch_size, seq_len),
            "income": torch.randn(batch_size, seq_len),
            "gender": torch.randint(0, 3, (batch_size, seq_len)),
            "country": torch.randint(0, 10, (batch_size, seq_len)),
            "target": torch.randint(0, 2, (batch_size, 1))  # エンコードされない追加データ
        }
        
        # 順伝播
        output = encoder(features)
        
        # 出力形状の確認 - 入力が2次元なので出力は3次元 (batch_size, seq_len, embedding_dim)
        assert output.dim() == 3
        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_len
        assert output.shape[2] == encoder.output_dim
    
    def test_forward_with_missing_features(self, mock_config, group_cardinalities):
        """一部の特徴量が欠けている場合のテスト"""
        encoder = FeatureEncoder(mock_config, group_cardinalities)
        batch_size = 2
        
        # 一部の特徴量のみを含む入力
        features = {
            "age": torch.tensor([25.0, 30.0]),
            "gender": torch.tensor([1, 2]),
            "target": torch.tensor([0, 1])  # エンコードされない追加データ
        }
        
        # 順伝播
        output = encoder(features)
        
        # 出力形状の確認
        assert output.shape[0] == batch_size
        # 仕様変更: encoder.forward は常に完全な出力次元を持つ
        assert output.shape[1] == encoder.output_dim
        
        # 埋め込まれた特徴量の位置の値がゼロ以外であることを確認
        age_start, age_end = encoder.feature_to_position["age"]
        gender_start, gender_end = encoder.feature_to_position["gender"]
        
        # age と gender の埋め込みは非ゼロであるべき
        assert not torch.allclose(output[:, age_start:age_end], torch.zeros_like(output[:, age_start:age_end]))
        assert not torch.allclose(output[:, gender_start:gender_end], torch.zeros_like(output[:, gender_start:gender_end]))
        
        # 欠損している特徴量の位置はゼロであるべき
        income_start, income_end = encoder.feature_to_position["income"]
        country_start, country_end = encoder.feature_to_position["country"]
        
        assert torch.allclose(output[:, income_start:income_end], torch.zeros_like(output[:, income_start:income_end]))
        assert torch.allclose(output[:, country_start:country_end], torch.zeros_like(output[:, country_start:country_end]))
    
    def test_suggest_embedding_dim(self, mock_config, group_cardinalities):
        """カーディナリティに基づく埋め込み次元の提案をテスト"""
        encoder = FeatureEncoder(mock_config, group_cardinalities)
        
        # 様々なカーディナリティでテスト
        cases = [
            (10, 4),    # 小さいカーディナリティ
            (100, 6),   # 中程度のカーディナリティ
            (10000, 20) # 大きいカーディナリティ
        ]
        
        for cardinality, expected_range in cases:
            dim = encoder._suggest_embedding_dim(cardinality)
            assert dim % 2 == 0  # 常に偶数
            assert dim >= 2      # 最小値は2
            assert dim <= 50     # 最大値は50
            # 期待される値の範囲内に収まっていることを確認（厳密な値ではなく範囲をチェック）
            assert abs(dim - expected_range) <= 4
    
    def test_forward_with_nan_values(self, mock_config, group_cardinalities):
        """NaN値が含まれる入力での順伝播をテスト"""
        encoder = FeatureEncoder(mock_config, group_cardinalities)
        batch_size = 2
        
        # NaN値を含むテスト用の入力特徴量を作成
        features = {
            "age": torch.tensor([25.0, float('nan')]),
            "income": torch.tensor([float('nan'), 60000.0]),
            "gender": torch.tensor([1, 2]),
            "country": torch.tensor([5, 0]),  # 0はパディング/不明値
            "target": torch.tensor([0, 1])    # エンコードされない追加データ
        }
        
        # 順伝播 - エラーなく実行できることを確認
        output = encoder(features)
        
        # 出力形状の確認
        assert output.dim() == 2
        assert output.shape[0] == batch_size
        assert output.shape[1] == encoder.output_dim
        
        # NaNを含む出力がないことを確認
        assert not torch.isnan(output).any(), "出力にNaN値が含まれています"