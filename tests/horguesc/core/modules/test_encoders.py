import torch
import pytest
from unittest.mock import MagicMock
from horguesc.core.modules.encoders import FeatureEncoder

class TestFeatureEncoder:
    @pytest.fixture
    def mock_config(self):
        """テスト用の設定オブジェクトを作成"""
        config = MagicMock()
        
        # 数値特徴量の設定
        config.get_numerical_feature_names.return_value = ["age", "income"]
        config.numerical_features = {
            "age": {"embedding_dim": 4},
            "income": {"embedding_dim": 8}
        }
        
        # カテゴリカル特徴量の設定
        config.get_categorical_feature_names.return_value = ["gender", "country"]
        config.categorical_features = {
            "gender": {"cardinality": 3, "embedding_dim": 2},
            "country": {"cardinality": 10, "embedding_dim": 5}
        }
        
        return config
    
    def test_encoder_initialization(self, mock_config):
        """エンコーダの初期化が正しく行われるかテスト"""
        encoder = FeatureEncoder(mock_config)
        
        # 数値特徴量の埋め込み層
        assert "age" in encoder.numerical_embeddings
        assert "income" in encoder.numerical_embeddings
        assert encoder.num_embedding_dims["age"] == 4
        assert encoder.num_embedding_dims["income"] == 8
        
        # カテゴリカル特徴量の埋め込み層
        assert "gender" in encoder.embedding_layers
        assert "country" in encoder.embedding_layers
        assert encoder.cat_embedding_dims["gender"] == 2
        assert encoder.cat_embedding_dims["country"] == 5
        
        # 出力次元
        assert encoder.numerical_total_dim == 12  # 4 + 8
        assert encoder.embedding_total_dim == 7   # 2 + 5
        assert encoder.output_dim == 19           # 12 + 7
    
    def test_forward_basic(self, mock_config):
        """基本的な順伝播をテスト"""
        encoder = FeatureEncoder(mock_config)
        batch_size = 2
        
        # テスト用の入力特徴量を作成 (batch_size,) の形状
        features = {
            "age": torch.tensor([25.0, 30.0]),
            "income": torch.tensor([50000.0, 60000.0]),
            "gender": torch.tensor([1, 2]),
            "country": torch.tensor([5, 7])
        }
        
        # 順伝播
        output = encoder(features)
        
        # 出力形状の確認
        assert output.shape == (batch_size, encoder.output_dim)
    
    def test_forward_with_missing_features(self, mock_config):
        """一部の特徴量が欠けている場合のテスト"""
        encoder = FeatureEncoder(mock_config)
        batch_size = 2
        
        # 一部の特徴量のみ含む入力 (batch_size,) の形状
        features = {
            "age": torch.tensor([25.0, 30.0]),
            "gender": torch.tensor([1, 2])
        }
        
        # 順伝播
        output = encoder(features)
        
        # 出力形状の確認 - age(4次元)とgender(2次元)のみなので合計6次元
        expected_dim = 6  # age(4) + gender(2)
        assert output.shape == (batch_size, expected_dim)
    
    def test_forward_with_nan_values(self, mock_config):
        """NaN値を含む数値特徴量のテスト"""
        encoder = FeatureEncoder(mock_config)
        batch_size = 2
        
        # NaN値を含む入力 (batch_size,) の形状
        features = {
            "age": torch.tensor([25.0, float('nan')]),
            "income": torch.tensor([50000.0, 60000.0]),
            "gender": torch.tensor([1, 2])
        }
        
        # 順伝播
        output = encoder(features)
        
        # 出力形状の確認 - 全ての特徴量を含まないので出力次元も少なくなる
        expected_dim = 14  # age(4) + income(8) + gender(2)
        assert output.shape == (batch_size, expected_dim)
        # NaNが正しく処理されていることを確認
        assert not torch.isnan(output).any()
    
    def test_forward_multidimensional(self, mock_config):
        """多次元入力のテスト"""
        encoder = FeatureEncoder(mock_config)
        batch_size = 2
        seq_len = 3
        
        # 多次元入力（例: シーケンスデータ）- countryなし
        features = {
            "age": torch.randn(batch_size, seq_len),
            "income": torch.randn(batch_size, seq_len),
            "gender": torch.randint(0, 3, (batch_size, seq_len)),
        }
        
        # 順伝播
        output = encoder(features)
        
        # 出力形状の確認 - countryがないので出力次元数は減少
        expected_dim = 14  # age(4) + income(8) + gender(2)
        assert output.shape == (batch_size, seq_len, expected_dim)