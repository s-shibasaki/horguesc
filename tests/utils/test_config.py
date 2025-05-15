import os
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock

# テスト対象のモジュールをインポート
from horguesc.utils.config import Config, load_config, find_config

class TestConfig:
    @pytest.fixture
    def sample_config_file(self):
        """テスト用の設定ファイルを作成するフィクスチャ"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ini') as f:
            f.write("""
[features]
numerical_features = feature1, feature2, feature3
categorical_features = cat1, cat2

[groups]
group1 = feature1, feature2
group2 = cat1, cat2

[training]
train_start_date = 2023-01-01
train_end_date = 2023-03-31
val_start_date = 2023-04-01
val_end_date = 2023-04-30
            """)
            temp_path = f.name
        
        yield temp_path
        
        # テスト後にファイルを削除
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            
    def test_config_load_from_file(self, sample_config_file):
        """設定ファイルからの読み込みをテスト"""
        config = Config()
        config.load_from_file(sample_config_file)
        
        assert config.has_section('features')
        assert config.has_option('features', 'numerical_features')
        assert 'feature1, feature2, feature3' == config.get('features', 'numerical_features')
    
    def test_parse_features(self, sample_config_file):
        """feature解析のテスト"""
        config = Config().load_from_file(sample_config_file).parse_features()
        
        # 数値特徴量のテスト
        assert config.numerical_features == ['feature1', 'feature2', 'feature3']
        assert config.categorical_features == ['cat1', 'cat2']
        
        # グループのテスト
        assert config.feature_groups['numerical']['feature1'] == 'group1'
        assert config.feature_groups['numerical']['feature2'] == 'group1'
        assert config.feature_groups['numerical']['feature3'] == 'feature3'  # 自分自身がグループ
        
        # グループごとの特徴量リスト
        assert set(config.group_features['numerical']['group1']) == {'feature1', 'feature2'}
    
    def test_validate_dates_valid(self, sample_config_file):
        """有効な日付フォーマットの検証テスト"""
        config = Config().load_from_file(sample_config_file)
        # 例外が発生しないことを確認
        config._validate_dates()
    
    def test_validate_dates_invalid(self):
        """無効な日付フォーマットの検証テスト"""
        config = Config()
        config.add_section('training')
        config.set('training', 'train_start_date', '2023/01/01')  # 不正なフォーマット
        
        with pytest.raises(ValueError) as excinfo:
            config._validate_dates()
        
        assert "日付の形式が正しくありません" in str(excinfo.value)
    
    @patch('horguesc.utils.config.find_config')
    def test_config_file_not_found(self, mock_find_config):
        """設定ファイルが見つからない場合のテスト"""
        mock_find_config.return_value = None
        
        with pytest.raises(FileNotFoundError):
            Config().load_from_file()
    
    def test_update_from_args(self, sample_config_file):
        """コマンドライン引数による設定更新のテスト"""
        import argparse
        config = Config().load_from_file(sample_config_file)
        
        # argparseのNamespaceオブジェクトを直接作成
        args = argparse.Namespace()
        args.command = 'train'
        args.model_type = 'lightgbm'
        args.training_batch_size = '64'
        
        # 辞書形式で変換して__dict__を設定
        args.__dict__ = {
            'command': 'train',
            'model.type': 'lightgbm',
            'training.batch_size': '64'
        }
        
        config.update_from_args(args)
        
        assert config.get('model', 'type') == 'lightgbm'
        assert config.get('training', 'batch_size') == '64'

    def test_load_config_integration(self, sample_config_file):
        """load_config関数の統合テスト"""
        with patch('horguesc.utils.config.find_config', return_value=sample_config_file):
            config = load_config()
            
        assert isinstance(config, Config)
        assert len(config.numerical_features) == 3
        assert len(config.categorical_features) == 2