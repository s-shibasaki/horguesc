import os
import pytest
from unittest import mock
from horguesc.utils.config import Config, find_config, load_config

def test_find_config():
    # 存在するパスをモックする
    with mock.patch('os.path.exists', return_value=True):
        assert find_config() is not None
    
    # すべてのパスが存在しない場合
    with mock.patch('os.path.exists', return_value=False):
        assert find_config() is None

def test_config_initialization():
    config = Config()
    assert config.numerical_features == {}
    assert config.categorical_features == {}
    assert config.num_features_count == 0
    assert config.cat_features_count == 0

def test_parse_features():
    config = Config()
    
    # 設定データを追加
    config['model'] = {'default_numerical_embedding_dim': '8'}
    config['features.age'] = {
        'type': 'numerical',
        'description': 'User age'
    }
    config['features.gender'] = {
        'type': 'categorical',
        'description': 'User gender',
        'cardinality': '3'
    }
    
    # 特徴量解析を実行
    config.parse_features()
    
    # 数値特徴量の検証
    assert len(config.numerical_features) == 1
    assert 'age' in config.numerical_features
    assert config.numerical_features['age']['embedding_dim'] == 8
    
    # カテゴリ特徴量の検証
    assert len(config.categorical_features) == 1
    assert 'gender' in config.categorical_features
    assert config.categorical_features['gender']['cardinality'] == 3
    assert config.categorical_features['gender']['embedding_dim'] > 0
    
    # カウント検証
    assert config.num_features_count == 1
    assert config.cat_features_count == 1

def test_get_feature_methods():
    config = Config()
    
    # テスト用データを設定
    config.numerical_features = {
        'age': {'name': 'age', 'embedding_dim': 8},
        'income': {'name': 'income', 'embedding_dim': 10}
    }
    config.categorical_features = {
        'gender': {'name': 'gender', 'cardinality': 3, 'embedding_dim': 2},
        'country': {'name': 'country', 'cardinality': 50, 'embedding_dim': 25}
    }
    
    # メソッドの戻り値を検証
    assert config.get_numerical_feature_names() == ['age', 'income']
    assert config.get_categorical_feature_names() == ['gender', 'country']

@mock.patch('horguesc.utils.config.find_config')
@mock.patch('configparser.ConfigParser.read')
def test_load_config(mock_read, mock_find_config):
    # find_configの戻り値を設定
    mock_find_config.return_value = 'path/to/config.ini'
    
    # 設定ファイル読み込み処理をモック
    config = load_config()
    
    # モック関数が呼び出されたことを確認
    mock_find_config.assert_called_once()
    mock_read.assert_called_once()
    
    # ConfigオブジェクトとConfigParserの機能を継承していることを確認
    assert isinstance(config, Config)

@mock.patch('horguesc.utils.config.find_config')
def test_load_config_file_not_found(mock_find_config):
    # 設定ファイルが見つからない場合
    mock_find_config.return_value = None
    
    # 例外が発生することを確認
    with pytest.raises(FileNotFoundError):
        load_config()