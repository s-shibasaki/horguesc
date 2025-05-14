import os
import pytest
from unittest import mock
from horguesc.utils.config import Config, find_config, load_config

def test_find_config():
    """設定ファイル検索機能をテスト"""
    # 存在するパスをモックする
    with mock.patch('os.path.exists', return_value=True):
        assert find_config() is not None
    
    # すべてのパスが存在しない場合
    with mock.patch('os.path.exists', return_value=False):
        assert find_config() is None

def test_config_initialization():
    """設定の初期化をテスト"""
    config = Config()
    assert not hasattr(config, 'numerical_features') or config.numerical_features == []
    assert not hasattr(config, 'categorical_features') or config.categorical_features == []

def test_parse_features_list():
    """特徴量リスト形式の構文解析をテスト"""
    config = Config()
    
    # 設定データを追加（新形式 - リスト形式）
    config['features'] = {
        'numerical_features': 'age, income, weight',
        'categorical_features': 'gender, country, city'
    }
    
    # 特徴量解析を実行
    config.parse_features()
    
    # 特徴量リストの検証
    assert config.numerical_features == ['age', 'income', 'weight']
    assert config.categorical_features == ['gender', 'country', 'city']
    
    # 特徴量グループが自動生成されているか検証
    assert 'numerical' in config.feature_groups
    assert 'categorical' in config.feature_groups
    
    # デフォルトでは各特徴量が自身のグループになっていることを検証
    assert config.feature_groups['numerical']['age'] == 'age'
    assert config.feature_groups['categorical']['gender'] == 'gender'
    
    # グループごとの特徴量リストも確認
    assert 'age' in config.group_features['numerical']
    assert config.group_features['numerical']['age'] == ['age']

def test_parse_features_with_groups():
    """グループを含む特徴量設定のテスト"""
    config = Config()
    
    # 特徴量リストを設定
    config['features'] = {
        'numerical_features': 'age, income, height, weight',
        'categorical_features': 'gender, country, city'
    }
    
    # グループ情報を追加
    config['groups'] = {
        'basic_info': 'age, gender',
        'financial': 'income',
        'physical': 'height, weight',
        'location': 'country, city'
    }
    
    # 特徴量解析を実行
    config.parse_features()
    
    # グループ情報の検証
    assert config.feature_groups['numerical']['age'] == 'basic_info'
    assert config.feature_groups['categorical']['gender'] == 'basic_info'
    assert config.feature_groups['numerical']['income'] == 'financial'
    assert config.feature_groups['numerical']['height'] == 'physical'
    assert config.feature_groups['numerical']['weight'] == 'physical'
    assert config.feature_groups['categorical']['country'] == 'location'
    assert config.feature_groups['categorical']['city'] == 'location'
    
    # グループごとの特徴量リストの検証
    assert 'basic_info' in config.group_features['numerical']
    assert config.group_features['numerical']['basic_info'] == ['age']
    assert 'basic_info' in config.group_features['categorical']
    assert config.group_features['categorical']['basic_info'] == ['gender']
    assert config.group_features['numerical']['physical'] == ['height', 'weight']

@mock.patch('horguesc.utils.config.find_config')
@mock.patch('configparser.ConfigParser.read')
def test_load_config(mock_read, mock_find_config):
    """設定ファイル読み込みをテスト"""
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
    """設定ファイルがない場合のエラー処理をテスト"""
    # 設定ファイルが見つからない場合
    mock_find_config.return_value = None
    
    # 例外が発生することを確認
    with pytest.raises(FileNotFoundError):
        load_config()