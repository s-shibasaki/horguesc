import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock

from horguesc.core.base.dataset import BaseDataset

# テスト用のデータセットサブクラス
class MockDataset(BaseDataset):  # TestDataset → MockDataset に変更
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _fetch_data(self):
        # テスト用の簡単なデータを用意
        self.raw_data = {
            'feature1': np.array([1, 2, 3, 4, 5]),
            'feature2': np.array(['a', 'b', 'c', 'd', 'e']),
            'target': np.array([0, 1, 0, 1, 0])
        }
    
    def get_name(self):
        return "MockDataset"

# 設定モック
@pytest.fixture
def mock_config():
    config = MagicMock()
    config.get_numerical_feature_names.return_value = ['feature1']
    config.get_categorical_feature_names.return_value = ['feature2']
    return config

# 特徴量プロセッサーモック
@pytest.fixture
def mock_feature_processor():
    processor = MagicMock()
    processor.normalize_numerical_feature.side_effect = lambda name, data: data * 0.1
    processor.encode_categorical_feature.side_effect = lambda name, data: np.array([0, 1, 2, 3, 4])
    return processor

def test_dataset_initialization(mock_config):
    """データセットの初期化が正しく行われるかテスト"""
    dataset = MockDataset(
        config=mock_config, 
        mode='train',
        batch_size=2,
        start_date='2023-01-01',
        end_date='2023-01-31',
        random_seed=42
    )
    
    assert dataset.mode == 'train'
    assert dataset.batch_size == 2
    assert dataset.start_date == datetime(2023, 1, 1)
    assert dataset.end_date == datetime(2023, 1, 31)

def test_fetch_data(mock_config):
    """データ取得メソッドが正しく機能するかテスト"""
    dataset = MockDataset(config=mock_config)
    dataset.fetch_data()
    
    assert dataset.raw_data is not None
    assert 'feature1' in dataset.raw_data
    assert 'feature2' in dataset.raw_data
    assert len(dataset.raw_data['feature1']) == 5

def test_process_features(mock_config, mock_feature_processor):
    """特徴量処理が正しく行われるかテスト"""
    dataset = MockDataset(config=mock_config)
    dataset.fetch_data()
    dataset.process_features(mock_feature_processor)
    
    assert dataset.processed_data is not None
    assert 'feature1' in dataset.processed_data
    assert 'feature2' in dataset.processed_data
    # 数値特徴量が0.1倍されているか確認
    np.testing.assert_array_equal(dataset.processed_data['feature1'], np.array([1, 2, 3, 4, 5]) * 0.1)

def test_batch_retrieval(mock_config, mock_feature_processor):
    """バッチ取得が正しく機能するかテスト"""
    dataset = MockDataset(config=mock_config, batch_size=2)
    dataset.fetch_data()
    dataset.process_features(mock_feature_processor)
    
    # 1つ目のバッチを取得
    batch1 = dataset.get_batch()
    assert len(batch1['feature1']) == 2
    
    # 2つ目のバッチを取得
    batch2 = dataset.get_batch()
    assert len(batch2['feature1']) == 2
    
    # 3つ目のバッチ（残り1つ）を取得
    batch3 = dataset.get_batch()
    assert len(batch3['feature1']) == 1
    
    # もう一度バッチを取得すると先頭から（シャッフルされている可能性あり）
    batch4 = dataset.get_batch()
    assert len(batch4['feature1']) == 2

def test_mode_validation(mock_config):
    """無効なモードで例外が発生するかテスト"""
    with pytest.raises(ValueError):
        MockDataset(config=mock_config, mode='invalid_mode')

def test_shuffle_in_train_mode(mock_config, mock_feature_processor):
    """訓練モードでシャッフルが行われるかテスト"""
    # シード固定でインスタンス化
    dataset = MockDataset(config=mock_config, mode='train', random_seed=42, batch_size=5)
    dataset.fetch_data()
    dataset.process_features(mock_feature_processor)
    
    # 最初のインデックス配列を保存
    original_indices = dataset._batch_indices.copy()
    
    # バッチを取得して一周させる
    _ = dataset.get_batch()
    
    # 再シャッフルされたインデックス配列と比較
    assert not np.array_equal(original_indices, dataset._batch_indices)

def test_no_shuffle_in_eval_mode(mock_config, mock_feature_processor):
    """評価モードでシャッフルが行われないかテスト"""
    dataset = MockDataset(config=mock_config, mode='eval', random_seed=42, batch_size=5)
    dataset.fetch_data()
    dataset.process_features(mock_feature_processor)
    
    # インデックス配列を確認（昇順になっているはず）
    assert np.array_equal(dataset._batch_indices, np.arange(5))
    
    # 最初のインデックス配列を保存
    original_indices = dataset._batch_indices.copy()
    
    # バッチを取得して一周させる
    _ = dataset.get_batch()
    
    # evalモードではシャッフルされないことを確認
    assert np.array_equal(original_indices, dataset._batch_indices)