import pytest
import numpy as np
import pandas as pd
import datetime
from unittest.mock import MagicMock, patch
from horguesc.tasks.trifecta.dataset import TrifectaDataset
from horguesc.utils.config import Config


@pytest.fixture
def mock_config():
    """テスト用の設定を作成する"""
    config = MagicMock(spec=Config)
    # 特徴量定義
    config.numerical_features = ['bataiju']
    config.categorical_features = ['umaban', 'ketto_toroku_bango']
    
    # グループマッピングを設定
    config.feature_groups = {
        'numerical': {'bataiju': 'weight'},
        'categorical': {
            'umaban': 'position',
            'ketto_toroku_bango': 'horse_id'
        }
    }
    return config


@pytest.fixture
def mock_db_ops():
    """データベース操作のモックを作成する"""
    mock_ops = MagicMock()
    
    # モックデータを用意（実際のDBクエリ結果に近い形式）
    mock_data = [
        # 第1レース (2頭出走)
        {
            'kaisai_date': '2023-01-01',
            'keibajo_code': '01',
            'kaisai_kai': 1,
            'kaisai_nichime': 1,
            'kyoso_bango': 1,
            'umaban': 1,
            'bataiju': 480,
            'ketto_toroku_bango': 123456789,
            'target': 1
        },
        {
            'kaisai_date': '2023-01-01',
            'keibajo_code': '01',
            'kaisai_kai': 1,
            'kaisai_nichime': 1,
            'kyoso_bango': 1,
            'umaban': 2,
            'bataiju': 460,
            'ketto_toroku_bango': 987654321,
            'target': 2
        },
        # 第2レース (3頭出走)
        {
            'kaisai_date': '2023-01-01',
            'keibajo_code': '01',
            'kaisai_kai': 1,
            'kaisai_nichime': 1,
            'kyoso_bango': 2,
            'umaban': 1,
            'bataiju': 470,
            'ketto_toroku_bango': 111222333,
            'target': 2
        },
        {
            'kaisai_date': '2023-01-01',
            'keibajo_code': '01',
            'kaisai_kai': 1,
            'kaisai_nichime': 1,
            'kyoso_bango': 2,
            'umaban': 2,
            'bataiju': 490,
            'ketto_toroku_bango': 444555666,
            'target': 3
        },
        {
            'kaisai_date': '2023-01-01',
            'keibajo_code': '01',
            'kaisai_kai': 1,
            'kaisai_nichime': 1,
            'kyoso_bango': 2,
            'umaban': 3,
            'bataiju': None,  # 欠損値のテスト
            'ketto_toroku_bango': 777888999,
            'target': 1
        }
    ]
    
    # execute_queryの戻り値を設定
    mock_ops.execute_query.return_value = mock_data
    return mock_ops


def test_init(mock_config):
    """初期化のテスト"""
    # 開始日と終了日を指定してデータセットを初期化
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    dataset = TrifectaDataset(
        config=mock_config,
        mode='train',
        batch_size=32,
        start_date=start_date,
        end_date=end_date
    )
    
    # 初期化が正しく行われたか確認
    assert dataset.mode == 'train'
    assert dataset.batch_size == 32
    assert dataset.start_date == datetime.datetime(2023, 1, 1)
    assert dataset.end_date == datetime.datetime(2023, 1, 31)


def test_fetch_data(mock_config, mock_db_ops):
    """データ取得のテスト"""
    dataset = TrifectaDataset(
        config=mock_config,
        mode='train',
        start_date='2023-01-01',
        end_date='2023-01-31'
    )
    
    # _fetch_dataメソッドをテスト
    dataset._fetch_data(db_ops=mock_db_ops)
    
    # SQLビルダーを使ったクエリが実行されたか確認
    mock_db_ops.execute_query.assert_called_once()
    
    # 引数にSQLクエリとパラメータが含まれ、fetch_allとas_dictが両方Trueで呼び出されたか確認
    args, kwargs = mock_db_ops.execute_query.call_args
    assert kwargs['fetch_all'] is True
    assert kwargs['as_dict'] is True
    
    # raw_dataが正しく構築されたかチェック
    assert 'kyoso_id' in dataset.raw_data
    assert len(dataset.raw_data['kyoso_id']) == 1  # モックデータでは3頭以上のレースは1つのみ
    
    # 特徴量データが正しく抽出されたか確認
    assert 'umaban' in dataset.raw_data
    assert 'bataiju' in dataset.raw_data
    assert 'ketto_toroku_bango' in dataset.raw_data
    
    # 'target'はFEATURE_DEFINITIONSに含まれていないので、代わりに'kakutei_chakujun'を確認
    assert 'kakutei_chakujun' in dataset.raw_data
    
    # 2次元配列に変換されているか確認
    assert dataset.raw_data['umaban'].shape == (1, 3)  # 1レース、最大3頭出走
    assert dataset.raw_data['kakutei_chakujun'].shape == (1, 3)  # 1レース、最大3頭出走


def test_convert_query_results_to_2d_arrays(mock_config):
    """クエリ結果から2次元配列への変換テスト"""
    dataset = TrifectaDataset(
        config=mock_config,
        mode='train'
    )
    
    # テスト用のクエリ結果 - 3頭のデータを用意して3連単予測の最小要件を満たす
    query_results = [
        {
            'kaisai_date': '2023-01-01',
            'keibajo_code': '01',
            'kaisai_kai': 1,
            'kaisai_nichime': 1,
            'kyoso_bango': 1,
            'umaban': 1,
            'bataiju': 480,
            'ketto_toroku_bango': 123456789,
            'target': 1,
            'kakutei_chakujun': 1
        },
        {
            'kaisai_date': '2023-01-01',
            'keibajo_code': '01',
            'kaisai_kai': 1,
            'kaisai_nichime': 1,
            'kyoso_bango': 1,
            'umaban': 2,
            'bataiju': 460,
            'ketto_toroku_bango': 987654321,
            'target': 2,
            'kakutei_chakujun': 2
        },
        {
            'kaisai_date': '2023-01-01', 
            'keibajo_code': '01',
            'kaisai_kai': 1,
            'kaisai_nichime': 1,
            'kyoso_bango': 1,
            'umaban': 3,
            'bataiju': 470,
            'ketto_toroku_bango': 555555555,
            'target': 3,
            'kakutei_chakujun': 3
        }
    ]
    
    result = dataset._convert_query_results_to_2d_arrays(query_results)
    
    # 結果の検証
    assert 'kyoso_id' in result
    assert len(result['kyoso_id']) == 1  # 1つのレース
    assert result['kyoso_id'][0] == '2023-01-01_01_1_1_1'  # ID形式の確認
    
    # 特徴量が正しく抽出されているか確認
    assert 'umaban' in result
    assert result['umaban'].shape == (1, 3)  # 1レース、3頭出走
    assert result['umaban'][0][0] == 1
    assert result['umaban'][0][1] == 2
    assert result['umaban'][0][2] == 3
    
    assert 'bataiju' in result
    assert result['bataiju'][0][0] == 480
    assert result['bataiju'][0][1] == 460
    assert result['bataiju'][0][2] == 470
    
    # 'target'はFEATURE_DEFINITIONSに含まれていないので、代わりに'kakutei_chakujun'を確認
    assert 'kakutei_chakujun' in result
    assert result['kakutei_chakujun'][0][0] == 1
    assert result['kakutei_chakujun'][0][1] == 2
    assert result['kakutei_chakujun'][0][2] == 3


def test_get_all_data(mock_config):
    """get_all_dataメソッドのテスト"""
    dataset = TrifectaDataset(
        config=mock_config,
        mode='train'
    )
    
    # 処理済みデータをセット
    processed_data = {
        'umaban': np.array([[1, 2]]),
        'bataiju': np.array([[0.5, -0.5]]),
        'target': np.array([[1, 2]])
    }
    dataset.processed_data = processed_data
    
    # get_all_dataメソッドを呼び出し
    result = dataset.get_all_data()
    
    # 結果の検証
    assert result == processed_data


def test_get_batch(mock_config):
    """get_batchメソッドのテスト"""
    dataset = TrifectaDataset(
        config=mock_config,
        mode='train',
        batch_size=1
    )
    
    # 複数のレース（3レース）に対する処理済みデータをセット
    processed_data = {
        'umaban': np.array([[1, 2], [3, 4], [5, 6]]),
        'bataiju': np.array([[0.5, -0.5], [0.2, -0.2], [0.1, -0.1]]),
        'target': np.array([[1, 2], [1, 2], [1, 2]])
    }
    dataset.processed_data = processed_data
    
    # バッチインデックスの初期化
    dataset._init_batch_indices()
    
    # 最初のバッチを取得
    batch1 = dataset.get_batch()
    
    # バッチサイズ1なので、1レース分のデータが含まれているはず
    assert batch1['umaban'].shape == (1, 2)
    assert batch1['bataiju'].shape == (1, 2)
    assert batch1['target'].shape == (1, 2)
    
    # さらに2回のバッチ取得でデータを使い切る
    batch2 = dataset.get_batch()
    batch3 = dataset.get_batch()
    
    # 4回目の取得では、データが最初から循環するはず
    batch4 = dataset.get_batch()
    
    # トレーニングモードの場合、インデックスがシャッフルされる可能性があるので
    # 形状のみ確認する
    assert batch4['umaban'].shape == (1, 2)


def test_with_feature_processor(mock_config, mock_db_ops):
    """特徴量プロセッサとの統合テスト"""
    dataset = TrifectaDataset(
        config=mock_config,
        mode='train',
        start_date='2023-01-01',
        end_date='2023-01-31'
    )
    
    # データ取得
    dataset._fetch_data(db_ops=mock_db_ops)
    
    # 特徴量プロセッサのモック
    mock_processor = MagicMock()
    
    # 特徴量を正規化する関数のモックを設定
    mock_processor.normalize_numerical_feature.return_value = torch.tensor([[0.5, -0.5, 0.0], [0.2, 0.3, 0.0]])
    mock_processor.encode_categorical_feature.return_value = torch.tensor([[1, 2, 3], [4, 5, 6]])
    
    # 特徴量処理を実行
    dataset.process_features(mock_processor)
    
    # 正規化関数が呼び出されたかチェック
    assert mock_processor.normalize_numerical_feature.called
    assert mock_processor.encode_categorical_feature.called
    
    # 処理済みデータが格納されているか確認
    assert dataset.processed_data is not None


@pytest.mark.parametrize(
    "mode,should_shuffle", [
        ('train', True),
        ('eval', False)
    ]
)
def test_init_batch_indices(mock_config, mode, should_shuffle):
    """バッチインデックスの初期化テスト (モードによるシャッフル動作の違いを確認)"""
    dataset = TrifectaDataset(
        config=mock_config,
        mode=mode,
        batch_size=1,
        random_seed=42  # 再現性のため固定シード
    )
    
    # 処理済みデータをセット
    dataset.processed_data = {
        'umaban': np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
    }
    
    # シャッフルのモック関数
    original_shuffle = dataset._shuffle_indices
    shuffle_called = [False]  # リストでラップして参照を維持
    
    def mock_shuffle():
        shuffle_called[0] = True
        original_shuffle()
    
    dataset._shuffle_indices = mock_shuffle
    
    # バッチインデックスの初期化
    dataset._init_batch_indices()
    
    # モードに応じて適切なシャッフル動作が行われたか確認
    assert shuffle_called[0] == should_shuffle


# 必要に応じてtorch.tensorをインポート
import torch