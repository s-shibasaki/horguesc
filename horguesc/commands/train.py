"""
horguescのトレインコマンドの実装。
"""
import logging
import sys
import os
import torch
from importlib import import_module
from horguesc.core.modules.encoders import FeatureEncoder
from horguesc.utils.config import load_config
from horguesc.core.trainer import MultitaskTrainer
from horguesc.core.features.processor import FeatureProcessor

logger = logging.getLogger(__name__)

def run(args):
    """
    トレインコマンドを実行します。
    
    Args:
        args: コマンドライン引数
        
    Returns:
        int: 終了コード
    """
    try:
        # 設定を読み込み
        config = load_config()
        logger.info("設定を正常に読み込みました")
        
        # 共通の特徴量プロセッサを作成
        feature_processor = FeatureProcessor(config)
        
        # モデルとデータセットを各タスクで初期化
        models = {}
        datasets = {}
        all_parameters = []

        # 共有エンコーダを作成
        shared_encoder = FeatureEncoder(config, feature_processor)
        all_parameters.extend(shared_encoder.parameters())

        # 設定からタスクを取得
        tasks = config.get('training', 'tasks').split(',')
        
        # まず全てのデータセットを初期化して特徴量を収集
        for task in tasks:
            task = task.strip()
            try:
                # タスクモジュールをインポート
                dataset_module = import_module(f"horguesc.tasks.{task}.dataset")
                
                # データセットを初期化
                dataset_class = getattr(dataset_module, f"{task.capitalize()}Dataset")
                dataset = dataset_class(config, feature_processor)
                
                # このタスクが担当する特徴量を設定
                numerical_features = config.get(f'tasks.{task}', 'numerical_features', fallback='').split(',')
                numerical_features = [f.strip() for f in numerical_features if f.strip()]
                
                categorical_features = config.get(f'tasks.{task}', 'categorical_features', fallback='').split(',')
                categorical_features = [f.strip() for f in categorical_features if f.strip()]
                
                dataset.set_features(numerical_features, categorical_features)
                
                # データを準備（生データの読み込みと前処理）
                dataset.prepare_data()
                
                datasets[task] = dataset
                logger.info(f"{task} データセットを初期化しました")
                
            except (ImportError, AttributeError) as e:
                logger.error(f"タスク {task} の初期化に失敗しました: {e}")
                return 1
        
        # ここでカテゴリエンコーダを保存（後で検証/推論に使用）
        encoder_path = os.path.join(config.get('paths', 'model_dir', fallback='models'), 'encoders.pt')
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        feature_processor.save_encoders(encoder_path)
        
        # 次にモデルを初期化
        for task in tasks:
            task = task.strip()
            try:
                # タスクモジュールをインポート
                model_module = import_module(f"horguesc.tasks.{task}.model")
                
                # モデルを初期化
                model_class = getattr(model_module, f"{task.capitalize()}Model")
                model = model_class(config, shared_encoder)
                
                models[task] = model
                
                # オプティマイザ用のパラメータを収集
                all_parameters.extend(model.parameters())
                
                logger.info(f"{task} モデルを初期化しました")
            except (ImportError, AttributeError) as e:
                logger.error(f"タスク {task} のモデル初期化に失敗しました: {e}")
                return 1
        
        # オプティマイザを作成
        optimizer_name = config.get('training', 'optimizer', fallback='Adam')
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(
            all_parameters,
            lr=config.getfloat('training', 'learning_rate', fallback=0.001)
        )
        
        # トレーナーを初期化してモデルをトレーニング
        trainer = MultitaskTrainer(config, models, datasets, optimizer)
        num_epochs = config.getint('training', 'num_epochs', fallback=10)
        trainer.train(num_epochs)
        
        logger.info("トレーニングが正常に完了しました")
        return 0
        
    except Exception as e:
        logger.error(f"トレーニング中にエラーが発生しました: {e}", exc_info=True)
        return 1