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
        train_datasets = {}
        val_datasets = {}
        all_parameters = []

        # 設定からタスクを取得
        tasks_str = config.get('tasks', 'active', fallback='')
        tasks = [task.strip() for task in tasks_str.split(',') if task.strip()]
        
        if not tasks:
            logger.error("アクティブなタスクが設定されていません。tasks.active を設定してください。")
            return 1
            
        logger.info(f"アクティブなタスク: {', '.join(tasks)}")
        
        # 学習・検証データの日付範囲を設定から取得
        train_start = config.get('training', 'train_start_date')
        train_end = config.get('training', 'train_end_date')
        val_start = config.get('training', 'val_start_date')
        val_end = config.get('training', 'val_end_date')
        
        # バッチサイズを設定から取得
        batch_size = config.getint('training', 'batch_size', fallback=32)

        # Step 1: 全データセットを初期化して生データを読み込み、訓練データのみで特徴量値を収集
        for task in tasks:
            task = task.strip()
            try:
                # タスクモジュールをインポート
                dataset_module = import_module(f"horguesc.tasks.{task}.dataset")
                
                # データセットクラスを取得
                dataset_class = getattr(dataset_module, f"{task.capitalize()}Dataset")
                
                # トレーニング用データセットの作成
                train_dataset = dataset_class(
                    config=config,
                    mode='train',
                    batch_size=batch_size,
                    start_date=train_start,
                    end_date=train_end
                )
                
                # 検証用データセットの作成
                val_dataset = dataset_class(
                    config=config,
                    mode='eval',
                    batch_size=batch_size,
                    start_date=val_start,
                    end_date=val_end
                )
                
                # データを取得
                train_dataset.fetch_data()
                val_dataset.fetch_data()
                
                # 訓練データのみで特徴量値を収集
                train_dataset.collect_features(feature_processor)
                
                train_datasets[task] = train_dataset
                val_datasets[task] = val_dataset
                
                logger.info(f"{task} データセットを初期化して生データを取得しました")
                
            except (ImportError, AttributeError) as e:
                logger.error(f"タスク {task} の初期化に失敗しました: {e}")
                return 1
        
        # Step 2: 収集した特徴量でエンコーダーをフィット
        logger.info("全データセットから収集した特徴量でエンコーダーをフィットします")
        feature_processor.fit()
        
        # 変更: 特徴量プロセッサでフィットした後に共有エンコーダを作成
        # 特徴量プロセッサでフィットした後に共有エンコーダを作成
        group_cardinalities = feature_processor.get_group_cardinalities()
        shared_encoder = FeatureEncoder(config, group_cardinalities)
        all_parameters.extend(shared_encoder.parameters())
        
        # Step 3: 各データセットのデータを処理
        for task in tasks:
            train_datasets[task].process_features(feature_processor)
            val_datasets[task].process_features(feature_processor)
            logger.info(f"{task} データセットの特徴量処理を完了しました")
        
        # エンコーダを保存（後で検証/推論に使用）
        encoder_path = os.path.join(config.get('paths', 'model_dir', fallback='models'), 'feature_processor.pt')
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        feature_processor.save_state(encoder_path)
        logger.info(f"特徴量処理器の状態を {encoder_path} に保存しました")
        
        # 次にモデルを初期化
        for task in tasks:
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
        
        # 学習用と検証用のデータセットを一つの辞書にまとめる
        combined_datasets = {}
        for task in tasks:
            combined_datasets[f"{task}.train"] = train_datasets[task]  # 学習用データセット
            combined_datasets[f"{task}.eval"] = val_datasets[task]     # 評価用データセット
        
        # トレーナーを初期化してモデルをトレーニング
        trainer = MultitaskTrainer(config, models, combined_datasets, optimizer)
        num_epochs = config.getint('training', 'num_epochs', fallback=10)
        trainer.train(num_epochs)
        
        logger.info("トレーニングが正常に完了しました")
        return 0
        
    except Exception as e:
        logger.error(f"トレーニング中にエラーが発生しました: {e}", exc_info=True)
        return 1