import logging
import numpy as np
import torch
from horguesc.core.base.dataset import BaseDataset
from horguesc.database.operations import DatabaseOperations

logger = logging.getLogger(__name__)

class TrifectaDataset(BaseDataset):
    """トリフェクタ予測タスクのデータセット。"""
    
    def __init__(self, config, feature_processor=None):
        super().__init__(config, feature_processor)
        self.db_ops = DatabaseOperations(config)
        
        # トレーニング用のデータインデックス
        self.train_indices = None
        self.current_index = 0
    
    def get_raw_data(self):
        """データベースから生データを読み込みます。"""
        logger.info("データベースからトリフェクタデータセットを読み込んでいます")

        query = """
        SELECT 
            r.race_id,
            r.race_date,
            r.track_id,
            r.track_condition,
            r.distance,
            h.horse_id,
            h.bloodline_id,
            h.birth_year,
            h.sex,
            h.country_id,
            re.finish_position,
            re.odds,
            re.popularity,
            re.final_time,
            re.weight,
            re.weight_diff,
            j.jockey_id,
            t.trainer_id
        FROM race r
        JOIN race_entry re ON r.race_id = re.race_id
        JOIN horse h ON re.horse_id = h.horse_id
        JOIN jockey j ON re.jockey_id = j.jockey_id
        JOIN trainer t ON re.trainer_id = t.trainer_id
        WHERE r.race_date BETWEEN ? AND ?
        """

        # 設定からデータパラメータを取得、またはデフォルトを使用
        from_date = self.config.get('dataset', 'from_date', fallback='2023-01-01')
        to_date = self.config.get('dataset', 'to_date', fallback='2023-12-31')
        logger.info(f"{from_date} から {to_date} までのデータを読み込んでいます")

        try:
            results = self.db_ops.execute_query(query, params=[from_date, to_date], fetch_all=True)

            if not results:
                logger.warning("指定された日付範囲のデータが見つかりません。")
                return {}
            
            logger.info(f"データベースから {len(results)} レコードを読み込みました")

            # 結果を特徴量と目標に処理
            raw_data = {
                # 数値特徴量
                'distance': [],
                'odds': [],
                'popularity': [],
                'weight': [],
                'weight_diff': [],
                'birth_year': [],
                
                # カテゴリカル特徴量
                'track_id': [],
                'track_condition': [],
                'horse_id': [],
                'bloodline_id': [],
                'sex': [],
                'country_id': [],
                'jockey_id': [],
                'trainer_id': [],
                
                # 目標
                'finish_position': []
            }

            for row in results:
                # 行から特徴量と目標を抽出
                raw_data['distance'].append(row[4])
                raw_data['track_id'].append(row[2])
                raw_data['track_condition'].append(row[3])
                raw_data['horse_id'].append(row[5])
                raw_data['bloodline_id'].append(row[6])
                raw_data['birth_year'].append(row[7])
                raw_data['sex'].append(row[8])
                raw_data['country_id'].append(row[9])
                raw_data['finish_position'].append(row[10])
                raw_data['odds'].append(row[11])
                raw_data['popularity'].append(row[12])
                raw_data['weight'].append(row[14])
                raw_data['weight_diff'].append(row[15])
                raw_data['jockey_id'].append(row[16])
                raw_data['trainer_id'].append(row[17])

            return raw_data

        except Exception as e:
            logger.error(f"トリフェクタデータの読み込みに失敗しました: {str(e)}")
            return {}
    
    def _split_data(self, processed_data):
        """処理済みデータをトレーニングデータと検証データに分割します。"""
        # データサイズを取得
        data_size = len(processed_data.get('finish_position', []))
        if data_size == 0:
            logger.warning("分割するデータがありません")
            self.train_data = {'features': {}, 'targets': {}}
            self.val_data = {'features': {}, 'targets': {}}
            return
        
        # データを分割（例: 80% トレーニング、20% 検証）
        split_idx = int(data_size * 0.8)
        indices = np.random.permutation(data_size)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # トレーニング用のインデックスをシャッフルして保存
        self.train_indices = train_indices
        np.random.shuffle(self.train_indices)
        self.current_index = 0
        
        # 特徴量と目標を分離
        train_features = {}
        val_features = {}
        
        for feature_name, feature_tensor in processed_data.items():
            if feature_name != 'finish_position':
                train_features[feature_name] = feature_tensor[train_indices]
                val_features[feature_name] = feature_tensor[val_indices]
        
        # 目標値を準備
        if 'finish_position' in processed_data:
            train_targets = {'labels': processed_data['finish_position'][train_indices]}
            val_targets = {'labels': processed_data['finish_position'][val_indices]}
        else:
            logger.warning("目標値 'finish_position' が見つかりません")
            train_targets = {'labels': torch.zeros(len(train_indices))}
            val_targets = {'labels': torch.zeros(len(val_indices))}
        
        # トレーニングデータと検証データを設定
        self.train_data = {
            'features': train_features,
            'targets': train_targets
        }
        
        self.val_data = {
            'features': val_features,
            'targets': val_targets
        }
        
        logger.info(f"データを分割しました: {len(train_indices)} トレーニングサンプル, {len(val_indices)} 検証サンプル")
    
    def get_next_batch(self, batch_size):
        """トリフェクタデータの次のバッチを取得します。
        
        Args:
            batch_size: バッチのサイズ
            
        Returns:
            dict: 'inputs'と'targets'キーを持つ辞書
        """
        if self.train_data is None or self.train_indices is None:
            logger.warning("データが準備されていません。get_next_batch を呼び出す前に prepare_data を実行してください")
            return {
                'inputs': {'features': torch.zeros(batch_size, 10)},
                'targets': {'labels': torch.zeros(batch_size)}
            }
        
        # バッチ用のインデックスを取得
        if self.current_index + batch_size > len(self.train_indices):
            # エポック終了時にインデックスをリセットしてシャッフル
            np.random.shuffle(self.train_indices)
            self.current_index = 0
        
        batch_indices = self.train_indices[self.current_index:self.current_index + batch_size]
        self.current_index += batch_size
        
        # バッチデータを準備
        batch_features = {}
        for name, tensor in self.train_data['features'].items():
            batch_features[name] = tensor[batch_indices]
        
        batch_targets = {
            'labels': self.train_data['targets']['labels'][batch_indices]
        }
        
        return {
            'inputs': {'features': batch_features},
            'targets': batch_targets
        }
    
    def get_validation_data(self):
        """トリフェクタタスクの検証データを取得します。
        
        Returns:
            dict: 検証データを含む辞書
        """
        if self.val_data is None:
            logger.warning("検証データが準備されていません。get_validation_data を呼び出す前に prepare_data を実行してください")
            return {
                'inputs': {'features': torch.zeros(100, 10)},
                'targets': {'labels': torch.zeros(100)}
            }
        
        return {
            'inputs': {'features': self.val_data['features']},
            'targets': self.val_data['targets']
        }
    
    def get_name(self):
        """タスク名を取得します。
        
        Returns:
            str: タスク名
        """
        return "trifecta"