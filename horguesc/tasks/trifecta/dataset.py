from horguesc.core.base.dataset import BaseDataset
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
from horguesc.database.sql_builder import SQLBuilder
import itertools
import torch
import pandas as pd

logger = logging.getLogger(__name__)

class TrifectaDataset(BaseDataset):
    """Trifectaタスク向けのデータセットクラス。"""
    
    # クラス変数として特徴量定義を追加（簡略化版）
    FEATURE_DEFINITIONS = {
        # 競走ID用のカラム
        'kaisai_date': 'se.kaisai_date',
        'keibajo_code': 'se.keibajo_code',
        'kaisai_kai': 'se.kaisai_kai',
        'kaisai_nichime': 'se.kaisai_nichime',
        'kyoso_bango': 'se.kyoso_bango',
        
        # 特徴量
        'umaban': 'se.umaban',
        'bataiju': 'CASE WHEN se.bataiju BETWEEN 2 AND 998 THEN se.bataiju ELSE NULL END',
        'ketto_toroku_bango': 'CASE WHEN se.ketto_toroku_bango != 0 THEN se.ketto_toroku_bango ELSE NULL END',
        'futan_juryo': 'se.futan_juryo',
        'blinker_shiyo_kubun': 'se.blinker_shiyo_kubun',
        'kishu_code': 'se.kishu_code',
        'kishu_minarai_code': 'se.kishu_minarai_code',
        
        # RAテーブルの関連特徴量
        'kyori': 'ra.kyori',
        'track_code': 'ra.track_code',
        'course_kubun': 'ra.course_kubun',
        'tenko_code': 'ra.tenko_code',
        'babajotai_code': 'ra.babajotai_code',
        
        # ターゲット変数
        'kakutei_chakujun': 'se.kakutei_chakujun',
    }
    
    # 競走IDを構成するカラム定義（FEATURE_DEFINITIONSのキーを使用）
    KYOSO_ID_COLUMNS = [
        'kaisai_date',
        'keibajo_code', 
        'kaisai_kai', 
        'kaisai_nichime', 
        'kyoso_bango'
    ]
    
    def __init__(self, *args, **kwargs):
        """TrifectaDatasetの初期化"""
        super().__init__(*args, **kwargs)
    
    def _fetch_data(self, db_ops=None, **kwargs) -> None:
        """データベースからTrifectaデータを取得する"""
        logger.info(f"Trifectaデータの取得を開始: {self.start_date} から {self.end_date} まで")
        
        # データベース接続を取得
        if db_ops is None:
            from horguesc.database.operations import DatabaseOperations
            db_ops = DatabaseOperations(self.config)
        
        # 日付範囲の文字列を準備
        start_date_str = self.start_date.strftime('%Y-%m-%d') if self.start_date else None
        end_date_str = self.end_date.strftime('%Y-%m-%d') if self.end_date else None
        
        # SQLBuilderでクエリを構築
        builder = SQLBuilder("se")
        
        # 全ての特徴量を選択（ID用カラムも含む）
        for feature_name, expression in self.FEATURE_DEFINITIONS.items():
            builder.select_as(expression, feature_name)
        
        # 必要なJOINを追加
        builder.join('''LEFT JOIN ra ON se.kaisai_date = ra.kaisai_date 
                    AND se.keibajo_code = ra.keibajo_code 
                    AND se.kaisai_kai = ra.kaisai_kai 
                    AND se.kaisai_nichime = ra.kaisai_nichime 
                    AND se.kyoso_bango = ra.kyoso_bango''')
        
        # データフィルタ条件を追加
        builder.where("se.data_type IN ('7', '2')")
        builder.where("se.kakutei_chakujun IS NOT NULL")  # 確定着順がある馬のみ
        builder.where("se.kakutei_chakujun BETWEEN 1 AND 18")  # 着順が有効な範囲内のもの
        
        # 日付範囲のフィルタ
        if start_date_str or end_date_str:
            builder.where_date_range("se.kaisai_date", start_date_str, end_date_str)
        
        # 結果を整理するための並び順を指定
        builder.order_by_columns("se.kaisai_date", "se.keibajo_code", 
                               "se.kaisai_kai", "se.kaisai_nichime", 
                               "se.kyoso_bango", "se.umaban")
        
        try:
            # クエリを実行
            query, params = builder.build()
            logger.debug(f"実行するクエリ: {query}")
            
            # データ取得
            results = db_ops.execute_query(query, params=params, fetch_all=True, as_dict=True)
            
            if not results:
                logger.warning(f"指定された期間のデータが見つかりませんでした: {start_date_str}〜{end_date_str}")
                self.raw_data = {}
                return
            
            # クエリ結果を2D配列形式のデータに変換
            self.raw_data = self._convert_query_results_to_2d_arrays(results)
            
            # 3連単のターゲットを作成
            self._create_trifecta_targets()
            
            logger.info(f"Trifectaデータの取得完了: {len(self.raw_data['kyoso_id'])}競走、{len(results)}頭")
            
        except Exception as e:
            logger.error(f"データ取得中にエラーが発生しました: {e}")
            raise
    
    def _convert_query_results_to_2d_arrays(self, query_results):
        """
        クエリ結果を2D配列形式のデータに変換する
        
        Args:
            query_results: SQLクエリの結果（辞書のリスト）
            
        Returns:
            dict: kyoso_idリストと特徴量の2D配列を含む辞書
        """
        # 競走IDごとにデータをグループ化
        kyoso_groups = defaultdict(list)
        
        for row in query_results:
            # 競走IDを生成
            kyoso_id_parts = [str(row[col]) for col in self.KYOSO_ID_COLUMNS]
            kyoso_id = '_'.join(kyoso_id_parts)
            kyoso_groups[kyoso_id].append(row)
        
        # 最大出走頭数を計算
        max_horses = max(len(horses) for horses in kyoso_groups.values())
        logger.info(f"最大出走頭数: {max_horses}")
        
        # 特徴量データの2D配列を準備
        kyoso_ids = []
        feature_arrays = defaultdict(list)
        horse_counts = []  # 各レースの実際の出走馬数を保存
        
        # 事前定義された特徴量名リストを使用
        features = list(self.FEATURE_DEFINITIONS.keys())
        
        # 各競走のデータを処理
        for kyoso_id, horses in kyoso_groups.items():
            # 最低3頭の出走馬がいるレースのみ処理（3連単予測のため）
            if len(horses) < 3:
                logger.debug(f"出走頭数が3頭未満のレースをスキップ: {kyoso_id}")
                continue
                
            kyoso_ids.append(kyoso_id)
            horse_counts.append(len(horses))  # 実際の出走馬数を記録
            
            # 各特徴量について処理
            for feature in features:
                # 馬ごとの特徴量値を取得
                values = [horse.get(feature) for horse in horses]
                
                # 最大頭数まで None で埋める
                while len(values) < max_horses:
                    values.append(None)
                
                feature_arrays[feature].append(values)
        
        # データを返す（kyoso_idはリストのまま、特徴量のみNumPy配列に変換）
        formatted_data = {
            'kyoso_id': kyoso_ids,
            'horse_count': np.array(horse_counts, dtype=np.int32)  # 追加: 各レースの実際の出走馬数
        }
        
        for feature, values_list in feature_arrays.items():
            formatted_data[feature] = np.array(values_list)
        
        return formatted_data
    
    def _create_trifecta_targets(self):
        """
        3連単のターゲットを作成する
        
        - 各レースについて、確定着順に基づいて3連単の正解インデックスを特定
        - インデックスベースで処理し、馬番ではなく配列の位置を使用
        - 同着の場合も正しく処理（着順の数字ではなく順位で判断）
        """
        logger.info("3連単ターゲット情報の作成を開始")
        
        if 'kakutei_chakujun' not in self.raw_data or len(self.raw_data['kakutei_chakujun']) == 0:
            logger.warning("確定着順データがないため、3連単ターゲットを作成できません")
            return
            
        # レース数と最大出走頭数を取得
        n_races = len(self.raw_data['kyoso_id'])
        max_horses = self.raw_data['kakutei_chakujun'].shape[1]
        
        # 各レースの実際の出走頭数
        horse_counts = self.raw_data['horse_count']
        
        # 全ての可能な3連単の組み合わせを生成
        all_combinations = list(itertools.permutations(range(max_horses), 3))
        
        # 各レースの正解3連単のインデックスを格納する配列
        target_trifecta = np.full(n_races, -1, dtype=np.int64)
        
        # 各レースについて処理
        for race_idx in range(n_races):
            # このレースの着順データを取得
            chakujun = self.raw_data['kakutei_chakujun'][race_idx]
            
            # 実際の出走頭数（馬の数）
            race_horse_count = int(horse_counts[race_idx])
            
            # NoneやNaNを大きな値に置き換えて、有効な馬だけを考慮
            chakujun_array = np.array([
                999 if (i >= race_horse_count or pd.isna(v) or v is None)
                else int(v) for i, v in enumerate(chakujun)
            ])
            
            # インデックスと着順のマッピングを作成
            idx_with_chakujun = [(i, place) for i, place in enumerate(chakujun_array)]
            
            # 着順で並べ替え
            idx_with_chakujun.sort(key=lambda x: x[1])
            
            # 上位3頭の馬インデックスを取得（同着を考慮、着順の値ではなく順位で判断）
            top3_indices = [pair[0] for pair in idx_with_chakujun[:3]]
            
            # 上位3頭が揃っていることを確認
            if len(top3_indices) == 3:
                # この組み合わせがall_combinationsの何番目かを特定
                target_combo = tuple(top3_indices)
                
                # all_combinations内でのインデックスを検索
                try:
                    combo_idx = all_combinations.index(target_combo)
                    target_trifecta[race_idx] = combo_idx
                except ValueError:
                    # 組み合わせがall_combinationsに含まれない場合（通常発生しない）
                    logger.warning(f"レース {self.raw_data['kyoso_id'][race_idx]} の組み合わせが見つかりません: {target_combo}")
            else:
                # 上位3頭が特定できない場合（データ異常）
                logger.warning(f"レース {self.raw_data['kyoso_id'][race_idx]} の上位3頭を特定できません")
        
        # ターゲットを追加
        # -1の値（組み合わせが見つからなかったレース）を除外
        valid_races = target_trifecta >= 0
        valid_indices = np.where(valid_races)[0]  # Convert boolean mask to indices
        
        if len(valid_indices) > 0:
            # 有効なレースのみを保持
            filtered_data = {}
            
            for key, value in self.raw_data.items():
                # Type-specific filtering based on data type
                if isinstance(value, np.ndarray):
                    filtered_data[key] = value[valid_races]
                elif isinstance(value, list):
                    filtered_data[key] = [value[i] for i in valid_indices]
                else:
                    # Handle other data types or copy as is if filtering not applicable
                    filtered_data[key] = value
                    
            # Update raw_data with filtered data
            self.raw_data = filtered_data
            
            # ターゲットデータを追加
            self.raw_data['target_trifecta'] = target_trifecta[valid_races]
            
            logger.info(f"3連単ターゲット作成完了: {len(valid_indices)}/{n_races} レースで有効なターゲットを作成")
        else:
            logger.warning("有効なターゲットが作成できませんでした")
