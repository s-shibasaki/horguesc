from horguesc.core.base.dataset import BaseDataset
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
from horguesc.database.sql_builder import SQLBuilder

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
        
        # ターゲット変数
        'target': 'se.kakutei_chakujun',
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
    
    def get_name(self) -> str:
        """データセット名を返す"""
        return "Trifecta"
        
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
        
        # 事前定義された特徴量名リストを使用
        features = list(self.FEATURE_DEFINITIONS.keys())
        
        # 各競走のデータを処理
        for kyoso_id, horses in kyoso_groups.items():
            kyoso_ids.append(kyoso_id)
            
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
            'kyoso_id': kyoso_ids
        }
        
        for feature, values_list in feature_arrays.items():
            formatted_data[feature] = np.array(values_list)
        
        return formatted_data