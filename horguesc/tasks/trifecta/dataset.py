from horguesc.core.base.dataset import BaseDataset
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
from horguesc.database.sql_builder import SQLBuilder
import itertools
import torch

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
        
        # 事前定義された特徴量名リストを使用
        features = list(self.FEATURE_DEFINITIONS.keys())
        
        # 各競走のデータを処理
        for kyoso_id, horses in kyoso_groups.items():
            # 最低3頭の出走馬がいるレースのみ処理（3連単予測のため）
            if len(horses) < 3:
                logger.debug(f"出走頭数が3頭未満のレースをスキップ: {kyoso_id}")
                continue
                
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
    
    def _create_trifecta_targets(self):
        """
        3連単のターゲットを作成する
        
        - 各レースについて、馬番と確定着順の対応を作成
        - 全ての可能な3連単順列を生成
        - 実際の着順1-2-3を正解とするターゲットを作成
        """
        if 'kakutei_chakujun' not in self.raw_data or 'umaban' not in self.raw_data:
            logger.error("確定着順またはumaban（馬番）データがありません")
            return
        
        num_races = len(self.raw_data['kyoso_id'])
        if num_races == 0:
            logger.warning("レースデータがありません")
            return
        
        max_horses = self.raw_data['umaban'].shape[1]
        
        # 全ての馬番での可能な3連単順列を生成
        all_permutations = list(itertools.permutations(range(1, max_horses + 1), 3))
        
        # ターゲットインデックス配列を初期化
        target_indices = np.zeros(num_races, dtype=np.int64)
        
        for race_idx in range(num_races):
            umaban_array = self.raw_data['umaban'][race_idx]
            chakujun_array = self.raw_data['kakutei_chakujun'][race_idx]
            
            # 有効な馬番と着順のペアを作成
            valid_horses = [(int(umaban), int(chakujun)) 
                           for umaban, chakujun in zip(umaban_array, chakujun_array) 
                           if umaban is not None and chakujun is not None]
            
            # 着順でソート
            valid_horses.sort(key=lambda x: x[1])
            
            # 上位3着の馬番を取得
            top3_horses = [horse[0] for horse in valid_horses[:3]]
            
            # 3頭揃わない場合はスキップ
            if len(top3_horses) < 3:
                logger.debug(f"レース {self.raw_data['kyoso_id'][race_idx]} は3着までの馬が揃っていません")
                continue
            
            # 正解の順列を作成
            correct_perm = tuple(top3_horses)
            
            # 正解順列のインデックスを見つける
            try:
                correct_index = all_permutations.index(correct_perm)
                target_indices[race_idx] = correct_index
            except ValueError:
                # 馬番がリスト範囲外の場合など
                logger.warning(f"レース {self.raw_data['kyoso_id'][race_idx]} の正解順列 {correct_perm} が見つかりませんでした")
        
        # ターゲットデータを追加
        self.raw_data['target_trifecta'] = target_indices
        self.raw_data['permutations'] = all_permutations
        
        logger.info(f"3連単ターゲットの作成完了: {len(all_permutations)}通りの順列、{num_races}レース")
    
