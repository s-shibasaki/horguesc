from horguesc.core.base.dataset import BaseDataset
import logging
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from horguesc.database.sql_builder import SQLBuilder
import itertools
import torch
import pandas as pd

logger = logging.getLogger(__name__)

class TrifectaDataset(BaseDataset):
    """Trifectaタスク向けのデータセットクラス。"""
    
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
        
        # Get base_date for days_before calculation
        if self.mode == self.MODE_TRAIN and self.end_date:
            # For training data, use end_date as the base date
            base_date_str = self.end_date.strftime('%Y-%m-%d')
            days_before_expr = f"CASE WHEN se.kaisai_date IS NOT NULL THEN DATE '{base_date_str}' - se.kaisai_date ELSE NULL END"
            logger.info(f"トレーニングモード: days_before の基準日を {base_date_str} に設定します")
        else:
            # For evaluation/inference mode, always use 0
            days_before_expr = "0"
            logger.info(f"評価/推論モード: days_before を常に 0 に設定します")
        
        # SQLBuilderでクエリを構築
        builder = SQLBuilder("se")
        
        # 競走IDのためのカラムを追加
        builder.select_as("se.kaisai_date", "kaisai_date")
        builder.select_as("se.keibajo_code", "keibajo_code")
        builder.select_as("se.kaisai_kai", "kaisai_kai")
        builder.select_as("se.kaisai_nichime", "kaisai_nichime")
        builder.select_as("se.kyoso_bango", "kyoso_bango")
        
        # 数値特徴量を追加
        builder.select_as("CASE WHEN se.wakuban != 0 THEN se.wakuban ELSE NULL END", "wakuban")
        builder.select_as("CASE WHEN se.umaban != 0 THEN se.umaban ELSE NULL END", "umaban")
        builder.select_as("""
            CASE 
                WHEN se.data_type = '2' THEN 
                    COALESCE(
                        (SELECT CASE WHEN jc.futan_juryo != 0 THEN jc.futan_juryo ELSE NULL END 
                        FROM jc 
                        WHERE jc.kaisai_date = se.kaisai_date
                        AND jc.keibajo_code = se.keibajo_code
                        AND jc.kaisai_kai = se.kaisai_kai
                        AND jc.kaisai_nichime = se.kaisai_nichime
                        AND jc.kyoso_bango = se.kyoso_bango
                        AND jc.umaban = se.umaban
                        ORDER BY jc.happyo_datetime DESC
                        LIMIT 1),
                        CASE WHEN se.futan_juryo != 0 THEN se.futan_juryo ELSE NULL END
                    )
                ELSE CASE WHEN se.futan_juryo != 0 THEN se.futan_juryo ELSE NULL END
            END
        """, "futan_juryo")
        builder.select_as("""
            CASE 
                WHEN se.data_type = '2' THEN 
                    COALESCE(
                        (SELECT CASE 
                            WHEN wh.bataiju BETWEEN 2 AND 998 THEN wh.bataiju 
                            ELSE NULL 
                        END 
                        FROM wh 
                        WHERE wh.kaisai_date = se.kaisai_date
                        AND wh.keibajo_code = se.keibajo_code
                        AND wh.kaisai_kai = se.kaisai_kai
                        AND wh.kaisai_nichime = se.kaisai_nichime
                        AND wh.kyoso_bango = se.kyoso_bango
                        AND wh.umaban = se.umaban
                        LIMIT 1),
                        CASE 
                            WHEN se.bataiju BETWEEN 2 AND 998 THEN se.bataiju 
                            ELSE NULL 
                        END
                    )
                ELSE 
                    CASE 
                        WHEN se.bataiju BETWEEN 2 AND 998 THEN se.bataiju 
                        ELSE NULL 
                    END
            END
        """, "bataiju")
        builder.select_as("""
            CASE 
                WHEN se.data_type = '2' THEN 
                    COALESCE(
                        (SELECT CASE 
                            WHEN wh.zogensa BETWEEN -998 AND 998 THEN wh.zogensa 
                            ELSE NULL 
                        END 
                        FROM wh 
                        WHERE wh.kaisai_date = se.kaisai_date
                        AND wh.keibajo_code = se.keibajo_code
                        AND wh.kaisai_kai = se.kaisai_kai
                        AND wh.kaisai_nichime = se.kaisai_nichime
                        AND wh.kyoso_bango = se.kyoso_bango
                        AND wh.umaban = se.umaban
                        LIMIT 1),
                        CASE 
                            WHEN se.zogensa BETWEEN -998 AND 998 THEN se.zogensa 
                            ELSE NULL 
                        END
                    )
                ELSE
                    CASE 
                        WHEN se.zogensa BETWEEN -998 AND 998 THEN se.zogensa 
                        ELSE NULL 
                    END
            END
        """, "zogensa")
        builder.select_as("""
            CASE 
                WHEN se.data_type = '2' THEN 
                    COALESCE(
                        (SELECT CASE
                            WHEN cc.kyori BETWEEN 0 AND 9999 THEN cc.kyori
                            ELSE NULL
                        END
                        FROM cc 
                        WHERE cc.kaisai_date = se.kaisai_date
                        AND cc.keibajo_code = se.keibajo_code
                        AND cc.kaisai_kai = se.kaisai_kai
                        AND cc.kaisai_nichime = se.kaisai_nichime
                        AND cc.kyoso_bango = se.kyoso_bango),
                        CASE
                            WHEN ra.kyori BETWEEN 0 AND 9999 THEN ra.kyori
                            ELSE NULL
                        END
                    )
                ELSE
                    CASE
                        WHEN ra.kyori != 0 THEN ra.kyori 
                        ELSE NULL 
                    END
            END
        """, "kyori")
        builder.select_as(days_before_expr, "days_before")
        builder.select_as("CASE WHEN ks.birth_date IS NOT NULL THEN EXTRACT(YEAR FROM AGE(se.kaisai_date, ks.birth_date)) ELSE NULL END", "kishu_age")
        builder.select_as("CASE WHEN um.birth_date IS NOT NULL THEN EXTRACT(YEAR FROM AGE(se.kaisai_date, um.birth_date)) ELSE NULL END", "uma_age")
        
        # カテゴリ特徴量を追加
        builder.select_as("CASE WHEN se.ketto_toroku_bango != 0 THEN se.ketto_toroku_bango ELSE NULL END", "ketto_toroku_bango")
        builder.select_as("se.blinker_shiyo_kubun", "blinker_shiyo_kubun")
        builder.select_as("""
            CASE 
                WHEN se.data_type = '2' THEN 
                    COALESCE(
                        (SELECT CASE
                            WHEN jc.kishu_code != 0 THEN jc.kishu_code
                            ELSE NULL
                        END 
                        FROM jc 
                        WHERE jc.kaisai_date = se.kaisai_date
                        AND jc.keibajo_code = se.keibajo_code
                        AND jc.kaisai_kai = se.kaisai_kai
                        AND jc.kaisai_nichime = se.kaisai_nichime
                        AND jc.kyoso_bango = se.kyoso_bango
                        AND jc.umaban = se.umaban
                        ORDER BY jc.happyo_datetime DESC
                        LIMIT 1),
                        CASE 
                            WHEN se.kishu_code != 0 THEN se.kishu_code
                            ELSE NULL
                        END
                    )
                ELSE 
                    CASE 
                        WHEN se.kishu_code != 0 THEN se.kishu_code
                        ELSE NULL
                    END
            END
        """, "kishu_code")
        builder.select_as("""
            CASE 
                WHEN se.data_type = '2' THEN 
                    COALESCE(
                        (SELECT jc.kishu_minarai_code 
                        FROM jc 
                        WHERE jc.kaisai_date = se.kaisai_date
                        AND jc.keibajo_code = se.keibajo_code
                        AND jc.kaisai_kai = se.kaisai_kai
                        AND jc.kaisai_nichime = se.kaisai_nichime
                        AND jc.kyoso_bango = se.kyoso_bango
                        AND jc.umaban = se.umaban
                        ORDER BY jc.happyo_datetime DESC
                        LIMIT 1),
                        se.kishu_minarai_code
                    )
                ELSE se.kishu_minarai_code
            END
        """, "kishu_minarai_code")
        builder.select_as("""
            CASE 
                WHEN se.data_type = '2' THEN 
                    COALESCE(
                        (SELECT CASE
                            WHEN cc.track_code != '00' THEN cc.track_code
                            ELSE NULL
                        END
                        FROM cc 
                        WHERE cc.kaisai_date = se.kaisai_date
                        AND cc.keibajo_code = se.keibajo_code
                        AND cc.kaisai_kai = se.kaisai_kai
                        AND cc.kaisai_nichime = se.kaisai_nichime
                        AND cc.kyoso_bango = se.kyoso_bango),
                        CASE 
                            WHEN ra.track_code != '00' THEN ra.track_code
                            ELSE NULL
                        END
                    )
                ELSE
                    CASE
                        WHEN ra.track_code != '00' THEN ra.track_code
                        ELSE NULL
                    END
            END
        """, "track_code")
        builder.select_as("CASE WHEN ra.course_kubun != '0' THEN ra.course_kubun ELSE NULL END", "course_kubun")
        builder.select_as("""
            CASE 
                WHEN se.data_type = '2' THEN 
                    (SELECT we.tenko_code
                    FROM we 
                    WHERE we.kaisai_date = ra.kaisai_date
                    AND we.keibajo_code = ra.keibajo_code
                    AND we.kaisai_kai = ra.kaisai_kai
                    AND we.kaisai_nichime = ra.kaisai_nichime
                    AND we.happyo_datetime < (ra.kaisai_date + ra.hasso_time)
                    AND we.tenko_code != '0'
                    ORDER BY we.happyo_datetime DESC
                    LIMIT 1)
                ELSE
                    CASE
                        WHEN ra.tenko_code != '0' THEN ra.tenko_code
                        ELSE NULL
                    END
            END
        """, "tenko_code")
        builder.select_as("""
            CASE 
                WHEN se.data_type = '2' THEN 
                    CASE 
                        WHEN CAST(ra.track_code AS INTEGER) BETWEEN 10 AND 22 
                          OR CAST(ra.track_code AS INTEGER) = 51 
                          OR CAST(ra.track_code AS INTEGER) BETWEEN 53 AND 59 
                        THEN 
                            (SELECT we.babajotai_code_shiba 
                            FROM we 
                            WHERE we.kaisai_date = ra.kaisai_date
                            AND we.keibajo_code = ra.keibajo_code
                            AND we.kaisai_kai = ra.kaisai_kai
                            AND we.kaisai_nichime = ra.kaisai_nichime
                            AND we.happyo_datetime < (ra.kaisai_date + ra.hasso_time)
                            AND we.babajotai_code_shiba != '0'
                            ORDER BY we.happyo_datetime DESC
                            LIMIT 1)
                        WHEN CAST(ra.track_code AS INTEGER) BETWEEN 23 AND 29 
                          OR CAST(ra.track_code AS INTEGER) = 52 
                        THEN 
                            (SELECT we.babajotai_code_dirt 
                            FROM we 
                            WHERE we.kaisai_date = ra.kaisai_date
                            AND we.keibajo_code = ra.keibajo_code
                            AND we.kaisai_kai = ra.kaisai_kai
                            AND we.kaisai_nichime = ra.kaisai_nichime
                            AND we.happyo_datetime < (ra.kaisai_date + ra.hasso_time)
                            AND we.babajotai_code_dirt != '0'
                            ORDER BY we.happyo_datetime DESC
                            LIMIT 1)
                        ELSE 
                            CASE
                                WHEN ra.babajotai_code != '0' THEN ra.babajotai_code
                                ELSE NULL
                            END
                    END
                ELSE 
                    CASE
                        WHEN ra.babajotai_code != '0' THEN ra.babajotai_code
                        ELSE NULL
                    END
            END
        """, "babajotai_code")
        builder.select_as("CASE WHEN um.chokyoshi_code != 0 THEN um.chokyoshi_code ELSE NULL END", "chokyoshi_code")
        builder.select_as("CASE WHEN ks.seibetsu_kubun != '0' AND ks.seibetsu_kubun IS NOT NULL THEN ks.seibetsu_kubun ELSE NULL END", "kishu_seibetsu")
        builder.select_as("CASE WHEN um.seibetsu_code != '0' AND um.seibetsu_code IS NOT NULL THEN um.seibetsu_code ELSE NULL END", "uma_seibetsu")
        
        # 3代血統情報を追加（条件に基づいて処理）
        for i in range(1, 15):
            idx_str = f"{i:02d}"
            builder.select_as(
                f"CASE "
                f"WHEN hn{idx_str}.ketto_toroku_bango IS NOT NULL AND hn{idx_str}.ketto_toroku_bango != 0 THEN hn{idx_str}.ketto_toroku_bango "
                f"WHEN um.hanshoku_toroku_bango_{idx_str} != 0 THEN um.hanshoku_toroku_bango_{idx_str} "
                f"ELSE NULL END", 
                f"hanshoku_toroku_bango_{idx_str}"
            )
        
        # ターゲット変数を追加
        builder.select_as("CASE WHEN se.kakutei_chakujun != 0 THEN se.kakutei_chakujun ELSE NULL END", "kakutei_chakujun")
        
        # 必要なJOINを追加
        builder.join('LEFT JOIN ra ON se.kaisai_date = ra.kaisai_date AND se.keibajo_code = ra.keibajo_code AND se.kaisai_kai = ra.kaisai_kai AND se.kaisai_nichime = ra.kaisai_nichime AND se.kyoso_bango = ra.kyoso_bango')
        builder.join('LEFT JOIN um ON se.ketto_toroku_bango = um.ketto_toroku_bango')
        builder.join('LEFT JOIN ks ON se.kishu_code = ks.kishu_code')
        
        # 血統情報のためのJOINを追加
        for i in range(1, 15):
            idx_str = f"{i:02d}"
            builder.join(f'LEFT JOIN hn hn{idx_str} ON um.hanshoku_toroku_bango_{idx_str} = hn{idx_str}.hanshoku_toroku_bango')
        
        # データフィルタ条件を追加
        builder.where("se.data_type IN ('7', '2')")
        builder.where("(se.ijo_kubun_code IS NULL OR se.ijo_kubun_code NOT IN ('1', '2', '3', '4', '5'))")

        # data_typeが'2'のときにavテーブルとの関連をチェックする条件を追加
        builder.where("""
            CASE 
                WHEN se.data_type = '2' THEN 
                    NOT EXISTS (
                        SELECT 1 
                        FROM av 
                        WHERE av.kaisai_date = se.kaisai_date 
                        AND av.keibajo_code = se.keibajo_code 
                        AND av.kaisai_kai = se.kaisai_kai 
                        AND av.kaisai_nichime = se.kaisai_nichime 
                        AND av.kyoso_bango = se.kyoso_bango 
                        AND av.umaban = se.umaban 
                        AND av.data_type IN ('1', '2')
                    )
                ELSE TRUE
            END
        """)

        # 推論モードではなく、訓練・評価モードのときだけ着順フィルタを適用
        if self.mode != self.MODE_INFERENCE:
            builder.where("(se.kakutei_chakujun IS NOT NULL AND se.kakutei_chakujun != 0)")  # 確定着順がある馬のみ
        
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
            
            # 推論モード以外の場合のみ、3連単のターゲットを作成
            if self.mode != self.MODE_INFERENCE:
                self._create_trifecta_targets()
                logger.info(f"Trifectaデータの取得完了: {len(self.raw_data['kyoso_id'])}競走、{len(results)}頭")
            else:
                logger.info(f"推論用Trifectaデータの取得完了（ターゲットなし）: {len(self.raw_data['kyoso_id'])}競走、{len(results)}頭")
        
        except Exception as e:
            logger.error(f"データ取得中にエラーが発生しました: {e}")
            raise
    
    def _convert_query_results_to_2d_arrays(self, query_results):
        """
        クエリ結果を2D配列形式のデータに変換する
        
        Args:
            query_results: SQLクエリの結果（辞書のリスト）
            
        Returns:
            dict: kyoso_idリストと特徴量の2D配列を含む辞
        """
        # 競走IDごとにデータをグループ化
        kyoso_groups = defaultdict(list)
        
        # 競走IDを構成するカラムをクエリから取得
        kyoso_id_columns = ['kaisai_date', 'keibajo_code', 'kaisai_kai', 'kaisai_nichime', 'kyoso_bango']
        
        for row in query_results:
            # 競走IDを生成
            kyoso_id_parts = [str(row[col]) for col in kyoso_id_columns]
            kyoso_id = '_'.join(kyoso_id_parts)
            kyoso_groups[kyoso_id].append(row)
        
        # 最大出走頭数を計算
        max_horses = max(len(horses) for horses in kyoso_groups.values())
        logger.info(f"最大出走頭数: {max_horses}")
        
        # 特徴量データの2D配列を準備
        kyoso_ids = []
        feature_arrays = defaultdict(list)
        horse_counts = []  # 各レースの実際の出走馬数を保存
        
        # 特徴量名一覧をクエリ結果から取得
        features = list(query_results[0].keys()) if query_results else []
        
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
            
            # 実際の出走馬数（馬の数）
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
