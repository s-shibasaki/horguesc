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
                # 追加の馬券種のターゲットを作成
                self._create_additional_bet_targets()
                logger.info(f"Trifectaデータの取得完了: {len(self.raw_data['kyoso_id'])}競走、{len(results)}頭")
            else:
                logger.info(f"推論用Trifectaデータの取得完了（ターゲットなし）: {len(self.raw_data['kyoso_id'])}競走、{len(results)}頭")
            
            # Add betting odds data
            if self.mode == self.MODE_TRAIN:
                logger.info("トレーニングモードではオッズデータを追加しません")
            else:
                logger.info("オッズデータを追加します")
                self._add_betting_odds(db_ops)
    
        except Exception as e:
            logger.error(f"データ取得中にエラーが発生しました: {e}")
            raise
    
    def _convert_query_results_to_2d_arrays(self, query_results):
        """
        クエリ結果を2D配列形式のデータに変換する。馬番をインデックスとして使用する。
        
        Args:
            query_results: SQLクエリの結果（辞書のリスト）
            
        Returns:
            dict: kyoso_idリストと特徴量の2D配列を含む辞書
        """
        # 競走IDごとにデータをグループ化
        kyoso_groups = defaultdict(list)
        
        for row in query_results:
            # 競走IDを生成 - 日付をYYYYmmDD形式に、他のコードは2桁に0埋め
            kaisai_date = row['kaisai_date']
            if isinstance(kaisai_date, str):
                date_str = kaisai_date.replace('-', '')
            else:  # datetime object
                date_str = kaisai_date.strftime('%Y%m%d')
            
            keibajo_code = str(row['keibajo_code']).zfill(2)
            kaisai_kai = str(row['kaisai_kai']).zfill(2)
            kaisai_nichime = str(row['kaisai_nichime']).zfill(2)
            kyoso_bango = str(row['kyoso_bango']).zfill(2)
            
            # アンダースコアなしで連結
            kyoso_id = f"{date_str}{keibajo_code}{kaisai_kai}{kaisai_nichime}{kyoso_bango}"
            kyoso_groups[kyoso_id].append(row)
        
        # 馬番の最大値を取得 (馬番は1から始まる)
        max_horse_number = 0
        for horses in kyoso_groups.values():
            for horse in horses:
                umaban = horse.get('umaban')
                if umaban is not None and not pd.isna(umaban):
                    max_horse_number = max(max_horse_number, int(umaban))
        
        # 安全マージンを取って、最大馬番を18(JRAの通常の最大値)にする
        max_horse_number = max(max_horse_number, 18)
        logger.info(f"最大馬番: {max_horse_number}")
        
        # 特徴量データの2D配列を準備
        kyoso_ids = []
        feature_arrays = defaultdict(list)
        
        # 特徴量名一覧をクエリ結果から取得
        features = list(query_results[0].keys()) if query_results else []
        
        # 各競走のデータを処理
        for kyoso_id, horses in kyoso_groups.items():
            # 最低3頭の出走馬がいるレースのみ処理（3連単予測のため）
            if len(horses) < 3:
                logger.debug(f"出走頭数が3頭未満のレースをスキップ: {kyoso_id}")
                continue
            
            kyoso_ids.append(kyoso_id)
            
            # 馬番のマスク配列を初期化 (0: 欠番または存在しない, 1: 存在する)
            horse_mask = np.zeros(max_horse_number, dtype=np.int8)
            
            # 各特徴量の値を格納する配列を初期化
            feature_values = {feature: np.full(max_horse_number, None) for feature in features}
            
            # 各馬の特徴量を適切な馬番位置に格納
            for horse in horses:
                umaban = horse.get('umaban')
                if umaban is not None and not pd.isna(umaban):
                    horse_idx = int(umaban) - 1  # 馬番は1から始まるため、0ベースのインデックスに変換
                    if 0 <= horse_idx < max_horse_number:
                        horse_mask[horse_idx] = 1
                        for feature in features:
                            feature_values[feature][horse_idx] = horse.get(feature)
            
            # マスクを保存
            feature_arrays['horse_mask'].append(horse_mask)
            
            # 各特徴量の値を保存
            for feature, values in feature_values.items():
                feature_arrays[feature].append(values)
        
        # データを返す（kyoso_idはリストのまま、特徴量のみNumPy配列に変換）
        formatted_data = {'kyoso_id': kyoso_ids}
        
        for feature, values_list in feature_arrays.items():
            formatted_data[feature] = np.array(values_list)
        
        return formatted_data
    
    def _create_trifecta_targets(self):
        """
        3連単のターゲットを作成する
        
        - 各レースについて、確定着順に基づいて3連単の正解インデックスを特定
        - 馬番をインデックスとして使用 (インデックス = 馬番 - 1)
        - 同着の場合も正しく処理（着順の数字ではなく順位で判断）
        """
        logger.info("3連単ターゲット情報の作成を開始")
        
        if 'kakutei_chakujun' not in self.raw_data or len(self.raw_data['kakutei_chakujun']) == 0:
            logger.warning("確定着順データがないため、3連単ターゲットを作成できません")
            return
            
        # レース数と最大馬番を取得
        n_races = len(self.raw_data['kyoso_id'])
        max_horses = self.raw_data['kakutei_chakujun'].shape[1]
        
        # マスクデータを取得 (存在する馬のみを考慮)
        horse_mask = self.raw_data['horse_mask']
        
        # 全ての可能な3連単の組み合わせを生成
        all_combinations = list(itertools.permutations(range(max_horses), 3))
        
        # 各レースの正解3連単のインデックスを格納する配列
        target_trifecta = np.full(n_races, -1, dtype=np.int64)
        
        # 各レースについて処理
        for race_idx in range(n_races):
            # このレースの着順データとマスクを取得
            chakujun = self.raw_data['kakutei_chakujun'][race_idx]
            mask = horse_mask[race_idx]
            
            # 欠番や無効な馬を除外するためのマスク処理
            valid_horses = np.where(mask > 0)[0]
            
            # 有効な馬の着順データを抽出
            valid_chakujun = np.array([
                999 if (pd.isna(chakujun[i]) or chakujun[i] is None or chakujun[i] == 0) 
                else int(chakujun[i]) for i in valid_horses
            ])
            
            # 有効な馬の馬番（インデックス）と着順のペアを作成
            idx_with_chakujun = [(i, place) for i, place in zip(valid_horses, valid_chakujun)]
            
            # 着順で並べ替え
            idx_with_chakujun.sort(key=lambda x: x[1])
            
            # 上位3頭の馬インデックスを取得
            top3_indices = [pair[0] for pair in idx_with_chakujun[:3] if pair[1] < 999]
            
            # 上位3頭が揃っていることを確認
            if len(top3_indices) == 3:
                # この組み合わせがall_combinationsの何番目かを特定
                target_combo = tuple(top3_indices)
                
                # all_combinations内でのインデックスを検索
                try:
                    combo_idx = all_combinations.index(target_combo)
                    target_trifecta[race_idx] = combo_idx
                except ValueError:
                    # 組み合わせがall_combinationsに含まれない場合
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
            self.raw_data['target_sanrentan'] = target_trifecta[valid_races]
            
            logger.info(f"3連単ターゲット作成完了: {len(valid_indices)}/{n_races} レースで有効なターゲットを作成")
        else:
            logger.warning("有効なターゲットが作成できませんでした")
    
    def _add_betting_odds(self, db_ops=None):
        """
        Add betting odds data to raw_data for each race.
        
        This method loads various betting odds types using OddsLoader and adds them to the dataset,
        structured as arrays with shape [num_races, ...] to ensure compatibility with batch processing.
        
        Args:
            db_ops: Optional database operations object. If None, a new one will be created.
        """
        if 'kyoso_id' not in self.raw_data or not self.raw_data['kyoso_id']:
            logger.warning("No race data available to add betting odds")
            return
            
        # Create db_ops if not provided
        if db_ops is None:
            from horguesc.database.operations import DatabaseOperations
            db_ops = DatabaseOperations(self.config)
        
        # Import and initialize OddsLoader
        from horguesc.core.betting.odds_loader import OddsLoader
        odds_loader = OddsLoader(db_ops)
        
        # Get race IDs and necessary arrays for odds loading
        race_ids = self.raw_data['kyoso_id']
        
        # Extract umaban and wakuban arrays
        umaban_array = self.raw_data.get('umaban')
        wakuban_array = self.raw_data.get('wakuban')
        
        # Load all odds types that we want to include
        # odds_types = [
        #     odds_loader.TANSHO,     # Win odds
        #     odds_loader.FUKUSHO,    # Place odds
        #     odds_loader.SANRENTAN,  # Trifecta odds - most important for this task
        # ]
        odds_types = None  # Load all odds types
        
        logger.info(f"Loading odds data for {len(race_ids)} races")
        
        try:
            # Load odds data using OddsLoader
            odds_data = odds_loader.load_odds(
                race_ids=race_ids,
                horse_numbers=umaban_array,
                frame_numbers=wakuban_array,
                odds_types=odds_types
            )
            
            # Add odds data to raw_data with prefix 'odds_'
            for odds_type, odds_array in odds_data.items():
                self.raw_data[f'odds_{odds_type}'] = odds_array
                logger.info(f"Added {odds_type} odds data with shape {odds_array.shape}")
                
        except Exception as e:
            logger.error(f"Error loading odds data: {e}", exc_info=True)
    
    def _create_additional_bet_targets(self):
        """
        三連単(trifecta)以外の馬券種のターゲットを作成する
    
        - 単勝 (tansho/win): 1着の馬
        - 複勝 (fukusho/place): 3着以内の馬
        - 枠連 (wakuren/bracket quinella): 1着と2着の枠番の組み合わせ（順不同）
        - 馬連 (umaren/quinella): 1着と2着の馬番の組み合わせ（順不同）
        - ワイド (wide/quinella place): 3着以内に2頭の馬が入る組み合わせ（順不同）
        - 馬単 (umatan/exacta): 1着と2着の馬番の組み合わせ（順序あり）
        - 三連複 (sanrenpuku/trio): 3着以内の馬番の組み合わせ（順不同）
        """
        logger.info("追加の馬券種ターゲット情報の作成を開始")
        
        if 'kakutei_chakujun' not in self.raw_data or len(self.raw_data['kakutei_chakujun']) == 0:
            logger.warning("確定着順データがないため、追加馬券種のターゲットを作成できません")
            return
        
        # レース数と最大馬番を取得
        n_races = len(self.raw_data['kyoso_id'])
        max_horses = self.raw_data['kakutei_chakujun'].shape[1]
        
        # マスクデータを取得
        horse_mask = self.raw_data['horse_mask']
        
        # 各馬券種のターゲットを格納する配列を初期化
        target_tansho = np.full(n_races, -1, dtype=np.int64)                        # 単勝
        target_fukusho = np.zeros((n_races, max_horses), dtype=np.int64)            # 複勝
        target_wakuren = np.full(n_races, -1, dtype=np.int64)                       # 枠連
        target_umaren = np.full(n_races, -1, dtype=np.int64)                        # 馬連
        target_wide = np.zeros((n_races, (max_horses * (max_horses-1))//2), dtype=np.int64)  # ワイド
        target_umatan = np.full(n_races, -1, dtype=np.int64)                        # 馬単
        target_sanrenpuku = np.full(n_races, -1, dtype=np.int64)                    # 3連複
        
        # 組み合わせインデックスのマッピングを作成
        # 馬連・ワイド・3連複用（順不同）
        umaren_combinations = list(itertools.combinations(range(max_horses), 2))
        sanrenpuku_combinations = list(itertools.combinations(range(max_horses), 3))
        
        # 馬単用（順序あり）
        umatan_combinations = list(itertools.permutations(range(max_horses), 2))
        
        # 枠連用（枠番の組み合わせ、順不同）
        max_frames = 8  # 日本の競馬では最大8枠
        wakuren_combinations = []
        for i in range(1, max_frames + 1):
            for j in range(i, max_frames + 1):  # i以上のjで組み合わせを作成（自枠との組み合わせも含む）
                wakuren_combinations.append((i, j))
        
        # 各レースについて処理
        for race_idx in range(n_races):
            # このレースの着順データとマスクを取得
            chakujun = self.raw_data['kakutei_chakujun'][race_idx]
            mask = horse_mask[race_idx]
            
            # 枠番データを取得（枠連用）
            wakuban = self.raw_data['wakuban'][race_idx] if 'wakuban' in self.raw_data else None
            
            # 欠番や無効な馬を除外するためのマスク処理
            valid_horses = np.where(mask > 0)[0]
            
            # 有効な馬の着順データを抽出
            valid_chakujun = np.array([
                999 if (pd.isna(chakujun[i]) or chakujun[i] is None or chakujun[i] == 0) 
                else int(chakujun[i]) for i in valid_horses
            ])
            
            # 有効な馬の馬番（インデックス）と着順のペアを作成
            idx_with_chakujun = [(i, place) for i, place in zip(valid_horses, valid_chakujun)]
            
            # 着順で並べ替え
            idx_with_chakujun.sort(key=lambda x: x[1])
            
            # 上位3頭の馬インデックスを取得
            top3_indices = [pair[0] for pair in idx_with_chakujun[:3] if pair[1] < 999]
            
            if len(top3_indices) >= 1:
                # 単勝（1着の馬）
                target_tansho[race_idx] = top3_indices[0]
            
            if len(top3_indices) >= 1:
                # 複勝（3着以内の馬にフラグ）
                for horse_idx in top3_indices:
                    if horse_idx < max_horses:  # 範囲内であることを確認
                        target_fukusho[race_idx, horse_idx] = 1
            
            if len(top3_indices) >= 2 and wakuban is not None:
                # 枠連（1着と2着の枠番の組み合わせ）
                frame1 = wakuban[top3_indices[0]]
                frame2 = wakuban[top3_indices[1]]
                
                # NoneやNaNをチェック
                if not (pd.isna(frame1) or pd.isna(frame2) or frame1 is None or frame2 is None):
                    frame1, frame2 = int(frame1), int(frame2)
                    
                    # 枠番の順序を正規化（小さい方を先に）
                    if frame1 > frame2:
                        frame1, frame2 = frame2, frame1
                    
                    # 組み合わせのインデックスを検索
                    try:
                        combo_idx = wakuren_combinations.index((frame1, frame2))
                        target_wakuren[race_idx] = combo_idx
                    except ValueError:
                        pass
            
            if len(top3_indices) >= 2:
                # 馬連（1着と2着の馬番の組み合わせ、順不同）
                horse1, horse2 = top3_indices[0], top3_indices[1]
                
                # 馬番の順序を正規化（小さい方を先に）
                if horse1 > horse2:
                    horse1, horse2 = horse2, horse1
                
                # 組み合わせのインデックスを検索
                try:
                    combo_idx = umaren_combinations.index((horse1, horse2))
                    target_umaren[race_idx] = combo_idx
                except ValueError:
                    pass
                
                # 馬単（1着と2着の馬番の組み合わせ、順序あり）
                try:
                    combo_idx = umatan_combinations.index((top3_indices[0], top3_indices[1]))
                    target_umatan[race_idx] = combo_idx
                except ValueError:
                    pass
            
            if len(top3_indices) >= 3:
                # 3連複（3着以内の馬番の組み合わせ、順不同）
                horses = sorted([top3_indices[0], top3_indices[1], top3_indices[2]])
                try:
                    combo_idx = sanrenpuku_combinations.index(tuple(horses))
                    target_sanrenpuku[race_idx] = combo_idx
                except ValueError:
                    pass
            
            # ワイド（3着以内の2頭の馬の組み合わせ、順不同）
            if len(top3_indices) >= 2:
                # より効率的な実装: 組み合わせをベクトル化
                pairs = list(itertools.combinations(top3_indices, 2))
                for pair in pairs:
                    horse1, horse2 = pair
                    # 馬番の順序を正規化（小さい方を先に）
                    if horse1 > horse2:
                        horse1, horse2 = horse2, horse1
                    
                    try:
                        combo_idx = umaren_combinations.index((horse1, horse2))
                        target_wide[race_idx, combo_idx] = 1
                    except ValueError:
                        pass
        
        # 有効なターゲットがあるもののみを追加
        if np.any(target_tansho >= 0):
            self.raw_data['target_tansho'] = target_tansho
            logger.info(f"単勝ターゲット作成完了: {np.sum(target_tansho >= 0)}/{n_races} レースで有効")
        
        if np.any(target_fukusho):
            self.raw_data['target_fukusho'] = target_fukusho
            logger.info(f"複勝ターゲット作成完了: {np.sum(np.any(target_fukusho, axis=1))}/{n_races} レースで有効")
        
        if np.any(target_wakuren >= 0):
            self.raw_data['target_wakuren'] = target_wakuren
            logger.info(f"枠連ターゲット作成完了: {np.sum(target_wakuren >= 0)}/{n_races} レースで有効")
        
        if np.any(target_umaren >= 0):
            self.raw_data['target_umaren'] = target_umaren
            logger.info(f"馬連ターゲット作成完了: {np.sum(target_umaren >= 0)}/{n_races} レースで有効")
        
        if np.any(target_wide):
            self.raw_data['target_wide'] = target_wide
            logger.info(f"ワイドターゲット作成完了: {np.sum(np.any(target_wide, axis=1))}/{n_races} レースで有効")
        
        if np.any(target_umatan >= 0):
            self.raw_data['target_umatan'] = target_umatan
            logger.info(f"馬単ターゲット作成完了: {np.sum(target_umatan >= 0)}/{n_races} レースで有効")
        
        if np.any(target_sanrenpuku >= 0):
            self.raw_data['target_sanrenpuku'] = target_sanrenpuku
            logger.info(f"3連複ターゲット作成完了: {np.sum(target_sanrenpuku >= 0)}/{n_races} レースで有効")
