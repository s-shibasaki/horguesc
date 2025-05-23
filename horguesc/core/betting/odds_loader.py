"""
Module for loading betting odds from the database.
"""
import logging
import numpy as np
import itertools
import time  # Added for timing measurements
from typing import List, Dict, Optional, Tuple, Any, Set
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class OddsLoader:
    """
    Loads betting odds from the database for specified races.
    """
    
    # Constants for odds types
    TANSHO = 'tansho'          # Win
    FUKUSHO = 'fukusho'        # Place
    WAKUREN = 'wakuren'        # Bracket Quinella
    UMAREN = 'umaren'          # Quinella
    WIDE = 'wide'              # Quinella Place
    UMATAN = 'umatan'          # Exacta
    SANRENPUKU = 'sanrenpuku'  # Trio
    SANRENTAN = 'sanrentan'    # Trifecta
    
    ALL_ODDS_TYPES = [TANSHO, FUKUSHO, WAKUREN, UMAREN, WIDE, UMATAN, SANRENPUKU, SANRENTAN]
    
    def __init__(self, db_ops, config):
        """
        Initialize the OddsLoader with database operations.
        
        Args:
            db_ops: Database operations object
        """
        self.db_ops = db_ops
        self.config = config
        self.max_frame_number = config.getint('features', 'max_frame_number', fallback=8)  # Maximum frame number (defined in requirements)
        self.max_horse_number = config.getint('features', 'max_horse_number', fallback=18)  # Maximum horse number (defined in requirements)
    
    def load_odds(self, 
                  race_ids: List[str], 
                  horse_numbers: np.ndarray, 
                  frame_numbers: np.ndarray, 
                  odds_types: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Load odds data for specified race IDs and odds types.
        
        Args:
            race_ids: List of race IDs
            horse_numbers: Array of horse numbers with shape (num_races, max_horses)
            frame_numbers: Array of frame numbers with shape (num_races, max_frames)
            odds_types: List of odds types to load (if None, load all types)
            
        Returns:
            Dictionary mapping odds types to numpy arrays of shape (num_races, combinations)
        """
        start_time = time.time()
        logger.info(f"Starting to load odds for {len(race_ids)} races")

        if odds_types is None:
            odds_types = self.ALL_ODDS_TYPES
            
        # Validate input
        if not race_ids:
            logger.warning("No race IDs provided")
            return {}
            
        # Get date range from race IDs
        min_date_str, max_date_str = self._get_date_range_from_race_ids(race_ids)
        logger.info(f"Date range: {min_date_str} to {max_date_str}")
        
        # Dictionary to store results
        odds_data = {}
        
        # Load each requested odds type
        for odds_type in odds_types:
            type_start_time = time.time()
            logger.info(f"Starting to load {odds_type} odds...")
            
            if odds_type == self.TANSHO:
                odds_data[odds_type] = self._load_tansho(race_ids, horse_numbers, min_date_str, max_date_str)
            elif odds_type == self.FUKUSHO:
                odds_data[odds_type] = self._load_fukusho(race_ids, horse_numbers, min_date_str, max_date_str)
            elif odds_type == self.WAKUREN:
                odds_data[odds_type] = self._load_wakuren(race_ids, frame_numbers, min_date_str, max_date_str)
            elif odds_type == self.UMAREN:
                odds_data[odds_type] = self._load_umaren(race_ids, horse_numbers, min_date_str, max_date_str)
            elif odds_type == self.WIDE:
                odds_data[odds_type] = self._load_wide(race_ids, horse_numbers, min_date_str, max_date_str)
            elif odds_type == self.UMATAN:
                odds_data[odds_type] = self._load_umatan(race_ids, horse_numbers, min_date_str, max_date_str)
            elif odds_type == self.SANRENPUKU:
                odds_data[odds_type] = self._load_sanrenpuku(race_ids, horse_numbers, min_date_str, max_date_str)
            elif odds_type == self.SANRENTAN:
                odds_data[odds_type] = self._load_sanrentan(race_ids, horse_numbers, min_date_str, max_date_str)
                
            type_elapsed_time = time.time() - type_start_time
            logger.info(f"Completed loading {odds_type} odds in {type_elapsed_time:.2f} seconds")
        
        total_elapsed_time = time.time() - start_time
        logger.info(f"Completed loading all odds in {total_elapsed_time:.2f} seconds")
        
        return odds_data
    
    def _get_date_range_from_race_ids(self, race_ids: List[str]) -> Tuple[str, str]:
        """
        Extract date range from race IDs.
        
        Args:
            race_ids: List of race IDs where first 8 chars represent date (YYYYMMDD)
            
        Returns:
            Tuple of (min_date_str, max_date_str) in YYYY-MM-DD format
        """
        dates = [race_id[:8] for race_id in race_ids]
        min_date_str = min(dates)
        max_date_str = max(dates)
        
        # Convert to YYYY-MM-DD format
        min_date_str = f"{min_date_str[:4]}-{min_date_str[4:6]}-{min_date_str[6:8]}"
        max_date_str = f"{max_date_str[:4]}-{max_date_str[4:6]}-{max_date_str[6:8]}"
        
        return min_date_str, max_date_str
    
    def _load_tansho(self, race_ids: List[str], horse_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load win odds (単勝) using pandas for optimization.
        """
        start_time = time.time()
        logger.info(f"Beginning tansho query...")
        
        # Query to get the latest tansho odds for each race
        query = """
        WITH latest_odds AS (
            SELECT 
                kaisai_date,
                keibajo_code,
                kaisai_kai,
                kaisai_nichime,
                kyoso_bango,
                umaban,
                odds,
                ROW_NUMBER() OVER(PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban 
                                  ORDER BY happyo_datetime DESC) as rn
            FROM 
                tansho
            WHERE 
                kaisai_date BETWEEN %s AND %s
        )
        SELECT 
            kaisai_date,
            keibajo_code,
            kaisai_kai,
            kaisai_nichime,
            kyoso_bango,
            umaban,
            odds
        FROM 
            latest_odds
        WHERE 
            rn = 1
        """
        
        try:
            # Execute query and get results as DataFrame
            query_start_time = time.time()
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            query_elapsed_time = time.time() - query_start_time
            logger.info(f"Tansho database query completed in {query_elapsed_time:.2f} seconds")
            
            if not results:
                logger.info("No tansho odds found for the specified races")
                return np.full_like(horse_numbers, np.nan, dtype=np.float32)
            
            logger.info(f"Tansho query returned {len(results)} rows")
                
            df_start_time = time.time()
            df = pd.DataFrame([dict(row) for row in results])
            df_elapsed_time = time.time() - df_start_time
            logger.info(f"Created pandas DataFrame in {df_elapsed_time:.2f} seconds")
            
            # Convert to numeric for calculations
            convert_start_time = time.time()
            df['umaban'] = df['umaban'].astype(int)
            df['odds'] = df['odds'].astype(float) / 10.0  # Convert to decimal odds
            convert_elapsed_time = time.time() - convert_start_time
            logger.info(f"Converted data types in {convert_elapsed_time:.2f} seconds")

            # umabanの最大値と最小値を出力
            max_umaban = df['umaban'].max()
            min_umaban = df['umaban'].min()
            logger.info(f"Max umaban: {max_umaban}, Min umaban: {min_umaban}")            
            
            # Create a dictionary to map race_id to index in the array
            index_start_time = time.time()
            race_id_to_index = {race_id: idx for idx, race_id in enumerate(race_ids)}
            index_elapsed_time = time.time() - index_start_time
            logger.info(f"Created race_id to index mapping in {index_elapsed_time:.2f} seconds")
            
            # Convert kaisai_date to datetime if it's not already
            date_start_time = time.time()
            df['kaisai_date'] = pd.to_datetime(df['kaisai_date'])

            # Then create race_id as before
            df['race_id'] = df['kaisai_date'].dt.strftime('%Y%m%d') + df['keibajo_code'] + df['kaisai_kai'].astype(str).str.zfill(2) + df['kaisai_nichime'].astype(str).str.zfill(2) + df['kyoso_bango'].astype(str).str.zfill(2)
            date_elapsed_time = time.time() - date_start_time
            logger.info(f"Created race_id field in {date_elapsed_time:.2f} seconds")

            # Initialize odds_array with NaN values
            init_start_time = time.time()
            odds_array = np.full_like(horse_numbers, np.nan, dtype=np.float32)
            init_elapsed_time = time.time() - init_start_time
            logger.info(f"Initialized odds array in {init_elapsed_time:.2f} seconds")
            
            # Filter df to only include race_ids that are in our list
            filter_start_time = time.time()
            valid_df = df[df['race_id'].isin(race_ids)]
            filter_elapsed_time = time.time() - filter_start_time
            logger.info(f"Filtered DataFrame in {filter_elapsed_time:.2f} seconds, {len(valid_df)} rows remain")
            
            if len(valid_df) > 0:
                # Calculate indices for array assignment
                indices_start_time = time.time()
                row_indices = [race_id_to_index[rid] for rid in valid_df['race_id']]
                col_indices = valid_df['umaban'].values - 1  # Adjust for 0-indexing
                indices_elapsed_time = time.time() - indices_start_time
                logger.info(f"Calculated indices in {indices_elapsed_time:.2f} seconds")
                
                # Filter out invalid indices (where horse number exceeds array dimensions)
                mask_start_time = time.time()
                valid_mask = col_indices < horse_numbers.shape[1]
                row_indices = np.array(row_indices)[valid_mask]
                col_indices = col_indices[valid_mask]
                odds_values = valid_df['odds'].values[valid_mask]
                mask_elapsed_time = time.time() - mask_start_time
                logger.info(f"Applied valid mask in {mask_elapsed_time:.2f} seconds, {np.sum(valid_mask)} valid entries")
                
                # Assign odds to the array using advanced indexing
                assign_start_time = time.time()
                odds_array[row_indices, col_indices] = odds_values
                assign_elapsed_time = time.time() - assign_start_time
                logger.info(f"Assigned values to array in {assign_elapsed_time:.2f} seconds")
            
            logger.info(f"Loaded tansho odds for {len(valid_df)} horse-race combinations")
            
        except Exception as e:
            logger.error(f"Error loading tansho odds: {e}")
            logger.exception(e)
            return np.full_like(horse_numbers, np.nan, dtype=np.float32)
            
        total_elapsed_time = time.time() - start_time
        logger.info(f"Total tansho loading time: {total_elapsed_time:.2f} seconds")
        return odds_array

    def _load_fukusho(self, race_ids: List[str], horse_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load place odds (複勝) using pandas for optimization.
        """
        start_time = time.time()
        logger.info(f"Beginning fukusho query...")
        
        # Query to get the latest fukusho odds for each race
        query = """
        WITH latest_odds AS (
            SELECT 
                kaisai_date,
                keibajo_code,
                kaisai_kai,
                kaisai_nichime,
                kyoso_bango,
                umaban,
                min_odds,
                ROW_NUMBER() OVER(PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban 
                                  ORDER BY happyo_datetime DESC) as rn
            FROM 
                fukusho
            WHERE 
                kaisai_date BETWEEN %s AND %s
        )
        SELECT 
            kaisai_date,
            keibajo_code,
            kaisai_kai,
            kaisai_nichime,
            kyoso_bango,
            umaban,
            min_odds
        FROM 
            latest_odds
        WHERE 
            rn = 1
        """
        
        try:
            # Execute query and get results as DataFrame
            query_start_time = time.time()
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            query_elapsed_time = time.time() - query_start_time
            logger.info(f"Fukusho database query completed in {query_elapsed_time:.2f} seconds")
            
            if not results:
                logger.info("No fukusho odds found for the specified races")
                return np.full_like(horse_numbers, np.nan, dtype=np.float32)
            
            logger.info(f"Fukusho query returned {len(results)} rows")
                
            df_start_time = time.time()
            df = pd.DataFrame([dict(row) for row in results])
            df_elapsed_time = time.time() - df_start_time
            logger.info(f"Created pandas DataFrame in {df_elapsed_time:.2f} seconds")
            
            # Convert to numeric for calculations
            convert_start_time = time.time()
            df['umaban'] = df['umaban'].astype(int)
            df['min_odds'] = df['min_odds'].astype(float) / 10.0  # Convert to decimal odds
            convert_elapsed_time = time.time() - convert_start_time
            logger.info(f"Converted data types in {convert_elapsed_time:.2f} seconds")

            # Create a dictionary to map race_id to index in the array
            index_start_time = time.time()
            race_id_to_index = {race_id: idx for idx, race_id in enumerate(race_ids)}
            index_elapsed_time = time.time() - index_start_time
            logger.info(f"Created race_id to index mapping in {index_elapsed_time:.2f} seconds")
            
            # Convert kaisai_date to datetime if it's not already
            date_start_time = time.time()
            df['kaisai_date'] = pd.to_datetime(df['kaisai_date'])

            # Then create race_id as before
            df['race_id'] = df['kaisai_date'].dt.strftime('%Y%m%d') + df['keibajo_code'] + df['kaisai_kai'].astype(str).str.zfill(2) + df['kaisai_nichime'].astype(str).str.zfill(2) + df['kyoso_bango'].astype(str).str.zfill(2)
            date_elapsed_time = time.time() - date_start_time
            logger.info(f"Created race_id field in {date_elapsed_time:.2f} seconds")

            # Initialize odds_array with NaN values
            init_start_time = time.time()
            odds_array = np.full_like(horse_numbers, np.nan, dtype=np.float32)
            init_elapsed_time = time.time() - init_start_time
            logger.info(f"Initialized odds array in {init_elapsed_time:.2f} seconds")
            
            # Filter df to only include race_ids that are in our list
            filter_start_time = time.time()
            valid_df = df[df['race_id'].isin(race_ids)]
            filter_elapsed_time = time.time() - filter_start_time
            logger.info(f"Filtered DataFrame in {filter_elapsed_time:.2f} seconds, {len(valid_df)} rows remain")
            
            if len(valid_df) > 0:
                # Calculate indices for array assignment
                indices_start_time = time.time()
                row_indices = [race_id_to_index[rid] for rid in valid_df['race_id']]
                col_indices = valid_df['umaban'].values - 1  # Adjust for 0-indexing
                indices_elapsed_time = time.time() - indices_start_time
                logger.info(f"Calculated indices in {indices_elapsed_time:.2f} seconds")
                
                # Filter out invalid indices (where horse number exceeds array dimensions)
                mask_start_time = time.time()
                valid_mask = col_indices < horse_numbers.shape[1]
                row_indices = np.array(row_indices)[valid_mask]
                col_indices = col_indices[valid_mask]
                odds_values = valid_df['min_odds'].values[valid_mask]
                mask_elapsed_time = time.time() - mask_start_time
                logger.info(f"Applied valid mask in {mask_elapsed_time:.2f} seconds, {np.sum(valid_mask)} valid entries")
                
                # Assign odds to the array using advanced indexing
                assign_start_time = time.time()
                odds_array[row_indices, col_indices] = odds_values
                assign_elapsed_time = time.time() - assign_start_time
                logger.info(f"Assigned values to array in {assign_elapsed_time:.2f} seconds")
        
            logger.info(f"Loaded fukusho odds for {len(valid_df)} horse-race combinations")
            
        except Exception as e:
            logger.error(f"Error loading fukusho odds: {e}")
            logger.exception(e)
            return np.full_like(horse_numbers, np.nan, dtype=np.float32)
            
        total_elapsed_time = time.time() - start_time
        logger.info(f"Total fukusho loading time: {total_elapsed_time:.2f} seconds")
        return odds_array

    def _load_wakuren(self, race_ids: List[str], frame_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load bracket quinella odds (枠連).
        枠連 is an unordered combination of frame numbers, including same-frame combinations.
        """
        start_time = time.time()
        logger.info(f"Beginning wakuren query...")
        
        # Calculate number of possible combinations for wakuren (unordered pairs with replacement)
        total_combinations = (self.max_frame_number * (self.max_frame_number + 1)) // 2
        logger.info(f"Wakuren combinations: {total_combinations}")
        
        # Query to get the latest wakuren odds for each race
        query = """
        WITH latest_odds AS (
            SELECT 
                kaisai_date,
                keibajo_code,
                kaisai_kai, 
                kaisai_nichime,
                kyoso_bango,
                wakuban_1,
                wakuban_2,
                odds,
                ROW_NUMBER() OVER(PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, wakuban_1, wakuban_2
                              ORDER BY happyo_datetime DESC) as rn
            FROM 
                wakuren
            WHERE 
                kaisai_date BETWEEN %s AND %s
        )
        SELECT 
            kaisai_date,
            keibajo_code,
            kaisai_kai,
            kaisai_nichime,
            kyoso_bango,
            wakuban_1,
            wakuban_2,
            odds
        FROM 
            latest_odds
        WHERE 
            rn = 1
        """
        
        try:
            # Execute query and get results as DataFrame
            query_start_time = time.time()
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            query_elapsed_time = time.time() - query_start_time
            logger.info(f"Wakuren database query completed in {query_elapsed_time:.2f} seconds")
            
            if not results:
                logger.info("No wakuren odds found for the specified races")
                return np.full((len(race_ids), total_combinations), np.nan, dtype=np.float32)
            
            logger.info(f"Wakuren query returned {len(results)} rows")
                
            df_start_time = time.time()
            df = pd.DataFrame([dict(row) for row in results])
            df_elapsed_time = time.time() - df_start_time
            logger.info(f"Created pandas DataFrame in {df_elapsed_time:.2f} seconds")
            
            # Convert to numeric for calculations
            convert_start_time = time.time()
            df['wakuban_1'] = df['wakuban_1'].astype(int)
            df['wakuban_2'] = df['wakuban_2'].astype(int)
            df['odds'] = df['odds'].astype(float) / 10.0  # Convert to decimal odds
            convert_elapsed_time = time.time() - convert_start_time
            logger.info(f"Converted data types in {convert_elapsed_time:.2f} seconds")
            
            # Create a dictionary to map race_id to index in the array
            index_start_time = time.time()
            race_id_to_index = {race_id: idx for idx, race_id in enumerate(race_ids)}
            index_elapsed_time = time.time() - index_start_time
            logger.info(f"Created race_id to index mapping in {index_elapsed_time:.2f} seconds")
            
            # Convert kaisai_date to datetime if it's not already
            date_start_time = time.time()
            df['kaisai_date'] = pd.to_datetime(df['kaisai_date'])

            # Then create race_id
            df['race_id'] = df['kaisai_date'].dt.strftime('%Y%m%d') + df['keibajo_code'] + df['kaisai_kai'].astype(str).str.zfill(2) + df['kaisai_nichime'].astype(str).str.zfill(2) + df['kyoso_bango'].astype(str).str.zfill(2)
            date_elapsed_time = time.time() - date_start_time
            logger.info(f"Created race_id field in {date_elapsed_time:.2f} seconds")

            # Initialize odds_array with NaN values
            init_start_time = time.time()
            odds_array = np.full((len(race_ids), total_combinations), np.nan, dtype=np.float32)
            init_elapsed_time = time.time() - init_start_time
            logger.info(f"Initialized odds array in {init_elapsed_time:.2f} seconds")
            
            # Filter df to only include race_ids that are in our list
            filter_start_time = time.time()
            valid_df = df[df['race_id'].isin(race_ids)].copy()
            filter_elapsed_time = time.time() - filter_start_time
            logger.info(f"Filtered DataFrame in {filter_elapsed_time:.2f} seconds, {len(valid_df)} rows remain")
            
            if len(valid_df) > 0:
                # Create a vectorized function to compute the index in the flattened array for a wakuban pair
                def pair_to_index(a, b):
                    # Ensure a <= b for consistent indexing (wakuren includes same-frame combinations)
                    a, b = min(a, b), max(a, b)
                    # Formula for combination with replacement index
                    return ((a - 1) * (2 * self.max_frame_number - a)) // 2 + (b - a)
                
                # Vectorize the pair_to_index function
                index_function_start_time = time.time()
                vectorized_pair_to_index = np.vectorize(pair_to_index)
                
                # Ensure wakuban_1 <= wakuban_2 for consistent indexing
                valid_df['min_wakuban'] = valid_df[['wakuban_1', 'wakuban_2']].min(axis=1)
                valid_df['max_wakuban'] = valid_df[['wakuban_1', 'wakuban_2']].max(axis=1)
                index_function_elapsed_time = time.time() - index_function_start_time
                logger.info(f"Prepared index function in {index_function_elapsed_time:.2f} seconds")
                
                # Calculate col_indices
                indices_start_time = time.time()
                col_indices = vectorized_pair_to_index(valid_df['min_wakuban'].values, valid_df['max_wakuban'].values)
                
                # Calculate row indices
                row_indices = [race_id_to_index[rid] for rid in valid_df['race_id']]
                row_indices = np.array(row_indices)
                indices_elapsed_time = time.time() - indices_start_time
                logger.info(f"Calculated indices in {indices_elapsed_time:.2f} seconds")
                
                # Filter out invalid indices
                mask_start_time = time.time()
                valid_mask = (col_indices >= 0) & (col_indices < total_combinations)
                row_indices = row_indices[valid_mask]
                col_indices = col_indices[valid_mask]
                odds_values = valid_df['odds'].values[valid_mask]
                mask_elapsed_time = time.time() - mask_start_time
                logger.info(f"Applied valid mask in {mask_elapsed_time:.2f} seconds, {np.sum(valid_mask)} valid entries")
                
                # Assign odds to the array using advanced indexing
                assign_start_time = time.time()
                odds_array[row_indices, col_indices] = odds_values
                assign_elapsed_time = time.time() - assign_start_time
                logger.info(f"Assigned values to array in {assign_elapsed_time:.2f} seconds")
        
            logger.info(f"Loaded wakuren odds for {len(valid_df)} combinations across {len(valid_df['race_id'].unique())} races")
        
        except Exception as e:
            logger.error(f"Error loading wakuren odds: {e}")
            logger.exception(e)
            return np.full((len(race_ids), total_combinations), np.nan, dtype=np.float32)
        
        total_elapsed_time = time.time() - start_time
        logger.info(f"Total wakuren loading time: {total_elapsed_time:.2f} seconds")
        return odds_array

    def _load_umaren(self, race_ids: List[str], horse_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load quinella odds (馬連).
        馬連 is an unordered combination of two different horse numbers.
        """
        start_time = time.time()
        logger.info(f"Beginning umaren query...")
        
        # Calculate number of possible combinations for umaren (unordered pairs)
        max_horses = horse_numbers.shape[1]
        num_combinations = max_horses * (max_horses - 1) // 2
        logger.info(f"Umaren combinations: {num_combinations}")
        
        # Query to get the latest umaren odds for each race
        query = """
        WITH latest_odds AS (
            SELECT 
                kaisai_date,
                keibajo_code,
                kaisai_kai,
                kaisai_nichime,
                kyoso_bango,
                umaban_1,
                umaban_2,
                odds,
                ROW_NUMBER() OVER(PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban_1, umaban_2
                              ORDER BY happyo_datetime DESC) as rn
            FROM 
                umaren
            WHERE 
                kaisai_date BETWEEN %s AND %s
        )
        SELECT 
            kaisai_date,
            keibajo_code,
            kaisai_kai,
            kaisai_nichime,
            kyoso_bango,
            umaban_1,
            umaban_2,
            odds
        FROM 
            latest_odds
        WHERE 
            rn = 1
        """
        
        try:
            # Execute query and get results as DataFrame
            query_start_time = time.time()
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            query_elapsed_time = time.time() - query_start_time
            logger.info(f"Umaren database query completed in {query_elapsed_time:.2f} seconds")
            
            if not results:
                logger.info("No umaren odds found for the specified races")
                return np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
                
            logger.info(f"Umaren query returned {len(results)} rows")
                
            df_start_time = time.time()
            df = pd.DataFrame([dict(row) for row in results])
            df_elapsed_time = time.time() - df_start_time
            logger.info(f"Created pandas DataFrame in {df_elapsed_time:.2f} seconds")
            
            # Convert to numeric for calculations
            convert_start_time = time.time()
            df['umaban_1'] = df['umaban_1'].astype(int)
            df['umaban_2'] = df['umaban_2'].astype(int)
            df['odds'] = df['odds'].astype(float) / 10.0  # Convert to decimal odds
            convert_elapsed_time = time.time() - convert_start_time
            logger.info(f"Converted data types in {convert_elapsed_time:.2f} seconds")
            
            # Create a dictionary to map race_id to index in the array
            index_start_time = time.time()
            race_id_to_index = {race_id: idx for idx, race_id in enumerate(race_ids)}
            index_elapsed_time = time.time() - index_start_time
            logger.info(f"Created race_id to index mapping in {index_elapsed_time:.2f} seconds")
            
            # Convert kaisai_date to datetime if it's not already
            date_start_time = time.time()
            df['kaisai_date'] = pd.to_datetime(df['kaisai_date'])

            # Then create race_id as before
            df['race_id'] = df['kaisai_date'].dt.strftime('%Y%m%d') + df['keibajo_code'] + df['kaisai_kai'].astype(str).str.zfill(2) + df['kaisai_nichime'].astype(str).str.zfill(2) + df['kyoso_bango'].astype(str).str.zfill(2)
            date_elapsed_time = time.time() - date_start_time
            logger.info(f"Created race_id field in {date_elapsed_time:.2f} seconds")

            # Initialize odds_array with NaN values
            init_start_time = time.time()
            odds_array = np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
            init_elapsed_time = time.time() - init_start_time
            logger.info(f"Initialized odds array in {init_elapsed_time:.2f} seconds")
            
            # Filter df to only include race_ids that are in our list
            filter_start_time = time.time()
            valid_df = df[df['race_id'].isin(race_ids)].copy()
            filter_elapsed_time = time.time() - filter_start_time
            logger.info(f"Filtered DataFrame in {filter_elapsed_time:.2f} seconds, {len(valid_df)} rows remain")
            
            if len(valid_df) > 0:
                # Create a function to compute the index in the flattened array for a horse pair
                function_start_time = time.time()
                def pair_to_index(a, b):
                    # Ensure a < b for consistent indexing
                    a, b = min(a, b), max(a, b)
                    # Formula for indexing combinations: (n*(n-1)/2 - (n-a)*(n-a-1)/2 + b - a - 1)
                    return (max_horses * (max_horses - 1) // 2) - ((max_horses - a) * (max_horses - a - 1) // 2) + (b - a - 1)
                
                # Vectorize the pair_to_index function
                vectorized_pair_to_index = np.vectorize(pair_to_index)
                function_elapsed_time = time.time() - function_start_time
                logger.info(f"Created index function in {function_elapsed_time:.2f} seconds")
                
                # Ensure umaban_1 < umaban_2 for consistent indexing
                min_max_start_time = time.time()
                valid_df['min_umaban'] = valid_df[['umaban_1', 'umaban_2']].min(axis=1)
                valid_df['max_umaban'] = valid_df[['umaban_1', 'umaban_2']].max(axis=1)
                min_max_elapsed_time = time.time() - min_max_start_time
                logger.info(f"Calculated min/max horse numbers in {min_max_elapsed_time:.2f} seconds")
                
                # Calculate col_indices
                indices_start_time = time.time()
                col_indices = vectorized_pair_to_index(valid_df['min_umaban'].values - 1, valid_df['max_umaban'].values - 1)
                
                # Calculate row indices
                row_indices = [race_id_to_index[rid] for rid in valid_df['race_id']]
                row_indices = np.array(row_indices)
                indices_elapsed_time = time.time() - indices_start_time
                logger.info(f"Calculated indices in {indices_elapsed_time:.2f} seconds")
                
                # Filter out invalid indices (where horse number exceeds array dimensions)
                mask_start_time = time.time()
                valid_mask = (col_indices >= 0) & (col_indices < num_combinations)
                row_indices = np.array(row_indices)[valid_mask]
                col_indices = col_indices[valid_mask]
                odds_values = valid_df['odds'].values[valid_mask]
                mask_elapsed_time = time.time() - mask_start_time
                logger.info(f"Applied valid mask in {mask_elapsed_time:.2f} seconds, {np.sum(valid_mask)} valid entries")
                
                # Assign odds to the array using advanced indexing
                assign_start_time = time.time()
                odds_array[row_indices, col_indices] = odds_values
                assign_elapsed_time = time.time() - assign_start_time
                logger.info(f"Assigned values to array in {assign_elapsed_time:.2f} seconds")
        
            logger.info(f"Loaded umaren odds for {len(valid_df)} combinations across {len(valid_df['race_id'].unique())} races")
            
        except Exception as e:
            logger.error(f"Error loading umaren odds: {e}")
            logger.exception(e)
            return np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
            
        total_elapsed_time = time.time() - start_time
        logger.info(f"Total umaren loading time: {total_elapsed_time:.2f} seconds")
        return odds_array

    def _load_wide(self, race_ids: List[str], horse_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load quinella place odds (ワイド).
        ワイド is an unordered combination of two different horse numbers where both horses finish in top positions.
        """
        start_time = time.time()
        logger.info(f"Beginning wide query...")
        
        # Calculate number of possible combinations for wide (unordered pairs)
        max_horses = horse_numbers.shape[1]
        num_combinations = max_horses * (max_horses - 1) // 2
        logger.info(f"Wide combinations: {num_combinations}")
        
        # Query to get the latest wide odds for each race
        query = """
        WITH latest_odds AS (
            SELECT 
                kaisai_date,
                keibajo_code,
                kaisai_kai,
                kaisai_nichime,
                kyoso_bango,
                umaban_1,
                umaban_2,
                min_odds,
                ROW_NUMBER() OVER(PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban_1, umaban_2
                          ORDER BY happyo_datetime DESC) as rn
            FROM 
                wide
            WHERE 
                kaisai_date BETWEEN %s AND %s
        )
        SELECT 
            kaisai_date,
            keibajo_code,
            kaisai_kai,
            kaisai_nichime,
            kyoso_bango,
            umaban_1,
            umaban_2,
            min_odds
        FROM 
            latest_odds
        WHERE 
            rn = 1
        """
    
        try:
            # Execute query and get results as DataFrame
            query_start_time = time.time()
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            query_elapsed_time = time.time() - query_start_time
            logger.info(f"Wide database query completed in {query_elapsed_time:.2f} seconds")
            
            if not results:
                logger.info("No wide odds found for the specified races")
                return np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
            
            logger.info(f"Wide query returned {len(results)} rows")
            
            df_start_time = time.time()
            df = pd.DataFrame([dict(row) for row in results])
            df_elapsed_time = time.time() - df_start_time
            logger.info(f"Created pandas DataFrame in {df_elapsed_time:.2f} seconds")
            
            # Convert to numeric for calculations
            convert_start_time = time.time()
            df['umaban_1'] = df['umaban_1'].astype(int)
            df['umaban_2'] = df['umaban_2'].astype(int)
            df['min_odds'] = df['min_odds'].astype(float) / 10.0  # Convert to decimal odds
            convert_elapsed_time = time.time() - convert_start_time
            logger.info(f"Converted data types in {convert_elapsed_time:.2f} seconds")
            
            # Create a dictionary to map race_id to index in the array
            index_start_time = time.time()
            race_id_to_index = {race_id: idx for idx, race_id in enumerate(race_ids)}
            index_elapsed_time = time.time() - index_start_time
            logger.info(f"Created race_id to index mapping in {index_elapsed_time:.2f} seconds")
            
            # Convert kaisai_date to datetime if it's not already
            date_start_time = time.time()
            df['kaisai_date'] = pd.to_datetime(df['kaisai_date'])

            # Then create race_id as before
            df['race_id'] = df['kaisai_date'].dt.strftime('%Y%m%d') + df['keibajo_code'] + df['kaisai_kai'].astype(str).str.zfill(2) + df['kaisai_nichime'].astype(str).str.zfill(2) + df['kyoso_bango'].astype(str).str.zfill(2)
            date_elapsed_time = time.time() - date_start_time
            logger.info(f"Created race_id field in {date_elapsed_time:.2f} seconds")

            # Initialize odds_array with NaN values
            init_start_time = time.time()
            odds_array = np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
            init_elapsed_time = time.time() - init_start_time
            logger.info(f"Initialized odds array in {init_elapsed_time:.2f} seconds")
            
            # Filter df to only include race_ids that are in our list
            filter_start_time = time.time()
            valid_df = df[df['race_id'].isin(race_ids)].copy()
            filter_elapsed_time = time.time() - filter_start_time
            logger.info(f"Filtered DataFrame in {filter_elapsed_time:.2f} seconds, {len(valid_df)} rows remain")
            
            if len(valid_df) > 0:
                # Create a function to compute the index in the flattened array for a horse pair
                function_start_time = time.time()
                def pair_to_index(a, b):
                    # Ensure a < b for consistent indexing
                    a, b = min(a, b), max(a, b)
                    # Formula for indexing combinations: (n*(n-1)/2 - (n-a)*(n-a-1)/2 + b - a - 1)
                    return (max_horses * (max_horses - 1) // 2) - ((max_horses - a) * (max_horses - a - 1) // 2) + (b - a - 1)
                
                # Vectorize the pair_to_index function
                vectorized_pair_to_index = np.vectorize(pair_to_index)
                function_elapsed_time = time.time() - function_start_time
                logger.info(f"Created index function in {function_elapsed_time:.2f} seconds")
                
                # Ensure umaban_1 < umaban_2 for consistent indexing
                min_max_start_time = time.time()
                valid_df['min_umaban'] = valid_df[['umaban_1', 'umaban_2']].min(axis=1)
                valid_df['max_umaban'] = valid_df[['umaban_1', 'umaban_2']].max(axis=1)
                min_max_elapsed_time = time.time() - min_max_start_time
                logger.info(f"Calculated min/max horse numbers in {min_max_elapsed_time:.2f} seconds")
                
                # Calculate col_indices
                indices_start_time = time.time()
                col_indices = vectorized_pair_to_index(valid_df['min_umaban'].values - 1, valid_df['max_umaban'].values - 1)
                
                # Calculate row indices
                row_indices = [race_id_to_index[rid] for rid in valid_df['race_id']]
                row_indices = np.array(row_indices)
                indices_elapsed_time = time.time() - indices_start_time
                logger.info(f"Calculated indices in {indices_elapsed_time:.2f} seconds")
                
                # Filter out invalid indices (where horse number exceeds array dimensions)
                mask_start_time = time.time()
                valid_mask = (col_indices >= 0) & (col_indices < num_combinations)
                row_indices = np.array(row_indices)[valid_mask]
                col_indices = col_indices[valid_mask]
                odds_values = valid_df['min_odds'].values[valid_mask]
                mask_elapsed_time = time.time() - mask_start_time
                logger.info(f"Applied valid mask in {mask_elapsed_time:.2f} seconds, {np.sum(valid_mask)} valid entries")
                
                # Assign odds to the array using advanced indexing
                assign_start_time = time.time()
                odds_array[row_indices, col_indices] = odds_values
                assign_elapsed_time = time.time() - assign_start_time
                logger.info(f"Assigned values to array in {assign_elapsed_time:.2f} seconds")
    
            logger.info(f"Loaded wide odds for {len(valid_df)} combinations across {len(valid_df['race_id'].unique())} races")
    
        except Exception as e:
            logger.error(f"Error loading wide odds: {e}")
            logger.exception(e)
            return np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
    
        total_elapsed_time = time.time() - start_time
        logger.info(f"Total wide loading time: {total_elapsed_time:.2f} seconds")
        return odds_array

    def _load_umatan(self, race_ids: List[str], horse_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load exacta odds (馬単).
        馬単 is an ordered combination of two different horse numbers.
        """
        start_time = time.time()
        logger.info(f"Beginning umatan query...")
        
        # Calculate number of possible combinations for umatan (ordered pairs)
        max_horses = horse_numbers.shape[1]
        num_combinations = max_horses * (max_horses - 1)
        logger.info(f"Umatan combinations: {num_combinations}")
        
        # Query to get the latest umatan odds for each race
        query = """
        WITH latest_odds AS (
            SELECT 
                kaisai_date,
                keibajo_code,
                kaisai_kai,
                kaisai_nichime,
                kyoso_bango,
                umaban_1,
                umaban_2,
                odds,
                ROW_NUMBER() OVER(PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban_1, umaban_2
                          ORDER BY happyo_datetime DESC) as rn
            FROM 
                umatan
            WHERE 
                kaisai_date BETWEEN %s AND %s
        )
        SELECT 
            kaisai_date,
            keibajo_code,
            kaisai_kai,
            kaisai_nichime,
            kyoso_bango,
            umaban_1,
            umaban_2,
            odds
        FROM 
            latest_odds
        WHERE 
            rn = 1
        """

        try:
            # Execute query and get results as DataFrame
            query_start_time = time.time()
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            query_elapsed_time = time.time() - query_start_time
            logger.info(f"Umatan database query completed in {query_elapsed_time:.2f} seconds")
            
            if not results:
                logger.info("No umatan odds found for the specified races")
                return np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
                
            logger.info(f"Umatan query returned {len(results)} rows")
                
            df_start_time = time.time()
            df = pd.DataFrame([dict(row) for row in results])
            df_elapsed_time = time.time() - df_start_time
            logger.info(f"Created pandas DataFrame in {df_elapsed_time:.2f} seconds")
            
            # Convert to numeric for calculations
            convert_start_time = time.time()
            df['umaban_1'] = df['umaban_1'].astype(int)
            df['umaban_2'] = df['umaban_2'].astype(int)
            df['odds'] = df['odds'].astype(float) / 10.0  # Convert to decimal odds
            convert_elapsed_time = time.time() - convert_start_time
            logger.info(f"Converted data types in {convert_elapsed_time:.2f} seconds")
            
            # Create a dictionary to map race_id to index in the array
            index_start_time = time.time()
            race_id_to_index = {race_id: idx for idx, race_id in enumerate(race_ids)}
            index_elapsed_time = time.time() - index_start_time
            logger.info(f"Created race_id to index mapping in {index_elapsed_time:.2f} seconds")
            
            # Convert kaisai_date to datetime if it's not already
            date_start_time = time.time()
            df['kaisai_date'] = pd.to_datetime(df['kaisai_date'])

            # Then create race_id as before
            df['race_id'] = df['kaisai_date'].dt.strftime('%Y%m%d') + df['keibajo_code'] + df['kaisai_kai'].astype(str).str.zfill(2) + df['kaisai_nichime'].astype(str).str.zfill(2) + df['kyoso_bango'].astype(str).str.zfill(2)
            date_elapsed_time = time.time() - date_start_time
            logger.info(f"Created race_id field in {date_elapsed_time:.2f} seconds")

            # Initialize odds_array with NaN values
            init_start_time = time.time()
            odds_array = np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
            init_elapsed_time = time.time() - init_start_time
            logger.info(f"Initialized odds array in {init_elapsed_time:.2f} seconds")
            
            # Filter df to only include race_ids that are in our list
            filter_start_time = time.time()
            valid_df = df[df['race_id'].isin(race_ids)].copy()
            filter_elapsed_time = time.time() - filter_start_time
            logger.info(f"Filtered DataFrame in {filter_elapsed_time:.2f} seconds, {len(valid_df)} rows remain")
            
            if len(valid_df) > 0:
                # Create a function to compute the index in the flattened array for an ordered horse pair
                function_start_time = time.time()
                def ordered_pair_to_index(a, b):
                    # Convert to 0-indexed
                    a_idx = a - 1
                    b_idx = b - 1
                    # Formula for ordered pairs: a_idx * (max_horses - 1) + (b_idx - (b_idx > a_idx))
                    return a_idx * (max_horses - 1) + (b_idx - (b_idx > a_idx))
        
                # Vectorize the ordered_pair_to_index function
                vectorized_ordered_pair_to_index = np.vectorize(ordered_pair_to_index)
                function_elapsed_time = time.time() - function_start_time
                logger.info(f"Created index function in {function_elapsed_time:.2f} seconds")
                
                # Calculate col_indices for the ordered pairs
                indices_start_time = time.time()
                col_indices = vectorized_ordered_pair_to_index(valid_df['umaban_1'].values, valid_df['umaban_2'].values)
                
                # Calculate row indices
                row_indices = [race_id_to_index[rid] for rid in valid_df['race_id']]
                row_indices = np.array(row_indices)
                indices_elapsed_time = time.time() - indices_start_time
                logger.info(f"Calculated indices in {indices_elapsed_time:.2f} seconds")
                
                # Filter out invalid indices
                mask_start_time = time.time()
                valid_mask = (col_indices >= 0) & (col_indices < num_combinations)
                row_indices = row_indices[valid_mask]
                col_indices = col_indices[valid_mask]
                odds_values = valid_df['odds'].values[valid_mask]
                mask_elapsed_time = time.time() - mask_start_time
                logger.info(f"Applied valid mask in {mask_elapsed_time:.2f} seconds, {np.sum(valid_mask)} valid entries")
                
                # Assign odds to the array using advanced indexing
                assign_start_time = time.time()
                odds_array[row_indices, col_indices] = odds_values
                assign_elapsed_time = time.time() - assign_start_time
                logger.info(f"Assigned values to array in {assign_elapsed_time:.2f} seconds")
            
            logger.info(f"Loaded umatan odds for {len(valid_df)} combinations across {len(valid_df['race_id'].unique())} races")
            
        except Exception as e:
            logger.error(f"Error loading umatan odds: {e}")
            logger.exception(e)
            return np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
            
        total_elapsed_time = time.time() - start_time
        logger.info(f"Total umatan loading time: {total_elapsed_time:.2f} seconds")
        return odds_array

    def _load_sanrenpuku(self, race_ids: List[str], horse_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load trio odds (三連複).
        三連複 is an unordered combination of three different horse numbers.
        """
        start_time = time.time()
        logger.info(f"Beginning sanrenpuku query...")
        
        # Calculate number of possible combinations for sanrenpuku (unordered triplets)
        max_horses = horse_numbers.shape[1]
        num_combinations = max_horses * (max_horses - 1) * (max_horses - 2) // 6
        logger.info(f"Sanrenpuku combinations: {num_combinations}")
        
        # Query to get the latest sanrenpuku odds for each race
        query = """
        WITH latest_odds AS (
            SELECT 
                kaisai_date,
                keibajo_code,
                kaisai_kai,
                kaisai_nichime,
                kyoso_bango,
                umaban_1,
                umaban_2,
                umaban_3,
                odds,
                ROW_NUMBER() OVER(PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban_1, umaban_2, umaban_3
                          ORDER BY happyo_datetime DESC) as rn
            FROM 
                sanrenpuku
            WHERE 
                kaisai_date BETWEEN %s AND %s
        )
        SELECT 
            kaisai_date,
            keibajo_code,
            kaisai_kai,
            kaisai_nichime,
            kyoso_bango,
            umaban_1,
            umaban_2,
            umaban_3,
            odds
        FROM 
            latest_odds
        WHERE 
            rn = 1
        """

        try:
            # Execute query and get results as DataFrame
            query_start_time = time.time()
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            query_elapsed_time = time.time() - query_start_time
            logger.info(f"Sanrenpuku database query completed in {query_elapsed_time:.2f} seconds")
            
            if not results:
                logger.info("No sanrenpuku odds found for the specified races")
                return np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
            
            logger.info(f"Sanrenpuku query returned {len(results)} rows")
            
            df_start_time = time.time()
            df = pd.DataFrame([dict(row) for row in results])
            df_elapsed_time = time.time() - df_start_time
            logger.info(f"Created pandas DataFrame in {df_elapsed_time:.2f} seconds")
            
            # Convert to numeric for calculations
            convert_start_time = time.time()
            df['umaban_1'] = df['umaban_1'].astype(int)
            df['umaban_2'] = df['umaban_2'].astype(int)
            df['umaban_3'] = df['umaban_3'].astype(int)
            df['odds'] = df['odds'].astype(float) / 10.0  # Convert to decimal odds
            convert_elapsed_time = time.time() - convert_start_time
            logger.info(f"Converted data types in {convert_elapsed_time:.2f} seconds")
            
            # Create a dictionary to map race_id to index in the array
            index_start_time = time.time()
            race_id_to_index = {race_id: idx for idx, race_id in enumerate(race_ids)}
            index_elapsed_time = time.time() - index_start_time
            logger.info(f"Created race_id to index mapping in {index_elapsed_time:.2f} seconds")
            
            # Convert kaisai_date to datetime if it's not already
            date_start_time = time.time()
            df['kaisai_date'] = pd.to_datetime(df['kaisai_date'])

            # Then create race_id as before
            df['race_id'] = df['kaisai_date'].dt.strftime('%Y%m%d') + df['keibajo_code'] + df['kaisai_kai'].astype(str).str.zfill(2) + df['kaisai_nichime'].astype(str).str.zfill(2) + df['kyoso_bango'].astype(str).str.zfill(2)
            date_elapsed_time = time.time() - date_start_time
            logger.info(f"Created race_id field in {date_elapsed_time:.2f} seconds")

            # Initialize odds_array with NaN values
            init_start_time = time.time()
            odds_array = np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
            init_elapsed_time = time.time() - init_start_time
            logger.info(f"Initialized odds array in {init_elapsed_time:.2f} seconds")
            
            # Filter df to only include race_ids that are in our list
            filter_start_time = time.time()
            valid_df = df[df['race_id'].isin(race_ids)].copy()
            filter_elapsed_time = time.time() - filter_start_time
            logger.info(f"Filtered DataFrame in {filter_elapsed_time:.2f} seconds, {len(valid_df)} rows remain")
            
            if len(valid_df) > 0:
                # Sort the horse numbers for each combination to ensure consistent indexing
                sort_start_time = time.time()
                sorted_values = np.sort(valid_df[['umaban_1', 'umaban_2', 'umaban_3']].values, axis=1)
                valid_df['umaban_min'] = sorted_values[:, 0]
                valid_df['umaban_mid'] = sorted_values[:, 1]
                valid_df['umaban_max'] = sorted_values[:, 2]
                sort_elapsed_time = time.time() - sort_start_time
                logger.info(f"Sorted horse numbers in {sort_elapsed_time:.2f} seconds")
                
                # Formula for indexing combinations of 3 elements from n elements (vectorized)
                indices_start_time = time.time()
                n = max_horses
                a = valid_df['umaban_min'].values - 1  # Convert to 0-indexed
                b = valid_df['umaban_mid'].values - 1
                c = valid_df['umaban_max'].values - 1
                
                col_indices = ((n*(n-1)*(n-2)//6) - 
                            ((n-a)*(n-a-1)*(n-a-2)//6) + 
                            ((n-a-1)*(n-a-2)//2) - 
                            ((n-b)*(n-b-1)//2) + 
                            (c-b-1))
                
                # Calculate row indices
                row_indices = np.array([race_id_to_index[rid] for rid in valid_df['race_id']])
                indices_elapsed_time = time.time() - indices_start_time
                logger.info(f"Calculated indices in {indices_elapsed_time:.2f} seconds")
                
                # Filter out invalid indices
                mask_start_time = time.time()
                valid_mask = (col_indices >= 0) & (col_indices < num_combinations)
                row_indices = row_indices[valid_mask]
                col_indices = col_indices[valid_mask]
                odds_values = valid_df['odds'].values[valid_mask]
                mask_elapsed_time = time.time() - mask_start_time
                logger.info(f"Applied valid mask in {mask_elapsed_time:.2f} seconds, {np.sum(valid_mask)} valid entries")
                
                # Assign odds to the array using advanced indexing
                assign_start_time = time.time()
                odds_array[row_indices, col_indices] = odds_values
                assign_elapsed_time = time.time() - assign_start_time
                logger.info(f"Assigned values to array in {assign_elapsed_time:.2f} seconds")

            logger.info(f"Loaded sanrenpuku odds for {len(valid_df)} combinations across {len(valid_df['race_id'].unique())} races")

        except Exception as e:
            logger.error(f"Error loading sanrenpuku odds: {e}")
            logger.exception(e)
            return np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
        
        total_elapsed_time = time.time() - start_time
        logger.info(f"Total sanrenpuku loading time: {total_elapsed_time:.2f} seconds")
        return odds_array

    def _load_sanrentan(self, race_ids: List[str], horse_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load trifecta odds (三連単).
        三連単 is an ordered combination of three different horse numbers.
        """
        start_time = time.time()
        logger.info(f"Beginning sanrentan query...")
        
        # Calculate number of possible combinations for sanrentan (ordered triplets)
        max_horses = horse_numbers.shape[1]
        num_combinations = max_horses * (max_horses - 1) * (max_horses - 2)
        logger.info(f"Sanrentan combinations: {num_combinations}")
        
        # Query to get the latest sanrentan odds for each race
        query = """
        WITH latest_odds AS (
            SELECT 
                kaisai_date,
                keibajo_code,
                kaisai_kai,
                kaisai_nichime,
                kyoso_bango,
                umaban_1,
                umaban_2,
                umaban_3,
                odds,
                ROW_NUMBER() OVER(PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban_1, umaban_2, umaban_3
                          ORDER BY happyo_datetime DESC) as rn
            FROM 
                sanrentan
            WHERE 
                kaisai_date BETWEEN %s AND %s
        )
        SELECT 
            kaisai_date,
            keibajo_code,
            kaisai_kai,
            kaisai_nichime,
            kyoso_bango,
            umaban_1,
            umaban_2,
            umaban_3,
            odds
        FROM 
            latest_odds
        WHERE 
            rn = 1
        """

        try:
            # Execute query and get results as DataFrame
            query_start_time = time.time()
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            query_elapsed_time = time.time() - query_start_time
            logger.info(f"Sanrentan database query completed in {query_elapsed_time:.2f} seconds")
            
            if not results:
                logger.info("No sanrentan odds found for the specified races")
                return np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
            
            logger.info(f"Sanrentan query returned {len(results)} rows")
            
            df_start_time = time.time()
            df = pd.DataFrame([dict(row) for row in results])
            df_elapsed_time = time.time() - df_start_time
            logger.info(f"Created pandas DataFrame in {df_elapsed_time:.2f} seconds")
            
            # Convert to numeric for calculations
            convert_start_time = time.time()
            df['umaban_1'] = df['umaban_1'].astype(int)
            df['umaban_2'] = df['umaban_2'].astype(int)
            df['umaban_3'] = df['umaban_3'].astype(int)
            df['odds'] = df['odds'].astype(float) / 10.0  # Convert to decimal odds
            convert_elapsed_time = time.time() - convert_start_time
            logger.info(f"Converted data types in {convert_elapsed_time:.2f} seconds")
            
            # Create a dictionary to map race_id to index in the array
            index_start_time = time.time()
            race_id_to_index = {race_id: idx for idx, race_id in enumerate(race_ids)}
            index_elapsed_time = time.time() - index_start_time
            logger.info(f"Created race_id to index mapping in {index_elapsed_time:.2f} seconds")
            
            # Convert kaisai_date to datetime if it's not already
            date_start_time = time.time()
            df['kaisai_date'] = pd.to_datetime(df['kaisai_date'])

            # Then create race_id as before
            df['race_id'] = df['kaisai_date'].dt.strftime('%Y%m%d') + df['keibajo_code'] + df['kaisai_kai'].astype(str).str.zfill(2) + df['kaisai_nichime'].astype(str).str.zfill(2) + df['kyoso_bango'].astype(str).str.zfill(2)
            date_elapsed_time = time.time() - date_start_time
            logger.info(f"Created race_id field in {date_elapsed_time:.2f} seconds")

            # Initialize odds_array with NaN values
            init_start_time = time.time()
            odds_array = np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
            init_elapsed_time = time.time() - init_start_time
            logger.info(f"Initialized odds array in {init_elapsed_time:.2f} seconds")
            
            # Filter df to only include race_ids that are in our list
            filter_start_time = time.time()
            valid_df = df[df['race_id'].isin(race_ids)].copy()
            filter_elapsed_time = time.time() - filter_start_time
            logger.info(f"Filtered DataFrame in {filter_elapsed_time:.2f} seconds, {len(valid_df)} rows remain")
            
            if len(valid_df) > 0:
                # Get the horse numbers as 0-indexed values
                indices_start_time = time.time()
                a_idx = valid_df['umaban_1'].values - 1
                b_idx = valid_df['umaban_2'].values - 1
                c_idx = valid_df['umaban_3'].values - 1
                
                # Create adjustment values for the second index
                b_adj = (b_idx > a_idx).astype(int)
                
                # Create adjustment values for the third index
                c_adj1 = (c_idx > a_idx).astype(int)
                c_adj2 = (c_idx > b_idx).astype(int)
                
                # Compute adjusted indices
                adjusted_b_idx = b_idx - b_adj
                adjusted_c_idx = c_idx - c_adj1 - c_adj2
                
                # Calculate column indices using the formula for ordered triplets
                col_indices = (a_idx * (max_horses - 1) * (max_horses - 2) + 
                              adjusted_b_idx * (max_horses - 2) + 
                              adjusted_c_idx)
                
                # Calculate row indices
                row_indices = np.array([race_id_to_index[rid] for rid in valid_df['race_id']])
                indices_elapsed_time = time.time() - indices_start_time
                logger.info(f"Calculated indices in {indices_elapsed_time:.2f} seconds")
                
                # Filter out invalid indices
                mask_start_time = time.time()
                valid_mask = (col_indices >= 0) & (col_indices < num_combinations)
                row_indices = row_indices[valid_mask]
                col_indices = col_indices[valid_mask]
                odds_values = valid_df['odds'].values[valid_mask]
                mask_elapsed_time = time.time() - mask_start_time
                logger.info(f"Applied valid mask in {mask_elapsed_time:.2f} seconds, {np.sum(valid_mask)} valid entries")
                
                # Assign odds to the array using advanced indexing
                assign_start_time = time.time()
                odds_array[row_indices, col_indices] = odds_values
                assign_elapsed_time = time.time() - assign_start_time
                logger.info(f"Assigned values to array in {assign_elapsed_time:.2f} seconds")

            logger.info(f"Loaded sanrentan odds for {len(valid_df)} combinations across {len(valid_df['race_id'].unique())} races")

        except Exception as e:
            logger.error(f"Error loading sanrentan odds: {e}")
            logger.exception(e)
            return np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
        
        total_elapsed_time = time.time() - start_time
        logger.info(f"Total sanrentan loading time: {total_elapsed_time:.2f} seconds")
        return odds_array
