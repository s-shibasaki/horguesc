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

        self.race_id_query = """
            CONCAT(
                TO_CHAR(kaisai_date, 'YYYYMMDD'),
                keibajo_code,
                LPAD(kaisai_kai::text, 2, '0'),
                LPAD(kaisai_nichime::text, 2, '0'),
                LPAD(kyoso_bango::text, 2, '0')
            ) as race_id,
        """
    
    def load_odds(self, 
                  race_ids: List[str], 
                  odds_types: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Load odds data for specified race IDs and odds types.
        
        Args:
            race_ids: List of race IDs
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
                odds_data[odds_type] = self._load_tansho(race_ids, min_date_str, max_date_str)
            elif odds_type == self.FUKUSHO:
                odds_data[odds_type] = self._load_fukusho(race_ids, min_date_str, max_date_str)
            elif odds_type == self.WAKUREN:
                odds_data[odds_type] = self._load_wakuren(race_ids, min_date_str, max_date_str)
            elif odds_type == self.UMAREN:
                odds_data[odds_type] = self._load_umaren(race_ids, min_date_str, max_date_str)
            elif odds_type == self.WIDE:
                odds_data[odds_type] = self._load_wide(race_ids, min_date_str, max_date_str)
            elif odds_type == self.UMATAN:
                odds_data[odds_type] = self._load_umatan(race_ids, min_date_str, max_date_str)
            elif odds_type == self.SANRENPUKU:
                odds_data[odds_type] = self._load_sanrenpuku(race_ids, min_date_str, max_date_str)
            elif odds_type == self.SANRENTAN:
                odds_data[odds_type] = self._load_sanrentan(race_ids, min_date_str, max_date_str)
                
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
    

    def _load_tansho(self, race_ids: List[str], min_date_str: str, max_date_str: str) -> np.ndarray:
        """
        Load tansho (win) odds for specified races.
        
        Args:
            race_ids: List of race IDs
            min_date_str: Minimum date in YYYY-MM-DD format
            max_date_str: Maximum date in YYYY-MM-DD format
            
        Returns:
            Numpy array of shape (num_races, max_horse_number) with odds values
        """
        # Create a mapping of race_id -> index for quick lookups
        race_id_to_index = {race_id: i for i, race_id in enumerate(race_ids)}
        
        # Initialize the result array with NaN
        result = np.full((len(race_ids), self.max_horse_number), np.nan, dtype=np.float32)

        # Build the query to get the latest odds for each race
        query = f"""
            WITH latest_odds AS (
                SELECT
                    {self.race_id_query}
                    umaban,
                    odds,
                    ROW_NUMBER() OVER (
                        PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban
                        ORDER BY happyo_datetime DESC
                    ) as row_num
                FROM
                    tansho
                WHERE
                    kaisai_date BETWEEN %s AND %s
            )
            SELECT
                race_id, umaban, odds
            FROM
                latest_odds
            WHERE
                row_num = 1
        """
        
        # Execute the query
        results = self.db_ops.execute_query(query, (min_date_str, max_date_str), fetch_all=True)
        
        # Process the results
        for row in results:
            race_id = row[0]
            # Skip if race_id is not in our target list
            if race_id in race_id_to_index:
                idx = race_id_to_index[race_id]
                umaban = row[1] - 1  # Convert 1-indexed to 0-indexed
                odds = row[2]
                
                # Skip invalid horse numbers
                if 0 <= umaban < self.max_horse_number and odds is not None:
                    # Odds are stored as integers with 10x multiplier, convert to float
                    result[idx, umaban] = odds / 10.0
        
        return result

    def _load_fukusho(self, race_ids: List[str], min_date_str: str, max_date_str: str) -> np.ndarray:
        """
        Load fukusho (place) odds for specified races.
        
        Args:
            race_ids: List of race IDs
            min_date_str: Minimum date in YYYY-MM-DD format
            max_date_str: Maximum date in YYYY-MM-DD format
            
        Returns:
            Numpy array of shape (num_races, max_horse_number) with odds values
        """
        # Create a mapping of race_id -> index for quick lookups
        race_id_to_index = {race_id: i for i, race_id in enumerate(race_ids)}
        
        # Initialize the result array with NaN
        result = np.full((len(race_ids), self.max_horse_number), np.nan, dtype=np.float32)

        # Build the query to get the latest odds for each race
        query = f"""
            WITH latest_odds AS (
                SELECT
                    {self.race_id_query}
                    umaban,
                    min_odds,
                    ROW_NUMBER() OVER (
                        PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban
                        ORDER BY happyo_datetime DESC
                    ) as row_num
                FROM
                    fukusho
                WHERE
                    kaisai_date BETWEEN %s AND %s
            )
            SELECT
                race_id, umaban, min_odds
            FROM
                latest_odds
            WHERE
                row_num = 1
        """
        
        # Execute the query
        results = self.db_ops.execute_query(query, (min_date_str, max_date_str), fetch_all=True)
        
        # Process the results
        for row in results:
            race_id = row[0]
            # Skip if race_id is not in our target list
            if race_id in race_id_to_index:
                idx = race_id_to_index[race_id]
                umaban = row[1] - 1  # Convert 1-indexed to 0-indexed
                min_odds = row[2]
                
                # Skip invalid horse numbers
                if 0 <= umaban < self.max_horse_number and min_odds is not None:
                    # Odds are stored as integers with 10x multiplier, convert to float
                    result[idx, umaban] = min_odds / 10.0
        
        return result

    def _load_wakuren(self, race_ids: List[str], min_date_str: str, max_date_str: str) -> np.ndarray:
        """
        Load wakuren (bracket quinella) odds for specified races.
        
        Args:
            race_ids: List of race IDs
            min_date_str: Minimum date in YYYY-MM-DD format
            max_date_str: Maximum date in YYYY-MM-DD format
            
        Returns:
            Numpy array of shape (num_races, combinations) with odds values
        """
        # Create a mapping of race_id -> index for quick lookups
        race_id_to_index = {race_id: i for i, race_id in enumerate(race_ids)}
        
        # Calculate number of combinations: 8x(8+1)/2 = 36 for bracket numbers 1-8
        num_combinations = self.max_frame_number * (self.max_frame_number + 1) // 2
        
        # Initialize the result array with NaN
        result = np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
        
        # Build the query to get the latest odds for each race
        query = f"""
            WITH latest_odds AS (
                SELECT
                    {self.race_id_query}
                    wakuban_1,
                    wakuban_2,
                    odds,
                    ROW_NUMBER() OVER (
                        PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, wakuban_1, wakuban_2
                        ORDER BY happyo_datetime DESC
                    ) as row_num
                FROM
                    wakuren
                WHERE
                    kaisai_date BETWEEN %s AND %s
            )
            SELECT
                race_id, wakuban_1, wakuban_2, odds
            FROM
                latest_odds
            WHERE
                row_num = 1
        """
        
        # Execute the query
        results = self.db_ops.execute_query(query, (min_date_str, max_date_str), fetch_all=True)
        
        # Create a dictionary to map bracket combinations to array indices
        wakuren_index = {}
        idx = 0
        for waku1 in range(1, self.max_frame_number + 1):
            for waku2 in range(waku1, self.max_frame_number + 1):
                wakuren_index[(waku1, waku2)] = idx
                if waku1 != waku2:
                    wakuren_index[(waku2, waku1)] = idx
                idx += 1
        
        # Process the results
        for row in results:
            race_id = row[0]
            # Skip if race_id is not in our target list
            if race_id in race_id_to_index:
                idx = race_id_to_index[race_id]
                wakuban_1 = row[1]
                wakuban_2 = row[2]
                odds = row[3]
                
                # Skip invalid bracket numbers
                if (1 <= wakuban_1 <= self.max_frame_number and 
                    1 <= wakuban_2 <= self.max_frame_number and 
                    odds is not None):
                    # Get index for this combination
                    combo_idx = wakuren_index[(wakuban_1, wakuban_2)]
                    # Odds are stored as integers with 10x multiplier, convert to float
                    result[idx, combo_idx] = odds / 10.0
        
        return result

    def _load_umaren(self, race_ids: List[str], min_date_str: str, max_date_str: str) -> np.ndarray:
        """
        Load umaren (quinella) odds for specified races.
        
        Args:
            race_ids: List of race IDs
            min_date_str: Minimum date in YYYY-MM-DD format
            max_date_str: Maximum date in YYYY-MM-DD format
            
        Returns:
            Numpy array of shape (num_races, combinations) with odds values
        """
        # Create a mapping of race_id -> index for quick lookups
        race_id_to_index = {race_id: i for i, race_id in enumerate(race_ids)}
        
        # Calculate number of combinations: n(n-1)/2 for n horses
        num_combinations = self.max_horse_number * (self.max_horse_number - 1) // 2
        
        # Initialize the result array with NaN
        result = np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
        
        # Build the query to get the latest odds for each race
        query = f"""
            WITH latest_odds AS (
                SELECT
                    {self.race_id_query}
                    umaban_1,
                    umaban_2,
                    odds,
                    ROW_NUMBER() OVER (
                        PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban_1, umaban_2
                        ORDER BY happyo_datetime DESC
                    ) as row_num
                FROM
                    umaren
                WHERE
                    kaisai_date BETWEEN %s AND %s
            )
            SELECT
                race_id, umaban_1, umaban_2, odds
            FROM
                latest_odds
            WHERE
                row_num = 1
        """
        
        # Execute the query
        results = self.db_ops.execute_query(query, (min_date_str, max_date_str), fetch_all=True)
        
        # Create a dictionary to map horse number combinations to array indices
        umaren_index = {}
        idx = 0
        for uma1 in range(1, self.max_horse_number + 1):
            for uma2 in range(uma1 + 1, self.max_horse_number + 1):
                umaren_index[(uma1, uma2)] = idx
                umaren_index[(uma2, uma1)] = idx
                idx += 1
        
        # Process the results
        for row in results:
            race_id = row[0]
            # Skip if race_id is not in our target list
            if race_id in race_id_to_index:
                idx = race_id_to_index[race_id]
                umaban_1 = row[1]
                umaban_2 = row[2]
                odds = row[3]
                
                # Skip invalid horse numbers or duplicate combinations
                if (1 <= umaban_1 <= self.max_horse_number and 
                    1 <= umaban_2 <= self.max_horse_number and 
                    umaban_1 != umaban_2 and 
                    odds is not None):
                    # Get index for this combination
                    combo_idx = umaren_index[(umaban_1, umaban_2)]
                    # Odds are stored as integers with 10x multiplier, convert to float
                    result[idx, combo_idx] = odds / 10.0
        
        return result

    def _load_wide(self, race_ids: List[str], min_date_str: str, max_date_str: str) -> np.ndarray:
        """
        Load wide (quinella place) odds for specified races.
        
        Args:
            race_ids: List of race IDs
            min_date_str: Minimum date in YYYY-MM-DD format
            max_date_str: Maximum date in YYYY-MM-DD format
            
        Returns:
            Numpy array of shape (num_races, combinations) with odds values
        """
        # Create a mapping of race_id -> index for quick lookups
        race_id_to_index = {race_id: i for i, race_id in enumerate(race_ids)}
        
        # Calculate number of combinations: n(n-1)/2 for n horses
        num_combinations = self.max_horse_number * (self.max_horse_number - 1) // 2
        
        # Initialize the result array with NaN
        result = np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
        
        # Build the query to get the latest odds for each race
        query = f"""
            WITH latest_odds AS (
                SELECT
                    {self.race_id_query}
                    umaban_1,
                    umaban_2,
                    min_odds,
                    ROW_NUMBER() OVER (
                        PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban_1, umaban_2
                        ORDER BY happyo_datetime DESC
                    ) as row_num
                FROM
                    wide
                WHERE
                    kaisai_date BETWEEN %s AND %s
            )
            SELECT
                race_id, umaban_1, umaban_2, min_odds
            FROM
                latest_odds
            WHERE
                row_num = 1
        """
        
        # Execute the query
        results = self.db_ops.execute_query(query, (min_date_str, max_date_str), fetch_all=True)
        
        # Create a dictionary to map horse number combinations to array indices
        wide_index = {}
        idx = 0
        for uma1 in range(1, self.max_horse_number + 1):
            for uma2 in range(uma1 + 1, self.max_horse_number + 1):
                wide_index[(uma1, uma2)] = idx
                wide_index[(uma2, uma1)] = idx
                idx += 1
        
        # Process the results
        for row in results:
            race_id = row[0]
            # Skip if race_id is not in our target list
            if race_id in race_id_to_index:
                idx = race_id_to_index[race_id]
                umaban_1 = row[1]
                umaban_2 = row[2]
                min_odds = row[3]  # Using min_odds as specified
                
                # Skip invalid horse numbers or duplicate combinations
                if (1 <= umaban_1 <= self.max_horse_number and 
                    1 <= umaban_2 <= self.max_horse_number and 
                    umaban_1 != umaban_2 and 
                    min_odds is not None):
                    # Get index for this combination
                    combo_idx = wide_index[(umaban_1, umaban_2)]
                    # Odds are stored as integers with 10x multiplier, convert to float
                    result[idx, combo_idx] = min_odds / 10.0
        
        return result

    def _load_umatan(self, race_ids: List[str], min_date_str: str, max_date_str: str) -> np.ndarray:
        """
        Load umatan (exacta) odds for specified races.
        
        Args:
            race_ids: List of race IDs
            min_date_str: Minimum date in YYYY-MM-DD format
            max_date_str: Maximum date in YYYY-MM-DD format
            
        Returns:
            Numpy array of shape (num_races, combinations) with odds values
        """
        # Create a mapping of race_id -> index for quick lookups
        race_id_to_index = {race_id: i for i, race_id in enumerate(race_ids)}
        
        # Calculate number of combinations: n(n-1) for n horses (order matters)
        num_combinations = self.max_horse_number * (self.max_horse_number - 1)
        
        # Initialize the result array with NaN
        result = np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
        
        # Build the query to get the latest odds for each race
        query = f"""
            WITH latest_odds AS (
                SELECT
                    {self.race_id_query}
                    umaban_1,
                    umaban_2,
                    odds,
                    ROW_NUMBER() OVER (
                        PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban_1, umaban_2
                        ORDER BY happyo_datetime DESC
                    ) as row_num
                FROM
                    umatan
                WHERE
                    kaisai_date BETWEEN %s AND %s
            )
            SELECT
                race_id, umaban_1, umaban_2, odds
            FROM
                latest_odds
            WHERE
                row_num = 1
        """
        
        # Execute the query
        results = self.db_ops.execute_query(query, (min_date_str, max_date_str), fetch_all=True)
        
        # Create a dictionary to map horse number combinations to array indices
        umatan_index = {}
        idx = 0
        for uma1 in range(1, self.max_horse_number + 1):
            for uma2 in range(1, self.max_horse_number + 1):
                if (uma1 != uma2):
                    umatan_index[(uma1, uma2)] = idx
                    idx += 1
        
        # Process the results
        for row in results:
            race_id = row[0]
            # Skip if race_id is not in our target list
            if race_id in race_id_to_index:
                idx = race_id_to_index[race_id]
                umaban_1 = row[1]
                umaban_2 = row[2]
                odds = row[3]
                
                # Skip invalid horse numbers or same horse numbers
                if (1 <= umaban_1 <= self.max_horse_number and 
                    1 <= umaban_2 <= self.max_horse_number and 
                    umaban_1 != umaban_2 and 
                    odds is not None):
                    # Get index for this combination
                    combo_idx = umatan_index[(umaban_1, umaban_2)]
                    # Odds are stored as integers with 10x multiplier, convert to float
                    result[idx, combo_idx] = odds / 10.0
        
        return result

    def _load_sanrenpuku(self, race_ids: List[str], min_date_str: str, max_date_str: str) -> np.ndarray:
        """
        Load sanrenpuku (trio) odds for specified races.
        
        Args:
            race_ids: List of race IDs
            min_date_str: Minimum date in YYYY-MM-DD format
            max_date_str: Maximum date in YYYY-MM-DD format
            
        Returns:
            Numpy array of shape (num_races, combinations) with odds values
        """
        # Create a mapping of race_id -> index for quick lookups
        race_id_to_index = {race_id: i for i, race_id in enumerate(race_ids)}
        
        # Calculate number of combinations: n choose 3 = n(n-1)(n-2)/6 for n horses
        num_combinations = self.max_horse_number * (self.max_horse_number - 1) * (self.max_horse_number - 2) // 6
        
        # Initialize the result array with NaN
        result = np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
        
        # Build the query to get the latest odds for each race
        query = f"""
            WITH latest_odds AS (
                SELECT
                    {self.race_id_query}
                    umaban_1,
                    umaban_2,
                    umaban_3,
                    odds,
                    ROW_NUMBER() OVER (
                        PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban_1, umaban_2, umaban_3
                        ORDER BY happyo_datetime DESC
                    ) as row_num
                FROM
                    sanrenpuku
                WHERE
                    kaisai_date BETWEEN %s AND %s
            )
            SELECT
                race_id, umaban_1, umaban_2, umaban_3, odds
            FROM
                latest_odds
            WHERE
                row_num = 1
        """
        
        # Execute the query
        results = self.db_ops.execute_query(query, (min_date_str, max_date_str), fetch_all=True)
        
        # Create a dictionary to map horse number combinations to array indices
        sanrenpuku_index = {}
        idx = 0
        for uma1 in range(1, self.max_horse_number + 1):
            for uma2 in range(uma1 + 1, self.max_horse_number + 1):
                for uma3 in range(uma2 + 1, self.max_horse_number + 1):
                    sanrenpuku_index[(uma1, uma2, uma3)] = idx
                    sanrenpuku_index[(uma1, uma3, uma2)] = idx
                    sanrenpuku_index[(uma2, uma1, uma3)] = idx
                    sanrenpuku_index[(uma2, uma3, uma1)] = idx
                    sanrenpuku_index[(uma3, uma1, uma2)] = idx
                    sanrenpuku_index[(uma3, uma2, uma1)] = idx
                    idx += 1
        
        # Process the results
        for row in results:
            race_id = row[0]
            # Skip if race_id is not in our target list
            if race_id in race_id_to_index:
                idx = race_id_to_index[race_id]
                umaban_1 = row[1]
                umaban_2 = row[2]
                umaban_3 = row[3]
                odds = row[4]
                
                # Skip invalid horse numbers or repeating numbers
                if (1 <= umaban_1 <= self.max_horse_number and 
                    1 <= umaban_2 <= self.max_horse_number and 
                    1 <= umaban_3 <= self.max_horse_number and 
                    len(set([umaban_1, umaban_2, umaban_3])) == 3 and 
                    odds is not None):
                    # Get index for this combination
                    combo_idx = sanrenpuku_index[(umaban_1, umaban_2, umaban_3)]
                    # Odds are stored as integers with 10x multiplier, convert to float
                    result[idx, combo_idx] = odds / 10.0
        
        return result

    def _load_sanrentan(self, race_ids: List[str], min_date_str: str, max_date_str: str) -> np.ndarray:
        """
        Load sanrentan (trifecta) odds for specified races.
        
        Args:
            race_ids: List of race IDs
            min_date_str: Minimum date in YYYY-MM-DD format
            max_date_str: Maximum date in YYYY-MM-DD format
            
        Returns:
            Numpy array of shape (num_races, combinations) with odds values
        """
        # Create a mapping of race_id -> index for quick lookups
        race_id_to_index = {race_id: i for i, race_id in enumerate(race_ids)}
        
        # Calculate number of combinations: n(n-1)(n-2) for n horses (order matters)
        num_combinations = self.max_horse_number * (self.max_horse_number - 1) * (self.max_horse_number - 2)
        
        # Initialize the result array with NaN
        result = np.full((len(race_ids), num_combinations), np.nan, dtype=np.float32)
        
        # Build the query to get the latest odds for each race
        query = f"""
            WITH latest_odds AS (
                SELECT
                    {self.race_id_query}
                    umaban_1,
                    umaban_2,
                    umaban_3,
                    odds,
                    ROW_NUMBER() OVER (
                        PARTITION BY kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban_1, umaban_2, umaban_3
                        ORDER BY happyo_datetime DESC
                    ) as row_num
                FROM
                    sanrentan
                WHERE
                    kaisai_date BETWEEN %s AND %s
            )
            SELECT
                race_id, umaban_1, umaban_2, umaban_3, odds
            FROM
                latest_odds
            WHERE
                row_num = 1
        """
        
        # Execute the query
        results = self.db_ops.execute_query(query, (min_date_str, max_date_str), fetch_all=True)
        
        # Create a dictionary to map horse number combinations to array indices
        sanrentan_index = {}
        idx = 0
        for uma1 in range(1, self.max_horse_number + 1):
            for uma2 in range(1, self.max_horse_number + 1):
                for uma3 in range(1, self.max_horse_number + 1):
                    if (uma1 != uma2) and (uma1 != uma3) and (uma2 != uma3):
                        sanrentan_index[(uma1, uma2, uma3)] = idx
                        idx += 1
        
        # Process the results
        for row in results:
            race_id = row[0]
            # Skip if race_id is not in our target list
            if race_id in race_id_to_index:
                idx = race_id_to_index[race_id]
                umaban_1 = row[1]
                umaban_2 = row[2]
                umaban_3 = row[3]
                odds = row[4]
                
                # Skip invalid horse numbers or repeating numbers
                if (1 <= umaban_1 <= self.max_horse_number and 
                    1 <= umaban_2 <= self.max_horse_number and 
                    1 <= umaban_3 <= self.max_horse_number and 
                    len(set([umaban_1, umaban_2, umaban_3])) == 3 and 
                    odds is not None):
                    # Get index for this combination
                    combo_idx = sanrentan_index[(umaban_1, umaban_2, umaban_3)]
                    # Odds are stored as integers with 10x multiplier, convert to float
                    result[idx, combo_idx] = odds / 10.0
        
        return result