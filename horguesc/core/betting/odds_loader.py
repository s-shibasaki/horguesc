"""
Module for loading betting odds from the database.
"""
import logging
import numpy as np
import itertools
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
    
    def __init__(self, db_ops):
        """
        Initialize the OddsLoader with database operations.
        
        Args:
            db_ops: Database operations object
        """
        self.db_ops = db_ops
        self.max_frame_number = 8  # Maximum frame number (defined in requirements)
    
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
        if odds_types is None:
            odds_types = self.ALL_ODDS_TYPES
            
        # Validate input
        if not race_ids:
            logger.warning("No race IDs provided")
            return {}
            
        # Get date range from race IDs
        min_date, max_date = self._get_date_range_from_race_ids(race_ids)
        logger.info(f"Loading odds for date range: {min_date} to {max_date}")
        
        # Dictionary to store results
        odds_data = {}
        
        # Get max horses count from data
        num_races, max_horses = horse_numbers.shape
        
        # Load each requested odds type
        for odds_type in odds_types:
            if odds_type == self.TANSHO:
                odds_data[odds_type] = self._load_tansho(race_ids, horse_numbers, min_date, max_date)
            elif odds_type == self.FUKUSHO:
                odds_data[odds_type] = self._load_fukusho(race_ids, horse_numbers, min_date, max_date)
            elif odds_type == self.WAKUREN:
                odds_data[odds_type] = self._load_wakuren(race_ids, frame_numbers, min_date, max_date)
            elif odds_type == self.UMAREN:
                odds_data[odds_type] = self._load_umaren(race_ids, horse_numbers, min_date, max_date)
            elif odds_type == self.WIDE:
                odds_data[odds_type] = self._load_wide(race_ids, horse_numbers, min_date, max_date)
            elif odds_type == self.UMATAN:
                odds_data[odds_type] = self._load_umatan(race_ids, horse_numbers, min_date, max_date)
            elif odds_type == self.SANRENPUKU:
                odds_data[odds_type] = self._load_sanrenpuku(race_ids, horse_numbers, min_date, max_date)
            elif odds_type == self.SANRENTAN:
                odds_data[odds_type] = self._load_sanrentan(race_ids, horse_numbers, min_date, max_date)
        
        return odds_data
    
    def _get_date_range_from_race_ids(self, race_ids: List[str]) -> Tuple[str, str]:
        """
        Extract date range from race IDs.
        
        Args:
            race_ids: List of race IDs where first 8 chars represent date (YYYYMMDD)
            
        Returns:
            Tuple of (min_date, max_date) in YYYY-MM-DD format
        """
        dates = [race_id[:8] for race_id in race_ids]
        min_date_str = min(dates)
        max_date_str = max(dates)
        
        # Convert to YYYY-MM-DD format
        min_date = f"{min_date_str[:4]}-{min_date_str[4:6]}-{min_date_str[6:8]}"
        max_date = f"{max_date_str[:4]}-{max_date_str[4:6]}-{max_date_str[6:8]}"
        
        return min_date, max_date
    
    def _extract_race_components(self, race_id: str) -> Tuple[str, str, str, str, str]:
        """
        Extract components from race ID.
        
        Args:
            race_id: Race ID in format YYYYMMDDKKRRNNRR
                    (date, track_code, kai, nichime, race_number)
            
        Returns:
            Tuple of (date, track_code, kai, nichime, race_number)
        """
        date = f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}"
        track_code = race_id[8:10]
        kai = race_id[10:12]
        nichime = race_id[12:14]
        race_number = race_id[14:16]
        
        return date, track_code, kai, nichime, race_number
    
    def _build_race_id_map(self, race_ids: List[str]) -> Dict[Tuple, int]:
        """
        Build a mapping from race components to indices.
        
        Args:
            race_ids: List of race IDs
            
        Returns:
            Dictionary mapping (date, track, kai, nichime, race) to index
        """
        race_id_map = {}
        for i, race_id in enumerate(race_ids):
            date, track, kai, nichime, race = self._extract_race_components(race_id)
            race_id_map[(date, track, kai, nichime, race)] = i
        return race_id_map
    
    def _load_tansho(self, race_ids: List[str], horse_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load win odds (単勝).
        
        Args:
            race_ids: List of race IDs
            horse_numbers: Array of horse numbers
            min_date: Minimum date in YYYY-MM-DD format
            max_date: Maximum date in YYYY-MM-DD format
            
        Returns:
            Array of win odds with shape (num_races, max_horses)
        """
        num_races, max_horses = horse_numbers.shape
        
        # Initialize odds array with NaN
        odds_array = np.full((num_races, max_horses), np.nan, dtype=np.float32)
        
        # Build race ID map
        race_id_map = self._build_race_id_map(race_ids)
        
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
            # Execute query
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            
            # Process results
            for row in results:
                # Convert date to string in YYYY-MM-DD format
                date_str = row['kaisai_date']
                if hasattr(date_str, 'strftime'):
                    date_str = date_str.strftime('%Y-%m-%d')
                
                # Create key for race_id_map
                key = (
                    date_str,
                    str(row['keibajo_code']).zfill(2),
                    str(row['kaisai_kai']).zfill(2),
                    str(row['kaisai_nichime']).zfill(2),
                    str(row['kyoso_bango']).zfill(2)
                )
                
                if key in race_id_map:
                    race_idx = race_id_map[key]
                    umaban = int(row['umaban']) - 1  # Convert to 0-based index
                    odds_value = row['odds']
                    
                    # Only set odds if umaban is within bounds and odds exists
                    if 0 <= umaban < max_horses and odds_value is not None:
                        # Convert to decimal odds (divide by 10 since DB stores odds*10)
                        odds_array[race_idx, umaban] = float(odds_value) / 10.0
            
            logger.info(f"Loaded tansho odds for {len(results)} horse-race combinations")
            
        except Exception as e:
            logger.error(f"Error loading tansho odds: {e}")
            
        return odds_array
    
    def _load_fukusho(self, race_ids: List[str], horse_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load place odds (複勝).
        
        Args:
            race_ids: List of race IDs
            horse_numbers: Array of horse numbers
            min_date: Minimum date in YYYY-MM-DD format
            max_date: Maximum date in YYYY-MM-DD format
            
        Returns:
            Array of place odds with shape (num_races, max_horses)
        """
        num_races, max_horses = horse_numbers.shape
        
        # Initialize odds array with NaN
        odds_array = np.full((num_races, max_horses), np.nan, dtype=np.float32)
        
        # Build race ID map
        race_id_map = self._build_race_id_map(race_ids)
        
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
            # Execute query
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            
            # Process results (use min_odds as specified in requirements)
            for row in results:
                # Convert date to string in YYYY-MM-DD format
                date_str = row['kaisai_date']
                if hasattr(date_str, 'strftime'):
                    date_str = date_str.strftime('%Y-%m-%d')
                
                # Create key for race_id_map
                key = (
                    date_str,
                    str(row['keibajo_code']).zfill(2),
                    str(row['kaisai_kai']).zfill(2),
                    str(row['kaisai_nichime']).zfill(2),
                    str(row['kyoso_bango']).zfill(2)
                )
                
                if key in race_id_map:
                    race_idx = race_id_map[key]
                    umaban = int(row['umaban']) - 1  # Convert to 0-based index
                    odds_value = row['min_odds']  # Use min_odds as specified
                    
                    # Only set odds if umaban is within bounds and odds exists
                    if 0 <= umaban < max_horses and odds_value is not None:
                        # Convert to decimal odds
                        odds_array[race_idx, umaban] = float(odds_value) / 10.0
            
            logger.info(f"Loaded fukusho odds for {len(results)} horse-race combinations")
            
        except Exception as e:
            logger.error(f"Error loading fukusho odds: {e}")
            
        return odds_array
    
    def _load_wakuren(self, race_ids: List[str], frame_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load bracket quinella odds (枠連).
        
        Args:
            race_ids: List of race IDs
            frame_numbers: Array of frame numbers
            min_date: Minimum date in YYYY-MM-DD format
            max_date: Maximum date in YYYY-MM-DD format
            
        Returns:
            Array of bracket quinella odds with shape (num_races, combinations)
        """
        num_races = len(race_ids)
        
        # For wakuren, we include same-frame combinations (e.g., 1-1, 2-2)
        # Total combinations: max_frames * (max_frames + 1) / 2
        max_frames = self.max_frame_number
        num_combinations = (max_frames * (max_frames + 1)) // 2
        
        # Initialize odds array with NaN
        odds_array = np.full((num_races, num_combinations), np.nan, dtype=np.float32)
        
        # Build race ID map
        race_id_map = self._build_race_id_map(race_ids)
        
        # Create combination to index mapping
        # For wakuren, we need to include same-frame combinations
        combo_to_idx = {}
        idx = 0
        for frame1 in range(1, max_frames + 1):
            for frame2 in range(frame1, max_frames + 1):
                combo_to_idx[(frame1, frame2)] = idx
                idx += 1
        
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
            # Execute query
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            
            # Process results
            for row in results:
                # Convert date to string in YYYY-MM-DD format
                date_str = row['kaisai_date']
                if hasattr(date_str, 'strftime'):
                    date_str = date_str.strftime('%Y-%m-%d')
                
                # Create key for race_id_map
                key = (
                    date_str,
                    str(row['keibajo_code']).zfill(2),
                    str(row['kaisai_kai']).zfill(2),
                    str(row['kaisai_nichime']).zfill(2),
                    str(row['kyoso_bango']).zfill(2)
                )
                
                if key in race_id_map:
                    race_idx = race_id_map[key]
                    wakuban_1 = int(row['wakuban_1'])
                    wakuban_2 = int(row['wakuban_2'])
                    
                    # Ensure wakuban_1 <= wakuban_2 for consistent ordering
                    if wakuban_1 > wakuban_2:
                        wakuban_1, wakuban_2 = wakuban_2, wakuban_1
                    
                    # Only process if both frame numbers are valid
                    if 1 <= wakuban_1 <= max_frames and 1 <= wakuban_2 <= max_frames:
                        combo_idx = combo_to_idx.get((wakuban_1, wakuban_2))
                        odds_value = row['odds']
                        
                        if combo_idx is not None and odds_value is not None:
                            odds_array[race_idx, combo_idx] = float(odds_value) / 10.0
            
            logger.info(f"Loaded wakuren odds for {len(results)} frame-race combinations")
            
        except Exception as e:
            logger.error(f"Error loading wakuren odds: {e}")
            
        return odds_array
    
    def _load_umaren(self, race_ids: List[str], horse_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load quinella odds (馬連).
        
        Args:
            race_ids: List of race IDs
            horse_numbers: Array of horse numbers
            min_date: Minimum date in YYYY-MM-DD format
            max_date: Maximum date in YYYY-MM-DD format
            
        Returns:
            Array of quinella odds with shape (num_races, combinations)
        """
        num_races, max_horses = horse_numbers.shape
        
        # For umaren, we need nC2 combinations (order doesn't matter)
        num_combinations = (max_horses * (max_horses - 1)) // 2
        
        # Initialize odds array with NaN
        odds_array = np.full((num_races, num_combinations), np.nan, dtype=np.float32)
        
        # Build race ID map
        race_id_map = self._build_race_id_map(race_ids)
        
        # Create combination to index mapping
        combo_to_idx = {}
        idx = 0
        for horse1 in range(1, max_horses + 1):
            for horse2 in range(horse1 + 1, max_horses + 1):
                combo_to_idx[(horse1, horse2)] = idx
                idx += 1
        
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
            # Execute query
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            
            # Process results
            for row in results:
                # Convert date to string in YYYY-MM-DD format
                date_str = row['kaisai_date']
                if hasattr(date_str, 'strftime'):
                    date_str = date_str.strftime('%Y-%m-%d')
                
                # Create key for race_id_map
                key = (
                    date_str,
                    str(row['keibajo_code']).zfill(2),
                    str(row['kaisai_kai']).zfill(2),
                    str(row['kaisai_nichime']).zfill(2),
                    str(row['kyoso_bango']).zfill(2)
                )
                
                if key in race_id_map:
                    race_idx = race_id_map[key]
                    umaban_1 = int(row['umaban_1'])
                    umaban_2 = int(row['umaban_2'])
                    
                    # Ensure umaban_1 < umaban_2 for consistent ordering
                    if umaban_1 > umaban_2:
                        umaban_1, umaban_2 = umaban_2, umaban_1
                    
                    # Only process if both horse numbers are valid
                    if 1 <= umaban_1 <= max_horses and 1 <= umaban_2 <= max_horses:
                        combo_idx = combo_to_idx.get((umaban_1, umaban_2))
                        odds_value = row['odds']
                        
                        if combo_idx is not None and odds_value is not None:
                            odds_array[race_idx, combo_idx] = float(odds_value) / 10.0
            
            logger.info(f"Loaded umaren odds for {len(results)} horse-race combinations")
            
        except Exception as e:
            logger.error(f"Error loading umaren odds: {e}")
            
        return odds_array
    
    def _load_wide(self, race_ids: List[str], horse_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load quinella place odds (ワイド).
        
        Args:
            race_ids: List of race IDs
            horse_numbers: Array of horse numbers
            min_date: Minimum date in YYYY-MM-DD format
            max_date: Maximum date in YYYY-MM-DD format
            
        Returns:
            Array of quinella place odds with shape (num_races, combinations)
        """
        num_races, max_horses = horse_numbers.shape
        
        # For wide, we need nC2 combinations (order doesn't matter)
        num_combinations = (max_horses * (max_horses - 1)) // 2
        
        # Initialize odds array with NaN
        odds_array = np.full((num_races, num_combinations), np.nan, dtype=np.float32)
        
        # Build race ID map
        race_id_map = self._build_race_id_map(race_ids)
        
        # Create combination to index mapping
        combo_to_idx = {}
        idx = 0
        for horse1 in range(1, max_horses + 1):
            for horse2 in range(horse1 + 1, max_horses + 1):
                combo_to_idx[(horse1, horse2)] = idx
                idx += 1
        
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
            # Execute query
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            
            # Process results (use min_odds as specified in requirements)
            for row in results:
                # Convert date to string in YYYY-MM-DD format
                date_str = row['kaisai_date']
                if hasattr(date_str, 'strftime'):
                    date_str = date_str.strftime('%Y-%m-%d')
                
                # Create key for race_id_map
                key = (
                    date_str,
                    str(row['keibajo_code']).zfill(2),
                    str(row['kaisai_kai']).zfill(2),
                    str(row['kaisai_nichime']).zfill(2),
                    str(row['kyoso_bango']).zfill(2)
                )
                
                if key in race_id_map:
                    race_idx = race_id_map[key]
                    umaban_1 = int(row['umaban_1'])
                    umaban_2 = int(row['umaban_2'])
                    
                    # Ensure umaban_1 < umaban_2 for consistent ordering
                    if umaban_1 > umaban_2:
                        umaban_1, umaban_2 = umaban_2, umaban_1
                    
                    # Only process if both horse numbers are valid
                    if 1 <= umaban_1 <= max_horses and 1 <= umaban_2 <= max_horses:
                        combo_idx = combo_to_idx.get((umaban_1, umaban_2))
                        odds_value = row['min_odds']  # Use min_odds as specified
                        
                        if combo_idx is not None and odds_value is not None:
                            odds_array[race_idx, combo_idx] = float(odds_value) / 10.0
            
            logger.info(f"Loaded wide odds for {len(results)} horse-race combinations")
            
        except Exception as e:
            logger.error(f"Error loading wide odds: {e}")
            
        return odds_array
    
    def _load_umatan(self, race_ids: List[str], horse_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load exacta odds (馬単).
        
        Args:
            race_ids: List of race IDs
            horse_numbers: Array of horse numbers
            min_date: Minimum date in YYYY-MM-DD format
            max_date: Maximum date in YYYY-MM-DD format
            
        Returns:
            Array of exacta odds with shape (num_races, combinations)
        """
        num_races, max_horses = horse_numbers.shape
        
        # For umatan, we need nP2 combinations (order matters)
        num_combinations = max_horses * (max_horses - 1)
        
        # Initialize odds array with NaN
        odds_array = np.full((num_races, num_combinations), np.nan, dtype=np.float32)
        
        # Build race ID map
        race_id_map = self._build_race_id_map(race_ids)
        
        # Create combination to index mapping
        combo_to_idx = {}
        idx = 0
        for horse1 in range(1, max_horses + 1):
            for horse2 in range(1, max_horses + 1):
                if horse1 != horse2:
                    combo_to_idx[(horse1, horse2)] = idx
                    idx += 1
        
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
            # Execute query
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            
            # Process results
            for row in results:
                # Convert date to string in YYYY-MM-DD format
                date_str = row['kaisai_date']
                if hasattr(date_str, 'strftime'):
                    date_str = date_str.strftime('%Y-%m-%d')
                
                # Create key for race_id_map
                key = (
                    date_str,
                    str(row['keibajo_code']).zfill(2),
                    str(row['kaisai_kai']).zfill(2),
                    str(row['kaisai_nichime']).zfill(2),
                    str(row['kyoso_bango']).zfill(2)
                )
                
                if key in race_id_map:
                    race_idx = race_id_map[key]
                    umaban_1 = int(row['umaban_1'])
                    umaban_2 = int(row['umaban_2'])
                    
                    # For umatan, order matters
                    if 1 <= umaban_1 <= max_horses and 1 <= umaban_2 <= max_horses and umaban_1 != umaban_2:
                        combo_idx = combo_to_idx.get((umaban_1, umaban_2))
                        odds_value = row['odds']
                        
                        if combo_idx is not None and odds_value is not None:
                            odds_array[race_idx, combo_idx] = float(odds_value) / 10.0
            
            logger.info(f"Loaded umatan odds for {len(results)} horse-race combinations")
            
        except Exception as e:
            logger.error(f"Error loading umatan odds: {e}")
            
        return odds_array
    
    def _load_sanrenpuku(self, race_ids: List[str], horse_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load trio odds (三連複).
        
        Args:
            race_ids: List of race IDs
            horse_numbers: Array of horse numbers
            min_date: Minimum date in YYYY-MM-DD format
            max_date: Maximum date in YYYY-MM-DD format
            
        Returns:
            Array of trio odds with shape (num_races, combinations)
        """
        num_races, max_horses = horse_numbers.shape
        
        # For sanrenpuku, we need nC3 combinations (order doesn't matter)
        num_combinations = (max_horses * (max_horses - 1) * (max_horses - 2)) // 6
        
        # Initialize odds array with NaN
        odds_array = np.full((num_races, num_combinations), np.nan, dtype=np.float32)
        
        # Build race ID map
        race_id_map = self._build_race_id_map(race_ids)
        
        # Create combination to index mapping
        combo_to_idx = {}
        idx = 0
        for horse1 in range(1, max_horses + 1):
            for horse2 in range(horse1 + 1, max_horses + 1):
                for horse3 in range(horse2 + 1, max_horses + 1):
                    combo_to_idx[(horse1, horse2, horse3)] = idx
                    idx += 1
        
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
            # Execute query
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            
            # Process results
            for row in results:
                # Convert date to string in YYYY-MM-DD format
                date_str = row['kaisai_date']
                if hasattr(date_str, 'strftime'):
                    date_str = date_str.strftime('%Y-%m-%d')
                
                # Create key for race_id_map
                key = (
                    date_str,
                    str(row['keibajo_code']).zfill(2),
                    str(row['kaisai_kai']).zfill(2),
                    str(row['kaisai_nichime']).zfill(2),
                    str(row['kyoso_bango']).zfill(2)
                )
                
                if key in race_id_map:
                    race_idx = race_id_map[key]
                    umaban_1 = int(row['umaban_1'])
                    umaban_2 = int(row['umaban_2'])
                    umaban_3 = int(row['umaban_3'])
                    
                    # Sort to get canonical ordering
                    umabans = sorted([umaban_1, umaban_2, umaban_3])
                    
                    # Only process if all three horse numbers are valid and different
                    if (1 <= umabans[0] <= max_horses and 
                        1 <= umabans[1] <= max_horses and 
                        1 <= umabans[2] <= max_horses and 
                        umabans[0] < umabans[1] < umabans[2]):
                        
                        combo_idx = combo_to_idx.get(tuple(umabans))
                        odds_value = row['odds']
                        
                        if combo_idx is not None and odds_value is not None:
                            odds_array[race_idx, combo_idx] = float(odds_value) / 10.0
            
            logger.info(f"Loaded sanrenpuku odds for {len(results)} horse-race combinations")
            
        except Exception as e:
            logger.error(f"Error loading sanrenpuku odds: {e}")
            
        return odds_array
    
    def _load_sanrentan(self, race_ids: List[str], horse_numbers: np.ndarray, min_date: str, max_date: str) -> np.ndarray:
        """
        Load trifecta odds (三連単).
        
        Args:
            race_ids: List of race IDs
            horse_numbers: Array of horse numbers
            min_date: Minimum date in YYYY-MM-DD format
            max_date: Maximum date in YYYY-MM-DD format
            
        Returns:
            Array of trifecta odds with shape (num_races, combinations)
        """
        num_races, max_horses = horse_numbers.shape
        
        # For sanrentan, we need nP3 combinations (order matters)
        num_combinations = max_horses * (max_horses - 1) * (max_horses - 2)
        
        # Initialize odds array with NaN
        odds_array = np.full((num_races, num_combinations), np.nan, dtype=np.float32)
        
        # Build race ID map
        race_id_map = self._build_race_id_map(race_ids)
        
        # Create combination to index mapping
        combo_to_idx = {}
        idx = 0
        for horse1 in range(1, max_horses + 1):
            for horse2 in range(1, max_horses + 1):
                if horse2 == horse1:
                    continue
                for horse3 in range(1, max_horses + 1):
                    if horse3 == horse1 or horse3 == horse2:
                        continue
                    combo_to_idx[(horse1, horse2, horse3)] = idx
                    idx += 1
        
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
            # Execute query
            results = self.db_ops.execute_query(query, params=[min_date, max_date], fetch_all=True, as_dict=True)
            
            # Process results
            for row in results:
                # Convert date to string in YYYY-MM-DD format
                date_str = row['kaisai_date']
                if hasattr(date_str, 'strftime'):
                    date_str = date_str.strftime('%Y-%m-%d')
                
                # Create key for race_id_map
                key = (
                    date_str,
                    str(row['keibajo_code']).zfill(2),
                    str(row['kaisai_kai']).zfill(2),
                    str(row['kaisai_nichime']).zfill(2),
                    str(row['kyoso_bango']).zfill(2)
                )
                
                if key in race_id_map:
                    race_idx = race_id_map[key]
                    umaban_1 = int(row['umaban_1'])
                    umaban_2 = int(row['umaban_2'])
                    umaban_3 = int(row['umaban_3'])
                    
                    # For sanrentan, order matters
                    if (1 <= umaban_1 <= max_horses and 
                        1 <= umaban_2 <= max_horses and 
                        1 <= umaban_3 <= max_horses and 
                        umaban_1 != umaban_2 and umaban_1 != umaban_3 and umaban_2 != umaban_3):
                        
                        combo_idx = combo_to_idx.get((umaban_1, umaban_2, umaban_3))
                        odds_value = row['odds']
                        
                        if combo_idx is not None and odds_value is not None:
                            odds_array[race_idx, combo_idx] = float(odds_value) / 10.0
            
            logger.info(f"Loaded sanrentan odds for {len(results)} horse-race combinations")
            
        except Exception as e:
            logger.error(f"Error loading sanrentan odds: {e}")
            
        return odds_array