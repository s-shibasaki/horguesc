"""
Database operations for horguesc.
"""
import logging
from .connection import DatabaseConnection

logger = logging.getLogger(__name__)

class DatabaseOperations:
    """Database operations for horguesc."""
    
    def __init__(self, config):
        """
        Initialize database operations.
        
        Args:
            config: ConfigParser object with database configuration
        """
        self.db_conn = DatabaseConnection.get_instance(config)
    
    def execute_query(self, query, params=None, fetch_all=False):
        """
        Execute a SQL query and return the results.
        
        Args:
            query (str): The SQL query to execute
            params (tuple, optional): Parameters for the query
            fetch_all (bool): Whether to fetch all results or just one
        
        Returns:
            The query results
        """
        try:
            if params and not isinstance(params, tuple):
                params = tuple(params)
                
            with self.db_conn.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                if fetch_all:
                    return cursor.fetchall()
                else:
                    return cursor.fetchone()
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def test_connection(self):
        """
        Test the database connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            with self.db_conn.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_table_list(self):
        """
        Get a list of tables in the database.
        
        Returns:
            list: List of table names
        """
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        """
        try:
            results = self.execute_query(query, fetch_all=True)
            return [row[0] for row in results]
        except Exception as e:
            logger.error(f"Failed to get table list: {str(e)}")
            raise