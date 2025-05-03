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
        Execute a database query and return the results.
        
        Args:
            query: SQL query string
            params: Optional parameters for the query
            fetch_all: Whether to fetch all results
            
        Returns:
            Query results if fetch_all is True, otherwise None
        """
        try:
            return self.db_conn.execute_query(query, params, fetch_all)
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
            self.db_conn.execute_query("SELECT 1")
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
            results = self.db_conn.execute_query(query, fetch_all=True)
            return [row[0] for row in results]
        except Exception as e:
            logger.error(f"Failed to get table list: {str(e)}")
            raise