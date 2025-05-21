"""
Database operations for horguesc.
"""
import logging
from .connection import DatabaseConnection
import psycopg2.extras

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
    
    def execute_query(self, query, params=None, fetch_all=False, as_dict=False):
        """
        Execute a SQL query and return the results.
        
        Args:
            query (str): The SQL query to execute
            params (tuple, optional): Parameters for the query
            fetch_all (bool): Whether to fetch all results or just one
            as_dict (bool): Whether to return results as dictionaries
        
        Returns:
            The query results
        """
        try:
            if params and not isinstance(params, tuple):
                params = tuple(params)
                
            conn = None
            try:
                conn = self.db_conn.get_connection()
                cursor_factory = psycopg2.extras.DictCursor if as_dict else None
                with conn.cursor(cursor_factory=cursor_factory) as cursor:
                    cursor.execute(query, params)
                    
                    if fetch_all:
                        return cursor.fetchall()
                    else:
                        return cursor.fetchone()
            finally:
                if conn:
                    self.db_conn.release_connection(conn)
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
            conn = None
            try:
                conn = self.db_conn.get_connection()
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return True
            finally:
                if conn:
                    self.db_conn.release_connection(conn)
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