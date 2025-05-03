"""
Database connection functionality for horguesc.
"""
import psycopg2
from psycopg2 import pool
import logging

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Manages database connections for horguesc."""
    
    _instance = None
    _connection_pool = None
    
    @classmethod
    def get_instance(cls, config=None):
        """
        Get the singleton instance of DatabaseConnection.
        
        Args:
            config: ConfigParser object with database configuration
            
        Returns:
            DatabaseConnection: The singleton instance
        """
        if cls._instance is None:
            cls._instance = DatabaseConnection(config)
        return cls._instance
    
    def __init__(self, config):
        """
        Initialize the database connection.
        
        Args:
            config: ConfigParser object with database configuration
        """
        self.config = config
        self._initialize_pool()
        
    def _initialize_pool(self):
        """Initialize the connection pool."""
        if not self.config:
            raise ValueError("Database configuration is required")
        
        try:
            db_config = self.config['database']
            self._connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 10,  # min_conn, max_conn
                host=db_config.get('Host', 'localhost'),
                port=db_config.getint('Port', 5432),
                database=db_config.get('Database', 'horguesc'),
                user=db_config.get('Username', 'postgres'),
                password=db_config.get('Password', 'postgres')
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {str(e)}")
            raise
    
    def get_connection(self):
        """
        Get a connection from the pool.
        
        Returns:
            connection: A database connection
        """
        if self._connection_pool is None:
            self._initialize_pool()
        
        connection = self._connection_pool.getconn()
        return connection
    
    def release_connection(self, connection):
        """
        Release a connection back to the pool.
        
        Args:
            connection: The connection to release
        """
        self._connection_pool.putconn(connection)
    
    def execute_query(self, query, params=None, fetch_all=False):
        """
        Execute a query and optionally return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch_all: Whether to fetch all results
            
        Returns:
            list or None: Query results if fetch_all is True
        """
        connection = None
        cursor = None
        result = None
        
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            cursor.execute(query, params or ())
            
            if fetch_all:
                result = cursor.fetchall()
            else:
                connection.commit()
                
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database query error: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                self.release_connection(connection)
                
        return result
    
    def close_all(self):
        """Close all connections in the pool."""
        if self._connection_pool:
            self._connection_pool.closeall()
            logger.info("All database connections closed")