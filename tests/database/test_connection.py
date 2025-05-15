import pytest
from configparser import ConfigParser
from unittest.mock import patch, MagicMock
from horguesc.database.connection import DatabaseConnection

@pytest.fixture
def mock_config():
    config = ConfigParser()
    config['database'] = {
        'Host': 'test_host',
        'Port': '5432',
        'Database': 'test_db',
        'Username': 'test_user',
        'Password': 'test_password'
    }
    return config


@pytest.fixture(autouse=True)
def mock_pool():
    """Mock psycopg2.pool.SimpleConnectionPool to prevent actual DB connections"""
    pool_mock = MagicMock()
    with patch('psycopg2.pool.SimpleConnectionPool', return_value=pool_mock) as mock:
        yield mock


def test_singleton_pattern(mock_config):
    # Reset the singleton for this test
    DatabaseConnection._instance = None
    
    # Test that get_instance returns the same instance
    instance1 = DatabaseConnection.get_instance(mock_config)
    instance2 = DatabaseConnection.get_instance()
    
    assert instance1 is instance2
    
    # Reset for other tests
    DatabaseConnection._instance = None


def test_initialization_requires_config():
    # Test that initialization without config raises ValueError
    with pytest.raises(ValueError):
        DatabaseConnection(None)


def test_initialize_pool(mock_config, mock_pool):
    # Test pool initialization with config
    connection = DatabaseConnection(mock_config)
    
    mock_pool.assert_called_once_with(
        1, 10,
        host='test_host',
        port=5432,
        database='test_db',
        user='test_user',
        password='test_password'
    )


def test_get_connection(mock_config):
    # Test get_connection method
    connection = DatabaseConnection(mock_config)
    connection._connection_pool = MagicMock()
    mock_conn = MagicMock()
    connection._connection_pool.getconn.return_value = mock_conn
    
    result = connection.get_connection()
    
    assert result == mock_conn
    connection._connection_pool.getconn.assert_called_once()


def test_release_connection(mock_config):
    # Test release_connection method
    connection = DatabaseConnection(mock_config)
    connection._connection_pool = MagicMock()
    mock_conn = MagicMock()
    
    connection.release_connection(mock_conn)
    connection._connection_pool.putconn.assert_called_once_with(mock_conn)


def test_close_all(mock_config):
    # Test close_all method
    connection = DatabaseConnection(mock_config)
    connection._connection_pool = MagicMock()
    
    connection.close_all()
    connection._connection_pool.closeall.assert_called_once()