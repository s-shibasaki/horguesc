import pytest
from unittest import mock
from horguesc.database.operations import DatabaseOperations  # Use absolute import instead of relative

class TestDatabaseOperations:
    
    @mock.patch('horguesc.database.connection.DatabaseConnection.get_instance')
    def test_initialization(self, mock_get_instance):
        # Arrange
        mock_config = mock.MagicMock()
        mock_db_conn = mock.MagicMock()
        mock_get_instance.return_value = mock_db_conn
        
        # Act
        db_ops = DatabaseOperations(mock_config)
        
        # Assert
        mock_get_instance.assert_called_once_with(mock_config)
        assert db_ops.db_conn == mock_db_conn
    
    @mock.patch('horguesc.database.connection.DatabaseConnection.get_instance')
    def test_execute_query_fetch_all(self, mock_get_instance):
        # Arrange
        mock_config = mock.MagicMock()
        mock_connection = mock.MagicMock()
        mock_cursor = mock.MagicMock()
        mock_cursor.fetchall.return_value = [('table1',), ('table2',)]
        mock_connection.cursor.return_value = mock_cursor
        mock_db_conn = mock.MagicMock()
        mock_db_conn.get_connection.return_value.__enter__.return_value = mock_connection
        mock_get_instance.return_value = mock_db_conn
        
        db_ops = DatabaseOperations(mock_config)
        
        # Act
        result = db_ops.execute_query("SELECT * FROM table", fetch_all=True)
        
        # Assert
        mock_cursor.execute.assert_called_once_with("SELECT * FROM table", None)
        mock_cursor.fetchall.assert_called_once()
        assert result == [('table1',), ('table2',)]
    
    @mock.patch('horguesc.database.connection.DatabaseConnection.get_instance')
    def test_execute_query_fetch_one(self, mock_get_instance):
        # Arrange
        mock_config = mock.MagicMock()
        mock_connection = mock.MagicMock()
        mock_cursor = mock.MagicMock()
        mock_cursor.fetchone.return_value = ('table1',)
        mock_connection.cursor.return_value = mock_cursor
        mock_db_conn = mock.MagicMock()
        mock_db_conn.get_connection.return_value.__enter__.return_value = mock_connection
        mock_get_instance.return_value = mock_db_conn
        
        db_ops = DatabaseOperations(mock_config)
        
        # Act
        result = db_ops.execute_query("SELECT * FROM table", fetch_all=False)
        
        # Assert
        mock_cursor.execute.assert_called_once_with("SELECT * FROM table", None)
        mock_cursor.fetchone.assert_called_once()
        assert result == ('table1',)
    
    @mock.patch('horguesc.database.connection.DatabaseConnection.get_instance')
    def test_execute_query_with_params(self, mock_get_instance):
        # Arrange
        mock_config = mock.MagicMock()
        mock_connection = mock.MagicMock()
        mock_cursor = mock.MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_db_conn = mock.MagicMock()
        mock_db_conn.get_connection.return_value.__enter__.return_value = mock_connection
        mock_get_instance.return_value = mock_db_conn
        
        db_ops = DatabaseOperations(mock_config)
        
        # Act
        db_ops.execute_query("SELECT * FROM table WHERE id = %s", [1])
        
        # Assert
        mock_cursor.execute.assert_called_once_with("SELECT * FROM table WHERE id = %s", (1,))
    
    @mock.patch('horguesc.database.connection.DatabaseConnection.get_instance')
    def test_execute_query_error(self, mock_get_instance):
        # Arrange
        mock_config = mock.MagicMock()
        mock_connection = mock.MagicMock()
        mock_cursor = mock.MagicMock()
        mock_cursor.execute.side_effect = Exception("Database error")
        mock_connection.cursor.return_value = mock_cursor
        mock_db_conn = mock.MagicMock()
        mock_db_conn.get_connection.return_value.__enter__.return_value = mock_connection
        mock_get_instance.return_value = mock_db_conn
        
        db_ops = DatabaseOperations(mock_config)
        
        # Act & Assert
        with pytest.raises(Exception):
            db_ops.execute_query("SELECT * FROM table")
    
    @mock.patch('horguesc.database.connection.DatabaseConnection.get_instance')
    def test_test_connection_success(self, mock_get_instance):
        # Arrange
        mock_config = mock.MagicMock()
        mock_connection = mock.MagicMock()
        mock_cursor = mock.MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_db_conn = mock.MagicMock()
        mock_db_conn.get_connection.return_value.__enter__.return_value = mock_connection
        mock_get_instance.return_value = mock_db_conn
        
        db_ops = DatabaseOperations(mock_config)
        
        # Act
        result = db_ops.test_connection()
        
        # Assert
        mock_cursor.execute.assert_called_once_with("SELECT 1")
        assert result is True
    
    @mock.patch('horguesc.database.connection.DatabaseConnection.get_instance')
    def test_test_connection_failure(self, mock_get_instance):
        # Arrange
        mock_config = mock.MagicMock()
        mock_db_conn = mock.MagicMock()
        mock_db_conn.get_connection.side_effect = Exception("Connection failed")
        mock_get_instance.return_value = mock_db_conn
        
        db_ops = DatabaseOperations(mock_config)
        
        # Act
        result = db_ops.test_connection()
        
        # Assert
        assert result is False
    
    @mock.patch('horguesc.database.connection.DatabaseConnection.get_instance')
    def test_get_table_list(self, mock_get_instance):
        # Arrange
        mock_config = mock.MagicMock()
        mock_db_conn = mock.MagicMock()
        mock_get_instance.return_value = mock_db_conn
        
        db_ops = DatabaseOperations(mock_config)
        
        # Mock execute_query to return specific results
        db_ops.execute_query = mock.MagicMock(return_value=[('table1',), ('table2',)])
        
        # Act
        result = db_ops.get_table_list()
        
        # Assert
        assert result == ['table1', 'table2']
        db_ops.execute_query.assert_called_once()
    
    @mock.patch('horguesc.database.connection.DatabaseConnection.get_instance')
    def test_get_table_list_error(self, mock_get_instance):
        # Arrange
        mock_config = mock.MagicMock()
        mock_db_conn = mock.MagicMock()
        mock_get_instance.return_value = mock_db_conn
        
        db_ops = DatabaseOperations(mock_config)
        
        # Mock execute_query to raise an exception
        db_ops.execute_query = mock.MagicMock(side_effect=Exception("Database error"))
        
        # Act & Assert
        with pytest.raises(Exception):
            db_ops.get_table_list()