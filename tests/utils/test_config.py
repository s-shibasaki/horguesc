import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from horguesc.utils.config import Config, load_config

@pytest.fixture
def sample_config_content():
    """Return sample configuration content for testing."""
    return """
[database]
Host=testhost
Port=5432
Database=testdb
Username=testuser
Password=testpass

[features]
numerical_features=bataiju,futan_juryo,kyori
categorical_features=umaban,track_code,course_kubun

[groups]
group1=bataiju,futan_juryo
group2=umaban,track_code

[training]
train_start_date=2023-01-01
train_end_date=2023-12-31
val_start_date=2024-01-01
val_end_date=2024-12-31
"""

@pytest.fixture
def config_file(sample_config_content):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.ini') as f:
        f.write(sample_config_content)
        path = f.name
    
    yield path
    
    # Clean up temp file after test
    os.unlink(path)

def test_config_load_from_file(config_file):
    """Test loading configuration from a file."""
    config = Config()
    config.load_from_file(config_file)
    
    # Check if sections were loaded
    assert 'database' in config
    assert 'features' in config
    assert 'groups' in config
    assert 'training' in config
    
    # Check specific values
    assert config['database']['Host'] == 'testhost'
    assert config['features']['numerical_features'] == 'bataiju,futan_juryo,kyori'

def test_config_parse_features(config_file):
    """Test parsing feature definitions from configuration."""
    config = Config()
    config.load_from_file(config_file)
    config.parse_features()
    
    # Check numerical and categorical features lists
    assert config.numerical_features == ['bataiju', 'futan_juryo', 'kyori']
    assert config.categorical_features == ['umaban', 'track_code', 'course_kubun']
    
    # Check feature groups
    assert config.feature_groups['numerical']['bataiju'] == 'group1'
    assert config.feature_groups['numerical']['futan_juryo'] == 'group1'
    assert config.feature_groups['numerical']['kyori'] == 'kyori'  # Self-mapping for ungrouped features
    
    assert config.feature_groups['categorical']['umaban'] == 'group2'
    assert config.feature_groups['categorical']['track_code'] == 'group2'
    assert config.feature_groups['categorical']['course_kubun'] == 'course_kubun'  # Self-mapping

def test_config_validate_dates(config_file):
    """Test date validation in configuration."""
    config = Config()
    config.load_from_file(config_file)
    
    # This should validate without errors
    config.validate()
    
    # Modify a date to be invalid
    config['training']['train_start_date'] = 'invalid-date'
    
    # Validation should now fail
    with pytest.raises(ValueError):
        config.validate()

def test_update_from_args(config_file):
    """Test updating configuration from command line arguments."""
    config = Config()
    config.load_from_file(config_file)
    
    # Create a mock args object
    args = MagicMock()
    args.train_start = '2023-02-01'
    # Add these missing attributes that the update_from_args method is trying to access
    args.train_end = None
    args.val_start = None
    args.val_end = None
    args.save_each_epoch = True
    args.skip_validation = True
    
    # Update configuration with mock args
    config.update_from_args(args)
    
    # Check if values were updated
    assert config['training']['train_start_date'] == '2023-02-01'
    assert config.getboolean('training', 'save_each_epoch') is True
    assert config.getboolean('training', 'skip_validation') is True

def test_load_config_function():
    """Test the load_config helper function."""
    with patch('horguesc.utils.config.Config') as MockConfig:
        # Create a mock config instance
        mock_config = MagicMock()
        MockConfig.return_value = mock_config
        
        # Setup the mock's method chain
        mock_config.load_from_file.return_value = mock_config
        mock_config.parse_features.return_value = mock_config
        mock_config.validate.return_value = mock_config
        
        # Call the function
        result = load_config()
        
        # Verify the expected methods were called
        mock_config.load_from_file.assert_called_once()
        mock_config.parse_features.assert_called_once()
        mock_config.validate.assert_called_once()
        
        # The function should return the config object
        assert result == mock_config