import pytest
import numpy as np
import os
import tempfile
import pandas as pd
from horguesc.core.features.processor import FeatureProcessor
from unittest.mock import MagicMock

class TestFeatureProcessor:
    """Test cases for the FeatureProcessor class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        mock_config = MagicMock()
        mock_config.numerical_features = ['weight', 'height']
        mock_config.categorical_features = ['color', 'type']
        mock_config.feature_groups = {
            'numerical': {'weight': 'physical', 'height': 'physical'},
            'categorical': {'color': 'appearance', 'type': 'classification'}
        }
        mock_config.group_features = {
            'numerical': {'physical': ['weight', 'height']},
            'categorical': {'appearance': ['color'], 'classification': ['type']}
        }
        return mock_config
    
    @pytest.fixture
    def processor(self, mock_config):
        """Create a FeatureProcessor instance for testing."""
        return FeatureProcessor(mock_config)

    def test_initialization(self, processor, mock_config):
        """Test that FeatureProcessor initializes correctly."""
        assert processor.config == mock_config
        assert hasattr(processor, 'feature_groups')
        assert hasattr(processor, 'numerical_parameters')
        assert hasattr(processor, 'categorical_encoders')

    def test_collect_values_for_fitting(self, processor):
        """Test collecting values for fitting."""
        # Sample data
        raw_data = {
            'weight': np.array([70.5, 65.3, 80.0]),
            'height': np.array([175.0, 162.5, 185.2]),
            'color': np.array(['red', 'blue', 'green']),
            'type': np.array(['A', 'B', 'A'])
        }
        
        processor.collect_values_for_fitting(raw_data)
        
        # The processor should store values by group rather than by feature
        assert 'physical' in processor.group_numerical_values
        assert 'appearance' in processor.group_observed_values
        assert 'classification' in processor.group_observed_values
        
        # Values for physical group should contain data from both weight and height
        assert len(processor.group_numerical_values['physical']) > 0
        
        # Check categorical values were collected
        assert 'red' in processor.group_observed_values['appearance']
        assert 'A' in processor.group_observed_values['classification']

    def test_fit(self, processor):
        """Test fitting the processor."""
        # Setup sample data
        processor.group_numerical_values = {
            'physical': [70.5, 65.3, 80.0, 175.0, 162.5, 185.2]
        }
        processor.group_observed_values = {
            'appearance': {'red', 'blue', 'green'},
            'classification': {'A', 'B'}
        }
        
        processor.fit()
        
        # Check numerical parameters were computed
        assert 'physical' in processor.numerical_parameters
        assert 'mean_value' in processor.numerical_parameters['physical']
        assert 'std_value' in processor.numerical_parameters['physical']
        
        # Check categorical encodings were created
        assert 'appearance' in processor.categorical_encoders
        assert 'classification' in processor.categorical_encoders

    def test_normalize_numerical_feature(self, processor):
        """Test normalization of numerical features."""
        # Setup parameters
        processor.numerical_parameters = {
            'physical': {'mean_value': 70.0, 'std_value': 10.0}
        }
        processor.feature_groups = {
            'numerical': {'weight': 'physical'},
            'categorical': {}
        }
        
        # Test normalization
        values = np.array([60.0, 70.0, 80.0])
        normalized = processor.normalize_numerical_feature('weight', values)
        
        # Values should be: (60-70)/10=-1, (70-70)/10=0, (80-70)/10=1
        expected = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_encode_categorical_feature(self, processor):
        """Test encoding of categorical features."""
        # Setup encoders
        processor.categorical_encoders = {
            'appearance': {'red': 1, 'blue': 2, 'green': 3}
        }
        processor.feature_groups = {
            'numerical': {},
            'categorical': {'color': 'appearance'}
        }
        
        # Test encoding
        values = np.array(['red', 'blue', 'unknown', 'green'])
        encoded = processor.encode_categorical_feature('color', values)
        
        # Expected: red=1, blue=2, unknown=0 (unknown/padding), green=3
        expected = np.array([1, 2, 0, 3], dtype=np.int64)
        np.testing.assert_array_equal(encoded, expected)

    def test_save_and_load_state(self, processor):
        """Test saving and loading processor state."""
        # Setup processor state
        processor.numerical_parameters = {
            'physical': {'mean_value': 70.0, 'std_value': 10.0}
        }
        processor.categorical_encoders = {
            'appearance': {'red': 1, 'blue': 2, 'green': 3},
            'classification': {'A': 1, 'B': 2}
        }
        
        # Save state to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = tmp.name
        
        processor.save_state(temp_path)
        
        # Create a new processor and load the state
        new_processor = FeatureProcessor(processor.config)
        new_processor.load_state(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        # Check state was loaded correctly
        assert new_processor.numerical_parameters == processor.numerical_parameters
        assert new_processor.categorical_encoders == processor.categorical_encoders

    def test_get_group_cardinalities(self, processor):
        """Test getting group cardinalities."""
        # Setup encoders
        processor.categorical_encoders = {
            'appearance': {'red': 1, 'blue': 2, 'green': 3},
            'classification': {'A': 1, 'B': 2}
        }
        
        # Get cardinalities
        cardinalities = processor.get_group_cardinalities()
        
        # Check results
        assert cardinalities['appearance'] == 3  # color group has 3 values
        assert cardinalities['classification'] == 2  # type group has 2 values