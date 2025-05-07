"""
Feature configuration utilities for horguesc.
"""
import configparser
import os
import logging
import ast
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class FeatureConfig:
    """Class to manage feature configurations with explicit named sections."""
    
    def __init__(self, config):
        """
        Initialize feature configuration.
        
        Args:
            config: ConfigParser object with feature configuration
        """
        self.config = config
        self.numerical_features = {}
        self.categorical_features = {}
        self._parse_config()
    
    def _parse_config(self):
        """Parse the configuration file and set up feature metadata."""
        # Scan all sections to find feature definitions
        feature_prefix = "features."
        
        for section in self.config.sections():
            # Check if this section defines a feature
            if section.startswith(feature_prefix):
                feature_name = section[len(feature_prefix):]  # Get the part after "feature."
                feature_type = self.config.get(section, 'type', fallback='').lower()
                
                # Common feature attributes
                feature_info = {
                    'name': feature_name,
                    'index': self.config.getint(section, 'index'),
                    'description': self.config.get(section, 'description', fallback="")
                }
                
                # Process based on feature type
                if feature_type == 'numerical':
                    feature_info.update({
                        'min_value': self.config.getfloat(section, 'min_value', fallback=None),
                        'max_value': self.config.getfloat(section, 'max_value', fallback=None),
                        'mean_value': self.config.getfloat(section, 'mean_value', fallback=None),
                        'std_value': self.config.getfloat(section, 'std_value', fallback=None),
                        'normalization': self.config.get(section, 'normalization', fallback="none")
                    })
                    self.numerical_features[feature_name] = feature_info
                    
                elif feature_type == 'categorical':
                    # カーディナリティに基づく埋め込み次元の動的計算
                    cardinality = self.config.getint(section, 'cardinality')
                    suggested_dim = min(50, (cardinality+1)//2)  # または int(cardinality**0.25)
                    embedding_dim = self.config.getint(section, 'embedding_dim', fallback=suggested_dim)
                    
                    feature_info.update({
                        'cardinality': cardinality,
                        'embedding_dim': embedding_dim,
                    })
                    self.categorical_features[feature_name] = feature_info
                    
                else:
                    logger.warning(f"Unknown feature type '{feature_type}' for {feature_name}")
        
        # Calculate counts after parsing
        self.num_features_count = len(self.numerical_features)
        self.cat_features_count = len(self.categorical_features)
        
        logger.info(f"Found {self.num_features_count} numerical features and {self.cat_features_count} categorical features")
        
        # Log pre-encoded features information
        pre_encoded_features = [name for name, info in self.categorical_features.items() 
                              if info.get('pre_encoded', False)]
        if pre_encoded_features:
            logger.info(f"Pre-encoded categorical features: {', '.join(pre_encoded_features)}")
    
    def get_numerical_feature_names(self) -> List[str]:
        """Get list of numerical feature names in order of index."""
        names = [""] * self.num_features_count
        for name, info in self.numerical_features.items():
            index = info['index']
            if 0 <= index < len(names):
                names[index] = name
        return names
    
    def get_categorical_feature_names(self) -> List[str]:
        """Get list of categorical feature names in order of index."""
        names = [""] * self.cat_features_count
        for name, info in self.categorical_features.items():
            index = info['index']
            if 0 <= index < len(names):
                names[index] = name
        return names
    
    def get_feature_info(self, name: str) -> Dict[str, Any]:
        """Get information about a specific feature by name."""
        if name in self.numerical_features:
            return self.numerical_features[name]
        elif name in self.categorical_features:
            return self.categorical_features[name]
        else:
            raise ValueError(f"Feature '{name}' not found in configuration")
    
    def get_categorical_cardinalities(self) -> List[int]:
        """Get list of cardinalities for categorical features in order of index."""
        if not self.cat_features_count:
            return []
            
        cardinalities = [0] * self.cat_features_count
        
        for name, info in self.categorical_features.items():
            index = info['index']
            if 0 <= index < self.cat_features_count:
                cardinalities[index] = info['cardinality']
        
        return cardinalities
    
    def get_embedding_dimensions(self) -> List[int]:
        """Get list of embedding dimensions for categorical features in order of index."""
        if not self.cat_features_count:
            return []
            
        dims = [4] * self.cat_features_count  # Default embedding dimension
        
        for name, info in self.categorical_features.items():
            index = info['index']
            if 0 <= index < self.cat_features_count:
                dims[index] = info.get('embedding_dim', 4)
        
        return dims
    
    def validate_indices(self) -> bool:
        """
        Validate that feature indices are continuous and don't have gaps.
        
        Returns:
            bool: True if indices are valid, False otherwise
        """
        # Check numerical features
        if self.num_features_count > 0:
            num_indices = sorted([info['index'] for info in self.numerical_features.values()])
            if num_indices != list(range(len(num_indices))):
                logger.warning("Numerical feature indices are not continuous")
                return False
        
        # Check categorical features
        if self.cat_features_count > 0:
            cat_indices = sorted([info['index'] for info in self.categorical_features.values()])
            if cat_indices != list(range(len(cat_indices))):
                logger.warning("Categorical feature indices are not continuous")
                return False
        
        return True

def load_feature_config(config_main):
    """
    Load feature configuration from main config.
    
    Args:
        config_main: Main configuration object
        
    Returns:
        FeatureConfig: Feature configuration object
    """
    # 既存のconfigからFeatureConfigを初期化
    feature_config = FeatureConfig(config_main)
    
    # インデックスの検証
    if not feature_config.validate_indices():
        logger.warning("Feature indices have gaps or are not continuous. This may cause issues.")
    
    logger.info(f"Loaded feature configuration: {feature_config.num_features_count} numerical and "
                f"{feature_config.cat_features_count} categorical features")
    return feature_config