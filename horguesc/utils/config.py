"""
Configuration utilities for horguesc.
"""
import configparser
import os
import logging
import sys
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# 複数の可能性のある場所から設定ファイルを探す
def find_config():
    search_paths = [
        os.path.join(os.path.dirname(sys.executable), 'horguesc.ini'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'horguesc.ini'),
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    return None

class Config(configparser.ConfigParser):
    """
    Main configuration class for horguesc that extends ConfigParser.
    Handles all application configuration including feature definitions.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the extended ConfigParser."""
        super().__init__(*args, **kwargs)
        self.numerical_features = {}  # 挿入順序が保持される
        self.categorical_features = {}  # 挿入順序が保持される
        self.num_features_count = 0
        self.cat_features_count = 0
    
    def parse_features(self):
        """Parse feature definitions from the configuration."""
        # Scan all sections to find feature definitions
        feature_prefix = "features."
        
        # モデル設定からデフォルトの数値特徴量埋め込み次元を取得
        default_numerical_dim = self.getint('model', 'default_numerical_embedding_dim', fallback=8)
        
        for section in self.sections():
            # Check if this section defines a feature
            if section.startswith(feature_prefix):
                feature_name = section[len(feature_prefix):]  # Get the part after "feature."
                feature_type = self.get(section, 'type', fallback='').lower()
                
                # Common feature attributes
                feature_info = {
                    'name': feature_name,
                    'description': self.get(section, 'description', fallback="")
                }
                
                # Process based on feature type
                if feature_type == 'numerical':
                    # 数値特徴量用のembedding_dimを取得（個別設定がなければデフォルト値を使用）
                    embedding_dim = self.getint(section, 'embedding_dim', fallback=default_numerical_dim)
                    
                    feature_info.update({
                        'normalization': self.get(section, 'normalization', fallback="none"),
                        'embedding_dim': embedding_dim
                    })
                    self.numerical_features[feature_name] = feature_info
                    
                elif feature_type == 'categorical':
                    # カーディナリティに基づく埋め込み次元の動的計算
                    cardinality = self.getint(section, 'cardinality')
                    suggested_dim = min(50, (cardinality+1)//2)  # または int(cardinality**0.25)
                    embedding_dim = self.getint(section, 'embedding_dim', fallback=suggested_dim)
                    
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
        return self
    
    def get_numerical_feature_names(self) -> List[str]:
        """Get list of numerical feature names in order of insertion."""
        return list(self.numerical_features.keys())
    
    def get_categorical_feature_names(self) -> List[str]:
        """Get list of categorical feature names in order of insertion."""
        return list(self.categorical_features.keys())
    
def load_config():
    """
    Load configuration from horguesc.ini file.
    
    Returns:
        Config: Configuration object with parsed features
    """
    config = Config()
    config_path = find_config()
    
    if config_path is None:
        logger.error("Configuration file not found in any of the expected locations")
        raise FileNotFoundError("Configuration file not found")
    
    try:
        config.read(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        # 特徴量情報を解析
        config.parse_features()
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise