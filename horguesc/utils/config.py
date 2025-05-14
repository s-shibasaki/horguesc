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

    def parse_features(self):
        """Parse feature definitions from the configuration."""
        # 特徴量リストを取得
        numerical_features_str = self.get('features', 'numerical_features', fallback='').strip()
        categorical_features_str = self.get('features', 'categorical_features', fallback='').strip()
        
        # 数値特徴量とカテゴリカル特徴量のリストを作成
        numerical_feature_names = [f.strip() for f in numerical_features_str.split(',') if f.strip()]
        categorical_feature_names = [f.strip() for f in categorical_features_str.split(',') if f.strip()]
        
        # 特徴量リストを保存
        self.numerical_features = numerical_feature_names
        self.categorical_features = categorical_feature_names
        
        # グループ情報を保持する辞書（特徴量名からグループ名へのマッピング）
        self.feature_groups = {
            'numerical': {},
            'categorical': {}
        }
        
        # グループ定義を取得
        if self.has_section('groups'):
            for group_name, features_str in self.items('groups'):
                # グループに含まれる特徴量のリストを作成
                group_features = [f.strip() for f in features_str.split(',') if f.strip()]
                
                # 各特徴量をそのグループにマッピング
                for feature in group_features:
                    if feature in numerical_feature_names:
                        self.feature_groups['numerical'][feature] = group_name
                    elif feature in categorical_feature_names:
                        self.feature_groups['categorical'][feature] = group_name
        
        # グループに属さない特徴量は自分自身がグループになる
        for feature in numerical_feature_names:
            if feature not in self.feature_groups['numerical']:
                self.feature_groups['numerical'][feature] = feature
                
        for feature in categorical_feature_names:
            if feature not in self.feature_groups['categorical']:
                self.feature_groups['categorical'][feature] = feature
        
        # グループごとの特徴量リストを作成（エンコーダー作成時の参考用）
        self.group_features = {
            'numerical': {},
            'categorical': {}
        }
        
        # 特徴量グループ -> 特徴量リストのマッピングを作成
        for feature_type in ['numerical', 'categorical']:
            for feature, group in self.feature_groups[feature_type].items():
                if group not in self.group_features[feature_type]:
                    self.group_features[feature_type][group] = []
                self.group_features[feature_type][group].append(feature)
        
        logger.info(f"Found {len(self.numerical_features)} numerical features and {len(self.categorical_features)} categorical features")
        logger.info(f"Configured {len(self.group_features['numerical'])} numerical groups and {len(self.group_features['categorical'])} categorical groups")
        return self
    

def load_config():
    """
    Load configuration from horguesc.ini file.
    
    Returns:
        Config: Configuration object with parsed features and tasks
    """
    config = Config()
    config_path = find_config()
    
    if config_path is None:
        logger.error("Configuration file not found in any of the expected locations")
        raise FileNotFoundError("Configuration file not found")
    
    try:
        config.read(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        config.parse_features()
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise