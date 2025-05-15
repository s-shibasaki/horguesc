"""
Configuration utilities for horguesc.
"""
import configparser
import os
import logging
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import argparse

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
        
        # Initialize feature and group properties
        self.numerical_features = []
        self.categorical_features = []
        self.feature_groups = {'numerical': {}, 'categorical': {}}
        self.group_features = {'numerical': {}, 'categorical': {}}

    def load_from_file(self, config_path=None):
        """
        Load configuration from a file path or try to find it automatically.
        
        Args:
            config_path: Optional specific path to configuration file
            
        Returns:
            self: For method chaining
            
        Raises:
            FileNotFoundError: If configuration file cannot be found
        """
        if not config_path:
            config_path = find_config()
        
        if not config_path or not os.path.exists(config_path):
            logger.error("Configuration file not found in any of the expected locations")
            raise FileNotFoundError("Configuration file not found")
        
        try:
            self.read(config_path)
            logger.info(f"Configuration loaded from {config_path}")
            return self
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def parse_features(self):
        """
        Parse feature definitions from the configuration.
        
        Returns:
            self: For method chaining
        """
        # 特徴量リストを取得
        numerical_features_str = self.get('features', 'numerical_features', fallback='').strip()
        categorical_features_str = self.get('features', 'categorical_features', fallback='').strip()
        
        # 数値特徴量とカテゴリカル特徴量のリストを作成
        self.numerical_features = [f.strip() for f in numerical_features_str.split(',') if f.strip()]
        self.categorical_features = [f.strip() for f in categorical_features_str.split(',') if f.strip()]
        
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
                    if feature in self.numerical_features:
                        self.feature_groups['numerical'][feature] = group_name
                    elif feature in self.categorical_features:
                        self.feature_groups['categorical'][feature] = group_name
        
        # グループに属さない特徴量は自分自身がグループになる
        for feature in self.numerical_features:
            if feature not in self.feature_groups['numerical']:
                self.feature_groups['numerical'][feature] = feature
                
        for feature in self.categorical_features:
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
    
    def update_from_args(self, args):
        """
        Update configuration based on command line arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            self: For method chaining
        """
        # 設定をコマンドライン引数で更新
        if hasattr(args, 'save_each_epoch'):
            self.set('training', 'save_each_epoch', str(args.save_each_epoch).lower())
            logger.info(f"save_each_epoch設定を上書き: {args.save_each_epoch}")
            
        # スキップ検証オプションの処理
        if hasattr(args, 'skip_validation'):
            self.set('training', 'skip_validation', str(args.skip_validation).lower())
            logger.info(f"skip_validation設定を上書き: {args.skip_validation}")
        
        # 学習期間の設定を上書き
        if hasattr(args, 'train_start') and args.train_start:
            self.set('training', 'train_start_date', args.train_start)
            logger.info(f"train_start_date設定を上書き: {args.train_start}")
            
        if hasattr(args, 'train_end') and args.train_end:
            self.set('training', 'train_end_date', args.train_end)
            logger.info(f"train_end_date設定を上書き: {args.train_end}")
            
        if hasattr(args, 'val_start') and args.val_start:
            self.set('training', 'val_start_date', args.val_start)
            logger.info(f"val_start_date設定を上書き: {args.val_start}")
            
        if hasattr(args, 'val_end') and args.val_end:
            self.set('training', 'val_end_date', args.val_end)
            logger.info(f"val_end_date設定を上書き: {args.val_end}")
            
        return self
    
    def validate(self):
        """
        Validate all configuration including dates.
        
        Returns:
            self: For method chaining
            
        Raises:
            ValueError: If any validation fails
        """
        self._validate_dates()
        # Add other validation methods as needed
        return self
    
    def _validate_dates(self):
        """
        Validate date formats in the configuration.
        
        Raises:
            ValueError: If any date has invalid format
        """
        # 日付フィールドの検証
        date_fields = [
            ('training', 'train_start_date'),
            ('training', 'train_end_date'),
            ('training', 'val_start_date'),
            ('training', 'val_end_date')
        ]
        
        invalid_dates = []
        
        for section, option in date_fields:
            try:
                date_str = self.get(section, option, fallback=None)
                if date_str:
                    datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError as e:
                error_msg = f"日付形式が不正です: {section}.{option}={date_str}. {e}"
                logger.error(error_msg)
                invalid_dates.append((section, option, date_str, str(e)))
        
        if invalid_dates:
            details = "\n".join([f"- {section}.{option}={value}: {error}" 
                               for section, option, value, error in invalid_dates])
            raise ValueError(f"設定に含まれる日付の形式が正しくありません:\n{details}")


def load_config(args=None):
    """
    Load, parse and validate configuration.
    
    Args:
        args: Optional parsed command line arguments to update config with
        
    Returns:
        Config: Fully initialized configuration object
        
    Raises:
        FileNotFoundError: If configuration file cannot be found
        ValueError: If configuration validation fails
    """
    config = Config()
    
    try:
        # Load from file, parse features and optionally update from args
        config.load_from_file().parse_features()
        
        if args:
            config.update_from_args(args)
            
        # Validate configuration
        config.validate()
        
        return config
        
    except Exception as e:
        logger.error(f"Error in configuration setup: {e}")
        raise