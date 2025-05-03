"""
Configuration utilities for horguesc.
"""
import configparser
import os
import logging
import sys

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

def load_config():
    """
    Load configuration from horguesc.ini file.
    
    Returns:
        configparser.ConfigParser: Loaded configuration
    """
    config = configparser.ConfigParser()
    config_path = find_config()
    
    if config_path is None:
        logger.error("Configuration file not found in any of the expected locations")
        raise FileNotFoundError("Configuration file not found")
    
    try:
        config.read(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise