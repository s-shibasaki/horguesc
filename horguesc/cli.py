"""
Command line interface for the horguesc package.
"""

import argparse
import logging
import sys
from horguesc.utils.config import load_config

# アプリケーション全体のログ設定をここで行う
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def main():
    """Entry point for the horguesc command line application."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", action="store_true", help="show version information"
    )
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")
    
    # Train command parser with all options
    train_parser = subparsers.add_parser('train', help='train a model')
    # Add train-specific arguments
    _add_common_arguments(train_parser)
    train_parser.add_argument('--training.train_start_date', help='Training data start date (YYYY-MM-DD)')
    train_parser.add_argument('--training.train_end_date', help='Training data end date (YYYY-MM-DD)')
    train_parser.add_argument('--training.val_start_date', help='Validation data start date (YYYY-MM-DD)')
    train_parser.add_argument('--training.val_end_date', help='Validation data end date (YYYY-MM-DD)')
    train_parser.add_argument('--training.save_each_epoch', help='Save model after each epoch (true/false)')
    train_parser.add_argument('--training.skip_validation', help='Skip validation (true/false)')
    
    # Test command parser
    test_parser = subparsers.add_parser('test', help='test a trained model')
    _add_common_arguments(test_parser)
    
    # Predict command parser
    predict_parser = subparsers.add_parser('predict', help='make predictions with a trained model')
    _add_common_arguments(predict_parser)
    
    # Export dataset command parser
    export_parser = subparsers.add_parser('export', help='export dataset samples to Excel for verification')
    _add_common_arguments(export_parser)
    export_parser.add_argument('--export.mode', choices=['train', 'eval', 'inference'], help='Dataset mode to export')
    export_parser.add_argument('--export.start_date', help='Start date for dataset (YYYY-MM-DD)')
    export_parser.add_argument('--export.end_date', help='End date for dataset (YYYY-MM-DD)')
    export_parser.add_argument('--export.sample_size', type=int, help='Number of samples to export')
    export_parser.add_argument('--export.output_dir', help='Directory to save Excel files')
    
    args = parser.parse_args()

    if args.version:
        from . import __version__
        print(f"horguesc version {__version__}")
        return 0
    
    # Command handling
    if args.command:
        try:
            # Load, update and validate config in one step
            config = load_config(args)
            
            if args.command == 'train':
                from .commands import train
                return train.run(config)
            
            elif args.command == 'test':
                from .commands import test
                return test.run(config)
            
            elif args.command == 'predict':
                from .commands import predict
                return predict.run(config)
                
            elif args.command == 'export':
                from .commands import export
                return export.run(config)
                
        except ValueError as e:
            logger.error(str(e))
            return 1
            
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1
    
    else:
        print("Welcome to horguesc!")
        print("\nPlease specify a command. Use --help for more information.")
        parser.print_help()
        return 1

    return 0

def _add_common_arguments(parser):
    """Add common command line arguments to a parser."""
    # 既存の引数
    parser.add_argument('--database.host', help='Database host')
    parser.add_argument('--database.port', type=int, help='Database port')
    parser.add_argument('--database.username', help='Database username')
    parser.add_argument('--database.password', help='Database password')
    
    # モデルディレクトリのコマンドライン引数を追加
    parser.add_argument('--paths.model_dir', help='Directory to save/load models')
