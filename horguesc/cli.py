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
    train_parser.add_argument('--save-each-epoch', action='store_true', 
                             help='Save model after each training epoch')
    train_parser.add_argument('--skip-validation', action='store_true',
                             help='Skip validation during training')
    
    # 学習期間を設定するオプションを追加
    train_parser.add_argument('--train-start', type=str, 
                             help='Training data start date (YYYY-MM-DD)')
    train_parser.add_argument('--train-end', type=str,
                             help='Training data end date (YYYY-MM-DD)')
    train_parser.add_argument('--val-start', type=str,
                             help='Validation data start date (YYYY-MM-DD)')
    train_parser.add_argument('--val-end', type=str,
                             help='Validation data end date (YYYY-MM-DD)')
    
    test_parser = subparsers.add_parser('test', help='test a trained model')
    predict_parser = subparsers.add_parser('predict', help='make predictions with a trained model')
    
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

if __name__ == "__main__":
    main()
