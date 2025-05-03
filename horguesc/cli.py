"""
Command line interface for the horguesc package.
"""

import argparse
import logging
import sys

# アプリケーション全体のログ設定をここで行う
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    """Entry point for the horguesc command line application."""
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", action="store_true", help="show version information"
    )
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")
    train_parser = subparsers.add_parser('train', help='train a model')
    test_parser = subparsers.add_parser('test', help='test a trained model')
    predict_parser = subparsers.add_parser('predict', help='make predictions with a trained model')
    args = parser.parse_args()

    if args.version:
        from . import __version__

        print(f"horguesc version {__version__}")
        return 0
    
    if args.command == 'train':
        from .commands import train
        return train.run(args)
    
    elif args.command == 'test':
        from .commands import test
        return test.run(args)
    
    elif args.command == 'predict':
        from .commands import predict
        return predict.run(args)
    
    else:
        print("Welcome to horguesc!")
        print("\nPlease specify a command. Use --help for more information.")
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    main()
