"""
Command line interface for the horguesc package.
"""
import argparse
import sys
from . import core

def main():
    """Entry point for the horguesc command line application."""
    parser = argparse.ArgumentParser(description='horguesc CLI')
    parser.add_argument('--version', action='store_true', help='show version')

    subparsers = parser.add_subparsers(dest='command', help='sub-command help')

    dataloader_parser = subparsers.add_parser('dataloader', help='Load data using C++ executable')
    dataloader_parser.add_argument('--exe', type=str, required=True, help='Path to the C++ executable')

    args = parser.parse_args()
    

    if args.version:
        from . import __version__
        print(f"horguesc version {__version__}")
        return 0
    

    if args.command == 'dataloader':
        try:
            dataset = core.load_dataset(args.exe)
            print(dataset)

        except Exception as e:
            print(f"Error loading dataset: {e}", file=sys.stderr)
            return 1
        
        return 0

    # Default behavior if no command specified
    print(core.hello())
    return 0

if __name__ == "__main__":
    main()