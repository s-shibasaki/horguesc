"""
Command line interface for the horguesc package.
"""

import argparse
import sys


def main():
    """Entry point for the horguesc command line application."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", action="store_true", help="show version information"
    )
    args = parser.parse_args()

    if args.version:
        from . import __version__

        print(f"horguesc version {__version__}")
        return 0

    print("Welcome to horguesc!")
    return 0


if __name__ == "__main__":
    main()
