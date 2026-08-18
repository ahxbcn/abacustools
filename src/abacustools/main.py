"""Command line interface for abacustools."""

import argparse

from abacustools import __version__


def main() -> None:
    """Entry point of the abacustools command line interface."""
    parser = argparse.ArgumentParser(prog="abacustools", description="Tools for accompanying using ABACUS")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args()


if __name__ == "__main__":
    main()