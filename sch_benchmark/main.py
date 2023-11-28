import argparse
from .operate import init, split, infer, merge, draw


def main():
    # support these subparsers:
    #   - init
    #   - split
    #   - infer
    #   - merge
    #   - show
    #   - draw

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("-d", "--dataset", type=str, required=True)
    init_parser.add_argument("-o", "--output", type=str, required=True)
    init_parser.set_defaults(func=init)