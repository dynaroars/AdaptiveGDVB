"""
"""
import argparse
import logging
import os
import sys

from contextlib import contextmanager
from functools import partial
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL


def add_arguments(parser: argparse.ArgumentParser):
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    verbosity_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show messages with finer-grained information",
    )
    verbosity_group.add_argument(
        "-q", "--quiet", action="store_true", help="suppress non-essential messages"
    )


@contextmanager
def suppress(level=logging.DEBUG, filter_level=logging.WARNING):
    if level >= filter_level:
        yield
        return
    with open(os.dup(sys.stdout.fileno()), "wb") as stdout_copy:
        with open(os.dup(sys.stderr.fileno()), "wb") as stderr_copy:
            sys.stdout.flush()
            sys.stderr.flush()
            with open(os.devnull, "wb") as devnull:
                os.dup2(devnull.fileno(), sys.stdout.fileno())
                os.dup2(devnull.fileno(), sys.stderr.fileno())
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(stdout_copy.fileno(), sys.stdout.fileno())
                os.dup2(stderr_copy.fileno(), sys.stderr.fileno())


def initialize(name: str, args: argparse.Namespace):
    global suppress
    logger = logging.getLogger(name)

    if args.debug:
        logger.setLevel(logging.DEBUG)
        suppress = partial(suppress, filter_level=logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
        suppress = partial(suppress, filter_level=logging.INFO)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
        suppress = partial(suppress, filter_level=logging.ERROR)

    formatter = logging.Formatter(f"%(levelname)-8s %(asctime)s (%(name)s) %(message)s")

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def getLogger(name: str):
    return logging.getLogger(name)
