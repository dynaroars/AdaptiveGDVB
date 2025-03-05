import sys
import logging

logging.basicConfig(
    stream=sys.stdout,
    format="[%(levelname)s] %(asctime)s %(message)s ",
    datefmt="%m/%d/%Y %I:%M:%S %p :",
)


def initialize(args):
    if args.debug:
        logging_level = logging.DEBUG
    elif args.dumb:
        logging_level = logging.WARN
    else:
        logging_level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(level=logging_level)
    return logger
