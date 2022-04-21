import argparse
import logging
from typing import Union


def setup_basic_logging(
    level: Union[int, str] = logging.WARNING,
    filename: str = None,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_to_console: bool = True,
):
    handlers = []
    if filename:
        handlers.append(logging.FileHandler(filename=filename))
    if log_to_console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(level=level, format=format, handlers=handlers)


def add_logging_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--loglevel",
        help="Log level (default: WARNING)",
        default="WARNING",
    )
    parser.add_argument(
        "--logfile",
        help="Log file",
    )
