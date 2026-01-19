import logging
import os
import sys

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s"


def configure_logging(
    level=logging.INFO,
    fmt=DEFAULT_FORMAT,
    stream=True,
    filepath=None,
    logger_name="hydrolib.dhydamo",
):
    """Configure logging for hydrolib.dhydamo without adding duplicate handlers."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = logging.Formatter(fmt)

    if stream and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if filepath:
        directory = os.path.dirname(filepath)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        if not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == filepath
            for h in logger.handlers
        ):
            file_handler = logging.FileHandler(filepath)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
