import logging
import sys
import os
FMT = "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s"

def initialize_logger(name='delft3dfmpy', path=None, log_level=20, fmt=FMT):
    """Set-up the logging on sys.stdout"""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console)
    if path is not None:
        add_filehandler(logger, path, log_level=log_level, fmt=fmt)
    return logger


def add_filehandler(logger, path, log_level=20, fmt=FMT):
    """Add file handler to logger."""
    if os.path.dirname(path) != "" and not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    isfile = os.path.isfile(path)
    if isfile:
        os.remove(path)
    ch = logging.FileHandler(path)
    ch.setFormatter(logging.Formatter(fmt))
    ch.setLevel(log_level)
    logger.addHandler(ch)
    if isfile:
        logger.debug(f"Overwriting log messages in file {path}.")
    else:
        logger.debug(f"Writing log messages to new file {path}.")

class ProgressLogger:

    def __init__(self, logger, total, step):
        self.logger = logger
        self.total = total
        self.lastp = -1
        self.step = step

    def set_step(self, i):
        percentage = int(round(((i+1) / (self.total)) * 100))
        if percentage % self.step == 0:
            if self.lastp == percentage:
                return None
            self.lastp = percentage
            self.logger.info(f'Processing raster: {percentage:3d} %')





