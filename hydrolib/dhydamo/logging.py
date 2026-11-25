"""Logging setup for hydrolib.dhydamo."""

import logging
import os
import sys
from pathlib import Path
from typing import ClassVar, Literal

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s"


class ColorFormatter(logging.Formatter):
    """Formatter that colors each message by its level, with ANSI escape codes."""

    # INFO keeps the default color of the terminal
    colors: ClassVar[dict[int, str]] = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    reset = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the record, colored by its level; INFO and unknown levels are not colored."""
        message = super().format(record)
        if record.levelno not in self.colors:
            return message
        return f"{self.colors[record.levelno]}{message}{self.reset}"


def configure_logging(
    level: int | str = logging.INFO,
    fmt: str = DEFAULT_FORMAT,
    stream: bool = True,
    filepath: str | Path | None = None,
    logger_name: str = "hydrolib.dhydamo",
    color: bool = False,
    mode: Literal["a", "w"] = "a",
) -> logging.Logger:
    """Configure logging for hydrolib.dhydamo without adding duplicate handlers.

    Calling it again with the same stream or file does not add another handler, so a notebook
    cell can be run more than once.

    Parameters
    ----------
    level : int or str, optional
        Lowest level that is logged, e.g. ``logging.DEBUG`` or ``"WARNING"``. Defaults to
        ``logging.INFO``.
    fmt : str, optional
        Format of each message, see ``logging.Formatter``. Defaults to ``DEFAULT_FORMAT``.
    stream : bool, optional
        Log to the screen (``sys.stdout``). Defaults to True. Messages are then no longer passed
        on to the root logger, so they do not appear twice.
    filepath : str or Path, optional
        Log file, in UTF-8. Missing folders are created. Defaults to None: no log file.
    logger_name : str, optional
        Logger to configure. Defaults to ``"hydrolib.dhydamo"``.
    color : bool, optional
        Color the messages on the screen by level, e.g. warnings yellow and errors red. Needs a
        terminal or notebook that shows ANSI colors. The log file is never colored. Defaults to
        False.
    mode : {"a", "w"}, optional
        ``"a"`` adds to an existing log file, ``"w"`` starts with an empty one. Defaults to
        ``"a"``.

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = logging.Formatter(fmt)

    has_stream_handler = False
    for h in logger.handlers:
        # a FileHandler is also a StreamHandler, so leave those out when looking for a screen handler
        if isinstance(h, logging.FileHandler):
            continue
        if isinstance(h, logging.StreamHandler):
            has_stream_handler = True
            break

    if stream and not has_stream_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        if color:
            handler.setFormatter(ColorFormatter(fmt))
        else:
            handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    if filepath:
        directory = os.path.dirname(filepath)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        # FileHandler stores the absolute path
        path = os.path.abspath(filepath)
        if not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == path
            for h in logger.handlers
        ):
            file_handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
