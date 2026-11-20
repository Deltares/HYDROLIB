import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from hydrolib.dhydamo.logging import ColorFormatter, configure_logging


@pytest.fixture
def logger_name(request: pytest.FixtureRequest) -> Iterator[str]:
    """Yield a logger name of its own for each test; its handlers are closed and removed afterwards."""
    name = f"hydrolib.dhydamo.test.{request.node.name}"
    yield name
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _handler_types(logger: logging.Logger) -> list[str]:
    return sorted(type(handler).__name__ for handler in logger.handlers)


def test_calling_twice_with_a_relative_path_adds_no_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, logger_name: str
) -> None:
    # the file handler stores the absolute path; a relative path used to add a second file handler
    monkeypatch.chdir(tmp_path)
    configure_logging(filepath="example.log", logger_name=logger_name)
    logger = configure_logging(filepath="example.log", logger_name=logger_name)

    assert _handler_types(logger) == ["FileHandler", "StreamHandler"]


def test_stream_handler_is_added_after_a_file_handler(tmp_path: Path, logger_name: str) -> None:
    # a FileHandler is also a StreamHandler; it used to prevent the screen handler
    configure_logging(stream=False, filepath=tmp_path / "example.log", logger_name=logger_name)
    logger = configure_logging(logger_name=logger_name)

    assert _handler_types(logger) == ["FileHandler", "StreamHandler"]
    assert logger.propagate is False


def test_log_file_is_utf8_and_mode_w_starts_empty(tmp_path: Path, logger_name: str) -> None:
    path = tmp_path / "logs" / "example.log"
    logger = configure_logging(stream=False, filepath=path, logger_name=logger_name, fmt="%(message)s")
    logger.info("first run: 5 m³/s")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    logger = configure_logging(stream=False, filepath=path, logger_name=logger_name, fmt="%(message)s", mode="w")
    logger.info("second run: Δh = 0.1 m")
    logger.handlers[0].flush()

    assert path.read_text(encoding="utf-8") == "second run: Δh = 0.1 m\n"


def test_color_formatter_colors_by_level() -> None:
    formatter = ColorFormatter("%(levelname)s %(message)s")

    def format_level(level: int) -> str:
        return formatter.format(logging.LogRecord("test", level, __file__, 1, "message", None, None))

    assert format_level(logging.INFO) == "INFO message"
    assert format_level(logging.WARNING) == "\033[33mWARNING message\033[0m"
    assert format_level(logging.ERROR) == "\033[31mERROR message\033[0m"


def test_color_only_on_the_screen(tmp_path: Path, logger_name: str) -> None:
    logger = configure_logging(filepath=tmp_path / "example.log", logger_name=logger_name, color=True)

    formatters = {type(handler).__name__: type(handler.formatter).__name__ for handler in logger.handlers}
    assert formatters == {"StreamHandler": "ColorFormatter", "FileHandler": "Formatter"}
