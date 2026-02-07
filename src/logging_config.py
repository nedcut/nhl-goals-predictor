"""
Logging configuration for NHL goals prediction pipeline.

Provides structured logging with configurable levels and optional file output.

Usage:
    from src.logging_config import get_logger, setup_logging

    # Setup logging at module level
    setup_logging(level="INFO")

    # Get a logger for your module
    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.warning("Missing data for game %s", game_id)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# Default format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Module-level flag to track if logging has been configured
_logging_configured = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: str = LOG_FORMAT,
) -> logging.Logger:
    """Configure logging for the NHL prediction pipeline.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : Path, optional
        If provided, also log to this file.
    format_string : str
        Log message format.

    Returns
    -------
    logging.Logger
        The root logger for the package.
    """
    global _logging_configured

    # Get the package root logger
    logger = logging.getLogger("nhl_predictor")
    logger_level = getattr(logging, level.upper())

    # If already configured, allow dynamic level updates (e.g. --verbose CLI flag).
    if _logging_configured:
        logger.setLevel(logger_level)
        for handler in logger.handlers:
            handler.setLevel(logger_level)
        return logger

    logger.setLevel(logger_level)

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=LOG_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logger_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    _logging_configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Parameters
    ----------
    name : str
        Usually __name__ of the calling module.

    Returns
    -------
    logging.Logger
        Logger instance for the module.
    """
    # Ensure logging is configured with defaults
    if not _logging_configured:
        setup_logging()

    # Create child logger under the package namespace
    if name.startswith("src."):
        name = name.replace("src.", "nhl_predictor.")
    elif not name.startswith("nhl_predictor."):
        name = f"nhl_predictor.{name}"

    return logging.getLogger(name)
