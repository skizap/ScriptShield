"""
Logging infrastructure for the obfuscator application.

This module provides configurable logging with console and file output,
log rotation, and hierarchical logger management.

Examples:
    >>> from obfuscator.utils.logger import setup_logger
    >>> logger = setup_logger("obfuscator", level="DEBUG", log_file=Path("logs/app.log"))
    >>> logger.info("Application started")
    >>> logger.debug("Debug information")
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from .path_utils import ensure_directory

# Valid log levels
VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Default log format strings
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Path | None = None
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Creates a logger with console output and optional file output.
    Prevents duplicate handlers if called multiple times.

    Args:
        name: Logger name (typically module name or "obfuscator").
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file. If provided, enables file logging.

    Returns:
        Configured logger instance.

    Raises:
        ValueError: If level is not a valid log level.

    Examples:
        >>> logger = setup_logger("obfuscator", level="DEBUG")
        >>> logger = setup_logger("obfuscator.gui", log_file=Path("logs/gui.log"))

    Note:
        File logs use detailed format, console logs use simple format.
        Log files are rotated at 10MB with 5 backup files.
    """
    if level.upper() not in VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log level: {level}. Must be one of {VALID_LOG_LEVELS}")

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Check for existing handlers to avoid duplicates
    has_console_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers)
    has_file_handler = any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers)

    # Add console handler if not present
    if not has_console_handler:
        add_console_handler(logger, level)

    # Add file handler if log_file is provided and not already present
    if log_file is not None and not has_file_handler:
        add_file_handler(logger, log_file, level="DEBUG")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve existing logger or create default logger.

    Args:
        name: Logger name.

    Returns:
        Logger instance.

    Examples:
        >>> logger = get_logger("obfuscator.utils")
        >>> logger.info("Message")

    Note:
        If logger doesn't exist and no parent handlers are available,
        creates one with INFO level and console output.
        Child loggers inherit handlers from parent via propagation.
    """
    logger = logging.getLogger(name)

    # Check if this logger or any parent has handlers configured
    # This prevents duplicate console handlers when parent is already set up
    if logger.hasHandlers():
        return logger

    # Check parent loggers for handlers (propagation will use them)
    parent_has_handlers = False
    parent_name = name.rsplit(".", 1)[0] if "." in name else ""
    while parent_name:
        parent_logger = logging.getLogger(parent_name)
        if parent_logger.handlers:
            parent_has_handlers = True
            break
        parent_name = parent_name.rsplit(".", 1)[0] if "." in parent_name else ""

    # Also check root logger
    if not parent_has_handlers and logging.getLogger().handlers:
        parent_has_handlers = True

    if parent_has_handlers:
        # Parent has handlers, just return the logger (propagation handles output)
        return logger

    # No handlers anywhere, create default logger
    return setup_logger(name)


def set_log_level(logger: logging.Logger, level: str) -> None:
    """
    Change logging level dynamically.
    
    Args:
        logger: Logger instance to modify.
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    
    Raises:
        ValueError: If level is not valid.
    
    Examples:
        >>> logger = get_logger("obfuscator")
        >>> set_log_level(logger, "DEBUG")
    """
    if level.upper() not in VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log level: {level}. Must be one of {VALID_LOG_LEVELS}")
    
    logger.setLevel(getattr(logging, level.upper()))


def add_file_handler(
    logger: logging.Logger,
    log_file: Path,
    level: str = "DEBUG"
) -> None:
    """
    Add file output handler to logger with rotation.
    
    Creates log directory if it doesn't exist. Uses rotating file handler
    with 10MB max size and 5 backup files.
    
    Args:
        logger: Logger instance to modify.
        log_file: Path to log file.
        level: Logging level for file handler.
    
    Raises:
        ValueError: If level is not valid.
        OSError: If log directory cannot be created.
    
    Examples:
        >>> logger = get_logger("obfuscator")
        >>> add_file_handler(logger, Path("logs/app.log"))
    """
    if level.upper() not in VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log level: {level}. Must be one of {VALID_LOG_LEVELS}")

    # Ensure log directory exists
    ensure_directory(log_file.parent)

    # Create rotating file handler (10MB max, 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, level.upper()))

    # Use detailed format for file logs
    formatter = logging.Formatter(DETAILED_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)


def add_console_handler(logger: logging.Logger, level: str = "INFO") -> None:
    """
    Add console output handler to logger.

    Args:
        logger: Logger instance to modify.
        level: Logging level for console handler.

    Raises:
        ValueError: If level is not valid.

    Examples:
        >>> logger = get_logger("obfuscator")
        >>> add_console_handler(logger, "WARNING")
    """
    if level.upper() not in VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log level: {level}. Must be one of {VALID_LOG_LEVELS}")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))

    # Use simple format for console logs
    formatter = logging.Formatter(SIMPLE_FORMAT)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)


def get_log_directory() -> Path:
    """
    Return the default logs directory path.

    Returns:
        Path to logs directory (relative to project root).

    Examples:
        >>> get_log_directory()
        PosixPath('logs')

    Note:
        This returns the path but does not create the directory.
        Use ensure_directory() to create it.
    """
    return Path("logs")

