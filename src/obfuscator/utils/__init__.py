"""
Utility modules for cross-platform operations and logging.

This package provides:
- Path utilities for file system operations
- Platform detection utilities
- Logging infrastructure with file and console output

Examples:
    >>> from obfuscator.utils import normalize_path, setup_logger, get_platform
    >>> path = normalize_path("~/documents/script.lua")
    >>> logger = setup_logger("obfuscator")
    >>> platform = get_platform()
"""

from .path_utils import (
    # Type alias
    PathLike,
    # Path normalization and resolution
    normalize_path,
    resolve_path,
    # Directory operations
    ensure_directory,
    get_relative_path,
    # Path validation
    is_safe_path,
    validate_lua_file,
    is_readable,
    is_writable,
    # File operations
    get_file_extension,
    change_extension,
    ensure_extension,
    # Platform detection
    get_platform,
    is_windows,
    is_macos,
    is_linux,
    get_path_separator,
)

from .logger import (
    # Logger setup
    setup_logger,
    get_logger,
    set_log_level,
    # Handler management
    add_file_handler,
    add_console_handler,
    # Utilities
    get_log_directory,
    # Constants
    VALID_LOG_LEVELS,
)

__all__ = [
    # Type alias
    "PathLike",
    # Path utilities
    "normalize_path",
    "resolve_path",
    "ensure_directory",
    "get_relative_path",
    "is_safe_path",
    "validate_lua_file",
    "is_readable",
    "is_writable",
    "get_file_extension",
    "change_extension",
    "ensure_extension",
    # Platform detection
    "get_platform",
    "is_windows",
    "is_macos",
    "is_linux",
    "get_path_separator",
    # Logger
    "setup_logger",
    "get_logger",
    "set_log_level",
    "add_file_handler",
    "add_console_handler",
    "get_log_directory",
    "VALID_LOG_LEVELS",
]

