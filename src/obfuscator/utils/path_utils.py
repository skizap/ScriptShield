"""
Path utilities for cross-platform file system operations.

This module provides utilities for path normalization, validation, file operations,
and platform detection. All functions use pathlib.Path for cross-platform compatibility.

Examples:
    >>> from obfuscator.utils.path_utils import normalize_path, get_platform
    >>> path = normalize_path("~/documents/script.lua")
    >>> platform = get_platform()
    >>> print(f"Running on {platform}")
"""

from __future__ import annotations

import os
import platform as platform_module
from pathlib import Path
from typing import Union

# Type alias for path-like objects
PathLike = Union[str, Path]

# Cache platform detection result
_PLATFORM_CACHE: str | None = None


def normalize_path(path: PathLike) -> Path:
    """
    Convert a string or Path object to a normalized absolute Path.
    
    This function handles path expansion (~), resolves relative paths,
    and normalizes path separators for the current platform.
    
    Args:
        path: A file system path as string or Path object.
    
    Returns:
        Normalized absolute Path object.
    
    Raises:
        ValueError: If path is empty or None.
    
    Examples:
        >>> normalize_path("~/documents/file.lua")
        PosixPath('/home/user/documents/file.lua')
        
        >>> normalize_path("../relative/path")
        PosixPath('/absolute/resolved/path')
    
    Note:
        On Windows, this handles both forward and backward slashes.
    """
    if path is None or (isinstance(path, str) and not path.strip()):
        raise ValueError("Path cannot be None or empty")
    
    path_obj = Path(path).expanduser().resolve()
    return path_obj


def ensure_directory(path: Path) -> Path:
    """
    Create directory if it doesn't exist, return Path.
    
    Args:
        path: Directory path to create.
    
    Returns:
        The directory Path object.
    
    Raises:
        OSError: If directory creation fails.
    
    Examples:
        >>> ensure_directory(Path("logs"))
        PosixPath('/project/logs')
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_relative_path(path: Path, base: Path) -> Path:
    """
    Get relative path from base directory.
    
    Args:
        path: The target path.
        base: The base directory.
    
    Returns:
        Relative path from base to path.
    
    Raises:
        ValueError: If paths are on different drives (Windows).
    
    Examples:
        >>> get_relative_path(Path("/home/user/docs/file.txt"), Path("/home/user"))
        PosixPath('docs/file.txt')
    """
    return path.relative_to(base)


def is_safe_path(path: Path, base: Path) -> bool:
    """
    Validate path doesn't escape base directory (security check).
    
    This prevents path traversal attacks by ensuring the resolved path
    is within the base directory.
    
    Args:
        path: Path to validate.
        base: Base directory that should contain the path.
    
    Returns:
        True if path is safe (within base), False otherwise.
    
    Examples:
        >>> is_safe_path(Path("/home/user/docs/file.txt"), Path("/home/user"))
        True
        >>> is_safe_path(Path("/etc/passwd"), Path("/home/user"))
        False
    """
    try:
        resolved_path = path.resolve()
        resolved_base = base.resolve()
        resolved_path.relative_to(resolved_base)
        return True
    except (ValueError, OSError):
        return False


def resolve_path(path: PathLike, base: Path | None = None) -> Path:
    """
    Resolve path relative to base or current working directory.

    Args:
        path: Path to resolve.
        base: Base directory for relative paths. If None, uses cwd.

    Returns:
        Resolved absolute Path object.

    Examples:
        >>> resolve_path("file.lua", Path("/project"))
        PosixPath('/project/file.lua')
    """
    path_obj = Path(path)
    if base is not None and not path_obj.is_absolute():
        return (base / path_obj).resolve()
    return path_obj.expanduser().resolve()


def get_file_extension(path: Path) -> str:
    """
    Extract file extension including the dot.

    Args:
        path: File path.

    Returns:
        File extension (e.g., ".lua", ".txt"). Empty string if no extension.

    Examples:
        >>> get_file_extension(Path("script.lua"))
        '.lua'
        >>> get_file_extension(Path("README"))
        ''
    """
    return path.suffix


def change_extension(path: Path, new_ext: str) -> Path:
    """
    Replace file extension with a new one.

    Args:
        path: Original file path.
        new_ext: New extension (with or without leading dot).

    Returns:
        Path with new extension.

    Examples:
        >>> change_extension(Path("script.lua"), ".txt")
        PosixPath('script.txt')
        >>> change_extension(Path("script.lua"), "py")
        PosixPath('script.py')
    """
    if not new_ext.startswith('.'):
        new_ext = '.' + new_ext
    return path.with_suffix(new_ext)


def ensure_extension(path: Path, ext: str) -> Path:
    """
    Add extension if missing.

    Args:
        path: File path.
        ext: Extension to ensure (with or without leading dot).

    Returns:
        Path with extension.

    Examples:
        >>> ensure_extension(Path("script"), ".lua")
        PosixPath('script.lua')
        >>> ensure_extension(Path("script.lua"), ".lua")
        PosixPath('script.lua')
    """
    if not ext.startswith('.'):
        ext = '.' + ext
    if path.suffix.lower() == ext.lower():
        return path
    return path.with_suffix(ext)


def validate_lua_file(path: Path) -> bool:
    """
    Check if path points to a valid .lua file.

    Args:
        path: Path to validate.

    Returns:
        True if path exists, is a file, and has .lua extension.

    Examples:
        >>> validate_lua_file(Path("script.lua"))
        True
        >>> validate_lua_file(Path("script.txt"))
        False
    """
    return path.exists() and path.is_file() and path.suffix.lower() == '.lua'


def is_readable(path: Path) -> bool:
    """
    Check if path has read permissions.

    Args:
        path: Path to check.

    Returns:
        True if path exists and is readable.

    Examples:
        >>> is_readable(Path("script.lua"))
        True
    """
    return path.exists() and os.access(path, os.R_OK)


def is_writable(path: Path) -> bool:
    """
    Check if path has write permissions.

    Args:
        path: Path to check.

    Returns:
        True if path exists and is writable.

    Examples:
        >>> is_writable(Path("script.lua"))
        True
    """
    return path.exists() and os.access(path, os.W_OK)


def get_platform() -> str:
    """
    Return current platform name.

    Returns:
        Platform name: "windows", "macos", or "linux".

    Examples:
        >>> get_platform()
        'linux'

    Note:
        Result is cached for performance.
    """
    global _PLATFORM_CACHE
    if _PLATFORM_CACHE is not None:
        return _PLATFORM_CACHE

    system = platform_module.system()
    if system == "Windows":
        _PLATFORM_CACHE = "windows"
    elif system == "Darwin":
        _PLATFORM_CACHE = "macos"
    elif system == "Linux":
        _PLATFORM_CACHE = "linux"
    else:
        _PLATFORM_CACHE = system.lower()

    return _PLATFORM_CACHE


def is_windows() -> bool:
    """
    Check if running on Windows.

    Returns:
        True if platform is Windows.

    Examples:
        >>> is_windows()
        False
    """
    return get_platform() == "windows"


def is_macos() -> bool:
    """
    Check if running on macOS.

    Returns:
        True if platform is macOS.

    Examples:
        >>> is_macos()
        False
    """
    return get_platform() == "macos"


def is_linux() -> bool:
    """
    Check if running on Linux.

    Returns:
        True if platform is Linux.

    Examples:
        >>> is_linux()
        True
    """
    return get_platform() == "linux"


def get_path_separator() -> str:
    """
    Return OS-specific path separator.

    Returns:
        Path separator: '\\' on Windows, '/' on Unix-like systems.

    Examples:
        >>> get_path_separator()
        '/'

    Note:
        This returns os.sep for the current platform.
    """
    return os.sep

