"""
Comprehensive tests for path_utils module.

Tests cover path normalization, validation, file operations,
and platform detection across different operating systems.
"""

import os
import platform
import pytest
from pathlib import Path
from unittest.mock import patch

from obfuscator.utils.path_utils import (
    normalize_path,
    ensure_directory,
    get_relative_path,
    is_safe_path,
    resolve_path,
    get_file_extension,
    change_extension,
    ensure_extension,
    validate_lua_file,
    is_readable,
    is_writable,
    get_platform,
    is_windows,
    is_macos,
    is_linux,
    get_path_separator,
)


class TestPathNormalization:
    """Test path normalization and resolution functions."""
    
    def test_normalize_path_converts_string_to_path(self):
        """Test that normalize_path converts string to Path object."""
        result = normalize_path("test.lua")
        assert isinstance(result, Path)
        assert result.is_absolute()
    
    def test_normalize_path_handles_path_object(self):
        """Test that normalize_path accepts Path objects."""
        path = Path("test.lua")
        result = normalize_path(path)
        assert isinstance(result, Path)
        assert result.is_absolute()
    
    def test_normalize_path_expands_home_directory(self, tmp_path):
        """Test that ~ is expanded to home directory."""
        result = normalize_path("~/test.lua")
        assert "~" not in str(result)
        assert result.is_absolute()
    
    def test_normalize_path_resolves_relative_paths(self):
        """Test that relative paths are resolved to absolute."""
        result = normalize_path("./test.lua")
        assert result.is_absolute()
    
    def test_normalize_path_raises_on_none(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="Path cannot be None or empty"):
            normalize_path(None)
    
    def test_normalize_path_raises_on_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Path cannot be None or empty"):
            normalize_path("")
    
    def test_normalize_path_raises_on_whitespace(self):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Path cannot be None or empty"):
            normalize_path("   ")
    
    def test_resolve_path_with_base(self, tmp_path):
        """Test resolve_path with base directory."""
        base = tmp_path
        result = resolve_path("test.lua", base)
        assert result == base / "test.lua"
    
    def test_resolve_path_without_base(self):
        """Test resolve_path without base uses cwd."""
        result = resolve_path("test.lua")
        assert result.is_absolute()
    
    def test_resolve_path_absolute_ignores_base(self, tmp_path):
        """Test that absolute paths ignore base parameter."""
        abs_path = tmp_path / "test.lua"
        result = resolve_path(abs_path, Path("/other/base"))
        assert result == abs_path


class TestDirectoryOperations:
    """Test directory creation and path operations."""
    
    def test_ensure_directory_creates_new_directory(self, tmp_path):
        """Test that ensure_directory creates new directory."""
        new_dir = tmp_path / "new_directory"
        assert not new_dir.exists()
        
        result = ensure_directory(new_dir)
        
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir
    
    def test_ensure_directory_creates_nested_directories(self, tmp_path):
        """Test that ensure_directory creates parent directories."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        assert not nested_dir.exists()
        
        result = ensure_directory(nested_dir)
        
        assert nested_dir.exists()
        assert result == nested_dir
    
    def test_ensure_directory_succeeds_if_exists(self, tmp_path):
        """Test that ensure_directory succeeds if directory exists."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        
        result = ensure_directory(existing_dir)
        
        assert result == existing_dir
    
    def test_get_relative_path(self, tmp_path):
        """Test get_relative_path calculation."""
        base = tmp_path
        target = tmp_path / "subdir" / "file.lua"
        
        result = get_relative_path(target, base)
        
        assert result == Path("subdir") / "file.lua"
    
    def test_get_relative_path_same_directory(self, tmp_path):
        """Test get_relative_path when paths are in same directory."""
        base = tmp_path
        target = tmp_path / "file.lua"
        
        result = get_relative_path(target, base)
        
        assert result == Path("file.lua")


class TestPathValidation:
    """Test path validation and security functions."""
    
    def test_is_safe_path_within_base(self, tmp_path):
        """Test is_safe_path returns True for path within base."""
        base = tmp_path
        safe_path = tmp_path / "subdir" / "file.lua"

        assert is_safe_path(safe_path, base) is True

    def test_is_safe_path_outside_base(self, tmp_path):
        """Test is_safe_path returns False for path outside base."""
        base = tmp_path / "restricted"
        unsafe_path = tmp_path / "other" / "file.lua"

        assert is_safe_path(unsafe_path, base) is False

    def test_is_safe_path_prevents_traversal(self, tmp_path):
        """Test is_safe_path prevents path traversal attacks."""
        base = tmp_path / "base"
        base.mkdir()
        traversal_path = base / ".." / ".." / "etc" / "passwd"

        # The path tries to escape base directory
        assert is_safe_path(traversal_path, base) is False

    def test_validate_lua_file_valid(self, tmp_path):
        """Test validate_lua_file returns True for valid .lua file."""
        lua_file = tmp_path / "script.lua"
        lua_file.write_text("-- Lua script")

        assert validate_lua_file(lua_file) is True

    def test_validate_lua_file_wrong_extension(self, tmp_path):
        """Test validate_lua_file returns False for non-.lua file."""
        txt_file = tmp_path / "script.txt"
        txt_file.write_text("text")

        assert validate_lua_file(txt_file) is False

    def test_validate_lua_file_nonexistent(self, tmp_path):
        """Test validate_lua_file returns False for nonexistent file."""
        lua_file = tmp_path / "nonexistent.lua"

        assert validate_lua_file(lua_file) is False

    def test_validate_lua_file_directory(self, tmp_path):
        """Test validate_lua_file returns False for directory."""
        lua_dir = tmp_path / "script.lua"
        lua_dir.mkdir()

        assert validate_lua_file(lua_dir) is False

    def test_is_readable_existing_file(self, tmp_path):
        """Test is_readable returns True for readable file."""
        file = tmp_path / "readable.txt"
        file.write_text("content")

        assert is_readable(file) is True

    def test_is_readable_nonexistent_file(self, tmp_path):
        """Test is_readable returns False for nonexistent file."""
        file = tmp_path / "nonexistent.txt"

        assert is_readable(file) is False

    def test_is_writable_existing_file(self, tmp_path):
        """Test is_writable returns True for writable file."""
        file = tmp_path / "writable.txt"
        file.write_text("content")

        assert is_writable(file) is True

    def test_is_writable_nonexistent_file(self, tmp_path):
        """Test is_writable returns False for nonexistent file."""
        file = tmp_path / "nonexistent.txt"

        assert is_writable(file) is False


class TestFileOperations:
    """Test file extension and name operations."""

    def test_get_file_extension_with_extension(self):
        """Test get_file_extension returns extension."""
        path = Path("script.lua")
        assert get_file_extension(path) == ".lua"

    def test_get_file_extension_no_extension(self):
        """Test get_file_extension returns empty string for no extension."""
        path = Path("README")
        assert get_file_extension(path) == ""

    def test_get_file_extension_multiple_dots(self):
        """Test get_file_extension with multiple dots."""
        path = Path("archive.tar.gz")
        assert get_file_extension(path) == ".gz"

    def test_change_extension_with_dot(self):
        """Test change_extension with dot prefix."""
        path = Path("script.lua")
        result = change_extension(path, ".txt")
        assert result == Path("script.txt")

    def test_change_extension_without_dot(self):
        """Test change_extension without dot prefix."""
        path = Path("script.lua")
        result = change_extension(path, "py")
        assert result == Path("script.py")

    def test_change_extension_no_original_extension(self):
        """Test change_extension on file without extension."""
        path = Path("README")
        result = change_extension(path, ".md")
        assert result == Path("README.md")

    def test_ensure_extension_adds_missing_extension(self):
        """Test ensure_extension adds extension if missing."""
        path = Path("script")
        result = ensure_extension(path, ".lua")
        assert result == Path("script.lua")

    def test_ensure_extension_keeps_existing_extension(self):
        """Test ensure_extension keeps extension if already present."""
        path = Path("script.lua")
        result = ensure_extension(path, ".lua")
        assert result == Path("script.lua")

    def test_ensure_extension_without_dot(self):
        """Test ensure_extension works without dot prefix."""
        path = Path("script")
        result = ensure_extension(path, "lua")
        assert result == Path("script.lua")

    def test_ensure_extension_case_insensitive(self):
        """Test ensure_extension is case-insensitive."""
        path = Path("script.LUA")
        result = ensure_extension(path, ".lua")
        assert result == Path("script.LUA")

    def test_ensure_extension_replaces_wrong_extension(self):
        """Test ensure_extension replaces wrong extension."""
        path = Path("script.txt")
        result = ensure_extension(path, ".lua")
        assert result == Path("script.lua")


class TestPlatformDetection:
    """Test platform detection functions."""

    def test_get_platform_returns_valid_value(self):
        """Test get_platform returns one of expected values."""
        platform_name = get_platform()
        assert platform_name in ["windows", "macos", "linux"]

    def test_get_platform_caches_result(self):
        """Test get_platform caches result for performance."""
        first_call = get_platform()
        second_call = get_platform()
        assert first_call == second_call

    @patch('obfuscator.utils.path_utils.platform_module.system')
    def test_get_platform_windows(self, mock_system):
        """Test get_platform detects Windows."""
        mock_system.return_value = "Windows"
        # Save and restore cache
        import obfuscator.utils.path_utils as pu
        original_cache = pu._PLATFORM_CACHE
        try:
            pu._PLATFORM_CACHE = None
            assert get_platform() == "windows"
        finally:
            pu._PLATFORM_CACHE = original_cache

    @patch('obfuscator.utils.path_utils.platform_module.system')
    def test_get_platform_macos(self, mock_system):
        """Test get_platform detects macOS."""
        mock_system.return_value = "Darwin"
        import obfuscator.utils.path_utils as pu
        original_cache = pu._PLATFORM_CACHE
        try:
            pu._PLATFORM_CACHE = None
            assert get_platform() == "macos"
        finally:
            pu._PLATFORM_CACHE = original_cache

    @patch('obfuscator.utils.path_utils.platform_module.system')
    def test_get_platform_linux(self, mock_system):
        """Test get_platform detects Linux."""
        mock_system.return_value = "Linux"
        import obfuscator.utils.path_utils as pu
        original_cache = pu._PLATFORM_CACHE
        try:
            pu._PLATFORM_CACHE = None
            assert get_platform() == "linux"
        finally:
            pu._PLATFORM_CACHE = original_cache

    def test_platform_boolean_functions(self):
        """Test is_windows, is_macos, is_linux functions."""
        # Exactly one should be True
        results = [is_windows(), is_macos(), is_linux()]
        assert sum(results) == 1

    def test_get_path_separator(self):
        """Test get_path_separator returns os.sep."""
        separator = get_path_separator()
        assert separator == os.sep
        assert separator in ['/', '\\']

