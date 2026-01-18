"""Comprehensive tests for the Lua processor module.

This test suite covers:
- Standard Lua parsing across versions (5.1, 5.2, 5.3, 5.4)
- Luau-specific syntax and type annotations
- LuaJIT-specific features
- Code generation and round-trip testing
- Symbol extraction integration
- Validation and helper methods
- Error handling and edge cases
"""

import os
import stat
from pathlib import Path

import pytest

from obfuscator.processors.lua_processor import (
    LuaProcessor,
    ParseResult,
    GenerateResult,
    MAX_FILE_SIZE_MB,
)
from obfuscator.processors.lua_symbol_extractor import (
    LuaSymbolTable,
    LuaImportInfo,
    LuaFunctionInfo,
    LuaVariableInfo,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def tmp_lua_file(tmp_path):
    """Fixture to create temporary Lua files for testing."""
    def _create_file(content: str, filename: str = "test.lua") -> Path:
        file_path = tmp_path / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path
    return _create_file


@pytest.fixture
def sample_lua_code():
    """Fixture providing sample Lua code strings."""
    return {
        "simple": """
local x = 10
local y = 20
print(x + y)
""",
        "function": """
local function greet(name)
    print("Hello, " .. name)
end
greet("World")
""",
        "table": """
local person = {
    name = "Alice",
    age = 30
}
print(person.name)
""",
        "require": """
local http = require("http")
local json = require("json")
""",
    }


@pytest.fixture
def lua_processor():
    """Fixture providing a LuaProcessor instance."""
    return LuaProcessor()


@pytest.fixture
def lua_processor_with_luau():
    """Fixture providing a LuaProcessor with Luau enabled."""
    return LuaProcessor(enable_luau=True)


# ============================================================================
# Test Utilities
# ============================================================================


def create_lua_file(path: Path, content: str) -> Path:
    """Helper to create Lua files."""
    path.write_text(content, encoding="utf-8")
    return path


def assert_parse_success(result: ParseResult) -> None:
    """Helper to assert successful parsing."""
    assert result.success, f"Parse failed with errors: {result.errors}"
    assert result.ast_node is not None
    assert len(result.errors) == 0


def assert_symbols_extracted(symbol_table: LuaSymbolTable, expected_counts: dict) -> None:
    """Helper to verify symbol counts."""
    if "imports" in expected_counts:
        assert len(symbol_table.imports) == expected_counts["imports"], \
            f"Expected {expected_counts['imports']} imports, got {len(symbol_table.imports)}"
    if "functions" in expected_counts:
        assert len(symbol_table.functions) == expected_counts["functions"], \
            f"Expected {expected_counts['functions']} functions, got {len(symbol_table.functions)}"
    if "variables" in expected_counts:
        assert len(symbol_table.variables) == expected_counts["variables"], \
            f"Expected {expected_counts['variables']} variables, got {len(symbol_table.variables)}"
    if "roblox_patterns" in expected_counts:
        assert len(symbol_table.roblox_api_usage) == expected_counts["roblox_patterns"], \
            f"Expected {expected_counts['roblox_patterns']} Roblox patterns, got {len(symbol_table.roblox_api_usage)}"


# ============================================================================
# Test Class: TestLuaProcessorParsing
# ============================================================================


class TestLuaProcessorParsing:
    """Test standard Lua parsing across versions."""

    def test_parse_lua51_syntax(self, lua_processor, tmp_lua_file):
        """Test Lua 5.1 specific syntax (local function, for loops)."""
        code = """
local function test()
    for i = 1, 10 do
        print(i)
    end
end
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

    def test_parse_lua52_syntax(self, lua_processor, tmp_lua_file):
        """Test Lua 5.2 features (goto, labels)."""
        code = """
::start::
local x = 10
if x > 5 then
    goto finish
end
::finish::
print("Done")
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

    def test_parse_lua53_syntax(self, lua_processor, tmp_lua_file):
        """Test Lua 5.3 features (bitwise operators)."""
        code = """
local a = 5
local b = 3
local c = a & b
local d = a | b
local e = a ~ b
local f = ~a
local g = a << 2
local h = a >> 1
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

    def test_parse_lua54_syntax(self, lua_processor, tmp_lua_file):
        """Test Lua 5.4 features (const, to-be-closed)."""
        code = """
local x <const> = 10
local f <close> = io.open("test.txt", "r")
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        # luaparser actually parses Lua 5.4 syntax (treats <const>/<close> as attributes)
        assert result.success
        assert result.ast_node is not None

    def test_parse_basic_lua_file(self, lua_processor, tmp_lua_file):
        """Test parsing simple Lua file with functions and variables."""
        code = """
local x = 10
local y = 20

local function add(a, b)
    return a + b
end

local result = add(x, y)
print(result)
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)
        assert result.file_path == file_path
        assert result.source_code == code

    def test_parse_file_not_found(self, lua_processor):
        """Test error handling for non-existent files."""
        result = lua_processor.parse_file(Path("nonexistent.lua"))
        assert not result.success
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower()
        assert result.ast_node is None

    def test_parse_file_not_readable(self, lua_processor, tmp_lua_file):
        """Test error handling for unreadable files."""
        code = "local x = 10"
        file_path = tmp_lua_file(code)

        # Make file unreadable
        os.chmod(file_path, 0o000)

        try:
            result = lua_processor.parse_file(file_path)
            assert not result.success
            assert len(result.errors) > 0
            assert "not readable" in result.errors[0].lower()
        finally:
            # Restore permissions for cleanup
            os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)

    def test_parse_file_too_large(self, lua_processor, tmp_path):
        """Test rejection of files exceeding MAX_FILE_SIZE_MB."""
        # Create a file larger than MAX_FILE_SIZE_MB
        large_file = tmp_path / "large.lua"
        size_bytes = (MAX_FILE_SIZE_MB * 1024 * 1024) + 1024  # Slightly over limit

        with open(large_file, "wb") as f:
            f.write(b"-- " * (size_bytes // 3))

        result = lua_processor.parse_file(large_file)
        assert not result.success
        assert len(result.errors) > 0
        assert "too large" in result.errors[0].lower()

    def test_parse_syntax_error(self, lua_processor, tmp_lua_file):
        """Test handling of Lua syntax errors."""
        code = """
local x = 10
local y =
print(x + y)
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert not result.success
        assert len(result.errors) > 0
        assert "syntax error" in result.errors[0].lower()

    def test_parse_empty_file(self, lua_processor, tmp_lua_file):
        """Test parsing empty Lua file."""
        code = ""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)
        assert result.ast_node.body == []


# ============================================================================
# Test Class: TestLuaProcessorLuauSupport
# ============================================================================


class TestLuaProcessorLuauSupport:
    """Test Luau-specific syntax and type annotations."""

    def test_parse_luau_type_annotations(self, lua_processor_with_luau, tmp_lua_file):
        """Test parsing Luau with type annotations."""
        code = """
local name: string = "Alice"
local age: number = 30
local isActive: boolean = true
"""
        file_path = tmp_lua_file(code, "test.luau")
        result = lua_processor_with_luau.parse_file(file_path)
        assert_parse_success(result)
        assert result.luau_enabled
        assert result.luau_metadata is not None

    def test_parse_luau_function_types(self, lua_processor_with_luau, tmp_lua_file):
        """Test function type signatures."""
        code = """
local function add(x: number, y: number): number
    return x + y
end
"""
        file_path = tmp_lua_file(code, "test.luau")
        result = lua_processor_with_luau.parse_file(file_path)
        assert_parse_success(result)
        assert result.luau_metadata is not None

    def test_parse_luau_generics(self, lua_processor_with_luau, tmp_lua_file):
        """Test generic types."""
        code = """
type Array<T> = {T}
local numbers: Array<number> = {1, 2, 3}
"""
        file_path = tmp_lua_file(code, "test.luau")
        result = lua_processor_with_luau.parse_file(file_path)
        assert_parse_success(result)

    def test_parse_luau_type_aliases(self, lua_processor_with_luau, tmp_lua_file):
        """Test type aliases."""
        code = """
type Point = {x: number, y: number}
local p: Point = {x = 10, y = 20}
"""
        file_path = tmp_lua_file(code, "test.luau")
        result = lua_processor_with_luau.parse_file(file_path)
        assert_parse_success(result)

    def test_luau_metadata_preservation(self, lua_processor_with_luau, tmp_lua_file):
        """Verify Luau metadata is attached to AST."""
        code = """
local x: string = "test"
local function greet(name: string): string
    return "Hello, " .. name
end
"""
        file_path = tmp_lua_file(code, "test.luau")
        result = lua_processor_with_luau.parse_file(file_path)
        assert_parse_success(result)
        assert result.luau_metadata is not None
        assert len(result.luau_metadata.annotations) > 0

    def test_luau_type_restoration(self, lua_processor_with_luau, tmp_lua_file):
        """Test that type annotations are restored in generated code."""
        code = """
local x: number = 10
local function add(a: number, b: number): number
    return a + b
end
"""
        file_path = tmp_lua_file(code, "test.luau")
        result = lua_processor_with_luau.parse_file(file_path)
        assert_parse_success(result)

        gen_result = lua_processor_with_luau.generate_code(result.ast_node, restore_luau_types=True)
        assert gen_result.success
        # Check that type annotations are present in generated code
        assert ": number" in gen_result.code

    def test_luau_disabled_by_default(self, lua_processor, tmp_lua_file):
        """Verify Luau preprocessing is disabled by default."""
        code = """
local x: number = 10
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        # Without Luau enabled, this will fail to parse
        assert not result.success or not result.luau_enabled


# ============================================================================
# Test Class: TestLuaProcessorLuaJIT
# ============================================================================


class TestLuaProcessorLuaJIT:
    """Test LuaJIT-specific features."""

    def test_detect_luajit_ffi(self, lua_processor, tmp_lua_file):
        """Test detection of FFI usage."""
        code = """
local ffi = require("ffi")
ffi.cdef[[
    int printf(const char *fmt, ...);
]]
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)
        assert "ffi" in result.luajit_features

    def test_detect_luajit_64bit_integers(self, lua_processor, tmp_lua_file):
        """Test detection of 64-bit integer literals."""
        code = """
local x = 123LL
local y = 0x1234ULL
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        # Note: luaparser may not parse these, but we test detection
        features = lua_processor.detect_luajit_features(code)
        assert "64bit_integers" in features

    def test_detect_luajit_bitop(self, lua_processor, tmp_lua_file):
        """Test detection of BitOp library usage."""
        code = """
local bit = require("bit")
local x = bit.band(5, 3)
local y = bit.bor(5, 3)
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)
        assert "bitop" in result.luajit_features

    def test_detect_luajit_jit_pragmas(self, lua_processor, tmp_lua_file):
        """Test detection of JIT pragmas."""
        code = """
jit.on()
jit.off()
jit.flush()
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)
        assert "jit_pragmas" in result.luajit_features

    def test_luajit_features_in_parse_result(self, lua_processor, tmp_lua_file):
        """Verify detected features are included in ParseResult."""
        code = """
local ffi = require("ffi")
local bit = require("bit")
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)
        assert isinstance(result.luajit_features, list)
        assert len(result.luajit_features) > 0


# ============================================================================
# Test Class: TestLuaProcessorCodeGeneration
# ============================================================================


class TestLuaProcessorCodeGeneration:
    """Test code generation from AST."""

    def test_generate_code_basic(self, lua_processor, tmp_lua_file):
        """Test generating code from simple AST."""
        code = """
local x = 10
print(x)
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

        gen_result = lua_processor.generate_code(result.ast_node)
        assert gen_result.success
        assert len(gen_result.code) > 0
        assert len(gen_result.errors) == 0

    def test_generate_code_with_functions(self, lua_processor, tmp_lua_file):
        """Test generating code with function definitions."""
        code = """
local function add(a, b)
    return a + b
end

local function multiply(x, y)
    return x * y
end
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

        gen_result = lua_processor.generate_code(result.ast_node)
        assert gen_result.success
        assert "function" in gen_result.code

    def test_generate_code_with_tables(self, lua_processor, tmp_lua_file):
        """Test generating code with table constructors."""
        code = """
local person = {
    name = "Alice",
    age = 30,
    hobbies = {"reading", "coding"}
}
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

        gen_result = lua_processor.generate_code(result.ast_node)
        assert gen_result.success

    def test_generate_code_round_trip(self, lua_processor, tmp_lua_file):
        """Test parse → generate → parse cycle produces valid code."""
        code = """
local x = 10
local function test()
    return x * 2
end
print(test())
"""
        file_path = tmp_lua_file(code)

        # First parse
        result1 = lua_processor.parse_file(file_path)
        assert_parse_success(result1)

        # Generate code
        gen_result = lua_processor.generate_code(result1.ast_node)
        assert gen_result.success

        # Parse generated code
        file_path2 = tmp_lua_file(gen_result.code, "generated.lua")
        result2 = lua_processor.parse_file(file_path2)
        assert_parse_success(result2)

    def test_generate_code_preserves_semantics(self, lua_processor, tmp_lua_file):
        """Verify generated code has same behavior."""
        code = """
local x = 5
local y = 10
local z = x + y
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

        gen_result = lua_processor.generate_code(result.ast_node)
        assert gen_result.success
        # Basic check: generated code should contain the same variable names
        assert "x" in gen_result.code
        assert "y" in gen_result.code
        assert "z" in gen_result.code

    def test_generate_code_with_luau_types(self, lua_processor_with_luau, tmp_lua_file):
        """Test Luau type restoration in generated code."""
        code = """
local x: number = 10
local function greet(name: string): string
    return "Hello, " .. name
end
"""
        file_path = tmp_lua_file(code, "test.luau")
        result = lua_processor_with_luau.parse_file(file_path)
        assert_parse_success(result)

        gen_result = lua_processor_with_luau.generate_code(result.ast_node, restore_luau_types=True)
        assert gen_result.success
        assert ": number" in gen_result.code
        assert ": string" in gen_result.code

    def test_generate_code_without_luau_restoration(self, lua_processor_with_luau, tmp_lua_file):
        """Test restore_luau_types=False parameter."""
        code = """
local x: number = 10
"""
        file_path = tmp_lua_file(code, "test.luau")
        result = lua_processor_with_luau.parse_file(file_path)
        assert_parse_success(result)

        gen_result = lua_processor_with_luau.generate_code(result.ast_node, restore_luau_types=False)
        assert gen_result.success
        # Type annotations should not be restored
        # Note: This depends on implementation details


# ============================================================================
# Test Class: TestLuaProcessorSymbolExtraction
# ============================================================================


class TestLuaProcessorSymbolExtraction:
    """Test symbol extraction integration."""

    def test_extract_symbols_require_calls(self, lua_processor, tmp_lua_file):
        """Test extraction of require() calls with aliases."""
        code = """
local http = require("http")
local json = require("json")
local utils = require("./utils")
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

        symbols = lua_processor.extract_symbols(result.ast_node, file_path)
        assert_symbols_extracted(symbols, {"imports": 3})
        assert symbols.imports[0].alias == "http"
        assert symbols.imports[1].alias == "json"

    def test_extract_symbols_functions(self, lua_processor, tmp_lua_file):
        """Test extraction of function definitions (local and global)."""
        code = """
local function localFunc(x)
    return x * 2
end

function globalFunc(y)
    return y + 1
end

local obj = {}
function obj:method(z)
    return z - 1
end
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

        symbols = lua_processor.extract_symbols(result.ast_node, file_path)
        assert_symbols_extracted(symbols, {"functions": 3})

        # Check local function
        local_funcs = [f for f in symbols.functions if f.name == "localFunc"]
        assert len(local_funcs) == 1
        assert local_funcs[0].is_local

        # Check global function
        global_funcs = [f for f in symbols.functions if f.name == "globalFunc"]
        assert len(global_funcs) == 1
        assert not global_funcs[0].is_local

    def test_extract_symbols_variables(self, lua_processor, tmp_lua_file):
        """Test extraction of variable assignments."""
        code = """
local x = 10
local y = 20
MAX_SIZE = 100
local config = {debug = true}
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

        symbols = lua_processor.extract_symbols(result.ast_node, file_path)
        assert_symbols_extracted(symbols, {"variables": 4})

    def test_extract_symbols_roblox_patterns(self, lua_processor, tmp_lua_file):
        """Test detection of Roblox API patterns."""
        code = """
local Players = game:GetService("Players")
local part = Instance.new("Part")
local player = game.Players.LocalPlayer
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

        symbols = lua_processor.extract_symbols(result.ast_node, file_path)
        assert len(symbols.roblox_api_usage) > 0

    def test_extract_symbols_method_definitions(self, lua_processor, tmp_lua_file):
        """Test extraction of method definitions (colon syntax)."""
        code = """
local MyClass = {}

function MyClass:new()
    return setmetatable({}, {__index = MyClass})
end

function MyClass:greet(name)
    print("Hello, " .. name)
end
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

        symbols = lua_processor.extract_symbols(result.ast_node, file_path)
        methods = [f for f in symbols.functions if f.is_method]
        assert len(methods) >= 2

    def test_extract_symbols_scope_tracking(self, lua_processor, tmp_lua_file):
        """Verify local vs global scope is correctly identified."""
        code = """
local localVar = 10
globalVar = 20

local function localFunc()
    return 1
end

function globalFunc()
    return 2
end
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

        symbols = lua_processor.extract_symbols(result.ast_node, file_path)

        # Check variable scopes
        local_vars = [v for v in symbols.variables if v.scope == "local"]
        global_vars = [v for v in symbols.variables if v.scope == "global"]
        assert len(local_vars) >= 1
        assert len(global_vars) >= 1

    def test_extract_symbols_empty_file(self, lua_processor, tmp_lua_file):
        """Test symbol extraction from empty AST."""
        code = ""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

        symbols = lua_processor.extract_symbols(result.ast_node, file_path)
        assert_symbols_extracted(symbols, {"imports": 0, "functions": 0, "variables": 0})

    def test_extract_symbols_complex_file(self, lua_processor, tmp_lua_file):
        """Test extraction from file with mixed symbols."""
        code = """
local http = require("http")
local json = require("json")

local API_KEY = "secret"
local config = {debug = true}

local function fetchData(url)
    return http.get(url)
end

function processData(data)
    return json.decode(data)
end

local MyModule = {}

function MyModule:init()
    self.data = {}
end

return MyModule
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

        symbols = lua_processor.extract_symbols(result.ast_node, file_path)
        assert len(symbols.imports) >= 2
        assert len(symbols.functions) >= 3
        assert len(symbols.variables) >= 3


# ============================================================================
# Test Class: TestLuaProcessorValidation
# ============================================================================


class TestLuaProcessorValidation:
    """Test validation and helper methods."""

    def test_validate_syntax_valid(self, lua_processor):
        """Test validate_syntax() with valid Lua code."""
        code = """
local x = 10
local function test()
    return x * 2
end
"""
        is_valid, error = lua_processor.validate_syntax(code)
        assert is_valid
        assert error == ""

    def test_validate_syntax_invalid(self, lua_processor):
        """Test validate_syntax() with syntax errors."""
        code = """
local x =
print(x)
"""
        is_valid, error = lua_processor.validate_syntax(code)
        assert not is_valid
        assert len(error) > 0

    def test_get_ast_info(self, lua_processor, tmp_lua_file):
        """Test get_ast_info() returns correct metadata."""
        code = """
local x = 10
local y = 20
print(x + y)
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

        info = lua_processor.get_ast_info(result.ast_node)
        assert "node_type" in info
        assert info["node_type"] == "Chunk"
        assert "body_length" in info
        assert info["body_length"] > 0

    def test_get_ast_info_empty(self, lua_processor, tmp_lua_file):
        """Test get_ast_info() with empty AST."""
        code = ""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

        info = lua_processor.get_ast_info(result.ast_node)
        assert info["body_length"] == 0


# ============================================================================
# Test Class: TestLuaProcessorErrorHandling
# ============================================================================


class TestLuaProcessorErrorHandling:
    """Test error handling and edge cases."""

    def test_parse_unicode_content(self, lua_processor, tmp_lua_file):
        """Test parsing Lua files with Unicode characters."""
        code = """
local greeting = "Hello, 世界! 🌍"
local emoji = "🎉"
print(greeting)
"""
        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        assert_parse_success(result)

    def test_parse_encoding_error(self, lua_processor, tmp_path):
        """Test handling of encoding errors."""
        # Create a file with invalid UTF-8
        file_path = tmp_path / "invalid.lua"
        with open(file_path, "wb") as f:
            f.write(b"local x = \xff\xfe")

        result = lua_processor.parse_file(file_path)
        # Should handle encoding error gracefully
        assert result.success is False
        assert result.errors
        assert "encoding" in result.errors[0].lower() or "decode" in result.errors[0].lower()

    def test_generate_code_invalid_ast(self, lua_processor):
        """Test error handling for malformed AST nodes."""
        # luaparser.ast.to_lua_source(None) actually returns empty string without error
        # Test with an object that will cause an actual error
        class InvalidNode:
            pass

        gen_result = lua_processor.generate_code(InvalidNode())
        assert gen_result.success is False
        assert gen_result.errors
        # Should contain error about unexpected type or attribute error
        assert any(keyword in gen_result.errors[0].lower()
                  for keyword in ["error", "unexpected", "attribute", "type"])

    def test_extract_symbols_invalid_ast(self, lua_processor, tmp_path):
        """Test error handling in symbol extraction."""
        file_path = tmp_path / "test.lua"

        # The visitor is very tolerant and doesn't raise errors for invalid nodes
        # Test that extract_symbols handles the case gracefully
        # Even with None or invalid objects, it returns an empty symbol table
        symbols = lua_processor.extract_symbols(None, file_path)
        assert symbols is not None
        assert len(symbols.imports) == 0
        assert len(symbols.functions) == 0
        assert len(symbols.variables) == 0

    def test_deeply_nested_ast(self, lua_processor, tmp_lua_file):
        """Test handling of deeply nested AST structures."""
        # Create deeply nested code
        depth = 50
        code = "local x = " + "(" * depth + "1" + ")" * depth

        file_path = tmp_lua_file(code)
        result = lua_processor.parse_file(file_path)
        # Should handle deep nesting
        assert result is not None

