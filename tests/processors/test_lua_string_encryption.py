"""Unit tests for Lua string encryption functionality.

This module provides tests for Lua-specific string encryption functionality,
including XOR encryption, runtime injection, and edge cases.
"""

import pytest

from obfuscator.processors.ast_transformer import (
    StringEncryptionTransformer,
    LUAPARSER_AVAILABLE,
)

# Skip all tests if luaparser is not available
pytestmark = pytest.mark.skipif(
    not LUAPARSER_AVAILABLE,
    reason="luaparser library not installed"
)


if LUAPARSER_AVAILABLE:
    from luaparser import ast as lua_ast
    from luaparser import astnodes as lua_nodes


@pytest.fixture
def lua_transformer():
    """Create a StringEncryptionTransformer for Lua."""
    return StringEncryptionTransformer()


class TestLuaXorEncryption:
    """Tests for Lua XOR encryption."""

    def test_xor_encryption_basic(self, lua_transformer):
        """Test basic XOR encryption."""
        plaintext = "Hello, Lua!"
        encrypted = lua_transformer._encrypt_string_xor(plaintext)

        # Encrypted bytes should be different from plaintext
        plaintext_bytes = plaintext.encode('utf-8')
        assert encrypted != plaintext_bytes
        assert len(encrypted) == len(plaintext_bytes)

    def test_xor_encryption_decryption(self, lua_transformer):
        """Test that XOR encryption is reversible."""
        plaintext = "Test string for encryption"
        encrypted = lua_transformer._encrypt_string_xor(plaintext)

        # XOR with the same key should give back plaintext
        key = lua_transformer.encryption_key
        key_len = len(key)
        decrypted_bytes = bytes(
            encrypted[i] ^ key[i % key_len]
            for i in range(len(encrypted))
        )
        decrypted = decrypted_bytes.decode('utf-8')

        assert decrypted == plaintext

    def test_xor_encryption_unicode(self, lua_transformer):
        """Test XOR encryption with Unicode characters."""
        plaintext = "Hello 世界 🌍"
        encrypted = lua_transformer._encrypt_string_xor(plaintext)

        # Verify decryption
        key = lua_transformer.encryption_key
        key_len = len(key)
        decrypted_bytes = bytes(
            encrypted[i] ^ key[i % key_len]
            for i in range(len(encrypted))
        )
        decrypted = decrypted_bytes.decode('utf-8')

        assert decrypted == plaintext


class TestLuaRuntimeGeneration:
    """Tests for Lua decryption runtime generation."""

    def test_lua_runtime_generation(self, lua_transformer):
        """Test that Lua runtime code is generated correctly."""
        runtime_code = lua_transformer._generate_lua_decryption_runtime()

        # Check that runtime contains the decryption function
        assert '_decrypt_string' in runtime_code
        assert '_decrypt_key' in runtime_code
        assert 'local function' in runtime_code
        assert 'string.byte' in runtime_code
        assert 'string.char' in runtime_code

    def test_lua_runtime_key_embedded(self, lua_transformer):
        """Test that encryption key is embedded in runtime."""
        runtime_code = lua_transformer._generate_lua_decryption_runtime()

        # Key should be embedded as escaped bytes
        assert '_decrypt_key = "' in runtime_code


class TestLuaSkipRequirePaths:
    """Tests for skipping require() paths in Lua."""

    def test_skip_short_strings(self, lua_transformer):
        """Test that short strings are skipped."""
        # The transformer's _should_skip_string should skip short strings
        assert lua_transformer._should_skip_string("ab", None)
        assert lua_transformer._should_skip_string("x", None)
        assert lua_transformer._should_skip_string("", None)

    def test_dont_skip_long_strings(self, lua_transformer):
        """Test that long strings are not skipped."""
        # Create a mock node for testing
        import ast
        mock_node = ast.Constant(value="test")

        assert not lua_transformer._should_skip_string("abc", mock_node)
        assert not lua_transformer._should_skip_string("longer string", mock_node)


class TestMixedStringsAndNumbers:
    """Tests to verify only strings are transformed in Lua context."""

    def test_xor_only_for_strings(self, lua_transformer):
        """Test that XOR encryption only works on strings."""
        # XOR encryption should only accept string input
        result = lua_transformer._encrypt_string_xor("test")
        assert isinstance(result, bytes)

    def test_encryption_key_properties(self, lua_transformer):
        """Test encryption key properties."""
        # Key should be the expected length
        assert len(lua_transformer.encryption_key) == lua_transformer.key_length

        # IV should be 12 bytes for AES-GCM
        assert len(lua_transformer.iv) == 12


class TestLuaRuntimeSyntax:
    """Tests to verify Lua runtime code syntax validity."""

    def test_lua_runtime_syntax_valid(self, lua_transformer):
        """Test that generated Lua runtime has valid syntax."""
        runtime_code = lua_transformer._generate_lua_decryption_runtime()

        # Try to parse the runtime code with luaparser
        try:
            tree = lua_ast.parse(runtime_code)
            assert tree is not None
        except Exception as e:
            pytest.fail(f"Lua runtime code failed to parse: {e}")

    def test_lua_runtime_contains_xor(self, lua_transformer):
        """Test that Lua runtime uses XOR operation."""
        runtime_code = lua_transformer._generate_lua_decryption_runtime()

        # Lua 5.3+ uses ~ for XOR
        assert '~' in runtime_code or 'bxor' in runtime_code.lower()


class TestLuaASTTransformation:
    """Tests for Lua AST transformation of String nodes into decryption calls."""

    def test_transform_single_string_to_decrypt_call(self, lua_transformer):
        """Test that a single String node is transformed into a _decrypt_string call."""
        # Parse Lua code with a string literal
        lua_code = 'local msg = "Hello, World!"'
        chunk = lua_ast.parse(lua_code)

        # Transform the AST
        result = lua_transformer.transform(chunk)

        assert result.success
        assert result.transformation_count >= 1

        # Generate Lua source from transformed AST
        output_code = lua_ast.to_lua_source(result.ast_node)

        # Verify the output contains _decrypt_string call
        assert '_decrypt_string(' in output_code
        # The raw string literal should be replaced
        assert '"Hello, World!"' not in output_code

    def test_transform_multiple_strings(self, lua_transformer):
        """Test that multiple String nodes are all transformed."""
        lua_code = '''
local a = "First string"
local b = "Second string"
local c = "Third string"
'''
        chunk = lua_ast.parse(lua_code)

        result = lua_transformer.transform(chunk)

        assert result.success
        assert result.transformation_count >= 3

        output_code = lua_ast.to_lua_source(result.ast_node)

        # All strings should be replaced with decrypt calls
        assert output_code.count('_decrypt_string(') >= 3
        assert '"First string"' not in output_code
        assert '"Second string"' not in output_code
        assert '"Third string"' not in output_code

    def test_transform_preserves_non_string_nodes(self, lua_transformer):
        """Test that non-string nodes are not affected."""
        lua_code = '''
local num = 42
local bool = true
local msg = "Encrypt me"
'''
        chunk = lua_ast.parse(lua_code)

        result = lua_transformer.transform(chunk)

        assert result.success

        output_code = lua_ast.to_lua_source(result.ast_node)

        # Numbers and booleans should be preserved
        assert '42' in output_code
        assert 'true' in output_code
        # String should be encrypted
        assert '_decrypt_string(' in output_code
        assert '"Encrypt me"' not in output_code

    def test_runtime_injection_at_chunk_start(self, lua_transformer):
        """Test that decryption runtime is injected at the beginning of the chunk."""
        lua_code = 'local msg = "Test message"'
        chunk = lua_ast.parse(lua_code)

        result = lua_transformer.transform(chunk)

        assert result.success

        output_code = lua_ast.to_lua_source(result.ast_node)

        # Runtime should be at the start
        assert '_decrypt_key' in output_code
        assert 'local function _decrypt_string' in output_code
        # The runtime should appear before the assignment
        decrypt_key_pos = output_code.find('_decrypt_key')
        msg_assign_pos = output_code.find('msg')
        assert decrypt_key_pos < msg_assign_pos, "Runtime should be injected before code"

    def test_runtime_injected_once(self, lua_transformer):
        """Test that runtime is injected only once even with multiple strings."""
        lua_code = '''
local a = "String one"
local b = "String two"
local c = "String three"
'''
        chunk = lua_ast.parse(lua_code)

        result = lua_transformer.transform(chunk)

        assert result.success

        output_code = lua_ast.to_lua_source(result.ast_node)

        # Runtime should appear exactly once
        assert output_code.count('local function _decrypt_string') == 1
        # _decrypt_key appears: 1 in declaration, 1 for #_decrypt_key, 1 for string.byte
        assert output_code.count('_decrypt_key') >= 2  # declaration + usage in function

    def test_short_strings_not_transformed(self, lua_transformer):
        """Test that strings shorter than min_string_length are not transformed."""
        lua_code = '''
local short = "ab"
local long = "This is long enough"
'''
        chunk = lua_ast.parse(lua_code)

        result = lua_transformer.transform(chunk)

        assert result.success
        # Only the long string should be transformed
        assert result.transformation_count == 1

    def test_string_in_function_call_transformed(self, lua_transformer):
        """Test that strings passed to function calls are transformed."""
        lua_code = 'print("Hello from Lua!")'
        chunk = lua_ast.parse(lua_code)

        result = lua_transformer.transform(chunk)

        assert result.success
        assert result.transformation_count >= 1

        output_code = lua_ast.to_lua_source(result.ast_node)

        # String should be replaced with decrypt call
        assert '_decrypt_string(' in output_code
        assert '"Hello from Lua!"' not in output_code

    def test_nested_strings_transformed(self, lua_transformer):
        """Test that strings in nested structures are transformed."""
        lua_code = '''
local t = {
    name = "John Doe",
    message = "Hello there"
}
'''
        chunk = lua_ast.parse(lua_code)

        result = lua_transformer.transform(chunk)

        assert result.success
        assert result.transformation_count >= 2

        output_code = lua_ast.to_lua_source(result.ast_node)

        # Both strings should be transformed
        assert output_code.count('_decrypt_string(') >= 2
        assert '"John Doe"' not in output_code
        assert '"Hello there"' not in output_code


class TestLuaDocumentation:
    """Tests documenting Lua execution requirements."""

    def test_lua_execution_note(self):
        """Document that actual Lua execution requires external interpreter.

        Note: Full Lua execution testing requires an external Lua interpreter.
        The luaparser library only handles parsing and AST manipulation.
        To verify runtime behavior, use:
        - lua5.3 or luajit command-line interpreter
        - LuaJIT bindings if available
        - Lua C API integration
        """
        # This test serves as documentation
        assert LUAPARSER_AVAILABLE, "luaparser is required for Lua AST manipulation"

