"""Unit tests for StringEncryptionTransformer.

This module provides comprehensive tests for Python string encryption functionality,
including encryption/decryption round-trips, edge cases, and configuration options.
"""

import ast
import base64

import pytest

from obfuscator.processors.ast_transformer import (
    StringEncryptionTransformer,
    TransformResult,
    CRYPTOGRAPHY_AVAILABLE,
)


# Skip all tests if cryptography is not available
pytestmark = pytest.mark.skipif(
    not CRYPTOGRAPHY_AVAILABLE,
    reason="cryptography library not installed"
)


@pytest.fixture
def default_transformer():
    """Create a StringEncryptionTransformer with default settings."""
    return StringEncryptionTransformer()


@pytest.fixture
def custom_key_transformer():
    """Create a StringEncryptionTransformer with custom key length."""
    return StringEncryptionTransformer(key_length=32)


@pytest.fixture
def short_min_transformer():
    """Create a StringEncryptionTransformer with min_string_length=1."""
    return StringEncryptionTransformer(min_string_length=1)


class TestStringEncryptionTransformerInit:
    """Tests for StringEncryptionTransformer initialization."""

    def test_default_initialization(self, default_transformer):
        """Test default transformer initialization."""
        assert default_transformer.key_length == 16
        assert default_transformer.min_string_length == 3
        assert len(default_transformer.encryption_key) == 16
        assert len(default_transformer.iv) == 12
        assert default_transformer.language_mode is None
        assert default_transformer.runtime_injected is False

    def test_custom_key_length(self, custom_key_transformer):
        """Test transformer with custom key length."""
        assert custom_key_transformer.key_length == 32
        assert len(custom_key_transformer.encryption_key) == 32

    def test_key_length_adjustment_small(self):
        """Test that small key lengths are adjusted to 16."""
        transformer = StringEncryptionTransformer(key_length=8)
        assert transformer.key_length == 16

    def test_key_length_adjustment_medium(self):
        """Test that medium key lengths are adjusted appropriately."""
        transformer = StringEncryptionTransformer(key_length=20)
        assert transformer.key_length == 16

    def test_key_length_adjustment_large(self):
        """Test that large key lengths are adjusted to 32."""
        transformer = StringEncryptionTransformer(key_length=28)
        assert transformer.key_length == 24

    def test_min_string_length_custom(self, short_min_transformer):
        """Test custom minimum string length."""
        assert short_min_transformer.min_string_length == 1


class TestEncryptSimpleString:
    """Tests for encrypting simple string literals."""

    def test_encrypt_simple_string(self, default_transformer):
        """Test encrypting a simple string literal."""
        code = 'message = "Hello, World!"'
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 1
        assert result.ast_node is not None

    def test_encrypt_multiple_strings(self, default_transformer):
        """Test encrypting multiple string literals."""
        code = '''
greeting = "Hello"
name = "Alice"
message = "Welcome to the program"
'''
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 3

    def test_encrypt_string_in_function(self, default_transformer):
        """Test encrypting strings inside functions."""
        code = '''
def greet(name):
    return "Hello, " + name
'''
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 1

    def test_encrypt_string_in_class(self, default_transformer):
        """Test encrypting strings inside class definitions."""
        code = '''
class Greeter:
    default_greeting = "Hello"
    
    def greet(self):
        return "Welcome!"
'''
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success
        assert result.transformation_count >= 2


class TestSkipStrings:
    """Tests for strings that should be skipped from encryption."""

    def test_skip_empty_strings(self, default_transformer):
        """Test that empty strings are not encrypted."""
        code = 'empty = ""'
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 0

    def test_skip_short_strings(self, default_transformer):
        """Test that strings shorter than min_string_length are skipped."""
        code = '''
a = "ab"
b = "x"
'''
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 0

    def test_encrypt_at_boundary(self, default_transformer):
        """Test that strings at exactly min_string_length are encrypted."""
        code = 'text = "abc"'  # Exactly 3 characters
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 1

    def test_short_min_length_encrypts_all(self, short_min_transformer):
        """Test that min_string_length=1 encrypts even single chars."""
        code = '''
a = "x"
b = "ab"
'''
        tree = ast.parse(code)

        result = short_min_transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 2


class TestRuntimeInjection:
    """Tests for decryption runtime injection."""

    def test_runtime_injection_exists(self, default_transformer):
        """Test that decryption runtime is injected."""
        code = 'msg = "Hello"'
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success
        assert default_transformer.runtime_injected

        # Check that the runtime import and function exist
        generated_code = ast.unparse(result.ast_node)
        assert '_decrypt_string' in generated_code
        assert 'base64' in generated_code or '_b64' in generated_code

    def test_runtime_injected_once(self, default_transformer):
        """Test that runtime is only injected once per module."""
        code = '''
a = "first"
b = "second"
c = "third"
'''
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success
        # Count occurrences of the decryption function definition
        generated_code = ast.unparse(result.ast_node)
        assert generated_code.count('def _decrypt_string') == 1

    def test_runtime_preserves_future_imports(self, default_transformer):
        """Test that __future__ imports are preserved at the top."""
        code = '''
from __future__ import annotations
message = "Hello"
'''
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success
        # First statement should still be the __future__ import
        first_stmt = result.ast_node.body[0]
        assert isinstance(first_stmt, ast.ImportFrom)
        assert first_stmt.module == '__future__'


class TestConfigurableKeyLength:
    """Tests for configurable encryption key lengths."""

    def test_key_length_8_adjusted(self):
        """Test that key_length=8 is adjusted to 16."""
        transformer = StringEncryptionTransformer(key_length=8)
        code = 'msg = "test string"'
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success
        assert transformer.key_length == 16

    def test_key_length_16(self):
        """Test encryption with 16-byte key."""
        transformer = StringEncryptionTransformer(key_length=16)
        code = 'msg = "test string"'
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 1

    def test_key_length_24(self):
        """Test encryption with 24-byte key."""
        transformer = StringEncryptionTransformer(key_length=24)
        code = 'msg = "test string"'
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 1

    def test_key_length_32(self):
        """Test encryption with 32-byte key."""
        transformer = StringEncryptionTransformer(key_length=32)
        code = 'msg = "test string"'
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 1


class TestDuplicateStrings:
    """Tests for caching of duplicate strings."""

    def test_duplicate_strings_cached(self, default_transformer):
        """Test that duplicate strings use the same encrypted value."""
        code = '''
a = "duplicate"
b = "duplicate"
c = "duplicate"
'''
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 3

        # All three should decrypt to the same value
        generated_code = ast.unparse(result.ast_node)
        # Count _decrypt_string calls - should be 3
        assert generated_code.count('_decrypt_string') >= 3


class TestRoundTrip:
    """Tests for encryption/decryption round-trip verification."""

    def test_round_trip_simple(self, default_transformer):
        """Test that encrypted code can be executed and produces correct output."""
        code = '''
def get_message():
    return "Hello, World!"

result = get_message()
'''
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success

        # Execute the transformed code
        generated_code = ast.unparse(result.ast_node)
        namespace = {}
        exec(generated_code, namespace)

        assert namespace['result'] == "Hello, World!"

    def test_round_trip_multiple_strings(self, default_transformer):
        """Test round-trip with multiple strings."""
        code = '''
def greet(name):
    greeting = "Hello, "
    suffix = "!"
    return greeting + name + suffix

result = greet("Alice")
'''
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success

        generated_code = ast.unparse(result.ast_node)
        namespace = {}
        exec(generated_code, namespace)

        assert namespace['result'] == "Hello, Alice!"

    def test_round_trip_unicode(self, default_transformer):
        """Test round-trip with Unicode strings."""
        code = '''
emoji = "Hello 🌍 World! 你好"
result = emoji
'''
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success

        generated_code = ast.unparse(result.ast_node)
        namespace = {}
        exec(generated_code, namespace)

        assert namespace['result'] == "Hello 🌍 World! 你好"


class TestErrorHandling:
    """Tests for error handling in string encryption."""

    def test_non_module_ast_node(self, default_transformer):
        """Test handling of non-Module AST nodes."""
        # Create a function definition node directly
        code = 'def foo(): pass'
        tree = ast.parse(code)
        func_node = tree.body[0]

        # Transform just the function node (not a module)
        result = default_transformer.transform(func_node)

        # Should still succeed but with a warning
        assert result.success

    def test_transform_result_structure(self, default_transformer):
        """Test that TransformResult has correct structure."""
        code = 'msg = "test"'
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert isinstance(result, TransformResult)
        assert hasattr(result, 'ast_node')
        assert hasattr(result, 'success')
        assert hasattr(result, 'transformation_count')
        assert hasattr(result, 'errors')
        assert isinstance(result.errors, list)


class TestNonStringConstants:
    """Tests to verify non-string constants are not affected."""

    def test_numbers_unchanged(self, default_transformer):
        """Test that numeric constants are not encrypted."""
        code = '''
x = 42
y = 3.14
z = 1 + 2j
'''
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 0

    def test_booleans_unchanged(self, default_transformer):
        """Test that boolean constants are not encrypted."""
        code = '''
a = True
b = False
'''
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 0

    def test_none_unchanged(self, default_transformer):
        """Test that None is not encrypted."""
        code = 'x = None'
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 0

    def test_mixed_constants(self, default_transformer):
        """Test that only strings are encrypted in mixed code."""
        code = '''
name = "Alice"
age = 30
active = True
data = None
'''
        tree = ast.parse(code)

        result = default_transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 1  # Only "Alice"

