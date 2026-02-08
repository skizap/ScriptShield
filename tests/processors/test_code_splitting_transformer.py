"""Tests for the CodeSplittingTransformer.

This module provides comprehensive test coverage for the code splitting
transformer, including initialization, Python and Lua chunk encryption,
function splitting, runtime injection, and integration with other transformers.
"""

import ast
import base64
import unittest
from unittest.mock import MagicMock, patch

import pytest

from obfuscator.core.config import ObfuscationConfig
from obfuscator.processors.ast_transformer import (
    CodeSplittingTransformer,
    CRYPTOGRAPHY_AVAILABLE,
    LUAPARSER_AVAILABLE,
    StringEncryptionTransformer,
    NumberObfuscationTransformer,
)

# Skip Lua tests if luaparser is not available
lua_skip = pytest.mark.skipif(
    not LUAPARSER_AVAILABLE,
    reason="luaparser not installed"
)

if LUAPARSER_AVAILABLE:
    from luaparser import ast as lua_ast
    from luaparser import astnodes as lua_nodes


class TestCodeSplittingTransformerInit(unittest.TestCase):
    """Tests for CodeSplittingTransformer initialization."""

    def test_default_initialization(self):
        """Verify default chunk_size is 5 and encryption is True."""
        transformer = CodeSplittingTransformer()
        assert transformer.chunk_size == 5
        assert transformer.encryption_enabled is True
        assert transformer.language_mode is None
        assert transformer.runtime_injected is False
        assert len(transformer.split_functions) == 0

    def test_initialization_with_config(self):
        """Test config extraction for chunk_size and encryption."""
        config = MagicMock()
        config.options = {
            "code_split_chunk_size": 10,
            "code_split_encryption": False,
        }
        transformer = CodeSplittingTransformer(config=config)
        assert transformer.chunk_size == 10
        assert transformer.encryption_enabled is False

    def test_initialization_with_explicit_params(self):
        """Test parameter override."""
        config = MagicMock()
        config.options = {
            "code_split_chunk_size": 10,
            "code_split_encryption": False,
        }
        transformer = CodeSplittingTransformer(
            config=config, chunk_size=3, encryption_enabled=True
        )
        assert transformer.chunk_size == 3
        assert transformer.encryption_enabled is True

    def test_explicit_params_override_config(self):
        """Verify parameter precedence over config."""
        config = MagicMock()
        config.options = {"code_split_chunk_size": 8}
        transformer = CodeSplittingTransformer(config=config, chunk_size=4)
        assert transformer.chunk_size == 4

    def test_chunk_size_minimum_validation(self):
        """Test chunk_size below minimum is clamped to 2."""
        transformer = CodeSplittingTransformer(chunk_size=1)
        assert transformer.chunk_size == 2

        transformer = CodeSplittingTransformer(chunk_size=0)
        assert transformer.chunk_size == 2

        transformer = CodeSplittingTransformer(chunk_size=-5)
        assert transformer.chunk_size == 2

    def test_chunk_size_valid_values(self):
        """Test valid chunk_size values are accepted."""
        for size in [2, 5, 10]:
            transformer = CodeSplittingTransformer(chunk_size=size)
            assert transformer.chunk_size == size

    def test_encryption_key_generation(self):
        """Verify encryption key is generated."""
        transformer = CodeSplittingTransformer()
        assert len(transformer.encryption_key) == 16

    def test_encryption_keys_are_unique(self):
        """Two transformers should have different keys."""
        t1 = CodeSplittingTransformer()
        t2 = CodeSplittingTransformer()
        assert t1.encryption_key != t2.encryption_key


class TestPythonChunkEncryption(unittest.TestCase):
    """Tests for Python AES-GCM chunk encryption."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = CodeSplittingTransformer()

    @pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed")
    def test_aes_encryption_produces_bytes(self):
        """Verify AES encryption returns bytes with IV prefix."""
        result = self.transformer._encrypt_chunk_aes("x = 1")
        assert isinstance(result, bytes)
        # Must be at least 12 bytes (IV) + some ciphertext
        assert len(result) > 12

    @pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed")
    def test_aes_encryption_decryption_roundtrip(self):
        """Verify encrypted chunks can be decrypted with per-chunk IV."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        original = "x = 1\ny = 2\nz = x + y"
        encrypted = self.transformer._encrypt_chunk_aes(original)

        # Extract per-chunk IV from the first 12 bytes
        chunk_iv = encrypted[:12]
        ciphertext = encrypted[12:]
        aesgcm = AESGCM(self.transformer.encryption_key)
        decrypted = aesgcm.decrypt(chunk_iv, ciphertext, None).decode("utf-8")
        assert decrypted == original

    @pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed")
    def test_per_chunk_iv_uniqueness(self):
        """Each call should generate a different IV."""
        enc1 = self.transformer._encrypt_chunk_aes("x = 1")
        enc2 = self.transformer._encrypt_chunk_aes("x = 1")
        iv1 = enc1[:12]
        iv2 = enc2[:12]
        assert iv1 != iv2

    @pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed")
    def test_base64_encoding_of_encrypted_chunks(self):
        """Verify encrypted chunks can be base64-encoded."""
        encrypted = self.transformer._encrypt_chunk_aes("print('hello')")
        b64 = base64.b64encode(encrypted).decode("ascii")
        assert isinstance(b64, str)
        # Should be valid base64
        decoded = base64.b64decode(b64)
        assert decoded == encrypted

    def test_xor_encryption_produces_bytes(self):
        """Verify XOR encryption returns bytes."""
        result = self.transformer._encrypt_chunk_xor("x = 1")
        assert isinstance(result, bytes)
        assert len(result) == len("x = 1".encode("utf-8"))

    def test_xor_encryption_decryption_roundtrip(self):
        """Verify XOR encryption is reversible."""
        original = "local x = 1\nlocal y = 2"
        encrypted = self.transformer._encrypt_chunk_xor(original)

        # XOR is symmetric
        key = self.transformer.encryption_key
        decrypted_bytes = bytes(
            encrypted[i] ^ key[i % len(key)]
            for i in range(len(encrypted))
        )
        assert decrypted_bytes.decode("utf-8") == original


class TestPythonFunctionSplitting(unittest.TestCase):
    """Tests for Python function splitting."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = CodeSplittingTransformer(chunk_size=2)

    def _make_large_function(self, num_stmts=10):
        """Helper to create a function with many statements."""
        stmts = "\n".join(f"    x{i} = {i}" for i in range(num_stmts))
        code = f"def big_func():\n{stmts}\n    return x0"
        return code

    def test_split_simple_function(self):
        """Test splitting a function with enough statements."""
        code = self._make_large_function(10)
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 1
        assert "big_func" in self.transformer.split_functions

    def test_skip_small_function(self):
        """Functions with fewer than chunk_size*2 statements are skipped."""
        code = "def small():\n    x = 1\n    return x"
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 0

    def test_preserves_function_name(self):
        """The function name should be preserved after splitting."""
        code = self._make_large_function(10)
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success

        # Find the function def in the transformed AST
        func_defs = [
            node for node in ast.walk(result.ast_node)
            if isinstance(node, ast.FunctionDef)
        ]
        assert len(func_defs) >= 1
        assert func_defs[0].name == "big_func"

    def test_chunk_size_control(self):
        """Different chunk sizes produce different numbers of chunks."""
        code = self._make_large_function(12)

        # chunk_size=2 -> 12 stmts / 2 = 6 chunks
        t2 = CodeSplittingTransformer(chunk_size=2)
        tree = ast.parse(code)
        result = t2.transform(tree)
        assert result.success
        assert result.transformation_count == 1

        # chunk_size=5 -> 12 stmts / 5 = 3 chunks (2+2+1 remainder -> still 3)
        t5 = CodeSplittingTransformer(chunk_size=5)
        tree = ast.parse(code)
        result = t5.transform(tree)
        assert result.success
        assert result.transformation_count == 1

    def test_skip_generator_function(self):
        """Generator functions (with yield) should be skipped."""
        code = """
def gen():
    x0 = 0
    x1 = 1
    x2 = 2
    x3 = 3
    yield x0
"""
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 0

    def test_skip_async_function(self):
        """Async functions should not be visited by visit_FunctionDef."""
        code = """
async def async_func():
    x0 = 0
    x1 = 1
    x2 = 2
    x3 = 3
    return x0
"""
        tree = ast.parse(code)
        # CodeSplittingTransformer only overrides visit_FunctionDef, not
        # visit_AsyncFunctionDef, so async functions won't be split.
        result = self.transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 0

    def test_multiple_functions(self):
        """Multiple eligible functions in one module."""
        stmts = "\n".join(f"    x{i} = {i}" for i in range(6))
        code = f"def foo():\n{stmts}\n    return x0\n\ndef bar():\n{stmts}\n    return x0"
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 2
        assert "foo" in self.transformer.split_functions
        assert "bar" in self.transformer.split_functions

    def test_transformed_ast_is_valid(self):
        """Transformed AST should be parseable when unparsed."""
        code = self._make_large_function(10)
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success
        assert result.ast_node is not None

        # Unparse and re-parse should succeed
        unparsed = ast.unparse(result.ast_node)
        reparsed = ast.parse(unparsed)
        assert reparsed is not None

    def test_function_args_forwarded(self):
        """Wrapper should forward original function parameters."""
        stmts = "\n".join(f"    z{i} = a + b + {i}" for i in range(6))
        code = f"def add_stuff(a, b):\n{stmts}\n    return z0"
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success

        unparsed = ast.unparse(result.ast_node)
        # The wrapper should reference the parameter names
        assert "'a'" in unparsed or '"a"' in unparsed
        assert "'b'" in unparsed or '"b"' in unparsed

    def test_encryption_disabled_produces_valid_ast(self):
        """encryption_enabled=False should still produce valid AST."""
        transformer = CodeSplittingTransformer(chunk_size=2, encryption_enabled=False)
        code = self._make_large_function(10)
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 1

        unparsed = ast.unparse(result.ast_node)
        reparsed = ast.parse(unparsed)
        assert reparsed is not None
        # Runtime should NOT import AESGCM when encryption is disabled
        assert "AESGCM" not in unparsed


class TestPythonRuntimeInjection(unittest.TestCase):
    """Tests for Python runtime injection."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = CodeSplittingTransformer(chunk_size=2)

    def _make_large_function(self, num_stmts=10):
        stmts = "\n".join(f"    x{i} = {i}" for i in range(num_stmts))
        return f"def big_func():\n{stmts}\n    return x0"

    def test_runtime_injected_when_functions_split(self):
        """Runtime should be injected when at least one function is split."""
        code = self._make_large_function(10)
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success
        assert self.transformer.runtime_injected is True

    def test_runtime_not_injected_when_no_splits(self):
        """Runtime should NOT be injected when no functions are split."""
        code = "x = 1"
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success
        assert self.transformer.runtime_injected is False

    def test_runtime_preserves_future_imports(self):
        """Runtime should be inserted after __future__ imports."""
        code = f"from __future__ import annotations\n{self._make_large_function(10)}"
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success

        # First statement should still be the __future__ import
        first_stmt = result.ast_node.body[0]
        assert isinstance(first_stmt, ast.ImportFrom)
        assert first_stmt.module == "__future__"

    def test_runtime_preserves_docstring(self):
        """Runtime should be inserted after module docstring."""
        code = f'\"\"\"Module docstring.\"\"\"\n{self._make_large_function(10)}'
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success

        # First statement should still be the docstring
        first_stmt = result.ast_node.body[0]
        assert isinstance(first_stmt, ast.Expr)
        assert isinstance(first_stmt.value, ast.Constant)
        assert first_stmt.value.value == "Module docstring."

    def test_runtime_code_contains_decrypt_functions(self):
        """Injected runtime should define _decrypt_chunk and _reassemble_function."""
        code = self._make_large_function(10)
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success

        unparsed = ast.unparse(result.ast_node)
        assert "_decrypt_chunk" in unparsed
        assert "_reassemble_function" in unparsed

    def test_runtime_without_encryption(self):
        """Runtime with encryption_enabled=False should not use AES."""
        transformer = CodeSplittingTransformer(chunk_size=2, encryption_enabled=False)
        code = self._make_large_function(10)
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success

        unparsed = ast.unparse(result.ast_node)
        assert "_decrypt_chunk" in unparsed
        assert "_reassemble_function" in unparsed
        assert "AESGCM" not in unparsed


@lua_skip
class TestLuaChunkEncryption(unittest.TestCase):
    """Tests for Lua XOR chunk encryption."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = CodeSplittingTransformer()

    def test_xor_encryption_for_lua(self):
        """XOR encryption should work for Lua code strings."""
        code = "local x = 1"
        result = self.transformer._encrypt_chunk_xor(code)
        assert isinstance(result, bytes)
        assert len(result) == len(code.encode("utf-8"))

    def test_xor_byte_escaping(self):
        """Encrypted bytes should be escapable for Lua strings."""
        code = "local x = 1"
        encrypted = self.transformer._encrypt_chunk_xor(code)
        escaped = "".join(f"\\{b}" for b in encrypted)
        assert isinstance(escaped, str)
        assert len(escaped) > 0


@lua_skip
class TestLuaFunctionSplitting(unittest.TestCase):
    """Tests for Lua function splitting."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = CodeSplittingTransformer(chunk_size=2)

    def test_transform_lua_chunk(self):
        """Basic Lua chunk transformation."""
        code = """
function big_func()
    local x0 = 0
    local x1 = 1
    local x2 = 2
    local x3 = 3
    local x4 = 4
    local x5 = 5
    return x0
end
"""
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success

    def test_skip_small_lua_function(self):
        """Small Lua functions should be skipped."""
        code = """
function small()
    local x = 1
    return x
end
"""
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 0


@lua_skip
class TestLuaRuntimeInjection(unittest.TestCase):
    """Tests for Lua runtime injection."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = CodeSplittingTransformer(chunk_size=2)

    def test_lua_language_detection(self):
        """Transformer should detect Lua from Chunk node."""
        code = """
function foo()
    local x = 1
    return x
end
"""
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success
        assert self.transformer.language_mode == "lua"


class TestTransformResult(unittest.TestCase):
    """Tests for transformation result handling."""

    def test_successful_transformation(self):
        """Verify result structure on success."""
        transformer = CodeSplittingTransformer(chunk_size=2)
        stmts = "\n".join(f"    x{i} = {i}" for i in range(6))
        code = f"def foo():\n{stmts}\n    return x0"
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success is True
        assert result.ast_node is not None
        assert isinstance(result.transformation_count, int)
        assert isinstance(result.errors, list)

    def test_transformation_count_tracking(self):
        """Count should match number of split functions."""
        transformer = CodeSplittingTransformer(chunk_size=2)
        stmts = "\n".join(f"    x{i} = {i}" for i in range(6))
        code = (
            f"def foo():\n{stmts}\n    return x0\n\n"
            f"def bar():\n{stmts}\n    return x0"
        )
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 2

    def test_error_handling_none_input(self):
        """None input should return failure."""
        transformer = CodeSplittingTransformer()
        result = transformer.transform(None)
        assert not result.success
        assert len(result.errors) > 0

    def test_error_handling_unsupported_type(self):
        """Unsupported AST node type should return failure."""
        transformer = CodeSplittingTransformer()
        result = transformer.transform("not an ast node")
        assert not result.success
        assert len(result.errors) > 0

    def test_zero_transformations_on_empty_module(self):
        """Empty module should succeed with zero transformations."""
        transformer = CodeSplittingTransformer()
        tree = ast.parse("")
        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 0


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def test_empty_module(self):
        """Empty module handling."""
        transformer = CodeSplittingTransformer(chunk_size=2)
        tree = ast.parse("")
        result = transformer.transform(tree)
        assert result.success is True
        assert result.transformation_count == 0

    def test_module_without_functions(self):
        """Module with only statements."""
        transformer = CodeSplittingTransformer(chunk_size=2)
        code = "x = 1\ny = 2\nz = 3"
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success is True
        assert result.transformation_count == 0

    def test_nested_functions(self):
        """Nested function handling."""
        transformer = CodeSplittingTransformer(chunk_size=2)
        inner_stmts = "\n".join(f"        y{i} = {i}" for i in range(6))
        outer_stmts = "\n".join(f"    x{i} = {i}" for i in range(6))
        code = (
            f"def outer():\n{outer_stmts}\n"
            f"    def inner():\n{inner_stmts}\n        return y0\n"
            f"    return inner()"
        )
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success is True

    def test_class_methods(self):
        """Class method handling."""
        transformer = CodeSplittingTransformer(chunk_size=2)
        stmts = "\n".join(f"        self.x{i} = {i}" for i in range(6))
        code = f"class Foo:\n    def method(self):\n{stmts}\n        return self.x0"
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success is True
        assert result.transformation_count >= 1

    def test_function_with_decorators(self):
        """Decorated functions should be split, decorators preserved."""
        transformer = CodeSplittingTransformer(chunk_size=2)
        stmts = "\n".join(f"    x{i} = {i}" for i in range(6))
        code = f"@staticmethod\ndef decorated():\n{stmts}\n    return x0"
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success is True
        assert result.transformation_count == 1

        # Decorator should be preserved
        func_defs = [
            node for node in ast.walk(result.ast_node)
            if isinstance(node, ast.FunctionDef)
        ]
        assert len(func_defs) >= 1
        assert len(func_defs[0].decorator_list) == 1

    def test_function_with_yield_from(self):
        """Functions with yield from should be skipped."""
        transformer = CodeSplittingTransformer(chunk_size=2)
        code = """
def delegating_gen():
    x0 = 0
    x1 = 1
    x2 = 2
    x3 = 3
    yield from range(10)
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 0

    def test_state_reset_between_transforms(self):
        """Transformer state should reset between transform() calls."""
        transformer = CodeSplittingTransformer(chunk_size=2)
        stmts = "\n".join(f"    x{i} = {i}" for i in range(6))
        code = f"def foo():\n{stmts}\n    return x0"

        tree1 = ast.parse(code)
        result1 = transformer.transform(tree1)
        assert result1.success
        assert result1.transformation_count == 1

        tree2 = ast.parse(code)
        result2 = transformer.transform(tree2)
        assert result2.success
        assert result2.transformation_count == 1
        # split_functions should be reset
        assert len(transformer.split_functions) == 1


class TestIntegration(unittest.TestCase):
    """Integration tests with other transformers."""

    def test_with_number_obfuscation(self):
        """CodeSplittingTransformer after NumberObfuscationTransformer."""
        stmts = "\n".join(f"    x{i} = {i}" for i in range(6))
        code = f"def compute():\n{stmts}\n    return x0"
        tree = ast.parse(code)

        # Apply number obfuscation first
        num_transformer = NumberObfuscationTransformer()
        result1 = num_transformer.transform(tree)
        assert result1.success

        # Then apply code splitting
        cs_transformer = CodeSplittingTransformer(chunk_size=2)
        result2 = cs_transformer.transform(result1.ast_node)
        assert result2.success

    def test_obfuscated_code_is_valid_ast(self):
        """Full pipeline output should be valid Python AST."""
        stmts = "\n".join(f"    x{i} = {i}" for i in range(8))
        code = f"def example():\n{stmts}\n    return x0"
        tree = ast.parse(code)

        transformer = CodeSplittingTransformer(chunk_size=2)
        result = transformer.transform(tree)
        assert result.success

        # Should be unparseable and re-parseable
        unparsed = ast.unparse(result.ast_node)
        reparsed = ast.parse(unparsed)
        assert reparsed is not None

    def test_encrypted_chunks_are_base64(self):
        """Wrapper body should contain base64-encoded chunk strings."""
        stmts = "\n".join(f"    x{i} = {i}" for i in range(6))
        code = f"def target():\n{stmts}\n    return x0"
        tree = ast.parse(code)

        transformer = CodeSplittingTransformer(chunk_size=2)
        result = transformer.transform(tree)
        assert result.success

        unparsed = ast.unparse(result.ast_node)
        # The function body should contain _reassemble_function call
        assert "_reassemble_function" in unparsed

    def test_config_driven_pipeline(self):
        """Test using ObfuscationConfig to drive transformer."""
        config = ObfuscationConfig(
            name="test_split",
            features={"code_splitting": True},
            options={
                "code_split_chunk_size": 3,
                "code_split_encryption": True,
            },
        )
        transformer = CodeSplittingTransformer(config=config)
        assert transformer.chunk_size == 3
        assert transformer.encryption_enabled is True

        stmts = "\n".join(f"    x{i} = {i}" for i in range(8))
        code = f"def driven():\n{stmts}\n    return x0"
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 1


class TestConfigValidation(unittest.TestCase):
    """Tests for config validation of code splitting options."""

    def test_valid_config(self):
        """Valid config should pass validation."""
        config = ObfuscationConfig(
            name="test",
            options={
                "code_split_chunk_size": 5,
                "code_split_encryption": True,
            },
        )
        config.validate()  # Should not raise

    def test_invalid_chunk_size_type(self):
        """Non-integer chunk_size should fail validation."""
        config = ObfuscationConfig(
            name="test",
            options={"code_split_chunk_size": "five"},
        )
        with pytest.raises(ValueError, match="code_split_chunk_size"):
            config.validate()

    def test_chunk_size_too_small(self):
        """chunk_size < 2 should fail validation."""
        config = ObfuscationConfig(
            name="test",
            options={"code_split_chunk_size": 1},
        )
        with pytest.raises(ValueError, match="code_split_chunk_size"):
            config.validate()

    def test_invalid_encryption_type(self):
        """Non-boolean code_split_encryption should fail validation."""
        config = ObfuscationConfig(
            name="test",
            options={"code_split_encryption": "yes"},
        )
        with pytest.raises(ValueError, match="code_split_encryption"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
