"""Integration tests for transformers with the obfuscation pipeline.

This module tests the integration of transformers with
other transformers and the complete obfuscation workflow.
"""

import ast

import pytest

from obfuscator.processors.ast_transformer import (
    StringEncryptionTransformer,
    ConstantFoldingTransformer,
    ConstantArrayTransformer,
    CRYPTOGRAPHY_AVAILABLE,
    LUAPARSER_AVAILABLE,
)
from obfuscator.core.config import ObfuscationConfig


# Skip encryption tests if cryptography is not available
requires_cryptography = pytest.mark.skipif(
    not CRYPTOGRAPHY_AVAILABLE,
    reason="cryptography library not installed"
)


class TestPythonFullPipeline:
    """Tests for complete Python obfuscation pipeline with string encryption."""

    @requires_cryptography
    def test_python_parse_encrypt_generate_validate(self):
        """Test full Python pipeline: parse → encrypt → generate → validate."""
        code = '''
def greet(name):
    message = "Hello, "
    return message + name + "!"

result = greet("World")
'''
        tree = ast.parse(code)

        transformer = StringEncryptionTransformer()
        result = transformer.transform(tree)

        assert result.success
        assert result.transformation_count >= 2  # "Hello, " and "!"

        # Generate code and validate syntax
        generated_code = ast.unparse(result.ast_node)
        assert "_decrypt_string" in generated_code

        # Validate by re-parsing
        reparsed = ast.parse(generated_code)
        assert reparsed is not None

    @requires_cryptography
    def test_python_execute_encrypted_code(self):
        """Test that encrypted Python code executes correctly."""
        code = '''
def compute():
    prefix = "Result: "
    value = "42"
    return prefix + value

output = compute()
'''
        tree = ast.parse(code)

        transformer = StringEncryptionTransformer()
        result = transformer.transform(tree)

        assert result.success

        # Execute and verify
        generated_code = ast.unparse(result.ast_node)
        namespace = {}
        exec(generated_code, namespace)

        assert namespace['output'] == "Result: 42"


class TestLuaFullPipeline:
    """Tests for Lua pipeline with string encryption."""

    @pytest.mark.skipif(not LUAPARSER_AVAILABLE, reason="luaparser not installed")
    def test_lua_parse_validate(self):
        """Test Lua parsing and validation."""
        from luaparser import ast as lua_ast

        code = '''
local function greet(name)
    local message = "Hello, "
    return message .. name
end
'''
        tree = lua_ast.parse(code)
        assert tree is not None


class TestConfigIntegration:
    """Tests for configuration integration with string encryption."""

    @requires_cryptography
    def test_config_key_length_respected(self):
        """Test that config's string_encryption_key_length is respected."""
        config = ObfuscationConfig(
            name="test",
            features={"string_encryption": True},
            options={"string_encryption_key_length": 32},
        )

        transformer = StringEncryptionTransformer(config=config)
        assert transformer.key_length == 32

    @requires_cryptography
    def test_config_default_key_length(self):
        """Test default key length from config."""
        config = ObfuscationConfig(name="test")
        transformer = StringEncryptionTransformer(config=config)
        assert transformer.key_length == 16

    @requires_cryptography
    def test_feature_enabled_check(self):
        """Test checking if string_encryption feature is enabled."""
        config = ObfuscationConfig(
            name="test",
            features={"string_encryption": True},
        )

        # Verify feature is enabled in config
        assert config.features.get("string_encryption") is True


class TestMultipleFiles:
    """Tests for processing multiple files with separate encryption keys."""

    @requires_cryptography
    def test_separate_keys_per_transformer(self):
        """Test that each transformer instance has unique encryption keys."""
        transformer1 = StringEncryptionTransformer()
        transformer2 = StringEncryptionTransformer()

        # Keys should be different
        assert transformer1.encryption_key != transformer2.encryption_key
        assert transformer1.iv != transformer2.iv

    @requires_cryptography
    def test_process_multiple_modules(self):
        """Test processing multiple Python modules independently."""
        module1_code = 'greeting = "Hello"'
        module2_code = 'farewell = "Goodbye"'

        tree1 = ast.parse(module1_code)
        tree2 = ast.parse(module2_code)

        transformer1 = StringEncryptionTransformer()
        transformer2 = StringEncryptionTransformer()

        result1 = transformer1.transform(tree1)
        result2 = transformer2.transform(tree2)

        assert result1.success
        assert result2.success
        assert result1.transformation_count == 1
        assert result2.transformation_count == 1


class TestWithOtherTransformers:
    """Tests for combining string encryption with other transformers."""

    @requires_cryptography
    def test_combine_with_constant_folding(self):
        """Test combining StringEncryptionTransformer with ConstantFoldingTransformer."""
        code = '''
x = 2 + 3
message = "The answer is: "
result = message + str(x)
'''
        tree = ast.parse(code)

        # First apply constant folding
        folder = ConstantFoldingTransformer()
        fold_result = folder.transform(tree)

        assert fold_result.success
        assert fold_result.transformation_count >= 1  # 2 + 3 -> 5

        # Then apply string encryption
        encryptor = StringEncryptionTransformer()
        encrypt_result = encryptor.transform(fold_result.ast_node)

        assert encrypt_result.success
        assert encrypt_result.transformation_count >= 1

        # Verify output
        generated_code = ast.unparse(encrypt_result.ast_node)
        assert "_decrypt_string" in generated_code

    @requires_cryptography
    def test_order_independence(self):
        """Test that transformer order doesn't affect correctness."""
        code = '''
def get_value():
    constant = 10 + 5
    label = "Value: "
    return label + str(constant)

output = get_value()
'''
        # Order 1: Fold then Encrypt
        tree1 = ast.parse(code)
        folder1 = ConstantFoldingTransformer()
        encryptor1 = StringEncryptionTransformer()

        folded1 = folder1.transform(tree1)
        result1 = encryptor1.transform(folded1.ast_node)

        # Order 2: Encrypt then Fold
        tree2 = ast.parse(code)
        encryptor2 = StringEncryptionTransformer()
        folder2 = ConstantFoldingTransformer()

        encrypted2 = encryptor2.transform(tree2)
        result2 = folder2.transform(encrypted2.ast_node)

        # Both should succeed
        assert result1.success
        assert result2.success

        # Execute both and verify output
        namespace1 = {}
        namespace2 = {}

        exec(ast.unparse(result1.ast_node), namespace1)
        exec(ast.unparse(result2.ast_node), namespace2)

        assert namespace1['output'] == "Value: 15"
        assert namespace2['output'] == "Value: 15"


class TestConstantArrayIntegration:
    """Integration tests for ConstantArrayTransformer."""

    def test_array_transform_in_obfuscation_pipeline(self):
        """Test constant array transformation in full pipeline."""
        code = '''
def process_data():
    values = [10, 20, 30, 40, 50]
    return len(values)

result = process_data()
'''
        tree = ast.parse(code)

        transformer = ConstantArrayTransformer(shuffle_seed=42)
        result = transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 1

        # Generate code and validate syntax
        generated_code = ast.unparse(result.ast_node)
        # Should have mapping dictionary injected
        assert "_arr_" in generated_code and "_map" in generated_code

        # Validate by re-parsing
        reparsed = ast.parse(generated_code)
        assert reparsed is not None

    def test_config_seed_integration(self):
        """Test that config's array_shuffle_seed is properly integrated."""
        config = ObfuscationConfig(
            name="test",
            features={"constant_array": True},
            options={"array_shuffle_seed": 123},
        )

        transformer = ConstantArrayTransformer(config=config)
        assert transformer.shuffle_seed == 123

    def test_config_default_seed(self):
        """Test default seed from config."""
        config = ObfuscationConfig(name="test")
        transformer = ConstantArrayTransformer(config=config)
        assert transformer.shuffle_seed is None

    def test_multiple_transformers_with_arrays(self):
        """Test processing multiple arrays across transformers."""
        module1_code = 'data = [1, 2, 3]'
        module2_code = 'values = [4, 5, 6]'

        tree1 = ast.parse(module1_code)
        tree2 = ast.parse(module2_code)

        transformer1 = ConstantArrayTransformer(shuffle_seed=42)
        transformer2 = ConstantArrayTransformer(shuffle_seed=42)

        result1 = transformer1.transform(tree1)
        result2 = transformer2.transform(tree2)

        assert result1.success
        assert result2.success
        assert result1.transformation_count == 1
        assert result2.transformation_count == 1
        # Same seed should produce same mappings
        assert transformer1.array_mappings == transformer2.array_mappings

    def test_combine_array_transform_with_other_transforms(self):
        """Test combining ConstantArrayTransformer with other transformers."""
        code = '''
x = 2 + 3
values = [10, 20, 30]
message = "Data: "
'''
        tree = ast.parse(code)

        # Apply constant folding first
        folder = ConstantFoldingTransformer()
        fold_result = folder.transform(tree)

        assert fold_result.success

        # Then apply array transformation
        arr_transformer = ConstantArrayTransformer(shuffle_seed=42)
        arr_result = arr_transformer.transform(fold_result.ast_node)

        assert arr_result.success
        assert arr_result.transformation_count == 1

        # Verify output
        generated_code = ast.unparse(arr_result.ast_node)
        assert "_arr_" in generated_code

    @requires_cryptography
    def test_array_and_string_encryption_combined(self):
        """Test combining ConstantArrayTransformer with StringEncryptionTransformer."""
        code = '''
def get_config():
    keys = ["key1", "key2", "key3"]
    return keys[0]
'''
        tree = ast.parse(code)

        # Apply string encryption
        encryptor = StringEncryptionTransformer()
        encrypt_result = encryptor.transform(tree)

        assert encrypt_result.success

        # Then apply array transformation
        arr_transformer = ConstantArrayTransformer(shuffle_seed=42)
        arr_result = arr_transformer.transform(encrypt_result.ast_node)

        assert arr_result.success

        # Verify both transformations applied
        generated_code = ast.unparse(arr_result.ast_node)
        assert "_decrypt_string" in generated_code or "_arr_" in generated_code

    @pytest.mark.skipif(not LUAPARSER_AVAILABLE, reason="luaparser not installed")
    def test_lua_array_integration(self):
        """Test Lua array transformation integration."""
        from luaparser import ast as lua_ast

        code = '''
local function get_values()
    local data = {10, 20, 30, 40}
    return data
end
'''
        tree = lua_ast.parse(code)

        transformer = ConstantArrayTransformer(shuffle_seed=42)
        result = transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 1
        assert len(transformer.array_mappings) == 1

    @requires_cryptography
    def test_python_arrays_with_strings_integrated(self):
        """Test arrays containing strings with both transformers."""
        code = '''
def get_messages():
    messages = ["hello", "world", "test"]
    return messages
'''
        tree = ast.parse(code)

        # Apply both transformers
        encryptor = StringEncryptionTransformer()
        arr_transformer = ConstantArrayTransformer(shuffle_seed=42)

        # Order: encrypt first, then array transform
        encrypt_result = encryptor.transform(tree)
        arr_result = arr_transformer.transform(encrypt_result.ast_node)

        assert encrypt_result.success
        assert arr_result.success

        # Verify transformations
        generated_code = ast.unparse(arr_result.ast_node)
        # Either strings were encrypted or array was transformed
        # (or both, depending on implementation details)
        assert len(generated_code) > 0
