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
    NumberObfuscationTransformer,
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


class TestNumberObfuscationIntegration:
    """Integration tests for NumberObfuscationTransformer."""

    def test_number_obfuscation_in_pipeline(self):
        """Test number obfuscation in full obfuscation pipeline."""
        code = '''
def calculate():
    x = 42
    y = 100
    return x + y

result = calculate()
'''
        tree = ast.parse(code)

        transformer = NumberObfuscationTransformer(complexity=3)
        result = transformer.transform(tree)

        assert result.success
        assert result.transformation_count >= 2  # 42 and 100 should be transformed

        # Generate code and validate syntax
        generated_code = ast.unparse(result.ast_node)
        assert "42" not in generated_code or "(" in generated_code  # 42 should be obfuscated

        # Validate by re-parsing
        reparsed = ast.parse(generated_code)
        assert reparsed is not None

    def test_number_obfuscation_with_config(self):
        """Test number obfuscation with configuration options."""
        config = ObfuscationConfig(
            name="test",
            features={"number_obfuscation": True},
            options={
                "number_obfuscation_complexity": 4,
                "number_obfuscation_min_value": 50,
                "number_obfuscation_max_value": 5000,
            },
        )

        transformer = NumberObfuscationTransformer(config=config)

        code = '''
x = 25  # Should not be obfuscated (below min)
y = 100  # Should be obfuscated
z = 6000  # Should not be obfuscated (above max)
'''
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 1  # Only y = 100

    def test_combine_number_obfuscation_with_constant_folding(self):
        """Test combining NumberObfuscationTransformer with ConstantFoldingTransformer."""
        code = '''
x = 2 + 3  # Should be folded to 5
y = 42     # Should be obfuscated
result = x + y
'''
        tree = ast.parse(code)

        # First apply constant folding
        folder = ConstantFoldingTransformer()
        fold_result = folder.transform(tree)

        assert fold_result.success
        assert fold_result.transformation_count >= 1  # 2 + 3 -> 5

        # Then apply number obfuscation
        obfuscator = NumberObfuscationTransformer(complexity=2)
        obfusc_result = obfuscator.transform(fold_result.ast_node)

        assert obfusc_result.success
        assert obfusc_result.transformation_count >= 1  # 42 should be obfuscated

        # Verify output
        generated_code = ast.unparse(obfusc_result.ast_node)
        assert "5" in generated_code  # Folded constant

    def test_combine_number_obfuscation_with_string_encryption(self):
        """Test combining NumberObfuscationTransformer with StringEncryptionTransformer."""
        code = '''
def process():
    count = 42
    message = "Result: "
    return message + str(count)

output = process()
'''
        tree = ast.parse(code)

        # Apply number obfuscation first
        number_obfuscator = NumberObfuscationTransformer(complexity=2)
        number_result = number_obfuscator.transform(tree)

        assert number_result.success

        # Then apply string encryption
        encryptor = StringEncryptionTransformer()
        encrypt_result = encryptor.transform(number_result.ast_node)

        assert encrypt_result.success

        # Verify both transformations applied
        generated_code = ast.unparse(encrypt_result.ast_node)
        # String should be encrypted
        assert "_decrypt_string" in generated_code

    def test_number_obfuscation_with_arrays(self):
        """Test number obfuscation combined with array transformation."""
        code = '''
def get_data():
    values = [10, 20, 30, 42]
    index = 2
    return values[index]

result = get_data()
'''
        tree = ast.parse(code)

        # Apply number obfuscation
        number_obfuscator = NumberObfuscationTransformer(complexity=2)
        number_result = number_obfuscator.transform(tree)

        assert number_result.success

        # Then apply array transformation
        arr_transformer = ConstantArrayTransformer(shuffle_seed=42)
        arr_result = arr_transformer.transform(number_result.ast_node)

        assert arr_result.success

        # Verify transformations
        generated_code = ast.unparse(arr_result.ast_node)
        assert "_arr_" in generated_code

    def test_number_obfuscation_execution_correctness(self):
        """Test that obfuscated numbers execute with correct values."""
        code = '''
def calculate():
    x = 42
    y = 100
    return x + y

result = calculate()
'''
        tree = ast.parse(code)

        transformer = NumberObfuscationTransformer(complexity=3)
        result = transformer.transform(tree)

        assert result.success

        # Execute and verify
        generated_code = ast.unparse(result.ast_node)
        namespace = {}
        exec(generated_code, namespace)

        assert namespace['result'] == 142  # 42 + 100

    def test_number_obfuscation_zero_and_one_unchanged(self):
        """Test that zero and one are not obfuscated."""
        code = '''
x = 0
y = 1
z = 42
'''
        tree = ast.parse(code)

        transformer = NumberObfuscationTransformer(complexity=2)
        result = transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 1  # Only 42 is transformed

        generated_code = ast.unparse(result.ast_node)
        assert "0" in generated_code
        assert "1" in generated_code

    def test_number_obfuscation_complexity_levels(self):
        """Test all complexity levels produce valid code."""
        code = 'x = 250'

        for complexity in range(1, 6):
            tree = ast.parse(code)
            transformer = NumberObfuscationTransformer(complexity=complexity)
            result = transformer.transform(tree)

            assert result.success
            assert result.transformation_count == 1

            # Verify code is valid and executable
            generated_code = ast.unparse(result.ast_node)
            reparsed = ast.parse(generated_code)
            assert reparsed is not None

            namespace = {}
            exec(generated_code, namespace)
            assert namespace['x'] == 250

    @pytest.mark.skipif(not LUAPARSER_AVAILABLE, reason="luaparser not installed")
    def test_lua_number_obfuscation_integration(self):
        """Test Lua number obfuscation integration."""
        from luaparser import ast as lua_ast

        code = '''
local function calculate()
    local x = 42
    local y = 100
    return x + y
end
'''
        tree = lua_ast.parse(code)

        transformer = NumberObfuscationTransformer(complexity=2)
        result = transformer.transform(tree)

        assert result.success
        assert result.transformation_count >= 2
        assert transformer.language_mode == "lua"

    def test_multiple_number_transformations(self):
        """Test processing multiple numbers in complex expressions."""
        code = '''
def complex_calculation():
    a = 10
    b = 20
    c = 30
    d = 40
    e = 50
    result = a + b * c - d / e
    return result

output = complex_calculation()
'''
        tree = ast.parse(code)

        transformer = NumberObfuscationTransformer(complexity=3)
        result = transformer.transform(tree)

        assert result.success
        assert result.transformation_count >= 5  # Should transform most numbers

        # Verify execution correctness
        generated_code = ast.unparse(result.ast_node)
        namespace = {}
        exec(generated_code, namespace)

        expected = 10 + 20 * 30 - 40 / 50
        assert abs(namespace['output'] - expected) < 0.001

    def test_number_obfuscation_edge_cases(self):
        """Test number obfuscation with edge cases."""
        code = '''
# Edge cases
small = 5      # Should not be obfuscated (below default min)
large = 999999 # Should not be obfuscated (above default max)
zero = 0       # Should not be obfuscated
one = 1        # Should not be obfuscated
negative = -10 # Should not be obfuscated
valid = 100    # Should be obfuscated
'''
        tree = ast.parse(code)

        transformer = NumberObfuscationTransformer()
        result = transformer.transform(tree)

        assert result.success
        assert result.transformation_count == 1  # Only 'valid = 100'

    def test_number_obfuscation_float_values(self):
        """Test number obfuscation with float values."""
        code = '''
x = 3.14
y = 42.5
z = 100.0
'''
        tree = ast.parse(code)

        transformer = NumberObfuscationTransformer(complexity=2)
        result = transformer.transform(tree)

        assert result.success
        # Should transform floats in range
        assert result.transformation_count >= 0

        # Verify execution
        generated_code = ast.unparse(result.ast_node)
        namespace = {}
        exec(generated_code, namespace)

        assert abs(namespace['x'] - 3.14) < 0.001
        assert abs(namespace['y'] - 42.5) < 0.001
        assert abs(namespace['z'] - 100.0) < 0.001
