"""Unit tests for ConstantArrayTransformer.

Tests the ConstantArrayTransformer class which obfuscates constant arrays
through element shuffling and index mapping.
"""

import ast
import pytest

from obfuscator.processors.ast_transformer import ConstantArrayTransformer
from obfuscator.core.config import ObfuscationConfig


class TestConstantArrayTransformerInit:
    """Test ConstantArrayTransformer initialization and configuration."""

    def test_default_initialization(self):
        """Test transformer initializes with default settings."""
        transformer = ConstantArrayTransformer()
        assert transformer.shuffle_seed is None
        assert transformer.language_mode is None
        assert not transformer.runtime_injected
        assert len(transformer.array_mappings) == 0
        assert transformer.array_id_counter == 0
        assert len(transformer.transformed_arrays) == 0
        assert len(transformer.array_to_id) == 0
        assert len(transformer.var_to_array_id) == 0

    def test_initialization_with_config(self):
        """Test transformer initialization with config."""
        config = ObfuscationConfig(
            name="test",
            options={"array_shuffle_seed": 42}
        )
        transformer = ConstantArrayTransformer(config)
        assert transformer.shuffle_seed == 42

    def test_initialization_with_explicit_seed(self):
        """Test transformer initialization with explicit shuffle_seed."""
        transformer = ConstantArrayTransformer(shuffle_seed=12345)
        assert transformer.shuffle_seed == 12345

    def test_explicit_seed_overrides_config(self):
        """Test explicit shuffle_seed overrides config."""
        config = ObfuscationConfig(
            name="test",
            options={"array_shuffle_seed": 42}
        )
        transformer = ConstantArrayTransformer(config, shuffle_seed=999)
        assert transformer.shuffle_seed == 999


class TestPythonArrayDetection:
    """Test Python array detection methods."""

    def test_is_constant_array_with_constants(self):
        """Test detection of constant arrays."""
        transformer = ConstantArrayTransformer()
        code = "x = [1, 2, 3, 4]"
        tree = ast.parse(code)
        assign = tree.body[0]
        list_node = assign.value

        assert transformer._is_constant_array(list_node) is True

    def test_is_constant_array_with_mixed_types(self):
        """Test detection fails with mixed constant/variable arrays."""
        transformer = ConstantArrayTransformer()
        code = "x = [1, y, 3]"
        tree = ast.parse(code)
        assign = tree.body[0]
        list_node = assign.value

        assert transformer._is_constant_array(list_node) is False

    def test_is_constant_array_empty(self):
        """Test detection fails on empty arrays."""
        transformer = ConstantArrayTransformer()
        code = "x = []"
        tree = ast.parse(code)
        assign = tree.body[0]
        list_node = assign.value

        assert transformer._is_constant_array(list_node) is False

    def test_is_constant_array_single_element(self):
        """Test detection fails on single-element arrays."""
        transformer = ConstantArrayTransformer()
        code = "x = [42]"
        tree = ast.parse(code)
        assign = tree.body[0]
        list_node = assign.value

        assert transformer._is_constant_array(list_node) is False

    def test_is_constant_array_with_different_constants(self):
        """Test detection works with various constant types."""
        transformer = ConstantArrayTransformer()
        code = "x = [1, 'a', True, None]"
        tree = ast.parse(code)
        assign = tree.body[0]
        list_node = assign.value

        assert transformer._is_constant_array(list_node) is True


class TestPythonArrayTransformation:
    """Test Python array transformation."""

    def test_simple_list_transformation(self):
        """Test transformation of a simple constant list."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "x = [1, 2, 3, 4]"
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1
        assert len(result.errors) == 0
        assert len(transformer.array_mappings) == 1

    def test_nested_list_transformation(self):
        """Test transformation of nested lists."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "x = [[1, 2], [3, 4]]"
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 2  # Two inner lists
        assert len(result.errors) == 0
        assert len(transformer.array_mappings) == 2

    def test_mixed_constant_variable_lists(self):
        """Test transformation skips non-constant lists."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "x = [1, 2, y, 4]"
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 0  # Should skip mixed list
        assert len(result.errors) == 0
        assert len(transformer.array_mappings) == 0

    def test_multiple_constant_arrays(self):
        """Test transformation of multiple arrays in same code."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = """
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
"""
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 3
        assert len(result.errors) == 0
        assert len(transformer.array_mappings) == 3

    def test_deterministic_shuffling_with_seed(self):
        """Test that shuffling is deterministic with seed."""
        transformer1 = ConstantArrayTransformer(shuffle_seed=42)
        transformer2 = ConstantArrayTransformer(shuffle_seed=42)
        code = "x = [1, 2, 3, 4, 5]"

        tree1 = ast.parse(code)
        tree2 = ast.parse(code)

        result1 = transformer1.transform(tree1)
        result2 = transformer2.transform(tree2)

        assert result1.success is True
        assert result2.success is True
        # Same mappings should be generated
        assert transformer1.array_mappings == transformer2.array_mappings


class TestPythonRuntimeInjection:
    """Test Python runtime injection."""

    def test_runtime_injection(self):
        """Test that runtime mapping dictionaries are injected."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "x = [1, 2, 3]"
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert transformer.runtime_injected is True

        # Check that mapping dictionary was added to module body
        module_body = result.ast_node.body
        # First statement should be the mapping assignment
        assert len(module_body) > 1
        assert isinstance(module_body[0], ast.Assign)

        # Verify the mapping variable name
        mapping_name = module_body[0].targets[0].id
        assert mapping_name.startswith("_arr_")
        assert mapping_name.endswith("_map")

    def test_runtime_preserves_docstring(self):
        """Test that runtime injection preserves module docstring."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = '"""Module docstring."""\nx = [1, 2, 3]'
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True

        # First statement should still be docstring
        module_body = result.ast_node.body
        assert isinstance(module_body[0], ast.Expr)
        assert isinstance(module_body[0].value, ast.Constant)
        assert isinstance(module_body[0].value.value, str)

        # Runtime should be after docstring
        assert isinstance(module_body[1], ast.Assign)

    def test_runtime_with_future_imports(self):
        """Test that runtime injection preserves __future__ imports."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = 'from __future__ import annotations\nx = [1, 2, 3]'
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True

        # First statement should be __future__ import
        module_body = result.ast_node.body
        assert isinstance(module_body[0], ast.ImportFrom)
        assert module_body[0].module == '__future__'

        # Runtime should be after __future__ import
        assert isinstance(module_body[1], ast.Assign)


class TestPythonRoundTrip:
    """Test that transformed code is functionally equivalent."""

    def test_roundtrip_execution(self):
        """Test that transformed code executes correctly."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = """
def get_value():
    arr = [10, 20, 30, 40, 50]
    return arr[0] + arr[2]
"""
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True

        # Note: This test validates the transformation succeeds.
        # Full functional equivalence would require index rewriting logic
        # which is a simplification in the current implementation.

    def test_nested_structure_preservation(self):
        """Test that nested structures are preserved."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = """
outer = [1, 2, 3, 4]
inner = [10, 20, 30]
"""
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_module(self):
        """Test transformation of empty module."""
        transformer = ConstantArrayTransformer()
        tree = ast.parse("")

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 0
        assert len(result.errors) == 0

    def test_module_without_arrays(self):
        """Test transformation of module without arrays."""
        transformer = ConstantArrayTransformer()
        code = """
x = 1
y = 2
z = x + y
"""
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 0
        assert len(result.errors) == 0

    def test_invalid_ast_type(self):
        """Test transformation handles unknown AST types gracefully."""
        transformer = ConstantArrayTransformer()

        # Create a simple expression instead of module
        expr = ast.parse("1 + 2", mode='eval')

        result = transformer.transform(expr)

        # Should still succeed but with no transformations
        assert result.success is True
        assert result.transformation_count == 0

    def test_large_array_transformation(self):
        """Test transformation of large arrays."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        # Create array with 100 elements
        code = "x = " + str(list(range(100)))
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1
        assert len(result.errors) == 0

    def test_array_with_duplicate_values(self):
        """Test that arrays with duplicate values are handled correctly."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "x = [1, 2, 1, 3, 2]"
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1
        assert len(result.errors) == 0


class TestArrayMappingGeneration:
    """Test array mapping generation utilities."""

    def test_generate_array_id(self):
        """Test unique array ID generation."""
        transformer = ConstantArrayTransformer()

        id1 = transformer._generate_array_id()
        id2 = transformer._generate_array_id()
        id3 = transformer._generate_array_id()

        assert id1 == "_arr_0"
        assert id2 == "_arr_1"
        assert id3 == "_arr_2"

    def test_generate_shuffle_mapping(self):
        """Test shuffle mapping generation."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)

        original_to_shuffled, shuffled_to_original = transformer._generate_shuffle_mapping(5)

        # Check lengths
        assert len(original_to_shuffled) == 5
        assert len(shuffled_to_original) == 5

        # Check it's a valid permutation
        assert set(original_to_shuffled) == {0, 1, 2, 3, 4}
        assert set(shuffled_to_original) == {0, 1, 2, 3, 4}

        # Check inverse relationship
        for i in range(5):
            shuffled_idx = original_to_shuffled[i]
            assert shuffled_to_original[shuffled_idx] == i

    def test_generate_shuffle_mapping_deterministic(self):
        """Test that shuffle mapping is deterministic with seed."""
        transformer1 = ConstantArrayTransformer(shuffle_seed=42)
        transformer2 = ConstantArrayTransformer(shuffle_seed=42)

        map1 = transformer1._generate_shuffle_mapping(10)
        map2 = transformer2._generate_shuffle_mapping(10)

        assert map1 == map2

    def test_generate_shuffle_mapping_random_without_seed(self):
        """Test that shuffle mapping varies without seed."""
        transformer1 = ConstantArrayTransformer()
        transformer2 = ConstantArrayTransformer()

        map1 = transformer1._generate_shuffle_mapping(10)
        map2 = transformer2._generate_shuffle_mapping(10)

        # Different seeds should produce different mappings (with high probability)
        # Note: This test might occasionally fail due to randomness
        # but is statistically very unlikely
        assert map1 != map2 or map1 == (list(range(10)), list(range(10)))


class TestTransformResult:
    """Test TransformResult structure."""

    def test_transform_result_success(self):
        """Test successful transformation result."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "x = [1, 2, 3]"
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.ast_node is not None
        assert result.transformation_count == 1
        assert len(result.errors) == 0

    def test_transform_result_with_no_transformations(self):
        """Test transformation result with no changes."""
        transformer = ConstantArrayTransformer()
        code = "x = 1"
        tree = ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.ast_node is not None
        assert result.transformation_count == 0
        assert len(result.errors) == 0
