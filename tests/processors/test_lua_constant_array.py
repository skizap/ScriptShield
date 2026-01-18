"""Unit tests for ConstantArrayTransformer Lua functionality.

Tests the Lua table transformation capabilities of ConstantArrayTransformer.
"""

import pytest

from obfuscator.processors.ast_transformer import ConstantArrayTransformer, LUAPARSER_AVAILABLE

# Skip all tests if luaparser is not available
pytestmark = pytest.mark.skipif(
    not LUAPARSER_AVAILABLE,
    reason="luaparser not available"
)

if LUAPARSER_AVAILABLE:
    from luaparser import ast as lua_ast


class TestLuaTableDetection:
    """Test Lua table detection methods."""

    def test_is_constant_table_with_constants(self):
        """Test detection of constant tables."""
        transformer = ConstantArrayTransformer()
        code = "local x = {1, 2, 3, 4}"
        tree = lua_ast.parse(code)
        table_node = tree.body.body[0].init

        assert transformer._is_constant_table(table_node) is True

    def test_is_constant_table_with_mixed_types(self):
        """Test detection fails with mixed constant/variable tables."""
        transformer = ConstantArrayTransformer()
        code = "local y = {1, x, 3}"
        tree = lua_ast.parse(code)
        table_node = tree.body.body[0].init

        assert transformer._is_constant_table(table_node) is False

    def test_is_constant_table_empty(self):
        """Test detection fails on empty tables."""
        transformer = ConstantArrayTransformer()
        code = "local x = {}"
        tree = lua_ast.parse(code)
        table_node = tree.body.body[0].init

        # Empty tables have no fields
        assert transformer._is_constant_table(table_node) is False

    def test_is_constant_table_single_element(self):
        """Test detection fails on single-element tables."""
        transformer = ConstantArrayTransformer()
        code = "local x = {42}"
        tree = lua_ast.parse(code)
        table_node = tree.body.body[0].init

        assert transformer._is_constant_table(table_node) is False

    def test_is_constant_table_with_different_constants(self):
        """Test detection works with various constant types."""
        transformer = ConstantArrayTransformer()
        code = "local x = {1, 'a', true, nil}"
        tree = lua_ast.parse(code)
        table_node = tree.body.body[0].init

        assert transformer._is_constant_table(table_node) is True


class TestLuaTableTransformation:
    """Test Lua table transformation."""

    def test_simple_table_transformation(self):
        """Test transformation of a simple constant table."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "local x = {1, 2, 3, 4}"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1
        assert len(result.errors) == 0
        assert len(transformer.array_mappings) == 1

    def test_nested_table_transformation(self):
        """Test transformation of nested tables."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "local x = {{1, 2}, {3, 4}}"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 2  # Two inner tables
        assert len(result.errors) == 0
        assert len(transformer.array_mappings) == 2

    def test_mixed_constant_variable_tables(self):
        """Test transformation skips non-constant tables."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "local x = {1, 2, y, 4}"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 0  # Should skip mixed table
        assert len(result.errors) == 0
        assert len(transformer.array_mappings) == 0

    def test_multiple_constant_tables(self):
        """Test transformation of multiple tables in same code."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = """
local x = {1, 2, 3}
local y = {4, 5, 6}
local z = {7, 8, 9}
"""
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 3
        assert len(result.errors) == 0
        assert len(transformer.array_mappings) == 3

    def test_deterministic_shuffling_with_seed(self):
        """Test that shuffling is deterministic with seed."""
        transformer1 = ConstantArrayTransformer(shuffle_seed=42)
        transformer2 = ConstantArrayTransformer(shuffle_seed=42)
        code = "local x = {1, 2, 3, 4, 5}"

        tree1 = lua_ast.parse(code)
        tree2 = lua_ast.parse(code)

        result1 = transformer1.transform(tree1)
        result2 = transformer2.transform(tree2)

        assert result1.success is True
        assert result2.success is True
        # Same mappings should be generated
        assert transformer1.array_mappings == transformer2.array_mappings


class TestLuaRuntimeInjection:
    """Test Lua runtime injection."""

    def test_runtime_injection(self):
        """Test that runtime mapping tables are injected."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "local x = {1, 2, 3}"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert transformer.runtime_injected is True

        # Check that mapping table was added to chunk body
        chunk_body = result.ast_node.body.body
        # First statement should be the mapping assignment
        assert len(chunk_body) > 1
        first_stmt = chunk_body[0]

        # Verify the mapping variable name (in unparsed form)
        # The runtime should be injected as a LocalAssignment
        assert hasattr(first_stmt, 'source')

    def test_runtime_generation(self):
        """Test that runtime code is generated correctly."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        array_id = "_arr_0"
        mapping = [2, 0, 1, 3]  # Sample mapping (0-based)

        runtime = transformer._generate_lua_runtime(array_id, mapping)

        # Check that runtime contains expected elements
        # Lua uses 1-based indexing, so keys and values are incremented by 1
        assert "local _arr_0_map" in runtime
        assert "[1] = 3" in runtime  # mapping[0] + 1 = 2 + 1 = 3
        assert "[2] = 1" in runtime  # mapping[1] + 1 = 0 + 1 = 1
        assert "[3] = 2" in runtime  # mapping[2] + 1 = 1 + 1 = 2
        assert "[4] = 4" in runtime  # mapping[3] + 1 = 3 + 1 = 4


class TestLuaASTTransformation:
    """Test full Lua AST transformation."""

    def test_lua_chunk_transformation(self):
        """Test transformation of a Lua chunk."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = """
local arr1 = {10, 20, 30}
local arr2 = {40, 50, 60}
print(arr1[1])
"""
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 2
        assert result.ast_node is not None

    def test_lua_table_in_function(self):
        """Test transformation of tables in functions."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = """
local function get_values()
    local data = {1, 2, 3, 4}
    return data
end
"""
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1

    def test_lua_table_with_string_constants(self):
        """Test transformation of tables with string constants."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "local x = {'a', 'b', 'c', 'd'}"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1
        assert len(result.errors) == 0

    def test_lua_table_with_boolean_constants(self):
        """Test transformation of tables with boolean constants."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "local x = {true, false, true}"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1
        assert len(result.errors) == 0


class TestLuaEdgeCases:
    """Test Lua-specific edge cases."""

    def test_empty_lua_module(self):
        """Test transformation of empty Lua module."""
        transformer = ConstantArrayTransformer()
        code = ""
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 0
        assert len(result.errors) == 0

    def test_lua_module_without_tables(self):
        """Test transformation of Lua module without tables."""
        transformer = ConstantArrayTransformer()
        code = """
local x = 1
local y = 2
local z = x + y
"""
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 0
        assert len(result.errors) == 0

    def test_lua_large_table_transformation(self):
        """Test transformation of large Lua tables."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        # Create table with 100 elements
        elements = ', '.join([str(i) for i in range(100)])
        code = f"local x = {{{elements}}}"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1
        assert len(result.errors) == 0

    def test_lua_table_with_duplicate_values(self):
        """Test that tables with duplicate values are handled correctly."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "local x = {1, 2, 1, 3, 2}"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1
        assert len(result.errors) == 0

    def test_lua_table_with_nil_values(self):
        """Test transformation of tables with nil values."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "local x = {1, nil, 3}"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1
        assert len(result.errors) == 0


class TestLuaTransformResult:
    """Test TransformResult structure for Lua."""

    def test_lua_transform_result_success(self):
        """Test successful Lua transformation result."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "local x = {1, 2, 3}"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.ast_node is not None
        assert result.transformation_count == 1
        assert len(result.errors) == 0

    def test_lua_transform_result_with_no_transformations(self):
        """Test Lua transformation result with no changes."""
        transformer = ConstantArrayTransformer()
        code = "local x = 1"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)

        assert result.success is True
        assert result.ast_node is not None
        assert result.transformation_count == 0
        assert len(result.errors) == 0


class TestLuaTraverseAST:
    """Test custom Lua AST traversal."""

    def test_traverse_lua_ast_with_tables(self):
        """Test that traversal finds and transforms table nodes."""
        transformer = ConstantArrayTransformer(shuffle_seed=42)
        code = "local x = {1, 2, 3}\nlocal y = {4, 5, 6}"
        tree = lua_ast.parse(code)

        transformer._traverse_lua_ast(tree)

        # Check that mappings were created
        assert len(transformer.array_mappings) == 2
        assert transformer.transformation_count == 2

    def test_traverse_lua_ast_without_tables(self):
        """Test that traversal handles code without tables."""
        transformer = ConstantArrayTransformer()
        code = "local x = 1\nlocal y = 2"
        tree = lua_ast.parse(code)

        transformer._traverse_lua_ast(tree)

        # No transformations should occur
        assert len(transformer.array_mappings) == 0
        assert transformer.transformation_count == 0


class TestLuaConfigurationIntegration:
    """Test integration with ObfuscationConfig for Lua."""

    def test_lua_with_config_seed(self):
        """Test Lua transformation using config-provided seed."""
        from obfuscator.core.config import ObfuscationConfig

        config = ObfuscationConfig(
            name="test",
            options={"array_shuffle_seed": 42}
        )
        transformer = ConstantArrayTransformer(config)
        code = "local x = {1, 2, 3}"

        tree = lua_ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert transformer.shuffle_seed == 42
        assert result.transformation_count == 1
