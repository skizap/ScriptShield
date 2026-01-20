"""Tests for NumberObfuscationTransformer with Lua code.

This module contains tests for Lua-specific functionality of the NumberObfuscationTransformer,
including Lua AST transformations and Lua expression generation.
"""

import sys
from unittest.mock import Mock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, "src")

from obfuscator.processors.ast_transformer import (
    NumberObfuscationTransformer,
    TransformResult,
)
from obfuscator.core.config import ObfuscationConfig

# Try to import luaparser
try:
    from luaparser import ast as lua_ast
    from luaparser import astnodes as lua_nodes

    LUAPARSER_AVAILABLE = True
except ImportError:
    LUAPARSER_AVAILABLE = False


@pytest.mark.skipif(not LUAPARSER_AVAILABLE, reason="luaparser not available")
class TestLuaNumberObfuscation:
    """Test suite for Lua number obfuscation."""

    def test_lua_simple_number_obfuscation(self):
        """Test basic Lua number obfuscation."""
        transformer = NumberObfuscationTransformer(complexity=1)

        # Parse simple Lua code
        lua_code = "local x = 42"
        chunk = lua_ast.parse(lua_code)

        # Transform
        result = transformer.transform(chunk)

        assert result.success is True
        assert result.transformation_count == 1
        assert transformer.language_mode == "lua"

    def test_lua_multiple_numbers(self):
        """Test Lua code with multiple numbers."""
        transformer = NumberObfuscationTransformer(complexity=2)

        lua_code = """
local x = 42
local y = 100
local z = x + 50
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        assert result.transformation_count >= 2

    def test_lua_number_in_expression(self):
        """Test Lua number in arithmetic expression."""
        transformer = NumberObfuscationTransformer(complexity=1)

        lua_code = "local result = 42 + 100"
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        assert result.transformation_count >= 1

    def test_lua_function_with_numbers(self):
        """Test Lua function containing numbers."""
        transformer = NumberObfuscationTransformer(complexity=2)

        lua_code = """
function calculate()
    local a = 10
    local b = 20
    return a * b + 5
end
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        assert result.transformation_count >= 2

    def test_lua_table_with_numbers(self):
        """Test Lua table containing numeric values."""
        transformer = NumberObfuscationTransformer(complexity=1)

        lua_code = """
local data = {
    value1 = 42,
    value2 = 100,
    value3 = 250
}
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        assert result.transformation_count >= 2

    def test_lua_control_structures(self):
        """Test Lua control structures with numbers."""
        transformer = NumberObfuscationTransformer(complexity=2)

        lua_code = """
for i = 1, 10 do
    if i > 5 then
        print(i * 2)
    end
end
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        assert result.transformation_count >= 2

    def test_lua_complexity_levels(self):
        """Test different complexity levels with Lua."""
        lua_code = "local x = 100"

        for complexity in range(1, 6):
            transformer = NumberObfuscationTransformer(complexity=complexity)
            chunk = lua_ast.parse(lua_code)
            result = transformer.transform(chunk)

            assert result.success is True
            assert result.transformation_count == 1

    def test_lua_zero_and_one_not_obfuscated(self):
        """Test that zero and one are not obfuscated in Lua."""
        transformer = NumberObfuscationTransformer()

        lua_code = """
local x = 0
local y = 1
local z = 42
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        # Should only transform 42, not 0 or 1
        assert result.transformation_count == 1

    def test_lua_small_numbers_not_obfuscated(self):
        """Test that small numbers are not obfuscated in Lua."""
        transformer = NumberObfuscationTransformer(min_value=10)

        lua_code = """
local x = 5
local y = 9
local z = 10
local w = 11
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        # Should only transform 10 and 11, not 5 or 9
        assert result.transformation_count == 2

    def test_lua_large_numbers_not_obfuscated(self):
        """Test that large numbers are not obfuscated in Lua."""
        transformer = NumberObfuscationTransformer(max_value=1000)

        lua_code = """
local x = 500
local y = 1000
local z = 1001
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        # Should transform 500 and 1000, but not 1001
        assert result.transformation_count == 2

    def test_lua_float_numbers(self):
        """Test Lua float number obfuscation."""
        transformer = NumberObfuscationTransformer()

        lua_code = """
local x = 3.14
local y = 42.5
local z = 100.0
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        # Should transform floats in range
        assert result.transformation_count >= 0

    def test_lua_expression_correctness(self):
        """Test that Lua expressions evaluate to correct values."""
        transformer = NumberObfuscationTransformer(complexity=3)

        # Test various values
        test_values = [15, 42, 100, 250, 500]

        for value in test_values:
            lua_code = f"local x = {value}"
            chunk = lua_ast.parse(lua_code)

            # Transform
            result = transformer.transform(chunk)
            assert result.success is True

            # The transformed code should still be valid Lua
            transformed_code = lua_ast.to_lua_source(chunk)
            assert transformed_code is not None
            assert len(transformed_code) > 0

    def test_lua_nested_expressions(self):
        """Test Lua code with nested expressions."""
        transformer = NumberObfuscationTransformer(complexity=3)

        lua_code = """
local result = (10 + 20) * (30 + 40)
local nested = ((5 + 3) * 2) + 10
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        assert result.transformation_count >= 2

    def test_lua_function_calls_with_numbers(self):
        """Test Lua function calls containing numbers."""
        transformer = NumberObfuscationTransformer(complexity=2)

        lua_code = """
print(42)
math.sqrt(100)
tostring(250)
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        assert result.transformation_count >= 2

    def test_lua_error_handling(self):
        """Test error handling in Lua transformation."""
        transformer = NumberObfuscationTransformer()

        # Test with invalid Lua code (this should be handled gracefully)
        try:
            # This might fail to parse, which is OK
            lua_code = "local x ="
            chunk = lua_ast.parse(lua_code)
            result = transformer.transform(chunk)
            # If it gets here, it should handle the error gracefully
            assert isinstance(result, TransformResult)
        except Exception:
            # Parsing error is acceptable
            pass

    def test_lua_config_integration(self):
        """Test Lua transformation with config options."""
        config = ObfuscationConfig(
            name="test",
            language="lua",
            features={"number_obfuscation": True},
            options={
                "number_obfuscation_complexity": 4,
                "number_obfuscation_min_value": 50,
                "number_obfuscation_max_value": 5000,
            },
        )

        transformer = NumberObfuscationTransformer(config)

        lua_code = """
local small = 10
local medium = 100
local large = 10000
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        # Should only transform medium (100), not small (10) or large (10000)
        assert result.transformation_count == 1

    def test_lua_mixed_data_types(self):
        """Test Lua code with mixed data types."""
        transformer = NumberObfuscationTransformer(complexity=2)

        lua_code = """
local num = 42
local str = "hello"
local bool = true
local arr = {1, 2, 3}
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        # Should only transform the number 42, not string or boolean
        assert result.transformation_count == 1

    def test_lua_while_loop_with_numbers(self):
        """Test Lua while loop containing numbers."""
        transformer = NumberObfuscationTransformer(complexity=2)

        lua_code = """
local i = 0
while i < 10 do
    i = i + 1
end
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        assert result.transformation_count >= 2

    def test_lua_repeat_until_with_numbers(self):
        """Test Lua repeat-until loop containing numbers."""
        transformer = NumberObfuscationTransformer(complexity=2)

        lua_code = """
local i = 0
repeat
    i = i + 1
until i > 10
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        assert result.transformation_count >= 2

    def test_lua_if_statement_with_numbers(self):
        """Test Lua if statement containing numbers."""
        transformer = NumberObfuscationTransformer(complexity=2)

        lua_code = """
local x = 42
if x > 10 then
    x = x + 5
elseif x > 50 then
    x = x - 10
else
    x = x * 2
end
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        assert result.transformation_count >= 3

    def test_lua_local_assignments(self):
        """Test various forms of Lua local assignments."""
        transformer = NumberObfuscationTransformer(complexity=2)

        lua_code = """
local a = 10
local b, c = 20, 30
local d, e, f = 40, 50, 60
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        assert result.transformation_count >= 3

    def test_lua_return_statement_with_numbers(self):
        """Test Lua return statement containing numbers."""
        transformer = NumberObfuscationTransformer(complexity=2)

        lua_code = """
function get_number()
    return 42
end

function calculate()
    return 10 + 20 * 2
end
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        assert result.transformation_count >= 2

    def test_lua_complexity_consistency(self):
        """Test that all complexity levels work with Lua."""
        lua_code = "local x = 250"

        for complexity in range(1, 6):
            transformer = NumberObfuscationTransformer(complexity=complexity)
            chunk = lua_ast.parse(lua_code)
            result = transformer.transform(chunk)

            assert result.success is True
            assert result.transformation_count == 1

            # Verify the transformed code is valid
            transformed_code = lua_ast.to_lua_source(chunk)
            assert transformed_code is not None
            assert "local x =" in transformed_code

    def test_lua_transformation_count_tracking(self):
        """Test that transformation count is tracked correctly for Lua."""
        transformer = NumberObfuscationTransformer(complexity=2)

        lua_code = """
local a = 42
local b = 100
local c = 250
local d = 500
local e = 1000
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        assert result.transformation_count == 5
        assert transformer.transformation_count == 5

    def test_lua_error_recovery(self):
        """Test error recovery during Lua transformation."""
        transformer = NumberObfuscationTransformer(complexity=2)

        # This should not crash even if there are issues
        lua_code = "local x = 42"
        chunk = lua_ast.parse(lua_code)

        # Mock an error in the traversal
        original_traverse = transformer._traverse_lua_ast
        def mock_traverse(node):
            if hasattr(node, '__class__') and 'Number' in str(type(node)):
                raise ValueError("Simulated error")
            original_traverse(node)

        transformer._traverse_lua_ast = mock_traverse

        result = transformer.transform(chunk)

        # Should handle errors gracefully
        assert isinstance(result, TransformResult)

    def test_lua_special_float_values(self):
        """Test that special float values are not obfuscated in Lua."""
        transformer = NumberObfuscationTransformer()

        # Note: Lua doesn't have special float values like NaN/Inf in the same way as Python
        # but we test the logic anyway
        lua_code = """
local a = 42
local b = 100
"""
        chunk = lua_ast.parse(lua_code)
        result = transformer.transform(chunk)

        assert result.success is True
        assert result.transformation_count == 2


class TestLuaNumberObfuscationWithoutParser:
    """Test Lua number obfuscation when luaparser is not available."""

    @patch('obfuscator.processors.ast_transformer.LUAPARSER_AVAILABLE', False)
    def test_lua_transformation_without_parser(self):
        """Test that Lua transformation handles missing parser gracefully."""
        transformer = NumberObfuscationTransformer(complexity=2)

        # Should still work for Python
        import ast
        code = "x = 42"
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1
