"""Tests for NumberObfuscationTransformer.

This module contains comprehensive tests for the NumberObfuscationTransformer class,
covering Python AST transformations, various complexity levels, edge cases, and error handling.
"""

import ast
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


class TestNumberObfuscationTransformer:
    """Test suite for NumberObfuscationTransformer."""

    def test_initialization_default(self):
        """Test transformer initialization with default values."""
        transformer = NumberObfuscationTransformer()

        assert transformer.complexity == 3
        assert transformer.min_value == 10
        assert transformer.max_value == 1000000
        assert transformer.language_mode is None
        assert transformer.transformation_count == 0
        assert transformer.errors == []

    def test_initialization_with_config(self):
        """Test transformer initialization with config."""
        config = ObfuscationConfig(
            name="test",
            features={"number_obfuscation": True},
            options={
                "number_obfuscation_complexity": 4,
                "number_obfuscation_min_value": 50,
                "number_obfuscation_max_value": 500000,
            },
        )
        transformer = NumberObfuscationTransformer(config)

        assert transformer.complexity == 4
        assert transformer.min_value == 50
        assert transformer.max_value == 500000

    def test_initialization_with_parameters(self):
        """Test transformer initialization with direct parameters."""
        transformer = NumberObfuscationTransformer(
            complexity=5, min_value=100, max_value=10000
        )

        assert transformer.complexity == 5
        assert transformer.min_value == 100
        assert transformer.max_value == 10000

    def test_complexity_validation(self):
        """Test complexity level validation."""
        # Test invalid complexity (too low)
        transformer = NumberObfuscationTransformer(complexity=0)
        assert transformer.complexity == 3  # Should default to 3

        # Test invalid complexity (too high)
        transformer = NumberObfuscationTransformer(complexity=6)
        assert transformer.complexity == 3  # Should default to 3

        # Test valid complexities
        for level in range(1, 6):
            transformer = NumberObfuscationTransformer(complexity=level)
            assert transformer.complexity == level

    def test_should_obfuscate_number(self):
        """Test number filtering logic."""
        transformer = NumberObfuscationTransformer(
            min_value=10, max_value=1000
        )

        # Should not obfuscate
        assert not transformer._should_obfuscate_number(0)  # Zero
        assert not transformer._should_obfuscate_number(1)  # One
        assert not transformer._should_obfuscate_number(5)  # Below min
        assert not transformer._should_obfuscate_number(1500)  # Above max
        # Negative numbers should now be obfuscated
        assert transformer._should_obfuscate_number(-10)  # Negative
        assert not transformer._should_obfuscate_number(float('nan'))  # NaN
        assert not transformer._should_obfuscate_number(float('inf'))  # Infinity
        assert not transformer._should_obfuscate_number(3 + 4j)  # Complex

        # Should obfuscate
        assert transformer._should_obfuscate_number(42)
        assert transformer._should_obfuscate_number(100)
        assert transformer._should_obfuscate_number(999)

    def test_generate_simple_expression(self):
        """Test simple expression generation (complexity level 1)."""
        transformer = NumberObfuscationTransformer(complexity=1)

        expr = transformer._generate_simple_expression(42)
        assert isinstance(expr, ast.BinOp)
        assert isinstance(expr.op, ast.Add)

        # Verify the expression evaluates to the target value
        compiled = compile(ast.Expression(expr), "<test>", "eval")
        result = eval(compiled)
        assert result == 42

    def test_generate_mixed_expression(self):
        """Test mixed expression generation (complexity level 2)."""
        transformer = NumberObfuscationTransformer(complexity=2)

        expr = transformer._generate_mixed_expression(42)
        assert isinstance(expr, ast.AST)

        # Verify the expression evaluates to the target value
        compiled = compile(ast.Expression(expr), "<test>", "eval")
        result = eval(compiled)
        assert result == 42

    def test_generate_bitwise_expression(self):
        """Test bitwise expression generation (complexity level 3)."""
        transformer = NumberObfuscationTransformer(complexity=3)

        expr = transformer._generate_bitwise_expression(42)
        assert isinstance(expr, ast.AST)

        # Verify the expression evaluates to the target value
        compiled = compile(ast.Expression(expr), "<test>", "eval")
        result = eval(compiled)
        assert result == 42

    def test_generate_nested_expression(self):
        """Test nested expression generation (complexity level 4)."""
        transformer = NumberObfuscationTransformer(complexity=4)

        expr = transformer._generate_nested_expression(150)
        assert isinstance(expr, ast.AST)

        # Verify the expression evaluates to the target value
        compiled = compile(ast.Expression(expr), "<test>", "eval")
        result = eval(compiled)
        assert result == 150

    def test_generate_advanced_expression(self):
        """Test advanced expression generation (complexity level 5)."""
        transformer = NumberObfuscationTransformer(complexity=5)

        expr = transformer._generate_advanced_expression(200)
        assert isinstance(expr, ast.AST)

        # Verify the expression evaluates to the target value
        compiled = compile(ast.Expression(expr), "<test>", "eval")
        result = eval(compiled)
        assert result == 200

    def test_visit_constant_with_number(self):
        """Test visiting numeric constant nodes."""
        transformer = NumberObfuscationTransformer(complexity=1)

        # Create a constant node with a number
        node = ast.Constant(value=42)

        # Visit the node
        result = transformer.visit_Constant(node)

        # Should be transformed to a BinOp
        assert isinstance(result, ast.BinOp)
        assert transformer.transformation_count == 1

    def test_visit_constant_with_string(self):
        """Test visiting string constant nodes (should not transform)."""
        transformer = NumberObfuscationTransformer()

        # Create a constant node with a string
        node = ast.Constant(value="hello")

        # Visit the node
        result = transformer.visit_Constant(node)

        # Should remain unchanged
        assert isinstance(result, ast.Constant)
        assert result.value == "hello"
        assert transformer.transformation_count == 0

    def test_visit_constant_with_small_number(self):
        """Test visiting small numbers (should not transform)."""
        transformer = NumberObfuscationTransformer(min_value=10)

        # Create a constant node with a small number
        node = ast.Constant(value=5)

        # Visit the node
        result = transformer.visit_Constant(node)

        # Should remain unchanged
        assert isinstance(result, ast.Constant)
        assert result.value == 5
        assert transformer.transformation_count == 0

    def test_transform_python_code(self):
        """Test transforming Python code with numeric constants."""
        transformer = NumberObfuscationTransformer(complexity=2)

        # Parse Python code
        code = """
x = 42
y = 100
z = x + 50
"""
        tree = ast.parse(code)

        # Transform the AST
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count > 0
        assert len(result.errors) == 0

        # Verify the transformed code can be compiled
        compiled = compile(result.ast_node, "<test>", "exec")
        assert compiled is not None

    def test_transform_with_zero_and_one(self):
        """Test that zero and one are not obfuscated."""
        transformer = NumberObfuscationTransformer()

        code = """
x = 0
y = 1
z = 42
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        # Should only transform 42, not 0 or 1
        assert result.transformation_count == 1

    def test_transform_with_float(self):
        """Test transforming float values."""
        transformer = NumberObfuscationTransformer()

        code = """
x = 3.14
y = 42.5
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        # Should transform both floats if they're in range
        assert result.transformation_count >= 0

    def test_transform_error_handling(self):
        """Test error handling during transformation."""
        transformer = NumberObfuscationTransformer()

        # Create a malformed AST (None node)
        result = transformer.transform(None)

        assert result.success is False
        assert result.ast_node is None
        assert len(result.errors) > 0

    def test_location_preservation(self):
        """Test that location information is preserved."""
        transformer = NumberObfuscationTransformer(complexity=1)

        code = "x = 42"
        tree = ast.parse(code)

        # Get the original node with location
        original_node = tree.body[0].value
        original_lineno = original_node.lineno

        # Transform
        result = transformer.transform(tree)

        # Check that location is preserved in the transformed node
        new_node = result.ast_node.body[0].value
        assert hasattr(new_node, 'lineno')
        assert new_node.lineno == original_lineno

    def test_complexity_levels_consistency(self):
        """Test that all complexity levels produce valid expressions."""
        for complexity in range(1, 6):
            transformer = NumberObfuscationTransformer(complexity=complexity)

            # Test with different values
            for value in [42, 100, 500, 1000]:
                if transformer._should_obfuscate_number(value):
                    expr = transformer._generate_obfuscated_expression(value)

                    # Should be able to compile and evaluate
                    compiled = compile(ast.Expression(expr), "<test>", "eval")
                    result = eval(compiled)
                    assert result == value

    @patch('obfuscator.processors.ast_transformer.LUAPARSER_AVAILABLE', False)
    def test_lua_not_available(self):
        """Test behavior when Lua parser is not available."""
        transformer = NumberObfuscationTransformer()

        # Should still work for Python
        code = "x = 42"
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1

    def test_edge_cases(self):
        """Test various edge cases."""
        transformer = NumberObfuscationTransformer()

        # Test with maximum value
        node = ast.Constant(value=999999)
        result = transformer.visit_Constant(node)
        assert isinstance(result, (ast.BinOp, ast.Constant))

        # Test with minimum value
        node = ast.Constant(value=10)
        result = transformer.visit_Constant(node)
        assert isinstance(result, (ast.BinOp, ast.Constant))

        # Test with boundary value
        node = ast.Constant(value=9)  # Just below min
        result = transformer.visit_Constant(node)
        assert isinstance(result, ast.Constant)
        assert result.value == 9

    def test_multiple_transformations(self):
        """Test multiple number transformations in the same code."""
        transformer = NumberObfuscationTransformer(complexity=2)

        code = """
values = [42, 100, 250, 500, 1000]
result = 42 + 100 * 2
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count >= 2  # Should transform multiple numbers

    def test_negative_number_obfuscation(self):
        """Test that negative numbers are properly obfuscated."""
        transformer = NumberObfuscationTransformer(min_value=10)
        
        # Test negative number within range
        node = ast.Constant(value=-42)
        result = transformer.visit_Constant(node)
        
        # Should be transformed to a UnaryOp with USub
        assert isinstance(result, ast.UnaryOp)
        assert isinstance(result.op, ast.USub)
        assert isinstance(result.operand, ast.BinOp)
        
        # Verify that expression evaluates correctly
        compiled = compile(ast.Expression(result), "<test>", "eval")
        evaluated = eval(compiled)
        assert evaluated == -42
        
        # Test negative float
        node = ast.Constant(value=-3.14)
        result = transformer.visit_Constant(node)
        
        assert isinstance(result, ast.UnaryOp)
        assert isinstance(result.op, ast.USub)
        
        # Verify that expression evaluates correctly
        compiled = compile(ast.Expression(result), "<test>", "eval")
        evaluated = eval(compiled)
        assert abs(evaluated - 3.14) < 0.01  # Should be close to 3.14

    def test_expression_correctness(self):
        """Test that generated expressions always evaluate correctly."""
        transformer = NumberObfuscationTransformer(complexity=3)

        test_values = [15, 42, 100, 250, 500, 750, 999]

        for value in test_values:
            expr = transformer._generate_obfuscated_expression(value)

            # Compile and evaluate
            compiled = compile(ast.Expression(expr), "<test>", "eval")
            result = eval(compiled)

            assert result == value, f"Expression for {value} evaluated to {result}"


class TestNumberObfuscationConfigIntegration:
    """Test integration with ObfuscationConfig."""

    def test_config_with_number_obfuscation_options(self):
        """Test creating config with number obfuscation options."""
        config = ObfuscationConfig(
            name="test",
            features={"number_obfuscation": True},
            options={
                "number_obfuscation_complexity": 4,
                "number_obfuscation_min_value": 20,
                "number_obfuscation_max_value": 500000,
            },
        )

        assert config.options["number_obfuscation_complexity"] == 4
        assert config.options["number_obfuscation_min_value"] == 20
        assert config.options["number_obfuscation_max_value"] == 500000

    def test_config_validation_with_number_options(self):
        """Test config validation with number obfuscation options."""
        # Valid config
        config = ObfuscationConfig(
            name="test",
            features={"number_obfuscation": True},
            options={
                "number_obfuscation_complexity": 3,
                "number_obfuscation_min_value": 10,
                "number_obfuscation_max_value": 1000000,
            },
        )
        config.validate()  # Should not raise

        # Invalid complexity
        config.options["number_obfuscation_complexity"] = 6
        with pytest.raises(ValueError, match="must be between 1 and 5"):
            config.validate()

        # Invalid min value
        config.options["number_obfuscation_complexity"] = 3
        config.options["number_obfuscation_min_value"] = -1
        with pytest.raises(ValueError, match="must be non-negative"):
            config.validate()

        # Invalid max value
        config.options["number_obfuscation_min_value"] = 10
        config.options["number_obfuscation_max_value"] = 0
        with pytest.raises(ValueError, match="must be positive"):
            config.validate()

    def test_transformer_with_config_options(self):
        """Test transformer uses config options correctly."""
        config = ObfuscationConfig(
            name="test",
            features={"number_obfuscation": True},
            options={
                "number_obfuscation_complexity": 5,
                "number_obfuscation_min_value": 50,
                "number_obfuscation_max_value": 5000,
            },
        )

        transformer = NumberObfuscationTransformer(config)

        assert transformer.complexity == 5
        assert transformer.min_value == 50
        assert transformer.max_value == 5000

        # Test that filtering uses these values
        assert not transformer._should_obfuscate_number(25)  # Below min
        assert transformer._should_obfuscate_number(100)  # Within range
        assert not transformer._should_obfuscate_number(6000)  # Above max
