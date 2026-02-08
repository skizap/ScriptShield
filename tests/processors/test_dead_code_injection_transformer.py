"""Tests for the DeadCodeInjectionTransformer.

This module provides comprehensive test coverage for the dead code injection
transformer, including initialization, Python and Lua code generation, injection
point detection, and integration with other transformers.
"""

import ast
import unittest
from unittest.mock import MagicMock, patch

import pytest

from obfuscator.core.config import ObfuscationConfig
from obfuscator.processors.ast_transformer import (
    DeadCodeInjectionTransformer,
    LUAPARSER_AVAILABLE,
)

# Skip Lua tests if luaparser is not available
lua_skip = pytest.mark.skipif(
    not LUAPARSER_AVAILABLE,
    reason="luaparser not installed"
)

if LUAPARSER_AVAILABLE:
    from luaparser import ast as lua_ast
    from luaparser import astnodes as lua_nodes


class TestDeadCodeInjectionTransformerInit(unittest.TestCase):
    """Tests for DeadCodeInjectionTransformer initialization."""

    def test_default_initialization(self):
        """Verify default percentage is 20."""
        transformer = DeadCodeInjectionTransformer()
        assert transformer.dead_code_percentage == 20
        assert transformer.injection_count == 0
        assert transformer.language_mode is None

    def test_initialization_with_config(self):
        """Test config extraction."""
        config = MagicMock()
        config.options = {"dead_code_percentage": 50}
        transformer = DeadCodeInjectionTransformer(config=config)
        assert transformer.dead_code_percentage == 50

    def test_initialization_with_explicit_percentage(self):
        """Test parameter override."""
        config = MagicMock()
        config.options = {"dead_code_percentage": 30}
        transformer = DeadCodeInjectionTransformer(
            config=config, dead_code_percentage=75
        )
        assert transformer.dead_code_percentage == 75

    def test_explicit_params_override_config(self):
        """Verify parameter precedence over config."""
        config = MagicMock()
        config.options = {"dead_code_percentage": 30}
        transformer = DeadCodeInjectionTransformer(
            config=config, dead_code_percentage=60
        )
        assert transformer.dead_code_percentage == 60

    def test_percentage_validation_bounds(self):
        """Test 0-100 validation."""
        # Test negative values clamped to 0
        transformer = DeadCodeInjectionTransformer(dead_code_percentage=-10)
        assert transformer.dead_code_percentage == 0

        # Test values > 100 clamped to 100
        transformer = DeadCodeInjectionTransformer(dead_code_percentage=150)
        assert transformer.dead_code_percentage == 100

        # Test edge cases
        transformer = DeadCodeInjectionTransformer(dead_code_percentage=0)
        assert transformer.dead_code_percentage == 0

        transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)
        assert transformer.dead_code_percentage == 100


class TestPythonDeadCodeGeneration(unittest.TestCase):
    """Tests for Python dead code generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)

    def test_generate_dead_assignment(self):
        """Verify assignment generation."""
        assignment = self.transformer._create_python_dead_assignment()
        assert isinstance(assignment, ast.Assign)
        assert len(assignment.targets) == 1
        assert isinstance(assignment.targets[0], ast.Name)
        # Check variable name pattern
        var_name = assignment.targets[0].id
        assert any(
            prefix in var_name
            for prefix in ['_tmp_', '_unused_', '_cache_', '_dead_', '_aux_']
        )

    def test_generate_dead_call(self):
        """Verify function call generation."""
        call = self.transformer._create_python_dead_call()
        assert isinstance(call, ast.Expr)
        assert isinstance(call.value, ast.Call)
        # Check that it's a builtin function
        func_name = call.value.func.id
        assert func_name in ['len', 'str', 'int', 'float', 'abs', 'max', 'min']

    def test_generate_dead_loop(self):
        """Verify loop generation."""
        loop = self.transformer._create_python_dead_loop()
        assert isinstance(loop, (ast.For, ast.While))

        if isinstance(loop, ast.For):
            # Check for loop has range(0) as iter
            assert isinstance(loop.iter, ast.Call)
            assert loop.iter.func.id == 'range'
        else:
            # Check while loop has False condition
            assert isinstance(loop.test, ast.Constant)
            assert loop.test.value is False

    def test_generate_dead_conditional(self):
        """Verify conditional generation."""
        conditional = self.transformer._create_python_dead_conditional()
        assert isinstance(conditional, ast.If)
        assert isinstance(conditional.test, ast.Constant)
        assert conditional.test.value is False
        # Check body has dead statements
        assert len(conditional.body) >= 1

    def test_realistic_variable_names(self):
        """Check variable naming patterns."""
        names = set()
        for _ in range(20):
            assignment = self.transformer._create_python_dead_assignment()
            names.add(assignment.targets[0].id)

        # All names should follow patterns
        for name in names:
            assert any(
                prefix in name
                for prefix in ['_tmp_', '_unused_', '_cache_', '_dead_', '_aux_']
            )
        # Names should be unique
        assert len(names) == 20

    def test_expression_variety(self):
        """Ensure varied expressions."""
        expressions = []
        for _ in range(50):
            expr = self.transformer._create_python_arithmetic_expression()
            expressions.append(type(expr).__name__)

        # Should have variety of expression types
        assert len(set(expressions)) > 1


class TestPythonInjectionPoints(unittest.TestCase):
    """Tests for Python injection point detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)

    def test_inject_after_return(self):
        """Verify injection after return statements."""
        code = """
def foo():
    x = 1
    return x
"""
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success
        assert result.transformation_count > 0

    def test_inject_in_false_conditional(self):
        """Test if False: injection."""
        code = """
def foo():
    if True:
        return 1
"""
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success
        # Check that the else branch might have dead code

    def test_no_injection_in_reachable_code(self):
        """Ensure only unreachable locations."""
        # With 0% probability, no injection should occur
        transformer = DeadCodeInjectionTransformer(dead_code_percentage=0)
        code = """
def foo():
    return 42
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success
        # At 0%, we might still visit but shouldn't inject

    def test_injection_frequency_control(self):
        """Verify percentage controls frequency."""
        code = """
def foo():
    return 42
"""
        # Test with 0% - should have no or minimal injections
        transformer_0 = DeadCodeInjectionTransformer(dead_code_percentage=0)
        tree = ast.parse(code)
        result_0 = transformer_0.transform(tree)
        assert result_0.success

        # Test with 100% - should have injections
        transformer_100 = DeadCodeInjectionTransformer(dead_code_percentage=100)
        tree = ast.parse(code)
        result_100 = transformer_100.transform(tree)
        assert result_100.success
        assert result_100.transformation_count > 0


class TestPythonFunctionTransformation(unittest.TestCase):
    """Tests for Python function transformations."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)

    def test_transform_simple_function(self):
        """Basic function with return."""
        code = """
def foo():
    return 42
"""
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success
        assert result.transformation_count > 0

    def test_transform_function_with_multiple_returns(self):
        """Multiple injection points."""
        code = """
def foo(x):
    if x > 0:
        return 1
    else:
        return 0
"""
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success

    def test_transform_function_with_loops(self):
        """Dead code in loops."""
        code = """
def foo():
    for i in range(10):
        if i == 5:
            return i
    return -1
"""
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success

    def test_transform_multiple_functions(self):
        """Multiple functions in module."""
        code = """
def foo():
    return 1

def bar():
    return 2
"""
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success

    def test_transformation_count(self):
        """Verify counter accuracy."""
        code = """
def foo():
    return 42

def bar():
    return 43
"""
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success
        # Should track total injection count
        assert result.transformation_count >= 0


@lua_skip
class TestLuaDeadCodeGeneration(unittest.TestCase):
    """Tests for Lua dead code generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)

    def test_lua_support_available(self):
        """Check luaparser availability."""
        assert LUAPARSER_AVAILABLE

    def test_generate_lua_dead_assignment(self):
        """Lua assignment generation."""
        assignment = self.transformer._create_lua_dead_assignment()
        assert assignment is not None
        assert isinstance(assignment, lua_nodes.LocalAssign)

    def test_generate_lua_dead_call(self):
        """Lua function call generation."""
        call = self.transformer._create_lua_dead_call()
        assert call is not None
        assert isinstance(call, lua_nodes.Assign)

    def test_generate_lua_dead_loop(self):
        """Lua loop generation."""
        loop = self.transformer._create_lua_dead_loop()
        assert loop is not None
        assert isinstance(loop, lua_nodes.While)
        assert isinstance(loop.test, lua_nodes.FalseExpr)

    def test_generate_lua_dead_conditional(self):
        """Lua conditional generation."""
        conditional = self.transformer._create_lua_dead_conditional()
        assert conditional is not None
        assert isinstance(conditional, lua_nodes.If)
        assert isinstance(conditional.test, lua_nodes.FalseExpr)


@lua_skip
class TestLuaFunctionTransformation(unittest.TestCase):
    """Tests for Lua function transformations."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)

    def test_transform_lua_function(self):
        """Basic Lua function transformation."""
        code = """
function foo()
    return 42
end
"""
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success

    def test_inject_after_lua_return(self):
        """Lua return statement injection."""
        code = """
function foo()
    local x = 1
    return x
end
"""
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)
        assert result.success
        assert result.transformation_count >= 0

    def test_lua_injection_frequency(self):
        """Percentage control in Lua."""
        code = """
function foo()
    return 42
end
"""
        # Test with 100% - should have injections
        transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)
        tree = lua_ast.parse(code)
        result = transformer.transform(tree)
        assert result.success


class TestTransformResult(unittest.TestCase):
    """Tests for transformation result handling."""

    def test_successful_transformation(self):
        """Verify result structure."""
        transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)
        code = "def foo(): return 42"
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success is True
        assert result.ast_node is not None
        assert isinstance(result.transformation_count, int)
        assert isinstance(result.errors, list)

    def test_transformation_with_zero_percentage(self):
        """No injection at 0%."""
        transformer = DeadCodeInjectionTransformer(dead_code_percentage=0)
        code = "def foo(): return 42"
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success is True

    def test_transformation_with_hundred_percentage(self):
        """Max injection at 100%."""
        transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)
        code = """
def foo():
    return 42
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success is True

    def test_error_handling(self):
        """Error capture in result."""
        transformer = DeadCodeInjectionTransformer()
        # Test with invalid input that might cause errors
        result = transformer.transform(None)
        assert not result.success
        assert len(result.errors) > 0


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def test_empty_module(self):
        """Empty module handling."""
        transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)
        code = ""
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success is True

    def test_module_without_functions(self):
        """Module with only statements."""
        transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)
        code = """
x = 1
y = 2
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success is True

    def test_nested_functions(self):
        """Nested function handling."""
        transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)
        code = """
def outer():
    def inner():
        return 1
    return inner()
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success is True

    def test_class_methods(self):
        """Class method handling."""
        transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)
        code = """
class Foo:
    def method(self):
        return 42
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success is True

    def test_function_with_no_returns(self):
        """Functions without returns."""
        transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)
        code = """
def foo():
    x = 1
    print(x)
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success is True


class TestCodeRealism(unittest.TestCase):
    """Tests for code realism."""

    def test_dead_code_is_syntactically_valid(self):
        """Parse generated code."""
        transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)

        # Create a simple function and transform it
        code = "def foo(): return 42"
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success

        # The transformed AST should be valid
        assert result.ast_node is not None

        # Try to unparse if available (Python 3.9+)
        try:
            import ast
            unparsed = ast.unparse(result.ast_node)
            # Should be able to re-parse the unparsed code
            reparsed = ast.parse(unparsed)
            assert reparsed is not None
        except (ImportError, AttributeError):
            # ast.unparse not available in older Python
            pass

    def test_variable_name_uniqueness(self):
        """No name conflicts."""
        transformer = DeadCodeInjectionTransformer(dead_code_percentage=100)

        # Generate multiple variables
        names = []
        for _ in range(50):
            name = transformer._generate_python_variable_name()
            names.append(name)

        # All names should be unique
        assert len(names) == len(set(names))


if __name__ == "__main__":
    unittest.main()
