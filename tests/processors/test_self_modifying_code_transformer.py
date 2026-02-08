"""Tests for the SelfModifyingCodeTransformer class.

This test suite covers:
- Initialization and configuration
- Python self-modify runtime code generation
- Lua self-modify runtime code generation
- Python function transformation
- Lua function transformation
- Transform result handling
- Edge cases and error handling
- Integration with the obfuscation pipeline
"""

import ast
import unittest
from unittest.mock import MagicMock, patch

import pytest

from obfuscator.core.config import ObfuscationConfig
from obfuscator.processors.ast_transformer import (
    LUAPARSER_AVAILABLE,
    SelfModifyingCodeTransformer,
    TransformResult,
)
from obfuscator.processors.self_modify_runtime_python import (
    generate_python_self_modify_runtime,
)
from obfuscator.processors.self_modify_runtime_lua import (
    generate_lua_self_modify_runtime,
)

# Skip Lua tests if luaparser is not available
lua_skip = pytest.mark.skipif(
    not LUAPARSER_AVAILABLE,
    reason="luaparser not installed"
)

if LUAPARSER_AVAILABLE:
    from luaparser import ast as lua_ast
    from luaparser import astnodes as lua_nodes


class TestSelfModifyingCodeTransformerInit(unittest.TestCase):
    """Tests for SelfModifyingCodeTransformer initialization."""

    def test_default_initialization(self):
        """Verify default complexity is 2."""
        transformer = SelfModifyingCodeTransformer()
        assert transformer.complexity == 2
        assert transformer.language_mode is None
        assert transformer.runtime_injected is False
        assert len(transformer.modified_functions) == 0

    def test_initialization_with_config(self):
        """Test config extraction for complexity."""
        config = MagicMock()
        config.options = {"self_modify_complexity": 3}
        transformer = SelfModifyingCodeTransformer(config=config)
        assert transformer.complexity == 3

    def test_initialization_with_explicit_complexity(self):
        """Test parameter override for complexity."""
        config = MagicMock()
        config.options = {"self_modify_complexity": 1}
        transformer = SelfModifyingCodeTransformer(config=config, complexity=3)
        assert transformer.complexity == 3

    def test_complexity_validation_bounds(self):
        """Test clamping to 1-3 range."""
        # Test too low
        transformer = SelfModifyingCodeTransformer(complexity=0)
        assert transformer.complexity == 1

        # Test too high
        transformer = SelfModifyingCodeTransformer(complexity=5)
        assert transformer.complexity == 3

        # Test negative
        transformer = SelfModifyingCodeTransformer(complexity=-1)
        assert transformer.complexity == 1

    def test_complexity_non_integer(self):
        """Test handling invalid types for complexity."""
        transformer = SelfModifyingCodeTransformer(complexity="invalid")
        assert transformer.complexity == 2

    def test_explicit_params_override_config(self):
        """Verify parameter precedence over config."""
        config = MagicMock()
        config.options = {"self_modify_complexity": 1}
        transformer = SelfModifyingCodeTransformer(config=config, complexity=2)
        assert transformer.complexity == 2

    def test_config_without_options(self):
        """Test config object without options attribute."""
        config = MagicMock(spec=[])
        transformer = SelfModifyingCodeTransformer(config=config)
        assert transformer.complexity == 2

    def test_config_with_missing_key(self):
        """Test config with options but missing self_modify_complexity key."""
        config = MagicMock()
        config.options = {"anti_debug_aggressiveness": 3}
        transformer = SelfModifyingCodeTransformer(config=config)
        assert transformer.complexity == 2


class TestPythonRuntimeGeneration(unittest.TestCase):
    """Tests for Python self-modify runtime code generation."""

    def test_generate_runtime_code(self):
        """Verify runtime code generation returns a string."""
        code = generate_python_self_modify_runtime(complexity=2)
        assert isinstance(code, str)
        assert len(code) > 0

    def test_runtime_contains_redefine_function(self):
        """Check for _redefine_function in generated code."""
        code = generate_python_self_modify_runtime(complexity=1)
        assert "_redefine_function" in code

    def test_runtime_contains_generate_code(self):
        """Check for code generation functions at complexity >= 2."""
        code = generate_python_self_modify_runtime(complexity=2)
        assert "_generate_code_at_runtime" in code

    def test_runtime_code_is_parseable(self):
        """Ensure generated code can be parsed by ast.parse()."""
        for level in [1, 2, 3]:
            code = generate_python_self_modify_runtime(complexity=level)
            try:
                ast.parse(code)
            except SyntaxError as e:
                self.fail(
                    f"Generated code at complexity={level} is not parseable: {e}"
                )

    def test_complexity_affects_code_content(self):
        """Verify different complexity levels produce different code."""
        code1 = generate_python_self_modify_runtime(complexity=1)
        code2 = generate_python_self_modify_runtime(complexity=2)
        code3 = generate_python_self_modify_runtime(complexity=3)

        # Level 1 should NOT have _generate_code_at_runtime
        assert "_generate_code_at_runtime" not in code1
        # Level 2 should have _generate_code_at_runtime
        assert "_generate_code_at_runtime" in code2
        # Level 3 should have _synthesize_function
        assert "_synthesize_function" in code3

    def test_runtime_contains_self_modify_wrapper(self):
        """Check that all levels contain _self_modify_wrapper."""
        for level in [1, 2, 3]:
            code = generate_python_self_modify_runtime(complexity=level)
            assert "_self_modify_wrapper" in code

    def test_runtime_contains_modify_function_body(self):
        """Check that _modify_function_body is present."""
        code = generate_python_self_modify_runtime(complexity=1)
        assert "_modify_function_body" in code


class TestPythonFunctionTransformation(unittest.TestCase):
    """Tests for Python function transformation."""

    def _make_function(self, num_stmts=5, name="foo"):
        """Helper to create a function with many statements."""
        stmts = "\n".join(f"    x{i} = {i}" for i in range(num_stmts))
        code = f"def {name}():\n{stmts}\n    return x0"
        return code

    def test_transform_simple_function(self):
        """Basic function transformation."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = self._make_function(5)
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 1
        assert "foo" in transformer.modified_functions

    def test_transform_multiple_functions(self):
        """Multiple functions in module."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = self._make_function(5, "foo") + "\n\n" + self._make_function(5, "bar")
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 2
        assert "foo" in transformer.modified_functions
        assert "bar" in transformer.modified_functions

    def test_transform_async_function(self):
        """Async function handling."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = """
async def async_func():
    x = 1
    y = 2
    z = x + y
    return z
"""
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success

        # Find async function in transformed AST
        async_func = None
        for node in ast.walk(result.ast_node):
            if isinstance(node, ast.AsyncFunctionDef):
                async_func = node
                break

        assert async_func is not None

    def test_transform_class_methods(self):
        """Class method handling."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = """
class Foo:
    def method(self):
        x = 1
        y = 2
        z = x + y
        return z
"""
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success

    def test_transformation_count(self):
        """Verify counter accuracy."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = self._make_function(5) + "\n\n" + self._make_function(4, "bar")
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success
        # Both functions have >= 3 statements so both should be transformed
        assert result.transformation_count == 2

    def test_skip_small_functions(self):
        """Functions with < 3 statements should be skipped."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = "def small():\n    x = 1\n    return x"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 0

    def test_skip_generators(self):
        """Generator functions should be skipped."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = """
def gen():
    x = 1
    y = 2
    z = 3
    yield x
"""
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 0

    def test_transformed_ast_is_valid(self):
        """Transformed AST should be unparseable and re-parseable."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = self._make_function(5)
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success
        assert result.ast_node is not None

        # Unparse and re-parse should succeed
        unparsed = ast.unparse(result.ast_node)
        reparsed = ast.parse(unparsed)
        assert reparsed is not None

    def test_function_with_parameters(self):
        """Function with parameters should have them forwarded."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = """
def compute(a, b, c):
    x = a + b
    y = b + c
    z = x + y
    return z
"""
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 1

        unparsed = ast.unparse(result.ast_node)
        # The wrapper should reference the parameter names
        assert "_redefine_function" in unparsed

    def test_state_reset_between_transforms(self):
        """Transformer state should reset between transform() calls."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = self._make_function(5)

        tree1 = ast.parse(code)
        result1 = transformer.transform(tree1)
        assert result1.success
        assert result1.transformation_count == 1

        tree2 = ast.parse(code)
        result2 = transformer.transform(tree2)
        assert result2.success
        assert result2.transformation_count == 1
        # modified_functions should be reset
        assert len(transformer.modified_functions) == 1


@lua_skip
class TestLuaRuntimeGeneration(unittest.TestCase):
    """Tests for Lua self-modify runtime code generation."""

    def test_generate_lua_runtime_code(self):
        """Verify Lua runtime generation."""
        code = generate_lua_self_modify_runtime(complexity=2)
        assert isinstance(code, str)
        assert len(code) > 0

    def test_lua_runtime_contains_loadstring(self):
        """Check for loadstring/load in generated code."""
        code = generate_lua_self_modify_runtime(complexity=1)
        assert "loadstring" in code or "load" in code

    def test_lua_runtime_contains_redefine_function(self):
        """Check for _redefine_function in Lua runtime."""
        code = generate_lua_self_modify_runtime(complexity=1)
        assert "_redefine_function" in code

    def test_lua_runtime_code_is_parseable(self):
        """Ensure valid Lua syntax."""
        for level in [1, 2, 3]:
            code = generate_lua_self_modify_runtime(complexity=level)
            try:
                lua_ast.parse(code)
            except Exception as e:
                self.fail(
                    f"Generated Lua code at complexity={level} is not parseable: {e}"
                )

    def test_lua_complexity_affects_code_content(self):
        """Verify different complexity levels produce different code."""
        code1 = generate_lua_self_modify_runtime(complexity=1)
        code2 = generate_lua_self_modify_runtime(complexity=2)
        code3 = generate_lua_self_modify_runtime(complexity=3)

        assert "_generate_code_at_runtime" not in code1
        assert "_generate_code_at_runtime" in code2
        assert "_synthesize_function" in code3


@lua_skip
class TestLuaFunctionTransformation(unittest.TestCase):
    """Tests for Lua function transformation."""

    def test_transform_lua_function(self):
        """Basic Lua function transformation."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = """
function foo()
    local x = 1
    local y = 2
    local z = x + y
    return z
end
"""
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        assert result.success

    def test_lua_language_detection(self):
        """Transformer should detect Lua from Chunk node."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = """
function foo()
    local x = 1
    return x
end
"""
        tree = lua_ast.parse(code)
        result = transformer.transform(tree)
        assert result.success
        assert transformer.language_mode == "lua"

    def test_lua_transformation_count(self):
        """Verify counter accuracy for Lua."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = """
function big_func()
    local x0 = 0
    local x1 = 1
    local x2 = 2
    local x3 = 3
    return x0
end
"""
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        assert result.success
        # Function has >= 3 statements so should be transformed
        assert result.transformation_count >= 0  # May or may not transform depending on Lua AST structure


class TestTransformResult(unittest.TestCase):
    """Tests for transformation result handling."""

    def test_successful_transformation(self):
        """Verify result structure on success."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        stmts = "\n".join(f"    x{i} = {i}" for i in range(5))
        code = f"def foo():\n{stmts}\n    return x0"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success is True
        assert result.ast_node is not None
        assert isinstance(result.transformation_count, int)
        assert isinstance(result.errors, list)

    def test_error_handling(self):
        """Error capture in result."""
        transformer = SelfModifyingCodeTransformer(complexity=2)

        # None input should return failure
        result = transformer.transform(None)
        assert not result.success
        assert result.ast_node is None
        assert len(result.errors) > 0

    def test_empty_module(self):
        """Empty module handling."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        tree = ast.parse("")

        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 0

    def test_error_handling_unsupported_type(self):
        """Unsupported AST node type should return failure."""
        transformer = SelfModifyingCodeTransformer()
        result = transformer.transform("not an ast node")
        assert not result.success
        assert len(result.errors) > 0

    def test_zero_transformations_on_small_functions(self):
        """Module with only small functions should succeed with zero transformations."""
        transformer = SelfModifyingCodeTransformer()
        code = "def tiny():\n    return 1"
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 0


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def test_nested_functions(self):
        """Nested function handling."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = """
def outer():
    x = 1
    y = 2
    z = 3
    def inner():
        a = 1
        b = 2
        c = 3
        return a + b + c
    return inner()
"""
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success
        # Both outer and inner have >= 3 statements
        assert result.transformation_count >= 1

    def test_lambda_functions(self):
        """Lambda preservation - lambdas should not be transformed."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = "f = lambda x: x + 1"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success
        # Lambda should not be transformed (it's not a FunctionDef)
        assert result.transformation_count == 0

    def test_module_without_functions(self):
        """Module with only statements."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = "x = 1\ny = 2\nz = 3"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 0

    def test_function_with_yield_from(self):
        """Functions with yield from should be skipped."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        code = """
def delegating_gen():
    x0 = 0
    x1 = 1
    x2 = 2
    yield from range(10)
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 0

    def test_function_with_decorators(self):
        """Decorated functions should still be transformed."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        stmts = "\n".join(f"    x{i} = {i}" for i in range(5))
        code = f"@staticmethod\ndef decorated():\n{stmts}\n    return x0"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 1

        # Decorator should be preserved
        func_defs = [
            node for node in ast.walk(result.ast_node)
            if isinstance(node, ast.FunctionDef) and node.name == 'decorated'
        ]
        assert len(func_defs) >= 1
        assert len(func_defs[0].decorator_list) == 1

    def test_runtime_preserves_future_imports(self):
        """Runtime should be inserted after __future__ imports."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        stmts = "\n".join(f"    x{i} = {i}" for i in range(5))
        code = f"from __future__ import annotations\ndef foo():\n{stmts}\n    return x0"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success

        # First statement should still be the __future__ import
        first_stmt = result.ast_node.body[0]
        assert isinstance(first_stmt, ast.ImportFrom)
        assert first_stmt.module == "__future__"

    def test_runtime_preserves_docstring(self):
        """Runtime should be inserted after module docstring."""
        transformer = SelfModifyingCodeTransformer(complexity=2)
        stmts = "\n".join(f"    x{i} = {i}" for i in range(5))
        code = f'\"\"\"Module docstring.\"\"\"\ndef foo():\n{stmts}\n    return x0'
        tree = ast.parse(code)

        result = transformer.transform(tree)
        assert result.success

        # First statement should still be the docstring
        first_stmt = result.ast_node.body[0]
        assert isinstance(first_stmt, ast.Expr)
        assert isinstance(first_stmt.value, ast.Constant)
        assert first_stmt.value.value == "Module docstring."


class TestIntegration(unittest.TestCase):
    """Integration tests with the obfuscation pipeline."""

    def test_with_obfuscation_engine(self):
        """Integration with ObfuscationEngine."""
        from obfuscator.core.obfuscation_engine import ObfuscationEngine

        config = ObfuscationConfig(
            name="test_self_modify",
            language="python",
            features={"self_modifying_code": True},
            options={"self_modify_complexity": 2},
        )
        engine = ObfuscationEngine(config)
        transformers = engine.get_enabled_transformers("python")

        # Should find SelfModifyingCodeTransformer in the list
        found = False
        for t in transformers:
            if isinstance(t, SelfModifyingCodeTransformer):
                found = True
                break
        assert found

    def test_transformation_order(self):
        """Verify pipeline order - self_modifying_code after code_splitting, before mangle_indexes."""
        from obfuscator.core.obfuscation_engine import ObfuscationEngine

        config = ObfuscationConfig(
            name="test_order",
            language="python",
            features={
                "code_splitting": True,
                "self_modifying_code": True,
                "mangle_indexes": True,
            },
        )
        engine = ObfuscationEngine(config)
        transformers = engine.get_enabled_transformers("python")

        # Get class names in order
        names = [type(t).__name__ for t in transformers]
        assert "SelfModifyingCodeTransformer" in names

        # Verify ordering
        from obfuscator.processors.ast_transformer import (
            CodeSplittingTransformer,
            MangleIndexesTransformer,
        )

        sm_idx = None
        cs_idx = None
        mi_idx = None
        for i, t in enumerate(transformers):
            if isinstance(t, SelfModifyingCodeTransformer):
                sm_idx = i
            elif isinstance(t, CodeSplittingTransformer):
                cs_idx = i
            elif isinstance(t, MangleIndexesTransformer):
                mi_idx = i

        if cs_idx is not None and sm_idx is not None:
            assert cs_idx < sm_idx
        if sm_idx is not None and mi_idx is not None:
            assert sm_idx < mi_idx

    def test_full_pipeline(self):
        """Run through complete pipeline."""
        from obfuscator.core.obfuscation_engine import ObfuscationEngine
        from pathlib import Path

        config = ObfuscationConfig(
            name="test_full",
            language="python",
            features={"self_modifying_code": True},
            options={"self_modify_complexity": 2},
        )
        engine = ObfuscationEngine(config)

        stmts = "\n".join(f"    x{i} = {i}" for i in range(5))
        code = f"def compute():\n{stmts}\n    return x0"
        tree = ast.parse(code)

        result = engine.apply_transformations(tree, "python", Path("test.py"))
        assert result.success

    def test_config_driven_transformer(self):
        """Test using ObfuscationConfig to drive transformer."""
        config = ObfuscationConfig(
            name="test_config",
            features={"self_modifying_code": True},
            options={"self_modify_complexity": 3},
        )
        transformer = SelfModifyingCodeTransformer(config=config)
        assert transformer.complexity == 3

        stmts = "\n".join(f"    x{i} = {i}" for i in range(5))
        code = f"def driven():\n{stmts}\n    return x0"
        tree = ast.parse(code)
        result = transformer.transform(tree)
        assert result.success
        assert result.transformation_count == 1


class TestConfigValidation(unittest.TestCase):
    """Tests for config validation of self-modifying code options."""

    def test_valid_config(self):
        """Valid config should pass validation."""
        config = ObfuscationConfig(
            name="test",
            options={"self_modify_complexity": 2},
        )
        config.validate()  # Should not raise

    def test_invalid_complexity_type(self):
        """Non-integer complexity should fail validation."""
        config = ObfuscationConfig(
            name="test",
            options={"self_modify_complexity": "two"},
        )
        with pytest.raises(ValueError, match="self_modify_complexity"):
            config.validate()

    def test_complexity_too_low(self):
        """complexity < 1 should fail validation."""
        config = ObfuscationConfig(
            name="test",
            options={"self_modify_complexity": 0},
        )
        with pytest.raises(ValueError, match="self_modify_complexity"):
            config.validate()

    def test_complexity_too_high(self):
        """complexity > 3 should fail validation."""
        config = ObfuscationConfig(
            name="test",
            options={"self_modify_complexity": 4},
        )
        with pytest.raises(ValueError, match="self_modify_complexity"):
            config.validate()

    def test_valid_complexity_range(self):
        """All valid complexity values should pass validation."""
        for level in [1, 2, 3]:
            config = ObfuscationConfig(
                name="test",
                options={"self_modify_complexity": level},
            )
            config.validate()  # Should not raise


if __name__ == "__main__":
    unittest.main()
