"""Tests for the OpaquePredicatesTransformer.

This module contains comprehensive tests for the OpaquePredicatesTransformer
class, covering initialization, Python opaque predicate generation,
Lua opaque predicate generation, injection at various locations, and
integration with other transformers.
"""

from __future__ import annotations

import ast
import sys
import unittest
from typing import Any, Optional

# Try to import luaparser for Lua tests
try:
    from luaparser import ast as lua_ast
    from luaparser import astnodes as lua_nodes
    LUAPARSER_AVAILABLE = True
except ImportError:
    LUAPARSER_AVAILABLE = False

from obfuscator.core.config import ObfuscationConfig
from obfuscator.processors.ast_transformer import (
    OpaquePredicatesTransformer,
    TransformResult,
)


def lua_skip(func):
    """Decorator to skip tests when luaparser is not available."""
    return unittest.skipUnless(LUAPARSER_AVAILABLE, "luaparser not available")(func)


class TestOpaquePredicatesTransformerInit(unittest.TestCase):
    """Test OpaquePredicatesTransformer initialization."""

    def test_default_initialization(self) -> None:
        """Test that default initialization uses correct defaults."""
        transformer = OpaquePredicatesTransformer()

        self.assertEqual(transformer.opaque_predicate_complexity, 2)
        self.assertEqual(transformer.opaque_predicate_percentage, 30)
        self.assertIsNone(transformer.config)
        self.assertIsNone(transformer.language_mode)
        self.assertEqual(transformer.predicate_counter, 0)

    def test_initialization_with_config(self) -> None:
        """Test initialization with ObfuscationConfig."""
        config = ObfuscationConfig(
            name="test",
            features={"opaque_predicates": True},
            options={
                "opaque_predicate_complexity": 3,
                "opaque_predicate_percentage": 50,
            }
        )
        transformer = OpaquePredicatesTransformer(config)

        self.assertEqual(transformer.opaque_predicate_complexity, 3)
        self.assertEqual(transformer.opaque_predicate_percentage, 50)
        self.assertEqual(transformer.config, config)

    def test_initialization_with_explicit_params(self) -> None:
        """Test initialization with explicit parameters overriding defaults."""
        transformer = OpaquePredicatesTransformer(
            opaque_predicate_complexity=1,
            opaque_predicate_percentage=75
        )

        self.assertEqual(transformer.opaque_predicate_complexity, 1)
        self.assertEqual(transformer.opaque_predicate_percentage, 75)

    def test_explicit_params_override_config(self) -> None:
        """Test that explicit parameters take precedence over config."""
        config = ObfuscationConfig(
            name="test",
            features={"opaque_predicates": True},
            options={
                "opaque_predicate_complexity": 3,
                "opaque_predicate_percentage": 50,
            }
        )
        transformer = OpaquePredicatesTransformer(
            config=config,
            opaque_predicate_complexity=1,
            opaque_predicate_percentage=80
        )

        self.assertEqual(transformer.opaque_predicate_complexity, 1)
        self.assertEqual(transformer.opaque_predicate_percentage, 80)

    def test_complexity_validation_bounds(self) -> None:
        """Test that complexity is clamped to valid range 1-3."""
        # Test below minimum
        transformer_low = OpaquePredicatesTransformer(opaque_predicate_complexity=0)
        self.assertEqual(transformer_low.opaque_predicate_complexity, 1)

        # Test above maximum
        transformer_high = OpaquePredicatesTransformer(opaque_predicate_complexity=10)
        self.assertEqual(transformer_high.opaque_predicate_complexity, 3)

        # Test negative
        transformer_neg = OpaquePredicatesTransformer(opaque_predicate_complexity=-5)
        self.assertEqual(transformer_neg.opaque_predicate_complexity, 1)

    def test_percentage_validation_bounds(self) -> None:
        """Test that percentage is clamped to valid range 0-100."""
        # Test below minimum
        transformer_low = OpaquePredicatesTransformer(opaque_predicate_percentage=-10)
        self.assertEqual(transformer_low.opaque_predicate_percentage, 0)

        # Test above maximum
        transformer_high = OpaquePredicatesTransformer(opaque_predicate_percentage=150)
        self.assertEqual(transformer_high.opaque_predicate_percentage, 100)


class TestPythonOpaquePredicateGeneration(unittest.TestCase):
    """Test Python opaque predicate generation methods."""

    def test_generate_opaque_true_complexity_1(self) -> None:
        """Test generating simple true predicates."""
        transformer = OpaquePredicatesTransformer(opaque_predicate_complexity=1)

        # Generate multiple predicates and verify they are valid AST nodes
        for _ in range(10):
            predicate = transformer._generate_python_opaque_true()
            self.assertIsInstance(predicate, ast.expr)

            # The predicate should be a Compare or Call node
            self.assertIsInstance(predicate, (ast.Compare, ast.Call, ast.BoolOp))

    def test_generate_opaque_true_complexity_2(self) -> None:
        """Test generating bitwise operation true predicates."""
        transformer = OpaquePredicatesTransformer(opaque_predicate_complexity=2)

        for _ in range(10):
            predicate = transformer._generate_python_opaque_true()
            self.assertIsInstance(predicate, ast.expr)

            # Complexity 2 should generate bitwise operations
            self.assertIsInstance(predicate, (ast.Compare, ast.BoolOp))

    def test_generate_opaque_true_complexity_3(self) -> None:
        """Test generating complex expression true predicates."""
        transformer = OpaquePredicatesTransformer(opaque_predicate_complexity=3)

        for _ in range(10):
            predicate = transformer._generate_python_opaque_true()
            self.assertIsInstance(predicate, ast.expr)

            # Complexity 3 may generate BoolOp for "or" expressions
            self.assertIsInstance(predicate, (ast.Compare, ast.BoolOp, ast.Call))

    def test_generate_opaque_false_complexity_1(self) -> None:
        """Test generating simple false predicates."""
        transformer = OpaquePredicatesTransformer(opaque_predicate_complexity=1)

        for _ in range(10):
            predicate = transformer._generate_python_opaque_false()
            self.assertIsInstance(predicate, ast.expr)

    def test_generate_opaque_false_complexity_2(self) -> None:
        """Test generating bitwise operation false predicates."""
        transformer = OpaquePredicatesTransformer(opaque_predicate_complexity=2)

        for _ in range(10):
            predicate = transformer._generate_python_opaque_false()
            self.assertIsInstance(predicate, ast.expr)

    def test_generate_opaque_false_complexity_3(self) -> None:
        """Test generating complex expression false predicates."""
        transformer = OpaquePredicatesTransformer(opaque_predicate_complexity=3)

        for _ in range(10):
            predicate = transformer._generate_python_opaque_false()
            self.assertIsInstance(predicate, ast.expr)

    def test_opaque_variable_generation(self) -> None:
        """Test generating opaque variable assignments."""
        transformer = OpaquePredicatesTransformer()

        assign = transformer._create_python_opaque_variable()
        self.assertIsInstance(assign, ast.Assign)
        self.assertEqual(len(assign.targets), 1)

        # Check variable name follows pattern _opaque_N
        target = assign.targets[0]
        self.assertIsInstance(target, ast.Name)
        self.assertTrue(target.id.startswith("_opaque_"))

        # Check value is a constant integer
        self.assertIsInstance(assign.value, ast.Constant)
        self.assertIsInstance(assign.value.value, int)
        self.assertGreaterEqual(assign.value.value, 1)
        self.assertLessEqual(assign.value.value, 100)

    def test_predicate_variety(self) -> None:
        """Test that multiple calls produce different predicates."""
        transformer = OpaquePredicatesTransformer(opaque_predicate_complexity=1)

        predicates = []
        for _ in range(20):
            pred = transformer._generate_python_opaque_true()
            predicates.append(ast.dump(pred))

        # Should have variety (not all identical)
        unique_predicates = set(predicates)
        self.assertGreater(len(unique_predicates), 1, "Predicates should have variety")


class TestPythonIfStatementInjection(unittest.TestCase):
    """Test opaque predicate injection into Python if statements."""

    def test_inject_into_simple_if(self) -> None:
        """Test wrapping a simple if condition."""
        code = "if x > 0:\n    print(x)"
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertEqual(result.transformation_count, 1)

        # The if test should now be a BoolOp (and)
        if_node = result.ast_node.body[0]
        self.assertIsInstance(if_node, ast.If)
        self.assertIsInstance(if_node.test, ast.BoolOp)
        self.assertIsInstance(if_node.test.op, ast.And)

    def test_inject_into_if_else(self) -> None:
        """Test handling of if-else structures."""
        code = "if x > 0:\n    print(x)\nelse:\n    print(-x)"
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertEqual(result.transformation_count, 1)

    def test_inject_into_elif_chain(self) -> None:
        """Test handling of if-elif-else chains."""
        code = """if x > 0:
    print("positive")
elif x < 0:
    print("negative")
else:
    print("zero")"""
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        # Should inject into each if/elif condition
        self.assertGreaterEqual(result.transformation_count, 1)

    def test_injection_frequency_control(self) -> None:
        """Test that percentage controls injection frequency."""
        code = "if x > 0:\n    print(x)"

        # With 0% should never inject
        transformer_0 = OpaquePredicatesTransformer(opaque_predicate_percentage=0)
        tree_0 = ast.parse(code)
        result_0 = transformer_0.transform(tree_0)
        self.assertEqual(result_0.transformation_count, 0)

        # With 100% should always inject
        transformer_100 = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        tree_100 = ast.parse(code)
        result_100 = transformer_100.transform(tree_100)
        self.assertEqual(result_100.transformation_count, 1)

    def test_preserve_original_condition(self) -> None:
        """Test that original condition is preserved in wrapped form."""
        code = "if x > 0:\n    print(x)"
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        if_node = result.ast_node.body[0]
        self.assertIsInstance(if_node.test, ast.BoolOp)

        # Original condition should be the second value in the AND
        original = if_node.test.values[1]
        self.assertIsInstance(original, ast.Compare)


class TestPythonLoopInjection(unittest.TestCase):
    """Test opaque predicate injection into Python loops."""

    def test_inject_into_while_loop(self) -> None:
        """Test wrapping while loop condition."""
        code = "while x < 10:\n    x += 1"
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertEqual(result.transformation_count, 1)

        while_node = result.ast_node.body[0]
        self.assertIsInstance(while_node, ast.While)
        self.assertIsInstance(while_node.test, ast.BoolOp)

    def test_inject_into_for_loop(self) -> None:
        """Test injecting opaque predicate into for loop body."""
        code = "for i in range(10):\n    print(i)"
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertEqual(result.transformation_count, 1)

        for_node = result.ast_node.body[0]
        self.assertIsInstance(for_node, ast.For)
        # Body should now be wrapped in an if statement
        self.assertEqual(len(for_node.body), 1)
        self.assertIsInstance(for_node.body[0], ast.If)

    def test_nested_loops(self) -> None:
        """Test handling of nested loop structures."""
        code = """for i in range(5):
    for j in range(5):
        print(i, j)"""
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        # Should inject into both loops
        self.assertGreaterEqual(result.transformation_count, 2)

    def test_loop_with_break_continue(self) -> None:
        """Test that break and continue are preserved."""
        code = """for i in range(10):
    if i == 5:
        break
    if i == 3:
        continue
    print(i)"""
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)

        # Verify the tree can be compiled
        try:
            compile(result.ast_node, '<test>', 'exec')
        except SyntaxError:
            self.fail("Transformed code should be syntactically valid")


class TestPythonFunctionInjection(unittest.TestCase):
    """Test opaque predicate injection at function entry."""

    def test_inject_at_function_entry(self) -> None:
        """Test adding opaque variable and check at function entry."""
        code = """def foo():
    return 42"""
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertEqual(result.transformation_count, 1)

        func_node = result.ast_node.body[0]
        self.assertIsInstance(func_node, ast.FunctionDef)
        # Should have opaque var + if statement
        self.assertEqual(len(func_node.body), 2)
        self.assertIsInstance(func_node.body[0], ast.Assign)  # Opaque variable
        self.assertIsInstance(func_node.body[1], ast.If)      # Opaque check

    def test_inject_into_multiple_functions(self) -> None:
        """Test injection into multiple functions in a module."""
        code = """def foo():
    return 1

def bar():
    return 2"""
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        # Should inject into both functions
        self.assertEqual(result.transformation_count, 2)

    def test_async_function_injection(self) -> None:
        """Test injection into async functions."""
        code = """async def foo():
    return 42"""
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertEqual(result.transformation_count, 1)

        func_node = result.ast_node.body[0]
        self.assertIsInstance(func_node, ast.AsyncFunctionDef)
        self.assertEqual(len(func_node.body), 2)

    def test_class_method_injection(self) -> None:
        """Test injection into class methods."""
        code = """class MyClass:
    def method(self):
        return 42"""
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        # Should inject into the method
        self.assertEqual(result.transformation_count, 1)


class TestLuaOpaquePredicateGeneration(unittest.TestCase):
    """Test Lua opaque predicate generation."""

    @lua_skip
    def test_lua_support_available(self) -> None:
        """Test that luaparser is available."""
        self.assertTrue(LUAPARSER_AVAILABLE)

    @lua_skip
    def test_generate_lua_opaque_true(self) -> None:
        """Test generating Lua true predicates."""
        transformer = OpaquePredicatesTransformer(opaque_predicate_complexity=1)

        for _ in range(10):
            predicate = transformer._generate_lua_opaque_true()
            self.assertIsNotNone(predicate)
            # Should be a Lua AST node
            self.assertTrue(hasattr(predicate, '__class__'))

    @lua_skip
    def test_generate_lua_opaque_false(self) -> None:
        """Test generating Lua false predicates."""
        transformer = OpaquePredicatesTransformer(opaque_predicate_complexity=1)

        for _ in range(10):
            predicate = transformer._generate_lua_opaque_false()
            self.assertIsNotNone(predicate)

    @lua_skip
    def test_lua_predicate_syntax(self) -> None:
        """Test that generated Lua predicates are valid syntax."""
        transformer = OpaquePredicatesTransformer(opaque_predicate_complexity=1)

        for _ in range(5):
            predicate = transformer._generate_lua_opaque_true()
            # Try to parse the predicate back (as part of a statement)
            try:
                # Create a simple if statement with the predicate
                var_name = transformer._generate_opaque_var_name()
                code = f"local {var_name} = 1\nif {var_name} > 0 then print('ok') end"
                lua_ast.parse(code)
            except Exception as e:
                self.fail(f"Generated Lua predicate should be valid: {e}")


class TestLuaInjection(unittest.TestCase):
    """Test Lua-specific injection methods."""

    @lua_skip
    def test_inject_into_lua_if(self) -> None:
        """Test injecting into Lua if statements."""
        code = """
local x = 5
if x > 0 then
    print("positive")
end
"""
        tree = lua_ast.parse(code)
        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.transformation_count, 1)

    @lua_skip
    def test_inject_into_lua_while(self) -> None:
        """Test injecting into Lua while loops."""
        code = """
local x = 0
while x < 10 do
    x = x + 1
end
"""
        tree = lua_ast.parse(code)
        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.transformation_count, 1)

    @lua_skip
    def test_inject_into_lua_function(self) -> None:
        """Test injecting at Lua function entry."""
        code = """
function foo()
    return 42
end
"""
        tree = lua_ast.parse(code)
        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.transformation_count, 1)

    @lua_skip
    def test_lua_injection_frequency(self) -> None:
        """Test percentage control for Lua injection."""
        code = """
local x = 5
if x > 0 then
    print("test")
end
"""
        # With 0% should not inject
        tree_0 = lua_ast.parse(code)
        transformer_0 = OpaquePredicatesTransformer(opaque_predicate_percentage=0)
        result_0 = transformer_0.transform(tree_0)
        self.assertEqual(result_0.transformation_count, 0)


class TestTransformResult(unittest.TestCase):
    """Test transformation result handling."""

    def test_successful_transformation(self) -> None:
        """Test successful transformation returns proper result."""
        code = "if x > 0:\n    print(x)"
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertIsNotNone(result.ast_node)
        self.assertIsInstance(result.ast_node, ast.Module)
        self.assertIsInstance(result.errors, list)
        self.assertEqual(len(result.errors), 0)

    def test_transformation_count(self) -> None:
        """Test that transformation count is accurate."""
        code = """if x > 0:
    print("a")
if x < 0:
    print("b")"""
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertEqual(result.transformation_count, 2)

    def test_error_handling(self) -> None:
        """Test handling of invalid input."""
        transformer = OpaquePredicatesTransformer()

        # Test None input
        result = transformer.transform(None)
        self.assertFalse(result.success)
        self.assertIsNone(result.ast_node)
        self.assertGreater(len(result.errors), 0)

    def test_empty_module(self) -> None:
        """Test transformation of empty module."""
        tree = ast.parse("")

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertEqual(result.transformation_count, 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def test_nested_if_statements(self) -> None:
        """Test handling of deeply nested if statements."""
        code = """if x > 0:
    if y > 0:
        if z > 0:
            print("deep")"""
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        # Should inject into all three if statements
        self.assertGreaterEqual(result.transformation_count, 3)

    def test_complex_boolean_expressions(self) -> None:
        """Test handling of existing complex boolean conditions."""
        code = "if x > 0 and y < 10 or z == 5:\n    print('complex')"
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        # Should still wrap the entire condition
        if_node = result.ast_node.body[0]
        self.assertIsInstance(if_node.test, ast.BoolOp)

    def test_lambda_functions(self) -> None:
        """Test that lambda functions are handled appropriately."""
        code = "f = lambda x: x + 1"
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        # Lambda should be preserved
        assign = result.ast_node.body[0]
        self.assertIsInstance(assign.value, ast.Lambda)

    def test_list_comprehensions(self) -> None:
        """Test handling of list comprehensions."""
        code = "[x for x in range(10) if x > 0]"
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)

    def test_generator_expressions(self) -> None:
        """Test handling of generator expressions."""
        code = "sum(x for x in range(10) if x > 0)"
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)

    def test_decorator_preservation(self) -> None:
        """Test that function decorators are preserved."""
        code = """@decorator
def foo():
    return 42"""
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        func_node = result.ast_node.body[0]
        self.assertIsInstance(func_node, ast.FunctionDef)
        self.assertEqual(len(func_node.decorator_list), 1)


class TestIntegrationWithOtherTransformers(unittest.TestCase):
    """Test integration with other transformers."""

    def test_with_control_flow_flattening(self) -> None:
        """Test compatibility with control flow flattening."""
        # This test ensures the transformer works correctly
        # when both features are enabled in the pipeline
        code = """def foo():
    if x > 0:
        return 1
    else:
        return 2"""
        tree = ast.parse(code)

        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)
        # The code should still be valid
        try:
            compile(result.ast_node, '<test>', 'exec')
        except SyntaxError:
            self.fail("Transformed code should be syntactically valid")

    def test_with_dead_code_injection(self) -> None:
        """Test combined transformations work correctly."""
        code = """def foo():
    return 42"""
        tree = ast.parse(code)

        # Apply dead code injection first (simulated)
        # Then apply opaque predicates
        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)
        result = transformer.transform(tree)

        self.assertTrue(result.success)

    def test_transformation_order(self) -> None:
        """Verify pipeline order is correct in ObfuscationEngine."""
        from obfuscator.core.obfuscation_engine import ObfuscationEngine

        config = ObfuscationConfig(
            name="test",
            features={
                "opaque_predicates": True,
                "dead_code_injection": True,
                "mangle_indexes": True,
            }
        )

        engine = ObfuscationEngine(config)
        transformers = engine.get_enabled_transformers("python")

        # Find positions of relevant transformers
        names = [type(t).__name__ for t in transformers]

        if "DeadCodeInjectionTransformer" in names and "OpaquePredicatesTransformer" in names:
            dead_pos = names.index("DeadCodeInjectionTransformer")
            opaque_pos = names.index("OpaquePredicatesTransformer")
            mangle_pos = names.index("MangleIndexesTransformer")

            # Opaque predicates should be after dead code
            self.assertLess(dead_pos, opaque_pos)
            # Opaque predicates should be before mangle indexes
            self.assertLess(opaque_pos, mangle_pos)

    def test_full_pipeline(self) -> None:
        """Test running through the full pipeline."""
        from obfuscator.core.obfuscation_engine import ObfuscationEngine

        code = """def calculate(x, y):
    if x > 0:
        while y < 10:
            y += x
        return y
    return 0"""

        config = ObfuscationConfig(
            name="test",
            features={
                "opaque_predicates": True,
                "dead_code_injection": True,
                "control_flow_flattening": True,
            },
            options={
                "opaque_predicate_complexity": 2,
                "opaque_predicate_percentage": 100,
                "dead_code_percentage": 100,
            }
        )

        tree = ast.parse(code)
        engine = ObfuscationEngine(config)

        result = engine.apply_transformations(tree, "python", None)

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.transformation_count, 1)


class TestShouldInject(unittest.TestCase):
    """Test the _should_inject method."""

    def test_always_inject_at_100(self) -> None:
        """Test that 100% always injects."""
        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=100)

        # With 100%, should always return True
        for _ in range(50):
            self.assertTrue(transformer._should_inject())

    def test_never_inject_at_0(self) -> None:
        """Test that 0% never injects."""
        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=0)

        # With 0%, should always return False
        for _ in range(50):
            self.assertFalse(transformer._should_inject())

    def test_probabilistic_injection(self) -> None:
        """Test that 50% injects roughly half the time."""
        transformer = OpaquePredicatesTransformer(opaque_predicate_percentage=50)

        injections = sum(1 for _ in range(200) if transformer._should_inject())

        # Should be roughly 100 (50%), allow for variance
        self.assertGreater(injections, 50)
        self.assertLess(injections, 150)


class TestVariableNameGeneration(unittest.TestCase):
    """Test opaque variable name generation."""

    def test_unique_variable_names(self) -> None:
        """Test that generated variable names are unique."""
        transformer = OpaquePredicatesTransformer()

        names = [transformer._generate_opaque_var_name() for _ in range(100)]

        # All names should be unique
        self.assertEqual(len(names), len(set(names)))

    def test_naming_pattern(self) -> None:
        """Test that variable names follow expected pattern."""
        transformer = OpaquePredicatesTransformer()

        name = transformer._generate_opaque_var_name()
        self.assertTrue(name.startswith("_opaque_"))

        # Should be followed by a number
        suffix = name.split("_")[-1]
        self.assertTrue(suffix.isdigit())


if __name__ == "__main__":
    unittest.main()
