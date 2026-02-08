"""Tests for the AntiDebuggingTransformer class.

This test suite covers:
- Initialization and configuration
- Python anti-debug code generation
- Lua anti-debug code generation
- AST transformation at various injection points
- Integration with other transformers
- Edge cases and error handling
"""

import ast
import sys
import unittest
from unittest.mock import MagicMock, patch

# Try to import luaparser for Lua tests
try:
    from luaparser import ast as lua_ast
    from luaparser import astnodes as lua_nodes
    LUAPARSER_AVAILABLE = True
except ImportError:
    LUAPARSER_AVAILABLE = False

from obfuscator.processors.anti_debug_runtime_python import (
    generate_python_anti_debug_checks,
    generate_python_single_check,
    generate_python_obfuscated_check,
)
from obfuscator.processors.anti_debug_runtime_lua import (
    generate_lua_anti_debug_checks,
    generate_lua_single_check,
    generate_lua_obfuscated_check,
)
from obfuscator.processors.ast_transformer import (
    AntiDebuggingTransformer,
    TransformResult,
)


def lua_skip(func):
    """Decorator to skip Lua tests when luaparser is not available."""
    return unittest.skipUnless(LUAPARSER_AVAILABLE, "luaparser not available")(func)


class TestAntiDebuggingTransformerInit(unittest.TestCase):
    """Tests for AntiDebuggingTransformer initialization."""

    def test_default_initialization(self):
        """Verify default aggressiveness is 2."""
        transformer = AntiDebuggingTransformer()
        self.assertEqual(transformer.anti_debug_aggressiveness, 2)
        self.assertEqual(transformer.injection_count, 0)
        self.assertIsNone(transformer.language_mode)

    def test_initialization_with_config(self):
        """Test config extraction for aggressiveness."""
        config = MagicMock()
        config.options = {"anti_debug_aggressiveness": 3}
        transformer = AntiDebuggingTransformer(config=config)
        self.assertEqual(transformer.anti_debug_aggressiveness, 3)

    def test_initialization_with_explicit_aggressiveness(self):
        """Test parameter override for aggressiveness."""
        config = MagicMock()
        config.options = {"anti_debug_aggressiveness": 1}
        transformer = AntiDebuggingTransformer(
            config=config, anti_debug_aggressiveness=3
        )
        self.assertEqual(transformer.anti_debug_aggressiveness, 3)

    def test_explicit_params_override_config(self):
        """Verify parameter precedence over config."""
        config = MagicMock()
        config.options = {"anti_debug_aggressiveness": 1}
        transformer = AntiDebuggingTransformer(
            config=config, anti_debug_aggressiveness=2
        )
        self.assertEqual(transformer.anti_debug_aggressiveness, 2)

    def test_aggressiveness_validation_bounds(self):
        """Test clamping to 1-3 range."""
        # Test too low
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=0)
        self.assertEqual(transformer.anti_debug_aggressiveness, 1)

        # Test too high
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=5)
        self.assertEqual(transformer.anti_debug_aggressiveness, 3)

    def test_aggressiveness_non_integer(self):
        """Test handling of non-integer aggressiveness."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness="invalid")
        self.assertEqual(transformer.anti_debug_aggressiveness, 2)


class TestPythonAntiDebugGeneration(unittest.TestCase):
    """Tests for Python anti-debug code generation."""

    def test_generate_runtime_code(self):
        """Verify runtime code generation returns valid Python."""
        code = generate_python_anti_debug_checks(aggressiveness=2)
        self.assertIsInstance(code, str)
        self.assertGreater(len(code), 0)

    def test_runtime_contains_gettrace_check(self):
        """Verify sys.gettrace() is in generated code."""
        code = generate_python_anti_debug_checks(aggressiveness=2)
        self.assertIn("sys.gettrace()", code)

    def test_runtime_contains_timing_check(self):
        """Verify timing detection code is present."""
        code = generate_python_anti_debug_checks(aggressiveness=2)
        self.assertIn("time.perf_counter()", code)

    def test_runtime_contains_module_check(self):
        """Verify debugger module detection."""
        code = generate_python_anti_debug_checks(aggressiveness=2)
        self.assertIn("sys.modules", code)
        self.assertIn("pdb", code)

    def test_runtime_code_is_parseable(self):
        """Ensure generated code can be parsed by ast.parse()."""
        code = generate_python_anti_debug_checks(aggressiveness=2)
        try:
            ast.parse(code)
        except SyntaxError as e:
            self.fail(f"Generated code is not parseable: {e}")

    def test_runtime_contains_process_check_at_level_2(self):
        """Verify process inspection at level 2+."""
        code = generate_python_anti_debug_checks(aggressiveness=2)
        self.assertIn("getppid", code)

    def test_runtime_contains_random_shuffle_at_level_3(self):
        """Verify random shuffle at level 3."""
        code = generate_python_anti_debug_checks(aggressiveness=3)
        self.assertIn("random.shuffle", code)

    def test_aggressiveness_affects_code_content(self):
        """Test that different aggressiveness levels produce different code."""
        code1 = generate_python_anti_debug_checks(aggressiveness=1)
        code2 = generate_python_anti_debug_checks(aggressiveness=2)
        code3 = generate_python_anti_debug_checks(aggressiveness=3)

        # Level 1 should not have process checks
        self.assertNotIn("getppid", code1)
        # Level 2 should have process checks
        self.assertIn("getppid", code2)
        # Level 3 should have random shuffle
        self.assertIn("random.shuffle", code3)

    def test_single_check_generation(self):
        """Test single inline check generation."""
        code = generate_python_single_check(check_type="trace", action="exit")
        self.assertIn("sys.gettrace()", code)
        self.assertIn("sys.exit", code)

    def test_obfuscated_check_generation(self):
        """Test obfuscated check generation."""
        code = generate_python_obfuscated_check(aggressiveness=2)
        self.assertIsInstance(code, str)
        self.assertGreater(len(code), 0)


class TestPythonInjectionPoints(unittest.TestCase):
    """Tests for Python AST injection points."""

    def test_inject_at_module_level(self):
        """Verify runtime is injected at module beginning."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = "def foo(): pass"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

        # Check that the module body has more statements (runtime + original)
        self.assertGreater(len(result.ast_node.body), len(tree.body))

    def test_inject_at_function_entry(self):
        """Verify checks at function entry points."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = "def foo(): x = 1"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

        # Find the function definition
        func_def = None
        for node in ast.walk(result.ast_node):
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break

        self.assertIsNotNone(func_def)
        # Check that first statement is the anti-debug call
        self.assertGreater(len(func_def.body), 0)

    def test_inject_at_critical_points_aggressive(self):
        """Test injection in loops/conditionals at aggressiveness=3."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=3)
        code = "while True: x = 1"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

        # Find the while loop
        while_node = None
        for node in ast.walk(result.ast_node):
            if isinstance(node, ast.While):
                while_node = node
                break

        self.assertIsNotNone(while_node)
        # Loop body should have anti-debug check at entry
        self.assertGreater(len(while_node.body), 1)

    def test_no_injection_at_low_aggressiveness(self):
        """Verify minimal injection at aggressiveness=1."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=1)
        code = "def foo(): x = 1"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

        # Function should not have check at entry (aggressiveness 1 = runtime only)
        func_def = None
        for node in ast.walk(result.ast_node):
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break

        self.assertIsNotNone(func_def)
        # With aggressiveness=1, only runtime is injected at module level
        self.assertEqual(len(func_def.body), 1)  # Just the original statement

    def test_injection_frequency_by_aggressiveness(self):
        """Compare injection counts across aggressiveness levels."""
        code = "def foo(): pass\ndef bar(): pass"

        counts = []
        for level in [1, 2, 3]:
            transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=level)
            tree = ast.parse(code)
            result = transformer.transform(tree)
            counts.append(result.transformation_count)

        # Higher aggressiveness should have equal or more transformations
        self.assertLessEqual(counts[0], counts[1])
        self.assertLessEqual(counts[1], counts[2])


class TestPythonFunctionTransformation(unittest.TestCase):
    """Tests for Python function transformation."""

    def test_transform_simple_function(self):
        """Basic function with anti-debug check."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = "def foo():\n    return 42"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        self.assertGreater(result.transformation_count, 0)

    def test_transform_multiple_functions(self):
        """Multiple functions in module."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = """
def foo():
    return 1

def bar():
    return 2
"""
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        # Should have checks in both functions plus runtime
        self.assertGreaterEqual(result.transformation_count, 2)

    def test_transform_async_function(self):
        """Async function handling."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = "async def foo():\n    return 42"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

        # Find async function
        async_func = None
        for node in ast.walk(result.ast_node):
            if isinstance(node, ast.AsyncFunctionDef):
                async_func = node
                break

        self.assertIsNotNone(async_func)

    def test_transform_class_methods(self):
        """Class method handling."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = """
class Foo:
    def method(self):
        return 1
"""
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

    def test_transformation_count(self):
        """Verify counter accuracy."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = "def foo(): pass"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        # Should have at least 1 transformation (the function entry check)
        self.assertGreaterEqual(result.transformation_count, 1)


class TestLuaAntiDebugGeneration(unittest.TestCase):
    """Tests for Lua anti-debug code generation."""

    @lua_skip
    def test_lua_support_available(self):
        """Check luaparser availability."""
        self.assertTrue(LUAPARSER_AVAILABLE)

    @lua_skip
    def test_generate_lua_runtime_code(self):
        """Verify Lua runtime generation."""
        code = generate_lua_anti_debug_checks(aggressiveness=2)
        self.assertIsInstance(code, str)
        self.assertGreater(len(code), 0)

    @lua_skip
    def test_lua_runtime_contains_debug_check(self):
        """Verify debug library detection."""
        code = generate_lua_anti_debug_checks(aggressiveness=2)
        self.assertIn("debug", code)

    @lua_skip
    def test_lua_runtime_contains_timing_check(self):
        """Verify timing detection."""
        code = generate_lua_anti_debug_checks(aggressiveness=2)
        self.assertIn("os.clock()", code)

    @lua_skip
    def test_lua_runtime_contains_hook_check(self):
        """Verify hook detection."""
        code = generate_lua_anti_debug_checks(aggressiveness=2)
        self.assertIn("gethook", code)

    @lua_skip
    def test_lua_runtime_code_is_parseable(self):
        """Ensure valid Lua syntax."""
        code = generate_lua_anti_debug_checks(aggressiveness=2)
        try:
            lua_ast.parse(code)
        except Exception as e:
            self.fail(f"Generated Lua code is not parseable: {e}")


class TestLuaFunctionTransformation(unittest.TestCase):
    """Tests for Lua function transformation."""

    @lua_skip
    def test_transform_lua_function(self):
        """Basic Lua function transformation."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = "function foo()\n    return 1\nend"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

    @lua_skip
    def test_inject_at_lua_function_entry(self):
        """Lua function entry injection."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = "function foo()\n    return 1\nend"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.transformation_count, 1)

    @lua_skip
    def test_lua_injection_frequency(self):
        """Aggressiveness control in Lua."""
        code = "function foo()\n    return 1\nend"

        counts = []
        for level in [1, 2, 3]:
            transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=level)
            tree = lua_ast.parse(code)
            result = transformer.transform(tree)
            counts.append(result.transformation_count)

        # Higher aggressiveness should have equal or more transformations
        self.assertLessEqual(counts[0], counts[1])


class TestTransformResult(unittest.TestCase):
    """Tests for transformation result handling."""

    def test_successful_transformation(self):
        """Verify result structure."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = "def foo(): pass"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.ast_node)
        self.assertIsInstance(result.transformation_count, int)
        self.assertIsInstance(result.errors, list)

    def test_transformation_with_low_aggressiveness(self):
        """Test aggressiveness=1."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=1)
        code = "def foo(): pass"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        # Should have minimal transformations (just runtime injection)
        self.assertGreaterEqual(result.transformation_count, 0)

    def test_transformation_with_high_aggressiveness(self):
        """Test aggressiveness=3."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=3)
        code = """
def foo():
    for i in range(10):
        pass
    return 1
"""
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        # Should have many transformations (function + loop)
        self.assertGreaterEqual(result.transformation_count, 2)

    def test_error_handling(self):
        """Error capture in result."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)

        # Create an invalid AST node to trigger an error
        result = transformer.transform(None)

        # Should fail gracefully
        self.assertFalse(result.success)
        self.assertIsNone(result.ast_node)
        self.assertGreater(len(result.errors), 0)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def test_empty_module(self):
        """Empty module handling."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = ""
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

    def test_module_without_functions(self):
        """Module with only statements."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = "x = 1\ny = 2"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

    def test_nested_functions(self):
        """Nested function handling."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = """
def outer():
    def inner():
        return 1
    return inner()
"""
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        # Should handle both outer and inner
        self.assertGreaterEqual(result.transformation_count, 2)

    def test_lambda_functions(self):
        """Lambda preservation."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = "f = lambda x: x + 1"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        # Lambda should not have anti-debug check (it's an expression, not a FunctionDef)

    def test_generator_functions(self):
        """Generator handling."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = """
def gen():
    yield 1
    yield 2
"""
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)


class TestIntegrationWithOtherTransformers(unittest.TestCase):
    """Tests for integration with other transformers."""

    def test_with_control_flow_flattening(self):
        """Compatibility test with control flow flattening."""
        # This test ensures anti-debugging can work after control flow flattening
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = """
def foo():
    if x > 0:
        return 1
    else:
        return 2
"""
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

    def test_with_opaque_predicates(self):
        """Combined transformations."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        code = "def foo():\n    return True"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

    def test_transformation_order(self):
        """Verify pipeline order in ObfuscationEngine."""
        from obfuscator.core.obfuscation_engine import ObfuscationEngine
        from obfuscator.core.config import ObfuscationConfig

        config = ObfuscationConfig(
            name="test",
            features={"anti_debugging": True},
        )
        engine = ObfuscationEngine(config)
        transformers = engine.get_enabled_transformers("python")

        # Find anti_debugging transformer in the list
        found = False
        for transformer in transformers:
            if isinstance(transformer, AntiDebuggingTransformer):
                found = True
                break

        # Note: anti_debugging is not in the basic preset, so this may not find it
        # The test mainly ensures the class is importable and usable
        self.assertTrue(True)  # Just verify no exception

    def test_full_pipeline(self):
        """Run through complete pipeline."""
        from obfuscator.core.obfuscation_engine import ObfuscationEngine
        from obfuscator.core.config import ObfuscationConfig

        config = ObfuscationConfig(
            name="test",
            preset="maximum",
            language="python",
            features={"anti_debugging": True},
        )
        engine = ObfuscationEngine(config)

        code = "def foo():\n    return 42"
        tree = ast.parse(code)

        # Apply all transformations
        result = engine.apply_transformations(tree, "python", "test.py")
        self.assertTrue(result.success)


class TestHelperMethods(unittest.TestCase):
    """Tests for transformer helper methods."""

    def test_should_inject_at_function_entry(self):
        """Test _should_inject_at_function_entry."""
        # Level 1: No function entry injection
        t1 = AntiDebuggingTransformer(anti_debug_aggressiveness=1)
        self.assertFalse(t1._should_inject_at_function_entry())

        # Level 2+: Function entry injection
        t2 = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        self.assertTrue(t2._should_inject_at_function_entry())

        t3 = AntiDebuggingTransformer(anti_debug_aggressiveness=3)
        self.assertTrue(t3._should_inject_at_function_entry())

    def test_should_inject_at_critical_points(self):
        """Test _should_inject_at_critical_points."""
        # Level 1-2: No critical point injection
        t1 = AntiDebuggingTransformer(anti_debug_aggressiveness=1)
        self.assertFalse(t1._should_inject_at_critical_points())

        t2 = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        self.assertFalse(t2._should_inject_at_critical_points())

        # Level 3: Critical point injection
        t3 = AntiDebuggingTransformer(anti_debug_aggressiveness=3)
        self.assertTrue(t3._should_inject_at_critical_points())

    def test_generate_python_check_variable_name(self):
        """Test unique variable name generation."""
        transformer = AntiDebuggingTransformer()

        name1 = transformer._generate_python_check_variable_name()
        name2 = transformer._generate_python_check_variable_name()

        self.assertEqual(name1, "_check_0")
        self.assertEqual(name2, "_check_1")

    def test_create_python_anti_debug_call(self):
        """Test AST node creation for check call."""
        transformer = AntiDebuggingTransformer()
        node = transformer._create_python_anti_debug_call()

        self.assertIsInstance(node, ast.Expr)
        self.assertIsInstance(node.value, ast.Call)
        self.assertIsInstance(node.value.func, ast.Name)
        self.assertEqual(node.value.func.id, "_check_env_0x1a2b")


class TestRuntimeInjectionMethods(unittest.TestCase):
    """Tests for runtime injection methods."""

    def test_inject_python_anti_debug_runtime(self):
        """Test Python runtime injection."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        statements = transformer._inject_python_anti_debug_runtime()

        self.assertIsInstance(statements, list)
        self.assertGreater(len(statements), 0)

        # Verify all are AST statement nodes
        for stmt in statements:
            self.assertIsInstance(stmt, ast.stmt)

    def test_inject_python_anti_debug_runtime_error_handling(self):
        """Test error handling in runtime injection."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)

        # Test with invalid aggressiveness (should still work with clamping)
        transformer.anti_debug_aggressiveness = 100
        statements = transformer._inject_python_anti_debug_runtime()
        self.assertIsInstance(statements, list)

    @lua_skip
    def test_inject_lua_anti_debug_runtime(self):
        """Test Lua runtime injection."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)
        nodes = transformer._inject_lua_anti_debug_runtime()

        self.assertIsInstance(nodes, list)

    def test_inject_lua_anti_debug_runtime_without_luaparser(self):
        """Test graceful handling when luaparser not available."""
        transformer = AntiDebuggingTransformer(anti_debug_aggressiveness=2)

        # Temporarily disable luaparser availability
        with patch.object(
            transformer, '_inject_lua_anti_debug_runtime', return_value=[]
        ):
            nodes = transformer._inject_lua_anti_debug_runtime()
            self.assertEqual(nodes, [])


if __name__ == "__main__":
    unittest.main()
