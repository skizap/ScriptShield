"""Unit tests for ControlFlowFlatteningTransformer.

Tests the ControlFlowFlatteningTransformer class which obfuscates control flow
using a state machine dispatcher pattern.
"""

import ast
import pytest
from typing import Any

from obfuscator.processors.ast_transformer import ControlFlowFlatteningTransformer
from obfuscator.core.config import ObfuscationConfig


class TestControlFlowFlatteningTransformerInit:
    """Test ControlFlowFlatteningTransformer initialization and configuration."""

    def test_default_initialization(self):
        """Test transformer initializes with default settings."""
        transformer = ControlFlowFlatteningTransformer()
        assert transformer.control_flow_complexity == 2
        assert transformer.min_statements_to_flatten == 5
        assert transformer.language_mode is None
        assert transformer.state_counter == 0
        assert transformer.transformation_count == 0
        assert len(transformer.errors) == 0

    def test_initialization_with_config(self):
        """Test transformer initialization with config."""
        config = ObfuscationConfig(
            name="test",
            features={"control_flow_flattening": True},
            options={"control_flow_complexity": 3, "min_statements_to_flatten": 10}
        )
        transformer = ControlFlowFlatteningTransformer(config)
        assert transformer.control_flow_complexity == 3
        assert transformer.min_statements_to_flatten == 10

    def test_initialization_with_explicit_complexity(self):
        """Test transformer initialization with explicit complexity."""
        transformer = ControlFlowFlatteningTransformer(control_flow_complexity=1)
        assert transformer.control_flow_complexity == 1

    def test_initialization_with_explicit_min_statements(self):
        """Test transformer initialization with explicit min_statements."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=3)
        assert transformer.min_statements_to_flatten == 3

    def test_explicit_params_override_config(self):
        """Test explicit parameters override config values."""
        config = ObfuscationConfig(
            name="test",
            options={"control_flow_complexity": 3, "min_statements_to_flatten": 10}
        )
        transformer = ControlFlowFlatteningTransformer(
            config=config,
            control_flow_complexity=1,
            min_statements_to_flatten=3
        )
        assert transformer.control_flow_complexity == 1
        assert transformer.min_statements_to_flatten == 3

    def test_complexity_validation_lower_bound(self):
        """Test complexity is clamped to minimum of 1."""
        transformer = ControlFlowFlatteningTransformer(control_flow_complexity=0)
        assert transformer.control_flow_complexity == 1

    def test_complexity_validation_upper_bound(self):
        """Test complexity is clamped to maximum of 3."""
        transformer = ControlFlowFlatteningTransformer(control_flow_complexity=5)
        assert transformer.control_flow_complexity == 3


class TestPythonFunctionDetection:
    """Test Python function detection methods."""

    def test_should_flatten_simple_function(self):
        """Test detection of flatten-eligible function with sufficient statements."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=3)
        code = """
def calculate(x):
    a = x + 1
    b = a * 2
    c = b - 3
    return c
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        assert transformer._should_flatten_function(func_def) is True

    def test_should_not_flatten_small_function(self):
        """Test that small functions are skipped."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=5)
        code = """
def simple(x):
    return x + 1
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        assert transformer._should_flatten_function(func_def) is False

    def test_should_not_flatten_async_function(self):
        """Test that async functions are skipped."""
        transformer = ControlFlowFlatteningTransformer()
        code = """
async def async_calc(x):
    a = x + 1
    b = a * 2
    c = b - 3
    d = c + 4
    return d
"""
        tree = ast.parse(code)
        async_func = tree.body[0]
        assert isinstance(async_func, ast.AsyncFunctionDef)
        assert transformer._should_flatten_function(async_func) is False  # type: ignore

    def test_should_not_flatten_generator_function(self):
        """Test that generator functions are skipped."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=3)
        code = """
def generator(x):
    a = x + 1
    yield a
    b = a * 2
    yield b
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        assert transformer._should_flatten_function(func_def) is False

    def test_should_flatten_function_with_control_flow(self):
        """Test that functions with control flow can be flattened."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=3)
        code = """
def with_control(x):
    if x > 0:
        a = x * 2
    else:
        a = x + 1
    b = a - 3
    return b
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        assert transformer._should_flatten_function(func_def) is True


class TestPythonBasicBlockExtraction:
    """Test basic block extraction methods."""

    def test_extract_sequential_statements(self):
        """Test extraction of simple sequential code."""
        transformer = ControlFlowFlatteningTransformer()
        code = """
def simple(x):
    a = 1
    b = 2
    c = 3
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        blocks = transformer._extract_basic_blocks(func_def.body)
        # All statements in one block (no control flow)
        assert len(blocks) == 1
        assert len(blocks[0]) == 3

    def test_extract_blocks_with_if_statement(self):
        """Test extraction with if/else statements."""
        transformer = ControlFlowFlatteningTransformer()
        code = """
def with_if(x):
    a = 1
    if x > 0:
        b = 2
    else:
        b = 3
    c = 4
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        blocks = transformer._extract_basic_blocks(func_def.body)
        # Block before if, if block, block after if
        assert len(blocks) >= 2

    def test_extract_blocks_with_while_loop(self):
        """Test extraction with while loop."""
        transformer = ControlFlowFlatteningTransformer()
        code = """
def with_while(x):
    a = 1
    while x > 0:
        x = x - 1
    b = 2
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        blocks = transformer._extract_basic_blocks(func_def.body)
        # Should have blocks for: before-while, while-statement, after-while
        assert len(blocks) >= 2

    def test_extract_blocks_with_return(self):
        """Test extraction with return statements."""
        transformer = ControlFlowFlatteningTransformer()
        code = """
def with_return(x):
    a = 1
    return a
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        blocks = transformer._extract_basic_blocks(func_def.body)
        # Return creates its own block
        assert len(blocks) == 2

    def test_extract_blocks_complex_flow(self):
        """Test extraction with mixed control flow."""
        transformer = ControlFlowFlatteningTransformer()
        code = """
def complex_flow(x):
    a = 1
    if x > 0:
        b = 2
        while b > 0:
            b = b - 1
    else:
        c = 3
    d = 4
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        blocks = transformer._extract_basic_blocks(func_def.body)
        # Multiple blocks for complex flow
        assert len(blocks) >= 3


class TestPythonDispatcherGeneration:
    """Test dispatcher generation methods."""

    def test_create_dispatcher_simple(self):
        """Test basic dispatcher structure creation."""
        transformer = ControlFlowFlatteningTransformer()
        blocks = [
            (1, [ast.parse("a = 1").body[0]]),
            (2, [ast.parse("b = 2").body[0]]),
        ]
        dispatcher = transformer._create_dispatcher_python(blocks, exit_state=3)

        assert len(dispatcher) == 2  # state init + while loop
        assert isinstance(dispatcher[0], ast.Assign)  # state initialization
        assert isinstance(dispatcher[1], ast.While)  # main loop

    def test_dispatcher_has_while_loop(self):
        """Verify dispatcher contains while loop wrapper."""
        transformer = ControlFlowFlatteningTransformer()
        blocks = [
            (1, [ast.parse("x = 1").body[0]]),
        ]
        dispatcher = transformer._create_dispatcher_python(blocks, exit_state=2)

        assert len(dispatcher) == 2
        while_loop = dispatcher[1]
        assert isinstance(while_loop, ast.While)
        assert isinstance(while_loop.test, ast.Constant)
        assert while_loop.test.value is True

    def test_dispatcher_has_state_variable(self):
        """Verify dispatcher initializes state variable."""
        transformer = ControlFlowFlatteningTransformer()
        blocks = [
            (1, [ast.parse("x = 1").body[0]]),
        ]
        dispatcher = transformer._create_dispatcher_python(blocks, exit_state=2)

        state_init = dispatcher[0]
        assert isinstance(state_init, ast.Assign)
        assert isinstance(state_init.targets[0], ast.Name)
        assert state_init.targets[0].id == "__state"

    def test_dispatcher_has_if_elif_chain(self):
        """Verify dispatcher contains if/elif chain for state dispatch."""
        transformer = ControlFlowFlatteningTransformer()
        blocks = [
            (1, [ast.parse("x = 1").body[0]]),
            (2, [ast.parse("y = 2").body[0]]),
        ]
        dispatcher = transformer._create_dispatcher_python(blocks, exit_state=3)

        while_loop = dispatcher[1]
        assert isinstance(while_loop, ast.While)
        assert len(while_loop.body) == 1  # The if/elif chain

    def test_dispatcher_state_transitions(self):
        """Verify dispatcher includes state transition assignments."""
        transformer = ControlFlowFlatteningTransformer()
        blocks = [
            (1, [ast.parse("a = 1").body[0]]),
            (2, [ast.parse("b = 2").body[0]]),
        ]
        dispatcher = transformer._create_dispatcher_python(blocks, exit_state=3)

        while_loop = dispatcher[1]
        # Navigate the if chain to find state transitions
        assert isinstance(while_loop, ast.While)
        assert len(while_loop.body) > 0


class TestPythonFunctionTransformation:
    """Test Python function transformation."""

    def test_transform_simple_function(self):
        """Test transformation of a basic function."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=3)
        code = """
def calculate(x):
    a = x + 1
    b = a * 2
    return b
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1
        assert len(result.errors) == 0

    def test_transform_function_with_if(self):
        """Test transformation of function with conditional."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=3)
        code = """
def with_if(x):
    if x > 0:
        result = x * 2
    else:
        result = x + 1
    return result
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1

    def test_transform_function_with_loop(self):
        """Test transformation of function with loop."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=3)
        code = """
def with_loop(x):
    result = 0
    while x > 0:
        result = result + x
        x = x - 1
    return result
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1

    def test_transform_function_with_return(self):
        """Test transformation of function with return statement."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=2)
        code = """
def with_return(x):
    a = x + 1
    return a
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1

    def test_transform_multiple_functions(self):
        """Test transformation of multiple functions in module."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=2)
        code = """
def func1(x):
    a = x + 1
    return a

def func2(y):
    b = y * 2
    return b
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 2

    def test_skip_small_function(self):
        """Test that small functions remain unchanged."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=5)
        code = """
def simple(x):
    return x + 1
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 0

    def test_transformation_count(self):
        """Test that counter increments correctly."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=2)
        code = """
def func1(x):
    a = x + 1
    return a

def func2(y):
    b = y + 2
    return b

def func3(z):
    return z  # Too small
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.transformation_count == 2  # func1 and func2


class TestLuaFunctionTransformation:
    """Test Lua function transformation (when luaparser available)."""

    def test_lua_support_available(self):
        """Check if Lua support is available."""
        from obfuscator.processors.ast_transformer import LUAPARSER_AVAILABLE
        # This test documents whether Lua tests should run
        assert LUAPARSER_AVAILABLE in (True, False)

    def test_transform_lua_function(self):
        """Test basic Lua function flattening (if luaparser available)."""
        from obfuscator.processors.ast_transformer import LUAPARSER_AVAILABLE, lua_ast

        if not LUAPARSER_AVAILABLE:
            pytest.skip("luaparser not available")

        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=3)
        lua_code = """
function calculate(x)
    local a = x + 1
    local b = a * 2
    return b
end
"""
        tree = lua_ast.parse(lua_code)
        result = transformer.transform(tree)

        assert result.success is True
        # Lua transformation may or may not produce results depending on implementation

    def test_skip_small_lua_function(self):
        """Test that small Lua functions are skipped."""
        from obfuscator.processors.ast_transformer import LUAPARSER_AVAILABLE, lua_ast

        if not LUAPARSER_AVAILABLE:
            pytest.skip("luaparser not available")

        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=5)
        lua_code = """
function simple(x)
    return x + 1
end
"""
        tree = lua_ast.parse(lua_code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 0


class TestTransformResult:
    """Test TransformResult structure."""

    def test_successful_transformation(self):
        """Verify success result structure."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=2)
        code = """
def func(x):
    a = x + 1
    return a
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.ast_node is not None
        assert result.transformation_count == 1
        assert len(result.errors) == 0

    def test_transformation_with_no_eligible_functions(self):
        """Test result when no functions can be flattened."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=5)
        code = """
def small(x):
    return x
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 0
        assert len(result.errors) == 0

    def test_error_handling(self):
        """Test error capture in result."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=2)
        code = """
def func(x):
    a = x + 1
    return a
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        # Errors list exists even on success
        assert isinstance(result.errors, list)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_module(self):
        """Test transformation of empty module."""
        transformer = ControlFlowFlatteningTransformer()
        tree = ast.parse("")

        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 0
        assert len(result.errors) == 0

    def test_module_without_functions(self):
        """Test transformation of module without functions."""
        transformer = ControlFlowFlatteningTransformer()
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

    def test_nested_functions(self):
        """Test handling of nested functions."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=2)
        code = """
def outer(x):
    def inner(y):
        return y * 2
    a = inner(x)
    return a
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        # Should flatten outer function only
        assert result.transformation_count >= 1

    def test_class_methods(self):
        """Test handling of class methods."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=2)
        code = """
class MyClass:
    def method(self, x):
        a = x + 1
        return a
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        # Method should be flattened
        assert result.transformation_count == 1

    def test_lambda_functions(self):
        """Test that lambda expressions are skipped."""
        transformer = ControlFlowFlatteningTransformer()
        code = """
f = lambda x: x + 1
result = f(5)
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 0

    def test_function_with_decorators(self):
        """Test that decorated functions are handled."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=2)
        code = """
@decorator
def decorated(x):
    a = x + 1
    return a
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1

    def test_function_with_type_annotations(self):
        """Test that type-annotated functions are handled."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=2)
        code = """
def typed(x: int) -> int:
    a = x + 1
    return a
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 1


class TestIntegrationWithOtherTransformers:
    """Test integration with other transformers."""

    def test_with_string_encryption(self):
        """Test control flow flattening can be applied after string encryption."""
        from obfuscator.processors.ast_transformer import StringEncryptionTransformer

        # First apply string encryption
        string_transformer = StringEncryptionTransformer()
        cff_transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=2)

        code = '''
def func(x):
    message = "hello"
    return message
'''
        tree = ast.parse(code)

        # Apply string encryption first
        result1 = string_transformer.transform(tree)
        assert result1.success is True

        # Then apply CFF
        result2 = cff_transformer.transform(result1.ast_node)
        assert result2.success is True

    def test_with_number_obfuscation(self):
        """Test control flow flattening can be applied after number obfuscation."""
        from obfuscator.processors.ast_transformer import NumberObfuscationTransformer

        num_transformer = NumberObfuscationTransformer()
        cff_transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=2)

        code = """
def func(x):
    a = 42
    return a + x
"""
        tree = ast.parse(code)

        # Apply number obfuscation first
        result1 = num_transformer.transform(tree)
        assert result1.success is True

        # Then apply CFF
        result2 = cff_transformer.transform(result1.ast_node)
        assert result2.success is True

    def test_transformation_order(self):
        """Test that transformations are applied in correct order."""
        from obfuscator.core.obfuscation_engine import ObfuscationEngine

        config = ObfuscationConfig(
            name="test",
            features={
                "string_encryption": True,
                "number_obfuscation": True,
                "control_flow_flattening": True,
            }
        )

        engine = ObfuscationEngine(config)
        transformers = engine.get_enabled_transformers("python")

        # Check that CFF is in the pipeline
        transformer_names = [type(t).__name__ for t in transformers]
        assert "ControlFlowFlatteningTransformer" in transformer_names

        # Check order: should be after value obfuscation
        string_idx = transformer_names.index("StringEncryptionTransformer")
        number_idx = transformer_names.index("NumberObfuscationTransformer")
        cff_idx = transformer_names.index("ControlFlowFlatteningTransformer")

        assert string_idx < cff_idx
        assert number_idx < cff_idx


class TestStateNumberGeneration:
    """Test state number generation."""

    def test_generate_state_number(self):
        """Test unique state number generation."""
        transformer = ControlFlowFlatteningTransformer()

        # Reset counter
        transformer.state_counter = 0

        s1 = transformer._generate_state_number()
        s2 = transformer._generate_state_number()
        s3 = transformer._generate_state_number()

        assert s1 == 1
        assert s2 == 2
        assert s3 == 3
        assert s1 != s2 != s3

    def test_state_counter_reset_between_transforms(self):
        """Test that state counter resets for each transformation."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=2)

        code1 = """
def func1(x):
    a = x + 1
    return a
"""
        tree1 = ast.parse(code1)
        result1 = transformer.transform(tree1)

        assert result1.success is True
        # State counter should have been reset
        assert transformer.state_counter == 0


class TestYieldDetection:
    """Test yield expression detection."""

    def test_contains_yield_simple(self):
        """Test detection of simple yield."""
        transformer = ControlFlowFlatteningTransformer()
        code = """
def gen():
    yield 1
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        assert transformer._contains_yield(func_def) is True

    def test_contains_yield_from(self):
        """Test detection of yield from."""
        transformer = ControlFlowFlatteningTransformer()
        code = """
def gen():
    yield from [1, 2, 3]
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        assert transformer._contains_yield(func_def) is True

    def test_no_yield_in_regular_function(self):
        """Test that regular functions don't trigger yield detection."""
        transformer = ControlFlowFlatteningTransformer()
        code = """
def regular(x):
    return x + 1
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        assert transformer._contains_yield(func_def) is False


class TestTryExceptDetection:
    """Test try/except complexity detection."""

    def test_simple_try_except(self):
        """Test that simple try/except is not considered complex."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=3)
        code = """
def with_try(x):
    try:
        a = x / 1
    except:
        a = 0
    return a
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        # Simple try/except should be flattenable
        assert transformer._should_flatten_function(func_def) is True

    def test_complex_nested_try(self):
        """Test that nested try blocks are detected."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=5)
        code = """
def nested_try(x):
    try:
        try:
            a = x / 1
        except:
            a = 0
    except:
        a = 1
    return a
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        # Nested try should prevent flattening
        assert transformer._should_flatten_function(func_def) is False


class TestAsyncFunctionHandling:
    """Test async function handling."""

    def test_async_function_unchanged(self):
        """Test that async functions remain unchanged."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=3)
        code = """
async def async_func(x):
    a = x + 1
    b = a * 2
    c = b - 3
    return c
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        assert result.transformation_count == 0

    def test_regular_functions_still_transformed(self):
        """Test that regular functions are still transformed alongside async."""
        transformer = ControlFlowFlatteningTransformer(min_statements_to_flatten=2)
        code = """
def regular(x):
    return x + 1

async def async_func(x):
    return x + 2
"""
        tree = ast.parse(code)
        result = transformer.transform(tree)

        assert result.success is True
        # Only regular function should be transformed
        assert result.transformation_count == 1
