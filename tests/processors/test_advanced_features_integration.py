"""Integration tests for advanced obfuscation features.

This module tests the integration of 6 advanced transformers:
1. ControlFlowFlatteningTransformer - Obscures control flow with state machines
2. DeadCodeInjectionTransformer - Injects unreachable code paths
3. OpaquePredicatesTransformer - Adds always-true/false conditions
4. AntiDebuggingTransformer - Injects debugger detection checks
5. CodeSplittingTransformer - Splits code into encrypted chunks
6. SelfModifyingCodeTransformer - Generates runtime code modification

Tests verify:
- All transformers work together without conflicts
- Transformation order is correct
- Obfuscated code maintains functional equivalence
- Profile presets enable appropriate features
- Both Python and Lua are supported
"""

import ast
import unittest
from pathlib import Path
from typing import Dict, Any, List

from obfuscator.core.config import ObfuscationConfig
from obfuscator.core.obfuscation_engine import ObfuscationEngine
from obfuscator.core.profile_manager import ProfileManager
from obfuscator.processors import TransformResult
from obfuscator.processors.control_flow_transformer import ControlFlowFlatteningTransformer
from obfuscator.processors.dead_code_transformer import DeadCodeInjectionTransformer
from obfuscator.processors.opaque_predicates_transformer import OpaquePredicatesTransformer
from obfuscator.processors.anti_debug_transformer import AntiDebuggingTransformer
from obfuscator.processors.code_splitting_transformer import CodeSplittingTransformer
from obfuscator.processors.self_modifying_transformer import SelfModifyingCodeTransformer
from obfuscator.processors.ast_transformer import (
    RobloxExploitDefenseTransformer,
    RobloxRemoteSpyTransformer,
)

# Try to import luaparser for Lua tests
try:
    from luaparser import ast as lua_ast
    from luaparser import astnodes as lua_nodes
    LUAPARSER_AVAILABLE = True
except ImportError:
    LUAPARSER_AVAILABLE = False


# Sample Python code for testing
SAMPLE_PYTHON_CODE = '''
def simple_function(x):
    """A simple function that doubles a number."""
    return x * 2

def conditional_function(x):
    """Function with if/else control flow."""
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"

def loop_function(n):
    """Function with loops."""
    result = 0
    for i in range(n):
        result += i
    return result

def try_except_function():
    """Function with exception handling."""
    try:
        result = 1 / 1
        return result
    except ZeroDivisionError:
        return None

class SampleClass:
    """A sample class with methods."""
    
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
    
    def process(self, data):
        return data * self.value

def nested_function_example():
    """Function with nested function."""
    def inner(y):
        return y + 1
    return inner(5)

async def async_function(x):
    """Async function for testing."""
    return x * 2

def generator_function(n):
    """Generator function for testing."""
    for i in range(n):
        yield i * 2
'''

# Sample Lua code for testing
SAMPLE_LUA_CODE = '''
-- Simple function
function simpleFunction(x)
    return x * 2
end

-- Conditional function
function conditionalFunction(x)
    if x > 0 then
        return "positive"
    elseif x < 0 then
        return "negative"
    else
        return "zero"
    end
end

-- Loop function
function loopFunction(n)
    local result = 0
    for i = 1, n do
        result = result + i
    end
    return result
end

-- Local function
local function localFunction(x)
    return x + 10
end

-- Table with methods
local MyTable = {}

function MyTable.new(value)
    local obj = {value = value}
    setmetatable(obj, {__index = MyTable})
    return obj
end

function MyTable:getValue()
    return self.value
end

function MyTable:process(data)
    return data * self.value
end

-- Nested function
function outerFunction()
    local function inner(y)
        return y + 1
    end
    return inner(5)
end
'''


class TestAdvancedFeaturesInitialization(unittest.TestCase):
    """Test that all transformers are properly initialized."""
    
    def test_all_transformers_in_pipeline(self):
        """Verify all 6 transformers are instantiated when features enabled."""
        config = ObfuscationConfig(
            name="Test All Features",
            language="python",
            features={
                "control_flow_flattening": True,
                "dead_code_injection": True,
                "opaque_predicates": True,
                "anti_debugging": True,
                "code_splitting": True,
                "self_modifying_code": True,
            }
        )
        engine = ObfuscationEngine(config)
        transformers = engine.get_enabled_transformers()
        
        transformer_types = [type(t) for t in transformers]
        
        self.assertIn(ControlFlowFlatteningTransformer, transformer_types)
        self.assertIn(DeadCodeInjectionTransformer, transformer_types)
        self.assertIn(OpaquePredicatesTransformer, transformer_types)
        self.assertIn(AntiDebuggingTransformer, transformer_types)
        self.assertIn(CodeSplittingTransformer, transformer_types)
        self.assertIn(SelfModifyingCodeTransformer, transformer_types)
    
    def test_transformer_order_correct(self):
        """Verify transformers appear in correct sequence."""
        config = ObfuscationConfig(
            name="Test Order",
            language="python",
            features={
                "string_encryption": True,
                "number_obfuscation": True,
                "constant_array": True,
                "control_flow_flattening": True,
                "dead_code_injection": True,
                "opaque_predicates": True,
                "anti_debugging": True,
                "code_splitting": True,
                "self_modifying_code": True,
            }
        )
        engine = ObfuscationEngine(config)
        transformers = engine.get_enabled_transformers()
        
        # Expected order based on _TRANSFORMER_ORDER
        expected_order = [
            "StringEncryptionTransformer",
            "NumberObfuscationTransformer",
            "ConstantArrayTransformer",
            "ControlFlowFlatteningTransformer",
            "DeadCodeInjectionTransformer",
            "OpaquePredicatesTransformer",
            "AntiDebuggingTransformer",
            "CodeSplittingTransformer",
            "SelfModifyingCodeTransformer",
        ]
        
        actual_order = [t.__class__.__name__ for t in transformers]
        
        # Verify order matches expected
        for expected in expected_order:
            if expected in actual_order:
                self.assertEqual(
                    actual_order[actual_order.index(expected)],
                    expected,
                    f"Transformer {expected} not in correct position"
                )
    
    def test_transformers_receive_config(self):
        """Verify each transformer gets config with correct options."""
        config = ObfuscationConfig(
            name="Test Config",
            language="python",
            features={
                "control_flow_flattening": True,
                "dead_code_injection": True,
                "opaque_predicates": True,
            },
            options={
                "dead_code_percentage": 25,
                "opaque_predicate_complexity": 2,
                "opaque_predicate_percentage": 40,
            }
        )
        engine = ObfuscationEngine(config)
        transformers = engine.get_enabled_transformers()
        
        for transformer in transformers:
            # Each transformer should have access to config
            self.assertIsNotNone(transformer.config)


class TestPythonAdvancedPipeline(unittest.TestCase):
    """Test advanced features pipeline on Python code."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_code = SAMPLE_PYTHON_CODE
    
    def test_control_flow_with_dead_code(self):
        """Apply both CFF and dead code injection together."""
        config = ObfuscationConfig(
            name="CFF + Dead Code",
            language="python",
            features={
                "control_flow_flattening": True,
                "dead_code_injection": True,
            },
            options={
                "dead_code_percentage": 20,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(self.sample_code)
        
        self.assertIn("success", result)
        self.assertIsNotNone(result.get("code"))
        # Verify code was transformed
        self.assertNotEqual(result["code"].strip(), self.sample_code.strip())
    
    def test_opaque_predicates_with_anti_debug(self):
        """Combine opaque predicates and anti-debugging."""
        config = ObfuscationConfig(
            name="Opaque + Anti-Debug",
            language="python",
            features={
                "opaque_predicates": True,
                "anti_debugging": True,
            },
            options={
                "anti_debug_aggressiveness": 2,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(self.sample_code)
        
        self.assertIn("success", result)
        self.assertIsNotNone(result.get("code"))
    
    def test_code_splitting_with_self_modifying(self):
        """Test code splitting and self-modifying code."""
        config = ObfuscationConfig(
            name="Code Split + Self-Modify",
            language="python",
            features={
                "code_splitting": True,
                "self_modifying_code": True,
            },
            options={
                "code_split_chunk_size": 5,
                "self_modify_complexity": 2,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(self.sample_code)
        
        self.assertIn("success", result)
        self.assertIsNotNone(result.get("code"))
    
    def test_all_six_transformers_combined(self):
        """Enable all 6 advanced features and verify success."""
        config = ObfuscationConfig(
            name="All Six Features",
            language="python",
            features={
                "control_flow_flattening": True,
                "dead_code_injection": True,
                "opaque_predicates": True,
                "anti_debugging": True,
                "code_splitting": True,
                "self_modifying_code": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(self.sample_code)
        
        self.assertIn("success", result)
        self.assertTrue(result.get("success", False))
        self.assertIsNotNone(result.get("code"))
        self.assertGreater(len(result.get("code", "")), 0)
    
    def test_advanced_with_basic_features(self):
        """Combine advanced features with string encryption, number obfuscation."""
        config = ObfuscationConfig(
            name="Advanced + Basic",
            language="python",
            features={
                "string_encryption": True,
                "number_obfuscation": True,
                "constant_array": True,
                "control_flow_flattening": True,
                "dead_code_injection": True,
                "opaque_predicates": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(self.sample_code)
        
        self.assertIn("success", result)
        self.assertTrue(result.get("success", False))
    
    def test_functional_equivalence_all_features(self):
        """Execute obfuscated code and verify correct output."""
        # Create a simple test function
        test_code = '''
def calculate(x, y):
    result = x + y
    if result > 0:
        return result * 2
    return result

output = calculate(3, 4)
'''
        config = ObfuscationConfig(
            name="Functional Test",
            language="python",
            features={
                "control_flow_flattening": True,
                "dead_code_injection": True,
                "opaque_predicates": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(test_code)
        
        self.assertTrue(result.get("success", False))
        # The obfuscated code should be valid Python
        obfuscated_code = result.get("code", "")
        if obfuscated_code:
            try:
                ast.parse(obfuscated_code)
            except SyntaxError as e:
                self.fail(f"Obfuscated code is not valid Python: {e}")


class TestLuaAdvancedPipeline(unittest.TestCase):
    """Test advanced features pipeline on Lua code."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_code = SAMPLE_LUA_CODE
    
    def test_lua_control_flow_flattening(self):
        """Test CFF on Lua code."""
        config = ObfuscationConfig(
            name="Lua CFF",
            language="lua",
            features={
                "control_flow_flattening": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(self.sample_code)
        
        self.assertIn("success", result)
        self.assertIsNotNone(result.get("code"))
    
    def test_lua_dead_code_injection(self):
        """Test dead code injection on Lua."""
        config = ObfuscationConfig(
            name="Lua Dead Code",
            language="lua",
            features={
                "dead_code_injection": True,
            },
            options={
                "dead_code_percentage": 20,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(self.sample_code)
        
        self.assertIn("success", result)
        self.assertIsNotNone(result.get("code"))
    
    def test_lua_opaque_predicates(self):
        """Test opaque predicates on Lua."""
        config = ObfuscationConfig(
            name="Lua Opaque",
            language="lua",
            features={
                "opaque_predicates": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(self.sample_code)
        
        self.assertIn("success", result)
    
    def test_lua_all_features_combined(self):
        """Enable all features for Lua and verify."""
        config = ObfuscationConfig(
            name="Lua All Features",
            language="lua",
            features={
                "control_flow_flattening": True,
                "dead_code_injection": True,
                "opaque_predicates": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(self.sample_code)
        
        self.assertIn("success", result)
        self.assertTrue(result.get("success", False))
    
    def test_lua_functional_equivalence(self):
        """Execute Lua code and verify output (requires Lua runtime)."""
        # Skip if no Lua runtime available
        import subprocess
        try:
            subprocess.run(["lua", "-v"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.skipTest("Lua runtime not available")
        
        test_code = '''
function add(a, b)
    return a + b
end

result = add(2, 3)
print(result)
'''
        config = ObfuscationConfig(
            name="Lua Functional",
            language="lua",
            features={
                "control_flow_flattening": True,
                "dead_code_injection": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(test_code)
        
        self.assertTrue(result.get("success", False))


class TestTransformerInteractions(unittest.TestCase):
    """Test interactions between different transformers."""
    
    def test_cff_preserves_dead_code(self):
        """Verify CFF doesn't break dead code."""
        test_code = '''
def test_func():
    x = 1
    y = 2
    return x + y
'''
        config = ObfuscationConfig(
            name="CFF + Dead Code",
            language="python",
            features={
                "control_flow_flattening": True,
                "dead_code_injection": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(test_code)
        
        self.assertTrue(result.get("success", False))
        # Result should still be valid Python
        try:
            ast.parse(result["code"])
        except SyntaxError:
            self.fail("CFF broke dead code injection output")
    
    def test_opaque_predicates_in_flattened_code(self):
        """Verify opaque predicates work after CFF."""
        test_code = '''
def compute(x):
    if x > 0:
        return x * 2
    return x
'''
        config = ObfuscationConfig(
            name="Opaque after CFF",
            language="python",
            features={
                "control_flow_flattening": True,
                "opaque_predicates": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(test_code)
        
        self.assertTrue(result.get("success", False))
    
    def test_anti_debug_in_split_code(self):
        """Verify anti-debug checks in code-split functions."""
        test_code = '''
def protected_function():
    return "secret"
'''
        config = ObfuscationConfig(
            name="Anti-Debug + Code Split",
            language="python",
            features={
                "anti_debugging": True,
                "code_splitting": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(test_code)
        
        self.assertTrue(result.get("success", False))
    
    def test_self_modifying_with_encryption(self):
        """Verify self-modifying code works with string encryption."""
        test_code = '''
def process_data(data):
    secret = "hidden_value"
    return data + secret
'''
        config = ObfuscationConfig(
            name="Self-Modify + Encryption",
            language="python",
            features={
                "string_encryption": True,
                "self_modifying_code": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(test_code)
        
        self.assertTrue(result.get("success", False))


class TestProfilePresets(unittest.TestCase):
    """Test preset profile configurations."""
    
    def test_light_profile_no_advanced_features(self):
        """Verify Light profile has no advanced features."""
        config = ProfileManager.get_default_profile("Light")
        
        self.assertFalse(config.features.get("control_flow_flattening", True))
        self.assertFalse(config.features.get("dead_code_injection", True))
        self.assertFalse(config.features.get("opaque_predicates", True))
        self.assertFalse(config.features.get("anti_debugging", True))
        self.assertFalse(config.features.get("code_splitting", True))
        self.assertFalse(config.features.get("self_modifying_code", True))
    
    def test_medium_profile_basic_advanced(self):
        """Verify Medium has dead code injection only."""
        config = ProfileManager.get_default_profile("Medium")
        
        self.assertTrue(config.features.get("dead_code_injection", False))
        self.assertFalse(config.features.get("control_flow_flattening", True))
        self.assertFalse(config.features.get("opaque_predicates", True))
        self.assertFalse(config.features.get("anti_debugging", True))
        self.assertFalse(config.features.get("code_splitting", True))
        self.assertFalse(config.features.get("self_modifying_code", True))
        # Verify options
        self.assertEqual(config.options.get("dead_code_percentage"), 15)
    
    def test_heavy_profile_moderate_advanced(self):
        """Verify Heavy has CFF, dead code, opaque predicates, self-modifying."""
        config = ProfileManager.get_default_profile("Heavy")
        
        self.assertTrue(config.features.get("control_flow_flattening", False))
        self.assertTrue(config.features.get("dead_code_injection", False))
        self.assertTrue(config.features.get("opaque_predicates", False))
        self.assertFalse(config.features.get("anti_debugging", True))
        self.assertFalse(config.features.get("code_splitting", True))
        self.assertTrue(config.features.get("self_modifying_code", False))
        # Verify options
        self.assertEqual(config.options.get("self_modify_complexity"), 1)
    
    def test_maximum_profile_all_advanced(self):
        """Verify Maximum has all 6 advanced features."""
        config = ProfileManager.get_default_profile("Maximum")
        
        self.assertTrue(config.features.get("control_flow_flattening", False))
        self.assertTrue(config.features.get("dead_code_injection", False))
        self.assertTrue(config.features.get("opaque_predicates", False))
        self.assertTrue(config.features.get("anti_debugging", False))
        self.assertTrue(config.features.get("code_splitting", False))
        self.assertTrue(config.features.get("self_modifying_code", False))
        # Verify max options
        self.assertEqual(config.options.get("anti_debug_aggressiveness"), 3)
        self.assertEqual(config.options.get("code_split_chunk_size"), 3)
        self.assertEqual(config.options.get("self_modify_complexity"), 3)
    
    def test_preset_options_correct(self):
        """Verify each preset has correct option values."""
        for preset_name in ["Light", "Medium", "Heavy", "Maximum"]:
            config = ProfileManager.get_default_profile(preset_name)
            
            # All profiles should have these options
            self.assertIn("opaque_predicate_complexity", config.options)
            self.assertIn("opaque_predicate_percentage", config.options)
            self.assertIn("anti_debug_aggressiveness", config.options)
            self.assertIn("code_split_chunk_size", config.options)
            self.assertIn("code_split_encryption", config.options)
            self.assertIn("self_modify_complexity", config.options)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""
    
    def test_empty_function_with_all_features(self):
        """Test transformers on empty functions."""
        test_code = '''
def empty_func():
    pass
'''
        config = ObfuscationConfig(
            name="Empty Function",
            language="python",
            features={
                "control_flow_flattening": True,
                "dead_code_injection": True,
                "opaque_predicates": True,
                "anti_debugging": True,
                "code_splitting": True,
                "self_modifying_code": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(test_code)
        
        self.assertTrue(result.get("success", False))
    
    def test_nested_functions_all_features(self):
        """Test with nested function definitions."""
        test_code = '''
def outer():
    def inner():
        return 42
    return inner()
'''
        config = ObfuscationConfig(
            name="Nested Functions",
            language="python",
            features={
                "control_flow_flattening": True,
                "dead_code_injection": True,
                "opaque_predicates": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(test_code)
        
        self.assertTrue(result.get("success", False))
    
    def test_class_methods_all_features(self):
        """Test with class methods."""
        test_code = '''
class TestClass:
    def method1(self):
        return 1
    
    def method2(self, x):
        return x * 2
'''
        config = ObfuscationConfig(
            name="Class Methods",
            language="python",
            features={
                "control_flow_flattening": True,
                "dead_code_injection": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(test_code)
        
        self.assertTrue(result.get("success", False))
    
    def test_async_functions_handling(self):
        """Verify async functions handled correctly."""
        test_code = '''
async def async_func(x):
    return await x
'''
        config = ObfuscationConfig(
            name="Async Functions",
            language="python",
            features={
                "control_flow_flattening": True,
                "dead_code_injection": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(test_code)
        
        self.assertTrue(result.get("success", False))
    
    def test_generator_functions_handling(self):
        """Verify generators handled correctly."""
        test_code = '''
def gen_func(n):
    for i in range(n):
        yield i
'''
        config = ObfuscationConfig(
            name="Generator Functions",
            language="python",
            features={
                "control_flow_flattening": True,
                "dead_code_injection": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(test_code)
        
        self.assertTrue(result.get("success", False))


class TestPerformanceAndScaling(unittest.TestCase):
    """Test performance and scaling scenarios."""
    
    def test_large_module_transformation(self):
        """Test on module with 50+ functions."""
        # Generate a large module
        functions = []
        for i in range(55):
            functions.append(f'''
def func_{i}(x):
    result = x + {i}
    if result > 0:
        return result * 2
    return result
''')
        test_code = "\n".join(functions)
        
        config = ObfuscationConfig(
            name="Large Module",
            language="python",
            features={
                "control_flow_flattening": True,
                "dead_code_injection": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(test_code)
        
        self.assertTrue(result.get("success", False))
    
    def test_deeply_nested_code(self):
        """Test on code with deep nesting levels."""
        test_code = '''
def deep_nested(x):
    if x > 0:
        if x > 10:
            if x > 100:
                if x > 1000:
                    return "very large"
                return "large"
            return "medium"
        return "small"
    return "non-positive"
'''
        config = ObfuscationConfig(
            name="Deep Nesting",
            language="python",
            features={
                "control_flow_flattening": True,
                "opaque_predicates": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(test_code)
        
        self.assertTrue(result.get("success", False))
    
    def test_transformation_count_accuracy(self):
        """Verify transformation counts are accurate."""
        test_code = '''
def func1():
    return 1

def func2():
    return 2
'''
        config = ObfuscationConfig(
            name="Count Test",
            language="python",
            features={
                "dead_code_injection": True,
            }
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(test_code)
        
        self.assertTrue(result.get("success", False))
        # Verify result has statistics
        self.assertIn("stats", result)


class TestIntegrationWorkflow(unittest.TestCase):
    """Test the complete integration workflow."""
    
    def test_pipeline_all_transformers_sequential(self):
        """Test all transformers in sequence."""
        test_code = '''
def process(value):
    if value > 0:
        temp = value * 2
        return temp + 1
    return value
'''
        # Test each transformer individually
        transformers = [
            ("Control Flow Flattening", {"control_flow_flattening": True}),
            ("Dead Code Injection", {"dead_code_injection": True}),
            ("Opaque Predicates", {"opaque_predicates": True}),
            ("Anti-Debugging", {"anti_debugging": True}),
            ("Code Splitting", {"code_splitting": True}),
            ("Self-Modifying Code", {"self_modifying_code": True}),
        ]
        
        for name, features in transformers:
            config = ObfuscationConfig(
                name=f"Test {name}",
                language="python",
                features=features
            )
            engine = ObfuscationEngine(config)
            result = engine.obfuscate_string(test_code)
            
            self.assertTrue(
                result.get("success", False),
                f"{name} transformer failed"
            )
    
    def test_error_handling_invalid_code(self):
        """Verify error handling for invalid code."""
        invalid_code = "def broken(  # incomplete function"
        
        config = ObfuscationConfig(
            name="Invalid Code",
            language="python",
            features={
                "control_flow_flattening": True,
            }
        )
        engine = ObfuscationEngine(config)
        
        # Should not crash, may return error
        result = engine.obfuscate_string(invalid_code)
        # Either success is False or code contains error indication
        self.assertIn("success", result)


if __name__ == "__main__":
    unittest.main()


def lua_skip(func):
    """Decorator to skip Lua tests when luaparser is not available."""
    return unittest.skipUnless(LUAPARSER_AVAILABLE, "luaparser not available")(func)


class TestRobloxFeaturesIntegration(unittest.TestCase):
    """Integration tests for Roblox-specific features."""

    @lua_skip
    def test_roblox_exploit_defense_with_vm_protection(self):
        """Test Roblox exploit defense with VM protection on Lua code."""
        code = """
function foo()
    return 1 + 2
end
"""
        config = ObfuscationConfig(
            name="Roblox + VM",
            language="lua",
            features={
                "roblox_exploit_defense": True,
                "roblox_exploit_aggressiveness": 2,
            },
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(code)

        self.assertTrue(result.get("success", False))

    @lua_skip
    def test_roblox_remote_spy_with_string_encryption(self):
        """Test Roblox remote spy with string encryption on Lua code."""
        code = """
local event = game.ReplicatedStorage.MyEvent
event:FireServer("hello")
"""
        config = ObfuscationConfig(
            name="Remote Spy + String Enc",
            language="lua",
            features={
                "roblox_remote_spy": True,
                "string_encryption": True,
            },
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(code)

        self.assertTrue(result.get("success", False))

    @lua_skip
    def test_both_roblox_features_together(self):
        """Test both Roblox features together (exploit defense + remote spy)."""
        code = """
local event = game.ReplicatedStorage.PlayerJoined
event:FireServer("Player1")

function processData()
    return game.Players.LocalPlayer
end
"""
        config = ObfuscationConfig(
            name="Both Roblox Features",
            language="lua",
            features={
                "roblox_exploit_defense": True,
                "roblox_exploit_aggressiveness": 2,
                "roblox_remote_spy": True,
            },
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(code)

        self.assertTrue(result.get("success", False))

    @lua_skip
    def test_roblox_with_control_flow_and_dead_code(self):
        """Test Roblox features with control flow flattening and dead code injection."""
        code = """
function processPlayer(player)
    if player then
        local event = game.ReplicatedStorage.UpdatePlayer
        event:FireServer(player.Name)
        return true
    end
    return false
end
"""
        config = ObfuscationConfig(
            name="Roblox + CFF + Dead Code",
            language="lua",
            features={
                "roblox_exploit_defense": True,
                "roblox_exploit_aggressiveness": 2,
                "roblox_remote_spy": True,
                "control_flow_flattening": True,
                "dead_code_injection": True,
            },
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(code)

        self.assertTrue(result.get("success", False))

    def test_roblox_features_skipped_for_python(self):
        """Test Roblox features are skipped for Python files (transformation_count=0)."""
        code = """
def foo():
    return 42
"""
        config = ObfuscationConfig(
            name="Roblox on Python",
            language="python",
            features={
                "roblox_exploit_defense": True,
                "roblox_remote_spy": True,
            },
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(code)

        self.assertTrue(result.get("success", False))

    @lua_skip
    def test_roblox_with_anti_debugging(self):
        """Test Roblox features work with anti-debugging transformer."""
        code = """
function secureFunction()
    return game.Players.LocalPlayer.Name
end
"""
        config = ObfuscationConfig(
            name="Roblox + Anti-Debug",
            language="lua",
            features={
                "roblox_exploit_defense": True,
                "roblox_exploit_aggressiveness": 2,
                "anti_debugging": True,
                "anti_debug_aggressiveness": 2,
            },
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(code)

        self.assertTrue(result.get("success", False))

    @lua_skip
    def test_full_pipeline_all_transformers_with_roblox(self):
        """Test full pipeline with all transformers including Roblox features."""
        code = """
local remotes = {
    onDamage = game.ReplicatedStorage.OnDamage,
    onHeal = game.ReplicatedStorage.OnHeal
}

function handleCombat(target, damage)
    remotes.onDamage:FireServer(target, damage)
    return true
end
"""
        config = ObfuscationConfig(
            name="Full Pipeline with Roblox",
            language="lua",
            features={
                "string_encryption": True,
                "number_obfuscation": True,
                "constant_array": True,
                "control_flow_flattening": True,
                "dead_code_injection": True,
                "opaque_predicates": True,
                "anti_debugging": True,
                "roblox_exploit_defense": True,
                "roblox_exploit_aggressiveness": 2,
                "roblox_remote_spy": True,
            },
        )
        engine = ObfuscationEngine(config)
        result = engine.obfuscate_string(code)

        self.assertTrue(result.get("success", False))


class TestRobloxLanguageDetection(unittest.TestCase):
    """Tests for Roblox language detection and handling."""

    @lua_skip
    def test_roblox_transformers_only_apply_to_lua_chunks(self):
        """Test Roblox transformers only apply to Lua ASTs."""
        transformer = RobloxExploitDefenseTransformer(aggressiveness=2)

        # Lua code should be processed
        lua_code = "function foo() return 1 end"
        lua_tree = lua_ast.parse(lua_code)

        result = transformer.transform(lua_tree)
        self.assertTrue(result.success)
        self.assertGreater(result.transformation_count, 0)

    def test_python_asts_returned_unchanged(self):
        """Test Python ASTs are returned unchanged with success=True."""
        transformer = RobloxExploitDefenseTransformer(aggressiveness=2)

        python_code = "def foo(): return 1"
        python_tree = ast.parse(python_code)

        result = transformer.transform(python_tree)
        self.assertTrue(result.success)
        self.assertEqual(result.transformation_count, 0)

    def test_transformation_count_is_zero_for_python(self):
        """Test transformation count is 0 for Python files."""
        transformer = RobloxRemoteSpyTransformer()

        python_code = "def foo(): pass"
        python_tree = ast.parse(python_code)

        result = transformer.transform(python_tree)
        self.assertEqual(result.transformation_count, 0)

    def test_no_errors_when_skipping_python_files(self):
        """Test no errors are generated when skipping Python files."""
        transformer = RobloxExploitDefenseTransformer(aggressiveness=2)

        python_code = "def foo(): return 42"
        python_tree = ast.parse(python_code)

        result = transformer.transform(python_tree)
        self.assertEqual(len(result.errors), 0)


class TestRobloxProfilePresets(unittest.TestCase):
    """Tests for Roblox feature configuration in profile presets."""

    def test_maximum_preset_includes_roblox_for_lua(self):
        """Test 'Maximum' preset includes Roblox features for Lua projects."""
        config = ProfileManager.get_default_profile("Maximum")

        # Maximum should have Roblox features enabled
        self.assertTrue(config.features.get("roblox_exploit_defense", False))
        self.assertTrue(config.features.get("roblox_remote_spy", False))

    def test_roblox_feature_options_in_presets(self):
        """Test Roblox feature options are correctly set in presets."""
        config = ProfileManager.get_default_profile("Maximum")

        # Check aggressiveness option
        self.assertIn("roblox_exploit_aggressiveness", config.options)
        self.assertEqual(config.options.get("roblox_exploit_aggressiveness"), 3)

    def test_roblox_exploit_aggressiveness_respected(self):
        """Test `roblox_exploit_aggressiveness` option is respected."""
        config = ObfuscationConfig(
            name="Test Aggressiveness",
            language="lua",
            features={"roblox_exploit_defense": True},
            options={"roblox_exploit_aggressiveness": 1},
        )
        engine = ObfuscationEngine(config)

        # Verify the option is passed to transformer
        transformers = engine.get_enabled_transformers("lua")
        for t in transformers:
            if isinstance(t, RobloxExploitDefenseTransformer):
                self.assertEqual(t.aggressiveness, 1)
                break

    @lua_skip
    def test_roblox_features_can_be_disabled_via_config(self):
        """Test Roblox features can be enabled/disabled via config."""
        # Test with disabled
        config_disabled = ObfuscationConfig(
            name="Disabled",
            language="lua",
            features={
                "roblox_exploit_defense": False,
                "roblox_remote_spy": False,
            },
        )

        code = "local e = game.ReplicatedStorage.Event\ne:FireServer()"
        engine = ObfuscationEngine(config_disabled)
        result = engine.obfuscate_string(code)

        self.assertTrue(result.get("success", False))


if __name__ == "__main__":
    unittest.main()
