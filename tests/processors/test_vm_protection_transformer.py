"""Unit tests for VMProtectionTransformer.

This module tests the VMProtectionTransformer class, which provides
VM-based code protection by converting functions to bytecode.
"""

import ast
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, 'src')

from obfuscator.processors.ast_transformer import VMProtectionTransformer, TransformResult
from obfuscator.core.config import ObfuscationConfig


class TestVMProtectionTransformer(unittest.TestCase):
    """Test cases for VMProtectionTransformer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ObfuscationConfig(
            name="test_vm_protection",
            features={"vm_protection": True},
            options={
                "vm_protection_complexity": 2,
                "vm_protect_all_functions": False,
                "vm_bytecode_encryption": True,
                "vm_protection_marker": "vm_protect"
            }
        )
        self.transformer = VMProtectionTransformer(config=self.config)

    def test_transformer_initialization(self):
        """Test VMProtectionTransformer initialization."""
        self.assertEqual(self.transformer.complexity, 2)
        self.assertFalse(self.transformer.protect_all_functions)
        self.assertTrue(self.transformer.bytecode_encryption)
        self.assertEqual(self.transformer.protection_marker, "vm_protect")
        self.assertEqual(self.transformer.transformation_count, 0)
        self.assertEqual(len(self.transformer.errors), 0)

    def test_transformer_initialization_with_parameters(self):
        """Test VMProtectionTransformer initialization with explicit parameters."""
        transformer = VMProtectionTransformer(
            complexity=3,
            protect_all_functions=True,
            bytecode_encryption=False,
            protection_marker="custom_marker"
        )
        self.assertEqual(transformer.complexity, 3)
        self.assertTrue(transformer.protect_all_functions)
        self.assertFalse(transformer.bytecode_encryption)
        self.assertEqual(transformer.protection_marker, "custom_marker")

    def test_transformer_initialization_invalid_complexity(self):
        """Test VMProtectionTransformer with invalid complexity level."""
        transformer = VMProtectionTransformer(complexity=5)
        # Should default to 2 for invalid complexity
        self.assertEqual(transformer.complexity, 2)

    def test_should_protect_function_with_decorator(self):
        """Test function protection detection with decorator."""
        code = '''
@vm_protect
def sensitive_function(x, y):
    return x + y
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        # Should detect the decorator
        result = self.transformer._should_protect_function(func_node)
        self.assertTrue(result)

    def test_should_not_protect_function_without_marker(self):
        """Test that functions without marker are not protected."""
        code = '''
def normal_function(x, y):
    return x + y
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        result = self.transformer._should_protect_function(func_node)
        self.assertFalse(result)

    def test_should_protect_all_functions(self):
        """Test protect_all_functions mode."""
        transformer = VMProtectionTransformer(protect_all_functions=True)
        
        code = '''
def any_function(x, y):
    return x + y
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        result = transformer._should_protect_function(func_node)
        self.assertTrue(result)

    def test_is_function_eligible_too_small(self):
        """Test that small functions are not eligible."""
        code = '''
def small_func():
    pass
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        result = self.transformer._is_function_eligible(func_node)
        self.assertFalse(result)

    def test_is_function_eligible_async_function(self):
        """Test that async functions are not eligible."""
        code = '''
async def async_func():
    await some_call()
    return result
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        result = self.transformer._is_function_eligible(func_node)
        self.assertFalse(result)

    def test_is_function_eligible_generator(self):
        """Test that generator functions are not eligible."""
        code = '''
def generator_func():
    yield 1
    yield 2
    yield 3
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        result = self.transformer._is_function_eligible(func_node)
        self.assertFalse(result)

    def test_is_function_eligible_valid_function(self):
        """Test that valid functions are eligible."""
        code = '''
def valid_function(x, y):
    result = x * 2
    result = result + y
    return result
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        result = self.transformer._is_function_eligible(func_node)
        self.assertTrue(result)

    def test_contains_yield(self):
        """Test yield detection in functions."""
        code = '''
def func_with_yield():
    yield 42
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        result = self.transformer._contains_yield(func_node)
        self.assertTrue(result)

    def test_compile_function_to_bytecode(self):
        """Test function compilation to bytecode."""
        code = '''
def test_func(x, y):
    result = x + y
    return result
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        # Mock the vm_bytecode module
        with patch.object(self.transformer, 'vm_bytecode') as mock_vm:
            mock_compiler = Mock()
            mock_compiler.compile_function.return_value = (
                [],  # instructions
                [1, 2, 3],  # constants
                3  # num_locals
            )
            
            mock_serializer = Mock()
            mock_serializer.serialize.return_value = [1, 2, 3, 4, 5]
            
            mock_vm.create_bytecode_compiler.return_value = mock_compiler
            mock_vm.BytecodeSerializer.return_value = mock_serializer
            mock_vm.encrypt_bytecode = lambda x, y: x
            
            bytecode, constants, num_locals = self.transformer._compile_function_to_bytecode(func_node)
            
            self.assertIsInstance(bytecode, list)
            self.assertIsInstance(constants, list)
            self.assertIsInstance(num_locals, int)

    def test_generate_python_wrapper(self):
        """Test Python wrapper function generation."""
        code = '''
def original_func(x, y):
    return x + y
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        bytecode = [1, 2, 3]
        constants = [42, "hello"]
        num_locals = 2
        
        wrapper = self.transformer._generate_python_wrapper(
            func_node, bytecode, constants, num_locals
        )
        
        self.assertIsInstance(wrapper, ast.FunctionDef)
        self.assertEqual(wrapper.name, "original_func")
        self.assertTrue(len(wrapper.body) > 0)
        
        # Check that wrapper contains bytecode assignments
        body_str = ast.unparse(wrapper)
        self.assertIn("__bytecode", body_str)
        self.assertIn("__constants", body_str)
        self.assertIn("__num_locals", body_str)

    def test_visit_functiondef_with_protection(self):
        """Test visiting FunctionDef node with protection."""
        code = '''
@vm_protect
def protected_func(x, y):
    result = x * 2
    result = result + y
    return result
'''
        tree = ast.parse(code)
        
        # Mock the compilation methods
        with patch.object(self.transformer, '_compile_function_to_bytecode') as mock_compile:
            with patch.object(self.transformer, '_generate_python_wrapper') as mock_wrapper:
                mock_compile.return_value = ([1, 2, 3], [42], 2)
                
                mock_wrapper.return_value = ast.FunctionDef(
                    name="protected_func",
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg='*args')],
                        vararg=None,
                        kwonlyargs=[],
                        kw_defaults=[],
                        kwarg=None,
                        defaults=[]
                    ),
                    body=[ast.Pass()],
                    decorator_list=[]
                )
                
                result = self.transformer.visit(tree)
                
                # Should have transformed the function
                self.assertEqual(self.transformer.transformation_count, 1)
                self.assertIn("protected_func", self.transformer.protected_functions)

    def test_visit_functiondef_without_protection(self):
        """Test visiting FunctionDef node without protection."""
        code = '''
def normal_func(x, y):
    return x + y
'''
        tree = ast.parse(code)
        
        result = self.transformer.visit(tree)
        
        # Should not have transformed the function
        self.assertEqual(self.transformer.transformation_count, 0)
        self.assertNotIn("normal_func", self.transformer.protected_functions)

    def test_transform_with_protected_function(self):
        """Test full transformation with protected function."""
        code = '''
@vm_protect
def sensitive_calculation(x, y):
    temp = x * 2
    result = temp + y
    return result

print(sensitive_calculation(5, 3))
'''
        tree = ast.parse(code)
        
        # Mock compilation to avoid actual bytecode generation
        with patch.object(self.transformer, '_compile_function_to_bytecode') as mock_compile:
            with patch.object(self.transformer, '_generate_python_wrapper') as mock_wrapper:
                mock_compile.return_value = ([1, 2, 3, 4, 5], [10, 20], 3)
                
                # Create a simple wrapper
                wrapper = ast.FunctionDef(
                    name="sensitive_calculation",
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg='*args')],
                        vararg=None,
                        kwonlyargs=[],
                        kw_defaults=[],
                        kwarg=None,
                        defaults=[]
                    ),
                    body=[ast.Return(value=ast.Constant(value=42))],
                    decorator_list=[]
                )
                mock_wrapper.return_value = wrapper
                
                result = self.transformer.transform(tree)
                
                self.assertTrue(result.success)
                self.assertEqual(result.transformation_count, 1)
                self.assertIn("sensitive_calculation", self.transformer.protected_functions)

    def test_transform_without_vm_bytecode_module(self):
        """Test transformation when vm_bytecode module is not available."""
        transformer = VMProtectionTransformer(config=self.config)
        transformer.vm_bytecode = None
        
        code = '''
@vm_protect
def test_func():
    return 42
'''
        tree = ast.parse(code)
        
        result = transformer.transform(tree)
        
        # Should fail gracefully
        self.assertFalse(result.success)
        self.assertTrue(len(result.errors) > 0)

    def test_transform_with_runtime_injection(self):
        """Test transformation includes runtime injection."""
        code = '''
@vm_protect
def test_func():
    return 42
'''
        tree = ast.parse(code)
        
        # Mock compilation
        with patch.object(self.transformer, '_compile_function_to_bytecode') as mock_compile:
            with patch.object(self.transformer, '_generate_python_wrapper') as mock_wrapper:
                mock_compile.return_value = ([1, 2, 3], [42], 1)
                
                wrapper = ast.FunctionDef(
                    name="test_func",
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg='*args')],
                        vararg=None,
                        kwonlyargs=[],
                        kw_defaults=[],
                        kwarg=None,
                        defaults=[]
                    ),
                    body=[ast.Return(value=ast.Constant(value=42))],
                    decorator_list=[]
                )
                mock_wrapper.return_value = wrapper
                
                result = self.transformer.transform(tree)
                
                self.assertTrue(result.success)
                # Runtime should be injected
                self.assertTrue(self.transformer.runtime_injected)

    def test_multiple_protected_functions(self):
        """Test transformation with multiple protected functions."""
        code = '''
@vm_protect
def func1(x):
    return x * 2

@vm_protect
def func2(y):
    return y + 10

@vm_protect
def func3(z):
    return z ** 2
'''
        tree = ast.parse(code)
        
        # Mock compilation for all functions
        with patch.object(self.transformer, '_compile_function_to_bytecode') as mock_compile:
            with patch.object(self.transformer, '_generate_python_wrapper') as mock_wrapper:
                mock_compile.return_value = ([1, 2, 3], [42], 1)
                
                wrapper = ast.FunctionDef(
                    name="test",
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg='*args')],
                        vararg=None,
                        kwonlyargs=[],
                        kw_defaults=[],
                        kwarg=None,
                        defaults=[]
                    ),
                    body=[ast.Return(value=ast.Constant(value=42))],
                    decorator_list=[]
                )
                mock_wrapper.return_value = wrapper
                
                result = self.transformer.transform(tree)
                
                self.assertTrue(result.success)
                self.assertEqual(result.transformation_count, 3)
                self.assertEqual(len(self.transformer.protected_functions), 3)

    def test_transform_preserves_unprotected_functions(self):
        """Test that unprotected functions are preserved."""
        code = '''
def normal_func1():
    return 1

@vm_protect
def protected_func():
    return 2

def normal_func2():
    return 3
'''
        tree = ast.parse(code)
        
        # Mock compilation
        with patch.object(self.transformer, '_compile_function_to_bytecode') as mock_compile:
            with patch.object(self.transformer, '_generate_python_wrapper') as mock_wrapper:
                mock_compile.return_value = ([1, 2, 3], [42], 1)
                
                wrapper = ast.FunctionDef(
                    name="protected_func",
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg='*args')],
                        vararg=None,
                        kwonlyargs=[],
                        kw_defaults=[],
                        kwarg=None,
                        defaults=[]
                    ),
                    body=[ast.Return(value=ast.Constant(value=42))],
                    decorator_list=[]
                )
                mock_wrapper.return_value = wrapper
                
                result = self.transformer.transform(tree)
                
                self.assertTrue(result.success)
                self.assertEqual(result.transformation_count, 1)
                self.assertIn("protected_func", self.transformer.protected_functions)
                self.assertNotIn("normal_func1", self.transformer.protected_functions)
                self.assertNotIn("normal_func2", self.transformer.protected_functions)

    def test_error_handling_during_compilation(self):
        """Test error handling during bytecode compilation."""
        code = '''
@vm_protect
def faulty_func():
    return 42
'''
        tree = ast.parse(code)
        
        # Mock compilation to raise an exception
        with patch.object(self.transformer, '_compile_function_to_bytecode') as mock_compile:
            mock_compile.side_effect = Exception("Compilation failed")
            
            result = self.transformer.transform(tree)
            
            # Should have errors but not crash
            self.assertTrue(len(result.errors) > 0)

    def test_complexity_levels(self):
        """Test different complexity levels."""
        for complexity in [1, 2, 3]:
            transformer = VMProtectionTransformer(complexity=complexity)
            self.assertEqual(transformer.complexity, complexity)

    def test_bytecode_encryption_option(self):
        """Test bytecode encryption option."""
        transformer = VMProtectionTransformer(bytecode_encryption=True)
        self.assertTrue(transformer.bytecode_encryption)
        
        transformer = VMProtectionTransformer(bytecode_encryption=False)
        self.assertFalse(transformer.bytecode_encryption)

    def test_custom_protection_marker(self):
        """Test custom protection marker."""
        code = '''
@custom_protect
def test_func():
    return 42
'''
        tree = ast.parse(code)
        
        transformer = VMProtectionTransformer(protection_marker="custom_protect")
        func_node = tree.body[0]
        
        result = transformer._should_protect_function(func_node)
        self.assertTrue(result)


class TestVMProtectionConfigIntegration(unittest.TestCase):
    """Test VM protection integration with ObfuscationConfig."""

    def test_config_with_vm_protection_options(self):
        """Test ObfuscationConfig with VM protection options."""
        config = ObfuscationConfig(
            name="test",
            features={"vm_protection": True},
            options={
                "vm_protection_complexity": 3,
                "vm_protect_all_functions": True,
                "vm_bytecode_encryption": False,
                "vm_protection_marker": "protect_me"
            }
        )
        
        transformer = VMProtectionTransformer(config=config)
        
        self.assertEqual(transformer.complexity, 3)
        self.assertTrue(transformer.protect_all_functions)
        self.assertFalse(transformer.bytecode_encryption)
        self.assertEqual(transformer.protection_marker, "protect_me")

    def test_config_validation_vm_options(self):
        """Test configuration validation with VM protection options."""
        config = ObfuscationConfig(
            name="test",
            features={"vm_protection": True},
            options={
                "vm_protection_complexity": 2,
                "vm_protect_all_functions": False,
                "vm_bytecode_encryption": True,
                "vm_protection_marker": "vm:protect"
            }
        )
        
        # Should not raise any exceptions
        config.validate()

    def test_invalid_vm_protection_complexity(self):
        """Test validation with invalid VM protection complexity."""
        config = ObfuscationConfig(
            name="test",
            features={"vm_protection": True},
            options={
                "vm_protection_complexity": 5,  # Invalid: should be 1-3
            }
        )
        
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_vm_protect_all_functions_type(self):
        """Test validation with invalid vm_protect_all_functions type."""
        config = ObfuscationConfig(
            name="test",
            features={"vm_protection": True},
            options={
                "vm_protect_all_functions": "yes",  # Should be bool
            }
        )
        
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_vm_bytecode_encryption_type(self):
        """Test validation with invalid vm_bytecode_encryption type."""
        config = ObfuscationConfig(
            name="test",
            features={"vm_protection": True},
            options={
                "vm_bytecode_encryption": "enabled",  # Should be bool
            }
        )
        
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_vm_protection_marker_type(self):
        """Test validation with invalid vm_protection_marker type."""
        config = ObfuscationConfig(
            name="test",
            features={"vm_protection": True},
            options={
                "vm_protection_marker": 123,  # Should be string
            }
        )
        
        with self.assertRaises(ValueError):
            config.validate()


if __name__ == "__main__":
    unittest.main()
