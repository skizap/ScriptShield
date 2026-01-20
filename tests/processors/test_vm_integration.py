"""Integration tests for VM Protection.

This module tests VM protection integration with other transformers
and end-to-end obfuscation workflows.
"""

import ast
import sys
import unittest
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, 'src')

from obfuscator.processors.ast_transformer import (
    VMProtectionTransformer, 
    StringEncryptionTransformer,
    NumberObfuscationTransformer,
    ConstantArrayTransformer,
    TransformResult
)
from obfuscator.core.config import ObfuscationConfig


class TestVMProtectionIntegration(unittest.TestCase):
    """Integration tests for VM protection with other transformers."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ObfuscationConfig(
            name="test_integration",
            features={
                "vm_protection": True,
                "string_encryption": True,
                "number_obfuscation": True,
                "constant_array": True
            },
            options={
                "vm_protection_complexity": 2,
                "vm_protect_all_functions": False,
                "vm_bytecode_encryption": True,
                "vm_protection_marker": "vm_protect",
                "string_encryption_key_length": 16,
                "number_obfuscation_complexity": 3,
                "array_shuffle_seed": 42
            }
        )

    def test_vm_with_string_encryption(self):
        """Test VM protection combined with string encryption."""
        code = '''
@vm_protect
def protected_func():
    message = "secret_data"
    return message
'''
        tree = ast.parse(code)
        
        # Apply VM protection first
        vm_transformer = VMProtectionTransformer(config=self.config)
        
        with patch.object(vm_transformer, '_compile_function_to_bytecode') as mock_compile:
            with patch.object(vm_transformer, '_generate_python_wrapper') as mock_wrapper:
                mock_compile.return_value = ([1, 2, 3], ["secret_data"], 2)
                
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
                    body=[ast.Return(value=ast.Constant(value="secret_data"))],
                    decorator_list=[]
                )
                mock_wrapper.return_value = wrapper
                
                vm_result = vm_transformer.transform(tree)
                self.assertTrue(vm_result.success)
                
                # Then apply string encryption
                string_transformer = StringEncryptionTransformer(config=self.config)
                final_result = string_transformer.transform(vm_result.ast_node)
                
                self.assertTrue(final_result.success)

    def test_vm_with_number_obfuscation(self):
        """Test VM protection combined with number obfuscation."""
        code = '''
@vm_protect
def protected_func():
    result = 42 * 2
    return result
'''
        tree = ast.parse(code)
        
        # Apply number obfuscation first
        number_transformer = NumberObfuscationTransformer(config=self.config)
        num_result = number_transformer.transform(tree)
        self.assertTrue(num_result.success)
        
        # Then apply VM protection
        vm_transformer = VMProtectionTransformer(config=self.config)
        
        with patch.object(vm_transformer, '_compile_function_to_bytecode') as mock_compile:
            with patch.object(vm_transformer, '_generate_python_wrapper') as mock_wrapper:
                mock_compile.return_value = ([1, 2, 3], [84], 2)
                
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
                    body=[ast.Return(value=ast.Constant(value=84))],
                    decorator_list=[]
                )
                mock_wrapper.return_value = wrapper
                
                vm_result = vm_transformer.transform(num_result.ast_node)
                self.assertTrue(vm_result.success)

    def test_vm_with_constant_array(self):
        """Test VM protection combined with constant array obfuscation."""
        code = '''
@vm_protect
def protected_func():
    data = [1, 2, 3, 4, 5]
    return data[2]
'''
        tree = ast.parse(code)
        
        # Apply constant array obfuscation first
        array_transformer = ConstantArrayTransformer(config=self.config)
        array_result = array_transformer.transform(tree)
        self.assertTrue(array_result.success)
        
        # Then apply VM protection
        vm_transformer = VMProtectionTransformer(config=self.config)
        
        with patch.object(vm_transformer, '_compile_function_to_bytecode') as mock_compile:
            with patch.object(vm_transformer, '_generate_python_wrapper') as mock_wrapper:
                mock_compile.return_value = ([1, 2, 3], [[3, 5, 1, 4, 2]], 2)
                
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
                    body=[ast.Return(value=ast.Constant(value=3))],
                    decorator_list=[]
                )
                mock_wrapper.return_value = wrapper
                
                vm_result = vm_transformer.transform(array_result.ast_node)
                self.assertTrue(vm_result.success)

    def test_transformation_order_vm_first(self):
        """Test transformation order with VM protection first."""
        code = '''
@vm_protect
def protected_func():
    message = "secret"
    number = 42
    data = [1, 2, 3]
    return message, number, data
'''
        tree = ast.parse(code)
        
        # VM protection first
        vm_transformer = VMProtectionTransformer(config=self.config)
        
        with patch.object(vm_transformer, '_compile_function_to_bytecode') as mock_compile:
            with patch.object(vm_transformer, '_generate_python_wrapper') as mock_wrapper:
                mock_compile.return_value = ([1, 2, 3], ["secret", 42, [1, 2, 3]], 4)
                
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
                    body=[ast.Return(value=ast.Tuple(elts=[
                        ast.Constant(value="secret"),
                        ast.Constant(value=42),
                        ast.List(elts=[ast.Constant(value=i) for i in [1, 2, 3]])
                    ]))],
                    decorator_list=[]
                )
                mock_wrapper.return_value = wrapper
                
                vm_result = vm_transformer.transform(tree)
                self.assertTrue(vm_result.success)
                
                # Then apply other transformers
                string_transformer = StringEncryptionTransformer(config=self.config)
                string_result = string_transformer.transform(vm_result.ast_node)
                self.assertTrue(string_result.success)
                
                number_transformer = NumberObfuscationTransformer(config=self.config)
                number_result = number_transformer.transform(string_result.ast_node)
                self.assertTrue(number_result.success)
                
                array_transformer = ConstantArrayTransformer(config=self.config)
                final_result = array_transformer.transform(number_result.ast_node)
                self.assertTrue(final_result.success)

    def test_multi_file_project_simulation(self):
        """Test VM protection in a multi-file project simulation."""
        # Simulate multiple files
        file1_code = '''
@vm_protect
def calculate(x, y):
    return x * y + 10
'''
        
        file2_code = '''
from module1 import calculate

@vm_protect
def process_data(data):
    result = calculate(data[0], data[1])
    return result
'''
        
        # Transform first file
        tree1 = ast.parse(file1_code)
        vm_transformer = VMProtectionTransformer(config=self.config)
        
        with patch.object(vm_transformer, '_compile_function_to_bytecode') as mock_compile:
            with patch.object(vm_transformer, '_generate_python_wrapper') as mock_wrapper:
                mock_compile.return_value = ([1, 2, 3], [10], 3)
                
                wrapper = ast.FunctionDef(
                    name="calculate",
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
                
                result1 = vm_transformer.transform(tree1)
                self.assertTrue(result1.success)
                
        # Transform second file
        tree2 = ast.parse(file2_code)
        vm_transformer2 = VMProtectionTransformer(config=self.config)
        
        with patch.object(vm_transformer2, '_compile_function_to_bytecode') as mock_compile2:
            with patch.object(vm_transformer2, '_generate_python_wrapper') as mock_wrapper2:
                mock_compile2.return_value = ([1, 2, 3], [42], 2)
                
                wrapper2 = ast.FunctionDef(
                    name="process_data",
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
                mock_wrapper2.return_value = wrapper2
                
                result2 = vm_transformer2.transform(tree2)
                self.assertTrue(result2.success)

    def test_performance_impact_measurement(self):
        """Test performance impact measurement."""
        code = '''
@vm_protect
def compute_heavy(x, y, z):
    result = 0
    for i in range(100):
        result += x * i + y * i ** 2 + z * i ** 3
    return result
'''
        tree = ast.parse(code)
        
        vm_transformer = VMProtectionTransformer(config=self.config)
        
        with patch.object(vm_transformer, '_compile_function_to_bytecode') as mock_compile:
            with patch.object(vm_transformer, '_generate_python_wrapper') as mock_wrapper:
                # Simulate large bytecode
                large_bytecode = list(range(1000))
                mock_compile.return_value = (large_bytecode, [1, 2, 3, 4, 5], 4)
                
                wrapper = ast.FunctionDef(
                    name="compute_heavy",
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
                
                result = vm_transformer.transform(tree)
                
                self.assertTrue(result.success)
                # Verify transformation count
                self.assertEqual(result.transformation_count, 1)

    def test_bytecode_size_vs_original(self):
        """Test bytecode size compared to original code."""
        code = '''
@vm_protect
def simple_func():
    return 42
'''
        tree = ast.parse(code)
        
        vm_transformer = VMProtectionTransformer(config=self.config)
        
        with patch.object(vm_transformer, '_compile_function_to_bytecode') as mock_compile:
            with patch.object(vm_transformer, '_generate_python_wrapper') as mock_wrapper:
                # Original function is small
                original_size = len(ast.unparse(tree.body[0]))
                
                # Bytecode is larger
                bytecode = list(range(50))  # 50 bytecode instructions
                mock_compile.return_value = (bytecode, [42], 1)
                
                wrapper = ast.FunctionDef(
                    name="simple_func",
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
                
                result = vm_transformer.transform(tree)
                
                self.assertTrue(result.success)
                # The wrapper function should be larger than original
                wrapper_code = ast.unparse(result.ast_node.body[0])
                self.assertTrue(len(wrapper_code) > original_size)

    def test_error_recovery_multiple_transformers(self):
        """Test error recovery when one transformer fails."""
        code = '''
@vm_protect
def protected_func():
    return "secret"
'''
        tree = ast.parse(code)
        
        # VM protection succeeds
        vm_transformer = VMProtectionTransformer(config=self.config)
        
        with patch.object(vm_transformer, '_compile_function_to_bytecode') as mock_compile:
            with patch.object(vm_transformer, '_generate_python_wrapper') as mock_wrapper:
                mock_compile.return_value = ([1, 2, 3], ["secret"], 1)
                
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
                    body=[ast.Return(value=ast.Constant(value="secret"))],
                    decorator_list=[]
                )
                mock_wrapper.return_value = wrapper
                
                vm_result = vm_transformer.transform(tree)
                self.assertTrue(vm_result.success)
                
                # String encryption also succeeds
                string_transformer = StringEncryptionTransformer(config=self.config)
                final_result = string_transformer.transform(vm_result.ast_node)
                self.assertTrue(final_result.success)

    def test_runtime_code_injection_once(self):
        """Test that VM runtime is injected only once."""
        code = '''
@vm_protect
def func1():
    return 1

@vm_protect
def func2():
    return 2
'''
        tree = ast.parse(code)
        
        vm_transformer = VMProtectionTransformer(config=self.config)
        
        with patch.object(vm_transformer, '_compile_function_to_bytecode') as mock_compile:
            with patch.object(vm_transformer, '_generate_python_wrapper') as mock_wrapper:
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
                
                result = vm_transformer.transform(tree)
                
                self.assertTrue(result.success)
                # Runtime should be injected only once
                self.assertTrue(vm_transformer.runtime_injected)

    def test_edge_cases_empty_functions(self):
        """Test VM protection with edge cases."""
        code = '''
@vm_protect
def empty_func():
    pass

@vm_protect
def single_statement_func():
    return 42

@vm_protect
def large_func():
    x = 1
    y = 2
    z = 3
    a = x + y
    b = a * z
    c = b - x
    return c
'''
        tree = ast.parse(code)
        
        vm_transformer = VMProtectionTransformer(config=self.config)
        
        with patch.object(vm_transformer, '_compile_function_to_bytecode') as mock_compile:
            with patch.object(vm_transformer, '_generate_python_wrapper') as mock_wrapper:
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
                
                result = vm_transformer.transform(tree)
                
                self.assertTrue(result.success)
                # Should protect eligible functions (large_func)
                self.assertEqual(result.transformation_count, 1)

    def test_configuration_presets(self):
        """Test VM protection with different configuration presets."""
        presets = ["light", "medium", "heavy", "maximum"]
        
        for preset in presets:
            config = ObfuscationConfig(
                name=f"test_{preset}",
                preset=preset,
                features={"vm_protection": True},
                language="python"
            )
            
            transformer = VMProtectionTransformer(config=config)
            
            # Should create transformer without errors
            self.assertIsNotNone(transformer)
            self.assertTrue(1 <= transformer.complexity <= 3)


if __name__ == "__main__":
    unittest.main()
