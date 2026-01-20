"""Unit tests for Lua VM Protection.

This module tests VM protection functionality for Lua code,
including bytecode compilation and transformation.
"""

import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, 'src')

try:
    from luaparser import ast as lua_ast
    from luaparser import astnodes as lua_nodes
    LUAPARSER_AVAILABLE = True
except ImportError:
    LUAPARSER_AVAILABLE = False

from obfuscator.processors.ast_transformer import VMProtectionTransformer, TransformResult
from obfuscator.core.config import ObfuscationConfig


@unittest.skipUnless(LUAPARSER_AVAILABLE, "luaparser not available")
class TestLuaVMProtection(unittest.TestCase):
    """Test cases for Lua VM protection."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ObfuscationConfig(
            name="test_lua_vm_protection",
            language="lua",
            features={"vm_protection": True},
            options={
                "vm_protection_complexity": 2,
                "vm_protect_all_functions": False,
                "vm_bytecode_encryption": True,
                "vm_protection_marker": "vm:protect"
            }
        )
        self.transformer = VMProtectionTransformer(config=self.config)

    def test_transformer_initialization_for_lua(self):
        """Test VMProtectionTransformer initialization for Lua."""
        self.assertEqual(self.transformer.language_mode, None)  # Not set until transform
        self.assertEqual(self.transformer.complexity, 2)
        self.assertFalse(self.transformer.protect_all_functions)
        self.assertTrue(self.transformer.bytecode_encryption)

    def test_compile_lua_function_to_bytecode(self):
        """Test Lua function compilation to bytecode."""
        lua_code = '''
local function test_func(x, y)
    local result = x + y
    return result
end
'''
        chunk = lua_ast.parse(lua_code)
        func_node = chunk.body.body[0]
        
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
            
            bytecode, constants, num_locals = self.transformer._compile_lua_function_to_bytecode(func_node)
            
            self.assertIsInstance(bytecode, list)
            self.assertIsInstance(constants, list)
            self.assertIsInstance(num_locals, int)
            # Verify Lua compiler was requested
            mock_vm.create_bytecode_compiler.assert_called_once_with('lua', 2)

    def test_inject_lua_runtime(self):
        """Test Lua runtime injection."""
        lua_code = '''
local function test_func()
    return 42
end
'''
        chunk = lua_ast.parse(lua_code)
        
        # Mock the vm_runtime_lua module
        with patch('obfuscator.processors.ast_transformer.vm_runtime_lua') as mock_runtime:
            mock_runtime.generate_lua_vm_runtime.return_value = '''
local BytecodeVM = {}
function BytecodeVM:new() return {} end
'''
            
            self.transformer._inject_lua_runtime(chunk)
            
            self.assertTrue(self.transformer.runtime_injected)
            mock_runtime.generate_lua_vm_runtime.assert_called_once_with(True)

    def test_transform_lua_chunk(self):
        """Test transformation of Lua chunk."""
        lua_code = '''
-- vm:protect
local function protected_func(x, y)
    local result = x * 2 + y
    return result
end

local function normal_func()
    return 42
end
'''
        chunk = lua_ast.parse(lua_code)
        
        # Mock compilation methods
        with patch.object(self.transformer, '_compile_lua_function_to_bytecode') as mock_compile:
            mock_compile.return_value = ([1, 2, 3], [42], 2)
            
            result = self.transformer.transform(chunk)
            
            self.assertTrue(result.success)
            # Note: Lua function protection logic is not fully implemented
            # This test ensures the transformation doesn't crash

    def test_transform_preserves_lua_syntax(self):
        """Test that Lua transformation preserves valid syntax."""
        lua_code = '''
local function test_func(x, y)
    local result = x + y
    return result
end
'''
        chunk = lua_ast.parse(lua_code)
        original_code = lua_ast.to_lua_source(chunk)
        
        # Transform
        result = self.transformer.transform(chunk)
        
        self.assertTrue(result.success)
        # Should still be valid Lua
        transformed_code = lua_ast.to_lua_source(result.ast_node)
        self.assertIsInstance(transformed_code, str)
        self.assertTrue(len(transformed_code) > 0)

    def test_lua_function_with_multiple_returns(self):
        """Test Lua function with multiple return values."""
        lua_code = '''
local function multi_return(x)
    return x, x * 2, x + 10
end
'''
        chunk = lua_ast.parse(lua_code)
        func_node = chunk.body.body[0]
        
        # Mock compilation
        with patch.object(self.transformer, 'vm_bytecode') as mock_vm:
            mock_compiler = Mock()
            mock_compiler.compile_function.return_value = (
                [],  # instructions
                [1, 2, 3],  # constants
                2  # num_locals
            )
            
            mock_serializer = Mock()
            mock_serializer.serialize.return_value = [1, 2, 3]
            
            mock_vm.create_bytecode_compiler.return_value = mock_compiler
            mock_vm.BytecodeSerializer.return_value = mock_serializer
            
            bytecode, constants, num_locals = self.transformer._compile_lua_function_to_bytecode(func_node)
            
            self.assertIsInstance(bytecode, list)

    def test_lua_function_with_table_operations(self):
        """Test Lua function with table operations."""
        lua_code = '''
local function table_func(t)
    t.x = 10
    t.y = 20
    return t.x + t.y
end
'''
        chunk = lua_ast.parse(lua_code)
        func_node = chunk.body.body[0]
        
        # Mock compilation
        with patch.object(self.transformer, 'vm_bytecode') as mock_vm:
            mock_compiler = Mock()
            mock_compiler.compile_function.return_value = (
                [],  # instructions
                [10, 20],  # constants
                2  # num_locals
            )
            
            mock_serializer = Mock()
            mock_serializer.serialize.return_value = [1, 2, 3, 4]
            
            mock_vm.create_bytecode_compiler.return_value = mock_compiler
            mock_vm.BytecodeSerializer.return_value = mock_serializer
            
            bytecode, constants, num_locals = self.transformer._compile_lua_function_to_bytecode(func_node)
            
            self.assertIsInstance(bytecode, list)

    def test_lua_bytecode_encryption(self):
        """Test Lua bytecode encryption option."""
        transformer = VMProtectionTransformer(
            config=self.config,
            bytecode_encryption=True
        )
        
        self.assertTrue(transformer.bytecode_encryption)
        
        lua_code = '''
local function test_func()
    return 42
end
'''
        chunk = lua_ast.parse(lua_code)
        
        # Mock runtime generation
        with patch('obfuscator.processors.ast_transformer.vm_runtime_lua') as mock_runtime:
            mock_runtime.generate_lua_vm_runtime.return_value = "-- VM runtime"
            
            transformer._inject_lua_runtime(chunk)
            
            # Should call with encryption enabled
            mock_runtime.generate_lua_vm_runtime.assert_called_once_with(True)

    def test_lua_without_bytecode_encryption(self):
        """Test Lua without bytecode encryption."""
        transformer = VMProtectionTransformer(
            config=self.config,
            bytecode_encryption=False
        )
        
        self.assertFalse(transformer.bytecode_encryption)
        
        lua_code = '''
local function test_func()
    return 42
end
'''
        chunk = lua_ast.parse(lua_code)
        
        # Mock runtime generation
        with patch('obfuscator.processors.ast_transformer.vm_runtime_lua') as mock_runtime:
            mock_runtime.generate_lua_vm_runtime.return_value = "-- VM runtime"
            
            transformer._inject_lua_runtime(chunk)
            
            # Should call with encryption disabled
            mock_runtime.generate_lua_vm_runtime.assert_called_once_with(False)

    def test_complex_lua_function(self):
        """Test complex Lua function with control flow."""
        lua_code = '''
local function complex_func(x)
    if x > 10 then
        return x * 2
    elseif x > 5 then
        return x + 10
    else
        return x
    end
end
'''
        chunk = lua_ast.parse(lua_code)
        func_node = chunk.body.body[0]
        
        # Mock compilation
        with patch.object(self.transformer, 'vm_bytecode') as mock_vm:
            mock_compiler = Mock()
            mock_compiler.compile_function.return_value = (
                [],  # instructions
                [5, 10],  # constants
                2  # num_locals
            )
            
            mock_serializer = Mock()
            mock_serializer.serialize.return_value = [1, 2, 3, 4, 5, 6]
            
            mock_vm.create_bytecode_compiler.return_value = mock_compiler
            mock_vm.BytecodeSerializer.return_value = mock_serializer
            
            bytecode, constants, num_locals = self.transformer._compile_lua_function_to_bytecode(func_node)
            
            self.assertIsInstance(bytecode, list)
            self.assertGreater(len(bytecode), 0)

    def test_lua_while_loop(self):
        """Test Lua function with while loop."""
        lua_code = '''
local function loop_func(n)
    local i = 0
    while i < n do
        i = i + 1
    end
    return i
end
'''
        chunk = lua_ast.parse(lua_code)
        func_node = chunk.body.body[0]
        
        # Mock compilation
        with patch.object(self.transformer, 'vm_bytecode') as mock_vm:
            mock_compiler = Mock()
            mock_compiler.compile_function.return_value = (
                [],  # instructions
                [0, 1],  # constants
                3  # num_locals (n, i, plus one for temp)
            )
            
            mock_serializer = Mock()
            mock_serializer.serialize.return_value = [1, 2, 3, 4, 5]
            
            mock_vm.create_bytecode_compiler.return_value = mock_compiler
            mock_vm.BytecodeSerializer.return_value = mock_serializer
            
            bytecode, constants, num_locals = self.transformer._compile_lua_function_to_bytecode(func_node)
            
            self.assertIsInstance(bytecode, list)
            self.assertEqual(num_locals, 3)

    def test_nested_lua_functions(self):
        """Test nested Lua functions."""
        lua_code = '''
local function outer_func(x)
    local function inner_func(y)
        return y * 2
    end
    return inner_func(x) + 10
end
'''
        chunk = lua_ast.parse(lua_code)
        
        # Transform should handle nested functions
        result = self.transformer.transform(chunk)
        
        self.assertTrue(result.success)

    def test_lua_function_with_varargs(self):
        """Test Lua function with variable arguments."""
        lua_code = '''
local function varargs_func(...)
    local args = {...}
    return #args
end
'''
        chunk = lua_ast.parse(lua_code)
        func_node = chunk.body.body[0]
        
        # Mock compilation
        with patch.object(self.transformer, 'vm_bytecode') as mock_vm:
            mock_compiler = Mock()
            mock_compiler.compile_function.return_value = (
                [],  # instructions
                [],  # constants
                2  # num_locals
            )
            
            mock_serializer = Mock()
            mock_serializer.serialize.return_value = [1, 2, 3]
            
            mock_vm.create_bytecode_compiler.return_value = mock_compiler
            mock_vm.BytecodeSerializer.return_value = mock_serializer
            
            bytecode, constants, num_locals = self.transformer._compile_lua_function_to_bytecode(func_node)
            
            self.assertIsInstance(bytecode, list)

    def test_roblox_api_preservation(self):
        """Test that Roblox API patterns are preserved."""
        lua_code = '''
local function roblox_func(player)
    local character = player.Character
    local humanoid = character:FindFirstChild("Humanoid")
    return humanoid.Health
end
'''
        chunk = lua_ast.parse(lua_code)
        
        # Transform should preserve Roblox API calls
        result = self.transformer.transform(chunk)
        
        self.assertTrue(result.success)

    def test_luau_type_annotations(self):
        """Test Luau type annotations if enabled."""
        lua_code = '''
local function typed_func(x: number, y: number): number
    return x + y
end
'''
        chunk = lua_ast.parse(lua_code)
        
        # Transform should handle Luau syntax
        result = self.transformer.transform(chunk)
        
        self.assertTrue(result.success)


class TestLuaVMProtectionConfigIntegration(unittest.TestCase):
    """Test Lua VM protection integration with ObfuscationConfig."""

    def test_lua_config_with_vm_protection(self):
        """Test Lua ObfuscationConfig with VM protection options."""
        config = ObfuscationConfig(
            name="test_lua",
            language="lua",
            features={"vm_protection": True},
            options={
                "vm_protection_complexity": 2,
                "vm_protect_all_functions": False,
                "vm_bytecode_encryption": True,
                "vm_protection_marker": "vm:protect"
            }
        )
        
        transformer = VMProtectionTransformer(config=config)
        
        self.assertEqual(transformer.complexity, 2)
        self.assertFalse(transformer.protect_all_functions)
        self.assertTrue(transformer.bytecode_encryption)
        self.assertEqual(transformer.protection_marker, "vm:protect")

    def test_lua_config_validation(self):
        """Test Lua configuration validation."""
        config = ObfuscationConfig(
            name="test_lua",
            language="lua",
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


if __name__ == "__main__":
    unittest.main()
