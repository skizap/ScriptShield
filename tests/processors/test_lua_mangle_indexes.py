"""Unit tests for MangleIndexesTransformer Lua support.

This module tests the Lua table access transformation and Roblox API
preservation features of MangleIndexesTransformer.
"""

import unittest

try:
    from luaparser import ast as lua_ast
    from luaparser import astnodes as lua_nodes
    LUAPARSER_AVAILABLE = True
except ImportError:
    LUAPARSER_AVAILABLE = False

from obfuscator.core.config import ObfuscationConfig
from obfuscator.processors.ast_transformer import MangleIndexesTransformer


@unittest.skipIf(not LUAPARSER_AVAILABLE, "luaparser not available")
class TestLuaTableAccess(unittest.TestCase):
    """Test Lua table access transformation."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = ObfuscationConfig(
            version="1.0",
            name="test",
            language="lua",
            preset="medium",
            features=set(),  # Disable Roblox preservation for these tests
            options={}
        )
        self.transformer = MangleIndexesTransformer(config)
        self.transformer.preserve_roblox_api = False  # Explicitly disable
    
    def test_simple_table_access(self):
        """Test transformation of simple table access."""
        code = '''
local data = {name = "John"}
print(data["name"])
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        self.assertGreater(result.transformation_count, 0)
        self.assertIn("name", self.transformer.key_mappings)
    
    def test_dot_notation_access(self):
        """Test transformation of dot notation table access."""
        code = '''
local data = {name = "John"}
print(data.name)
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)

        self.assertTrue(result.success)
        # Dot notation should now be transformed
        self.assertGreater(result.transformation_count, 0)
        self.assertIn("name", self.transformer.key_mappings)
    
    def test_nested_table_access(self):
        """Test transformation of nested table access."""
        code = '''
local data = {user = {name = "John"}}
print(data["user"]["name"])
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        # Should transform both "user" and "name" keys
        self.assertIn("user", self.transformer.key_mappings)
        self.assertIn("name", self.transformer.key_mappings)
    
    def test_skip_already_obfuscated(self):
        """Test that already obfuscated keys are skipped."""
        code = '''
local data = {["_0x1"] = "value"}
print(data["_0x1"])
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        # Should not create mapping for already obfuscated key
        self.assertNotIn("_0x1", self.transformer.key_mappings)
    
    def test_runtime_injection(self):
        """Test that Lua runtime is injected."""
        code = '''
local data = {key = "value"}
print(data["key"])
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        self.assertTrue(self.transformer.runtime_injected)
        self.assertGreater(len(self.transformer.key_mappings), 0)


@unittest.skipIf(not LUAPARSER_AVAILABLE, "luaparser not available")
class TestLuaRobloxPreservation(unittest.TestCase):
    """Test Roblox API pattern preservation."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = ObfuscationConfig(
            version="1.0",
            name="test",
            language="lua",
            preset="medium",
            features={'roblox_exploit_defense'},
            options={}
        )
        self.transformer = MangleIndexesTransformer(config)
    
    def test_preserve_game_get_service(self):
        """Test preservation of game:GetService() pattern."""
        code = '''
local Players = game:GetService("Players")
print(Players)
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        # "game" and "GetService" should not be obfuscated
        self.assertNotIn("game", self.transformer.key_mappings)
        self.assertNotIn("GetService", self.transformer.key_mappings)
    
    def test_preserve_instance_new(self):
        """Test preservation of Instance.new() pattern."""
        code = '''
local part = Instance.new("Part")
print(part)
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        # "Instance" and "new" should not be obfuscated
        self.assertNotIn("Instance", self.transformer.key_mappings)
        self.assertNotIn("new", self.transformer.key_mappings)
    
    def test_preserve_roblox_services(self):
        """Test preservation of Roblox service names."""
        code = '''
local storage = game:GetService("ReplicatedStorage")
local scripts = game:GetService("ServerScriptService")
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        # Service names should not be obfuscated
        self.assertNotIn("ReplicatedStorage", self.transformer.key_mappings)
        self.assertNotIn("ServerScriptService", self.transformer.key_mappings)
    
    def test_preserve_remote_objects(self):
        """Test preservation of Roblox remote object types."""
        code = '''
local event = Instance.new("RemoteEvent")
local func = Instance.new("RemoteFunction")
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        # Remote object types should not be obfuscated
        self.assertNotIn("RemoteEvent", self.transformer.key_mappings)
        self.assertNotIn("RemoteFunction", self.transformer.key_mappings)


    def test_mixed_roblox_and_custom(self):
        """Test that custom keys are obfuscated while Roblox APIs are preserved."""
        code = '''
local Players = game:GetService("Players")
local customData = {myKey = "myValue"}
print(customData["myKey"])
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)

        self.assertTrue(result.success)
        # Roblox APIs should not be obfuscated
        self.assertNotIn("game", self.transformer.key_mappings)
        self.assertNotIn("GetService", self.transformer.key_mappings)
        self.assertNotIn("Players", self.transformer.key_mappings)
        # Custom key should be obfuscated
        self.assertIn("myKey", self.transformer.key_mappings)

    def test_extractor_integration(self):
        """Test that LuaSymbolExtractor patterns are integrated."""
        code = '''
local service = game:GetService("HttpService")
local part = Instance.new("Part")
local customData = {myKey = "myValue"}
print(customData["myKey"])
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)

        self.assertTrue(result.success)
        # Roblox APIs detected by extractor should not be obfuscated
        self.assertNotIn("game", self.transformer.key_mappings)
        self.assertNotIn("GetService", self.transformer.key_mappings)
        self.assertNotIn("HttpService", self.transformer.key_mappings)
        self.assertNotIn("Instance", self.transformer.key_mappings)
        self.assertNotIn("new", self.transformer.key_mappings)
        # Custom key should be obfuscated
        self.assertIn("myKey", self.transformer.key_mappings)


@unittest.skipIf(not LUAPARSER_AVAILABLE, "luaparser not available")
class TestLuaRuntimeInjection(unittest.TestCase):
    """Test Lua runtime mapping injection."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = MangleIndexesTransformer()

    def test_runtime_injection_format(self):
        """Test that Lua runtime has correct format."""
        code = '''
local data = {key1 = "value1", key2 = "value2"}
print(data["key1"])
print(data["key2"])
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertTrue(self.transformer.runtime_injected)

        # Generate runtime to check format
        runtime = self.transformer._generate_lua_runtime()
        self.assertIn("local _key_map", runtime)
        self.assertIn("key1", runtime)
        self.assertIn("key2", runtime)

    def test_runtime_not_injected_without_transformations(self):
        """Test that runtime is not injected if no transformations occur."""
        code = '''
local x = 1
local y = 2
print(x + y)
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertFalse(self.transformer.runtime_injected)
        self.assertEqual(len(self.transformer.key_mappings), 0)

    def test_runtime_escapes_special_characters(self):
        """Test that runtime properly escapes special characters in keys."""
        # This test verifies the runtime generation handles edge cases
        self.transformer.key_mappings = {
            'key"with"quotes': '_0x1',
            'key\\with\\backslash': '_0x2'
        }

        runtime = self.transformer._generate_lua_runtime()

        # Should escape quotes and backslashes
        self.assertIn('\\"', runtime)
        self.assertIn('\\\\', runtime)


@unittest.skipIf(not LUAPARSER_AVAILABLE, "luaparser not available")
class TestLuaEdgeCases(unittest.TestCase):
    """Test edge cases in Lua transformation."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = MangleIndexesTransformer()
        self.transformer.preserve_roblox_api = False

    def test_empty_table(self):
        """Test transformation with empty table."""
        code = '''
local data = {}
print(data)
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertEqual(len(self.transformer.key_mappings), 0)

    def test_numeric_keys(self):
        """Test that numeric keys are not transformed."""
        code = '''
local data = {[1] = "first", [2] = "second"}
print(data[1])
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)

        self.assertTrue(result.success)
        # Numeric keys should not be transformed
        self.assertEqual(len(self.transformer.key_mappings), 0)

    def test_function_table_access(self):
        """Test transformation of table access in function."""
        code = '''
local function getData()
    local data = {key = "value"}
    return data["key"]
end
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertIn("key", self.transformer.key_mappings)

    def test_multiple_tables_same_key(self):
        """Test that same key in different tables uses same obfuscation."""
        code = '''
local table1 = {key = "value1"}
local table2 = {key = "value2"}
print(table1["key"])
print(table2["key"])
'''
        tree = lua_ast.parse(code)
        result = self.transformer.transform(tree)

        self.assertTrue(result.success)
        # "key" should appear only once in mappings
        self.assertIn("key", self.transformer.key_mappings)
        self.assertEqual(len(self.transformer.key_mappings), 1)


if __name__ == '__main__':
    unittest.main()

