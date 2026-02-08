"""Integration tests for verification comment fixes.

This module tests the three verification fixes:
1. Attribute mangling with Store/Del contexts
2. Lua dot-notation handling
3. Roblox API preservation with LuaSymbolExtractor integration
"""

import ast
import unittest

try:
    from luaparser import ast as lua_ast
    LUAPARSER_AVAILABLE = True
except ImportError:
    LUAPARSER_AVAILABLE = False

from obfuscator.core.config import ObfuscationConfig
from obfuscator.processors.ast_transformer import MangleIndexesTransformer


class TestVerificationFixes(unittest.TestCase):
    """Integration tests for all verification fixes."""
    
    def test_python_attribute_all_contexts(self):
        """Test Comment 1: Attribute mangling handles Load, Store, and Del contexts."""
        config = ObfuscationConfig(
            version="1.0",
            name="test",
            language="python",
            preset="medium",
            features=set(),
            options={}
        )
        transformer = MangleIndexesTransformer(config)
        
        code = '''
class MyClass:
    def __init__(self):
        self.data = {}

obj = MyClass()
obj.attr = 42          # Store context
value = obj.attr       # Load context
obj.attr += 10         # AugAssign (both Load and Store)
del obj.attr           # Del context
'''
        tree = ast.parse(code)
        result = transformer.transform(tree)
        
        self.assertTrue(result.success, f"Transformation failed: {result.errors}")
        self.assertIn("attr", transformer.key_mappings)
        
        # Verify all three operations are present
        code_str = ast.unparse(result.ast_node)
        self.assertIn("setattr", code_str, "Store context should use setattr")
        self.assertIn("getattr", code_str, "Load context should use getattr")
        self.assertIn("delattr", code_str, "Del context should use delattr")
        
        # Verify the code compiles without errors
        try:
            compile(result.ast_node, '<test>', 'exec')
        except SyntaxError as e:
            self.fail(f"Transformed code has syntax errors: {e}")
    
    @unittest.skipIf(not LUAPARSER_AVAILABLE, "luaparser not available")
    def test_lua_dot_notation_transformation(self):
        """Test Comment 2: Lua dot-notation accesses are obfuscated."""
        config = ObfuscationConfig(
            version="1.0",
            name="test",
            language="lua",
            preset="medium",
            features=set(),  # Disable Roblox preservation
            options={}
        )
        transformer = MangleIndexesTransformer(config)
        transformer.preserve_roblox_api = False
        
        code = '''
local data = {name = "John", age = 30}
print(data.name)      -- Dot notation
print(data["age"])    -- Bracket notation
'''
        tree = lua_ast.parse(code)
        result = transformer.transform(tree)
        
        self.assertTrue(result.success, f"Transformation failed: {result.errors}")
        
        # Both dot and bracket notation should be transformed
        self.assertIn("name", transformer.key_mappings, "Dot notation key should be mapped")
        self.assertIn("age", transformer.key_mappings, "Bracket notation key should be mapped")
        self.assertGreater(result.transformation_count, 0, "Should transform at least one access")
    
    @unittest.skipIf(not LUAPARSER_AVAILABLE, "luaparser not available")
    def test_roblox_api_extractor_integration(self):
        """Test Comment 3: Roblox API preservation uses LuaSymbolExtractor."""
        config = ObfuscationConfig(
            version="1.0",
            name="test",
            language="lua",
            preset="medium",
            features={'roblox_exploit_defense'},
            options={}
        )
        transformer = MangleIndexesTransformer(config)
        
        code = '''
-- Roblox API calls that should be detected by LuaSymbolExtractor
local Players = game:GetService("Players")
local HttpService = game:GetService("HttpService")
local part = Instance.new("Part")
local event = Instance.new("RemoteEvent")

-- Custom code that should be obfuscated
local myData = {customKey = "value"}
print(myData.customKey)
print(myData["customKey"])
'''
        tree = lua_ast.parse(code)
        result = transformer.transform(tree)
        
        self.assertTrue(result.success, f"Transformation failed: {result.errors}")
        
        # Roblox API names should NOT be in mappings (preserved)
        self.assertNotIn("game", transformer.key_mappings, "game should be preserved")
        self.assertNotIn("GetService", transformer.key_mappings, "GetService should be preserved")
        self.assertNotIn("Instance", transformer.key_mappings, "Instance should be preserved")
        self.assertNotIn("new", transformer.key_mappings, "new should be preserved")
        
        # Service names detected by extractor should be preserved
        # Note: These might be in the patterns set even if not in key_mappings
        self.assertIn("Players", transformer.roblox_patterns, "Players should be in patterns")
        self.assertIn("HttpService", transformer.roblox_patterns, "HttpService should be in patterns")
        
        # Custom keys should be obfuscated
        self.assertIn("customKey", transformer.key_mappings, "Custom key should be obfuscated")
    
    @unittest.skipIf(not LUAPARSER_AVAILABLE, "luaparser not available")
    def test_lua_dot_notation_with_roblox_preservation(self):
        """Test that dot notation works correctly with Roblox API preservation."""
        config = ObfuscationConfig(
            version="1.0",
            name="test",
            language="lua",
            preset="medium",
            features={'roblox_exploit_defense'},
            options={}
        )
        transformer = MangleIndexesTransformer(config)
        
        code = '''
-- Roblox API with dot notation (should be preserved)
local service = game.GetService
local newInstance = Instance.new

-- Custom data with dot notation (should be obfuscated)
local myTable = {myField = 123}
local value = myTable.myField
'''
        tree = lua_ast.parse(code)
        result = transformer.transform(tree)
        
        self.assertTrue(result.success, f"Transformation failed: {result.errors}")
        
        # Roblox APIs should be preserved
        self.assertNotIn("GetService", transformer.key_mappings)
        self.assertNotIn("new", transformer.key_mappings)
        
        # Custom field should be obfuscated
        self.assertIn("myField", transformer.key_mappings)


if __name__ == '__main__':
    unittest.main()

