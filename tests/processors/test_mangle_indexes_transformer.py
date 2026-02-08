"""Unit tests for MangleIndexesTransformer.

This module tests the MangleIndexesTransformer class which obfuscates
dictionary and attribute access in Python code.
"""

import ast
import unittest
from typing import Any

from obfuscator.core.config import ObfuscationConfig
from obfuscator.processors.ast_transformer import MangleIndexesTransformer


class TestMangleIndexesTransformerInit(unittest.TestCase):
    """Test MangleIndexesTransformer initialization and configuration."""
    
    def test_init_without_config(self):
        """Test initialization without configuration."""
        transformer = MangleIndexesTransformer()
        self.assertEqual(transformer.identifier_prefix, '_0x')
        self.assertTrue(transformer.preserve_roblox_api)
        self.assertEqual(transformer.transformation_count, 0)
        self.assertEqual(len(transformer.key_mappings), 0)
    
    def test_init_with_config(self):
        """Test initialization with custom configuration."""
        config = ObfuscationConfig(
            version="1.0",
            name="test",
            language="python",
            preset="medium",
            features=set(),
            options={'identifier_prefix': '_custom_'}
        )
        transformer = MangleIndexesTransformer(config)
        self.assertEqual(transformer.identifier_prefix, '_custom_')
    
    def test_init_with_roblox_defense(self):
        """Test initialization with Roblox defense enabled."""
        config = ObfuscationConfig(
            version="1.0",
            name="test",
            language="lua",
            preset="medium",
            features={'roblox_exploit_defense'},
            options={}
        )
        transformer = MangleIndexesTransformer(config)
        self.assertTrue(transformer.preserve_roblox_api)


class TestPythonDictionaryAccess(unittest.TestCase):
    """Test Python dictionary subscript access transformation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.transformer = MangleIndexesTransformer()
    
    def test_simple_dictionary_access(self):
        """Test transformation of simple dictionary access."""
        code = '''
data = {"name": "John"}
print(data["name"])
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        self.assertGreater(result.transformation_count, 0)
        self.assertIn("name", self.transformer.key_mappings)
        
        # Verify runtime was injected
        self.assertTrue(self.transformer.runtime_injected)
        self.assertIsInstance(result.ast_node.body[0], ast.Assign)
        self.assertEqual(result.ast_node.body[0].targets[0].id, '_key_map')
    
    def test_nested_dictionary_access(self):
        """Test transformation of nested dictionary access."""
        code = '''
data = {"user": {"name": "John"}}
print(data["user"]["name"])
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        self.assertIn("user", self.transformer.key_mappings)
        self.assertIn("name", self.transformer.key_mappings)
    
    def test_skip_already_obfuscated(self):
        """Test that already obfuscated keys are skipped."""
        code = '''
data = {"_0x1": "value"}
print(data["_0x1"])
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        # Should not create mapping for already obfuscated key
        self.assertNotIn("_0x1", self.transformer.key_mappings)
    
    def test_skip_non_string_keys(self):
        """Test that non-string keys are not transformed."""
        code = '''
data = {0: "value", 1: "another"}
print(data[0])
print(data[1])
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        # Should not transform integer keys
        self.assertEqual(len(self.transformer.key_mappings), 0)


class TestPythonAttributeAccess(unittest.TestCase):
    """Test Python attribute access transformation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.transformer = MangleIndexesTransformer()
    
    def test_simple_attribute_access(self):
        """Test transformation of simple attribute access."""
        code = '''
class MyClass:
    def __init__(self):
        self.custom_attr = 42

obj = MyClass()
print(obj.custom_attr)
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        self.assertIn("custom_attr", self.transformer.key_mappings)
    
    def test_skip_builtin_attributes(self):
        """Test that built-in attributes are preserved."""
        code = '''
class MyClass:
    pass

print(MyClass.__name__)
print(MyClass.__dict__)
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        # Should not transform built-in attributes
        self.assertNotIn("__name__", self.transformer.key_mappings)
        self.assertNotIn("__dict__", self.transformer.key_mappings)
    
    def test_skip_common_methods(self):
        """Test that common standard library methods are preserved."""
        code = '''
data = []
data.append(1)
data.extend([2, 3])
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        # Should not transform common methods
        self.assertNotIn("append", self.transformer.key_mappings)
        self.assertNotIn("extend", self.transformer.key_mappings)


class TestPythonRuntimeInjection(unittest.TestCase):
    """Test Python runtime mapping injection."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = MangleIndexesTransformer()

    def test_runtime_injection_position(self):
        """Test that runtime is injected at the correct position."""
        code = '''
"""Module docstring."""
from __future__ import annotations

data = {"key": "value"}
print(data["key"])
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertTrue(self.transformer.runtime_injected)

        # Runtime should be after docstring and __future__ import
        # Body should be: [docstring, __future__, _key_map, data assignment, print]
        self.assertGreater(len(result.ast_node.body), 3)

        # Find the _key_map assignment
        key_map_found = False
        for stmt in result.ast_node.body:
            if isinstance(stmt, ast.Assign):
                if isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == '_key_map':
                    key_map_found = True
                    break

        self.assertTrue(key_map_found, "Runtime mapping not found in transformed code")

    def test_runtime_not_injected_without_transformations(self):
        """Test that runtime is not injected if no transformations occur."""
        code = '''
x = 1
y = 2
print(x + y)
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertFalse(self.transformer.runtime_injected)
        self.assertEqual(len(self.transformer.key_mappings), 0)

    def test_multiple_transformations_single_runtime(self):
        """Test that runtime is only injected once for multiple transformations."""
        code = '''
data1 = {"key1": "value1"}
data2 = {"key2": "value2"}
print(data1["key1"])
print(data2["key2"])
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)

        self.assertTrue(result.success)
        self.assertTrue(self.transformer.runtime_injected)

        # Count _key_map assignments (should be exactly 1)
        key_map_count = 0
        for stmt in result.ast_node.body:
            if isinstance(stmt, ast.Assign):
                if isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == '_key_map':
                    key_map_count += 1

        self.assertEqual(key_map_count, 1, "Runtime should be injected exactly once")


class TestPythonRoundTrip(unittest.TestCase):
    """Test functional equivalence of transformed code."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = MangleIndexesTransformer()

    def test_dictionary_access_equivalence(self):
        """Test that transformed dictionary access produces same results."""
        code = '''
data = {"name": "John", "age": 30}
result = data["name"]
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)

        self.assertTrue(result.success)

        # Execute original code
        original_namespace = {}
        exec(compile(ast.parse(code), '<string>', 'exec'), original_namespace)

        # Execute transformed code
        transformed_namespace = {}
        exec(compile(result.ast_node, '<string>', 'exec'), transformed_namespace)

        # Results should be the same
        self.assertEqual(original_namespace['result'], transformed_namespace['result'])

    def test_deterministic_key_generation(self):
        """Test that key generation is deterministic within a transformation."""
        code = '''
data = {"key": "value"}
x = data["key"]
y = data["key"]
'''
        tree1 = ast.parse(code)
        result1 = self.transformer.transform(tree1)

        # Reset transformer
        transformer2 = MangleIndexesTransformer()
        tree2 = ast.parse(code)
        result2 = transformer2.transform(tree2)

        # Both should generate the same mapping for "key"
        self.assertEqual(
            self.transformer.key_mappings["key"],
            transformer2.key_mappings["key"]
        )


class TestMixedAccessPatterns(unittest.TestCase):
    """Test mixed dictionary and attribute access patterns."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = MangleIndexesTransformer()

    def test_mixed_subscript_and_attribute(self):
        """Test transformation of mixed access patterns."""
        code = '''
class Container:
    def __init__(self):
        self.data = {"key": "value"}

obj = Container()
result = obj.data["key"]
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)

        self.assertTrue(result.success)
        # Should transform both "data" attribute and "key" subscript
        self.assertIn("data", self.transformer.key_mappings)
        self.assertIn("key", self.transformer.key_mappings)

    def test_chained_attribute_access(self):
        """Test transformation of chained attribute access."""
        code = '''
class Inner:
    def __init__(self):
        self.value = 42

class Outer:
    def __init__(self):
        self.inner = Inner()

obj = Outer()
result = obj.inner.value
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)

        self.assertTrue(result.success)
        # Should transform both "inner" and "value" attributes
        self.assertIn("inner", self.transformer.key_mappings)
        self.assertIn("value", self.transformer.key_mappings)


if __name__ == '__main__':
    unittest.main()

