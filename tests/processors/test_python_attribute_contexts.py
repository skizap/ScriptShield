"""Unit tests for MangleIndexesTransformer Python attribute context handling.

This module tests that attribute access, assignment, and deletion are properly
transformed with the correct contexts (Load, Store, Del).
"""

import ast
import unittest

from obfuscator.core.config import ObfuscationConfig
from obfuscator.processors.ast_transformer import MangleIndexesTransformer


class TestPythonAttributeContexts(unittest.TestCase):
    """Test Python attribute transformation for different contexts."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = ObfuscationConfig(
            version="1.0",
            name="test",
            language="python",
            preset="medium",
            features=set(),
            options={}
        )
        self.transformer = MangleIndexesTransformer(config)
    
    def test_attribute_load_context(self):
        """Test transformation of attribute access (Load context)."""
        code = '''
class MyClass:
    def __init__(self):
        self.my_attr = 42

obj = MyClass()
value = obj.my_attr
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        self.assertGreater(result.transformation_count, 0)
        self.assertIn("my_attr", self.transformer.key_mappings)
        
        # Verify the transformed code contains getattr
        code_str = ast.unparse(result.ast_node)
        self.assertIn("getattr", code_str)
        self.assertIn("_key_map", code_str)
    
    def test_attribute_store_context(self):
        """Test transformation of attribute assignment (Store context)."""
        code = '''
class MyClass:
    pass

obj = MyClass()
obj.my_attr = 42
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        self.assertGreater(result.transformation_count, 0)
        self.assertIn("my_attr", self.transformer.key_mappings)
        
        # Verify the transformed code contains setattr
        code_str = ast.unparse(result.ast_node)
        self.assertIn("setattr", code_str)
        self.assertIn("_key_map", code_str)
    
    def test_attribute_del_context(self):
        """Test transformation of attribute deletion (Del context)."""
        code = '''
class MyClass:
    def __init__(self):
        self.my_attr = 42

obj = MyClass()
del obj.my_attr
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        self.assertGreater(result.transformation_count, 0)
        self.assertIn("my_attr", self.transformer.key_mappings)
        
        # Verify the transformed code contains delattr
        code_str = ast.unparse(result.ast_node)
        self.assertIn("delattr", code_str)
        self.assertIn("_key_map", code_str)
    
    def test_augmented_assignment(self):
        """Test transformation of augmented attribute assignment."""
        code = '''
class MyClass:
    def __init__(self):
        self.counter = 0

obj = MyClass()
obj.counter += 1
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        self.assertGreater(result.transformation_count, 0)
        self.assertIn("counter", self.transformer.key_mappings)
        
        # Verify the transformed code contains both getattr and setattr
        code_str = ast.unparse(result.ast_node)
        self.assertIn("setattr", code_str)
        self.assertIn("getattr", code_str)
        self.assertIn("_key_map", code_str)
    
    def test_builtin_attributes_preserved(self):
        """Test that built-in attributes are not transformed."""
        code = '''
class MyClass:
    pass

obj = MyClass()
name = obj.__class__.__name__
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        # Built-in attributes should not be in mappings
        self.assertNotIn("__class__", self.transformer.key_mappings)
        self.assertNotIn("__name__", self.transformer.key_mappings)
    
    def test_mixed_contexts(self):
        """Test that different contexts are handled correctly in the same code."""
        code = '''
class MyClass:
    pass

obj = MyClass()
obj.attr1 = 10
value = obj.attr1
obj.attr1 += 5
del obj.attr1
'''
        tree = ast.parse(code)
        result = self.transformer.transform(tree)
        
        self.assertTrue(result.success)
        self.assertIn("attr1", self.transformer.key_mappings)
        
        # Verify all three operations are present
        code_str = ast.unparse(result.ast_node)
        self.assertIn("setattr", code_str)
        self.assertIn("getattr", code_str)
        self.assertIn("delattr", code_str)


if __name__ == '__main__':
    unittest.main()

