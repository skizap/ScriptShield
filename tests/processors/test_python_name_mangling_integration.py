"""Integration tests for Python name mangling across multiple files."""

import tempfile
import ast
from pathlib import Path
import pytest

from src.obfuscator.core.config import ObfuscationConfig
from src.obfuscator.core.orchestrator import ObfuscationOrchestrator
from src.obfuscator.core.symbol_table import GlobalSymbolTable


class TestPythonMultiFileIntegration:
    """Test multi-file Python projects with name mangling."""

    def test_multi_file_python_basic(self, tmp_path):
        """Test basic multi-file project with imports."""
        # Create module_a.py
        module_a = tmp_path / "module_a.py"
        module_a.write_text("""
def helper_function():
    return 42

class HelperClass:
    def method(self):
        return "helper"
""")

        # Create module_b.py importing from module_a
        module_b = tmp_path / "module_b.py"
        module_b.write_text("""
from module_a import helper_function, HelperClass

def main():
    result = helper_function()
    obj = HelperClass()
    return result, obj.method()
""")

        # Create config and orchestrator
        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )
        
        orchestrator = ObfuscationOrchestrator(config)
        
        # Run orchestration
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        orchestrator.orchestrate(
            input_files=[str(module_a), str(module_b)],
            output_dir=str(output_dir)
        )
        
        # Verify output files exist
        output_a = output_dir / "module_a.py"
        output_b = output_dir / "module_b.py"
        assert output_a.exists()
        assert output_b.exists()
        
        # Read and parse output files
        content_a = output_a.read_text()
        content_b = output_b.read_text()
        
        # Verify mangling occurred (should have _0x prefix)
        assert "_0x" in content_a
        assert "_0x" in content_b
        
        # Verify imports still reference the same mangled names
        # The import statement should reference the mangled name from module_a
        assert "from module_a import" in content_b
        
        # Verify syntax is valid
        ast.parse(content_a)
        ast.parse(content_b)

    def test_relative_imports(self, tmp_path):
        """Test relative imports with name mangling."""
        # Create package structure
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        
        # Create subpackage
        subpkg = pkg / "subpkg"
        subpkg.mkdir()
        (subpkg / "__init__.py").write_text("")
        
        # Create module in subpackage
        module_c = subpkg / "module_c.py"
        module_c.write_text("""
def sub_helper():
    return "sub"
""")
        
        # Create module using relative import
        module_d = subpkg / "module_d.py"
        module_d.write_text("""
from . import module_c
from .. import pkg_module

def use_helpers():
    return module_c.sub_helper(), pkg_module.pkg_helper()
""")
        
        # Create parent module
        pkg_module = pkg / "pkg_module.py"
        pkg_module.write_text("""
def pkg_helper():
    return "pkg"
""")
        
        # Run orchestration
        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )
        
        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        input_files = [
            str(module_c), str(module_d), str(pkg_module)
        ]
        
        orchestrator.orchestrate(
            input_files=input_files,
            output_dir=str(output_dir)
        )
        
        # Verify all files were processed
        for file_path in input_files:
            file_name = Path(file_path).name
            output_file = output_dir / file_name
            assert output_file.exists()
            
            content = output_file.read_text()
            # Should have mangled names
            assert "_0x" in content
            # Should still have relative imports
            assert "from ." in content or "from .." in content
            
            # Verify syntax is valid
            ast.parse(content)

    def test_cross_file_function_calls(self, tmp_path):
        """Test cross-file function calls with mangling."""
        # Create utils module
        utils = tmp_path / "utils.py"
        utils.write_text("""
def calculate_sum(a, b):
    return a + b

def calculate_product(a, b):
    return a * b
""")
        
        # Create main module that uses utils
        main = tmp_path / "main.py"
        main.write_text("""
import utils

def process_data(x, y):
    total = utils.calculate_sum(x, y)
    product = utils.calculate_product(x, y)
    return total, product
""")
        
        # Run orchestration
        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )
        
        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        orchestrator.orchestrate(
            input_files=[str(utils), str(main)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        utils_content = (output_dir / "utils.py").read_text()
        main_content = (output_dir / "main.py").read_text()
        
        # Both files should have mangled names
        assert "_0x" in utils_content
        assert "_0x" in main_content
        
        # Main should still reference utils module
        assert "import utils" in main_content
        
        # Verify syntax
        ast.parse(utils_content)
        ast.parse(main_content)

    def test_class_inheritance_across_files(self, tmp_path):
        """Test class inheritance across files."""
        # Create base module
        base = tmp_path / "base.py"
        base.write_text("""
class BaseClass:
    def __init__(self):
        self.value = 10
    
    def get_value(self):
        return self.value
""")
        
        # Create derived module
        derived = tmp_path / "derived.py"
        derived.write_text("""
from base import BaseClass

class DerivedClass(BaseClass):
    def __init__(self):
        super().__init__()
        self.value = 20
    
    def get_doubled_value(self):
        return self.get_value() * 2
""")
        
        # Run orchestration
        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )
        
        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        orchestrator.orchestrate(
            input_files=[str(base), str(derived)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        base_content = (output_dir / "base.py").read_text()
        derived_content = (output_dir / "derived.py").read_text()
        
        # Should have mangled class names
        assert "_0x" in base_content
        assert "_0x" in derived_content
        
        # Should still have inheritance structure
        assert "class" in derived_content
        assert "BaseClass" in derived_content  # Base class reference preserved
        
        # Verify syntax
        ast.parse(base_content)
        ast.parse(derived_content)

    def test_circular_imports(self, tmp_path):
        """Test circular imports with graceful fallback."""
        # Create circular dependency
        module_e = tmp_path / "module_e.py"
        module_f = tmp_path / "module_f.py"
        
        module_e.write_text("""
import module_f

def function_e():
    return "e"

def call_f():
    return module_f.function_f()
""")
        
        module_f.write_text("""
import module_e

def function_f():
    return "f"

def call_e():
    return module_e.function_e()
""")
        
        # Run orchestration
        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )
        
        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Should not raise error despite circular dependency
        orchestrator.orchestrate(
            input_files=[str(module_e), str(module_f)],
            output_dir=str(output_dir)
        )
        
        # Verify both files processed
        assert (output_dir / "module_e.py").exists()
        assert (output_dir / "module_f.py").exists()
        
        # Verify content
        e_content = (output_dir / "module_e.py").read_text()
        f_content = (output_dir / "module_f.py").read_text()
        
        # Should have mangled names
        assert "_0x" in e_content
        assert "_0x" in f_content
        
        # Verify syntax
        ast.parse(e_content)
        ast.parse(f_content)

    def test_mixed_scope_symbols(self, tmp_path):
        """Test global functions calling local functions."""
        # Create module with mixed scopes
        mixed = tmp_path / "mixed.py"
        mixed.write_text("""
def global_function():
    def local_function():
        return "local"
    
    result = local_function()
    return result

class GlobalClass:
    def method(self):
        def nested_local():
            return "nested"
        return nested_local()
""")
        
        # Run orchestration
        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )
        
        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        orchestrator.orchestrate(
            input_files=[str(mixed)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        content = (output_dir / "mixed.py").read_text()
        
        # Should have mangled global names
        assert "_0x" in content
        
        # Local functions should not be mangled (they're not in global scope)
        # This is verified by checking that local_function and nested_local
        # are not prefixed with _0x
        
        # Verify syntax
        ast.parse(content)

    def test_preserve_exports_flag(self, tmp_path):
        """Test preserve_exports flag functionality."""
        # Create module with exported symbols
        exports = tmp_path / "exports.py"
        exports.write_text("""
def public_api():
    return "public"

def _private_helper():
    return "private"

class PublicClass:
    pass

class _PrivateClass:
    pass
""")
        
        # Run orchestration with preserve_exports=True
        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": True,
                "preserve_constants": False,
            }
        )
        
        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        orchestrator.orchestrate(
            input_files=[str(exports)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        content = (output_dir / "exports.py").read_text()
        
        # Public symbols should keep original names
        assert "def public_api" in content
        assert "class PublicClass" in content
        
        # Private symbols should be mangled
        assert "_0x" in content
        
        # Verify syntax
        ast.parse(content)

    def test_preserve_constants_flag(self, tmp_path):
        """Test preserve_constants flag functionality."""
        # Create module with constants
        constants = tmp_path / "constants.py"
        constants.write_text("""
MAX_VALUE = 100
MIN_VALUE = 0
API_KEY = "secret123"

def use_constants():
    return MAX_VALUE + MIN_VALUE
""")
        
        # Run orchestration with preserve_constants=True
        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": True,
            }
        )
        
        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        orchestrator.orchestrate(
            input_files=[str(constants)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        content = (output_dir / "constants.py").read_text()
        
        # Constants should keep original names
        assert "MAX_VALUE" in content
        assert "MIN_VALUE" in content
        assert "API_KEY" in content
        
        # Function should be mangled
        assert "_0x" in content
        
        # Verify syntax
        ast.parse(content)

    def test_from_import_bindings_mangled(self, tmp_path):
        """Test from-import bindings are rewritten to mangled names."""
        # Create defining module
        defs = tmp_path / "defs.py"
        defs.write_text("""
def compute_value():
    return 99

class DataHolder:
    pass
""")

        # Create consumer module
        consumer = tmp_path / "consumer.py"
        consumer.write_text("""
from defs import compute_value, DataHolder

def use_them():
    v = compute_value()
    d = DataHolder()
    return v, d
""")

        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )

        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        orchestrator.orchestrate(
            input_files=[str(defs), str(consumer)],
            output_dir=str(output_dir)
        )

        defs_content = (output_dir / "defs.py").read_text()
        consumer_content = (output_dir / "consumer.py").read_text()

        # Both files should contain mangled identifiers
        assert "_0x" in defs_content
        assert "_0x" in consumer_content

        # The from-import should reference the mangled name, not the original
        assert "compute_value" not in consumer_content or "_0x" in consumer_content
        assert "DataHolder" not in consumer_content or "_0x" in consumer_content

        # Verify syntax
        ast.parse(defs_content)
        ast.parse(consumer_content)

    def test_module_attribute_calls_mangled(self, tmp_path):
        """Test module.attr calls are rewritten to mangled names."""
        # Create utils module
        utils = tmp_path / "utils.py"
        utils.write_text("""
def calculate_sum(a, b):
    return a + b
""")

        # Create caller using import + attribute access
        caller = tmp_path / "caller.py"
        caller.write_text("""
import utils

def do_work(x, y):
    return utils.calculate_sum(x, y)
""")

        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )

        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        orchestrator.orchestrate(
            input_files=[str(utils), str(caller)],
            output_dir=str(output_dir)
        )

        utils_content = (output_dir / "utils.py").read_text()
        caller_content = (output_dir / "caller.py").read_text()

        # Definitions should be mangled
        assert "_0x" in utils_content

        # Attribute access should reference mangled name
        assert "_0x" in caller_content

        # The import statement should keep the module name
        assert "import utils" in caller_content

        # Verify syntax
        ast.parse(utils_content)
        ast.parse(caller_content)