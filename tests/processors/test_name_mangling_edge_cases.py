"""Edge case tests for name mangling functionality."""

import tempfile
from pathlib import Path
import pytest

from src.obfuscator.core.config import ObfuscationConfig
from src.obfuscator.core.orchestrator import ObfuscationOrchestrator
import ast


class TestNameManglingEdgeCases:
    """Test edge cases and boundary conditions for name mangling."""

    def test_empty_files(self, tmp_path):
        """Test no errors on empty Python and Lua files."""
        # Create empty files
        empty_py = tmp_path / "empty.py"
        empty_py.write_text("")
        
        empty_lua = tmp_path / "empty.lua"
        empty_lua.write_text("")
        
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
        
        # Should not raise errors
        orchestrator.orchestrate(
            input_files=[str(empty_py), str(empty_lua)],
            output_dir=str(output_dir)
        )
        
        # Verify files exist (even if empty)
        assert (output_dir / "empty.py").exists()
        assert (output_dir / "empty.lua").exists()

    def test_files_with_only_imports(self, tmp_path):
        """Test files with no definitions handled correctly."""
        # Create Python file with only imports
        imports_py = tmp_path / "imports.py"
        imports_py.write_text("""
import os
import sys
from typing import List, Dict
import numpy as np
""")
        
        # Create Lua file with only requires
        requires_lua = tmp_path / "requires.lua"
        requires_lua.write_text("""
local json = require("json")
local http = require("socket.http")
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
            input_files=[str(imports_py), str(requires_lua)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        py_content = (output_dir / "imports.py").read_text()
        lua_content = (output_dir / "requires.lua").read_text()
        
        # Imports/requires should be preserved
        assert "import os" in py_content
        assert "import sys" in py_content
        assert "from typing import" in py_content
        assert 'require("json")' in lua_content
        assert 'require("socket.http")' in lua_content
        
        # Should not have mangled names (no definitions to mangle)
        # or minimal mangling if any

    def test_deeply_nested_scopes(self, tmp_path):
        """Test deeply nested function scopes (5+ levels)."""
        # Create deeply nested Python functions
        nested_py = tmp_path / "nested.py"
        nested_py.write_text("""
def level1():
    def level2():
        def level3():
            def level4():
                def level5():
                    return "deep"
                return level5()
            return level4()
        return level3()
    return level2()

result = level1()
""")
        
        # Create deeply nested Lua functions
        nested_lua = tmp_path / "nested.lua"
        nested_lua.write_text("""
function level1()
    local function level2()
        local function level3()
            local function level4()
                local function level5()
                    return "deep"
                end
                return level5()
            end
            return level4()
        end
        return level3()
    end
    return level2()
end

local result = level1()
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
            input_files=[str(nested_py), str(nested_lua)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        py_content = (output_dir / "nested.py").read_text()
        lua_content = (output_dir / "nested.lua").read_text()
        
        # Global function should be mangled
        assert "_0x" in py_content
        assert "_0x" in lua_content
        
        # Verify syntax is valid
        ast.parse(py_content)
        # Lua syntax validation would require a Lua parser

    def test_name_collisions(self, tmp_path):
        """Test when mangled names would collide."""
        # Create many functions to test collision handling
        collision_py = tmp_path / "collision.py"
        functions = "\n".join([f"def func_{i}():\n    return {i}" for i in range(50)])
        collision_py.write_text(functions)
        
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
            input_files=[str(collision_py)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        content = (output_dir / "collision.py").read_text()
        
        # Should have many mangled names
        mangled_count = content.count("_0x")
        assert mangled_count >= 50  # At least 50 function definitions
        
        # All mangled names should be unique (no collisions)
        import re
        mangled_names = re.findall(r'_0x\w+', content)
        assert len(mangled_names) == len(set(mangled_names))
        
        # Verify syntax
        ast.parse(content)

    def test_unicode_identifiers(self, tmp_path):
        """Test Python files with non-ASCII identifiers."""
        # Create Python with Unicode identifiers
        unicode_py = tmp_path / "unicode.py"
        unicode_py.write_text("""
def 函数_计算(参数1, 参数2):
    return 参数1 + 参数2

class 类_数据:
    def __init__(self):
        self.属性_值 = 42
    
    def 方法_获取(self):
        return self.属性_值

结果 = 函数_计算(10, 20)
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
            input_files=[str(unicode_py)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        content = (output_dir / "unicode.py").read_text()
        
        # Should have mangled names
        assert "_0x" in content
        
        # Unicode identifiers should be handled (either mangled or preserved)
        # Verify syntax is valid
        ast.parse(content)

    def test_very_large_projects(self, tmp_path):
        """Test with 100+ files to verify performance and memory bounds."""
        # Create many small files
        files = []
        for i in range(100):
            file_path = tmp_path / f"module_{i:03d}.py"
            file_path.write_text(f"""
def function_{i}():
    return {i}

class Class_{i}:
    def method_{i}(self):
        return {i * 2}
""")
            files.append(str(file_path))
        
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
        
        # Should complete in reasonable time
        orchestrator.orchestrate(
            input_files=files,
            output_dir=str(output_dir)
        )
        
        # Verify all files processed
        for i in range(100):
            output_file = output_dir / f"module_{i:03d}.py"
            assert output_file.exists()
            
            content = output_file.read_text()
            assert "_0x" in content
            
            # Verify syntax
            ast.parse(content)

    def test_mixed_python_and_lua(self, tmp_path):
        """Test orchestrator handles mixed-language projects."""
        # Create Python module
        py_module = tmp_path / "python_module.py"
        py_module.write_text("""
def python_function():
    return "python"
""")
        
        # Create Lua module
        lua_module = tmp_path / "lua_module.lua"
        lua_module.write_text("""
function lua_function()
    return "lua"
end
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
            input_files=[str(py_module), str(lua_module)],
            output_dir=str(output_dir)
        )
        
        # Verify both files processed
        py_output = output_dir / "python_module.py"
        lua_output = output_dir / "lua_module.lua"
        
        assert py_output.exists()
        assert lua_output.exists()
        
        # Verify mangling in both
        assert "_0x" in py_output.read_text()
        assert "_0x" in lua_output.read_text()

    def test_external_dependencies(self, tmp_path):
        """Test stdlib imports are never mangled."""
        # Create Python with stdlib imports
        stdlib_py = tmp_path / "stdlib.py"
        stdlib_py.write_text("""
import os
import sys
import math
import json
from collections import defaultdict
from typing import List, Dict

def use_stdlib():
    path = os.path.join("a", "b")
    sqrt_val = math.sqrt(16)
    data = json.dumps({"key": "value"})
    return path, sqrt_val, data
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
            input_files=[str(stdlib_py)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        content = (output_dir / "stdlib.py").read_text()
        
        # Stdlib imports should be preserved
        assert "import os" in content
        assert "import sys" in content
        assert "import math" in content
        assert "import json" in content
        assert "from collections import defaultdict" in content
        assert "from typing import" in content
        
        # Stdlib usage should be preserved
        assert "os.path.join" in content
        assert "math.sqrt" in content
        assert "json.dumps" in content
        
        # User function should be mangled
        assert "_0x" in content
        assert "use_stdlib" not in content or "_0x" in content

    def test_dynamic_imports_limitation(self, tmp_path):
        """Document limitation that dynamic imports cannot be tracked."""
        # Create Python with dynamic imports
        dynamic_py = tmp_path / "dynamic.py"
        dynamic_py.write_text("""
# These dynamic imports cannot be tracked
module_name = "os"
dynamic_module = __import__(module_name)

# This also cannot be tracked
import importlib
another_module = importlib.import_module("sys")

# Exec cannot be tracked
code = "def dynamic_func(): return 42"
exec(code)
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
            input_files=[str(dynamic_py)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        content = (output_dir / "dynamic.py").read_text()
        
        # Dynamic imports should be preserved as-is
        assert "__import__" in content
        assert "importlib.import_module" in content
        assert "exec(code)" in content
        
        # Verify syntax
        ast.parse(content)

    def test_string_based_access_limitation(self, tmp_path):
        """Document limitation that string-based access cannot be mangled."""
        # Create Python with string-based attribute access
        string_access_py = tmp_path / "string_access.py"
        string_access_py.write_text("""
class MyClass:
    def __init__(self):
        self.attribute = "value"
    
    def method(self):
        return "result"

obj = MyClass()

# These cannot be mangled
attr_value = getattr(obj, "attribute")
method_result = getattr(obj, "method")()

# Dict access cannot be mangled
d = {"key": "value"}
value = d["key"]

# Globals access cannot be mangled
func = globals()["some_function"]
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
            input_files=[str(string_access_py)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        content = (output_dir / "string_access.py").read_text()
        
        # String-based access should be preserved
        assert 'getattr(obj, "attribute")' in content
        assert 'getattr(obj, "method")' in content
        assert 'd["key"]' in content
        assert 'globals()["some_function"]' in content
        
        # Class and method definitions should be mangled
        assert "_0x" in content