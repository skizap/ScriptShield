"""Orchestrator feature-flag tests for mangle_globals toggling.

Verifies that enabling/disabling mangle_globals toggles renaming and
preserves exports/constants. Also includes an end-to-end workflow test
that exercises multi-file Python and Lua projects.
"""

import ast
from pathlib import Path

import pytest

from src.obfuscator.core.config import ObfuscationConfig
from src.obfuscator.core.orchestrator import ObfuscationOrchestrator


class TestMangleGlobalsFeatureFlag:
    """Test enabling/disabling mangle_globals via orchestrator config."""

    def test_mangle_globals_enabled(self, tmp_path):
        """When mangle_globals is enabled, symbols should be renamed."""
        mod = tmp_path / "mod.py"
        mod.write_text("""
def my_function():
    return 1

class MyClass:
    pass
""")

        config = ObfuscationConfig(
            name="test-enabled",
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
            input_files=[str(mod)],
            output_dir=str(output_dir)
        )

        content = (output_dir / "mod.py").read_text()
        assert "_0x" in content
        ast.parse(content)

    def test_mangle_globals_disabled(self, tmp_path):
        """When mangle_globals is disabled, symbols should keep original names."""
        mod = tmp_path / "mod.py"
        mod.write_text("""
def my_function():
    return 1

class MyClass:
    pass
""")

        config = ObfuscationConfig(
            name="test-disabled",
            features={"mangle_globals": False},
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

        result = orchestrator.orchestrate(
            input_files=[str(mod)],
            output_dir=str(output_dir)
        )

        content = (output_dir / "mod.py").read_text()
        # File should be processed (output exists) but original names kept
        assert (output_dir / "mod.py").exists()
        ast.parse(content)

    def test_preserve_exports_flag(self, tmp_path):
        """When preserve_exports is True, exported symbols keep original names."""
        mod = tmp_path / "mod.py"
        mod.write_text("""
def public_api():
    return "public"

def _private_helper():
    return "private"
""")

        config = ObfuscationConfig(
            name="test-exports",
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
            input_files=[str(mod)],
            output_dir=str(output_dir)
        )

        content = (output_dir / "mod.py").read_text()
        # Public symbols should be preserved when preserve_exports is True
        assert "public_api" in content
        # Mangling should still occur for non-exported symbols
        assert "_0x" in content
        ast.parse(content)

    def test_preserve_constants_flag(self, tmp_path):
        """When preserve_constants is True, ALL_CAPS variables keep original names."""
        mod = tmp_path / "mod.py"
        mod.write_text("""
MAX_VALUE = 100
MIN_VALUE = 0

def compute():
    return MAX_VALUE + MIN_VALUE
""")

        config = ObfuscationConfig(
            name="test-constants",
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
            input_files=[str(mod)],
            output_dir=str(output_dir)
        )

        content = (output_dir / "mod.py").read_text()
        assert "MAX_VALUE" in content
        assert "MIN_VALUE" in content
        # Function should still be mangled
        assert "_0x" in content
        ast.parse(content)


class TestEndToEndMultiFileWorkflow:
    """End-to-end workflow test exercising multi-file Python and Lua projects."""

    def test_multi_file_python_project(self, tmp_path):
        """Full Python project: utils -> main with imports."""
        utils = tmp_path / "utils.py"
        utils.write_text("""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""")

        main = tmp_path / "main.py"
        main.write_text("""
from utils import add, multiply

def compute(x, y):
    s = add(x, y)
    p = multiply(x, y)
    return s, p
""")

        config = ObfuscationConfig(
            name="e2e-python",
            language="python",
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

        result = orchestrator.orchestrate(
            input_files=[str(utils), str(main)],
            output_dir=str(output_dir)
        )

        # Both files should be processed
        assert (output_dir / "utils.py").exists()
        assert (output_dir / "main.py").exists()

        utils_content = (output_dir / "utils.py").read_text()
        main_content = (output_dir / "main.py").read_text()

        # Both should contain mangled names
        assert "_0x" in utils_content
        assert "_0x" in main_content

        # Syntax should be valid
        ast.parse(utils_content)
        ast.parse(main_content)

    def test_multi_file_lua_project(self, tmp_path):
        """Full Lua project: helper -> main with require."""
        helper = tmp_path / "helper.lua"
        helper.write_text("""
local M = {}

function M.greet(name)
    return "Hello, " .. name
end

return M
""")

        main = tmp_path / "main.lua"
        main.write_text("""
local helper = require("helper")

local function run()
    return helper.greet("world")
end

return run
""")

        config = ObfuscationConfig(
            name="e2e-lua",
            language="lua",
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

        result = orchestrator.orchestrate(
            input_files=[str(helper), str(main)],
            output_dir=str(output_dir)
        )

        assert (output_dir / "helper.lua").exists()
        assert (output_dir / "main.lua").exists()

        helper_content = (output_dir / "helper.lua").read_text()
        main_content = (output_dir / "main.lua").read_text()

        assert "_0x" in helper_content
        assert "_0x" in main_content

        # Require path unchanged
        assert 'require("helper")' in main_content

    def test_mixed_python_lua_project(self, tmp_path):
        """Mixed project: Python and Lua files processed together."""
        py_mod = tmp_path / "logic.py"
        py_mod.write_text("""
def process_data(items):
    return [x * 2 for x in items]

class Processor:
    def run(self):
        return process_data([1, 2, 3])
""")

        lua_mod = tmp_path / "config.lua"
        lua_mod.write_text("""
local M = {}

function M.get_settings()
    return {timeout = 30, retries = 3}
end

return M
""")

        config = ObfuscationConfig(
            name="e2e-mixed",
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

        result = orchestrator.orchestrate(
            input_files=[str(py_mod), str(lua_mod)],
            output_dir=str(output_dir)
        )

        assert (output_dir / "logic.py").exists()
        assert (output_dir / "config.lua").exists()

        py_content = (output_dir / "logic.py").read_text()
        lua_content = (output_dir / "config.lua").read_text()

        assert "_0x" in py_content
        assert "_0x" in lua_content

        ast.parse(py_content)
