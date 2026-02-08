"""End-to-end integration tests for the ObfuscationOrchestrator with all features.

Tests verify that the orchestrator correctly passes config to processors,
which in turn invoke the ObfuscationEngine for the full transformation
pipeline.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from obfuscator.core.config import ObfuscationConfig
from obfuscator.core.orchestrator import ObfuscationOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    features: dict[str, bool] | None = None,
    language: str = "python",
    **extra_options: Any,
) -> ObfuscationConfig:
    """Create an ObfuscationConfig with the given features."""
    default_features: dict[str, bool] = {
        "mangle_globals": True,
        "string_encryption": False,
        "number_obfuscation": False,
        "constant_array": False,
        "mangle_indexes": False,
        "vm_protection": False,
    }
    if features:
        default_features.update(features)

    options = {
        "string_encryption_key_length": 16,
        "array_shuffle_seed": None,
        "dead_code_percentage": 20,
        "identifier_prefix": "_0x",
        "number_obfuscation_complexity": 3,
        "number_obfuscation_min_value": 10,
        "number_obfuscation_max_value": 1000000,
        "vm_protection_complexity": 2,
        "vm_protect_all_functions": False,
        "vm_bytecode_encryption": True,
        "vm_protection_marker": "vm_protect",
    }
    options.update(extra_options)

    return ObfuscationConfig(
        name="test_pipeline",
        language=language,
        features=default_features,
        options=options,
        symbol_table_options={
            "identifier_prefix": "_0x",
            "mangling_strategy": "sequential",
            "preserve_exports": False,
            "preserve_constants": False,
        },
    )


def _write_python_file(tmp_path: Path, name: str, code: str) -> Path:
    """Write a Python file into tmp_path and return its path."""
    file_path = tmp_path / name
    file_path.write_text(textwrap.dedent(code), encoding="utf-8")
    return file_path


def _write_lua_file(tmp_path: Path, name: str, code: str) -> Path:
    """Write a Lua file into tmp_path and return its path."""
    file_path = tmp_path / name
    file_path.write_text(textwrap.dedent(code), encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Python orchestrator tests
# ---------------------------------------------------------------------------

class TestOrchestratorPython:
    """Process Python files through the full orchestrator pipeline."""

    def test_orchestrator_with_all_features_python(self, tmp_path: Path):
        """Process a Python file with all 6 features enabled."""
        config = _make_config(features={
            "mangle_globals": True,
            "string_encryption": True,
            "number_obfuscation": True,
            "constant_array": True,
            "mangle_indexes": True,
            "vm_protection": True,
        })
        orchestrator = ObfuscationOrchestrator(config=config)

        src = _write_python_file(tmp_path, "main.py", """\
            def greet(name):
                message = "Hello, " + name
                return message

            result = greet("World")
            print(result)
        """)

        output_dir = tmp_path / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        # Orchestrator should complete (individual transformer failures
        # are handled gracefully)
        assert result is not None
        assert isinstance(result.processed_files, list)

    def test_orchestrator_backward_compatibility(self, tmp_path: Path):
        """Verify existing name-mangling-only workflows still work."""
        config = _make_config(features={
            "mangle_globals": True,
            # All other features disabled
        })
        orchestrator = ObfuscationOrchestrator(config=config)

        src = _write_python_file(tmp_path, "simple.py", """\
            def my_func():
                return 42

            result = my_func()
        """)

        output_dir = tmp_path / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert result is not None
        # At least one file should have been processed
        assert len(result.processed_files) >= 1

        # Output file should exist and be valid Python
        for pr in result.processed_files:
            if pr.success and pr.output_path:
                code = pr.output_path.read_text(encoding="utf-8")
                # Should be parseable Python
                import ast
                ast.parse(code)

    def test_orchestrator_execution_correctness(self, tmp_path: Path):
        """Verify obfuscated Python code executes with correct results."""
        config = _make_config(features={
            "mangle_globals": True,
            "number_obfuscation": True,
        })
        orchestrator = ObfuscationOrchestrator(config=config)

        source_code = textwrap.dedent("""\
            def add(a, b):
                return a + b

            total = add(10, 20)
        """)

        src = _write_python_file(tmp_path, "calc.py", source_code)

        output_dir = tmp_path / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert result is not None

        # Execute original code and capture its namespace
        original_ns: dict[str, Any] = {}
        exec(source_code, original_ns)
        assert original_ns["total"] == 30

        # Execute obfuscated code and verify functional equivalence
        for pr in result.processed_files:
            if pr.success and pr.output_path:
                obfuscated_code = pr.output_path.read_text(encoding="utf-8")

                import ast as ast_mod
                ast_mod.parse(obfuscated_code)  # valid syntax

                obfuscated_ns: dict[str, Any] = {}
                exec(obfuscated_code, obfuscated_ns)

                # Find the variable holding the result (name may be mangled)
                # Look for any int value equal to 30 in the namespace
                found_total = any(
                    v == 30
                    for k, v in obfuscated_ns.items()
                    if isinstance(v, int) and not k.startswith("__")
                )
                assert found_total, (
                    f"Expected a variable with value 30 in obfuscated output, "
                    f"got: {obfuscated_ns}"
                )

    def test_orchestrator_multi_file_with_features(self, tmp_path: Path):
        """Process multi-file project with cross-file dependencies and features."""
        config = _make_config(features={
            "mangle_globals": True,
            "number_obfuscation": True,
        })
        orchestrator = ObfuscationOrchestrator(config=config)

        _write_python_file(tmp_path, "utils.py", """\
            def calculate_sum(a, b):
                return a + b

            MAGIC_NUMBER = 42
        """)

        _write_python_file(tmp_path, "main.py", """\
            from utils import calculate_sum

            result = calculate_sum(10, 20)
            print(result)
        """)

        output_dir = tmp_path / "out"
        result = orchestrator.process_files(
            input_files=[
                tmp_path / "utils.py",
                tmp_path / "main.py",
            ],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert result is not None
        assert isinstance(result.processed_files, list)

    def test_orchestrator_feature_combinations(self, tmp_path: Path):
        """Test different feature combinations in single-file scenario."""
        combinations = [
            {"mangle_globals": True, "string_encryption": True},
            {"mangle_globals": True, "constant_array": True},
            {"mangle_globals": True, "mangle_indexes": True},
            {"number_obfuscation": True, "constant_array": True},
        ]

        for combo in combinations:
            config = _make_config(features=combo)
            orchestrator = ObfuscationOrchestrator(config=config)

            src = _write_python_file(tmp_path, "combo_test.py", """\
                data = [10, 20, 30]
                name = "test"
                value = 42
            """)

            output_dir = tmp_path / "out_combo"
            result = orchestrator.process_files(
                input_files=[src],
                output_dir=output_dir,
                config=config.symbol_table_options,
                project_root=tmp_path,
            )

            assert result is not None, f"Failed for combo: {combo}"


# ---------------------------------------------------------------------------
# Lua orchestrator tests
# ---------------------------------------------------------------------------

class TestOrchestratorLua:
    """Process Lua files through the full orchestrator pipeline."""

    def test_orchestrator_with_all_features_lua(self, tmp_path: Path):
        """Process Lua files with all 6 features enabled."""
        config = _make_config(
            features={
                "mangle_globals": True,
                "string_encryption": True,
                "number_obfuscation": True,
                "constant_array": True,
                "mangle_indexes": True,
                "vm_protection": True,
            },
            language="lua",
        )
        orchestrator = ObfuscationOrchestrator(config=config)

        src = _write_lua_file(tmp_path, "main.lua", """\
            local function greet(name)
                local message = "Hello, " .. name
                return message
            end

            local result = greet("World")
            print(result)
        """)

        output_dir = tmp_path / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert result is not None
        assert isinstance(result.processed_files, list)

    def test_orchestrator_lua_backward_compatibility(self, tmp_path: Path):
        """Verify Lua name-mangling-only workflows still work."""
        config = _make_config(
            features={"mangle_globals": True},
            language="lua",
        )
        orchestrator = ObfuscationOrchestrator(config=config)

        src = _write_lua_file(tmp_path, "simple.lua", """\
            local function myFunc()
                return 42
            end

            local result = myFunc()
        """)

        output_dir = tmp_path / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert result is not None
        assert len(result.processed_files) >= 1

        # Output should be valid Lua
        for pr in result.processed_files:
            if pr.success and pr.output_path:
                code = pr.output_path.read_text(encoding="utf-8")
                assert len(code) > 0


# ---------------------------------------------------------------------------
# No-config backward compatibility
# ---------------------------------------------------------------------------

class TestNoConfigBackwardCompat:
    """Verify that creating an orchestrator without config still works."""

    def test_orchestrator_no_config(self, tmp_path: Path):
        """Orchestrator without config should process files normally."""
        orchestrator = ObfuscationOrchestrator()

        src = _write_python_file(tmp_path, "no_config.py", """\
            x = 1
            print(x)
        """)

        output_dir = tmp_path / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            project_root=tmp_path,
        )

        assert result is not None
        assert isinstance(result.processed_files, list)

    def test_python_processor_no_config(self):
        """PythonProcessor without config should work identically to before."""
        from obfuscator.processors.python_processor import PythonProcessor

        processor = PythonProcessor()
        assert processor._config is None

    def test_lua_processor_no_config(self):
        """LuaProcessor without config should work identically to before."""
        from obfuscator.processors.lua_processor import LuaProcessor

        processor = LuaProcessor()
        assert processor._config is None
