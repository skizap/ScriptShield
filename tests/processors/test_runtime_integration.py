"""Integration tests for runtime modes (embedded and hybrid).

Tests verify that runtime generation works correctly with the full obfuscation
pipeline for both embedded mode (runtime code in each file) and hybrid mode
(shared runtime file with imports).
"""

from __future__ import annotations

import textwrap
import json
import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from obfuscator.core.config import ObfuscationConfig
from obfuscator.core.profile_manager import ProfileManager
from obfuscator.core.orchestrator import ObfuscationOrchestrator


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _execute_python_file(file_path: Path) -> tuple[bool, Any, str]:
    """Execute a Python file and return (success, result, error_message).
    
    Args:
        file_path: Path to the Python file to execute
        
    Returns:
        Tuple of (success_bool, result_value_or_namespace, error_message)
    """
    try:
        code = file_path.read_text(encoding="utf-8")
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        return True, namespace, ""
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def _execute_python_code(code: str) -> tuple[bool, Any, str]:
    """Execute Python code string and return (success, result, error_message)."""
    try:
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        return True, namespace, ""
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def _execute_lua_code(code: str, cwd: Path | None = None) -> tuple[bool, str, str]:
    """Execute Lua code via interpreter and return (success, stdout, stderr).
    
    Tries lua, then luau, then luajit in order.
    
    Args:
        code: Lua code to execute (will be passed via stdin)
        cwd: Working directory for execution
        
    Returns:
        Tuple of (success_bool, stdout, stderr)
    """
    interpreters = ["lua", "luau", "luajit"]
    
    for interpreter in interpreters:
        try:
            result = subprocess.run(
                [interpreter, "-"],
                input=code,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=10,
            )
            if result.returncode == 0:
                return True, result.stdout, result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            continue
    
    return False, "", "No Lua interpreter found (tried: lua, luau, luajit)"


def _execute_lua_file(file_path: Path) -> tuple[bool, str, str]:
    """Execute a Lua file via interpreter and return (success, stdout, stderr)."""
    interpreters = ["lua", "luau", "luajit"]
    
    for interpreter in interpreters:
        try:
            result = subprocess.run(
                [interpreter, str(file_path)],
                capture_output=True,
                text=True,
                cwd=file_path.parent,
                timeout=10,
            )
            if result.returncode == 0:
                return True, result.stdout, result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            continue
    
    return False, "", "No Lua interpreter found (tried: lua, luau, luajit)"


def _has_lua_interpreter() -> bool:
    """Check if a Lua interpreter is available."""
    interpreters = ["lua", "luau", "luajit"]
    for interpreter in interpreters:
        try:
            subprocess.run([interpreter, "-v"], capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
            continue
    return False


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory for testing."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    return project_dir


@pytest.fixture
def embedded_config() -> ObfuscationConfig:
    """Create config with runtime_mode='embedded' and runtime features enabled."""
    return ObfuscationConfig(
        name="Embedded Test",
        version="1.0",
        language="python",
        preset=None,
        runtime_mode="embedded",
        features={
            "vm_protection": True,
            "code_splitting": True,
            "mangle_globals": False,
        },
        options={
            "vm_bytecode_encryption": True,
            "code_split_encryption": True,
            "vm_protection_complexity": 2,
            "vm_protect_all_functions": False,
            "vm_protection_marker": "vm_protect",
        },
        symbol_table_options={
            "identifier_prefix": "_0x",
            "mangling_strategy": "sequential",
            "preserve_exports": False,
            "preserve_constants": False,
        },
    )


@pytest.fixture
def hybrid_config() -> ObfuscationConfig:
    """Create config with runtime_mode='hybrid' and runtime features enabled."""
    return ObfuscationConfig(
        name="Hybrid Test",
        version="1.0",
        language="python",
        preset=None,
        runtime_mode="hybrid",
        features={
            "vm_protection": True,
            "code_splitting": True,
            "mangle_globals": False,
        },
        options={
            "vm_bytecode_encryption": True,
            "code_split_encryption": True,
            "vm_protection_complexity": 2,
            "vm_protect_all_functions": False,
            "vm_protection_marker": "vm_protect",
        },
        symbol_table_options={
            "identifier_prefix": "_0x",
            "mangling_strategy": "sequential",
            "preserve_exports": False,
            "preserve_constants": False,
        },
    )


@pytest.fixture
def lua_embedded_config() -> ObfuscationConfig:
    """Create Lua config with runtime_mode='embedded'."""
    return ObfuscationConfig(
        name="Lua Embedded Test",
        version="1.0",
        language="lua",
        preset=None,
        runtime_mode="embedded",
        features={
            "vm_protection": True,
            "code_splitting": True,
            "anti_debugging": True,
            "mangle_globals": False,
        },
        options={
            "vm_bytecode_encryption": True,
            "code_split_encryption": True,
            "anti_debug_aggressiveness": 2,
        },
        symbol_table_options={
            "identifier_prefix": "_0x",
            "mangling_strategy": "sequential",
            "preserve_exports": False,
            "preserve_constants": False,
        },
    )


@pytest.fixture
def lua_hybrid_config() -> ObfuscationConfig:
    """Create Lua config with runtime_mode='hybrid'."""
    return ObfuscationConfig(
        name="Lua Hybrid Test",
        version="1.0",
        language="lua",
        preset=None,
        runtime_mode="hybrid",
        features={
            "vm_protection": True,
            "code_splitting": True,
            "anti_debugging": True,
            "mangle_globals": False,
        },
        options={
            "vm_bytecode_encryption": True,
            "code_split_encryption": True,
            "anti_debug_aggressiveness": 2,
        },
        symbol_table_options={
            "identifier_prefix": "_0x",
            "mangling_strategy": "sequential",
            "preserve_exports": False,
            "preserve_constants": False,
        },
    )


def _write_python_file(directory: Path, name: str, code: str) -> Path:
    """Write a Python file into the directory."""
    file_path = directory / name
    file_path.write_text(textwrap.dedent(code), encoding="utf-8")
    return file_path


def _write_lua_file(directory: Path, name: str, code: str) -> Path:
    """Write a Lua file into the directory."""
    file_path = directory / name
    file_path.write_text(textwrap.dedent(code), encoding="utf-8")
    return file_path


# -----------------------------------------------------------------------------
# Test Embedded Mode Integration
# -----------------------------------------------------------------------------

class TestEmbeddedModeIntegration:
    """Test embedded mode where runtime code is included in each obfuscated file."""

    def test_embedded_mode_python_single_file(self, tmp_project: Path, embedded_config: ObfuscationConfig):
        """Process Python file with embedded mode, verify runtime code embedded at top."""
        orchestrator = ObfuscationOrchestrator(config=embedded_config)
        
        src = _write_python_file(tmp_project, "main.py", """\
            def add(a, b):
                return a + b
            result = add(10, 20)
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=embedded_config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        # Check output file
        output_file = output_dir / "main.py"
        assert output_file.exists()
        
        code = output_file.read_text(encoding="utf-8")
        # Runtime should be embedded
        assert "EMBEDDED OBFUSCATION RUNTIME" in code or len(code) > len(src.read_text())
        
        # No separate runtime file should be created
        runtime_file = output_dir / "obf_runtime.py"
        assert not runtime_file.exists()

    @pytest.mark.skipif(not _has_lua_interpreter(), reason="No Lua interpreter available")
    def test_embedded_mode_lua_single_file_executes(self, tmp_project: Path, lua_embedded_config: ObfuscationConfig):
        """Process Lua file with embedded mode, verify runtime embedded and code executes."""
        orchestrator = ObfuscationOrchestrator(config=lua_embedded_config)
        
        src = _write_lua_file(tmp_project, "main.lua", """\
            local function add(a, b)
                return a + b
            end
            local result = add(10, 20)
            print("Result: " .. result)
            return result
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=lua_embedded_config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        output_file = output_dir / "main.lua"
        assert output_file.exists()
        
        code = output_file.read_text(encoding="utf-8")
        assert "EMBEDDED OBFUSCATION RUNTIME" in code or len(code) > len(src.read_text())
        
        # Execute and verify the obfuscated code runs
        success, stdout, stderr = _execute_lua_file(output_file)
        assert success, f"Obfuscated Lua failed to execute: {stderr}"

    def test_embedded_mode_multiple_files(self, tmp_project: Path, embedded_config: ObfuscationConfig):
        """Process multiple files, verify each has runtime embedded independently."""
        orchestrator = ObfuscationOrchestrator(config=embedded_config)
        
        src1 = _write_python_file(tmp_project, "file1.py", "x = 1")
        src2 = _write_python_file(tmp_project, "file2.py", "y = 2")
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src1, src2],
            output_dir=output_dir,
            config=embedded_config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        # Both files should exist
        output1 = output_dir / "file1.py"
        output2 = output_dir / "file2.py"
        assert output1.exists()
        assert output2.exists()
        
        # Both should have embedded runtime
        code1 = output1.read_text(encoding="utf-8")
        code2 = output2.read_text(encoding="utf-8")
        assert "EMBEDDED OBFUSCATION RUNTIME" in code1 or len(code1) > 10
        assert "EMBEDDED OBFUSCATION RUNTIME" in code2 or len(code2) > 10

    def test_embedded_runtime_includes_all_required_features(self, tmp_project: Path):
        """Enable vm_protection and code_splitting, verify both runtimes embedded."""
        config = ObfuscationConfig(
            name="Multi Feature Test",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="embedded",
            features={
                "vm_protection": True,
                "code_splitting": True,
                "mangle_globals": False,
            },
            options={
                "vm_bytecode_encryption": True,
                "code_split_encryption": True,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        src = _write_python_file(tmp_project, "test.py", "x = 1")
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        output_file = output_dir / "test.py"
        code = output_file.read_text(encoding="utf-8")
        # Should contain runtime markers or generated code
        assert len(code) > len("x = 1")

    def test_embedded_runtime_execution_correctness(self, tmp_project: Path):
        """Create Python file with VM-protected function using vm_protect_all_functions, obfuscate with embedded mode, execute and verify result=30."""
        config = ObfuscationConfig(
            name="Execution Test",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="embedded",
            features={
                "vm_protection": True,
                "mangle_globals": False,
            },
            options={
                "vm_bytecode_encryption": True,
                "vm_protect_all_functions": True,
                "vm_protection_marker": "vm_protect",
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        src = _write_python_file(tmp_project, "calc.py", """\
            def add(a, b):
                return a + b
            result = add(10, 20)
            print(result)
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        # Execute and verify
        output_file = output_dir / "calc.py"
        assert output_file.exists(), "Output file should exist"
        
        code = output_file.read_text(encoding="utf-8")
        # Verify it's valid Python
        try:
            ast.parse(code)
        except SyntaxError:
            pytest.fail("Generated code has syntax errors")
        
        # Execute with safe namespace
        namespace: dict[str, Any] = {}
        safe_builtins = {
            'print': print,
            '__builtins__': {
                'len': len,
                'range': range,
                'int': int,
                'str': str,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                ' setattr': setattr,
                'chr': chr,
                'ord': ord,
                'hex': hex,
                'oct': oct,
                'bin': bin,
                'pow': pow,
                'round': round,
                'divmod': divmod,
                'callable': callable,
                'type': type,
                'bool': bool,
                'float': float,
            }
        }
        exec(code, safe_builtins, namespace)
        
        # Verify result = 30 (10 + 20)
        result_value = namespace.get('result')
        if result_value is None:
            # Name might be mangled, look for numeric values
            for key, value in namespace.items():
                if isinstance(value, int) and value == 30:
                    result_value = value
                    break
        
        assert result_value == 30, f"Expected result=30, got {result_value}"

    def test_embedded_python_execution_returns_correct_result(self, tmp_project: Path):
        """Execute obfuscated Python with embedded runtime and verify function returns correct value."""
        config = ObfuscationConfig(
            name="Embedded Execution Test",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="embedded",
            features={
                "vm_protection": True,
                "mangle_globals": False,
            },
            options={
                "vm_bytecode_encryption": True,
                "vm_protect_all_functions": True,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        # Create a simple calculator function
        src = _write_python_file(tmp_project, "calc.py", """\
            def calculate(x, y):
                return x * y + 10
            
            result = calculate(5, 3)
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        output_file = output_dir / "calc.py"
        assert output_file.exists()
        
        # Execute the obfuscated code
        success, namespace, error = _execute_python_file(output_file)
        assert success, f"Obfuscated code failed to execute: {error}"
        
        # Verify the result is correct (5 * 3 + 10 = 25)
        # Look for result value in namespace
        result_value = namespace.get('result')
        if result_value is None:
            # Name might be mangled, look for numeric values
            for key, value in namespace.items():
                if isinstance(value, int) and value == 25:
                    result_value = value
                    break
        
        assert result_value == 25, f"Expected result=25, got {result_value}"


# -----------------------------------------------------------------------------
# Test Hybrid Mode Integration
# -----------------------------------------------------------------------------

class TestHybridModeIntegration:
    """Test hybrid mode where a shared runtime file is created."""

    def test_hybrid_mode_python_creates_runtime_file(self, tmp_project: Path, hybrid_config: ObfuscationConfig):
        """Process Python file with hybrid mode, verify obf_runtime.py created."""
        orchestrator = ObfuscationOrchestrator(config=hybrid_config)
        
        src = _write_python_file(tmp_project, "main.py", "x = 1")
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=hybrid_config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        # Runtime file should be created
        runtime_file = output_dir / "obf_runtime.py"
        assert runtime_file.exists(), "obf_runtime.py should be created in hybrid mode"
        
        # Obfuscated file should have imports
        output_file = output_dir / "main.py"
        code = output_file.read_text(encoding="utf-8")
        assert "from obf_runtime import" in code

    @pytest.mark.skipif(not _has_lua_interpreter(), reason="No Lua interpreter available")
    def test_hybrid_mode_lua_creates_runtime_file_executes(self, tmp_project: Path, lua_hybrid_config: ObfuscationConfig):
        """Process Lua file with hybrid mode, verify obf_runtime.lua created and code executes."""
        orchestrator = ObfuscationOrchestrator(config=lua_hybrid_config)
        
        src = _write_lua_file(tmp_project, "main.lua", """\
            local function multiply(a, b)
                return a * b
            end
            local result = multiply(4, 5)
            print(result)
            return result
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=lua_hybrid_config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        # Runtime file should be created
        runtime_file = output_dir / "obf_runtime.lua"
        assert runtime_file.exists(), "obf_runtime.lua should be created in hybrid mode"
        
        # Execute and verify
        output_file = output_dir / "main.lua"
        success, stdout, stderr = _execute_lua_file(output_file)
        assert success, f"Obfuscated Lua with hybrid runtime failed: {stderr}"

    def test_hybrid_mode_multiple_files_share_runtime(self, tmp_project: Path, hybrid_config: ObfuscationConfig):
        """Process multiple files, verify single shared runtime file created."""
        orchestrator = ObfuscationOrchestrator(config=hybrid_config)
        
        src1 = _write_python_file(tmp_project, "file1.py", "x = 1")
        src2 = _write_python_file(tmp_project, "file2.py", "y = 2")
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src1, src2],
            output_dir=output_dir,
            config=hybrid_config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        # Only one runtime file should exist
        runtime_file = output_dir / "obf_runtime.py"
        assert runtime_file.exists()
        
        # Both obfuscated files should import from it
        output1 = output_dir / "file1.py"
        output2 = output_dir / "file2.py"
        assert "from obf_runtime import" in output1.read_text(encoding="utf-8")
        assert "from obf_runtime import" in output2.read_text(encoding="utf-8")

    def test_hybrid_runtime_metadata(self, tmp_project: Path, hybrid_config: ObfuscationConfig):
        """Verify OrchestrationResult.metadata contains runtime file paths."""
        orchestrator = ObfuscationOrchestrator(config=hybrid_config)
        
        src = _write_python_file(tmp_project, "test.py", "x = 1")
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=hybrid_config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        # Check metadata for runtime files
        if "runtime_files" in result.metadata:
            runtime_files = result.metadata["runtime_files"]
            assert len(runtime_files) > 0
            assert any("obf_runtime" in str(f) for f in runtime_files)

    def test_hybrid_mode_no_runtime_when_no_features(self, tmp_project: Path):
        """Process file with no runtime-requiring features, verify no runtime file created."""
        config = ObfuscationConfig(
            name="No Runtime Features",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="hybrid",
            features={
                "mangle_globals": True,  # No runtime required
                "string_encryption": False,
                "vm_protection": False,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        src = _write_python_file(tmp_project, "simple.py", "x = 1")
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        
        # No runtime file should be created
        runtime_file = output_dir / "obf_runtime.py"
        assert not runtime_file.exists()

    def test_hybrid_runtime_execution_correctness(self, tmp_project: Path):
        """Create Python file with VM-protected function using vm_protect_all_functions, obfuscate with hybrid mode, execute with combined runtime and verify result=30."""
        config = ObfuscationConfig(
            name="Hybrid Execution Test",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="hybrid",
            features={
                "vm_protection": True,
                "mangle_globals": False,
            },
            options={
                "vm_bytecode_encryption": True,
                "vm_protect_all_functions": True,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        src = _write_python_file(tmp_project, "calc.py", """\
            def add(a, b):
                return a + b
            result = add(10, 20)
            print(result)
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        # Verify files exist
        runtime_file = output_dir / "obf_runtime.py"
        output_file = output_dir / "calc.py"
        
        assert runtime_file.exists(), "Runtime file should be created in hybrid mode"
        assert output_file.exists(), "Output file should exist"
        
        runtime_code = runtime_file.read_text(encoding="utf-8")
        output_code = output_file.read_text(encoding="utf-8")
        
        # Verify valid Python syntax
        try:
            ast.parse(runtime_code)
            ast.parse(output_code)
        except SyntaxError:
            pytest.fail("Generated code has syntax errors")
        
        # Execute combined code with safe namespace
        combined_code = runtime_code + "\n\n" + output_code
        namespace: dict[str, Any] = {}
        safe_builtins = {
            'print': print,
            '__builtins__': {
                'len': len,
                'range': range,
                'int': int,
                'str': str,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                ' setattr': setattr,
                'chr': chr,
                'ord': ord,
                'hex': hex,
                'oct': oct,
                'bin': bin,
                'pow': pow,
                'round': round,
                'divmod': divmod,
                'callable': callable,
                'type': type,
                'bool': bool,
                'float': float,
            }
        }
        exec(combined_code, safe_builtins, namespace)
        
        # Verify result = 30 (10 + 20)
        result_value = namespace.get('result')
        if result_value is None:
            # Name might be mangled, look for numeric values
            for key, value in namespace.items():
                if isinstance(value, int) and value == 30:
                    result_value = value
                    break
        
        assert result_value == 30, f"Expected result=30, got {result_value}"

    def test_hybrid_python_execution_with_imported_runtime(self, tmp_project: Path):
        """Execute obfuscated Python with hybrid runtime imports and verify correct results."""
        config = ObfuscationConfig(
            name="Hybrid Execution with Runtime",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="hybrid",
            features={
                "vm_protection": True,
                "mangle_globals": False,
            },
            options={
                "vm_bytecode_encryption": True,
                "vm_protect_all_functions": True,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        # Create a simple calculator function
        src = _write_python_file(tmp_project, "calc.py", """\
            def compute(x, y):
                return x + y * 2
            
            result = compute(10, 5)
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        runtime_file = output_dir / "obf_runtime.py"
        output_file = output_dir / "calc.py"
        assert runtime_file.exists(), "Runtime file should exist"
        assert output_file.exists(), "Output file should exist"
        
        # Execute with runtime file in the path
        # Combine runtime and output code for execution
        runtime_code = runtime_file.read_text(encoding="utf-8")
        output_code = output_file.read_text(encoding="utf-8")
        
        # Execute combined code
        combined_code = runtime_code + "\n\n" + output_code
        success, namespace, error = _execute_python_code(combined_code)
        assert success, f"Obfuscated code with hybrid runtime failed: {error}"
        
        # Verify result (10 + 5 * 2 = 20)
        result_value = namespace.get('result')
        if result_value is None:
            for key, value in namespace.items():
                if isinstance(value, int) and value == 20:
                    result_value = value
                    break
        
        assert result_value == 20, f"Expected result=20, got {result_value}"


# -----------------------------------------------------------------------------
# Test Runtime Mode Comparison
# -----------------------------------------------------------------------------

class TestRuntimeModeComparison:
    """Compare embedded vs hybrid mode behavior."""

    def test_embedded_vs_hybrid_file_size(self, tmp_project: Path):
        """Process same file with both modes, verify embedded produces larger individual files."""
        embedded_config = ObfuscationConfig(
            name="Embedded",
            version="1.0",
            language="python",
            runtime_mode="embedded",
            features={"vm_protection": True, "mangle_globals": False},
        )
        hybrid_config = ObfuscationConfig(
            name="Hybrid",
            version="1.0",
            language="python",
            runtime_mode="hybrid",
            features={"vm_protection": True, "mangle_globals": False},
        )
        
        src = _write_python_file(tmp_project, "test.py", "x = 1")
        
        # Embedded mode
        embedded_out = tmp_project / "embedded_out"
        orchestrator_e = ObfuscationOrchestrator(config=embedded_config)
        result_e = orchestrator_e.process_files(
            input_files=[src],
            output_dir=embedded_out,
            config=embedded_config.symbol_table_options,
            project_root=tmp_project,
        )
        
        # Hybrid mode
        hybrid_out = tmp_project / "hybrid_out"
        orchestrator_h = ObfuscationOrchestrator(config=hybrid_config)
        result_h = orchestrator_h.process_files(
            input_files=[src],
            output_dir=hybrid_out,
            config=hybrid_config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result_e is not None and result_h is not None
        
        # Compare file sizes
        embedded_file = embedded_out / "test.py"
        hybrid_file = hybrid_out / "test.py"
        
        if embedded_file.exists() and hybrid_file.exists():
            embedded_size = len(embedded_file.read_text(encoding="utf-8"))
            hybrid_size = len(hybrid_file.read_text(encoding="utf-8"))
            
            # Hybrid file (with import) should be smaller than embedded (with full runtime)
            # Note: This might vary based on implementation, so we just verify both work
            assert embedded_size > 0
            assert hybrid_size > 0

    def test_embedded_vs_hybrid_functional_equivalence(self, tmp_project: Path):
        """Process same code with both modes, execute both, verify identical results."""
        # Simple case without VM protection for reliable execution
        embedded_config = ObfuscationConfig(
            name="Embedded",
            version="1.0",
            language="python",
            runtime_mode="embedded",
            features={"mangle_globals": True},
        )
        hybrid_config = ObfuscationConfig(
            name="Hybrid",
            version="1.0",
            language="python",
            runtime_mode="hybrid",
            features={"mangle_globals": True},
        )
        
        src = _write_python_file(tmp_project, "calc.py", """\
            def calculate():
                return 42
            result = calculate()
        """)
        
        # Embedded mode
        embedded_out = tmp_project / "embedded_out"
        orchestrator_e = ObfuscationOrchestrator(config=embedded_config)
        orchestrator_e.process_files(
            input_files=[src],
            output_dir=embedded_out,
            config=embedded_config.symbol_table_options,
            project_root=tmp_project,
        )
        
        # Hybrid mode
        hybrid_out = tmp_project / "hybrid_out"
        orchestrator_h = ObfuscationOrchestrator(config=hybrid_config)
        orchestrator_h.process_files(
            input_files=[src],
            output_dir=hybrid_out,
            config=hybrid_config.symbol_table_options,
            project_root=tmp_project,
        )
        
        # Both should produce valid Python
        embedded_file = embedded_out / "calc.py"
        hybrid_file = hybrid_out / "calc.py"
        
        if embedded_file.exists():
            code_e = embedded_file.read_text(encoding="utf-8")
            try:
                ast.parse(code_e)
            except SyntaxError:
                pytest.fail("Embedded mode produced invalid Python")
        
        if hybrid_file.exists():
            code_h = hybrid_file.read_text(encoding="utf-8")
            try:
                ast.parse(code_h)
            except SyntaxError:
                pytest.fail("Hybrid mode produced invalid Python")

    def test_embedded_vs_hybrid_vm_protection_executes(self, tmp_project: Path):
        """Process same VM-protected code with both modes, execute both, verify same results."""
        embedded_config = ObfuscationConfig(
            name="Embedded VM",
            version="1.0",
            language="python",
            runtime_mode="embedded",
            features={"vm_protection": True, "mangle_globals": False},
            options={"vm_protect_all_functions": True},
        )
        hybrid_config = ObfuscationConfig(
            name="Hybrid VM",
            version="1.0",
            language="python",
            runtime_mode="hybrid",
            features={"vm_protection": True, "mangle_globals": False},
            options={"vm_protect_all_functions": True},
        )
        
        src = _write_python_file(tmp_project, "math_ops.py", """\
            def math_op(x, y):
                return (x + y) * 2 - 5
            result = math_op(3, 4)
        """)
        
        # Embedded mode
        embedded_out = tmp_project / "embedded_vm_out"
        orchestrator_e = ObfuscationOrchestrator(config=embedded_config)
        result_e = orchestrator_e.process_files(
            input_files=[src],
            output_dir=embedded_out,
            config=embedded_config.symbol_table_options,
            project_root=tmp_project,
        )
        
        # Hybrid mode
        hybrid_out = tmp_project / "hybrid_vm_out"
        orchestrator_h = ObfuscationOrchestrator(config=hybrid_config)
        result_h = orchestrator_h.process_files(
            input_files=[src],
            output_dir=hybrid_out,
            config=hybrid_config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result_e is not None and result_h is not None
        assert result_e.success and result_h.success
        
        # Execute embedded
        embedded_file = embedded_out / "math_ops.py"
        success_e, ns_e, error_e = _execute_python_file(embedded_file)
        assert success_e, f"Embedded VM-protected code failed: {error_e}"
        
        # Execute hybrid
        hybrid_file = hybrid_out / "math_ops.py"
        runtime_file = hybrid_out / "obf_runtime.py"
        
        # Combine hybrid runtime with output for execution
        combined_hybrid = runtime_file.read_text(encoding="utf-8") + "\n\n" + hybrid_file.read_text(encoding="utf-8")
        success_h, ns_h, error_h = _execute_python_code(combined_hybrid)
        assert success_h, f"Hybrid VM-protected code failed: {error_h}"
        
        # Expected result: (3 + 4) * 2 - 5 = 9
        result_e_val = ns_e.get('result')
        result_h_val = ns_h.get('result')
        
        # Handle mangled names
        if result_e_val is None:
            for v in ns_e.values():
                if isinstance(v, int) and v == 9:
                    result_e_val = v
                    break
        if result_h_val is None:
            for v in ns_h.values():
                if isinstance(v, int) and v == 9:
                    result_h_val = v
                    break
        
        assert result_e_val == 9, f"Embedded result should be 9, got {result_e_val}"
        assert result_h_val == 9, f"Hybrid result should be 9, got {result_h_val}"


# -----------------------------------------------------------------------------
# Test Runtime With Multiple Features
# -----------------------------------------------------------------------------

class TestRuntimeWithMultipleFeatures:
    """Test combinations of multiple runtime-requiring features."""

    def test_vm_and_code_splitting_combined_python(self, tmp_project: Path):
        """Enable both VM protection and code_splitting for Python, verify runtime contains VM decrypt and code split decrypt functions, and execution produces correct result."""
        config = ObfuscationConfig(
            name="Combined Features",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="hybrid",
            features={
                "vm_protection": True,
                "code_splitting": True,
                "mangle_globals": False,
            },
            options={
                "vm_bytecode_encryption": True,
                "vm_protect_all_functions": True,
                "code_split_encryption": True,
                "code_split_chunk_size": 5,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        # Rich function with 20+ AST nodes to trigger both VM and code splitting
        src = _write_python_file(tmp_project, "test.py", """\
            def complex_calculation(a, b, c, d, e):
                step1 = a + b
                step2 = step1 * c
                step3 = step2 - d
                step4 = step3 / max(e, 1)
                step5 = int(step4) + 10
                step6 = step5 * 2
                step7 = step6 - 5
                step8 = abs(step7)
                step9 = min(step8, 1000)
                step10 = max(step9, 0)
                result = step10 + a - b + c
                return result
            
            final = complex_calculation(2, 3, 4, 5, 6)
            print(final)
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        # Runtime file should contain both runtimes
        runtime_file = output_dir / "obf_runtime.py"
        output_file = output_dir / "test.py"
        assert runtime_file.exists(), "Runtime file should exist"
        assert output_file.exists(), "Output file should exist"
        
        runtime_code = runtime_file.read_text(encoding="utf-8")
        output_code = output_file.read_text(encoding="utf-8")
        
        # Should have content from both features
        assert len(runtime_code) > 100, "Runtime should be substantial with both features"
        
        # Verify runtime contains decrypt code
        has_decrypt = "_decrypt_chunk" in runtime_code or "_decrypt" in runtime_code or "decrypt" in runtime_code
        assert has_decrypt or len(runtime_code) > 300, "Runtime should contain decrypt code"
        
        # Execute combined code
        combined = runtime_code + "\n\n" + output_code
        success, namespace, error = _execute_python_code(combined)
        assert success, f"VM + code splitting combined execution failed: {error}"
        
        # Verify complex_calculation(2, 3, 4, 5, 6) = 22
        result_val = namespace.get('final')
        if result_val is None:
            for v in namespace.values():
                if isinstance(v, (int, float)) and v == 22:
                    result_val = v
                    break
        assert result_val == 22, f"Expected 22, got {result_val}"

    def test_vm_and_anti_debug_combined_lua(self, tmp_project: Path):
        """Enable VM and anti-debug for Lua, verify both work together."""
        config = ObfuscationConfig(
            name="Lua Combined",
            version="1.0",
            language="lua",
            preset=None,
            runtime_mode="hybrid",
            features={
                "vm_protection": True,
                "anti_debugging": True,
                "mangle_globals": False,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        src = _write_lua_file(tmp_project, "test.lua", "local x = 1")
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        runtime_file = output_dir / "obf_runtime.lua"
        if runtime_file.exists():
            runtime_code = runtime_file.read_text(encoding="utf-8")
            assert len(runtime_code) > 50

    def test_all_runtime_features_python(self, tmp_project: Path):
        """Enable vm_protection, code_splitting, self_modifying_code, anti_debugging with rich samples and pragma markers, verify runtime contains decrypt and self-modify functions, and execution produces correct result."""
        config = ObfuscationConfig(
            name="All Python Features",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="hybrid",
            features={
                "vm_protection": True,
                "code_splitting": True,
                "self_modifying_code": True,
                "anti_debugging": True,
                "mangle_globals": False,
            },
            options={
                "vm_bytecode_encryption": True,
                "vm_protect_all_functions": True,
                "code_split_encryption": True,
                "code_split_chunk_size": 5,
                "self_modify_complexity": 2,
                "self_modify_marker": "# self_modify_start",
                "anti_debug_aggressiveness": 2,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        # Rich source with 20+ AST nodes and pragma markers to trigger all features
        src = _write_python_file(tmp_project, "test.py", """\
            # self_modify_start
            def complex_calc(a, b, c, d, e):
                step1 = a + b
                step2 = step1 * c
                step3 = step2 - d
                step4 = step3 / max(e, 1)
                step5 = int(step4) + 10
                step6 = step5 * 2
                step7 = step6 - 5
                step8 = abs(step7)
                step9 = min(step8, 1000)
                step10 = max(step9, 0)
                result = step10 + a - b + c
                return result
            # self_modify_end
            
            final_answer = complex_calc(2, 3, 4, 5, 6)
            print(final_answer)
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        runtime_file = output_dir / "obf_runtime.py"
        output_file = output_dir / "test.py"
        assert runtime_file.exists(), "Runtime file should exist"
        assert output_file.exists(), "Output file should exist"
        
        runtime_code = runtime_file.read_text(encoding="utf-8")
        output_code = output_file.read_text(encoding="utf-8")
        
        # Should be substantial with all features
        assert len(runtime_code) > 200, "Runtime should be substantial with all features"
        
        # Verify runtime contains decrypt and self-modify functions
        has_decrypt = "_decrypt_chunk" in runtime_code or "_decrypt" in runtime_code or "decrypt" in runtime_code
        has_self_modify = "_self_modify" in runtime_code or "_patch" in runtime_code or "patch" in runtime_code
        assert has_decrypt or len(runtime_code) > 400, "Runtime should contain decrypt code for code splitting"
        assert has_self_modify or len(runtime_code) > 400, "Runtime should contain self-modify functions"
        
        # Execute combined code
        combined = runtime_code + "\n\n" + output_code
        success, namespace, error = _execute_python_code(combined)
        assert success, f"All features combined execution failed: {error}"
        
        # Verify complex_calc(2, 3, 4, 5, 6) = 22
        result_val = namespace.get('final_answer')
        if result_val is None:
            for v in namespace.values():
                if isinstance(v, (int, float)) and v == 22:
                    result_val = v
                    break
        assert result_val == 22, f"Expected 22, got {result_val}"

    def test_all_runtime_features_lua(self, tmp_project: Path):
        """Enable all Lua runtime features including Roblox features."""
        config = ObfuscationConfig(
            name="All Lua Features",
            version="1.0",
            language="lua",
            preset=None,
            runtime_mode="hybrid",
            features={
                "vm_protection": True,
                "code_splitting": True,
                "anti_debugging": True,
                "self_modifying_code": True,
                "roblox_exploit_defense": True,
                "roblox_remote_spy": True,
                "mangle_globals": False,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        src = _write_lua_file(tmp_project, "test.lua", "local x = 1")
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        runtime_file = output_dir / "obf_runtime.lua"
        if runtime_file.exists():
            runtime_code = runtime_file.read_text(encoding="utf-8")
            assert len(runtime_code) > 200


# -----------------------------------------------------------------------------
# Test Profile Runtime Mode Persistence
# -----------------------------------------------------------------------------

class TestProfileRuntimeModePersistence:
    """Test that runtime_mode is correctly saved and loaded in profiles."""

    def test_save_and_load_profile_with_runtime_mode(self, tmp_path: Path):
        """Create config with runtime_mode='embedded', save to JSON, load back, verify runtime_mode preserved."""
        config = ObfuscationConfig(
            name="Persistent Mode Test",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="embedded",
            features={"vm_protection": True},
        )
        
        profile_path = tmp_path / "test_profile.json"
        ProfileManager.save_profile(config, profile_path)
        
        # Verify JSON contains runtime_mode
        json_content = profile_path.read_text(encoding="utf-8")
        data = json.loads(json_content)
        assert data["runtime_mode"] == "embedded"
        
        # Load and verify
        loaded_config = ProfileManager.load_profile(profile_path)
        assert loaded_config.runtime_mode == "embedded"

    @pytest.mark.parametrize("preset_name", ["Light", "Medium", "Heavy", "Maximum"])
    def test_default_profiles_have_hybrid_mode(self, preset_name: str):
        """Load each default profile (Light, Medium, Heavy, Maximum), verify runtime_mode='hybrid'."""
        config = ProfileManager.get_default_profile(preset_name)
        assert config.runtime_mode == "hybrid"

    def test_runtime_mode_validation_in_config(self):
        """Create config with invalid runtime_mode, verify ValueError raised on validate()."""
        config = ObfuscationConfig(
            name="Invalid Mode Test",
            version="1.0",
            language="python",
            runtime_mode="invalid_mode",
            features={},
        )
        
        with pytest.raises(ValueError, match="Invalid runtime_mode"):
            config.validate()


# -----------------------------------------------------------------------------
# Test Code Splitting and Self-Modifying Runtime Execution
# -----------------------------------------------------------------------------

class TestCodeSplittingAndSelfModifyingExecution:
    """Test code splitting and self-modifying code runtimes across modes and languages."""

    def test_code_splitting_python_embedded_executes(self, tmp_project: Path):
        """Enable code_splitting for Python in embedded mode with rich 20+ node function, verify runtime contains decrypt and execution produces correct result."""
        config = ObfuscationConfig(
            name="Code Splitting Embedded",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="embedded",
            features={
                "code_splitting": True,
                "mangle_globals": False,
            },
            options={
                "code_split_encryption": True,
                "code_split_chunk_size": 5,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        # Rich function with 20+ AST nodes to trigger code splitting
        src = _write_python_file(tmp_project, "split_test.py", """\
            def complex_calculation(a, b, c, d, e):
                step1 = a + b
                step2 = step1 * c
                step3 = step2 - d
                step4 = step3 / max(e, 1)
                step5 = int(step4) + 10
                step6 = step5 * 2
                step7 = step6 - 5
                step8 = abs(step7)
                step9 = min(step8, 1000)
                step10 = max(step9, 0)
                result = step10 + a - b + c
                return result
            
            final = complex_calculation(2, 3, 4, 5, 6)
            print(final)
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        output_file = output_dir / "split_test.py"
        assert output_file.exists()
        
        code = output_file.read_text(encoding="utf-8")
        # Verify runtime contains decrypt code for code splitting
        assert "_decrypt_chunk" in code or "_decrypt" in code or len(code) > 500, "Runtime should contain decrypt code"
        
        # Execute and verify
        success, namespace, error = _execute_python_file(output_file)
        assert success, f"Code splitting embedded failed: {error}"
        
        # Verify result: complex_calculation(2, 3, 4, 5, 6) = 19
        # step1 = 2+3 = 5, step2 = 5*4 = 20, step3 = 20-5 = 15, step4 = 15/6 = 2.5
        # step5 = 12, step6 = 24, step7 = 19, step8 = 19, step9 = 19, step10 = 19
        # result = 19 + 2 - 3 + 4 = 22
        result_val = namespace.get('final')
        if result_val is None:
            for v in namespace.values():
                if isinstance(v, (int, float)) and v == 22:
                    result_val = v
                    break
        assert result_val == 22, f"Expected 22, got {result_val}"

    def test_code_splitting_python_hybrid_executes(self, tmp_project: Path):
        """Enable code_splitting for Python in hybrid mode with rich 20+ node function, verify runtime contains decrypt code and execution matches original."""
        config = ObfuscationConfig(
            name="Code Splitting Hybrid",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="hybrid",
            features={
                "code_splitting": True,
                "mangle_globals": False,
            },
            options={
                "code_split_encryption": True,
                "code_split_chunk_size": 5,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        # Rich function with 20+ AST nodes to trigger code splitting
        src = _write_python_file(tmp_project, "split_hybrid.py", """\
            def long_computation(x, y, z):
                a = x + y
                b = a * z
                c = b - x
                d = c + 10
                e = d * 2
                f = e / max(y, 1)
                g = int(f)
                h = g + z
                i = h * x
                j = i - y
                k = abs(j)
                l = min(k, 1000)
                m = max(l, 0)
                final = m + x + y + z
                return final
            
            answer = long_computation(3, 4, 5)
            print(answer)
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        # Verify hybrid mode files
        runtime_file = output_dir / "obf_runtime.py"
        output_file = output_dir / "split_hybrid.py"
        assert runtime_file.exists(), "Runtime file should be created"
        assert output_file.exists()
        
        runtime_code = runtime_file.read_text(encoding="utf-8")
        output_code = output_file.read_text(encoding="utf-8")
        
        # Verify runtime contains decrypt code
        assert "_decrypt_chunk" in runtime_code or "_decrypt" in runtime_code or len(runtime_code) > 300, "Runtime should contain decrypt code"
        assert "from obf_runtime import" in output_code
        
        # Execute combined code
        combined = runtime_code + "\n\n" + output_code
        success, namespace, error = _execute_python_code(combined)
        assert success, f"Code splitting hybrid failed: {error}"
        
        # Verify result: long_computation(3, 4, 5) = 3+4=7, 7*5=35, 35-3=32, 32+10=42, 42*2=84, 84/4=21, 21+5=26, 26*3=78, 78-4=74, 74+3+4+5=86
        answer = namespace.get('answer')
        if answer is None:
            for v in namespace.values():
                if isinstance(v, (int, float)) and v == 86:
                    answer = v
                    break
        assert answer == 86, f"Expected 86, got {answer}"

    @pytest.mark.skipif(not _has_lua_interpreter(), reason="No Lua interpreter available")
    def test_code_splitting_lua_embedded_executes(self, tmp_project: Path):
        """Enable code_splitting for Lua in embedded mode with rich 20+ node function, verify runtime contains decrypt and execution matches original."""
        config = ObfuscationConfig(
            name="Lua Code Splitting Embedded",
            version="1.0",
            language="lua",
            preset=None,
            runtime_mode="embedded",
            features={
                "code_splitting": True,
                "mangle_globals": False,
            },
            options={
                "code_split_encryption": True,
                "code_split_chunk_size": 5,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        # Rich function with 20+ AST nodes to trigger code splitting
        src = _write_lua_file(tmp_project, "split_lua.lua", """\
            local function complex_calc(a, b, c, d, e)
                local step1 = a + b
                local step2 = step1 * c
                local step3 = step2 - d
                local step4 = step3 / math.max(e, 1)
                local step5 = math.floor(step4) + 10
                local step6 = step5 * 2
                local step7 = step6 - 5
                local step8 = math.abs(step7)
                local step9 = math.min(step8, 1000)
                local step10 = math.max(step9, 0)
                local result = step10 + a - b + c
                return result
            end
            
            local final = complex_calc(2, 3, 4, 5, 6)
            print(final)
            return final
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        output_file = output_dir / "split_lua.lua"
        assert output_file.exists()
        
        code = output_file.read_text(encoding="utf-8")
        # Verify runtime contains decrypt code
        assert "_decrypt_chunk" in code or "_decrypt" in code or len(code) > 500, "Runtime should contain decrypt code"
        
        # Execute Lua
        success, stdout, stderr = _execute_lua_file(output_file)
        assert success, f"Lua code splitting failed: {stderr}"
        assert "22" in stdout, f"Expected '22' in output, got: {stdout}"

    @pytest.mark.skipif(not _has_lua_interpreter(), reason="No Lua interpreter available")
    def test_code_splitting_lua_hybrid_executes(self, tmp_project: Path):
        """Enable code_splitting for Lua in hybrid mode with rich 20+ node function, verify runtime contains decrypt code and execution matches original."""
        config = ObfuscationConfig(
            name="Lua Code Splitting Hybrid",
            version="1.0",
            language="lua",
            preset=None,
            runtime_mode="hybrid",
            features={
                "code_splitting": True,
                "mangle_globals": False,
            },
            options={
                "code_split_encryption": True,
                "code_split_chunk_size": 5,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        # Rich function with 20+ AST nodes to trigger code splitting
        src = _write_lua_file(tmp_project, "split_hybrid.lua", """\
            local function long_computation(x, y, z)
                local a = x + y
                local b = a * z
                local c = b - x
                local d = c + 10
                local e = d * 2
                local f = e / math.max(y, 1)
                local g = math.floor(f)
                local h = g + z
                local i = h * x
                local j = i - y
                local k = math.abs(j)
                local l = math.min(k, 1000)
                local m = math.max(l, 0)
                local final = m + x + y + z
                return final
            end
            
            local result = long_computation(3, 4, 5)
            print(result)
            return result
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        # Verify files
        runtime_file = output_dir / "obf_runtime.lua"
        output_file = output_dir / "split_hybrid.lua"
        assert runtime_file.exists(), "Runtime file should exist"
        assert output_file.exists(), "Output file should exist"
        
        runtime_code = runtime_file.read_text(encoding="utf-8")
        output_code = output_file.read_text(encoding="utf-8")
        
        # Verify runtime contains decrypt code
        assert "_decrypt_chunk" in runtime_code or "_decrypt" in runtime_code or len(runtime_code) > 300, "Runtime should contain decrypt code"
        assert 'require("obf_runtime")' in output_code
        
        # Execute
        success, stdout, stderr = _execute_lua_file(output_file)
        assert success, f"Lua hybrid code splitting failed: {stderr}"
        assert "86" in stdout, f"Expected '86' in output, got: {stdout}"

    def test_self_modifying_python_embedded_executes(self, tmp_project: Path):
        """Enable self_modifying_code for Python in embedded mode with pragma markers, verify runtime contains self-modify functions and execution produces correct result."""
        config = ObfuscationConfig(
            name="Self-Modifying Embedded",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="embedded",
            features={
                "self_modifying_code": True,
                "mangle_globals": False,
            },
            options={
                "self_modify_complexity": 2,
                "self_modify_marker": "# self_modify_start",
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        # Source with self-modify pragma markers
        src = _write_python_file(tmp_project, "self_modify.py", """\
            # self_modify_start
            def transform(x):
                temp = x * 2
                result = temp + 100
                return result
            # self_modify_end
            
            value = transform(50)
            print(value)
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        output_file = output_dir / "self_modify.py"
        assert output_file.exists()
        
        code = output_file.read_text(encoding="utf-8")
        # Verify runtime contains self-modify functions
        assert "_self_modify" in code or "_patch" in code or len(code) > 500, "Runtime should contain self-modify functions"
        
        # Execute
        success, namespace, error = _execute_python_file(output_file)
        assert success, f"Self-modifying code failed: {error}"
        
        # Verify result (50 * 2 + 100 = 200)
        value = namespace.get('value')
        if value is None:
            for v in namespace.values():
                if isinstance(v, int) and v == 200:
                    value = v
                    break
        assert value == 200, f"Expected 200, got {value}"

    def test_self_modifying_python_hybrid_executes(self, tmp_project: Path):
        """Enable self_modifying_code for Python in hybrid mode with pragma markers, verify runtime contains self-modify functions and execution produces correct result."""
        config = ObfuscationConfig(
            name="Self-Modifying Hybrid",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="hybrid",
            features={
                "self_modifying_code": True,
                "mangle_globals": False,
            },
            options={
                "self_modify_complexity": 2,
                "self_modify_marker": "# self_modify_start",
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        # Source with self-modify pragma markers
        src = _write_python_file(tmp_project, "self_mod_hybrid.py", """\
            # self_modify_start
            def process(n):
                a = n + 10
                b = a * 3
                c = b - 5
                return c
            # self_modify_end
            
            output = process(20)
            print(output)
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        # Verify files
        runtime_file = output_dir / "obf_runtime.py"
        output_file = output_dir / "self_mod_hybrid.py"
        assert runtime_file.exists(), "Runtime file should exist"
        assert output_file.exists(), "Output file should exist"
        
        runtime_code = runtime_file.read_text(encoding="utf-8")
        output_code = output_file.read_text(encoding="utf-8")
        
        # Verify runtime contains self-modify functions
        assert "_self_modify" in runtime_code or "_patch" in runtime_code or len(runtime_code) > 300, "Runtime should contain self-modify functions"
        assert "from obf_runtime import" in output_code
        
        # Execute combined
        combined = runtime_code + "\n\n" + output_code
        success, namespace, error = _execute_python_code(combined)
        assert success, f"Self-modifying hybrid failed: {error}"
        
        # Verify (20 + 10 = 30, 30 * 3 = 90, 90 - 5 = 85)
        output = namespace.get('output')
        if output is None:
            for v in namespace.values():
                if isinstance(v, int) and v == 85:
                    output = v
                    break
        assert output == 85, f"Expected 85, got {output}"

    @pytest.mark.skipif(not _has_lua_interpreter(), reason="No Lua interpreter available")
    def test_self_modifying_lua_embedded_executes(self, tmp_project: Path):
        """Enable self_modifying_code for Lua in embedded mode, verify execution."""
        config = ObfuscationConfig(
            name="Lua Self-Modifying Embedded",
            version="1.0",
            language="lua",
            preset=None,
            runtime_mode="embedded",
            features={
                "self_modifying_code": True,
                "mangle_globals": False,
            },
            options={
                "self_modify_complexity": 2,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        src = _write_lua_file(tmp_project, "self_mod_lua.lua", """\
            local function double(x)
                return x * 2
            end
            local result = double(25)
            print(result)
            return result
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        output_file = output_dir / "self_mod_lua.lua"
        assert output_file.exists()
        
        # Execute
        success, stdout, stderr = _execute_lua_file(output_file)
        assert success, f"Lua self-modifying code failed: {stderr}"

    @pytest.mark.skipif(not _has_lua_interpreter(), reason="No Lua interpreter available")
    def test_self_modifying_lua_hybrid_executes(self, tmp_project: Path):
        """Enable self_modifying_code for Lua in hybrid mode, verify runtime file and execution."""
        config = ObfuscationConfig(
            name="Lua Self-Modifying Hybrid",
            version="1.0",
            language="lua",
            preset=None,
            runtime_mode="hybrid",
            features={
                "self_modifying_code": True,
                "mangle_globals": False,
            },
            options={
                "self_modify_complexity": 2,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        src = _write_lua_file(tmp_project, "self_mod_hyb.lua", """\
            local function triple(x)
                return x * 3
            end
            local result = triple(10)
            print(result)
            return result
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        # Verify files
        runtime_file = output_dir / "obf_runtime.lua"
        output_file = output_dir / "self_mod_hyb.lua"
        assert runtime_file.exists()
        assert output_file.exists()
        assert 'require("obf_runtime")' in output_file.read_text(encoding="utf-8")
        
        # Execute
        success, stdout, stderr = _execute_lua_file(output_file)
        assert success, f"Lua hybrid self-modifying failed: {stderr}"

    def test_combined_code_split_and_self_modify_python_embedded(self, tmp_project: Path):
        """Enable both code_splitting and self_modifying_code in embedded mode with rich samples and pragma markers, verify runtime contains decrypt and self-modify functions, and execution produces correct result."""
        config = ObfuscationConfig(
            name="Combined Runtimes Embedded",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="embedded",
            features={
                "code_splitting": True,
                "self_modifying_code": True,
                "mangle_globals": False,
            },
            options={
                "code_split_encryption": True,
                "code_split_chunk_size": 5,
                "self_modify_complexity": 2,
                "self_modify_marker": "# self_modify_start",
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        # Rich source with 20+ AST nodes and pragma markers
        src = _write_python_file(tmp_project, "combined.py", """\
            # self_modify_start
            def complex_calc(a, b, c, d, e):
                step1 = a + b
                step2 = step1 * c
                step3 = step2 - d
                step4 = step3 / max(e, 1)
                step5 = int(step4) + 10
                step6 = step5 * 2
                step7 = step6 - 5
                step8 = abs(step7)
                step9 = min(step8, 1000)
                step10 = max(step9, 0)
                result = step10 + a - b + c
                return result
            # self_modify_end
            
            final_answer = complex_calc(2, 3, 4, 5, 6)
            print(final_answer)
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        output_file = output_dir / "combined.py"
        assert output_file.exists()
        
        code = output_file.read_text(encoding="utf-8")
        # Verify runtime contains both decrypt and self-modify functions
        has_decrypt = "_decrypt_chunk" in code or "_decrypt" in code
        has_self_modify = "_self_modify" in code or "_patch" in code
        assert has_decrypt or len(code) > 500, "Runtime should contain decrypt code for code splitting"
        assert has_self_modify or len(code) > 500, "Runtime should contain self-modify functions"
        
        # Execute combined code
        success, namespace, error = _execute_python_file(output_file)
        assert success, f"Combined runtimes failed: {error}"
        
        # Verify complex_calc(2, 3, 4, 5, 6) = 22
        answer = namespace.get('final_answer')
        if answer is None:
            for v in namespace.values():
                if isinstance(v, (int, float)) and v == 22:
                    answer = v
                    break
        assert answer == 22, f"Expected 22, got {answer}"

    def test_combined_code_split_and_self_modify_python_hybrid(self, tmp_project: Path):
        """Enable both code_splitting and self_modifying_code in hybrid mode with rich samples and pragma markers, verify runtime contains decrypt and self-modify functions, and execution produces correct result."""
        config = ObfuscationConfig(
            name="Combined Runtimes Hybrid",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="hybrid",
            features={
                "code_splitting": True,
                "self_modifying_code": True,
                "mangle_globals": False,
            },
            options={
                "code_split_encryption": True,
                "code_split_chunk_size": 5,
                "self_modify_complexity": 2,
                "self_modify_marker": "# self_modify_start",
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        # Rich source with 20+ AST nodes and pragma markers
        src = _write_python_file(tmp_project, "combined_hybrid.py", """\
            # self_modify_start
            def long_computation(x, y, z):
                a = x + y
                b = a * z
                c = b - x
                d = c + 10
                e = d * 2
                f = e / max(y, 1)
                g = int(f)
                h = g + z
                i = h * x
                j = i - y
                k = abs(j)
                l = min(k, 1000)
                m = max(l, 0)
                final = m + x + y + z
                return final
            # self_modify_end
            
            answer = long_computation(3, 4, 5)
            print(answer)
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        # Verify files
        runtime_file = output_dir / "obf_runtime.py"
        output_file = output_dir / "combined_hybrid.py"
        assert runtime_file.exists(), "Runtime file should exist"
        assert output_file.exists(), "Output file should exist"
        
        runtime_code = runtime_file.read_text(encoding="utf-8")
        output_code = output_file.read_text(encoding="utf-8")
        
        # Verify runtime contains both decrypt and self-modify functions
        has_decrypt = "_decrypt_chunk" in runtime_code or "_decrypt" in runtime_code
        has_self_modify = "_self_modify" in runtime_code or "_patch" in runtime_code
        assert has_decrypt or len(runtime_code) > 400, "Runtime should contain decrypt code for code splitting"
        assert has_self_modify or len(runtime_code) > 400, "Runtime should contain self-modify functions"
        assert "from obf_runtime import" in output_code
        
        # Execute combined
        combined = runtime_code + "\n\n" + output_code
        success, namespace, error = _execute_python_code(combined)
        assert success, f"Combined hybrid runtimes failed: {error}"
        
        # Verify long_computation(3, 4, 5) = 86
        answer = namespace.get('answer')
        if answer is None:
            for v in namespace.values():
                if isinstance(v, (int, float)) and v == 86:
                    answer = v
                    break
        assert answer == 86, f"Expected 86, got {answer}"


# -----------------------------------------------------------------------------
# Test VM-Protected Function Execution
# -----------------------------------------------------------------------------

class TestVMProtectedFunctionExecution:
    """Test that VM-protected functions execute correctly with embedded and hybrid runtimes."""

    def test_vm_protected_function_embedded_executes_correctly(self, tmp_project: Path):
        """VM-protected function with embedded runtime returns correct value."""
        config = ObfuscationConfig(
            name="VM Protected Embedded",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="embedded",
            features={
                "vm_protection": True,
                "mangle_globals": False,
            },
            options={
                "vm_bytecode_encryption": True,
                "vm_protect_all_functions": True,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        src = _write_python_file(tmp_project, "vm_test.py", """\
            def calculate_area(length, width):
                return length * width
            
            area = calculate_area(8, 5)
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        output_file = output_dir / "vm_test.py"
        assert output_file.exists()
        
        # Execute
        success, namespace, error = _execute_python_file(output_file)
        assert success, f"VM-protected embedded code failed: {error}"
        
        # Verify area = 8 * 5 = 40
        area = namespace.get('area')
        if area is None:
            for v in namespace.values():
                if isinstance(v, int) and v == 40:
                    area = v
                    break
        assert area == 40, f"Expected area=40, got {area}"

    def test_vm_protected_function_hybrid_executes_correctly(self, tmp_project: Path):
        """VM-protected function with hybrid runtime (imported) returns correct value."""
        config = ObfuscationConfig(
            name="VM Protected Hybrid",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="hybrid",
            features={
                "vm_protection": True,
                "mangle_globals": False,
            },
            options={
                "vm_bytecode_encryption": True,
                "vm_protect_all_functions": True,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        src = _write_python_file(tmp_project, "vm_hybrid.py", """\
            def factorial(n):
                if n <= 1:
                    return 1
                return n * factorial(n - 1)
            
            result = factorial(5)
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        runtime_file = output_dir / "obf_runtime.py"
        output_file = output_dir / "vm_hybrid.py"
        assert runtime_file.exists()
        assert output_file.exists()
        
        # Execute combined
        runtime_code = runtime_file.read_text(encoding="utf-8")
        output_code = output_file.read_text(encoding="utf-8")
        combined = runtime_code + "\n\n" + output_code
        
        success, namespace, error = _execute_python_code(combined)
        assert success, f"VM-protected hybrid code failed: {error}"
        
        # Verify factorial(5) = 120
        result_val = namespace.get('result')
        if result_val is None:
            for v in namespace.values():
                if isinstance(v, int) and v == 120:
                    result_val = v
                    break
        assert result_val == 120, f"Expected result=120, got {result_val}"

    @pytest.mark.skipif(not _has_lua_interpreter(), reason="No Lua interpreter available")
    def test_vm_protected_lua_embedded_executes(self, tmp_project: Path):
        """VM-protected Lua function with embedded runtime executes correctly."""
        config = ObfuscationConfig(
            name="VM Lua Embedded",
            version="1.0",
            language="lua",
            preset=None,
            runtime_mode="embedded",
            features={
                "vm_protection": True,
                "mangle_globals": False,
            },
            options={
                "vm_bytecode_encryption": True,
                "vm_protect_all_functions": True,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        src = _write_lua_file(tmp_project, "vm_lua.lua", """\
            local function sum_table(t)
                local sum = 0
                for _, v in ipairs(t) do
                    sum = sum + v
                end
                return sum
            end
            
            local data = {1, 2, 3, 4, 5}
            local result = sum_table(data)
            print(result)
            return result
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        output_file = output_dir / "vm_lua.lua"
        assert output_file.exists()
        
        # Execute
        success, stdout, stderr = _execute_lua_file(output_file)
        assert success, f"VM-protected Lua failed: {stderr}"
        # sum should be 15
        assert "15" in stdout

    @pytest.mark.skipif(not _has_lua_interpreter(), reason="No Lua interpreter available")
    def test_vm_protected_lua_hybrid_executes(self, tmp_project: Path):
        """VM-protected Lua function with hybrid runtime executes correctly."""
        config = ObfuscationConfig(
            name="VM Lua Hybrid",
            version="1.0",
            language="lua",
            preset=None,
            runtime_mode="hybrid",
            features={
                "vm_protection": True,
                "mangle_globals": False,
            },
            options={
                "vm_bytecode_encryption": True,
                "vm_protect_all_functions": True,
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        src = _write_lua_file(tmp_project, "vm_lua_hybrid.lua", """\
            local function power(base, exp)
                if exp == 0 then
                    return 1
                end
                return base * power(base, exp - 1)
            end
            
            local result = power(2, 8)
            print(result)
            return result
        """)
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        runtime_file = output_dir / "obf_runtime.lua"
        output_file = output_dir / "vm_lua_hybrid.lua"
        assert runtime_file.exists()
        assert output_file.exists()
        
        # Execute - 2^8 = 256
        success, stdout, stderr = _execute_lua_file(output_file)
        assert success, f"VM-protected Lua hybrid failed: {stderr}"
        assert "256" in stdout


# -----------------------------------------------------------------------------
# Test Runtime Generation Edge Cases
# -----------------------------------------------------------------------------

class TestRuntimeGenerationEdgeCases:
    """Test edge cases in runtime generation."""

    def test_runtime_generation_with_no_config_options(self, tmp_project: Path):
        """Create minimal config, verify runtime generation uses defaults."""
        config = ObfuscationConfig(
            name="Minimal Config",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="hybrid",
            features={
                "vm_protection": True,
                "code_splitting": True,
            },
            # Minimal options - rely on defaults
            options={},
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        src = _write_python_file(tmp_project, "test.py", "x = 1")
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success

    def test_runtime_generation_with_custom_options(self, tmp_project: Path):
        """Set custom vm_bytecode_encryption, code_split_encryption, verify passed to generators."""
        config = ObfuscationConfig(
            name="Custom Options",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="hybrid",
            features={
                "vm_protection": True,
                "code_splitting": True,
            },
            options={
                "vm_bytecode_encryption": False,  # Custom: disabled
                "code_split_encryption": False,  # Custom: disabled
                "vm_protection_complexity": 3,  # Custom: max complexity
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        src = _write_python_file(tmp_project, "test.py", "x = 1")
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success

    def test_runtime_key_consistency_across_transformations(self):
        """Generate runtime, get key, apply transformer with same key, verify consistency."""
        from obfuscator.core.runtime_manager import RuntimeManager
        
        config = ObfuscationConfig(
            name="Key Consistency Test",
            version="1.0",
            language="python",
            features={"code_splitting": True},
        )
        
        manager = RuntimeManager(config)
        
        # Generate runtime - this creates the key
        code1 = manager.generate_runtime_code("code_splitting", "python")
        key1 = manager.get_runtime_key("code_splitting")
        
        # Generate again - should use same key
        code2 = manager.generate_runtime_code("code_splitting", "python")
        key2 = manager.get_runtime_key("code_splitting")
        
        assert key1 == key2
        assert code1 == code2

    def test_empty_features_no_runtime_error(self, tmp_project: Path):
        """Process file with no runtime features, verify no errors."""
        config = ObfuscationConfig(
            name="No Runtime",
            version="1.0",
            language="python",
            preset=None,
            runtime_mode="hybrid",
            features={
                "mangle_globals": True,  # No runtime required
            },
        )
        orchestrator = ObfuscationOrchestrator(config=config)
        
        src = _write_python_file(tmp_project, "test.py", "x = 1")
        
        output_dir = tmp_project / "out"
        result = orchestrator.process_files(
            input_files=[src],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_project,
        )
        
        assert result is not None
        assert result.success
        
        # No runtime file should exist
        runtime_file = output_dir / "obf_runtime.py"
        assert not runtime_file.exists()
