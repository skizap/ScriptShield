"""Shared fixtures for orchestrator workflow tests."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from obfuscator.core.config import ObfuscationConfig
from obfuscator.core.orchestrator import (
    ConflictStrategy,
    ErrorStrategy,
    ObfuscationOrchestrator,
    ProgressInfo,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def create_test_file(directory: Path, name: str, content: str = "") -> Path:
    """Create a test file with the given name and content."""
    file_path = directory / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not content:
        ext = file_path.suffix.lower()
        if ext in (".py", ".pyw"):
            content = "x = 1\n"
        elif ext in (".lua", ".luau"):
            content = "local x = 1\n"
        else:
            content = ""
    file_path.write_text(textwrap.dedent(content), encoding="utf-8")
    return file_path


def create_readonly_file(directory: Path, name: str, content: str = "x = 1\n") -> Path:
    """Create a file and remove read permissions."""
    file_path = create_test_file(directory, name, content)
    os.chmod(file_path, 0o000)
    return file_path


def assert_validation_passed(result) -> None:
    """Assert that a ValidationResult indicates success."""
    assert result.success is True, f"Validation failed with errors: {result.errors}"
    assert len(result.errors) == 0


def assert_state_transition(log_records: list, from_state: str, to_state: str) -> bool:
    """Check that a state transition appears in log records."""
    pattern = f"{from_state} -> {to_state}"
    return any(pattern in record.message for record in log_records)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary directory structure with sample files."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return tmp_path


@pytest.fixture
def sample_python_files(tmp_project: Path) -> list[Path]:
    """Return list of Path objects for test Python files."""
    src = tmp_project / "src"
    files = [
        create_test_file(src, "main.py", "def main():\n    return 1\n"),
        create_test_file(src, "utils.py", "def helper():\n    return 2\n"),
        create_test_file(src, "config.py", "SETTING = 'value'\n"),
    ]
    return files


@pytest.fixture
def sample_lua_files(tmp_project: Path) -> list[Path]:
    """Return list of Path objects for test Lua files."""
    src = tmp_project / "src"
    files = [
        create_test_file(src, "main.lua", "local function main()\n    return 1\nend\n"),
        create_test_file(src, "utils.lua", "local function helper()\n    return 2\nend\n"),
    ]
    return files


@pytest.fixture
def sample_config() -> ObfuscationConfig:
    """Return a basic ObfuscationConfig for testing."""
    return ObfuscationConfig(
        name="test_workflow",
        language="python",
        features={"mangle_globals": True},
        options={
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
        },
        symbol_table_options={
            "identifier_prefix": "_0x",
            "mangling_strategy": "sequential",
            "preserve_exports": False,
            "preserve_constants": False,
        },
        conflict_strategy="overwrite",
    )


@pytest.fixture
def orchestrator_instance(sample_config: ObfuscationConfig) -> ObfuscationOrchestrator:
    """Return an ObfuscationOrchestrator configured for testing."""
    return ObfuscationOrchestrator(config=sample_config)


@pytest.fixture
def mock_progress_callback() -> MagicMock:
    """Return a mock function that captures progress updates."""
    callback = MagicMock()
    callback.captured: list[ProgressInfo] = []

    def side_effect(progress_info: ProgressInfo) -> None:
        callback.captured.append(progress_info)

    callback.side_effect = side_effect
    return callback


@pytest.fixture
def mock_error_callback() -> MagicMock:
    """Return a mock function that simulates user error decisions (continue)."""
    callback = MagicMock(return_value=True)
    return callback


@pytest.fixture
def mock_conflict_callback() -> MagicMock:
    """Return a mock function that simulates user conflict decisions."""
    callback = MagicMock(return_value=ConflictStrategy.OVERWRITE)
    return callback


@pytest.fixture
def orchestrator_with_callbacks(
    sample_config: ObfuscationConfig,
    mock_progress_callback: MagicMock,
    mock_error_callback: MagicMock,
) -> dict[str, Any]:
    """Return configured orchestrator with all callbacks attached."""
    orchestrator = ObfuscationOrchestrator(config=sample_config)
    return {
        "orchestrator": orchestrator,
        "progress_callback": mock_progress_callback,
        "error_callback": mock_error_callback,
    }
