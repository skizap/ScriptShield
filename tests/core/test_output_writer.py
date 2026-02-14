"""Comprehensive tests for the OutputWriter class.

Covers:
- Initialization and metadata setup
- Atomic and direct write flows
- Conflict resolution strategies
- Directory structure preservation
- Runtime library writing
- Summary report generation
- Permission validation
- Metadata management
- Cross-platform path handling
- Batch operations
- Edge cases
- Integration with RuntimeManager
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import tempfile
import time
import textwrap
from dataclasses import FrozenInstanceError
from pathlib import Path, PureWindowsPath
from unittest.mock import MagicMock, patch

import pytest

from obfuscator.core.config import ObfuscationConfig
from obfuscator.core.orchestrator import ConflictStrategy
from obfuscator.core.output_writer import OutputWriter, WriteMetadata, WriteResult
from obfuscator.core.runtime_manager import RuntimeManager
from obfuscator.utils.path_utils import normalize_path as real_normalize_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def output_writer_basic(tmp_path: Path) -> OutputWriter:
    """Create OutputWriter with OVERWRITE strategy."""
    return OutputWriter(
        output_dir=tmp_path / "output",
        conflict_strategy=ConflictStrategy.OVERWRITE,
    )


@pytest.fixture
def output_writer_rename(tmp_path: Path) -> OutputWriter:
    """Create OutputWriter with RENAME strategy."""
    return OutputWriter(
        output_dir=tmp_path / "output",
        conflict_strategy=ConflictStrategy.RENAME,
    )


@pytest.fixture
def output_writer_skip(tmp_path: Path) -> OutputWriter:
    """Create OutputWriter with SKIP strategy."""
    return OutputWriter(
        output_dir=tmp_path / "output",
        conflict_strategy=ConflictStrategy.SKIP,
    )


@pytest.fixture
def mock_runtime_manager() -> MagicMock:
    """Create RuntimeManager mock with configurable runtime responses."""
    manager = MagicMock(spec=RuntimeManager)
    manager.has_runtime_requirements.return_value = True
    manager.collect_required_runtimes.return_value = ["vm_protection"]
    manager.get_combined_runtime.return_value = (
        "# combined runtime code\nRUNTIME_READY = True\n"
    )
    return manager


def _default_options() -> dict[str, object]:
    """Return options used by test ObfuscationConfig fixtures."""
    return {
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
        "opaque_predicate_complexity": 2,
        "opaque_predicate_percentage": 30,
        "anti_debug_aggressiveness": 2,
        "code_split_chunk_size": 5,
        "code_split_encryption": True,
        "self_modify_complexity": 2,
        "roblox_exploit_aggressiveness": 2,
        "roblox_exploit_action": "exit",
    }


@pytest.fixture
def sample_python_config() -> ObfuscationConfig:
    """Create Python ObfuscationConfig for hybrid runtime library tests."""
    return ObfuscationConfig(
        name="python_runtime_test",
        language="python",
        features={
            "vm_protection": True,
            "code_splitting": True,
            "anti_debugging": True,
        },
        options=_default_options(),
        runtime_mode="hybrid",
        conflict_strategy="overwrite",
    )


@pytest.fixture
def sample_lua_config() -> ObfuscationConfig:
    """Create Lua ObfuscationConfig for hybrid runtime library tests."""
    return ObfuscationConfig(
        name="lua_runtime_test",
        language="lua",
        features={
            "vm_protection": True,
            "anti_debugging": True,
            "roblox_exploit_defense": True,
            "roblox_remote_spy": True,
        },
        options=_default_options(),
        runtime_mode="hybrid",
        conflict_strategy="overwrite",
    )


@pytest.fixture
def conflict_callback_mock() -> MagicMock:
    """Create conflict callback mock returning OVERWRITE for ASK strategy."""
    return MagicMock(return_value=ConflictStrategy.OVERWRITE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_test_file(
    directory: Path,
    name: str,
    content: str = "",
) -> Path:
    """Create a test file with sensible default content by extension."""
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


def create_readonly_file(
    directory: Path,
    name: str,
    content: str = "x = 1\n",
) -> Path:
    """Create a file and remove permissions to emulate read-only behavior."""
    file_path = create_test_file(directory, name, content)
    os.chmod(file_path, 0o000)
    return file_path

def _write_ok(
    writer: OutputWriter,
    path: Path,
    content: str = "x = 1\n",
) -> WriteResult:
    """Write a valid test file and assert success for setup convenience."""
    result = writer.write_file(path, content)
    assert result.success, f"Setup write failed for {path}: {result.error}"
    return result


def _make_python_runtime_manager(config: ObfuscationConfig) -> RuntimeManager:
    """Create RuntimeManager helper for readability in integration tests."""
    return RuntimeManager(config)


# ---------------------------------------------------------------------------
# Test OutputWriter Basics
# ---------------------------------------------------------------------------

class TestOutputWriterBasics:
    """Initialization and validation tests for OutputWriter."""

    def test_initialization_creates_output_dir(self, tmp_path: Path):
        """Verify write operations create the output directory tree when needed."""
        output_dir = tmp_path / "nested" / "out"
        writer = OutputWriter(
            output_dir=output_dir,
            conflict_strategy=ConflictStrategy.OVERWRITE,
        )

        assert writer.output_dir == output_dir.resolve()
        assert writer.output_dir.parent.exists(), (
            "Expected parent directory to be created"
        )

        result = writer.write_file(writer.output_dir / "main.py", "x = 1\n")
        assert result.success is True
        assert writer.output_dir.exists(), "Expected output directory to exist"

    def test_initialization_normalizes_path(self, tmp_path: Path):
        """Verify normalize_path() is used during initialization."""
        raw_output_dir = tmp_path / "./folder" / ".." / "output"

        with patch(
            "obfuscator.core.output_writer.normalize_path",
            wraps=real_normalize_path,
        ) as mock_normalize:
            writer = OutputWriter(
                output_dir=raw_output_dir,
                conflict_strategy=ConflictStrategy.OVERWRITE,
            )

        assert mock_normalize.called, (
            "normalize_path should be called during initialization"
        )
        assert writer.output_dir.is_absolute(), (
            "Normalized output_dir should be absolute"
        )

    def test_initialization_sets_attributes(self, tmp_path: Path):
        """Verify output_dir, strategy, and atomic-write flag are stored."""
        writer = OutputWriter(
            output_dir=tmp_path / "out",
            conflict_strategy=ConflictStrategy.RENAME,
            use_atomic_writes=False,
        )

        assert writer.conflict_strategy == ConflictStrategy.RENAME
        assert writer.use_atomic_writes is False
        assert writer.output_dir == (tmp_path / "out").resolve()

    def test_initialization_creates_metadata(self, output_writer_basic: OutputWriter):
        """Verify WriteMetadata object is created with start_time set."""
        metadata = output_writer_basic.get_metadata()

        assert isinstance(metadata, WriteMetadata)
        assert metadata.start_time is not None

    def test_initialization_with_nonexistent_parent_creates_it(
        self,
        tmp_path: Path,
    ):
        """Verify non-existent parent directories are created."""
        output_dir = tmp_path / "deep" / "nested" / "output"
        assert not output_dir.parent.exists()

        OutputWriter(
            output_dir=output_dir,
            conflict_strategy=ConflictStrategy.OVERWRITE,
        )

        assert output_dir.parent.exists(), "Parent directory should be created"

    def test_initialization_with_conflict_callback(
        self,
        tmp_path: Path,
        conflict_callback_mock: MagicMock,
    ):
        """Verify provided conflict callback is stored on writer."""
        writer = OutputWriter(
            output_dir=tmp_path / "out",
            conflict_strategy=ConflictStrategy.ASK,
            conflict_callback=conflict_callback_mock,
        )

        assert writer._conflict_callback is conflict_callback_mock

    def test_metadata_start_time_set_on_init(self, output_writer_basic: OutputWriter):
        """Verify metadata.start_time is a valid UNIX timestamp."""
        now = time.time()
        start_time = output_writer_basic.get_metadata().start_time

        assert start_time is not None
        assert 0 < start_time <= now


# ---------------------------------------------------------------------------
# Test Atomic File Operations
# ---------------------------------------------------------------------------

class TestAtomicFileOperations:
    """Atomic write path and fallback behavior tests."""

    def test_write_file_creates_temp_file_first(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify atomic write creates a temporary file before final move."""
        target = tmp_path / "atomic" / "file.py"

        fake_fd = MagicMock()
        fake_fd.name = str(Path(tempfile.gettempdir()) / "tmp_atomic_file.py")
        fake_fd.fileno.return_value = 11

        with (
            patch(
                "obfuscator.core.output_writer.tempfile.NamedTemporaryFile",
                return_value=fake_fd,
            ) as mock_temp,
            patch("obfuscator.core.output_writer.os.fsync") as mock_fsync,
            patch("obfuscator.core.output_writer.shutil.move") as mock_move,
        ):
            ok = output_writer_basic._write_atomic(target, "print('ok')\n")

        assert ok is True
        assert mock_temp.called, "NamedTemporaryFile should be called"
        assert fake_fd.write.called, "Content should be written to temp file"
        mock_fsync.assert_called_once_with(11)
        mock_move.assert_called_once_with(fake_fd.name, str(target))

    def test_write_file_atomic_rename(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify write_file uses atomic path and final output exists."""
        target = tmp_path / "atomic_rename.py"

        with patch(
            "obfuscator.core.output_writer.shutil.move",
            wraps=shutil.move,
        ) as mock_move:
            result = output_writer_basic.write_file(target, "x = 42\n")

        assert result.success is True
        assert result.was_atomic is True
        assert mock_move.called, "Expected atomic move operation"
        assert target.read_text(encoding="utf-8") == "x = 42\n"

    def test_write_file_atomic_success(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify successful atomic write stores exact content."""
        target = tmp_path / "atomic_success.py"
        content = "def f():\n    return 'atomic'\n"

        result = output_writer_basic.write_file(target, content)

        assert result.success is True
        assert target.exists()
        assert target.read_text(encoding="utf-8") == content

    def test_write_file_atomic_flag_in_result(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify WriteResult.was_atomic is True on successful atomic writes."""
        result = output_writer_basic.write_file(tmp_path / "flag_test.py", "x = 1\n")

        assert result.success is True
        assert result.was_atomic is True

    def test_write_file_fallback_to_direct_write(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify atomic failure falls back to direct write and still succeeds."""
        target = tmp_path / "fallback.py"

        with (
            patch.object(
                output_writer_basic,
                "_write_atomic",
                side_effect=OSError("atomic failed"),
            ),
            patch.object(
                output_writer_basic,
                "_write_direct",
                return_value=True,
            ) as mock_direct,
        ):
            result = output_writer_basic.write_file(target, "x = 2\n")

        assert result.success is True
        assert result.was_atomic is False
        mock_direct.assert_called_once()

    def test_write_file_direct_write_when_atomic_disabled(self, tmp_path: Path):
        """Verify direct write path is used when atomic writes are disabled."""
        writer = OutputWriter(
            output_dir=tmp_path / "out",
            conflict_strategy=ConflictStrategy.OVERWRITE,
            use_atomic_writes=False,
        )
        target = tmp_path / "direct_only.py"

        with (
            patch.object(writer, "_write_atomic") as mock_atomic,
            patch.object(writer, "_write_direct", return_value=True) as mock_direct,
        ):
            result = writer.write_file(target, "x = 3\n")

        assert result.success is True
        assert result.was_atomic is False
        mock_atomic.assert_not_called()
        mock_direct.assert_called_once()

    def test_atomic_write_rollback_on_failure(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify temp file cleanup is attempted when atomic write fails."""
        target = tmp_path / "rollback.py"
        temp_path = str(tmp_path / "rollback_tmp.py")

        class FailingTempFile:
            name = temp_path

            def write(self, _content: str) -> None:
                raise OSError("forced write failure")

            def flush(self) -> None:
                return None

            def fileno(self) -> int:
                return 0

            def close(self) -> None:
                return None

        with (
            patch(
                "obfuscator.core.output_writer.tempfile.NamedTemporaryFile",
                return_value=FailingTempFile(),
            ),
            patch("obfuscator.core.output_writer.os.unlink") as mock_unlink,
        ):
            with pytest.raises(OSError, match="forced write failure"):
                output_writer_basic._write_atomic(target, "content")

        mock_unlink.assert_called_once_with(temp_path)

    def test_atomic_write_preserves_content_encoding(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify UTF-8 content with non-ASCII characters is preserved."""
        target = tmp_path / "encoding.py"
        content = "message = 'héllo 🌍 – こんにちは – مرحبا'\n"

        result = output_writer_basic.write_file(target, content)

        assert result.success is True
        assert target.read_text(encoding="utf-8") == content

    def test_write_file_updates_metadata_counters(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify total_writes and successful_writes increment."""
        output_writer_basic.write_file(tmp_path / "meta_counter.py", "x = 99\n")
        metadata = output_writer_basic.get_metadata()

        assert metadata.total_writes == 1
        assert metadata.successful_writes == 1
        assert metadata.failed_writes == 0


# ---------------------------------------------------------------------------
# Test Conflict Resolution
# ---------------------------------------------------------------------------

class TestConflictResolution:
    """Tests for OVERWRITE, SKIP, RENAME, and ASK conflict behavior."""

    def test_overwrite_strategy_replaces_existing_file(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify OVERWRITE replaces existing file content."""
        target = create_test_file(tmp_path, "overwrite.py", "old = True\n")
        result = output_writer_basic.write_file(target, "old = False\n")

        assert result.success is True
        assert target.read_text(encoding="utf-8") == "old = False\n"

    def test_overwrite_strategy_sets_conflict_resolution(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify OVERWRITE records conflict_resolution='overwritten'."""
        target = create_test_file(tmp_path, "overwrite_resolution.py", "x = 1\n")
        result = output_writer_basic.write_file(target, "x = 2\n")

        assert result.conflict_resolution == "overwritten"

    def test_overwrite_metadata_counter(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify overwritten_writes counter increments."""
        target = create_test_file(tmp_path, "overwrite_meta.py", "x = 1\n")
        output_writer_basic.write_file(target, "x = 2\n")

        assert output_writer_basic.get_metadata().overwritten_writes == 1

    def test_skip_strategy_leaves_existing_file(
        self,
        output_writer_skip: OutputWriter,
        tmp_path: Path,
    ):
        """Verify SKIP leaves existing file unchanged."""
        target = create_test_file(tmp_path, "skip.py", "original = True\n")
        result = output_writer_skip.write_file(target, "original = False\n")

        assert result.success is True
        assert target.read_text(encoding="utf-8") == "original = True\n"

    def test_skip_strategy_returns_none_output_path(
        self,
        output_writer_skip: OutputWriter,
        tmp_path: Path,
    ):
        """Verify SKIP returns WriteResult with output_path=None."""
        target = create_test_file(tmp_path, "skip_none.py", "x = 1\n")
        result = output_writer_skip.write_file(target, "x = 2\n")

        assert result.success is True
        assert result.output_path is None

    def test_skip_strategy_sets_conflict_resolution(
        self,
        output_writer_skip: OutputWriter,
        tmp_path: Path,
    ):
        """Verify SKIP records conflict_resolution='skipped'."""
        target = create_test_file(tmp_path, "skip_resolution.py", "x = 1\n")
        result = output_writer_skip.write_file(target, "x = 2\n")

        assert result.conflict_resolution == "skipped"

    def test_skip_metadata_counter(
        self,
        output_writer_skip: OutputWriter,
        tmp_path: Path,
    ):
        """Verify skipped_writes counter increments."""
        target = create_test_file(tmp_path, "skip_meta.py", "x = 1\n")
        output_writer_skip.write_file(target, "x = 2\n")

        assert output_writer_skip.get_metadata().skipped_writes == 1

    def test_rename_strategy_creates_timestamped_file(
        self,
        output_writer_rename: OutputWriter,
        tmp_path: Path,
    ):
        """Verify RENAME creates filename with YYYYMMDD_HHMMSS timestamp."""
        target = create_test_file(tmp_path, "rename.py", "x = 1\n")
        result = output_writer_rename.write_file(target, "x = 2\n")

        assert result.success is True
        assert result.output_path is not None
        assert result.output_path != target
        assert re.search(r"rename_\d{8}_\d{6}\.py$", result.output_path.name)

    @pytest.mark.parametrize("filename", ["module.py", "script.lua"])
    def test_rename_strategy_preserves_extension(
        self,
        output_writer_rename: OutputWriter,
        tmp_path: Path,
        filename: str,
    ):
        """Verify RENAME keeps the original file extension."""
        target = create_test_file(tmp_path, filename, "x = 1\n")
        result = output_writer_rename.write_file(target, "x = 2\n")

        assert result.output_path is not None
        assert result.output_path.suffix == target.suffix

    def test_rename_strategy_handles_collision(
        self,
        output_writer_rename: OutputWriter,
        tmp_path: Path,
    ):
        """Verify RENAME adds counter suffix when timestamped name exists."""
        target = create_test_file(tmp_path, "collision.py", "x = 1\n")
        fixed_timestamp = "20260214_133500"
        create_test_file(
            tmp_path,
            f"collision_{fixed_timestamp}.py",
            "existing = True\n",
        )

        with patch("obfuscator.core.output_writer.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = fixed_timestamp
            result = output_writer_rename.write_file(target, "x = 2\n")

        assert result.output_path is not None
        assert result.output_path.name == f"collision_{fixed_timestamp}_1.py"

    def test_rename_strategy_sets_conflict_resolution(
        self,
        output_writer_rename: OutputWriter,
        tmp_path: Path,
    ):
        """Verify RENAME records conflict_resolution='renamed'."""
        target = create_test_file(tmp_path, "rename_resolution.py", "x = 1\n")
        result = output_writer_rename.write_file(target, "x = 2\n")

        assert result.conflict_resolution == "renamed"

    def test_rename_metadata_counter(
        self,
        output_writer_rename: OutputWriter,
        tmp_path: Path,
    ):
        """Verify renamed_writes counter increments."""
        target = create_test_file(tmp_path, "rename_meta.py", "x = 1\n")
        output_writer_rename.write_file(target, "x = 2\n")

        assert output_writer_rename.get_metadata().renamed_writes == 1

    def test_ask_strategy_with_callback(
        self,
        tmp_path: Path,
        conflict_callback_mock: MagicMock,
    ):
        """Verify ASK strategy invokes callback with conflicting path."""
        writer = OutputWriter(
            output_dir=tmp_path / "out",
            conflict_strategy=ConflictStrategy.ASK,
            conflict_callback=conflict_callback_mock,
        )
        target = create_test_file(tmp_path, "ask_callback.py", "x = 1\n")

        result = writer.write_file(target, "x = 2\n")

        assert result.success is True
        conflict_callback_mock.assert_called_once_with(target.resolve())

    def test_ask_strategy_callback_decision_applied(self, tmp_path: Path):
        """Verify callback decision (SKIP) is applied by ASK strategy."""
        callback = MagicMock(return_value=ConflictStrategy.SKIP)
        writer = OutputWriter(
            output_dir=tmp_path / "out",
            conflict_strategy=ConflictStrategy.ASK,
            conflict_callback=callback,
        )
        target = create_test_file(tmp_path, "ask_skip.py", "x = 1\n")

        result = writer.write_file(target, "x = 2\n")

        assert result.success is True
        assert result.output_path is None
        assert result.conflict_resolution == "skipped"

    def test_ask_strategy_caches_decisions(self, tmp_path: Path):
        """Verify ASK strategy caches decisions for repeated conflicts."""
        callback = MagicMock(return_value=ConflictStrategy.OVERWRITE)
        writer = OutputWriter(
            output_dir=tmp_path / "out",
            conflict_strategy=ConflictStrategy.ASK,
            conflict_callback=callback,
        )
        target = create_test_file(tmp_path, "ask_cached.py", "x = 1\n")

        first = writer.write_file(target, "x = 2\n")
        second = writer.write_file(target, "x = 3\n")

        assert first.success and second.success
        assert callback.call_count == 1

    def test_ask_strategy_without_callback_raises_error(self, tmp_path: Path):
        """Verify ASK conflict resolver raises ValueError without callback."""
        writer = OutputWriter(
            output_dir=tmp_path / "out",
            conflict_strategy=ConflictStrategy.ASK,
        )
        target = create_test_file(tmp_path, "ask_no_callback.py", "x = 1\n")

        with pytest.raises(ValueError, match="requires a conflict_callback"):
            writer._resolve_conflict(target)

    def test_no_conflict_when_file_not_exists(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify conflict_resolution is None when writing a new file."""
        target = tmp_path / "no_conflict.py"
        result = output_writer_basic.write_file(target, "x = 1\n")

        assert result.success is True
        assert result.conflict_resolution is None


# ---------------------------------------------------------------------------
# Test Directory Structure Preservation
# ---------------------------------------------------------------------------

class TestDirectoryStructurePreservation:
    """Tests for write_with_structure() behavior."""

    def test_write_with_structure_preserves_relative_path(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify relative path under project_root is preserved in output_base."""
        project_root = tmp_path / "src"
        input_path = create_test_file(project_root, "pkg/module.py", "x = 1\n")
        output_base = tmp_path / "output_base"

        result = output_writer_basic.write_with_structure(
            input_path=input_path,
            output_base=output_base,
            content="x = 2\n",
            project_root=project_root,
        )

        expected = (output_base / "pkg/module.py").resolve()
        assert result.success is True
        assert result.output_path == expected
        assert expected.read_text(encoding="utf-8") == "x = 2\n"

    def test_write_with_structure_without_project_root(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify flat output structure when project_root is not provided."""
        input_path = create_test_file(tmp_path / "src" / "pkg", "module.py", "x = 1\n")
        output_base = tmp_path / "flat_out"

        result = output_writer_basic.write_with_structure(
            input_path=input_path,
            output_base=output_base,
            content="x = 2\n",
            project_root=None,
        )

        expected = (output_base / "module.py").resolve()
        assert result.success is True
        assert result.output_path == expected

    def test_write_with_structure_creates_nested_directories(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify intermediate nested directories are created automatically."""
        project_root = tmp_path / "project"
        input_path = create_test_file(project_root, "a/b/c/deep.py", "x = 1\n")
        output_base = tmp_path / "build"

        result = output_writer_basic.write_with_structure(
            input_path=input_path,
            output_base=output_base,
            content="x = 2\n",
            project_root=project_root,
        )

        assert result.success is True
        assert result.output_path is not None
        assert result.output_path.parent.exists()

    def test_write_with_structure_handles_different_drives_windows(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify ValueError from relative path computation falls back to flat path."""
        input_path = create_test_file(tmp_path / "src", "module.py", "x = 1\n")
        output_base = tmp_path / "out"

        with (
            patch("platform.system", return_value="Windows"),
            patch(
                "obfuscator.core.output_writer.get_relative_path",
                side_effect=ValueError("different drives"),
            ),
        ):
            result = output_writer_basic.write_with_structure(
                input_path=input_path,
                output_base=output_base,
                content="x = 2\n",
                project_root=tmp_path / "src",
            )

        expected = (output_base / "module.py").resolve()
        assert result.success is True
        assert result.output_path == expected

    def test_write_with_structure_handles_relative_path_error(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify generic relative-path errors trigger flat output fallback."""
        input_path = create_test_file(tmp_path / "src", "relative_error.py", "x = 1\n")
        output_base = tmp_path / "out"

        with patch(
            "obfuscator.core.output_writer.get_relative_path",
            side_effect=ValueError("cannot compute"),
        ):
            result = output_writer_basic.write_with_structure(
                input_path=input_path,
                output_base=output_base,
                content="x = 2\n",
                project_root=tmp_path / "src",
            )

        assert result.success is True
        assert result.output_path == (output_base / "relative_error.py").resolve()

    def test_write_with_structure_returns_write_result(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify write_with_structure returns a WriteResult object."""
        input_path = create_test_file(tmp_path / "src", "typed.py", "x = 1\n")
        output_base = tmp_path / "out"
        result = output_writer_basic.write_with_structure(
            input_path=input_path,
            output_base=output_base,
            content="x = 2\n",
            project_root=tmp_path / "src",
        )

        assert isinstance(result, WriteResult)
        assert result.original_path == (output_base / "typed.py").resolve()

    def test_write_with_structure_with_conflict(
        self,
        output_writer_skip: OutputWriter,
        tmp_path: Path,
    ):
        """Verify conflict strategy applies for structured-output collisions."""
        project_root = tmp_path / "src"
        input_path = create_test_file(project_root, "pkg/conflict.py", "x = 1\n")
        output_base = tmp_path / "out"
        existing = create_test_file(
            output_base / "pkg",
            "conflict.py",
            "existing = True\n",
        )

        result = output_writer_skip.write_with_structure(
            input_path=input_path,
            output_base=output_base,
            content="x = 2\n",
            project_root=project_root,
        )

        assert result.success is True
        assert result.output_path is None
        assert result.conflict_resolution == "skipped"
        assert existing.read_text(encoding="utf-8") == "existing = True\n"


# ---------------------------------------------------------------------------
# Test Runtime Library Writing
# ---------------------------------------------------------------------------

class TestRuntimeLibraryWriting:
    """Tests for write_runtime_library() behavior."""

    def test_write_runtime_library_python_hybrid_mode(
        self,
        output_writer_basic: OutputWriter,
        mock_runtime_manager: MagicMock,
    ):
        """Verify Python hybrid runtime writes obf_runtime.py."""
        result = output_writer_basic.write_runtime_library(
            runtime_manager=mock_runtime_manager,
            language="python",
            runtime_mode="hybrid",
        )

        assert result.success is True
        assert result.output_path is not None
        assert result.output_path.name == "obf_runtime.py"
        assert result.output_path.exists()

    def test_write_runtime_library_lua_hybrid_mode(
        self,
        output_writer_basic: OutputWriter,
        mock_runtime_manager: MagicMock,
    ):
        """Verify Lua hybrid runtime writes obf_runtime.lua."""
        result = output_writer_basic.write_runtime_library(
            runtime_manager=mock_runtime_manager,
            language="lua",
            runtime_mode="hybrid",
        )

        assert result.success is True
        assert result.output_path is not None
        assert result.output_path.name == "obf_runtime.lua"
        assert result.output_path.exists()

    def test_write_runtime_library_calls_get_combined_runtime(
        self,
        output_writer_basic: OutputWriter,
        mock_runtime_manager: MagicMock,
    ):
        """Verify RuntimeManager.get_combined_runtime called with target language."""
        output_writer_basic.write_runtime_library(mock_runtime_manager, "python", "hybrid")
        mock_runtime_manager.get_combined_runtime.assert_called_once_with("python")

    def test_write_runtime_library_embedded_mode_fails(
        self,
        output_writer_basic: OutputWriter,
        mock_runtime_manager: MagicMock,
    ):
        """Verify embedded runtime mode fails and writes no runtime file."""
        result = output_writer_basic.write_runtime_library(
            mock_runtime_manager,
            "python",
            "embedded",
        )

        assert result.success is False
        assert result.output_path is None
        assert "embedded mode" in (result.error or "")

    def test_write_runtime_library_invalid_mode_fails(
        self,
        output_writer_basic: OutputWriter,
        mock_runtime_manager: MagicMock,
    ):
        """Verify unsupported runtime mode fails with descriptive message."""
        result = output_writer_basic.write_runtime_library(
            mock_runtime_manager,
            "python",
            "invalid",
        )

        assert result.success is False
        assert result.output_path is None
        assert "received runtime mode 'invalid'" in (result.error or "")

    def test_write_runtime_library_invalid_language_fails(
        self,
        output_writer_basic: OutputWriter,
        mock_runtime_manager: MagicMock,
    ):
        """Verify invalid language returns failed WriteResult."""
        result = output_writer_basic.write_runtime_library(
            mock_runtime_manager,
            "invalid",
            "hybrid",
        )

        assert result.success is False
        assert "Invalid language" in (result.error or "")

    def test_write_runtime_library_no_requirements_skips(
        self,
        output_writer_basic: OutputWriter,
        mock_runtime_manager: MagicMock,
    ):
        """Verify no runtime requirements leads to a skipped runtime write."""
        mock_runtime_manager.has_runtime_requirements.return_value = False

        result = output_writer_basic.write_runtime_library(
            mock_runtime_manager,
            "python",
            "hybrid",
        )

        assert result.success is True
        assert result.output_path is None
        assert output_writer_basic.get_metadata().skipped_writes >= 1

    def test_write_runtime_library_empty_code_skips(
        self,
        output_writer_basic: OutputWriter,
        mock_runtime_manager: MagicMock,
    ):
        """Verify empty generated runtime code causes skip with warning."""
        mock_runtime_manager.get_combined_runtime.return_value = "   \n"

        result = output_writer_basic.write_runtime_library(
            mock_runtime_manager,
            "python",
            "hybrid",
        )

        assert result.success is True
        assert result.output_path is None
        assert any(
            "empty" in warning.lower()
            for warning in output_writer_basic.get_metadata().warnings
        )

    def test_write_runtime_library_updates_metadata(
        self,
        output_writer_basic: OutputWriter,
        mock_runtime_manager: MagicMock,
    ):
        """Verify runtime library counters and paths are updated in metadata."""
        result = output_writer_basic.write_runtime_library(
            mock_runtime_manager,
            "python",
            "hybrid",
        )
        metadata = output_writer_basic.get_metadata()

        assert result.success is True
        assert metadata.runtime_libraries_written == 1
        assert len(metadata.runtime_library_paths) == 1
        assert metadata.runtime_library_paths[0].name == "obf_runtime.py"

    def test_write_runtime_library_with_conflict_resolution(
        self,
        output_writer_rename: OutputWriter,
        mock_runtime_manager: MagicMock,
    ):
        """Verify runtime write respects conflict strategy on existing files."""
        target = output_writer_rename.output_dir / "obf_runtime.py"
        create_test_file(target.parent, target.name, "old runtime\n")

        result = output_writer_rename.write_runtime_library(
            mock_runtime_manager,
            "python",
            "hybrid",
        )

        assert result.success is True
        assert result.conflict_resolution == "renamed"
        assert result.output_path is not None
        assert result.output_path.name != "obf_runtime.py"

    def test_write_runtime_library_custom_output_dir(
        self,
        output_writer_basic: OutputWriter,
        mock_runtime_manager: MagicMock,
        tmp_path: Path,
    ):
        """Verify output_dir override writes runtime into custom directory."""
        custom_dir = tmp_path / "custom_runtime"

        result = output_writer_basic.write_runtime_library(
            runtime_manager=mock_runtime_manager,
            language="python",
            runtime_mode="hybrid",
            output_dir=custom_dir,
        )

        assert result.success is True
        assert result.output_path is not None
        assert result.output_path.parent == custom_dir.resolve()

    def test_write_runtime_library_exception_handling(
        self,
        output_writer_basic: OutputWriter,
        mock_runtime_manager: MagicMock,
    ):
        """Verify unexpected runtime generation exceptions are captured."""
        mock_runtime_manager.get_combined_runtime.side_effect = RuntimeError(
            "runtime explosion"
        )

        result = output_writer_basic.write_runtime_library(
            mock_runtime_manager,
            "python",
            "hybrid",
        )

        assert result.success is False
        assert "runtime explosion" in (result.error or "")


# ---------------------------------------------------------------------------
# Test Summary Report Generation
# ---------------------------------------------------------------------------

class TestSummaryReportGeneration:
    """Tests for summary report content generation and writing."""

    def test_generate_summary_report_text_format(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify text summary contains expected section headers."""
        _write_ok(output_writer_basic, tmp_path / "a.py")
        report = output_writer_basic.generate_summary_report(format="text")

        assert "=== Obfuscation Output Summary ===" in report
        assert "File Statistics:" in report
        assert "Runtime Libraries:" in report

    def test_generate_summary_report_json_format(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify JSON summary contains expected top-level keys."""
        _write_ok(output_writer_basic, tmp_path / "a.py")
        payload = json.loads(output_writer_basic.generate_summary_report(format="json"))

        assert "summary" in payload
        assert "written_files" in payload
        assert "failed_files" in payload

    def test_summary_report_includes_elapsed_time(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify summary includes elapsed_seconds and formatted_elapsed."""
        _write_ok(output_writer_basic, tmp_path / "a.py")
        summary = json.loads(output_writer_basic.generate_summary_report(format="json"))["summary"]

        assert "formatted_elapsed" in summary
        assert "elapsed_seconds" in summary

    def test_summary_report_includes_file_statistics(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify all expected file counters are in summary payload."""
        _write_ok(output_writer_basic, tmp_path / "a.py")
        output_writer_basic.write_file(tmp_path / "b.txt", "invalid")
        summary = json.loads(output_writer_basic.generate_summary_report(format="json"))["summary"]

        assert "total_files" in summary
        assert "successful_writes" in summary
        assert "failed_writes" in summary
        assert "skipped_writes" in summary
        assert "renamed_writes" in summary
        assert "overwritten_writes" in summary

    def test_summary_report_includes_runtime_libraries(
        self,
        output_writer_basic: OutputWriter,
        mock_runtime_manager: MagicMock,
    ):
        """Verify runtime libraries appear with language labels in JSON report."""
        output_writer_basic.write_runtime_library(mock_runtime_manager, "python", "hybrid")
        payload = json.loads(output_writer_basic.generate_summary_report(format="json"))

        assert payload["runtime_libraries"], "Expected runtime library entries"
        assert payload["runtime_libraries"][0]["language"] == "python"

    def test_summary_report_includes_warnings(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify warnings are included after a warning-producing write."""
        output_writer_basic.write_file(tmp_path / "unsupported.txt", "hello")
        payload = json.loads(output_writer_basic.generate_summary_report(format="json"))

        assert payload["warnings"], "Expected warnings list to be populated"

    def test_summary_report_includes_written_files_details(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify written_files entries include path/conflict/atomic details."""
        result = _write_ok(output_writer_basic, tmp_path / "detail.py")
        payload = json.loads(output_writer_basic.generate_summary_report(format="json"))

        assert payload["written_files"], "Expected written file details"
        first = payload["written_files"][0]
        assert first["path"] == str(result.output_path)
        assert "conflict_resolution" in first
        assert "was_atomic" in first

    def test_summary_report_includes_failed_files_details(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify failed_files entries include error details."""
        output_writer_basic.write_file(tmp_path / "bad.ext", "oops")
        payload = json.loads(output_writer_basic.generate_summary_report(format="json"))

        assert payload["failed_files"], "Expected failed file details"
        assert "error" in payload["failed_files"][0]

    def test_summary_report_without_file_details(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify include_file_details=False omits per-file entries."""
        _write_ok(output_writer_basic, tmp_path / "a.py")
        output_writer_basic.write_file(tmp_path / "b.ext", "x")
        payload = json.loads(
            output_writer_basic.generate_summary_report(
                format="json",
                include_file_details=False,
            )
        )

        assert payload["written_files"] == []
        assert payload["failed_files"] == []

    def test_summary_report_sets_end_time(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify metadata.end_time is set during report generation."""
        _write_ok(output_writer_basic, tmp_path / "a.py")
        assert output_writer_basic.get_metadata().end_time is None

        output_writer_basic.generate_summary_report(format="text")
        assert output_writer_basic.get_metadata().end_time is not None

    def test_summary_report_invalid_format_raises_error(
        self,
        output_writer_basic: OutputWriter,
    ):
        """Verify invalid report format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid summary report format"):
            output_writer_basic.generate_summary_report(format="invalid")

    def test_write_summary_report_to_file(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify write_summary_report() writes report content to disk."""
        _write_ok(output_writer_basic, tmp_path / "a.py")
        output_path = tmp_path / "report.txt"

        result = output_writer_basic.write_summary_report(
            output_path=output_path,
            format="text",
        )

        assert result.success is True
        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8")

    def test_write_summary_report_default_filename(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify default summary report names for text and JSON formats."""
        _write_ok(output_writer_basic, tmp_path / "a.py")

        text_result = output_writer_basic.write_summary_report(format="text")
        json_result = output_writer_basic.write_summary_report(format="json")

        assert text_result.success is True
        assert json_result.success is True
        assert text_result.output_path is not None
        assert json_result.output_path is not None
        assert text_result.output_path.name == "obfuscation_summary.txt"
        assert json_result.output_path.name == "obfuscation_summary.json"

    def test_write_summary_report_custom_path(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify custom summary output path is honored."""
        _write_ok(output_writer_basic, tmp_path / "a.py")
        custom = tmp_path / "reports" / "custom_summary.json"

        result = output_writer_basic.write_summary_report(
            output_path=custom,
            format="json",
        )

        assert result.success is True
        assert custom.exists()
        assert result.output_path == custom

    def test_get_summary_report_returns_string(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify get_summary_report returns a string without writing a file."""
        _write_ok(output_writer_basic, tmp_path / "a.py")
        report = output_writer_basic.get_summary_report(format="text")

        assert isinstance(report, str)
        assert "Obfuscation Output Summary" in report


# ---------------------------------------------------------------------------
# Test Permission Validation
# ---------------------------------------------------------------------------

class TestPermissionValidation:
    """Permission-check and failure-path tests."""

    def test_validate_write_permissions_writable_file(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify writable file passes permission validation."""
        target = create_test_file(tmp_path, "writable.py", "x = 1\n")
        valid, error = output_writer_basic._validate_write_permissions(target)

        assert valid is True
        assert error is None

    def test_validate_write_permissions_readonly_file(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify read-only file fails permission validation."""
        target = create_readonly_file(tmp_path, "readonly.py", "x = 1\n")

        def _is_writable(path: Path) -> bool:
            return Path(path).resolve() != target.resolve()

        try:
            with patch("obfuscator.core.output_writer.is_writable", side_effect=_is_writable):
                valid, error = output_writer_basic._validate_write_permissions(target)
            assert valid is False
            assert error is not None
            assert "not writable" in error.lower()
        finally:
            os.chmod(target, 0o644)

    def test_validate_write_permissions_readonly_parent(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify non-existing target in read-only parent fails validation."""
        readonly_parent = tmp_path / "readonly_parent"
        readonly_parent.mkdir(parents=True)
        os.chmod(readonly_parent, 0o555)

        target = readonly_parent / "new.py"

        def _is_writable(path: Path) -> bool:
            return Path(path).resolve() != readonly_parent.resolve()

        try:
            with patch("obfuscator.core.output_writer.is_writable", side_effect=_is_writable):
                valid, error = output_writer_basic._validate_write_permissions(target)
            assert valid is False
            assert error is not None
            assert "parent directory is not writable" in error.lower()
        finally:
            os.chmod(readonly_parent, 0o755)

    def test_validate_write_permissions_nonexistent_file_writable_parent(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify new file in writable parent passes validation."""
        target = tmp_path / "writable_parent" / "new.py"
        target.parent.mkdir(parents=True)

        valid, error = output_writer_basic._validate_write_permissions(target)

        assert valid is True
        assert error is None

    def test_write_file_fails_on_permission_error(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify write_file fails when permission check reports non-writable path."""
        target = create_test_file(tmp_path, "permission_fail.py", "x = 1\n")

        with patch("obfuscator.core.output_writer.is_writable", return_value=False):
            result = output_writer_basic.write_file(target, "x = 2\n")

        assert result.success is False
        assert result.error is not None
        assert "not writable" in result.error.lower()

    def test_permission_error_adds_warning(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify permission failures append warnings to metadata."""
        target = create_test_file(tmp_path, "permission_warning.py", "x = 1\n")

        with patch("obfuscator.core.output_writer.is_writable", return_value=False):
            output_writer_basic.write_file(target, "x = 2\n")

        warnings = output_writer_basic.get_metadata().warnings
        assert any("not writable" in warning.lower() for warning in warnings)

    def test_permission_error_updates_failed_counter(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify failed_writes counter increments after permission error."""
        target = create_test_file(tmp_path, "permission_count.py", "x = 1\n")

        with patch("obfuscator.core.output_writer.is_writable", return_value=False):
            output_writer_basic.write_file(target, "x = 2\n")

        assert output_writer_basic.get_metadata().failed_writes == 1


# ---------------------------------------------------------------------------
# Test Metadata Management
# ---------------------------------------------------------------------------

class TestMetadataManagement:
    """Tests for metadata accessors and reset behavior."""

    def test_get_metadata_returns_copy(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify get_metadata returns a copy, not internal mutable state."""
        _write_ok(output_writer_basic, tmp_path / "copy_test.py")

        snapshot = output_writer_basic.get_metadata()
        snapshot.total_writes = 999
        snapshot.written_files.append(Path("/fake/path.py"))

        fresh = output_writer_basic.get_metadata()
        assert fresh.total_writes != 999
        assert Path("/fake/path.py") not in fresh.written_files

    def test_get_metadata_includes_all_fields(self, output_writer_basic: OutputWriter):
        """Verify metadata snapshot includes every WriteMetadata field."""
        metadata = output_writer_basic.get_metadata()
        required_fields = {
            "total_writes",
            "successful_writes",
            "failed_writes",
            "skipped_writes",
            "renamed_writes",
            "overwritten_writes",
            "written_files",
            "runtime_libraries_written",
            "runtime_library_paths",
            "start_time",
            "end_time",
            "warnings",
            "file_results",
        }

        for field_name in required_fields:
            assert hasattr(metadata, field_name), f"Missing metadata field: {field_name}"

    def test_reset_metadata_clears_counters(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify reset_metadata resets counters and lists."""
        _write_ok(output_writer_basic, tmp_path / "a.py")
        output_writer_basic.write_file(tmp_path / "b.txt", "invalid")
        assert output_writer_basic.get_metadata().total_writes > 0

        output_writer_basic.reset_metadata()
        metadata = output_writer_basic.get_metadata()

        assert metadata.total_writes == 0
        assert metadata.successful_writes == 0
        assert metadata.failed_writes == 0
        assert metadata.written_files == []
        assert metadata.file_results == []

    def test_metadata_written_files_list(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify written_files contains all successfully written paths."""
        first = _write_ok(output_writer_basic, tmp_path / "written_1.py")
        second = _write_ok(output_writer_basic, tmp_path / "written_2.py")
        metadata = output_writer_basic.get_metadata()

        assert first.output_path in metadata.written_files
        assert second.output_path in metadata.written_files
        assert len(metadata.written_files) == 2

    def test_metadata_file_results_list(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify file_results tracks every write attempt as WriteResult."""
        _write_ok(output_writer_basic, tmp_path / "ok.py")
        output_writer_basic.write_file(tmp_path / "bad.txt", "invalid")
        metadata = output_writer_basic.get_metadata()

        assert len(metadata.file_results) == 2
        assert all(isinstance(item, WriteResult) for item in metadata.file_results)

    def test_metadata_elapsed_seconds_property(self, output_writer_basic: OutputWriter):
        """Verify elapsed_seconds computes end-start correctly."""
        output_writer_basic._metadata.start_time = 100.0
        output_writer_basic._metadata.end_time = 112.5

        metadata = output_writer_basic.get_metadata()
        assert metadata.elapsed_seconds == pytest.approx(12.5)

    def test_metadata_formatted_elapsed_property(self, output_writer_basic: OutputWriter):
        """Verify formatted_elapsed renders MM:SS from elapsed seconds."""
        output_writer_basic._metadata.start_time = 1.0
        output_writer_basic._metadata.end_time = 66.0

        metadata = output_writer_basic.get_metadata()
        assert metadata.formatted_elapsed == "01:05"


# ---------------------------------------------------------------------------
# Test Cross-Platform Path Handling
# ---------------------------------------------------------------------------

class TestCrossPlatformPathHandling:
    """Cross-platform path compatibility tests."""

    def test_windows_path_separators(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify writer handles paths derived from Windows-style separators."""
        windows_rel = str(PureWindowsPath("nested") / "windows_file.py")
        target = tmp_path / windows_rel.replace("\\", os.sep)

        with patch("platform.system", return_value="Windows"):
            result = output_writer_basic.write_file(target, "x = 1\n")

        assert result.success is True
        assert target.resolve().exists()

    def test_linux_path_separators(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify forward-slash Linux-style paths are processed correctly."""
        with patch("platform.system", return_value="Linux"):
            target = tmp_path / "linux" / "nested" / "linux_file.py"
            result = output_writer_basic.write_file(target, "x = 1\n")

        assert result.success is True
        assert target.exists()

    def test_macos_path_handling(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify macOS-style path handling works for standard writes."""
        with patch("platform.system", return_value="Darwin"):
            target = tmp_path / "macos" / "module.py"
            result = output_writer_basic.write_file(target, "x = 1\n")

        assert result.success is True
        assert target.exists()

    def test_unicode_paths(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify writes succeed for file names containing Unicode characters."""
        target = tmp_path / "ユニコード_файл_📦.py"
        result = output_writer_basic.write_file(target, "x = 1\n")

        assert result.success is True
        assert target.exists()

    def test_long_paths_windows(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify long path writes can be handled in a simulated Windows context."""
        long_root = tmp_path
        while len(str(long_root / "long_path.py")) <= 270:
            long_root = long_root / ("segment_" + "x" * 20)
        target = long_root / "long_path.py"

        with patch("platform.system", return_value="Windows"):
            result = output_writer_basic.write_file(target, "x = 1\n")

        assert len(str(target)) > 260
        assert result.success is True
        assert target.exists()

    def test_path_normalization_across_platforms(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify normalize_path is called and path resolves consistently."""
        messy_path = Path(str(tmp_path / "norm" / ".." / "normalized.py"))

        with patch(
            "obfuscator.core.output_writer.normalize_path",
            wraps=real_normalize_path,
        ) as mock_normalize:
            result = output_writer_basic.write_file(messy_path, "x = 1\n")

        assert result.success is True
        assert mock_normalize.called
        assert result.output_path == real_normalize_path(messy_path)

    def test_relative_path_resolution(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify relative path resolution under project roots behaves correctly."""
        project_root = tmp_path / "project_root"
        input_path = create_test_file(project_root, "sub/module.py", "x = 1\n")
        output_base = tmp_path / "build"

        result = output_writer_basic.write_with_structure(
            input_path=input_path,
            output_base=output_base,
            content="x = 2\n",
            project_root=project_root,
        )

        assert result.success is True
        assert result.output_path == (output_base / "sub/module.py").resolve()


# ---------------------------------------------------------------------------
# Test Batch Operations
# ---------------------------------------------------------------------------

class TestBatchOperations:
    """Tests for write_files() batch behavior."""

    def test_write_files_batch_success(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify all files in a valid batch are written successfully."""
        files = [
            (tmp_path / "a.py", "a = 1\n", None),
            (tmp_path / "b.py", "b = 2\n", None),
            (tmp_path / "c.py", "c = 3\n", None),
        ]
        results = output_writer_basic.write_files(files)

        assert len(results) == 3
        assert all(item.success for item in results)
        assert (tmp_path / "a.py").exists()
        assert (tmp_path / "b.py").exists()
        assert (tmp_path / "c.py").exists()

    def test_write_files_batch_partial_failure(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify mixed-validity batch returns both success and failure results."""
        files = [
            (tmp_path / "ok_1.py", "x = 1\n", None),
            (tmp_path / "bad.txt", "unsupported\n", None),
            (tmp_path / "ok_2.py", "x = 2\n", None),
        ]
        results = output_writer_basic.write_files(files)

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

    def test_write_files_batch_continues_on_error(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify batch processing continues after an intermediate failure."""
        files = [
            (tmp_path / "first.py", "x = 1\n", None),
            (tmp_path / "second.invalid", "bad\n", None),
            (tmp_path / "third.py", "x = 3\n", None),
        ]
        results = output_writer_basic.write_files(files)

        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True
        assert (tmp_path / "third.py").exists(), "Third file should still be processed"

    def test_write_files_returns_results_in_order(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify batch result ordering matches input ordering."""
        first = tmp_path / "order_1.py"
        second = tmp_path / "order_2.py"
        third = tmp_path / "order_3.py"
        files = [
            (first, "x = 1\n", None),
            (second, "x = 2\n", None),
            (third, "x = 3\n", None),
        ]

        results = output_writer_basic.write_files(files)

        assert [item.original_path for item in results] == [first, second, third]

    def test_write_files_updates_metadata(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify metadata counters reflect all batch write attempts."""
        files = [
            (tmp_path / "meta_1.py", "x = 1\n", None),
            (tmp_path / "meta_bad.txt", "bad\n", None),
            (tmp_path / "meta_2.py", "x = 2\n", None),
        ]
        output_writer_basic.write_files(files)
        metadata = output_writer_basic.get_metadata()

        assert metadata.total_writes == 3
        assert metadata.successful_writes == 2
        assert metadata.failed_writes == 1


# ---------------------------------------------------------------------------
# Test Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge-case and robustness tests for write behavior."""

    def test_write_empty_content(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify writing an empty string creates an empty file."""
        target = tmp_path / "empty.py"
        result = output_writer_basic.write_file(target, "")

        assert result.success is True
        assert target.exists()
        assert target.read_text(encoding="utf-8") == ""

    def test_write_very_large_content(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify large payloads (>1MB) are written correctly."""
        target = tmp_path / "large.py"
        content = "x" * (1024 * 1024 + 256)

        result = output_writer_basic.write_file(target, content)

        assert result.success is True
        assert target.stat().st_size == len(content)

    def test_write_binary_content_fails_gracefully(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify non-string payloads fail in a predictable way."""
        target = tmp_path / "binary.py"
        with pytest.raises(TypeError):
            output_writer_basic.write_file(target, b"\x00\x01")

    def test_write_to_symlink(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify writes to symlink paths follow symlink targets correctly."""
        if not hasattr(os, "symlink"):
            pytest.skip("Symlink support not available on this platform")

        real_target = create_test_file(tmp_path, "real.py", "x = 1\n")
        symlink_path = tmp_path / "link.py"

        try:
            os.symlink(real_target, symlink_path)
        except OSError:
            pytest.skip("Symlink creation not permitted in this environment")

        result = output_writer_basic.write_file(symlink_path, "x = 2\n")

        assert result.success is True
        assert real_target.read_text(encoding="utf-8") == "x = 2\n"

    def test_write_with_special_characters_in_filename(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify filenames with spaces/symbols can be written."""
        target = tmp_path / "special name.v1 @test.py"
        result = output_writer_basic.write_file(target, "x = 1\n")

        assert result.success is True
        assert target.exists()

    def test_concurrent_writes_to_same_path(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify repeated writes to same path stay deterministic and valid."""
        target = tmp_path / "concurrent.py"
        payloads = ["value = 1\n", "value = 2\n", "value = 3\n"]
        results = [
            output_writer_basic.write_file(target, payload)
            for payload in payloads
        ]

        assert len(results) == 3
        assert all(item.success for item in results)
        assert target.read_text(encoding="utf-8") == payloads[-1]

    def test_temp_file_cleanup_on_exception(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify temp files are cleaned up when atomic write path errors."""
        target = tmp_path / "cleanup.py"
        temp_path = str(tmp_path / "cleanup_tmp.py")

        class FailingTempFile:
            name = temp_path

            def write(self, _content: str) -> None:
                raise OSError("forced atomic failure")

            def flush(self) -> None:
                return None

            def fileno(self) -> int:
                return 0

            def close(self) -> None:
                return None

        with (
            patch(
                "obfuscator.core.output_writer.tempfile.NamedTemporaryFile",
                return_value=FailingTempFile(),
            ),
            patch("obfuscator.core.output_writer.os.unlink") as mock_unlink,
            patch.object(output_writer_basic, "_write_direct", return_value=False),
        ):
            result = output_writer_basic.write_file(target, "x = 1\n")

        assert result.success is False
        assert mock_unlink.called

    def test_invalid_extension_handling(
        self,
        output_writer_basic: OutputWriter,
        tmp_path: Path,
    ):
        """Verify unsupported extension writes fail with warning metadata."""
        result = output_writer_basic.write_file(tmp_path / "unsupported.xyz", "x")
        metadata = output_writer_basic.get_metadata()

        assert result.success is False
        assert any("Unsupported file extension" in warning for warning in metadata.warnings)

    def test_write_result_dataclass_immutability(self):
        """Verify WriteResult mutability behavior based on dataclass frozen config."""
        result = WriteResult(success=True, output_path=None, original_path=Path("a.py"))
        is_frozen = WriteResult.__dataclass_params__.frozen

        if is_frozen:
            with pytest.raises(FrozenInstanceError):
                result.success = False
        else:
            result.success = False
            assert result.success is False


# ---------------------------------------------------------------------------
# Test Integration With RuntimeManager
# ---------------------------------------------------------------------------

class TestIntegrationWithRuntimeManager:
    """Integration tests using real RuntimeManager instances."""

    def test_integration_python_runtime_full_workflow(
        self,
        sample_python_config: ObfuscationConfig,
        tmp_path: Path,
    ):
        """Verify end-to-end Python runtime generation and write workflow."""
        runtime_manager = _make_python_runtime_manager(sample_python_config)
        writer = OutputWriter(
            output_dir=tmp_path / "python_runtime_out",
            conflict_strategy=ConflictStrategy.OVERWRITE,
        )

        result = writer.write_runtime_library(
            runtime_manager=runtime_manager,
            language="python",
            runtime_mode="hybrid",
        )

        assert result.success is True
        assert result.output_path is not None
        content = result.output_path.read_text(encoding="utf-8")
        assert "Obfuscation Runtime Code - Combined" in content

    def test_integration_lua_runtime_full_workflow(
        self,
        sample_lua_config: ObfuscationConfig,
        tmp_path: Path,
    ):
        """Verify end-to-end Lua runtime generation for Roblox-enabled config."""
        runtime_manager = RuntimeManager(sample_lua_config)
        writer = OutputWriter(
            output_dir=tmp_path / "lua_runtime_out",
            conflict_strategy=ConflictStrategy.OVERWRITE,
        )

        result = writer.write_runtime_library(
            runtime_manager=runtime_manager,
            language="lua",
            runtime_mode="hybrid",
        )

        assert result.success is True
        assert result.output_path is not None
        content = result.output_path.read_text(encoding="utf-8")
        assert "Obfuscation Runtime Code - Combined" in content
        assert "--" in content

    def test_integration_runtime_manager_key_consistency(self):
        """Verify runtime encryption keys remain consistent across generations."""
        config = ObfuscationConfig(
            name="key_consistency",
            language="python",
            features={"code_splitting": True},
            options=_default_options(),
            runtime_mode="hybrid",
            conflict_strategy="overwrite",
        )
        runtime_manager = RuntimeManager(config)

        runtime_manager.get_combined_runtime("python")
        first_key = runtime_manager.get_runtime_key("code_splitting")

        runtime_manager.get_combined_runtime("python")
        second_key = runtime_manager.get_runtime_key("code_splitting")

        assert first_key is not None
        assert second_key is not None
        assert first_key == second_key

    def test_integration_multiple_runtime_features(self, tmp_path: Path):
        """Verify combined runtime generation with multiple enabled features."""
        config = ObfuscationConfig(
            name="multi_feature_runtime",
            language="python",
            features={
                "vm_protection": True,
                "anti_debugging": True,
                "code_splitting": True,
                "self_modifying_code": True,
            },
            options=_default_options(),
            runtime_mode="hybrid",
            conflict_strategy="overwrite",
        )
        runtime_manager = RuntimeManager(config)
        combined = runtime_manager.get_combined_runtime("python")

        assert combined
        assert "Runtime:" in combined or "Obfuscation Runtime Code" in combined

        writer = OutputWriter(
            output_dir=tmp_path / "multi_runtime_out",
            conflict_strategy=ConflictStrategy.OVERWRITE,
        )
        result = writer.write_runtime_library(
            runtime_manager=runtime_manager,
            language="python",
            runtime_mode="hybrid",
        )

        assert result.success is True
        assert result.output_path is not None

    def test_integration_runtime_library_conflict_resolution(
        self,
        sample_python_config: ObfuscationConfig,
        tmp_path: Path,
    ):
        """Verify runtime library conflicts respect SKIP and RENAME strategies."""
        runtime_manager = RuntimeManager(sample_python_config)
        shared_output = tmp_path / "shared_runtime"

        writer_overwrite = OutputWriter(
            output_dir=shared_output,
            conflict_strategy=ConflictStrategy.OVERWRITE,
        )
        first = writer_overwrite.write_runtime_library(
            runtime_manager=runtime_manager,
            language="python",
            runtime_mode="hybrid",
        )
        assert first.success is True
        assert first.output_path is not None

        writer_skip = OutputWriter(
            output_dir=shared_output,
            conflict_strategy=ConflictStrategy.SKIP,
        )
        skipped = writer_skip.write_runtime_library(
            runtime_manager=runtime_manager,
            language="python",
            runtime_mode="hybrid",
        )
        assert skipped.success is True
        assert skipped.output_path is None
        assert skipped.conflict_resolution == "skipped"

        writer_rename = OutputWriter(
            output_dir=shared_output,
            conflict_strategy=ConflictStrategy.RENAME,
        )
        renamed = writer_rename.write_runtime_library(
            runtime_manager=runtime_manager,
            language="python",
            runtime_mode="hybrid",
        )
        assert renamed.success is True
        assert renamed.output_path is not None
        assert renamed.conflict_resolution == "renamed"
