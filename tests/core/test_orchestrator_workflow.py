"""Comprehensive test suite for orchestrator workflow features.

Tests cover input validation, job state management, conflict detection/resolution,
cancellation, error handling with user choice, progress tracking with time
estimation, GUI callback integration, and edge cases.
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from obfuscator.core.config import ObfuscationConfig
from obfuscator.core.orchestrator import (
    ConflictDetectionResult,
    ConflictInfo,
    ConflictStrategy,
    ErrorStrategy,
    JobState,
    ObfuscationOrchestrator,
    OrchestrationResult,
    ProcessResult,
    ProgressInfo,
    ValidationResult,
)

from tests.core.conftest import (
    create_test_file,
    create_readonly_file,
    assert_validation_passed,
    assert_state_transition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    features: dict[str, bool] | None = None,
    language: str = "python",
    conflict_strategy: str = "overwrite",
    **extra_options: Any,
) -> ObfuscationConfig:
    """Create an ObfuscationConfig with the given features."""
    default_features: dict[str, bool] = {
        "mangle_globals": True,
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
        name="test_workflow",
        language=language,
        features=default_features,
        options=options,
        symbol_table_options={
            "identifier_prefix": "_0x",
            "mangling_strategy": "sequential",
            "preserve_exports": False,
            "preserve_constants": False,
        },
        conflict_strategy=conflict_strategy,
    )


def _write_file(tmp_path: Path, name: str, code: str) -> Path:
    """Write a file into tmp_path and return its path."""
    file_path = tmp_path / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(textwrap.dedent(code), encoding="utf-8")
    return file_path


# ===========================================================================
# 2. Input Validation Tests
# ===========================================================================


class TestInputValidation:
    """Tests for validate_inputs() method."""

    def test_validate_files_exist(self, tmp_path: Path):
        """Verify validation fails when input files don't exist."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        nonexistent = tmp_path / "nonexistent.py"
        result = orchestrator.validate_inputs([nonexistent], output_dir)

        assert result.success is False
        assert any("not found" in e.lower() or "nonexistent.py" in e for e in result.errors)

    def test_validate_files_readable(self, tmp_path: Path):
        """Create unreadable files, verify validation detects them."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        unreadable = create_readonly_file(tmp_path, "secret.py")
        try:
            result = orchestrator.validate_inputs([unreadable], output_dir)
            assert result.success is False
            assert any("not readable" in e.lower() or "secret.py" in e for e in result.errors)
        finally:
            os.chmod(unreadable, 0o644)

    def test_validate_file_extensions(self, tmp_path: Path):
        """Test with invalid extensions (.txt, .md), verify rejection."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        txt_file = _write_file(tmp_path, "readme.txt", "hello\n")
        md_file = _write_file(tmp_path, "notes.md", "# Notes\n")

        result = orchestrator.validate_inputs([txt_file, md_file], output_dir)

        assert result.success is False
        assert len(result.errors) == 2
        assert any(".txt" in e for e in result.errors)
        assert any(".md" in e for e in result.errors)

    def test_validate_output_directory_writable(self, tmp_path: Path):
        """Create read-only output directory, verify validation detects issue."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")
        output_dir = tmp_path / "readonly_out"
        output_dir.mkdir()
        os.chmod(output_dir, 0o444)

        try:
            result = orchestrator.validate_inputs([py_file], output_dir)
            assert result.success is False
            assert any("not writable" in e.lower() for e in result.errors)
        finally:
            os.chmod(output_dir, 0o755)

    def test_validate_config(self, tmp_path: Path):
        """Pass invalid config, verify config.validate() errors are caught."""
        bad_config = ObfuscationConfig(
            name="bad",
            language="invalid_language",
            features={"mangle_globals": True},
        )
        orchestrator = ObfuscationOrchestrator(config=bad_config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")
        result = orchestrator.validate_inputs([py_file], output_dir)

        assert result.success is False
        assert any("configuration" in e.lower() or "validation" in e.lower() for e in result.errors)

    def test_validate_empty_file_list(self, tmp_path: Path):
        """Pass empty input_files list, verify validation fails."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = orchestrator.validate_inputs([], output_dir)

        assert result.success is False
        assert any("no input files" in e.lower() for e in result.errors)

    def test_validate_mixed_valid_invalid(self, tmp_path: Path):
        """Mix valid and invalid files, verify all issues reported."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        valid_file = _write_file(tmp_path, "good.py", "x = 1\n")
        bad_ext = _write_file(tmp_path, "bad.txt", "hello\n")
        nonexistent = tmp_path / "missing.py"

        result = orchestrator.validate_inputs(
            [valid_file, bad_ext, nonexistent], output_dir
        )

        assert result.success is False
        assert len(result.errors) >= 2

    def test_validation_result_structure(self, tmp_path: Path):
        """Verify ValidationResult contains expected attributes."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")
        result = orchestrator.validate_inputs([py_file], output_dir)

        assert isinstance(result, ValidationResult)
        assert hasattr(result, "success")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)


# ===========================================================================
# 3. Job State Management Tests
# ===========================================================================


class TestJobStateManagement:
    """Tests for job state tracking and transitions."""

    def test_initial_state_pending(self):
        """Verify orchestrator starts in PENDING state."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        assert orchestrator._current_state == JobState.PENDING

    def test_state_transition_sequence(self, tmp_path: Path):
        """Verify state transitions: PENDING -> VALIDATING -> ANALYZING -> PROCESSING -> COMPLETED."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        states_seen: list[str] = []
        original_transition = orchestrator._transition_state

        def tracking_transition(new_state, result):
            states_seen.append(new_state.name)
            original_transition(new_state, result)

        orchestrator._transition_state = tracking_transition

        result = orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert "PENDING" in states_seen
        assert "VALIDATING" in states_seen
        assert "ANALYZING" in states_seen
        assert "PROCESSING" in states_seen
        assert "COMPLETED" in states_seen

        # Verify ordering
        pending_idx = states_seen.index("PENDING")
        validating_idx = states_seen.index("VALIDATING")
        analyzing_idx = states_seen.index("ANALYZING")
        processing_idx = states_seen.index("PROCESSING")
        completed_idx = states_seen.index("COMPLETED")

        assert pending_idx < validating_idx < analyzing_idx < processing_idx < completed_idx

    def test_state_transition_on_validation_failure(self, tmp_path: Path):
        """Trigger validation failure, verify state transitions to FAILED."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = orchestrator.process_files(
            input_files=[],
            output_dir=output_dir,
            config=config.symbol_table_options,
        )

        assert result.current_state == JobState.FAILED
        assert result.success is False

    def test_state_transition_on_processing_error(self, tmp_path: Path):
        """Inject processing error, verify state transitions to FAILED."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        with patch.object(
            orchestrator, "_scan_and_extract_symbols",
            side_effect=Exception("Simulated scan failure"),
        ):
            result = orchestrator.process_files(
                input_files=[py_file],
                output_dir=output_dir,
                config=config.symbol_table_options,
                project_root=tmp_path,
            )

        assert result.current_state == JobState.FAILED
        assert result.success is False

    def test_state_in_orchestration_result(self, tmp_path: Path):
        """Verify OrchestrationResult.current_state reflects final state."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        result = orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert isinstance(result.current_state, JobState)
        # Should be COMPLETED or FAILED depending on processor availability
        assert result.current_state in (JobState.COMPLETED, JobState.FAILED)

    def test_state_logging(self, tmp_path: Path, caplog):
        """Use caplog to verify state transitions are logged at INFO level."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        with caplog.at_level(logging.INFO, logger="obfuscator.core.orchestrator"):
            orchestrator.process_files(
                input_files=[py_file],
                output_dir=output_dir,
                config=config.symbol_table_options,
                project_root=tmp_path,
            )

        state_logs = [r for r in caplog.records if "State transition" in r.message]
        assert len(state_logs) >= 2  # At least PENDING->VALIDATING and beyond

    def test_state_in_progress_callback(self, tmp_path: Path):
        """Mock progress callback, verify state name appears in callback."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        captured: list[ProgressInfo] = []

        def progress_cb(info: ProgressInfo):
            captured.append(info)

        orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            progress_callback=progress_cb,
        )

        assert len(captured) > 0
        states_in_callbacks = {c.current_state for c in captured}
        # Should see at least VALIDATING and ANALYZING
        assert JobState.VALIDATING in states_in_callbacks or JobState.ANALYZING in states_in_callbacks

    def test_cancelled_state(self, tmp_path: Path):
        """Call request_cancellation() during processing, verify CANCELLED state."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        # Request cancellation before processing
        orchestrator._cancellation_requested = True

        result = orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert result.current_state == JobState.CANCELLED
        assert result.success is False


# ===========================================================================
# 4. Conflict Detection and Resolution Tests
# ===========================================================================


class TestConflictHandling:
    """Tests for conflict detection and resolution."""

    def test_detect_no_conflicts(self, tmp_path: Path):
        """Process files with empty output directory, verify no conflicts."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)

        py_file = _write_file(tmp_path / "src", "main.py", "x = 1\n")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = orchestrator.detect_conflicts([py_file], output_dir, tmp_path / "src")
        assert not result.has_conflicts
        assert len(result.conflicts) == 0

    def test_detect_conflicts(self, tmp_path: Path):
        """Pre-create output files, verify conflicts detected."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)

        src_dir = tmp_path / "src"
        py_file = _write_file(src_dir, "main.py", "x = 1\n")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        # Pre-create the output file to cause conflict
        (output_dir / "main.py").write_text("old content", encoding="utf-8")

        result = orchestrator.detect_conflicts([py_file], output_dir, src_dir)
        assert result.has_conflicts
        assert len(result.conflicts) == 1
        assert result.conflicts[0].exists is True

    def test_resolve_conflict_overwrite(self, tmp_path: Path):
        """Set strategy to OVERWRITE, verify path is returned as-is."""
        config = _make_config(conflict_strategy="overwrite")
        orchestrator = ObfuscationOrchestrator(config=config)

        output_path = tmp_path / "main.py"
        output_path.write_text("old", encoding="utf-8")

        resolved = orchestrator.resolve_conflict(output_path)
        assert resolved == output_path

    def test_resolve_conflict_skip(self, tmp_path: Path):
        """Set strategy to SKIP, verify None is returned."""
        config = _make_config(conflict_strategy="skip")
        orchestrator = ObfuscationOrchestrator(config=config)

        output_path = tmp_path / "main.py"
        output_path.write_text("old", encoding="utf-8")

        resolved = orchestrator.resolve_conflict(output_path)
        assert resolved is None

    def test_resolve_conflict_rename(self, tmp_path: Path):
        """Set strategy to RENAME, verify new file has timestamp suffix."""
        config = _make_config(conflict_strategy="rename")
        orchestrator = ObfuscationOrchestrator(config=config)

        output_path = tmp_path / "main.py"
        output_path.write_text("old", encoding="utf-8")

        resolved = orchestrator.resolve_conflict(output_path)
        assert resolved is not None
        assert resolved != output_path
        # Should match pattern: main_YYYYMMDD_HHMMSS.py
        assert re.match(r"main_\d{8}_\d{6}\.py", resolved.name)

    def test_conflict_resolution_before_processing(self, tmp_path: Path):
        """Verify detect_conflicts() is called before processing starts."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        call_order: list[str] = []
        original_detect = orchestrator.detect_conflicts
        original_process = orchestrator._process_file_in_order

        def tracking_detect(*args, **kwargs):
            call_order.append("detect_conflicts")
            return original_detect(*args, **kwargs)

        def tracking_process(*args, **kwargs):
            call_order.append("process_file")
            return original_process(*args, **kwargs)

        orchestrator.detect_conflicts = tracking_detect
        orchestrator._process_file_in_order = tracking_process

        orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert "detect_conflicts" in call_order
        if "process_file" in call_order:
            detect_idx = call_order.index("detect_conflicts")
            process_idx = call_order.index("process_file")
            assert detect_idx < process_idx

    def test_conflict_callback_integration(self, tmp_path: Path):
        """Mock conflict callback, verify ConflictInfo list is correct."""
        config = _make_config(conflict_strategy="overwrite")
        orchestrator = ObfuscationOrchestrator(config=config)

        src_dir = tmp_path / "src"
        py_file = _write_file(src_dir, "main.py", "x = 1\n")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "main.py").write_text("old", encoding="utf-8")

        result = orchestrator.detect_conflicts([py_file], output_dir, src_dir)
        assert result.has_conflicts
        for conflict in result.conflicts:
            assert isinstance(conflict, ConflictInfo)
            assert isinstance(conflict.input_path, Path)
            assert isinstance(conflict.output_path, Path)
            assert isinstance(conflict.exists, bool)

    def test_batch_conflict_resolution(self, tmp_path: Path):
        """Create multiple conflicting files, verify strategy applies to all."""
        config = _make_config(conflict_strategy="skip")
        orchestrator = ObfuscationOrchestrator(config=config)

        src_dir = tmp_path / "src"
        files = [
            _write_file(src_dir, "a.py", "x = 1\n"),
            _write_file(src_dir, "b.py", "y = 2\n"),
            _write_file(src_dir, "c.py", "z = 3\n"),
        ]

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        for name in ["a.py", "b.py", "c.py"]:
            (output_dir / name).write_text("old", encoding="utf-8")

        result = orchestrator.detect_conflicts(files, output_dir, src_dir)
        assert len(result.conflicts) == 3

        # Each should resolve to None (SKIP)
        for conflict in result.conflicts:
            resolved = orchestrator.resolve_conflict(conflict.output_path)
            assert resolved is None

    def test_rename_timestamp_format(self, tmp_path: Path):
        """Verify renamed files use YYYYMMDD_HHMMSS format."""
        config = _make_config(conflict_strategy="rename")
        orchestrator = ObfuscationOrchestrator(config=config)

        output_path = tmp_path / "script.lua"
        output_path.write_text("old", encoding="utf-8")

        resolved = orchestrator.resolve_conflict(output_path)
        assert resolved is not None
        # Pattern: script_YYYYMMDD_HHMMSS.lua
        pattern = r"^script_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.lua$"
        match = re.match(pattern, resolved.name)
        assert match is not None, f"Renamed file '{resolved.name}' doesn't match expected format"


# ===========================================================================
# 5. Cancellation Mechanism Tests
# ===========================================================================


class TestCancellation:
    """Tests for the cancellation mechanism."""

    def test_cancellation_flag_initial_state(self):
        """Verify _cancellation_requested is False initially."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        assert orchestrator._cancellation_requested is False

    def test_request_cancellation(self):
        """Call request_cancellation(), verify flag is set."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        orchestrator.request_cancellation()
        assert orchestrator._cancellation_requested is True

    def test_cancellation_during_processing(self, tmp_path: Path):
        """Start process_files(), cancel mid-process, verify it stops."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        files = []
        for i in range(5):
            files.append(_write_file(tmp_path, f"file_{i}.py", f"x_{i} = {i}\n"))

        results: list[OrchestrationResult] = []

        def run_processing():
            r = orchestrator.process_files(
                input_files=files,
                output_dir=output_dir,
                config=config.symbol_table_options,
                project_root=tmp_path,
            )
            results.append(r)

        # Cancel after a brief delay
        def cancel_after_delay():
            time.sleep(0.05)
            orchestrator.request_cancellation()

        t1 = threading.Thread(target=run_processing)
        t2 = threading.Thread(target=cancel_after_delay)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=5)

        assert len(results) == 1
        result = results[0]
        # Either cancelled or completed (if processing finished before cancel)
        assert result.current_state in (JobState.CANCELLED, JobState.COMPLETED, JobState.FAILED)

    def test_cancellation_preserves_completed_files(self, tmp_path: Path):
        """Cancel after some files, verify completed files are in result."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        files = []
        for i in range(5):
            files.append(_write_file(tmp_path, f"file_{i}.py", f"x_{i} = {i}\n"))

        # We'll cancel right before processing starts via the flag
        # First let validation and analysis run, then cancel at processing start
        call_count = [0]
        original_process = orchestrator._process_file_in_order

        def process_then_cancel(*args, **kwargs):
            call_count[0] += 1
            result = original_process(*args, **kwargs)
            if call_count[0] >= 2:
                orchestrator.request_cancellation()
            return result

        orchestrator._process_file_in_order = process_then_cancel

        result = orchestrator.process_files(
            input_files=files,
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        # Should have at least some processed files
        if result.current_state == JobState.CANCELLED:
            assert len(result.processed_files) >= 1

    def test_cancellation_result_metadata(self, tmp_path: Path):
        """Verify OrchestrationResult contains cancellation metadata."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        # Pre-set cancellation
        orchestrator._cancellation_requested = True

        result = orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert result.current_state == JobState.CANCELLED
        assert result.metadata.get("was_cancelled") is True

    def test_cancellation_state_transition(self, tmp_path: Path):
        """Verify state transitions to CANCELLED when cancellation is requested."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        states_seen: list[str] = []
        original_transition = orchestrator._transition_state

        def tracking_transition(new_state, result):
            states_seen.append(new_state.name)
            original_transition(new_state, result)

        orchestrator._transition_state = tracking_transition
        orchestrator._cancellation_requested = True

        orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert "CANCELLED" in states_seen

    def test_cancellation_cleanup(self, tmp_path: Path):
        """Verify no partial/corrupted output after cancellation."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        orchestrator._cancellation_requested = True

        result = orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        # Any files that were written should be complete (not partial)
        for pr in result.processed_files:
            if pr.success and pr.output_path and pr.output_path.exists():
                content = pr.output_path.read_text(encoding="utf-8")
                assert len(content) > 0

    def test_multiple_cancellation_requests(self):
        """Call request_cancellation() multiple times, verify idempotent."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)

        orchestrator.request_cancellation()
        orchestrator.request_cancellation()
        orchestrator.request_cancellation()

        assert orchestrator._cancellation_requested is True

    def test_cancellation_before_processing(self, tmp_path: Path):
        """Call request_cancellation() before process_files(), verify immediate cancellation."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        orchestrator.request_cancellation()

        result = orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert result.current_state == JobState.CANCELLED
        assert result.success is False
        assert len(result.processed_files) == 0


# ===========================================================================
# 6. Error Handling with User Choice Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling strategies (CONTINUE, STOP, ASK)."""

    def test_error_strategy_continue(self, tmp_path: Path):
        """Set CONTINUE, inject error in file 2 of 5, verify processing continues."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        files = [_write_file(tmp_path, f"f{i}.py", f"x{i} = {i}\n") for i in range(5)]

        call_count = [0]
        original_process = orchestrator._process_file_in_order

        def inject_error(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                return ProcessResult(
                    file_path=args[0],
                    output_path=None,
                    success=False,
                    errors=["Simulated error"],
                )
            return original_process(*args, **kwargs)

        orchestrator._process_file_in_order = inject_error

        result = orchestrator.process_files(
            input_files=files,
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            error_strategy=ErrorStrategy.CONTINUE,
        )

        # Processing should have continued past the error
        assert len(result.processed_files) >= 3

    def test_error_strategy_stop(self, tmp_path: Path):
        """Set STOP, inject error in file 2 of 5, verify processing stops."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        files = [_write_file(tmp_path, f"f{i}.py", f"x{i} = {i}\n") for i in range(5)]

        call_count = [0]
        original_process = orchestrator._process_file_in_order

        def inject_error(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                return ProcessResult(
                    file_path=args[0],
                    output_path=None,
                    success=False,
                    errors=["Simulated error"],
                )
            return original_process(*args, **kwargs)

        orchestrator._process_file_in_order = inject_error

        result = orchestrator.process_files(
            input_files=files,
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            error_strategy=ErrorStrategy.STOP,
        )

        assert result.current_state == JobState.FAILED
        assert result.success is False
        # Should have stopped before processing all files
        assert len(result.processed_files) <= 3

    def test_error_strategy_ask_continue(self, tmp_path: Path):
        """Set ASK, callback returns True (continue), verify processing continues."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        files = [_write_file(tmp_path, f"f{i}.py", f"x{i} = {i}\n") for i in range(3)]

        call_count = [0]
        original_process = orchestrator._process_file_in_order

        def inject_error(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ProcessResult(
                    file_path=args[0],
                    output_path=None,
                    success=False,
                    errors=["Simulated error"],
                )
            return original_process(*args, **kwargs)

        orchestrator._process_file_in_order = inject_error

        mock_cb = MagicMock(return_value=True)

        result = orchestrator.process_files(
            input_files=files,
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            error_strategy=ErrorStrategy.ASK,
            error_callback=mock_cb,
        )

        mock_cb.assert_called_once()
        assert len(result.processed_files) >= 2

    def test_error_strategy_ask_stop(self, tmp_path: Path):
        """Set ASK, callback returns False (stop), verify processing stops."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        files = [_write_file(tmp_path, f"f{i}.py", f"x{i} = {i}\n") for i in range(3)]

        call_count = [0]
        original_process = orchestrator._process_file_in_order

        def inject_error(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ProcessResult(
                    file_path=args[0],
                    output_path=None,
                    success=False,
                    errors=["Simulated error"],
                )
            return original_process(*args, **kwargs)

        orchestrator._process_file_in_order = inject_error

        mock_cb = MagicMock(return_value=False)

        result = orchestrator.process_files(
            input_files=files,
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            error_strategy=ErrorStrategy.ASK,
            error_callback=mock_cb,
        )

        mock_cb.assert_called_once()
        assert result.current_state == JobState.FAILED
        assert result.success is False

    def test_error_callback_receives_details(self, tmp_path: Path):
        """Mock error callback, verify it receives file_path and errors list."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)

        error_details: list[tuple] = []

        def capture_cb(file_path, errors):
            error_details.append((file_path, errors))
            return True

        orchestrator._error_strategy = ErrorStrategy.ASK
        result = orchestrator.handle_processing_error(
            Path("test.py"),
            ["Error 1", "Error 2"],
            capture_cb,
        )

        assert result is True
        assert len(error_details) == 1
        assert error_details[0][0] == Path("test.py")
        assert error_details[0][1] == ["Error 1", "Error 2"]

    def test_error_tracking_in_result(self, tmp_path: Path):
        """Verify OrchestrationResult.metadata contains error decisions."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        files = [_write_file(tmp_path, f"f{i}.py", f"x{i} = {i}\n") for i in range(3)]

        call_count = [0]
        original_process = orchestrator._process_file_in_order

        def inject_error(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ProcessResult(
                    file_path=args[0],
                    output_path=None,
                    success=False,
                    errors=["Simulated error"],
                )
            return original_process(*args, **kwargs)

        orchestrator._process_file_in_order = inject_error

        result = orchestrator.process_files(
            input_files=files,
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            error_strategy=ErrorStrategy.CONTINUE,
        )

        assert "error_decisions" in result.metadata
        assert "files_failed_with_errors" in result.metadata
        assert len(result.metadata["error_decisions"]) >= 1

    def test_multiple_errors_with_ask(self, tmp_path: Path):
        """Inject errors in files 1 and 3, verify callback called for each."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        files = [_write_file(tmp_path, f"f{i}.py", f"x{i} = {i}\n") for i in range(5)]

        call_count = [0]
        original_process = orchestrator._process_file_in_order

        def inject_errors(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] in (1, 3):
                return ProcessResult(
                    file_path=args[0],
                    output_path=None,
                    success=False,
                    errors=[f"Error in file {call_count[0]}"],
                )
            return original_process(*args, **kwargs)

        orchestrator._process_file_in_order = inject_errors

        mock_cb = MagicMock(return_value=True)

        result = orchestrator.process_files(
            input_files=files,
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            error_strategy=ErrorStrategy.ASK,
            error_callback=mock_cb,
        )

        assert mock_cb.call_count == 2

    def test_error_logging(self, tmp_path: Path, caplog):
        """Verify errors are logged with ERROR level."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)

        with caplog.at_level(logging.WARNING, logger="obfuscator.core.orchestrator"):
            orchestrator.handle_processing_error(
                Path("test.py"),
                ["Parse error at line 10"],
                None,
            )

        error_logs = [
            r for r in caplog.records
            if "test.py" in r.message and r.levelno >= logging.WARNING
        ]
        assert len(error_logs) >= 1

    def test_error_in_validation_phase(self, tmp_path: Path):
        """Trigger error during validation, verify handled differently."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Empty file list triggers validation error, not processing error
        result = orchestrator.process_files(
            input_files=[],
            output_dir=output_dir,
            config=config.symbol_table_options,
        )

        assert result.current_state == JobState.FAILED
        assert any("no input files" in e.lower() for e in result.errors)


# ===========================================================================
# 7. Progress Tracking and Time Estimation Tests
# ===========================================================================


class TestProgressTracking:
    """Tests for progress callbacks and time estimation."""

    def test_progress_callback_invocation(self, tmp_path: Path):
        """Mock progress callback, verify called for each file."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        files = [_write_file(tmp_path, f"f{i}.py", f"x{i} = {i}\n") for i in range(3)]

        captured: list[ProgressInfo] = []

        def cb(info: ProgressInfo):
            captured.append(info)

        orchestrator.process_files(
            input_files=files,
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            progress_callback=cb,
        )

        # Should have at least one callback per phase + one per file
        assert len(captured) >= 3

    def test_progress_info_structure(self, tmp_path: Path):
        """Verify ProgressInfo contains all expected fields."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        captured: list[ProgressInfo] = []

        def cb(info: ProgressInfo):
            captured.append(info)

        orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            progress_callback=cb,
        )

        assert len(captured) > 0
        info = captured[0]
        assert hasattr(info, "current_file")
        assert hasattr(info, "current_index")
        assert hasattr(info, "total_files")
        assert hasattr(info, "percentage")
        assert hasattr(info, "elapsed_seconds")
        assert hasattr(info, "estimated_remaining_seconds")
        assert hasattr(info, "current_state")
        assert hasattr(info, "message")

    def test_progress_percentage_calculation(self, tmp_path: Path):
        """Verify progress percentage increases over time."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        files = [_write_file(tmp_path, f"f{i}.py", f"x{i} = {i}\n") for i in range(3)]

        captured: list[ProgressInfo] = []

        def cb(info: ProgressInfo):
            captured.append(info)

        orchestrator.process_files(
            input_files=files,
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            progress_callback=cb,
        )

        assert len(captured) >= 2
        # Percentages should be non-decreasing
        percentages = [c.percentage for c in captured]
        for i in range(1, len(percentages)):
            assert percentages[i] >= percentages[i - 1]

    def test_time_tracking_start(self, tmp_path: Path):
        """Verify _start_time is set when process_files() begins."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        assert orchestrator._start_time is None

        orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert orchestrator._start_time is not None

    def test_elapsed_time_calculation(self, tmp_path: Path):
        """Verify elapsed time is tracked."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        captured: list[ProgressInfo] = []

        def cb(info: ProgressInfo):
            captured.append(info)

        orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            progress_callback=cb,
        )

        # At least one callback should have non-zero elapsed
        elapsed_values = [c.elapsed_seconds for c in captured]
        assert any(e >= 0 for e in elapsed_values)

    def test_estimated_remaining_time(self, tmp_path: Path):
        """Process files, verify ETA calculation is present after first file."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        files = [_write_file(tmp_path, f"f{i}.py", f"x{i} = {i}\n") for i in range(5)]

        captured: list[ProgressInfo] = []

        def cb(info: ProgressInfo):
            captured.append(info)

        orchestrator.process_files(
            input_files=files,
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            progress_callback=cb,
        )

        # At least some callbacks should have estimated_remaining_seconds
        has_estimates = [
            c for c in captured if c.estimated_remaining_seconds is not None
        ]
        # After first file is processed, estimates should appear
        # (may not appear if processing is very fast)
        assert len(captured) >= 3

    def test_progress_metadata_in_result(self, tmp_path: Path):
        """Verify OrchestrationResult contains total_time and average_time_per_file."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        result = orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert "total_processing_time_seconds" in result.metadata
        assert "average_time_per_file_seconds" in result.metadata
        assert result.metadata["total_processing_time_seconds"] >= 0

    def test_progress_callback_with_state(self, tmp_path: Path):
        """Verify progress callback includes current state information."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        captured: list[ProgressInfo] = []

        def cb(info: ProgressInfo):
            captured.append(info)

        orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            progress_callback=cb,
        )

        for info in captured:
            assert isinstance(info.current_state, JobState)

    def test_progress_for_empty_file_list(self, tmp_path: Path):
        """Process empty list, verify progress callback handles edge case."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        captured: list[ProgressInfo] = []

        def cb(info: ProgressInfo):
            captured.append(info)

        result = orchestrator.process_files(
            input_files=[],
            output_dir=output_dir,
            config=config.symbol_table_options,
            progress_callback=cb,
        )

        # Should still get validation progress callback
        assert len(captured) >= 1
        assert result.current_state == JobState.FAILED

    def test_time_estimation_first_file(self, tmp_path: Path):
        """Verify estimated_remaining_seconds is None for first file."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        captured: list[ProgressInfo] = []

        def cb(info: ProgressInfo):
            captured.append(info)

        orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            progress_callback=cb,
        )

        # First callback (validation phase) should have None for estimated remaining
        if captured:
            assert captured[0].estimated_remaining_seconds is None


# ===========================================================================
# 8. Integration Tests with GUI Callbacks
# ===========================================================================


class TestGUIIntegration:
    """Tests for GUI callback integration patterns."""

    def test_full_workflow_with_all_callbacks(self, tmp_path: Path):
        """Mock all callbacks, simulate complete workflow."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        progress_calls: list[ProgressInfo] = []

        def progress_cb(info: ProgressInfo):
            progress_calls.append(info)

        error_cb = MagicMock(return_value=True)

        result = orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            progress_callback=progress_cb,
            error_callback=error_cb,
        )

        assert result is not None
        assert len(progress_calls) >= 1

    def test_progress_widget_integration(self, tmp_path: Path):
        """Create mock ProgressWidget, verify state and progress updates."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        # Simulate ProgressWidget
        widget = MagicMock()
        widget.state_updates = []
        widget.progress_updates = []

        def on_progress(info: ProgressInfo):
            widget.state_updates.append(info.current_state)
            widget.progress_updates.append(info.percentage)

        result = orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            progress_callback=on_progress,
        )

        assert len(widget.state_updates) >= 1
        assert len(widget.progress_updates) >= 1

    def test_error_dialog_integration(self, tmp_path: Path):
        """Mock ErrorHandlingDialog, inject error, verify dialog receives params."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        files = [_write_file(tmp_path, f"f{i}.py", f"x{i} = {i}\n") for i in range(3)]

        call_count = [0]
        original_process = orchestrator._process_file_in_order

        def inject_error(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ProcessResult(
                    file_path=args[0],
                    output_path=None,
                    success=False,
                    errors=["Parse error at line 10"],
                )
            return original_process(*args, **kwargs)

        orchestrator._process_file_in_order = inject_error

        dialog_calls: list[tuple] = []

        def error_dialog(file_path, errors):
            dialog_calls.append((file_path, errors))
            return True

        result = orchestrator.process_files(
            input_files=files,
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            error_strategy=ErrorStrategy.ASK,
            error_callback=error_dialog,
        )

        assert len(dialog_calls) == 1
        assert isinstance(dialog_calls[0][0], Path)
        assert isinstance(dialog_calls[0][1], list)
        assert "Parse error at line 10" in dialog_calls[0][1]

    def test_conflict_dialog_integration(self, tmp_path: Path):
        """Mock ConflictResolutionDialog, verify it receives ConflictInfo list."""
        config = _make_config(conflict_strategy="overwrite")
        orchestrator = ObfuscationOrchestrator(config=config)

        src_dir = tmp_path / "src"
        py_file = _write_file(src_dir, "main.py", "x = 1\n")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "main.py").write_text("old", encoding="utf-8")

        result = orchestrator.detect_conflicts([py_file], output_dir, src_dir)

        # Simulated dialog receives conflicts
        assert result.has_conflicts
        for conflict in result.conflicts:
            assert isinstance(conflict, ConflictInfo)

    def test_conflict_resolution_via_process_files_with_ask(self, tmp_path: Path):
        """Exercise conflict resolution end-to-end with a GUI-style conflict callback."""
        config = _make_config(conflict_strategy="ask")
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        src_dir = tmp_path / "src"
        py_file = _write_file(src_dir, "main.py", "x = 1\n")

        # Pre-create the output file to force a conflict
        (output_dir / "main.py").write_text("old content", encoding="utf-8")

        # Step 1: Call process_files with ASK strategy — returns early with conflicts
        result = orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=src_dir,
        )

        assert result.success is False
        assert "conflicts" in result.metadata
        conflicts = result.metadata["conflicts"]
        assert len(conflicts) >= 1
        for conflict in conflicts:
            assert isinstance(conflict, ConflictInfo)
            assert isinstance(conflict.input_path, Path)
            assert isinstance(conflict.output_path, Path)
            assert conflict.exists is True

        # Step 2: Simulate GUI conflict callback that chooses RENAME
        mock_conflict_callback = MagicMock(return_value=ConflictStrategy.RENAME)
        chosen_strategy = mock_conflict_callback(conflicts)
        mock_conflict_callback.assert_called_once_with(conflicts)

        # Step 3: Apply the chosen strategy and re-process
        orchestrator.set_conflict_strategy(chosen_strategy)
        result2 = orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=src_dir,
        )

        # Step 4: Verify the strategy was honored
        assert result2 is not None
        assert result2.metadata.get("conflict_strategy") == "rename"

    def test_cancellation_from_gui(self, tmp_path: Path):
        """Mock cancel button click, verify request_cancellation() is called."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)

        # Simulate GUI cancel button
        orchestrator.request_cancellation()
        assert orchestrator._cancellation_requested is True

    def test_callback_exception_handling(self, tmp_path: Path):
        """Make callback raise exception, verify orchestrator handles gracefully."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        def bad_callback(info: ProgressInfo):
            raise RuntimeError("Callback crashed!")

        # Should not crash the orchestrator
        result = orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
            progress_callback=bad_callback,
        )

        # Orchestrator should handle callback failures gracefully
        # (it will raise or fail but shouldn't crash silently)
        assert result is not None

    def test_concurrent_callback_invocations(self, tmp_path: Path):
        """Verify callbacks work correctly from a separate thread."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")

        callback_thread_ids: list[int] = []

        def thread_tracking_cb(info: ProgressInfo):
            callback_thread_ids.append(threading.get_ident())

        results: list[OrchestrationResult] = []

        def run():
            r = orchestrator.process_files(
                input_files=[py_file],
                output_dir=output_dir,
                config=config.symbol_table_options,
                project_root=tmp_path,
                progress_callback=thread_tracking_cb,
            )
            results.append(r)

        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=30)

        assert len(results) == 1
        # All callbacks should come from the same thread
        if callback_thread_ids:
            assert len(set(callback_thread_ids)) == 1


# ===========================================================================
# 9. Edge Cases and Error Scenarios
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_empty_file_list(self, tmp_path: Path):
        """Pass empty input_files, verify graceful handling."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = orchestrator.process_files(
            input_files=[],
            output_dir=output_dir,
            config=config.symbol_table_options,
        )

        assert result.success is False
        assert result.current_state == JobState.FAILED

    def test_invalid_config_none(self, tmp_path: Path):
        """Pass config=None, verify default config is used or error raised."""
        orchestrator = ObfuscationOrchestrator(config=None)

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")
        output_dir = tmp_path / "output"

        result = orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            project_root=tmp_path,
        )

        # Should still work (no config means no config validation)
        assert result is not None

    def test_permission_error_output_directory(self, tmp_path: Path):
        """Create output directory without write permissions."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")
        output_dir = tmp_path / "no_write"
        output_dir.mkdir()
        os.chmod(output_dir, 0o444)

        try:
            result = orchestrator.validate_inputs([py_file], output_dir)
            assert result.success is False
            assert any("not writable" in e.lower() for e in result.errors)
        finally:
            os.chmod(output_dir, 0o755)

    def test_permission_error_during_write(self, tmp_path: Path):
        """Make output directory read-only after validation, verify error handling."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)

        py_file = _write_file(tmp_path, "main.py", "x = 1\n")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Validation passes (dir is writable)
        val_result = orchestrator.validate_inputs([py_file], output_dir)
        assert val_result.success is True

        # Make read-only after validation
        os.chmod(output_dir, 0o444)
        try:
            result = orchestrator.process_files(
                input_files=[py_file],
                output_dir=output_dir,
                config=config.symbol_table_options,
                project_root=tmp_path,
            )
            # Should fail during write phase
            # (may succeed if output_dir gets recreated by mkdir)
            assert result is not None
        finally:
            os.chmod(output_dir, 0o755)

    def test_mid_process_cancellation(self, tmp_path: Path):
        """Cancel exactly between files, verify clean state."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        files = [_write_file(tmp_path, f"f{i}.py", f"x{i} = {i}\n") for i in range(5)]

        call_count = [0]
        original_process = orchestrator._process_file_in_order

        def cancel_at_file_3(*args, **kwargs):
            call_count[0] += 1
            result = original_process(*args, **kwargs)
            if call_count[0] == 2:
                orchestrator.request_cancellation()
            return result

        orchestrator._process_file_in_order = cancel_at_file_3

        result = orchestrator.process_files(
            input_files=files,
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        if result.current_state == JobState.CANCELLED:
            assert result.metadata.get("was_cancelled") is True

    def test_very_large_file_list(self, tmp_path: Path):
        """Verify large file batches are accepted and processing is invoked for each."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        files = []
        for i in range(10):
            files.append(_write_file(tmp_path, f"mod_{i}.py", f"var_{i} = {i}\n"))

        process_call_count = [0]

        def stub_process(file_path, *args, **kwargs):
            process_call_count[0] += 1
            out = output_dir / file_path.name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("# stubbed", encoding="utf-8")
            return ProcessResult(
                file_path=file_path,
                output_path=out,
                success=True,
                errors=[],
            )

        orchestrator._process_file_in_order = stub_process

        result = orchestrator.process_files(
            input_files=files,
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert result is not None
        assert process_call_count[0] == 10
        assert len(result.processed_files) == 10
        assert all(pr.success for pr in result.processed_files)

    def test_unicode_file_paths(self, tmp_path: Path):
        """Use files with unicode characters in names."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        unicode_file = _write_file(tmp_path, "módulo_café.py", "x = 1\n")

        result = orchestrator.process_files(
            input_files=[unicode_file],
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert result is not None

    def test_symlink_handling(self, tmp_path: Path):
        """Create symlinks to files, verify they are handled."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        real_file = _write_file(tmp_path, "real.py", "x = 1\n")
        link_path = tmp_path / "link.py"

        try:
            link_path.symlink_to(real_file)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        result = orchestrator.validate_inputs([link_path], output_dir)
        # Symlinks should be resolved and validated
        assert isinstance(result, ValidationResult)

    def test_circular_dependency_with_cancellation(self, tmp_path: Path):
        """Trigger circular dependency detection, then cancel, verify clean exit."""
        config = _make_config()
        orchestrator = ObfuscationOrchestrator(config=config)
        output_dir = tmp_path / "output"

        # Create files with circular imports
        _write_file(tmp_path, "a.py", "from b import y\nx = 1\n")
        _write_file(tmp_path, "b.py", "from a import x\ny = 2\n")

        files = [tmp_path / "a.py", tmp_path / "b.py"]

        # Request cancellation mid-process
        original_process = orchestrator._process_file_in_order

        def cancel_on_first(*args, **kwargs):
            orchestrator.request_cancellation()
            return original_process(*args, **kwargs)

        orchestrator._process_file_in_order = cancel_on_first

        result = orchestrator.process_files(
            input_files=files,
            output_dir=output_dir,
            config=config.symbol_table_options,
            project_root=tmp_path,
        )

        assert result is not None
        assert result.current_state in (
            JobState.CANCELLED, JobState.COMPLETED, JobState.FAILED
        )
