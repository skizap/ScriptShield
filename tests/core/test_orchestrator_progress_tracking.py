"""Comprehensive progress tracking tests for the orchestration workflow."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from obfuscator.core.config import ObfuscationConfig
from obfuscator.core.orchestrator import (
    ConflictStrategy,
    ErrorStrategy,
    JobState,
    ObfuscationOrchestrator,
    ProcessResult,
    ProgressInfo,
)
from obfuscator.core.output_writer import WriteResult


_BASE_PHASE_MESSAGES = {
    "Validating inputs...",
    "Checking for file conflicts...",
    "Scanning files and extracting symbols...",
    "Building dependency graph...",
    "Pre-computing symbol table...",
}


@pytest.fixture
def orchestrator_instance(sample_config: ObfuscationConfig) -> ObfuscationOrchestrator:
    """Build an orchestrator without running heavy processor initialization.

    This local fixture overrides the shared fixture from ``tests/core/conftest.py``
    to avoid importing processor engine dependencies during unit-level progress
    callback tests.
    """
    sample_config.runtime_mode = "embedded"
    orchestrator = ObfuscationOrchestrator.__new__(ObfuscationOrchestrator)
    orchestrator._logger = MagicMock()
    orchestrator._project_root = None
    orchestrator._config = sample_config
    orchestrator._processed_engines = []
    orchestrator._current_state = JobState.PENDING
    orchestrator._conflict_strategy = ConflictStrategy(sample_config.conflict_strategy)
    orchestrator._conflict_decisions = {}
    orchestrator._cancellation_requested = False
    orchestrator._error_strategy = ErrorStrategy.CONTINUE
    orchestrator._start_time = None
    orchestrator._file_processing_times = []
    return orchestrator


class _GraphStub:
    """Minimal dependency graph stub that preserves a deterministic order."""

    def __init__(self, processing_order: list[Path]) -> None:
        self._processing_order = list(processing_order)
        self.nodes = set(processing_order)
        self.edges: dict[Path, set[Path]] = {}

    def get_processing_order(self) -> list[Path]:
        """Return files in insertion order."""
        return list(self._processing_order)


def _configure_progress_harness(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator: ObfuscationOrchestrator,
    files: list[Path],
    *,
    sleep_per_file: float = 0.0,
    cancel_after: int | None = None,
    register_runtime_engines: bool = False,
) -> list[Path]:
    """Patch heavy internals so tests can focus on progress callback semantics."""
    resolved_files = [file_path.resolve() for file_path in files]

    def _scan_and_extract_symbols(input_files, result):
        return set(resolved_files), {path: MagicMock() for path in resolved_files}

    monkeypatch.setattr(orchestrator, "_scan_and_extract_symbols", _scan_and_extract_symbols)
    monkeypatch.setattr(
        orchestrator,
        "_build_dependency_graph",
        lambda *args, **kwargs: _GraphStub(resolved_files),
    )
    monkeypatch.setattr(
        orchestrator,
        "_build_global_symbol_table",
        lambda *args, **kwargs: MagicMock(),
    )

    call_counter = {"count": 0}

    def _process_file_in_order(
        file_path,
        global_table,
        output_dir,
        config,
        output_writer,
        emit_log_callback=None,
    ):
        call_counter["count"] += 1
        if sleep_per_file > 0:
            time.sleep(sleep_per_file)

        if register_runtime_engines:
            language = "lua" if file_path.suffix.lower() in {".lua", ".luau"} else "python"
            fake_engine = MagicMock(name=f"{language}_engine")
            fake_engine.required_runtimes = {f"{language}_runtime_helper"}
            fake_engine.runtime_manager = MagicMock(name=f"{language}_runtime_manager")
            orchestrator._processed_engines.append(
                SimpleNamespace(engine=fake_engine, language=language)
            )

        if cancel_after is not None and call_counter["count"] >= cancel_after:
            orchestrator.request_cancellation()

        return ProcessResult(
            file_path=file_path,
            output_path=output_dir / file_path.name,
            success=True,
        )

    monkeypatch.setattr(orchestrator, "_process_file_in_order", _process_file_in_order)
    return resolved_files


def _run_orchestration(
    orchestrator: ObfuscationOrchestrator,
    files: list[Path],
    tmp_project: Path,
    sample_config: ObfuscationConfig,
    progress_callback: MagicMock | None = None,
):
    """Execute process_files using shared fixture defaults."""
    return orchestrator.process_files(
        input_files=files,
        output_dir=tmp_project / "output",
        config=sample_config.symbol_table_options,
        progress_callback=progress_callback,
        project_root=tmp_project / "src",
    )


def _step_messages_only(captured: list[ProgressInfo]) -> list[ProgressInfo]:
    """Return only callbacks generated by report_progress increments."""
    return [
        info
        for info in captured
        if info.message in _BASE_PHASE_MESSAGES or info.message.startswith("Processing ")
    ]


class TestProgressCallbackInvocation:
    """Validate progress callback invocation patterns across orchestration phases."""

    def test_progress_callback_called_for_each_phase(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_project: Path,
        sample_python_files: list[Path],
        sample_config: ObfuscationConfig,
        orchestrator_instance: ObfuscationOrchestrator,
        mock_progress_callback: MagicMock,
    ) -> None:
        """Progress callback should receive all base phase messages and file messages."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files)

        result = _run_orchestration(
            orchestrator_instance,
            sample_python_files,
            tmp_project,
            sample_config,
            mock_progress_callback,
        )

        assert result.success
        messages = [info.message for info in mock_progress_callback.captured]
        for phase_message in _BASE_PHASE_MESSAGES:
            assert phase_message in messages
        for file_path in sample_python_files:
            assert f"Processing {file_path.name}..." in messages

    def test_progress_callback_receives_progressinfo_objects(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_project: Path,
        sample_python_files: list[Path],
        sample_config: ObfuscationConfig,
        orchestrator_instance: ObfuscationOrchestrator,
        mock_progress_callback: MagicMock,
    ) -> None:
        """Every callback payload should be a ProgressInfo instance."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files)
        _run_orchestration(
            orchestrator_instance,
            sample_python_files,
            tmp_project,
            sample_config,
            mock_progress_callback,
        )

        assert mock_progress_callback.captured
        assert all(isinstance(info, ProgressInfo) for info in mock_progress_callback.captured)

    def test_progress_callback_not_called_when_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_project: Path,
        sample_python_files: list[Path],
        sample_config: ObfuscationConfig,
        orchestrator_instance: ObfuscationOrchestrator,
    ) -> None:
        """Processing should still succeed when no progress callback is provided."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files)
        result = _run_orchestration(
            orchestrator_instance,
            sample_python_files,
            tmp_project,
            sample_config,
            None,
        )

        assert result.success
        assert result.current_state == JobState.COMPLETED

    def test_progress_callback_invocation_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_project: Path,
        sample_python_files: list[Path],
        sample_config: ObfuscationConfig,
        orchestrator_instance: ObfuscationOrchestrator,
        mock_progress_callback: MagicMock,
    ) -> None:
        """Invocation count should include 5 phase steps, file steps, and completion log."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files)
        _run_orchestration(
            orchestrator_instance,
            sample_python_files,
            tmp_project,
            sample_config,
            mock_progress_callback,
        )

        runtime_extra = 1 if sample_config.runtime_mode == "hybrid" else 0
        expected_calls = 5 + len(sample_python_files) + 1 + runtime_extra
        assert mock_progress_callback.call_count == expected_calls


class TestProgressInfoAccuracy:
    """Verify ProgressInfo field accuracy for callback payloads."""

    def test_progressinfo_percentage_calculation(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """Percentage should match current_index / total_files for each callback."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files)
        _run_orchestration(orchestrator_instance, sample_python_files, tmp_project, sample_config, mock_progress_callback)
        for info in mock_progress_callback.captured:
            expected = (info.current_index / info.total_files) * 100 if info.total_files > 0 else 0.0
            assert info.percentage == pytest.approx(expected)

    def test_progressinfo_current_file_tracking(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """File processing callbacks should track current_file in processing order."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files)
        _run_orchestration(orchestrator_instance, sample_python_files, tmp_project, sample_config, mock_progress_callback)
        processing_infos = [info for info in mock_progress_callback.captured if info.message.startswith("Processing ")]
        assert [info.current_file for info in processing_infos] == [path.name for path in sample_python_files]

    def test_progressinfo_current_index_increments(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """Step callbacks emitted by report_progress should increment indices by one."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files)
        _run_orchestration(orchestrator_instance, sample_python_files, tmp_project, sample_config, mock_progress_callback)
        step_infos = _step_messages_only(mock_progress_callback.captured)
        assert [info.current_index for info in step_infos] == list(range(1, len(step_infos) + 1))

    def test_progressinfo_total_files_accuracy(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """total_files should remain constant at 5 + input_file_count throughout callbacks."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files)
        _run_orchestration(orchestrator_instance, sample_python_files, tmp_project, sample_config, mock_progress_callback)
        expected_total_steps = 5 + len(sample_python_files)
        assert all(info.total_files == expected_total_steps for info in mock_progress_callback.captured)

    def test_progressinfo_state_transitions(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """Callback states should include validating, analyzing, processing, and completed."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files)
        _run_orchestration(orchestrator_instance, sample_python_files, tmp_project, sample_config, mock_progress_callback)
        states = {info.current_state for info in mock_progress_callback.captured}
        assert {JobState.VALIDATING, JobState.ANALYZING, JobState.PROCESSING, JobState.COMPLETED}.issubset(states)

    def test_progressinfo_message_content(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """All callback messages should contain non-empty human-readable text."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files)
        _run_orchestration(orchestrator_instance, sample_python_files, tmp_project, sample_config, mock_progress_callback)
        assert all(info.message.strip() for info in mock_progress_callback.captured)

    def test_progressinfo_log_level_defaults_to_info(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """Non-runtime callbacks should default to info log level."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files)
        _run_orchestration(orchestrator_instance, sample_python_files, tmp_project, sample_config, mock_progress_callback)
        assert all(info.log_level == "info" for info in mock_progress_callback.captured)


class TestTimeEstimation:
    """Validate elapsed/remaining time fields and formatting helpers."""

    def test_elapsed_seconds_increases(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """Elapsed seconds should be monotonic as processing advances."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files, sleep_per_file=0.01)
        _run_orchestration(orchestrator_instance, sample_python_files, tmp_project, sample_config, mock_progress_callback)
        elapsed_values = [info.elapsed_seconds for info in mock_progress_callback.captured]
        assert elapsed_values == sorted(elapsed_values)
        assert any(value > 0 for value in elapsed_values)

    def test_elapsed_seconds_starts_at_zero(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """First callback should start near zero elapsed runtime."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files)
        _run_orchestration(orchestrator_instance, sample_python_files, tmp_project, sample_config, mock_progress_callback)
        first_elapsed = mock_progress_callback.captured[0].elapsed_seconds
        assert 0 <= first_elapsed < 0.25

    def test_estimated_remaining_none_initially(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """Estimated remaining time should start as None before any file completes."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files)
        _run_orchestration(orchestrator_instance, sample_python_files, tmp_project, sample_config, mock_progress_callback)
        assert mock_progress_callback.captured[0].estimated_remaining_seconds is None

    def test_estimated_remaining_calculated_after_first_file(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """After the first file finishes, later callbacks should include an ETA."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files, sleep_per_file=0.01)
        _run_orchestration(orchestrator_instance, sample_python_files, tmp_project, sample_config, mock_progress_callback)
        callbacks_after_first_file = [info for info in mock_progress_callback.captured if info.current_index > 6]
        assert any(info.estimated_remaining_seconds is not None for info in callbacks_after_first_file)

    def test_formatted_elapsed_property(self):
        """formatted_elapsed should render MM:SS correctly."""
        info = ProgressInfo(None, 1, 1, 100.0, 125.9, None, JobState.PROCESSING, "done")
        assert info.formatted_elapsed == "02:05"

    def test_formatted_remaining_property(self):
        """formatted_remaining should handle both unknown and known estimates."""
        info_unknown = ProgressInfo(None, 1, 1, 100.0, 5.0, None, JobState.PROCESSING, "step")
        info_known = ProgressInfo(None, 1, 1, 100.0, 5.0, 61.0, JobState.PROCESSING, "step")
        assert info_unknown.formatted_remaining == "Calculating..."
        assert info_known.formatted_remaining == "01:01"


class TestRuntimeLibraryProgressTracking:
    """Verify runtime generation progress callbacks and log levels in hybrid mode."""

    def test_runtime_generation_emits_progress(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """Hybrid runtime generation should emit start, created, and completion logs."""
        sample_config.runtime_mode = "hybrid"
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files[:1], register_runtime_engines=True)

        runtime_path = (tmp_project / "output") / "obf_runtime.py"
        with patch(
            "obfuscator.core.output_writer.OutputWriter.write_runtime_library",
            return_value=WriteResult(success=True, output_path=runtime_path, original_path=runtime_path),
        ):
            _run_orchestration(orchestrator_instance, sample_python_files[:1], tmp_project, sample_config, mock_progress_callback)

        messages = [info.message for info in mock_progress_callback.captured]
        assert "Generating hybrid runtime libraries..." in messages
        assert any("Runtime library created:" in message for message in messages)
        assert any("Hybrid runtime generation completed" in message for message in messages)

    def test_runtime_success_log_level(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """Runtime creation message should be emitted at success level."""
        sample_config.runtime_mode = "hybrid"
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files[:1], register_runtime_engines=True)

        runtime_path = (tmp_project / "output") / "obf_runtime.py"
        with patch(
            "obfuscator.core.output_writer.OutputWriter.write_runtime_library",
            return_value=WriteResult(success=True, output_path=runtime_path, original_path=runtime_path),
        ):
            _run_orchestration(orchestrator_instance, sample_python_files[:1], tmp_project, sample_config, mock_progress_callback)

        created_infos = [info for info in mock_progress_callback.captured if "Runtime library created:" in info.message]
        assert created_infos
        assert all(info.log_level == "success" for info in created_infos)

    def test_runtime_warning_on_skip(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """Skipped runtime writes should emit warning-level log entries."""
        sample_config.runtime_mode = "hybrid"
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files[:1], register_runtime_engines=True)

        runtime_path = (tmp_project / "output") / "obf_runtime.py"
        with patch(
            "obfuscator.core.output_writer.OutputWriter.write_runtime_library",
            return_value=WriteResult(success=True, output_path=None, original_path=runtime_path, conflict_resolution="skipped"),
        ):
            _run_orchestration(orchestrator_instance, sample_python_files[:1], tmp_project, sample_config, mock_progress_callback)

        skipped_infos = [info for info in mock_progress_callback.captured if "Runtime library skipped for python" in info.message]
        assert skipped_infos
        assert all(info.log_level == "warning" for info in skipped_infos)

    def test_runtime_error_on_failure(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """Runtime write failures should emit error-level log entries."""
        sample_config.runtime_mode = "hybrid"
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files[:1], register_runtime_engines=True)

        runtime_path = (tmp_project / "output") / "obf_runtime.py"
        with patch(
            "obfuscator.core.output_writer.OutputWriter.write_runtime_library",
            return_value=WriteResult(success=False, output_path=None, original_path=runtime_path, error="write failed"),
        ):
            _run_orchestration(orchestrator_instance, sample_python_files[:1], tmp_project, sample_config, mock_progress_callback)

        failed_infos = [info for info in mock_progress_callback.captured if "Failed to write runtime library for python" in info.message]
        assert failed_infos
        assert all(info.log_level == "error" for info in failed_infos)

    def test_runtime_generation_for_python_and_lua(self, monkeypatch, tmp_project, sample_python_files, sample_lua_files, sample_config, orchestrator_instance, mock_progress_callback):
        """Hybrid mode should attempt runtime generation for both Python and Lua."""
        sample_config.runtime_mode = "hybrid"
        mixed_files = [sample_python_files[0], sample_lua_files[0]]
        _configure_progress_harness(monkeypatch, orchestrator_instance, mixed_files, register_runtime_engines=True)

        def _runtime_result(*args, **kwargs):
            language = kwargs["language"]
            output_dir = kwargs["output_dir"]
            suffix = "py" if language == "python" else "lua"
            path = (output_dir or (tmp_project / "output")) / f"obf_runtime.{suffix}"
            return WriteResult(success=True, output_path=path, original_path=path)

        with patch("obfuscator.core.output_writer.OutputWriter.write_runtime_library", side_effect=_runtime_result) as patched_runtime:
            _run_orchestration(orchestrator_instance, mixed_files, tmp_project, sample_config, mock_progress_callback)

        called_languages = {call.kwargs["language"] for call in patched_runtime.call_args_list}
        assert called_languages == {"python", "lua"}


class TestProgressWithEdgeCases:
    """Cover edge behavior for callback delivery and progress accounting."""

    def test_progress_with_empty_file_list(self, tmp_project, sample_config, orchestrator_instance, mock_progress_callback):
        """Empty input should still emit validation progress then fail gracefully."""
        result = orchestrator_instance.process_files(
            input_files=[],
            output_dir=tmp_project / "output",
            config=sample_config.symbol_table_options,
            progress_callback=mock_progress_callback,
        )
        assert mock_progress_callback.captured
        assert result.current_state == JobState.FAILED

    def test_progress_with_single_file(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """Single-file runs should report accurate total step counts."""
        one_file = [sample_python_files[0]]
        _configure_progress_harness(monkeypatch, orchestrator_instance, one_file)
        _run_orchestration(orchestrator_instance, one_file, tmp_project, sample_config, mock_progress_callback)
        assert any(info.current_file == one_file[0].name for info in mock_progress_callback.captured)
        assert all(info.total_files == 6 for info in mock_progress_callback.captured)

    def test_progress_with_large_batch(self, monkeypatch, tmp_project, sample_config, orchestrator_instance, mock_progress_callback):
        """Large file sets should preserve bounded percentage and callback ordering."""
        large_files = []
        for index in range(120):
            file_path = (tmp_project / "src") / f"batch_{index}.py"
            file_path.write_text(f"value_{index} = {index}\n", encoding="utf-8")
            large_files.append(file_path)

        _configure_progress_harness(monkeypatch, orchestrator_instance, large_files)
        result = _run_orchestration(orchestrator_instance, large_files, tmp_project, sample_config, mock_progress_callback)
        assert result.success
        assert max(info.percentage for info in mock_progress_callback.captured) <= 100.0
        runtime_extra = 1 if sample_config.runtime_mode == "hybrid" else 0
        assert mock_progress_callback.call_count == 5 + len(large_files) + 1 + runtime_extra

    def test_progress_during_cancellation(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance, mock_progress_callback):
        """Cancellation during processing should emit CANCELLED state updates."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files, cancel_after=1)
        result = _run_orchestration(orchestrator_instance, sample_python_files, tmp_project, sample_config, mock_progress_callback)
        assert result.current_state == JobState.CANCELLED
        assert any(info.current_state == JobState.CANCELLED for info in mock_progress_callback.captured)
        assert any("Job cancelled by user" in info.message for info in mock_progress_callback.captured)

    def test_progress_callback_exception_handling(self, monkeypatch, tmp_project, sample_python_files, sample_config, orchestrator_instance):
        """Progress callback exceptions should propagate to the caller."""
        _configure_progress_harness(monkeypatch, orchestrator_instance, sample_python_files)

        state = {"raised": False}

        def flaky_callback(progress_info: ProgressInfo) -> None:
            if not state["raised"]:
                state["raised"] = True
                raise RuntimeError("intentional callback failure")

        with pytest.raises(RuntimeError, match="intentional callback failure"):
            _run_orchestration(
                orchestrator_instance,
                sample_python_files,
                tmp_project,
                sample_config,
                flaky_callback,
            )
