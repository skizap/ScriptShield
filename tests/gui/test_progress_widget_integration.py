"""Integration tests for ProgressWidget state, rendering, and callback behavior."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from obfuscator.core.orchestrator import JobState, ProgressInfo

if TYPE_CHECKING:
    from obfuscator.gui.widgets.progress_widget import ProgressWidget


def _flush_events() -> None:
    """Flush pending GUI events to stabilize assertions."""
    QApplication.processEvents()
    QTest.qWait(5)


def _log_labels(progress_widget: ProgressWidget) -> list:
    """Return all QLabel log entries, excluding the terminal layout stretch."""
    labels = []
    for index in range(progress_widget.log_layout.count() - 1):
        item = progress_widget.log_layout.itemAt(index)
        widget = item.widget() if item is not None else None
        if widget is not None:
            labels.append(widget)
    return labels


def _latest_log(progress_widget: ProgressWidget):
    """Return the newest rendered log label."""
    labels = _log_labels(progress_widget)
    assert labels
    return labels[-1]


def _on_progress_like_main_window(
    progress_widget: ProgressWidget,
    progress_info: ProgressInfo,
    *,
    total_files: int,
) -> None:
    """Apply ProgressInfo to ProgressWidget using MainWindow callback logic."""
    batch_size = 100
    batch_threshold = 1000

    progress_widget.set_progress(int(progress_info.percentage))
    progress_widget.set_state(progress_info.current_state.name)
    progress_widget.set_time_info(
        progress_info.elapsed_seconds,
        progress_info.estimated_remaining_seconds,
    )
    progress_widget.set_current_file(progress_info.current_file)

    if total_files > batch_threshold:
        non_file_step_count = max(progress_info.total_files - total_files, 0)
        file_index = max(progress_info.current_index - non_file_step_count, 0)
        total_batches = (total_files + batch_size - 1) // batch_size
        current_batch = min((file_index // batch_size) + 1, total_batches)
        progress_widget.set_batch_info(current_batch, total_batches)
    else:
        progress_widget.set_batch_info(1, 1)

    progress_widget.add_log_entry(
        progress_info.message,
        progress_info.log_level or "info",
    )
    _flush_events()


class TestProgressWidgetStateUpdates:
    """Verify state label updates and color coding."""

    def test_set_state_updates_label(self, progress_widget: ProgressWidget) -> None:
        """set_state should update label text with state name."""
        progress_widget.set_state("PROCESSING")
        assert progress_widget.state_label.text() == "Current State: PROCESSING"

    def test_set_state_color_coding(self, progress_widget: ProgressWidget) -> None:
        """set_state should map state names to expected color values."""
        expected_colors = {
            "PENDING": "#888888",
            "VALIDATING": "#FFA500",
            "ANALYZING": "#0088FF",
            "PROCESSING": "#00AA00",
            "COMPLETED": "#00CC00",
            "FAILED": "#CC0000",
            "CANCELLED": "#888888",
        }
        for state_name, color in expected_colors.items():
            progress_widget.set_state(state_name)
            assert color in progress_widget.state_label.styleSheet()

    def test_get_state_returns_current_state(self, progress_widget: ProgressWidget) -> None:
        """get_state should return the same value shown in state label."""
        progress_widget.set_state("ANALYZING")
        assert progress_widget.get_state() == "Current State: ANALYZING"

    def test_initial_state_is_pending(self, progress_widget: ProgressWidget) -> None:
        """Widget should start in the PENDING state."""
        assert progress_widget.get_state() == "Current State: PENDING"


class TestProgressBarUpdates:
    """Validate progress bar value updates and formatting."""

    def test_set_progress_updates_value(self, progress_widget: ProgressWidget) -> None:
        """set_progress should update the progress bar's numeric value."""
        progress_widget.set_progress(47)
        assert progress_widget.progress_bar.value() == 47

    def test_get_progress_returns_value(self, progress_widget: ProgressWidget) -> None:
        """get_progress should return the latest progress value."""
        progress_widget.set_progress(63)
        assert progress_widget.get_progress() == 63

    def test_progress_increments_sequentially(self, progress_widget: ProgressWidget) -> None:
        """Sequential updates should be reflected exactly."""
        for value in (0, 10, 25, 50, 75, 100):
            progress_widget.set_progress(value)
            assert progress_widget.get_progress() == value

    def test_progress_bar_format_shows_percentage(self, progress_widget: ProgressWidget) -> None:
        """Progress bar should display percentage formatting token."""
        assert progress_widget.progress_bar.format() == "%p%"


class TestLogEntryRendering:
    """Validate log entry creation, iconography, and style behavior."""

    def test_add_log_entry_creates_label(self, progress_widget: ProgressWidget) -> None:
        """add_log_entry should append a new label to the log layout."""
        before = len(_log_labels(progress_widget))
        progress_widget.add_log_entry("Started", "info")
        _flush_events()
        after = len(_log_labels(progress_widget))
        assert after == before + 1

    def test_log_entry_success_icon(self, progress_widget: ProgressWidget) -> None:
        """Success log entries should render with a check icon."""
        progress_widget.add_log_entry("Completed", "success")
        _flush_events()
        assert "✓" in _latest_log(progress_widget).text()

    def test_log_entry_warning_icon(self, progress_widget: ProgressWidget) -> None:
        """Warning log entries should render with a warning icon."""
        progress_widget.add_log_entry("Potential issue", "warning")
        _flush_events()
        assert "⚠" in _latest_log(progress_widget).text()

    def test_log_entry_error_icon(self, progress_widget: ProgressWidget) -> None:
        """Error log entries should render with an error icon."""
        progress_widget.add_log_entry("Failed", "error")
        _flush_events()
        assert "✗" in _latest_log(progress_widget).text()

    def test_log_entry_info_no_icon(self, progress_widget: ProgressWidget) -> None:
        """Info log entries should not include success/warning/error icons."""
        progress_widget.add_log_entry("Informational", "info")
        _flush_events()
        text = _latest_log(progress_widget).text()
        assert "Informational" in text
        assert "✓" not in text
        assert "⚠" not in text
        assert "✗" not in text

    def test_log_entry_timestamp_format(self, progress_widget: ProgressWidget) -> None:
        """Log entries should include a ``[HH:MM:SS]`` timestamp prefix."""

        class _FixedDateTime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 1, 2, 3, 4, 5)

        with patch("datetime.datetime", _FixedDateTime):
            progress_widget.add_log_entry("Timestamp test", "info")
        _flush_events()

        text = _latest_log(progress_widget).text()
        assert text.startswith("[03:04:05]")
        assert re.match(r"^\[\d{2}:\d{2}:\d{2}\]", text)

    def test_log_entry_styling(self, progress_widget: ProgressWidget, widget_style_getter) -> None:
        """Each log level should use the expected stylesheet snippet."""
        progress_widget.add_log_entry("ok", "success")
        progress_widget.add_log_entry("warn", "warning")
        progress_widget.add_log_entry("bad", "error")
        progress_widget.add_log_entry("note", "info")
        _flush_events()

        labels = _log_labels(progress_widget)
        assert labels[-4].styleSheet() == widget_style_getter("log_entry_success")
        assert labels[-3].styleSheet() == widget_style_getter("log_entry_warning")
        assert labels[-2].styleSheet() == widget_style_getter("log_entry_error")
        assert labels[-1].styleSheet() == widget_style_getter("log_entry_info")

    def test_clear_logs_removes_entries(self, progress_widget: ProgressWidget) -> None:
        """clear_logs should remove all labels and reset entry counter."""
        progress_widget.add_log_entry("one", "info")
        progress_widget.add_log_entry("two", "warning")
        _flush_events()

        assert len(_log_labels(progress_widget)) == 2
        progress_widget.clear_logs()
        _flush_events()

        assert len(_log_labels(progress_widget)) == 0
        assert progress_widget._log_entry_counter == 0
        assert progress_widget.log_layout.count() == 1


class TestCurrentFileDisplay:
    """Validate current-file label behavior across path formats."""

    def test_set_current_file_updates_label(self, progress_widget: ProgressWidget) -> None:
        """set_current_file should show explicit filenames."""
        progress_widget.set_current_file("module.py")
        assert progress_widget.current_file_label.text() == "Current: module.py"

    def test_set_current_file_extracts_basename(self, progress_widget: ProgressWidget) -> None:
        """set_current_file should collapse POSIX paths to basename."""
        progress_widget.set_current_file("/tmp/project/pkg/module.py")
        assert progress_widget.current_file_label.text() == "Current: module.py"

    def test_set_current_file_none_shows_placeholder(self, progress_widget: ProgressWidget) -> None:
        """set_current_file(None) should restore placeholder text."""
        progress_widget.set_current_file(None)
        assert progress_widget.current_file_label.text() == "Current: --"

    def test_set_current_file_handles_backslashes(self, progress_widget: ProgressWidget) -> None:
        """set_current_file should normalize Windows-style separators."""
        windows_path = str(Path("C:/repo/src/module.lua")).replace("/", "\\")
        progress_widget.set_current_file(windows_path)
        assert progress_widget.current_file_label.text() == "Current: module.lua"


class TestBatchProgressDisplay:
    """Validate batch label content and visibility rules."""

    def test_set_batch_info_shows_label(self, progress_widget: ProgressWidget) -> None:
        """Multiple batches should show descriptive batch text."""
        progress_widget.show_progress()
        progress_widget.set_batch_info(2, 5)
        assert progress_widget.batch_label.isVisible()
        assert progress_widget.batch_label.text() == "Processing batch 2 of 5"

    def test_set_batch_info_visibility(self, progress_widget: ProgressWidget) -> None:
        """Batch label should be visible whenever total_batches > 1."""
        progress_widget.show_progress()
        progress_widget.set_batch_info(1, 3)
        assert progress_widget.batch_label.isVisible()

    def test_set_batch_info_hidden_for_single_batch(self, progress_widget: ProgressWidget) -> None:
        """Batch label should hide when only one batch exists."""
        progress_widget.set_batch_info(1, 1)
        assert not progress_widget.batch_label.isVisible()
        assert progress_widget.batch_label.text() == ""

    def test_set_batch_info_hidden_initially(self, progress_widget: ProgressWidget) -> None:
        """Batch label should start hidden by default."""
        assert not progress_widget.batch_label.isVisible()


class TestTimeDisplayUpdates:
    """Validate elapsed/remaining labels and internal formatter logic."""

    def test_set_time_info_elapsed(self, progress_widget: ProgressWidget) -> None:
        """set_time_info should format elapsed seconds as MM:SS."""
        progress_widget.set_time_info(75.2, None)
        assert progress_widget.time_label.text() == "Elapsed: 01:15"

    def test_set_time_info_eta_calculating(self, progress_widget: ProgressWidget) -> None:
        """Unknown ETA should render as Calculating..."""
        progress_widget.set_time_info(10.0, None)
        assert progress_widget.eta_label.text() == "Remaining: Calculating..."

    def test_set_time_info_eta_formatted(self, progress_widget: ProgressWidget) -> None:
        """Known ETA should be rendered as MM:SS."""
        progress_widget.set_time_info(10.0, 130.0)
        assert progress_widget.eta_label.text() == "Remaining: 02:10"

    def test_format_time_helper(self, progress_widget: ProgressWidget) -> None:
        """_format_time should return zero-padded MM:SS values."""
        assert progress_widget._format_time(0) == "00:00"
        assert progress_widget._format_time(59) == "00:59"
        assert progress_widget._format_time(125) == "02:05"


class TestProgressWidgetIntegrationWithOrchestrator:
    """Exercise ProgressInfo-driven widget updates using orchestration semantics."""

    def test_full_workflow_simulation(self, progress_widget: ProgressWidget, progress_info_factory) -> None:
        """A representative workflow should update progress, state, logs, and file label."""
        total_files = 3
        total_steps = 5 + total_files
        sequence = [
            progress_info_factory(
                current_file=None,
                current_index=1,
                total_files=total_steps,
                percentage=12.5,
                elapsed_seconds=0.1,
                estimated_remaining_seconds=None,
                current_state=JobState.VALIDATING,
                message="Validating inputs...",
                log_level="info",
            ),
            progress_info_factory(
                current_file="main.py",
                current_index=6,
                total_files=total_steps,
                percentage=75.0,
                elapsed_seconds=1.2,
                estimated_remaining_seconds=0.5,
                current_state=JobState.PROCESSING,
                message="Processing main.py...",
                log_level="info",
            ),
            progress_info_factory(
                current_file=None,
                current_index=total_steps,
                total_files=total_steps,
                percentage=100.0,
                elapsed_seconds=1.6,
                estimated_remaining_seconds=0.0,
                current_state=JobState.COMPLETED,
                message="Job completed",
                log_level="success",
            ),
        ]

        for info in sequence:
            _on_progress_like_main_window(progress_widget, info, total_files=total_files)

        assert progress_widget.get_progress() == 100
        assert progress_widget.get_state() == "Current State: COMPLETED"
        assert progress_widget.current_file_label.text() == "Current: --"
        assert len(_log_labels(progress_widget)) == len(sequence)

    def test_large_project_batch_display(self, progress_widget: ProgressWidget, progress_info_factory) -> None:
        """Large projects should show bounded batch values that never overflow."""
        progress_widget.show_progress()
        total_files = 1200
        total_steps = total_files + 5
        final_file_info = progress_info_factory(
            current_file="file_1199.py",
            current_index=total_steps,
            total_files=total_steps,
            percentage=100.0,
            elapsed_seconds=30.0,
            estimated_remaining_seconds=0.0,
            current_state=JobState.PROCESSING,
            message="Processing file_1199.py...",
            log_level="info",
        )

        _on_progress_like_main_window(progress_widget, final_file_info, total_files=total_files)

        assert progress_widget.batch_label.isVisible()
        assert progress_widget.batch_label.text() == "Processing batch 12 of 12"

    def test_runtime_generation_log_entries(self, progress_widget: ProgressWidget, progress_info_factory) -> None:
        """Runtime-generation progress logs should render level-specific entries."""
        runtime_start = progress_info_factory(
            current_file=None,
            current_index=8,
            total_files=8,
            percentage=100.0,
            elapsed_seconds=2.0,
            estimated_remaining_seconds=0.0,
            current_state=JobState.COMPLETED,
            message="Generating hybrid runtime libraries...",
            log_level="info",
        )
        runtime_success = progress_info_factory(
            current_file=None,
            current_index=8,
            total_files=8,
            percentage=100.0,
            elapsed_seconds=2.1,
            estimated_remaining_seconds=0.0,
            current_state=JobState.COMPLETED,
            message="Runtime library created: obf_runtime.py",
            log_level="success",
        )

        _on_progress_like_main_window(progress_widget, runtime_start, total_files=1)
        _on_progress_like_main_window(progress_widget, runtime_success, total_files=1)

        latest = _latest_log(progress_widget)
        assert "Runtime library created: obf_runtime.py" in latest.text()
        assert "✓" in latest.text()

    def test_error_handling_log_entries(self, progress_widget: ProgressWidget, progress_info_factory, widget_style_getter) -> None:
        """Error progress logs should render the error icon and styling."""
        error_info = progress_info_factory(
            current_file="main.py",
            current_index=7,
            total_files=8,
            percentage=87.5,
            elapsed_seconds=1.5,
            estimated_remaining_seconds=0.2,
            current_state=JobState.PROCESSING,
            message="Error processing main.py: Parse error",
            log_level="error",
        )

        _on_progress_like_main_window(progress_widget, error_info, total_files=3)

        latest = _latest_log(progress_widget)
        assert "✗" in latest.text()
        assert latest.styleSheet() == widget_style_getter("log_entry_error")

    def test_cancellation_workflow(
        self,
        progress_widget: ProgressWidget,
        progress_info_factory,
        mock_orchestrator: MagicMock,
    ) -> None:
        """Cancellation should propagate through cancel signal and state/log updates."""
        progress_widget.cancel_requested.connect(mock_orchestrator.request_cancellation)
        progress_widget.show_progress()
        progress_widget.cancel_button.click()

        cancellation_info = progress_info_factory(
            current_file=None,
            current_index=6,
            total_files=8,
            percentage=75.0,
            elapsed_seconds=1.3,
            estimated_remaining_seconds=None,
            current_state=JobState.CANCELLED,
            message="Job cancelled by user",
            log_level="warning",
        )
        _on_progress_like_main_window(progress_widget, cancellation_info, total_files=3)

        assert mock_orchestrator.request_cancellation.call_count == 1
        assert progress_widget.get_state() == "Current State: CANCELLED"
        assert "⚠" in _latest_log(progress_widget).text()


class TestProgressWidgetReset:
    """Verify reset behavior for state, visibility, and control defaults."""

    def test_reset_clears_all_state(self, progress_widget: ProgressWidget) -> None:
        """reset should restore default labels, hide batch, and clear logs."""
        progress_widget.show_progress()
        progress_widget.set_progress(66)
        progress_widget.set_state("PROCESSING")
        progress_widget.set_time_info(95.0, 20.0)
        progress_widget.set_current_file("script.py")
        progress_widget.set_batch_info(2, 4)
        progress_widget.add_log_entry("working", "info")

        progress_widget.reset()
        _flush_events()

        assert progress_widget.get_progress() == 0
        assert progress_widget.get_state() == "Current State: PENDING"
        assert progress_widget.time_label.text() == "Elapsed: 00:00"
        assert progress_widget.eta_label.text() == "Remaining: --:--"
        assert progress_widget.current_file_label.text() == "Current: --"
        assert progress_widget.batch_label.text() == ""
        assert not progress_widget.batch_label.isVisible()
        assert len(_log_labels(progress_widget)) == 0

    def test_reset_hides_widget(self, progress_widget: ProgressWidget) -> None:
        """reset should hide the progress widget after cleanup."""
        progress_widget.show_progress()
        assert progress_widget.is_visible()

        progress_widget.reset()

        assert not progress_widget.is_visible()
        assert not progress_widget.isVisible()

    def test_reset_enables_cancel_button(self, progress_widget: ProgressWidget) -> None:
        """reset should re-enable cancel button if it was previously disabled."""
        progress_widget.show_progress()
        progress_widget.cancel_button.setEnabled(False)

        progress_widget.reset()

        assert progress_widget.cancel_button.isEnabled()
