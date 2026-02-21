"""
Main window implementation for the Python & Lua Obfuscator application.

This module provides the MainWindow class which serves as the primary
application window containing all GUI widgets and layouts.
"""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QGridLayout,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from obfuscator.gui.widgets import (
    ActionWidget,
    CancellationConfirmDialog,
    ErrorReportDialog,
    FileSelectionWidget,
    InfoPanelWidget,
    OutputWidget,
    ProfileWidget,
    ProgressWidget,
    ResumeCheckpointDialog,
    SecurityConfigWidget,
    StartConfirmationDialog,
)

from obfuscator.gui.styles.stylesheet import get_application_stylesheet
from obfuscator.utils.error_formatting import extract_line_column, parse_error
from obfuscator.utils.logger import get_logger
from obfuscator.utils.path_utils import get_platform, normalize_path
from obfuscator.core.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MEMORY_THRESHOLD_PERCENT,
    DEFAULT_MULTIPROCESSING_THRESHOLD,
    ObfuscationConfig,
)
from obfuscator.core import CheckpointManager
from obfuscator.core.orchestrator import ObfuscationOrchestrator, JobState, ErrorStrategy, ProgressInfo

# Module-level logger
logger = get_logger("obfuscator.gui.main_window")


class MainWindow(QMainWindow):
    """
    Main application window for the Python & Lua Obfuscator.

    This window serves as the container for all GUI components including
    file selection, configuration panels, and obfuscation controls.
    Supports Python, Lua, and Luau files with mixed-language detection.

    Attributes:
        DEFAULT_WIDTH: Default window width in pixels.
        DEFAULT_HEIGHT: Default window height in pixels.
        MIN_WIDTH: Minimum window width in pixels.
        MIN_HEIGHT: Minimum window height in pixels.
        WINDOW_TITLE: Window title text.
    """

    DEFAULT_WIDTH = 1200
    DEFAULT_HEIGHT = 800
    MIN_WIDTH = 800
    MIN_HEIGHT = 600
    WINDOW_TITLE = "Python & Lua Obfuscator"

    def __init__(self) -> None:
        """Initialize the main window with all components."""
        super().__init__()

        logger.info(f"Initializing MainWindow on {get_platform()}")

        self._setup_window_properties()
        self._setup_icon()
        self._setup_central_widget()
        self._connect_signals()
        self._center_window()
        self._current_orchestrator: ObfuscationOrchestrator | None = None
        self._active_total_files: int = 0

        logger.info("MainWindow initialization complete")

    def _setup_window_properties(self) -> None:
        """Configure window title, size, and constraints."""
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)

        # Apply global application stylesheet
        self.setStyleSheet(get_application_stylesheet())

        logger.info("Window properties configured")

    def _setup_icon(self) -> None:
        """Set up the application window icon."""
        # Define potential icon paths based on platform
        platform = get_platform()
        if platform == "windows":
            icon_filename = "app.ico"
        else:
            icon_filename = "app.png"

        # Build icon path relative to project root (parent of src/obfuscator/gui)
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        icon_path = project_root / "resources" / "icons" / icon_filename
        icon_path = normalize_path(icon_path)

        if icon_path.exists():
            icon = QIcon(str(icon_path))
            self.setWindowIcon(icon)
            logger.info(f"Application icon loaded from {icon_path}")
        else:
            logger.warning(
                f"Application icon not found at {icon_path} - continuing without icon"
            )

    def _setup_central_widget(self) -> None:
        """Set up the central widget with 2-column grid layout."""
        # Create main container to hold both grid and progress widget
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(16)

        # Create grid widget
        central_widget = QWidget()
        central_widget.setProperty("data-element-id", "dashboard-grid")

        # Create main grid layout
        grid = QGridLayout(central_widget)
        grid.setContentsMargins(16, 16, 16, 16)
        grid.setSpacing(16)

        # Left column container
        left_column = QWidget()
        left_column.setProperty("data-element-id", "dashboard-left-column")
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Add file selection widget to left column
        self.file_selection = FileSelectionWidget()
        left_layout.addWidget(self.file_selection)

        # Add security configuration widget to left column
        self.security_config = SecurityConfigWidget()
        self.security_config.set_file_selection_widget(self.file_selection)
        left_layout.addWidget(self.security_config)

        left_layout.addStretch()

        # Right column container
        right_column = QWidget()
        right_column.setProperty("data-element-id", "dashboard-right-column")
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        # Profile widget
        self.profile_widget = ProfileWidget()
        right_layout.addWidget(self.profile_widget)

        # Output widget
        self.output_widget = OutputWidget()
        right_layout.addWidget(self.output_widget)

        # Action widget
        self.action_widget = ActionWidget()
        right_layout.addWidget(self.action_widget)

        # Info panel widget
        self.info_panel = InfoPanelWidget()
        right_layout.addWidget(self.info_panel)

        right_layout.addStretch()

        # Add columns to grid with equal stretch
        grid.addWidget(left_column, 0, 0)
        grid.addWidget(right_column, 0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # Add grid to main container
        main_layout.addWidget(central_widget)

        # Create and add progress widget
        self.progress_widget = ProgressWidget()
        main_layout.addWidget(self.progress_widget)

        self.setCentralWidget(main_container)
        logger.info(
            "Central widget with 2-column grid layout setup complete "
            "(file selection and security config widgets added)"
        )

    def _center_window(self) -> None:
        """Center the window on the primary screen."""
        screen = QApplication.primaryScreen()
        if screen is None:
            logger.warning("Could not get primary screen - skipping window centering")
            return

        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        window_width = self.width()
        window_height = self.height()

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.move(x, y)
        logger.info(f"Window centered at position ({x}, {y})")

    def _connect_signals(self) -> None:
        """Connect widget signals to handlers."""
        # Connect file selection changes to enable/disable start button
        self.file_selection.files_changed.connect(self._on_files_changed)

        # Connect action widget start button
        self.action_widget.start_clicked.connect(self._on_start_obfuscation)

        # Connect profile widget signals
        self.profile_widget.profile_save_requested.connect(self._on_profile_save)
        self.profile_widget.profile_load_requested.connect(self._on_profile_load)

        # Connect progress widget cancel signal
        self.progress_widget.cancel_requested.connect(self._on_cancel_obfuscation)

        logger.debug("Widget signals connected")

    def _on_files_changed(self, files: list) -> None:
        """Handle file selection changes - enable/disable start button and trigger output auto-suggestion."""
        has_files = len(files) > 0
        self.action_widget.set_enabled(has_files)
        self.output_widget.suggest_output_path(files)
        logger.debug(f"Files changed, count: {len(files)}, start enabled: {has_files}, triggering output auto-suggestion")

    def _format_error_for_log(
        self,
        error: str,
        fallback_file: str | Path | None = None,
    ) -> tuple[str, str]:
        """Format an error string for progress logs and infer log level by type."""
        parsed_error = parse_error(error)
        if parsed_error is not None:
            error_type = str(parsed_error.get("error_type") or "Error")
            file_name = Path(str(parsed_error.get("file_path") or "unknown")).name
            line = int(parsed_error.get("line") or 0)
            column = int(parsed_error.get("column") or 0)
            message = str(parsed_error.get("message") or "Unknown error")
            formatted = f"[ERROR] {file_name}:{line}:{column} - {error_type}: {message}"

            if error_type in {"ParseError", "SyntaxError", "IndentationError", "TabError"}:
                return formatted, "warning"
            return formatted, "error"

        line, column = extract_line_column(error)
        fallback_name = Path(str(fallback_file)).name if fallback_file else None

        if fallback_name and line is not None and column is not None:
            return f"[ERROR] {fallback_name}:{line}:{column} - {error}", "error"
        if fallback_name:
            return f"[ERROR] {fallback_name} - {error}", "error"
        if line is not None and column is not None:
            return f"[ERROR] line {line}, column {column} - {error}", "error"
        return f"[ERROR] {error}", "error"

    def _on_start_obfuscation(self) -> None:
        """Handle start obfuscation button click.

        Initiates the obfuscation workflow using the ObfuscationOrchestrator.
        The orchestrator handles dependency analysis, symbol table construction,
        and processes files in topological order for consistent cross-file
        symbol references.
        """
        files = self.file_selection.get_files()
        output_path = self.output_widget.get_output_path()
        security_config = self.security_config.get_config()

        files_with_languages = self.file_selection.get_files_with_languages()
        language = "lua"
        if files_with_languages:
            normalized_languages = {
                str(file_language).lower()
                for file_language in files_with_languages.values()
            }
            language = "python" if "python" in normalized_languages else "lua"
        elif any(Path(file_path).suffix.lower() in {".py", ".pyw"} for file_path in files):
            language = "python"

        orchestrator_config = ObfuscationConfig.from_gui_config(
            name="active-session",
            preset=security_config.get("preset"),
            features=security_config.get("features", {}),
            language=language,
            enable_multiprocessing=security_config.get("enable_multiprocessing", True),
            max_workers=security_config.get("max_workers", DEFAULT_MAX_WORKERS),
            batch_size=security_config.get("batch_size", DEFAULT_BATCH_SIZE),
            multiprocessing_threshold=security_config.get(
                "multiprocessing_threshold",
                DEFAULT_MULTIPROCESSING_THRESHOLD,
            ),
            memory_threshold_percent=security_config.get(
                "memory_threshold_percent",
                DEFAULT_MEMORY_THRESHOLD_PERCENT,
            ),
        )
        orchestrator_config.runtime_mode = security_config.get("runtime_mode", "hybrid")

        logger.info(
            f"Start obfuscation requested: {len(files)} files, "
            f"output: {output_path}, preset: {security_config.get('preset')}, "
            f"multiprocessing: {orchestrator_config.enable_multiprocessing}, "
            f"max_workers: {orchestrator_config.max_workers}, "
            f"batch_size: {orchestrator_config.batch_size}, "
            f"parallel_threshold: {orchestrator_config.multiprocessing_threshold}, "
            f"memory_threshold_percent: {orchestrator_config.memory_threshold_percent}"
        )

        output_dir = Path(output_path) if output_path else Path.cwd() / "obfuscated"
        preset = security_config.get("preset", "Unknown")
        runtime_mode = security_config.get("runtime_mode", "hybrid")

        checkpoint_path = CheckpointManager.find_latest_checkpoint(output_dir)
        resume_mode = False

        if checkpoint_path is not None:
            temp_mgr = CheckpointManager(output_dir / ".checkpoints")
            try:
                checkpoint_data = temp_mgr.restore_checkpoint(checkpoint_path)
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"Failed to restore checkpoint: {e}")
                ckpt_session_id = checkpoint_path.parent.name if checkpoint_path else ""
                temp_mgr.cleanup_checkpoints(ckpt_session_id)
                checkpoint_path = None
                resume_mode = False
            else:
                ckpt_timestamp = checkpoint_data.get("timestamp", "Unknown")
                ckpt_progress = checkpoint_data.get("progress", {})
                files_completed = ckpt_progress.get("files_completed", 0)
                total_files_ckpt = ckpt_progress.get("total_files", 0)
                ckpt_session_id = checkpoint_data.get("session_id", "")

                resume_dialog = ResumeCheckpointDialog(
                    timestamp=ckpt_timestamp,
                    files_completed=files_completed,
                    total_files=total_files_ckpt,
                    parent=self,
                )
                if (
                    resume_dialog.exec() == QDialog.DialogCode.Accepted
                    and resume_dialog.get_user_decision()
                ):
                    resume_mode = True
                else:
                    temp_mgr.cleanup_checkpoints(ckpt_session_id)

        if not resume_mode:
            dialog = StartConfirmationDialog(
                file_count=len(files),
                preset=preset,
                output_path=output_dir,
                runtime_mode=runtime_mode.capitalize(),
                parent=self,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.get_user_decision():
                self.progress_widget.add_log_entry("Obfuscation cancelled by user", "warning")
                self.action_widget.set_enabled(True)
                return

        # Reset and show progress widget
        self.progress_widget.reset()
        self.progress_widget.show_progress()
        self.progress_widget.add_log_entry("Starting obfuscation...", "info")

        # Disable start button during processing
        self.action_widget.set_enabled(False)

        # Convert file paths to Path objects
        input_files = [Path(f) for f in files]
        self._active_total_files = len(files)

        # Early conflict detection before creating main orchestrator
        conflict_result = None
        temp_orchestrator = None
        if not resume_mode:
            temp_orchestrator = ObfuscationOrchestrator(config=orchestrator_config)
            conflict_result = temp_orchestrator.detect_conflicts(input_files, output_dir)

            if conflict_result.has_conflicts:
                from obfuscator.gui.widgets import ConflictResolutionDialog
                dialog = ConflictResolutionDialog(conflict_result.conflicts, parent=self)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    strategy = dialog.get_selected_strategy()
                    if strategy:
                        temp_orchestrator.set_conflict_strategy(strategy)
                        self.progress_widget.add_log_entry(
                            f"Conflict resolution: {strategy.value}", "info"
                        )
                    else:
                        # User accepted but no strategy selected - cancel
                        self.progress_widget.add_log_entry(
                            "Obfuscation cancelled - no conflict resolution selected", "warning"
                        )
                        self.action_widget.set_enabled(True)
                        return
                else:
                    # User cancelled the dialog
                    self.progress_widget.add_log_entry(
                        "Obfuscation cancelled by user", "warning"
                    )
                    self.action_widget.set_enabled(True)
                    return

        total_files = len(files)
        BATCH_SIZE = 100
        BATCH_THRESHOLD = 1000

        # Define progress callback for GUI updates
        def on_progress(progress_info: ProgressInfo) -> None:
            self.progress_widget.set_progress(int(progress_info.percentage))
            self.progress_widget.set_state(progress_info.current_state.name)
            self.progress_widget.set_time_info(
                progress_info.elapsed_seconds,
                progress_info.estimated_remaining_seconds,
            )
            self.progress_widget.set_current_file(progress_info.current_file)

            if total_files > BATCH_THRESHOLD:
                non_file_step_count = max(progress_info.total_files - total_files, 0)
                file_index = max(progress_info.current_index - non_file_step_count, 0)
                total_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE
                current_batch = min((file_index // BATCH_SIZE) + 1, total_batches)
                self.progress_widget.set_batch_info(current_batch, total_batches)
            else:
                self.progress_widget.set_batch_info(1, 1)

            self.progress_widget.add_log_entry(
                progress_info.message,
                progress_info.log_level or "info",
            )

            # Process events to keep GUI responsive
            QApplication.processEvents()

        # Define error callback for handling file processing errors
        def on_error(file_path: Path, errors: list[str]) -> bool:
            """Handle file processing error by showing error dialog to user.

            Args:
                file_path: Path to the file that failed processing
                errors: List of error messages from the processing failure

            Returns:
                True to continue processing remaining files, False to stop
            """
            from obfuscator.gui.widgets import ErrorHandlingDialog
            
            dialog = ErrorHandlingDialog(file_path, errors, parent=self)
            result = dialog.exec()
            
            # Get user decision
            continue_processing = dialog.get_user_decision()
            
            # Log decision to progress widget
            decision_msg = (
                f"User chose to {'continue' if continue_processing else 'stop'} "
                f"after error in {file_path.name}"
            )
            log_level = "info" if continue_processing else "warning"
            self.progress_widget.add_log_entry(decision_msg, log_level)
            
            logger.info(f"Error handling decision for {file_path.name}: {decision_msg}")
            
            return continue_processing

        try:
            # Create orchestrator and apply conflict strategy if set
            orchestrator = ObfuscationOrchestrator(config=orchestrator_config)
            self._current_orchestrator = orchestrator
            if (temp_orchestrator is not None
                and conflict_result is not None
                and conflict_result.has_conflicts):
                # Copy the strategy from temp orchestrator
                orchestrator.set_conflict_strategy(temp_orchestrator._conflict_strategy)

            if resume_mode:
                result = orchestrator.resume_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    input_files=input_files,
                    output_dir=output_dir,
                    config=orchestrator_config.symbol_table_options,
                    progress_callback=on_progress,
                    error_callback=on_error,
                    error_strategy=ErrorStrategy.ASK,
                )
            else:
                result = orchestrator.process_files(
                    input_files=input_files,
                    output_dir=output_dir,
                    config=orchestrator_config.symbol_table_options,
                    progress_callback=on_progress,
                    error_callback=on_error,
                    error_strategy=ErrorStrategy.ASK,
                )

            # Check for cancelled state
            if result.current_state == JobState.CANCELLED:
                completed_count = len(result.metadata.get("files_completed_before_cancel", []))
                total_count = result.metadata.get("total_files_planned", 0)
                self.progress_widget.add_log_entry("Obfuscation cancelled by user", "warning")
                self.progress_widget.add_log_entry(
                    f"Completed {completed_count}/{total_count} file(s) before cancellation", "info"
                )
            # Report results
            # Always display errors first, regardless of success flag
            for error in result.errors:
                formatted_error, level = self._format_error_for_log(error)
                self.progress_widget.add_log_entry(formatted_error, level)

            warning_count = len(result.warnings)
            if result.success:
                success_count = sum(1 for pr in result.processed_files if pr.success)
                completion_message = (
                    f"Obfuscation complete: {success_count}/{len(result.processed_files)} "
                    f"files processed successfully"
                )
                if warning_count > 0:
                    completion_message += f", {warning_count} warning(s)"
                self.progress_widget.add_log_entry(
                    completion_message,
                    "success"
                )
                self.progress_widget.set_progress(100)

                # Log conflict resolution info
                skipped_count = len(result.metadata.get("skipped_files", []))
                if skipped_count > 0:
                    self.progress_widget.add_log_entry(
                        f"{skipped_count} file(s) skipped due to conflicts", "warning"
                    )
                resolved_count = result.metadata.get("conflicts_resolved", 0)
                if resolved_count > 0:
                    self.progress_widget.add_log_entry(
                        f"{resolved_count} file conflict(s) resolved", "info"
                    )
                
                # Log error handling summary if errors were encountered
                error_decisions = result.metadata.get("error_decisions", [])
                if error_decisions:
                    self.progress_widget.add_log_entry(
                        f"{len(error_decisions)} error(s) encountered during processing", "warning"
                    )
                    # Display each failed file with first error message
                    for decision in error_decisions:
                        failed_file = decision.get("file", "Unknown")
                        errors = decision.get("errors", [])
                        first_error = errors[0] if errors else "Unknown error"
                        formatted_error, level = self._format_error_for_log(
                            first_error,
                            fallback_file=failed_file,
                        )
                        self.progress_widget.add_log_entry(f"  - {formatted_error}", level)
            else:
                # Show a summary message for failures
                failure_message = "Obfuscation completed with errors. See error messages above."
                if warning_count > 0:
                    failure_message += f" {warning_count} warning(s) detected."
                self.progress_widget.add_log_entry(
                    failure_message,
                    "error"
                )
                
                # Log error handling summary even on failure
                error_decisions = result.metadata.get("error_decisions", [])
                if error_decisions:
                    self.progress_widget.add_log_entry(
                        f"{len(error_decisions)} error(s) encountered during processing", "warning"
                    )
                    for decision in error_decisions:
                        failed_file = decision.get("file", "Unknown")
                        errors = decision.get("errors", [])
                        first_error = errors[0] if errors else "Unknown error"
                        formatted_error, level = self._format_error_for_log(
                            first_error,
                            fallback_file=failed_file,
                        )
                        self.progress_widget.add_log_entry(f"  - {formatted_error}", level)

            if warning_count > 0:
                self.progress_widget.add_log_entry(
                    f"{warning_count} warning(s) detected during processing",
                    "warning",
                )
                for warning in result.warnings[:10]:
                    self.progress_widget.add_log_entry(f"  {warning}", "warning")
                if warning_count > 10:
                    self.progress_widget.add_log_entry(
                        f"  ... and {warning_count - 10} more warning(s)",
                        "warning",
                    )

            if hasattr(result, "detailed_errors") and result.detailed_errors:
                error_dialog = ErrorReportDialog(
                    detailed_errors=result.detailed_errors,
                    parent=self,
                )
                error_dialog.exec()

        except Exception as e:
            logger.error(f"Obfuscation failed: {e}", exc_info=True)
            self.progress_widget.add_log_entry(f"Obfuscation failed: {e}", "error")

        finally:
            # Re-enable start button
            has_files = len(self.file_selection.get_files()) > 0
            self.action_widget.set_enabled(has_files)
            # Clear orchestrator reference
            self._current_orchestrator = None
            self._active_total_files = 0

    def _on_cancel_obfuscation(self) -> None:
        """Handle obfuscation cancellation request from user."""
        logger.info("Cancellation requested by user")

        if self._current_orchestrator is not None:
            progress_percentage = self.progress_widget.get_progress()
            total_count = self._active_total_files or len(self.file_selection.get_files())
            completed_count = int((progress_percentage / 100) * total_count) if total_count > 0 else 0

            current_file_text = self.progress_widget.current_file_label.text().strip()
            current_file: str | None = None
            if current_file_text.startswith("Current:"):
                current_file = current_file_text.split(":", 1)[1].strip()
            elif current_file_text:
                current_file = current_file_text

            if current_file in {"", "--", "N/A"}:
                current_file = None

            dialog = CancellationConfirmDialog(
                completed_count=completed_count,
                total_count=total_count,
                current_file=current_file,
                parent=self,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.get_user_decision():
                self.progress_widget.add_log_entry("Cancellation aborted by user", "info")
                return

            # Notify the orchestrator to cancel
            self._current_orchestrator.request_cancellation()
            logger.info("Cancellation requested - orchestrator notified")
            self.progress_widget.add_log_entry("Cancellation requested...", "warning")
            # Disable the cancel button to prevent multiple clicks
            self.progress_widget.cancel_button.setEnabled(False)
        else:
            # No active orchestrator, just hide the progress widget
            logger.warning("No active orchestrator to cancel")
            self.progress_widget.hide_progress()
            # Re-enable start button
            has_files = len(self.file_selection.get_files()) > 0
            self.action_widget.set_enabled(has_files)

    def _on_profile_save(self) -> None:
        """
        Handle profile save request - gather configuration and pass to ProfileWidget.

        Note: Profiles contain only obfuscation configuration (preset, features, options, language),
        not application state like output paths or file selections. ProfileWidget extracts only
        the security_config portion when saving to ensure profiles are reusable configuration templates.
        """
        # Determine language from first file, or default to "lua"
        files_with_languages = self.file_selection.get_files_with_languages()
        language = "lua"
        if files_with_languages:
            # Get language from first file and convert to lowercase for config
            first_language = next(iter(files_with_languages.values()))
            language = first_language.lower()

        # Default options structure
        options = {
            "string_encryption_key_length": 16,
            "dead_code_percentage": 20,
            "identifier_prefix": "_0x",
        }

        config = {
            "security_config": self.security_config.get_config(),
            "language": language,
            "options": options,
            "output_path": str(self.output_widget.get_output_path())
            if self.output_widget.get_output_path()
            else None,
            "files": files_with_languages,
        }
        self.profile_widget.save_profile(config)

    def _on_profile_load(self, config: dict) -> None:
        """
        Handle profile load - apply configuration to widgets.

        Applies both obfuscation configuration (preset, features) and workflow state
        (output_path, files) to restore the complete saved state.

        Args:
            config: Configuration dictionary containing security_config, output_path, and files
        """
        if not config:
            # Empty config means reset to defaults
            self.security_config.reset()
            self.output_widget.clear()
            self.file_selection.clear_files()
            logger.debug("Profile loaded: reset to defaults")
            return

        # Validate and extract security configuration
        security_config = config.get("security_config")
        if not security_config:
            logger.warning("Profile loaded but no security_config found - configuration may be malformed")
            QMessageBox.warning(
                self,
                "Invalid Profile",
                "The loaded profile does not contain valid security configuration."
            )
            return

        # Validate preset and features
        preset = security_config.get("preset")
        features = security_config.get("features")
        runtime_mode = security_config.get("runtime_mode")
        enable_multiprocessing = security_config.get("enable_multiprocessing", True)
        max_workers = security_config.get("max_workers", DEFAULT_MAX_WORKERS)
        batch_size = security_config.get("batch_size", DEFAULT_BATCH_SIZE)
        multiprocessing_threshold = security_config.get(
            "multiprocessing_threshold",
            DEFAULT_MULTIPROCESSING_THRESHOLD,
        )
        memory_threshold_percent = security_config.get(
            "memory_threshold_percent",
            DEFAULT_MEMORY_THRESHOLD_PERCENT,
        )

        if preset is None and not features:
            logger.error("Profile loaded but both preset and features are missing")
            QMessageBox.critical(
                self,
                "Invalid Configuration",
                "The profile must contain either a preset or custom features."
            )
            return

        # Apply security configuration to widget
        try:
            self.security_config.set_config(
                preset=preset,
                features=features,
                runtime_mode=runtime_mode,
                enable_multiprocessing=enable_multiprocessing,
                max_workers=max_workers,
                batch_size=batch_size,
                multiprocessing_threshold=multiprocessing_threshold,
                memory_threshold_percent=memory_threshold_percent,
            )
            logger.info(
                "Profile loaded successfully - Preset: %s, Features: %d, Runtime Mode: %s, Multiprocessing: %s, Max Workers: %s, Batch Size: %s, Parallel Threshold: %s, Memory Threshold: %s",
                preset,
                len(features) if features else 0,
                runtime_mode,
                enable_multiprocessing,
                max_workers,
                batch_size,
                multiprocessing_threshold,
                memory_threshold_percent,
            )
        except Exception as e:
            logger.error(f"Failed to apply security configuration: {e}")
            QMessageBox.critical(
                self,
                "Configuration Error",
                f"Failed to apply security configuration: {str(e)}"
            )
            return

        # Apply output path if present
        output_path = config.get("output_path")
        if output_path:
            try:
                from pathlib import Path
                self.output_widget.set_output_path(Path(output_path))
                logger.debug(f"Restored output path: {output_path}")
            except Exception as e:
                logger.warning(f"Failed to restore output path: {e}")
        else:
            # Clear output path if not in profile
            self.output_widget.clear()

        # Apply files if present
        files = config.get("files")
        if files:
            try:
                self.file_selection.set_files_with_languages(files)
                logger.debug(f"Restored {len(files)} file(s)")
            except Exception as e:
                logger.warning(f"Failed to restore files: {e}")
        else:
            # Clear files if not in profile
            self.file_selection.clear_files()

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Handle window close event.

        Args:
            event: The close event to handle.
        """
        logger.info("Application shutdown requested")
        event.accept()
        super().closeEvent(event)
