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
    FileSelectionWidget,
    InfoPanelWidget,
    OutputWidget,
    ProfileWidget,
    ProgressWidget,
    SecurityConfigWidget,
)

from obfuscator.gui.styles.stylesheet import get_application_stylesheet
from obfuscator.utils.logger import get_logger
from obfuscator.utils.path_utils import get_platform, normalize_path
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

    def _on_start_obfuscation(self) -> None:
        """Handle start obfuscation button click.

        Initiates the obfuscation workflow using the ObfuscationOrchestrator.
        The orchestrator handles dependency analysis, symbol table construction,
        and processes files in topological order for consistent cross-file
        symbol references.
        """
        files = self.file_selection.get_files()
        output_path = self.output_widget.get_output_path()
        config = self.security_config.get_config()

        logger.info(
            f"Start obfuscation requested: {len(files)} files, "
            f"output: {output_path}, preset: {config.get('preset')}"
        )

        # Reset and show progress widget
        self.progress_widget.reset()
        self.progress_widget.show_progress()
        self.progress_widget.add_log_entry("Starting obfuscation...", "info")

        # Disable start button during processing
        self.action_widget.set_enabled(False)

        # Convert file paths to Path objects
        input_files = [Path(f) for f in files]
        output_dir = Path(output_path) if output_path else Path.cwd() / "obfuscated"

        # Early conflict detection before creating main orchestrator
        temp_orchestrator = ObfuscationOrchestrator()
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

        # Define progress callback for GUI updates
        def on_progress(progress_info: ProgressInfo) -> None:
            self.progress_widget.set_progress(int(progress_info.percentage))
            self.progress_widget.set_state(progress_info.current_state.name)
            self.progress_widget.set_time_info(
                progress_info.elapsed_seconds,
                progress_info.estimated_remaining_seconds,
            )
            self.progress_widget.add_log_entry(progress_info.message, "info")

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
            orchestrator = ObfuscationOrchestrator()
            self._current_orchestrator = orchestrator
            if conflict_result.has_conflicts:
                # Copy the strategy from temp orchestrator
                orchestrator.set_conflict_strategy(temp_orchestrator._conflict_strategy)

            result = orchestrator.process_files(
                input_files=input_files,
                output_dir=output_dir,
                config=config,
                progress_callback=on_progress,
                error_callback=on_error,
                error_strategy=ErrorStrategy.ASK
            )

            # Check for cancelled state
            if result.current_state == JobState.CANCELLED:
                completed_count = len(result.metadata.get("files_completed_before_cancel", []))
                total_count = result.metadata.get("total_files_planned", 0)
                self.progress_widget.add_log_entry("Obfuscation cancelled by user", "warning")
                self.progress_widget.add_log_entry(
                    f"Completed {completed_count}/{total_count} file(s) before cancellation", "info"
                )
                if result.warnings:
                    for warning in result.warnings:
                        self.progress_widget.add_log_entry(f"Warning: {warning}", "warning")
            # Report results
            # Always display errors first, regardless of success flag
            for error in result.errors:
                self.progress_widget.add_log_entry(f"Error: {error}", "error")

            if result.success:
                success_count = sum(1 for pr in result.processed_files if pr.success)
                self.progress_widget.add_log_entry(
                    f"Obfuscation complete: {success_count}/{len(result.processed_files)} "
                    f"files processed successfully",
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
                        self.progress_widget.add_log_entry(
                            f"  - {failed_file}: {first_error}", "error"
                        )
            else:
                # Show a summary message for failures
                self.progress_widget.add_log_entry(
                    "Obfuscation completed with errors. See error messages above.",
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
                        self.progress_widget.add_log_entry(
                            f"  - {failed_file}: {first_error}", "error"
                        )

            # Log warnings
            for warning in result.warnings:
                self.progress_widget.add_log_entry(f"Warning: {warning}", "warning")

        except Exception as e:
            logger.error(f"Obfuscation failed: {e}", exc_info=True)
            self.progress_widget.add_log_entry(f"Obfuscation failed: {e}", "error")

        finally:
            # Re-enable start button
            has_files = len(self.file_selection.get_files()) > 0
            self.action_widget.set_enabled(has_files)
            # Clear orchestrator reference
            self._current_orchestrator = None

    def _on_cancel_obfuscation(self) -> None:
        """Handle obfuscation cancellation request from user."""
        logger.info("Cancellation requested by user")

        if self._current_orchestrator is not None:
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
            self.security_config.set_config(preset=preset, features=features, runtime_mode=runtime_mode)
            logger.info(f"Profile loaded successfully - Preset: {preset}, Features: {len(features) if features else 0}, Runtime Mode: {runtime_mode}")
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
