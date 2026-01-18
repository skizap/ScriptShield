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
from obfuscator.core.orchestrator import ObfuscationOrchestrator

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

        # Define progress callback for GUI updates
        def on_progress(message: str, current: int, total: int) -> None:
            progress_percent = int((current / total) * 100) if total > 0 else 0
            self.progress_widget.set_progress(progress_percent)
            self.progress_widget.add_log_entry(message, "info")
            # Process events to keep GUI responsive
            QApplication.processEvents()

        try:
            # Create orchestrator and process files
            orchestrator = ObfuscationOrchestrator()
            result = orchestrator.process_files(
                input_files=input_files,
                output_dir=output_dir,
                config=config,
                progress_callback=on_progress
            )

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
            else:
                # Show a summary message for failures
                self.progress_widget.add_log_entry(
                    "Obfuscation completed with errors. See error messages above.",
                    "error"
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

    def _on_cancel_obfuscation(self) -> None:
        """Handle obfuscation cancellation."""
        logger.info("Obfuscation cancelled")
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
            self.security_config.set_config(preset=preset, features=features)
            logger.info(f"Profile loaded successfully - Preset: {preset}, Features: {len(features) if features else 0}")
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
