"""
Profile management widget for the Python Obfuscator GUI.

Provides save/load/delete functionality for configuration profiles
using file dialogs for file-based persistence.
"""

import json
from typing import Optional
from pathlib import Path

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QFileDialog,
)
from PyQt6.QtGui import QCursor

from obfuscator.utils.logger import get_logger
from obfuscator.utils.path_utils import ensure_directory
from obfuscator.gui.styles.stylesheet import get_widget_style
from obfuscator.core.profile_manager import ProfileManager
from obfuscator.core.config import ObfuscationConfig

logger = get_logger("obfuscator.gui.widgets.profile_widget")


class ProfileWidget(QWidget):
    """
    Widget for managing obfuscation configuration profiles using file dialogs.

    Provides save/load/delete functionality with .obfprofile files.
    Communicates with MainWindow via signals for profile operations.
    """

    profile_load_requested = pyqtSignal(dict)
    profile_save_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the profile widget."""
        super().__init__(parent)
        self._last_directory = str(Path.home())
        self._setup_ui()
        logger.debug("ProfileWidget initialized")

    def _setup_ui(self) -> None:
        """Set up the widget UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Title
        title_label = QLabel("Configuration Profiles")
        title_label.setStyleSheet(get_widget_style("title_label"))
        title_label.setProperty("data-element-id", "profile-section-title")
        layout.addWidget(title_label)

        # Buttons row
        row_layout = QHBoxLayout()
        row_layout.setSpacing(8)

        # Save button
        self._save_btn = QPushButton("💾")
        self._save_btn.setProperty("data-element-id", "profile-save-button")
        self._save_btn.setToolTip("Save current configuration to file")
        self._save_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._save_btn.setStyleSheet(get_widget_style("icon_button"))
        self._save_btn.clicked.connect(self._on_save_clicked)
        row_layout.addWidget(self._save_btn)

        # Load button
        self._load_btn = QPushButton("📂")
        self._load_btn.setProperty("data-element-id", "profile-load-button")
        self._load_btn.setToolTip("Load configuration from file")
        self._load_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._load_btn.setStyleSheet(get_widget_style("icon_button"))
        self._load_btn.clicked.connect(self._on_load_clicked)
        row_layout.addWidget(self._load_btn)

        # Delete button
        self._delete_btn = QPushButton("🗑️")
        self._delete_btn.setProperty("data-element-id", "profile-delete-button")
        self._delete_btn.setToolTip("Delete a profile file")
        self._delete_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._delete_btn.setStyleSheet(get_widget_style("icon_button"))
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        row_layout.addWidget(self._delete_btn)

        layout.addLayout(row_layout)

    def _on_save_clicked(self) -> None:
        """
        Handle save button click - emit signal to request config from MainWindow.

        The actual file dialog and save operation happens in save_profile() method
        which is called by MainWindow after gathering configuration.
        """
        self.profile_save_requested.emit()
        logger.debug("Profile save requested")

    def save_profile(self, config: dict) -> None:
        """
        Save configuration data to a file using file dialog.

        Args:
            config: Configuration dictionary from MainWindow containing security_config, etc.
        """
        # Open file dialog for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Profile",
            self._last_directory,
            "Obfuscation Profiles (*.obfprofile);;All Files (*)",
        )

        if not file_path:
            logger.debug("Profile save cancelled by user")
            return

        # Ensure .obfprofile extension
        if not file_path.endswith('.obfprofile'):
            file_path += '.obfprofile'

        # Update last directory
        self._last_directory = str(Path(file_path).parent)

        # Extract profile name from filename
        profile_name = Path(file_path).stem

        try:
            # Convert config dict to ObfuscationConfig
            # Extract security_config containing preset, features, and runtime_mode
            security_config = config.get("security_config", {})
            preset = security_config.get("preset")  # Don't default to "medium"
            features = security_config.get("features", {})
            runtime_mode = security_config.get("runtime_mode", "hybrid")

            # Extract language and options from config
            language = config.get("language", "lua")
            options = config.get("options", {
                "string_encryption_key_length": 16,
                "array_shuffle_seed": None,
                "dead_code_percentage": 20,
                "identifier_prefix": "_0x",
            })

            # Create ObfuscationConfig using from_gui_config
            # Pass None for preset when unset so custom feature sets persist
            config_obj = ObfuscationConfig.from_gui_config(
                name=profile_name,
                preset=preset,  # Can be None for custom configurations
                features=features,
                language=language
            )

            # Set runtime_mode from GUI config
            config_obj.runtime_mode = runtime_mode

            # Set options from GUI values
            config_obj.options = options

            # Create extended profile data that includes workflow state
            # This includes both the obfuscation config and UI state (output_path, files)
            profile_data = config_obj.to_dict()

            # Add workflow state fields
            output_path = config.get("output_path")
            files = config.get("files", {})

            if output_path:
                profile_data["output_path"] = output_path
            if files:
                profile_data["files"] = files

            # Save extended profile data directly to JSON
            # We bypass ProfileManager.save_profile to include extra fields
            ensure_directory(Path(file_path).parent)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=2)

            QMessageBox.information(
                self,
                "Success",
                f"Profile '{profile_name}' saved successfully!"
            )
            logger.info(f"Profile saved: {profile_name} at {file_path}")

        except (ValueError, OSError) as e:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save profile:\n{str(e)}"
            )
            logger.error(f"Failed to save profile: {e}")

    def _on_load_clicked(self) -> None:
        """
        Handle load button click - open file dialog and load profile.

        Loads profile from file, converts to dict format, and emits signal to MainWindow.
        """
        # Open file dialog for profile selection
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Profile",
            self._last_directory,
            "Obfuscation Profiles (*.obfprofile);;All Files (*)",
        )

        if not file_path:
            logger.debug("Profile load cancelled by user")
            return

        # Update last directory
        self._last_directory = str(Path(file_path).parent)

        try:
            # Load profile data directly from JSON to get all fields including workflow state
            with open(file_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)

            # Create config object from the data for validation
            config_obj = ObfuscationConfig.from_dict(profile_data)
            config_obj.validate()

            # Convert preset to title case for GUI (e.g., "medium" -> "Medium")
            preset_gui = config_obj.preset.title() if config_obj.preset else None

            # Create reverse mapping from JSON feature names to GUI labels
            from obfuscator.core.config import GUI_TO_JSON_FEATURE_MAP
            json_to_gui_map = {v: k for k, v in GUI_TO_JSON_FEATURE_MAP.items()}

            # Convert JSON feature names to GUI labels
            gui_features = {}
            for json_name, enabled in config_obj.features.items():
                gui_name = json_to_gui_map.get(json_name)
                if gui_name:
                    gui_features[gui_name] = enabled

            # Get runtime_mode from config_obj or profile_data
            runtime_mode = getattr(config_obj, 'runtime_mode', None) or profile_data.get('runtime_mode', 'hybrid')

            # Transform to match current GUI format
            # Include security_config, output_path, and files
            gui_config = {
                "security_config": {
                    "preset": preset_gui,
                    "features": gui_features,
                    "runtime_mode": runtime_mode,
                }
            }

            # Add workflow state fields if present in the profile
            if "output_path" in profile_data:
                gui_config["output_path"] = profile_data["output_path"]
            if "files" in profile_data:
                gui_config["files"] = profile_data["files"]

            # Emit signal to MainWindow
            self.profile_load_requested.emit(gui_config)
            logger.info(f"Profile loaded: {config_obj.name} from {file_path}")

        except FileNotFoundError:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Profile file not found:\n{file_path}"
            )
            logger.error(f"Profile file not found: {file_path}")

        except ValueError as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Invalid profile format:\n{str(e)}"
            )
            logger.error(f"Invalid profile format: {e}")

        except json.JSONDecodeError as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Corrupted profile file:\n{str(e)}"
            )
            logger.error(f"Corrupted profile file: {e}")

        except OSError as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to read profile file:\n{file_path}\n\n{str(e)}"
            )
            logger.error(f"File I/O error loading profile: {e}")

    def _on_delete_clicked(self) -> None:
        """
        Handle delete button click - open file dialog and delete profile file.

        Shows confirmation dialog before deletion.
        """
        # Open file dialog for profile selection
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Profile to Delete",
            self._last_directory,
            "Obfuscation Profiles (*.obfprofile);;All Files (*)",
        )

        if not file_path:
            logger.debug("Profile deletion cancelled by user")
            return

        # Update last directory
        self._last_directory = str(Path(file_path).parent)

        # Get filename for confirmation message
        filename = Path(file_path).name

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Delete Profile",
            f"Are you sure you want to delete '{filename}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Delete the file
                Path(file_path).unlink()

                QMessageBox.information(
                    self,
                    "Success",
                    f"Profile '{filename}' deleted successfully!"
                )
                logger.info(f"Profile deleted: {filename}")

            except OSError as e:
                QMessageBox.critical(
                    self,
                    "Delete Error",
                    f"Failed to delete profile:\n{str(e)}"
                )
                logger.error(f"Failed to delete profile: {e}")
