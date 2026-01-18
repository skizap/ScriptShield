"""
Output location widget for the Python Obfuscator GUI.

Provides directory/file path selection for obfuscation output.
"""

import os
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
)
from PyQt6.QtGui import QCursor

from obfuscator.utils.logger import get_logger
from obfuscator.utils.path_utils import normalize_path
from obfuscator.gui.styles.stylesheet import get_widget_style

logger = get_logger("obfuscator.gui.widgets.output_widget")


class OutputWidget(QWidget):
    """Widget for selecting output directory/file path."""

    output_changed = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the output widget."""
        super().__init__(parent)
        self._setup_ui()
        logger.debug("OutputWidget initialized")

    def _setup_ui(self) -> None:
        """Set up the widget UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Title
        title_label = QLabel("Output Location")
        title_label.setStyleSheet(get_widget_style("title_label"))
        title_label.setProperty("data-element-id", "output-section-title")
        layout.addWidget(title_label)

        # Path input row
        row_layout = QHBoxLayout()
        row_layout.setSpacing(8)

        # Path field
        self._path_field = QLineEdit()
        self._path_field.setProperty("data-element-id", "output-path-field")
        self._path_field.setPlaceholderText("Select output location...")
        self._path_field.setStyleSheet(get_widget_style("input_field"))
        self._path_field.editingFinished.connect(self._on_path_edited)
        row_layout.addWidget(self._path_field, 1)

        # Browse button
        self._browse_btn = QPushButton("📁")
        self._browse_btn.setProperty("data-element-id", "output-browse-button")
        self._browse_btn.setToolTip("Browse for output directory")
        self._browse_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._browse_btn.setStyleSheet(get_widget_style("icon_button"))
        self._browse_btn.clicked.connect(self._on_browse_clicked)
        row_layout.addWidget(self._browse_btn)

        layout.addLayout(row_layout)

    def _on_browse_clicked(self) -> None:
        """Handle browse button click - open directory picker."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self._path_field.text() or str(Path.home()),
        )
        if directory:
            self._set_path(directory)
            logger.info(f"Output directory selected: {directory}")

    def _on_path_edited(self) -> None:
        """Handle manual path entry."""
        path_text = self._path_field.text().strip()
        if path_text:
            self._validate_and_emit(path_text)

    def _set_path(self, path: str) -> None:
        """Set the path field and emit change signal."""
        self._path_field.setText(path)
        self._validate_and_emit(path)

    def _validate_and_emit(self, path: str) -> None:
        """Validate path and emit output_changed signal."""
        try:
            normalized = normalize_path(path)
            self.output_changed.emit(str(normalized))
            logger.debug(f"Output path changed: {normalized}")
        except ValueError as e:
            logger.warning(f"Invalid output path: {e}")

    def get_output_path(self) -> Optional[Path]:
        """Return current output path as Path object or None if empty."""
        path_text = self._path_field.text().strip()
        if not path_text:
            return None
        try:
            return normalize_path(path_text)
        except ValueError:
            return None

    def set_output_path(self, path: Path) -> None:
        """Programmatically set output path."""
        self._set_path(str(path))

    def clear(self) -> None:
        """Clear the output path field."""
        self._path_field.clear()
        self.output_changed.emit("")
        logger.debug("Output path cleared")

    def suggest_output_path(self, source_paths: List[str]) -> None:
        """
        Suggest an output path based on source file locations.

        Finds a common parent directory from selected files and proposes
        {parent}/obfuscated as the output path, but only when the output field
        is empty to respect user's manual selections.

        Args:
            source_paths: List of file path strings selected for obfuscation.
        """
        # Check if output field is already set by user
        if self._path_field.text().strip():
            logger.debug("Output path already set, skipping auto-suggestion")
            return

        # Handle empty source files list
        if not source_paths:
            logger.debug("No source files, skipping auto-suggestion")
            return

        try:
            # Convert source paths to Path objects
            paths = []
            for path_str in source_paths:
                try:
                    paths.append(normalize_path(path_str))
                except ValueError as e:
                    logger.warning(f"Failed to normalize path {path_str}: {e}")
                    continue

            if not paths:
                logger.debug("No valid paths found, skipping auto-suggestion")
                return

            # Determine common parent directory
            common_parent = self._find_common_parent(paths)
            if not common_parent:
                logger.warning("Failed to determine common parent directory")
                return

            # Create suggested output path
            suggested_path = common_parent / "obfuscated"
            self.set_output_path(suggested_path)
            logger.info(f"Auto-suggested output path: {suggested_path}")

        except Exception as e:
            logger.warning(f"Failed to determine common parent directory: {e}")

    def _find_common_parent(self, paths: List[Path]) -> Optional[Path]:
        """
        Find the common parent directory for a list of file paths.

        Args:
            paths: List of Path objects representing file locations.

        Returns:
            The common parent Path, or None if it cannot be determined.
        """
        if not paths:
            return None

        # Single file case: use its parent directory
        if len(paths) == 1:
            return paths[0].parent

        # Multiple files case: find common ancestor
        try:
            # Use os.path.commonpath for cross-platform compatibility
            path_strings = [str(p) for p in paths]
            common = os.path.commonpath(path_strings)
            common_path = Path(common)
            
            # Check if common path is a filesystem root or drive root
            if self._is_root_path(common_path):
                logger.debug(
                    f"Common path is a filesystem root: {common_path}, "
                    "falling back to first file's parent"
                )
                return paths[0].parent
            
            return common_path
        except ValueError:
            # Files are on different drives (Windows) or no common path
            # Fall back to first file's parent
            return paths[0].parent
    
    def _is_root_path(self, path: Path) -> bool:
        """
        Check if the given path is a filesystem root or drive root.

        Args:
            path: Path object to check.

        Returns:
            True if the path is a root, False otherwise.
        """
        # Method 1: Check if parent equals the path (root has no parent)
        if path.parent == path:
            return True
        
        # Method 2: Check if path is '/' (Unix root)
        if str(path) == '/' or str(path) == '//':
            return True
        
        # Method 3: Check for Windows drive root patterns (e.g., 'C:\', 'D:/')
        path_str = str(path)
        if len(path_str) == 3 and path_str[1] in {':', '\\', '/'} and path_str[0].isalpha():
            return True
        
        # Method 4: Check if resolved path is root (handles some edge cases)
        try:
            resolved = path.resolve()
            if resolved.parent == resolved:
                return True
        except (OSError, RuntimeError):
            pass
        
        return False
