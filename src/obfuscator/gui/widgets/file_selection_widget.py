"""
File selection widget with drag-and-drop support.

This module provides a FileSelectionWidget for selecting Python/Lua/Luau files
via drag-and-drop or file browser dialog, with language detection and
file management capabilities.
"""

from pathlib import Path
from typing import Dict, List, Optional
import itertools

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from obfuscator.utils.logger import get_logger
from obfuscator.utils.path_utils import normalize_path
from obfuscator.gui.styles.stylesheet import get_widget_style, COLORS

# Module-level logger
logger = get_logger("obfuscator.gui.widgets.file_selection_widget")

# Supported file extensions
SUPPORTED_EXTENSIONS = {".lua", ".luau", ".py"}


class FileSelectionWidget(QWidget):
    """
    Widget for selecting and managing Python/Lua/Luau files.

    Provides drag-and-drop functionality, file browser dialog, and
    a list view showing selected files with language badges and
    management controls. Supports mixed-language detection.

    Signals:
        files_changed: Emitted when the file list changes with list of paths.
    """

    files_changed = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the file selection widget.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._files: Dict[str, str] = {}  # path_str -> language
        self._id_counter = itertools.count()  # Monotonic ID generator for unique data-element-ids
        self._setup_ui()
        logger.info("FileSelectionWidget initialized")

    def _setup_ui(self) -> None:
        """Set up the widget UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Title
        title_label = QLabel("Select Files")
        title_label.setStyleSheet(get_widget_style("title_label"))
        layout.addWidget(title_label)

        # File count label
        self._file_count_label = QLabel("0 files selected")
        self._file_count_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        self._file_count_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._file_count_label.setVisible(False)
        layout.addWidget(self._file_count_label)

        # Drag-and-drop zone
        self._drop_zone = self._create_drop_zone()
        layout.addWidget(self._drop_zone)
 +++++++ REPLACE

        # File list
        self._file_list = QListWidget()
        self._file_list.setObjectName("fileList")
        self._file_list.setProperty("data-element-id", "file-list")
        self._file_list.setMinimumHeight(200)
        self._file_list.setStyleSheet(get_widget_style("file_list"))
        layout.addWidget(self._file_list)

        # Empty state label
        self._empty_label = QLabel("No files selected")
        self._empty_label.setStyleSheet(f"color: {COLORS['text_disabled']}; font-style: italic;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty_label)

        self._update_empty_state()

    def _create_drop_zone(self) -> QFrame:
        """Create the drag-and-drop zone frame."""
        drop_zone = QFrame()
        drop_zone.setObjectName("dropZone")
        drop_zone.setProperty("data-element-id", "file-drop-zone")
        drop_zone.setStyleSheet(get_widget_style("drop_zone"))
        drop_zone.setAcceptDrops(True)

        # Override drag/drop events
        drop_zone.dragEnterEvent = self._on_drag_enter
        drop_zone.dragMoveEvent = self._on_drag_move
        drop_zone.dragLeaveEvent = self._on_drag_leave
        drop_zone.dropEvent = self._on_drop

        # Drop zone content
        drop_layout = QVBoxLayout(drop_zone)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        drop_icon = QLabel("📁")
        drop_icon.setStyleSheet("font-size: 32px;")
        drop_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(drop_icon)

        drop_text = QLabel("Drag and drop Python/Lua files or folders here")
        drop_text.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        drop_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(drop_text)
 +++++++ REPLACE

        or_label = QLabel("— or —")
        or_label.setStyleSheet(f"color: {COLORS['text_disabled']}; font-size: 11px;")
        or_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(or_label)

        # Button widget for Browse Files and Browse Folder
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)
        
        browse_files_btn = QPushButton("Browse Files")
        browse_files_btn.setObjectName("browseFilesButton")
        browse_files_btn.setProperty("data-element-id", "file-browse-files-button")
        browse_files_btn.setFixedWidth(120)
        browse_files_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_files_btn.clicked.connect(self._on_browse_clicked)
        browse_files_btn.setStyleSheet(get_widget_style("primary_button"))
        button_layout.addWidget(browse_files_btn)
        
        browse_folder_btn = QPushButton("Browse Folder")
        browse_folder_btn.setObjectName("browseFolderButton")
        browse_folder_btn.setProperty("data-element-id", "file-browse-folder-button")
        browse_folder_btn.setFixedWidth(120)
        browse_folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_folder_btn.clicked.connect(self._on_browse_folder_clicked)
        browse_folder_btn.setStyleSheet(get_widget_style("secondary_button"))
        button_layout.addWidget(browse_folder_btn)
        
        drop_layout.addWidget(button_widget, alignment=Qt.AlignmentFlag.AlignCenter)
 +++++++ REPLACE
 +++++++ REPLACE

        return drop_zone

    def _on_drag_enter(self, event) -> None:
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            has_valid = any(
                Path(url.toLocalFile()).suffix.lower() in SUPPORTED_EXTENSIONS
                for url in urls if url.isLocalFile() and Path(url.toLocalFile()).is_file()
            )
            has_directory = any(
                Path(url.toLocalFile()).is_dir()
                for url in urls if url.isLocalFile()
            )
            if has_valid or has_directory:
                event.acceptProposedAction()
                self._drop_zone.setStyleSheet(get_widget_style("drop_zone_hover"))
                logger.debug(f"Drag enter with {len(urls)} item(s)")
                return
        event.ignore()
 +++++++ REPLACE

    def _on_drag_move(self, event) -> None:
        """Handle drag move event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def _on_drag_leave(self, event) -> None:
        """Handle drag leave event."""
        self._drop_zone.setStyleSheet(get_widget_style("drop_zone"))
        event.accept()

    def _on_drop(self, event) -> None:
        """Handle drop event."""
        self._drop_zone.setStyleSheet(get_widget_style("drop_zone"))
        if event.mimeData().hasUrls():
            paths = []
            folder_count = 0
            file_count = 0
            
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = Path(url.toLocalFile())
                    
                    # Handle directory
                    if file_path.is_dir():
                        folder_count += 1
                        discovered = self._scan_folder_recursively(file_path)
                        paths.extend(discovered)
                    
                    # Handle file
                    elif file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        file_count += 1
                        paths.append(file_path)
            
            if paths:
                self._add_files(paths)
                event.acceptProposedAction()
                if folder_count > 0:
                    logger.info(f"Dropped {file_count} file(s) and {folder_count} folder(s), discovered {len(paths)} total files")
                else:
                    logger.info(f"Dropped {file_count} file(s)")
                return
        event.ignore()
 +++++++ REPLACE

    def _on_browse_clicked(self) -> None:
        """Handle browse button click with file/folder selection option."""
        # Show dialog to choose between files and folder
        from PyQt6.QtWidgets import QMessageBox
        
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Select Files or Folder")
        dialog.setText("What would you like to select?")
        
        files_btn = dialog.addButton("Select Files", QMessageBox.ButtonRole.ActionRole)
        folder_btn = dialog.addButton("Select Folder", QMessageBox.ButtonRole.ActionRole)
        cancel_btn = dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        
        dialog.exec()
        
        if dialog.clickedButton() == files_btn:
            # Select files
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Python/Lua Files",
                "",
                "Script Files (*.py *.lua *.luau);;All Files (*.*)"
            )
            if files:
                paths = [Path(f) for f in files]
                self._add_files(paths)
                logger.info(f"Selected {len(paths)} file(s) via browser")
        
        elif dialog.clickedButton() == folder_btn:
            # Select folder
            folder_path = QFileDialog.getExistingDirectory(
                self,
                "Select Python/Lua/Luau Folder"
            )
            if folder_path:
                path = Path(folder_path)
                logger.info(f"Selected folder: {path}")
                discovered = self._scan_folder_recursively(path)
                if discovered:
                    self._add_files(discovered)
                    logger.info(f"Added {len(discovered)} files from folder")
 +++++++ REPLACE

    def _on_browse_folder_clicked(self) -> None:
        """Handle browse folder button click."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Python/Lua/Luau Folder"
        )
        if folder_path:
            path = Path(folder_path)
            logger.info(f"Selected folder: {path}")
            discovered = self._scan_folder_recursively(path)
            if discovered:
                self._add_files(discovered)
                logger.info(f"Added {len(discovered)} files from folder")
 +++++++ REPLACE

    def _scan_folder_recursively(self, folder_path: Path) -> List[Path]:
        """
        Scan a folder recursively for supported file types.

        Args:
            folder_path: Path to the folder to scan.

        Returns:
            List of discovered file paths.
        """
        discovered_files = []
        logger.info(f"Scanning folder: {folder_path}")

        try:
            # Scan for each supported extension
            for ext in SUPPORTED_EXTENSIONS:
                try:
                    pattern = f"*{ext}"
                    for file_path in folder_path.rglob(pattern):
                        if file_path.is_file():
                            discovered_files.append(file_path)
                except (OSError, PermissionError) as e:
                    logger.warning(f"Cannot access folder during scan: {folder_path} - {e}")

            logger.info(f"Discovered {len(discovered_files)} files in {folder_path}")

        except (OSError, PermissionError) as e:
            logger.warning(f"Cannot access folder: {folder_path} - {e}")

        return discovered_files
 +++++++ REPLACE

    def _add_files(self, paths: List[Path]) -> None:
        """
        Add files to the list.

        Args:
            paths: List of file paths to add.
        """
        added_count = 0
        for path in paths:
            try:
                normalized = normalize_path(path)
                path_str = str(normalized)

                # Skip duplicates
                if path_str in self._files:
                    logger.debug(f"Skipping duplicate: {path_str}")
                    continue

                # Skip if file doesn't exist
                if not normalized.exists():
                    logger.warning(f"File not found: {path_str}")
                    continue

                # Detect language and add to internal dict
                language = self._detect_language(normalized)
                self._files[path_str] = language

                # Create list item
                self._create_file_item(path_str, language)
                added_count += 1

            except (ValueError, OSError) as e:
                logger.error(f"Error adding file {path}: {e}")

        if added_count > 0:
            self._update_empty_state()
            self._update_language_badges()
            self.files_changed.emit(self.get_files())
            logger.info(f"Added {added_count} file(s), total: {len(self._files)}")

    def _create_file_item(self, path_str: str, language: str) -> None:
        """
        Create a list widget item for a file.

        Args:
            path_str: The file path string.
            language: The detected language (Lua/Luau/Python/Mixed).
        """
        # Badge style mapping
        badge_style_map = {
            "Lua": "language_badge_lua",
            "Luau": "language_badge_luau",
            "Python": "language_badge_python",
            "Mixed": "language_badge_mixed",
        }

        # Generate unique monotonic ID for this item
        unique_id = next(self._id_counter)

        # Create item widget
        item_widget = QWidget()
        item_widget.setProperty("data-element-id", f"file-item-{unique_id}")
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(8, 4, 8, 4)
        item_layout.setSpacing(8)

        # File name label
        file_name = Path(path_str).name
        name_label = QLabel(file_name)
        name_label.setToolTip(path_str)
        name_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        item_layout.addWidget(name_label, 1)

        # Language badge
        badge = QLabel(language)
        badge.setProperty("data-element-id", f"language-badge-{unique_id}")
        badge_style = badge_style_map.get(language, "language_badge_lua")
        badge.setStyleSheet(get_widget_style(badge_style))
        item_layout.addWidget(badge)

        # Change language button
        change_btn = QPushButton("⇄")
        change_btn.setProperty("data-element-id", f"change-language-button-{unique_id}")
        change_btn.setFixedSize(28, 28)
        change_btn.setToolTip("Toggle language")
        change_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        change_btn.setStyleSheet(get_widget_style("icon_button"))
        # Accept checked bool from clicked signal to avoid TypeError
        change_btn.clicked.connect(lambda checked: self._change_language(path_str))
        item_layout.addWidget(change_btn)

        # Remove button
        remove_btn = QPushButton("✕")
        remove_btn.setProperty("data-element-id", f"remove-file-button-{unique_id}")
        remove_btn.setFixedSize(28, 28)
        remove_btn.setToolTip("Remove file")
        remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_btn.setStyleSheet(get_widget_style("danger_button"))
        # Accept checked bool from clicked signal to avoid TypeError
        remove_btn.clicked.connect(lambda checked: self._remove_file(path_str))
        item_layout.addWidget(remove_btn)

        # Create list item and set widget
        list_item = QListWidgetItem(self._file_list)
        list_item.setData(Qt.ItemDataRole.UserRole, path_str)
        list_item.setSizeHint(item_widget.sizeHint())
        self._file_list.setItemWidget(list_item, item_widget)

    def _remove_file(self, path_str: str) -> None:
        """
        Remove a file from the list.

        Args:
            path_str: The file path string to remove.
        """
        if path_str not in self._files:
            return

        # Find and remove the list item
        for i in range(self._file_list.count()):
            item = self._file_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == path_str:
                self._file_list.takeItem(i)
                break

        # Remove from internal dict
        del self._files[path_str]

        self._update_empty_state()
        self._update_language_badges()
        self.files_changed.emit(self.get_files())
        logger.info(f"Removed file: {Path(path_str).name}")

    def _change_language(self, path_str: str) -> None:
        """
        Toggle the language for a file. Python files cannot be toggled.
        Lua files toggle between Lua and Luau.

        Args:
            path_str: The file path string.
        """
        if path_str not in self._files:
            return

        current = self._files[path_str]

        # Python files cannot be toggled
        if current == "Python":
            logger.debug(f"Cannot toggle language for Python file: {Path(path_str).name}")
            return

        # Toggle between Lua and Luau for .lua files
        new_lang = "Luau" if current == "Lua" else "Lua"
        self._files[path_str] = new_lang

        # Update all badges to handle mixed-language projects correctly
        self._update_language_badges()

        self.files_changed.emit(self.get_files())
        logger.info(f"Changed language for {Path(path_str).name} to {new_lang}")

    def _detect_language(self, path: Path) -> str:
        """
        Detect the language based on file extension.

        Args:
            path: The file path.

        Returns:
            "Python", "Luau", or "Lua" based on extension.
        """
        ext = path.suffix.lower()
        if ext == ".py":
            return "Python"
        if ext == ".luau":
            return "Luau"
        return "Lua"

    def _detect_project_language(self) -> str:
        """
        Analyze all files to determine if the project has mixed languages.

        Treats Lua and Luau as the same language family. Returns "Mixed" only
        when both Python and Lua/Luau files are present.

        Returns:
            Single language name if all files are the same, or "Mixed" if Python
            and Lua/Luau files are both present.
        """
        if not self._files:
            return "Lua"  # Default when no files

        unique_languages = set(self._files.values())

        # If only one language, return it
        if len(unique_languages) == 1:
            return next(iter(unique_languages))

        # Check if we have Python files
        has_python = "Python" in unique_languages
        has_lua_family = "Lua" in unique_languages or "Luau" in unique_languages

        # Mixed only if both Python and Lua/Luau are present
        if has_python and has_lua_family:
            return "Mixed"

        # If only Lua and Luau (no Python), return the first one found
        # This maintains backward compatibility for Lua+Luau projects
        return next(iter(unique_languages))

    def _update_language_badges(self) -> None:
        """
        Refresh all badges based on project language detection.

        If the project contains files of multiple languages, all badges
        show "Mixed". Otherwise, individual file languages are shown.
        """
        project_language = self._detect_project_language()
        is_mixed = project_language == "Mixed"

        if is_mixed:
            logger.info("Mixed-language project detected, updating badges")

        # Badge style mapping
        badge_style_map = {
            "Lua": "language_badge_lua",
            "Luau": "language_badge_luau",
            "Python": "language_badge_python",
            "Mixed": "language_badge_mixed",
        }

        for i in range(self._file_list.count()):
            item = self._file_list.item(i)
            if item:
                path_str = item.data(Qt.ItemDataRole.UserRole)
                widget = self._file_list.itemWidget(item)
                if widget and path_str in self._files:
                    # Find the badge label
                    for child in widget.findChildren(QLabel):
                        if child.text() in ("Lua", "Luau", "Python", "Mixed"):
                            if is_mixed:
                                child.setText("Mixed")
                                child.setStyleSheet(get_widget_style("language_badge_mixed"))
                            else:
                                file_lang = self._files[path_str]
                                child.setText(file_lang)
                                badge_style = badge_style_map.get(file_lang, "language_badge_lua")
                                child.setStyleSheet(get_widget_style(badge_style))
                            break

    def _update_empty_state(self) -> None:
        """Update visibility of empty state label."""
        is_empty = len(self._files) == 0
        self._empty_label.setVisible(is_empty)
        self._file_list.setVisible(not is_empty)
        self._update_file_count_display()

    def _update_file_count_display(self) -> None:
        """Update the file count label text and visibility."""
        count = len(self._files)
        if count == 1:
            self._file_count_label.setText("1 file selected")
        else:
            self._file_count_label.setText(f"{count} files selected")
        
        self._file_count_label.setVisible(count > 0)
 +++++++ REPLACE

    def get_files(self) -> List[str]:
        """
        Get the list of selected file paths.

        Returns:
            List of file path strings.
        """
        return list(self._files.keys())

    def get_files_with_languages(self) -> Dict[str, str]:
        """
        Get the selected files with their languages.

        Returns:
            Dictionary mapping file paths to their languages.
        """
        return dict(self._files)

    def clear_files(self) -> None:
        """Clear all selected files."""
        self._files.clear()
        self._file_list.clear()
        self._update_empty_state()
        self.files_changed.emit([])
        logger.info("Cleared all files")

    def set_files_with_languages(self, files_dict: Dict[str, str]) -> None:
        """
        Set the file selection from a dictionary of paths to languages.

        This clears any existing files and adds the provided files.
        Used for restoring file selection from saved profiles.

        Args:
            files_dict: Dictionary mapping file path strings to language strings.
        """
        # Clear existing files
        self._files.clear()
        self._file_list.clear()

        # Add each file with its specified language
        for path_str, language in files_dict.items():
            path = Path(path_str)
            # Skip if file doesn't exist
            if not path.exists():
                logger.warning(f"Skipping non-existent file from profile: {path_str}")
                continue

            # Add to internal dict with the saved language
            self._files[path_str] = language

            # Create list item
            self._create_file_item(path_str, language)

        self._update_empty_state()
        self._update_language_badges()
        self.files_changed.emit(self.get_files())
        logger.info(f"Set {len(self._files)} file(s) from profile")
