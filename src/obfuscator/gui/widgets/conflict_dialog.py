"""Conflict resolution dialog for handling file output conflicts.

This module provides the ConflictResolutionDialog class which displays
a dialog for users to choose how to handle file conflicts during obfuscation.
"""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QButtonGroup,
)

from obfuscator.gui.styles.stylesheet import COLORS, FONTS, SPACING
from obfuscator.core.orchestrator import ConflictStrategy, ConflictInfo


class ConflictResolutionDialog(QDialog):
    """Dialog for resolving file conflicts during obfuscation.

    This dialog displays conflicting files and allows users to choose
    a resolution strategy: OVERWRITE, SKIP, or RENAME.

    Attributes:
        conflicts: List of ConflictInfo objects representing file conflicts
        selected_strategy: The conflict resolution strategy chosen by user
    """

    def __init__(self, conflicts: list[ConflictInfo], parent: QWidget | None = None) -> None:
        """Initialize the conflict resolution dialog.

        Args:
            conflicts: List of ConflictInfo objects with conflict details
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.conflicts = conflicts
        self.selected_strategy: ConflictStrategy | None = None
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self) -> None:
        """Set up the dialog UI components."""
        self.setWindowTitle("File Conflicts Detected")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title label
        title_label = QLabel("File Conflicts Detected")
        title_label.setStyleSheet(f"""
            QLabel {{
                font-size: {FONTS['size_title']};
                font-weight: {FONTS['weight_bold']};
                color: {COLORS['text_primary']};
            }}
        """)
        layout.addWidget(title_label)

        # Description label
        desc_label = QLabel(
            f"The following {len(self.conflicts)} output file(s) already exist:\n"
            "Choose how to handle these conflicts:"
        )
        desc_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(desc_label)

        # Scrollable list of conflicts
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['border_default']};
                border-radius: {SPACING['radius_sm']};
            }}
        """)

        list_container = QWidget()
        list_layout = QVBoxLayout(list_container)
        list_layout.setContentsMargins(8, 8, 8, 8)
        list_layout.setSpacing(4)

        for conflict in self.conflicts:
            path_label = QLabel(f"• {conflict.output_path}")
            path_label.setStyleSheet(f"""
                QLabel {{
                    color: {COLORS['text_primary']};
                    font-family: monospace;
                    font-size: 12px;
                    padding: 4px;
                }}
            """)
            list_layout.addWidget(path_label)

        list_layout.addStretch()
        scroll.setWidget(list_container)
        layout.addWidget(scroll)

        # Strategy selection group
        strategy_label = QLabel("Resolution Strategy:")
        strategy_label.setStyleSheet(f"""
            QLabel {{
                font-weight: {FONTS['weight_bold']};
                color: {COLORS['text_primary']};
                padding-top: 8px;
            }}
        """)
        layout.addWidget(strategy_label)

        # Radio button group for strategies
        self.strategy_group = QButtonGroup(self)

        self.overwrite_radio = QRadioButton("Overwrite all existing files")
        self.overwrite_radio.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.strategy_group.addButton(self.overwrite_radio, 1)
        layout.addWidget(self.overwrite_radio)

        self.skip_radio = QRadioButton("Skip all existing files")
        self.skip_radio.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.strategy_group.addButton(self.skip_radio, 2)
        layout.addWidget(self.skip_radio)

        self.rename_radio = QRadioButton("Rename all conflicting files (append timestamp)")
        self.rename_radio.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.strategy_group.addButton(self.rename_radio, 3)
        layout.addWidget(self.rename_radio)

        # Default to rename as safest option
        self.rename_radio.setChecked(True)

        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _apply_styles(self) -> None:
        """Apply stylesheet to dialog."""
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_medium']};
            }}
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
            }}
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: {SPACING['radius_sm']};
                font-weight: {FONTS['weight_bold']};
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_hover']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['primary_pressed']};
            }}
        """)

    def _on_accept(self) -> None:
        """Handle OK button click - set selected strategy."""
        checked_id = self.strategy_group.checkedId()
        if checked_id == 1:
            self.selected_strategy = ConflictStrategy.OVERWRITE
        elif checked_id == 2:
            self.selected_strategy = ConflictStrategy.SKIP
        elif checked_id == 3:
            self.selected_strategy = ConflictStrategy.RENAME
        self.accept()

    def get_selected_strategy(self) -> ConflictStrategy | None:
        """Get the selected conflict resolution strategy.

        Returns:
            The selected ConflictStrategy, or None if dialog was cancelled
        """
        return self.selected_strategy
