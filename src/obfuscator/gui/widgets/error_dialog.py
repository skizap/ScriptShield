"""Error handling dialog for displaying file processing errors and user decisions.

This module provides the ErrorHandlingDialog class which displays file processing
errors to users and captures their decision to continue or stop obfuscation.
"""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from obfuscator.gui.styles.stylesheet import COLORS, FONTS, SPACING


class ErrorHandlingDialog(QDialog):
    """Dialog for handling file processing errors during obfuscation.

    This dialog displays file errors to users and allows them to choose
    whether to continue processing remaining files or stop obfuscation.

    Attributes:
        file_path: Path to the file that failed processing
        errors: List of error messages from the processing failure
        _user_decision: Boolean indicating user choice (True = continue, False = stop)
    """

    def __init__(
        self,
        file_path: Path,
        errors: list[str],
        parent: QWidget | None = None
    ) -> None:
        """Initialize the error handling dialog.

        Args:
            file_path: Path to the file that encountered an error
            errors: List of error messages from processing failure
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.file_path = file_path
        self.errors = errors
        self._user_decision = False
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self) -> None:
        """Set up the dialog UI components."""
        self.setWindowTitle("Error Processing File")
        self.setMinimumWidth(500)
        self.setMinimumHeight(350)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title label
        title_label = QLabel("Error Processing File")
        title_label.setStyleSheet(f"""
            QLabel {{
                font-size: {FONTS['size_title']};
                font-weight: {FONTS['weight_bold']};
                color: {COLORS['text_primary']};
            }}
        """)
        layout.addWidget(title_label)

        # File name display (bold, monospace)
        file_label = QLabel(f"File: {self.file_path.name}")
        file_label.setStyleSheet(f"""
            QLabel {{
                font-weight: {FONTS['weight_bold']};
                color: {COLORS['text_primary']};
                font-family: monospace;
                font-size: 14px;
            }}
        """)
        layout.addWidget(file_label)

        # Full path label (secondary color)
        path_label = QLabel(f"Path: {self.file_path}")
        path_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_secondary']};
                font-family: monospace;
                font-size: 11px;
            }}
        """)
        layout.addWidget(path_label)

        # Error section label
        error_section_label = QLabel("Error Messages:")
        error_section_label.setStyleSheet(f"""
            QLabel {{
                font-weight: {FONTS['weight_bold']};
                color: {COLORS['text_primary']};
                padding-top: 8px;
            }}
        """)
        layout.addWidget(error_section_label)

        # Scrollable error message list
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
        list_layout.setContentsMargins(12, 12, 12, 12)
        list_layout.setSpacing(8)

        for error in self.errors:
            error_label = QLabel(f"• {error}")
            error_label.setStyleSheet(f"""
                QLabel {{
                    color: {COLORS['error']};
                    font-family: monospace;
                    font-size: 12px;
                    padding: 4px;
                }}
            """)
            error_label.setWordWrap(True)
            list_layout.addWidget(error_label)

        list_layout.addStretch()
        scroll.setWidget(list_container)
        layout.addWidget(scroll)

        # Question label
        question_label = QLabel("How would you like to proceed?")
        question_label.setStyleSheet(f"""
            QLabel {{
                font-weight: {FONTS['weight_bold']};
                color: {COLORS['text_primary']};
                padding-top: 8px;
            }}
        """)
        layout.addWidget(question_label)

        # Button box with custom buttons
        button_box = QDialogButtonBox()
        
        # Continue button
        continue_button = button_box.addButton(
            "Continue with Remaining Files",
            QDialogButtonBox.ButtonRole.AcceptRole
        )
        continue_button.setStyleSheet(f"""
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
        """)
        
        # Stop button
        stop_button = button_box.addButton(
            "Stop Obfuscation",
            QDialogButtonBox.ButtonRole.RejectRole
        )
        stop_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['danger']};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: {SPACING['radius_sm']};
                font-weight: {FONTS['weight_bold']};
            }}
            QPushButton:hover {{
                background-color: #aa3333;
            }}
        """)
        
        button_box.accepted.connect(self._on_continue)
        button_box.rejected.connect(self._on_stop)
        layout.addWidget(button_box)

    def _apply_styles(self) -> None:
        """Apply stylesheet to dialog."""
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_medium']};
            }}
        """)

    def _on_continue(self) -> None:
        """Handle continue button click - user chose to continue."""
        self._user_decision = True
        self.accept()

    def _on_stop(self) -> None:
        """Handle stop button click - user chose to stop."""
        self._user_decision = False
        self.reject()

    def get_user_decision(self) -> bool:
        """Get the user's decision from the dialog.

        Returns:
            True if user chose to continue processing remaining files,
            False if user chose to stop obfuscation.
        """
        return self._user_decision
