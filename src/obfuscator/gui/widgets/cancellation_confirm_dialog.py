"""Cancellation confirmation dialog for stopping active obfuscation jobs."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget

from obfuscator.gui.styles.stylesheet import COLORS, FONTS, SPACING


class CancellationConfirmDialog(QDialog):
    """Dialog that confirms cancellation with progress context."""

    def __init__(
        self,
        completed_count: int,
        total_count: int,
        current_file: str | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.completed_count = max(0, completed_count)
        self.total_count = max(0, total_count)
        self.current_file = current_file.strip() if current_file else None
        self._user_decision = False

        self._setup_ui()
        self._apply_styles()
        self.setModal(True)

    def _setup_ui(self) -> None:
        """Set up dialog UI components."""
        self.setWindowTitle("Confirm Cancellation")
        self.setMinimumWidth(450)
        self.setMinimumHeight(300)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("Confirm Cancellation")
        title_label.setStyleSheet(
            f"""
            QLabel {{
                font-size: {FONTS['size_title']};
                font-weight: {FONTS['weight_bold']};
                color: {COLORS['text_primary']};
            }}
            """
        )
        layout.addWidget(title_label)

        warning_label = QLabel("Are you sure you want to stop the obfuscation process?")
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(warning_label)

        summary_container = QWidget()
        summary_container.setObjectName("progressSummary")
        summary_layout = QVBoxLayout(summary_container)
        summary_layout.setSpacing(8)
        summary_layout.setContentsMargins(12, 12, 12, 12)

        completed_label = QLabel(
            f"Files completed: {self.completed_count} / {self.total_count}"
        )
        completed_label.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-family: monospace;"
        )
        summary_layout.addWidget(completed_label)

        current_file_display = self.current_file or "N/A"
        current_file_label = QLabel(f"Current file: {current_file_display}")
        current_file_label.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-family: monospace;"
        )
        current_file_label.setWordWrap(True)
        summary_layout.addWidget(current_file_label)

        percentage_label = QLabel(f"Progress: {self._calculate_percentage()}%")
        percentage_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-weight: {FONTS['weight_bold']};"
        )
        summary_layout.addWidget(percentage_label)

        progress_hint = QLabel(self._get_progress_hint())
        progress_hint.setWordWrap(True)
        progress_hint.setStyleSheet(
            f"color: {COLORS['warning']}; font-size: {FONTS['size_small']};"
        )
        summary_layout.addWidget(progress_hint)

        layout.addWidget(summary_container)

        question_label = QLabel("Obfuscation will stop after the current file completes.")
        question_label.setWordWrap(True)
        question_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_body']};"
        )
        layout.addWidget(question_label)

        button_box = QDialogButtonBox()

        continue_button = button_box.addButton(
            "Continue Processing",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        continue_button.setStyleSheet(
            f"""
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
            """
        )

        stop_button = button_box.addButton(
            "Stop Obfuscation",
            QDialogButtonBox.ButtonRole.RejectRole,
        )
        stop_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLORS['danger']};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: {SPACING['radius_sm']};
                font-weight: {FONTS['weight_bold']};
            }}
            QPushButton:hover {{
                background-color: {COLORS['error']};
            }}
            """
        )

        button_box.accepted.connect(self._on_continue)
        button_box.rejected.connect(self._on_stop)
        layout.addWidget(button_box)

    def _apply_styles(self) -> None:
        """Apply stylesheet to dialog and summary container."""
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {COLORS['bg_medium']};
            }}
            QWidget#progressSummary {{
                background-color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['border_default']};
                border-radius: {SPACING['radius_sm']};
            }}
            """
        )

    def _calculate_percentage(self) -> int:
        """Calculate progress percentage safely."""
        if self.total_count <= 0:
            return 0
        percentage = int((self.completed_count / self.total_count) * 100)
        return max(0, min(percentage, 100))

    def _get_progress_hint(self) -> str:
        """Get context-aware progress hint message."""
        if self.total_count > 0 and self.completed_count >= self.total_count:
            return "Processing is nearly complete"
        if self.completed_count == 0:
            return "No files have been completed yet"
        return "Completed files will remain in the output directory"

    def _on_continue(self) -> None:
        """Handle continue button click - keep processing."""
        self._user_decision = False
        self.reject()

    def _on_stop(self) -> None:
        """Handle stop button click - confirm cancellation."""
        self._user_decision = True
        self.accept()

    def get_user_decision(self) -> bool:
        """Return True if user chose to stop, False to continue."""
        return self._user_decision

    def showEvent(self, event) -> None:
        """Center the dialog on its parent window when shown."""
        super().showEvent(event)
        parent = self.parentWidget()
        if parent is None:
            return

        parent_rect = parent.frameGeometry()
        self.move(parent_rect.center() - self.rect().center())
