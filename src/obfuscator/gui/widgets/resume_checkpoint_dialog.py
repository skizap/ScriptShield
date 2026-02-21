"""Resume checkpoint dialog for continuing a previous obfuscation session."""

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


class ResumeCheckpointDialog(QDialog):
    """Dialog that prompts the user to resume from the latest checkpoint."""

    def __init__(
        self,
        timestamp: str,
        files_completed: int,
        total_files: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.timestamp = timestamp
        self.files_completed = files_completed
        self.total_files = total_files
        self._user_decision: bool = False

        self._setup_ui()
        self._apply_styles()
        self.setModal(True)

    def _setup_ui(self) -> None:
        """Set up dialog UI components."""
        self.setWindowTitle("Resume Previous Session")
        self.setMinimumWidth(480)
        self.setMinimumHeight(300)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("Resume Previous Session")
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

        description_label = QLabel(
            "A previous obfuscation session was found. Would you like to resume it?"
        )
        description_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        summary_scroll = QScrollArea()
        summary_scroll.setWidgetResizable(True)
        summary_scroll.setStyleSheet(
            f"""
            QScrollArea {{
                background-color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['border_default']};
                border-radius: {SPACING['radius_sm']};
            }}
            """
        )

        summary_container = QWidget()
        summary_container.setObjectName("summaryContainer")
        summary_layout = QVBoxLayout(summary_container)
        summary_layout.setSpacing(10)
        summary_layout.setContentsMargins(12, 12, 12, 12)

        formatted_timestamp = (self.timestamp or "Unknown").replace("T", " ")
        if "." in formatted_timestamp:
            formatted_timestamp = formatted_timestamp.split(".", maxsplit=1)[0]

        progress_percent = (
            int(self.files_completed / self.total_files * 100) if self.total_files else 0
        )

        self._add_summary_row(
            summary_layout,
            "Session timestamp",
            formatted_timestamp,
            monospace=True,
        )
        self._add_summary_row(
            summary_layout,
            "Files completed",
            f"{self.files_completed} / {self.total_files}",
            monospace=True,
        )
        self._add_summary_row(
            summary_layout,
            "Progress",
            f"{progress_percent}%",
            monospace=True,
        )

        summary_layout.addStretch()
        summary_scroll.setWidget(summary_container)
        layout.addWidget(summary_scroll)

        button_box = QDialogButtonBox()

        resume_button = button_box.addButton(
            "Resume",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        resume_button.setStyleSheet(
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

        start_fresh_button = button_box.addButton(
            "Start Fresh",
            QDialogButtonBox.ButtonRole.RejectRole,
        )
        start_fresh_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLORS['bg_lighter']};
                color: {COLORS['text_primary']};
                border: none;
                padding: 8px 16px;
                border-radius: {SPACING['radius_sm']};
                font-weight: {FONTS['weight_normal']};
            }}
            QPushButton:hover {{
                background-color: {COLORS['border_default']};
            }}
            """
        )

        button_box.accepted.connect(self._on_resume)
        button_box.rejected.connect(self._on_start_fresh)
        layout.addWidget(button_box)

    def _add_summary_row(
        self,
        parent_layout: QVBoxLayout,
        label_text: str,
        value_text: str,
        value_color: str = COLORS["text_primary"],
        monospace: bool = False,
    ) -> None:
        """Add a label/value row to the summary panel."""
        label = QLabel(f"{label_text}:")
        label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_body']};"
        )
        parent_layout.addWidget(label)

        value_font_family = "font-family: monospace;" if monospace else ""
        value = QLabel(value_text)
        value.setWordWrap(True)
        value.setStyleSheet(
            f"color: {value_color}; {value_font_family} font-size: {FONTS['size_body']};"
        )
        parent_layout.addWidget(value)

    def _apply_styles(self) -> None:
        """Apply stylesheet to dialog and summary container."""
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {COLORS['bg_medium']};
            }}
            QWidget#summaryContainer {{
                background-color: {COLORS['bg_dark']};
                border-radius: {SPACING['radius_sm']};
            }}
            """
        )

    def _on_resume(self) -> None:
        """Handle resume button click."""
        self._user_decision = True
        self.accept()

    def _on_start_fresh(self) -> None:
        """Handle start fresh button click."""
        self._user_decision = False
        self.reject()

    def get_user_decision(self) -> bool:
        """Return True if user chose to resume, False to start fresh."""
        return self._user_decision

    def showEvent(self, event) -> None:
        """Center the dialog on parent when shown."""
        super().showEvent(event)
        parent = self.parentWidget()
        if parent is None:
            return

        parent_rect = parent.frameGeometry()
        self.move(parent_rect.center() - self.rect().center())
