"""Start confirmation dialog for reviewing obfuscation settings."""

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


class StartConfirmationDialog(QDialog):
    """Dialog that summarizes obfuscation settings before execution."""

    def __init__(
        self,
        file_count: int,
        preset: str,
        output_path: Path,
        runtime_mode: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.file_count = max(0, file_count)
        self.preset = preset or "Unknown"
        self.output_path = output_path
        self.runtime_mode = (runtime_mode or "Unknown").capitalize()
        self._user_decision = False

        self._setup_ui()
        self._apply_styles()
        self.setModal(True)

    def _setup_ui(self) -> None:
        """Set up dialog UI components."""
        self.setWindowTitle("Confirm Obfuscation")
        self.setMinimumWidth(500)
        self.setMinimumHeight(350)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("Confirm Obfuscation")
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

        description_label = QLabel("Review the obfuscation settings before processing:")
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

        files_value = str(self.file_count)
        if self.file_count > 100:
            files_value = f"{self.file_count} [!] Large batch"

        self._add_summary_row(
            summary_layout,
            "Files to process",
            files_value,
            value_color=COLORS["warning"] if self.file_count > 100 else COLORS["text_primary"],
            monospace=True,
        )
        self._add_summary_row(summary_layout, "Security level", self.preset)
        self._add_summary_row(summary_layout, "Runtime mode", self.runtime_mode)
        self._add_summary_row(
            summary_layout,
            "Output destination",
            self._format_output_path(self.output_path),
            monospace=True,
        )
        self._add_summary_row(
            summary_layout,
            "Estimated time",
            self._estimate_time(self.file_count),
            monospace=True,
        )

        if self.file_count > 100:
            warning_label = QLabel("[!] This is a large batch and may take longer than estimated.")
            warning_label.setWordWrap(True)
            warning_label.setStyleSheet(f"color: {COLORS['warning']};")
            summary_layout.addWidget(warning_label)

        summary_layout.addStretch()
        summary_scroll.setWidget(summary_container)
        layout.addWidget(summary_scroll)

        info_label = QLabel(
            "This operation may take several minutes depending on file count and security level."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_body']};"
        )
        layout.addWidget(info_label)

        button_box = QDialogButtonBox()

        start_button = button_box.addButton(
            "Start Obfuscation",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        start_button.setStyleSheet(
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

        cancel_button = button_box.addButton(
            "Cancel",
            QDialogButtonBox.ButtonRole.RejectRole,
        )
        cancel_button.setStyleSheet(
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

        button_box.accepted.connect(self._on_start)
        button_box.rejected.connect(self._on_cancel)
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

    def _estimate_time(self, file_count: int) -> str:
        """Estimate execution time based on a fixed per-file heuristic."""
        total_seconds = max(0, file_count) * 2
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _format_output_path(self, output_path: Path) -> str:
        """Format output path to keep long paths readable."""
        output_text = str(output_path) if output_path else "N/A"
        if len(output_text) <= 70:
            return output_text

        path_parts = output_path.parts
        if len(path_parts) <= 3:
            return output_text

        return f".../{'/'.join(path_parts[-3:])}"

    def _on_start(self) -> None:
        """Handle start button click."""
        self._user_decision = True
        self.accept()

    def _on_cancel(self) -> None:
        """Handle cancel button click."""
        self._user_decision = False
        self.reject()

    def get_user_decision(self) -> bool:
        """Return True if user chose to start, False to cancel."""
        return self._user_decision

    def showEvent(self, event) -> None:
        """Center the dialog on parent when shown."""
        super().showEvent(event)
        parent = self.parentWidget()
        if parent is None:
            return

        parent_rect = parent.frameGeometry()
        self.move(parent_rect.center() - self.rect().center())
