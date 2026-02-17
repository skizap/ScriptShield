"""Error report dialog for viewing and exporting detailed obfuscation errors."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from obfuscator.gui.styles.stylesheet import COLORS, FONTS, SPACING


class ErrorReportDialog(QDialog):
    """Dialog for reviewing structured processing errors and exporting a report."""

    def __init__(
        self,
        detailed_errors: list[dict[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.detailed_errors = detailed_errors or []

        self._setup_ui()
        self._apply_styles()
        self.setModal(True)

    def _setup_ui(self) -> None:
        """Set up dialog UI components."""
        self.setWindowTitle("Error Report")
        self.setMinimumWidth(650)
        self.setMinimumHeight(420)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("Error Report")
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

        summary_label = QLabel(f"Total errors: {len(self.detailed_errors)}")
        summary_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_body']};"
        )
        layout.addWidget(summary_label)

        error_scroll = QScrollArea()
        error_scroll.setWidgetResizable(True)
        error_scroll.setStyleSheet(
            f"""
            QScrollArea {{
                background-color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['border_default']};
                border-radius: {SPACING['radius_sm']};
            }}
            """
        )

        error_container = QWidget()
        error_container.setObjectName("errorReportList")
        error_layout = QVBoxLayout(error_container)
        error_layout.setSpacing(8)
        error_layout.setContentsMargins(12, 12, 12, 12)

        if self.detailed_errors:
            for error in self.detailed_errors:
                error_text = self._format_error_line(error)
                error_label = QLabel(error_text)
                error_label.setWordWrap(True)
                error_label.setStyleSheet(
                    f"""
                    QLabel {{
                        color: {COLORS['error']};
                        font-family: monospace;
                        font-size: {FONTS['size_small']};
                        padding: 2px;
                    }}
                    """
                )
                error_layout.addWidget(error_label)
        else:
            no_error_label = QLabel("No detailed errors available.")
            no_error_label.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_body']};"
            )
            error_layout.addWidget(no_error_label)

        error_layout.addStretch()
        error_scroll.setWidget(error_container)
        layout.addWidget(error_scroll)

        button_box = QDialogButtonBox()

        export_button = button_box.addButton(
            "Export Report",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        export_button.setStyleSheet(
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

        close_button = button_box.addButton(
            "Close",
            QDialogButtonBox.ButtonRole.RejectRole,
        )
        close_button.setStyleSheet(
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

        export_button.clicked.connect(self._on_export)
        close_button.clicked.connect(self.reject)
        layout.addWidget(button_box)

    def _apply_styles(self) -> None:
        """Apply stylesheet to dialog and error list container."""
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {COLORS['bg_medium']};
            }}
            QWidget#errorReportList {{
                background-color: {COLORS['bg_dark']};
                border-radius: {SPACING['radius_sm']};
            }}
            """
        )

    def _format_error_line(self, error: dict[str, Any]) -> str:
        """Format a detailed error for compact monospace display."""
        file_path = str(error.get("file_path") or "Unknown file")
        line = error.get("line")
        column = error.get("column")
        error_type = str(error.get("error_type") or "UnknownError")
        message = str(error.get("message") or "Unknown error")

        if line is not None and column is not None:
            location = f"{file_path}:{line}:{column}"
        elif line is not None:
            location = f"{file_path}:{line}"
        elif column is not None:
            location = f"{file_path}:{column}"
        else:
            location = file_path

        return f"{location}: {error_type}: {message}"

    def _format_grouped_error(self, error: dict[str, Any]) -> str:
        """Format a detailed error for grouped export output."""
        line = error.get("line")
        column = error.get("column")
        error_type = str(error.get("error_type") or "UnknownError")
        message = str(error.get("message") or "Unknown error")

        location_parts: list[str] = []
        if line is not None:
            location_parts.append(f"Line {line}")
        if column is not None:
            location_parts.append(f"Column {column}")

        if location_parts:
            return f"{', '.join(location_parts)}: {error_type}: {message}"
        return f"{error_type}: {message}"

    def _generate_report_text(self) -> str:
        """Generate a text error report grouped by file and error type."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        errors_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
        errors_by_type: dict[str, int] = defaultdict(int)

        for error in self.detailed_errors:
            file_path = str(error.get("file_path") or "Unknown file")
            error_type = str(error.get("error_type") or "UnknownError")

            errors_by_file[file_path].append(error)
            errors_by_type[error_type] += 1

        lines = [
            "=== Obfuscation Error Report ===",
            f"Generated: {timestamp}",
            f"Total Errors: {len(self.detailed_errors)}",
            "",
            "Errors by File:",
        ]

        if errors_by_file:
            for file_path, file_errors in errors_by_file.items():
                lines.append(f"  {file_path} ({len(file_errors)} errors):")
                for error in file_errors:
                    lines.append(f"    - {self._format_grouped_error(error)}")
                lines.append("")
        else:
            lines.append("  None")
            lines.append("")

        lines.append("Errors by Type:")
        if errors_by_type:
            for error_type, count in sorted(errors_by_type.items()):
                lines.append(f"  {error_type}: {count}")
        else:
            lines.append("  None")

        return "\n".join(lines).rstrip()

    def _on_export(self) -> None:
        """Export the current error report to a text file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"error_report_{timestamp}.txt"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Error Report",
            default_name,
            "Text Files (*.txt);;All Files (*)",
        )
        if not file_path:
            return

        report_content = self._generate_report_text()

        try:
            with open(file_path, "w", encoding="utf-8") as report_file:
                report_file.write(report_content)
                report_file.write("\n")

            QMessageBox.information(
                self,
                "Export Successful",
                f"Error report exported to:\n{file_path}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export error report:\n{exc}",
            )

    def showEvent(self, event) -> None:
        """Center the dialog on parent when shown."""
        super().showEvent(event)
        parent = self.parentWidget()
        if parent is None:
            return

        parent_rect = parent.frameGeometry()
        self.move(parent_rect.center() - self.rect().center())
