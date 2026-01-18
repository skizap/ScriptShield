"""Progress widget for displaying obfuscation progress and logs."""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
)
from PyQt6.QtCore import pyqtSignal, QTimer, Qt
from PyQt6.QtGui import QFont, QCursor

from obfuscator.utils.logger import get_logger
from obfuscator.gui.styles.stylesheet import get_widget_style


class ProgressWidget(QWidget):
    """Widget for displaying obfuscation progress and logs."""

    # Signal emitted when cancel button is clicked
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the progress widget."""
        super().__init__(parent)
        self.logger = get_logger("obfuscator.gui.widgets.progress_widget")
        self._is_visible = False
        self._log_entry_counter = 0

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        # Set widget properties
        self.setObjectName("progressWidget")
        self.setProperty("data-element-id", "progress-widget-container")
        self.setStyleSheet(get_widget_style("progress_container"))

        # Create main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Add title label
        title_label = QLabel("Progress")
        title_label.setProperty("data-element-id", "progress-title")
        title_label.setStyleSheet(get_widget_style("title_label"))
        layout.addWidget(title_label)

        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setProperty("data-element-id", "progress-bar")
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(get_widget_style("progress_bar"))
        self.progress_bar.setMinimumHeight(25)
        layout.addWidget(self.progress_bar)

        # Create log container
        self.log_scroll_area = QScrollArea()
        self.log_scroll_area.setProperty("data-element-id", "log-container")
        self.log_scroll_area.setWidgetResizable(True)
        self.log_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.log_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.log_scroll_area.setStyleSheet(get_widget_style("log_container"))
        self.log_scroll_area.setMinimumHeight(200)

        # Create inner widget for log entries
        log_inner_widget = QWidget()
        log_inner_widget.setObjectName("logContent")
        self.log_layout = QVBoxLayout(log_inner_widget)
        self.log_layout.setContentsMargins(10, 10, 10, 10)
        self.log_layout.setSpacing(5)
        self.log_layout.addStretch()

        self.log_scroll_area.setWidget(log_inner_widget)
        layout.addWidget(self.log_scroll_area)

        # Create cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setProperty("data-element-id", "cancel-button")
        self.cancel_button.setStyleSheet(get_widget_style("danger_button"))
        self.cancel_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.cancel_button.setVisible(False)
        layout.addWidget(self.cancel_button)

        # Initially hide the widget
        self.setVisible(False)

    def show_progress(self):
        """Show the progress widget."""
        self._is_visible = True
        self.setVisible(True)
        self.cancel_button.setVisible(True)
        self.logger.info("Progress widget shown")

    def hide_progress(self):
        """Hide the progress widget."""
        self._is_visible = False
        self.setVisible(False)
        self.cancel_button.setVisible(False)
        self.logger.info("Progress widget hidden")

    def is_visible(self) -> bool:
        """Check if the progress widget is visible."""
        return self._is_visible

    def set_progress(self, value: int):
        """Set the progress bar value (0-100)."""
        self.progress_bar.setValue(value)
        self.logger.debug(f"Progress updated to {value}%")

    def get_progress(self) -> int:
        """Get the current progress bar value."""
        return self.progress_bar.value()

    def add_log_entry(self, message: str, level: str = "info"):
        """Add a log entry to the log container."""
        from datetime import datetime

        # Create timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"

        # Create label
        log_label = QLabel(full_message)
        log_label.setProperty("data-element-id", f"log-entry-{self._log_entry_counter}")
        log_label.setWordWrap(True)

        # Apply style based on level
        if level == "success":
            log_label.setStyleSheet(get_widget_style("log_entry_success"))
        elif level == "warning":
            log_label.setStyleSheet(get_widget_style("log_entry_warning"))
        elif level == "error":
            log_label.setStyleSheet(get_widget_style("log_entry_error"))
        else:  # info
            log_label.setStyleSheet(get_widget_style("log_entry_info"))

        # Add to layout (before the stretch)
        self.log_layout.insertWidget(self.log_layout.count() - 1, log_label)
        self._log_entry_counter += 1

        # Scroll to bottom
        self._scroll_to_bottom()

        self.logger.debug(f"Log entry added: [{level}] {message}")

    def clear_logs(self):
        """Clear all log entries."""
        # Remove all widgets except the stretch
        while self.log_layout.count() > 1:
            item = self.log_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._log_entry_counter = 0
        self.logger.info("Logs cleared")

    def reset(self):
        """Reset the progress widget to initial state."""
        self.set_progress(0)
        self.clear_logs()
        self.hide_progress()
        self.logger.info("Progress widget reset")

    def _scroll_to_bottom(self):
        """Scroll the log container to the bottom."""
        QTimer.singleShot(0, lambda: self.log_scroll_area.verticalScrollBar().setValue(
            self.log_scroll_area.verticalScrollBar().maximum()
        ))

    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.logger.info("Cancel requested by user")
        self.cancel_requested.emit()
