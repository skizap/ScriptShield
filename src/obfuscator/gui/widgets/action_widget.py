"""
Action widget for the Python Obfuscator GUI.

Provides the primary action button for starting obfuscation.
"""

from typing import Optional

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
)
from PyQt6.QtGui import QCursor

from obfuscator.utils.logger import get_logger
from obfuscator.gui.styles.stylesheet import get_widget_style

logger = get_logger("obfuscator.gui.widgets.action_widget")


class ActionWidget(QWidget):
    """Widget with the primary action button for starting obfuscation."""

    start_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the action widget."""
        super().__init__(parent)
        self._setup_ui()
        logger.debug("ActionWidget initialized")

    def _setup_ui(self) -> None:
        """Set up the widget UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Start button
        self._start_btn = QPushButton("Start Obfuscation")
        self._start_btn.setProperty("data-element-id", "start-obfuscation-button")
        self._start_btn.setToolTip("Start the obfuscation process with current settings")
        self._start_btn.setStyleSheet(get_widget_style("primary_button_disabled"))
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_clicked)
        layout.addWidget(self._start_btn)

    def _on_clicked(self) -> None:
        """Handle button click - emit start_clicked signal."""
        self.start_clicked.emit()
        logger.info("Start obfuscation button clicked")

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the start button."""
        self._start_btn.setEnabled(enabled)
        if enabled:
            self._start_btn.setStyleSheet(get_widget_style("primary_button"))
            self._start_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        else:
            self._start_btn.setStyleSheet(get_widget_style("primary_button_disabled"))
            self._start_btn.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        logger.debug(f"Start button enabled: {enabled}")

    def is_enabled(self) -> bool:
        """Return whether the start button is enabled."""
        return self._start_btn.isEnabled()
