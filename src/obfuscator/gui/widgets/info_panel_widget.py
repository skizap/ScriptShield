"""
Info panel widget for the Python Obfuscator GUI.

Displays helpful tips and information to guide users.
"""

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFrame,
)

from obfuscator.utils.logger import get_logger
from obfuscator.gui.styles.stylesheet import get_widget_style

logger = get_logger("obfuscator.gui.widgets.info_panel_widget")

# Tips to display in the info panel
TIPS = [
    "💡 Select multiple files for batch obfuscation",
    "🔒 Higher security levels increase processing time",
    "💾 Save your configuration as a profile for reuse",
    "📁 Output files will be saved to your selected location",
]


class InfoPanelWidget(QWidget):
    """Widget displaying helpful tips and information."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the info panel widget."""
        super().__init__(parent)
        self._setup_ui()
        logger.debug("InfoPanelWidget initialized")

    def _setup_ui(self) -> None:
        """Set up the widget UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Title
        title_label = QLabel("Quick Tips")
        title_label.setStyleSheet(get_widget_style("title_label"))
        title_label.setProperty("data-element-id", "info-panel-title")
        layout.addWidget(title_label)

        # Tips container frame
        tips_frame = QFrame()
        tips_frame.setProperty("data-element-id", "info-panel-content")
        tips_frame.setStyleSheet(get_widget_style("info_frame"))

        tips_layout = QVBoxLayout(tips_frame)
        tips_layout.setContentsMargins(12, 12, 12, 12)
        tips_layout.setSpacing(6)

        # Add tip labels
        for tip in TIPS:
            tip_lbl = QLabel(tip)
            tip_lbl.setStyleSheet(get_widget_style("tip_label"))
            tip_lbl.setWordWrap(True)
            tips_layout.addWidget(tip_lbl)

        layout.addWidget(tips_frame)
