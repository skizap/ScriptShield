"""
Application entry point for the Lua Obfuscator GUI.

This module initializes the PyQt6 application and creates the main window.

Example:
    Run the application from command line:
    $ python -m obfuscator.main
"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from obfuscator.gui import MainWindow
from obfuscator.utils.logger import setup_logger

# Application metadata
APP_NAME = "Lua Obfuscator"
APP_VERSION = "0.1.0"
ORGANIZATION_NAME = "Obfuscator Developers"

# Baseline stylesheet for consistent appearance
BASELINE_STYLESHEET = """
QMainWindow {
    background-color: #2b2b2b;
}

QWidget {
    font-family: "Segoe UI", "Ubuntu", "Roboto", sans-serif;
    font-size: 13px;
    color: #e0e0e0;
}

QLabel {
    color: #e0e0e0;
}

QPushButton {
    background-color: #3c3f41;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 6px 12px;
    color: #e0e0e0;
}

QPushButton:hover {
    background-color: #4a4d4f;
    border-color: #666666;
}

QPushButton:pressed {
    background-color: #2d2f31;
}

QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #1e1e1e;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px;
    color: #e0e0e0;
}

QGroupBox {
    border: 1px solid #555555;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
"""


def main() -> int:
    """
    Initialize and run the Lua Obfuscator application.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # Setup application logger
    logger = setup_logger(
        "obfuscator",
        level="INFO",
        log_file=Path("logs/obfuscator.log")
    )

    logger.info(f"Application starting - version {APP_VERSION}")

    try:
        # Create Qt application
        app = QApplication(sys.argv)

        # Set application metadata
        app.setApplicationName(APP_NAME)
        app.setOrganizationName(ORGANIZATION_NAME)
        app.setApplicationVersion(APP_VERSION)

        # Apply baseline stylesheet
        app.setStyleSheet(BASELINE_STYLESHEET)

        # Create and show main window
        window = MainWindow()
        window.show()

        logger.info("Main window displayed")

        # Start event loop
        return app.exec()

    except Exception as e:
        logger.exception(f"Fatal error during application startup: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

