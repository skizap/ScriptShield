"""
GUI package for the Lua Obfuscator application.

This package provides the graphical user interface components built with PyQt6.
It includes the main window and all associated widgets for file selection,
configuration, and obfuscation control.

Example:
    >>> from obfuscator.gui import MainWindow
    >>> window = MainWindow()
    >>> window.show()
"""

from .main_window import MainWindow

__all__ = ["MainWindow"]

