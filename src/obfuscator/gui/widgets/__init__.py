"""
Widgets package for the Lua Obfuscator GUI.

This package provides reusable GUI widgets including file selection,
configuration panels, and other UI components.

Example:
    >>> from obfuscator.gui.widgets import FileSelectionWidget, SecurityConfigWidget
    >>> file_widget = FileSelectionWidget()
    >>> security_widget = SecurityConfigWidget()
"""

from .action_widget import ActionWidget
from .file_selection_widget import FileSelectionWidget
from .info_panel_widget import InfoPanelWidget
from .output_widget import OutputWidget
from .profile_widget import ProfileWidget
from .progress_widget import ProgressWidget
from .security_config_widget import SecurityConfigWidget

__all__ = [
    "ActionWidget",
    "FileSelectionWidget",
    "InfoPanelWidget",
    "OutputWidget",
    "ProfileWidget",
    "ProgressWidget",
    "SecurityConfigWidget",
]

