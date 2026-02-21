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
from .cancellation_confirm_dialog import CancellationConfirmDialog
from .conflict_dialog import ConflictResolutionDialog
from .error_dialog import ErrorHandlingDialog
from .error_report_dialog import ErrorReportDialog
from .file_selection_widget import FileSelectionWidget
from .info_panel_widget import InfoPanelWidget
from .output_widget import OutputWidget
from .profile_widget import ProfileWidget
from .progress_widget import ProgressWidget
from .resume_checkpoint_dialog import ResumeCheckpointDialog
from .security_config_widget import SecurityConfigWidget
from .start_confirmation_dialog import StartConfirmationDialog

__all__ = [
    "ActionWidget",
    "CancellationConfirmDialog",
    "ConflictResolutionDialog",
    "ErrorHandlingDialog",
    "ErrorReportDialog",
    "FileSelectionWidget",
    "InfoPanelWidget",
    "OutputWidget",
    "ProfileWidget",
    "ProgressWidget",
    "ResumeCheckpointDialog",
    "SecurityConfigWidget",
    "StartConfirmationDialog",
]

