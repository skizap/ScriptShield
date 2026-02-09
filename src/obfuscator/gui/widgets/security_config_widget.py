"""
Security Configuration Widget for the Python Obfuscator GUI.

Provides preset security levels and advanced feature toggles for configuring
obfuscation settings.
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QFrame,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QCursor

from typing import TYPE_CHECKING

from obfuscator.utils.logger import get_logger
from obfuscator.gui.styles.stylesheet import get_widget_style, COLORS

if TYPE_CHECKING:
    from obfuscator.gui.widgets.file_selection_widget import FileSelectionWidget

logger = get_logger("obfuscator.gui.widgets.security_config_widget")

# Feature tooltips
FEATURE_TOOLTIPS = {
    "Variable Renaming": "Renames variables to obscure names",
    "Function Renaming": "Renames functions to obscure names",
    "String Encryption": "Encrypts string literals",
    "Number Obfuscation": "Obfuscates numeric constants",
    "Dead Code Injection": "Adds unreachable code to confuse analysis",
    "Comment Removal": "Removes all comments from code",
    "Control Flow Flattening": "Restructures control flow to make it harder to follow",
    "Opaque Predicates": "Adds always-true/false conditions",
    "Constant Folding": "Pre-computes constant expressions",
    "Anti-Debug": "Adds debugger detection checks",
    "VM Protection": "Wraps code in virtual machine layer",
    "Bytecode Compilation": "Compiles to bytecode format",
    "Roblox Exploit Defense": "Detects and blocks Roblox exploit executors (Synapse, KRNL, Script-Ware) with integrity checks and environment fingerprinting",
    "Roblox Remote Spy Protection": "Encrypts RemoteEvent/RemoteFunction names and obfuscates argument patterns to prevent remote spy tools",
}

# Preset tooltips
PRESET_TOOLTIPS = {
    "Light": "Basic protection: Variable/function renaming and comment removal. Fast processing.",
    "Medium": "Balanced security: Adds string encryption and number obfuscation. Moderate processing time.",
    "Heavy": "Strong protection: Includes control flow flattening and dead code injection. Slower processing.",
    "Maximum": "Ultimate security: All features enabled including VM protection. Longest processing time.",
}

# Feature definitions organized by category
CORE_FEATURES = [
    "Variable Renaming",
    "Function Renaming",
    "String Encryption",
    "Number Obfuscation",
    "Dead Code Injection",
    "Comment Removal",
]

ADVANCED_FEATURES = [
    "Control Flow Flattening",
    "Opaque Predicates",
    "Constant Folding",
    "Anti-Debug",
    "VM Protection",
    "Bytecode Compilation",
]

ROBLOX_FEATURES = [
    "Roblox Exploit Defense",
    "Roblox Remote Spy Protection",
]

# Preset configurations mapping preset names to enabled features
PRESET_CONFIGS = {
    "Light": ["Variable Renaming", "Function Renaming", "Comment Removal"],
    "Medium": [
        "Variable Renaming",
        "Function Renaming",
        "Comment Removal",
        "String Encryption",
        "Number Obfuscation",
    ],
    "Heavy": [
        "Variable Renaming",
        "Function Renaming",
        "Comment Removal",
        "String Encryption",
        "Number Obfuscation",
        "Control Flow Flattening",
        "Dead Code Injection",
        "Opaque Predicates",
        "Constant Folding",
    ],
    "Maximum": CORE_FEATURES + ADVANCED_FEATURES + ROBLOX_FEATURES,
}


class SecurityConfigWidget(QWidget):
    """Widget for configuring security/obfuscation settings.

    This widget provides preset security levels and advanced feature toggles
    for configuring obfuscation settings. It supports language-aware feature
    visibility - Roblox-specific features are only shown for Lua/Luau projects.

    Language Detection:
        The widget connects to the FileSelectionWidget to monitor file changes
        and automatically show/hide Roblox-specific features based on the
        detected project language. Roblox features are shown for Lua, Luau,
        and Mixed language projects, but hidden for Python-only projects.

    Features:
        - Preset security levels (Light, Medium, Heavy, Maximum)
        - Advanced feature toggles with "Lua Only" badges for Roblox features
        - Dynamic language-aware visibility for language-specific features
        - Profile save/load compatibility with language filtering
    """

    config_changed = pyqtSignal(dict)

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._current_preset: str | None = "Light"
        self._features: dict[str, bool] = {}
        self._advanced_expanded: bool = False
        self._preset_buttons: dict[str, QPushButton] = {}
        self._feature_checkboxes: dict[str, QCheckBox] = {}
        self._current_language: str = "Lua"  # Track current project language
        self._file_selection_widget: "FileSelectionWidget | None" = None  # Reference to file selection widget
        self._roblox_section_widgets: list[QWidget] = []  # Track Roblox section widgets for visibility

        self._init_features()
        self._setup_ui()
        self._on_preset_clicked("Light")

        logger.debug("SecurityConfigWidget initialized")

    def _init_features(self) -> None:
        """Initialize all features to False."""
        all_features = CORE_FEATURES + ADVANCED_FEATURES + ROBLOX_FEATURES
        for feature in all_features:
            self._features[feature] = False

    def set_file_selection_widget(self, file_selection_widget: "FileSelectionWidget") -> None:
        """Connect to file selection widget for language detection.

        This method establishes a connection between the security configuration
        widget and the file selection widget. It allows the security widget to
        monitor file selection changes and automatically show/hide Roblox-specific
        features based on the detected project language.

        The language detection flow:
        1. User adds/removes files in FileSelectionWidget
        2. FileSelectionWidget emits files_changed signal
        3. _on_files_changed handler detects project language
        4. _update_language_visibility shows/hides Roblox features accordingly

        Args:
            file_selection_widget: The FileSelectionWidget instance to connect to.

        Note:
            This should be called during widget initialization, after both
            widgets are created but before the main window is shown.
        """
        self._file_selection_widget = file_selection_widget
        self._file_selection_widget.files_changed.connect(self._on_files_changed)
        self._update_language_visibility()
        logger.debug("Connected to FileSelectionWidget for language detection")

    def _on_files_changed(self, files: list) -> None:
        """Handle file selection changes to update language-specific features.

        Args:
            files: List of file paths (not used directly, we get languages from widget).
        """
        if not self._file_selection_widget:
            return

        # Get files with languages from file selection widget
        files_with_languages = self._file_selection_widget.get_files_with_languages()

        # Detect project language using file selection widget's method
        detected_language = self._file_selection_widget._detect_project_language()
        self._current_language = detected_language

        # Check if any Python files are present (for mixed projects, treat as Python-only)
        languages = set(files_with_languages.values()) if files_with_languages else set()
        self._python_present = "Python" in languages

        logger.debug(f"Language detected: {detected_language}, Python present: {self._python_present} from {len(files_with_languages)} files")
        self._update_language_visibility()

    def _update_language_visibility(self) -> None:
        """Show/hide Roblox features based on current language.

        Shows Roblox features section only when no Python files are present.
        Hides Roblox features section when Python files are part of the selection,
        even if Lua/Luau files are also present (mixed projects).
        Also disables Roblox feature checkboxes when hidden.

        Handles edge cases:
        - Mixed projects (Python + Lua/Luau): Hide Roblox features
        - Empty file selection: Default to showing Roblox features (assume Lua)
        - Rapid file selection changes: Handled by Qt's signal system
        """
        # Determine if Roblox features should be shown
        # Hide when Python files are present (treat mixed as Python-only for feature visibility)
        # Show only when no Python files and we have Lua/Luau files (or empty/default)
        should_show_roblox = not getattr(self, '_python_present', False) and self._current_language in ("Lua", "Luau", "Mixed")

        # Show/hide all Roblox section widgets
        for widget in self._roblox_section_widgets:
            widget.setVisible(should_show_roblox)

        # Disable Roblox feature checkboxes when hidden
        for feature in ROBLOX_FEATURES:
            if feature in self._feature_checkboxes:
                checkbox = self._feature_checkboxes[feature]
                checkbox.setEnabled(should_show_roblox)
                if not should_show_roblox:
                    # Uncheck Roblox features when hidden
                    checkbox.setChecked(False)
                    self._features[feature] = False

        if should_show_roblox:
            logger.debug("Roblox features visible (no Python files in selection)")
        else:
            logger.debug("Roblox features hidden (Python files present in selection)")

    def _setup_ui(self) -> None:
        """Set up the widget UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(12)

        # Title label
        title_label = QLabel("Security Configuration")
        title_label.setStyleSheet(get_widget_style("title_label"))
        title_label.setProperty("data-element-id", "security-config-title")
        main_layout.addWidget(title_label)

        # Preset buttons section
        self._setup_preset_buttons(main_layout)

        # Advanced options toggle
        self._setup_advanced_toggle(main_layout)

        # Features panel (collapsible)
        self._setup_features_panel(main_layout)

    def _setup_preset_buttons(self, parent_layout: QVBoxLayout) -> None:
        """Create and configure preset selection buttons."""
        presets_container = QWidget()
        presets_layout = QHBoxLayout(presets_container)
        presets_layout.setContentsMargins(0, 0, 0, 0)
        presets_layout.setSpacing(8)

        for preset_name in ["Light", "Medium", "Heavy", "Maximum"]:
            btn = QPushButton(preset_name)
            btn.setProperty("data-element-id", f"preset-{preset_name.lower()}-button")
            btn.setToolTip(PRESET_TOOLTIPS.get(preset_name, ""))
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setStyleSheet(get_widget_style("preset_button"))
            btn.clicked.connect(lambda checked, p=preset_name: self._on_preset_clicked(p))
            self._preset_buttons[preset_name] = btn
            presets_layout.addWidget(btn)

        parent_layout.addWidget(presets_container)

    def _setup_advanced_toggle(self, parent_layout: QVBoxLayout) -> None:
        """Create the advanced options toggle button."""
        self._advanced_toggle_btn = QPushButton("▼ Advanced Options")
        self._advanced_toggle_btn.setProperty("data-element-id", "advanced-options-toggle")
        self._advanced_toggle_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._advanced_toggle_btn.setStyleSheet(get_widget_style("advanced_toggle"))
        self._advanced_toggle_btn.clicked.connect(self._on_advanced_toggle_clicked)
        parent_layout.addWidget(self._advanced_toggle_btn)

    def _setup_features_panel(self, parent_layout: QVBoxLayout) -> None:
        """Create the collapsible features panel with checkboxes."""
        self._features_panel = QFrame()
        self._features_panel.setProperty("data-element-id", "features-panel")
        self._features_panel.setStyleSheet(get_widget_style("features_panel"))
        self._features_panel.setVisible(False)

        panel_layout = QVBoxLayout(self._features_panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(6)

        # Core features section
        self._add_feature_section(panel_layout, "Core Features", CORE_FEATURES)

        # Advanced features section
        self._add_feature_section(panel_layout, "Advanced Features", ADVANCED_FEATURES)

        # Roblox features section
        self._add_feature_section(panel_layout, "Roblox-Specific Features", ROBLOX_FEATURES)

        parent_layout.addWidget(self._features_panel)

    def _add_feature_section(
        self, parent_layout: QVBoxLayout, section_title: str, features: list[str]
    ) -> None:
        """Add a section of feature checkboxes to the panel.

        For Roblox-Specific Features section, adds "Lua Only" badges next to
        each feature checkbox to indicate these features only apply to Lua/Luau files.
        Also tracks all Roblox section widgets for language-based visibility toggling.
        """
        is_roblox_section = section_title == "Roblox-Specific Features"

        section_label = QLabel(section_title)
        section_label.setStyleSheet(get_widget_style("section_label"))
        parent_layout.addWidget(section_label)

        # Track Roblox section label for visibility toggling
        if is_roblox_section:
            self._roblox_section_widgets.append(section_label)

        for feature in features:
            # Create horizontal layout for checkbox and optional badge
            feature_layout = QHBoxLayout()
            feature_layout.setSpacing(8)
            feature_layout.setContentsMargins(0, 0, 0, 0)

            checkbox = QCheckBox(feature)
            slug = feature.lower().replace(" ", "-")
            checkbox.setProperty("data-element-id", f"feature-checkbox-{slug}")
            checkbox.setToolTip(FEATURE_TOOLTIPS.get(feature, ""))
            checkbox.setStyleSheet(get_widget_style("checkbox"))
            checkbox.stateChanged.connect(
                lambda state, f=feature: self._on_feature_toggled(f, state == 2)
            )
            self._feature_checkboxes[feature] = checkbox
            feature_layout.addWidget(checkbox)

            # Add "Lua Only" badge for Roblox features
            if is_roblox_section:
                badge = QLabel("Lua Only")
                badge.setStyleSheet(
                    f"font-size: 10px; padding: 2px 6px; border-radius: 3px; "
                    f"background-color: {COLORS.get('accent', '#2196F3')}; "
                    f"color: white;"
                )
                badge.setToolTip("This feature only applies to Lua and Luau files")
                feature_layout.addWidget(badge)
                feature_layout.addStretch()

                # Create container widget to track for visibility
                feature_container = QWidget()
                feature_container.setLayout(feature_layout)
                self._roblox_section_widgets.append(feature_container)
                parent_layout.addWidget(feature_container)
            else:
                feature_layout.addStretch()
                feature_container = QWidget()
                feature_container.setLayout(feature_layout)
                parent_layout.addWidget(feature_container)

    def _on_preset_clicked(self, preset_name: str) -> None:
        """Handle preset button click."""
        self._current_preset = preset_name
        enabled_features = PRESET_CONFIGS.get(preset_name, [])

        # Filter out Roblox features if Python files are present
        # This applies to both Python-only and mixed (Python+Lua/Luau) projects
        if getattr(self, '_python_present', False) or self._current_language == "Python":
            enabled_features = [f for f in enabled_features if f not in ROBLOX_FEATURES]
            logger.debug(f"Filtered out Roblox features (Python present) in {preset_name} preset")

        # Update internal features state
        for feature in self._features:
            self._features[feature] = feature in enabled_features

        self._update_checkboxes()
        self._update_preset_buttons()
        self._emit_config_changed()

        logger.debug(f"Preset selected: {preset_name}")

    def _on_feature_toggled(self, feature_name: str, checked: bool) -> None:
        """Handle individual feature checkbox toggle."""
        self._features[feature_name] = checked

        # Check if current configuration matches any preset
        self._current_preset = self._find_matching_preset()

        self._update_preset_buttons()
        self._emit_config_changed()

        logger.debug(f"Feature toggled: {feature_name} = {checked}")

    def _find_matching_preset(self) -> str | None:
        """Find if current feature configuration matches any preset."""
        current_enabled = {f for f, enabled in self._features.items() if enabled}

        for preset_name, preset_features in PRESET_CONFIGS.items():
            if current_enabled == set(preset_features):
                return preset_name

        return None

    def _on_advanced_toggle_clicked(self) -> None:
        """Handle advanced options toggle button click."""
        self._advanced_expanded = not self._advanced_expanded
        self._features_panel.setVisible(self._advanced_expanded)

        if self._advanced_expanded:
            self._advanced_toggle_btn.setText("▲ Advanced Options")
        else:
            self._advanced_toggle_btn.setText("▼ Advanced Options")

        logger.debug(f"Advanced options expanded: {self._advanced_expanded}")

    def _update_checkboxes(self) -> None:
        """Sync checkbox states with internal features dictionary."""
        for feature, checkbox in self._feature_checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(self._features.get(feature, False))
            checkbox.blockSignals(False)

    def _update_preset_buttons(self) -> None:
        """Update visual state of preset buttons."""
        for preset_name, btn in self._preset_buttons.items():
            if preset_name == self._current_preset:
                btn.setStyleSheet(get_widget_style("preset_button_active"))
            else:
                btn.setStyleSheet(get_widget_style("preset_button"))

    def _emit_config_changed(self) -> None:
        """Emit the config_changed signal with current configuration."""
        config = self.get_config()
        self.config_changed.emit(config)

    def get_config(self) -> dict:
        """Get the current configuration."""
        return {
            "preset": self._current_preset,
            "features": dict(self._features),
        }

    def set_config(self, preset: str = None, features: dict = None) -> None:
        """Set configuration programmatically.

        When loading a profile with Roblox features enabled but Python files are
        present (including mixed projects), a warning is logged and Roblox features
        are auto-disabled.
        """
        if preset is not None and preset in PRESET_CONFIGS:
            self._on_preset_clicked(preset)
        elif features is not None:
            # Reset all features to False before applying incoming features
            # This ensures missing features default to unchecked
            for feature in self._features:
                self._features[feature] = False

            # Check for language mismatch with Roblox features
            # Auto-disable if Python files are present (Python-only or mixed projects)
            roblox_enabled = any(features.get(f, False) for f in ROBLOX_FEATURES)
            python_present = getattr(self, '_python_present', False) or self._current_language == "Python"
            if roblox_enabled and python_present:
                logger.warning(
                    "Roblox features enabled in profile but Python files are present. "
                    "Roblox features will be auto-disabled."
                )
                # Auto-disable Roblox features when Python is present
                for roblox_feature in ROBLOX_FEATURES:
                    if roblox_feature in features:
                        features[roblox_feature] = False

            self._features.update(features)
            self._current_preset = self._find_matching_preset()
            self._update_checkboxes()
            self._update_preset_buttons()
            self._emit_config_changed()

    def reset(self) -> None:
        """Reset to default configuration (Light preset)."""
        self._on_preset_clicked("Light")
