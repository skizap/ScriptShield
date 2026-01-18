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

from obfuscator.utils.logger import get_logger
from obfuscator.gui.styles.stylesheet import get_widget_style, COLORS

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
    "Roblox API Preservation": "Preserves Roblox-specific APIs",
    "Luau Type Stripping": "Removes Luau type annotations",
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
    "Roblox API Preservation",
    "Luau Type Stripping",
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
    """Widget for configuring security/obfuscation settings."""

    config_changed = pyqtSignal(dict)

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._current_preset: str | None = "Light"
        self._features: dict[str, bool] = {}
        self._advanced_expanded: bool = False
        self._preset_buttons: dict[str, QPushButton] = {}
        self._feature_checkboxes: dict[str, QCheckBox] = {}

        self._init_features()
        self._setup_ui()
        self._on_preset_clicked("Light")

        logger.debug("SecurityConfigWidget initialized")

    def _init_features(self) -> None:
        """Initialize all features to False."""
        all_features = CORE_FEATURES + ADVANCED_FEATURES + ROBLOX_FEATURES
        for feature in all_features:
            self._features[feature] = False

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
        """Add a section of feature checkboxes to the panel."""
        section_label = QLabel(section_title)
        section_label.setStyleSheet(get_widget_style("section_label"))
        parent_layout.addWidget(section_label)

        for feature in features:
            checkbox = QCheckBox(feature)
            slug = feature.lower().replace(" ", "-")
            checkbox.setProperty("data-element-id", f"feature-checkbox-{slug}")
            checkbox.setToolTip(FEATURE_TOOLTIPS.get(feature, ""))
            checkbox.setStyleSheet(get_widget_style("checkbox"))
            checkbox.stateChanged.connect(
                lambda state, f=feature: self._on_feature_toggled(f, state == 2)
            )
            self._feature_checkboxes[feature] = checkbox
            parent_layout.addWidget(checkbox)

    def _on_preset_clicked(self, preset_name: str) -> None:
        """Handle preset button click."""
        self._current_preset = preset_name
        enabled_features = PRESET_CONFIGS.get(preset_name, [])

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
        """Set configuration programmatically."""
        if preset is not None and preset in PRESET_CONFIGS:
            self._on_preset_clicked(preset)
        elif features is not None:
            # Reset all features to False before applying incoming features
            # This ensures missing features default to unchecked
            for feature in self._features:
                self._features[feature] = False

            self._features.update(features)
            self._current_preset = self._find_matching_preset()
            self._update_checkboxes()
            self._update_preset_buttons()
            self._emit_config_changed()

    def reset(self) -> None:
        """Reset to default configuration (Light preset)."""
        self._on_preset_clicked("Light")
