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
    QRadioButton,
    QButtonGroup,
    QSpinBox,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QCursor

from typing import TYPE_CHECKING

from obfuscator.utils.logger import get_logger
from obfuscator.gui.styles.stylesheet import get_widget_style, COLORS
from obfuscator.core.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENABLE_MULTIPROCESSING,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MEMORY_THRESHOLD_PERCENT,
    DEFAULT_MULTIPROCESSING_THRESHOLD,
)

if TYPE_CHECKING:
    from obfuscator.gui.widgets.file_selection_widget import FileSelectionWidget

logger = get_logger("obfuscator.gui.widgets.security_config_widget")

_UNSET = object()

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
        
        # Runtime mode tracking
        self._runtime_mode: str = "hybrid"
        self._runtime_button_group: QButtonGroup = None
        self._hybrid_radio: QRadioButton = None
        self._embedded_radio: QRadioButton = None
        self._enable_multiprocessing: bool = DEFAULT_ENABLE_MULTIPROCESSING
        self._max_workers: int | None = DEFAULT_MAX_WORKERS
        self._batch_size: int = DEFAULT_BATCH_SIZE
        self._multiprocessing_threshold: int = DEFAULT_MULTIPROCESSING_THRESHOLD
        self._memory_threshold_percent: int = DEFAULT_MEMORY_THRESHOLD_PERCENT
        self._multiprocessing_checkbox: QCheckBox = None
        self._max_workers_spinbox: QSpinBox = None
        self._batch_size_spinbox: QSpinBox = None
        self._multiprocessing_threshold_spinbox: QSpinBox = None
        self._memory_threshold_spinbox: QSpinBox = None

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

        # Runtime mode section
        self._setup_runtime_mode_section(main_layout)

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

    def _setup_runtime_mode_section(self, parent_layout: QVBoxLayout) -> None:
        """Create the runtime mode selection section with radio buttons."""
        # Section label
        section_label = QLabel("Runtime Mode")
        section_label.setStyleSheet(get_widget_style("section_label"))
        parent_layout.addWidget(section_label)

        # Create button group for mutual exclusivity
        self._runtime_button_group = QButtonGroup(self)

        # Create horizontal layout for radio buttons
        radio_layout = QHBoxLayout()
        radio_layout.setSpacing(16)
        radio_layout.setContentsMargins(0, 0, 0, 0)

        # Hybrid radio button
        self._hybrid_radio = QRadioButton("Hybrid")
        self._hybrid_radio.setProperty("data-element-id", "runtime-mode-hybrid-radio")
        self._hybrid_radio.setToolTip(
            "Shared runtime files across project. Smaller output size, requires runtime files to be distributed with obfuscated code."
        )
        self._hybrid_radio.setStyleSheet(get_widget_style("checkbox"))
        self._hybrid_radio.setChecked(True)
        self._runtime_button_group.addButton(self._hybrid_radio)
        radio_layout.addWidget(self._hybrid_radio)

        # Embedded radio button
        self._embedded_radio = QRadioButton("Embedded")
        self._embedded_radio.setProperty("data-element-id", "runtime-mode-embedded-radio")
        self._embedded_radio.setToolTip(
            "Runtime code embedded in each file. Larger output size, fully self-contained files."
        )
        self._embedded_radio.setStyleSheet(get_widget_style("checkbox"))
        self._runtime_button_group.addButton(self._embedded_radio)
        radio_layout.addWidget(self._embedded_radio)

        radio_layout.addStretch()

        # Create container widget
        radio_container = QWidget()
        radio_container.setLayout(radio_layout)
        parent_layout.addWidget(radio_container)

        # Connect signals
        self._hybrid_radio.toggled.connect(self._on_runtime_mode_changed)
        self._embedded_radio.toggled.connect(self._on_runtime_mode_changed)

    def _on_runtime_mode_changed(self) -> None:
        """Handle runtime mode radio button toggle."""
        if self._hybrid_radio.isChecked():
            self._runtime_mode = "hybrid"
        else:
            self._runtime_mode = "embedded"
        
        self._emit_config_changed()
        logger.debug(f"Runtime mode changed to: {self._runtime_mode}")

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

        # Performance settings section
        performance_label = QLabel("Performance Settings")
        performance_label.setStyleSheet(get_widget_style("section_label"))
        performance_label.setToolTip(
            "Advanced multiprocessing controls. Worker count and batch size are auto-tuned by default."
        )
        panel_layout.addWidget(performance_label)

        self._multiprocessing_checkbox = QCheckBox("Enable Multiprocessing")
        self._multiprocessing_checkbox.setProperty(
            "data-element-id",
            "enable-multiprocessing-checkbox",
        )
        self._multiprocessing_checkbox.setStyleSheet(get_widget_style("checkbox"))
        self._multiprocessing_checkbox.setChecked(self._enable_multiprocessing)
        self._multiprocessing_checkbox.setToolTip(
            "Use multiple CPU cores for faster processing of large projects (100+ files). "
            "Disable for debugging or memory-constrained systems. Batch size and worker count are "
            "auto-tuned by default. Multiprocessing is skipped for small projects and memory pressure "
            "dynamically reduces batch sizes."
        )
        self._multiprocessing_checkbox.stateChanged.connect(
            lambda state: self._on_multiprocessing_toggled(state == 2)
        )
        panel_layout.addWidget(self._multiprocessing_checkbox)

        max_workers_layout = QHBoxLayout()
        max_workers_layout.setContentsMargins(0, 0, 0, 0)
        max_workers_layout.setSpacing(8)
        max_workers_label = QLabel("Max Workers")
        max_workers_label.setToolTip(
            "Maximum worker processes for parallel execution. Set to Auto to use CPU count - 1 (up to 8)."
        )
        max_workers_layout.addWidget(max_workers_label)
        max_workers_layout.addStretch()

        self._max_workers_spinbox = QSpinBox()
        self._max_workers_spinbox.setProperty(
            "data-element-id",
            "max-workers-spinbox",
        )
        self._max_workers_spinbox.setRange(0, 16)
        self._max_workers_spinbox.setSpecialValueText("Auto")
        self._max_workers_spinbox.setValue(
            0 if self._max_workers is None else self._max_workers
        )
        self._max_workers_spinbox.setToolTip(
            "Worker process limit. Auto chooses CPU count - 1 (capped at 8). "
            "Manual values are validated in the range 1-16."
        )
        self._max_workers_spinbox.valueChanged.connect(self._on_max_workers_changed)
        max_workers_layout.addWidget(self._max_workers_spinbox)
        panel_layout.addLayout(max_workers_layout)

        batch_size_layout = QHBoxLayout()
        batch_size_layout.setContentsMargins(0, 0, 0, 0)
        batch_size_layout.setSpacing(8)
        batch_size_label = QLabel("Batch Size")
        batch_size_label.setToolTip(
            "Files assigned to each worker batch. Larger values improve throughput but increase memory usage."
        )
        batch_size_layout.addWidget(batch_size_label)
        batch_size_layout.addStretch()

        self._batch_size_spinbox = QSpinBox()
        self._batch_size_spinbox.setProperty(
            "data-element-id",
            "batch-size-spinbox",
        )
        self._batch_size_spinbox.setRange(10, 200)
        self._batch_size_spinbox.setValue(self._batch_size)
        self._batch_size_spinbox.setToolTip(
            "Initial files per multiprocessing batch. Valid range: 10-200. "
            "May be reduced at runtime when memory pressure is detected."
        )
        self._batch_size_spinbox.valueChanged.connect(self._on_batch_size_changed)
        batch_size_layout.addWidget(self._batch_size_spinbox)
        panel_layout.addLayout(batch_size_layout)

        multiprocessing_threshold_layout = QHBoxLayout()
        multiprocessing_threshold_layout.setContentsMargins(0, 0, 0, 0)
        multiprocessing_threshold_layout.setSpacing(8)
        multiprocessing_threshold_label = QLabel("Parallel Threshold")
        multiprocessing_threshold_label.setToolTip(
            "Minimum file count required before multiprocessing is used. "
            "Smaller projects run sequentially below this threshold."
        )
        multiprocessing_threshold_layout.addWidget(multiprocessing_threshold_label)
        multiprocessing_threshold_layout.addStretch()

        self._multiprocessing_threshold_spinbox = QSpinBox()
        self._multiprocessing_threshold_spinbox.setProperty(
            "data-element-id",
            "multiprocessing-threshold-spinbox",
        )
        self._multiprocessing_threshold_spinbox.setRange(10, 1000)
        self._multiprocessing_threshold_spinbox.setValue(self._multiprocessing_threshold)
        self._multiprocessing_threshold_spinbox.setToolTip(
            "Minimum number of files required to trigger multiprocessing. "
            "Valid range: 10-1000."
        )
        self._multiprocessing_threshold_spinbox.valueChanged.connect(
            self._on_multiprocessing_threshold_changed
        )
        multiprocessing_threshold_layout.addWidget(self._multiprocessing_threshold_spinbox)
        panel_layout.addLayout(multiprocessing_threshold_layout)

        memory_threshold_layout = QHBoxLayout()
        memory_threshold_layout.setContentsMargins(0, 0, 0, 0)
        memory_threshold_layout.setSpacing(8)
        memory_threshold_label = QLabel("Memory Threshold (%)")
        memory_threshold_label.setToolTip(
            "Memory pressure threshold for adaptive batch downsizing. "
            "When usage exceeds this percentage, batches are reduced."
        )
        memory_threshold_layout.addWidget(memory_threshold_label)
        memory_threshold_layout.addStretch()

        self._memory_threshold_spinbox = QSpinBox()
        self._memory_threshold_spinbox.setProperty(
            "data-element-id",
            "memory-threshold-spinbox",
        )
        self._memory_threshold_spinbox.setRange(50, 95)
        self._memory_threshold_spinbox.setValue(self._memory_threshold_percent)
        self._memory_threshold_spinbox.setSuffix("%")
        self._memory_threshold_spinbox.setToolTip(
            "Maximum memory usage percentage before worker batch sizes are reduced. "
            "Valid range: 50-95%."
        )
        self._memory_threshold_spinbox.valueChanged.connect(
            self._on_memory_threshold_percent_changed
        )
        memory_threshold_layout.addWidget(self._memory_threshold_spinbox)
        panel_layout.addLayout(memory_threshold_layout)

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

    def _on_multiprocessing_toggled(self, enabled: bool) -> None:
        """Handle multiprocessing checkbox toggle."""
        self._enable_multiprocessing = enabled
        self._emit_config_changed()
        logger.debug(f"Multiprocessing enabled: {enabled}")

    def _on_max_workers_changed(self, value: int) -> None:
        """Handle max workers spinbox changes."""
        self._max_workers = None if value == 0 else int(value)
        self._emit_config_changed()
        logger.debug(
            "Max workers changed to: %s",
            "auto" if self._max_workers is None else self._max_workers,
        )

    def _on_batch_size_changed(self, value: int) -> None:
        """Handle batch size spinbox changes."""
        self._batch_size = int(value)
        self._emit_config_changed()
        logger.debug("Batch size changed to: %d", self._batch_size)

    def _on_multiprocessing_threshold_changed(self, value: int) -> None:
        """Handle multiprocessing threshold spinbox changes."""
        self._multiprocessing_threshold = int(value)
        self._emit_config_changed()
        logger.debug(
            "Multiprocessing threshold changed to: %d",
            self._multiprocessing_threshold,
        )

    def _on_memory_threshold_percent_changed(self, value: int) -> None:
        """Handle memory threshold percent spinbox changes."""
        self._memory_threshold_percent = int(value)
        self._emit_config_changed()
        logger.debug(
            "Memory threshold percent changed to: %d",
            self._memory_threshold_percent,
        )

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
            "runtime_mode": self._runtime_mode,
            "enable_multiprocessing": self._enable_multiprocessing,
            "max_workers": self._max_workers,
            "batch_size": self._batch_size,
            "multiprocessing_threshold": self._multiprocessing_threshold,
            "memory_threshold_percent": self._memory_threshold_percent,
        }

    def set_config(
        self,
        preset: str = None,
        features: dict = None,
        runtime_mode: str = None,
        enable_multiprocessing: bool = None,
        max_workers: int | None | object = _UNSET,
        batch_size: int | object = _UNSET,
        multiprocessing_threshold: int | object = _UNSET,
        memory_threshold_percent: int | object = _UNSET,
    ) -> None:
        """Set configuration programmatically.

        When loading a profile with Roblox features enabled but Python files are
        present (including mixed projects), a warning is logged and Roblox features
        are auto-disabled.
        """
        # Handle runtime_mode first to avoid triggering change events during preset application
        if runtime_mode is not None and runtime_mode in ("hybrid", "embedded"):
            self._runtime_mode = runtime_mode
            # Update radio buttons without triggering signals
            self._hybrid_radio.blockSignals(True)
            self._embedded_radio.blockSignals(True)
            self._hybrid_radio.setChecked(runtime_mode == "hybrid")
            self._embedded_radio.setChecked(runtime_mode == "embedded")
            self._hybrid_radio.blockSignals(False)
            self._embedded_radio.blockSignals(False)
            logger.debug(f"Runtime mode set to: {runtime_mode}")
        elif runtime_mode is not None:
            # Invalid runtime_mode value
            logger.warning(f"Invalid runtime_mode '{runtime_mode}', defaulting to 'hybrid'")
            self._runtime_mode = "hybrid"
            self._hybrid_radio.blockSignals(True)
            self._embedded_radio.blockSignals(True)
            self._hybrid_radio.setChecked(True)
            self._embedded_radio.setChecked(False)
            self._hybrid_radio.blockSignals(False)
            self._embedded_radio.blockSignals(False)

        if enable_multiprocessing is not None:
            self._enable_multiprocessing = bool(enable_multiprocessing)
            if self._multiprocessing_checkbox is not None:
                self._multiprocessing_checkbox.blockSignals(True)
                self._multiprocessing_checkbox.setChecked(self._enable_multiprocessing)
                self._multiprocessing_checkbox.blockSignals(False)
            logger.debug(
                "Multiprocessing setting set to: %s",
                self._enable_multiprocessing,
            )

        if max_workers is not _UNSET:
            if max_workers is None:
                self._max_workers = None
            else:
                try:
                    parsed_max_workers = int(max_workers)
                except (TypeError, ValueError):
                    logger.warning("Invalid max_workers '%s', defaulting to auto", max_workers)
                    parsed_max_workers = 0

                self._max_workers = None if parsed_max_workers <= 0 else max(1, min(16, parsed_max_workers))

            if self._max_workers_spinbox is not None:
                self._max_workers_spinbox.blockSignals(True)
                self._max_workers_spinbox.setValue(
                    0 if self._max_workers is None else self._max_workers
                )
                self._max_workers_spinbox.blockSignals(False)

            logger.debug(
                "Max workers setting set to: %s",
                "auto" if self._max_workers is None else self._max_workers,
            )

        if batch_size is not _UNSET:
            try:
                parsed_batch_size = int(batch_size)
            except (TypeError, ValueError):
                logger.warning("Invalid batch_size '%s', using default %d", batch_size, DEFAULT_BATCH_SIZE)
                parsed_batch_size = DEFAULT_BATCH_SIZE

            self._batch_size = max(10, min(200, parsed_batch_size))
            if self._batch_size_spinbox is not None:
                self._batch_size_spinbox.blockSignals(True)
                self._batch_size_spinbox.setValue(self._batch_size)
                self._batch_size_spinbox.blockSignals(False)

            logger.debug("Batch size setting set to: %d", self._batch_size)

        if multiprocessing_threshold is not _UNSET:
            try:
                parsed_threshold = int(multiprocessing_threshold)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid multiprocessing_threshold '%s', using default %d",
                    multiprocessing_threshold,
                    DEFAULT_MULTIPROCESSING_THRESHOLD,
                )
                parsed_threshold = DEFAULT_MULTIPROCESSING_THRESHOLD

            self._multiprocessing_threshold = max(10, min(1000, parsed_threshold))
            if self._multiprocessing_threshold_spinbox is not None:
                self._multiprocessing_threshold_spinbox.blockSignals(True)
                self._multiprocessing_threshold_spinbox.setValue(
                    self._multiprocessing_threshold
                )
                self._multiprocessing_threshold_spinbox.blockSignals(False)

            logger.debug(
                "Multiprocessing threshold setting set to: %d",
                self._multiprocessing_threshold,
            )

        if memory_threshold_percent is not _UNSET:
            try:
                parsed_memory_threshold = int(memory_threshold_percent)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid memory_threshold_percent '%s', using default %d",
                    memory_threshold_percent,
                    DEFAULT_MEMORY_THRESHOLD_PERCENT,
                )
                parsed_memory_threshold = DEFAULT_MEMORY_THRESHOLD_PERCENT

            self._memory_threshold_percent = max(50, min(95, parsed_memory_threshold))
            if self._memory_threshold_spinbox is not None:
                self._memory_threshold_spinbox.blockSignals(True)
                self._memory_threshold_spinbox.setValue(self._memory_threshold_percent)
                self._memory_threshold_spinbox.blockSignals(False)

            logger.debug(
                "Memory threshold percent setting set to: %d",
                self._memory_threshold_percent,
            )

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
        # Reset runtime mode to hybrid
        self._runtime_mode = "hybrid"
        self._hybrid_radio.blockSignals(True)
        self._embedded_radio.blockSignals(True)
        self._hybrid_radio.setChecked(True)
        self._embedded_radio.setChecked(False)
        self._hybrid_radio.blockSignals(False)
        self._embedded_radio.blockSignals(False)

        self._enable_multiprocessing = DEFAULT_ENABLE_MULTIPROCESSING
        if self._multiprocessing_checkbox is not None:
            self._multiprocessing_checkbox.blockSignals(True)
            self._multiprocessing_checkbox.setChecked(DEFAULT_ENABLE_MULTIPROCESSING)
            self._multiprocessing_checkbox.blockSignals(False)

        self._max_workers = DEFAULT_MAX_WORKERS
        if self._max_workers_spinbox is not None:
            self._max_workers_spinbox.blockSignals(True)
            self._max_workers_spinbox.setValue(
                0 if self._max_workers is None else self._max_workers
            )
            self._max_workers_spinbox.blockSignals(False)

        self._batch_size = DEFAULT_BATCH_SIZE
        if self._batch_size_spinbox is not None:
            self._batch_size_spinbox.blockSignals(True)
            self._batch_size_spinbox.setValue(DEFAULT_BATCH_SIZE)
            self._batch_size_spinbox.blockSignals(False)

        self._multiprocessing_threshold = DEFAULT_MULTIPROCESSING_THRESHOLD
        if self._multiprocessing_threshold_spinbox is not None:
            self._multiprocessing_threshold_spinbox.blockSignals(True)
            self._multiprocessing_threshold_spinbox.setValue(
                DEFAULT_MULTIPROCESSING_THRESHOLD
            )
            self._multiprocessing_threshold_spinbox.blockSignals(False)

        self._memory_threshold_percent = DEFAULT_MEMORY_THRESHOLD_PERCENT
        if self._memory_threshold_spinbox is not None:
            self._memory_threshold_spinbox.blockSignals(True)
            self._memory_threshold_spinbox.setValue(DEFAULT_MEMORY_THRESHOLD_PERCENT)
            self._memory_threshold_spinbox.blockSignals(False)

        logger.debug("Runtime mode reset to: hybrid")
