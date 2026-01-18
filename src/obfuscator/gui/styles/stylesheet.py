"""
Centralized stylesheet system for the Python Obfuscator GUI.
Defines all colors, fonts, and QSS styles as reusable constants.
"""

from typing import Optional

# Color Palette
COLORS = {
    "bg_dark": "#1e1e1e",      # Input fields, log container
    "bg_medium": "#2d2d2d",    # Widget containers, frames
    "bg_light": "#3d3d3d",     # Hover states, disabled elements
    "bg_lighter": "#444444",   # Secondary buttons
    "text_primary": "#ffffff", # Main text content
    "text_secondary": "#aaaaaa", # Tips, secondary info
    "text_tertiary": "#cccccc",  # Placeholder text
    "text_disabled": "#888888",  # Disabled button text
    "border_default": "#555555", # Default borders
    "border_light": "#3d3d3d",   # Subtle borders
    "primary": "#4a9eff",        # Primary buttons, links
    "primary_hover": "#3a8eef",  # Primary button hover
    "primary_pressed": "#2a7edf",# Primary button pressed
    "success": "#4caf50",        # Success messages, checkmarks
    "warning": "#ff9800",        # Warning messages
    "error": "#f44336",          # Error messages
    "danger": "#cc4444",         # Cancel/delete buttons
    "lua_badge": "#1e5a9e",      # Lua language indicator
    "luau_badge": "#2e7d32",     # Luau language indicator
    "python_badge": "#1976d2",   # Python language indicator
    "mixed_badge": "#757575",    # Mixed language indicator
}

# Typography
FONTS = {
    "family": '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif',
    "size_title": "16px",
    "size_body": "13px",
    "size_small": "11px",
    "weight_normal": "normal",
    "weight_bold": "bold",
}

# Spacing and Layout
SPACING = {
    "padding_xs": "4px",
    "padding_sm": "8px",
    "padding_md": "12px",
    "padding_lg": "16px",
    "radius_sm": "4px",
    "radius_md": "6px",
    "radius_lg": "8px",
    "min_height_button": "32px",
    "min_height_input": "32px",
}

# Component Styles
STYLES = {
    "primary_button": f"""
        QPushButton {{
            background-color: {COLORS['primary']};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: {SPACING['radius_sm']};
            font-weight: {FONTS['weight_bold']};
            font-size: {FONTS['size_body']};
            min-height: {SPACING['min_height_button']};
        }}
        QPushButton:hover {{
            background-color: {COLORS['primary_hover']};
        }}
        QPushButton:pressed {{
            background-color: {COLORS['primary_pressed']};
        }}
    """,
    "primary_button_disabled": f"""
        QPushButton {{
            background-color: {COLORS['bg_light']};
            color: {COLORS['text_disabled']};
            border: none;
            padding: 8px 16px;
            border-radius: {SPACING['radius_sm']};
            font-weight: {FONTS['weight_bold']};
            font-size: {FONTS['size_body']};
            min-height: {SPACING['min_height_button']};
        }}
    """,
    "secondary_button": f"""
        QPushButton {{
            background-color: {COLORS['bg_lighter']};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: {SPACING['radius_sm']};
            font-size: {FONTS['size_body']};
            min-height: {SPACING['min_height_button']};
        }}
        QPushButton:hover {{
            background-color: {COLORS['border_default']};
        }}
    """,
    "danger_button": f"""
        QPushButton {{
            background-color: {COLORS['bg_lighter']};
            color: white;
            border: none;
            padding: 8px;
            border-radius: {SPACING['radius_sm']};
            font-size: 14px;
            min-height: {SPACING['min_height_button']};
        }}
        QPushButton:hover {{
            background-color: {COLORS['danger']};
        }}
    """,
    "icon_button": f"""
        QPushButton {{
            background-color: {COLORS['bg_lighter']};
            color: white;
            border: none;
            padding: 8px;
            border-radius: {SPACING['radius_sm']};
            font-size: 14px;
            min-height: {SPACING['min_height_button']};
        }}
        QPushButton:hover {{
            background-color: {COLORS['border_default']};
        }}
    """,
    "input_field": f"""
        QLineEdit {{
            background-color: {COLORS['bg_dark']};
            color: white;
            border: 1px solid {COLORS['border_default']};
            border-radius: {SPACING['radius_sm']};
            padding: 4px 8px;
            font-size: {FONTS['size_body']};
            min-height: {SPACING['min_height_input']};
        }}
        QLineEdit:focus {{
            border: 1px solid {COLORS['primary']};
        }}
    """,
    "dropdown": f"""
        QComboBox {{
            background-color: {COLORS['bg_medium']};
            color: white;
            border: 1px solid {COLORS['border_default']};
            border-radius: {SPACING['radius_sm']};
            padding: 8px 12px;
            font-size: {FONTS['size_body']};
            min-height: 20px;
        }}
        QComboBox:hover {{
            border-color: {COLORS['primary']};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox::down-arrow {{
            width: 10px;
            height: 10px;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid {COLORS['text_secondary']};
            margin-right: 8px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {COLORS['bg_medium']};
            color: white;
            selection-background-color: {COLORS['primary']};
            border: 1px solid {COLORS['border_default']};
        }}
    """,
    "checkbox": f"""
        QCheckBox {{
            color: white;
            spacing: 8px;
            font-size: {FONTS['size_body']};
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
        }}
    """,
    "advanced_toggle": f"""
        QPushButton {{
            background-color: {COLORS['bg_light']};
            color: {COLORS['text_tertiary']};
            padding: 10px;
            border: none;
            border-radius: {SPACING['radius_sm']};
            text-align: left;
            font-size: {FONTS['size_body']};
        }}
        QPushButton:hover {{
            background-color: {COLORS['bg_lighter']};
        }}
    """,
    "features_panel": f"""
        QFrame {{
            background-color: {COLORS['bg_medium']};
            border: 1px solid {COLORS['border_default']};
            border-radius: {SPACING['radius_lg']};
            padding: 12px;
        }}
    """,
    "section_label": f"""
        QLabel {{
            color: {COLORS['text_secondary']};
            font-weight: {FONTS['weight_bold']};
            font-size: 12px;
            padding-top: 8px;
            padding-bottom: 4px;
        }}
    """,
    "title_label": f"""
        QLabel {{
            font-size: {FONTS['size_title']};
            font-weight: {FONTS['weight_bold']};
            color: white;
        }}
    """,
    "drop_zone": f"""
        QFrame#dropZone {{
            border: 2px dashed {COLORS['border_default']};
            border-radius: {SPACING['radius_lg']};
            background-color: {COLORS['bg_medium']};
            min-height: 120px;
        }}
    """,
    "drop_zone_hover": f"""
        QFrame#dropZone {{
            border: 2px dashed {COLORS['primary']};
            border-radius: {SPACING['radius_lg']};
            background-color: #3d3d4d;
            min-height: 120px;
        }}
    """,
    "language_badge_lua": f"""
        QLabel {{
            background-color: {COLORS['lua_badge']};
            color: white;
            padding: 2px 8px;
            border-radius: {SPACING['radius_sm']};
            font-size: {FONTS['size_small']};
            font-weight: {FONTS['weight_bold']};
        }}
    """,
    "language_badge_luau": f"""
        QLabel {{
            background-color: {COLORS['luau_badge']};
            color: white;
            padding: 2px 8px;
            border-radius: {SPACING['radius_sm']};
            font-size: {FONTS['size_small']};
            font-weight: {FONTS['weight_bold']};
        }}
    """,
    "language_badge_python": f"""
        QLabel {{
            background-color: {COLORS['python_badge']};
            color: white;
            padding: 2px 8px;
            border-radius: {SPACING['radius_sm']};
            font-size: {FONTS['size_small']};
            font-weight: {FONTS['weight_bold']};
        }}
    """,
    "language_badge_mixed": f"""
        QLabel {{
            background-color: {COLORS['mixed_badge']};
            color: white;
            padding: 2px 8px;
            border-radius: {SPACING['radius_sm']};
            font-size: {FONTS['size_small']};
            font-weight: {FONTS['weight_bold']};
        }}
    """,
    "file_list": f"""
        QListWidget {{
            background-color: {COLORS['bg_medium']};
            border: 1px solid {COLORS['border_default']};
            border-radius: {SPACING['radius_sm']};
        }}
        QListWidget::item {{
            padding: 4px;
            border-bottom: 1px solid {COLORS['border_light']};
        }}
    """,
    "preset_button": f"""
        QPushButton {{
            background-color: {COLORS['bg_lighter']};
            color: white;
            padding: 8px 16px;
            border-radius: {SPACING['radius_sm']};
            border: none;
            font-size: {FONTS['size_body']};
        }}
        QPushButton:hover {{
            background-color: {COLORS['border_default']};
        }}
    """,
    "preset_button_active": f"""
        QPushButton {{
            background-color: {COLORS['primary']};
            color: white;
            padding: 8px 16px;
            border-radius: {SPACING['radius_sm']};
            border: 2px solid {COLORS['primary_hover']};
            font-weight: {FONTS['weight_bold']};
            font-size: {FONTS['size_body']};
        }}
    """,
    "info_frame": f"""
        QFrame {{
            background-color: {COLORS['bg_medium']};
            border: 1px solid {COLORS['border_default']};
            border-radius: {SPACING['radius_lg']};
            padding: {SPACING['padding_md']};
        }}
    """,
    "tip_label": f"""
        QLabel {{
            color: {COLORS['text_secondary']};
            font-size: {FONTS['size_body']};
        }}
    """,
    "progress_container": f"""
        QWidget#progressWidget {{
            background-color: {COLORS['bg_medium']};
            border: 1px solid {COLORS['border_default']};
            border-radius: {SPACING['radius_lg']};
        }}
    """,
    "progress_bar": f"""
        QProgressBar {{
            background-color: {COLORS['bg_dark']};
            border: 1px solid {COLORS['border_default']};
            border-radius: {SPACING['radius_sm']};
            text-align: center;
            color: white;
            height: 20px;
        }}
        QProgressBar::chunk {{
            background-color: {COLORS['primary']};
            border-radius: 3px;
        }}
    """,
    "log_container": f"""
        QScrollArea {{
            background-color: {COLORS['bg_dark']};
            border: 1px solid {COLORS['border_default']};
            border-radius: {SPACING['radius_sm']};
        }}
        QWidget#logContent {{
            background-color: {COLORS['bg_dark']};
        }}
    """,
    "log_entry_success": f"color: {COLORS['success']}; font-family: monospace; font-size: 11px; padding: 2px;",
    "log_entry_warning": f"color: {COLORS['warning']}; font-family: monospace; font-size: 11px; padding: 2px;",
    "log_entry_error": f"color: {COLORS['error']}; font-family: monospace; font-size: 11px; padding: 2px;",
    "log_entry_info": "color: white; font-family: monospace; font-size: 11px; padding: 2px;",
}

def get_application_stylesheet() -> str:
    \"\"\"Returns the complete QSS for the entire application.\"\"\"
    return f\"\"\"
        QMainWindow, QDialog {{
            background-color: {COLORS['bg_medium']};
            color: {COLORS['text_primary']};
            font-family: {FONTS['family']};
        }}
        QWidget {{
            color: {COLORS['text_primary']};
            font-family: {FONTS['family']};
        }}
        QLabel {{
            color: {COLORS['text_primary']};
        }}
    \"\"\"

def get_widget_style(widget_type: str) -> str:
    \"\"\"Returns QSS for specific widget type.\"\"\"
    return STYLES.get(widget_type, "")

def apply_hover_effect(base_style: str, hover_color: str) -> str:
    \"\"\"Adds hover state to styles.\"\"\"
    return f\"\"\"{base_style}
        QPushButton:hover {{
            background-color: {hover_color};
        }}
    \"\"\"
