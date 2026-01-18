"""Configuration data model for obfuscation profiles.

This module defines the ObfuscationConfig dataclass that represents
obfuscation configuration profiles. It handles conversion between GUI
feature names and JSON schema feature names, validation, and serialization.

Example:
    Creating a configuration from GUI settings:
    
    >>> config = ObfuscationConfig.from_gui_config(
    ...     preset="medium",
    ...     features={"Variable Renaming": True, "String Encryption": True},
    ...     name="My Custom Profile",
    ...     language="lua"
    ... )
    >>> config.validate()
    
    Converting to dictionary for JSON serialization:
    
    >>> config_dict = config.to_dict()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from pathlib import Path

from obfuscator.utils.logger import get_logger

logger = get_logger("obfuscator.core.config")

# Mapping from GUI feature names to JSON schema feature names
GUI_TO_JSON_FEATURE_MAP = {
    "Variable Renaming": "mangle_globals",
    "Function Renaming": "mangle_globals",
    "String Encryption": "string_encryption",
    "Number Obfuscation": "number_obfuscation",
    "Dead Code Injection": "dead_code_injection",
    "Comment Removal": "comment_removal",  # Not in JSON schema, handled separately
    "Control Flow Flattening": "control_flow_flattening",
    "Opaque Predicates": "opaque_predicates",
    "Constant Folding": "constant_array",
    "Anti-Debug": "anti_debugging",
    "VM Protection": "vm_protection",
    "Bytecode Compilation": "code_splitting",
    "Roblox API Preservation": "roblox_exploit_defense",
    "Luau Type Stripping": "roblox_remote_spy",
}

# Valid feature names from JSON schema
VALID_FEATURES = {
    "mangle_globals",
    "string_encryption",
    "number_obfuscation",
    "dead_code_injection",
    "control_flow_flattening",
    "opaque_predicates",
    "constant_array",
    "anti_debugging",
    "vm_protection",
    "code_splitting",
    "roblox_exploit_defense",
    "roblox_remote_spy",
    "anti_tamper",
    "virtualization",
}

# Valid preset names
VALID_PRESETS = {"light", "medium", "heavy", "maximum"}


@dataclass
class ObfuscationConfig:
    """Obfuscation configuration profile.

    This dataclass represents a complete obfuscation configuration profile
    matching the JSON schema format. It includes validation and conversion
    methods for working with GUI configurations.

    Attributes:
        version: Schema version (currently "1.0")
        name: Profile name
        language: Target language ("python" or "lua")
        preset: Preset name (light/medium/heavy/maximum) or None for custom
        features: Dictionary of feature flags (feature_name -> enabled)
        options: Dictionary of additional options with default values
        symbol_table_options: Configuration for the global symbol table
    """

    name: str
    version: str = "1.0"
    language: str = "lua"
    preset: Optional[str] = None
    features: Dict[str, bool] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=lambda: {
        "string_encryption_key_length": 16,
        "dead_code_percentage": 20,
        "identifier_prefix": "_0x",
    })
    symbol_table_options: Dict[str, Any] = field(default_factory=lambda: {
        "identifier_prefix": "_0x",
        "mangling_strategy": "sequential",
        "preserve_exports": False,
        "preserve_constants": False,
    })
    
    def validate(self) -> None:
        """Validate the configuration.
        
        Raises:
            ValueError: If any validation check fails
        """
        # Check version
        if self.version != "1.0":
            raise ValueError(f"Invalid version: {self.version}. Expected '1.0'")
        
        # Check language
        if self.language not in ("python", "lua"):
            raise ValueError(
                f"Invalid language: {self.language}. Expected 'python' or 'lua'"
            )
        
        # Check preset
        if self.preset is not None:
            preset_lower = self.preset.lower()
            if preset_lower not in VALID_PRESETS:
                raise ValueError(
                    f"Invalid preset: {self.preset}. "
                    f"Expected one of {VALID_PRESETS} or None"
                )
        
        # Check features
        for feature_name in self.features:
            if feature_name not in VALID_FEATURES:
                raise ValueError(
                    f"Unknown feature '{feature_name}' in configuration. "
                    f"Valid features: {VALID_FEATURES}"
                )
        
        # Check options types
        if "string_encryption_key_length" in self.options:
            if not isinstance(self.options["string_encryption_key_length"], int):
                raise ValueError(
                    "Option 'string_encryption_key_length' must be an integer"
                )
        
        if "dead_code_percentage" in self.options:
            if not isinstance(self.options["dead_code_percentage"], int):
                raise ValueError("Option 'dead_code_percentage' must be an integer")
        
        if "identifier_prefix" in self.options:
            if not isinstance(self.options["identifier_prefix"], str):
                raise ValueError("Option 'identifier_prefix' must be a string")

        # Check symbol_table_options
        valid_strategies = {"sequential", "random", "minimal"}
        if "mangling_strategy" in self.symbol_table_options:
            strategy = self.symbol_table_options["mangling_strategy"]
            if strategy not in valid_strategies:
                raise ValueError(
                    f"Invalid mangling_strategy: {strategy}. "
                    f"Expected one of {valid_strategies}"
                )

        if "identifier_prefix" in self.symbol_table_options:
            if not isinstance(self.symbol_table_options["identifier_prefix"], str):
                raise ValueError(
                    "symbol_table_options 'identifier_prefix' must be a string"
                )

        if "preserve_exports" in self.symbol_table_options:
            if not isinstance(self.symbol_table_options["preserve_exports"], bool):
                raise ValueError(
                    "symbol_table_options 'preserve_exports' must be a boolean"
                )

        if "preserve_constants" in self.symbol_table_options:
            if not isinstance(self.symbol_table_options["preserve_constants"], bool):
                raise ValueError(
                    "symbol_table_options 'preserve_constants' must be a boolean"
                )

        logger.debug(f"Configuration '{self.name}' validated successfully")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the configuration

        Example:
            >>> config = ObfuscationConfig(name="Test")
            >>> config_dict = config.to_dict()
            >>> isinstance(config_dict, dict)
            True
        """
        return {
            "version": self.version,
            "name": self.name,
            "language": self.language,
            "preset": self.preset,
            "features": self.features.copy(),
            "options": self.options.copy(),
            "symbol_table_options": self.symbol_table_options.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObfuscationConfig:
        """Create configuration from dictionary.

        Args:
            data: Dictionary containing configuration data

        Returns:
            ObfuscationConfig instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If data validation fails

        Example:
            >>> data = {
            ...     "version": "1.0",
            ...     "name": "Test Profile",
            ...     "language": "lua",
            ...     "preset": "medium",
            ...     "features": {"mangle_globals": True},
            ...     "options": {}
            ... }
            >>> config = ObfuscationConfig.from_dict(data)
        """
        try:
            config = cls(
                version=data["version"],
                name=data["name"],
                language=data.get("language", "lua"),
                preset=data.get("preset"),
                features=data.get("features", {}),
                options=data.get("options", {
                    "string_encryption_key_length": 16,
                    "dead_code_percentage": 20,
                    "identifier_prefix": "_0x",
                }),
                symbol_table_options=data.get("symbol_table_options", {
                    "identifier_prefix": "_0x",
                    "mangling_strategy": "sequential",
                    "preserve_exports": False,
                    "preserve_constants": False,
                }),
            )
            logger.debug(f"Created configuration from dictionary: {config.name}")
            return config
        except KeyError as e:
            raise KeyError(f"Missing required field in configuration: {e}")

    @classmethod
    def from_gui_config(
        cls,
        preset: Optional[str],
        features: Dict[str, bool],
        name: str,
        language: str = "lua"
    ) -> ObfuscationConfig:
        """Create configuration from GUI feature settings.

        Converts GUI feature names to JSON schema feature names using
        the GUI_TO_JSON_FEATURE_MAP.

        Args:
            preset: Preset name (light/medium/heavy/maximum) or None
            features: Dictionary mapping GUI feature names to enabled state
            name: Profile name
            language: Target language (default: "lua")

        Returns:
            ObfuscationConfig instance

        Example:
            >>> config = ObfuscationConfig.from_gui_config(
            ...     preset="medium",
            ...     features={"Variable Renaming": True, "String Encryption": True},
            ...     name="My Profile"
            ... )
        """
        # Convert GUI feature names to JSON schema feature names
        json_features: Dict[str, bool] = {}

        for gui_name, enabled in features.items():
            json_name = GUI_TO_JSON_FEATURE_MAP.get(gui_name)
            if json_name and json_name != "comment_removal":
                # Skip comment_removal as it's not in JSON schema
                json_features[json_name] = enabled

        config = cls(
            version="1.0",
            name=name,
            language=language,
            preset=preset.lower() if preset else None,
            features=json_features,
            options={
                "string_encryption_key_length": 16,
                "dead_code_percentage": 20,
                "identifier_prefix": "_0x",
            },
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )

        logger.debug(
            f"Created configuration from GUI settings: {name} "
            f"(preset: {preset}, {len(json_features)} features)"
        )

        return config

