"""Configuration data model for obfuscation profiles.

This module defines the ObfuscationConfig dataclass that represents
obfuscation configuration profiles. It handles conversion between GUI
feature names and JSON schema feature names, validation, and serialization.

The configuration also includes multiprocessing controls for large projects:
- ``enable_multiprocessing`` toggles parallel processing globally.
- ``max_workers`` controls worker process count (or auto-detect when ``None``).
- ``batch_size`` controls files processed per worker batch.
- ``multiprocessing_threshold`` sets the minimum file count before parallel
  processing is used.

Disable multiprocessing when debugging transform behavior or when running in
memory-constrained environments. Worker count and adaptive batch sizing remain
auto-tuned by default.

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

import multiprocessing
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
    "Index Mangling": "mangle_indexes",
    "Roblox Exploit Defense": "roblox_exploit_defense",
    "Roblox Remote Spy Protection": "roblox_remote_spy",
}

# Valid feature names from JSON schema
VALID_FEATURES = {
    "mangle_globals",
    "mangle_indexes",
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
    "self_modifying_code",
}

# Valid preset names
VALID_PRESETS = {"light", "medium", "heavy", "maximum"}

DEFAULT_MEMORY_THRESHOLD_PERCENT = 80
DEFAULT_MIN_BATCH_SIZE = 25
DEFAULT_ENABLE_MULTIPROCESSING = True
DEFAULT_MAX_WORKERS: int | None = None
DEFAULT_BATCH_SIZE = 75
DEFAULT_MULTIPROCESSING_THRESHOLD = 100


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

    Features:
        mangle_globals: Rename global functions, classes, and variables
            - Renames all global symbols with generated identifiers
            - Preserves language builtins and reserved names
            - Maintains cross-file consistency via GlobalSymbolTable
            - Respects preserve_exports and preserve_constants flags
            - Handles both Python and Lua with language-specific rules
            
    Symbol Table Options:
        identifier_prefix: Prefix for mangled names (default: "_0x")
        mangling_strategy: "sequential" | "random" | "minimal"
            - sequential: _0x1, _0x2, _0x3, ... (deterministic)
            - random: _0xa3f2, _0x7b1c, ... (non-deterministic)
            - minimal: a, b, c, ..., aa, ab, ... (shortest names)
        preserve_exports: If True, exported symbols keep original names
        preserve_constants: If True, ALL_CAPS variables keep original names
    """

    name: str
    version: str = "1.0"
    language: str = "lua"
    preset: Optional[str] = None
    features: Dict[str, bool] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=lambda: {
        "string_encryption_key_length": 16,
        "array_shuffle_seed": None,
        "dead_code_percentage": 20,
        "identifier_prefix": "_0x",
        "number_obfuscation_complexity": 3,
        "number_obfuscation_min_value": 10,
        "number_obfuscation_max_value": 1000000,
        "vm_protection_complexity": 2,
        "vm_protect_all_functions": False,
        "vm_bytecode_encryption": True,
        "vm_protection_marker": "vm_protect",
        "opaque_predicate_complexity": 2,
        "opaque_predicate_percentage": 30,
        "anti_debug_aggressiveness": 2,
        "code_split_chunk_size": 5,
        "code_split_encryption": True,
        "self_modify_complexity": 2,
        "roblox_exploit_aggressiveness": 2,
        "roblox_exploit_action": "exit",
    })
    symbol_table_options: Dict[str, Any] = field(default_factory=lambda: {
        "identifier_prefix": "_0x",
        "mangling_strategy": "sequential",
        "preserve_exports": False,
        "preserve_constants": False,
    })
    runtime_mode: str = "hybrid"
    conflict_strategy: str = "ask"
    memory_threshold_percent: int = DEFAULT_MEMORY_THRESHOLD_PERCENT
    min_batch_size: int = DEFAULT_MIN_BATCH_SIZE
    enable_multiprocessing: bool = DEFAULT_ENABLE_MULTIPROCESSING  # Master switch for parallel processing. Disable for debugging or memory-constrained environments.
    max_workers: int | None = DEFAULT_MAX_WORKERS  # Maximum worker processes. None = auto-detect (CPU count - 1, max 8).
    batch_size: int = DEFAULT_BATCH_SIZE  # Files per batch. Auto-adjusted based on memory pressure during processing.
    multiprocessing_threshold: int = DEFAULT_MULTIPROCESSING_THRESHOLD  # Minimum file count to trigger multiprocessing. Below this, sequential processing is used.
    checkpoint_enabled: bool = True
    checkpoint_interval_files: int = 100
    checkpoint_interval_seconds: int = 300
    checkpoint_dir: str | None = None
    
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
        
        # Check runtime_mode
        if self.runtime_mode not in ("hybrid", "embedded"):
            raise ValueError(
                f"Invalid runtime_mode: {self.runtime_mode}. Expected 'hybrid' or 'embedded'"
            )

        # Check conflict_strategy
        valid_conflict_strategies = {"overwrite", "skip", "rename", "ask"}
        if self.conflict_strategy not in valid_conflict_strategies:
            raise ValueError(
                f"Invalid conflict_strategy: {self.conflict_strategy}. "
                f"Expected one of {valid_conflict_strategies}"
            )

        if not 50 <= self.memory_threshold_percent <= 95:
            raise ValueError(
                "memory_threshold_percent must be between 50 and 95"
            )

        if not 10 <= self.min_batch_size <= 50:
            raise ValueError("min_batch_size must be between 10 and 50")

        if not isinstance(self.enable_multiprocessing, bool):
            raise ValueError("enable_multiprocessing must be a boolean")

        if self.max_workers is not None:
            if not isinstance(self.max_workers, int) or isinstance(self.max_workers, bool):
                raise ValueError("max_workers must be an integer or None")
            if not 1 <= self.max_workers <= 16:
                raise ValueError("max_workers must be between 1 and 16")

            cpu_count = multiprocessing.cpu_count()
            if self.max_workers > cpu_count:
                logger.warning(
                    "max_workers=%d exceeds available CPU cores (%d)",
                    self.max_workers,
                    cpu_count,
                )

        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise ValueError("batch_size must be an integer")
        if not 10 <= self.batch_size <= 200:
            raise ValueError("batch_size must be between 10 and 200")
        if self.batch_size < self.min_batch_size:
            raise ValueError("batch_size must be greater than or equal to min_batch_size")

        if (
            not isinstance(self.multiprocessing_threshold, int)
            or isinstance(self.multiprocessing_threshold, bool)
        ):
            raise ValueError("multiprocessing_threshold must be an integer")
        if not 10 <= self.multiprocessing_threshold <= 1000:
            raise ValueError("multiprocessing_threshold must be between 10 and 1000")
        if self.multiprocessing_threshold < 50:
            logger.warning(
                "multiprocessing_threshold=%d is low and may add multiprocessing overhead",
                self.multiprocessing_threshold,
            )
            
        if not isinstance(self.checkpoint_enabled, bool):
            raise ValueError("checkpoint_enabled must be a boolean")
            
        if not isinstance(self.checkpoint_interval_files, int) or isinstance(self.checkpoint_interval_files, bool):
            raise ValueError("checkpoint_interval_files must be an integer")
        if not 1 <= self.checkpoint_interval_files <= 10000:
            raise ValueError("checkpoint_interval_files must be between 1 and 10000")
            
        if not isinstance(self.checkpoint_interval_seconds, int) or isinstance(self.checkpoint_interval_seconds, bool):
            raise ValueError("checkpoint_interval_seconds must be an integer")
        if not 30 <= self.checkpoint_interval_seconds <= 86400:
            raise ValueError("checkpoint_interval_seconds must be between 30 and 86400")
            
        if self.checkpoint_dir is not None and not isinstance(self.checkpoint_dir, str):
            raise ValueError("checkpoint_dir must be a string or None")
        
        # Check features
        for feature_name in self.features:
            if feature_name not in VALID_FEATURES:
                raise ValueError(
                    f"Unknown feature '{feature_name}' in configuration. "
                    f"Valid features: {VALID_FEATURES}"
                )
        
        # Check options types
        if "string_encryption_key_length" in self.options:
            key_length = self.options["string_encryption_key_length"]
            if not isinstance(key_length, int):
                raise ValueError(
                    "Option 'string_encryption_key_length' must be an integer"
                )
            if key_length <= 0:
                raise ValueError(
                    "Option 'string_encryption_key_length' must be a positive integer"
                )
            # Valid AES key lengths are 16, 24, 32 bytes; warn if too small
            if key_length < 16:
                logger.warning(
                    f"string_encryption_key_length={key_length} is small; "
                    "16 bytes or more is recommended for security"
                )
        
        if "dead_code_percentage" in self.options:
            if not isinstance(self.options["dead_code_percentage"], int):
                raise ValueError("Option 'dead_code_percentage' must be an integer")
        
        if "identifier_prefix" in self.options:
            if not isinstance(self.options["identifier_prefix"], str):
                raise ValueError("Option 'identifier_prefix' must be a string")

        if "array_shuffle_seed" in self.options:
            seed = self.options["array_shuffle_seed"]
            if seed is not None and not isinstance(seed, int):
                raise ValueError(
                    "Option 'array_shuffle_seed' must be an integer or None"
                )

        if "number_obfuscation_complexity" in self.options:
            complexity = self.options["number_obfuscation_complexity"]
            if not isinstance(complexity, int):
                raise ValueError(
                    "Option 'number_obfuscation_complexity' must be an integer"
                )
            if not 1 <= complexity <= 5:
                raise ValueError(
                    "Option 'number_obfuscation_complexity' must be between 1 and 5"
                )

        if "number_obfuscation_min_value" in self.options:
            min_val = self.options["number_obfuscation_min_value"]
            if not isinstance(min_val, int):
                raise ValueError(
                    "Option 'number_obfuscation_min_value' must be an integer"
                )
            if min_val < 0:
                raise ValueError(
                    "Option 'number_obfuscation_min_value' must be non-negative"
                )

        if "number_obfuscation_max_value" in self.options:
            max_val = self.options["number_obfuscation_max_value"]
            if not isinstance(max_val, int):
                raise ValueError(
                    "Option 'number_obfuscation_max_value' must be an integer"
                )
            if max_val <= 0:
                raise ValueError(
                    "Option 'number_obfuscation_max_value' must be positive"
                )

        # VM Protection options validation
        if "vm_protection_complexity" in self.options:
            vm_complexity = self.options["vm_protection_complexity"]
            if not isinstance(vm_complexity, int):
                raise ValueError(
                    "Option 'vm_protection_complexity' must be an integer"
                )
            if not 1 <= vm_complexity <= 3:
                raise ValueError(
                    "Option 'vm_protection_complexity' must be between 1 and 3"
                )

        if "vm_protect_all_functions" in self.options:
            if not isinstance(self.options["vm_protect_all_functions"], bool):
                raise ValueError(
                    "Option 'vm_protect_all_functions' must be a boolean"
                )

        if "vm_bytecode_encryption" in self.options:
            if not isinstance(self.options["vm_bytecode_encryption"], bool):
                raise ValueError(
                    "Option 'vm_bytecode_encryption' must be a boolean"
                )

        if "vm_protection_marker" in self.options:
            if not isinstance(self.options["vm_protection_marker"], str):
                raise ValueError(
                    "Option 'vm_protection_marker' must be a string"
                )

        # Opaque predicates options validation
        if "opaque_predicate_complexity" in self.options:
            complexity = self.options["opaque_predicate_complexity"]
            if not isinstance(complexity, int):
                raise ValueError(
                    "Option 'opaque_predicate_complexity' must be an integer"
                )
            if not 1 <= complexity <= 3:
                raise ValueError(
                    "Option 'opaque_predicate_complexity' must be between 1 and 3"
                )

        if "opaque_predicate_percentage" in self.options:
            percentage = self.options["opaque_predicate_percentage"]
            if not isinstance(percentage, int):
                raise ValueError(
                    "Option 'opaque_predicate_percentage' must be an integer"
                )
            if not 0 <= percentage <= 100:
                raise ValueError(
                    "Option 'opaque_predicate_percentage' must be between 0 and 100"
                )

        # Anti-debugging options validation
        if "anti_debug_aggressiveness" in self.options:
            aggressiveness = self.options["anti_debug_aggressiveness"]
            if not isinstance(aggressiveness, int):
                raise ValueError(
                    "Option 'anti_debug_aggressiveness' must be an integer"
                )
            if not 1 <= aggressiveness <= 3:
                raise ValueError(
                    "Option 'anti_debug_aggressiveness' must be between 1 and 3"
                )

        # Roblox exploit defense options validation
        if "roblox_exploit_aggressiveness" in self.options:
            aggressiveness = self.options["roblox_exploit_aggressiveness"]
            if not isinstance(aggressiveness, int):
                raise ValueError(
                    "Option 'roblox_exploit_aggressiveness' must be an integer"
                )
            if not 1 <= aggressiveness <= 3:
                raise ValueError(
                    "Option 'roblox_exploit_aggressiveness' must be between 1 and 3"
                )

        if "roblox_exploit_action" in self.options:
            action = self.options["roblox_exploit_action"]
            if action not in ("exit", "loop", "exception"):
                raise ValueError(
                    "Option 'roblox_exploit_action' must be 'exit', 'loop', or 'exception'"
                )

        # Code splitting options validation
        if "code_split_chunk_size" in self.options:
            chunk_size = self.options["code_split_chunk_size"]
            if not isinstance(chunk_size, int):
                raise ValueError(
                    "Option 'code_split_chunk_size' must be an integer"
                )
            if chunk_size < 2:
                raise ValueError(
                    "Option 'code_split_chunk_size' must be >= 2"
                )

        if "code_split_encryption" in self.options:
            if not isinstance(self.options["code_split_encryption"], bool):
                raise ValueError(
                    "Option 'code_split_encryption' must be a boolean"
                )

        # Self-modifying code options validation
        if "self_modify_complexity" in self.options:
            sm_complexity = self.options["self_modify_complexity"]
            if not isinstance(sm_complexity, int):
                raise ValueError(
                    "Option 'self_modify_complexity' must be an integer"
                )
            if not 1 <= sm_complexity <= 3:
                raise ValueError(
                    "Option 'self_modify_complexity' must be between 1 and 3"
                )

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
            "runtime_mode": self.runtime_mode,
            "conflict_strategy": self.conflict_strategy,
            "memory_threshold_percent": self.memory_threshold_percent,
            "min_batch_size": self.min_batch_size,
            "enable_multiprocessing": self.enable_multiprocessing,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "multiprocessing_threshold": self.multiprocessing_threshold,
            "checkpoint_enabled": self.checkpoint_enabled,
            "checkpoint_interval_files": self.checkpoint_interval_files,
            "checkpoint_interval_seconds": self.checkpoint_interval_seconds,
            "checkpoint_dir": self.checkpoint_dir,
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
                    "array_shuffle_seed": None,
                    "dead_code_percentage": 20,
                    "identifier_prefix": "_0x",
                    "number_obfuscation_complexity": 3,
                    "number_obfuscation_min_value": 10,
                    "number_obfuscation_max_value": 1000000,
                    "vm_protection_complexity": 2,
                    "vm_protect_all_functions": False,
                    "vm_bytecode_encryption": True,
                    "vm_protection_marker": "vm:protect",
                    "opaque_predicate_complexity": 2,
                    "opaque_predicate_percentage": 30,
                    "anti_debug_aggressiveness": 2,
                    "code_split_chunk_size": 5,
                    "code_split_encryption": True,
                }),
                symbol_table_options=data.get("symbol_table_options", {
                    "identifier_prefix": "_0x",
                    "mangling_strategy": "sequential",
                    "preserve_exports": False,
                    "preserve_constants": False,
                }),
                runtime_mode=data.get("runtime_mode", "hybrid"),
                conflict_strategy=data.get("conflict_strategy", "ask"),
                memory_threshold_percent=data.get(
                    "memory_threshold_percent",
                    DEFAULT_MEMORY_THRESHOLD_PERCENT,
                ),
                min_batch_size=data.get("min_batch_size", DEFAULT_MIN_BATCH_SIZE),
                enable_multiprocessing=data.get(
                    "enable_multiprocessing",
                    DEFAULT_ENABLE_MULTIPROCESSING,
                ),
                max_workers=data.get("max_workers", DEFAULT_MAX_WORKERS),
                batch_size=data.get("batch_size", DEFAULT_BATCH_SIZE),
                multiprocessing_threshold=data.get(
                    "multiprocessing_threshold",
                    DEFAULT_MULTIPROCESSING_THRESHOLD,
                ),
                checkpoint_enabled=data.get("checkpoint_enabled", True),
                checkpoint_interval_files=data.get("checkpoint_interval_files", 100),
                checkpoint_interval_seconds=data.get("checkpoint_interval_seconds", 300),
                checkpoint_dir=data.get("checkpoint_dir", None),
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
        language: str = "lua",
        enable_multiprocessing: bool = DEFAULT_ENABLE_MULTIPROCESSING,
        max_workers: int | None = DEFAULT_MAX_WORKERS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        multiprocessing_threshold: int = DEFAULT_MULTIPROCESSING_THRESHOLD,
        memory_threshold_percent: int = DEFAULT_MEMORY_THRESHOLD_PERCENT,
    ) -> ObfuscationConfig:
        """Create configuration from GUI feature settings.

        Converts GUI feature names to JSON schema feature names using
        the GUI_TO_JSON_FEATURE_MAP.

        Args:
            preset: Preset name (light/medium/heavy/maximum) or None
            features: Dictionary mapping GUI feature names to enabled state
            name: Profile name
            language: Target language (default: "lua")
            enable_multiprocessing: Master multiprocessing toggle (default: True)
            max_workers: Maximum worker process count, or None for auto-detect
            batch_size: Initial multiprocessing batch size (default: 75)
            multiprocessing_threshold: File-count threshold for parallel processing
            memory_threshold_percent: Memory usage threshold for adaptive batch reduction

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
                "array_shuffle_seed": None,
                "dead_code_percentage": 20,
                "identifier_prefix": "_0x",
                "number_obfuscation_complexity": 3,
                "number_obfuscation_min_value": 10,
                "number_obfuscation_max_value": 1000000,
                "vm_protection_complexity": 2,
                "vm_protect_all_functions": False,
                "vm_bytecode_encryption": True,
                "vm_protection_marker": "vm:protect",
                "opaque_predicate_complexity": 2,
                "opaque_predicate_percentage": 30,
                "anti_debug_aggressiveness": 2,
                "code_split_chunk_size": 5,
                "code_split_encryption": True,
            },
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            },
            runtime_mode="hybrid",
            conflict_strategy="ask",
            memory_threshold_percent=memory_threshold_percent,
            min_batch_size=DEFAULT_MIN_BATCH_SIZE,
            enable_multiprocessing=enable_multiprocessing,
            max_workers=max_workers,
            batch_size=batch_size,
            multiprocessing_threshold=multiprocessing_threshold,
        )

        logger.debug(
            f"Created configuration from GUI settings: {name} "
            f"(preset: {preset}, {len(json_features)} features)"
        )

        return config
