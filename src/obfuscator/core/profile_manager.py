"""Profile management for obfuscation configurations.

This module provides the ProfileManager class for saving, loading, and
validating obfuscation configuration profiles. It handles JSON serialization,
file I/O, and provides default preset profiles.

Example:
    Saving a profile:
    
    >>> from pathlib import Path
    >>> manager = ProfileManager()
    >>> config = ObfuscationConfig(name="My Profile", language="lua")
    >>> manager.save_profile(config, Path("my_profile.json"))
    
    Loading a profile:
    
    >>> config = manager.load_profile(Path("my_profile.json"))
    >>> print(config.name)
    My Profile
    
    Getting a default profile:
    
    >>> config = manager.get_default_profile("Medium")
    >>> print(config.preset)
    medium
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any
from copy import deepcopy

from obfuscator.utils.logger import get_logger
from obfuscator.utils.path_utils import normalize_path, ensure_directory
from obfuscator.core.config import ObfuscationConfig

logger = get_logger("obfuscator.core.profile_manager")

# Default profile configurations based on security_config_widget.py PRESET_CONFIGS
DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "Light": {
        "version": "1.0",
        "name": "Light Profile",
        "language": "lua",
        "preset": "light",
        "features": {
            "mangle_globals": True,
            "string_encryption": False,
            "number_obfuscation": False,
            "dead_code_injection": False,
            "control_flow_flattening": False,
            "opaque_predicates": False,
            "constant_array": False,
            "anti_debugging": False,
            "vm_protection": False,
            "code_splitting": False,
            "roblox_exploit_defense": False,
            "roblox_remote_spy": False,
            "self_modifying_code": False,
            "virtualization": False,
        },
        "options": {
            "string_encryption_key_length": 16,
            "dead_code_percentage": 20,
            "identifier_prefix": "_0x",
            "opaque_predicate_complexity": 2,
            "opaque_predicate_percentage": 30,
            "anti_debug_aggressiveness": 2,
            "code_split_chunk_size": 5,
            "code_split_encryption": True,
            "self_modify_complexity": 2,
        },
        "symbol_table_options": {
            "identifier_prefix": "_0x",
            "mangling_strategy": "sequential",
            "preserve_exports": False,
            "preserve_constants": False,
        }
    },
    "Medium": {
        "version": "1.0",
        "name": "Medium Profile",
        "language": "lua",
        "preset": "medium",
        "features": {
            "mangle_globals": True,
            "string_encryption": True,
            "number_obfuscation": True,
            "dead_code_injection": True,
            "control_flow_flattening": False,
            "opaque_predicates": False,
            "constant_array": False,
            "anti_debugging": False,
            "vm_protection": False,
            "code_splitting": False,
            "roblox_exploit_defense": False,
            "roblox_remote_spy": False,
            "self_modifying_code": False,
            "virtualization": False,
        },
        "options": {
            "string_encryption_key_length": 16,
            "dead_code_percentage": 15,
            "identifier_prefix": "_0x",
            "opaque_predicate_complexity": 2,
            "opaque_predicate_percentage": 30,
            "anti_debug_aggressiveness": 2,
            "code_split_chunk_size": 5,
            "code_split_encryption": True,
            "self_modify_complexity": 2,
        },
        "symbol_table_options": {
            "identifier_prefix": "_0x",
            "mangling_strategy": "sequential",
            "preserve_exports": False,
            "preserve_constants": False,
        }
    },
    "Heavy": {
        "version": "1.0",
        "name": "Heavy Profile",
        "language": "lua",
        "preset": "heavy",
        "features": {
            "mangle_globals": True,
            "string_encryption": True,
            "number_obfuscation": True,
            "dead_code_injection": True,
            "control_flow_flattening": True,
            "opaque_predicates": True,
            "constant_array": True,
            "anti_debugging": False,
            "vm_protection": False,
            "code_splitting": False,
            "roblox_exploit_defense": False,
            "roblox_remote_spy": False,
            "self_modifying_code": True,
            "virtualization": False,
        },
        "options": {
            "string_encryption_key_length": 16,
            "dead_code_percentage": 20,
            "identifier_prefix": "_0x",
            "opaque_predicate_complexity": 2,
            "opaque_predicate_percentage": 30,
            "anti_debug_aggressiveness": 2,
            "code_split_chunk_size": 5,
            "code_split_encryption": True,
            "self_modify_complexity": 1,
        },
        "symbol_table_options": {
            "identifier_prefix": "_0x",
            "mangling_strategy": "random",
            "preserve_exports": False,
            "preserve_constants": True,
        }
    },
    "Maximum": {
        "version": "1.0",
        "name": "Maximum Profile",
        "language": "lua",
        "preset": "maximum",
        "features": {
            "mangle_globals": True,
            "string_encryption": True,
            "number_obfuscation": True,
            "dead_code_injection": True,
            "control_flow_flattening": True,
            "opaque_predicates": True,
            "constant_array": True,
            "anti_debugging": True,
            "vm_protection": True,
            "code_splitting": True,
            "roblox_exploit_defense": True,
            "roblox_remote_spy": True,
            "self_modifying_code": True,
            "virtualization": False,
        },
        "options": {
            "string_encryption_key_length": 16,
            "dead_code_percentage": 20,
            "identifier_prefix": "_0x",
            "opaque_predicate_complexity": 2,
            "opaque_predicate_percentage": 30,
            "anti_debug_aggressiveness": 3,
            "code_split_chunk_size": 3,
            "code_split_encryption": True,
            "self_modify_complexity": 3,
        },
        "symbol_table_options": {
            "identifier_prefix": "_0x",
            "mangling_strategy": "random",
            "preserve_exports": False,
            "preserve_constants": True,
        }
    }
}


class ProfileManager:
    """Manager for obfuscation configuration profiles.

    This class provides methods for saving, loading, validating, and
    managing obfuscation configuration profiles. It handles JSON
    serialization and provides access to default preset profiles.
    """

    @staticmethod
    def save_profile(config: ObfuscationConfig, file_path: Path) -> None:
        """Save a configuration profile to a JSON file.

        Args:
            config: ObfuscationConfig instance to save
            file_path: Path where the profile should be saved

        Raises:
            ValueError: If configuration validation fails
            OSError: If file write operation fails
            json.JSONEncodeError: If JSON serialization fails

        Example:
            >>> config = ObfuscationConfig(name="Test", language="lua")
            >>> ProfileManager.save_profile(config, Path("test.json"))
        """
        try:
            # Validate configuration
            config.validate()

            # Convert to dictionary
            config_dict = config.to_dict()

            # Ensure parent directory exists
            ensure_directory(file_path.parent)

            # Write JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)

            logger.debug(f"Profile '{config.name}' saved to {file_path}")

        except ValueError as e:
            logger.error(f"Validation failed for profile '{config.name}': {e}")
            raise ValueError(f"Configuration validation failed: {e}")
        except OSError as e:
            logger.error(f"Failed to write profile to {file_path}: {e}")
            raise OSError(f"Failed to write profile file: {e}")
        except TypeError as e:
            logger.error(f"Failed to serialize profile '{config.name}': {e}")
            raise TypeError(f"Failed to serialize configuration: {e}")

    @staticmethod
    def load_profile(file_path: Path) -> ObfuscationConfig:
        """Load a configuration profile from a JSON file.

        Args:
            file_path: Path to the profile JSON file

        Returns:
            ObfuscationConfig instance loaded from file

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If JSON is invalid or validation fails

        Example:
            >>> config = ProfileManager.load_profile(Path("test.json"))
            >>> print(config.name)
        """
        # Check if file exists
        if not file_path.exists():
            logger.error(f"Profile file not found: {file_path}")
            raise FileNotFoundError(f"Profile file not found: {file_path}")

        try:
            # Read and parse JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Create config from dictionary
            config = ObfuscationConfig.from_dict(data)

            # Validate the loaded config
            config.validate()

            logger.debug(f"Profile '{config.name}' loaded from {file_path}")
            return config

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in profile file {file_path}: {e}")
            raise ValueError(f"Invalid JSON format in profile file: {e}")
        except KeyError as e:
            logger.error(f"Missing required field in profile {file_path}: {e}")
            raise ValueError(f"Missing required field in profile: {e}")
        except ValueError as e:
            logger.error(f"Validation failed for profile from {file_path}: {e}")
            raise ValueError(f"Profile validation failed: {e}")

    @staticmethod
    def validate_profile(file_path: Path) -> bool:
        """Validate a profile file without loading it.

        Args:
            file_path: Path to the profile JSON file

        Returns:
            True if profile is valid, False otherwise

        Example:
            >>> is_valid = ProfileManager.validate_profile(Path("test.json"))
            >>> if is_valid:
            ...     print("Profile is valid")
        """
        try:
            ProfileManager.load_profile(file_path)
            logger.debug(f"Profile validation successful: {file_path}")
            return True
        except Exception as e:
            logger.warning(f"Profile validation failed for {file_path}: {e}")
            return False

    @staticmethod
    def get_default_profile(preset_name: str) -> ObfuscationConfig:
        """Get a default preset profile.

        Args:
            preset_name: Name of the preset (case-insensitive)
                        Valid values: "Light", "Medium", "Heavy", "Maximum"

        Returns:
            ObfuscationConfig instance for the requested preset

        Raises:
            ValueError: If preset_name is not valid

        Example:
            >>> config = ProfileManager.get_default_profile("Medium")
            >>> print(config.preset)
            medium
        """
        # Normalize preset name to title case
        preset_title = preset_name.strip().title()

        if preset_title not in DEFAULT_PROFILES:
            valid_presets = list(DEFAULT_PROFILES.keys())
            logger.error(f"Invalid preset name: {preset_name}")
            raise ValueError(
                f"Invalid preset name: {preset_name}. "
                f"Valid presets: {valid_presets}"
            )

        # Create a deep copy to avoid modifying the default
        profile_data = deepcopy(DEFAULT_PROFILES[preset_title])
        config = ObfuscationConfig.from_dict(profile_data)

        logger.debug(f"Retrieved default profile: {preset_title}")
        return config

    @staticmethod
    def list_default_profiles() -> List[str]:
        """Get list of available default profile names.

        Returns:
            List of default profile names

        Example:
            >>> profiles = ProfileManager.list_default_profiles()
            >>> print(profiles)
            ['Light', 'Medium', 'Heavy', 'Maximum']
        """
        return list(DEFAULT_PROFILES.keys())

