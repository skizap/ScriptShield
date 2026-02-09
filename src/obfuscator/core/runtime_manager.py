"""
Centralized runtime manager for obfuscation runtime code generation.

This module provides the RuntimeManager class that acts as a facade over all
runtime generation modules. It maps feature flags to their corresponding runtime
generators, extracts configuration parameters from ObfuscationConfig, and
orchestrates runtime code generation for both Python and Lua.

Example:
    Typical usage workflow:

    >>> from obfuscator.core.config import ObfuscationConfig
    >>> from obfuscator.core.runtime_manager import RuntimeManager
    >>>
    >>> config = ObfuscationConfig.from_dict({
    ...     "version": "1.0",
    ...     "name": "Test Profile",
    ...     "language": "lua",
    ...     "features": {
    ...         "vm_protection": True,
    ...         "anti_debugging": True
    ...     },
    ...     "options": {}
    ... })
    >>>
    >>> manager = RuntimeManager(config)
    >>> required = manager.collect_required_runtimes("lua")
    >>> print(f"Required runtimes: {required}")
    >>>
    >>> runtime_code = manager.generate_runtime_code("vm_protection", "lua")
    >>> imports = manager.get_runtime_imports("vm_protection", "lua")
"""

import secrets
import logging
from typing import Dict, List, Any, Callable, Optional

from obfuscator.core.config import ObfuscationConfig
from obfuscator.processors import (
    vm_runtime_python,
    vm_runtime_lua,
    code_split_runtime_python,
    code_split_runtime_lua,
    self_modify_runtime_python,
    self_modify_runtime_lua,
    anti_debug_runtime_python,
    anti_debug_runtime_lua,
    roblox_exploit_runtime_lua,
    roblox_remote_spy_runtime_lua,
)

logger = logging.getLogger("obfuscator.core.runtime_manager")


class RuntimeManager:
    """
    Centralized manager for obfuscation runtime code generation.

    The RuntimeManager acts as a facade over all runtime generation modules,
    providing a unified interface for generating runtime code based on feature
    flags and configuration options. It handles:

    - Mapping feature flags to runtime generator functions
    - Extracting configuration parameters for each runtime type
    - Generating runtime code for both Python and Lua
    - Providing import statements for runtime integration
    - Batch generation of all required runtimes

    Attributes:
        config: The ObfuscationConfig instance containing feature flags and options

    Example:
        >>> config = ObfuscationConfig(name="My Profile", features={"vm_protection": True})
        >>> manager = RuntimeManager(config)
        >>> runtimes = manager.generate_all_runtimes("lua")
    """

    # Mapping of feature flags to Python runtime generator functions
    _PYTHON_RUNTIME_GENERATORS: Dict[str, Callable[..., str]] = {
        "vm_protection": vm_runtime_python.generate_python_vm_runtime,
        "code_splitting": code_split_runtime_python.generate_python_chunk_decryption_runtime,
        "self_modifying_code": self_modify_runtime_python.generate_python_self_modify_runtime,
        "anti_debugging": anti_debug_runtime_python.generate_python_anti_debug_checks,
    }

    # Mapping of feature flags to Lua runtime generator functions
    _LUA_RUNTIME_GENERATORS: Dict[str, Callable[..., str]] = {
        "vm_protection": vm_runtime_lua.generate_lua_vm_runtime,
        "code_splitting": code_split_runtime_lua.generate_lua_chunk_decryption_runtime,
        "self_modifying_code": self_modify_runtime_lua.generate_lua_self_modify_runtime,
        "anti_debugging": anti_debug_runtime_lua.generate_lua_anti_debug_checks,
        "roblox_exploit_defense": roblox_exploit_runtime_lua.generate_roblox_exploit_defense_checks,
        "roblox_remote_spy": roblox_remote_spy_runtime_lua.generate_roblox_remote_spy_runtime,
    }

    def __init__(self, config: ObfuscationConfig) -> None:
        """
        Initialize the RuntimeManager with an obfuscation configuration.

        Args:
            config: The ObfuscationConfig instance containing feature flags and options
        """
        self.config = config
        # Store encryption keys for runtimes that need them
        # Keys are generated once and reused to ensure consistency between
        # transformer and runtime generation
        self._runtime_keys: Dict[str, bytes] = {}
        logger.debug(f"RuntimeManager initialized for config: {config.name}")

    def collect_required_runtimes(self, language: str) -> List[str]:
        """
        Collect the list of feature names that require runtime code generation.

        Iterates through the config.features dictionary and identifies which
        enabled features have corresponding runtime generators for the specified
        language.

        Args:
            language: Target language ("python" or "lua")

        Returns:
            List of feature names requiring runtime code generation

        Raises:
            ValueError: If language is not "python" or "lua"

        Example:
            >>> required = manager.collect_required_runtimes("lua")
            >>> print(required)  # ['vm_protection', 'anti_debugging']
        """
        if language not in ("python", "lua"):
            raise ValueError(f"Invalid language: {language}. Expected 'python' or 'lua'")

        # Select appropriate generator mapping
        generator_map = (
            self._PYTHON_RUNTIME_GENERATORS if language == "python"
            else self._LUA_RUNTIME_GENERATORS
        )

        required_runtimes: List[str] = []

        # Check each enabled feature
        for feature_name, enabled in self.config.features.items():
            if enabled and feature_name in generator_map:
                required_runtimes.append(feature_name)
                logger.debug(f"Runtime required for feature: {feature_name}")

        logger.info(f"Collected {len(required_runtimes)} required runtimes for {language}")
        return required_runtimes

    def _get_runtime_params(self, runtime_type: str, language: str) -> Dict[str, Any]:
        """
        Extract runtime parameters from configuration for a specific runtime type.

        This internal method maps configuration options to the parameters
        expected by each runtime generator function.

        Args:
            runtime_type: The type of runtime (e.g., "vm_protection", "anti_debugging")
            language: Target language ("python" or "lua")

        Returns:
            Dictionary of parameters to pass to the runtime generator

        Example:
            >>> params = manager._get_runtime_params("vm_protection", "lua")
            >>> print(params)  # {'bytecode_obfuscation': True}
        """
        params: Dict[str, Any] = {}

        if runtime_type == "vm_protection":
            encryption_flag = self.config.options.get("vm_bytecode_encryption", True)
            if language == "python":
                params = {"bytecode_encryption": encryption_flag}
            else:  # lua
                params = {"bytecode_obfuscation": encryption_flag}

        elif runtime_type == "code_splitting":
            encryption_flag = self.config.options.get("code_split_encryption", True)
            # Use stored key or generate and store a new 16-byte encryption key
            if "code_splitting" not in self._runtime_keys:
                self._runtime_keys["code_splitting"] = secrets.token_bytes(16)
            encryption_key = self._runtime_keys["code_splitting"]
            params = {
                "encryption_key": encryption_key,
                "encryption_enabled": encryption_flag,
            }

        elif runtime_type == "self_modifying_code":
            complexity = self.config.options.get("self_modify_complexity", 2)
            params = {"complexity": complexity}

        elif runtime_type == "anti_debugging":
            aggressiveness = self.config.options.get("anti_debug_aggressiveness", 2)
            # Determine action from roblox_exploit_action option (default to "exit")
            action = self.config.options.get("roblox_exploit_action", "exit")
            params = {"aggressiveness": aggressiveness, "action": action}

        elif runtime_type == "roblox_exploit_defense":
            aggressiveness = self.config.options.get("roblox_exploit_aggressiveness", 2)
            action = self.config.options.get("roblox_exploit_action", "exit")
            # script_hash is None for now (hash computation can be added later)
            params = {
                "aggressiveness": aggressiveness,
                "action": action,
                "script_hash": None,
            }

        elif runtime_type == "roblox_remote_spy":
            # Use stored key or generate and store a new 16-byte encryption key
            if "roblox_remote_spy" not in self._runtime_keys:
                self._runtime_keys["roblox_remote_spy"] = secrets.token_bytes(16)
            encryption_key = self._runtime_keys["roblox_remote_spy"]
            # Empty map as placeholder (actual map would come from transformer)
            params = {
                "encryption_key": encryption_key,
                "remote_name_map": {},
            }

        else:
            # Unknown runtime type - return empty dict
            logger.warning(f"Unknown runtime type: {runtime_type}")

        return params

    def generate_runtime_code(self, runtime_type: str, language: str) -> str:
        """
        Generate runtime code for a specific runtime type and language.

        Retrieves the appropriate runtime generator from the mapping, extracts
        configuration parameters, and invokes the generator with those parameters.

        Args:
            runtime_type: The type of runtime to generate (e.g., "vm_protection")
            language: Target language ("python" or "lua")

        Returns:
            Generated runtime code as a string, or empty string if generation fails

        Raises:
            ValueError: If language is not "python" or "lua"

        Example:
            >>> code = manager.generate_runtime_code("anti_debugging", "python")
            >>> print(len(code))  # Size of generated code
        """
        if language not in ("python", "lua"):
            raise ValueError(f"Invalid language: {language}. Expected 'python' or 'lua'")

        # Select appropriate generator mapping
        generator_map = (
            self._PYTHON_RUNTIME_GENERATORS if language == "python"
            else self._LUA_RUNTIME_GENERATORS
        )

        # Check if runtime_type exists in mapping
        if runtime_type not in generator_map:
            logger.warning(f"No runtime generator found for type: {runtime_type}")
            return ""

        # Get the generator function
        generator = generator_map[runtime_type]

        # Get parameters for this runtime type
        params = self._get_runtime_params(runtime_type, language)

        try:
            # Invoke generator with unpacked parameters
            generated_code = generator(**params)
            code_size = len(generated_code)
            logger.debug(f"Generated runtime code for {runtime_type} ({language}): {code_size} chars")
            return generated_code
        except Exception as e:
            logger.error(f"Error generating runtime code for {runtime_type}: {e}")
            return ""

    def get_runtime_imports(self, runtime_type: str, language: str) -> str:
        """
        Get import statements required for a specific runtime type.

        Returns Python import statements or Lua require statements based on
        the runtime type and language.

        Args:
            runtime_type: The type of runtime (e.g., "vm_protection")
            language: Target language ("python" or "lua")

        Returns:
            Import/require statements as a string, or empty string if none needed

        Raises:
            ValueError: If language is not "python" or "lua"

        Example:
            >>> imports = manager.get_runtime_imports("vm_protection", "python")
            >>> print(imports)  # 'from obf_runtime import BytecodeVM, execute_protected_function'
        """
        if language not in ("python", "lua"):
            raise ValueError(f"Invalid language: {language}. Expected 'python' or 'lua'")

        if language == "python":
            # Python import statements
            import_map = {
                "vm_protection": "# VM Protection Runtime\nfrom obf_runtime import BytecodeVM, execute_protected_function",
                "code_splitting": "# Code Splitting Runtime\nfrom obf_runtime import _decrypt_chunk, _reassemble_function",
                "self_modifying_code": "# Self-Modifying Code Runtime\nfrom obf_runtime import _redefine_function, _generate_code_at_runtime, _modify_function_body, _self_modify_wrapper",
                "anti_debugging": "# Anti-Debugging Runtime\nfrom obf_runtime import _check_env_0x1a2b",
            }
            return import_map.get(runtime_type, "")

        else:  # lua
            # Lua require statements - bind result to local variable for accessing exported functions
            # The runtime modules return a table with exported functions
            import_map = {
                "vm_protection": "-- VM Protection Runtime\nlocal rt = require(\"obf_runtime\")",
                "code_splitting": "-- Code Splitting Runtime\nlocal rt = require(\"obf_runtime\")",
                "self_modifying_code": "-- Self-Modifying Code Runtime\nlocal rt = require(\"obf_runtime\")",
                "anti_debugging": "-- Anti-Debugging Runtime\nlocal rt = require(\"obf_runtime\")",
                "roblox_exploit_defense": "-- Roblox Exploit Defense Runtime\nlocal rt = require(\"obf_runtime\")",
                "roblox_remote_spy": "-- Roblox Remote Spy Protection Runtime\nlocal rt = require(\"obf_runtime\")",
            }
            return import_map.get(runtime_type, "")

    def generate_all_runtimes(self, language: str) -> Dict[str, str]:
        """
        Generate runtime code for all required features.

        Collects the list of required runtime types and generates code
        for each one, returning a mapping of runtime types to their generated code.

        Args:
            language: Target language ("python" or "lua")

        Returns:
            Dictionary mapping runtime types to generated code strings

        Raises:
            ValueError: If language is not "python" or "lua"

        Example:
            >>> all_runtimes = manager.generate_all_runtimes("lua")
            >>> for runtime_type, code in all_runtimes.items():
            ...     print(f"{runtime_type}: {len(code)} chars")
        """
        if language not in ("python", "lua"):
            raise ValueError(f"Invalid language: {language}. Expected 'python' or 'lua'")

        required_runtimes = self.collect_required_runtimes(language)
        runtime_codes: Dict[str, str] = {}

        for runtime_type in required_runtimes:
            code = self.generate_runtime_code(runtime_type, language)
            if code:  # Only store non-empty results
                runtime_codes[runtime_type] = code

        logger.info(f"Generated {len(runtime_codes)} runtimes for {language}")
        return runtime_codes

    def has_runtime_requirements(self, language: str) -> bool:
        """
        Check if any features requiring runtime code are enabled.

        Args:
            language: Target language ("python" or "lua")

        Returns:
            True if any runtime-generating features are enabled, False otherwise

        Raises:
            ValueError: If language is not "python" or "lua"

        Example:
            >>> if manager.has_runtime_requirements("lua"):
            ...     print("Runtime code is needed")
        """
        required = self.collect_required_runtimes(language)
        return len(required) > 0

    def get_combined_runtime(self, language: str) -> str:
        """
        Generate and combine all required runtime code into a single string.

        Generates runtime code for all required features and combines them
        with section separators and comments for clarity.

        Args:
            language: Target language ("python" or "lua")

        Returns:
            Combined runtime code as a single string

        Raises:
            ValueError: If language is not "python" or "lua"

        Example:
            >>> combined = manager.get_combined_runtime("python")
            >>> print(len(combined))  # Total size of all runtime code
        """
        if language not in ("python", "lua"):
            raise ValueError(f"Invalid language: {language}. Expected 'python' or 'lua'")

        all_runtimes = self.generate_all_runtimes(language)

        if not all_runtimes:
            return ""

        parts: List[str] = []

        # Add header comment
        if language == "python":
            parts.append('# Obfuscation Runtime Code - Combined')
            parts.append('# Generated by ScriptShield RuntimeManager')
            parts.append('')
        else:  # lua
            parts.append('-- Obfuscation Runtime Code - Combined')
            parts.append('-- Generated by ScriptShield RuntimeManager')
            parts.append('')

        # Combine all runtime code with separators
        for i, (runtime_type, code) in enumerate(all_runtimes.items()):
            if i > 0:
                # Add separator between runtimes
                if language == "python":
                    parts.append(f'\n# {"=" * 60}')
                    parts.append(f'# Runtime: {runtime_type}')
                    parts.append(f'# {"=" * 60}\n')
                else:  # lua
                    parts.append(f'\n-- {"=" * 60}')
                    parts.append(f'-- Runtime: {runtime_type}')
                    parts.append(f'-- {"=" * 60}\n')

            parts.append(code)

        return '\n'.join(parts)

    def get_runtime_function_names(self, runtime_type: str) -> List[str]:
        """
        Get the list of function names exported by a runtime.

        This is useful for generating import statements and understanding
        what functions are available from each runtime module.

        Args:
            runtime_type: The type of runtime (e.g., "vm_protection")

        Returns:
            List of function names exported by the runtime

        Example:
            >>> funcs = manager.get_runtime_function_names("vm_protection")
            >>> print(funcs)  # ['BytecodeVM', 'execute_protected_function']
        """
        function_map: Dict[str, List[str]] = {
            "vm_protection": ["BytecodeVM", "execute_protected_function"],
            "code_splitting": ["_decrypt_chunk", "_reassemble_function"],
            "self_modifying_code": [
                "_redefine_function",
                "_generate_code_at_runtime",
                "_modify_function_body",
                "_self_modify_wrapper",
            ],
            "anti_debugging": ["_check_env_0x1a2b"],
            "roblox_exploit_defense": [
                "_roblox_check_0x1f3a",
                "_exploit_detect_0x2b4c",
                "_integrity_verify_0x3d5e",
                "_env_fingerprint_0x4f6a",
            ],
            "roblox_remote_spy": [
                "_decrypt_remote_name_0x7b8c",
                "_resolve_remote_0x6a5f",
            ],
        }

        return function_map.get(runtime_type, [])

    def get_runtime_key(self, runtime_type: str) -> Optional[bytes]:
        """
        Get the encryption key for a specific runtime type.

        This method allows transformers to retrieve the same encryption key
        that was used during runtime generation, ensuring consistency for
        encryption/decryption operations.

        Args:
            runtime_type: The type of runtime (e.g., "code_splitting", "roblox_remote_spy")

        Returns:
            The encryption key bytes if one has been generated, None otherwise

        Example:
            >>> manager = RuntimeManager(config)
            >>> # First, generate runtime code which creates the key
            >>> code = manager.generate_runtime_code("code_splitting", "lua")
            >>> # Then retrieve the key for transformer use
            >>> key = manager.get_runtime_key("code_splitting")
            >>> print(f"Key length: {len(key)} bytes")
        """
        return self._runtime_keys.get(runtime_type)

    def set_runtime_key(self, runtime_type: str, key: bytes) -> None:
        """
        Set or override the encryption key for a specific runtime type.

        This method allows transformers or external code to provide a
        specific encryption key instead of having one auto-generated.

        Args:
            runtime_type: The type of runtime (e.g., "code_splitting", "roblox_remote_spy")
            key: The encryption key bytes to use

        Raises:
            ValueError: If key is not bytes or is empty

        Example:
            >>> manager = RuntimeManager(config)
            >>> import secrets
            >>> key = secrets.token_bytes(16)
            >>> manager.set_runtime_key("code_splitting", key)
        """
        if not isinstance(key, bytes):
            raise ValueError(f"Key must be bytes, got {type(key).__name__}")
        if len(key) == 0:
            raise ValueError("Key cannot be empty")
        self._runtime_keys[runtime_type] = key
        logger.debug(f"Set encryption key for {runtime_type} ({len(key)} bytes)")

    def get_all_runtime_keys(self) -> Dict[str, bytes]:
        """
        Get all stored encryption keys.

        Returns a copy of the runtime keys dictionary, useful for
        serialization or debugging purposes.

        Returns:
            Dictionary mapping runtime types to their encryption keys

        Example:
            >>> manager = RuntimeManager(config)
            >>> # Generate some runtimes with keys
            >>> code = manager.generate_runtime_code("code_splitting", "lua")
            >>> # Get all keys
            >>> keys = manager.get_all_runtime_keys()
            >>> for runtime, key in keys.items():
            ...     print(f"{runtime}: {len(key)} bytes")
        """
        return self._runtime_keys.copy()
