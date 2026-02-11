"""Unit tests for the RuntimeManager class.

Tests cover all public methods including runtime collection, code generation,
encryption key management, and utility functions for both Python and Lua.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from typing import Any

from obfuscator.core.runtime_manager import RuntimeManager
from obfuscator.core.config import ObfuscationConfig


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_config() -> ObfuscationConfig:
    """Create an ObfuscationConfig with various runtime features enabled."""
    return ObfuscationConfig(
        name="Test Profile",
        version="1.0",
        language="python",
        preset=None,
        features={
            "vm_protection": True,
            "code_splitting": True,
            "self_modifying_code": True,
            "anti_debugging": True,
            "mangle_globals": False,
            "string_encryption": False,
        },
        options={
            "vm_bytecode_encryption": True,
            "code_split_encryption": True,
            "self_modify_complexity": 2,
            "anti_debug_aggressiveness": 2,
            "roblox_exploit_action": "exit",
            "string_encryption_key_length": 16,
            "array_shuffle_seed": None,
            "dead_code_percentage": 20,
            "identifier_prefix": "_0x",
            "number_obfuscation_complexity": 3,
            "number_obfuscation_min_value": 10,
            "number_obfuscation_max_value": 1000000,
        },
        symbol_table_options={
            "identifier_prefix": "_0x",
            "mangling_strategy": "sequential",
            "preserve_exports": False,
            "preserve_constants": False,
        },
        runtime_mode="hybrid",
    )


@pytest.fixture
def runtime_manager(sample_config: ObfuscationConfig) -> RuntimeManager:
    """Create a RuntimeManager instance with the sample config."""
    return RuntimeManager(sample_config)


@pytest.fixture
def lua_config() -> ObfuscationConfig:
    """Create an ObfuscationConfig for Lua-specific testing with Roblox features."""
    return ObfuscationConfig(
        name="Lua Test Profile",
        version="1.0",
        language="lua",
        preset=None,
        features={
            "vm_protection": True,
            "code_splitting": True,
            "anti_debugging": True,
            "roblox_exploit_defense": True,
            "roblox_remote_spy": True,
            "self_modifying_code": True,
            "mangle_globals": False,
        },
        options={
            "vm_bytecode_encryption": True,
            "code_split_encryption": True,
            "anti_debug_aggressiveness": 2,
            "roblox_exploit_aggressiveness": 2,
            "roblox_exploit_action": "exit",
            "string_encryption_key_length": 16,
            "array_shuffle_seed": None,
            "dead_code_percentage": 20,
            "identifier_prefix": "_0x",
            "number_obfuscation_complexity": 3,
            "number_obfuscation_min_value": 10,
            "number_obfuscation_max_value": 1000000,
            "vm_protection_complexity": 2,
            "vm_protect_all_functions": False,
            "vm_protection_marker": "vm_protect",
            "opaque_predicate_complexity": 2,
            "opaque_predicate_percentage": 30,
            "code_split_chunk_size": 5,
            "self_modify_complexity": 2,
        },
        symbol_table_options={
            "identifier_prefix": "_0x",
            "mangling_strategy": "sequential",
            "preserve_exports": False,
            "preserve_constants": False,
        },
        runtime_mode="hybrid",
    )


# -----------------------------------------------------------------------------
# Test RuntimeManager Basics
# -----------------------------------------------------------------------------

class TestRuntimeManagerBasics:
    """Test basic RuntimeManager initialization and validation."""

    def test_initialization(self, sample_config: ObfuscationConfig):
        """Verify RuntimeManager initializes with config and empty runtime_keys dict."""
        manager = RuntimeManager(sample_config)
        assert manager.config is sample_config
        assert manager._runtime_keys == {}
        assert isinstance(manager._runtime_keys, dict)

    def test_invalid_language_raises_error(self, runtime_manager: RuntimeManager):
        """Verify ValueError raised for invalid language in all methods."""
        invalid_language = "invalid_lang"
        
        # Test collect_required_runtimes
        with pytest.raises(ValueError, match=f"Invalid language: {invalid_language}"):
            runtime_manager.collect_required_runtimes(invalid_language)
        
        # Test generate_runtime_code
        with pytest.raises(ValueError, match=f"Invalid language: {invalid_language}"):
            runtime_manager.generate_runtime_code("vm_protection", invalid_language)
        
        # Test get_runtime_imports
        with pytest.raises(ValueError, match=f"Invalid language: {invalid_language}"):
            runtime_manager.get_runtime_imports("vm_protection", invalid_language)
        
        # Test generate_all_runtimes
        with pytest.raises(ValueError, match=f"Invalid language: {invalid_language}"):
            runtime_manager.generate_all_runtimes(invalid_language)
        
        # Test has_runtime_requirements
        with pytest.raises(ValueError, match=f"Invalid language: {invalid_language}"):
            runtime_manager.has_runtime_requirements(invalid_language)
        
        # Test get_combined_runtime
        with pytest.raises(ValueError, match=f"Invalid language: {invalid_language}"):
            runtime_manager.get_combined_runtime(invalid_language)


# -----------------------------------------------------------------------------
# Test Collect Required Runtimes
# -----------------------------------------------------------------------------

class TestCollectRequiredRuntimes:
    """Test runtime requirement collection for different features."""

    def test_collect_python_runtimes(self, runtime_manager: RuntimeManager):
        """Enable vm_protection and code_splitting, verify both collected."""
        required = runtime_manager.collect_required_runtimes("python")
        assert "vm_protection" in required
        assert "code_splitting" in required
        assert "self_modifying_code" in required
        assert "anti_debugging" in required

    def test_collect_lua_runtimes(self, lua_config: ObfuscationConfig):
        """Enable vm_protection, anti_debugging, roblox_exploit_defense, verify all collected."""
        manager = RuntimeManager(lua_config)
        required = manager.collect_required_runtimes("lua")
        assert "vm_protection" in required
        assert "anti_debugging" in required
        assert "roblox_exploit_defense" in required
        assert "roblox_remote_spy" in required
        assert "code_splitting" in required
        assert "self_modifying_code" in required

    def test_collect_empty_when_no_features(self):
        """Config with all features disabled returns empty list."""
        config = ObfuscationConfig(
            name="Empty Config",
            version="1.0",
            language="python",
            features={
                "vm_protection": False,
                "code_splitting": False,
                "self_modifying_code": False,
                "anti_debugging": False,
                "mangle_globals": False,
            },
        )
        manager = RuntimeManager(config)
        required = manager.collect_required_runtimes("python")
        assert required == []

    def test_collect_only_features_with_generators(self):
        """Enable feature without generator (e.g., mangle_globals), verify not collected."""
        config = ObfuscationConfig(
            name="Partial Config",
            version="1.0",
            language="python",
            features={
                "vm_protection": True,
                "mangle_globals": True,  # No runtime generator
                "string_encryption": True,  # No runtime generator
            },
        )
        manager = RuntimeManager(config)
        required = manager.collect_required_runtimes("python")
        assert "vm_protection" in required
        assert "mangle_globals" not in required
        assert "string_encryption" not in required

    def test_has_runtime_requirements_true(self, runtime_manager: RuntimeManager):
        """Verify returns True when features enabled."""
        assert runtime_manager.has_runtime_requirements("python") is True
        assert runtime_manager.has_runtime_requirements("lua") is True

    def test_has_runtime_requirements_false(self):
        """Verify returns False when no runtime features enabled."""
        config = ObfuscationConfig(
            name="No Runtime Config",
            version="1.0",
            language="python",
            features={
                "mangle_globals": True,
                "string_encryption": True,
            },
        )
        manager = RuntimeManager(config)
        assert manager.has_runtime_requirements("python") is False


# -----------------------------------------------------------------------------
# Test Generate Runtime Code
# -----------------------------------------------------------------------------

class TestGenerateRuntimeCode:
    """Test runtime code generation for various types and languages."""

    def test_generate_vm_protection_python(self, runtime_manager: RuntimeManager):
        """Verify generates non-empty code, contains expected VM functions."""
        code = runtime_manager.generate_runtime_code("vm_protection", "python")
        assert len(code) > 0
        assert "BytecodeVM" in code or "execute" in code.lower()

    def test_generate_vm_protection_lua(self, lua_config: ObfuscationConfig):
        """Verify generates Lua VM runtime code."""
        manager = RuntimeManager(lua_config)
        code = manager.generate_runtime_code("vm_protection", "lua")
        assert len(code) > 0
        assert "function" in code

    def test_generate_code_splitting_python(self, runtime_manager: RuntimeManager):
        """Verify generates decryption runtime, encryption key created."""
        code = runtime_manager.generate_runtime_code("code_splitting", "python")
        assert len(code) > 0
        # Key should be generated and stored
        assert "code_splitting" in runtime_manager._runtime_keys
        assert len(runtime_manager._runtime_keys["code_splitting"]) == 16

    def test_generate_code_splitting_lua(self, lua_config: ObfuscationConfig):
        """Verify Lua code splitting runtime."""
        manager = RuntimeManager(lua_config)
        code = manager.generate_runtime_code("code_splitting", "lua")
        assert len(code) > 0
        assert "code_splitting" in manager._runtime_keys

    def test_generate_self_modifying_python(self, runtime_manager: RuntimeManager):
        """Verify self-modify runtime with complexity parameter."""
        code = runtime_manager.generate_runtime_code("self_modifying_code", "python")
        assert len(code) > 0
        # Complexity of 2 should produce valid code
        assert "function" in code.lower() or "def" in code

    def test_generate_anti_debugging_python(self, runtime_manager: RuntimeManager):
        """Verify anti-debug runtime with aggressiveness parameter."""
        code = runtime_manager.generate_runtime_code("anti_debugging", "python")
        assert len(code) > 0
        assert "check" in code.lower() or "debug" in code.lower()

    def test_generate_roblox_exploit_defense_lua(self, lua_config: ObfuscationConfig):
        """Verify Roblox exploit defense runtime."""
        manager = RuntimeManager(lua_config)
        code = manager.generate_runtime_code("roblox_exploit_defense", "lua")
        assert len(code) > 0
        assert "function" in code

    def test_generate_roblox_remote_spy_lua(self, lua_config: ObfuscationConfig):
        """Verify remote spy runtime with encryption key."""
        manager = RuntimeManager(lua_config)
        code = manager.generate_runtime_code("roblox_remote_spy", "lua")
        assert len(code) > 0
        # Key should be generated
        assert "roblox_remote_spy" in manager._runtime_keys
        assert len(manager._runtime_keys["roblox_remote_spy"]) == 16

    def test_generate_unknown_runtime_returns_empty(self, runtime_manager: RuntimeManager):
        """Verify unknown runtime type returns empty string."""
        code = runtime_manager.generate_runtime_code("unknown_runtime", "python")
        assert code == ""

    def test_runtime_params_extracted_correctly(self):
        """Verify config options passed to generators (vm_bytecode_encryption, code_split_encryption, etc.)."""
        config = ObfuscationConfig(
            name="Param Test",
            version="1.0",
            language="python",
            features={
                "vm_protection": True,
                "code_splitting": True,
                "self_modifying_code": True,
                "anti_debugging": True,
            },
            options={
                "vm_bytecode_encryption": False,  # Disabled
                "code_split_encryption": False,  # Disabled
                "self_modify_complexity": 3,  # Custom
                "anti_debug_aggressiveness": 1,  # Custom
            },
        )
        manager = RuntimeManager(config)
        
        # Generate runtimes - should not raise errors with custom params
        vm_code = manager.generate_runtime_code("vm_protection", "python")
        cs_code = manager.generate_runtime_code("code_splitting", "python")
        sm_code = manager.generate_runtime_code("self_modifying_code", "python")
        ad_code = manager.generate_runtime_code("anti_debugging", "python")
        
        assert len(vm_code) > 0
        assert len(cs_code) > 0
        assert len(sm_code) > 0
        assert len(ad_code) > 0


# -----------------------------------------------------------------------------
# Test Get Runtime Imports
# -----------------------------------------------------------------------------

class TestGetRuntimeImports:
    """Test import statement generation for different runtimes."""

    def test_get_python_vm_imports(self, runtime_manager: RuntimeManager):
        """Verify returns correct Python import statement for VM."""
        imports = runtime_manager.get_runtime_imports("vm_protection", "python")
        assert "from obf_runtime import" in imports
        assert "BytecodeVM" in imports

    def test_get_python_code_splitting_imports(self, runtime_manager: RuntimeManager):
        """Verify import for code splitting functions."""
        imports = runtime_manager.get_runtime_imports("code_splitting", "python")
        assert "from obf_runtime import" in imports
        assert "_decrypt_chunk" in imports

    def test_get_lua_vm_imports(self, runtime_manager: RuntimeManager):
        """Verify Lua require statement."""
        imports = runtime_manager.get_runtime_imports("vm_protection", "lua")
        assert 'require("obf_runtime")' in imports

    def test_get_lua_roblox_imports(self, lua_config: ObfuscationConfig):
        """Verify Roblox-specific imports."""
        manager = RuntimeManager(lua_config)
        imports = manager.get_runtime_imports("roblox_exploit_defense", "lua")
        assert 'require("obf_runtime")' in imports
        imports2 = manager.get_runtime_imports("roblox_remote_spy", "lua")
        assert 'require("obf_runtime")' in imports2

    def test_get_unknown_runtime_imports_empty(self, runtime_manager: RuntimeManager):
        """Verify unknown runtime returns empty string."""
        imports = runtime_manager.get_runtime_imports("unknown_runtime", "python")
        assert imports == ""
        imports_lua = runtime_manager.get_runtime_imports("unknown_runtime", "lua")
        assert imports_lua == ""


# -----------------------------------------------------------------------------
# Test Encryption Key Management
# -----------------------------------------------------------------------------

class TestEncryptionKeyManagement:
    """Test encryption key generation, retrieval, and management."""

    def test_encryption_key_generated_for_code_splitting(self, runtime_manager: RuntimeManager):
        """Generate code_splitting runtime, verify key created."""
        # Key should not exist yet
        assert runtime_manager.get_runtime_key("code_splitting") is None
        
        # Generate runtime code
        code = runtime_manager.generate_runtime_code("code_splitting", "python")
        assert len(code) > 0
        
        # Key should now exist
        key = runtime_manager.get_runtime_key("code_splitting")
        assert key is not None
        assert isinstance(key, bytes)
        assert len(key) == 16

    def test_get_runtime_key(self, runtime_manager: RuntimeManager):
        """Verify get_runtime_key returns correct key."""
        # Generate code to create key
        runtime_manager.generate_runtime_code("code_splitting", "python")
        
        key = runtime_manager.get_runtime_key("code_splitting")
        assert key is not None
        assert isinstance(key, bytes)

    def test_set_runtime_key(self, runtime_manager: RuntimeManager):
        """Verify set_runtime_key stores key correctly."""
        custom_key = b"custom_key_12345"
        runtime_manager.set_runtime_key("custom_runtime", custom_key)
        
        retrieved_key = runtime_manager.get_runtime_key("custom_runtime")
        assert retrieved_key == custom_key

    def test_set_runtime_key_validation(self, runtime_manager: RuntimeManager):
        """Verify ValueError for non-bytes or empty key."""
        # Non-bytes key
        with pytest.raises(ValueError, match="Key must be bytes"):
            runtime_manager.set_runtime_key("test", "not_bytes")
        
        with pytest.raises(ValueError, match="Key must be bytes"):
            runtime_manager.set_runtime_key("test", 12345)
        
        # Empty key
        with pytest.raises(ValueError, match="Key cannot be empty"):
            runtime_manager.set_runtime_key("test", b"")

    def test_get_all_runtime_keys(self, runtime_manager: RuntimeManager):
        """Generate multiple runtimes, verify all keys returned."""
        # Generate runtimes that create keys
        runtime_manager.generate_runtime_code("code_splitting", "python")
        
        lua_config = ObfuscationConfig(
            name="Lua Keys Test",
            version="1.0",
            language="lua",
            features={
                "roblox_remote_spy": True,
            },
        )
        lua_manager = RuntimeManager(lua_config)
        lua_manager.generate_runtime_code("roblox_remote_spy", "lua")
        
        # Get all keys
        all_keys = runtime_manager.get_all_runtime_keys()
        assert "code_splitting" in all_keys
        assert isinstance(all_keys["code_splitting"], bytes)
        
        lua_keys = lua_manager.get_all_runtime_keys()
        assert "roblox_remote_spy" in lua_keys

    def test_keys_consistent_across_calls(self, runtime_manager: RuntimeManager):
        """Generate same runtime twice, verify same key used."""
        # First generation
        code1 = runtime_manager.generate_runtime_code("code_splitting", "python")
        key1 = runtime_manager.get_runtime_key("code_splitting")
        
        # Second generation should use same key
        code2 = runtime_manager.generate_runtime_code("code_splitting", "python")
        key2 = runtime_manager.get_runtime_key("code_splitting")
        
        assert key1 == key2
        assert code1 == code2  # Code should be identical with same key


# -----------------------------------------------------------------------------
# Test Generate All Runtimes
# -----------------------------------------------------------------------------

class TestGenerateAllRuntimes:
    """Test batch generation of all required runtimes."""

    def test_generate_all_python_runtimes(self, runtime_manager: RuntimeManager):
        """Enable multiple features, verify all generated."""
        all_runtimes = runtime_manager.generate_all_runtimes("python")
        
        assert "vm_protection" in all_runtimes
        assert "code_splitting" in all_runtimes
        assert "self_modifying_code" in all_runtimes
        assert "anti_debugging" in all_runtimes
        
        for name, code in all_runtimes.items():
            assert len(code) > 0, f"Runtime {name} generated empty code"

    def test_generate_all_lua_runtimes(self, lua_config: ObfuscationConfig):
        """Enable Lua features, verify dict with all runtime codes."""
        manager = RuntimeManager(lua_config)
        all_runtimes = manager.generate_all_runtimes("lua")
        
        assert "vm_protection" in all_runtimes
        assert "code_splitting" in all_runtimes
        assert "anti_debugging" in all_runtimes
        assert "roblox_exploit_defense" in all_runtimes
        assert "roblox_remote_spy" in all_runtimes
        assert "self_modifying_code" in all_runtimes

    def test_generate_all_empty_when_no_features(self):
        """Verify empty dict when no features enabled."""
        config = ObfuscationConfig(
            name="No Features",
            version="1.0",
            language="python",
            features={},
        )
        manager = RuntimeManager(config)
        all_runtimes = manager.generate_all_runtimes("python")
        assert all_runtimes == {}


# -----------------------------------------------------------------------------
# Test Get Combined Runtime
# -----------------------------------------------------------------------------

class TestGetCombinedRuntime:
    """Test combining multiple runtimes into a single string."""

    def test_combined_runtime_python(self, runtime_manager: RuntimeManager):
        """Verify combined runtime has header, separators, all runtime code."""
        combined = runtime_manager.get_combined_runtime("python")
        
        assert len(combined) > 0
        # Check for header
        assert "Obfuscation Runtime Code - Combined" in combined
        assert "ScriptShield RuntimeManager" in combined
        # Check for separators between runtimes
        assert "=" * 30 in combined or "Runtime:" in combined

    def test_combined_runtime_lua(self, lua_config: ObfuscationConfig):
        """Verify Lua combined runtime format."""
        manager = RuntimeManager(lua_config)
        combined = manager.get_combined_runtime("lua")
        
        assert len(combined) > 0
        assert "-- Obfuscation Runtime Code - Combined" in combined
        assert "--" in combined  # Lua comments

    def test_combined_runtime_empty_when_no_features(self):
        """Verify empty string when no features."""
        config = ObfuscationConfig(
            name="No Features",
            version="1.0",
            language="python",
            features={},
        )
        manager = RuntimeManager(config)
        combined = manager.get_combined_runtime("python")
        assert combined == ""


# -----------------------------------------------------------------------------
# Test Get Runtime Function Names
# -----------------------------------------------------------------------------

class TestGetRuntimeFunctionNames:
    """Test retrieval of function names exported by runtimes."""

    def test_get_vm_protection_function_names(self, runtime_manager: RuntimeManager):
        """Verify returns ["BytecodeVM", "execute_protected_function"]."""
        names = runtime_manager.get_runtime_function_names("vm_protection")
        assert "BytecodeVM" in names
        assert "execute_protected_function" in names

    def test_get_code_splitting_function_names(self, runtime_manager: RuntimeManager):
        """Verify returns decryption function names."""
        names = runtime_manager.get_runtime_function_names("code_splitting")
        assert "_decrypt_chunk" in names
        assert "_reassemble_function" in names

    def test_get_unknown_runtime_function_names(self, runtime_manager: RuntimeManager):
        """Verify returns empty list for unknown runtime."""
        names = runtime_manager.get_runtime_function_names("unknown_runtime")
        assert names == []

    def test_get_all_function_names(self, runtime_manager: RuntimeManager):
        """Verify all known runtimes have function names defined."""
        known_runtimes = [
            "vm_protection",
            "code_splitting",
            "self_modifying_code",
            "anti_debugging",
            "roblox_exploit_defense",
            "roblox_remote_spy",
        ]
        
        for runtime in known_runtimes:
            names = runtime_manager.get_runtime_function_names(runtime)
            assert isinstance(names, list)
            # All should have at least one function name
            assert len(names) > 0, f"Runtime {runtime} should have function names"
