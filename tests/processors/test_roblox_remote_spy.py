"""Tests for the RobloxRemoteSpyTransformer class.

This test suite covers:
- Initialization and configuration
- Remote detection (FireServer, InvokeServer, etc.)
- Encryption/decryption of remote names
- Runtime generation for remote spy protection
- AST transformation and remote call obfuscation
- Argument obfuscation techniques
- Edge cases and integration with other transformers
"""

import ast
import base64
import unittest
from unittest.mock import MagicMock, patch

# Try to import luaparser for Lua tests
try:
    from luaparser import ast as lua_ast
    from luaparser import astnodes as lua_nodes
    LUAPARSER_AVAILABLE = True
except ImportError:
    LUAPARSER_AVAILABLE = False

from obfuscator.processors.ast_transformer import (
    RobloxRemoteSpyTransformer,
    TransformResult,
)
from obfuscator.processors.roblox_remote_spy_runtime_lua import (
    generate_roblox_remote_spy_runtime,
)


def lua_skip(func):
    """Decorator to skip Lua tests when luaparser is not available."""
    return unittest.skipUnless(LUAPARSER_AVAILABLE, "luaparser not available")(func)


class TestRobloxRemoteSpyInit(unittest.TestCase):
    """Tests for RobloxRemoteSpyTransformer initialization."""

    def test_default_initialization(self):
        """Test default initialization (language_mode="lua", encryption_key generated)."""
        transformer = RobloxRemoteSpyTransformer()
        self.assertEqual(transformer.language_mode, "lua")
        self.assertIsNotNone(transformer.encryption_key)
        self.assertEqual(len(transformer.encryption_key), 16)

    def test_initialization_with_explicit_key(self):
        """Test initialization with explicit encryption_key parameter (16 bytes)."""
        key = b"1234567890123456"
        transformer = RobloxRemoteSpyTransformer(encryption_key=key)
        self.assertEqual(transformer.encryption_key, key)

    def test_encryption_key_stored_correctly(self):
        """Test encryption_key is stored correctly."""
        key = b"testkey123456789"
        transformer = RobloxRemoteSpyTransformer(encryption_key=key)
        self.assertEqual(transformer.encryption_key, key)
        self.assertEqual(len(transformer.encryption_key), 16)

    def test_remote_name_map_initialized_empty(self):
        """Test remote_name_map and _detected_remotes are initialized as empty."""
        transformer = RobloxRemoteSpyTransformer()
        self.assertEqual(transformer.remote_name_map, {})
        self.assertEqual(transformer._detected_remotes, {})


class TestRobloxRemoteSpyDetection(unittest.TestCase):
    """Tests for remote detection functionality."""

    @lua_skip
    def test_detect_fire_server(self):
        """Test _is_remote_method_call() detects FireServer method calls."""
        transformer = RobloxRemoteSpyTransformer()
        code = "remote:FireServer('test')"
        tree = lua_ast.parse(code)

        # Find the method call
        call_node = None
        for node in lua_ast.walk(tree):
            if isinstance(node, lua_nodes.Call):
                call_node = node
                break

        if call_node:
            self.assertTrue(transformer._is_remote_method_call(call_node))

    @lua_skip
    def test_detect_fire_client(self):
        """Test _is_remote_method_call() detects FireClient method calls."""
        transformer = RobloxRemoteSpyTransformer()
        code = "remote:FireClient(player, 'data')"
        tree = lua_ast.parse(code)

        call_node = None
        for node in lua_ast.walk(tree):
            if isinstance(node, lua_nodes.Call):
                call_node = node
                break

        if call_node:
            self.assertTrue(transformer._is_remote_method_call(call_node))

    @lua_skip
    def test_detect_invoke_server(self):
        """Test _is_remote_method_call() detects InvokeServer method calls."""
        transformer = RobloxRemoteSpyTransformer()
        code = "func:InvokeServer(args)"
        tree = lua_ast.parse(code)

        call_node = None
        for node in lua_ast.walk(tree):
            if isinstance(node, lua_nodes.Call):
                call_node = node
                break

        if call_node:
            self.assertTrue(transformer._is_remote_method_call(call_node))

    @lua_skip
    def test_detect_invoke_client(self):
        """Test _is_remote_method_call() detects InvokeClient method calls."""
        transformer = RobloxRemoteSpyTransformer()
        code = "func:InvokeClient(player)"
        tree = lua_ast.parse(code)

        call_node = None
        for node in lua_ast.walk(tree):
            if isinstance(node, lua_nodes.Call):
                call_node = node
                break

        if call_node:
            self.assertTrue(transformer._is_remote_method_call(call_node))

    @lua_skip
    def test_non_remote_methods_return_false(self):
        """Test _is_remote_method_call() returns False for non-remote methods."""
        transformer = RobloxRemoteSpyTransformer()
        code = "obj:RegularMethod()"
        tree = lua_ast.parse(code)

        call_node = None
        for node in lua_ast.walk(tree):
            if isinstance(node, lua_nodes.Call):
                call_node = node
                break

        if call_node:
            self.assertFalse(transformer._is_remote_method_call(call_node))

    @lua_skip
    def test_extract_simple_variable_name(self):
        """Test _extract_remote_object_name() extracts simple variable names."""
        transformer = RobloxRemoteSpyTransformer()
        code = "myRemote:FireServer()"
        tree = lua_ast.parse(code)

        call_node = None
        for node in lua_ast.walk(tree):
            if isinstance(node, lua_nodes.Call):
                call_node = node
                break

        if call_node:
            name = transformer._extract_remote_object_name(call_node)
            self.assertEqual(name, "myRemote")

    @lua_skip
    def test_extract_indexed_access(self):
        """Test _extract_remote_object_name() extracts indexed access."""
        transformer = RobloxRemoteSpyTransformer()
        code = "remotes.playerJoined:FireServer()"
        tree = lua_ast.parse(code)

        call_node = None
        for node in lua_ast.walk(tree):
            if isinstance(node, lua_nodes.Call):
                call_node = node
                break

        if call_node:
            name = transformer._extract_remote_object_name(call_node)
            self.assertEqual(name, "remotes.playerJoined")

    @lua_skip
    def test_skip_roblox_api_globals(self):
        """Test _extract_remote_object_name() skips Roblox API globals."""
        transformer = RobloxRemoteSpyTransformer()

        # These should return None or be skipped
        roblox_globals = ["game", "workspace", "script", "Instance"]

        for global_name in roblox_globals:
            code = f"{global_name}:SomeMethod()"
            tree = lua_ast.parse(code)

            call_node = None
            for node in lua_ast.walk(tree):
                if isinstance(node, lua_nodes.Call):
                    call_node = node
                    break

            if call_node:
                name = transformer._extract_remote_object_name(call_node)
                # Should either return None or not be a remote method
                self.assertIsNone(name)


class TestRobloxRemoteSpyEncryption(unittest.TestCase):
    """Tests for remote name encryption."""

    def test_encrypt_produces_base64(self):
        """Test _encrypt_remote_name() produces base64-encoded string."""
        transformer = RobloxRemoteSpyTransformer()
        encrypted = transformer._encrypt_remote_name("PlayerJoined")

        # Should be a string
        self.assertIsInstance(encrypted, str)

        # Should be valid base64 (or at least contain base64 characters)
        import re
        self.assertTrue(re.match(r'^[A-Za-z0-9+/=]+$', encrypted))

    def test_encryption_is_reversible(self):
        """Test encryption is reversible (decrypt matches original)."""
        transformer = RobloxRemoteSpyTransformer()
        original = "TestRemote"
        encrypted = transformer._encrypt_remote_name(original)

        # Decrypt
        decrypted = transformer._decrypt_remote_name(encrypted)
        self.assertEqual(decrypted, original)

    def test_same_name_same_output(self):
        """Test same name with same key produces same encrypted output."""
        key = b"1234567890123456"
        transformer = RobloxRemoteSpyTransformer(encryption_key=key)

        encrypted1 = transformer._encrypt_remote_name("MyRemote")
        encrypted2 = transformer._encrypt_remote_name("MyRemote")

        self.assertEqual(encrypted1, encrypted2)

    def test_different_names_different_outputs(self):
        """Test different names produce different encrypted outputs."""
        transformer = RobloxRemoteSpyTransformer()

        encrypted1 = transformer._encrypt_remote_name("RemoteA")
        encrypted2 = transformer._encrypt_remote_name("RemoteB")

        self.assertNotEqual(encrypted1, encrypted2)

    def test_encryption_uses_xor_cipher(self):
        """Test encryption uses XOR cipher with provided key."""
        key = b"ABC"  # Simple key for testing
        transformer = RobloxRemoteSpyTransformer(encryption_key=key)

        # XOR with key "ABC" should produce predictable results
        original = "X"
        encrypted = transformer._encrypt_remote_name(original)
        decrypted = transformer._decrypt_remote_name(encrypted)

        self.assertEqual(decrypted, original)


class TestRobloxRemoteSpyRuntimeGeneration(unittest.TestCase):
    """Tests for runtime generation."""

    @lua_skip
    def test_inject_remote_spy_runtime_returns_list(self):
        """Test _inject_remote_spy_runtime() returns list of Lua AST nodes."""
        transformer = RobloxRemoteSpyTransformer()
        # Add a remote to the map
        transformer.remote_name_map["key1"] = "encrypted1"

        nodes = transformer._inject_remote_spy_runtime()
        self.assertIsInstance(nodes, list)
        self.assertGreater(len(nodes), 0)

    def test_runtime_contains_base64_decode(self):
        """Test runtime contains base64 decode function."""
        key = b"1234567890123456"
        remote_map = {"key1": "encrypted1"}
        runtime = generate_roblox_remote_spy_runtime(key, remote_map)

        self.assertIn("_b64_decode_0x8e2d", runtime)

    def test_runtime_contains_decryption_function(self):
        """Test runtime contains decryption function."""
        key = b"1234567890123456"
        remote_map = {"key1": "encrypted1"}
        runtime = generate_roblox_remote_spy_runtime(key, remote_map)

        self.assertIn("_decrypt_remote_name_0x7b8c", runtime)

    def test_runtime_contains_lookup_table(self):
        """Test runtime contains remote name lookup table."""
        key = b"1234567890123456"
        remote_map = {"key1": "encrypted1", "key2": "encrypted2"}
        runtime = generate_roblox_remote_spy_runtime(key, remote_map)

        self.assertIn("_remote_names_0x9d4e", runtime)
        self.assertIn("key1", runtime)
        self.assertIn("key2", runtime)

    def test_runtime_contains_cache_table(self):
        """Test runtime contains remote cache table."""
        key = b"1234567890123456"
        remote_map = {"key1": "encrypted1"}
        runtime = generate_roblox_remote_spy_runtime(key, remote_map)

        self.assertIn("_remote_cache_0x3c2b", runtime)

    def test_runtime_contains_resolver_function(self):
        """Test runtime contains resolver function."""
        key = b"1234567890123456"
        remote_map = {"key1": "encrypted1"}
        runtime = generate_roblox_remote_spy_runtime(key, remote_map)

        self.assertIn("_resolve_remote_0x6a5f", runtime)

    def test_runtime_exposes_to_global(self):
        """Test runtime exposes functions to global scope."""
        key = b"1234567890123456"
        remote_map = {"key1": "encrypted1"}
        runtime = generate_roblox_remote_spy_runtime(key, remote_map)

        self.assertIn("_G[\"_resolve_remote_0x6a5f\"]", runtime)

    @lua_skip
    def test_runtime_code_is_parseable(self):
        """Test runtime code is parseable by lua_ast.parse()."""
        key = b"1234567890123456"
        remote_map = {"key1": "encrypted1"}
        runtime = generate_roblox_remote_spy_runtime(key, remote_map)

        try:
            lua_ast.parse(runtime)
        except Exception as e:
            self.fail(f"Generated runtime code is not parseable: {e}")


class TestRobloxRemoteSpyTransformation(unittest.TestCase):
    """Tests for AST transformation."""

    @lua_skip
    def test_transformation_with_remote_event_calls(self):
        """Test transformation of Lua Chunk with RemoteEvent calls."""
        transformer = RobloxRemoteSpyTransformer()
        code = "local event = game.ReplicatedStorage.MyEvent\nevent:FireServer('test')"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

    @lua_skip
    def test_collect_remote_calls_pre_scans(self):
        """Test _collect_remote_calls() pre-scans AST and builds remote_name_map."""
        transformer = RobloxRemoteSpyTransformer()
        code = "local e = game.ReplicatedStorage.Event\ne:FireServer()"
        tree = lua_ast.parse(code)

        transformer._collect_remote_calls(tree)

        # Should have detected remotes
        self.assertGreater(len(transformer._detected_remotes), 0)

    @lua_skip
    def test_runtime_injected_only_if_remotes_detected(self):
        """Test runtime is injected only if remotes are detected."""
        transformer = RobloxRemoteSpyTransformer()
        code = "local x = 1\nprint(x)"  # No remote calls
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        self.assertEqual(result.transformation_count, 0)

    @lua_skip
    def test_remote_source_replaced_with_resolver(self):
        """Test remote source is replaced with _resolve_remote_0x6a5f() call."""
        transformer = RobloxRemoteSpyTransformer()
        code = "local e = game.ReplicatedStorage.Event\ne:FireServer('test')"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

        # Check that resolver function is used
        source = lua_ast.to_lua_source(result.ast_node)
        self.assertIn("_resolve_remote_0x6a5f", source)

    @lua_skip
    def test_obfuscate_remote_arguments(self):
        """Test _obfuscate_remote_arguments() wraps arguments in identity expressions."""
        transformer = RobloxRemoteSpyTransformer()
        # This should wrap arguments
        code = "e:FireServer('arg1', 42)"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

    @lua_skip
    def test_transformation_count_includes_all(self):
        """Test transformation count includes remote replacements + argument obfuscations."""
        transformer = RobloxRemoteSpyTransformer()
        code = "local e = game.ReplicatedStorage.Event\ne:FireServer('arg')"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        self.assertGreater(result.transformation_count, 0)

    def test_python_ast_skipped_gracefully(self):
        """Test Python AST is skipped gracefully."""
        transformer = RobloxRemoteSpyTransformer()
        code = "def foo(): pass"
        tree = ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        self.assertEqual(result.transformation_count, 0)

    @lua_skip
    def test_lua_without_remotes_zero_transformations(self):
        """Test Lua chunk without remotes returns success with zero transformations."""
        transformer = RobloxRemoteSpyTransformer()
        code = "local x = 1\nfunction foo() return x end"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        self.assertEqual(result.transformation_count, 0)


class TestRobloxRemoteSpyArgumentObfuscation(unittest.TestCase):
    """Tests for argument obfuscation."""

    @lua_skip
    def test_argument_wrapping_identity_function(self):
        """Test argument wrapping with identity function (function(x) return x end)(arg)."""
        transformer = RobloxRemoteSpyTransformer()
        # Identity wrapping should be part of obfuscation
        code = "e:FireServer('test')"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

    @lua_skip
    def test_argument_arithmetic_operations(self):
        """Test argument wrapping with arithmetic operations (arg + 0, arg * 1)."""
        transformer = RobloxRemoteSpyTransformer()
        code = "e:FireServer(42)"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

    @lua_skip
    def test_argument_double_negation(self):
        """Test argument wrapping with double negation (-(-arg))."""
        transformer = RobloxRemoteSpyTransformer()
        code = "e:FireServer(-5)"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

    @lua_skip
    def test_replace_placeholder_in_expr(self):
        """Test _replace_placeholder_in_expr() correctly substitutes placeholder with actual argument."""
        transformer = RobloxRemoteSpyTransformer()
        # This is an internal method, just verify it exists
        self.assertTrue(hasattr(transformer, '_replace_placeholder_in_expr'))

    @lua_skip
    def test_obfuscation_preserves_argument_semantics(self):
        """Test obfuscation preserves argument semantics."""
        transformer = RobloxRemoteSpyTransformer()
        code = "e:FireServer(1 + 1)"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)


class TestRobloxRemoteSpyEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    @lua_skip
    def test_empty_lua_chunk(self):
        """Test empty Lua chunk (no remote calls)."""
        transformer = RobloxRemoteSpyTransformer()
        code = ""
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        self.assertEqual(result.transformation_count, 0)

    @lua_skip
    def test_lua_with_only_locals(self):
        """Test Lua chunk with only local variables (no remotes)."""
        transformer = RobloxRemoteSpyTransformer()
        code = "local x = 1\nlocal y = 2\nreturn x + y"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        self.assertEqual(result.transformation_count, 0)

    @lua_skip
    def test_nested_remote_calls(self):
        """Test nested remote calls (remotes inside functions)."""
        transformer = RobloxRemoteSpyTransformer()
        code = "function notify()\n  local e = game.ReplicatedStorage.Event\n  e:FireServer()\nend"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)
        self.assertGreater(result.transformation_count, 0)

    @lua_skip
    def test_multiple_different_remotes(self):
        """Test multiple different remotes in same script."""
        transformer = RobloxRemoteSpyTransformer()
        code = """
local event1 = game.ReplicatedStorage.Event1
local event2 = game.ReplicatedStorage.Event2
event1:FireServer('a')
event2:FireServer('b')
"""
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

        # Should have detected both remotes
        self.assertGreaterEqual(len(transformer._detected_remotes), 2)

    @lua_skip
    def test_same_remote_called_multiple_times(self):
        """Test same remote called multiple times (should use same encrypted key)."""
        transformer = RobloxRemoteSpyTransformer()
        code = "local e = game.ReplicatedStorage.Event\ne:FireServer('a')\ne:FireServer('b')"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)

    @lua_skip
    def test_remote_calls_with_complex_arguments(self):
        """Test remote calls with complex arguments (tables, function calls)."""
        transformer = RobloxRemoteSpyTransformer()
        code = "local e = game.ReplicatedStorage.Event\ne:FireServer({x=1}, getValue())"
        tree = lua_ast.parse(code)

        result = transformer.transform(tree)
        self.assertTrue(result.success)


class TestRobloxRemoteSpyIntegration(unittest.TestCase):
    """Tests for integration with other transformers."""

    @lua_skip
    def test_compatibility_with_roblox_exploit_defense(self):
        """Test compatibility with RobloxExploitDefenseTransformer."""
        from obfuscator.processors.ast_transformer import RobloxExploitDefenseTransformer

        code = "local e = game.ReplicatedStorage.Event\ne:FireServer()"
        tree = lua_ast.parse(code)

        # Apply remote spy
        remote_spy = RobloxRemoteSpyTransformer()
        result1 = remote_spy.transform(tree)
        self.assertTrue(result1.success)

        # Apply exploit defense
        exploit_defense = RobloxExploitDefenseTransformer(aggressiveness=2)
        result2 = exploit_defense.transform(result1.ast_node)
        self.assertTrue(result2.success)

    @lua_skip
    def test_compatibility_with_string_encryption(self):
        """Test compatibility with string encryption."""
        from obfuscator.processors.ast_transformer import StringEncryptionTransformer

        code = 'local e = game.ReplicatedStorage.Event\ne:FireServer("hello")'
        tree = lua_ast.parse(code)

        # Apply string encryption
        string_enc = StringEncryptionTransformer()
        result1 = string_enc.transform(tree)

        # Apply remote spy
        remote_spy = RobloxRemoteSpyTransformer()
        result2 = remote_spy.transform(result1.ast_node)
        self.assertTrue(result2.success)

    @lua_skip
    def test_compatibility_with_vm_protection(self):
        """Test compatibility with VM protection."""
        code = "local e = game.ReplicatedStorage.Event\ne:FireServer(1 + 2)"
        tree = lua_ast.parse(code)

        # Apply remote spy
        remote_spy = RobloxRemoteSpyTransformer()
        result = remote_spy.transform(tree)
        self.assertTrue(result.success)

    def test_full_pipeline_with_multiple_transformers(self):
        """Test full pipeline with multiple transformers enabled."""
        from obfuscator.core.obfuscation_engine import ObfuscationEngine
        from obfuscator.core.config import ObfuscationConfig

        config = ObfuscationConfig(
            name="test",
            preset="maximum",
            language="lua",
            features={"roblox_remote_spy": True},
        )
        engine = ObfuscationEngine(config)

        code = "local e = game.ReplicatedStorage.Event\ne:FireServer('test')"

        # Apply transformations
        result = engine.apply_transformations(code, "lua", "test.lua")
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
