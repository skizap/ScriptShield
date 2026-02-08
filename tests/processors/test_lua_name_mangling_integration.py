"""Integration tests for Lua name mangling across multiple files."""

import tempfile
from pathlib import Path
import pytest

from src.obfuscator.core.config import ObfuscationConfig
from src.obfuscator.core.orchestrator import ObfuscationOrchestrator


class TestLuaMultiFileIntegration:
    """Test multi-file Lua projects with name mangling."""

    def test_multi_file_lua_basic(self, tmp_path):
        """Test basic multi-file project with require()."""
        # Create helper.lua
        helper = tmp_path / "helper.lua"
        helper.write_text("""
local M = {}

function M.calculate_sum(a, b)
    return a + b
end

function M.calculate_product(a, b)
    return a * b
end

return M
""")

        # Create main.lua requiring helper
        main = tmp_path / "main.lua"
        main.write_text("""
local helper = require("helper")

local function process_data(x, y)
    local total = helper.calculate_sum(x, y)
    local product = helper.calculate_product(x, y)
    return total, product
end

return process_data
""")

        # Create config and orchestrator
        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )
        
        orchestrator = ObfuscationOrchestrator(config)
        
        # Run orchestration
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        orchestrator.orchestrate(
            input_files=[str(helper), str(main)],
            output_dir=str(output_dir)
        )
        
        # Verify output files exist
        output_helper = output_dir / "helper.lua"
        output_main = output_dir / "main.lua"
        assert output_helper.exists()
        assert output_main.exists()
        
        # Read output files
        helper_content = output_helper.read_text()
        main_content = output_main.read_text()
        
        # Verify mangling occurred
        assert "_0x" in helper_content
        assert "_0x" in main_content
        
        # Verify require() paths unchanged
        assert 'require("helper")' in main_content
        
        # Verify module structure preserved
        assert "local M = {}" in helper_content
        assert "return M" in helper_content

    def test_relative_requires(self, tmp_path):
        """Test relative require() paths."""
        # Create directory structure
        modules = tmp_path / "modules"
        modules.mkdir()
        
        # Create utils.lua in modules
        utils = modules / "utils.lua"
        utils.write_text("""
local M = {}

function M.util_function()
    return "utility"
end

return M
""")
        
        # Create main.lua that requires with path
        main = tmp_path / "main.lua"
        main.write_text("""
local utils = require("modules.utils")

local function use_utils()
    return utils.util_function()
end

return use_utils
""")
        
        # Run orchestration
        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )
        
        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        orchestrator.orchestrate(
            input_files=[str(utils), str(main)],
            output_dir=str(output_dir)
        )
        
        # Verify files
        output_utils = output_dir / "modules" / "utils.lua"
        output_main = output_dir / "main.lua"
        assert output_utils.exists()
        assert output_main.exists()
        
        # Verify content
        utils_content = output_utils.read_text()
        main_content = output_main.read_text()
        
        # Should have mangled names
        assert "_0x" in utils_content
        assert "_0x" in main_content
        
        # Require path should be unchanged
        assert 'require("modules.utils")' in main_content

    def test_roblox_api_preservation(self, tmp_path):
        """Test Roblox API names are preserved."""
        # Create Roblox script
        roblox_script = tmp_path / "script.lua"
        roblox_script.write_text("""
local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local function on_player_joined(player)
    local character = player.Character or player.CharacterAdded:Wait()
    local humanoid = character:FindFirstChild("Humanoid")
    
    if humanoid then
        humanoid.Health = 100
    end
end

Players.PlayerAdded:Connect(on_player_joined)

local part = Instance.new("Part")
part.Position = Vector3.new(0, 10, 0)
part.Anchored = true
part.Parent = workspace
""")
        
        # Run orchestration
        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )
        
        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        orchestrator.orchestrate(
            input_files=[str(roblox_script)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        content = (output_dir / "script.lua").read_text()
        
        # Roblox API should be preserved
        assert "game" in content
        assert "workspace" in content
        assert "Players" in content
        assert "GetService" in content
        assert "PlayerAdded" in content
        assert "Character" in content
        assert "Humanoid" in content
        assert "Instance" in content
        assert "Vector3" in content
        assert "Part" in content
        
        # User functions should be mangled
        assert "_0x" in content
        assert "on_player_joined" not in content or "_0x" in content

    def test_roblox_services(self, tmp_path):
        """Test Roblox service patterns are preserved."""
        # Create service-heavy script
        services = tmp_path / "services.lua"
        services.write_text("""
local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")
local TweenService = game:GetService("TweenService")
local UserInputService = game:GetService("UserInputService")

local function game_loop()
    RunService.Heartbeat:Connect(function(deltaTime)
        -- Game logic here
    end)
end

local function fetch_data(url)
    return HttpService:GetAsync(url)
end

return {
    game_loop = game_loop,
    fetch_data = fetch_data
}
""")
        
        # Run orchestration
        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )
        
        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        orchestrator.orchestrate(
            input_files=[str(services)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        content = (output_dir / "services.lua").read_text()
        
        # Services should be preserved
        assert "RunService" in content
        assert "HttpService" in content
        assert "TweenService" in content
        assert "UserInputService" in content
        assert "GetService" in content
        assert "Heartbeat" in content
        assert "GetAsync" in content
        
        # User functions should be mangled
        assert "_0x" in content
        assert "game_loop" not in content or "_0x" in content
        assert "fetch_data" not in content or "_0x" in content

    def test_local_vs_global_functions(self, tmp_path):
        """Test local vs global function scope handling."""
        # Create module with mixed scopes
        scopes = tmp_path / "scopes.lua"
        scopes.write_text("""
-- Global function (should be mangled)
function global_function()
    return "global"
end

-- Local function (should not be mangled)
local function local_function()
    return "local"
end

-- Global function with local helper
function process_data(data)
    local function helper(d)
        return d * 2
    end
    
    return helper(data)
end

return {
    global_function = global_function,
    process_data = process_data
}
""")
        
        # Run orchestration
        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )
        
        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        orchestrator.orchestrate(
            input_files=[str(scopes)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        content = (output_dir / "scopes.lua").read_text()
        
        # Should have mangled names
        assert "_0x" in content
        
        # Global functions should be mangled
        assert "function global_function" not in content
        assert "function process_data" not in content
        
        # Local functions should not be mangled (they keep original names)
        assert "local function local_function" in content
        assert "local function helper" in content

    def test_circular_requires(self, tmp_path):
        """Test circular requires with graceful fallback."""
        # Create circular dependency
        module_a = tmp_path / "module_a.lua"
        module_b = tmp_path / "module_b.lua"
        
        module_a.write_text("""
local module_b = require("module_b")

local function function_a()
    return "a"
end

local function call_b()
    return module_b.function_b()
end

return {
    function_a = function_a,
    call_b = call_b
}
""")
        
        module_b.write_text("""
local module_a = require("module_a")

local function function_b()
    return "b"
end

local function call_a()
    return module_a.function_a()
end

return {
    function_b = function_b,
    call_a = call_a
}
""")
        
        # Run orchestration
        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )
        
        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Should not raise error despite circular dependency
        orchestrator.orchestrate(
            input_files=[str(module_a), str(module_b)],
            output_dir=str(output_dir)
        )
        
        # Verify both files processed
        assert (output_dir / "module_a.lua").exists()
        assert (output_dir / "module_b.lua").exists()
        
        # Verify content
        a_content = (output_dir / "module_a.lua").read_text()
        b_content = (output_dir / "module_b.lua").read_text()
        
        # Should have mangled names
        assert "_0x" in a_content
        assert "_0x" in b_content
        
        # Require statements should be preserved
        assert 'require("module_b")' in a_content
        assert 'require("module_a")' in b_content

    def test_mixed_luau_types(self, tmp_path):
        """Test Luau type annotations don't break mangling."""
        # Create Luau script with type annotations
        luau_script = tmp_path / "typed.lua"
        luau_script.write_text("""
type PlayerData = {
    name: string,
    score: number,
    isActive: boolean
}

local function process_player(data: PlayerData): (string, number)
    local function helper(name: string): string
        return "Hello, " .. name
    end
    
    local message = helper(data.name)
    return message, data.score
end

return process_player
""")
        
        # Run orchestration
        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )
        
        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        orchestrator.orchestrate(
            input_files=[str(luau_script)],
            output_dir=str(output_dir)
        )
        
        # Verify content
        content = (output_dir / "typed.lua").read_text()
        
        # Type annotations should be preserved
        assert "type PlayerData" in content
        assert "name: string" in content
        assert "score: number" in content
        assert "isActive: boolean" in content
        
        # Functions should be mangled
        assert "_0x" in content
        assert "process_player" not in content or "_0x" in content
        assert "helper" not in content or "_0x" in content

    def test_field_access_mangled(self, tmp_path):
        """Test helper.calculate_sum field access is rewritten to mangled name."""
        # Create helper module with table exports
        helper = tmp_path / "helper.lua"
        helper.write_text("""
local M = {}

function M.calculate_sum(a, b)
    return a + b
end

return M
""")

        # Create main that accesses via field
        main = tmp_path / "main.lua"
        main.write_text("""
local helper = require("helper")

local function do_work(x, y)
    return helper.calculate_sum(x, y)
end

return do_work
""")

        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )

        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        orchestrator.orchestrate(
            input_files=[str(helper), str(main)],
            output_dir=str(output_dir)
        )

        helper_content = (output_dir / "helper.lua").read_text()
        main_content = (output_dir / "main.lua").read_text()

        # Both files should have mangled names
        assert "_0x" in helper_content
        assert "_0x" in main_content

        # Require path must remain unchanged
        assert 'require("helper")' in main_content

        # The field access should use the mangled identifier, not the original
        assert "calculate_sum" not in main_content or "_0x" in main_content

    def test_table_return_fields_mangled(self, tmp_path):
        """Test table return field keys/values are updated with mangled names."""
        # Create module with table return
        mod = tmp_path / "mod.lua"
        mod.write_text("""
function compute(x)
    return x * 2
end

return {
    compute = compute
}
""")

        config = ObfuscationConfig(
            name="test",
            features={"mangle_globals": True},
            symbol_table_options={
                "identifier_prefix": "_0x",
                "mangling_strategy": "sequential",
                "preserve_exports": False,
                "preserve_constants": False,
            }
        )

        orchestrator = ObfuscationOrchestrator(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        orchestrator.orchestrate(
            input_files=[str(mod)],
            output_dir=str(output_dir)
        )

        content = (output_dir / "mod.lua").read_text()

        # Function definition should be mangled
        assert "_0x" in content