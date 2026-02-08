"""Anti-debugging runtime code generator for Lua.

This module generates Lua anti-debugging detection code that can be
injected into obfuscated scripts to detect and respond to debugging attempts.
"""

import textwrap
from typing import Optional


def generate_lua_anti_debug_checks(aggressiveness: int = 2, action: str = "exit") -> str:
    """Generate Lua anti-debugging runtime code.

    Generates a suite of anti-debugging detection functions that check for:
    - Debug library presence and accessibility
    - Timing-based detection (execution stepping)
    - Environment inspection (debug-related globals)
    - Hook detection via debug.gethook()

    Args:
        aggressiveness: Level of anti-debug checks (1=minimal, 2=moderate, 3=aggressive)
        action: Defensive action when debugging detected ("exit", "loop", "exception")

    Returns:
        Lua code as a string containing all detection functions
    """
    # Define obfuscated function names
    check_func_name = "_check_env_0x1a2b"
    debug_check_name = "_check_debug_0x3c4d"
    timing_check_name = "_check_timing_0x5e6f"
    hook_check_name = "_check_hook_0x7a8b"
    env_check_name = "_check_env_0x9c0d"
    defensive_action_name = "_defensive_action_0xd1e2"

    # Defensive action code based on selected action
    if action == "exit":
        action_code = "os.exit(1)"
    elif action == "loop":
        action_code = "while true do end"
    else:  # exception
        action_code = "error('Security violation detected')"

    code_parts = []

    # Defensive action function
    defensive_func = f"""local function {defensive_action_name}()
    -- Execute defensive action when debugging is detected
    {action_code}
end"""
    code_parts.append(defensive_func)
    code_parts.append("")

    # Debug library detection - always included
    debug_check = f"""local function {debug_check_name}()
    -- Check if debug library exists and is accessible
    if debug ~= nil then
        -- Additional check: see if debug functions are callable
        local _ok, _ = pcall(function() return debug.getinfo end)
        if _ok then
            {defensive_action_name}()
        end
    end
end"""
    code_parts.append(debug_check)
    code_parts.append("")

    # Timing-based detection - always included
    timing_threshold = "0.5" if aggressiveness >= 3 else "1.0"
    timing_check = f"""local function {timing_check_name}()
    -- Detect debugging via timing analysis
    local _start = os.clock()
    local _counter = 0
    for _ = 1, 1000 do
        _counter = _counter + 1
    end
    local _elapsed = os.clock() - _start
    if _elapsed > {timing_threshold} then
        {defensive_action_name}()
    end
end"""
    code_parts.append(timing_check)
    code_parts.append("")

    # Hook detection - always included
    hook_check = f"""local function {hook_check_name}()
    -- Check for active debug hooks
    if debug ~= nil and debug.gethook ~= nil then
        local _ok, _hook = pcall(debug.gethook)
        if _ok and _hook ~= nil then
            {defensive_action_name}()
        end
    end
end"""
    code_parts.append(hook_check)
    code_parts.append("")

    # Environment inspection - moderate and above
    if aggressiveness >= 2:
        env_check = f"""local function {env_check_name}()
    -- Check for debug-related global variables
    local _debug_vars = {{
        'debugger', 'mobdebug', 'ldbg', 'clidebug',
        'debugger_active', 'debug_mode'
    }}
    for _, _var in ipairs(_debug_vars) do
        if _G[_var] ~= nil then
            {defensive_action_name}()
        end
    end
end"""
        code_parts.append(env_check)
        code_parts.append("")

    # Main check function that calls all checks
    checks_to_call = [debug_check_name, timing_check_name, hook_check_name]
    if aggressiveness >= 2:
        checks_to_call.append(env_check_name)

    if aggressiveness >= 3:
        # At aggressive level, add randomized check order
        main_check = f"""local function {check_func_name}()
    -- Main anti-debugging check function
    local _checks = {{
        {debug_check_name},
        {timing_check_name},
        {hook_check_name},
        {env_check_name},
    }}
    -- Randomize check order to avoid pattern detection
    local _n = #_checks
    for _i = _n, 2, -1 do
        local _j = math.random(_i)
        _checks[_i], _checks[_j] = _checks[_j], _checks[_i]
    end
    for _, _check in ipairs(_checks) do
        local _ok, _ = pcall(_check)
        if not _ok then
            -- Ignore errors in individual checks
        end
    end
end"""
    else:
        # Standard sequential checks
        check_calls = "\n    ".join(f"{c}()" for c in checks_to_call)
        main_check = f"""local function {check_func_name}()
    -- Main anti-debugging check function
    {check_calls}
end"""

    code_parts.append(main_check)

    return "\n\n".join(code_parts)


def generate_lua_single_check(check_type: str = "debug", action: str = "exit") -> str:
    """Generate a single anti-debugging check as an inline statement.

    Useful for injecting quick checks without the full runtime.

    Args:
        check_type: Type of check ("debug", "timing", "hook")
        action: Defensive action ("exit", "loop", "exception")

    Returns:
        Lua code string for single check
    """
    if action == "exit":
        action_code = "os.exit(1)"
    elif action == "loop":
        action_code = "while true do end"
    else:
        action_code = "error('Security violation detected')"

    if check_type == "debug":
        return f"""if debug ~= nil then {action_code} end"""
    elif check_type == "timing":
        return f"""local _s = os.clock(); for _ = 1, 100 do end; if os.clock() - _s > 0.1 then {action_code} end"""
    elif check_type == "hook":
        return f"""if debug ~= nil and debug.gethook ~= nil then local _ok, _h = pcall(debug.gethook); if _ok and _h ~= nil then {action_code} end end"""
    else:
        return f"""if debug ~= nil then {action_code} end"""


def generate_lua_obfuscated_check(aggressiveness: int = 2) -> str:
    """Generate obfuscated anti-debugging check code.

    Creates more stealthy checks that are harder to detect and remove.

    Args:
        aggressiveness: Obfuscation level (affects complexity)

    Returns:
        Lua code string with obfuscated checks
    """
    code = '''
-- Obfuscated debug check using indirect access
local _g = _G
local _d = _g["debug"]

if _d ~= nil then
    local _e = _g["os"]["exit"]
    _e(1)
end

-- Indirect timing check
local _c = _g["os"]["clock"]
local _s = _c()
for _ = 1, 100 do end
if (_c() - _s) > 0.1 then
    _g["os"]["exit"](1)
end
'''
    return code.strip()
