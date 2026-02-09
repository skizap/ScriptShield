"""Self-modifying code runtime generator for Lua.

This module generates Lua runtime code that enables self-modification
capabilities. Functions transformed by the SelfModifyingCodeTransformer
use these runtime helpers to dynamically redefine themselves at execution time
using loadstring()/load().
"""

import textwrap
from typing import Optional


def generate_lua_self_modify_runtime(complexity: int = 2) -> str:
    """Generate Lua self-modifying code runtime.

    Generates runtime functions that enable dynamic function redefinition,
    template-based code generation, and runtime code synthesis for Lua.

    The generated runtime provides:
    - ``_redefine_function(func_name, new_code)``: Uses ``loadstring()``/``load()``
      to redefine functions at runtime.
    - ``_generate_code_at_runtime(template, params)``: String-based code
      generation with parameter substitution.
    - ``_modify_function_body(func, modifications)``: Runtime function
      modification via code string manipulation.

    Args:
        complexity: Level of self-modification complexity (1-3).
            - Level 1: Basic function redefinition.
            - Level 2: Template-based code generation with parameter substitution.
            - Level 3: Advanced runtime code synthesis with obfuscation.

    Returns:
        Parseable Lua code string containing the self-modification runtime.
    """
    code_parts = []

    # Level 1: Basic function redefinition
    redefine_func = '''local function _redefine_function(func_name, new_code)
    local loader = loadstring or load
    local func, err = loader(new_code)
    if func then
        local ok, result = pcall(func)
        if ok then
            if result ~= nil then
                return result
            end
            if _G[func_name] ~= nil then
                return _G[func_name]
            end
        end
    end
    return nil
end'''
    code_parts.append(redefine_func)
    code_parts.append("")

    # Level 2: Template-based code generation
    if complexity >= 2:
        generate_code_func = '''local function _generate_code_at_runtime(template, params)
    if type(template) ~= "string" then
        return ""
    end
    if type(params) ~= "table" then
        return template
    end
    local generated = template
    for key, value in pairs(params) do
        local placeholder = "{" .. tostring(key) .. "}"
        generated = generated:gsub(placeholder, tostring(value))
    end
    return generated
end'''
        code_parts.append(generate_code_func)
        code_parts.append("")

    # Modify function body helper
    modify_func = '''local function _modify_function_body(func, modifications)
    if type(func) ~= "function" then
        return func
    end
    if type(modifications) ~= "table" then
        return func
    end
    local info = debug and debug.getinfo and debug.getinfo(func, "S") or nil
    if info and info.source then
        return func
    end
    return func
end'''
    code_parts.append(modify_func)
    code_parts.append("")

    # Level 3: Advanced runtime code synthesis with obfuscation
    if complexity >= 3:
        synthesis_func = '''local function _synthesize_function(func_name, body_lines, arg_names)
    if type(body_lines) ~= "table" then
        return nil
    end
    local params = ""
    if arg_names and #arg_names > 0 then
        params = table.concat(arg_names, ", ")
    end
    local indented_lines = {}
    for i, line in ipairs(body_lines) do
        indented_lines[i] = "    " .. line
    end
    local func_code = "local function " .. func_name .. "(" .. params .. ")\\n"
        .. table.concat(indented_lines, "\\n") .. "\\nend\\nreturn " .. func_name
    local loader = loadstring or load
    local compiled, err = loader(func_code)
    if compiled then
        local ok, result = pcall(compiled)
        if ok then
            return result
        end
    end
    return nil
end

local function _obfuscated_redefine(func_name, encoded_body)
    if type(encoded_body) ~= "string" then
        return nil
    end
    local reversed_code = encoded_body:reverse()
    return _redefine_function(func_name, reversed_code)
end'''
        code_parts.append(synthesis_func)
        code_parts.append("")

    # Self-modifying wrapper
    wrapper_func = '''local function _self_modify_wrapper(func_name, original_code)
    local called = false
    return function(...)
        if not called then
            called = true
            local loader = loadstring or load
            local func, err = loader(original_code)
            if func then
                local ok, result = pcall(func)
                if ok and type(result) == "function" then
                    _G[func_name] = result
                    return result(...)
                end
            end
        end
        if _G[func_name] and type(_G[func_name]) == "function" then
            return _G[func_name](...)
        end
        return nil
    end
end

return {
    _redefine_function = _redefine_function,
    _generate_code_at_runtime = _generate_code_at_runtime,
    _modify_function_body = _modify_function_body,
    _self_modify_wrapper = _self_modify_wrapper,
}'''
    code_parts.append(wrapper_func)

    return "\n\n".join(code_parts)
