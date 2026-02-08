"""Roblox remote spy protection runtime code generator for Lua.

This module generates Lua runtime code that protects RemoteEvent and RemoteFunction
calls from being intercepted by remote spy exploits. It encrypts remote object
names and uses dynamic lookups to resolve remote references at runtime.
"""

import base64
from typing import Optional


def generate_roblox_remote_spy_runtime(
    encryption_key: bytes,
    remote_name_map: dict[str, str],
) -> str:
    """Generate Lua Roblox remote spy protection runtime code.

    Generates runtime functions that:
    - Decrypt encrypted remote names using XOR cipher
    - Maintain a lookup table of encrypted remote names
    - Resolve remote objects dynamically via game:GetDescendants()
    - Cache resolved remotes to avoid repeated lookups

    Args:
        encryption_key: The XOR encryption key bytes
        remote_name_map: Dictionary mapping encrypted keys to encrypted remote names

    Returns:
        Lua code as a string containing all protection functions
    """
    # Encode key for Lua string literal (escape non-printable bytes)
    key_escaped = ''.join(f'\\{b}' for b in encryption_key)

    # Build the encrypted name lookup table
    # Note: encrypted_name is already base64-encoded from _encrypt_remote_name
    table_entries = []
    for key, encrypted_name in remote_name_map.items():
        # Store the base64-encoded encrypted name directly (no double encoding)
        table_entries.append(f'        ["{key}"] = "{encrypted_name}"')

    table_content = ',\n'.join(table_entries) if table_entries else ''

    # Define obfuscated function names
    decrypt_name = "_decrypt_remote_name_0x7b8c"
    names_table = "_remote_names_0x9d4e"
    resolve_remote = "_resolve_remote_0x6a5f"
    cache_table = "_remote_cache_0x3c2b"

    code_parts = []

    # Base64 decode helper function (internal)
    b64_decode_func = '''local function _b64_decode_0x8e2d(data)
    -- Base64 decode without external dependencies
    local b64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    local result = {}
    local padding = 0
    for i = 1, #data do
        local char = string.sub(data, i, i)
        if char == "=" then
            padding = padding + 1
        end
    end
    local len = #data - padding
    local i = 1
    while i <= len do
        local c1 = string.find(b64_chars, string.sub(data, i, i), 1, true) - 1
        local c2 = string.find(b64_chars, string.sub(data, i + 1, i + 1), 1, true) - 1
        local c3 = 0
        local c4 = 0
        if i + 2 <= #data and string.sub(data, i + 2, i + 2) ~= "=" then
            c3 = string.find(b64_chars, string.sub(data, i + 2, i + 2), 1, true) - 1
        end
        if i + 3 <= #data and string.sub(data, i + 3, i + 3) ~= "=" then
            c4 = string.find(b64_chars, string.sub(data, i + 3, i + 3), 1, true) - 1
        end
        local b1 = (c1 << 2) | (c2 >> 4)
        local b2 = ((c2 & 0xF) << 4) | (c3 >> 2)
        local b3 = ((c3 & 0x3) << 6) | c4
        table.insert(result, string.char(b1))
        if c3 ~= 64 and i + 2 <= len and string.sub(data, i + 2, i + 2) ~= "=" then
            table.insert(result, string.char(b2))
        end
        if c4 ~= 64 and i + 3 <= len and string.sub(data, i + 3, i + 3) ~= "=" then
            table.insert(result, string.char(b3))
        end
        i = i + 4
    end
    return table.concat(result)
end'''
    code_parts.append(b64_decode_func)
    code_parts.append("")

    # Encryption key constant
    key_def = f'''local _decrypt_key_0x4a1b = "{key_escaped}"'''
    code_parts.append(key_def)
    code_parts.append("")

    # Decryption function - always base64 decodes then XOR decrypts
    decrypt_func = f'''local function {decrypt_name}(encrypted_data_b64)
    -- Base64 decode then decrypt remote name using XOR cipher
    local encrypted_data = _b64_decode_0x8e2d(encrypted_data_b64)
    local result = {{}}
    local key_len = #_decrypt_key_0x4a1b
    for i = 1, #encrypted_data do
        local encrypted_byte = string.byte(encrypted_data, i)
        local key_byte = string.byte(_decrypt_key_0x4a1b, ((i - 1) % key_len) + 1)
        result[i] = string.char(encrypted_byte ~ key_byte)
    end
    return table.concat(result)
end'''
    code_parts.append(decrypt_func)
    code_parts.append("")

    # Encrypted name lookup table
    lookup_table = f'''local {names_table} = {{
{table_content}
}}'''
    code_parts.append(lookup_table)
    code_parts.append("")

    # Cache table for resolved remotes
    cache_def = f'''local {cache_table} = {{}}'''
    code_parts.append(cache_def)
    code_parts.append("")

    # Remote resolver function
    resolver_func = f'''local function {resolve_remote}(encrypted_key)
    -- Check cache first
    if {cache_table}[encrypted_key] ~= nil then
        return {cache_table}[encrypted_key]
    end
    
    -- Get encrypted name from lookup table
    local encrypted_name_b64 = {names_table}[encrypted_key]
    if not encrypted_name_b64 then
        return nil
    end
    
    -- Decrypt the remote name
    local remote_name = {decrypt_name}(encrypted_name_b64)
    
    -- Resolve remote object via game:GetDescendants()
    local _ok, _result = pcall(function()
        local _game = _G["game"] or game
        if _game and _game.GetDescendants then
            local _descendants = _game:GetDescendants()
            for _, _obj in ipairs(_descendants) do
                if _obj.Name == remote_name then
                    -- Verify it's a RemoteEvent or RemoteFunction
                    local _class = _obj.ClassName
                    if _class == "RemoteEvent" or _class == "RemoteFunction" then
                        return _obj
                    end
                end
            end
        end
        return nil
    end)
    
    if _ok and _result then
        {cache_table}[encrypted_key] = _result
        return _result
    end
    
    return nil
end'''
    code_parts.append(resolver_func)
    code_parts.append("")

    # Expose functions to global scope for transformed code
    expose_code = f'''_G["_resolve_remote_0x6a5f"] = {resolve_remote}
_G["_decrypt_remote_name_0x7b8c"] = {decrypt_name}'''
    code_parts.append(expose_code)

    return "\n\n".join(code_parts)
