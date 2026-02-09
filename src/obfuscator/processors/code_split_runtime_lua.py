"""
Lua Runtime Generator for Code Splitting.

This module generates the Lua runtime code that decrypts and reassembles
function chunks that were split by the CodeSplittingTransformer. Uses XOR
cipher for chunk decryption, matching the encryption approach in
StringEncryptionTransformer for Lua compatibility.
"""


def generate_lua_chunk_decryption_runtime(
    encryption_key: bytes, encryption_enabled: bool = True
) -> str:
    """
    Generate Lua runtime code for chunk decryption and reassembly.

    The generated runtime provides two functions:
    - ``_decrypt_chunk(encrypted_data)``: Decrypts a single XOR-encrypted chunk
      when encryption is enabled, or returns data as-is when disabled.
    - ``_reassemble_function(chunks, ...)``: Decrypts all chunks, concatenates
      them, loads the combined code via ``loadstring()``/``load()``, and
      executes it forwarding any varargs so that function arguments are
      preserved in the reassembled body.

    Args:
        encryption_key: XOR encryption key bytes.
        encryption_enabled: Whether XOR encryption is active. When False the
            runtime treats chunk data as plain text without decryption.

    Returns:
        Lua code string containing the chunk decryption runtime.

    Example:
        >>> import secrets
        >>> key = secrets.token_bytes(16)
        >>> runtime_code = generate_lua_chunk_decryption_runtime(key)
        >>> isinstance(runtime_code, str)
        True
    """
    # Escape key bytes for Lua string literal using octal escape sequences
    key_escaped = ''.join(f'\\{b}' for b in encryption_key)

    if encryption_enabled:
        runtime = f'''
local _cs_decrypt_key = "{key_escaped}"

local function _decrypt_chunk(encrypted_data)
    local result = {{}}
    local key_len = #_cs_decrypt_key
    for i = 1, #encrypted_data do
        local encrypted_byte = string.byte(encrypted_data, i)
        local key_byte = string.byte(_cs_decrypt_key, ((i - 1) % key_len) + 1)
        result[i] = string.char(encrypted_byte ~ key_byte)
    end
    return table.concat(result)
end

local function _reassemble_function(chunks, ...)
    local decrypted_parts = {{}}
    for i = 1, #chunks do
        local ok, part = pcall(_decrypt_chunk, chunks[i])
        if not ok then
            error("Chunk decryption failed: " .. tostring(part))
        end
        decrypted_parts[i] = part
    end
    local full_code = table.concat(decrypted_parts, "\\n")
    local func, err = loadstring(full_code)
    if not func then
        func, err = load(full_code)
    end
    if not func then
        error("Function reassembly failed: " .. tostring(err))
    end
    return func(...)
end

-- Return the module with exported functions
return {{
    _decrypt_chunk = _decrypt_chunk,
    _reassemble_function = _reassemble_function,
}}
'''
    else:
        runtime = '''
local function _decrypt_chunk(data)
    return data
end

local function _reassemble_function(chunks, ...)
    local parts = {}
    for i = 1, #chunks do
        parts[i] = _decrypt_chunk(chunks[i])
    end
    local full_code = table.concat(parts, "\\n")
    local func, err = loadstring(full_code)
    if not func then
        func, err = load(full_code)
    end
    if not func then
        error("Function reassembly failed: " .. tostring(err))
    end
    return func(...)
end

-- Return the module with exported functions
return {{
    _decrypt_chunk = _decrypt_chunk,
    _reassemble_function = _reassemble_function,
}}
'''
    return runtime.strip()
