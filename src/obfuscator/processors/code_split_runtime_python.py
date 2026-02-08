"""
Python Runtime Generator for Code Splitting.

This module generates the Python runtime code that decrypts and reassembles
function chunks that were split by the CodeSplittingTransformer. Uses AES-GCM
for chunk decryption (with per-chunk IVs), matching the encryption approach in
StringEncryptionTransformer.
"""

import base64
from typing import Optional


def generate_python_chunk_decryption_runtime(
    encryption_key: bytes, encryption_enabled: bool = True
) -> str:
    """
    Generate Python runtime code for chunk decryption and reassembly.

    The generated runtime provides two functions:
    - ``_decrypt_chunk(encrypted_b64)``: Decrypts a single base64-encoded chunk.
      When encryption is enabled, extracts a 12-byte per-chunk IV prefix from
      the ciphertext before AES-GCM decryption. When disabled, plain base64
      decode only.
    - ``_reassemble_function(chunks, ...)``: Decrypts all chunks, concatenates
      them, wraps the code in an inner function definition to preserve
      ``return`` semantics, executes the wrapper, and returns the result.

    Args:
        encryption_key: AES encryption key bytes (16, 24, or 32 bytes).
        encryption_enabled: Whether AES encryption is active. When False the
            runtime treats chunks as plain base64-encoded UTF-8.

    Returns:
        Python code string containing the chunk decryption runtime.

    Example:
        >>> import secrets
        >>> key = secrets.token_bytes(16)
        >>> runtime_code = generate_python_chunk_decryption_runtime(key)
        >>> isinstance(runtime_code, str)
        True
    """
    key_b64 = base64.b64encode(encryption_key).decode('ascii')

    if encryption_enabled:
        runtime = f'''
import base64 as _cs_b64
import textwrap as _cs_textwrap
from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _CS_AESGCM

_cs_decrypt_key = _cs_b64.b64decode("{key_b64}")
_cs_aesgcm = _CS_AESGCM(_cs_decrypt_key)


def _decrypt_chunk(encrypted_b64: str) -> str:
    """Decrypt a single base64-encoded encrypted chunk (per-chunk IV)."""
    try:
        raw = _cs_b64.b64decode(encrypted_b64)
        _cs_iv = raw[:12]
        ciphertext = raw[12:]
        decrypted = _cs_aesgcm.decrypt(_cs_iv, ciphertext, None)
        return decrypted.decode("utf-8")
    except Exception as _cs_err:
        raise RuntimeError("Chunk decryption failed") from _cs_err


def _reassemble_function(chunks, _cs_globals=None, _cs_locals=None,
                          _cs_arg_names=None, _cs_arg_values=None):
    """Decrypt, reassemble, wrap in a function, and execute.

    Args:
        chunks: List of base64-encoded encrypted code chunks.
        _cs_globals: Global namespace for execution.
        _cs_locals: Local namespace for execution.
        _cs_arg_names: List of parameter name strings for the inner wrapper.
        _cs_arg_values: List of argument values to pass to the inner wrapper.

    Returns:
        The return value of the reassembled function body.
    """
    try:
        decrypted_parts = [_decrypt_chunk(c) for c in chunks]
        full_code = "\\n".join(decrypted_parts)
        indented = _cs_textwrap.indent(full_code, "    ")
        params = ", ".join(_cs_arg_names) if _cs_arg_names else ""
        wrapper = "def _cs_inner(" + params + "):\\n" + indented + "\\n"
        if _cs_globals is None:
            _cs_globals = globals()
        _cs_ns = dict(_cs_globals)
        if _cs_locals is not None:
            _cs_ns.update(_cs_locals)
        exec(compile(wrapper, "<code_split>", "exec"), _cs_ns)
        return _cs_ns["_cs_inner"](*(_cs_arg_values if _cs_arg_values is not None else []))
    except Exception as _cs_err:
        raise RuntimeError("Function reassembly failed") from _cs_err
'''
    else:
        runtime = '''
import base64 as _cs_b64
import textwrap as _cs_textwrap


def _decrypt_chunk(encoded_b64: str) -> str:
    """Decode a base64-encoded plain chunk (no encryption)."""
    try:
        return _cs_b64.b64decode(encoded_b64).decode("utf-8")
    except Exception as _cs_err:
        raise RuntimeError("Chunk decoding failed") from _cs_err


def _reassemble_function(chunks, _cs_globals=None, _cs_locals=None,
                          _cs_arg_names=None, _cs_arg_values=None):
    """Decode, reassemble, wrap in a function, and execute.

    Args:
        chunks: List of base64-encoded plain code chunks.
        _cs_globals: Global namespace for execution.
        _cs_locals: Local namespace for execution.
        _cs_arg_names: List of parameter name strings for the inner wrapper.
        _cs_arg_values: List of argument values to pass to the inner wrapper.

    Returns:
        The return value of the reassembled function body.
    """
    try:
        decrypted_parts = [_decrypt_chunk(c) for c in chunks]
        full_code = "\\n".join(decrypted_parts)
        indented = _cs_textwrap.indent(full_code, "    ")
        params = ", ".join(_cs_arg_names) if _cs_arg_names else ""
        wrapper = "def _cs_inner(" + params + "):\\n" + indented + "\\n"
        if _cs_globals is None:
            _cs_globals = globals()
        _cs_ns = dict(_cs_globals)
        if _cs_locals is not None:
            _cs_ns.update(_cs_locals)
        exec(compile(wrapper, "<code_split>", "exec"), _cs_ns)
        return _cs_ns["_cs_inner"](*(_cs_arg_values if _cs_arg_values is not None else []))
    except Exception as _cs_err:
        raise RuntimeError("Function reassembly failed") from _cs_err
'''
    return runtime.strip()
