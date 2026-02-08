"""Self-modifying code runtime generator for Python.

This module generates Python runtime code that enables self-modification
capabilities. Functions transformed by the SelfModifyingCodeTransformer
use these runtime helpers to dynamically redefine themselves at execution time.
"""

import textwrap
from typing import Optional


def generate_python_self_modify_runtime(complexity: int = 2) -> str:
    """Generate Python self-modifying code runtime.

    Generates runtime functions that enable dynamic function redefinition,
    template-based code generation, and runtime code synthesis.

    The generated runtime provides:
    - ``_redefine_function(func_name, new_code, namespace)``: Dynamically
      redefines a function using ``compile()`` and ``exec()``.
    - ``_generate_code_at_runtime(template, params)``: Generates code from
      templates at runtime via string substitution.
    - ``_modify_function_body(func, modifications)``: Modifies function
      internals dynamically by recompiling with modifications applied.

    Args:
        complexity: Level of self-modification complexity (1-3).
            - Level 1: Basic function redefinition.
            - Level 2: Template-based code generation with parameter substitution.
            - Level 3: Advanced runtime code synthesis with obfuscation.

    Returns:
        Parseable Python code string containing the self-modification runtime.
    """
    code_parts = []

    # Base imports always included
    code_parts.append("import types")
    code_parts.append("import textwrap")
    code_parts.append("")

    # Level 1: Basic function redefinition
    redefine_func = '''
def _redefine_function(func_name, new_code, namespace=None):
    """Dynamically redefine a function using compile() and exec()."""
    try:
        if namespace is None:
            namespace = {}
        compiled = compile(new_code, "<self_modify>", "exec")
        exec(compiled, namespace)
        if func_name in namespace:
            return namespace[func_name]
        return None
    except Exception:
        return None
'''
    code_parts.append(textwrap.dedent(redefine_func).strip())
    code_parts.append("")

    # Level 2: Template-based code generation
    if complexity >= 2:
        generate_code_func = '''
def _generate_code_at_runtime(template, params):
    """Generate code from a template with parameter substitution at runtime."""
    try:
        if not isinstance(template, str):
            return ""
        if not isinstance(params, dict):
            params = {}
        generated = template
        for key, value in params.items():
            placeholder = "{" + str(key) + "}"
            generated = generated.replace(placeholder, str(value))
        return generated
    except Exception:
        return template
'''
        code_parts.append(textwrap.dedent(generate_code_func).strip())
        code_parts.append("")

    # Modify function body helper
    modify_func = '''
def _modify_function_body(func, modifications):
    """Modify function internals dynamically by recompiling with modifications."""
    try:
        import inspect
        if not callable(func):
            return func
        if not isinstance(modifications, dict):
            return func
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            return func
        modified_source = source
        for old_val, new_val in modifications.items():
            modified_source = modified_source.replace(str(old_val), str(new_val))
        namespace = {}
        compiled = compile(modified_source, "<self_modify>", "exec")
        exec(compiled, namespace)
        func_name = func.__name__
        if func_name in namespace:
            return namespace[func_name]
        return func
    except Exception:
        return func
'''
    code_parts.append(textwrap.dedent(modify_func).strip())
    code_parts.append("")

    # Level 3: Advanced runtime code synthesis with obfuscation
    if complexity >= 3:
        synthesis_func = '''
def _synthesize_function(func_name, body_lines, arg_names=None, namespace=None):
    """Synthesize a function from individual body lines at runtime.

    Constructs a complete function definition from parts, compiles, and
    registers it in the given namespace.
    """
    try:
        if namespace is None:
            namespace = {}
        if arg_names is None:
            arg_names = []
        params = ", ".join(arg_names)
        indented_body = textwrap.indent("\\n".join(body_lines), "    ")
        func_code = f"def {func_name}({params}):\\n{indented_body}\\n"
        compiled = compile(func_code, "<self_modify_synth>", "exec")
        exec(compiled, namespace)
        if func_name in namespace:
            return namespace[func_name]
        return None
    except Exception:
        return None


def _obfuscated_redefine(func_name, encoded_body, namespace=None):
    """Redefine a function from an obfuscated (reversed + encoded) body."""
    try:
        import base64
        if namespace is None:
            namespace = {}
        decoded = base64.b64decode(encoded_body).decode("utf-8")
        reversed_code = decoded[::-1]
        return _redefine_function(func_name, reversed_code, namespace)
    except Exception:
        return None
'''
        code_parts.append(textwrap.dedent(synthesis_func).strip())
        code_parts.append("")

    # Self-modifying wrapper that redefines on first call
    wrapper_func = '''
def _self_modify_wrapper(func_name, original_code, namespace=None):
    """Create a self-modifying wrapper that redefines the function on first call.

    Returns a wrapper function that, on its first invocation, redefines
    the target function using the provided code, then calls the redefined
    version with the original arguments.
    """
    _sm_called = [False]

    def _sm_wrapper(*args, **kwargs):
        if not _sm_called[0]:
            _sm_called[0] = True
            ns = namespace if namespace is not None else globals()
            new_func = _redefine_function(func_name, original_code, ns)
            if new_func is not None and callable(new_func):
                if func_name in ns:
                    pass
                return new_func(*args, **kwargs)
        if namespace is not None and func_name in namespace:
            return namespace[func_name](*args, **kwargs)
        return None

    return _sm_wrapper
'''
    code_parts.append(textwrap.dedent(wrapper_func).strip())

    return "\n\n".join(code_parts)
