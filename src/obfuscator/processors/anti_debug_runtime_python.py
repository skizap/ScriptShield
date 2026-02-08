"""Anti-debugging runtime code generator for Python.

This module generates Python anti-debugging detection code that can be
injected into obfuscated scripts to detect and respond to debugging attempts.
"""

import textwrap
from typing import Optional


def generate_python_anti_debug_checks(aggressiveness: int = 2, action: str = "exit") -> str:
    """Generate Python anti-debugging runtime code.

    Generates a suite of anti-debugging detection functions that check for:
    - sys.gettrace() detection (debugger attached)
    - Timing-based detection (execution stepping)
    - Debugger module detection (pdb, pydevd, debugpy, etc.)
    - Process inspection (parent process name)

    Args:
        aggressiveness: Level of anti-debug checks (1=minimal, 2=moderate, 3=aggressive)
        action: Defensive action when debugging detected ("exit", "loop", "exception")

    Returns:
        Python code as a string containing all detection functions
    """
    # Define obfuscated function names
    check_func_name = "_check_env_0x1a2b"
    trace_check_name = "_check_trace_0x3c4d"
    timing_check_name = "_check_timing_0x5e6f"
    module_check_name = "_check_modules_0x7a8b"
    process_check_name = "_check_process_0x9c0d"
    defensive_action_name = "_defensive_action_0xd1e2"

    # Defensive action code based on selected action
    if action == "exit":
        action_code = "sys.exit(1)"
    elif action == "loop":
        action_code = "while True: pass"
    else:  # exception
        action_code = "raise RuntimeError('Security violation detected')"

    # Base imports always included
    code_parts = [
        "import sys",
        "import time",
    ]

    # Add os import for process checking at higher aggressiveness
    if aggressiveness >= 2:
        code_parts.append("import os")

    # Add platform import for platform-specific checks
    if aggressiveness >= 3:
        code_parts.append("import platform")

    code_parts.append("")  # Empty line after imports

    # Defensive action function
    defensive_func = f"""def {defensive_action_name}():
    # Execute defensive action when debugging is detected
    {action_code}
"""
    code_parts.append(textwrap.dedent(defensive_func).strip())
    code_parts.append("")

    # sys.gettrace() detection - always included
    trace_check = f"""def {trace_check_name}():
    # Check if a trace function is set (debugger attached)
    if sys.gettrace() is not None:
        {defensive_action_name}()
"""
    code_parts.append(textwrap.dedent(trace_check).strip())
    code_parts.append("")

    # Timing-based detection - always included
    timing_threshold = "0.5" if aggressiveness >= 3 else "1.0"
    timing_check = f"""def {timing_check_name}():
    # Detect debugging via timing analysis
    _start = time.perf_counter()
    _counter = 0
    for _ in range(1000):
        _counter += 1
    _elapsed = time.perf_counter() - _start
    if _elapsed > {timing_threshold}:
        {defensive_action_name}()
"""
    code_parts.append(textwrap.dedent(timing_check).strip())
    code_parts.append("")

    # Debugger module detection - always included
    module_check = f"""def {module_check_name}():
    # Check for presence of debugger modules
    _debug_modules = {{'pdb', 'pydevd', 'debugpy', 'ipdb', 'pdbpp', 'wdb'}}
    for _mod in _debug_modules:
        if _mod in sys.modules:
            {defensive_action_name}()
"""
    code_parts.append(textwrap.dedent(module_check).strip())
    code_parts.append("")

    # Process inspection - moderate and above
    if aggressiveness >= 2:
        process_check = f"""def {process_check_name}():
    # Check parent process for known debuggers
    try:
        if hasattr(os, 'getppid'):
            _ppid = os.getppid()
            _debugger_names = ['gdb', 'lldb', 'strace', 'ltrace', 'idbg']
            # Check process name if possible (platform-specific)
            if hasattr(os, 'uname') and os.uname().sysname == 'Linux':
                try:
                    with open(f'/proc/{{_ppid}}/comm', 'r') as _f:
                        _comm = _f.read().strip()
                        if any(_d in _comm.lower() for _d in _debugger_names):
                            {defensive_action_name}()
                except:
                    pass
    except:
        pass
"""
        code_parts.append(textwrap.dedent(process_check).strip())
        code_parts.append("")

    # Main check function that calls all checks
    checks_to_call = [trace_check_name, timing_check_name, module_check_name]
    if aggressiveness >= 2:
        checks_to_call.append(process_check_name)

    if aggressiveness >= 3:
        # At aggressive level, add randomized check selection
        main_check = f"""def {check_func_name}():
    # Main anti-debugging check function
    import random
    _checks = [
        {trace_check_name},
        {timing_check_name},
        {module_check_name},
        {process_check_name},
    ]
    # Randomize check order to avoid pattern detection
    random.shuffle(_checks)
    for _check in _checks:
        try:
            _check()
        except:
            pass
"""
    else:
        # Standard sequential checks
        check_calls = "\n    ".join(f"{c}()" for c in checks_to_call)
        main_check = f"""def {check_func_name}():
    # Main anti-debugging check function
    {check_calls}
"""

    code_parts.append(textwrap.dedent(main_check).strip())

    return "\n\n".join(code_parts)


def generate_python_single_check(check_type: str = "trace", action: str = "exit") -> str:
    """Generate a single anti-debugging check as an inline statement.

    Useful for injecting quick checks without the full runtime.

    Args:
        check_type: Type of check ("trace", "timing", "modules")
        action: Defensive action ("exit", "loop", "exception")

    Returns:
        Python code string for single check
    """
    if action == "exit":
        action_code = "sys.exit(1)"
    elif action == "loop":
        action_code = "while True: pass"
    else:
        action_code = "raise RuntimeError('Security violation detected')"

    if check_type == "trace":
        return f"""import sys; (lambda: ({action_code}) if sys.gettrace() is not None else None)()"""
    elif check_type == "timing":
        return f"""import time; _s = time.perf_counter(); [0 for _ in range(100)]; (lambda: ({action_code}) if time.perf_counter() - _s > 0.1 else None)()"""
    elif check_type == "modules":
        return f"""import sys; (lambda: ({action_code}) if any(m in sys.modules for m in ['pdb', 'pydevd', 'debugpy']) else None)()"""
    else:
        return f"""import sys; (lambda: ({action_code}) if sys.gettrace() is not None else None)()"""


def generate_python_obfuscated_check(aggressiveness: int = 2) -> str:
    """Generate obfuscated anti-debugging check code.

    Creates more stealthy checks that are harder to detect and remove.

    Args:
        aggressiveness: Obfuscation level (affects complexity)

    Returns:
        Python code string with obfuscated checks
    """
    # Use string obfuscation and indirect execution
    code = '''
import sys
import time
import types

# Obfuscated check using indirect attribute access
_g = globals()
_s = __import__('sys')
_t = __import__('time')

# Indirect trace check
if hasattr(_s, 'gettrace'):
    _tr = getattr(_s, 'gettrace')
    if callable(_tr) and _tr() is not None:
        _e = getattr(_s, 'exit')
        _e(1)

# Indirect module check
_m = getattr(_s, 'modules')
_d = ['pdb', 'pydevd', 'debugpy']
for _mod in _d:
    if _mod in _m:
        _e = getattr(_s, 'exit')
        _e(1)
'''
    return code.strip()
