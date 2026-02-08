# Name Mangling Limitations

This document captures known edge cases and limitations of the `mangle_globals` obfuscation feature. These are inherent constraints of static analysis–based renaming and apply to both Python and Lua processors.

## Dynamic Imports / Requires

- **Python `__import__()`**: Symbols imported via `__import__(module_name)` with a computed string cannot be tracked or mangled.
- **Python `importlib.import_module()`**: Dynamic module loading bypasses static import analysis.
- **Lua computed `require()`**: `require(variable)` where the argument is not a string literal cannot be resolved to a file.

## String-Based Access

- **Python `getattr(obj, "name")`**: Attribute names passed as string literals are not rewritten when definitions are mangled.
- **Python `globals()["name"]` / `locals()["name"]`**: Dictionary-style access to the symbol table uses string keys that are invisible to AST transformation.
- **Lua `table["key"]`**: Bracket-notation field access with string literals is not mangled.

## Reflection and Introspection

- **Python `inspect` module**: Code that inspects function names, signatures, or source at runtime will observe mangled identifiers.
- **Python `__dict__` access**: Direct dictionary access on modules or classes uses string keys that are not updated.
- **Lua `debug` library**: `debug.getinfo()` and related functions will report mangled names.

## Eval / Exec / Loadstring

- **Python `eval()` / `exec()`**: Code embedded in string arguments is not parsed or transformed.
- **Lua `loadstring()` / `load()`**: Dynamically compiled code is not analyzed for symbol references.

## Metatables and Metamethods (Lua)

- Symbols accessed via `__index` metamethods on metatables cannot be statically resolved.
- Proxy tables that delegate field access dynamically will not have their target fields mangled.

## C Extensions and Native Modules

- **Python C extensions**: Symbols defined in compiled `.so`/`.pyd` modules cannot be mangled.
- **Lua C modules**: Symbols exported from C-based Lua modules are opaque to the obfuscator.

## Roblox-Specific Caveats

- **RemoteEvent / RemoteFunction names**: Event names passed as string arguments (e.g., `RemoteEvent:FireServer("EventName")`) are not mangled.
- **Roblox service method strings**: `game:GetService("ServiceName")` string arguments are preserved, but user-defined service wrappers using dynamic strings are not tracked.
- **Instance.Name property**: Assigning or comparing `Instance.Name` with string literals referencing mangled symbols will break.

## Forward References in Type Annotations

- **Python quoted forward references**: Type hints like `"MyClass"` in string form may reference the original name after the class definition has been mangled.
- **Luau type aliases**: `type` declarations are preserved but may reference mangled identifiers if their targets are renamed.

## Cross-File Consistency Caveats

- **Circular dependencies**: When circular imports/requires are detected, files are processed in original order rather than topological order. Cross-file symbol consistency may be affected.
- **Multiple definitions with same name**: If two files define a global symbol with the same name, cross-file attribute resolution uses the first match found in the symbol table. This may produce incorrect mangling when the wrong definition is matched.

## Serialization and Persistence

- Pickled Python objects, JSON configs, or Lua data files that reference symbol names by string will not be updated.
- Database column names, API endpoint strings, and configuration keys that correspond to code symbols are not mangled.
