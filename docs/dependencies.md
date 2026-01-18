# Dependencies Documentation

## Overview

This document describes all dependencies required for the Python Obfuscator project. The project is designed to work with Python 3.9 and above, ensuring compatibility across modern Python versions.

Each dependency has been carefully selected to provide specific functionality while maintaining a minimal footprint and ensuring long-term maintainability.

**Python Version Requirement:** `>=3.9`

---

## Core Dependencies

| Dependency | Version | Purpose | License |
|------------|---------|---------|---------|
| PyQt6 | 6.10.0 | GUI framework for the obfuscator interface | GPL v3 / Commercial |
| luaparser | 4.0.0 | Lua code parsing and AST generation | MIT |
| cryptography | >=41.0.0 | String encryption using AES-256-GCM | Apache 2.0 / BSD |
| pathlib | stdlib | Cross-platform path handling | PSF |

### PyQt6 (6.10.0)

**Purpose:** Provides the graphical user interface framework for the obfuscator application.

**Why PyQt6:**
- Modern Qt6 bindings for Python with excellent cross-platform support
- Comprehensive widget library for building professional desktop applications
- Active development and strong community support
- Compatible with Python 3.9-3.13

**License Note:** PyQt6 is dual-licensed under GPL v3 and commercial licenses. For open-source projects, GPL v3 is appropriate. Commercial projects may require a commercial license.

### luaparser (4.0.0)

**Purpose:** Parses Lua source code into an Abstract Syntax Tree (AST) for analysis and transformation.

**Why luaparser:**
- Pure Python implementation for easy integration
- Supports modern Lua syntax
- Provides comprehensive AST representation for obfuscation operations
- MIT license allows flexible usage

**Key Features:**
- Parses Lua 5.3 syntax
- Generates detailed AST nodes for all Lua constructs
- Enables programmatic code transformation

### cryptography (>=41.0.0)

**Purpose:** Provides secure cryptographic primitives for string encryption obfuscation.

**Why cryptography:**
- Industry-standard cryptographic library with extensive security auditing
- Provides AES-256-GCM authenticated encryption for Python string obfuscation
- Well-maintained with regular security updates
- Apache 2.0 / BSD dual-license allows flexible usage

**Key Features:**
- AES-256-GCM authenticated encryption for Python code
- Secure random key and IV generation
- High-performance implementation using OpenSSL bindings
- Cross-platform support (Windows, Linux, macOS)

**Usage in Obfuscator:**
- String literals in Python code are encrypted using AES-256-GCM
- Encryption key length is configurable (default: 16 bytes)
- A decryption runtime is injected into obfuscated code
- Lua uses simpler XOR-based encryption for compatibility

### pathlib (Standard Library)

**Purpose:** Provides object-oriented filesystem path handling with cross-platform compatibility.

**Why pathlib:**
- Part of Python standard library (no installation required)
- Modern, Pythonic API for path operations
- Cross-platform compatibility (Windows, Linux, macOS)
- Preferred over `os.path` for new Python projects

---

## Development Dependencies

Development dependencies are specified in `requirements-dev.txt` and include tools for testing, code quality, and packaging.

| Dependency | Purpose |
|------------|---------|
| pytest | Testing framework for unit and integration tests |
| black | Code formatter for consistent style |
| flake8 | Linting tool for code quality checks |
| mypy | Static type checker for Python |
| build | PEP 517 build frontend for creating distributions |
| twine | Tool for publishing packages to PyPI |

### Installation

```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

---

## Installation Instructions

### Basic Installation

Install only the core dependencies required to run the obfuscator:

```bash
# Install core dependencies
pip install -r requirements.txt
```

### Development Installation

Install both core and development dependencies for contributing to the project:

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Verification

After installation, verify that all dependencies are correctly installed:

```bash
# Run the verification script
python scripts/verify_dependencies.py
```

Expected output:
```
✓ PyQt6 6.10.0 installed successfully
✓ luaparser 4.0.0 installed successfully
✓ pathlib available (standard library)
✓ All dependencies verified successfully!
```

---

## Version Pinning Strategy

### Why Pin Exact Versions?

This project uses **exact version pinning** (e.g., `PyQt6==6.10.0`) rather than minimum versions (e.g., `PyQt6>=6.10.0`) for the following reasons:

1. **Reproducibility:** Ensures all developers and users have identical dependency versions
2. **Stability:** Prevents unexpected breaking changes from automatic updates
3. **Testing:** Guarantees that tested configurations match production environments
4. **Debugging:** Eliminates version mismatches as a source of bugs

### Update Policy

Dependencies should be updated when:

- **Security vulnerabilities** are discovered in current versions
- **Bug fixes** in newer versions address issues affecting the project
- **New features** in dependencies provide significant value
- **Compatibility** requirements change (e.g., new Python version support)

### How to Update Dependencies

1. Update version numbers in `requirements.txt`
2. Update version numbers in `pyproject.toml` dependencies list
3. Test thoroughly with the new versions
4. Run `python scripts/verify_dependencies.py` to verify installation
5. Update this documentation with any compatibility notes
6. Commit changes with a clear description of what was updated and why

---

## Compatibility Notes

### Python Version Compatibility

| Python Version | PyQt6 6.10.0 | luaparser 4.0.0 | Status |
|----------------|--------------|-----------------|--------|
| 3.9 | ✓ | ✓ | Supported |
| 3.10 | ✓ | ✓ | Supported |
| 3.11 | ✓ | ✓ | Supported |
| 3.12 | ✓ | ✓ | Supported |
| 3.13 | ✓ | ✓ | Supported |

### Platform Compatibility

All dependencies are cross-platform and support:
- **Linux** (Ubuntu, Debian, Fedora, etc.)
- **macOS** (10.13+)
- **Windows** (10+)

---

## Alternative Considerations

### PySide6 vs PyQt6

**PySide6** is an alternative to PyQt6 with the following differences:

| Feature | PyQt6 | PySide6 |
|---------|-------|---------|
| License | GPL v3 / Commercial | LGPL v3 |
| Maintainer | Riverbank Computing | Qt Company |
| API | Nearly identical | Nearly identical |
| Performance | Excellent | Excellent |

**Why PyQt6 was chosen:**
- Slightly more mature ecosystem
- Extensive documentation and community resources
- GPL v3 license is acceptable for this open-source project

**Note:** If LGPL licensing is required, PySide6 can be used as a drop-in replacement with minimal code changes.

### pathlib vs os.path

**pathlib** is preferred over `os.path` for:
- Object-oriented API (more Pythonic)
- Better cross-platform path handling
- Cleaner, more readable code
- Modern Python best practices

---

## Known Issues

### PyQt6 on Wayland (Linux)

Some Linux distributions using Wayland may experience issues with PyQt6. If you encounter problems:

```bash
# Force X11 backend
export QT_QPA_PLATFORM=xcb
python -m obfuscator
```

### luaparser Limitations

- Supports Lua 5.3 syntax (not Lua 5.4 features)
- Some edge cases in complex Lua code may not parse correctly
- Report parsing issues to: https://github.com/boolangery/py-lua-parser

---

## Troubleshooting

### Import Errors

If you encounter import errors after installation:

1. Verify Python version: `python --version` (should be >=3.9)
2. Verify pip installation: `pip list | grep -E "PyQt6|luaparser"`
3. Run verification script: `python scripts/verify_dependencies.py`
4. Try reinstalling: `pip install --force-reinstall -r requirements.txt`

### Version Conflicts

If you have conflicting versions installed:

```bash
# Uninstall existing versions
pip uninstall PyQt6 luaparser

# Reinstall from requirements
pip install -r requirements.txt
```

---

## Dependency Analysis System

The obfuscator includes a dependency analysis system for multi-file projects
that ensures consistent symbol mangling across files with import relationships.

### Internal Dependencies

The dependency analysis system uses these internal modules:

| Module | Purpose |
|--------|---------|
| `core/dependency_graph.py` | Dependency graph construction and topological sorting |
| `core/symbol_table.py` | Global symbol table with pre-computed mangled names |
| `core/orchestrator.py` | Workflow coordination for multi-file processing |

### How It Works

1. **Import Analysis:** The system analyzes Python `import` statements and Lua
   `require()` calls to build a dependency graph.

2. **Topological Sorting:** Files are processed in dependency order using
   Kahn's algorithm, ensuring dependencies are obfuscated before dependents.

3. **Symbol Pre-computation:** A global symbol table is built with all mangled
   names computed before any file is transformed, ensuring cross-file
   consistency.

### Configuration Options

The dependency analysis system respects these configuration options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `preserve_exports` | bool | `false` | Keep exported symbol names unchanged |
| `preserve_constants` | bool | `false` | Keep constant names unchanged |
| `mangling_strategy` | string | `"sequential"` | Name generation strategy |
| `identifier_prefix` | string | `"_0x"` | Prefix for mangled names |

### Circular Dependency Handling

If circular dependencies are detected:

1. A `CircularDependencyError` is raised with the cycle path
2. The orchestrator logs a warning and falls back to original file order
3. Processing continues with best-effort symbol consistency

### Reserved Names

The system automatically preserves:

- **Python:** Built-in functions, magic methods (`__init__`, `__str__`, etc.)
- **Lua:** Keywords, Roblox API globals (when applicable)

---

## Additional Resources

- **PyQt6 Documentation:** https://www.riverbankcomputing.com/static/Docs/PyQt6/
- **luaparser Documentation:** https://github.com/boolangery/py-lua-parser
- **pathlib Documentation:** https://docs.python.org/3/library/pathlib.html
- **Python Packaging Guide:** https://packaging.python.org/

---

*Last Updated: 2026-01-18*

