# Obfuscator

Obfuscator is a cross-platform Lua code obfuscator with a PyQt6-based graphical
interface. It is designed to help Lua developers and game developers protect
their source code by transforming it into harder-to-read, harder-to-reverse
forms while preserving runtime behavior.

## Project Overview

**Key features (current and planned):**

- GUI-based workflow built with **PyQt6**
- Cross-platform support for **Windows**, **macOS**, and **Linux**
- Lua parsing and AST manipulation powered by **luaparser**
- Safe, cross-platform file handling utilities
- **VM Protection** - Convert functions to bytecode executed by a virtual machine
- **String Encryption** - Encrypt string literals with AES-256-GCM (Python) or XOR (Lua)
- **Number Obfuscation** - Replace numeric constants with arithmetic expressions
- **Constant Array Obfuscation** - Shuffle array elements and rewrite access patterns
- Configurable obfuscation strategies
- Focus on testability and extensibility

**Technology stack:**

- **Language:** Python 3.9+
- **GUI Framework:** PyQt6
- **Lua Parser:** luaparser
- **Testing:** pytest
- **Code Quality:** black, flake8, mypy

## Project Status & Roadmap

🚧 **Under Development** – The project is in an early alpha stage. The
architecture, utilities, and test infrastructure are being put in place before
the full GUI and obfuscation engine are implemented.

Planned milestones include:

- Core obfuscation engine (AST transforms and code generation)
- Initial PyQt6 GUI for selecting input/output files and obfuscation options
- Configuration system for reusable obfuscation profiles
- Plugin system for custom obfuscation strategies

## Prerequisites

- **Python:** 3.9 or newer (3.9–3.12 are targeted)
- **Operating systems:**
  - Windows 10 or newer
  - macOS 10.13 (High Sierra) or newer
  - Most modern Linux distributions
- **Tools:**
  - Git (for cloning the repository)
  - A C compiler toolchain (optional, only if you install additional native
    dependencies)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/python-obfuscator.git
cd python-obfuscator
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux (bash/zsh):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Runtime dependencies:

```bash
pip install -r requirements.txt
```

Development dependencies:

```bash
pip install -r requirements-dev.txt
```

### 4. Verify your environment

You can run the dependency verification script to ensure that versions and
platform-specific requirements are satisfied:

```bash
python scripts/verify_dependencies.py
```

## Quick Start

The main GUI entry point is still under active development. The example below
illustrates how the project is expected to be used once the GUI wiring is
complete (command may change):

```bash
# Placeholder – actual entry point will be documented once implemented
python -m obfuscator
```

In the meantime, you can run the test suite to validate your environment and
the core utilities:

```bash
pytest
```

## Obfuscation Features

### VM Protection

VM Protection converts functions to custom bytecode that is executed by a virtual machine, providing strong obfuscation:

```python
# Original code
@vm_protect
def sensitive_calculation(x, y):
    result = x * 2 + y
    return result

# Obfuscated code (conceptual)
def sensitive_calculation(*args):
    __bytecode = [1, 2, 3, ...]  # Encoded bytecode
    __constants = [2, ...]       # Constant pool
    __num_locals = 3
    return execute_protected_function(__bytecode, __constants, __num_locals, *args)
```

**Configuration:**
- `vm_protection_complexity`: 1-3 (basic to advanced instruction set)
- `vm_protect_all_functions`: Protect all functions automatically
- `vm_bytecode_encryption`: Enable bytecode encryption
- `vm_protection_marker`: Decorator/comment marker (default: "vm:protect")

### String Encryption

Encrypt string literals to hide them from static analysis:

```python
# Before
message = "secret_key"

# After
message = _decrypt_string("encrypted_string_data")
```

**Configuration:**
- `string_encryption_key_length`: 16, 24, or 32 bytes

### Number Obfuscation

Replace numeric constants with equivalent arithmetic expressions:

```python
# Before
value = 42

# After
value = (21 * 2) + 0  # Evaluates to 42
```

**Configuration:**
- `number_obfuscation_complexity`: 1-5 (simple to advanced expressions)
- `number_obfuscation_min_value`: Minimum value to obfuscate
- `number_obfuscation_max_value`: Maximum value to obfuscate

### Constant Array Obfuscation

Shuffle array elements and rewrite access patterns:

```python
# Before
data = [1, 2, 3, 4, 5]
result = data[2]  # Returns 3

# After
data = [3, 5, 1, 4, 2]  # Shuffled
result = data[_arr_0_map[2]]  # Still returns 3
```

**Configuration:**
- `array_shuffle_seed`: Seed for deterministic shuffling (None for random)

## Project Structure

A high-level view of the repository layout:

```text
.
├─ src/
│  └─ obfuscator/
│     ├─ processors/          # AST transformers and obfuscation logic
│     │  ├─ ast_transformer.py
│     │  ├─ vm_bytecode.py
│     │  ├─ vm_runtime_python.py
│     │  └─ vm_runtime_lua.py
│     ├─ core/                # Core orchestration and configuration
│     │  ├─ config.py
│     │  └─ orchestrator.py
│     └─ utils/               # Reusable utilities
│        ├─ path_utils.py
│        └─ logger.py
├─ tests/                     # Pytest-based unit and integration tests
├─ docs/                      # Project documentation
│  ├─ dependencies.md
│  └─ architecture.md
├─ scripts/                   # Helper scripts
├─ pyproject.toml
└─ README.md
```

- **src/obfuscator/processors/** – AST transformers and obfuscation logic
  - `ast_transformer.py` - VM protection, string encryption, number obfuscation, constant array transformers
  - `vm_bytecode.py` - Bytecode instruction set and compilers for Python/Lua
  - `vm_runtime_python.py` - Python VM interpreter runtime generator
  - `vm_runtime_lua.py` - Lua VM interpreter runtime generator
- **src/obfuscator/core/** – Core orchestration and configuration
- **src/obfuscator/utils/** – Reusable utilities (path handling, logging, etc.)
- **tests/** – Pytest-based unit and integration tests
- **docs/** – Project documentation, including dependencies and architecture
- **scripts/** – Helper scripts (e.g., dependency verification)

For a deeper dive into the internals, see
[`docs/architecture.md`](docs/architecture.md).

## Documentation

- **Architecture:** [`docs/architecture.md`](docs/architecture.md)
- **Dependencies and environment notes:**
  [`docs/dependencies.md`](docs/dependencies.md)
- **Contributing guide:** [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Troubleshooting

- **Virtual environment issues**
  - Ensure you are using the correct Python version (3.9+)
  - On Windows, run PowerShell or cmd **as Administrator** if you encounter
    execution policy errors when activating `.venv`

- **Dependency installation problems**
  - Upgrade pip first: `python -m pip install --upgrade pip`
  - Confirm you are in the activated virtual environment before installing
    packages

- **PyQt6 / Linux / Wayland quirks**
  - Some combinations of desktop environments and display servers (especially
    Wayland) may require additional environment variables or platform plugins.
  - See [`docs/dependencies.md`](docs/dependencies.md) for up-to-date notes on
    PyQt6 and platform-specific caveats.

- **Issues and bugs**
  - If you run into problems, please open an issue on the project's issue
    tracker (for example:
    `https://github.com/<your-username>/python-obfuscator/issues`). Include
    your OS, Python version, and a minimal reproduction if possible.

## License

This project is licensed under the **MIT License**. See
[`LICENSE`](LICENSE) for details.
