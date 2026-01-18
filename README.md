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
- Configurable obfuscation strategies (planned)
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

## Project Structure

A high-level view of the repository layout:

```text
.
├─ src/
│  └─ obfuscator/
│     └─ utils/
├─ tests/
├─ docs/
│  ├─ dependencies.md
│  └─ architecture.md
├─ scripts/
├─ pyproject.toml
└─ README.md
```

- **src/obfuscator/** – Main package containing application code
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
  - If you run into problems, please open an issue on the project’s issue
    tracker (for example:
    `https://github.com/<your-username>/python-obfuscator/issues`). Include
    your OS, Python version, and a minimal reproduction if possible.

## License

This project is licensed under the **MIT License**. See
[`LICENSE`](LICENSE) for details.
