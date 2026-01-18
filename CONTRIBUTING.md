# Contributing to Obfuscator

Thank you for taking the time to contribute to **Obfuscator**. This project aims
to provide a cross-platform, GUI-based Lua code obfuscator with a strong focus
on code quality, testability, and extensibility.

This guide explains how to set up a development environment, the preferred
workflow, and the standards expected for code, tests, and documentation.

## Getting Started

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/<your-username>/python-obfuscator.git
   cd python-obfuscator
   ```

2. **Create and activate a virtual environment** (see details in
   [`README.md`](README.md)).

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Verify your environment**
   ```bash
   python scripts/verify_dependencies.py
   ```

5. **Run the tests** to ensure everything passes before you start making
   changes:
   ```bash
   pytest
   ```

For a deeper understanding of the internal design, read
[`docs/architecture.md`](docs/architecture.md).

## Development Workflow

- Create a branch for your work:
  - Features: `feature/<short-description>`
  - Bug fixes: `bugfix/<issue-id-or-short-description>`
  - Docs-only changes: `docs/<topic>`

- Use **small, focused commits** that do one thing well.

- Prefer **Conventional Commits**-style messages where possible, e.g.:
  - `feat(gui): add basic project open dialog`
  - `fix(utils): handle tilde expansion on Windows`
  - `docs: document architecture and utilities`

- Open a **draft pull request** early if you want feedback on direction.

- Ensure your branch is up to date with the main branch before requesting
  review.

## Code Style Guidelines

- Follow **PEP 8** as the baseline coding standard.

- Use **type hints** throughout the codebase. Existing utility modules such as
  `src/obfuscator/utils/path_utils.py` should be used as a reference for style
  and typing.

- Format code with **black**:
  ```bash
  black src/ tests/
  ```

- Lint code with **flake8**:
  ```bash
  flake8 src/ tests/
  ```

- Keep functions and classes **small and focused**. Prefer clear, descriptive
  names over abbreviations.

- Use **Google-style docstrings** (or the existing project convention) for all
  public modules, classes, and functions. See
  `src/obfuscator/utils/logger.py` and other utils for examples.

## Testing Standards

- All new features and bug fixes should include **unit tests**.

- Place tests in the `tests/` directory, following existing patterns, e.g.:
  - `tests/utils/test_path_utils.py`

- Test discovery is configured in `pyproject.toml` to look for:
  - Files: `test_*.py`
  - Classes: `Test*`
  - Functions: `test_*`

- Run the full test suite locally before opening a pull request:
  ```bash
  pytest
  ```

- Aim to keep or improve coverage. Coverage configuration is defined under the
  `[tool.pytest.ini_options]` and `[tool.coverage.*]` sections in
  `pyproject.toml`.

## Documentation Standards

- Update or add docstrings for any new public functions, classes, or modules.

- When changing behavior that is user-visible or architectural, update the
  relevant documentation:
  - `README.md` for high-level usage and setup
  - `docs/architecture.md` for design and structure
  - `docs/dependencies.md` for dependency-related changes

- Keep examples in docstrings simple and focused.

## Pull Request Checklist

Before marking your pull request as ready for review, please verify that:

- [ ] Code follows the style guidelines (PEP 8, black, flake8)
- [ ] Type hints are added/updated where appropriate
- [ ] Tests are added or updated for all code changes
- [ ] All tests pass locally (`pytest`)
- [ ] Relevant documentation is updated (`README.md`, `docs/*`, docstrings)
- [ ] The change set is focused and well-scoped

## Reporting Issues

When opening an issue (bug report or feature request), please include:

- A clear, descriptive title
- A concise description of the problem or request
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Your environment details:
  - OS and version (e.g. Windows 11, Ubuntu 24.04, macOS Sonoma)
  - Python version (`python --version`)
  - Relevant dependency versions (see `docs/dependencies.md`)

## Code of Conduct

Please be respectful and professional in all interactions. We aim to foster an
inclusive, welcoming environment for contributors of all backgrounds and
experience levels.

Harassment, discrimination, or personal attacks are not tolerated. If you
encounter behavior that violates these principles, please contact a project
maintainer privately or open an issue describing the concern.
