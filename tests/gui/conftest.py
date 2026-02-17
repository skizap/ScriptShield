"""Shared fixtures and helpers for GUI progress widget integration tests."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication

from obfuscator.core.orchestrator import JobState, ProgressInfo


def _load_module_from_path(module_name: str, module_path: Path):
    """Load a module by absolute path without traversing package __init__ files."""
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {module_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_progress_widget_dependencies() -> tuple[type, Callable[[str], str]]:
    """Load ``ProgressWidget`` and ``get_widget_style`` from source files directly."""
    repo_root = Path(__file__).resolve().parents[2]
    stylesheet_path = repo_root / "src" / "obfuscator" / "gui" / "styles" / "stylesheet.py"
    progress_widget_path = repo_root / "src" / "obfuscator" / "gui" / "widgets" / "progress_widget.py"

    # Create lightweight namespace packages to satisfy absolute imports in module code.
    if "obfuscator.gui" not in sys.modules:
        gui_pkg = types.ModuleType("obfuscator.gui")
        gui_pkg.__path__ = []
        sys.modules["obfuscator.gui"] = gui_pkg
    if "obfuscator.gui.styles" not in sys.modules:
        styles_pkg = types.ModuleType("obfuscator.gui.styles")
        styles_pkg.__path__ = []
        sys.modules["obfuscator.gui.styles"] = styles_pkg
    if "obfuscator.gui.widgets" not in sys.modules:
        widgets_pkg = types.ModuleType("obfuscator.gui.widgets")
        widgets_pkg.__path__ = []
        sys.modules["obfuscator.gui.widgets"] = widgets_pkg

    stylesheet_module = _load_module_from_path(
        "obfuscator.gui.styles.stylesheet",
        stylesheet_path,
    )
    progress_widget_module = _load_module_from_path(
        "obfuscator.gui.widgets.progress_widget",
        progress_widget_path,
    )

    return progress_widget_module.ProgressWidget, stylesheet_module.get_widget_style


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Provide a QApplication instance for all GUI widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def progress_widget_class() -> type:
    """Expose the dynamically loaded ``ProgressWidget`` class."""
    progress_widget_type, _ = _load_progress_widget_dependencies()
    return progress_widget_type


@pytest.fixture
def widget_style_getter() -> Callable[[str], str]:
    """Expose the stylesheet getter used by ProgressWidget rendering tests."""
    _, style_getter = _load_progress_widget_dependencies()
    return style_getter


@pytest.fixture
def progress_widget(qapp: QApplication, progress_widget_class: type) -> Any:
    """Create a fresh ProgressWidget for each test."""
    _ = qapp
    widget = progress_widget_class()
    yield widget
    widget.hide()
    widget.deleteLater()


@pytest.fixture
def mock_orchestrator() -> MagicMock:
    """Return a lightweight orchestrator mock for signal/callback testing."""
    orchestrator = MagicMock()
    orchestrator.request_cancellation = MagicMock()
    return orchestrator


@pytest.fixture
def progress_info_factory():
    """Build ``ProgressInfo`` instances with concise defaults for tests."""

    def _factory(
        *,
        current_file: str | None = None,
        current_index: int = 1,
        total_files: int = 8,
        percentage: float = 0.0,
        elapsed_seconds: float = 0.0,
        estimated_remaining_seconds: float | None = None,
        current_state: JobState = JobState.PENDING,
        message: str = "Progress update",
        log_level: str = "info",
    ) -> ProgressInfo:
        return ProgressInfo(
            current_file=current_file,
            current_index=current_index,
            total_files=total_files,
            percentage=percentage,
            elapsed_seconds=elapsed_seconds,
            estimated_remaining_seconds=estimated_remaining_seconds,
            current_state=current_state,
            message=message,
            log_level=log_level,
        )

    return _factory
