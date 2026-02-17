"""Utilities for structured obfuscation error formatting and parsing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ERROR_FORMAT = "{file_path}:{line}:{column}: {error_type}: {message}"
"""Canonical error format used across processors, engine, and GUI."""

_STRUCTURED_ERROR_PATTERN = re.compile(
    r"^(?P<file_path>.+):(?P<line>\d+):(?P<column>\d+):\s*(?P<error_type>[^:]+):\s*(?P<message>.*)$"
)

_LINE_COLUMN_PATTERN = re.compile(
    r"\bline\s+(?P<line>\d+)\s*,\s*column\s+(?P<column>\d+)",
    flags=re.IGNORECASE,
)
_LINE_PATTERN = re.compile(r"\bline\s+(?P<line>\d+)", flags=re.IGNORECASE)
_COLUMN_PATTERN = re.compile(r"\bcolumn\s+(?P<column>\d+)", flags=re.IGNORECASE)


def _normalize_location_value(value: int | None) -> int:
    """Normalize optional location values to non-negative integers."""
    if value is None:
        return 0

    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return 0

    return max(0, numeric_value)


def format_error(
    file_path: Path | str,
    line: int | None,
    column: int | None,
    error_type: str,
    message: str,
) -> str:
    """Format an error using the canonical structured error format."""
    return ERROR_FORMAT.format(
        file_path=str(file_path),
        line=_normalize_location_value(line),
        column=_normalize_location_value(column),
        error_type=str(error_type),
        message=str(message),
    )


def parse_error(error_string: str) -> dict[str, Any] | None:
    """Parse a structured error string into components.

    Returns None if the input does not match the canonical ERROR_FORMAT.
    """
    if not error_string:
        return None

    match = _STRUCTURED_ERROR_PATTERN.match(error_string.strip())
    if not match:
        return None

    return {
        "file_path": match.group("file_path"),
        "line": int(match.group("line")),
        "column": int(match.group("column")),
        "error_type": match.group("error_type").strip(),
        "message": match.group("message").strip(),
    }


def extract_line_column(error_string: str) -> tuple[int | None, int | None]:
    """Extract line/column from structured or natural-language error text."""
    parsed = parse_error(error_string)
    if parsed is not None:
        return parsed.get("line"), parsed.get("column")

    line_column_match = _LINE_COLUMN_PATTERN.search(error_string)
    if line_column_match:
        return (
            int(line_column_match.group("line")),
            int(line_column_match.group("column")),
        )

    line_match = _LINE_PATTERN.search(error_string)
    column_match = _COLUMN_PATTERN.search(error_string)

    line = int(line_match.group("line")) if line_match else None
    column = int(column_match.group("column")) if column_match else None

    return line, column


def has_location_info(error_string: str) -> bool:
    """Return True when an error string contains line and/or column information."""
    line, column = extract_line_column(error_string)
    return line is not None or column is not None
