"""Custom exception and warning types for the obfuscator core."""

from __future__ import annotations

from pathlib import Path


class UnsupportedFeatureWarning(Warning):
    """Structured warning for unsupported non-fatal language features."""

    def __init__(
        self,
        feature_name: str,
        file_path: Path | str,
        line_number: int = 0,
        column_offset: int = 0,
        message: str = "",
        suggestion: str | None = None,
    ) -> None:
        self.feature_name = feature_name
        self.file_path = Path(file_path)
        self.line_number = line_number
        self.column_offset = column_offset
        self.message = message
        self.suggestion = suggestion
        super().__init__(message)

    def __str__(self) -> str:
        """Return a canonical warning string with source location context."""
        return (
            f"{self.file_path}:{self.line_number}:{self.column_offset}: "
            f"UnsupportedFeature: {self.feature_name} - {self.message}"
        )
