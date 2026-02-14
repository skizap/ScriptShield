"""Output writer module for atomic file writing with conflict resolution.

This module provides the ``OutputWriter`` class that centralizes all file
writing operations with atomic guarantees, comprehensive error handling,
and detailed logging.  It leverages temporary files with atomic rename
operations to ensure write safety and implements all conflict resolution
strategies (OVERWRITE, SKIP, RENAME, ASK) with timestamp-based renaming.

The writer tracks metadata for every write operation and supports both
single-file and batch writing modes, as well as directory-structure-preserving
writes for multi-file projects.

Example:
    Basic single-file write::

        from pathlib import Path
        from obfuscator.core.output_writer import OutputWriter
        from obfuscator.core.orchestrator import ConflictStrategy

        writer = OutputWriter(
            output_dir=Path("./obfuscated"),
            conflict_strategy=ConflictStrategy.OVERWRITE,
        )
        result = writer.write_file(
            output_path=Path("./obfuscated/main.py"),
            content="print('hello')",
        )
        if result.success:
            print(f"Written to {result.output_path}")

    Batch write with structure preservation::

        results = writer.write_files([
            (Path("out/a.py"), "code_a", Path("src/a.py")),
            (Path("out/b.py"), "code_b", Path("src/b.py")),
        ])
        meta = writer.get_metadata()
        print(f"{meta.successful_writes}/{meta.total_writes} succeeded")

    Preserving directory structure::

        result = writer.write_with_structure(
            input_path=Path("src/pkg/module.py"),
            output_base=Path("./obfuscated"),
            content="obfuscated_code",
            project_root=Path("src"),
        )
        # Writes to ./obfuscated/pkg/module.py

    Generate and write summary reports::

        report_text = writer.get_summary_report(format="text")
        print(report_text)

        report_result = writer.write_summary_report(format="json")
        if report_result.success:
            print(f"Summary written to {report_result.output_path}")
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from obfuscator.core.orchestrator import ConflictStrategy
from obfuscator.utils.path_utils import (
    normalize_path,
    ensure_directory,
    is_writable,
    get_relative_path,
)
from obfuscator.utils.logger import get_logger

if TYPE_CHECKING:
    from obfuscator.core.runtime_manager import RuntimeManager


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WriteResult:
    """Result of a single file write operation.

    Attributes:
        success: Whether the write operation succeeded.
        output_path: Final path the file was written to, or ``None`` if the
            write was skipped (e.g. SKIP conflict strategy).
        original_path: The originally requested output path before any
            conflict resolution renaming.
        conflict_resolution: A short label describing the resolution that was
            applied: ``"overwritten"``, ``"skipped"``, or ``"renamed"``.
            ``None`` when there was no conflict.
        error: Human-readable error message if the write failed,
            otherwise ``None``.
        was_atomic: ``True`` if the write used the atomic temp-file strategy,
            ``False`` if it fell back to a direct write.
    """

    success: bool
    output_path: Path | None
    original_path: Path
    conflict_resolution: str | None = None
    error: str | None = None
    was_atomic: bool = False


@dataclass
class WriteMetadata:
    """Aggregated statistics for all write operations performed by an
    ``OutputWriter`` instance.

    Attributes:
        total_writes: Total number of write attempts.
        successful_writes: Number of writes that completed successfully.
        failed_writes: Number of writes that failed.
        skipped_writes: Number of writes skipped due to conflict resolution.
        renamed_writes: Number of writes that used a renamed output path.
        overwritten_writes: Number of writes that overwrote existing files.
        written_files: Ordered list of paths that were successfully written.
        runtime_libraries_written: Number of runtime library files written.
        runtime_library_paths: Ordered list of written runtime library paths.
        start_time: Unix timestamp when write tracking started.
        end_time: Unix timestamp when summary/reporting was last generated.
        warnings: Collected warning messages for reporting.
        file_results: Ordered list of all individual write outcomes.
    """

    total_writes: int = 0
    successful_writes: int = 0
    failed_writes: int = 0
    skipped_writes: int = 0
    renamed_writes: int = 0
    overwritten_writes: int = 0
    written_files: list[Path] = field(default_factory=list)
    runtime_libraries_written: int = 0
    runtime_library_paths: list[Path] = field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None
    warnings: list[str] = field(default_factory=list)
    file_results: list[WriteResult] = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        """Return elapsed seconds between start and end timestamps."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return max(0.0, self.end_time - self.start_time)

    @property
    def formatted_elapsed(self) -> str:
        """Return elapsed time as ``MM:SS``."""
        minutes = int(self.elapsed_seconds) // 60
        seconds = int(self.elapsed_seconds) % 60
        return f"{minutes:02d}:{seconds:02d}"


# ---------------------------------------------------------------------------
# Supported extensions for validation
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS: set[str] = {".py", ".pyw", ".lua", ".luau"}


# ---------------------------------------------------------------------------
# OutputWriter
# ---------------------------------------------------------------------------

class OutputWriter:
    """Centralised file writer with atomic operations and conflict resolution.

    ``OutputWriter`` wraps every disk write in a safe workflow:

    1. Validate permissions on the target path.
    2. Detect and resolve conflicts using the configured
       :class:`ConflictStrategy`.
    3. Write content via an atomic *temp-file + rename* pattern (with a
       direct-write fallback).
    4. Record the outcome in :class:`WriteMetadata`.

    Summary reports can be generated in text or JSON format using
    :meth:`generate_summary_report`, :meth:`get_summary_report`, and
    :meth:`write_summary_report`. Metadata ``end_time`` is set when a
    summary report is generated.

    Attributes:
        output_dir: Base output directory (normalised absolute path).
        conflict_strategy: Active conflict resolution strategy.
        use_atomic_writes: Whether to prefer atomic writes.

    Args:
        output_dir: Base output directory.  Will be normalised and its
            parent validated.
        conflict_strategy: Strategy for handling existing files
            (default :attr:`ConflictStrategy.ASK`).
        use_atomic_writes: If ``True`` (default), use temp-file + rename;
            fall back to direct write on failure.
        conflict_callback: Callable invoked when ``ConflictStrategy.ASK``
            encounters a conflict.  Receives the conflicting ``Path`` and
            must return a :class:`ConflictStrategy`.

    Raises:
        ValueError: If *output_dir* is ``None`` or empty.
        OSError: If *output_dir* parent cannot be created.

    Example::

        writer = OutputWriter(
            output_dir=Path("./build"),
            conflict_strategy=ConflictStrategy.RENAME,
        )
        result = writer.write_file(Path("./build/app.py"), code)
    """

    def __init__(
        self,
        output_dir: Path,
        conflict_strategy: ConflictStrategy = ConflictStrategy.ASK,
        use_atomic_writes: bool = True,
        conflict_callback: Callable[[Path], ConflictStrategy] | None = None,
    ) -> None:
        self._logger = get_logger("obfuscator.core.output_writer")
        self.output_dir: Path = normalize_path(output_dir)
        self.conflict_strategy: ConflictStrategy = conflict_strategy
        self.use_atomic_writes: bool = use_atomic_writes
        self._conflict_callback = conflict_callback
        self._metadata: WriteMetadata = WriteMetadata()
        self._metadata.start_time = time.time()
        self._conflict_decisions: dict[Path, str] = {}

        # Ensure the output directory (or its parent) can be created
        if not self.output_dir.exists():
            parent = self.output_dir.parent
            if not parent.exists():
                try:
                    ensure_directory(parent)
                except OSError:
                    self._logger.error(
                        f"Cannot create parent directory: {parent}"
                    )
                    raise

        self._logger.debug(
            f"OutputWriter initialised: dir={self.output_dir}, "
            f"strategy={self.conflict_strategy.value}, "
            f"atomic={self.use_atomic_writes}"
        )

    def _record_result(self, result: WriteResult) -> WriteResult:
        """Store a ``WriteResult`` in metadata and return it unchanged."""
        self._metadata.file_results.append(result)
        return result

    def _record_warning(self, message: str) -> None:
        """Store warning messages for summary report generation."""
        self._metadata.warnings.append(message)

    # ------------------------------------------------------------------
    # Permission validation
    # ------------------------------------------------------------------

    def _validate_write_permissions(
        self, output_path: Path
    ) -> tuple[bool, str | None]:
        """Check whether *output_path* (or its parent) is writable.

        Args:
            output_path: Target file path to validate.

        Returns:
            A ``(is_valid, error_message)`` tuple.  *error_message* is
            ``None`` when the path is valid.
        """
        if output_path.exists():
            if not is_writable(output_path):
                msg = f"File is not writable: {output_path}"
                self._logger.warning(msg)
                self._record_warning(msg)
                return False, msg
        else:
            parent = output_path.parent
            if parent.exists():
                if not is_writable(parent):
                    msg = f"Parent directory is not writable: {parent}"
                    self._logger.warning(msg)
                    self._record_warning(msg)
                    return False, msg
            # Parent doesn't exist yet – we'll try to create it later
        return True, None

    # ------------------------------------------------------------------
    # Conflict resolution
    # ------------------------------------------------------------------

    def _resolve_conflict(
        self, output_path: Path
    ) -> tuple[Path | None, str]:
        """Resolve an existing-file conflict for *output_path*.

        Args:
            output_path: The conflicting output path.

        Returns:
            ``(resolved_path, resolution_type)`` where *resolved_path* is
            ``None`` for the SKIP strategy.

        Raises:
            ValueError: If ``ConflictStrategy.ASK`` is active but no
                callback is available and no cached decision exists.
        """
        strategy = self.conflict_strategy

        if strategy == ConflictStrategy.OVERWRITE:
            self._logger.info(
                f"Conflict resolved for {output_path.name}: OVERWRITE"
            )
            return output_path, "overwritten"

        if strategy == ConflictStrategy.SKIP:
            self._logger.info(
                f"Conflict resolved for {output_path.name}: SKIP"
            )
            return None, "skipped"

        if strategy == ConflictStrategy.RENAME:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = output_path.stem
            suffix = output_path.suffix
            new_name = f"{stem}_{timestamp}{suffix}"
            new_path = output_path.parent / new_name

            # Handle the unlikely collision where the renamed file exists
            counter = 1
            while new_path.exists():
                new_name = f"{stem}_{timestamp}_{counter}{suffix}"
                new_path = output_path.parent / new_name
                counter += 1

            self._logger.info(
                f"Conflict resolved for {output_path.name}: "
                f"RENAME -> {new_name}"
            )
            return new_path, "renamed"

        if strategy == ConflictStrategy.ASK:
            # Check decision cache first
            if output_path in self._conflict_decisions:
                cached = self._conflict_decisions[output_path]
                self._logger.info(
                    f"Using cached conflict decision for "
                    f"{output_path.name}: {cached}"
                )
                # Re-resolve with the cached strategy label
                if cached == "overwritten":
                    return output_path, "overwritten"
                if cached == "skipped":
                    return None, "skipped"
                # For renamed, generate a fresh name
                return self._resolve_conflict_as_rename(output_path)

            if self._conflict_callback is not None:
                decision = self._conflict_callback(output_path)
                # Delegate to the chosen strategy and cache
                prev_strategy = self.conflict_strategy
                self.conflict_strategy = decision
                try:
                    resolved_path, resolution = self._resolve_conflict(
                        output_path
                    )
                finally:
                    self.conflict_strategy = prev_strategy
                self._conflict_decisions[output_path] = resolution
                return resolved_path, resolution

            raise ValueError(
                f"ConflictStrategy.ASK requires a conflict_callback, but "
                f"none was provided and no cached decision exists for "
                f"{output_path}"
            )

        # Unreachable for known strategies, but be defensive
        self._logger.warning(
            f"Unknown conflict strategy {strategy}; defaulting to OVERWRITE"
        )
        return output_path, "overwritten"

    def _resolve_conflict_as_rename(
        self, output_path: Path
    ) -> tuple[Path, str]:
        """Helper to generate a renamed path (used by cached ASK decisions).

        Args:
            output_path: The conflicting output path.

        Returns:
            ``(new_path, "renamed")``
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = output_path.stem
        suffix = output_path.suffix
        new_name = f"{stem}_{timestamp}{suffix}"
        new_path = output_path.parent / new_name

        counter = 1
        while new_path.exists():
            new_name = f"{stem}_{timestamp}_{counter}{suffix}"
            new_path = output_path.parent / new_name
            counter += 1

        self._logger.info(
            f"Conflict resolved for {output_path.name}: RENAME -> {new_name}"
        )
        return new_path, "renamed"

    # ------------------------------------------------------------------
    # Low-level writers
    # ------------------------------------------------------------------

    def _write_atomic(self, output_path: Path, content: str) -> bool:
        """Write *content* to *output_path* atomically.

        The strategy is:
        1. Write to a temporary file **in the same directory** (so that
           ``shutil.move`` is a same-filesystem rename).
        2. Flush and ``fsync`` to guarantee durability.
        3. Rename the temp file to *output_path*.

        Args:
            output_path: Destination file path.
            content: Text content to write.

        Returns:
            ``True`` on success.

        Raises:
            OSError: On file-system errors (propagated after cleanup).
        """
        ensure_directory(output_path.parent)
        temp_path: str | None = None

        try:
            fd = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=str(output_path.parent),
                suffix=output_path.suffix,
            )
            temp_path = fd.name
            self._logger.debug(
                f"Atomic write: temp file created at {temp_path}"
            )

            fd.write(content)
            fd.flush()
            os.fsync(fd.fileno())
            fd.close()

            shutil.move(temp_path, str(output_path))
            self._logger.debug(
                f"Atomic write: renamed {temp_path} -> {output_path}"
            )
            return True

        except (OSError, IOError) as exc:
            self._logger.error(
                f"Atomic write failed for {output_path}: {exc}"
            )
            # Clean up the temporary file if it still exists
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    self._logger.error(
                        f"Failed to clean up temp file: {temp_path}"
                    )
            raise

    def _write_direct(self, output_path: Path, content: str) -> bool:
        """Write *content* directly to *output_path* (non-atomic fallback).

        Args:
            output_path: Destination file path.
            content: Text content to write.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        try:
            ensure_directory(output_path.parent)
            output_path.write_text(content, encoding="utf-8")
            self._logger.debug(f"Direct write succeeded: {output_path}")
            return True
        except (OSError, IOError) as exc:
            self._logger.error(
                f"Direct write failed for {output_path}: {exc}"
            )
            return False

    # ------------------------------------------------------------------
    # Extension validation
    # ------------------------------------------------------------------

    def _validate_extension(
        self, file_path: Path
    ) -> tuple[bool, str]:
        """Validate that *file_path* has a supported extension.

        Args:
            file_path: Path to validate.

        Returns:
            ``(is_valid, language)`` where *language* is ``"python"``,
            ``"lua"``, or ``"unknown"``.
        """
        ext = file_path.suffix.lower()
        if ext in {".py", ".pyw"}:
            return True, "python"
        if ext in {".lua", ".luau"}:
            return True, "lua"
        warning_message = (
            f"Unsupported file extension '{ext}' for: {file_path}"
        )
        self._logger.warning(warning_message)
        self._record_warning(warning_message)
        return False, "unknown"

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def write_file(
        self,
        output_path: Path,
        content: str,
        input_path: Path | None = None,
    ) -> WriteResult:
        """Write *content* to *output_path* with conflict resolution.

        This is the primary entry point for writing a single file.  It
        validates permissions, resolves conflicts, and writes using the
        atomic strategy (with a direct-write fallback).

        Args:
            output_path: Desired output file path.
            content: Text content to write.
            input_path: Optional source file path (for logging context).

        Returns:
            A :class:`WriteResult` describing the outcome.

        Example::

            result = writer.write_file(Path("out/main.py"), code)
            if result.success:
                print(f"Wrote {result.output_path}")
            else:
                print(f"Error: {result.error}")
        """
        self._metadata.total_writes += 1
        original_path = output_path

        try:
            output_path = normalize_path(output_path)
        except ValueError as exc:
            self._metadata.failed_writes += 1
            return self._record_result(
                WriteResult(
                    success=False,
                    output_path=None,
                    original_path=original_path,
                    error=f"Invalid output path: {exc}",
                )
            )

        # --- extension validation ---
        ext_valid, language = self._validate_extension(output_path)
        if not ext_valid:
            self._metadata.failed_writes += 1
            return self._record_result(
                WriteResult(
                    success=False,
                    output_path=None,
                    original_path=original_path,
                    error=(
                        f"Unsupported file extension '{output_path.suffix}' "
                        f"for: {output_path}. "
                        f"Supported: .py, .pyw, .lua, .luau"
                    ),
                )
            )

        # --- permission check ---
        valid, perm_error = self._validate_write_permissions(output_path)
        if not valid:
            self._metadata.failed_writes += 1
            return self._record_result(
                WriteResult(
                    success=False,
                    output_path=None,
                    original_path=original_path,
                    error=perm_error,
                )
            )

        # --- conflict resolution ---
        conflict_resolution: str | None = None
        if output_path.exists():
            try:
                resolved, conflict_resolution = self._resolve_conflict(
                    output_path
                )
            except ValueError as exc:
                self._metadata.failed_writes += 1
                return self._record_result(
                    WriteResult(
                        success=False,
                        output_path=None,
                        original_path=original_path,
                        error=str(exc),
                    )
                )

            if resolved is None:
                # SKIP strategy
                self._metadata.skipped_writes += 1
                self._logger.info(
                    f"Skipped writing {original_path.name} (conflict: SKIP)"
                )
                return self._record_result(
                    WriteResult(
                        success=True,
                        output_path=None,
                        original_path=original_path,
                        conflict_resolution="skipped",
                    )
                )

            output_path = resolved

        # --- write ---
        was_atomic = False
        try:
            if self.use_atomic_writes:
                try:
                    self._write_atomic(output_path, content)
                    was_atomic = True
                except (OSError, IOError):
                    warning_message = (
                        f"Atomic write failed for {output_path.name}, "
                        "falling back to direct write"
                    )
                    self._logger.warning(warning_message)
                    self._record_warning(warning_message)
                    if not self._write_direct(output_path, content):
                        raise OSError(
                            f"Both atomic and direct writes failed for "
                            f"{output_path}"
                        )
            else:
                if not self._write_direct(output_path, content):
                    raise OSError(
                        f"Direct write failed for {output_path}"
                    )
        except (OSError, IOError, PermissionError) as exc:
            self._metadata.failed_writes += 1
            return self._record_result(
                WriteResult(
                    success=False,
                    output_path=None,
                    original_path=original_path,
                    conflict_resolution=conflict_resolution,
                    error=str(exc),
                )
            )

        # --- bookkeeping ---
        self._metadata.successful_writes += 1
        self._metadata.written_files.append(output_path)

        if conflict_resolution == "overwritten":
            self._metadata.overwritten_writes += 1
        elif conflict_resolution == "renamed":
            self._metadata.renamed_writes += 1

        input_label = f" (source: {input_path})" if input_path else ""
        self._logger.info(
            f"Wrote {output_path}{input_label} "
            f"[atomic={was_atomic}, conflict={conflict_resolution}]"
        )

        return self._record_result(
            WriteResult(
                success=True,
                output_path=output_path,
                original_path=original_path,
                conflict_resolution=conflict_resolution,
                was_atomic=was_atomic,
            )
        )

    def write_files(
        self,
        files: list[tuple[Path, str, Path | None]],
    ) -> list[WriteResult]:
        """Write multiple files in sequence.

        Each element of *files* is a ``(output_path, content, input_path)``
        tuple.  Processing continues even if individual writes fail.

        Args:
            files: List of ``(output_path, content, input_path)`` tuples.
                *input_path* may be ``None``.

        Returns:
            A list of :class:`WriteResult` objects in the same order as
            *files*.

        Example::

            results = writer.write_files([
                (Path("out/a.py"), code_a, Path("src/a.py")),
                (Path("out/b.py"), code_b, None),
            ])
            for r in results:
                print(r.success, r.output_path)
        """
        self._logger.info(f"Batch write started: {len(files)} file(s)")
        results: list[WriteResult] = []

        for output_path, content, input_path in files:
            result = self.write_file(output_path, content, input_path)
            results.append(result)

        succeeded = sum(1 for r in results if r.success)
        self._logger.info(
            f"Batch write completed: {succeeded}/{len(files)} succeeded"
        )
        return results

    # ------------------------------------------------------------------
    # Directory-structure preservation
    # ------------------------------------------------------------------

    def write_with_structure(
        self,
        input_path: Path,
        output_base: Path,
        content: str,
        project_root: Path | None = None,
    ) -> WriteResult:
        """Write a file while preserving its directory structure.

        When *project_root* is provided the relative path from
        *project_root* to *input_path* is replicated under *output_base*.
        If the relative path cannot be computed (e.g. different drives on
        Windows) the file is written flat into *output_base*.

        Args:
            input_path: Original source file path.
            output_base: Base output directory.
            content: Text content to write.
            project_root: Optional project root for relative path
                computation.

        Returns:
            A :class:`WriteResult` for the write operation.

        Example::

            result = writer.write_with_structure(
                input_path=Path("src/pkg/mod.py"),
                output_base=Path("./build"),
                content=code,
                project_root=Path("src"),
            )
            # Writes to ./build/pkg/mod.py
        """
        if project_root is not None:
            try:
                relative = get_relative_path(
                    input_path.resolve(), project_root.resolve()
                )
                output_path = output_base / relative
                self._logger.debug(
                    f"Structure preserved: {input_path} -> {output_path}"
                )
            except ValueError:
                # Paths on different drives or incompatible – fall back
                output_path = output_base / input_path.name
                self._logger.debug(
                    f"Relative path failed, using flat structure: "
                    f"{input_path} -> {output_path}"
                )
        else:
            output_path = output_base / input_path.name
            self._logger.debug(
                f"No project root; flat structure: "
                f"{input_path} -> {output_path}"
            )

        return self.write_file(output_path, content, input_path)

    def write_runtime_library(
        self,
        runtime_manager: RuntimeManager,
        language: str,
        runtime_mode: str,
        output_dir: Path | None = None,
    ) -> WriteResult:
        """Write a consolidated hybrid runtime library file for a language.

        This method integrates :class:`RuntimeManager` with :class:`OutputWriter`
        to generate and persist hybrid runtime libraries with the same atomic
        write guarantees and conflict resolution used for obfuscated source files.
        Runtime libraries are only written when ``runtime_mode`` is ``"hybrid"``.

        Args:
            runtime_manager: Runtime manager used to inspect requirements and
                generate consolidated runtime code.
            language: Target language, either ``"python"`` or ``"lua"``.
            runtime_mode: Runtime mode from configuration (``"hybrid"`` or
                ``"embedded"``).
            output_dir: Optional output directory override. Defaults to the
                writer's configured ``output_dir``.

        Returns:
            A :class:`WriteResult` describing whether the runtime library was
            written, skipped, or failed.

        Example::

            from obfuscator.core.runtime_manager import RuntimeManager
            from obfuscator.core.config import ObfuscationConfig

            config = ObfuscationConfig(name="Test", runtime_mode="hybrid")
            runtime_manager = RuntimeManager(config)
            writer = OutputWriter(output_dir=Path("./build"))

            result = writer.write_runtime_library(
                runtime_manager=runtime_manager,
                language="python",
                runtime_mode=config.runtime_mode
            )

            if result.success and result.output_path:
                print(f"Runtime library written to {result.output_path}")
        """
        self._metadata.total_writes += 1
        fallback_original_path = self.output_dir / "obf_runtime"

        try:
            target_dir = output_dir or self.output_dir
            filename_map = {
                "python": "obf_runtime.py",
                "lua": "obf_runtime.lua",
            }
            filename = filename_map.get(language, "obf_runtime")
            output_path = target_dir / filename

            if runtime_mode != "hybrid":
                if runtime_mode == "embedded":
                    error = (
                        "Runtime libraries are only written in hybrid mode, "
                        "not embedded mode"
                    )
                else:
                    error = (
                        "Runtime libraries are only written in hybrid mode, "
                        f"received runtime mode '{runtime_mode}'"
                    )
                self._logger.error(error)
                self._metadata.failed_writes += 1
                return self._record_result(
                    WriteResult(
                        success=False,
                        output_path=None,
                        original_path=output_path,
                        error=error,
                    )
                )

            if language not in ("python", "lua"):
                error = (
                    f"Invalid language: {language}. "
                    "Expected 'python' or 'lua'"
                )
                self._logger.error(error)
                self._metadata.failed_writes += 1
                return self._record_result(
                    WriteResult(
                        success=False,
                        output_path=None,
                        original_path=output_path,
                        error=error,
                    )
                )

            if not runtime_manager.has_runtime_requirements(language):
                self._logger.info(
                    f"No runtime requirements for {language}, "
                    "skipping runtime library generation"
                )
                self._metadata.skipped_writes += 1
                return self._record_result(
                    WriteResult(
                        success=True,
                        output_path=None,
                        original_path=output_path,
                    )
                )

            self._logger.info(
                f"Generating hybrid runtime library for {language}"
            )
            runtime_code = runtime_manager.get_combined_runtime(language)

            if not runtime_code.strip():
                warning_message = (
                    f"Generated runtime code for {language} is empty, "
                    "skipping runtime library generation"
                )
                self._logger.warning(warning_message)
                self._record_warning(warning_message)
                self._metadata.skipped_writes += 1
                return self._record_result(
                    WriteResult(
                        success=True,
                        output_path=None,
                        original_path=output_path,
                    )
                )

            # write_file already manages metadata counters for this path.
            self._metadata.total_writes -= 1
            result = self.write_file(
                output_path=output_path,
                content=runtime_code,
                input_path=None,
            )

            if result.success and result.output_path is not None:
                self._metadata.runtime_libraries_written += 1
                self._metadata.runtime_library_paths.append(result.output_path)
                self._logger.info(
                    f"Hybrid runtime library written: {filename} "
                    f"({len(runtime_code)} chars)"
                )
                if result.conflict_resolution:
                    self._logger.info(
                        "Runtime library conflict resolution "
                        f"({language}): {result.conflict_resolution}"
                    )
            elif result.success and result.conflict_resolution == "skipped":
                self._logger.info(
                    f"Hybrid runtime library write skipped for {language} "
                    "due to conflict strategy"
                )

            return result

        except Exception as exc:
            self._logger.exception(
                f"Failed to write hybrid runtime library for {language}: {exc}"
            )
            self._metadata.failed_writes += 1
            return self._record_result(
                WriteResult(
                    success=False,
                    output_path=None,
                    original_path=locals().get(
                        "output_path", fallback_original_path
                    ),
                    error=str(exc),
                )
            )

    def _validate_report_format(self, format: str) -> None:
        """Validate summary report format argument."""
        if format not in {"text", "json"}:
            raise ValueError(
                "Invalid summary report format. Expected 'text' or 'json', "
                f"got '{format}'"
            )

    def _infer_language_from_path(self, file_path: Path) -> str:
        """Infer source language label from file extension."""
        suffix = file_path.suffix.lower()
        if suffix in {".py", ".pyw"}:
            return "python"
        if suffix in {".lua", ".luau"}:
            return "lua"
        return "unknown"

    def generate_summary_report(
        self, format: str = "text", include_file_details: bool = True
    ) -> str:
        """Generate an obfuscation output summary report.

        Args:
            format: Report format, either ``"text"`` or ``"json"``.
            include_file_details: Include per-file write details when ``True``.

        Returns:
            Rendered summary report content.

        Raises:
            ValueError: If *format* is not ``"text"`` or ``"json"``.
        """
        self._validate_report_format(format)
        self._logger.debug(
            "Generating summary report "
            f"(format={format}, include_file_details={include_file_details})"
        )

        self._metadata.end_time = time.time()

        if self._metadata.start_time is None:
            warning_message = (
                "Summary report generated without a start_time; "
                "elapsed time defaults to 00:00"
            )
            self._logger.warning(warning_message)
            if warning_message not in self._metadata.warnings:
                self._record_warning(warning_message)
        elif self._metadata.end_time < self._metadata.start_time:
            warning_message = (
                "Summary report end_time is earlier than start_time; "
                "elapsed time defaults to 00:00"
            )
            self._logger.warning(warning_message)
            if warning_message not in self._metadata.warnings:
                self._record_warning(warning_message)

        written_files: list[dict[str, str | bool | None]] = []
        failed_files: list[dict[str, str]] = []

        for index, result in enumerate(self._metadata.file_results):
            if not isinstance(result, WriteResult):
                warning_message = (
                    f"Unexpected write result at index {index}; "
                    "skipping invalid entry"
                )
                self._logger.warning(warning_message)
                if warning_message not in self._metadata.warnings:
                    self._record_warning(warning_message)
                continue

            resolved_path = result.output_path or result.original_path
            resolved_path_str = str(resolved_path) if resolved_path else "<unknown>"

            if result.success and result.output_path is not None:
                written_files.append(
                    {
                        "path": str(result.output_path),
                        "original_path": str(result.original_path),
                        "conflict_resolution": result.conflict_resolution,
                        "was_atomic": result.was_atomic,
                    }
                )
            elif not result.success:
                failed_files.append(
                    {
                        "path": resolved_path_str,
                        "error": result.error or "Unknown error",
                    }
                )

        runtime_libraries = [
            {
                "path": str(runtime_path),
                "language": self._infer_language_from_path(runtime_path),
            }
            for runtime_path in self._metadata.runtime_library_paths
            if runtime_path is not None
        ]

        warnings = list(self._metadata.warnings)
        warnings_count = len(warnings)
        summary_payload = {
            "elapsed_seconds": self._metadata.elapsed_seconds,
            "formatted_elapsed": self._metadata.formatted_elapsed,
            "total_files": self._metadata.total_writes,
            "successful_writes": self._metadata.successful_writes,
            "failed_writes": self._metadata.failed_writes,
            "skipped_writes": self._metadata.skipped_writes,
            "renamed_writes": self._metadata.renamed_writes,
            "overwritten_writes": self._metadata.overwritten_writes,
            "warnings_count": warnings_count,
            "runtime_libraries_written": (
                self._metadata.runtime_libraries_written
            ),
        }

        if format == "json":
            json_report = {
                "summary": summary_payload,
                "runtime_libraries": runtime_libraries,
                "written_files": written_files if include_file_details else [],
                "failed_files": failed_files if include_file_details else [],
                "warnings": warnings,
            }
            self._logger.info(
                "Summary report generated in JSON format "
                f"(files={self._metadata.total_writes})"
            )
            return json.dumps(json_report, indent=2)

        lines: list[str] = [
            "=== Obfuscation Output Summary ===",
            "",
            (
                "Execution Time: "
                f"{self._metadata.formatted_elapsed} "
                f"({self._metadata.elapsed_seconds:.2f} seconds)"
            ),
            "",
            "File Statistics:",
            f"  Total Files Processed: {self._metadata.total_writes}",
            f"  Successfully Written: {self._metadata.successful_writes}",
            f"  Failed: {self._metadata.failed_writes}",
            f"  Skipped: {self._metadata.skipped_writes}",
            f"  Renamed: {self._metadata.renamed_writes}",
            f"  Overwritten: {self._metadata.overwritten_writes}",
            f"  Warnings: {warnings_count}",
            "",
            "Runtime Libraries:",
            (
                "  Libraries Written: "
                f"{self._metadata.runtime_libraries_written}"
            ),
        ]

        if runtime_libraries:
            for runtime_library in runtime_libraries:
                lines.append(
                    "  - "
                    f"{runtime_library['path']} "
                    f"({runtime_library['language']})"
                )
        else:
            lines.append("  None")

        if include_file_details:
            lines.extend(["", "Written Files:"])
            if written_files:
                for entry in written_files:
                    conflict_label = entry["conflict_resolution"] or "none"
                    lines.append(
                        "  - "
                        f"{entry['path']} "
                        f"(original: {entry['original_path']}, "
                        f"conflict: {conflict_label}, "
                        f"atomic: {entry['was_atomic']})"
                    )
            else:
                lines.append("  None")

        lines.extend(["", "Warnings:"])
        if warnings:
            for warning in warnings:
                lines.append(f"  - {warning}")
        else:
            lines.append("  None")

        lines.extend(["", "Errors:"])
        if include_file_details:
            if failed_files:
                for failed in failed_files:
                    lines.append(
                        f"  - {failed['path']}: {failed['error']}"
                    )
            else:
                lines.append("  None")
        elif self._metadata.failed_writes > 0:
            lines.append("  File details omitted (include_file_details=False)")
        else:
            lines.append("  None")

        self._logger.info(
            "Summary report generated in text format "
            f"(files={self._metadata.total_writes})"
        )
        return "\n".join(lines)

    def write_summary_report(
        self,
        output_path: Path | None = None,
        format: str = "text",
        include_file_details: bool = True,
    ) -> WriteResult:
        """Generate and write summary report content to disk.

        Args:
            output_path: Optional destination path. If omitted, defaults to
                ``obfuscation_summary.txt`` or ``obfuscation_summary.json`` in
                ``self.output_dir``.
            format: Report format, either ``"text"`` or ``"json"``.
            include_file_details: Include per-file details when ``True``.

        Returns:
            A :class:`WriteResult` for the report write operation.

        Raises:
            ValueError: If *format* is not ``"text"`` or ``"json"``.
        """
        self._validate_report_format(format)
        default_name = (
            "obfuscation_summary.json"
            if format == "json"
            else "obfuscation_summary.txt"
        )
        resolved_output_path = output_path or (self.output_dir / default_name)

        try:
            report_content = self.generate_summary_report(
                format=format,
                include_file_details=include_file_details,
            )
            ensure_directory(resolved_output_path.parent)

            if format == "text":
                resolved_output_path.write_text(report_content, encoding="utf-8")
            else:
                report_data = json.loads(report_content)
                with resolved_output_path.open("w", encoding="utf-8") as f:
                    json.dump(report_data, f, indent=2)
                    f.write("\n")

            self._logger.info(
                "Summary report written successfully: "
                f"{resolved_output_path}"
            )
            return WriteResult(
                success=True,
                output_path=resolved_output_path,
                original_path=resolved_output_path,
                was_atomic=False,
            )

        except Exception as exc:
            error_message = (
                "Failed to write summary report "
                f"to {resolved_output_path}: {exc}"
            )
            self._logger.error(error_message)
            return WriteResult(
                success=False,
                output_path=None,
                original_path=resolved_output_path,
                error=error_message,
                was_atomic=False,
            )

    def get_summary_report(
        self, format: str = "text", include_file_details: bool = True
    ) -> str:
        """Return summary report content without writing it to disk."""
        return self.generate_summary_report(
            format=format,
            include_file_details=include_file_details,
        )

    # ------------------------------------------------------------------
    # Metadata access
    # ------------------------------------------------------------------

    def get_metadata(self) -> WriteMetadata:
        """Return a snapshot of the current write metadata.

        Returns:
            A **copy** of the internal :class:`WriteMetadata` so that
            callers cannot mutate the writer's state.

        Example::

            meta = writer.get_metadata()
            print(f"{meta.successful_writes} of {meta.total_writes} OK")
        """
        return WriteMetadata(
            total_writes=self._metadata.total_writes,
            successful_writes=self._metadata.successful_writes,
            failed_writes=self._metadata.failed_writes,
            skipped_writes=self._metadata.skipped_writes,
            renamed_writes=self._metadata.renamed_writes,
            overwritten_writes=self._metadata.overwritten_writes,
            written_files=list(self._metadata.written_files),
            runtime_libraries_written=self._metadata.runtime_libraries_written,
            runtime_library_paths=list(self._metadata.runtime_library_paths),
            start_time=self._metadata.start_time,
            end_time=self._metadata.end_time,
            warnings=list(self._metadata.warnings),
            file_results=list(self._metadata.file_results),
        )

    def reset_metadata(self) -> None:
        """Reset all write metadata and clear the conflict decision cache.

        Example::

            writer.reset_metadata()
            assert writer.get_metadata().total_writes == 0
        """
        self._metadata = WriteMetadata(start_time=time.time())
        self._conflict_decisions.clear()
        self._logger.debug("Write metadata and conflict cache reset")

    def get_written_files(self) -> list[Path]:
        """Return the list of successfully written file paths.

        Returns:
            A copy of the internal written-files list.

        Example::

            for p in writer.get_written_files():
                print(p)
        """
        return list(self._metadata.written_files)
