"""Obfuscation orchestrator module for coordinating multi-file processing.

This module provides the ObfuscationOrchestrator class that coordinates the
complete obfuscation workflow: scanning files, building dependency graphs,
pre-computing symbol tables, and processing files in topological order.

The orchestrator implements a stateful workflow with the following phases:

1. **Validation** (VALIDATING): Check file existence, readability, extensions,
   output directory writability, and configuration validity.
2. **Conflict Detection**: Scan output directory for existing files that would
   be overwritten, and resolve via the configured ConflictStrategy.
3. **Analysis** (ANALYZING): Parse files, extract symbols, build dependency
   graph, and pre-compute global symbol table.
4. **Processing** (PROCESSING): Obfuscate files in topological order with
   progress tracking, error handling, cancellation support, and atomic
   output writes via ``OutputWriter``.
5. **Completion** (COMPLETED/FAILED/CANCELLED): Final state with metadata,
   including optional output summary reports.

State Transitions::

    PENDING -> VALIDATING -> ANALYZING -> PROCESSING -> COMPLETED
                  |                         |              |
                  v                         v              v
                FAILED                   CANCELLED       FAILED

Example:
    Basic usage with progress tracking and error handling::

        from pathlib import Path
        from obfuscator.core.config import ObfuscationConfig
        from obfuscator.core.orchestrator import (
            ObfuscationOrchestrator, ErrorStrategy, ProgressInfo,
        )

        config = ObfuscationConfig(name="my_project", language="python")
        orchestrator = ObfuscationOrchestrator(config=config)

        def on_progress(info: ProgressInfo):
            print(f"[{info.current_state.name}] {info.message} "
                  f"({info.percentage:.0f}%) ETA: {info.formatted_remaining}")

        def on_error(file_path, errors):
            print(f"Error in {file_path}: {errors}")
            return True  # continue processing

        result = orchestrator.process_files(
            input_files=[Path("main.py"), Path("utils.py")],
            output_dir=Path("./obfuscated"),
            config={"identifier_prefix": "_0x", "mangling_strategy": "sequential"},
            progress_callback=on_progress,
            error_callback=on_error,
            error_strategy=ErrorStrategy.ASK,
            generate_summary=True,
        )

        if result.success:
            print(f"Processed {len(result.processed_files)} files")
            print(f"Total time: {result.metadata['total_processing_time_seconds']:.1f}s")
            if result.summary_report:
                print(result.summary_report)
        else:
            print(f"Failed: {result.errors}")
"""

from __future__ import annotations

import multiprocessing
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from obfuscator.core.dependency_graph import (
    DependencyAnalyzer,
    DependencyGraph,
    CircularDependencyError,
    DependencyResolutionError,
)
from obfuscator.core.symbol_table import (
    GlobalSymbolTable,
    SymbolTableBuilder,
)
from obfuscator.processors.symbol_extractor import SymbolTable
from obfuscator.processors.lua_symbol_extractor import LuaSymbolTable
from obfuscator.core.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENABLE_MULTIPROCESSING,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MEMORY_THRESHOLD_PERCENT,
    DEFAULT_MIN_BATCH_SIZE,
    DEFAULT_MULTIPROCESSING_THRESHOLD,
    ObfuscationConfig,
)
from obfuscator.core.runtime_manager import RuntimeManager
from obfuscator.core.worker import (
    MemoryMonitor,
    WorkerResult,
    WorkerTask,
    process_file_batch,
    set_cancellation_event,
)
from obfuscator.core.exceptions import UnsupportedFeatureWarning
from obfuscator.utils.error_formatting import extract_line_column, parse_error
from obfuscator.utils.logger import get_logger
from obfuscator.utils.path_utils import is_readable, is_writable

if TYPE_CHECKING:
    from obfuscator.processors.python_processor import PythonProcessor
    from obfuscator.processors.lua_processor import LuaProcessor
    from obfuscator.core.output_writer import OutputWriter, WriteResult

logger = get_logger("obfuscator.core.orchestrator")

MULTIPROCESSING_THRESHOLD = DEFAULT_MULTIPROCESSING_THRESHOLD
DEFAULT_PARALLEL_BATCH_SIZE = DEFAULT_BATCH_SIZE
DEFAULT_BATCH_TIMEOUT_SECONDS = 300.0
DEFAULT_CANCELLATION_GRACE_PERIOD = 5.0


def _init_worker(cancel_event: multiprocessing.Event) -> None:
    """Initialize each worker process with the shared cancellation event."""
    set_cancellation_event(cancel_event)


class JobState(Enum):
    """Enumeration of possible job states during obfuscation orchestration.

    The job state tracks the current phase of the obfuscation process,
    providing clear feedback about where the orchestration currently stands.

    States:
        PENDING: Initial state, job created but not started
        VALIDATING: Input validation phase (files, directories, config)
        ANALYZING: Scanning files and building dependency graph
        PROCESSING: Obfuscating files in topological order
        COMPLETED: All files processed successfully
        FAILED: Error occurred during processing
        CANCELLED: Job was cancelled by user (future phase)
    """
    PENDING = "pending"
    VALIDATING = "validating"
    ANALYZING = "analyzing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorStrategy(Enum):
    """Enumeration of error handling strategies during obfuscation.

    When errors occur during file processing, these strategies determine
    how the orchestrator handles the situation.

    Strategies:
        CONTINUE: Skip failed files and continue processing remaining files.
                 Errors are logged but processing continues for other files.
        STOP: Halt obfuscation immediately on first error.
              No further files are processed after an error occurs.
        ASK: Prompt user for decision (requires error callback).
             The error_callback is invoked with file path and error messages.
             User decides whether to continue or stop for each error.
    """
    CONTINUE = "continue"
    STOP = "stop"
    ASK = "ask"


class ConflictStrategy(Enum):
    """Enumeration of conflict resolution strategies for file output.

    When output files already exist, these strategies determine how
    the orchestrator handles the conflict.

    Strategies:
        OVERWRITE: Replace existing files without prompting
        SKIP: Skip files that already exist
        RENAME: Append timestamp to filename (e.g., `file_20260213_143022.py`)
        ASK: Prompt user for each conflict (GUI only)
    """
    OVERWRITE = "overwrite"
    SKIP = "skip"
    RENAME = "rename"
    ASK = "ask"


@dataclass
class ProcessedEngine:
    """Tracks an engine and its associated language for hybrid mode.
    
    Attributes:
        engine: The ObfuscationEngine instance used for transformations.
        language: The language of the file processed ("python" or "lua").
    """
    engine: "ObfuscationEngine"
    language: str


@dataclass
class ProcessResult:
    """Result of processing a single file.

    Attributes:
        file_path: Path to the processed file
        output_path: Path where output was written
        success: Whether processing succeeded
        errors: List of error messages
        detailed_errors: List of structured error metadata dictionaries
        warnings: List of warning messages
        conflict_resolution: How conflict was resolved (e.g., "renamed", "skipped", "overwritten")
        was_cancelled: Whether this file result represents cancellation-based skipping
    """
    file_path: Path
    output_path: Path | None
    success: bool
    errors: list[str] = field(default_factory=list)
    detailed_errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    conflict_resolution: str | None = None
    was_cancelled: bool = False


@dataclass
class OrchestrationResult:
    """Result of the complete orchestration process.

    Attributes:
        success: Whether the overall process succeeded
        current_state: Current state of the job (PENDING, VALIDATING, etc.)
        processed_files: List of ProcessResult for each file
        dependency_graph: The constructed dependency graph
        global_symbol_table: The pre-computed symbol table
        errors: List of global error messages
        detailed_errors: List of structured error metadata dictionaries
        warnings: List of global warning messages
        metadata: Additional metadata about the process
        summary_report: Optional text summary generated by OutputWriter
    """
    success: bool
    current_state: JobState = field(default=JobState.PENDING)
    processed_files: list[ProcessResult] = field(default_factory=list)
    dependency_graph: DependencyGraph | None = None
    global_symbol_table: GlobalSymbolTable | None = None
    errors: list[str] = field(default_factory=list)
    detailed_errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    summary_report: str | None = None


@dataclass
class ValidationResult:
    """Result of input validation checks.
    
    Attributes:
        success: Whether all validation checks passed
        errors: List of validation error messages
        warnings: List of validation warning messages
    """
    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ConflictInfo:
    """Information about a single file conflict.
    
    Attributes:
        input_path: Original input file path
        output_path: Conflicting output file path
        exists: Whether the file exists
        is_writable: Whether the file is writable (if exists)
    """
    input_path: Path
    output_path: Path
    exists: bool
    is_writable: bool = True


@dataclass
class ConflictDetectionResult:
    """Result of conflict detection scan.
    
    Attributes:
        conflicts: List of detected conflicts
    """
    conflicts: list[ConflictInfo] = field(default_factory=list)

    @property
    def has_conflicts(self) -> bool:
        """Returns True if any conflicts were detected."""
        return len(self.conflicts) > 0


@dataclass
class ProgressInfo:
    """Encapsulates all progress-related information including time tracking.

    Attributes:
        current_file: Name of the file currently being processed, or None
        current_index: Current step index in the processing sequence
        total_files: Total number of steps in the processing sequence
        percentage: Progress percentage (0.0 to 100.0)
        elapsed_seconds: Elapsed time in seconds since processing started
        estimated_remaining_seconds: Estimated remaining time in seconds, or None if unknown
        current_state: Current JobState of the orchestration
        message: Human-readable progress message
        log_level: Log severity for UI rendering (success, warning, error, info)
    """
    current_file: str | None
    current_index: int
    total_files: int
    percentage: float
    elapsed_seconds: float
    estimated_remaining_seconds: float | None
    current_state: JobState
    message: str
    log_level: str = "info"

    @property
    def is_complete(self) -> bool:
        """Returns True when current_index >= total_files."""
        return self.current_index >= self.total_files

    @property
    def formatted_elapsed(self) -> str:
        """Returns elapsed time as 'MM:SS' format."""
        minutes = int(self.elapsed_seconds) // 60
        seconds = int(self.elapsed_seconds) % 60
        return f"{minutes:02d}:{seconds:02d}"

    @property
    def formatted_remaining(self) -> str:
        """Returns estimated remaining time as 'MM:SS' format or 'Calculating...' if None."""
        if self.estimated_remaining_seconds is None:
            return "Calculating..."
        minutes = int(self.estimated_remaining_seconds) // 60
        seconds = int(self.estimated_remaining_seconds) % 60
        return f"{minutes:02d}:{seconds:02d}"


class ObfuscationOrchestrator:
    """Coordinates the complete obfuscation workflow for multi-file projects.

    The orchestrator implements a stateful, multi-phase workflow:

    1. **Validation** (VALIDATING): Pre-flight checks on inputs, config, and
       output directory.
    2. **Conflict Detection**: Identify and resolve output file conflicts using
       the configured ``ConflictStrategy`` (OVERWRITE, SKIP, RENAME, ASK).
    3. **Analysis** (ANALYZING): Scan files, extract symbols, build dependency
       graph, and pre-compute global symbol table.
    4. **Processing** (PROCESSING): Obfuscate files in topological order with
       progress callbacks, error handling, and cancellation support.
    5. **Completion**: Transition to COMPLETED, FAILED, or CANCELLED.

    Attributes:
        python_processor: Processor for Python files
        lua_processor: Processor for Lua files
        _logger: Logger instance
        _current_state: Current ``JobState`` of the orchestrator
        _cancellation_requested: Flag for graceful cancellation
        _conflict_strategy: Active conflict resolution strategy
        _error_strategy: Active error handling strategy

    Examples:
        Simple usage without callbacks::

            orchestrator = ObfuscationOrchestrator()
            result = orchestrator.process_files(
                input_files=[Path("main.py")],
                output_dir=Path("./out"),
                config={"mangling_strategy": "sequential"},
            )

        With progress tracking and cancellation::

            orchestrator = ObfuscationOrchestrator(config=my_config)

            def on_progress(info: ProgressInfo):
                print(f"{info.percentage:.0f}% - {info.message}")

            # Start in a thread to allow cancellation
            result = orchestrator.process_files(
                input_files=files,
                output_dir=Path("./out"),
                progress_callback=on_progress,
            )
            # From another thread: orchestrator.request_cancellation()

        With error handling via ASK strategy::

            def on_error(file_path, errors):
                # Show dialog, return True to continue or False to stop
                return True

            result = orchestrator.process_files(
                input_files=files,
                output_dir=Path("./out"),
                error_strategy=ErrorStrategy.ASK,
                error_callback=on_error,
            )
    """

    def __init__(self, config: ObfuscationConfig | None = None) -> None:
        """Initialize the orchestrator with processors.

        Args:
            config: Optional ObfuscationConfig instance. If provided,
                    the orchestrate() convenience method can be used.
        """
        # Import processors lazily to keep module-level imports lightweight.
        # This allows importing enums (e.g., ConflictStrategy) without
        # requiring the full processor/engine stack to initialize.
        from obfuscator.processors.python_processor import PythonProcessor
        from obfuscator.processors.lua_processor import LuaProcessor

        self.python_processor = PythonProcessor(config=config)
        self.lua_processor = LuaProcessor(config=config)
        self._logger = logger
        self._project_root: Path | None = None
        self._config = config
        self._processed_engines: list = []
        self._current_state: JobState = JobState.PENDING
        # Initialize conflict strategy from config if provided, with safe fallback
        if config is not None:
            try:
                self._conflict_strategy = ConflictStrategy(config.conflict_strategy)
            except ValueError:
                self._logger.warning(
                    f"Invalid conflict_strategy '{config.conflict_strategy}' in config, "
                    "falling back to ASK"
                )
                self._conflict_strategy = ConflictStrategy.ASK
        else:
            self._conflict_strategy = ConflictStrategy.ASK
        self._conflict_decisions: dict[Path, str] = {}
        self._cancellation_requested: bool = False
        self._error_strategy: ErrorStrategy = ErrorStrategy.CONTINUE
        self._start_time: float | None = None
        self._file_processing_times: list[float] = []
        self._parallel_processing_metadata: dict[str, Any] = {}
        self._active_pool_manager: ProcessPoolManager | None = None

    def _transition_state(self, new_state: JobState, result: OrchestrationResult) -> None:
        """Transition the orchestrator to a new state.

        Updates both the internal _current_state and the result.current_state,
        and logs the transition for debugging purposes.

        Args:
            new_state: The state to transition to
            result: The OrchestrationResult to update with the new state
        """
        old_state = self._current_state
        self._current_state = new_state
        result.current_state = new_state
        self._logger.info(f"State transition: {old_state.name} -> {new_state.name}")

    def request_cancellation(self) -> None:
        """Request cancellation of the current obfuscation job.

        Sets the cancellation flag which is checked at strategic points
        during processing to allow graceful shutdown while preserving
        partial results.

        The flag is checked:
        - Before the processing loop begins (immediate cancellation)
        - At the start of each file iteration (between-file cancellation)

        Files already written to disk are preserved.  The returned
        ``OrchestrationResult`` will have ``current_state == CANCELLED``
        and ``metadata["was_cancelled"] == True``, along with lists of
        completed and skipped files.

        This method is safe to call from any thread (e.g. a GUI cancel
        button handler) and is idempotent.

        Example::

            # From a GUI thread while process_files() runs in a worker:
            orchestrator.request_cancellation()
        """
        self._cancellation_requested = True
        if self._active_pool_manager is not None:
            self._active_pool_manager.signal_cancellation()
            self._logger.info("Cancellation signal propagated to multiprocessing workers")
        self._logger.info("Cancellation requested by user")

    def _check_cancellation(self) -> bool:
        """Check if cancellation has been requested.

        Returns:
            True if cancellation was requested, False otherwise.
        """
        if self._cancellation_requested:
            self._logger.debug("Cancellation detected during processing")
        return self._cancellation_requested

    def _calculate_time_estimates(
        self, total_input_files: int
    ) -> tuple[float, float | None]:
        """Calculate elapsed time and estimated remaining time.

        Args:
            total_input_files: Total number of input files to process

        Returns:
            Tuple of (elapsed_seconds, estimated_remaining_seconds or None)
        """
        if self._start_time is None:
            return (0.0, None)
        elapsed = time.time() - self._start_time
        if not self._file_processing_times:
            return (elapsed, None)
        average_time = sum(self._file_processing_times) / len(self._file_processing_times)
        files_remaining = max(0, total_input_files - len(self._file_processing_times))
        estimated_remaining = average_time * files_remaining
        return (elapsed, estimated_remaining)

    def orchestrate(
        self,
        input_files: list[str | Path],
        output_dir: str | Path,
        progress_callback: Callable[[ProgressInfo], None] | None = None,
        project_root: str | Path | None = None
    ) -> OrchestrationResult:
        """Orchestrate obfuscation using the stored ObfuscationConfig.

        This is a convenience alias for process_files() that uses the config
        provided at construction time and accepts string paths.

        Args:
            input_files: List of input file paths (str or Path)
            output_dir: Output directory (str or Path)
            progress_callback: Optional callback receiving ProgressInfo objects
            project_root: Optional project root directory

        Returns:
            OrchestrationResult with processing details
        """
        paths = [Path(f) for f in input_files]
        out_path = Path(output_dir)
        proj_root = Path(project_root) if project_root else None

        config_dict: dict[str, Any] = {}
        if self._config:
            config_dict.update(self._config.symbol_table_options)

        return self.process_files(
            input_files=paths,
            output_dir=out_path,
            config=config_dict,
            progress_callback=progress_callback,
            project_root=proj_root
        )

    def validate_inputs(
        self,
        input_files: list[Path],
        output_dir: Path
    ) -> ValidationResult:
        """Validate all inputs before processing begins.
        
        Performs comprehensive pre-flight checks including file existence,
        readability, valid extensions (.py, .pyw, .lua, .luau), output
        directory writability, and ``ObfuscationConfig.validate()`` if a
        config was provided at construction time.
        
        Args:
            input_files: List of input file paths to validate
            output_dir: Output directory to validate
            
        Returns:
            ValidationResult with ``success`` flag and ``errors``/``warnings``
            lists.  All detected issues are reported in a single result so
            the caller can display them together.

        Examples:
            Successful validation::

                result = orchestrator.validate_inputs([Path("main.py")], Path("./out"))
                assert result.success
                assert len(result.errors) == 0

            Validation failure (nonexistent file + bad extension)::

                result = orchestrator.validate_inputs(
                    [Path("missing.py"), Path("notes.txt")],
                    Path("./out"),
                )
                assert not result.success
                # result.errors contains:
                #   "File not found: missing.py"
                #   "Unsupported file extension '.txt' for file: notes.txt. ..."
        """
        errors: list[str] = []
        warnings: list[str] = []
        
        self._logger.info("Starting input validation...")
        
        # Empty file list check
        if not input_files:
            errors.append("No input files provided")
            self._logger.debug("Input file list is empty")
        else:
            self._logger.debug(f"Checking {len(input_files)} input files...")
            
            # Valid extensions
            valid_extensions = {".py", ".pyw", ".lua", ".luau"}
            
            for file_path in input_files:
                # Resolve path for consistent handling (handles symlinks)
                resolved_path = file_path.resolve()
                
                # File existence check
                if not resolved_path.exists():
                    errors.append(f"File not found: {file_path}")
                    continue
                
                # File readability check
                if not is_readable(resolved_path):
                    errors.append(f"File is not readable: {file_path}")
                    continue
                
                # File type validation
                extension = resolved_path.suffix.lower()
                if extension not in valid_extensions:
                    errors.append(
                        f"Unsupported file extension '{extension}' for file: {file_path}. "
                        f"Supported: .py, .pyw, .lua, .luau"
                    )
        
        # ObfuscationConfig validation
        if self._config is not None:
            try:
                self._config.validate()
            except ValueError as e:
                errors.append(f"Configuration validation failed: {e}")
        
        # Output directory writability check
        resolved_output = output_dir.resolve()
        
        if resolved_output.exists():
            if resolved_output.is_dir():
                # Check if directory is writable using utility
                if not is_writable(resolved_output):
                    errors.append(f"Output directory is not writable: {output_dir}")
            else:
                errors.append(f"Output path exists but is not a directory: {output_dir}")
        else:
            # Output directory doesn't exist, check if parent exists and is readable/writable
            parent_dir = resolved_output.parent
            if not parent_dir.exists():
                errors.append(f"Cannot create output directory (parent does not exist): {output_dir}")
            else:
                # Check if parent is readable and writable using utilities
                if not is_readable(parent_dir):
                    errors.append(f"Cannot create output directory (parent not readable): {output_dir}")
                if not is_writable(parent_dir):
                    errors.append(f"Cannot create output directory (parent not writable): {output_dir}")
        
        self._logger.info(f"Validation completed: {len(errors)} errors, {len(warnings)} warnings")
        
        return ValidationResult(
            success=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def set_conflict_strategy(self, strategy: ConflictStrategy) -> None:
        """Set the conflict resolution strategy.

        Args:
            strategy: The conflict resolution strategy to use
        """
        self._conflict_strategy = strategy
        self._logger.info(f"Conflict strategy set to: {strategy.value}")

    def handle_processing_error(
        self,
        file_path: Path,
        errors: list[str],
        error_callback: Callable[[Path, list[str]], bool] | None,
    ) -> bool:
        """Handle a file processing error based on the configured error strategy.

        This method implements the error handling logic according to the
        ErrorStrategy setting: CONTINUE, STOP, or ASK.

        Args:
            file_path: Path to the file that failed processing
            errors: List of error messages from the processing failure
            error_callback: Optional callback to invoke for user decision.
                Required when error_strategy is ASK.
                Signature: error_callback(file_path: Path, errors: list[str]) -> bool
                Returns True to continue, False to stop.

        Returns:
            bool: True if processing should continue, False if it should stop

        Raises:
            ValueError: If error_strategy is ASK but no error_callback is provided

        Example:
            >>> should_continue = orchestrator.handle_processing_error(
            ...     file_path=Path("main.py"),
            ...     errors=["Parse error at line 10"],
            ...     error_callback=my_callback
            ... )
        """
        if self._error_strategy == ErrorStrategy.CONTINUE:
            self._logger.warning(
                f"Error processing {file_path.name}: {errors[0] if errors else 'Unknown error'}. "
                "Continuing with remaining files (CONTINUE strategy)."
            )
            return True

        elif self._error_strategy == ErrorStrategy.STOP:
            self._logger.error(
                f"Error processing {file_path.name}: {errors[0] if errors else 'Unknown error'}. "
                "Halting obfuscation immediately (STOP strategy)."
            )
            return False

        elif self._error_strategy == ErrorStrategy.ASK:
            if error_callback is None:
                raise ValueError(
                    "ErrorStrategy.ASK requires an error_callback to be provided. "
                    "Use ErrorStrategy.CONTINUE or ErrorStrategy.STOP, or provide a callback."
                )
            self._logger.info(
                f"Error processing {file_path.name}. "
                "Prompting user for decision (ASK strategy)."
            )
            decision = error_callback(file_path, errors)
            action = "continue" if decision else "stop"
            self._logger.info(f"User chose to {action} after error in {file_path.name}")
            return decision

        else:
            self._logger.warning(
                f"Unknown error strategy: {self._error_strategy}. "
                "Defaulting to CONTINUE behavior."
            )
            return True

    def detect_conflicts(
        self,
        input_files: list[Path],
        output_dir: Path,
        project_root: Path | None = None
    ) -> ConflictDetectionResult:
        """Detect file conflicts before processing begins.

        For each input file, computes the expected output path and checks
        if a file already exists at that location.

        Args:
            input_files: List of input file paths to check
            output_dir: Output directory where files will be written
            project_root: Optional project root for relative path computation.
                         If None, computed from the common path of all input files.

        Returns:
            ``ConflictDetectionResult`` whose ``.has_conflicts`` property
            indicates whether any conflicts exist, and ``.conflicts`` is a
            list of ``ConflictInfo`` objects with ``input_path``,
            ``output_path``, ``exists``, and ``is_writable`` fields.

        Example::

            result = orchestrator.detect_conflicts(files, Path("./out"), project_root)
            if result.has_conflicts:
                for c in result.conflicts:
                    print(f"{c.output_path} already exists (writable={c.is_writable})")
        """
        conflicts: list[ConflictInfo] = []

        # Compute project root if not provided
        if project_root is None and input_files:
            resolved_paths = [f.resolve() for f in input_files]
            try:
                common_path = Path(os.path.commonpath(resolved_paths))
                project_root = common_path if common_path.is_dir() else common_path.parent
            except ValueError:
                project_root = Path.cwd()

        for input_path in input_files:
            # Compute output path using same logic as _process_file_in_order
            if project_root:
                try:
                    relative_path = input_path.resolve().relative_to(project_root.resolve())
                    output_path = output_dir / relative_path
                except ValueError:
                    output_path = output_dir / input_path.name
            else:
                output_path = output_dir / input_path.name

            # Check if output file exists
            exists = output_path.exists()
            is_writable = True

            if exists:
                # Check if writable
                is_writable = os.access(output_path, os.W_OK)
                conflicts.append(ConflictInfo(
                    input_path=input_path,
                    output_path=output_path,
                    exists=True,
                    is_writable=is_writable
                ))

        if conflicts:
            self._logger.info(f"Detected {len(conflicts)} file conflict(s)")
        else:
            self._logger.info("No file conflicts detected")

        return ConflictDetectionResult(conflicts=conflicts)

    def resolve_conflict(
        self,
        output_path: Path,
        strategy: ConflictStrategy | None = None
    ) -> Path | None:
        """Resolve a file conflict using the specified strategy.

        Args:
            output_path: The path where the file would be written
            strategy: The strategy to use. If None, uses ``self._conflict_strategy``.

        Returns:
            - **OVERWRITE**: Returns ``output_path`` unchanged.
            - **SKIP**: Returns ``None`` (file should not be written).
            - **RENAME**: Returns a new path with a ``_YYYYMMDD_HHMMSS``
              timestamp suffix inserted before the extension.
            - **ASK**: Returns the path based on a previously cached decision.

        Raises:
            ValueError: If ASK strategy is used and no cached decision exists.

        Examples::

            # OVERWRITE – returns same path
            resolved = orchestrator.resolve_conflict(Path("out/main.py"))

            # SKIP – returns None
            orchestrator.set_conflict_strategy(ConflictStrategy.SKIP)
            resolved = orchestrator.resolve_conflict(Path("out/main.py"))
            assert resolved is None

            # RENAME – returns timestamped path
            orchestrator.set_conflict_strategy(ConflictStrategy.RENAME)
            resolved = orchestrator.resolve_conflict(Path("out/main.py"))
            # e.g. Path("out/main_20260213_143022.py")
        """
        effective_strategy = strategy if strategy is not None else self._conflict_strategy

        if effective_strategy == ConflictStrategy.OVERWRITE:
            self._logger.info(f"Resolved conflict for {output_path.name}: OVERWRITE")
            return output_path

        elif effective_strategy == ConflictStrategy.SKIP:
            self._logger.info(f"Resolved conflict for {output_path.name}: SKIP")
            return None

        elif effective_strategy == ConflictStrategy.RENAME:
            # Generate new path with timestamp suffix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Insert timestamp before file extension
            stem = output_path.stem
            suffix = output_path.suffix
            new_name = f"{stem}_{timestamp}{suffix}"
            new_path = output_path.parent / new_name

            # Handle edge case where renamed file also exists
            counter = 1
            while new_path.exists():
                new_name = f"{stem}_{timestamp}_{counter}{suffix}"
                new_path = output_path.parent / new_name
                counter += 1

            self._logger.info(f"Resolved conflict for {output_path.name}: RENAME -> {new_name}")
            return new_path

        elif effective_strategy == ConflictStrategy.ASK:
            # Check for cached decision
            if output_path in self._conflict_decisions:
                decision = self._conflict_decisions[output_path]
                if decision == "skip":
                    self._logger.info(f"Resolved conflict for {output_path.name}: SKIP (cached)")
                    return None
                elif decision == "overwrite":
                    self._logger.info(f"Resolved conflict for {output_path.name}: OVERWRITE (cached)")
                    return output_path
                elif decision.startswith("rename:"):
                    new_path = Path(decision.replace("rename:", ""))
                    self._logger.info(f"Resolved conflict for {output_path.name}: RENAME -> {new_path.name} (cached)")
                    return new_path
            else:
                raise ValueError(
                    f"ASK strategy requires cached decision for {output_path}. "
                    "GUI must provide decision before calling resolve_conflict."
                )

        else:
            self._logger.warning(f"Unknown conflict strategy: {effective_strategy}, defaulting to overwrite")
            return output_path

    def process_files(
        self,
        input_files: list[Path],
        output_dir: Path,
        config: dict[str, Any] | None = None,
        progress_callback: Callable[[ProgressInfo], None] | None = None,
        project_root: Path | None = None,
        error_callback: Callable[[Path, list[str]], bool] | None = None,
        error_strategy: ErrorStrategy = ErrorStrategy.CONTINUE,
        generate_summary: bool = False,
    ) -> OrchestrationResult:
        """Process multiple files with dependency-aware obfuscation.

        This is the main entry point for the orchestration workflow.  It drives
        the orchestrator through the following state transitions::

            PENDING -> VALIDATING -> ANALYZING -> PROCESSING -> COMPLETED
                          |                         |
                          v                         v
                        FAILED                   CANCELLED

        The ``progress_callback`` receives a ``ProgressInfo`` object at each
        major step (validation, conflict check, scanning, graph build, symbol
        table build, and once per file).  ``ProgressInfo`` includes
        ``current_state``, ``percentage``, ``elapsed_seconds``,
        ``estimated_remaining_seconds``, and a human-readable ``message``.

        Cancellation can be requested at any time via
        ``request_cancellation()``.  The flag is checked before the processing
        loop and at the start of each file iteration.  Already-written files
        are preserved; the result includes metadata about which files were
        completed and which were skipped.

        Args:
            input_files: List of input file paths to process
            output_dir: Directory to write obfuscated files
            config: Configuration options for obfuscation
            progress_callback: Optional callback for progress updates.
                Signature: ``callback(progress_info: ProgressInfo)``
            project_root: Explicit project root directory for import resolution.
                If None, computed from the common path of all input files.
            error_callback: Optional callback for handling file processing errors.
                Signature: ``error_callback(file_path: Path, errors: list[str]) -> bool``
                Returns True to continue processing, False to stop.
                Required when error_strategy is ASK.
            error_strategy: Strategy for handling file processing errors.
                - ``CONTINUE``: Skip failed files and continue.
                - ``STOP``: Halt on first error.
                - ``ASK``: Prompt user via error_callback for each error.
            generate_summary: If ``True``, generate a text summary report from
                ``OutputWriter`` and store it in ``result.summary_report``.

        Returns:
            OrchestrationResult with ``success``, ``current_state``,
            ``processed_files``, ``errors``, ``warnings``, and ``metadata``
            (including ``total_processing_time_seconds``,
            ``average_time_per_file_seconds``, ``was_cancelled``, etc.).

        Examples:
            With progress and error callbacks::

                def on_progress(info: ProgressInfo):
                    print(f"{info.message} ({info.percentage:.0f}%)")

                def on_error(file_path, errors):
                    return True  # continue

                result = orchestrator.process_files(
                    input_files=[Path("main.py")],
                    output_dir=Path("./out"),
                    progress_callback=on_progress,
                    error_callback=on_error,
                    error_strategy=ErrorStrategy.ASK,
                    project_root=Path("/path/to/project"),
                    generate_summary=True,
                )
        """
        from obfuscator.core.output_writer import OutputWriter, WriteResult

        config = config or {}
        resolved_parallel_batch_size = DEFAULT_PARALLEL_BATCH_SIZE
        resolved_parallel_threshold = MULTIPROCESSING_THRESHOLD
        resolved_enable_multiprocessing = DEFAULT_ENABLE_MULTIPROCESSING
        resolved_max_workers: int | None = DEFAULT_MAX_WORKERS

        if self._config is not None:
            resolved_parallel_batch_size = self._config.batch_size
            resolved_parallel_threshold = self._config.multiprocessing_threshold
            resolved_enable_multiprocessing = self._config.enable_multiprocessing
            resolved_max_workers = self._config.max_workers
        else:
            if isinstance(config.get("batch_size"), int) and not isinstance(
                config.get("batch_size"), bool
            ):
                resolved_parallel_batch_size = int(config["batch_size"])

            if isinstance(config.get("multiprocessing_threshold"), int) and not isinstance(
                config.get("multiprocessing_threshold"), bool
            ):
                resolved_parallel_threshold = int(config["multiprocessing_threshold"])

            if isinstance(config.get("enable_multiprocessing"), bool):
                resolved_enable_multiprocessing = config["enable_multiprocessing"]

            max_workers_value = config.get("max_workers")
            if isinstance(max_workers_value, int) and not isinstance(max_workers_value, bool):
                resolved_max_workers = int(max_workers_value)
            elif max_workers_value is None:
                resolved_max_workers = None

        self._processed_engines = []
        self._cancellation_requested = False  # Reset cancellation flag
        self._error_strategy = error_strategy  # Store error strategy
        self._start_time = time.time()
        self._file_processing_times = []
        self._parallel_processing_metadata = {}
        self._active_pool_manager = None
        result = OrchestrationResult(success=True)
        # 5 pre-file phases: validation, conflict detection, scanning,
        # dependency build, symbol-table build — plus one step per file
        total_steps = 5 + len(input_files)
        current_step = 0
        num_input_files = len(input_files)

        # Initialize state to PENDING
        self._transition_state(JobState.PENDING, result)

        def _build_progress_info(
            message: str,
            current_file: str | None = None,
            log_level: str = "info",
        ) -> ProgressInfo:
            elapsed, estimated_remaining = self._calculate_time_estimates(num_input_files)
            percentage = (current_step / total_steps) * 100 if total_steps > 0 else 0
            return ProgressInfo(
                current_file=current_file,
                current_index=current_step,
                total_files=total_steps,
                percentage=percentage,
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=estimated_remaining,
                current_state=self._current_state,
                message=message,
                log_level=log_level,
            )

        def report_progress(message: str, current_file: str | None = None) -> None:
            """Report a progress step (increments the step counter)."""
            nonlocal current_step
            current_step += 1
            progress_info = _build_progress_info(message, current_file)
            if progress_callback:
                progress_callback(progress_info)
            self._logger.info(message)

        def emit_log(
            message: str,
            current_file: str | None = None,
            log_level: str = "info",
        ) -> None:
            """Emit a log-only message without incrementing the step counter."""
            nonlocal current_step

            effective_log_level = log_level
            if effective_log_level == "progress_step":
                current_step += 1
                effective_log_level = "info"

            progress_info = _build_progress_info(
                message,
                current_file,
                effective_log_level,
            )
            if progress_callback:
                progress_callback(progress_info)

            logger_method = getattr(self._logger, effective_log_level, self._logger.info)
            logger_method(message)

        # Transition to VALIDATING and validate inputs before processing
        self._transition_state(JobState.VALIDATING, result)
        report_progress("Validating inputs...")
        validation_result = self.validate_inputs(input_files, output_dir)

        if not validation_result.success:
            result.success = False
            result.errors.extend(validation_result.errors)
            result.warnings.extend(validation_result.warnings)
            self._logger.error(f"Input validation failed with {len(validation_result.errors)} error(s)")
            # Transition to FAILED before returning
            self._transition_state(JobState.FAILED, result)
            return result

        # Add any validation warnings to result
        if validation_result.warnings:
            result.warnings.extend(validation_result.warnings)

        # Conflict detection after validation succeeds
        report_progress("Checking for file conflicts...")
        conflict_result = self.detect_conflicts(input_files, output_dir, project_root)

        if conflict_result.has_conflicts:
            conflict_count = len(conflict_result.conflicts)

            if self._conflict_strategy == ConflictStrategy.ASK:
                # Return early with conflicts info for GUI to handle
                result.success = False
                result.errors.append(
                    "File conflicts detected. Please resolve conflicts before proceeding."
                )
                result.metadata["conflicts"] = conflict_result.conflicts
                result.metadata["conflicts_detected"] = conflict_count
                result.metadata["conflict_strategy"] = self._conflict_strategy.value
                self._logger.warning(
                    f"Detected {conflict_count} file conflict(s) - returning for GUI resolution"
                )
                self._transition_state(JobState.FAILED, result)
                return result

            elif self._conflict_strategy == ConflictStrategy.SKIP:
                result.warnings.append(
                    f"{conflict_count} file(s) will be skipped due to conflicts"
                )
                self._logger.info(f"{conflict_count} file(s) will be skipped")

            elif self._conflict_strategy == ConflictStrategy.RENAME:
                result.warnings.append(
                    f"{conflict_count} file(s) will be renamed to avoid conflicts"
                )
                self._logger.info(f"{conflict_count} file(s) will be renamed")

            elif self._conflict_strategy == ConflictStrategy.OVERWRITE:
                result.warnings.append(
                    f"{conflict_count} existing file(s) will be overwritten"
                )
                self._logger.info(f"{conflict_count} existing file(s) will be overwritten")

        # Compute project root from all input files if not explicitly provided
        if project_root is None and input_files:
            resolved_paths = [f.resolve() for f in input_files]
            try:
                common_path = Path(os.path.commonpath(resolved_paths))
                # If common path is a file, use its parent directory
                project_root = common_path if common_path.is_dir() else common_path.parent
                self._logger.info(f"Computed project root from input files: {project_root}")
            except ValueError:
                # commonpath raises ValueError if paths are on different drives (Windows)
                project_root = Path.cwd()
                self._logger.warning(
                    "Could not compute common path from input files, using cwd as project root"
                )

        # Store project_root for use in _process_file_in_order
        self._project_root = project_root

        try:
            output_writer = OutputWriter(
                output_dir=output_dir,
                conflict_strategy=self._conflict_strategy,
                use_atomic_writes=True,
                conflict_callback=None,
            )

            # Transition to ANALYZING for scan and symbol extraction
            self._transition_state(JobState.ANALYZING, result)

            # Phase 1: Scan and extract symbols (ASTs are discarded to bound memory)
            report_progress("Scanning files and extracting symbols...")
            successfully_parsed, symbol_tables = self._scan_and_extract_symbols(
                input_files, result
            )

            if not successfully_parsed:
                result.success = False
                result.errors.append("No files were successfully parsed")
                # Transition to FAILED before returning
                self._transition_state(JobState.FAILED, result)
                return result

            # Phase 2: Build dependency graph and symbol table
            report_progress("Building dependency graph...")
            graph = self._build_dependency_graph(
                successfully_parsed, symbol_tables, config, result, project_root
            )
            result.dependency_graph = graph

            report_progress("Pre-computing symbol table...")
            global_table = self._build_global_symbol_table(
                graph, symbol_tables, config, result
            )
            result.global_symbol_table = global_table

            # Phase 3: Process files in topological order (re-parsing just-in-time)
            try:
                processing_order = graph.get_processing_order()
            except CircularDependencyError as e:
                result.success = False
                result.errors.append(f"Circular dependency detected: {e.message}")
                result.warnings.append(
                    "Processing files in original order due to circular dependencies"
                )
                processing_order = list(successfully_parsed)

            use_multiprocessing = (
                resolved_enable_multiprocessing
                and len(processing_order) >= resolved_parallel_threshold
            )
            if use_multiprocessing:
                estimated_batches = max(
                    1,
                    (len(processing_order) + resolved_parallel_batch_size - 1)
                    // resolved_parallel_batch_size,
                )
                total_steps = 5 + estimated_batches
                worker_count = (
                    resolved_max_workers
                    if resolved_max_workers is not None
                    else max(1, min(multiprocessing.cpu_count() - 1, 8))
                )
                self._logger.info(
                    "Processing %d files using multiprocessing (%d workers, batch size %d)",
                    len(processing_order),
                    worker_count,
                    resolved_parallel_batch_size,
                )
            else:
                if not resolved_enable_multiprocessing:
                    self._logger.info(
                        "Processing %d files sequentially (multiprocessing disabled by configuration)",
                        len(processing_order),
                    )
                else:
                    self._logger.info(
                        "Processing %d files sequentially (below multiprocessing threshold: %d)",
                        len(processing_order),
                        resolved_parallel_threshold,
                    )

            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Transition to PROCESSING before file processing loop
            self._transition_state(JobState.PROCESSING, result)

            # Check for cancellation before starting the loop
            if self._check_cancellation():
                self._transition_state(JobState.CANCELLED, result)
                result.success = False
                cancel_total_time = time.time() - self._start_time
                result.metadata["cancellation_message"] = "Job cancelled by user before processing began"
                result.metadata["cancellation_time"] = datetime.now().isoformat()
                result.metadata["was_cancelled"] = True
                result.metadata["total_files_planned"] = len(processing_order)
                result.metadata["files_completed_before_cancel"] = []
                result.metadata["files_skipped_due_to_cancel"] = list(processing_order)
                result.metadata["total_processing_time_seconds"] = cancel_total_time
                result.metadata["average_time_per_file_seconds"] = 0
                result.metadata["individual_file_times"] = []
                result.metadata["processing_start_time"] = datetime.fromtimestamp(self._start_time).isoformat()
                result.metadata["processing_end_time"] = datetime.now().isoformat()
                result.warnings.append(f"0 of {len(processing_order)} file(s) completed before cancellation")
                emit_log("Job cancelled by user")
                return result

            warnings_by_file: dict[str, list[str]] = defaultdict(list)
            parallel_worker_count = 0
            parallel_batch_count = 0
            parallel_batch_size = 0
            parallel_batch_times: list[float] = []

            if use_multiprocessing:
                processed_start = len(result.processed_files)
                errors_start = len(result.errors)
                warnings_start = len(result.warnings)
                detailed_errors_start = len(result.detailed_errors)
                processing_times_start = len(self._file_processing_times)

                try:
                    self._process_files_parallel(
                        processing_order=processing_order,
                        global_table=global_table,
                        graph=graph,
                        output_dir=output_dir,
                        result=result,
                        progress_callback=progress_callback,
                        emit_log=emit_log,
                        error_callback=error_callback,
                        max_workers=resolved_max_workers,
                        batch_size=resolved_parallel_batch_size,
                    )

                    parallel_metadata = self._parallel_processing_metadata
                    parallel_worker_count = int(parallel_metadata.get("worker_count", 0))
                    parallel_batch_count = int(parallel_metadata.get("batch_count", 0))
                    parallel_batch_size = int(parallel_metadata.get("batch_size", 0))
                    parallel_batch_times = list(
                        parallel_metadata.get("batch_times_seconds", [])
                    )
                    parallel_warnings_by_file = parallel_metadata.get("warnings_by_file", {})
                    if isinstance(parallel_warnings_by_file, dict):
                        for file_key, file_warnings in parallel_warnings_by_file.items():
                            if isinstance(file_warnings, list):
                                warnings_by_file[str(file_key)].extend(
                                    [str(warning) for warning in file_warnings]
                                )
                except Exception as mp_error:
                    self._logger.warning(
                        "Multiprocessing failed: %s. Falling back to sequential processing.",
                        mp_error,
                        exc_info=True,
                    )
                    emit_log(
                        "Multiprocessing error, using sequential fallback",
                        log_level="warning",
                    )

                    # Discard partial multiprocessing results before sequential fallback.
                    del result.processed_files[processed_start:]
                    del result.errors[errors_start:]
                    del result.warnings[warnings_start:]
                    del result.detailed_errors[detailed_errors_start:]
                    del self._file_processing_times[processing_times_start:]
                    self._parallel_processing_metadata = {}

                    # Restore sequential-style step accounting so fallback progress
                    # remains bounded for per-file reporting.
                    total_steps = current_step + len(input_files)
                    use_multiprocessing = False

            if not use_multiprocessing:
                for file_path in processing_order:
                    # Check for cancellation at the start of each iteration
                    if self._check_cancellation():
                        self._transition_state(JobState.CANCELLED, result)
                        result.success = False
                        cancel_total_time = time.time() - self._start_time
                        result.metadata["cancellation_message"] = "Job cancelled by user during processing"
                        result.metadata["cancellation_time"] = datetime.now().isoformat()
                        result.metadata["was_cancelled"] = True
                        result.metadata["total_files_planned"] = len(processing_order)
                        completed_files = [pr.file_path for pr in result.processed_files if pr.success]
                        result.metadata["files_completed_before_cancel"] = [str(f) for f in completed_files]
                        skipped_files = [f for f in processing_order if f not in completed_files]
                        result.metadata["files_skipped_due_to_cancel"] = [str(f) for f in skipped_files]
                        result.metadata["total_processing_time_seconds"] = cancel_total_time
                        result.metadata["average_time_per_file_seconds"] = (
                            sum(self._file_processing_times) / len(self._file_processing_times)
                            if self._file_processing_times else 0
                        )
                        result.metadata["individual_file_times"] = self._file_processing_times
                        result.metadata["processing_start_time"] = datetime.fromtimestamp(self._start_time).isoformat()
                        result.metadata["processing_end_time"] = datetime.now().isoformat()
                        result.warnings.append(f"{len(completed_files)} of {len(processing_order)} file(s) completed before cancellation")
                        emit_log("Job cancelled by user")
                        return result

                    if file_path not in successfully_parsed:
                        continue

                    report_progress(f"Processing {file_path.name}...", current_file=file_path.name)
                    # Re-parse file just-in-time (AST not cached)
                    file_start_time = time.time()
                    process_result = self._process_file_in_order(
                        file_path,
                        global_table,
                        output_dir,
                        config,
                        output_writer,
                        emit_log_callback=emit_log,
                    )
                    self._file_processing_times.append(time.time() - file_start_time)
                    result.processed_files.append(process_result)

                    if process_result.warnings:
                        result.warnings.extend(process_result.warnings)
                        warnings_by_file[str(file_path)].extend(process_result.warnings)

                    if process_result.detailed_errors:
                        result.detailed_errors.extend(process_result.detailed_errors)

                    if not process_result.success:
                        # Append all per-file errors to the orchestration result
                        result.errors.extend(process_result.errors)

                        # Report all error messages via progress callback
                        if process_result.errors:
                            joined_errors = "; ".join(process_result.errors)
                            error_msg = f"Error processing {file_path.name}: {joined_errors}"
                        else:
                            error_msg = f"Error processing {file_path.name}: Unknown error"
                        emit_log(error_msg, current_file=file_path.name)

                        # Handle error based on strategy
                        should_continue = self.handle_processing_error(
                            file_path,
                            process_result.errors,
                            error_callback
                        )

                        # Track error decision in metadata
                        if "error_decisions" not in result.metadata:
                            result.metadata["error_decisions"] = []
                        if "files_failed_with_errors" not in result.metadata:
                            result.metadata["files_failed_with_errors"] = []

                        decision = "continue" if should_continue else "stop"
                        result.metadata["error_decisions"].append({
                            "file": str(file_path),
                            "errors": process_result.errors,
                            "decision": decision
                        })
                        result.metadata["files_failed_with_errors"].append(str(file_path))

                        # Report user decision
                        emit_log(f"User chose to {decision} after error", current_file=file_path.name)

                        # Break loop if user chose to stop
                        if not should_continue:
                            self._transition_state(JobState.FAILED, result)
                            result.success = False
                            break

            # Update overall success based on individual results
            failed_count = sum(
                1 for pr in result.processed_files if not pr.success and not pr.was_cancelled
            )
            cancelled_count = sum(1 for pr in result.processed_files if pr.was_cancelled)
            if failed_count > 0:
                result.warnings.append(
                    f"{failed_count} file(s) had processing errors"
                )
            if cancelled_count > 0:
                result.metadata["cancelled_files"] = [
                    str(pr.file_path) for pr in result.processed_files if pr.was_cancelled
                ]

            # Calculate total processing time
            total_time = time.time() - self._start_time

            result.metadata["total_files"] = len(input_files)
            result.metadata["processed_files"] = len(result.processed_files)
            result.metadata["failed_files"] = failed_count
            result.metadata["runtime_mode"] = self._config.runtime_mode if self._config else "embedded"
            result.metadata["runtime_files"] = []
            result.metadata["runtime_file_count"] = 0
            result.metadata["conflict_strategy"] = self._conflict_strategy.value
            result.metadata["error_strategy"] = self._error_strategy.value
            result.metadata["was_cancelled"] = self._cancellation_requested
            result.metadata["multiprocessing_enabled"] = use_multiprocessing
            if use_multiprocessing:
                result.metadata["worker_count"] = parallel_worker_count
                result.metadata["batch_count"] = parallel_batch_count
                result.metadata["batch_size"] = parallel_batch_size
                result.metadata["average_batch_time_seconds"] = (
                    sum(parallel_batch_times) / len(parallel_batch_times)
                    if parallel_batch_times
                    else 0
                )
            result.metadata["total_processing_time_seconds"] = total_time
            result.metadata["average_time_per_file_seconds"] = (
                sum(self._file_processing_times) / len(self._file_processing_times)
                if self._file_processing_times else 0
            )
            result.metadata["individual_file_times"] = self._file_processing_times
            result.metadata["processing_start_time"] = datetime.fromtimestamp(self._start_time).isoformat()
            result.metadata["processing_end_time"] = datetime.now().isoformat()

            errors_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
            errors_by_type: dict[str, int] = defaultdict(int)
            for detailed_error in result.detailed_errors:
                file_key = str(detailed_error.get("file_path") or "Unknown file")
                error_type = str(detailed_error.get("error_type") or "UnknownError")
                errors_by_file[file_key].append(detailed_error)
                errors_by_type[error_type] += 1

            result.metadata["detailed_errors"] = result.detailed_errors
            result.metadata["error_summary"] = {
                "total": len(result.detailed_errors),
                "by_type": dict(errors_by_type),
            }
            result.metadata["error_report"] = {
                "total_errors": len(result.detailed_errors),
                "errors_by_file": dict(errors_by_file),
                "errors_by_type": dict(errors_by_type),
                "timestamp": time.time(),
            }
            result.metadata["warning_count"] = len(result.warnings)
            result.metadata["warnings_by_file"] = dict(warnings_by_file)

            # Track skipped files and conflict resolutions
            skipped_files = [
                pr.file_path for pr in result.processed_files
                if pr.conflict_resolution == "skipped"
            ]
            if skipped_files:
                result.metadata["skipped_files"] = [str(f) for f in skipped_files]

            # Track resolved conflicts
            resolved_conflicts = [
                pr for pr in result.processed_files
                if pr.conflict_resolution in ("renamed", "overwritten")
            ]
            if resolved_conflicts:
                result.metadata["conflicts_resolved"] = len(resolved_conflicts)

            # Transition to COMPLETED only if the job didn't fail (e.g. stop-on-error)
            if result.success and result.current_state != JobState.FAILED:
                self._transition_state(JobState.COMPLETED, result)
                emit_log("Job completed")

            # Generate hybrid runtime files if in hybrid mode using OutputWriter
            if self._config is not None and self._config.runtime_mode == "hybrid":
                emit_log("Generating hybrid runtime libraries...", log_level="info")
                python_runtimes: set[str] = set()
                lua_runtimes: set[str] = set()

                for entry in self._processed_engines:
                    engine = entry.engine
                    language = entry.language
                    if hasattr(engine, "required_runtimes") and engine.required_runtimes:
                        if language == "python":
                            python_runtimes.update(engine.required_runtimes)
                        elif language == "lua":
                            lua_runtimes.update(engine.required_runtimes)

                runtime_files: list[str] = []
                runtime_write_results: list[WriteResult] = []

                for language, runtimes in (("python", python_runtimes), ("lua", lua_runtimes)):
                    if not runtimes:
                        continue

                    target_engine = next(
                        (
                            entry.engine
                            for entry in self._processed_engines
                            if entry.language == language
                        ),
                        None,
                    )
                    if target_engine is None or not hasattr(target_engine, "runtime_manager"):
                        warning_msg = (
                            f"No engine with runtime_manager found for {language}; "
                            "skipping hybrid runtime generation"
                        )
                        self._logger.warning(warning_msg)
                        emit_log(
                            f"Skipping hybrid runtime for {language} "
                            "(no runtime manager found)",
                            log_level="warning",
                        )
                        result.warnings.append(warning_msg)
                        continue

                    runtime_write_result = output_writer.write_runtime_library(
                        runtime_manager=target_engine.runtime_manager,
                        language=language,
                        runtime_mode=self._config.runtime_mode,
                        output_dir=output_dir,
                    )
                    runtime_write_results.append(runtime_write_result)

                    if runtime_write_result.success and runtime_write_result.output_path is not None:
                        runtime_files.append(str(runtime_write_result.output_path))
                        runtime_filename = runtime_write_result.output_path.name
                        emit_log(
                            f"Runtime library created: {runtime_filename}",
                            log_level="success",
                        )
                        self._logger.info(
                            f"Hybrid runtime library written for {language}: "
                            f"{runtime_write_result.output_path}"
                        )
                    elif runtime_write_result.success:
                        conflict_resolution = runtime_write_result.conflict_resolution or "unknown"
                        emit_log(
                            f"Runtime library skipped for {language} "
                            f"(conflict resolution: {conflict_resolution})",
                            log_level="warning",
                        )
                        self._logger.info(
                            f"Hybrid runtime library skipped for {language} "
                            f"(conflict={runtime_write_result.conflict_resolution})"
                        )
                    else:
                        runtime_error = (
                            runtime_write_result.error
                            or "Unknown runtime library write error"
                        )
                        warning_msg = (
                            f"Failed to write hybrid runtime library for {language}: "
                            f"{runtime_error}"
                        )
                        emit_log(
                            f"Failed to write runtime library for {language}: {runtime_error}",
                            log_level="error",
                        )
                        self._logger.warning(warning_msg)
                        result.warnings.append(warning_msg)

                result.metadata["runtime_files"] = runtime_files
                result.metadata["runtime_file_count"] = len(runtime_files)

                if runtime_write_results:
                    emit_log(
                        "Hybrid runtime generation completed "
                        f"({len(runtime_files)} libraries written)",
                        log_level="success",
                    )
                    self._logger.info(
                        "Hybrid runtime generation finished with "
                        f"{len(runtime_write_results)} write attempt(s)"
                    )

            if generate_summary:
                try:
                    result.summary_report = output_writer.generate_summary_report(
                        format="text",
                        include_file_details=True,
                    )
                    self._logger.info("Generated orchestration output summary report")
                except Exception as summary_error:
                    warning_msg = f"Failed to generate summary report: {summary_error}"
                    self._logger.warning(warning_msg)
                    result.warnings.append(warning_msg)

            writer_metadata = output_writer.get_metadata()
            result.metadata["output_writer_stats"] = {
                "total_writes": writer_metadata.total_writes,
                "successful_writes": writer_metadata.successful_writes,
                "failed_writes": writer_metadata.failed_writes,
                "skipped_writes": writer_metadata.skipped_writes,
                "renamed_writes": writer_metadata.renamed_writes,
                "overwritten_writes": writer_metadata.overwritten_writes,
                "runtime_libraries_written": writer_metadata.runtime_libraries_written,
            }
            result.metadata["output_writer_warnings"] = list(writer_metadata.warnings)

        except Exception as e:
            result.success = False
            result.errors.append(f"Orchestration failed: {e}")
            self._logger.error(f"Orchestration failed: {e}", exc_info=True)
            # Transition to FAILED on exception
            self._transition_state(JobState.FAILED, result)
            emit_log("Job failed")

        return result

    def _convert_worker_result_to_process_result(
        self,
        worker_result: WorkerResult,
    ) -> ProcessResult:
        """Convert a worker ``WorkerResult`` to orchestrator ``ProcessResult``."""
        output_path = (
            Path(worker_result.output_path)
            if worker_result.output_path is not None
            else None
        )

        detailed_errors: list[dict[str, Any]] = []
        for entry in worker_result.detailed_errors:
            if isinstance(entry, dict):
                detailed_errors.append(dict(entry))
            else:
                detailed_errors.append(
                    {
                        "file_path": str(worker_result.file_path),
                        "line": None,
                        "column": None,
                        "error_type": "UnexpectedException",
                        "message": str(entry),
                    }
                )

        return ProcessResult(
            file_path=Path(worker_result.file_path),
            output_path=output_path,
            success=worker_result.success,
            errors=[str(error) for error in worker_result.errors],
            detailed_errors=detailed_errors,
            warnings=[str(warning) for warning in worker_result.warnings],
            conflict_resolution=worker_result.conflict_resolution,
            was_cancelled=worker_result.was_cancelled,
        )

    def _process_files_parallel(
        self,
        processing_order: list[Path],
        global_table: GlobalSymbolTable,
        graph: DependencyGraph,
        output_dir: Path,
        result: OrchestrationResult,
        progress_callback: Callable[[ProgressInfo], None] | None,
        emit_log: Callable[[str, str | None, str], None],
        error_callback: Callable[[Path, list[str]], bool] | None,
        max_workers: int | None = None,
        batch_size: int = DEFAULT_PARALLEL_BATCH_SIZE,
    ) -> None:
        """Process files in parallel batches using worker processes."""
        _ = progress_callback

        warnings_by_file: dict[str, list[str]] = defaultdict(list)
        batch_times: list[float] = []
        worker_count_used = 0
        batch_size_used = batch_size
        batch_count = 0
        batch_size_adjustments: list[dict[str, Any]] = []
        worker_transformation_counts: dict[str, int] = {}
        total_success = 0
        halted_due_to_worker_errors = False
        skipped_due_to_halt: list[str] = []
        cancellation_detected = False
        cancellation_started_at: float | None = None
        cancelled_files: list[str] = []
        cancelled_uncollected_batches: list[int] = []
        graceful_shutdown_invoked = False

        if not processing_order:
            self._parallel_processing_metadata = {
                "worker_count": 0,
                "batch_count": batch_count,
                "batch_size": batch_size_used,
                "batch_size_adjustments": [],
                "batch_times_seconds": batch_times,
                "warnings_by_file": {},
                "cancelled_files": [],
                "cancelled_uncollected_batches": [],
            }
            return

        runtime_mode = self._config.runtime_mode if self._config else "embedded"

        if self._config is not None:
            config_dict = self._config.to_dict()
        else:
            inferred_language = "lua"
            if any(path.suffix.lower() in {".py", ".pyw"} for path in processing_order):
                inferred_language = "python"

            config_dict = ObfuscationConfig(
                name="parallel-default",
                language=inferred_language,
            ).to_dict()

        config_dict["runtime_mode"] = runtime_mode
        config_dict["conflict_strategy"] = self._conflict_strategy.value
        memory_threshold_percent = int(
            config_dict.get(
                "memory_threshold_percent",
                DEFAULT_MEMORY_THRESHOLD_PERCENT,
            )
        )
        min_batch_size = int(
            config_dict.get(
                "min_batch_size",
                DEFAULT_MIN_BATCH_SIZE,
            )
        )
        configured_batch_size = int(config_dict.get("batch_size", batch_size))
        configured_max_workers_raw = config_dict.get("max_workers", max_workers)
        configured_max_workers = (
            int(configured_max_workers_raw)
            if configured_max_workers_raw is not None
            else None
        )
        config_dict["memory_threshold_percent"] = memory_threshold_percent
        config_dict["min_batch_size"] = min_batch_size
        config_dict["batch_size"] = configured_batch_size
        config_dict["max_workers"] = configured_max_workers

        global_table_dict = global_table.to_dict()
        dependency_graph_dict = graph.to_dict()
        project_root_str = str(self._project_root) if self._project_root is not None else None

        def build_batch_failure_results(
            task_id: str,
            batch_paths: list[Path],
            error_message: str,
            processing_time: float = 0.0,
        ) -> list[WorkerResult]:
            """Build synthetic failure results for an entire batch."""
            failure_results: list[WorkerResult] = []
            for file_path in batch_paths:
                failure_results.append(
                    WorkerResult(
                        task_id=task_id,
                        file_path=str(file_path),
                        success=False,
                        output_path=None,
                        errors=[error_message],
                        warnings=[],
                        detailed_errors=[
                            self._create_detailed_error(
                                file_path=file_path,
                                error_type="UnexpectedException",
                                message=error_message,
                            )
                        ],
                        transformation_count=0,
                        processing_time=processing_time,
                    )
                )
            return failure_results

        def build_batch_cancelled_results(
            task_id: str,
            batch_paths: list[Path],
            warning_message: str = "Skipped due to cancellation request",
        ) -> list[WorkerResult]:
            """Build synthetic cancellation results for an entire batch."""
            cancelled_results: list[WorkerResult] = []
            for file_path in batch_paths:
                cancelled_results.append(
                    WorkerResult(
                        task_id=task_id,
                        file_path=str(file_path),
                        success=False,
                        output_path=None,
                        errors=[],
                        warnings=[warning_message],
                        detailed_errors=[],
                        transformation_count=0,
                        processing_time=0.0,
                        was_cancelled=True,
                    )
                )
            return cancelled_results

        pool_manager = ProcessPoolManager(
            max_workers=configured_max_workers,
            batch_size=configured_batch_size,
            memory_threshold_percent=memory_threshold_percent,
            min_batch_size=min_batch_size,
            grace_period=DEFAULT_CANCELLATION_GRACE_PERIOD,
        )
        self._active_pool_manager = pool_manager

        try:
            with pool_manager:
                worker_count_used = pool_manager.worker_count
                batch_size_used = pool_manager.batch_size

                emit_log(
                    f"Initializing process pool with {worker_count_used} workers",
                    log_level="info",
                )

                batches = pool_manager.create_batches(
                    processing_order,
                    pool_manager.batch_size,
                )
                batch_size_used = pool_manager.current_batch_size
                total_batches = len(batches)
                batch_count = total_batches
                batch_size_adjustments = list(pool_manager.batch_size_adjustments)

                if batch_size_adjustments:
                    adjustment_count = len(batch_size_adjustments)
                    emit_log(
                        (
                            "Batch size dynamically adjusted "
                            f"{adjustment_count} times due to memory pressure"
                        ),
                        log_level="info",
                    )
                    self._logger.info(
                        "Adaptive batch sizing applied %d adjustment(s)",
                        adjustment_count,
                    )

                emit_log(
                    f"Created {total_batches} batches from {len(processing_order)} files",
                    log_level="info",
                )

                if total_batches == 0:
                    self._parallel_processing_metadata = {
                        "worker_count": worker_count_used,
                        "batch_count": batch_count,
                        "batch_size": batch_size_used,
                        "batch_size_adjustments": batch_size_adjustments,
                        "batch_times_seconds": batch_times,
                        "warnings_by_file": {},
                        "cancelled_files": [],
                        "cancelled_uncollected_batches": [],
                    }
                    return

                pending_async_results: dict[int, AsyncResult] = {}
                batch_start_times: dict[int, float] = {}
                results_by_batch: dict[int, list[WorkerResult]] = {}

                for batch_index, batch in enumerate(batches, start=1):
                    task_id = f"batch-{batch_index}"

                    if self._check_cancellation():
                        if not cancellation_detected:
                            cancellation_detected = True
                            cancellation_started_at = time.time()
                            result.success = False
                            self._transition_state(JobState.CANCELLED, result)
                            pool_manager.signal_cancellation()
                            emit_log(
                                "Cancellation requested; stopping new batch submissions",
                                log_level="warning",
                            )

                        results_by_batch[batch_index] = build_batch_cancelled_results(
                            task_id=task_id,
                            batch_paths=batch,
                        )
                        batch_times.append(0.0)
                        cancelled_uncollected_batches.append(batch_index)
                        emit_log(
                            f"Batch {batch_index} cancelled before submission",
                            log_level="progress_step",
                        )
                        continue

                    try:
                        task = WorkerTask(
                            task_id=task_id,
                            file_paths=[str(path) for path in batch],
                            global_table_dict=global_table_dict,
                            dependency_graph_dict=dependency_graph_dict,
                            config_dict=config_dict,
                            output_dir=str(output_dir),
                            project_root=project_root_str,
                            runtime_mode=runtime_mode,
                            conflict_strategy=self._conflict_strategy.value,
                        )
                        pending_async_results[batch_index] = pool_manager.submit_batch(task)
                        batch_start_times[batch_index] = time.time()
                        self._logger.info("Submitted batch %d/%d", batch_index, total_batches)
                        emit_log(
                            f"Processing batch {batch_index}/{total_batches} ({len(batch)} files)...",
                            log_level="info",
                        )
                    except Exception as submission_error:
                        error_message = (
                            "Batch submission failed: "
                            f"{type(submission_error).__name__}: {submission_error}"
                        )
                        self._logger.warning(
                            "Failed to submit batch %d/%d: %s",
                            batch_index,
                            total_batches,
                            submission_error,
                            exc_info=True,
                        )

                        emit_log(
                            f"Batch {batch_index} submission failed: {submission_error}",
                            log_level="warning",
                        )

                        results_by_batch[batch_index] = build_batch_failure_results(
                            task_id=task_id,
                            batch_paths=batch,
                            error_message=error_message,
                        )
                        batch_times.append(0.0)
                        emit_log(
                            f"Batch {batch_index} completed: 0/{len(batch)} files succeeded",
                            log_level="progress_step",
                        )

                pending_batch_ids = set(pending_async_results.keys())
                while pending_batch_ids:
                    if self._check_cancellation() and not cancellation_detected:
                        cancellation_detected = True
                        cancellation_started_at = time.time()
                        result.success = False
                        self._transition_state(JobState.CANCELLED, result)
                        pool_manager.signal_cancellation()
                        emit_log(
                            "Cancellation requested; waiting for in-flight workers to finish current files",
                            log_level="warning",
                        )

                    made_progress = False

                    for batch_index in sorted(list(pending_batch_ids)):
                        async_result = pending_async_results[batch_index]
                        if not async_result.ready():
                            continue

                        batch = batches[batch_index - 1]
                        elapsed = time.time() - batch_start_times.get(batch_index, time.time())
                        batch_times.append(elapsed)

                        batch_results = pool_manager.collect_results(
                            [async_result],
                            timeout=0.1,
                        )

                        if not batch_results:
                            if cancellation_detected:
                                batch_results = build_batch_cancelled_results(
                                    task_id=f"batch-{batch_index}",
                                    batch_paths=batch,
                                )
                            else:
                                error_message = "Batch completed without returning worker results"
                                self._logger.warning(
                                    "Batch %d returned no results; synthesizing failures",
                                    batch_index,
                                )
                                batch_results = build_batch_failure_results(
                                    task_id=f"batch-{batch_index}",
                                    batch_paths=batch,
                                    error_message=error_message,
                                    processing_time=elapsed,
                                )

                        results_by_batch[batch_index] = batch_results
                        pending_batch_ids.remove(batch_index)
                        made_progress = True

                        success_count = sum(1 for batch_result in batch_results if batch_result.success)
                        cancelled_count = sum(
                            1 for batch_result in batch_results if batch_result.was_cancelled
                        )
                        processed_count = len(batch_results) - cancelled_count

                        batch_summary = (
                            f"Batch {batch_index} completed: {success_count}/{processed_count} "
                            "files succeeded"
                        )
                        if cancelled_count > 0:
                            batch_summary += f" ({cancelled_count} skipped due to cancellation)"

                        emit_log(batch_summary, log_level="progress_step")
                        self._logger.info(
                            "Batch %d completed in %.2fs (processed=%d, cancelled=%d)",
                            batch_index,
                            elapsed,
                            processed_count,
                            cancelled_count,
                        )

                    now = time.time()

                    if cancellation_detected and cancellation_started_at is not None:
                        cancellation_elapsed = now - cancellation_started_at
                        if cancellation_elapsed >= pool_manager.grace_period and pending_batch_ids:
                            force_message = (
                                "Cancellation grace period exceeded "
                                f"({pool_manager.grace_period:.1f}s); forcing worker shutdown"
                            )
                            self._logger.warning(force_message)
                            emit_log(force_message, log_level="warning")

                            if pool_manager.pool is not None:
                                pool_manager.shutdown_pool(grace_period=pool_manager.grace_period)
                                graceful_shutdown_invoked = True

                            for remaining_batch_index in sorted(list(pending_batch_ids)):
                                remaining_batch = batches[remaining_batch_index - 1]
                                remaining_elapsed = (
                                    now - batch_start_times.get(remaining_batch_index, now)
                                )
                                batch_times.append(max(0.0, remaining_elapsed))
                                results_by_batch[remaining_batch_index] = (
                                    build_batch_cancelled_results(
                                        task_id=f"batch-{remaining_batch_index}",
                                        batch_paths=remaining_batch,
                                    )
                                )
                                cancelled_uncollected_batches.append(remaining_batch_index)
                                pending_batch_ids.remove(remaining_batch_index)
                                emit_log(
                                    (
                                        f"Batch {remaining_batch_index} completed: 0/0 files "
                                        f"succeeded ({len(remaining_batch)} skipped due to cancellation)"
                                    ),
                                    log_level="progress_step",
                                )
                            continue

                    for batch_index in sorted(list(pending_batch_ids)):
                        elapsed = now - batch_start_times.get(batch_index, now)
                        if elapsed <= DEFAULT_BATCH_TIMEOUT_SECONDS:
                            continue

                        batch = batches[batch_index - 1]
                        timeout_message = (
                            f"Batch {batch_index} timed out after {elapsed:.1f}s "
                            f"(limit={DEFAULT_BATCH_TIMEOUT_SECONDS:.1f}s)"
                        )
                        self._logger.error(timeout_message)
                        emit_log(timeout_message, log_level="warning")

                        results_by_batch[batch_index] = build_batch_failure_results(
                            task_id=f"batch-{batch_index}",
                            batch_paths=batch,
                            error_message=timeout_message,
                            processing_time=elapsed,
                        )
                        batch_times.append(elapsed)
                        pending_batch_ids.remove(batch_index)
                        made_progress = True

                        emit_log(
                            f"Batch {batch_index} completed: 0/{len(batch)} files succeeded",
                            log_level="progress_step",
                        )

                    if not made_progress and pending_batch_ids:
                        time.sleep(0.2 if cancellation_detected else 0.5)

                if (
                    cancellation_detected
                    and pool_manager.pool is not None
                    and not graceful_shutdown_invoked
                ):
                    pool_manager.shutdown_pool(grace_period=pool_manager.grace_period)
                    graceful_shutdown_invoked = True

                worker_results: list[WorkerResult] = []
                for batch_index in range(1, total_batches + 1):
                    batch_results = results_by_batch.get(batch_index)
                    if batch_results is None:
                        batch = batches[batch_index - 1]
                        if cancellation_detected:
                            batch_results = build_batch_cancelled_results(
                                task_id=f"batch-{batch_index}",
                                batch_paths=batch,
                            )
                            cancelled_uncollected_batches.append(batch_index)
                        else:
                            error_message = f"Batch {batch_index} was not processed"
                            batch_results = build_batch_failure_results(
                                task_id=f"batch-{batch_index}",
                                batch_paths=batch,
                                error_message=error_message,
                            )
                    worker_results.extend(batch_results)

                for worker_index, worker_result in enumerate(worker_results):
                    process_result = self._convert_worker_result_to_process_result(worker_result)
                    result.processed_files.append(process_result)
                    self._file_processing_times.append(float(worker_result.processing_time))
                    worker_transformation_counts[str(process_result.file_path)] = int(
                        worker_result.transformation_count
                    )

                    if process_result.warnings:
                        result.warnings.extend(process_result.warnings)
                        warnings_by_file[str(process_result.file_path)].extend(
                            process_result.warnings
                        )

                    if process_result.detailed_errors:
                        result.detailed_errors.extend(process_result.detailed_errors)

                    if process_result.was_cancelled:
                        cancelled_files.append(str(process_result.file_path))
                        continue

                    if process_result.success:
                        total_success += 1
                        continue

                    result.errors.extend(process_result.errors)
                    joined_errors = "; ".join(process_result.errors) if process_result.errors else "Unknown error"
                    emit_log(
                        f"Error processing {process_result.file_path.name}: {joined_errors}",
                        current_file=process_result.file_path.name,
                        log_level="error",
                    )

                    if self._error_strategy not in {ErrorStrategy.STOP, ErrorStrategy.ASK}:
                        continue

                    if "error_decisions" not in result.metadata:
                        result.metadata["error_decisions"] = []
                    if "files_failed_with_errors" not in result.metadata:
                        result.metadata["files_failed_with_errors"] = []

                    should_continue = False
                    decision = "stop"

                    if self._error_strategy == ErrorStrategy.STOP:
                        self._logger.error(
                            "Error processing %s. Halting multiprocessing (STOP strategy).",
                            process_result.file_path.name,
                        )
                    elif error_callback is None:
                        callback_missing_message = (
                            "ErrorStrategy.ASK requires error_callback during multiprocessing; "
                            f"defaulting to STOP after error in {process_result.file_path.name}"
                        )
                        self._logger.error(callback_missing_message)
                        emit_log(
                            callback_missing_message,
                            current_file=process_result.file_path.name,
                            log_level="error",
                        )
                    else:
                        self._logger.info(
                            "Error processing %s. Prompting user for decision (ASK strategy).",
                            process_result.file_path.name,
                        )
                        try:
                            should_continue = bool(
                                error_callback(process_result.file_path, process_result.errors)
                            )
                        except Exception as callback_error:
                            callback_error_message = (
                                "Error callback raised exception during multiprocessing: "
                                f"{type(callback_error).__name__}: {callback_error}"
                            )
                            self._logger.error(callback_error_message, exc_info=True)
                            result.errors.append(callback_error_message)
                            emit_log(
                                callback_error_message,
                                current_file=process_result.file_path.name,
                                log_level="error",
                            )
                            should_continue = False

                        decision = "continue" if should_continue else "stop"
                        self._logger.info(
                            "User chose to %s after error in %s",
                            decision,
                            process_result.file_path.name,
                        )
                        emit_log(
                            f"User chose to {decision} after error",
                            current_file=process_result.file_path.name,
                        )

                    result.metadata["error_decisions"].append(
                        {
                            "file": str(process_result.file_path),
                            "errors": list(process_result.errors),
                            "decision": decision,
                        }
                    )
                    result.metadata["files_failed_with_errors"].append(
                        str(process_result.file_path)
                    )

                    if should_continue:
                        continue

                    halted_due_to_worker_errors = True
                    result.success = False
                    self._transition_state(JobState.FAILED, result)

                    remaining_results = worker_results[worker_index + 1 :]
                    skipped_due_to_halt = [
                        str(Path(remaining_result.file_path))
                        for remaining_result in remaining_results
                    ]

                    result.metadata["halted_due_to_worker_errors"] = True
                    result.metadata["halted_by_error_strategy"] = self._error_strategy.value
                    result.metadata["halted_on_file"] = str(process_result.file_path)
                    result.metadata["halted_on_errors"] = list(process_result.errors)
                    result.metadata["files_skipped_due_to_error_strategy"] = skipped_due_to_halt

                    emit_log(
                        f"Halting multiprocessing due to worker error in {process_result.file_path.name}",
                        current_file=process_result.file_path.name,
                        log_level="error",
                    )

                    if pool_manager.pool is not None:
                        pool_manager.terminate_pool()

                    break
        finally:
            self._active_pool_manager = None

        if cancellation_detected:
            result.success = False
            if result.current_state != JobState.CANCELLED:
                self._transition_state(JobState.CANCELLED, result)
            result.metadata["was_cancelled"] = True
            result.metadata["cancelled_files"] = list(cancelled_files)
            result.metadata["cancelled_uncollected_batches"] = sorted(
                set(cancelled_uncollected_batches)
            )
            emit_log(
                f"Multiprocessing cancelled: {total_success} file(s) processed before cancellation",
                log_level="warning",
            )
        elif halted_due_to_worker_errors:
            emit_log(
                "Multiprocessing halted due to worker errors",
                log_level="warning",
            )
        else:
            emit_log(
                f"Multiprocessing complete: {total_success}/{len(processing_order)} files processed",
                log_level="success",
            )

        result.metadata["worker_transformation_counts"] = worker_transformation_counts

        self._parallel_processing_metadata = {
            "worker_count": worker_count_used,
            "batch_count": batch_count,
            "batch_size": batch_size_used,
            "batch_size_adjustments": batch_size_adjustments,
            "batch_times_seconds": batch_times,
            "warnings_by_file": dict(warnings_by_file),
            "halted_due_to_worker_errors": halted_due_to_worker_errors,
            "files_skipped_due_to_error_strategy": skipped_due_to_halt,
            "cancelled_files": list(cancelled_files),
            "cancelled_uncollected_batches": sorted(set(cancelled_uncollected_batches)),
        }

    def _scan_and_extract_symbols(
        self,
        input_files: list[Path],
        result: OrchestrationResult
    ) -> tuple[set[Path], dict[Path, SymbolTable | LuaSymbolTable]]:
        """Scan all files and extract symbols without caching ASTs.

        This method parses files to extract symbols but discards ASTs immediately
        to keep memory bounded for large projects. Files are re-parsed during
        the processing phase.

        Args:
            input_files: List of files to scan
            result: OrchestrationResult to update with warnings/errors

        Returns:
            Tuple of (successfully_parsed_files set, symbol_tables dict)
        """
        successfully_parsed: set[Path] = set()
        symbol_tables: dict[Path, SymbolTable | LuaSymbolTable] = {}

        for file_path in input_files:
            resolved_path = file_path.resolve()
            language = self._detect_language(resolved_path)

            try:
                if language == "python":
                    parse_result = self.python_processor.parse_file(resolved_path)
                    if parse_result.success and parse_result.ast_node:
                        # Extract symbols then discard AST to bound memory
                        symbols = self.python_processor.extract_symbols(
                            parse_result.ast_node, resolved_path
                        )
                        symbol_tables[resolved_path] = symbols
                        successfully_parsed.add(resolved_path)
                        # AST is not stored - will be garbage collected
                    else:
                        result.warnings.append(
                            f"Failed to parse {file_path.name}: {parse_result.errors}"
                        )
                elif language == "lua":
                    parse_result = self.lua_processor.parse_file(resolved_path)
                    if parse_result.success and parse_result.ast_node:
                        # Extract symbols then discard AST to bound memory
                        symbols = self.lua_processor.extract_symbols(
                            parse_result.ast_node, resolved_path
                        )
                        symbol_tables[resolved_path] = symbols
                        successfully_parsed.add(resolved_path)
                        # AST is not stored - will be garbage collected
                    else:
                        result.warnings.append(
                            f"Failed to parse {file_path.name}: {parse_result.errors}"
                        )
                else:
                    result.warnings.append(
                        f"Unsupported file type: {file_path.name}"
                    )
            except Exception as e:
                result.warnings.append(
                    f"Error scanning {file_path.name}: {e}"
                )
                self._logger.warning(f"Error scanning {file_path}: {e}")

        self._logger.info(
            f"Scanned {len(successfully_parsed)} files, extracted symbols from "
            f"{len(symbol_tables)} files (ASTs discarded to bound memory)"
        )

        return successfully_parsed, symbol_tables

    def _build_dependency_graph(
        self,
        parsed_files: set[Path],
        symbol_tables: dict[Path, SymbolTable | LuaSymbolTable],
        config: dict[str, Any],
        result: OrchestrationResult,
        project_root: Path | None = None
    ) -> DependencyGraph:
        """Build the dependency graph from parsed files.

        Args:
            parsed_files: Set of successfully parsed file paths
            symbol_tables: Dict of file paths to symbol tables
            config: Configuration options
            result: OrchestrationResult to update
            project_root: Explicit project root directory (if None, computed from files)

        Returns:
            Constructed DependencyGraph
        """
        # Determine project root from input files or use provided root
        if project_root is not None:
            root = project_root
        elif parsed_files:
            first_file = next(iter(parsed_files))
            root = first_file.parent
        else:
            root = Path.cwd()

        analyzer = DependencyAnalyzer(root)

        # Build graph incrementally from symbol tables
        files_with_symbols = [
            (path, symbol_tables[path])
            for path in parsed_files
            if path in symbol_tables
        ]

        try:
            graph = analyzer.build_graph_incremental(files_with_symbols)
            self._logger.info(
                f"Built dependency graph with {len(graph.nodes)} nodes, "
                f"{len(graph.edges)} edges"
            )
        except DependencyResolutionError as e:
            result.warnings.append(f"Unresolved import: {e.message}")
            # Create a minimal graph with just the nodes
            graph = DependencyGraph()
            for path in parsed_files:
                language = self._detect_language(path)
                symbols = symbol_tables.get(path)
                exports = []
                if symbols:
                    exports = symbols.get_exported_symbols()
                graph.add_node(path, language, [], exports)

        return graph

    def _build_global_symbol_table(
        self,
        graph: DependencyGraph,
        symbol_tables: dict[Path, SymbolTable | LuaSymbolTable],
        config: dict[str, Any],
        result: OrchestrationResult
    ) -> GlobalSymbolTable:
        """Build the global symbol table from dependency graph.

        Args:
            graph: The dependency graph
            symbol_tables: Dict of file paths to symbol tables
            config: Configuration options
            result: OrchestrationResult to update

        Returns:
            Pre-computed GlobalSymbolTable (frozen)
        """
        builder = SymbolTableBuilder()

        try:
            global_table = builder.build_from_dependency_graph(
                graph, symbol_tables, config
            )
            self._logger.info(
                f"Built global symbol table with "
                f"{len(global_table.get_all_symbols())} symbols"
            )
        except CircularDependencyError as e:
            # Surface clear error message about circular dependency
            result.success = False
            error_msg = f"Circular dependency detected during symbol table build: {e.message}"
            result.errors.append(error_msg)
            self._logger.error(error_msg)

            # Use deterministic fallback order (original file order) so mangling still occurs
            result.warnings.append(
                "Building symbol table using original file order as fallback due to circular dependencies. "
                "Mangling will still occur but cross-file symbol consistency may be affected."
            )
            self._logger.warning(
                "Using original file order fallback for symbol table build"
            )

            # Build symbol table with fallback order (files in symbol_tables dict order)
            global_table = builder.build_with_fallback_order(
                symbol_tables, config
            )
            self._logger.info(
                f"Built global symbol table (fallback) with "
                f"{len(global_table.get_all_symbols())} symbols"
            )
        except Exception as e:
            # For other errors, still provide a meaningful message
            error_msg = f"Error building symbol table: {e}"
            result.errors.append(error_msg)
            self._logger.error(error_msg, exc_info=True)

            # Create a fallback table so processing can continue
            result.warnings.append(
                "Building symbol table with fallback order due to error. "
                "Mangling will still occur but may be affected."
            )
            global_table = builder.build_with_fallback_order(
                symbol_tables, config
            )

        return global_table

    def _extract_error_location(self, error_message: str) -> tuple[int | None, int | None]:
        """Extract line and column information from an error string when available."""
        parsed = parse_error(error_message)
        if parsed is not None:
            return parsed.get("line"), parsed.get("column")

        file_style_match = re.search(r":(\d+):(\d+):", error_message)
        if file_style_match:
            return int(file_style_match.group(1)), int(file_style_match.group(2))

        line_column_match = re.search(
            r"\bline\s+(\d+)\s*,\s*column\s+(\d+)",
            error_message,
            flags=re.IGNORECASE,
        )
        if line_column_match:
            return int(line_column_match.group(1)), int(line_column_match.group(2))

        return extract_line_column(error_message)

    def _create_detailed_error(
        self,
        file_path: Path,
        error_type: str,
        message: str,
        line: int | None = None,
        column: int | None = None,
        location: tuple[int | None, int | None] | None = None,
    ) -> dict[str, Any]:
        """Build a structured detailed error dictionary for reporting."""
        resolved_line = line
        resolved_column = column

        if location is not None:
            resolved_line, resolved_column = location

        if resolved_line is None and resolved_column is None:
            extracted_line, extracted_column = self._extract_error_location(str(message))
            resolved_line = extracted_line
            resolved_column = extracted_column

        return {
            "file_path": str(file_path),
            "line": resolved_line,
            "column": resolved_column,
            "error_type": error_type,
            "message": str(message),
        }

    def _format_feature_warning(
        self,
        warning: Any,
        fallback_file: Path,
    ) -> tuple[str, str]:
        """Normalize feature warnings into canonical warning and progress strings."""
        feature_name = getattr(warning, "feature_name", "UnsupportedFeature")
        warning_message = getattr(warning, "description", str(warning))
        file_path_value = getattr(warning, "file_path", None) or fallback_file
        line_number = getattr(warning, "line_number", 0) or 0
        column_offset = getattr(warning, "column_offset", 0) or 0
        suggestion = getattr(warning, "suggestion", None)

        structured_warning = UnsupportedFeatureWarning(
            feature_name=feature_name,
            file_path=file_path_value,
            line_number=line_number,
            column_offset=column_offset,
            message=warning_message,
            suggestion=suggestion,
        )
        warning_text = str(structured_warning)
        warning_path = Path(file_path_value)
        progress_text = (
            f"Warning in {warning_path.name}: {feature_name} at line {line_number}"
        )
        return warning_text, progress_text

    def _process_file_in_order(
        self,
        file_path: Path,
        global_table: GlobalSymbolTable,
        output_dir: Path,
        config: dict[str, Any],
        output_writer: OutputWriter,
        emit_log_callback: Callable[[str, str | None, str], None] | None = None,
    ) -> ProcessResult:
        """Process a single file using the global symbol table.

        Re-parses the file just-in-time to avoid caching ASTs in memory.
        AST is released immediately after writing output to bound memory.

        Args:
            file_path: Path to the file
            global_table: Pre-computed GlobalSymbolTable
            output_dir: Output directory
            config: Configuration options
            output_writer: OutputWriter instance for conflict-safe file writing
            emit_log_callback: Optional log emitter for progress warning messages

        Returns:
            ProcessResult for this file
        """
        language = self._detect_language(file_path)

        try:
            # Re-parse file just-in-time to bound memory usage
            if language == "python":
                parse_result = self.python_processor.parse_file(file_path)
                parse_warning_messages: list[str] = []
                if parse_result.warnings:
                    for parse_warning in parse_result.warnings:
                        warning_text, progress_warning = self._format_feature_warning(
                            parse_warning,
                            file_path,
                        )
                        parse_warning_messages.append(warning_text)
                        self._logger.warning(warning_text)
                        if emit_log_callback:
                            emit_log_callback(
                                progress_warning,
                                current_file=file_path.name,
                                log_level="warning",
                            )

                if not parse_result.success or not parse_result.ast_node:
                    parse_errors = parse_result.errors or ["Failed to re-parse file"]
                    detailed_errors: list[dict[str, Any]] = []
                    for parse_error in parse_errors:
                        line, column = self._extract_error_location(parse_error)
                        detailed_errors.append(
                            self._create_detailed_error(
                                file_path=file_path,
                                line=line,
                                column=column,
                                error_type="ParseError",
                                message=parse_error,
                            )
                        )
                    return ProcessResult(
                        file_path=file_path,
                        output_path=None,
                        success=False,
                        errors=parse_errors,
                        detailed_errors=detailed_errors,
                        warnings=parse_warning_messages,
                    )

                # Create engine once for this file to ensure consistent runtime keys
                engine = None
                if self._config is not None:
                    from obfuscator.core.obfuscation_engine import ObfuscationEngine
                    engine = ObfuscationEngine(self._config)

                ast_node = parse_result.ast_node
                transform_result = self.python_processor.obfuscate_with_symbol_table(
                    ast_node, file_path, global_table, engine=engine
                )

                # Get the engine that was used (from processor if not provided)
                used_engine = transform_result.get("engine") or self.python_processor.get_engine() or engine

                # Track engine for hybrid mode runtime generation with its language
                if used_engine:
                    processed_entry = ProcessedEngine(engine=used_engine, language="python")
                    if processed_entry not in self._processed_engines:
                        self._processed_engines.append(processed_entry)

                if transform_result["success"]:
                    gen_result = self.python_processor.generate_code(
                        transform_result["ast_node"]
                    )
                    # Release AST references to allow garbage collection
                    del ast_node
                    del transform_result

                    if gen_result.success:
                        final_code = self._inject_embedded_runtime(
                            gen_result.code, "python", used_engine
                        )
                        write_result = output_writer.write_with_structure(
                            input_path=file_path,
                            output_base=output_dir,
                            content=final_code,
                            project_root=self._project_root,
                        )

                        if not write_result.success:
                            write_error = (
                                write_result.error
                                or f"Failed to write output for {file_path.name}"
                            )
                            self._logger.error(
                                f"Write failed for {file_path}: {write_error}"
                            )
                            return ProcessResult(
                                file_path=file_path,
                                output_path=None,
                                success=False,
                                errors=[write_error],
                                detailed_errors=[
                                    self._create_detailed_error(
                                        file_path=file_path,
                                        error_type="WriteError",
                                        message=write_error,
                                    )
                                ],
                                warnings=parse_warning_messages,
                                conflict_resolution=write_result.conflict_resolution,
                            )

                        if write_result.output_path is None:
                            # File was skipped
                            skip_warning = (
                                f"Skipped {file_path.name} - file exists at output path"
                            )
                            return ProcessResult(
                                file_path=file_path,
                                output_path=None,
                                success=True,
                                warnings=[*parse_warning_messages, skip_warning],
                                conflict_resolution=write_result.conflict_resolution,
                            )

                        return ProcessResult(
                            file_path=file_path,
                            output_path=write_result.output_path,
                            success=True,
                            warnings=parse_warning_messages,
                            conflict_resolution=write_result.conflict_resolution,
                        )
                    else:
                        generation_errors = gen_result.errors or ["Code generation failed"]
                        detailed_generation_errors: list[dict[str, Any]] = []
                        for error in generation_errors:
                            line, column = self._extract_error_location(error)
                            detailed_generation_errors.append(
                                self._create_detailed_error(
                                    file_path=file_path,
                                    error_type="CodeGenerationError",
                                    message=error,
                                    location=(line, column),
                                )
                            )
                        return ProcessResult(
                            file_path=file_path,
                            output_path=None,
                            success=False,
                            errors=generation_errors,
                            detailed_errors=detailed_generation_errors,
                            warnings=parse_warning_messages,
                        )
                else:
                    transform_errors = transform_result.get("errors", ["Transformation failed"])
                    detailed_transform_errors: list[dict[str, Any]] = []
                    for error in transform_errors:
                        parsed_error = parse_error(error)
                        if parsed_error is not None:
                            line = parsed_error.get("line")
                            column = parsed_error.get("column")
                        else:
                            line, column = self._extract_error_location(error)

                        detailed_transform_errors.append(
                            self._create_detailed_error(
                                file_path=file_path,
                                error_type="TransformationError",
                                message=error,
                                location=(line, column),
                            )
                        )

                    return ProcessResult(
                        file_path=file_path,
                        output_path=None,
                        success=False,
                        errors=transform_errors,
                        detailed_errors=detailed_transform_errors,
                        warnings=parse_warning_messages,
                    )

            elif language == "lua":
                parse_result = self.lua_processor.parse_file(file_path)
                parse_warning_messages: list[str] = []
                if parse_result.warnings:
                    for parse_warning in parse_result.warnings:
                        warning_text, progress_warning = self._format_feature_warning(
                            parse_warning,
                            file_path,
                        )
                        parse_warning_messages.append(warning_text)
                        self._logger.warning(warning_text)
                        if emit_log_callback:
                            emit_log_callback(
                                progress_warning,
                                current_file=file_path.name,
                                log_level="warning",
                            )

                if not parse_result.success or not parse_result.ast_node:
                    parse_errors = parse_result.errors or ["Failed to re-parse file"]
                    detailed_errors = []
                    for parse_error in parse_errors:
                        line, column = self._extract_error_location(parse_error)
                        detailed_errors.append(
                            self._create_detailed_error(
                                file_path=file_path,
                                line=line,
                                column=column,
                                error_type="ParseError",
                                message=parse_error,
                            )
                        )
                    return ProcessResult(
                        file_path=file_path,
                        output_path=None,
                        success=False,
                        errors=parse_errors,
                        detailed_errors=detailed_errors,
                        warnings=parse_warning_messages,
                    )

                # Create engine once for this file to ensure consistent runtime keys
                engine = None
                if self._config is not None:
                    from obfuscator.core.obfuscation_engine import ObfuscationEngine
                    engine = ObfuscationEngine(self._config)

                ast_node = parse_result.ast_node
                transform_result = self.lua_processor.obfuscate_with_symbol_table(
                    ast_node, file_path, global_table, engine=engine
                )

                # Get the engine that was used (from processor if not provided)
                used_engine = transform_result.get("engine") or self.lua_processor.get_engine() or engine

                # Track engine for hybrid mode runtime generation with its language
                if used_engine:
                    processed_entry = ProcessedEngine(engine=used_engine, language="lua")
                    if processed_entry not in self._processed_engines:
                        self._processed_engines.append(processed_entry)

                # Release AST reference to allow garbage collection
                del ast_node

                if transform_result["success"]:
                    final_code = self._inject_embedded_runtime(
                        transform_result["code"], "lua", used_engine
                    )
                    write_result = output_writer.write_with_structure(
                        input_path=file_path,
                        output_base=output_dir,
                        content=final_code,
                        project_root=self._project_root,
                    )

                    if not write_result.success:
                        write_error = (
                            write_result.error
                            or f"Failed to write output for {file_path.name}"
                        )
                        self._logger.error(
                            f"Write failed for {file_path}: {write_error}"
                        )
                        return ProcessResult(
                            file_path=file_path,
                            output_path=None,
                            success=False,
                            errors=[write_error],
                            detailed_errors=[
                                self._create_detailed_error(
                                    file_path=file_path,
                                    error_type="WriteError",
                                    message=write_error,
                                )
                            ],
                            warnings=parse_warning_messages,
                            conflict_resolution=write_result.conflict_resolution,
                        )

                    if write_result.output_path is None:
                        # File was skipped
                        skip_warning = (
                            f"Skipped {file_path.name} - file exists at output path"
                        )
                        return ProcessResult(
                            file_path=file_path,
                            output_path=None,
                            success=True,
                            warnings=[*parse_warning_messages, skip_warning],
                            conflict_resolution=write_result.conflict_resolution,
                        )

                    return ProcessResult(
                        file_path=file_path,
                        output_path=write_result.output_path,
                        success=True,
                        warnings=parse_warning_messages,
                        conflict_resolution=write_result.conflict_resolution,
                    )
                else:
                    transform_errors = transform_result.get("errors", ["Transformation failed"])
                    detailed_transform_errors: list[dict[str, Any]] = []
                    for error in transform_errors:
                        parsed_error = parse_error(error)
                        if parsed_error is not None:
                            line = parsed_error.get("line")
                            column = parsed_error.get("column")
                        else:
                            line, column = self._extract_error_location(error)

                        detailed_transform_errors.append(
                            self._create_detailed_error(
                                file_path=file_path,
                                error_type="TransformationError",
                                message=error,
                                location=(line, column),
                            )
                        )

                    return ProcessResult(
                        file_path=file_path,
                        output_path=None,
                        success=False,
                        errors=transform_errors,
                        detailed_errors=detailed_transform_errors,
                        warnings=parse_warning_messages,
                    )
            else:
                unsupported_error = f"Unsupported language: {language}"
                return ProcessResult(
                    file_path=file_path,
                    output_path=None,
                    success=False,
                    errors=[unsupported_error],
                    detailed_errors=[
                        self._create_detailed_error(
                            file_path=file_path,
                            error_type="UnexpectedException",
                            message=unsupported_error,
                        )
                    ],
                )

        except Exception as e:
            exception_error = str(e)
            self._logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            return ProcessResult(
                file_path=file_path,
                output_path=None,
                success=False,
                errors=[exception_error],
                detailed_errors=[
                    self._create_detailed_error(
                        file_path=file_path,
                        error_type="UnexpectedException",
                        message=exception_error,
                    )
                ],
            )

    def _inject_embedded_runtime(
        self,
        code: str,
        language: str,
        engine: "ObfuscationEngine | None" = None
    ) -> str:
        """Inject embedded runtime code into obfuscated code.

        Uses the provided ObfuscationEngine's runtime_manager (if available) or
        creates a new RuntimeManager to check if runtime requirements exist,
        and prepends formatted runtime code if needed.

        Args:
            code: The obfuscated code to inject runtime into.
            language: Target language ("python" or "lua").
            engine: Optional ObfuscationEngine instance used for transformations.
                   If provided, its runtime_manager will be used to ensure key
                   consistency between transformed code and runtime code.

        Returns:
            Code with embedded runtime prepended, or original code if no runtime needed.
        """
        try:
            # Check if in hybrid mode - inject imports instead of embedded runtime
            if self._config is not None and self._config.runtime_mode == "hybrid":
                return self._inject_hybrid_imports(code, language, engine)

            # Only proceed with embedded runtime if explicitly in embedded mode
            if self._config is None or self._config.runtime_mode != "embedded":
                return code
            # Use the engine's tracked required_runtimes when available, otherwise fallback
            if engine is not None:
                # Use engine's tracked features (only those actually applied)
                has_runtime = engine.has_runtime_requirements()
                if has_runtime:
                    runtime_code = engine.get_required_runtime_code(language)
                else:
                    runtime_code = ""
            else:
                # Fallback: create a new RuntimeManager to check all enabled features
                runtime_manager = RuntimeManager(self._config)
                has_runtime = runtime_manager.has_runtime_requirements(language)
                if has_runtime:
                    runtime_code = runtime_manager.get_combined_runtime(language)
                else:
                    runtime_code = ""

            if not runtime_code:
                return code

            # Format runtime code with language-specific headers
            if language == "python":
                separator = "# " + "=" * 70
                header = (
                    f"{separator}\n"
                    f"# EMBEDDED OBFUSCATION RUNTIME\n"
                    f"{separator}\n\n"
                )
                footer = (
                    f"\n\n{separator}\n"
                    f"# END RUNTIME CODE\n"
                    f"{separator}\n\n"
                )
            elif language == "lua":
                separator = "-- " + "=" * 70
                header = (
                    f"{separator}\n"
                    f"-- EMBEDDED OBFUSCATION RUNTIME\n"
                    f"{separator}\n\n"
                )
                footer = (
                    f"\n\n{separator}\n"
                    f"-- END RUNTIME CODE\n"
                    f"{separator}\n\n"
                )
            else:
                # Fallback for unsupported languages
                return code

            self._logger.info(
                f"Injected embedded runtime code for {language} ({len(runtime_code)} chars)"
            )

            return header + runtime_code + footer + code

        except Exception as e:
            self._logger.warning(f"Failed to inject runtime code: {e}")
            return code

    def _inject_hybrid_imports(
        self,
        code: str,
        language: str,
        engine: "ObfuscationEngine | None" = None
    ) -> str:
        """Inject import statements for hybrid mode runtime.

        Instead of embedding runtime code, this method prepends import/require
        statements that reference the shared runtime file (obf_runtime.py or
        obf_runtime.lua) generated after all files are processed.

        Args:
            code: The obfuscated code to inject imports into.
            language: Target language ("python" or "lua").
            engine: Optional ObfuscationEngine instance used for transformations.

        Returns:
            Code with import statements prepended, or original code if no imports needed.
        """
        try:
            # Check if engine has runtime requirements
            if engine is None or not engine.has_runtime_requirements():
                return code

            # Get required runtime types from the engine
            required_runtimes = engine.required_runtimes
            if not required_runtimes:
                return code

            # Generate import statements for each required runtime
            runtime_manager = RuntimeManager(self._config) if self._config else None
            if runtime_manager is None:
                return code

            import_statements: list[str] = []

            if language == "python":
                # Add comment header for Python
                import_statements.append("# Obfuscation Runtime Imports")

                # Collect all function names from required runtimes
                all_functions: list[str] = []
                for runtime_type in required_runtimes:
                    functions = runtime_manager.get_runtime_function_names(runtime_type)
                    all_functions.extend(functions)

                # Generate single import statement for all functions
                if all_functions:
                    import_statements.append(
                        f"from obf_runtime import {', '.join(all_functions)}"
                    )

                import_statements.append("")  # Empty line after imports
                import_statements.append(f"# {'=' * 70}")
                import_statements.append("")

            elif language == "lua":
                # Add comment header for Lua
                import_statements.append("-- Obfuscation Runtime Imports")
                # Single require statement for all runtimes
                import_statements.append('local rt = require("obf_runtime")')
                import_statements.append("")  # Empty line after imports
                import_statements.append(f"-- {'=' * 70}")
                import_statements.append("")

            else:
                # Unsupported language - return code unchanged
                return code

            combined_imports = "\n".join(import_statements)
            self._logger.info(
                f"Injected hybrid imports for {language} ({len(required_runtimes)} runtime modules)"
            )

            return combined_imports + code

        except Exception as e:
            self._logger.warning(f"Failed to inject hybrid imports: {e}")
            return code

    def _generate_hybrid_runtime_files(
        self,
        output_dir: Path,
        result: OrchestrationResult
    ) -> None:
        """DEPRECATED: Generate shared runtime files for hybrid mode.

        DEPRECATED: Replaced by ``OutputWriter.write_runtime_library`` in
        ``process_files``. This method is retained for backward compatibility
        with direct callers.

        Collects all unique runtime requirements from tracked engines and generates
        consolidated runtime files (obf_runtime.py and/or obf_runtime.lua) in
        the output directory root.

        Args:
            output_dir: Directory to write runtime files to.
            result: OrchestrationResult to update with runtime file metadata.
        """
        self._logger.warning(
            "DEPRECATED: _generate_hybrid_runtime_files() is deprecated; "
            "use OutputWriter.write_runtime_library() instead."
        )

        try:
            # Validate output directory
            if not output_dir.exists():
                self._logger.warning(
                    f"Output directory does not exist: {output_dir}, skipping runtime generation"
                )
                return

            if not output_dir.is_dir():
                self._logger.warning(
                    f"Output path is not a directory: {output_dir}, skipping runtime generation"
                )
                return

            # Collect all unique runtime requirements from tracked engines
            python_runtimes: set[str] = set()
            lua_runtimes: set[str] = set()

            for entry in self._processed_engines:
                engine = entry.engine
                language = entry.language
                
                if hasattr(engine, 'required_runtimes') and engine.required_runtimes:
                    if language == "python":
                        python_runtimes.update(engine.required_runtimes)
                    elif language == "lua":
                        lua_runtimes.update(engine.required_runtimes)

            # Generate runtime files for each language
            runtime_files: list[str] = []

            for language, runtimes in [("python", python_runtimes), ("lua", lua_runtimes)]:
                if not runtimes:
                    continue

                try:
                    # Find the first engine for this language to use its runtime_manager
                    target_engine = None
                    for entry in self._processed_engines:
                        if entry.language == language:
                            target_engine = entry.engine
                            break
                    
                    if target_engine is None or not hasattr(target_engine, 'runtime_manager'):
                        self._logger.warning(
                            f"No engine with runtime_manager found for {language}, skipping runtime generation"
                        )
                        continue

                    runtime_manager = target_engine.runtime_manager
                    
                    # Generate runtime code only for required runtimes using the engine's stored keys
                    parts: list[str] = []
                    
                    # Add header comment
                    if language == "python":
                        parts.append('# Obfuscation Runtime Code - Applied Features')
                        parts.append('# Generated by ScriptShield RuntimeManager')
                        parts.append('')
                    else:  # lua
                        parts.append('-- Obfuscation Runtime Code - Applied Features')
                        parts.append('-- Generated by ScriptShield RuntimeManager')
                        parts.append('')
                    
                    # Generate runtime code only for required runtimes
                    for i, runtime_type in enumerate(runtimes):
                        code = runtime_manager.generate_runtime_code(runtime_type, language)
                        if code:
                            if i > 0:
                                # Add separator between runtimes
                                if language == "python":
                                    parts.append(f'\n# {"=" * 60}')
                                    parts.append(f'# Runtime: {runtime_type}')
                                    parts.append(f'# {"=" * 60}\n')
                                else:  # lua
                                    parts.append(f'\n-- {"=" * 60}')
                                    parts.append(f'-- Runtime: {runtime_type}')
                                    parts.append(f'-- {"=" * 60}\n')
                            parts.append(code)
                            self._logger.debug(f"Generated runtime code for {runtime_type} ({language})")
                    
                    runtime_code = '\n'.join(parts)

                    if not runtime_code:
                        self._logger.debug(
                            f"No runtime code generated for {language}, skipping file"
                        )
                        continue

                    # Determine output filename
                    filename = "obf_runtime.py" if language == "python" else "obf_runtime.lua"
                    runtime_path = output_dir / filename

                    # Write runtime file
                    runtime_path.write_text(runtime_code, encoding="utf-8")
                    runtime_files.append(str(runtime_path))

                    self._logger.info(
                        f"Generated hybrid runtime file: {filename} "
                        f"({len(runtime_code)} chars, {len(runtimes)} modules)"
                    )

                except Exception as e:
                    self._logger.warning(
                        f"Failed to generate {language} runtime file: {e}"
                    )
                    # Don't fail the entire orchestration - continue with other languages

            # Update result metadata with runtime file information
            if runtime_files:
                if "runtime_files" not in result.metadata:
                    result.metadata["runtime_files"] = []
                result.metadata["runtime_files"].extend(runtime_files)
                result.metadata["runtime_file_count"] = len(runtime_files)

        except Exception as e:
            self._logger.warning(f"Failed to generate hybrid runtime files: {e}")
            # Don't fail the entire orchestration - log and continue

    def _write_output(self, output_path: Path, code: str) -> tuple[Path | None, str | None]:
        """DEPRECATED: Write obfuscated code to output file with conflict resolution.

        DEPRECATED: Replaced by ``OutputWriter.write_with_structure`` in
        ``process_files``. This method is retained for backward compatibility
        with direct callers.

        Args:
            output_path: Path to write to
            code: Obfuscated code content

        Returns:
            Tuple of (resolved_path, conflict_resolution) where:
            - resolved_path: The path actually written to (None if skipped)
            - conflict_resolution: How the conflict was resolved (None if no conflict)
        """
        self._logger.warning(
            "DEPRECATED: _write_output() is deprecated; use "
            "OutputWriter.write_with_structure() instead."
        )

        resolved_path = output_path
        conflict_resolution = None

        # Check for conflicts if file exists and strategy is not OVERWRITE
        if output_path.exists() and self._conflict_strategy != ConflictStrategy.OVERWRITE:
            resolved_path = self.resolve_conflict(output_path)

            if resolved_path is None:
                # SKIP strategy - don't write
                self._logger.info(f"Skipped writing {output_path.name} (file exists)")
                return None, "skipped"

            # If path changed, it was renamed
            if resolved_path != output_path:
                conflict_resolution = "renamed"
            else:
                conflict_resolution = "overwritten"
        elif output_path.exists() and self._conflict_strategy == ConflictStrategy.OVERWRITE:
            conflict_resolution = "overwritten"

        # Create parent directories and write file
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(code, encoding="utf-8")
        self._logger.debug(f"Wrote output to {resolved_path}")

        return resolved_path, conflict_resolution

    def _detect_language(self, file_path: Path) -> str:
        """Detect the language of a file based on extension.

        Args:
            file_path: Path to the file

        Returns:
            Language identifier ('python', 'lua', or 'unknown')
        """
        suffix = file_path.suffix.lower()
        if suffix in (".py", ".pyw"):
            return "python"
        elif suffix in (".lua", ".luau"):
            return "lua"
        else:
            return "unknown"


class ProcessPoolManager:
    """Lifecycle manager for multiprocessing worker pool operations."""

    def __init__(
        self,
        worker_count: int | None = None,
        max_workers: int | None = None,
        batch_size: int = DEFAULT_PARALLEL_BATCH_SIZE,
        memory_threshold_percent: int = DEFAULT_MEMORY_THRESHOLD_PERCENT,
        min_batch_size: int = DEFAULT_MIN_BATCH_SIZE,
        grace_period: float = DEFAULT_CANCELLATION_GRACE_PERIOD,
    ) -> None:
        cpu_count = multiprocessing.cpu_count()
        default_worker_count = max(1, min(cpu_count - 1, 8))
        configured_worker_count = max_workers if max_workers is not None else worker_count
        if configured_worker_count is not None and (
            not isinstance(configured_worker_count, int)
            or isinstance(configured_worker_count, bool)
        ):
            raise ValueError("worker_count/max_workers must be an integer or None")

        resolved_worker_count = (
            default_worker_count
            if configured_worker_count is None
            else int(configured_worker_count)
        )

        if not 1 <= resolved_worker_count <= 16:
            raise ValueError("worker_count/max_workers must be between 1 and 16")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError("batch_size must be an integer")
        if not 10 <= batch_size <= 200:
            raise ValueError("batch_size must be between 10 and 200")
        if not 50 <= memory_threshold_percent <= 95:
            raise ValueError("memory_threshold_percent must be between 50 and 95")
        if not 10 <= min_batch_size <= 50:
            raise ValueError("min_batch_size must be between 10 and 50")
        if min_batch_size > batch_size:
            raise ValueError("min_batch_size must be less than or equal to batch_size")
        if grace_period <= 0:
            raise ValueError("grace_period must be > 0")

        self.worker_count = resolved_worker_count
        self.batch_size = batch_size
        self.memory_threshold_percent = memory_threshold_percent
        self.min_batch_size = min_batch_size
        self.grace_period = grace_period
        self.pool: multiprocessing.pool.Pool | None = None
        self.cancellation_event: multiprocessing.Event | None = None
        self.logger = get_logger("obfuscator.core.orchestrator.pool")
        self.memory_monitor = MemoryMonitor(threshold_percent=memory_threshold_percent)
        self.current_batch_size = batch_size
        self.batch_size_adjustments: list[dict[str, Any]] = []
        self._memory_monitoring_enabled = self.memory_monitor.monitoring_available

    def __enter__(self) -> ProcessPoolManager:
        self.start_pool()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.shutdown_pool()

    def start_pool(self) -> None:
        """Start the worker process pool."""
        if self.pool is not None:
            return

        self.logger.info(
            "Initializing process pool with %d workers (batch_size=%d)",
            self.worker_count,
            self.batch_size,
        )

        try:
            self.cancellation_event = multiprocessing.Event()
            self.pool = multiprocessing.Pool(
                processes=self.worker_count,
                initializer=_init_worker,
                initargs=(self.cancellation_event,),
            )
        except Exception:
            self.logger.error("Failed to initialize process pool", exc_info=True)
            self.pool = None
            self.cancellation_event = None
            raise

    def shutdown_pool(
        self,
        timeout: float = 10.0,
        grace_period: float | None = None,
    ) -> None:
        """Shutdown the worker pool, terminating hung workers after timeout."""
        if self.pool is None:
            self.cancellation_event = None
            return

        self.logger.info("Shutting down process pool")
        pool = self.pool
        resolved_grace_period = (
            self.grace_period if grace_period is None else max(0.1, grace_period)
        )

        try:
            pool.close()

            grace_deadline = time.time() + resolved_grace_period
            worker_processes = getattr(pool, "_pool", [])
            while worker_processes and any(proc.is_alive() for proc in worker_processes):
                if time.time() >= grace_deadline:
                    self.logger.warning(
                        (
                            "Graceful shutdown exceeded %.1fs; "
                            "transitioning to forced termination"
                        ),
                        resolved_grace_period,
                    )
                    pool.terminate()
                    break
                time.sleep(0.1)

            if timeout > 0:
                join_deadline = time.time() + timeout
                while worker_processes and any(proc.is_alive() for proc in worker_processes):
                    if time.time() >= join_deadline:
                        self.logger.warning(
                            "Pool join exceeded %.1fs after shutdown; forcing termination",
                            timeout,
                        )
                        pool.terminate()
                        break
                    time.sleep(0.1)

            pool.join()
        except Exception:
            self.logger.warning("Process pool shutdown encountered an error", exc_info=True)
            try:
                pool.terminate()
                pool.join()
            except Exception:
                self.logger.warning("Forced process pool termination failed", exc_info=True)
        finally:
            self.pool = None
            self.cancellation_event = None

    def terminate_pool(self) -> None:
        """Immediately terminate the worker pool."""
        if self.pool is None:
            self.cancellation_event = None
            return

        self.logger.warning("Terminating process pool")
        pool = self.pool
        try:
            pool.terminate()
            pool.join()
        except Exception:
            self.logger.warning("Forced process pool termination failed", exc_info=True)
        finally:
            self.pool = None
            self.cancellation_event = None

    def signal_cancellation(self) -> None:
        """Propagate cancellation to active worker processes."""
        if self.cancellation_event is None:
            return
        if not self.cancellation_event.is_set():
            self.cancellation_event.set()
            self.logger.info("Propagated cancellation signal to worker processes")

    def _check_and_adjust_batch_size(self, total_files: int) -> None:
        """Adjust batch size under memory pressure without changing file ordering."""
        if total_files < self.min_batch_size:
            self.logger.info(
                "Total files (%d) below minimum batch size (%d); skipping adaptive batching",
                total_files,
                self.min_batch_size,
            )
            return

        if total_files < self.current_batch_size:
            self.logger.info(
                "Total files (%d) below current batch size (%d); no adjustment needed",
                total_files,
                self.current_batch_size,
            )
            return

        if not self._memory_monitoring_enabled:
            self.current_batch_size = self.batch_size
            return

        if not self.memory_monitor.monitoring_available:
            self._memory_monitoring_enabled = False
            self.current_batch_size = self.batch_size
            self.logger.warning(
                "Memory monitor unavailable; disabling adaptive batching and using original batch size (%d)",
                self.batch_size,
            )
            return

        pressure_detected = self.memory_monitor.is_memory_pressure()
        memory_info = self.memory_monitor.get_memory_info()
        memory_percent = float(memory_info.get("percent", 0.0))

        if not self.memory_monitor.monitoring_available:
            self._memory_monitoring_enabled = False
            self.current_batch_size = self.batch_size
            self.logger.warning(
                "Memory monitor failed; disabling adaptive batching for this session and using original batch size (%d)",
                self.batch_size,
            )
            return

        # Batch size adjustment does not affect topological ordering.
        while pressure_detected and self.current_batch_size > self.min_batch_size:
            old_size = self.current_batch_size
            new_size = max(self.min_batch_size, old_size // 2)
            self.current_batch_size = new_size
            self.batch_size_adjustments.append(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "memory_percent": memory_percent,
                    "old_size": old_size,
                    "new_size": new_size,
                }
            )
            self.logger.warning(
                "Memory pressure (%.1f%%) detected - reducing batch size from %d to %d files",
                memory_percent,
                old_size,
                new_size,
            )

            pressure_detected = self.memory_monitor.is_memory_pressure()
            memory_info = self.memory_monitor.get_memory_info()
            memory_percent = float(memory_info.get("percent", 0.0))

            if not self.memory_monitor.monitoring_available:
                self._memory_monitoring_enabled = False
                self.current_batch_size = self.batch_size
                self.logger.warning(
                    "Memory monitor failed; disabling adaptive batching for this session and using original batch size (%d)",
                    self.batch_size,
                )
                return

            if not pressure_detected:
                self.logger.info(
                    "Memory pressure resolved at %.1f%% after adjusting batch size to %d files",
                    memory_percent,
                    self.current_batch_size,
                )
                return

        if pressure_detected and self.current_batch_size <= self.min_batch_size:
            self.logger.warning(
                (
                    "Batch size already at minimum (%d) - cannot reduce further "
                    "despite memory pressure (%.1f%%)"
                ),
                self.min_batch_size,
                memory_percent,
            )
            return

        self.logger.info(
            "Memory usage (%.1f%%) within safe limits - maintaining batch size of %d files",
            memory_percent,
            self.current_batch_size,
        )

    def create_batches(
        self,
        file_paths: list[Path],
        requested_batch_size: int,
    ) -> list[list[Path]]:
        """Split file paths into topologically ordered batches.

        Preserves topological ordering from input list by sequential slicing.
        """
        if not 10 <= requested_batch_size <= 200:
            raise ValueError("batch_size must be between 10 and 200")

        if not file_paths:
            return []

        self.current_batch_size = requested_batch_size
        self._check_and_adjust_batch_size(len(file_paths))

        memory_info = self.memory_monitor.get_memory_info()
        memory_percent = float(memory_info.get("percent", 0.0))

        batches: list[list[Path]] = []
        for index in range(0, len(file_paths), self.current_batch_size):
            batches.append(file_paths[index:index + self.current_batch_size])

        self.logger.info(
            (
                "Creating %d batches with size %d "
                "(memory: %.1f%%, %d adjustments made)"
            ),
            len(batches),
            self.current_batch_size,
            memory_percent,
            len(self.batch_size_adjustments),
        )
        return batches

    def submit_batch(self, task: WorkerTask) -> AsyncResult:
        """Submit a batch task to the process pool."""
        if self.pool is None:
            raise RuntimeError("Process pool is not initialized")

        return self.pool.apply_async(process_file_batch, args=(task,))

    def collect_results(
        self,
        async_results: list[AsyncResult],
        timeout: float | None = None,
    ) -> list[WorkerResult]:
        """Collect and flatten worker results from async handles."""
        flattened_results: list[WorkerResult] = []

        for async_result in async_results:
            cancellation_active = bool(
                self.cancellation_event is not None and self.cancellation_event.is_set()
            )
            effective_timeout = timeout
            if cancellation_active and (effective_timeout is None or effective_timeout > 1.0):
                effective_timeout = 1.0

            if cancellation_active and not async_result.ready():
                self.logger.info(
                    "Cancellation active; skipping wait for unfinished worker result"
                )
                continue

            try:
                raw_results = (
                    async_result.get(timeout=effective_timeout)
                    if effective_timeout is not None
                    else async_result.get()
                )
            except multiprocessing.TimeoutError as timeout_error:
                if cancellation_active:
                    self.logger.info(
                        "Timed out collecting worker result during cancellation; "
                        "returning partial results"
                    )
                    continue
                self.logger.error(
                    "Timed out collecting worker batch result",
                    exc_info=True,
                )
                raise TimeoutError("Timed out collecting worker batch result") from timeout_error
            except Exception:
                self.logger.error("Failed collecting worker batch result", exc_info=True)
                continue

            if not isinstance(raw_results, list):
                self.logger.warning(
                    "Unexpected worker payload type during collection: %s",
                    type(raw_results).__name__,
                )
                continue

            for entry in raw_results:
                if isinstance(entry, WorkerResult):
                    flattened_results.append(entry)
                elif isinstance(entry, dict):
                    try:
                        flattened_results.append(WorkerResult.from_dict(entry))
                    except Exception:
                        self.logger.warning(
                            "Invalid worker result dictionary payload encountered",
                            exc_info=True,
                        )
                else:
                    self.logger.warning(
                        "Ignoring unexpected worker result entry type: %s",
                        type(entry).__name__,
                    )

        return flattened_results

