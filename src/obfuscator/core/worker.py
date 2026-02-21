"""Multiprocessing worker utilities for batched file obfuscation.

This module implements the worker side of a parallel file-processing
architecture. Workers are process-isolated, consume serialized task payloads,
rebuild read-only state, and return structured per-file results.

Architecture
------------
Data flow follows a strict pipeline:

``WorkerTask -> process_file_batch() -> WorkerProcess -> list[WorkerResult]``

Each worker:
1. Deserializes shared state (global symbol table, dependency graph, config).
2. Parses files just-in-time.
3. Applies obfuscation transformations with a fresh engine per file.
4. Generates code and writes output atomically.
5. Releases AST references immediately after each file.

Serialization requirements
--------------------------
Shared structures are transferred as explicit dictionaries (``to_dict()`` /
``from_dict()`` style payloads). This avoids opaque pickle coupling and keeps
IPC payloads inspectable.

Memory management
-----------------
Workers do not retain AST caches. Parse, transform, and generation data are
processed per file and dereferenced immediately to bound memory in large jobs.

Thread-safety and process-safety
--------------------------------
Workers are independent processes with no shared mutable state. Logging uses
process-safe handlers and supports optional queue forwarding for centralized
aggregation.

Testing strategy
----------------
- Unit-test ``WorkerTask`` / ``WorkerResult`` serialization round-trips.
- Unit-test ``WorkerProcess`` with mocked parse/transform/write failures.
- Integration-test ``process_file_batch`` with mixed-success batches.
- Validate structured error typing for parse, transform, generation, and write
  failures.

Timeout note
------------
Timeout enforcement should be performed by the pool manager layer. Workers are
kept deterministic and single-responsibility.

Example
-------
>>> task = WorkerTask(
...     task_id="batch-1",
...     file_paths=["/path/to/file1.py", "/path/to/file2.py"],
...     global_table_dict=global_table.to_dict(),
...     dependency_graph_dict=graph.to_dict(),
...     config_dict=config.to_dict(),
...     output_dir="/path/to/output",
...     project_root="/path/to/project",
...     runtime_mode="hybrid",
...     conflict_strategy="rename",
... )
>>> results = process_file_batch(task)
>>> for result in results:
...     if result.success:
...         print(f"Processed {result.file_path} -> {result.output_path}")
...     else:
...         print(f"Failed {result.file_path}: {result.errors}")
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import re
import time
from dataclasses import dataclass, field
from logging.handlers import QueueHandler
from multiprocessing.queues import Queue as MultiprocessingQueue
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency guard
    psutil = None  # type: ignore[assignment]

from obfuscator.core.config import ObfuscationConfig
from obfuscator.core.dependency_graph import DependencyGraph
from obfuscator.core.exceptions import UnsupportedFeatureWarning
from obfuscator.core.symbol_table import GlobalSymbolTable, SymbolEntry
from obfuscator.utils.error_formatting import extract_line_column, parse_error

_VALID_RUNTIME_MODES: set[str] = {"hybrid", "embedded"}
_VALID_CONFLICT_STRATEGIES: set[str] = {"overwrite", "skip", "rename", "ask"}
_WORKER_CANCELLATION_EVENT: multiprocessing.Event | None = None


def set_cancellation_event(event: multiprocessing.Event | None) -> None:
    """Set or clear the worker-global cancellation event reference."""
    global _WORKER_CANCELLATION_EVENT
    _WORKER_CANCELLATION_EVENT = event


@dataclass
class WorkerTask:
    """Serializable task payload for batch file processing.

    Attributes:
        task_id: Unique batch identifier for tracking.
        file_paths: Source file paths as strings (IPC-safe).
        global_table_dict: Serialized ``GlobalSymbolTable``.
        dependency_graph_dict: Serialized ``DependencyGraph``.
        config_dict: Serialized ``ObfuscationConfig``.
        output_dir: Base output directory.
        project_root: Optional project root for relative output paths.
        runtime_mode: ``"hybrid"`` or ``"embedded"``.
        conflict_strategy: Output conflict behavior.
    """

    task_id: str
    file_paths: list[str]
    global_table_dict: dict[str, Any]
    dependency_graph_dict: dict[str, Any]
    config_dict: dict[str, Any]
    output_dir: str
    project_root: str | None
    runtime_mode: str
    conflict_strategy: str

    def __post_init__(self) -> None:
        """Validate task payload invariants."""
        if not isinstance(self.file_paths, list) or not self.file_paths:
            raise ValueError("WorkerTask.file_paths must be a non-empty list")

        if any(not isinstance(path, str) or not path for path in self.file_paths):
            raise ValueError("WorkerTask.file_paths must contain non-empty strings")

        required_global_keys = {"is_frozen", "symbols"}
        missing_global = required_global_keys.difference(self.global_table_dict.keys())
        if missing_global:
            raise ValueError(
                "WorkerTask.global_table_dict missing required keys: "
                f"{sorted(missing_global)}"
            )

        required_graph_keys = {"nodes", "edges"}
        missing_graph = required_graph_keys.difference(self.dependency_graph_dict.keys())
        if missing_graph:
            raise ValueError(
                "WorkerTask.dependency_graph_dict missing required keys: "
                f"{sorted(missing_graph)}"
            )

        required_config_keys = {"version", "name", "features", "options"}
        missing_config = required_config_keys.difference(self.config_dict.keys())
        if missing_config:
            raise ValueError(
                f"WorkerTask.config_dict missing required keys: {sorted(missing_config)}"
            )

        if self.runtime_mode not in _VALID_RUNTIME_MODES:
            raise ValueError(
                f"Invalid runtime_mode: {self.runtime_mode}. "
                f"Expected one of {_VALID_RUNTIME_MODES}"
            )

        if self.conflict_strategy not in _VALID_CONFLICT_STRATEGIES:
            raise ValueError(
                f"Invalid conflict_strategy: {self.conflict_strategy}. "
                f"Expected one of {_VALID_CONFLICT_STRATEGIES}"
            )

        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("WorkerTask.output_dir must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this task to a plain dictionary."""
        return {
            "task_id": self.task_id,
            "file_paths": list(self.file_paths),
            "global_table_dict": dict(self.global_table_dict),
            "dependency_graph_dict": dict(self.dependency_graph_dict),
            "config_dict": dict(self.config_dict),
            "output_dir": self.output_dir,
            "project_root": self.project_root,
            "runtime_mode": self.runtime_mode,
            "conflict_strategy": self.conflict_strategy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerTask:
        """Deserialize a task from dictionary data."""
        return cls(
            task_id=str(data["task_id"]),
            file_paths=[str(path) for path in data["file_paths"]],
            global_table_dict=dict(data["global_table_dict"]),
            dependency_graph_dict=dict(data["dependency_graph_dict"]),
            config_dict=dict(data["config_dict"]),
            output_dir=str(data["output_dir"]),
            project_root=(
                str(data["project_root"])
                if data.get("project_root") is not None
                else None
            ),
            runtime_mode=str(data["runtime_mode"]),
            conflict_strategy=str(data["conflict_strategy"]),
        )


@dataclass
class WorkerResult:
    """Serializable per-file worker result.

    Attributes:
        task_id: Task identifier matching :class:`WorkerTask.task_id`.
        file_path: Source file path.
        success: Whether file processing succeeded.
        output_path: Result output path when written.
        errors: Human-readable error list.
        warnings: Human-readable warning list.
        detailed_errors: Structured error dictionaries.
        conflict_resolution: Conflict strategy outcome, when relevant.
        transformation_count: Number of transformations applied.
        processing_time: Elapsed processing time in seconds.
        was_cancelled: Whether processing for this file was skipped due to
            cancellation of the active batch.
    """

    task_id: str
    file_path: str
    success: bool
    output_path: str | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    detailed_errors: list[dict[str, Any]] = field(default_factory=list)
    conflict_resolution: str | None = None
    transformation_count: int = 0
    processing_time: float = 0.0
    was_cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to a dictionary for transport/storage."""
        return {
            "task_id": self.task_id,
            "file_path": self.file_path,
            "success": self.success,
            "output_path": self.output_path,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "detailed_errors": list(self.detailed_errors),
            "conflict_resolution": self.conflict_resolution,
            "transformation_count": self.transformation_count,
            "processing_time": self.processing_time,
            "was_cancelled": self.was_cancelled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerResult:
        """Deserialize a worker result from dictionary data."""
        return cls(
            task_id=str(data["task_id"]),
            file_path=str(data["file_path"]),
            success=bool(data["success"]),
            output_path=(
                str(data["output_path"])
                if data.get("output_path") is not None
                else None
            ),
            errors=[str(error) for error in data.get("errors", [])],
            warnings=[str(warning) for warning in data.get("warnings", [])],
            detailed_errors=list(data.get("detailed_errors", [])),
            conflict_resolution=(
                str(data["conflict_resolution"])
                if data.get("conflict_resolution") is not None
                else None
            ),
            transformation_count=int(data.get("transformation_count", 0)),
            processing_time=float(data.get("processing_time", 0.0)),
            was_cancelled=bool(data.get("was_cancelled", False)),
        )


def setup_worker_logging(
    worker_id: int,
    log_queue: MultiprocessingQueue | None = None,
) -> logging.Logger:
    """Set up process-safe logging for a worker.

    Args:
        worker_id: Numeric worker identifier.
        log_queue: Optional queue for centralized log aggregation.

    Returns:
        Configured logger with worker-specific context.
    """
    logger_name = f"obfuscator.core.worker.worker-{worker_id}"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    base_logger = multiprocessing.get_logger()
    level_name = os.getenv("SCRIPTSHIELD_WORKER_LOG_LEVEL", "").upper()
    if level_name and hasattr(logging, level_name):
        level = getattr(logging, level_name)
    elif base_logger.level not in (0, logging.NOTSET):
        level = base_logger.level
    else:
        level = logging.INFO

    handler: logging.Handler
    if log_queue is not None:
        handler = QueueHandler(log_queue)
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(
        logging.Formatter(
            fmt=f"%(asctime)s [worker-{worker_id}] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _detect_language(file_path: Path) -> str:
    """Detect file language by extension.

    Returns:
        ``"python"`` for ``.py`` / ``.pyw`` files.
        ``"lua"`` for ``.lua`` / ``.luau`` files.

    Raises:
        ValueError: For unsupported file extensions.
    """
    suffix = file_path.suffix.lower()
    if suffix in {".py", ".pyw"}:
        return "python"
    if suffix in {".lua", ".luau"}:
        return "lua"
    raise ValueError(f"Unsupported file extension: {suffix} ({file_path})")


def _extract_error_location(error_message: str) -> tuple[int | None, int | None]:
    """Extract line/column location from an error string."""
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
    file_path: Path,
    error_type: str,
    message: str,
    line: int | None = None,
    column: int | None = None,
) -> dict[str, Any]:
    """Create a standardized detailed error dictionary."""
    resolved_line = line
    resolved_column = column
    if resolved_line is None and resolved_column is None:
        resolved_line, resolved_column = _extract_error_location(message)

    return {
        "file_path": str(file_path),
        "line": resolved_line,
        "column": resolved_column,
        "error_type": error_type,
        "message": str(message),
    }


def _extract_worker_id(process_name: str) -> int:
    """Extract trailing numeric id from multiprocessing process name."""
    match = re.search(r"(\d+)$", process_name)
    if match:
        return int(match.group(1))
    return os.getpid()


class MemoryMonitor:
    """Lightweight memory monitor used to detect memory pressure."""

    def __init__(self, threshold_percent: int = 80) -> None:
        if not 1 <= threshold_percent <= 100:
            raise ValueError("threshold_percent must be between 1 and 100")

        self.threshold_percent = threshold_percent
        self.logger = logging.getLogger("obfuscator.core.worker.memory_monitor")
        self._monitoring_available = psutil is not None

        if not self._monitoring_available:
            self.logger.error(
                "psutil is not available; memory monitoring will use safe defaults"
            )

    @property
    def monitoring_available(self) -> bool:
        """Whether psutil-backed memory monitoring is available."""
        return self._monitoring_available

    def get_memory_usage(self) -> float:
        """Return current system memory usage percentage."""
        if not self._monitoring_available or psutil is None:
            return 0.0

        try:
            usage_percent = float(psutil.virtual_memory().percent)
            self.logger.debug(
                "Memory check: %.1f%% used (threshold: %d%%)",
                usage_percent,
                self.threshold_percent,
            )
            return usage_percent
        except psutil.Error as exc:
            self._monitoring_available = False
            self.logger.error("Failed to read memory usage: %s", exc, exc_info=True)
            return 0.0

    def is_memory_pressure(self) -> bool:
        """Return ``True`` when memory usage exceeds configured threshold."""
        usage_percent = self.get_memory_usage()
        pressure_detected = usage_percent > float(self.threshold_percent)
        if pressure_detected:
            self.logger.warning(
                "Memory pressure detected: %.1f%% exceeds %d%%",
                usage_percent,
                self.threshold_percent,
            )
        return pressure_detected

    def get_memory_info(self) -> dict[str, int | float]:
        """Return current memory snapshot for diagnostics and logging."""
        if not self._monitoring_available or psutil is None:
            return {
                "total": 0,
                "available": 0,
                "used": 0,
                "percent": 0.0,
            }

        try:
            memory = psutil.virtual_memory()
            memory_info: dict[str, int | float] = {
                "total": int(memory.total),
                "available": int(memory.available),
                "used": int(memory.used),
                "percent": float(memory.percent),
            }
            self.logger.debug(
                "Memory check: %.1f%% used (threshold: %d%%)",
                memory_info["percent"],
                self.threshold_percent,
            )
            return memory_info
        except psutil.Error as exc:
            self._monitoring_available = False
            self.logger.error("Failed to read memory info: %s", exc, exc_info=True)
            return {
                "total": 0,
                "available": 0,
                "used": 0,
                "percent": 0.0,
            }


class WorkerProcess:
    """Worker runtime that processes file batches independently.

    Initialization:
    - Deserializes and validates ``GlobalSymbolTable`` payload (must be frozen).
    - Deserializes ``DependencyGraph`` payload (must include nodes).
    - Builds and validates ``ObfuscationConfig``.
    - Creates dedicated processors and ``OutputWriter``.
    """

    def __init__(
        self,
        worker_id: int,
        global_table_dict: dict[str, Any],
        dependency_graph_dict: dict[str, Any],
        config_dict: dict[str, Any],
    ) -> None:
        """Initialize worker-local processing state."""
        self.worker_id = worker_id
        self.task_id = ""
        self.logger = setup_worker_logging(worker_id)

        self.global_table = self._deserialize_global_symbol_table(global_table_dict)
        if not self.global_table.is_frozen:
            raise ValueError("GlobalSymbolTable must be frozen after deserialization")

        self.dependency_graph = self._deserialize_dependency_graph(
            dependency_graph_dict
        )
        if not self.dependency_graph.nodes:
            raise ValueError("DependencyGraph must contain at least one node")

        self.config = ObfuscationConfig.from_dict(config_dict)
        self.config.validate()

        # Imported lazily so WorkerTask/WorkerResult can be imported without
        # requiring the full processor stack.
        from obfuscator.processors.lua_processor import LuaProcessor
        from obfuscator.processors.python_processor import PythonProcessor

        self.python_processor = PythonProcessor(config=self.config)
        self.lua_processor = LuaProcessor(config=self.config)

        from obfuscator.core.plugin_manager import PluginManager
        self.plugin_manager = PluginManager()

        # Imported lazily to avoid module import cycles when orchestrator imports
        # worker utilities at module load time.
        from obfuscator.core.orchestrator import ConflictStrategy
        from obfuscator.core.output_writer import OutputWriter

        try:
            conflict_strategy = ConflictStrategy(self.config.conflict_strategy)
        except ValueError as exc:
            raise ValueError(
                f"Invalid conflict strategy: {self.config.conflict_strategy}"
            ) from exc

        self.output_writer = OutputWriter(
            output_dir=Path.cwd(),
            conflict_strategy=conflict_strategy,
            use_atomic_writes=True,
        )

        self.logger.info(
            "Worker initialized with %d symbols, %d graph nodes, %d graph edges",
            len(self.global_table.get_all_symbols()),
            len(self.dependency_graph.nodes),
            len(self.dependency_graph.edges),
        )

    def _deserialize_global_symbol_table(
        self,
        table_dict: dict[str, Any],
    ) -> GlobalSymbolTable:
        """Deserialize a frozen ``GlobalSymbolTable`` from dictionary data."""
        if not isinstance(table_dict, dict):
            raise ValueError("global_table_dict must be a dictionary")

        if not table_dict.get("is_frozen", False):
            raise ValueError(
                "Serialized GlobalSymbolTable is not frozen; worker requires "
                "an immutable symbol table"
            )

        symbols = table_dict.get("symbols", [])
        if not isinstance(symbols, list):
            raise ValueError("global_table_dict['symbols'] must be a list")

        global_table = GlobalSymbolTable(table_dict.get("config", {}))

        for symbol in symbols:
            if not isinstance(symbol, dict):
                continue

            entry = SymbolEntry(
                original_name=str(symbol.get("original_name", "")),
                mangled_name=str(symbol.get("mangled_name", "")),
                scope=str(symbol.get("scope", "global")),
                language=str(symbol.get("language", "python")),
                file_path=Path(str(symbol.get("file_path", ""))).resolve(),
                line_number=int(symbol.get("line_number", 0) or 0),
                symbol_type=str(symbol.get("symbol_type", "variable")),
                is_exported=bool(symbol.get("is_exported", False)),
            )
            global_table.add_symbol(entry)

        global_table.freeze()
        return global_table

    def _deserialize_dependency_graph(
        self,
        graph_dict: dict[str, Any],
    ) -> DependencyGraph:
        """Deserialize ``DependencyGraph`` from dictionary payload."""
        if not isinstance(graph_dict, dict):
            raise ValueError("dependency_graph_dict must be a dictionary")

        graph = DependencyGraph()

        nodes_data = graph_dict.get("nodes", {})
        if not isinstance(nodes_data, dict):
            raise ValueError("dependency_graph_dict['nodes'] must be a dictionary")

        for raw_path, node_data in nodes_data.items():
            if not isinstance(node_data, dict):
                continue

            imports = node_data.get("imports", [])
            exports = node_data.get("exports", [])
            metadata = node_data.get("metadata", {})

            node = graph.add_node(
                file_path=Path(str(raw_path)),
                language=str(node_data.get("language", "python")),
                imports=list(imports) if isinstance(imports, list) else [],
                exports=list(exports) if isinstance(exports, list) else [],
                metadata=dict(metadata) if isinstance(metadata, dict) else {},
            )
            node.is_processed = bool(node_data.get("is_processed", True))

        edges_data = graph_dict.get("edges", [])
        if isinstance(edges_data, list):
            for edge_data in edges_data:
                if not isinstance(edge_data, dict):
                    continue
                try:
                    graph.add_edge(
                        from_path=Path(str(edge_data.get("from", ""))),
                        to_path=Path(str(edge_data.get("to", ""))),
                        import_type=str(edge_data.get("import_type", "absolute")),
                        symbols=list(edge_data.get("imported_symbols", [])),
                        line_num=int(edge_data.get("line_number", 0) or 0),
                    )
                except ValueError as exc:
                    self.logger.warning(
                        "Skipping invalid dependency edge during deserialization: %s",
                        exc,
                    )

        return graph

    def _format_feature_warning(self, warning: Any, fallback_file: Path) -> str:
        """Format parse warnings into canonical warning strings."""
        if isinstance(warning, UnsupportedFeatureWarning):
            return str(warning)

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
        return str(structured_warning)

    def _extract_transformation_count(self, transform_result: dict[str, Any]) -> int:
        """Best-effort extraction of transformation count from result payload."""
        count_value = transform_result.get("transformation_count")
        if isinstance(count_value, int):
            return count_value

        stats = transform_result.get("stats")
        if isinstance(stats, dict):
            stats_count = stats.get("transformation_count")
            if isinstance(stats_count, int):
                return stats_count

        return 0

    def _is_cancellation_requested(self) -> bool:
        """Check whether pool-level cancellation has been requested."""
        event = _WORKER_CANCELLATION_EVENT
        return bool(event is not None and event.is_set())

    def _inject_embedded_runtime(
        self,
        code: str,
        language: str,
        engine: ObfuscationEngine,
    ) -> str:
        """Inject embedded runtime preamble into generated output code."""
        if self.config.runtime_mode != "embedded":
            return code

        if not engine.has_runtime_requirements():
            return code

        runtime_code = engine.get_required_runtime_code(language)
        if not runtime_code:
            return code

        if language == "python":
            separator = "# " + "=" * 70
            header = (
                f"{separator}\n"
                "# EMBEDDED OBFUSCATION RUNTIME\n"
                f"{separator}\n\n"
            )
            footer = (
                f"\n\n{separator}\n"
                "# END RUNTIME CODE\n"
                f"{separator}\n\n"
            )
        elif language == "lua":
            separator = "-- " + "=" * 70
            header = (
                f"{separator}\n"
                "-- EMBEDDED OBFUSCATION RUNTIME\n"
                f"{separator}\n\n"
            )
            footer = (
                f"\n\n{separator}\n"
                "-- END RUNTIME CODE\n"
                f"{separator}\n\n"
            )
        else:
            return code

        self.logger.info(
            "Injected embedded runtime for %s (%d chars)",
            language,
            len(runtime_code),
        )
        return header + runtime_code + footer + code

    def process_file(
        self,
        file_path: Path,
        output_dir: Path,
        project_root: Path | None,
    ) -> WorkerResult:
        """Process a single file and return a structured worker result.

        This method catches all exceptions to prevent worker process crashes and
        converts failures into ``WorkerResult(success=False)``.
        """
        start_time = time.perf_counter()
        warnings: list[str] = []
        detailed_errors: list[dict[str, Any]] = []

        parse_result: Any | None = None
        transform_result: dict[str, Any] | None = None
        ast_node: Any | None = None
        transformed_ast: Any | None = None

        try:
            language = _detect_language(file_path)
            self.logger.info("Processing file: %s (%s)", file_path, language)

            # Imported lazily to avoid importing transformation dependencies when
            # worker utilities are imported only for task/result dataclasses.
            from obfuscator.core.obfuscation_engine import ObfuscationEngine

            if language == "python":
                parse_result = self.python_processor.parse_file(file_path)
            else:
                parse_result = self.lua_processor.parse_file(file_path)

            for warning in parse_result.warnings:
                warning_text = self._format_feature_warning(warning, file_path)
                warnings.append(warning_text)
                self.logger.warning(warning_text)

            if not parse_result.success or parse_result.ast_node is None:
                parse_errors = parse_result.errors or ["Failed to parse file"]
                for parse_error in parse_errors:
                    line, column = _extract_error_location(str(parse_error))
                    detailed_errors.append(
                        _create_detailed_error(
                            file_path=file_path,
                            error_type="ParseError",
                            message=str(parse_error),
                            line=line,
                            column=column,
                        )
                    )
                return WorkerResult(
                    task_id=self.task_id,
                    file_path=str(file_path),
                    success=False,
                    output_path=None,
                    errors=[str(error) for error in parse_errors],
                    warnings=warnings,
                    detailed_errors=detailed_errors,
                    transformation_count=0,
                    processing_time=time.perf_counter() - start_time,
                )

            ast_node = parse_result.ast_node
            engine = ObfuscationEngine(self.config, plugin_manager=self.plugin_manager)
            transformation_count = 0

            if language == "python":
                transform_result = self.python_processor.obfuscate_with_symbol_table(
                    ast_node,
                    file_path,
                    self.global_table,
                    engine=engine,
                )
                transformation_count = self._extract_transformation_count(
                    transform_result
                )

                if not transform_result.get("success", False):
                    transform_errors = transform_result.get(
                        "errors",
                        ["Transformation failed"],
                    )
                    for error in transform_errors:
                        line, column = _extract_error_location(str(error))
                        detailed_errors.append(
                            _create_detailed_error(
                                file_path=file_path,
                                error_type="TransformationError",
                                message=str(error),
                                line=line,
                                column=column,
                            )
                        )
                    return WorkerResult(
                        task_id=self.task_id,
                        file_path=str(file_path),
                        success=False,
                        output_path=None,
                        errors=[str(error) for error in transform_errors],
                        warnings=warnings,
                        detailed_errors=detailed_errors,
                        transformation_count=transformation_count,
                        processing_time=time.perf_counter() - start_time,
                    )

                transformed_ast = transform_result.get("ast_node")
                if transformed_ast is None:
                    transform_error = "Transformation produced no AST node"
                    detailed_errors.append(
                        _create_detailed_error(
                            file_path=file_path,
                            error_type="TransformationError",
                            message=transform_error,
                        )
                    )
                    return WorkerResult(
                        task_id=self.task_id,
                        file_path=str(file_path),
                        success=False,
                        output_path=None,
                        errors=[transform_error],
                        warnings=warnings,
                        detailed_errors=detailed_errors,
                        transformation_count=transformation_count,
                        processing_time=time.perf_counter() - start_time,
                    )

                generate_result = self.python_processor.generate_code(transformed_ast)
                if not generate_result.success:
                    generation_errors = generate_result.errors or ["Code generation failed"]
                    for error in generation_errors:
                        line, column = _extract_error_location(str(error))
                        detailed_errors.append(
                            _create_detailed_error(
                                file_path=file_path,
                                error_type="CodeGenerationError",
                                message=str(error),
                                line=line,
                                column=column,
                            )
                        )
                    return WorkerResult(
                        task_id=self.task_id,
                        file_path=str(file_path),
                        success=False,
                        output_path=None,
                        errors=[str(error) for error in generation_errors],
                        warnings=warnings,
                        detailed_errors=detailed_errors,
                        transformation_count=transformation_count,
                        processing_time=time.perf_counter() - start_time,
                    )

                generated_code = generate_result.code
            else:
                transform_result = self.lua_processor.obfuscate_with_symbol_table(
                    ast_node,
                    file_path,
                    self.global_table,
                    engine=engine,
                )
                transformation_count = self._extract_transformation_count(
                    transform_result
                )

                if not transform_result.get("success", False):
                    transform_errors = transform_result.get(
                        "errors",
                        ["Transformation failed"],
                    )
                    for error in transform_errors:
                        line, column = _extract_error_location(str(error))
                        detailed_errors.append(
                            _create_detailed_error(
                                file_path=file_path,
                                error_type="TransformationError",
                                message=str(error),
                                line=line,
                                column=column,
                            )
                        )
                    return WorkerResult(
                        task_id=self.task_id,
                        file_path=str(file_path),
                        success=False,
                        output_path=None,
                        errors=[str(error) for error in transform_errors],
                        warnings=warnings,
                        detailed_errors=detailed_errors,
                        transformation_count=transformation_count,
                        processing_time=time.perf_counter() - start_time,
                    )

                generated_code = str(transform_result.get("code", ""))
                if not generated_code:
                    transformed_ast = transform_result.get("ast_node", ast_node)
                    generate_result = self.lua_processor.generate_code(
                        transformed_ast,
                        prepend_runtime="",
                    )
                    if not generate_result.success:
                        generation_errors = generate_result.errors or [
                            "Code generation failed"
                        ]
                        for error in generation_errors:
                            line, column = _extract_error_location(str(error))
                            detailed_errors.append(
                                _create_detailed_error(
                                    file_path=file_path,
                                    error_type="CodeGenerationError",
                                    message=str(error),
                                    line=line,
                                    column=column,
                                )
                            )
                        return WorkerResult(
                            task_id=self.task_id,
                            file_path=str(file_path),
                            success=False,
                            output_path=None,
                            errors=[str(error) for error in generation_errors],
                            warnings=warnings,
                            detailed_errors=detailed_errors,
                            transformation_count=transformation_count,
                            processing_time=time.perf_counter() - start_time,
                        )
                    generated_code = generate_result.code

            if self.config.runtime_mode == "embedded" and engine.has_runtime_requirements():
                generated_code = self._inject_embedded_runtime(
                    generated_code,
                    language,
                    engine,
                )

            write_result = self.output_writer.write_with_structure(
                input_path=file_path,
                output_base=output_dir,
                content=generated_code,
                project_root=project_root,
            )

            if not write_result.success:
                write_error = write_result.error or "Failed to write output"
                detailed_errors.append(
                    _create_detailed_error(
                        file_path=file_path,
                        error_type="WriteError",
                        message=write_error,
                    )
                )
                return WorkerResult(
                    task_id=self.task_id,
                    file_path=str(file_path),
                    success=False,
                    output_path=None,
                    errors=[write_error],
                    warnings=warnings,
                    detailed_errors=detailed_errors,
                    conflict_resolution=write_result.conflict_resolution,
                    transformation_count=transformation_count,
                    processing_time=time.perf_counter() - start_time,
                )

            if write_result.output_path is None:
                warnings.append(
                    f"Skipped {file_path.name} - file exists at output path"
                )

            return WorkerResult(
                task_id=self.task_id,
                file_path=str(file_path),
                success=True,
                output_path=(
                    str(write_result.output_path)
                    if write_result.output_path is not None
                    else None
                ),
                errors=[],
                warnings=warnings,
                detailed_errors=[],
                conflict_resolution=write_result.conflict_resolution,
                transformation_count=transformation_count,
                processing_time=time.perf_counter() - start_time,
            )

        except Exception as exc:  # pragma: no cover - resilience path
            error_message = f"{type(exc).__name__}: {exc}"
            self.logger.error(
                "Unexpected worker error while processing %s: %s",
                file_path,
                error_message,
                exc_info=True,
            )
            detailed_errors = [
                _create_detailed_error(
                    file_path=file_path,
                    error_type="UnexpectedException",
                    message=error_message,
                )
            ]
            return WorkerResult(
                task_id=self.task_id,
                file_path=str(file_path),
                success=False,
                output_path=None,
                errors=[error_message],
                warnings=warnings,
                detailed_errors=detailed_errors,
                transformation_count=0,
                processing_time=time.perf_counter() - start_time,
            )

        finally:
            # Explicitly drop AST references to encourage timely memory release.
            parse_result = None
            transform_result = None
            ast_node = None
            transformed_ast = None

    def process_batch(
        self,
        file_paths: list[Path],
        output_dir: Path,
        project_root: Path | None,
    ) -> list[WorkerResult]:
        """Process a batch of files and return per-file results."""
        self.logger.info(
            "Starting batch processing: %d file(s), output=%s",
            len(file_paths),
            output_dir,
        )

        results: list[WorkerResult] = []
        for index, file_path in enumerate(file_paths, start=1):
            if self._is_cancellation_requested():
                processed_count = len(results)
                skipped_paths = file_paths[processed_count:]
                self.logger.info(
                    "Cancellation detected in worker; stopping batch after %d/%d file(s)",
                    processed_count,
                    len(file_paths),
                )

                for skipped_path in skipped_paths:
                    results.append(
                        WorkerResult(
                            task_id=self.task_id,
                            file_path=str(skipped_path),
                            success=False,
                            output_path=None,
                            errors=[],
                            warnings=["Skipped due to cancellation request"],
                            detailed_errors=[],
                            transformation_count=0,
                            processing_time=0.0,
                            was_cancelled=True,
                        )
                    )
                break

            self.logger.info("Batch progress %d/%d: %s", index, len(file_paths), file_path)
            result = self.process_file(file_path, output_dir, project_root)
            results.append(result)

        success_count = sum(1 for result in results if result.success)
        cancelled_count = sum(1 for result in results if result.was_cancelled)
        self.logger.info(
            "Batch complete: %d/%d succeeded (%d cancelled/skipped)",
            success_count,
            len(results),
            cancelled_count,
        )
        return results


def process_file_batch(task: WorkerTask) -> list[WorkerResult]:
    """Worker entry point for multiprocessing execution.

    Args:
        task: :class:`WorkerTask` payload with serialized shared state and batch
            details.

    Returns:
        List of :class:`WorkerResult` objects for every input file. Initialization
        failures are converted to synthetic failure results instead of raising.
    """
    process_name = multiprocessing.current_process().name
    worker_id = _extract_worker_id(process_name)
    logger = setup_worker_logging(worker_id)

    normalized_task: WorkerTask | None = None

    try:
        if isinstance(task, WorkerTask):
            normalized_task = task
        elif isinstance(task, dict):
            normalized_task = WorkerTask.from_dict(task)
        else:
            raise TypeError(f"Unsupported task type: {type(task).__name__}")

        logger.info(
            "Worker startup: process=%s task_id=%s files=%d",
            process_name,
            normalized_task.task_id,
            len(normalized_task.file_paths),
        )

        # Ensure task-level runtime settings are authoritative.
        config_dict = dict(normalized_task.config_dict)
        config_dict["runtime_mode"] = normalized_task.runtime_mode
        config_dict["conflict_strategy"] = normalized_task.conflict_strategy

        worker = WorkerProcess(
            worker_id=worker_id,
            global_table_dict=normalized_task.global_table_dict,
            dependency_graph_dict=normalized_task.dependency_graph_dict,
            config_dict=config_dict,
        )
        worker.task_id = normalized_task.task_id

        file_paths = [Path(path) for path in normalized_task.file_paths]
        output_dir = Path(normalized_task.output_dir)
        project_root = (
            Path(normalized_task.project_root)
            if normalized_task.project_root is not None
            else None
        )

        results = worker.process_batch(file_paths, output_dir, project_root)

        success_count = sum(1 for result in results if result.success)
        logger.info(
            "Worker finished task_id=%s: %d/%d succeeded",
            normalized_task.task_id,
            success_count,
            len(results),
        )
        return results

    except Exception as exc:  # pragma: no cover - resilience path
        error_message = f"Worker initialization failed: {type(exc).__name__}: {exc}"
        logger.error(error_message, exc_info=True)

        task_id_for_fallback = "<unknown-task>"
        file_paths_for_fallback: list[str] = []

        if normalized_task is not None:
            task_id_for_fallback = normalized_task.task_id
            file_paths_for_fallback = list(normalized_task.file_paths)
        elif isinstance(task, dict):
            raw_task_id = task.get("task_id")
            if raw_task_id is not None:
                raw_task_id_str = str(raw_task_id).strip()
                if raw_task_id_str:
                    task_id_for_fallback = raw_task_id_str

            raw_file_paths = task.get("file_paths")
            if isinstance(raw_file_paths, list):
                for raw_file_path in raw_file_paths:
                    raw_file_path_str = str(raw_file_path).strip()
                    if raw_file_path_str:
                        file_paths_for_fallback.append(raw_file_path_str)

        fallback_results: list[WorkerResult] = []
        for file_path in file_paths_for_fallback:
            fallback_results.append(
                WorkerResult(
                    task_id=task_id_for_fallback,
                    file_path=str(file_path),
                    success=False,
                    output_path=None,
                    errors=[error_message],
                    warnings=[],
                    detailed_errors=[
                        _create_detailed_error(
                            file_path=Path(file_path),
                            error_type="UnexpectedException",
                            message=error_message,
                        )
                    ],
                    transformation_count=0,
                    processing_time=0.0,
                )
            )

        if not fallback_results:
            fallback_results.append(
                WorkerResult(
                    task_id=task_id_for_fallback,
                    file_path="<worker-initialization>",
                    success=False,
                    output_path=None,
                    errors=[error_message],
                    warnings=[],
                    detailed_errors=[
                        _create_detailed_error(
                            file_path=Path("<worker-initialization>"),
                            error_type="UnexpectedException",
                            message=error_message,
                        )
                    ],
                    transformation_count=0,
                    processing_time=0.0,
                )
            )

        return fallback_results

    finally:
        shutdown_task_id = "<unknown-task>"
        if normalized_task is not None:
            shutdown_task_id = normalized_task.task_id
        elif isinstance(task, dict):
            raw_task_id = task.get("task_id")
            if raw_task_id is not None:
                raw_task_id_str = str(raw_task_id).strip()
                if raw_task_id_str:
                    shutdown_task_id = raw_task_id_str

        logger.info(
            "Worker shutdown: process=%s task_id=%s",
            process_name,
            shutdown_task_id,
        )


__all__ = [
    "MemoryMonitor",
    "WorkerTask",
    "WorkerResult",
    "WorkerProcess",
    "process_file_batch",
    "set_cancellation_event",
    "setup_worker_logging",
]

