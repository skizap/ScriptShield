"""Obfuscation orchestrator module for coordinating multi-file processing.

This module provides the ObfuscationOrchestrator class that coordinates the
complete obfuscation workflow: scanning files, building dependency graphs,
pre-computing symbol tables, and processing files in topological order.

Example:
    >>> from pathlib import Path
    >>> from obfuscator.core.orchestrator import ObfuscationOrchestrator
    >>> 
    >>> orchestrator = ObfuscationOrchestrator()
    >>> result = orchestrator.process_files(
    ...     input_files=[Path("main.py"), Path("utils.py")],
    ...     output_dir=Path("./obfuscated"),
    ...     config={"identifier_prefix": "_0x", "mangling_strategy": "sequential"}
    ... )
    >>> if result.success:
    ...     print(f"Processed {len(result.processed_files)} files")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

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
from obfuscator.processors.python_processor import PythonProcessor
from obfuscator.processors.lua_processor import LuaProcessor
from obfuscator.processors.symbol_extractor import SymbolTable
from obfuscator.processors.lua_symbol_extractor import LuaSymbolTable
from obfuscator.core.config import ObfuscationConfig
from obfuscator.core.runtime_manager import RuntimeManager
from obfuscator.utils.logger import get_logger
from obfuscator.utils.path_utils import is_readable, is_writable

logger = get_logger("obfuscator.core.orchestrator")


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
        warnings: List of warning messages
    """
    file_path: Path
    output_path: Path | None
    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


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
        warnings: List of global warning messages
        metadata: Additional metadata about the process
    """
    success: bool
    current_state: JobState = field(default=JobState.PENDING)
    processed_files: list[ProcessResult] = field(default_factory=list)
    dependency_graph: DependencyGraph | None = None
    global_symbol_table: GlobalSymbolTable | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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


class ObfuscationOrchestrator:
    """Coordinates the complete obfuscation workflow for multi-file projects.

    The orchestrator implements a three-phase workflow:
    1. Scan & Extract: Parse all files and extract symbols
    2. Build Graph & Pre-compute: Build dependency graph and symbol table
    3. Obfuscate: Process files in topological order

    Attributes:
        python_processor: Processor for Python files
        lua_processor: Processor for Lua files
        _logger: Logger instance

    Example:
        >>> orchestrator = ObfuscationOrchestrator()
        >>> result = orchestrator.process_files(
        ...     input_files=[Path("main.py")],
        ...     output_dir=Path("./out"),
        ...     config={"mangling_strategy": "sequential"}
        ... )
    """

    def __init__(self, config: ObfuscationConfig | None = None) -> None:
        """Initialize the orchestrator with processors.

        Args:
            config: Optional ObfuscationConfig instance. If provided,
                    the orchestrate() convenience method can be used.
        """
        self.python_processor = PythonProcessor(config=config)
        self.lua_processor = LuaProcessor(config=config)
        self._logger = logger
        self._project_root: Path | None = None
        self._config = config
        self._processed_engines: list = []
        self._current_state: JobState = JobState.PENDING

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

    def orchestrate(
        self,
        input_files: list[str | Path],
        output_dir: str | Path,
        progress_callback: Callable[[str, int, int], None] | None = None,
        project_root: str | Path | None = None
    ) -> OrchestrationResult:
        """Orchestrate obfuscation using the stored ObfuscationConfig.

        This is a convenience alias for process_files() that uses the config
        provided at construction time and accepts string paths.

        Args:
            input_files: List of input file paths (str or Path)
            output_dir: Output directory (str or Path)
            progress_callback: Optional progress callback
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
        readability, valid extensions, and output directory writability.
        
        Args:
            input_files: List of input file paths to validate
            output_dir: Output directory to validate
            
        Returns:
            ValidationResult with success flag and error/warning messages
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

    def process_files(
        self,
        input_files: list[Path],
        output_dir: Path,
        config: dict[str, Any] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
        project_root: Path | None = None
    ) -> OrchestrationResult:
        """Process multiple files with dependency-aware obfuscation.

        This is the main entry point for the orchestration workflow.

        Args:
            input_files: List of input file paths to process
            output_dir: Directory to write obfuscated files
            config: Configuration options for obfuscation
            progress_callback: Optional callback for progress updates
                Signature: callback(message: str, current: int, total: int)
            project_root: Explicit project root directory for import resolution.
                If None, computed from the common path of all input files.

        Returns:
            OrchestrationResult with processing details

        Example:
            >>> def on_progress(msg, current, total):
            ...     print(f"{msg} ({current}/{total})")
            >>> result = orchestrator.process_files(
            ...     input_files=[Path("main.py")],
            ...     output_dir=Path("./out"),
            ...     progress_callback=on_progress,
            ...     project_root=Path("/path/to/project")
            ... )
        """
        config = config or {}
        self._processed_engines = []
        result = OrchestrationResult(success=True)
        total_steps = len(input_files) + 4  # validation + scan + build + pre-compute + one per file
        current_step = 0

        # Initialize state to PENDING
        self._transition_state(JobState.PENDING, result)

        def report_progress(message: str) -> None:
            nonlocal current_step
            current_step += 1
            state_prefix = f"State: {self._current_state.name} - "
            full_message = state_prefix + message
            if progress_callback:
                progress_callback(full_message, current_step, total_steps)
            self._logger.info(message)

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

            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Transition to PROCESSING before file processing loop
            self._transition_state(JobState.PROCESSING, result)

            for file_path in processing_order:
                if file_path not in successfully_parsed:
                    continue

                report_progress(f"Processing {file_path.name}...")
                # Re-parse file just-in-time (AST not cached)
                process_result = self._process_file_in_order(
                    file_path,
                    global_table,
                    output_dir,
                    config
                )
                result.processed_files.append(process_result)

                if not process_result.success:
                    result.warnings.extend(process_result.errors)

            # Update overall success based on individual results
            failed_count = sum(
                1 for pr in result.processed_files if not pr.success
            )
            if failed_count > 0:
                result.warnings.append(
                    f"{failed_count} file(s) had processing errors"
                )

            result.metadata["total_files"] = len(input_files)
            result.metadata["processed_files"] = len(result.processed_files)
            result.metadata["failed_files"] = failed_count
            result.metadata["runtime_mode"] = self._config.runtime_mode if self._config else "embedded"
            result.metadata["runtime_files"] = []

            # Transition to COMPLETED after successful processing
            self._transition_state(JobState.COMPLETED, result)
            report_progress("Job completed")

            # Generate hybrid runtime files if in hybrid mode
            if self._config is not None and self._config.runtime_mode == "hybrid":
                self._generate_hybrid_runtime_files(output_dir, result)

        except Exception as e:
            result.success = False
            result.errors.append(f"Orchestration failed: {e}")
            self._logger.error(f"Orchestration failed: {e}", exc_info=True)
            # Transition to FAILED on exception
            self._transition_state(JobState.FAILED, result)
            report_progress("Job failed")

        return result

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

    def _process_file_in_order(
        self,
        file_path: Path,
        global_table: GlobalSymbolTable,
        output_dir: Path,
        config: dict[str, Any]
    ) -> ProcessResult:
        """Process a single file using the global symbol table.

        Re-parses the file just-in-time to avoid caching ASTs in memory.
        AST is released immediately after writing output to bound memory.

        Args:
            file_path: Path to the file
            global_table: Pre-computed GlobalSymbolTable
            output_dir: Output directory
            config: Configuration options

        Returns:
            ProcessResult for this file
        """
        language = self._detect_language(file_path)

        # Compute relative path from project root to preserve directory structure
        if self._project_root:
            try:
                relative_path = file_path.resolve().relative_to(self._project_root.resolve())
                output_path = output_dir / relative_path
            except ValueError:
                # File is not relative to project root, use just the filename
                self._logger.warning(
                    f"File {file_path} is not relative to project root {self._project_root}, "
                    f"using filename only"
                )
                output_path = output_dir / file_path.name
        else:
            # No project root, use just the filename (fallback)
            output_path = output_dir / file_path.name

        try:
            # Re-parse file just-in-time to bound memory usage
            if language == "python":
                parse_result = self.python_processor.parse_file(file_path)
                if not parse_result.success or not parse_result.ast_node:
                    return ProcessResult(
                        file_path=file_path,
                        output_path=None,
                        success=False,
                        errors=parse_result.errors or ["Failed to re-parse file"]
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
                        self._write_output(output_path, final_code)
                        return ProcessResult(
                            file_path=file_path,
                            output_path=output_path,
                            success=True,
                            warnings=[]
                        )
                    else:
                        return ProcessResult(
                            file_path=file_path,
                            output_path=None,
                            success=False,
                            errors=gen_result.errors
                        )
                else:
                    return ProcessResult(
                        file_path=file_path,
                        output_path=None,
                        success=False,
                        errors=transform_result.get("errors", ["Transformation failed"])
                    )

            elif language == "lua":
                parse_result = self.lua_processor.parse_file(file_path)
                if not parse_result.success or not parse_result.ast_node:
                    return ProcessResult(
                        file_path=file_path,
                        output_path=None,
                        success=False,
                        errors=parse_result.errors or ["Failed to re-parse file"]
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
                    self._write_output(output_path, final_code)
                    return ProcessResult(
                        file_path=file_path,
                        output_path=output_path,
                        success=True,
                        warnings=[]
                    )
                else:
                    return ProcessResult(
                        file_path=file_path,
                        output_path=None,
                        success=False,
                        errors=transform_result.get("errors", ["Transformation failed"])
                    )
            else:
                return ProcessResult(
                    file_path=file_path,
                    output_path=None,
                    success=False,
                    errors=[f"Unsupported language: {language}"]
                )

        except Exception as e:
            self._logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            return ProcessResult(
                file_path=file_path,
                output_path=None,
                success=False,
                errors=[str(e)]
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
        """Generate shared runtime files for hybrid mode.

        Collects all unique runtime requirements from tracked engines and generates
        consolidated runtime files (obf_runtime.py and/or obf_runtime.lua) in
        the output directory root.

        Args:
            output_dir: Directory to write runtime files to.
            result: OrchestrationResult to update with runtime file metadata.
        """
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

    def _write_output(self, output_path: Path, code: str) -> None:
        """Write obfuscated code to output file.

        Args:
            output_path: Path to write to
            code: Obfuscated code content
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(code, encoding="utf-8")
        self._logger.debug(f"Wrote output to {output_path}")

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

