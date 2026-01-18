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
from obfuscator.utils.logger import get_logger

logger = get_logger("obfuscator.core.orchestrator")


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
        processed_files: List of ProcessResult for each file
        dependency_graph: The constructed dependency graph
        global_symbol_table: The pre-computed symbol table
        errors: List of global error messages
        warnings: List of global warning messages
        metadata: Additional metadata about the process
    """
    success: bool
    processed_files: list[ProcessResult] = field(default_factory=list)
    dependency_graph: DependencyGraph | None = None
    global_symbol_table: GlobalSymbolTable | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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

    def __init__(self) -> None:
        """Initialize the orchestrator with processors."""
        self.python_processor = PythonProcessor()
        self.lua_processor = LuaProcessor()
        self._logger = logger
        self._project_root: Path | None = None

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
        result = OrchestrationResult(success=True)
        total_steps = len(input_files) * 2 + 2  # scan + build + process each file
        current_step = 0

        def report_progress(message: str) -> None:
            nonlocal current_step
            current_step += 1
            if progress_callback:
                progress_callback(message, current_step, total_steps)
            self._logger.info(message)

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
            # Phase 1: Scan and extract symbols (ASTs are discarded to bound memory)
            report_progress("Scanning files and extracting symbols...")
            successfully_parsed, symbol_tables = self._scan_and_extract_symbols(
                input_files, result
            )

            if not successfully_parsed:
                result.success = False
                result.errors.append("No files were successfully parsed")
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

        except Exception as e:
            result.success = False
            result.errors.append(f"Orchestration failed: {e}")
            self._logger.error(f"Orchestration failed: {e}", exc_info=True)

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

                ast_node = parse_result.ast_node
                transform_result = self.python_processor.obfuscate_with_symbol_table(
                    ast_node, file_path, global_table
                )
                if transform_result.success:
                    gen_result = self.python_processor.generate_code(
                        transform_result.ast_node
                    )
                    # Release AST references to allow garbage collection
                    del ast_node
                    del transform_result

                    if gen_result.success:
                        self._write_output(output_path, gen_result.code)
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
                        errors=transform_result.errors
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

                ast_node = parse_result.ast_node
                gen_result = self.lua_processor.obfuscate_with_symbol_table(
                    ast_node, file_path, global_table
                )
                # Release AST reference to allow garbage collection
                del ast_node

                if gen_result.success:
                    self._write_output(output_path, gen_result.code)
                    return ProcessResult(
                        file_path=file_path,
                        output_path=output_path,
                        success=True,
                        warnings=gen_result.warnings
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

