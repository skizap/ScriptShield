"""Dependency graph module for analyzing file relationships.

This module provides data structures and algorithms for building dependency
graphs from Python and Lua source files, detecting circular dependencies,
and determining the optimal processing order via topological sorting.

Example:
    >>> from pathlib import Path
    >>> from obfuscator.core.dependency_graph import DependencyAnalyzer
    >>> 
    >>> analyzer = DependencyAnalyzer(Path("/project"))
    >>> analyzer.analyze_file(file_path, symbol_table)
    >>> order = analyzer.get_processing_order()
    >>> cycles = analyzer.detect_cycles()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from obfuscator.utils.logger import get_logger


# ============================================================================
# Custom Exceptions
# ============================================================================


class CircularDependencyError(Exception):
    """Exception raised when circular dependencies are detected.
    
    Attributes:
        cycle: List of file paths forming the circular dependency
        message: Detailed error message with line numbers
        
    Example:
        >>> cycle = [Path("a.py"), Path("b.py"), Path("a.py")]
        >>> raise CircularDependencyError(cycle)
    """
    
    def __init__(self, cycle: list[Path], line_info: dict[tuple[Path, Path], int] | None = None):
        """Initialize the exception with cycle information.
        
        Args:
            cycle: List of paths forming the cycle
            line_info: Optional dict mapping (from, to) edges to line numbers
        """
        self.cycle = cycle
        self.line_info = line_info or {}
        
        # Build detailed message with line numbers
        parts = []
        for i in range(len(cycle) - 1):
            from_path = cycle[i]
            to_path = cycle[i + 1]
            line_num = self.line_info.get((from_path, to_path), 0)
            if line_num:
                parts.append(f"{from_path.name}:{line_num}")
            else:
                parts.append(str(from_path.name))
        
        self.message = f"Circular dependency detected: {' -> '.join(parts)}"
        super().__init__(self.message)


class DependencyResolutionError(Exception):
    """Exception raised when an import cannot be resolved to a file.
    
    Attributes:
        file_path: Path of the file containing the import
        import_statement: The import statement that couldn't be resolved
        message: Detailed error message
        
    Example:
        >>> raise DependencyResolutionError(Path("main.py"), "missing_module")
    """
    
    def __init__(self, file_path: Path, import_statement: str, details: str = ""):
        """Initialize the exception with resolution failure details.
        
        Args:
            file_path: Path of the file with the unresolved import
            import_statement: The import that couldn't be resolved
            details: Optional additional details about the failure
        """
        self.file_path = file_path
        self.import_statement = import_statement
        detail_suffix = f": {details}" if details else ""
        self.message = (
            f"Cannot resolve import '{import_statement}' "
            f"in {file_path}{detail_suffix}"
        )
        super().__init__(self.message)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class DependencyNode:
    """Represents a single file in the dependency graph.
    
    Each node tracks the file's imports and exports for dependency analysis.
    Nodes are hashable and can be used in sets and as dict keys.
    
    Attributes:
        file_path: Absolute path to the source file
        language: Language identifier ('python' or 'lua')
        imports: List of module names this file imports
        exports: List of symbol names this file exports
        is_processed: Whether the file has been analyzed
        metadata: Additional metadata for extensibility
        
    Example:
        >>> node = DependencyNode(
        ...     file_path=Path("/project/main.py").resolve(),
        ...     language="python",
        ...     imports=["os", "sys"],
        ...     exports=["main", "Config"]
        ... )
    """
    file_path: Path
    language: str
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    is_processed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate node fields after initialization."""
        if not self.file_path.is_absolute():
            self.file_path = self.file_path.resolve()
        
        if self.language not in ("python", "lua"):
            raise ValueError(f"Invalid language: {self.language}. Must be 'python' or 'lua'")
    
    def __hash__(self) -> int:
        """Return hash based on file path for set/dict usage."""
        return hash(self.file_path)
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on file path."""
        if not isinstance(other, DependencyNode):
            return NotImplemented
        return self.file_path == other.file_path


@dataclass
class DependencyEdge:
    """Represents a dependency relationship between two files.

    An edge indicates that one file imports/requires another.
    Self-loops are prevented via validation.

    Attributes:
        from_node: Path of the importing file
        to_node: Path of the imported file
        import_type: Type of import ('absolute' or 'relative')
        imported_symbols: List of specific symbols imported
        line_number: Source line number of the import statement

    Example:
        >>> edge = DependencyEdge(
        ...     from_node=Path("/project/main.py"),
        ...     to_node=Path("/project/utils.py"),
        ...     import_type="relative",
        ...     imported_symbols=["helper"],
        ...     line_number=5
        ... )
    """
    from_node: Path
    to_node: Path
    import_type: str = "absolute"
    imported_symbols: list[str] = field(default_factory=list)
    line_number: int = 0

    def __post_init__(self) -> None:
        """Validate edge fields and prevent self-loops."""
        # Resolve paths to absolute
        if not self.from_node.is_absolute():
            self.from_node = self.from_node.resolve()
        if not self.to_node.is_absolute():
            self.to_node = self.to_node.resolve()

        # Prevent self-loops
        if self.from_node == self.to_node:
            raise ValueError(
                f"Self-loop detected: {self.from_node} cannot import itself"
            )

        if self.import_type not in ("absolute", "relative"):
            raise ValueError(
                f"Invalid import_type: {self.import_type}. "
                "Must be 'absolute' or 'relative'"
            )


@dataclass
class DependencyGraph:
    """Main graph container with incremental construction support.

    Maintains nodes, edges, and adjacency lists for efficient dependency
    queries. Supports incremental construction for processing large projects.

    Attributes:
        nodes: Dictionary mapping file paths to DependencyNode objects
        edges: List of all dependency edges
        adjacency_list: Forward adjacency (file -> files it depends on)
        reverse_adjacency: Reverse adjacency (file -> files depending on it)

    Example:
        >>> graph = DependencyGraph()
        >>> node = graph.add_node(Path("main.py"), "python", ["utils"], ["main"])
        >>> graph.add_edge(Path("main.py"), Path("utils.py"), "relative", [], 1)
    """
    nodes: dict[Path, DependencyNode] = field(default_factory=dict)
    edges: list[DependencyEdge] = field(default_factory=list)
    adjacency_list: dict[Path, set[Path]] = field(default_factory=dict)
    reverse_adjacency: dict[Path, set[Path]] = field(default_factory=dict)

    def add_node(
        self,
        file_path: Path,
        language: str,
        imports: list[str],
        exports: list[str],
        metadata: dict[str, Any] | None = None
    ) -> DependencyNode:
        """Add or update a node in the graph.

        If the node already exists, it will be updated with the new information.

        Args:
            file_path: Path to the source file
            language: Language identifier ('python' or 'lua')
            imports: List of module names imported by this file
            exports: List of symbols exported by this file
            metadata: Optional additional metadata

        Returns:
            The created or updated DependencyNode

        Example:
            >>> graph = DependencyGraph()
            >>> node = graph.add_node(
            ...     Path("/project/main.py"),
            ...     "python",
            ...     ["os", "sys"],
            ...     ["main"]
            ... )
        """
        resolved_path = file_path.resolve() if not file_path.is_absolute() else file_path

        node = DependencyNode(
            file_path=resolved_path,
            language=language,
            imports=imports,
            exports=exports,
            is_processed=True,
            metadata=metadata or {}
        )

        self.nodes[resolved_path] = node

        # Initialize adjacency entries if needed
        if resolved_path not in self.adjacency_list:
            self.adjacency_list[resolved_path] = set()
        if resolved_path not in self.reverse_adjacency:
            self.reverse_adjacency[resolved_path] = set()

        return node

    def add_edge(
        self,
        from_path: Path,
        to_path: Path,
        import_type: str,
        symbols: list[str],
        line_num: int
    ) -> None:
        """Add a dependency edge and update adjacency lists.

        Args:
            from_path: Path of the importing file
            to_path: Path of the imported file
            import_type: Type of import ('absolute' or 'relative')
            symbols: List of specific symbols imported
            line_num: Line number of the import statement

        Example:
            >>> graph = DependencyGraph()
            >>> graph.add_edge(
            ...     Path("/project/main.py"),
            ...     Path("/project/utils.py"),
            ...     "relative",
            ...     ["helper"],
            ...     5
            ... )
        """
        from_resolved = from_path.resolve() if not from_path.is_absolute() else from_path
        to_resolved = to_path.resolve() if not to_path.is_absolute() else to_path

        edge = DependencyEdge(
            from_node=from_resolved,
            to_node=to_resolved,
            import_type=import_type,
            imported_symbols=symbols,
            line_number=line_num
        )

        self.edges.append(edge)

        # Update adjacency lists
        if from_resolved not in self.adjacency_list:
            self.adjacency_list[from_resolved] = set()
        self.adjacency_list[from_resolved].add(to_resolved)

        if to_resolved not in self.reverse_adjacency:
            self.reverse_adjacency[to_resolved] = set()
        self.reverse_adjacency[to_resolved].add(from_resolved)

    def get_node(self, file_path: Path) -> DependencyNode | None:
        """Retrieve a node by its file path.

        Args:
            file_path: Path to look up

        Returns:
            The DependencyNode if found, None otherwise
        """
        resolved = file_path.resolve() if not file_path.is_absolute() else file_path
        return self.nodes.get(resolved)

    def get_dependencies(self, file_path: Path) -> set[Path]:
        """Get direct dependencies of a file.

        Args:
            file_path: Path of the file to query

        Returns:
            Set of paths this file depends on

        Example:
            >>> deps = graph.get_dependencies(Path("main.py"))
            >>> print(f"main.py depends on: {deps}")
        """
        resolved = file_path.resolve() if not file_path.is_absolute() else file_path
        return self.adjacency_list.get(resolved, set()).copy()

    def get_dependents(self, file_path: Path) -> set[Path]:
        """Get files that depend on this file.

        Args:
            file_path: Path of the file to query

        Returns:
            Set of paths that depend on this file

        Example:
            >>> dependents = graph.get_dependents(Path("utils.py"))
            >>> print(f"Files depending on utils.py: {dependents}")
        """
        resolved = file_path.resolve() if not file_path.is_absolute() else file_path
        return self.reverse_adjacency.get(resolved, set()).copy()

    def has_path(self, from_path: Path, to_path: Path) -> bool:
        """Check if a dependency path exists between two files.

        Uses BFS to determine if there's a path from from_path to to_path.

        Args:
            from_path: Starting file path
            to_path: Target file path

        Returns:
            True if a path exists, False otherwise
        """
        from_resolved = from_path.resolve() if not from_path.is_absolute() else from_path
        to_resolved = to_path.resolve() if not to_path.is_absolute() else to_path

        if from_resolved == to_resolved:
            return True

        visited: set[Path] = set()
        queue = [from_resolved]

        while queue:
            current = queue.pop(0)
            if current == to_resolved:
                return True

            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.adjacency_list.get(current, set()):
                if neighbor not in visited:
                    queue.append(neighbor)

        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph for debugging/logging.

        Returns:
            Dictionary representation of the graph
        """
        return {
            "nodes": {
                str(path): {
                    "language": node.language,
                    "imports": node.imports,
                    "exports": node.exports,
                    "is_processed": node.is_processed,
                    "metadata": node.metadata
                }
                for path, node in self.nodes.items()
            },
            "edges": [
                {
                    "from": str(edge.from_node),
                    "to": str(edge.to_node),
                    "import_type": edge.import_type,
                    "imported_symbols": edge.imported_symbols,
                    "line_number": edge.line_number
                }
                for edge in self.edges
            ],
            "node_count": len(self.nodes),
            "edge_count": len(self.edges)
        }

    def get_processing_order(self) -> list[Path]:
        """Get files in topological order for processing.

        Uses Kahn's algorithm to determine an order where each file
        is processed after all its dependencies.

        Returns:
            List of file paths in processing order (dependencies first)

        Raises:
            CircularDependencyError: If the graph contains cycles

        Example:
            >>> order = graph.get_processing_order()
            >>> for path in order:
            ...     process_file(path)
        """
        # Calculate in-degrees (number of dependencies)
        in_degree: dict[Path, int] = {path: 0 for path in self.nodes}

        for path in self.nodes:
            for dep in self.adjacency_list.get(path, set()):
                # Only count dependencies on files we know about
                if dep in in_degree:
                    in_degree[path] += 1

        # Start with nodes that have no dependencies
        queue = [path for path, degree in in_degree.items() if degree == 0]
        result: list[Path] = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # Decrease in-degree for nodes that depend on current
            for dependent in self.reverse_adjacency.get(current, set()):
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Check for cycles
        if len(result) != len(self.nodes):
            # Build edge line info for error message
            edge_line_info: dict[tuple[Path, Path], int] = {}
            for edge in self.edges:
                edge_line_info[(edge.from_node, edge.to_node)] = edge.line_number
            # There's a cycle - find it using DFS
            cycles = self._detect_cycles()
            if cycles:
                raise CircularDependencyError(cycles[0], edge_line_info)

        return result

    def _detect_cycles(self) -> list[list[Path]]:
        """Detect cycles in the graph using DFS.

        Returns:
            List of cycles found (each cycle is a list of paths)
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[Path, int] = {path: WHITE for path in self.nodes}
        parent: dict[Path, Path | None] = {path: None for path in self.nodes}
        cycles: list[list[Path]] = []

        def dfs(node: Path) -> None:
            color[node] = GRAY
            for neighbor in self.adjacency_list.get(node, set()):
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    # Found a cycle - reconstruct it
                    cycle = [neighbor]
                    current = node
                    while current != neighbor:
                        cycle.append(current)
                        current = parent.get(current)
                        if current is None:
                            break
                    cycle.append(neighbor)
                    cycles.append(cycle[::-1])
                elif color[neighbor] == WHITE:
                    parent[neighbor] = node
                    dfs(neighbor)
            color[node] = BLACK

        for node in self.nodes:
            if color[node] == WHITE:
                dfs(node)

        return cycles


# ============================================================================
# Dependency Analyzer
# ============================================================================


# Type alias for symbol tables from both processors
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from obfuscator.processors.symbol_extractor import SymbolTable
    from obfuscator.processors.lua_symbol_extractor import LuaSymbolTable


class DependencyAnalyzer:
    """Analyzes dependencies between source files and builds a dependency graph.

    This class provides the main analysis engine for extracting dependencies
    from symbol tables, resolving imports to file paths, and building a graph
    that can be used for topological sorting and cycle detection.

    Attributes:
        root_directory: Base directory for resolving relative imports
        graph: The dependency graph being built

    Example:
        >>> from pathlib import Path
        >>> analyzer = DependencyAnalyzer(Path("/project"))
        >>> analyzer.analyze_file(file_path, symbol_table)
        >>> order = analyzer.get_processing_order()
    """

    def __init__(self, root_directory: Path) -> None:
        """Initialize the analyzer with a root directory.

        Args:
            root_directory: Base directory for resolving relative imports
        """
        self.root_directory = root_directory.resolve()
        self.graph = DependencyGraph()
        self._logger = get_logger("obfuscator.core.dependency_graph")
        self._edge_line_info: dict[tuple[Path, Path], int] = {}

    def analyze_file(
        self,
        file_path: Path,
        symbol_table: "SymbolTable | LuaSymbolTable"
    ) -> DependencyNode:
        """Analyze a file and add it to the dependency graph.

        Extracts imports from the symbol table, resolves them to file paths,
        and creates nodes and edges in the graph.

        Args:
            file_path: Path to the source file
            symbol_table: Symbol table from the file's extraction

        Returns:
            The created DependencyNode for this file

        Example:
            >>> node = analyzer.analyze_file(Path("main.py"), symbols)
            >>> print(f"Found {len(node.imports)} imports")
        """
        resolved_path = file_path.resolve()

        # Determine language from file extension or symbol table type
        language = self._detect_language(file_path, symbol_table)

        # Extract imports from symbol table
        import_names: list[str] = []

        for import_info in symbol_table.imports:
            if language == "python":
                # Python ImportInfo has module_name
                import_names.append(import_info.module_name)
            else:
                # Lua LuaImportInfo has module_path
                import_names.append(import_info.module_path)

        # Get exported symbols
        exports = symbol_table.get_exported_symbols()

        # Create/update node
        node = self.graph.add_node(
            file_path=resolved_path,
            language=language,
            imports=import_names,
            exports=exports
        )

        self._logger.debug(f"Analyzed file: {file_path.name} ({len(import_names)} imports)")

        # Process each import and try to resolve to file path
        for import_info in symbol_table.imports:
            self._process_import(resolved_path, import_info, language)

        return node

    def _detect_language(
        self,
        file_path: Path,
        symbol_table: "SymbolTable | LuaSymbolTable"
    ) -> str:
        """Detect the language based on file extension or symbol table type.

        Args:
            file_path: Path to the file
            symbol_table: The symbol table object

        Returns:
            Language identifier ('python' or 'lua')
        """
        suffix = file_path.suffix.lower()
        if suffix in (".lua", ".luau"):
            return "lua"
        elif suffix in (".py", ".pyw"):
            return "python"

        # Fall back to checking symbol table type
        class_name = symbol_table.__class__.__name__
        if "Lua" in class_name:
            return "lua"
        return "python"

    def _process_import(
        self,
        file_path: Path,
        import_info: Any,
        language: str
    ) -> None:
        """Process a single import and add an edge if resolved.

        Args:
            file_path: Path of the importing file
            import_info: Import information object
            language: Language identifier
        """
        if language == "python":
            module_name = import_info.module_name
            level = import_info.level
            is_relative = level > 0
            line_number = import_info.line_number
            symbols = import_info.imported_names

            # For Python, pass level and imported_names for proper resolution
            resolved = self._resolve_python_import(
                file_path, module_name, is_relative, level, symbols
            )
        else:
            module_name = import_info.module_path
            is_relative = import_info.is_relative
            line_number = import_info.line_number
            symbols = [import_info.alias] if import_info.alias else []

            resolved = self._resolve_lua_import(file_path, module_name, is_relative)

        if resolved:
            try:
                self.graph.add_edge(
                    from_path=file_path,
                    to_path=resolved,
                    import_type="relative" if is_relative else "absolute",
                    symbols=symbols,
                    line_num=line_number
                )
                self._edge_line_info[(file_path, resolved)] = line_number
                self._logger.debug(f"Added edge: {file_path.name} -> {resolved.name}")
            except ValueError as e:
                # Self-loop or invalid edge
                self._logger.warning(f"Skipped invalid edge: {e}")
        else:
            self._logger.warning(
                f"Could not resolve import '{module_name}' in {file_path.name}:{line_number}"
            )

    def _resolve_python_import(
        self,
        current_file: Path,
        module_name: str,
        is_relative: bool,
        level: int = 0,
        imported_names: list[str] | None = None
    ) -> Path | None:
        """Resolve a Python import to a file path.

        Args:
            current_file: Path of the importing file
            module_name: The module name (e.g., 'os.path' or 'utils')
            is_relative: Whether this is a relative import
            level: Relative import level (0 for absolute, 1 for '.', 2 for '..', etc.)
            imported_names: List of names being imported (for 'from . import name')

        Returns:
            Resolved file path or None
        """
        imported_names = imported_names or []

        # Determine base directory
        if is_relative:
            # Relative import: start from current file's directory
            # and walk up (level - 1) parents for multi-level imports
            base_dir = current_file.parent
            # level=1 means '.', level=2 means '..', etc.
            # For level=1, we stay in parent directory
            # For level=2, we go one more level up, etc.
            for _ in range(level - 1):
                base_dir = base_dir.parent
        else:
            # Absolute import: search from root
            base_dir = self.root_directory

        # Convert module name to path components
        parts = module_name.split(".") if module_name else []

        # Case 1: 'from . import sibling' or 'from .. import something'
        # module_name is empty, resolve each imported_name as a module
        if not parts and imported_names:
            for name in imported_names:
                # Try as module (.py file)
                module_path = base_dir / f"{name}.py"
                if module_path.exists():
                    return module_path.resolve()

                # Try as package (__init__.py)
                package_path = base_dir / name / "__init__.py"
                if package_path.exists():
                    return package_path.resolve()

            return None

        if not parts:
            return None

        # Case 2: 'from pkg import submodule' - try resolving module_name/imported_name first
        # This handles 'from utils import helper' where helper is a submodule
        if imported_names:
            for name in imported_names:
                # Try module_name/imported_name as a module
                submodule_path = base_dir / "/".join(parts) / f"{name}.py"
                if submodule_path.exists():
                    return submodule_path.resolve()

                # Try as package
                subpackage_path = base_dir / "/".join(parts) / name / "__init__.py"
                if subpackage_path.exists():
                    return subpackage_path.resolve()

        # Case 3: Standard module resolution
        # Try as package (__init__.py)
        package_path = base_dir / "/".join(parts) / "__init__.py"
        if package_path.exists():
            return package_path.resolve()

        # Try as module (.py file)
        if len(parts) > 1:
            module_path = base_dir / "/".join(parts[:-1]) / f"{parts[-1]}.py"
        else:
            module_path = base_dir / f"{parts[0]}.py"
        if module_path.exists():
            return module_path.resolve()

        # Try directly as module
        direct_path = base_dir / f"{'/'.join(parts)}.py"
        if direct_path.exists():
            return direct_path.resolve()

        return None

    def _resolve_lua_import(
        self,
        current_file: Path,
        require_path: str,
        is_relative: bool
    ) -> Path | None:
        """Resolve a Lua require() to a file path.

        Args:
            current_file: Path of the requiring file
            require_path: The require path (e.g., 'module.submodule')
            is_relative: Whether this is a relative require

        Returns:
            Resolved file path or None
        """
        # Detect and strip relative prefixes BEFORE replacing dots
        # Handle: "./" for current directory, "../" for parent, or leading "."
        cleaned_path = require_path

        if require_path.startswith("../"):
            # Parent directory reference
            cleaned_path = require_path[3:]
            is_relative = True
        elif require_path.startswith("./"):
            # Current directory reference
            cleaned_path = require_path[2:]
            is_relative = True
        elif require_path.startswith("."):
            # Leading dot (e.g., ".helper" as relative)
            cleaned_path = require_path[1:]
            is_relative = True

        # Now convert dots to path separators for module paths
        path_parts = cleaned_path.replace(".", "/")

        # Ensure path_parts doesn't start with "/" (which would make it absolute)
        if path_parts.startswith("/"):
            path_parts = path_parts.lstrip("/")

        if is_relative:
            base_dir = current_file.parent
        else:
            base_dir = self.root_directory

        # Try with .lua extension
        lua_path = base_dir / f"{path_parts}.lua"
        if lua_path.exists():
            return lua_path.resolve()

        # Try with .luau extension
        luau_path = base_dir / f"{path_parts}.luau"
        if luau_path.exists():
            return luau_path.resolve()

        # Try as directory with init.lua
        init_path = base_dir / path_parts / "init.lua"
        if init_path.exists():
            return init_path.resolve()

        return None

    def build_graph_incremental(
        self,
        files: list[tuple[Path, "SymbolTable | LuaSymbolTable"]]
    ) -> DependencyGraph:
        """Build dependency graph incrementally from a list of files.

        Processes files one at a time without loading all into memory,
        making it suitable for large projects.

        Args:
            files: List of (file_path, symbol_table) tuples

        Returns:
            The completed DependencyGraph

        Example:
            >>> files = [(Path("main.py"), main_symbols), (Path("utils.py"), utils_symbols)]
            >>> graph = analyzer.build_graph_incremental(files)
        """
        total = len(files)
        self._logger.info(f"Building dependency graph for {total} files")

        for i, (file_path, symbol_table) in enumerate(files):
            self.analyze_file(file_path, symbol_table)

            # Log progress every 100 files
            if (i + 1) % 100 == 0:
                self._logger.info(f"Processed {i + 1}/{total} files")

        self._logger.info(
            f"Dependency graph complete: {len(self.graph.nodes)} nodes, "
            f"{len(self.graph.edges)} edges"
        )

        return self.graph

    def get_processing_order(self) -> list[Path]:
        """Get files in topological order for processing.

        Uses Kahn's algorithm to determine an order where each file
        is processed after all its dependencies.

        Returns:
            List of file paths in processing order (dependencies first)

        Raises:
            CircularDependencyError: If the graph contains cycles

        Example:
            >>> order = analyzer.get_processing_order()
            >>> for path in order:
            ...     process_file(path)
        """
        # Calculate in-degrees (number of dependencies)
        in_degree: dict[Path, int] = {path: 0 for path in self.graph.nodes}

        for path in self.graph.nodes:
            for dep in self.graph.adjacency_list.get(path, set()):
                # Only count dependencies on files we know about
                if dep in in_degree:
                    in_degree[path] += 1

        # Start with nodes that have no dependencies
        queue = [path for path, degree in in_degree.items() if degree == 0]
        result: list[Path] = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # Decrease in-degree for nodes that depend on current
            for dependent in self.graph.reverse_adjacency.get(current, set()):
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Check for cycles
        if len(result) != len(self.graph.nodes):
            # There's a cycle - find it
            cycles = self.detect_cycles()
            if cycles:
                raise CircularDependencyError(cycles[0], self._edge_line_info)

        self._logger.info(f"Topological sort complete: {len(result)} files ordered")
        return result

    def detect_cycles(self) -> list[list[Path]]:
        """Detect all cycles in the dependency graph.

        Uses depth-first search with color marking to find strongly
        connected components representing cycles.

        Returns:
            List of cycles, where each cycle is a list of file paths

        Example:
            >>> cycles = analyzer.detect_cycles()
            >>> if cycles:
            ...     print(f"Found {len(cycles)} cycles")
            ...     for cycle in cycles:
            ...         print(" -> ".join(str(p) for p in cycle))
        """
        # Color states: 0 = white (unvisited), 1 = gray (in progress), 2 = black (done)
        color: dict[Path, int] = {path: 0 for path in self.graph.nodes}
        parent: dict[Path, Path | None] = {path: None for path in self.graph.nodes}
        cycles: list[list[Path]] = []

        def dfs(node: Path, path: list[Path]) -> None:
            """Depth-first search with cycle detection."""
            color[node] = 1  # Mark as in progress

            for neighbor in self.graph.adjacency_list.get(node, set()):
                if neighbor not in color:
                    # External dependency, skip
                    continue

                if color[neighbor] == 0:
                    # Unvisited, recurse
                    parent[neighbor] = node
                    dfs(neighbor, path + [neighbor])
                elif color[neighbor] == 1:
                    # Back edge found - cycle detected
                    cycle_start_idx = path.index(neighbor) if neighbor in path else -1
                    if cycle_start_idx >= 0:
                        cycle = path[cycle_start_idx:] + [neighbor]
                        cycles.append(cycle)
                    else:
                        # Neighbor is in progress but not in current path
                        # Reconstruct cycle
                        cycles.append([neighbor, node, neighbor])

            color[node] = 2  # Mark as done

        # Run DFS from each unvisited node
        for node in self.graph.nodes:
            if color[node] == 0:
                dfs(node, [node])

        if cycles:
            self._logger.warning(f"Detected {len(cycles)} cycle(s) in dependency graph")
            for cycle in cycles:
                cycle_str = " -> ".join(p.name for p in cycle)
                self._logger.warning(f"  Cycle: {cycle_str}")
        else:
            self._logger.debug("No cycles detected in dependency graph")

        return cycles

