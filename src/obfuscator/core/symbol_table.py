"""Global symbol table module for pre-computing mangled names.

This module provides a centralized symbol table system that pre-computes all
mangled names before obfuscation begins, ensuring consistency across files.
The table is immutable after initialization and supports both Python and Lua.

Example:
    >>> from pathlib import Path
    >>> from obfuscator.core.dependency_graph import DependencyAnalyzer
    >>> from obfuscator.core.symbol_table import SymbolTableBuilder, GlobalSymbolTable
    >>> 
    >>> # 1. Build dependency graph
    >>> analyzer = DependencyAnalyzer(Path("/project"))
    >>> files = [(path, extract_symbols(path)) for path in project_files]
    >>> graph = analyzer.build_graph_incremental(files)
    >>> 
    >>> # 2. Pre-compute symbol table
    >>> builder = SymbolTableBuilder()
    >>> config = {"identifier_prefix": "_0x", "mangling_strategy": "sequential"}
    >>> global_table = builder.build_from_dependency_graph(graph, symbol_tables, config)
    >>> 
    >>> # 3. Use during obfuscation
    >>> mangled = global_table.get_mangled_name(file_path, "my_function", "global")
"""

from __future__ import annotations

import hashlib
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from obfuscator.utils.logger import get_logger

if TYPE_CHECKING:
    from obfuscator.core.dependency_graph import DependencyGraph
    from obfuscator.processors.symbol_extractor import SymbolTable
    from obfuscator.processors.lua_symbol_extractor import LuaSymbolTable

logger = get_logger("obfuscator.core.symbol_table")

# Valid scope values
VALID_SCOPES = {"global", "local", "upvalue", "nonlocal"}

# Valid symbol types
VALID_SYMBOL_TYPES = {"function", "class", "variable", "import"}

# Valid languages
VALID_LANGUAGES = {"python", "lua"}


@dataclass
class SymbolEntry:
    """Represents a single symbol with comprehensive metadata.

    Attributes:
        original_name: The original symbol name
        mangled_name: The obfuscated name (assigned during pre-computation)
        scope: Symbol scope ("global", "local", "upvalue", "nonlocal")
        language: Source language ("python" or "lua")
        file_path: Absolute path to the source file
        line_number: Source line number where symbol is defined
        symbol_type: Type of symbol ("function", "class", "variable", "import")
        is_exported: Whether symbol is exported (public API)
        references: Tuple of (file_path, line_number) tuples for cross-file refs
        metadata: Additional extensible metadata (becomes immutable after freeze)

    Example:
        >>> entry = SymbolEntry(
        ...     original_name="my_function",
        ...     mangled_name="_0x1",
        ...     scope="global",
        ...     language="python",
        ...     file_path=Path("/project/main.py"),
        ...     line_number=10,
        ...     symbol_type="function"
        ... )
    """
    original_name: str
    mangled_name: str
    scope: str
    language: str
    file_path: Path
    line_number: int
    symbol_type: str
    is_exported: bool = False
    references: list[tuple[Path, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _frozen: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if self.scope not in VALID_SCOPES:
            raise ValueError(
                f"Invalid scope: {self.scope}. Must be one of {VALID_SCOPES}"
            )

        if self.language not in VALID_LANGUAGES:
            raise ValueError(
                f"Invalid language: {self.language}. Must be one of {VALID_LANGUAGES}"
            )

        if self.symbol_type not in VALID_SYMBOL_TYPES:
            raise ValueError(
                f"Invalid symbol_type: {self.symbol_type}. "
                f"Must be one of {VALID_SYMBOL_TYPES}"
            )

        # Ensure file_path is absolute
        if not self.file_path.is_absolute():
            object.__setattr__(self, 'file_path', self.file_path.resolve())

    def _freeze(self) -> None:
        """Make this entry immutable by converting references to tuple."""
        if not self._frozen:
            # Convert mutable references list to immutable tuple
            object.__setattr__(self, 'references', tuple(self.references))
            # Convert mutable metadata dict to immutable frozenset of items
            object.__setattr__(self, 'metadata', dict(self.metadata))  # Keep as dict but mark frozen
            object.__setattr__(self, '_frozen', True)

    def __hash__(self) -> int:
        """Hash based on (file_path, original_name, scope) for set/dict usage."""
        return hash((self.file_path, self.original_name, self.scope))

    def __eq__(self, other: object) -> bool:
        """Check equality based on (file_path, original_name, scope)."""
        if not isinstance(other, SymbolEntry):
            return NotImplemented
        return (
            self.file_path == other.file_path
            and self.original_name == other.original_name
            and self.scope == other.scope
        )


# Python reserved names (builtins and keywords)
PYTHON_BUILTINS = {
    # Built-in functions
    "abs", "aiter", "all", "anext", "any", "ascii", "bin", "bool", "breakpoint",
    "bytearray", "bytes", "callable", "chr", "classmethod", "compile", "complex",
    "delattr", "dict", "dir", "divmod", "enumerate", "eval", "exec", "filter",
    "float", "format", "frozenset", "getattr", "globals", "hasattr", "hash",
    "help", "hex", "id", "input", "int", "isinstance", "issubclass", "iter",
    "len", "list", "locals", "map", "max", "memoryview", "min", "next", "object",
    "oct", "open", "ord", "pow", "print", "property", "range", "repr", "reversed",
    "round", "set", "setattr", "slice", "sorted", "staticmethod", "str", "sum",
    "super", "tuple", "type", "vars", "zip", "__import__",
    # Keywords
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
    # Special names
    "__name__", "__doc__", "__package__", "__loader__", "__spec__", "__path__",
    "__file__", "__cached__", "__builtins__", "__annotations__",
}

# Python magic methods (dunder methods)
PYTHON_MAGIC_METHODS = {
    "__init__", "__new__", "__del__", "__repr__", "__str__", "__bytes__",
    "__format__", "__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__",
    "__hash__", "__bool__", "__getattr__", "__getattribute__", "__setattr__",
    "__delattr__", "__dir__", "__get__", "__set__", "__delete__", "__set_name__",
    "__init_subclass__", "__class_getitem__", "__call__", "__len__", "__length_hint__",
    "__getitem__", "__setitem__", "__delitem__", "__missing__", "__iter__",
    "__reversed__", "__contains__", "__add__", "__sub__", "__mul__", "__matmul__",
    "__truediv__", "__floordiv__", "__mod__", "__divmod__", "__pow__", "__lshift__",
    "__rshift__", "__and__", "__xor__", "__or__", "__radd__", "__rsub__", "__rmul__",
    "__rmatmul__", "__rtruediv__", "__rfloordiv__", "__rmod__", "__rdivmod__",
    "__rpow__", "__rlshift__", "__rrshift__", "__rand__", "__rxor__", "__ror__",
    "__iadd__", "__isub__", "__imul__", "__imatmul__", "__itruediv__", "__ifloordiv__",
    "__imod__", "__ipow__", "__ilshift__", "__irshift__", "__iand__", "__ixor__",
    "__ior__", "__neg__", "__pos__", "__abs__", "__invert__", "__complex__",
    "__int__", "__float__", "__index__", "__round__", "__trunc__", "__floor__",
    "__ceil__", "__enter__", "__exit__", "__await__", "__aiter__", "__anext__",
    "__aenter__", "__aexit__",
}

# Lua keywords
LUA_KEYWORDS = {
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function",
    "goto", "if", "in", "local", "nil", "not", "or", "repeat", "return", "then",
    "true", "until", "while",
}

# Roblox API names to preserve
ROBLOX_API = {
    # Global objects
    "game", "workspace", "script", "plugin", "shared", "settings", "tick",
    # Services
    "Players", "ReplicatedStorage", "ServerScriptService", "ServerStorage",
    "StarterGui", "StarterPack", "StarterPlayer", "Lighting", "SoundService",
    "TweenService", "UserInputService", "RunService", "HttpService",
    "MarketplaceService", "DataStoreService", "MessagingService", "PathfindingService",
    "CollectionService", "Teams", "Chat", "TextService", "LocalizationService",
    "VRService", "ContentProvider", "CoreGui", "TestService", "ReplicatedFirst",
    # Classes
    "Instance", "Vector2", "Vector3", "CFrame", "Color3", "UDim", "UDim2",
    "Enum", "BrickColor", "Ray", "Region3", "Rect", "TweenInfo", "NumberSequence",
    "ColorSequence", "NumberRange", "PhysicalProperties", "Random", "DateTime",
    # Methods
    "GetService", "FindFirstChild", "WaitForChild", "Clone", "Destroy", "new",
    "Connect", "Disconnect", "Fire", "Invoke", "BindToRenderStep", "UnbindFromRenderStep",
    "spawn", "wait", "delay", "coroutine", "require", "typeof", "getmetatable",
    "setmetatable", "rawget", "rawset", "rawequal", "pcall", "xpcall", "error",
    "assert", "print", "warn", "select", "pairs", "ipairs", "next", "unpack",
    "tonumber", "tostring", "type", "table", "string", "math", "os", "debug",
    # Common instance properties/methods
    "Name", "Parent", "ClassName", "Changed", "GetChildren", "GetDescendants",
    "IsA", "FindFirstAncestor", "FindFirstChildOfClass", "FindFirstChildWhichIsA",
    "GetAttribute", "SetAttribute", "GetPropertyChangedSignal",
    # Events
    "PlayerAdded", "PlayerRemoving", "CharacterAdded", "CharacterRemoving",
    "Stepped", "Heartbeat", "RenderStepped",
}


class SymbolMangler:
    """Generates unique mangled names for symbols.

    Supports multiple mangling strategies and language-specific reserved names.

    Attributes:
        _used_names: Set of already generated names
        _config: Configuration options (prefix, charset, strategy)
        _counter: Counter for sequential naming

    Example:
        >>> mangler = SymbolMangler({"identifier_prefix": "_0x", "mangling_strategy": "sequential"})
        >>> name1 = mangler.generate_name("my_func", "global", "python")
        >>> name2 = mangler.generate_name("other_func", "global", "python")
        >>> print(name1, name2)
        _0x1 _0x2
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the mangler with configuration.

        Args:
            config: Configuration dict with keys:
                - identifier_prefix: Prefix for mangled names (default: "_0x")
                - mangling_strategy: "sequential", "random", or "minimal"
                - charset: Characters for minimal strategy
        """
        self._config = config or {}
        self._used_names: set[str] = set()
        self._counter: int = 0
        self._minimal_counter: int = 0

        # Add all reserved names to used_names
        self._reserved = self._get_all_reserved()
        self._used_names.update(self._reserved)

    def generate_name(self, original: str, scope: str, language: str) -> str:
        """Generate a unique mangled name.

        Args:
            original: Original symbol name
            scope: Symbol scope
            language: Source language

        Returns:
            Unique mangled name
        """
        # Check if name should be preserved
        if self.is_reserved(original, language):
            return original

        strategy = self._config.get("mangling_strategy", "sequential")
        prefix = self._config.get("identifier_prefix", "_0x")

        if strategy == "sequential":
            return self._generate_sequential(prefix)
        elif strategy == "random":
            return self._generate_random(prefix)
        elif strategy == "minimal":
            return self._generate_minimal()
        else:
            return self._generate_sequential(prefix)

    def _generate_sequential(self, prefix: str) -> str:
        """Generate sequential names like _0x1, _0x2, etc."""
        while True:
            self._counter += 1
            name = f"{prefix}{self._counter:x}"
            if name not in self._used_names:
                self._used_names.add(name)
                return name

    def _generate_random(self, prefix: str) -> str:
        """Generate random hex names like _0xa3f2, etc."""
        while True:
            hex_part = ''.join(random.choices('0123456789abcdef', k=4))
            name = f"{prefix}{hex_part}"
            if name not in self._used_names:
                self._used_names.add(name)
                return name

    def _generate_minimal(self) -> str:
        """Generate minimal names like a, b, ..., aa, ab, etc."""
        chars = string.ascii_lowercase
        while True:
            self._minimal_counter += 1
            name = self._int_to_base(self._minimal_counter, chars)
            if name not in self._used_names and not name[0].isdigit():
                self._used_names.add(name)
                return name

    def _int_to_base(self, num: int, chars: str) -> str:
        """Convert integer to base-N string using given characters."""
        if num == 0:
            return chars[0]
        result = []
        base = len(chars)
        while num:
            num, remainder = divmod(num - 1, base)
            result.append(chars[remainder])
        return ''.join(reversed(result))

    def is_reserved(self, name: str, language: str) -> bool:
        """Check if a name is reserved and should not be mangled.

        Args:
            name: Name to check
            language: Source language

        Returns:
            True if name should be preserved
        """
        # Check Python reserved names
        if language == "python":
            if name in PYTHON_BUILTINS:
                return True
            if name in PYTHON_MAGIC_METHODS:
                return True
            # Preserve dunder names
            if name.startswith("__") and name.endswith("__"):
                return True

        # Check Lua reserved names
        if language == "lua":
            if name in LUA_KEYWORDS:
                return True
            if name in ROBLOX_API:
                return True

        return False

    def _get_all_reserved(self) -> set[str]:
        """Get all reserved names for both languages."""
        reserved = set()
        reserved.update(PYTHON_BUILTINS)
        reserved.update(PYTHON_MAGIC_METHODS)
        reserved.update(LUA_KEYWORDS)
        reserved.update(ROBLOX_API)
        return reserved

    def _get_python_reserved(self) -> set[str]:
        """Return Python builtins and keywords."""
        return PYTHON_BUILTINS | PYTHON_MAGIC_METHODS

    def _get_lua_reserved(self) -> set[str]:
        """Return Lua keywords and Roblox API names."""
        return LUA_KEYWORDS | ROBLOX_API


class GlobalSymbolTable:
    """Main immutable symbol registry for pre-computed mangled names.

    This class maintains a mapping of symbols to their mangled names.
    Once frozen, the table becomes immutable to prevent accidental
    modifications during obfuscation.

    Attributes:
        _symbols: Dict mapping (file_path, original_name, scope) to SymbolEntry
        _name_to_mangled: Fast lookup dict for mangled names
        _is_frozen: Whether the table is immutable
        _reserved_names: Set of names that cannot be mangled
        _mangling_config: Configuration options

    Example:
        >>> table = GlobalSymbolTable()
        >>> entry = SymbolEntry(
        ...     original_name="func", mangled_name="_0x1", scope="global",
        ...     language="python", file_path=Path("/test.py"),
        ...     line_number=1, symbol_type="function"
        ... )
        >>> table.add_symbol(entry)
        >>> table.freeze()
        >>> print(table.get_mangled_name(Path("/test.py"), "func", "global"))
        _0x1
    """

    def __init__(self, mangling_config: dict[str, Any] | None = None) -> None:
        """Initialize the symbol table.

        Args:
            mangling_config: Configuration for the mangler
        """
        self._symbols: dict[tuple[Path, str, str], SymbolEntry] = {}
        self._name_to_mangled: dict[tuple[Path, str, str], str] = {}
        self._is_frozen: bool = False
        self._reserved_names: set[str] = set()
        self._mangling_config = mangling_config or {}

    def add_symbol(self, entry: SymbolEntry) -> None:
        """Add a symbol entry to the table.

        Args:
            entry: The SymbolEntry to add

        Raises:
            RuntimeError: If table is frozen
        """
        if self._is_frozen:
            raise RuntimeError(
                "Cannot add symbol to frozen GlobalSymbolTable. "
                "Table is immutable after freeze()."
            )

        key = (entry.file_path, entry.original_name, entry.scope)
        self._symbols[key] = entry
        self._name_to_mangled[key] = entry.mangled_name

        logger.debug(
            f"Added symbol: {entry.original_name} -> {entry.mangled_name} "
            f"({entry.scope}, {entry.file_path.name})"
        )

    def freeze(self) -> None:
        """Make the table immutable.

        After calling this method, no more symbols can be added and all
        SymbolEntry objects become immutable (references converted to tuples).
        """
        # Freeze all individual symbol entries
        for entry in self._symbols.values():
            entry._freeze()

        self._is_frozen = True
        logger.info(
            f"GlobalSymbolTable frozen with {len(self._symbols)} symbols"
        )

    @property
    def is_frozen(self) -> bool:
        """Check if table is immutable."""
        return self._is_frozen

    def get_mangled_name(
        self,
        file_path: Path,
        original_name: str,
        scope: str
    ) -> str | None:
        """Retrieve the mangled name for a symbol.

        Args:
            file_path: Path to the source file
            original_name: Original symbol name
            scope: Symbol scope

        Returns:
            Mangled name or None if not found
        """
        resolved_path = file_path.resolve() if not file_path.is_absolute() else file_path
        key = (resolved_path, original_name, scope)
        return self._name_to_mangled.get(key)

    def get_symbol(
        self,
        file_path: Path,
        original_name: str,
        scope: str
    ) -> SymbolEntry | None:
        """Retrieve full symbol entry (defensive copy if frozen).

        Args:
            file_path: Path to the source file
            original_name: Original symbol name
            scope: Symbol scope

        Returns:
            SymbolEntry or None if not found. If table is frozen, returns
            the frozen entry directly (immutable after freeze).
        """
        resolved_path = file_path.resolve() if not file_path.is_absolute() else file_path
        key = (resolved_path, original_name, scope)
        entry = self._symbols.get(key)
        return entry  # Entry is already frozen after freeze()

    def add_reference(
        self,
        file_path: Path,
        original_name: str,
        scope: str,
        ref_file: Path,
        ref_line: int
    ) -> None:
        """Track a cross-file reference to a symbol.

        Args:
            file_path: Path to file where symbol is defined
            original_name: Symbol name
            scope: Symbol scope
            ref_file: Path to file containing the reference
            ref_line: Line number of the reference

        Raises:
            RuntimeError: If table is frozen
        """
        if self._is_frozen:
            raise RuntimeError("Cannot add reference to frozen GlobalSymbolTable")

        resolved_path = file_path.resolve() if not file_path.is_absolute() else file_path
        key = (resolved_path, original_name, scope)

        if key in self._symbols:
            ref_path = ref_file.resolve() if not ref_file.is_absolute() else ref_file
            self._symbols[key].references.append((ref_path, ref_line))
            logger.debug(
                f"Added reference: {original_name} referenced from "
                f"{ref_file.name}:{ref_line}"
            )

    def get_all_symbols(self) -> list[SymbolEntry]:
        """Return all symbols.

        Returns:
            List of all SymbolEntry objects. After freeze(), entries are immutable.
        """
        return list(self._symbols.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/debugging.

        Returns:
            Dictionary representation of the table
        """
        return {
            "is_frozen": self._is_frozen,
            "symbol_count": len(self._symbols),
            "symbols": [
                {
                    "original_name": entry.original_name,
                    "mangled_name": entry.mangled_name,
                    "scope": entry.scope,
                    "language": entry.language,
                    "file_path": str(entry.file_path),
                    "line_number": entry.line_number,
                    "symbol_type": entry.symbol_type,
                    "is_exported": entry.is_exported,
                    "reference_count": len(entry.references),
                }
                for entry in self._symbols.values()
            ],
            "config": self._mangling_config,
        }


class SymbolTableBuilder:
    """Builder for constructing GlobalSymbolTable from dependency graph.

    This class processes symbol tables from multiple files and creates
    a unified GlobalSymbolTable with pre-computed mangled names.

    Example:
        >>> builder = SymbolTableBuilder()
        >>> config = {"identifier_prefix": "_0x", "mangling_strategy": "sequential"}
        >>> global_table = builder.build_from_dependency_graph(graph, symbol_tables, config)
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._logger = get_logger("obfuscator.core.symbol_table")

    def build_from_dependency_graph(
        self,
        graph: "DependencyGraph",
        symbol_tables: dict[Path, "SymbolTable | LuaSymbolTable"],
        config: dict[str, Any] | None = None
    ) -> GlobalSymbolTable:
        """Build GlobalSymbolTable from dependency graph and symbol tables.

        Args:
            graph: The dependency graph with topological order
            symbol_tables: Dict mapping file paths to their symbol tables
            config: Configuration options for mangling

        Returns:
            Frozen GlobalSymbolTable with all mangled names
        """
        config = config or {}
        mangler = SymbolMangler(config)
        global_table = GlobalSymbolTable(config)

        # Extract preservation flags from config
        preserve_exports = config.get("preserve_exports", False)
        preserve_constants = config.get("preserve_constants", False)

        self._logger.info(
            f"Building global symbol table from {len(symbol_tables)} files "
            f"(preserve_exports={preserve_exports}, preserve_constants={preserve_constants})"
        )

        # Get processing order from dependency graph deterministically
        # This uses Kahn's algorithm for topological sort
        processing_order = graph.get_processing_order() if graph.nodes else []

        # Process files in topological order
        for file_path in processing_order:
            if file_path not in symbol_tables:
                continue

            symbol_table = symbol_tables[file_path]
            node = graph.get_node(file_path)
            language = node.language if node else self._detect_language(file_path)

            # Extract and process symbols
            entries = self._extract_symbols_from_file(
                file_path, symbol_table, language
            )

            # Assign mangled names and add to global table
            for entry in entries:
                mangled_name = self._compute_mangled_name(
                    entry, mangler, preserve_exports, preserve_constants
                )
                entry.mangled_name = mangled_name
                global_table.add_symbol(entry)

        # Process any files not in the graph (files without dependencies)
        for file_path, symbol_table in symbol_tables.items():
            if file_path not in processing_order:
                language = self._detect_language(file_path)
                entries = self._extract_symbols_from_file(
                    file_path, symbol_table, language
                )
                for entry in entries:
                    mangled_name = self._compute_mangled_name(
                        entry, mangler, preserve_exports, preserve_constants
                    )
                    entry.mangled_name = mangled_name
                    global_table.add_symbol(entry)

        # Detect cross-file references (only exported/global symbols)
        self._detect_cross_file_references(graph, global_table)

        # Freeze the table
        global_table.freeze()

        self._logger.info(
            f"Global symbol table built with {len(global_table.get_all_symbols())} symbols"
        )

        return global_table

    def build_with_fallback_order(
        self,
        symbol_tables: dict[Path, "SymbolTable | LuaSymbolTable"],
        config: dict[str, Any] | None = None
    ) -> GlobalSymbolTable:
        """Build GlobalSymbolTable using original file order (fallback for circular dependencies).

        This method processes files in dictionary order without relying on the dependency
        graph's topological sort. It's used when circular dependencies prevent normal
        processing order determination.

        Args:
            symbol_tables: Dict mapping file paths to their symbol tables
            config: Configuration options for mangling

        Returns:
            Frozen GlobalSymbolTable with all mangled names
        """
        config = config or {}
        mangler = SymbolMangler(config)
        global_table = GlobalSymbolTable(config)

        # Extract preservation flags from config
        preserve_exports = config.get("preserve_exports", False)
        preserve_constants = config.get("preserve_constants", False)

        self._logger.info(
            f"Building global symbol table (fallback order) from {len(symbol_tables)} files "
            f"(preserve_exports={preserve_exports}, preserve_constants={preserve_constants})"
        )

        # Process files in dictionary order (deterministic in Python 3.7+)
        for file_path, symbol_table in symbol_tables.items():
            language = self._detect_language(file_path)
            entries = self._extract_symbols_from_file(
                file_path, symbol_table, language
            )

            # Assign mangled names and add to global table
            for entry in entries:
                mangled_name = self._compute_mangled_name(
                    entry, mangler, preserve_exports, preserve_constants
                )
                entry.mangled_name = mangled_name
                global_table.add_symbol(entry)

        # Freeze the table
        global_table.freeze()

        self._logger.info(
            f"Global symbol table built (fallback) with {len(global_table.get_all_symbols())} symbols"
        )

        return global_table

    def _compute_mangled_name(
        self,
        entry: SymbolEntry,
        mangler: SymbolMangler,
        preserve_exports: bool,
        preserve_constants: bool
    ) -> str:
        """Compute the mangled name for a symbol respecting preservation flags.

        Args:
            entry: The symbol entry to mangle
            mangler: The mangler instance to use
            preserve_exports: If True, preserve exported symbols
            preserve_constants: If True, preserve constant symbols

        Returns:
            The mangled name (or original name if preserved)
        """
        # Check if symbol should be preserved due to export flag
        if preserve_exports and entry.is_exported:
            self._logger.debug(
                f"Preserving exported symbol: {entry.original_name}"
            )
            return entry.original_name

        # Check if symbol should be preserved due to constant flag
        if preserve_constants and self._is_constant(entry):
            self._logger.debug(
                f"Preserving constant symbol: {entry.original_name}"
            )
            return entry.original_name

        # Apply mangling
        return mangler.generate_name(
            entry.original_name, entry.scope, entry.language
        )

    def _is_constant(self, entry: SymbolEntry) -> bool:
        """Check if a symbol entry represents a constant.

        Args:
            entry: The symbol entry to check

        Returns:
            True if the symbol is a constant
        """
        # Check metadata for explicit constant flag
        if entry.metadata.get("is_constant", False):
            return True

        # Check symbol type
        if entry.symbol_type == "constant":
            return True

        # Python convention: ALL_CAPS names are constants
        if entry.language == "python" and entry.symbol_type == "variable":
            name = entry.original_name
            # Check for ALL_CAPS pattern (e.g., MAX_SIZE, DEFAULT_VALUE)
            if name.isupper() and "_" in name:
                return True
            # Single word uppercase (e.g., PI, DEBUG)
            if name.isupper() and len(name) > 1:
                return True

        return False

    def _extract_symbols_from_file(
        self,
        file_path: Path,
        symbol_table: "SymbolTable | LuaSymbolTable",
        language: str
    ) -> list[SymbolEntry]:
        """Convert extractor output to SymbolEntry objects.

        Args:
            file_path: Path to the source file
            symbol_table: The symbol table from extraction
            language: Source language

        Returns:
            List of SymbolEntry objects
        """
        if language == "python":
            return self._process_python_symbols(symbol_table, file_path)
        else:
            return self._process_lua_symbols(symbol_table, file_path)

    def _process_python_symbols(
        self,
        symbol_table: "SymbolTable",
        file_path: Path
    ) -> list[SymbolEntry]:
        """Handle Python-specific symbol extraction.

        Args:
            symbol_table: Python SymbolTable
            file_path: Source file path

        Returns:
            List of SymbolEntry objects
        """
        entries: list[SymbolEntry] = []
        resolved_path = file_path.resolve()
        exports = set(symbol_table.get_exported_symbols())

        # Process functions
        for func in symbol_table.functions:
            entries.append(SymbolEntry(
                original_name=func.name,
                mangled_name="",  # Will be assigned later
                scope=func.scope,
                language="python",
                file_path=resolved_path,
                line_number=func.line_number,
                symbol_type="function",
                is_exported=func.name in exports,
                metadata={"is_async": func.is_async, "parent_class": func.parent_class}
            ))

        # Process classes
        for cls in symbol_table.classes:
            entries.append(SymbolEntry(
                original_name=cls.name,
                mangled_name="",
                scope=cls.scope,
                language="python",
                file_path=resolved_path,
                line_number=cls.line_number,
                symbol_type="class",
                is_exported=cls.name in exports,
                metadata={"bases": cls.bases, "methods": cls.methods}
            ))

        # Process variables
        for var in symbol_table.variables:
            entries.append(SymbolEntry(
                original_name=var.name,
                mangled_name="",
                scope=var.scope,
                language="python",
                file_path=resolved_path,
                line_number=var.line_number,
                symbol_type="variable",
                is_exported=False,
                metadata={"is_constant": var.is_constant, "context": var.context}
            ))

        return entries

    def _process_lua_symbols(
        self,
        symbol_table: "LuaSymbolTable",
        file_path: Path
    ) -> list[SymbolEntry]:
        """Handle Lua-specific symbol extraction.

        Args:
            symbol_table: Lua LuaSymbolTable
            file_path: Source file path

        Returns:
            List of SymbolEntry objects
        """
        entries: list[SymbolEntry] = []
        resolved_path = file_path.resolve()
        exports = set(symbol_table.get_exported_symbols())

        # Process functions
        for func in symbol_table.functions:
            # Map Lua scope to our scope values
            scope = "local" if func.is_local else func.scope
            if scope not in VALID_SCOPES:
                scope = "global"

            entries.append(SymbolEntry(
                original_name=func.name,
                mangled_name="",
                scope=scope,
                language="lua",
                file_path=resolved_path,
                line_number=func.line_number,
                symbol_type="function",
                is_exported=func.name in exports,
                metadata={
                    "is_method": func.is_method,
                    "parent_table": func.parent_table,
                    "parameters": func.parameters
                }
            ))

        # Process variables
        for var in symbol_table.variables:
            scope = var.scope if var.scope in VALID_SCOPES else "global"

            entries.append(SymbolEntry(
                original_name=var.name,
                mangled_name="",
                scope=scope,
                language="lua",
                file_path=resolved_path,
                line_number=var.line_number,
                symbol_type="variable",
                is_exported=var.name in exports,
                metadata={"is_constant": var.is_constant, "context": var.context}
            ))

        return entries

    def _detect_cross_file_references(
        self,
        graph: "DependencyGraph",
        global_table: GlobalSymbolTable
    ) -> None:
        """Analyze imports to track cross-file references.

        Only binds references to exported/global symbols to avoid incorrectly
        binding imports to local symbols when globals are missing.

        Args:
            graph: The dependency graph with edges
            global_table: The symbol table to update with references
        """
        for edge in graph.edges:
            from_file = edge.from_node
            to_file = edge.to_node

            # Get the node's exports list to verify symbol is exported
            node = graph.get_node(to_file)
            exported_symbols = set(node.exports) if node else set()

            # For each imported symbol, find the export and add reference
            for symbol_name in edge.imported_symbols:
                # Only check global scope - cross-file references must be to
                # exported/global symbols, never local symbols
                symbol = global_table.get_symbol(to_file, symbol_name, "global")

                # Verify the symbol exists and is either exported or global scope
                if symbol and (symbol.is_exported or symbol_name in exported_symbols):
                    global_table.add_reference(
                        file_path=to_file,
                        original_name=symbol_name,
                        scope="global",
                        ref_file=from_file,
                        ref_line=edge.line_number
                    )
                    self._logger.debug(
                        f"Cross-file reference: {from_file.name} imports "
                        f"{symbol_name} from {to_file.name}"
                    )
                elif symbol:
                    # Symbol exists but is not exported - log warning
                    self._logger.warning(
                        f"Import {symbol_name} from {to_file.name} exists but "
                        f"is not exported (is_exported={symbol.is_exported})"
                    )

    def _detect_language(self, file_path: Path) -> str:
        """Detect language from file extension.

        Args:
            file_path: Path to the file

        Returns:
            Language identifier
        """
        suffix = file_path.suffix.lower()
        if suffix in (".lua", ".luau"):
            return "lua"
        return "python"

