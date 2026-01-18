"""Luau and LuaJIT syntax extension support for the Lua processor.

This module provides preprocessing and postprocessing capabilities to handle Luau-specific
type annotations and LuaJIT-specific syntax patterns. Since luaparser only supports standard
Lua 5.1-5.4 syntax, this module strips Luau type annotations before parsing and restores
them after code generation.

Supported Luau features:
    - Variable type annotations: `local x: number = 5`
    - Function parameter types: `function(param: type)`
    - Function return types: `function(): type`
    - Type aliases: `type Name = TypeDefinition`
    - Type casts: `(expression :: type)`
    - Table types: `{field: type}`
    - Union types: `type1 | type2`
    - Intersection types: `type1 & type2`
    - Optional types: `type?`
    - Generic types: `Type<T>`

Detected LuaJIT features (for warnings):
    - FFI usage: `require("ffi")`
    - Integer literals with suffixes: `1ll`, `1ull`
    - BitOp library: `bit.*` operations
    - JIT pragmas: `jit.on()`, `jit.off()`

Example:
    >>> from pathlib import Path
    >>> preprocessor = LuauPreprocessor()
    >>> stripped_code, metadata = preprocessor.strip_type_annotations(luau_source)
    >>> # Parse with luaparser, modify AST, generate code
    >>> generator = LuauCodeGenerator()
    >>> restored_code = generator.restore_type_annotations(lua_code, metadata)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from obfuscator.utils.logger import get_logger

# Initialize logger
logger = get_logger("obfuscator.processors.luau_extensions")

# =============================================================================
# Luau Built-in Types Registry (Step 6)
# =============================================================================

LUAU_BUILTIN_TYPES: set[str] = {
    "any", "nil", "boolean", "number", "string", "thread",
    "unknown", "never", "void", "true", "false",
}

LUAU_TYPE_OPERATORS: set[str] = {"|", "&", "?"}

# =============================================================================
# Regex Patterns for Luau Syntax (Step 10)
# =============================================================================

# Pattern to match content inside strings (to avoid stripping type-like syntax in strings)
STRING_PATTERN = re.compile(
    r'(?P<string>'
    r'"(?:[^"\\]|\\.)*"'       # Double-quoted strings
    r"|'(?:[^'\\]|\\.)*'"      # Single-quoted strings
    r'|\[\[.*?\]\]'            # Long strings [[...]]
    r'|\[=+\[.*?\]=+\]'        # Long strings [=[...]=], [==[...]==], etc.
    r')',
    re.DOTALL
)

# Pattern to match comments (to avoid stripping type-like syntax in comments)
COMMENT_PATTERN = re.compile(
    r'(?P<comment>'
    r'--\[\[.*?\]\]'           # Multi-line comments --[[...]]
    r'|--\[=+\[.*?\]=+\]'      # Multi-line comments with equals --[=[...]=]
    r'|--[^\n]*'               # Single-line comments
    r')',
    re.DOTALL
)

# Variable type annotation: local x: type = value OR x: type = value
VARIABLE_TYPE_ANNOTATION_PATTERN = re.compile(
    r'(?P<prefix>(?:local\s+)?(?P<varname>[a-zA-Z_][a-zA-Z0-9_]*))\s*'
    r':\s*(?P<type>[^=,\n]+?)\s*(?=[=,\n]|$)'
)

# Function parameter type: (param: type) or (param: type, param2: type)
FUNCTION_PARAM_TYPE_PATTERN = re.compile(
    r'(?P<param>[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(?P<type>[^,\)]+?)(?=[,\)])'
)

# Function return type: function(...): type or function(...) -> type
FUNCTION_RETURN_TYPE_PATTERN = re.compile(
    r'\)\s*(?::|->)\s*(?P<returntype>[^\n{]+?)(?=\s*(?:\n|$|{|end|local|function|if|for|while|repeat|return))'
)

# Type alias: type Name = TypeDefinition
TYPE_ALIAS_PATTERN = re.compile(
    r'^(?P<indent>\s*)type\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*'
    r'(?:<(?P<generics>[^>]+)>)?\s*=\s*(?P<typedef>.+?)(?=\n(?:\s*type\s|\s*local\s|\s*function\s|\s*$)|$)',
    re.MULTILINE
)

# Type cast: (expression :: type)
TYPE_CAST_PATTERN = re.compile(
    r'(?P<expr>[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*(?:\[[^\]]+\])?)\s*::\s*(?P<type>[a-zA-Z_][a-zA-Z0-9_<>,\s\|\&\?]*)'
)

# Table type: {field: type} or {[key]: value}
TABLE_TYPE_PATTERN = re.compile(
    r'\{\s*(?P<fields>(?:[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*[^,}]+(?:\s*,\s*)?)+)\s*\}'
)

# Union type: type1 | type2
UNION_TYPE_PATTERN = re.compile(r'(?P<type1>[a-zA-Z_][a-zA-Z0-9_<>]*)\s*\|\s*(?P<type2>[a-zA-Z_][a-zA-Z0-9_<>]*)')

# Intersection type: type1 & type2
INTERSECTION_TYPE_PATTERN = re.compile(r'(?P<type1>[a-zA-Z_][a-zA-Z0-9_<>]*)\s*&\s*(?P<type2>[a-zA-Z_][a-zA-Z0-9_<>]*)')

# Optional type: type?
OPTIONAL_TYPE_PATTERN = re.compile(r'(?P<type>[a-zA-Z_][a-zA-Z0-9_<>]*)\?')

# Variadic type: ...: type
VARIADIC_TYPE_PATTERN = re.compile(r'\.\.\.\s*:\s*(?P<type>[a-zA-Z_][a-zA-Z0-9_<>\|\&\?]+)')

# Generic type: Type<T> or Type<T, U>
GENERIC_TYPE_PATTERN = re.compile(r'(?P<base>[a-zA-Z_][a-zA-Z0-9_]*)<(?P<params>[^>]+)>')

# Export type: export type Name = ...
EXPORT_TYPE_PATTERN = re.compile(
    r'^(?P<indent>\s*)export\s+type\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*'
    r'(?:<(?P<generics>[^>]+)>)?\s*=\s*(?P<typedef>.+?)(?=\n(?:\s*type\s|\s*export\s|\s*local\s|\s*function\s|\s*$)|$)',
    re.MULTILINE
)


# =============================================================================
# Luau Type Information Dataclasses (Step 1)
# =============================================================================

@dataclass
class LuauTypeAnnotation:
    """Store type annotation information for a variable.

    Attributes:
        variable_name: Name of the annotated variable.
        type_string: The type annotation string (e.g., "number", "string?").
        line_number: 1-based line number where the annotation appears.
        column_offset: 0-based column offset in the line.
        is_local: Whether this is a local variable declaration.
        original_text: The original text that was stripped.
    """
    variable_name: str
    type_string: str
    line_number: int
    column_offset: int
    is_local: bool = False
    original_text: str = ""


@dataclass
class LuauFunctionType:
    """Store function type signature information.

    Attributes:
        function_name: Name of the function (may be empty for anonymous functions).
        parameters: List of (param_name, type_string) tuples.
        return_types: List of return type strings (functions can return multiple values).
        generics: List of generic type parameter names (e.g., ["T", "U"]).
        line_number: 1-based line number where the function is defined.
        column_offset: 0-based column offset in the line.
        original_param_text: Original parameter text with types.
        original_return_text: Original return type text.
    """
    function_name: str
    parameters: list[tuple[str, str]] = field(default_factory=list)
    return_types: list[str] = field(default_factory=list)
    generics: list[str] = field(default_factory=list)
    line_number: int = 0
    column_offset: int = 0
    original_param_text: str = ""
    original_return_text: str = ""


@dataclass
class LuauTableType:
    """Store table type definition information.

    Attributes:
        fields: Dictionary mapping field names to their type strings.
        index_type: Optional type for table indexer (e.g., [number]: string).
        line_number: 1-based line number where the table type appears.
        column_offset: 0-based column offset in the line.
        original_text: The original table type text.
    """
    fields: dict[str, str] = field(default_factory=dict)
    index_type: Optional[tuple[str, str]] = None  # (key_type, value_type)
    line_number: int = 0
    column_offset: int = 0
    original_text: str = ""


@dataclass
class LuauTypeAlias:
    """Store type alias definition information.

    Attributes:
        alias_name: Name of the type alias.
        type_definition: The type definition string.
        generics: List of generic type parameter names.
        line_number: 1-based line number where the alias is defined.
        is_exported: Whether this type is exported.
        original_text: The original type alias declaration.
    """
    alias_name: str
    type_definition: str
    generics: list[str] = field(default_factory=list)
    line_number: int = 0
    is_exported: bool = False
    original_text: str = ""


@dataclass
class LuauTypeCast:
    """Store type cast expression information.

    Attributes:
        expression: The expression being cast.
        target_type: The target type for the cast.
        line_number: 1-based line number where the cast appears.
        column_offset: 0-based column offset in the line.
        original_text: The original cast text.
    """
    expression: str
    target_type: str
    line_number: int = 0
    column_offset: int = 0
    original_text: str = ""


@dataclass
class LuauTypeMetadata:
    """Aggregate all Luau type information for a file.

    Attributes:
        annotations: List of variable type annotations.
        function_types: List of function type signatures.
        table_types: List of table type definitions.
        type_aliases: List of type alias definitions.
        type_casts: List of type cast expressions.
        original_source: The original source code before stripping.
        stripped_source: The source code after stripping type annotations.
    """
    annotations: list[LuauTypeAnnotation] = field(default_factory=list)
    function_types: list[LuauFunctionType] = field(default_factory=list)
    table_types: list[LuauTableType] = field(default_factory=list)
    type_aliases: list[LuauTypeAlias] = field(default_factory=list)
    type_casts: list[LuauTypeCast] = field(default_factory=list)
    original_source: str = ""
    stripped_source: str = ""


# =============================================================================
# Luau Type Utilities (Step 6 continued)
# =============================================================================

def is_luau_builtin_type(type_str: str) -> bool:
    """Check if a type string is a built-in Luau type.

    Args:
        type_str: The type string to check.

    Returns:
        True if the type is a built-in Luau type.

    Example:
        >>> is_luau_builtin_type("number")
        True
        >>> is_luau_builtin_type("MyCustomType")
        False
    """
    # Strip whitespace and optional marker
    clean_type = type_str.strip().rstrip("?")
    return clean_type in LUAU_BUILTIN_TYPES


def parse_luau_type_expression(type_str: str) -> dict:
    """Parse a complex Luau type expression into its components.

    Args:
        type_str: The type expression to parse.

    Returns:
        Dictionary containing parsed type information:
        - kind: "simple", "union", "intersection", "optional", "generic", "function"
        - base: The base type (for simple/generic/optional)
        - types: List of types (for union/intersection)
        - params: Generic parameters (for generic types)
        - is_optional: Whether the type is optional

    Example:
        >>> parse_luau_type_expression("number?")
        {'kind': 'optional', 'base': 'number', 'is_optional': True}
        >>> parse_luau_type_expression("string | nil")
        {'kind': 'union', 'types': ['string', 'nil'], 'is_optional': False}
    """
    type_str = type_str.strip()

    # Check for optional type
    if type_str.endswith("?"):
        base = type_str[:-1].strip()
        return {"kind": "optional", "base": base, "is_optional": True}

    # Check for union type
    if "|" in type_str and "&" not in type_str:
        types = [t.strip() for t in type_str.split("|")]
        return {"kind": "union", "types": types, "is_optional": False}

    # Check for intersection type
    if "&" in type_str and "|" not in type_str:
        types = [t.strip() for t in type_str.split("&")]
        return {"kind": "intersection", "types": types, "is_optional": False}

    # Check for generic type
    generic_match = GENERIC_TYPE_PATTERN.match(type_str)
    if generic_match:
        return {
            "kind": "generic",
            "base": generic_match.group("base"),
            "params": [p.strip() for p in generic_match.group("params").split(",")],
            "is_optional": False,
        }

    # Check for function type: (params) -> return
    if "->" in type_str or (type_str.startswith("(") and ")" in type_str):
        return {"kind": "function", "raw": type_str, "is_optional": False}

    # Simple type
    return {"kind": "simple", "base": type_str, "is_optional": False}


def validate_luau_type_syntax(type_str: str) -> tuple[bool, str]:
    """Validate Luau type syntax and return error messages.

    Args:
        type_str: The type string to validate.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.

    Example:
        >>> validate_luau_type_syntax("number")
        (True, "")
        >>> validate_luau_type_syntax("Array<")
        (False, "Unclosed generic type bracket")
    """
    type_str = type_str.strip()

    if not type_str:
        return (False, "Empty type string")

    # Check for balanced angle brackets (generics)
    open_count = type_str.count("<")
    close_count = type_str.count(">")
    if open_count != close_count:
        return (False, "Unclosed generic type bracket")

    # Check for balanced parentheses (function types)
    open_parens = type_str.count("(")
    close_parens = type_str.count(")")
    if open_parens != close_parens:
        return (False, "Unclosed parenthesis in type")

    # Check for balanced braces (table types)
    open_braces = type_str.count("{")
    close_braces = type_str.count("}")
    if open_braces != close_braces:
        return (False, "Unclosed brace in table type")

    # Check for invalid characters
    invalid_chars = set("@#$%^`~")
    for char in invalid_chars:
        if char in type_str:
            return (False, f"Invalid character '{char}' in type")

    return (True, "")


# =============================================================================
# Luau Syntax Preprocessor (Step 2)
# =============================================================================

class LuauPreprocessor:
    """Preprocessor for stripping Luau type annotations from source code.

    This class extracts and removes Luau-specific type annotations from source code,
    allowing it to be parsed by standard Lua parsers. The extracted type information
    is stored in metadata for later restoration.

    Example:
        >>> preprocessor = LuauPreprocessor()
        >>> source = "local x: number = 5"
        >>> stripped, metadata = preprocessor.strip_type_annotations(source)
        >>> print(stripped)  # "local x = 5"
        >>> print(len(metadata.annotations))  # 1
    """

    def __init__(self, debug_mode: bool = False) -> None:
        """Initialize the Luau preprocessor.

        Args:
            debug_mode: If True, preserves comments indicating where annotations were stripped.
        """
        self.debug_mode = debug_mode
        self.logger = get_logger("obfuscator.processors.luau_extensions")

    def strip_type_annotations(self, source_code: str) -> tuple[str, LuauTypeMetadata]:
        """Strip Luau type annotations from source code.

        Args:
            source_code: The Luau source code to process.

        Returns:
            Tuple of (stripped_code, metadata) where stripped_code is valid Lua 5.1-5.4
            and metadata contains all extracted type information.
        """
        self.logger.info("Applying Luau preprocessing to strip type annotations")

        metadata = LuauTypeMetadata(original_source=source_code)

        # First, protect strings and comments from modification
        protected_regions = self._find_protected_regions(source_code)

        # Extract type information before stripping
        metadata.type_aliases = self._extract_type_aliases(source_code, protected_regions)
        metadata.type_casts = self._extract_type_casts(source_code, protected_regions)
        metadata.annotations = self._extract_variable_annotations(source_code, protected_regions)
        metadata.function_types = self._extract_function_types(source_code, protected_regions)
        metadata.table_types = self._extract_table_types(source_code, protected_regions)

        # Strip type annotations from code
        stripped_code = self._strip_all_annotations(source_code, protected_regions, metadata)

        metadata.stripped_source = stripped_code

        self.logger.debug(
            f"Extracted: {len(metadata.annotations)} annotations, "
            f"{len(metadata.function_types)} function types, "
            f"{len(metadata.type_aliases)} type aliases, "
            f"{len(metadata.type_casts)} type casts"
        )

        return stripped_code, metadata

    def _find_protected_regions(self, source_code: str) -> list[tuple[int, int]]:
        """Find regions that should not be modified (strings and comments).

        Args:
            source_code: The source code to analyze.

        Returns:
            List of (start, end) tuples for protected regions.
        """
        regions = []

        # Find all string literals
        for match in STRING_PATTERN.finditer(source_code):
            regions.append((match.start(), match.end()))

        # Find all comments
        for match in COMMENT_PATTERN.finditer(source_code):
            regions.append((match.start(), match.end()))

        # Sort by start position and merge overlapping regions
        regions.sort(key=lambda x: x[0])
        merged = []
        for start, end in regions:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        return merged

    def _is_in_protected_region(self, pos: int, protected_regions: list[tuple[int, int]]) -> bool:
        """Check if a position is within a protected region.

        Args:
            pos: The character position to check.
            protected_regions: List of protected (start, end) regions.

        Returns:
            True if the position is within a protected region.
        """
        for start, end in protected_regions:
            if start <= pos < end:
                return True
            if start > pos:
                break  # Regions are sorted, no need to check further
        return False

    def _get_line_col(self, source_code: str, pos: int) -> tuple[int, int]:
        """Get line number and column offset for a position.

        Args:
            source_code: The source code.
            pos: Character position.

        Returns:
            Tuple of (line_number, column_offset), both 1-based.
        """
        lines = source_code[:pos].split("\n")
        return len(lines), len(lines[-1]) if lines else 0

    def _extract_variable_annotations(
        self, source_code: str, protected_regions: list[tuple[int, int]]
    ) -> list[LuauTypeAnnotation]:
        """Extract variable type annotations from source code.

        Args:
            source_code: The source code to analyze.
            protected_regions: Regions to skip (strings/comments).

        Returns:
            List of extracted type annotations.
        """
        annotations = []

        # Pattern to find variable declarations with type annotations
        # Matches: local x: type = value or x: type = value
        pattern = re.compile(
            r'(local\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^=,\n\)]+?)\s*(?=[=,\n\)]|$)'
        )

        for match in pattern.finditer(source_code):
            if self._is_in_protected_region(match.start(), protected_regions):
                continue

            is_local = match.group(1) is not None
            var_name = match.group(2)
            type_str = match.group(3).strip()
            line_num, col = self._get_line_col(source_code, match.start())

            # Skip if this looks like a table field definition or function param
            # (those are handled separately)
            context_before = source_code[max(0, match.start() - 20):match.start()]
            if "{" in context_before and "}" not in context_before:
                continue  # Inside table constructor

            annotations.append(LuauTypeAnnotation(
                variable_name=var_name,
                type_string=type_str,
                line_number=line_num,
                column_offset=col,
                is_local=is_local,
                original_text=match.group(0),
            ))

        return annotations

    def _extract_function_types(
        self, source_code: str, protected_regions: list[tuple[int, int]]
    ) -> list[LuauFunctionType]:
        """Extract function type signatures from source code.

        Args:
            source_code: The source code to analyze.
            protected_regions: Regions to skip (strings/comments).

        Returns:
            List of extracted function type signatures.
        """
        function_types = []

        # Find function definitions
        func_pattern = re.compile(
            r'(local\s+)?function\s*([a-zA-Z_][a-zA-Z0-9_.:]*)?'
            r'\s*(?:<([^>]+)>)?\s*\(([^)]*)\)\s*(?::\s*([^\n{]+?))?(?=\s*[\n{])',
            re.MULTILINE
        )

        for match in func_pattern.finditer(source_code):
            if self._is_in_protected_region(match.start(), protected_regions):
                continue

            func_name = match.group(2) or ""
            generics_str = match.group(3) or ""
            params_str = match.group(4) or ""
            return_type_str = match.group(5) or ""

            line_num, col = self._get_line_col(source_code, match.start())

            # Parse parameters with types
            parameters = []
            if params_str.strip():
                # Match each parameter with optional type
                param_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::\s*([^,]+))?')
                for param_match in param_pattern.finditer(params_str):
                    param_name = param_match.group(1)
                    param_type = param_match.group(2).strip() if param_match.group(2) else ""
                    if param_type:
                        parameters.append((param_name, param_type))

            # Parse generics
            generics = []
            if generics_str:
                generics = [g.strip() for g in generics_str.split(",")]

            # Parse return types
            return_types = []
            if return_type_str.strip():
                # Handle multiple return types separated by comma
                return_types = [r.strip() for r in return_type_str.split(",")]

            if parameters or return_types or generics:
                function_types.append(LuauFunctionType(
                    function_name=func_name,
                    parameters=parameters,
                    return_types=return_types,
                    generics=generics,
                    line_number=line_num,
                    column_offset=col,
                    original_param_text=params_str,
                    original_return_text=return_type_str,
                ))

        return function_types

    def _extract_type_aliases(
        self, source_code: str, protected_regions: list[tuple[int, int]]
    ) -> list[LuauTypeAlias]:
        """Extract type alias definitions from source code.

        Args:
            source_code: The source code to analyze.
            protected_regions: Regions to skip (strings/comments).

        Returns:
            List of extracted type aliases.
        """
        aliases = []

        # Find both regular and exported type aliases
        for pattern, is_exported in [
            (TYPE_ALIAS_PATTERN, False),
            (EXPORT_TYPE_PATTERN, True),
        ]:
            for match in pattern.finditer(source_code):
                if self._is_in_protected_region(match.start(), protected_regions):
                    continue

                alias_name = match.group("name")
                type_def = match.group("typedef").strip()
                generics_str = match.group("generics") or ""
                line_num, _ = self._get_line_col(source_code, match.start())

                # Parse generics
                generics = []
                if generics_str:
                    generics = [g.strip() for g in generics_str.split(",")]

                aliases.append(LuauTypeAlias(
                    alias_name=alias_name,
                    type_definition=type_def,
                    generics=generics,
                    line_number=line_num,
                    is_exported=is_exported,
                    original_text=match.group(0),
                ))

        return aliases

    def _extract_type_casts(
        self, source_code: str, protected_regions: list[tuple[int, int]]
    ) -> list[LuauTypeCast]:
        """Extract type cast expressions from source code.

        Args:
            source_code: The source code to analyze.
            protected_regions: Regions to skip (strings/comments).

        Returns:
            List of extracted type casts.
        """
        casts = []

        for match in TYPE_CAST_PATTERN.finditer(source_code):
            if self._is_in_protected_region(match.start(), protected_regions):
                continue

            expr = match.group("expr")
            target_type = match.group("type").strip()
            line_num, col = self._get_line_col(source_code, match.start())

            casts.append(LuauTypeCast(
                expression=expr,
                target_type=target_type,
                line_number=line_num,
                column_offset=col,
                original_text=match.group(0),
            ))

        return casts

    def _extract_table_types(
        self, source_code: str, protected_regions: list[tuple[int, int]]
    ) -> list[LuauTableType]:
        """Extract table type definitions from source code.

        Args:
            source_code: The source code to analyze.
            protected_regions: Regions to skip (strings/comments).

        Returns:
            List of extracted table types.
        """
        table_types = []

        # This pattern matches table type syntax within type definitions
        # Example: { name: string, age: number }
        pattern = re.compile(
            r'\{\s*([a-zA-Z_][a-zA-Z0-9_]*\s*:\s*[^,}]+(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*[^,}]+)*)\s*\}'
        )

        for match in pattern.finditer(source_code):
            if self._is_in_protected_region(match.start(), protected_regions):
                continue

            fields_str = match.group(1)
            line_num, col = self._get_line_col(source_code, match.start())

            # Parse fields
            fields = {}
            field_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^,}]+)')
            for field_match in field_pattern.finditer(fields_str):
                field_name = field_match.group(1)
                field_type = field_match.group(2).strip()
                fields[field_name] = field_type

            if fields:
                table_types.append(LuauTableType(
                    fields=fields,
                    line_number=line_num,
                    column_offset=col,
                    original_text=match.group(0),
                ))

        return table_types

    def _strip_all_annotations(
        self,
        source_code: str,
        protected_regions: list[tuple[int, int]],
        metadata: LuauTypeMetadata,
    ) -> str:
        """Strip all Luau type annotations from source code.

        Args:
            source_code: The source code to process.
            protected_regions: Regions to skip (strings/comments).
            metadata: Metadata containing extracted type information.

        Returns:
            Source code with type annotations stripped.
        """
        # Replace protected regions with placeholders to avoid modifying strings/comments
        placeholders: list[tuple[str, str]] = []
        result = source_code

        # Sort protected regions by start position in reverse to maintain indices
        sorted_regions = sorted(protected_regions, key=lambda r: -r[0])
        for i, (start, end) in enumerate(sorted_regions):
            placeholder = f"\x00PROTECTED_{i}\x00"
            original_text = result[start:end]
            placeholders.append((placeholder, original_text))
            result = result[:start] + placeholder + result[end:]

        # Step 1: Remove type alias declarations (type Name = ...)
        # Process in reverse order by line number to maintain positions
        sorted_aliases = sorted(metadata.type_aliases, key=lambda a: -a.line_number)
        for alias in sorted_aliases:
            prefix = "export type" if alias.is_exported else "type"
            generics_part = f"<{', '.join(alias.generics)}>" if alias.generics else ""
            # Pattern to match the entire type alias line
            pattern = re.compile(
                rf'^(\s*){re.escape(prefix)}\s+{re.escape(alias.alias_name)}'
                rf'{re.escape(generics_part)}\s*=\s*{re.escape(alias.type_definition)}\s*$',
                re.MULTILINE
            )
            if self.debug_mode:
                result = pattern.sub(r'\1-- [LUAU TYPE STRIPPED]', result)
            else:
                result = pattern.sub('', result)

        # Step 2: Remove type casts (expr :: type -> expr)
        for cast in sorted(metadata.type_casts, key=lambda c: -c.column_offset):
            cast_pattern = re.compile(
                rf'{re.escape(cast.expression)}\s*::\s*{re.escape(cast.target_type)}'
            )
            result = cast_pattern.sub(cast.expression, result)

        # Step 3: Strip function return type annotations
        # Match ): type or ) -> type and replace with just )
        result = self._apply_protected_substitution(
            result,
            r'\)\s*(?::|->)\s*([a-zA-Z_][a-zA-Z0-9_<>,\s\|\&\?]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_<>,\s\|\&\?]*)*)(?=\s*[\n{])',
            ')',
        )

        # Step 4: Strip function parameter type annotations using state machine
        result = self._strip_function_params_with_state_machine(result)

        # Step 5: Strip generic type parameters from function definitions
        # Match function<T, U> and replace with function
        result = self._apply_protected_substitution(
            result,
            r'(function\s*)(<[^>]+>)',
            r'\1',
        )

        # Step 6: Strip variable type annotations
        # Handle local x: type = value (with assignment)
        result = self._apply_protected_substitution(
            result,
            r'(local\s+[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*[^=\n]+?(\s*=)',
            r'\1\2',
        )

        # Step 6b: Handle local x: type (without assignment - Comment 2)
        result = self._apply_protected_substitution(
            result,
            r'^(\s*local\s+[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*[^=\n,]+?(\s*)$',
            r'\1\2',
            flags=re.MULTILINE,
        )

        # Step 6c: Handle non-local (global) name: type = value (Comment 2)
        # Must be at start of line or after semicolon, and use word boundary
        result = self._apply_protected_substitution(
            result,
            r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*[^=\n]+?(\s*=)',
            r'\1\2\3',
            flags=re.MULTILINE,
        )

        # Step 7: Clean up any empty lines created by removing type aliases
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)

        # Restore protected regions from placeholders (in reverse order of insertion)
        for placeholder, original_text in placeholders:
            result = result.replace(placeholder, original_text)

        return result

    def _apply_protected_substitution(
        self,
        text: str,
        pattern: str,
        replacement: str,
        flags: int = 0,
    ) -> str:
        """Apply regex substitution only outside of placeholder-protected regions.

        Args:
            text: The text to process (may contain placeholders).
            pattern: The regex pattern to match.
            replacement: The replacement string.
            flags: Optional regex flags.

        Returns:
            Text with substitutions applied outside placeholders.
        """
        # Placeholders contain \x00, skip any match that overlaps with them
        compiled = re.compile(pattern, flags)

        def safe_replace(match: re.Match) -> str:
            matched_text = match.group(0)
            # If the match contains a placeholder marker, don't modify it
            if '\x00' in matched_text:
                return matched_text
            # Otherwise apply the replacement
            return match.expand(replacement)

        return compiled.sub(safe_replace, text)

    def _strip_function_params_with_state_machine(self, source: str) -> str:
        """Strip type annotations from function parameters using a state machine.

        This handles:
        - Generic types with commas (e.g., Map<string, number>)
        - Nested generics (e.g., Array<Map<K, V>>)
        - Multiline parameter lists
        - Table types in parameters (e.g., {x: number, y: number})

        Args:
            source: The source code to process.

        Returns:
            Source code with parameter types stripped.
        """
        result = []
        i = 0
        n = len(source)

        while i < n:
            # Skip placeholders
            if source[i] == '\x00':
                # Find the end of the placeholder
                end = source.find('\x00', i + 1)
                if end != -1:
                    result.append(source[i:end + 1])
                    i = end + 1
                    continue
                else:
                    result.append(source[i])
                    i += 1
                    continue

            # Look for function parameter lists
            if source[i] == '(' and i > 0:
                # Check if this could be a function parameter list
                # (not a function call - look for 'function' keyword before)
                before = source[max(0, i - 50):i]
                if re.search(r'function\s*(?:<[^>]*>)?\s*$', before) or \
                   re.search(r'function\s+[a-zA-Z_][a-zA-Z0-9_.:]*\s*(?:<[^>]*>)?\s*$', before):
                    # Parse the parameter list with nesting awareness
                    params_start = i
                    params_content, params_end = self._parse_nested_parens(source, i)

                    if params_content is not None:
                        # Strip types from parameters
                        stripped_params = self._strip_params_from_content(params_content)
                        result.append(f'({stripped_params})')
                        i = params_end + 1
                        continue

            result.append(source[i])
            i += 1

        return ''.join(result)

    def _parse_nested_parens(self, source: str, start: int) -> tuple[str | None, int]:
        """Parse content inside parentheses, handling nested brackets.

        Args:
            source: The source code.
            start: Index of the opening parenthesis.

        Returns:
            Tuple of (content inside parens, index of closing paren) or (None, start) if not found.
        """
        if start >= len(source) or source[start] != '(':
            return None, start

        depth_paren = 1
        depth_angle = 0
        depth_brace = 0
        i = start + 1
        n = len(source)

        while i < n and depth_paren > 0:
            c = source[i]

            # Skip placeholders
            if c == '\x00':
                end = source.find('\x00', i + 1)
                if end != -1:
                    i = end + 1
                    continue

            # Handle string literals (simple check)
            if c in '"\'':
                quote = c
                i += 1
                while i < n and source[i] != quote:
                    if source[i] == '\\' and i + 1 < n:
                        i += 2
                    else:
                        i += 1
                i += 1
                continue

            # Track nesting
            if c == '(':
                depth_paren += 1
            elif c == ')':
                depth_paren -= 1
            elif c == '<':
                depth_angle += 1
            elif c == '>':
                depth_angle -= 1
            elif c == '{':
                depth_brace += 1
            elif c == '}':
                depth_brace -= 1

            i += 1

        if depth_paren == 0:
            return source[start + 1:i - 1], i - 1
        return None, start

    def _strip_params_from_content(self, params_content: str) -> str:
        """Strip type annotations from parameter content.

        Handles nested generics and table types when splitting on commas.

        Args:
            params_content: The content inside function parentheses.

        Returns:
            Parameters with type annotations stripped.
        """
        if not params_content.strip():
            return params_content

        # Split parameters respecting nesting
        params = self._split_params_respecting_nesting(params_content)
        stripped_params = []

        for param in params:
            param = param.strip()
            if not param:
                stripped_params.append(param)
                continue

            # Handle variadics: ...: type -> ...
            if param.startswith('...'):
                if ':' in param:
                    stripped_params.append('...')
                else:
                    stripped_params.append(param)
                continue

            # Handle regular parameters: name: type -> name
            colon_idx = self._find_type_colon(param)
            if colon_idx != -1:
                stripped_params.append(param[:colon_idx].strip())
            else:
                stripped_params.append(param)

        return ', '.join(stripped_params)

    def _split_params_respecting_nesting(self, content: str) -> list[str]:
        """Split parameter string on commas, respecting nested brackets.

        Args:
            content: The parameter content to split.

        Returns:
            List of individual parameters.
        """
        params = []
        current = []
        depth_angle = 0
        depth_brace = 0
        depth_paren = 0
        i = 0
        n = len(content)

        while i < n:
            c = content[i]

            # Skip placeholders
            if c == '\x00':
                end = content.find('\x00', i + 1)
                if end != -1:
                    current.append(content[i:end + 1])
                    i = end + 1
                    continue

            # Handle string literals
            if c in '"\'':
                quote = c
                current.append(c)
                i += 1
                while i < n and content[i] != quote:
                    if content[i] == '\\' and i + 1 < n:
                        current.append(content[i:i + 2])
                        i += 2
                    else:
                        current.append(content[i])
                        i += 1
                if i < n:
                    current.append(content[i])
                    i += 1
                continue

            # Track nesting
            if c == '<':
                depth_angle += 1
            elif c == '>':
                depth_angle -= 1
            elif c == '{':
                depth_brace += 1
            elif c == '}':
                depth_brace -= 1
            elif c == '(':
                depth_paren += 1
            elif c == ')':
                depth_paren -= 1
            elif c == ',' and depth_angle == 0 and depth_brace == 0 and depth_paren == 0:
                params.append(''.join(current))
                current = []
                i += 1
                continue

            current.append(c)
            i += 1

        if current:
            params.append(''.join(current))

        return params

    def _find_type_colon(self, param: str) -> int:
        """Find the colon that separates parameter name from type.

        Ignores colons inside nested brackets.

        Args:
            param: A single parameter string.

        Returns:
            Index of the type colon, or -1 if not found.
        """
        depth_angle = 0
        depth_brace = 0
        depth_paren = 0
        i = 0
        n = len(param)

        while i < n:
            c = param[i]

            if c == '<':
                depth_angle += 1
            elif c == '>':
                depth_angle -= 1
            elif c == '{':
                depth_brace += 1
            elif c == '}':
                depth_brace -= 1
            elif c == '(':
                depth_paren += 1
            elif c == ')':
                depth_paren -= 1
            elif c == ':' and depth_angle == 0 and depth_brace == 0 and depth_paren == 0:
                return i

            i += 1

        return -1


# =============================================================================
# Luau AST Node Extensions (Step 3)
# =============================================================================

# Attribute name for storing Luau metadata on AST nodes
LUAU_METADATA_ATTR = "_luau_type_metadata"


def attach_luau_metadata(ast_node, metadata: LuauTypeMetadata) -> None:
    """Attach Luau metadata to an AST node.

    Args:
        ast_node: The luaparser AST node to attach metadata to.
        metadata: The Luau type metadata to attach.

    Example:
        >>> ast_node = luaparser.ast.parse(stripped_code)
        >>> attach_luau_metadata(ast_node, metadata)
    """
    setattr(ast_node, LUAU_METADATA_ATTR, metadata)
    logger.debug(f"Attached Luau metadata to AST node of type {type(ast_node).__name__}")


def get_luau_metadata(ast_node) -> Optional[LuauTypeMetadata]:
    """Retrieve Luau metadata from an AST node.

    Args:
        ast_node: The luaparser AST node to retrieve metadata from.

    Returns:
        The Luau type metadata if present, None otherwise.

    Example:
        >>> metadata = get_luau_metadata(ast_node)
        >>> if metadata:
        ...     print(f"Found {len(metadata.annotations)} annotations")
    """
    return getattr(ast_node, LUAU_METADATA_ATTR, None)


def has_luau_metadata(ast_node) -> bool:
    """Check if an AST node has Luau metadata attached.

    Args:
        ast_node: The luaparser AST node to check.

    Returns:
        True if the node has Luau metadata attached.
    """
    return hasattr(ast_node, LUAU_METADATA_ATTR)


# =============================================================================
# Luau Code Generator (Step 4)
# =============================================================================

class LuauCodeGenerator:
    """Generator for restoring Luau type annotations to generated Lua code.

    This class takes generated Lua code and metadata containing extracted type
    information, and restores the Luau-specific syntax.

    Example:
        >>> generator = LuauCodeGenerator()
        >>> luau_code = generator.restore_type_annotations(lua_code, metadata)
    """

    def __init__(self, debug_mode: bool = False) -> None:
        """Initialize the Luau code generator.

        Args:
            debug_mode: If True, adds comments indicating where annotations were restored.
        """
        self.debug_mode = debug_mode
        self.logger = get_logger("obfuscator.processors.luau_extensions")

    def restore_type_annotations(
        self, lua_code: str, metadata: LuauTypeMetadata
    ) -> str:
        """Restore Luau type annotations to generated Lua code.

        Args:
            lua_code: The generated Lua code without type annotations.
            metadata: The metadata containing extracted type information.

        Returns:
            Lua code with Luau type annotations restored.
        """
        self.logger.info("Restoring Luau type annotations to generated code")

        result = lua_code

        # Restore in order: type aliases, variable annotations, function types, type casts
        result = self._restore_type_aliases(result, metadata.type_aliases)
        result = self._restore_variable_annotations(result, metadata.annotations)
        result = self._restore_function_types(result, metadata.function_types)
        result = self._restore_type_casts(result, metadata.type_casts)

        self.logger.debug("Completed restoring Luau type annotations")

        return result

    def _restore_type_aliases(self, code: str, aliases: list[LuauTypeAlias]) -> str:
        """Insert type alias declarations into the code.

        Args:
            code: The generated Lua code.
            aliases: List of type aliases to restore.

        Returns:
            Code with type aliases restored.
        """
        if not aliases:
            return code

        # Sort aliases by line number to insert in correct order
        sorted_aliases = sorted(aliases, key=lambda a: a.line_number)

        lines = code.split("\n")
        inserted_count = 0

        for alias in sorted_aliases:
            # Build the type alias declaration
            prefix = "export type" if alias.is_exported else "type"
            generics_part = f"<{', '.join(alias.generics)}>" if alias.generics else ""
            declaration = f"{prefix} {alias.alias_name}{generics_part} = {alias.type_definition}"

            # Insert at the appropriate line (adjusted for previous insertions)
            insert_line = min(alias.line_number - 1 + inserted_count, len(lines))
            lines.insert(insert_line, declaration)
            inserted_count += 1

            if self.debug_mode:
                self.logger.debug(f"Restored type alias '{alias.alias_name}' at line {insert_line + 1}")

        return "\n".join(lines)

    def _restore_variable_annotations(
        self, code: str, annotations: list[LuauTypeAnnotation]
    ) -> str:
        """Insert variable type annotations into the code.

        Uses stored line_number/column_offset to target original declaration sites.
        Uses word boundaries and anchors to avoid matching comparison operators
        (==, ~=, <=, >=).

        Args:
            code: The generated Lua code.
            annotations: List of variable annotations to restore.

        Returns:
            Code with variable type annotations restored.
        """
        if not annotations:
            return code

        # Work with lines to use line_number for targeting
        lines = code.split('\n')

        # Sort annotations by line number (descending) to avoid offset issues
        sorted_annotations = sorted(annotations, key=lambda a: -a.line_number)

        for annotation in sorted_annotations:
            line_idx = annotation.line_number - 1

            # Ensure line index is valid
            if line_idx < 0 or line_idx >= len(lines):
                # Fall back to searching entire code if line number is out of range
                self._restore_annotation_fallback(lines, annotation)
                continue

            line = lines[line_idx]

            # Build pattern that matches assignment but not comparisons
            # Use word boundary \b and negative lookbehind/lookahead for comparison ops
            var_name = re.escape(annotation.variable_name)

            if annotation.is_local:
                # Match: local varname = (but not local varname ==, ~=, <=, >=)
                # Anchor to start of line allowing leading whitespace
                pattern = rf'^(\s*local\s+{var_name})(\s*)(=)(?!=|~|<|>)'
                replacement = rf'\1: {annotation.type_string}\2\3'
            else:
                # Match: varname = at start of line (but not ==, ~=, <=, >=)
                # Anchor to start of line allowing leading whitespace
                pattern = rf'^(\s*)({var_name})(\s*)(=)(?!=|~|<|>)'
                replacement = rf'\1\2: {annotation.type_string}\3\4'

            new_line, count = re.subn(pattern, replacement, line, count=1)

            if count > 0:
                lines[line_idx] = new_line
                if self.debug_mode:
                    self.logger.debug(
                        f"Restored type annotation '{annotation.type_string}' "
                        f"for variable '{annotation.variable_name}' at line {annotation.line_number}"
                    )
            else:
                # Try fallback if exact line match failed
                self._restore_annotation_fallback(lines, annotation)

        return '\n'.join(lines)

    def _restore_annotation_fallback(
        self, lines: list[str], annotation: LuauTypeAnnotation
    ) -> None:
        """Fallback method to restore annotation when line-based matching fails.

        Searches for the first valid declaration site in the code.

        Args:
            lines: List of code lines (modified in place).
            annotation: The annotation to restore.
        """
        var_name = re.escape(annotation.variable_name)

        for i, line in enumerate(lines):
            if annotation.is_local:
                # Match local declaration, avoiding comparisons
                pattern = rf'^(\s*local\s+{var_name})(\s*)(=)(?!=|~|<|>)'
                replacement = rf'\1: {annotation.type_string}\2\3'
            else:
                # Match global assignment at line start, avoiding comparisons
                pattern = rf'^(\s*)({var_name})(\s*)(=)(?!=|~|<|>)'
                replacement = rf'\1\2: {annotation.type_string}\3\4'

            new_line, count = re.subn(pattern, replacement, line, count=1)

            if count > 0:
                lines[i] = new_line
                if self.debug_mode:
                    self.logger.debug(
                        f"Restored type annotation '{annotation.type_string}' "
                        f"for variable '{annotation.variable_name}' at line {i + 1} (fallback)"
                    )
                return

    def _restore_function_types(
        self, code: str, function_types: list[LuauFunctionType]
    ) -> str:
        """Insert function type signatures into the code.

        Args:
            code: The generated Lua code.
            function_types: List of function types to restore.

        Returns:
            Code with function type signatures restored.
        """
        if not function_types:
            return code

        result = code

        for func_type in function_types:
            if not func_type.parameters and not func_type.return_types and not func_type.generics:
                continue

            # Build pattern to find the function definition
            func_name_pattern = re.escape(func_type.function_name) if func_type.function_name else ""

            # Restore generics
            if func_type.generics:
                generics_str = f"<{', '.join(func_type.generics)}>"
                if func_type.function_name:
                    pattern = rf'(function\s+{func_name_pattern})\s*\('
                    replacement = rf'\1{generics_str}('
                else:
                    pattern = r'(function)\s*\('
                    replacement = rf'\1{generics_str}('
                result = re.sub(pattern, replacement, result, count=1)

            # Restore parameter types
            if func_type.parameters:
                # Find the function and its parameters
                if func_type.function_name:
                    func_pattern = rf'function\s+{func_name_pattern}(?:<[^>]+>)?\s*\(([^)]*)\)'
                else:
                    func_pattern = r'function(?:<[^>]+>)?\s*\(([^)]*)\)'

                def add_param_types(match):
                    params_str = match.group(1)
                    params = [p.strip() for p in params_str.split(",") if p.strip()]
                    typed_params = []
                    for param in params:
                        # Find type for this parameter
                        param_type = None
                        for pname, ptype in func_type.parameters:
                            if pname == param:
                                param_type = ptype
                                break
                        if param_type:
                            typed_params.append(f"{param}: {param_type}")
                        else:
                            typed_params.append(param)
                    return match.group(0).replace(params_str, ", ".join(typed_params))

                result = re.sub(func_pattern, add_param_types, result, count=1)

            # Restore return types
            if func_type.return_types:
                return_str = ", ".join(func_type.return_types)
                if func_type.function_name:
                    pattern = rf'(function\s+{func_name_pattern}(?:<[^>]+>)?\s*\([^)]*\))\s*'
                    replacement = rf'\1: {return_str} '
                else:
                    pattern = r'(function(?:<[^>]+>)?\s*\([^)]*\))\s*'
                    replacement = rf'\1: {return_str} '
                result = re.sub(pattern, replacement, result, count=1)

            if self.debug_mode:
                self.logger.debug(
                    f"Restored function types for '{func_type.function_name or 'anonymous'}'"
                )

        return result

    def _restore_type_casts(self, code: str, casts: list[LuauTypeCast]) -> str:
        """Insert type cast expressions into the code.

        Args:
            code: The generated Lua code.
            casts: List of type casts to restore.

        Returns:
            Code with type casts restored.
        """
        if not casts:
            return code

        result = code

        for cast in casts:
            # Find the expression and add the type cast
            pattern = rf'(?<![:\w])({re.escape(cast.expression)})(?![:\w])'
            replacement = rf'\1 :: {cast.target_type}'
            result = re.sub(pattern, replacement, result, count=1)

            if self.debug_mode:
                self.logger.debug(
                    f"Restored type cast '{cast.expression} :: {cast.target_type}'"
                )

        return result

    def _restore_table_types(self, code: str, table_types: list[LuauTableType]) -> str:
        """Insert table type definitions into the code.

        Note: Table types are typically part of type aliases or variable annotations,
        so this method mainly handles standalone table type syntax restoration.

        Args:
            code: The generated Lua code.
            table_types: List of table types to restore.

        Returns:
            Code with table types restored.
        """
        # Table types are usually embedded in type aliases and handled there
        # This method handles any edge cases where table types need direct restoration
        return code


# =============================================================================
# LuaJIT Syntax Detection (Step 5)
# =============================================================================

class LuaJITDetector:
    """Detector for LuaJIT-specific syntax patterns.

    This class detects LuaJIT-specific features in Lua source code and returns
    warnings for features that may not be compatible with standard Lua parsers.

    Example:
        >>> detector = LuaJITDetector()
        >>> features = detector.detect_luajit_features(source_code)
        >>> for feature in features:
        ...     print(f"Detected: {feature}")
    """

    # LuaJIT FFI patterns
    FFI_REQUIRE_PATTERN = re.compile(r'require\s*\(\s*["\']ffi["\']\s*\)')
    FFI_CDEF_PATTERN = re.compile(r'ffi\.cdef')
    FFI_NEW_PATTERN = re.compile(r'ffi\.new')
    FFI_CAST_PATTERN = re.compile(r'ffi\.cast')
    FFI_TYPEOF_PATTERN = re.compile(r'ffi\.typeof')

    # LuaJIT integer literal patterns (64-bit integers)
    INTEGER_LL_PATTERN = re.compile(r'\b(\d+)[lL][lL]\b')
    INTEGER_ULL_PATTERN = re.compile(r'\b(\d+)[uU][lL][lL]\b')

    # LuaJIT BitOp patterns
    BITOP_PATTERN = re.compile(r'\bbit\.(band|bor|bxor|bnot|lshift|rshift|arshift|rol|ror|bswap|tobit|tohex)\b')

    # LuaJIT JIT pragma patterns
    JIT_ON_PATTERN = re.compile(r'\bjit\.on\s*\(')
    JIT_OFF_PATTERN = re.compile(r'\bjit\.off\s*\(')
    JIT_FLUSH_PATTERN = re.compile(r'\bjit\.flush\s*\(')
    JIT_STATUS_PATTERN = re.compile(r'\bjit\.status\s*\(')
    JIT_OPT_PATTERN = re.compile(r'\bjit\.opt\.')

    def __init__(self) -> None:
        """Initialize the LuaJIT detector."""
        self.logger = get_logger("obfuscator.processors.luau_extensions")

    def detect_luajit_features(self, source_code: str) -> list[str]:
        """Detect LuaJIT-specific syntax patterns in source code.

        Args:
            source_code: The Lua source code to analyze.

        Returns:
            List of detected LuaJIT feature names.
        """
        features = []

        if self._detect_ffi_usage(source_code):
            features.append("ffi")
            self.logger.info("Detected LuaJIT FFI usage")

        integer_literals = self._detect_integer_literals(source_code)
        if integer_literals:
            features.append("64bit_integers")
            self.logger.info(f"Detected {len(integer_literals)} LuaJIT 64-bit integer literals")

        if self._detect_bitop_usage(source_code):
            features.append("bitop")
            self.logger.info("Detected LuaJIT BitOp library usage")

        jit_pragmas = self._detect_jit_pragmas(source_code)
        if jit_pragmas:
            features.append("jit_pragmas")
            self.logger.info(f"Detected LuaJIT pragmas: {', '.join(jit_pragmas)}")

        return features

    def _detect_ffi_usage(self, source_code: str) -> bool:
        """Check for FFI usage in source code.

        Args:
            source_code: The source code to check.

        Returns:
            True if FFI usage is detected.
        """
        patterns = [
            self.FFI_REQUIRE_PATTERN,
            self.FFI_CDEF_PATTERN,
            self.FFI_NEW_PATTERN,
            self.FFI_CAST_PATTERN,
            self.FFI_TYPEOF_PATTERN,
        ]
        return any(pattern.search(source_code) for pattern in patterns)

    def _detect_integer_literals(self, source_code: str) -> list[tuple[int, str]]:
        """Find integer literals with ll or ull suffixes.

        Args:
            source_code: The source code to check.

        Returns:
            List of (line_number, literal_text) tuples.
        """
        literals = []

        for pattern in [self.INTEGER_LL_PATTERN, self.INTEGER_ULL_PATTERN]:
            for match in pattern.finditer(source_code):
                line_num = source_code[:match.start()].count("\n") + 1
                literals.append((line_num, match.group(0)))

        return literals

    def _detect_bitop_usage(self, source_code: str) -> bool:
        """Check for BitOp library usage in source code.

        Args:
            source_code: The source code to check.

        Returns:
            True if BitOp usage is detected.
        """
        return bool(self.BITOP_PATTERN.search(source_code))

    def _detect_jit_pragmas(self, source_code: str) -> list[str]:
        """Find JIT compiler pragmas in source code.

        Args:
            source_code: The source code to check.

        Returns:
            List of detected pragma names (e.g., ["jit.on", "jit.off"]).
        """
        pragmas = []

        if self.JIT_ON_PATTERN.search(source_code):
            pragmas.append("jit.on")
        if self.JIT_OFF_PATTERN.search(source_code):
            pragmas.append("jit.off")
        if self.JIT_FLUSH_PATTERN.search(source_code):
            pragmas.append("jit.flush")
        if self.JIT_STATUS_PATTERN.search(source_code):
            pragmas.append("jit.status")
        if self.JIT_OPT_PATTERN.search(source_code):
            pragmas.append("jit.opt")

        return pragmas


# =============================================================================
# Luau Syntax Validation (Step 9)
# =============================================================================

def validate_luau_syntax(source_code: str) -> tuple[bool, list[str]]:
    """Validate Luau-specific syntax before preprocessing.

    This function checks for common Luau syntax errors and unsupported features
    before attempting to strip type annotations.

    Args:
        source_code: The Luau source code to validate.

    Returns:
        Tuple of (is_valid, error_messages). If valid, error_messages is empty.

    Example:
        >>> is_valid, errors = validate_luau_syntax(source_code)
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(f"Validation error: {error}")
    """
    errors = []

    # Check for balanced angle brackets in generic types
    lines = source_code.split("\n")
    for line_num, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith("--"):
            continue

        # Check for type annotations with unbalanced brackets
        open_angles = line.count("<")
        close_angles = line.count(">")

        # Simple heuristic: if we have more opens than closes on a single line
        # that contains type-like syntax, it might be an error
        if ":" in line or "type " in line:
            if open_angles > close_angles + 1:  # Allow for comparison operators
                errors.append(f"Line {line_num}: Possibly unclosed generic type bracket")

    # Check for invalid type alias syntax
    type_alias_pattern = re.compile(r'^(\s*)(export\s+)?type\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<[^>]*>)?\s*(?!=)', re.MULTILINE)
    for match in type_alias_pattern.finditer(source_code):
        line_num = source_code[:match.start()].count("\n") + 1
        errors.append(f"Line {line_num}: Type alias '{match.group(3)}' missing '=' and type definition")

    # Check for unsupported Luau features
    # Generic constraints (not fully supported)
    if re.search(r'<\s*[A-Z]\s*:\s*[a-zA-Z]', source_code):
        line_num = source_code[:re.search(r'<\s*[A-Z]\s*:\s*[a-zA-Z]', source_code).start()].count("\n") + 1
        errors.append(f"Line {line_num}: Generic constraints are not fully supported")

    # Log validation results
    if errors:
        logger.error(f"Luau syntax validation found {len(errors)} error(s)")
        for error in errors:
            logger.error(f"  {error}")

    return (len(errors) == 0, errors)

