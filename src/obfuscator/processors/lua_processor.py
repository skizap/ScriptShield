"""Lua code processor for parsing and generating Lua source code.

This module provides functionality to parse Lua source files into Abstract Syntax Trees (AST)
and generate Lua code from AST nodes. It uses the luaparser library for parsing and code
generation, supporting Lua 5.1, 5.2, 5.3, 5.4, Luau, and LuaJIT.

Luau Support:
    When enable_luau=True is passed to the LuaProcessor constructor, the processor will:
    - Strip Luau type annotations before parsing (allowing luaparser to process the code)
    - Store type information in metadata attached to the AST
    - Restore type annotations during code generation

Example:
    >>> from pathlib import Path
    >>> processor = LuaProcessor()
    >>> result = processor.parse_file(Path("script.lua"))
    >>> if result.success:
    ...     print(f"Parsed {len(result.ast_node.body)} statements")
    ...     gen_result = processor.generate_code(result.ast_node)
    ...     if gen_result.success:
    ...         print(gen_result.code)

    # With Luau support:
    >>> processor = LuaProcessor(enable_luau=True)
    >>> result = processor.parse_file(Path("script.luau"))
    >>> # Luau type annotations are preserved in metadata
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import luaparser.ast
import luaparser.astnodes

from obfuscator.utils.logger import get_logger

if TYPE_CHECKING:
    from obfuscator.core.config import ObfuscationConfig
    from obfuscator.core.symbol_table import GlobalSymbolTable
from obfuscator.processors.luau_extensions import (
    LuauCodeGenerator,
    LuaJITDetector,
    LuauPreprocessor,
    LuauTypeMetadata,
    attach_luau_metadata,
    get_luau_metadata,
    validate_luau_syntax,
)
from obfuscator.processors.lua_symbol_extractor import (
    LuaSymbolExtractor,
    LuaSymbolTable,
)

# Module constants
SUPPORTED_LUA_VERSIONS: tuple[str, ...] = ("5.1", "5.2", "5.3", "5.4", "Luau", "LuaJIT")
MAX_FILE_SIZE_MB: int = 10

# Initialize logger
logger = get_logger("obfuscator.processors.lua_processor")


@dataclass
class ParseResult:
    """Result of parsing a Lua source file.

    Attributes:
        ast_node: The parsed Lua AST root node (Chunk), or None if parsing failed.
        source_code: Original Lua source code that was parsed.
        file_path: Path to the Lua source file.
        success: Whether parsing succeeded.
        errors: List of error messages encountered during parsing.
        warnings: List of feature warnings (reserved for future integration).
        luau_metadata: Luau type metadata if Luau preprocessing was applied, None otherwise.
        luau_enabled: Whether Luau preprocessing was applied to this parse.
        luajit_features: List of detected LuaJIT feature names for warnings.

    Example:
        >>> result = processor.parse_file("script.lua")
        >>> if result.success:
        ...     print(f"Parsed {result.file_path}")
        ...     print(f"AST type: {type(result.ast_node).__name__}")
        ... else:
        ...     for error in result.errors:
        ...         print(f"Error: {error}")

        # With Luau support:
        >>> if result.luau_enabled and result.luau_metadata:
        ...     print(f"Extracted {len(result.luau_metadata.annotations)} type annotations")
    """

    ast_node: luaparser.astnodes.Chunk | None
    source_code: str
    file_path: Path
    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[Any] = field(default_factory=list)  # For future FeatureWarning integration
    luau_metadata: Optional[LuauTypeMetadata] = None
    luau_enabled: bool = False
    luajit_features: list[str] = field(default_factory=list)


@dataclass
class GenerateResult:
    """Result of generating Lua code from an AST.

    Attributes:
        code: Generated Lua source code.
        success: Whether code generation succeeded.
        errors: List of error messages encountered during generation.
        warnings: List of warning messages encountered during generation.
        metadata: Additional metadata about the generation process.

    Example:
        >>> result = processor.generate_code(ast_node)
        >>> if result.success:
        ...     print(result.code)
        ... else:
        ...     for error in result.errors:
        ...         print(f"Error: {error}")
    """

    code: str
    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class LuaProcessor:
    """Processor for parsing and generating Lua source code.

    This class provides methods to parse Lua source files into AST nodes and generate
    Lua code from AST nodes. It handles file validation, error reporting, and supports
    multiple Lua versions (5.1, 5.2, 5.3, 5.4, Luau, LuaJIT).

    Attributes:
        enable_luau: If True, enables Luau type annotation preprocessing.
        debug_mode: If True, enables debug output for Luau processing.

    Example:
        >>> processor = LuaProcessor()
        >>> parse_result = processor.parse_file("script.lua")
        >>> if parse_result.success:
        ...     # Modify AST here if needed
        ...     gen_result = processor.generate_code(parse_result.ast_node)
        ...     if gen_result.success:
        ...         print(gen_result.code)

        # With Luau support:
        >>> processor = LuaProcessor(enable_luau=True)
        >>> parse_result = processor.parse_file("script.luau")
    """

    def __init__(
        self,
        enable_luau: bool = False,
        debug_mode: bool = False,
        config: "ObfuscationConfig | None" = None,
    ) -> None:
        """Initialize the Lua processor.

        Args:
            enable_luau: If True, enables Luau type annotation preprocessing.
                         Type annotations will be stripped before parsing and
                         restored during code generation.
            debug_mode: If True, enables debug output for Luau processing,
                        including comments indicating stripped/restored annotations.
            config: Optional ObfuscationConfig instance. When provided,
                    the engine will apply additional transformations
                    (string encryption, number obfuscation, etc.) after
                    name mangling in ``obfuscate_with_symbol_table()``.
        """
        self.logger = get_logger("obfuscator.processors.lua_processor")
        self.enable_luau = enable_luau
        self.debug_mode = debug_mode
        self._config = config
        self._luau_preprocessor: Optional[LuauPreprocessor] = None
        self._luau_generator: Optional[LuauCodeGenerator] = None
        self._luajit_detector: Optional[LuaJITDetector] = None

        if enable_luau:
            self._luau_preprocessor = LuauPreprocessor(debug_mode=debug_mode)
            self._luau_generator = LuauCodeGenerator(debug_mode=debug_mode)
            self.logger.info("LuaProcessor initialized with Luau support enabled")
        else:
            self.logger.debug("LuaProcessor initialized")

    def parse_file(self, file_path: Path | str) -> ParseResult:
        """Parse a Lua source file into an AST.

        Validates the file (existence, readability, size), reads the source code,
        and parses it into a Lua AST using luaparser.

        Args:
            file_path: Path to the Lua source file to parse.

        Returns:
            ParseResult containing the AST node, source code, and any errors.

        Example:
            >>> result = processor.parse_file("script.lua")
            >>> if result.success:
            ...     print(f"Parsed {len(result.ast_node.body)} statements")
            ... else:
            ...     print(f"Parse failed: {result.errors}")

        Note:
            Files larger than MAX_FILE_SIZE_MB (10 MB) will be rejected for safety.
        """
        # Convert to Path object
        path = Path(file_path)

        # Validate file exists
        if not path.exists():
            error_msg = f"File not found: {path}"
            self.logger.error(error_msg)
            return ParseResult(
                ast_node=None,
                source_code="",
                file_path=path,
                success=False,
                errors=[error_msg],
            )

        # Check file is readable
        if not os.access(path, os.R_OK):
            error_msg = f"File is not readable: {path}"
            self.logger.error(error_msg)
            return ParseResult(
                ast_node=None,
                source_code="",
                file_path=path,
                success=False,
                errors=[error_msg],
            )

        # Check file size
        try:
            file_size = path.stat().st_size
            max_size = MAX_FILE_SIZE_MB * 1024 * 1024
            if file_size > max_size:
                error_msg = (
                    f"File too large: {path} ({file_size / 1024 / 1024:.2f} MB). "
                    f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB"
                )
                self.logger.error(error_msg)
                return ParseResult(
                    ast_node=None,
                    source_code="",
                    file_path=path,
                    success=False,
                    errors=[error_msg],
                )
        except OSError as e:
            error_msg = f"Failed to check file size for {path}: {e}"
            self.logger.error(error_msg)
            return ParseResult(
                ast_node=None,
                source_code="",
                file_path=path,
                success=False,
                errors=[error_msg],
            )

        # Read source code
        try:
            source_code = path.read_text(encoding="utf-8")
        except OSError as e:
            error_msg = f"Failed to read file {path}: {e}"
            self.logger.error(error_msg)
            return ParseResult(
                ast_node=None,
                source_code="",
                file_path=path,
                success=False,
                errors=[error_msg],
            )
        except ValueError as e:
            error_msg = f"Encoding error reading file {path}: {e}"
            self.logger.error(error_msg)
            return ParseResult(
                ast_node=None,
                source_code="",
                file_path=path,
                success=False,
                errors=[error_msg],
            )

        # Initialize variables for Luau support
        luau_metadata: Optional[LuauTypeMetadata] = None
        luajit_features: list[str] = []
        code_to_parse = source_code

        # Apply Luau preprocessing if enabled
        if self.enable_luau and self._luau_preprocessor:
            # Validate Luau syntax first
            is_valid, validation_errors = validate_luau_syntax(source_code)
            if not is_valid:
                self.logger.warning(
                    f"Luau syntax validation warnings for {path}: {validation_errors}"
                )
                # Continue processing despite validation warnings

            # Strip type annotations
            code_to_parse, luau_metadata = self._luau_preprocessor.strip_type_annotations(
                source_code
            )
            self.logger.debug(
                f"Luau preprocessing: extracted {len(luau_metadata.annotations)} annotations, "
                f"{len(luau_metadata.function_types)} function types, "
                f"{len(luau_metadata.type_aliases)} type aliases"
            )

        # Detect LuaJIT features (always, for warnings)
        if self._luajit_detector is None:
            self._luajit_detector = LuaJITDetector()
        luajit_features = self._luajit_detector.detect_luajit_features(source_code)
        if luajit_features:
            self.logger.info(f"Detected LuaJIT features in {path}: {luajit_features}")

        # Parse Lua code
        try:
            ast_node = luaparser.ast.parse(code_to_parse)
            self.logger.info(f"Successfully parsed Lua file: {path}")
            self.logger.debug(f"AST root type: {type(ast_node).__name__}")

            # Attach Luau metadata to AST if available
            if luau_metadata:
                attach_luau_metadata(ast_node, luau_metadata)

            return ParseResult(
                ast_node=ast_node,
                source_code=source_code,
                file_path=path,
                success=True,
                errors=[],
                luau_metadata=luau_metadata,
                luau_enabled=self.enable_luau,
                luajit_features=luajit_features,
            )
        except SyntaxError as e:
            # Format error message with line/column if available
            if hasattr(e, "lineno") and hasattr(e, "offset"):
                error_msg = f"Syntax error in {path} at line {e.lineno}, column {e.offset}: {e.msg}"
            else:
                error_msg = f"Syntax error in {path}: {e}"
            self.logger.error(error_msg)
            return ParseResult(
                ast_node=None,
                source_code=source_code,
                file_path=path,
                success=False,
                errors=[error_msg],
                luau_metadata=luau_metadata,
                luau_enabled=self.enable_luau,
                luajit_features=luajit_features,
            )
        except Exception as e:
            error_msg = f"Unexpected error parsing {path}: {e}"
            self.logger.error(error_msg, exc_info=True)
            return ParseResult(
                ast_node=None,
                source_code=source_code,
                file_path=path,
                success=False,
                errors=[error_msg],
                luau_metadata=luau_metadata,
                luau_enabled=self.enable_luau,
                luajit_features=luajit_features,
            )

    def generate_code(
        self,
        ast_node: luaparser.astnodes.Node,
        restore_luau_types: bool = True,
    ) -> GenerateResult:
        """Generate Lua source code from an AST node.

        Converts a Lua AST node back into source code using luaparser's code generator.
        If Luau metadata is attached to the AST node and restore_luau_types is True,
        type annotations will be restored to the generated code.

        Args:
            ast_node: The Lua AST node to generate code from.
            restore_luau_types: If True and Luau metadata exists, restore type annotations.
                               Defaults to True.

        Returns:
            GenerateResult containing the generated code and any errors.

        Example:
            >>> result = processor.generate_code(ast_node)
            >>> if result.success:
            ...     print(result.code)
            ... else:
            ...     print(f"Generation failed: {result.errors}")

            # Skip Luau type restoration:
            >>> result = processor.generate_code(ast_node, restore_luau_types=False)

        Note:
            Deeply nested ASTs may cause RecursionError during generation.
        """
        try:
            code = luaparser.ast.to_lua_source(ast_node)
            self.logger.debug(f"Successfully generated Lua code ({len(code)} characters)")

            # Restore Luau type annotations if metadata exists and restoration is enabled
            luau_metadata = get_luau_metadata(ast_node)
            if restore_luau_types and luau_metadata and self._luau_generator:
                code = self._luau_generator.restore_type_annotations(code, luau_metadata)
                self.logger.debug("Restored Luau type annotations to generated code")

            return GenerateResult(
                code=code,
                success=True,
                errors=[],
            )
        except ValueError as e:
            error_msg = f"Failed to generate code: {e}"
            self.logger.error(error_msg)
            return GenerateResult(
                code="",
                success=False,
                errors=[error_msg],
            )
        except RecursionError:
            error_msg = "AST is too deeply nested to generate code (RecursionError)"
            self.logger.error(error_msg)
            return GenerateResult(
                code="",
                success=False,
                errors=[error_msg],
            )
        except Exception as e:
            error_msg = f"Unexpected error during code generation: {e}"
            self.logger.error(error_msg, exc_info=True)
            return GenerateResult(
                code="",
                success=False,
                errors=[error_msg],
            )

    def validate_syntax(self, code: str) -> tuple[bool, str]:
        """Validate Lua syntax without creating a full ParseResult.

        Attempts to parse the code string to check for syntax errors.

        Args:
            code: Lua source code string to validate.

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is empty.

        Example:
            >>> is_valid, error = processor.validate_syntax("local x = 10")
            >>> if is_valid:
            ...     print("Syntax is valid")
            ... else:
            ...     print(f"Syntax error: {error}")
        """
        try:
            luaparser.ast.parse(code)
            return (True, "")
        except SyntaxError as e:
            # Format error message with line/column if available
            if hasattr(e, "lineno") and hasattr(e, "offset"):
                error_msg = f"Line {e.lineno}, column {e.offset}: {e.msg}"
            else:
                error_msg = f"Syntax error: {e}"
            return (False, error_msg)
        except Exception as e:
            error_msg = f"Unexpected error validating syntax: {e}"
            return (False, error_msg)

    def get_ast_info(self, ast_node: luaparser.astnodes.Chunk) -> dict[str, Any]:
        """Extract metadata information from a Lua AST node.

        Args:
            ast_node: The Lua AST Chunk node to extract information from.

        Returns:
            Dictionary containing AST metadata (node_type, body_length, has_shebang).

        Example:
            >>> info = processor.get_ast_info(ast_node)
            >>> print(f"Node type: {info['node_type']}")
            >>> print(f"Statements: {info['body_length']}")
        """
        info = {
            "node_type": type(ast_node).__name__,
            "body_length": len(ast_node.body) if hasattr(ast_node, "body") else 0,
            "has_shebang": False,
        }

        # Check for shebang in the original source (Lua supports shebangs)
        # Note: This would require access to the original source code
        # For now, we'll just set it to False as AST doesn't preserve comments

        self.logger.debug(f"AST info: {info}")
        return info

    def detect_luajit_features(self, source_code: str) -> list[str]:
        """Detect LuaJIT-specific features in source code.

        This method analyzes the source code for LuaJIT-specific syntax patterns
        such as FFI usage, 64-bit integer literals, BitOp library usage, and JIT pragmas.

        Args:
            source_code: The Lua source code to analyze.

        Returns:
            List of detected LuaJIT feature names (e.g., ["ffi", "64bit_integers"]).

        Example:
            >>> processor = LuaProcessor()
            >>> features = processor.detect_luajit_features(source_code)
            >>> if "ffi" in features:
            ...     print("Warning: Code uses LuaJIT FFI")
        """
        if self._luajit_detector is None:
            self._luajit_detector = LuaJITDetector()
        return self._luajit_detector.detect_luajit_features(source_code)

    def extract_symbols(
        self, ast_node: luaparser.astnodes.Chunk, file_path: Path | str
    ) -> LuaSymbolTable:
        """Extract all symbols from a Lua AST for dependency analysis.

        Traverses the provided AST node to extract comprehensive symbol
        information including require() calls, function definitions, and
        variable assignments. This data is suitable for constructing
        dependency graphs and managing symbol tables for obfuscation.

        Args:
            ast_node: The Lua AST Chunk node to analyze. If None, returns an empty symbol table.
            file_path: Path to the source file (str or Path object).

        Returns:
            LuaSymbolTable containing all extracted symbols with metadata.

        Raises:
            RuntimeError: If symbol extraction fails unexpectedly.

        Example:
            >>> processor = LuaProcessor()
            >>> result = processor.parse_file("script.lua")
            >>> if result.success:
            ...     symbols = processor.extract_symbols(result.ast_node, result.file_path)
            ...     print(f"Found {len(symbols.imports)} require() calls")
            ...     print(f"Found {len(symbols.functions)} functions")
            ...     print(f"Found {len(symbols.variables)} variables")
            ...     print(f"Roblox patterns: {symbols.get_roblox_patterns()}")

        Note:
            - The provided ast_node must be a Chunk node (from luaparser.ast.parse())
            - If ast_node is None or invalid, returns an empty LuaSymbolTable
            - All symbols are extracted with comprehensive metadata
            - Scope information is tracked for proper name mangling
            - Roblox API patterns are detected for preservation
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path

        # Handle invalid AST gracefully by returning empty symbol table
        if ast_node is None:
            self.logger.warning(f"Invalid AST (None) provided for {path}, returning empty symbol table")
            return LuaSymbolTable(file_path=path)

        try:
            # Create extractor and traverse AST
            extractor = LuaSymbolExtractor(path)
            extractor.visit(ast_node)

            # Get the complete symbol table
            symbol_table = extractor.get_symbol_table()

            # Log extraction summary
            self.logger.info(
                f"Extracted symbols from {path}: "
                f"{len(symbol_table.imports)} imports, "
                f"{len(symbol_table.functions)} functions, "
                f"{len(symbol_table.variables)} variables, "
                f"{len(symbol_table.roblox_api_usage)} Roblox patterns"
            )

            return symbol_table

        except Exception as e:
            error_msg = f"Failed to extract symbols from {path}: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def obfuscate_with_symbol_table(
        self,
        ast_node: luaparser.astnodes.Chunk,
        file_path: Path | str,
        global_table: "GlobalSymbolTable"
    ) -> GenerateResult:
        """Obfuscate a Lua AST using a pre-computed GlobalSymbolTable.

        This method applies name mangling transformations to the AST using
        the pre-computed mangled names from the GlobalSymbolTable. It ensures
        consistent symbol renaming across files when processing in topological
        order.

        Args:
            ast_node: The Lua AST Chunk node to transform
            file_path: Path to the source file
            global_table: Pre-computed GlobalSymbolTable with mangled names

        Returns:
            GenerateResult containing the generated code and metadata

        Example:
            >>> processor = LuaProcessor()
            >>> result = processor.parse_file("script.lua")
            >>> if result.success:
            ...     gen_result = processor.obfuscate_with_symbol_table(
            ...         result.ast_node,
            ...         result.file_path,
            ...         global_table
            ...     )
            ...     if gen_result.success:
            ...         print(gen_result.code)

        Note:
            - The GlobalSymbolTable must be frozen before calling this method
            - Files should be processed in topological order for consistency
            - Symbols not in the table are left unchanged
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path

        try:
            # Check whether name mangling is enabled via mangle_globals flag.
            # Default to True when no config is provided (backward compat).
            mangle_enabled = True
            if self._config is not None:
                mangle_enabled = self._config.features.get(
                    "mangle_globals", True
                )

            if mangle_enabled:
                # Create the name mangling visitor and apply transformations
                visitor = LuaNameManglingVisitor(global_table, path)
                visitor.visit(ast_node)
            else:
                self.logger.info(
                    f"Name mangling disabled for {path.name}; skipping LuaNameManglingVisitor"
                )

            # Apply additional transformations via ObfuscationEngine if config exists
            if self._config is not None:
                from obfuscator.core.obfuscation_engine import ObfuscationEngine

                engine = ObfuscationEngine(self._config)
                engine_result = engine.apply_transformations(
                    ast_node, "lua", path
                )

                if engine_result.success:
                    self.logger.info(
                        f"Engine transformations succeeded for {path.name}: "
                        f"{engine_result.transformation_count} additional transformations"
                    )
                    # Use the engine-transformed AST for code generation
                    ast_node = engine_result.ast_node
                else:
                    self.logger.error(
                        f"Engine transformations failed for {path.name}: "
                        f"{engine_result.errors}"
                    )
                    return GenerateResult(
                        code="",
                        success=False,
                        errors=engine_result.errors,
                        warnings=[],
                    )

            # Generate code from the transformed AST
            gen_result = self.generate_code(ast_node)

            if gen_result.success:
                self.logger.info(
                    f"Successfully obfuscated {path.name} using GlobalSymbolTable"
                )
            else:
                self.logger.warning(
                    f"Obfuscation of {path.name} completed with errors: {gen_result.errors}"
                )

            return gen_result

        except Exception as e:
            error_msg = f"Failed to obfuscate {path}: {e}"
            self.logger.error(error_msg, exc_info=True)
            return GenerateResult(
                code="",
                success=False,
                errors=[error_msg],
                warnings=[],
                metadata={"error": str(e)}
            )

    def apply_transformations(
        self,
        ast_node: luaparser.astnodes.Chunk,
        transformers: list[Any],
    ) -> GenerateResult:
        """Apply a pipeline of AST transformations to a Lua chunk.

        Applies each transformer in sequence to the AST, accumulating results.
        Validates the Lua AST by attempting code generation after all
        transformations complete.

        Args:
            ast_node: The Lua AST Chunk node to transform.
            transformers: List of transformer instances to apply in sequence.
                         Each must have a ``transform()`` method returning a
                         ``TransformResult``.

        Returns:
            GenerateResult containing the generated code if successful,
            or error messages if any transformation or validation failed.
        """
        if not transformers:
            return self.generate_code(ast_node)

        current_ast = ast_node
        total_count = 0
        all_errors: list[str] = []

        for idx, transformer in enumerate(transformers):
            name = type(transformer).__name__
            self.logger.debug(
                f"Applying Lua transformer {idx + 1}/{len(transformers)}: {name}"
            )

            result = transformer.transform(current_ast)

            if not result.success:
                error_msg = (
                    f"Lua transformer {name} failed: "
                    f"{', '.join(result.errors)}"
                )
                self.logger.error(error_msg)
                all_errors.extend(result.errors)
                return GenerateResult(
                    code="",
                    success=False,
                    errors=all_errors,
                )

            current_ast = result.ast_node
            total_count += result.transformation_count
            self.logger.debug(
                f"{name} completed: {result.transformation_count} nodes transformed"
            )

        # Validate by attempting code generation
        gen_result = self.generate_code(current_ast)
        if not gen_result.success:
            self.logger.warning(
                f"Lua AST validation failed after {total_count} transformations"
            )
        else:
            self.logger.info(
                f"Lua transformation pipeline completed: "
                f"{total_count} total transformations"
            )

        return gen_result


class LuaNameManglingVisitor(luaparser.ast.ASTVisitor):
    """AST visitor that renames symbols using a pre-computed GlobalSymbolTable.

    This visitor traverses the Lua AST and replaces original symbol names with
    their mangled counterparts from the GlobalSymbolTable. It tracks scope using
    a scope stack similar to LuaSymbolExtractor.

    Attributes:
        global_table: The pre-computed GlobalSymbolTable
        file_path: Path to the file being transformed
        _scope_stack: Stack tracking current scope names

    Example:
        >>> visitor = LuaNameManglingVisitor(global_table, Path("script.lua"))
        >>> visitor.visit(ast_node)
        >>> # AST is now modified in-place with mangled names

    Limitations and Edge Cases:
        1. Dynamic requires: Cannot track require() with computed strings
        2. String-based access: Cannot mangle table["key"] with string literals
        3. Loadstring: Code in loadstring() is not analyzed
        4. Metatables: Cannot track symbols accessed via __index metamethods
        5. C modules: Cannot mangle symbols from C-based Lua modules
        6. Roblox RemoteEvents: Event names passed as strings are not mangled
        
    Cross-File Consistency:
        - Requires GlobalSymbolTable to be frozen before transformation
        - Files must be processed in topological order from DependencyGraph
        - Circular requires fall back to original file order (may affect consistency)
        
    Preserved Symbols:
        - Lua keywords (local, function, end, if, then, etc.)
        - Roblox global objects (game, workspace, script, plugin, etc.)
        - Roblox services (Players, ReplicatedStorage, RunService, etc.)
        - Roblox API methods (GetService, FindFirstChild, WaitForChild, etc.)
        - Roblox datatypes (Vector3, CFrame, Color3, Instance, etc.)
        - Symbols marked with is_exported=True (if preserve_exports config enabled)
    """

    def __init__(self, global_table: "GlobalSymbolTable", file_path: Path) -> None:
        """Initialize the name mangling visitor.

        Args:
            global_table: Pre-computed GlobalSymbolTable with mangled names
            file_path: Path to the source file being transformed
        """
        super().__init__()
        self.global_table = global_table
        self.file_path = file_path.resolve() if not file_path.is_absolute() else file_path
        self._scope_stack: list[str] = []
        self._logger = get_logger("obfuscator.processors.lua_name_mangler")

    def _get_current_scope(self) -> str:
        """Get the current scope for symbol lookup.

        Returns:
            "global" if at module level, "local" otherwise
        """
        return "global" if len(self._scope_stack) == 0 else "local"

    def _try_mangle_name(self, name: str) -> str:
        """Attempt to get mangled name from global table.

        Tries current scope first, then falls back to global scope.

        Args:
            name: Original symbol name

        Returns:
            Mangled name if found, original name otherwise
        """
        scope = self._get_current_scope()

        # Try current scope first
        mangled = self.global_table.get_mangled_name(self.file_path, name, scope)
        if mangled:
            return mangled

        # Fall back to global scope if we're in local scope
        if scope == "local":
            mangled = self.global_table.get_mangled_name(self.file_path, name, "global")
            if mangled:
                return mangled

        # Return original name if not found in symbol table
        return name

    def _find_cross_file_mangled_name(self, name: str) -> str | None:
        """Search all files in the global table for a mangled name.

        Looks up a symbol by original_name in global scope across all files.
        Used for resolving field/member accesses on required modules.

        Args:
            name: Original symbol name to look up

        Returns:
            Mangled name if found (and different from original), None otherwise
        """
        for entry in self.global_table.get_all_symbols():
            if entry.original_name == name and entry.scope == "global":
                if entry.mangled_name != name:
                    return entry.mangled_name
        return None

    def visit_Name(self, node: luaparser.astnodes.Name) -> None:
        """Transform Name nodes by replacing with mangled names.

        Args:
            node: Name AST node
        """
        if hasattr(node, 'id') and node.id:
            original = node.id
            mangled = self._try_mangle_name(original)
            if mangled != original:
                self._logger.debug(f"Mangling Name: {original} -> {mangled}")
                node.id = mangled

    def visit_LocalFunction(self, node: luaparser.astnodes.LocalFunction) -> None:
        """Transform LocalFunction nodes by renaming function and traversing body.

        Args:
            node: LocalFunction AST node
        """
        # Mangle function name
        if hasattr(node, 'name') and node.name:
            func_name = node.name.id if hasattr(node.name, 'id') else str(node.name)
            mangled = self._try_mangle_name(func_name)
            if mangled != func_name:
                self._logger.debug(f"Mangling LocalFunction: {func_name} -> {mangled}")
                if hasattr(node.name, 'id'):
                    node.name.id = mangled

        # Push scope and traverse body
        self._scope_stack.append(func_name if hasattr(node, 'name') and node.name else "<anonymous>")
        self._visit_children(node)
        self._scope_stack.pop()

    def visit_Function(self, node: luaparser.astnodes.Function) -> None:
        """Transform Function nodes by renaming function and traversing body.

        Args:
            node: Function AST node
        """
        func_name = "<anonymous>"
        # Mangle function name if it exists
        if hasattr(node, 'name') and node.name:
            if hasattr(node.name, 'id'):
                func_name = node.name.id
                mangled = self._try_mangle_name(func_name)
                if mangled != func_name:
                    self._logger.debug(f"Mangling Function: {func_name} -> {mangled}")
                    node.name.id = mangled

        # Push scope and traverse body
        self._scope_stack.append(func_name)
        self._visit_children(node)
        self._scope_stack.pop()

    def visit_Assign(self, node: luaparser.astnodes.Assign) -> None:
        """Transform Assign nodes by renaming target variables.

        Args:
            node: Assign AST node
        """
        # Mangle target names
        if hasattr(node, 'targets') and node.targets:
            for target in node.targets:
                if hasattr(target, 'id') and target.id:
                    original = target.id
                    mangled = self._try_mangle_name(original)
                    if mangled != original:
                        self._logger.debug(f"Mangling Assign target: {original} -> {mangled}")
                        target.id = mangled

        # Continue traversal
        self._visit_children(node)

    def visit_Index(self, node: luaparser.astnodes.Index) -> None:
        """Transform Index nodes by mangling field names for dot/bracket access.

        For ``helper.calculate_sum``, if ``calculate_sum`` is a global symbol
        in the GlobalSymbolTable, rewrites the field identifier to the mangled
        name. Require paths and built-in names are left intact.

        Args:
            node: Index AST node
        """
        # Handle field name mangling for dot access (idx is a Name node)
        if hasattr(node, 'idx') and node.idx:
            if hasattr(node.idx, 'id') and node.idx.id:
                field_name = node.idx.id
                mangled = self._find_cross_file_mangled_name(field_name)
                if mangled and mangled != field_name:
                    self._logger.debug(f"Mangling Index field: {field_name} -> {mangled}")
                    node.idx.id = mangled

        # Only traverse the value child (skip idx to prevent double-mangling)
        if hasattr(node, 'value') and node.value:
            if hasattr(node.value, '__class__') and hasattr(node.value.__class__, '__module__'):
                if 'astnodes' in node.value.__class__.__module__:
                    self.visit(node.value)

    def visit_Invoke(self, node: luaparser.astnodes.Invoke) -> None:
        """Transform Invoke nodes by mangling method names for colon calls.

        For ``obj:method()``, if ``method`` is a global symbol in the
        GlobalSymbolTable, rewrites the method identifier to the mangled name.

        Args:
            node: Invoke AST node
        """
        # Handle method name mangling for colon calls: obj:method()
        if hasattr(node, 'func') and node.func:
            if hasattr(node.func, 'id') and node.func.id:
                method_name = node.func.id
                mangled = self._find_cross_file_mangled_name(method_name)
                if mangled and mangled != method_name:
                    self._logger.debug(f"Mangling Invoke method: {method_name} -> {mangled}")
                    node.func.id = mangled

        # Traverse source and args
        self._visit_children(node)

    def visit_LocalAssign(self, node: luaparser.astnodes.LocalAssign) -> None:
        """Transform LocalAssign nodes by renaming target variables.

        Args:
            node: LocalAssign AST node
        """
        # Mangle target names
        if hasattr(node, 'targets') and node.targets:
            for target in node.targets:
                if hasattr(target, 'id') and target.id:
                    original = target.id
                    mangled = self._try_mangle_name(original)
                    if mangled != original:
                        self._logger.debug(f"Mangling LocalAssign target: {original} -> {mangled}")
                        target.id = mangled

        # Continue traversal
        self._visit_children(node)

    def _visit_children(self, node: Any) -> None:
        """Manually traverse child nodes to continue AST traversal.

        Args:
            node: Current AST node
        """
        for attr_name in dir(node):
            if attr_name.startswith('_'):
                continue
            attr = getattr(node, attr_name, None)
            if attr is None:
                continue
            # Handle lists of nodes
            if isinstance(attr, list):
                for item in attr:
                    if hasattr(item, '__class__') and hasattr(item.__class__, '__module__'):
                        if 'astnodes' in item.__class__.__module__:
                            self.visit(item)
            # Handle single nodes
            elif hasattr(attr, '__class__') and hasattr(attr.__class__, '__module__'):
                if 'astnodes' in attr.__class__.__module__:
                    self.visit(attr)
