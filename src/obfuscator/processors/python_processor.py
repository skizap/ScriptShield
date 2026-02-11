"""Python source code processor for AST parsing and code generation.

This module provides the PythonProcessor class for parsing Python source files
into Abstract Syntax Trees (AST) and generating Python code from AST nodes.
It uses Python's standard `ast` module and requires Python 3.9+ for native
`ast.unparse()` support.

Example:
    Basic parsing and code generation workflow:

    >>> from obfuscator.processors import PythonProcessor
    >>> processor = PythonProcessor()
    >>>
    >>> # Parse a Python file
    >>> result = processor.parse_file("example.py")
    >>> if result.success:
    ...     print(f"Parsed {len(result.ast_node.body)} top-level statements")
    ...
    ...     # Generate code from AST
    ...     gen_result = processor.generate_code(result.ast_node)
    ...     if gen_result.success:
    ...         print(gen_result.code)
    ... else:
    ...     print(f"Errors: {result.errors}")

Note:
    Comments and exact whitespace are not preserved when parsing and regenerating
    code. This is a fundamental limitation of AST-based processing.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from obfuscator.processors.ast_transformer import (
    ASTTransformer,
    ConstantFoldingTransformer,
    TransformResult,
)
from obfuscator.processors.feature_detector import (
    FeatureWarning,
    UnsupportedFeatureDetector,
)
from obfuscator.processors.symbol_extractor import (
    SymbolExtractor,
    SymbolTable,
)
from obfuscator.utils.logger import get_logger

# Import for type checking only to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from obfuscator.core.config import ObfuscationConfig
    from obfuscator.core.symbol_table import GlobalSymbolTable
from obfuscator.core.obfuscation_engine import ObfuscationEngine

logger = get_logger("obfuscator.processors.python_processor")

# Module constants
SUPPORTED_PYTHON_VERSIONS: tuple[str, ...] = ("3.9", "3.10", "3.11", "3.12", "3.13")
"""Supported Python versions for AST processing (requires ast.unparse)."""

MAX_FILE_SIZE_MB: int = 10
"""Maximum file size in megabytes to prevent memory issues with large files."""


@dataclass
class ParseResult:
    """Result of parsing a Python source file.

    This dataclass encapsulates the result of parsing a Python file into an AST,
    including success status, the AST node, original source code, any errors,
    and warnings about unsupported features.

    Attributes:
        ast_node: The parsed AST module node (None if parsing failed)
        source_code: Original source code that was parsed
        file_path: Path to the source file
        success: Whether parsing completed successfully
        errors: List of error messages (empty if successful)
        warnings: List of FeatureWarning objects about unsupported features

    Example:
        >>> result = processor.parse_file("example.py")
        >>> if result.success:
        ...     print(f"AST has {len(result.ast_node.body)} statements")
        ...     for warning in result.warnings:
        ...         print(f"Warning: {warning.feature_name} at line {warning.line_number}")
        ... else:
        ...     for error in result.errors:
        ...         print(f"Error: {error}")
    """

    ast_node: ast.Module | None
    source_code: str
    file_path: Path
    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[FeatureWarning] = field(default_factory=list)


@dataclass
class GenerateResult:
    """Result of generating Python code from an AST.

    This dataclass encapsulates the result of generating Python source code
    from an AST node, including success status and any errors.

    Attributes:
        code: Generated Python source code (empty string if generation failed)
        success: Whether code generation completed successfully
        errors: List of error messages (empty if successful)

    Example:
        >>> gen_result = processor.generate_code(ast_node)
        >>> if gen_result.success:
        ...     print(gen_result.code)
        ... else:
        ...     print(f"Generation failed: {gen_result.errors}")

    Note:
        Generated code will not preserve comments or exact whitespace from
        the original source, as this information is not stored in the AST.
    """

    code: str
    success: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class TransformationPipelineResult:
    """Result of applying a pipeline of AST transformations.

    This dataclass encapsulates the outcome of applying multiple transformers
    in sequence to an AST, including the final transformed AST, success status,
    transformation counts, and any validation errors.

    Attributes:
        ast_node: The transformed AST Module node (None if pipeline failed)
        success: Whether all transformations completed successfully
        total_transformations: Total number of nodes transformed across all transformers
        errors: List of error messages from transformations
        validation_errors: List of validation errors detected after transformations

    Example:
        >>> transformers = [ConstantFoldingTransformer()]
        >>> result = processor.apply_transformations(ast_node, transformers)
        >>> if result.success:
        ...     print(f"Transformed {result.total_transformations} nodes")
        ...     if result.validation_errors:
        ...         print(f"Validation warnings: {result.validation_errors}")
        ... else:
        ...     print(f"Failed: {result.errors}")
    """

    ast_node: ast.Module | None
    success: bool
    total_transformations: int
    errors: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)


class PythonProcessor:
    """Processor for parsing Python files to AST and generating code from AST.

    This class provides methods for:
    - Parsing Python source files into Abstract Syntax Trees
    - Generating Python code from AST nodes
    - Validating Python syntax
    - Extracting basic AST information

    The processor uses Python's standard `ast` module and requires Python 3.9+
    for the `ast.unparse()` function.

    Example:
        >>> processor = PythonProcessor()
        >>>
        >>> # Parse a file
        >>> result = processor.parse_file("script.py")
        >>> if result.success:
        ...     # Modify AST here if needed
        ...
        ...     # Generate code back
        ...     gen_result = processor.generate_code(result.ast_node)
        ...     if gen_result.success:
        ...         print(gen_result.code)

    Note:
        Comments and exact formatting are not preserved during the parse-generate
        cycle. This is a fundamental limitation of AST-based code processing.
    """

    def __init__(self, config: "ObfuscationConfig | None" = None, engine: "ObfuscationEngine | None" = None) -> None:
        """Initialize the Python processor.

        Creates a new PythonProcessor instance ready to parse files
        and generate code.

        Args:
            config: Optional ObfuscationConfig instance. When provided,
                    the engine will apply additional transformations
                    (string encryption, number obfuscation, etc.) after
                    name mangling in ``obfuscate_with_symbol_table()``.
            engine: Optional ObfuscationEngine instance. When provided,
                    it will be used for transformations; otherwise a new engine is created.
        """
        self._config = config or ObfuscationConfig()
        self._current_engine: ObfuscationEngine | None = engine
        logger.debug("PythonProcessor initialized")

    def get_engine(self) -> ObfuscationEngine | None:
        """Get the engine instance used in the last obfuscation.

        Returns:
            The ObfuscationEngine instance from the last call to
            obfuscate_with_symbol_table(), or None if not yet called.
        """
        return self._current_engine

    def parse_file(self, file_path: Path | str) -> ParseResult:
        """Parse a Python source file into an Abstract Syntax Tree.

        Reads the specified Python file and parses it into an AST using
        Python's `ast.parse()` function. Performs validation for file
        existence, readability, and size limits.

        Args:
            file_path: Path to the Python source file (str or Path object)

        Returns:
            ParseResult containing the AST node if successful, or error
            messages if parsing failed.

        Example:
            >>> processor = PythonProcessor()
            >>> result = processor.parse_file("example.py")
            >>> if result.success:
            ...     print(f"Parsed: {result.file_path}")
            ...     print(f"Statements: {len(result.ast_node.body)}")
            ... else:
            ...     print(f"Failed: {result.errors}")

        Note:
            - Files larger than MAX_FILE_SIZE_MB (10MB) will be rejected
            - Files must be valid UTF-8 encoded Python source
            - Type comments are preserved if present (type_comments=True)
        """
        path = Path(file_path)

        # Validate file exists
        if not path.exists():
            error_msg = f"File not found: {path}"
            logger.error(error_msg)
            return ParseResult(
                ast_node=None,
                source_code="",
                file_path=path,
                success=False,
                errors=[error_msg]
            )

        # Validate file is readable
        if not os.access(path, os.R_OK):
            error_msg = f"File is not readable: {path}"
            logger.error(error_msg)
            return ParseResult(
                ast_node=None,
                source_code="",
                file_path=path,
                success=False,
                errors=[error_msg]
            )

        # Check file size
        try:
            file_size = path.stat().st_size
            max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
            if file_size > max_size_bytes:
                error_msg = (
                    f"File exceeds maximum size of {MAX_FILE_SIZE_MB}MB: "
                    f"{path} ({file_size / (1024 * 1024):.2f}MB)"
                )
                logger.error(error_msg)
                return ParseResult(
                    ast_node=None,
                    source_code="",
                    file_path=path,
                    success=False,
                    errors=[error_msg]
                )
        except OSError as e:
            error_msg = f"Failed to get file size for {path}: {e}"
            logger.error(error_msg)
            return ParseResult(
                ast_node=None,
                source_code="",
                file_path=path,
                success=False,
                errors=[error_msg]
            )

        # Read file content
        try:
            source_code = path.read_text(encoding="utf-8")
        except OSError as e:
            error_msg = f"Failed to read file {path}: {e}"
            logger.error(error_msg)
            return ParseResult(
                ast_node=None,
                source_code="",
                file_path=path,
                success=False,
                errors=[error_msg]
            )
        except ValueError as e:
            error_msg = f"Encoding error reading file {path}: {e}"
            logger.error(error_msg)
            return ParseResult(
                ast_node=None,
                source_code="",
                file_path=path,
                success=False,
                errors=[error_msg]
            )

        # Parse source code to AST
        try:
            ast_node = ast.parse(
                source_code,
                filename=str(path),
                type_comments=True
            )
        except SyntaxError as e:
            error_msg = (
                f"Syntax error in {path} at line {e.lineno}, "
                f"column {e.offset}: {e.msg}"
            )
            logger.error(error_msg)
            return ParseResult(
                ast_node=None,
                source_code=source_code,
                file_path=path,
                success=False,
                errors=[error_msg]
            )
        except ValueError as e:
            error_msg = f"Value error parsing {path}: {e}"
            logger.error(error_msg)
            return ParseResult(
                ast_node=None,
                source_code=source_code,
                file_path=path,
                success=False,
                errors=[error_msg]
            )

        logger.info(f"Successfully parsed {path}")

        # Detect unsupported features
        detector = UnsupportedFeatureDetector(path)
        detector.visit(ast_node)
        warnings = detector.get_warnings()

        # Log warnings by severity
        for warning in warnings:
            log_message = (
                f"{warning.file_path}:{warning.line_number}:{warning.column_offset} - "
                f"{warning.feature_name}: {warning.description}"
            )
            if warning.severity == "critical":
                logger.error(log_message)
            elif warning.severity == "error":
                logger.error(log_message)
            elif warning.severity == "warning":
                logger.warning(log_message)

        # Log summary
        logger.info(f"Detected {len(warnings)} unsupported feature(s) in {path}")

        return ParseResult(
            ast_node=ast_node,
            source_code=source_code,
            file_path=path,
            success=True,
            errors=[],
            warnings=warnings
        )


    def generate_code(self, ast_node: ast.AST) -> GenerateResult:
        """Generate Python source code from an AST node.

        Converts an Abstract Syntax Tree back into Python source code using
        Python's `ast.unparse()` function. Applies `ast.fix_missing_locations()`
        to ensure all nodes have proper location information.

        Args:
            ast_node: Any AST node to convert to source code (Module, Expression,
                FunctionDef, ClassDef, statements, expressions, etc.)

        Returns:
            GenerateResult containing the generated code if successful, or
            error messages if generation failed.

        Example:
            >>> processor = PythonProcessor()
            >>> result = processor.parse_file("example.py")
            >>> if result.success:
            ...     gen_result = processor.generate_code(result.ast_node)
            ...     if gen_result.success:
            ...         print(gen_result.code)
            >>>
            >>> # Generate code from individual nodes
            >>> import ast
            >>> func_node = ast.parse("def foo(): pass").body[0]
            >>> gen_result = processor.generate_code(func_node)

        Note:
            - Comments and exact whitespace from the original source are not
              preserved, as this information is not stored in the AST.
            - Very deeply nested ASTs may cause RecursionError.
        """
        # Fix missing location information
        ast.fix_missing_locations(ast_node)

        # Generate code from AST
        try:
            code = ast.unparse(ast_node)
        except ValueError as e:
            error_msg = f"Failed to generate code: {e}"
            logger.error(error_msg)
            return GenerateResult(
                code="",
                success=False,
                errors=[error_msg]
            )
        except RecursionError:
            error_msg = "AST is too deeply nested to generate code (RecursionError)"
            logger.error(error_msg)
            return GenerateResult(
                code="",
                success=False,
                errors=[error_msg]
            )

        logger.debug(f"Successfully generated {len(code)} characters of code")
        return GenerateResult(
            code=code,
            success=True,
            errors=[]
        )

    def validate_syntax(self, code: str) -> tuple[bool, str]:
        """Validate Python syntax without creating a full ParseResult.

        This is a lightweight method to check if a string contains valid
        Python syntax. It does not create file-related result objects.

        Args:
            code: Python source code string to validate

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is
            an empty string. If invalid, error_message contains the syntax
            error description.

        Example:
            >>> processor = PythonProcessor()
            >>> is_valid, error = processor.validate_syntax("x = 1 + 2")
            >>> print(is_valid)  # True
            >>>
            >>> is_valid, error = processor.validate_syntax("x = ")
            >>> print(is_valid)  # False
            >>> print(error)     # "unexpected EOF while parsing..."
        """
        try:
            ast.parse(code)
            return (True, "")
        except SyntaxError as e:
            error_msg = f"Line {e.lineno}, column {e.offset}: {e.msg}"
            return (False, error_msg)

    def get_ast_info(self, ast_node: ast.Module) -> dict[str, Any]:
        """Extract basic information from an AST for debugging and logging.

        Provides metadata about the AST structure without modifying it.
        Useful for debugging and understanding the parsed code structure.

        Args:
            ast_node: The AST Module node to analyze

        Returns:
            Dictionary containing:
            - node_type: Type name of the AST node
            - body_length: Number of top-level statements
            - has_docstring: Whether the module has a docstring

        Example:
            >>> processor = PythonProcessor()
            >>> result = processor.parse_file("example.py")
            >>> if result.success:
            ...     info = processor.get_ast_info(result.ast_node)
            ...     print(f"Statements: {info['body_length']}")
            ...     print(f"Has docstring: {info['has_docstring']}")
        """
        info: dict[str, Any] = {
            "node_type": type(ast_node).__name__,
            "body_length": len(ast_node.body) if hasattr(ast_node, "body") else 0,
            "has_docstring": False
        }

        # Check for module docstring
        if ast_node.body:
            first_stmt = ast_node.body[0]
            if isinstance(first_stmt, ast.Expr):
                if isinstance(first_stmt.value, ast.Constant):
                    if isinstance(first_stmt.value.value, str):
                        info["has_docstring"] = True

        logger.debug(f"AST info: {info}")
        return info

    def extract_symbols(self, ast_node: ast.Module, file_path: Path | str) -> SymbolTable:
        """Extract all symbols from an AST for dependency analysis.

        Traverses the provided AST node to extract comprehensive symbol
        information including imports, functions, classes, and variables.
        This data is suitable for constructing dependency graphs and
        managing symbol tables for obfuscation.

        Args:
            ast_node: The AST Module node to analyze
            file_path: Path to the source file (str or Path object)

        Returns:
            SymbolTable containing all extracted symbols with metadata

        Raises:
            RuntimeError: If symbol extraction fails unexpectedly

        Example:
            >>> processor = PythonProcessor()
            >>> result = processor.parse_file("example.py")
            >>> if result.success:
            ...     symbols = processor.extract_symbols(result.ast_node, result.file_path)
            ...     print(f"Found {len(symbols.imports)} imports")
            ...     print(f"Found {len(symbols.functions)} functions")
            ...     print(f"Found {len(symbols.classes)} classes")
            ...     print(f"Found {len(symbols.variables)} variables")
            ...     print(f"Exports: {symbols.get_exported_symbols()}")

        Note:
            - The provided ast_node must be a Module node (from ast.parse())
            - All symbols are extracted with comprehensive metadata
            - Scope information is tracked for proper name mangling
            - Constants are identified by uppercase naming convention
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path

        try:
            # Create extractor and traverse AST
            extractor = SymbolExtractor(path)
            extractor.visit(ast_node)
            
            # Get the complete symbol table
            symbol_table = extractor.get_symbol_table()
            
            # Log extraction summary
            logger.info(
                f"Extracted symbols from {path}: "
                f"{len(symbol_table.imports)} imports, "
                f"{len(symbol_table.functions)} functions, "
                f"{len(symbol_table.classes)} classes, "
                f"{len(symbol_table.variables)} variables"
            )
            
            return symbol_table
            
        except Exception as e:
            error_msg = f"Failed to extract symbols from {path}: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def obfuscate_with_symbol_table(
        self,
        ast_node: ast.AST,
        file_path: Path,
        symbol_table: SymbolTable,
        engine: ObfuscationEngine | None = None,
    ) -> dict[str, Any]:
        """Obfuscate using a pre-computed symbol table with optional engine.

        This method applies obfuscation transformations using an optional
        external engine instance, enabling consistent runtime key generation
        across the obfuscation pipeline.

        Args:
            ast_node: The AST node to obfuscate.
            file_path: Path to the source file (for context).
            symbol_table: Pre-computed global symbol table for collision avoidance.
            engine: Optional ObfuscationEngine instance. If provided, it will be
                   used for transformations; otherwise a new engine is created.

        Returns:
            Dictionary with obfuscation results containing:
            - ast_node: The obfuscated AST node
            - success: Whether obfuscation succeeded
            - errors: List of any error messages
            - engine: The ObfuscationEngine instance used (for runtime injection)

        Example:
            >>> processor = PythonProcessor()
            >>> result = processor.parse_file("example.py")
            >>> if result.success:
            ...     transform_result = processor.obfuscate_with_symbol_table(
            ...         result.ast_node,
            ...         result.file_path,
            ...         global_table
            ...     )
            ...     if transform_result.success:
            ...         code = processor.generate_code(transform_result.ast_node)
            ...         print(code.code)

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
                # Create the name mangling transformer
                transformer = NameManglingTransformer(symbol_table, path)

                # Apply the transformation
                result = transformer.transform(ast_node)

                if result.success:
                    logger.info(
                        f"Successfully obfuscated {path.name} using GlobalSymbolTable"
                    )
                else:
                    logger.warning(
                        f"Obfuscation of {path.name} completed with errors: {result.errors}"
                    )
                    return {
                        "ast_node": ast_node,
                        "success": False,
                        "errors": result.errors,
                        "engine": None,
                    }
            else:
                logger.info(
                    f"Name mangling disabled for {path.name}; skipping NameManglingTransformer"
                )
                result = TransformResult(
                    ast_node=ast_node,
                    success=True,
                    transformation_count=0,
                    errors=[],
                )

            # Use provided engine or create new one for additional transformations
            if self._config is not None and result.ast_node is not None:
                if engine is not None:
                    self._current_engine = engine
                    logger.debug("Using provided ObfuscationEngine instance")
                else:
                    from obfuscator.core.obfuscation_engine import ObfuscationEngine
                    self._current_engine = ObfuscationEngine(self._config)
                    logger.debug("Created new ObfuscationEngine instance")

                engine_result = self._current_engine.apply_transformations(
                    result.ast_node, "python", path
                )

                if engine_result.success:
                    logger.info(
                        f"Engine transformations succeeded for {path.name}: "
                        f"{engine_result.transformation_count} additional transformations"
                    )
                    final_ast = engine_result.ast_node

                    # For embedded runtime mode, embed runtime into AST
                    if self._config.runtime_mode == "embedded" and self._current_engine.has_runtime_requirements():
                        final_ast = self._current_engine.embed_runtime_in_ast(
                            engine_result.ast_node, "python"
                        )
                        logger.debug("Embedded runtime code into Python AST")

                    return {
                        "ast_node": final_ast,
                        "success": True,
                        "errors": result.errors + engine_result.errors,
                        "engine": self._current_engine,
                    }
                else:
                    logger.error(
                        f"Engine transformations failed for {path.name}: "
                        f"{engine_result.errors}"
                    )
                    return {
                        "ast_node": result.ast_node,
                        "success": False,
                        "errors": result.errors + engine_result.errors,
                        "engine": self._current_engine,
                    }

            # No config, just return the name mangling result
            return {
                "ast_node": result.ast_node,
                "success": result.success,
                "errors": result.errors,
                "engine": None,
            }

        except Exception as e:
            error_msg = f"Failed to obfuscate {path}: {e}"
            logger.error(error_msg, exc_info=True)
            return {
                "ast_node": ast_node,
                "success": False,
                "errors": [error_msg],
                "engine": None,
            }

    def apply_transformations(
        self,
        ast_node: ast.Module,
        transformers: list[ASTTransformer]
    ) -> TransformationPipelineResult:
        """Apply a pipeline of AST transformations to a module.

        Applies each transformer in sequence to the AST, accumulating results
        and validating the integrity of the transformed AST. If any transformation
        fails, the pipeline stops and returns the error.

        Args:
            ast_node: The AST Module node to transform
            transformers: List of ASTTransformer instances to apply in sequence

        Returns:
            TransformationPipelineResult containing the final transformed AST,
            success status, total transformation count, and any errors.

        Example:
            >>> processor = PythonProcessor()
            >>> result = processor.parse_file("example.py")
            >>> if result.success:
            ...     # Apply constant folding transformation
            ...     transformers = [ConstantFoldingTransformer()]
            ...     transform_result = processor.apply_transformations(
            ...         result.ast_node,
            ...         transformers
            ...     )
            ...     if transform_result.success:
            ...         # Generate code from transformed AST
            ...         code_result = processor.generate_code(transform_result.ast_node)
            ...         print(f"Transformed {transform_result.total_transformations} nodes")

        Note:
            - All transformers must be instances of ASTTransformer or its subclasses
            - Transformers are applied in the order provided
            - The pipeline stops on the first transformation failure
            - AST integrity is validated after all transformations complete
        """
        # Validate inputs
        if not isinstance(ast_node, ast.Module):
            error_msg = f"Expected ast.Module, got {type(ast_node).__name__}"
            logger.error(error_msg)
            return TransformationPipelineResult(
                ast_node=None,
                success=False,
                total_transformations=0,
                errors=[error_msg],
                validation_errors=[]
            )

        if not transformers:
            error_msg = "Transformer list cannot be empty"
            logger.error(error_msg)
            return TransformationPipelineResult(
                ast_node=None,
                success=False,
                total_transformations=0,
                errors=[error_msg],
                validation_errors=[]
            )

        # Validate all transformers
        for i, transformer in enumerate(transformers):
            if not isinstance(transformer, ASTTransformer):
                error_msg = (
                    f"Transformer at index {i} is not an ASTTransformer instance: "
                    f"{type(transformer).__name__}"
                )
                logger.error(error_msg)
                return TransformationPipelineResult(
                    ast_node=None,
                    success=False,
                    total_transformations=0,
                    errors=[error_msg],
                    validation_errors=[]
                )

        logger.info(f"Starting transformation pipeline with {len(transformers)} transformer(s)")
        current_ast = ast_node
        total_transformations = 0
        all_errors: list[str] = []

        # Apply each transformer in sequence
        for idx, transformer in enumerate(transformers):
            logger.debug(f"Applying transformer {idx + 1}/{len(transformers)}: {type(transformer).__name__}")

            # Apply the transformation
            transform_result: TransformResult = transformer.transform(current_ast)

            if not transform_result.success:
                error_msg = (
                    f"Transformer {type(transformer).__name__} failed: "
                    f"{', '.join(transform_result.errors)}"
                )
                logger.error(error_msg)
                all_errors.extend(transform_result.errors)

                return TransformationPipelineResult(
                    ast_node=None,
                    success=False,
                    total_transformations=total_transformations,
                    errors=all_errors,
                    validation_errors=[]
                )

            # Update AST and accumulate transformation count
            current_ast = transform_result.ast_node
            total_transformations += transform_result.transformation_count

            logger.debug(
                f"Transformer {type(transformer).__name__} completed: "
                f"{transform_result.transformation_count} nodes transformed"
            )

        # Validate AST integrity after all transformations
        logger.debug("Validating AST integrity after transformations")
        is_valid, validation_errors = self.validate_ast_integrity(current_ast)

        if not is_valid:
            logger.warning(f"AST validation failed: {validation_errors}")
        else:
            logger.debug("AST validation passed")

        # Log pipeline completion
        logger.info(
            f"Transformation pipeline completed: "
            f"{total_transformations} total transformations, "
            f"{len(validation_errors)} validation errors"
        )

        return TransformationPipelineResult(
            ast_node=current_ast,
            success=is_valid,
            total_transformations=total_transformations,
            errors=all_errors,
            validation_errors=validation_errors
        )

    def validate_ast_integrity(self, ast_node: ast.AST) -> tuple[bool, list[str]]:
        """Validate the integrity of an AST node.

        Performs comprehensive validation to ensure the AST is structurally
        sound and can be successfully unparsed and reparsed.

        Args:
            ast_node: The AST node to validate (typically an ast.Module)

        Returns:
            Tuple of (is_valid, error_messages). If valid, error_messages is
            an empty list. If invalid, contains descriptive error messages.

        Example:
            >>> processor = PythonProcessor()
            >>> result = processor.parse_file("example.py")
            >>> if result.success:
            ...     is_valid, errors = processor.validate_ast_integrity(result.ast_node)
            ...     if not is_valid:
            ...         print(f"AST errors: {errors}")

        Note:
            - Checks that AST can be unparsed using ast.unparse()
            - Verifies the unparsed code can be reparsed using ast.parse()
            - Detects common AST structural issues
            - Logs validation results at debug level
        """
        errors: list[str] = []

        # Check 1: Verify AST can be unparsed
        try:
            code = ast.unparse(ast_node)
            logger.debug(f"Successfully unparsed AST ({len(code)} characters)")
        except Exception as e:
            error_msg = f"Failed to unparse AST: {e.__class__.__name__}: {e}"
            errors.append(error_msg)
            logger.debug(error_msg)
            return (False, errors)

        # Check 2: Verify unparsed code can be reparsed
        try:
            reparsed_ast = ast.parse(code)
            logger.debug("Successfully reparsed unparsed code")
        except SyntaxError as e:
            error_msg = (
                f"Unparsed code has syntax errors at line {e.lineno}, "
                f"column {e.offset}: {e.msg}"
            )
            errors.append(error_msg)
            logger.debug(error_msg)
            return (False, errors)
        except Exception as e:
            error_msg = f"Failed to reparse unparsed code: {e.__class__.__name__}: {e}"
            errors.append(error_msg)
            logger.debug(error_msg)
            return (False, errors)

        # Check 3: Verify module has required attributes (if it's a Module)
        if isinstance(ast_node, ast.Module):
            required_attrs = ['body', 'type_ignores']
            for attr in required_attrs:
                if not hasattr(ast_node, attr):
                    error_msg = f"AST Module missing required attribute: {attr}"
                    errors.append(error_msg)
                    logger.debug(error_msg)

        # Check 4: Verify all nodes have valid types
        class NodeValidator(ast.NodeVisitor):
            """Visitor to validate all nodes in the AST."""

            def __init__(self) -> None:
                self.errors: list[str] = []

            def generic_visit(self, node: ast.AST) -> None:
                # Check for None nodes in lists
                if hasattr(node, 'body'):
                    if None in node.body:  # type: ignore
                        self.errors.append(
                            f"Found None node in body of {type(node).__name__}"
                        )
                super().generic_visit(node)

        try:
            validator = NodeValidator()
            validator.visit(ast_node)
            errors.extend(validator.errors)
        except Exception as e:
            error_msg = f"Error during node validation: {e.__class__.__name__}: {e}"
            errors.append(error_msg)
            logger.debug(error_msg)

        # Log final validation result
        if errors:
            logger.debug(f"AST validation found {len(errors)} issue(s): {errors}")
        else:
            logger.debug("AST validation passed successfully")

        return (len(errors) == 0, errors)


class NameManglingTransformer(ASTTransformer):
    """AST transformer that renames symbols using a pre-computed GlobalSymbolTable.

    This transformer queries the GlobalSymbolTable for mangled names and replaces
    original symbol names in the AST. It tracks scope using a scope stack similar
    to SymbolExtractor to correctly resolve local vs global symbols.

    Attributes:
        global_table: The pre-computed GlobalSymbolTable
        file_path: Path to the file being transformed
        _scope_stack: Stack tracking current scope names
        _class_stack: Stack tracking enclosing class contexts

    Example:
        >>> from pathlib import Path
        >>> transformer = NameManglingTransformer(global_table, Path("example.py"))
        >>> result = transformer.transform(ast_node)
        >>> if result.success:
        ...     print("Transformation successful")

    Limitations and Edge Cases:
        1. Dynamic imports: Cannot track symbols from __import__(), importlib, or exec()
        2. String-based access: Cannot mangle symbols accessed via getattr(obj, "name")
        3. Reflection: Cannot handle inspect module or __dict__ access patterns
        4. Eval/exec: Code in eval() or exec() strings is not analyzed
        5. C extensions: Cannot mangle symbols from compiled extensions
        6. Type annotations: Forward references in quotes may break if mangled
        
    Cross-File Consistency:
        - Requires GlobalSymbolTable to be frozen before transformation
        - Files must be processed in topological order from DependencyGraph
        - Circular dependencies fall back to original file order (may affect consistency)
        
    Preserved Symbols:
        - Python builtins (print, len, str, etc.)
        - Python keywords (if, for, class, def, etc.)
        - Magic methods (__init__, __str__, etc.)
        - Dunder names (__name__, __file__, etc.)
        - Symbols marked with is_exported=True (if preserve_exports config enabled)
        - ALL_CAPS constants (if preserve_constants config enabled)
    """

    def __init__(self, global_table: "GlobalSymbolTable", file_path: Path) -> None:
        """Initialize the name mangling transformer.

        Args:
            global_table: Pre-computed GlobalSymbolTable with mangled names
            file_path: Path to the source file being transformed
        """
        super().__init__()
        self.global_table = global_table
        self.file_path = file_path.resolve() if not file_path.is_absolute() else file_path
        self._scope_stack: list[str] = []
        self._class_stack: list[str] = []
        # Track locally bound identifiers per scope to avoid incorrect global fallback
        # Maps scope depth to set of locally bound names (parameters, assignments, etc.)
        self._local_bindings: dict[int, set[str]] = {}
        # Track import name mappings: original_name -> mangled_name for cross-file refs
        self._import_name_mapping: dict[str, str] = {}
        # Track module-level imports for attribute access resolution
        self._module_imports: set[str] = set()

    def _get_current_scope(self) -> str:
        """Get the current scope for symbol lookup.

        Returns:
            "global" if at module level, "local" otherwise
        """
        return "global" if len(self._scope_stack) == 0 else "local"

    def _try_mangle_name(self, name: str) -> str:
        """Attempt to get mangled name from global table.

        Tries current scope first, then falls back to global scope only if
        the name is not locally bound in the current scope.

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
        # BUT only if the name is not locally bound (e.g., not a parameter)
        if scope == "local":
            # Check if this name is locally bound in the current scope
            scope_depth = len(self._scope_stack)
            is_locally_bound = scope_depth in self._local_bindings and name in self._local_bindings[scope_depth]

            # Only fall back to global if not locally bound
            if not is_locally_bound:
                mangled = self.global_table.get_mangled_name(self.file_path, name, "global")
                if mangled:
                    return mangled

        # Return original name if not found in symbol table
        return name

    def _find_cross_file_mangled_name(self, name: str) -> str | None:
        """Search all files in the global table for a mangled name.

        Looks up a symbol by original_name in global scope across all files.
        Used for resolving imported symbols and attribute accesses.

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

    def visit_Import(self, node: ast.Import) -> ast.Import:
        """Transform Import nodes by tracking module names for attribute resolution.

        Records module names so visit_Attribute can mangle member accesses
        like ``utils.calculate_sum`` when ``calculate_sum`` is a global symbol.

        Args:
            node: Import AST node

        Returns:
            The Import node (unchanged, but module names are tracked)
        """
        for alias in node.names:
            local_name = alias.asname or alias.name
            self._module_imports.add(local_name)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        """Transform ImportFrom nodes by mapping imported names to mangled counterparts.

        For ``from module_a import helper_function``, looks up ``helper_function``
        in the GlobalSymbolTable (using the defining file path and global scope)
        and rewrites the alias name to the mangled identifier. Also tracks the
        mapping so that subsequent Name references are updated consistently.

        Args:
            node: ImportFrom AST node

        Returns:
            Transformed ImportFrom node with mangled import names
        """
        if node.names:
            for alias in node.names:
                if alias.name == '*':
                    continue
                original_name = alias.name
                mangled = self._find_cross_file_mangled_name(original_name)
                if mangled and mangled != original_name:
                    logger.debug(
                        f"Mangling ImportFrom: {original_name} -> {mangled}"
                    )
                    alias.name = mangled
                    if alias.asname is None:
                        self._import_name_mapping[original_name] = mangled
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        """Transform Attribute nodes by mangling attr for global symbol references.

        For ``utils.calculate_sum``, if ``utils`` is an imported module and
        ``calculate_sum`` is a global symbol in the GlobalSymbolTable, rewrites
        ``attr`` to the mangled identifier. Leaves preserved/built-in names intact.

        Args:
            node: Attribute AST node

        Returns:
            Transformed Attribute node with mangled attr
        """
        # Visit the value sub-expression first
        node.value = self.visit(node.value)

        # Only mangle attributes on imported modules
        if isinstance(node.value, ast.Name) and node.value.id in self._module_imports:
            mangled = self._find_cross_file_mangled_name(node.attr)
            if mangled and mangled != node.attr:
                logger.debug(
                    f"Mangling Attribute: {node.attr} -> {mangled}"
                )
                node.attr = mangled

        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Transform Name nodes by replacing with mangled names.

        Checks the import name mapping first (for cross-file from-import
        bindings), then falls back to the normal symbol table lookup.

        Args:
            node: Name AST node

        Returns:
            Transformed Name node with mangled id
        """
        # Check import mapping first for cross-file from-import bindings
        if node.id in self._import_name_mapping:
            mangled = self._import_name_mapping[node.id]
            logger.debug(f"Mangling Name (import mapping): {node.id} -> {mangled}")
            node.id = mangled
            return node

        mangled = self._try_mangle_name(node.id)
        if mangled != node.id:
            logger.debug(f"Mangling Name: {node.id} -> {mangled}")
            node.id = mangled
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Transform FunctionDef nodes by renaming function and traversing body.

        Args:
            node: FunctionDef AST node

        Returns:
            Transformed FunctionDef node
        """
        # Mangle function name
        mangled = self._try_mangle_name(node.name)
        if mangled != node.name:
            logger.debug(f"Mangling FunctionDef: {node.name} -> {mangled}")
            node.name = mangled

        # Push scope
        self._scope_stack.append(node.name)
        scope_depth = len(self._scope_stack)

        # Track function parameters as locally bound identifiers
        self._local_bindings[scope_depth] = set()
        for arg in node.args.args:
            if arg.arg and arg.arg != "self":  # Skip 'self' parameter
                self._local_bindings[scope_depth].add(arg.arg)

        # Traverse body
        self.generic_visit(node)

        # Clean up scope
        self._local_bindings.pop(scope_depth, None)
        self._scope_stack.pop()

        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Transform AsyncFunctionDef nodes by renaming function and traversing body.

        Args:
            node: AsyncFunctionDef AST node

        Returns:
            Transformed AsyncFunctionDef node
        """
        # Mangle function name
        mangled = self._try_mangle_name(node.name)
        if mangled != node.name:
            logger.debug(f"Mangling AsyncFunctionDef: {node.name} -> {mangled}")
            node.name = mangled

        # Push scope
        self._scope_stack.append(node.name)
        scope_depth = len(self._scope_stack)

        # Track function parameters as locally bound identifiers
        self._local_bindings[scope_depth] = set()
        for arg in node.args.args:
            if arg.arg and arg.arg != "self":  # Skip 'self' parameter
                self._local_bindings[scope_depth].add(arg.arg)

        # Traverse body
        self.generic_visit(node)

        # Clean up scope
        self._local_bindings.pop(scope_depth, None)
        self._scope_stack.pop()

        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Transform ClassDef nodes by renaming class and traversing body.

        Args:
            node: ClassDef AST node

        Returns:
            Transformed ClassDef node
        """
        # Mangle class name
        mangled = self._try_mangle_name(node.name)
        if mangled != node.name:
            logger.debug(f"Mangling ClassDef: {node.name} -> {mangled}")
            node.name = mangled

        # Push scope and class context, then traverse body
        self._scope_stack.append(node.name)
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()
        self._scope_stack.pop()

        return node
