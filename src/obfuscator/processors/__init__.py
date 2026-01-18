"""Python and Lua source code processing module.

This module provides the processors package for parsing source files
into Abstract Syntax Trees (AST), generating code from AST nodes, extracting
symbol information for dependency analysis, applying AST transformations
for code obfuscation, and detecting unsupported features.

Classes:
    PythonProcessor: Main processor class for parsing, code generation, and transformations
    SymbolExtractor: AST visitor for extracting symbol information
    ASTTransformer: Base class for AST transformations
    ConstantFoldingTransformer: Example transformer for constant folding
    UnsupportedFeatureDetector: Detector for unsupported or problematic Python features
    LuaProcessor: Main processor class for Lua parsing and code generation
    LuaSymbolExtractor: AST visitor for extracting Lua symbol information
    LuaFeatureDetector: Detector for unsupported or problematic Lua features

Data Classes:
    SymbolTable: Complete symbol information for a Python file
    ImportInfo: Information about an import statement
    FunctionInfo: Information about a function definition
    ClassInfo: Information about a class definition
    VariableInfo: Information about a variable assignment
    TransformResult: Result of applying a single AST transformer
    TransformationPipelineResult: Result of applying multiple transformers
    FeatureWarning: Warning about an unsupported or problematic feature
    LuaSymbolTable: Complete symbol information for a Lua file
    LuaImportInfo: Information about a Lua require() call
    LuaFunctionInfo: Information about a Lua function definition
    LuaVariableInfo: Information about a Lua variable assignment
    LuaParseResult: Result of parsing a Lua source file
    LuaGenerateResult: Result of generating Lua code from an AST

Example:
    >>> from obfuscator.processors import PythonProcessor
    >>> processor = PythonProcessor()
    >>> result = processor.parse_file("example.py")
    >>> if result.success:
    ...     # Generate code
    ...     generated = processor.generate_code(result.ast_node)
    ...     # Extract symbols
    ...     symbols = processor.extract_symbols(result.ast_node, result.file_path)
    ...     print(f"Found {len(symbols.functions)} functions")
    ...     # Apply transformations
    ...     from obfuscator.processors import ConstantFoldingTransformer
    ...     transformers = [ConstantFoldingTransformer()]
    ...     transform_result = processor.apply_transformations(
    ...         result.ast_node, transformers
    ...     )
    ...     # Check for warnings
    ...     for warning in result.warnings:
    ...         print(f"Warning: {warning.feature_name} at line {warning.line_number}")
"""

from obfuscator.processors.ast_transformer import (
    ASTTransformer,
    ConstantFoldingTransformer,
    TransformResult,
)
from obfuscator.processors.feature_detector import (
    FeatureWarning,
    UnsupportedFeatureDetector,
)
from obfuscator.processors.lua_feature_detector import LuaFeatureDetector
from obfuscator.processors.lua_processor import (
    LuaProcessor,
    ParseResult as LuaParseResult,
    GenerateResult as LuaGenerateResult,
)
from obfuscator.processors.lua_symbol_extractor import (
    LuaSymbolExtractor,
    LuaSymbolTable,
    LuaImportInfo,
    LuaFunctionInfo,
    LuaVariableInfo,
)
from obfuscator.processors.python_processor import (
    TransformationPipelineResult,
    PythonProcessor,
)
from obfuscator.processors.symbol_extractor import (
    ClassInfo,
    FunctionInfo,
    ImportInfo,
    SymbolExtractor,
    SymbolTable,
    VariableInfo,
)

__all__ = [
    "PythonProcessor",
    "SymbolExtractor",
    "SymbolTable",
    "ImportInfo",
    "FunctionInfo",
    "ClassInfo",
    "VariableInfo",
    "ASTTransformer",
    "TransformResult",
    "ConstantFoldingTransformer",
    "TransformationPipelineResult",
    "UnsupportedFeatureDetector",
    "LuaProcessor",
    "LuaParseResult",
    "LuaGenerateResult",
    "LuaSymbolExtractor",
    "LuaSymbolTable",
    "LuaImportInfo",
    "LuaFunctionInfo",
    "LuaVariableInfo",
    "LuaFeatureDetector",
    "FeatureWarning",
]
