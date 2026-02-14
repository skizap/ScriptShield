"""Public API for source processors with lazy imports.

This package intentionally avoids eager imports of heavy processor modules to
reduce startup overhead and prevent circular-import chains during module
initialization (for example, when only runtime helpers are needed).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "PythonProcessor": (
        "obfuscator.processors.python_processor",
        "PythonProcessor",
    ),
    "TransformationPipelineResult": (
        "obfuscator.processors.python_processor",
        "TransformationPipelineResult",
    ),
    "SymbolExtractor": (
        "obfuscator.processors.symbol_extractor",
        "SymbolExtractor",
    ),
    "SymbolTable": (
        "obfuscator.processors.symbol_extractor",
        "SymbolTable",
    ),
    "ImportInfo": (
        "obfuscator.processors.symbol_extractor",
        "ImportInfo",
    ),
    "FunctionInfo": (
        "obfuscator.processors.symbol_extractor",
        "FunctionInfo",
    ),
    "ClassInfo": (
        "obfuscator.processors.symbol_extractor",
        "ClassInfo",
    ),
    "VariableInfo": (
        "obfuscator.processors.symbol_extractor",
        "VariableInfo",
    ),
    "ASTTransformer": (
        "obfuscator.processors.ast_transformer",
        "ASTTransformer",
    ),
    "TransformResult": (
        "obfuscator.processors.ast_transformer",
        "TransformResult",
    ),
    "ConstantFoldingTransformer": (
        "obfuscator.processors.ast_transformer",
        "ConstantFoldingTransformer",
    ),
    "SelfModifyingCodeTransformer": (
        "obfuscator.processors.ast_transformer",
        "SelfModifyingCodeTransformer",
    ),
    "UnsupportedFeatureDetector": (
        "obfuscator.processors.feature_detector",
        "UnsupportedFeatureDetector",
    ),
    "FeatureWarning": (
        "obfuscator.processors.feature_detector",
        "FeatureWarning",
    ),
    "LuaProcessor": (
        "obfuscator.processors.lua_processor",
        "LuaProcessor",
    ),
    "LuaParseResult": (
        "obfuscator.processors.lua_processor",
        "ParseResult",
    ),
    "LuaGenerateResult": (
        "obfuscator.processors.lua_processor",
        "GenerateResult",
    ),
    "LuaSymbolExtractor": (
        "obfuscator.processors.lua_symbol_extractor",
        "LuaSymbolExtractor",
    ),
    "LuaSymbolTable": (
        "obfuscator.processors.lua_symbol_extractor",
        "LuaSymbolTable",
    ),
    "LuaImportInfo": (
        "obfuscator.processors.lua_symbol_extractor",
        "LuaImportInfo",
    ),
    "LuaFunctionInfo": (
        "obfuscator.processors.lua_symbol_extractor",
        "LuaFunctionInfo",
    ),
    "LuaVariableInfo": (
        "obfuscator.processors.lua_symbol_extractor",
        "LuaVariableInfo",
    ),
    "LuaFeatureDetector": (
        "obfuscator.processors.lua_feature_detector",
        "LuaFeatureDetector",
    ),
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str) -> Any:
    """Resolve package exports lazily."""
    target = _EXPORT_MAP.get(name)
    if target is None:
        raise AttributeError(f"module 'obfuscator.processors' has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
