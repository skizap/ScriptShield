"""Coordinated obfuscation engine for applying transformation pipelines.

This module provides the ObfuscationEngine class that reads feature flags from
ObfuscationConfig, instantiates enabled transformers in the correct dependency
order, and provides a unified interface for applying all transformations.

The transformation order is:
1. StringEncryptionTransformer (string_encryption)
2. NumberObfuscationTransformer (number_obfuscation)
3. ConstantArrayTransformer (constant_array)
4. ControlFlowFlatteningTransformer (control_flow_flattening)
5. DeadCodeInjectionTransformer (dead_code_injection)
6. OpaquePredicatesTransformer (opaque_predicates)
7. AntiDebuggingTransformer (anti_debugging)
8. RobloxExploitDefenseTransformer (roblox_exploit_defense)
9. RobloxRemoteSpyTransformer (roblox_remote_spy)
10. CodeSplittingTransformer (code_splitting)
11. SelfModifyingCodeTransformer (self_modifying_code)
12. MangleIndexesTransformer (mangle_indexes)
13. VMProtectionTransformer (vm_protection)

This ensures that value-level obfuscation (strings, numbers, arrays) happens
before structural changes (index mangling, VM wrapping).

Example:
    >>> from obfuscator.core.config import ObfuscationConfig
    >>> from obfuscator.core.obfuscation_engine import ObfuscationEngine
    >>> config = ObfuscationConfig(
    ...     name="full",
    ...     features={"string_encryption": True, "number_obfuscation": True},
    ... )
    >>> engine = ObfuscationEngine(config)
    >>> transformers = engine.get_enabled_transformers("python")
    >>> result = engine.apply_transformations(ast_node, "python", file_path)
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional

from obfuscator.core.config import ObfuscationConfig
from obfuscator.core.runtime_manager import RuntimeManager
from obfuscator.processors.ast_transformer import (
    AntiDebuggingTransformer,
    ASTTransformer,
    CodeSplittingTransformer,
    ConstantArrayTransformer,
    SelfModifyingCodeTransformer,
    ControlFlowFlatteningTransformer,
    DeadCodeInjectionTransformer,
    MangleIndexesTransformer,
    NumberObfuscationTransformer,
    OpaquePredicatesTransformer,
    RobloxExploitDefenseTransformer,
    RobloxRemoteSpyTransformer,
    StringEncryptionTransformer,
    TransformResult,
    VMProtectionTransformer,
)
from obfuscator.processors.python_processor import PythonProcessor
from obfuscator.processors.lua_processor import LuaProcessor
from obfuscator.utils.logger import get_logger

logger = get_logger("obfuscator.core.obfuscation_engine")


class ObfuscationEngine:
    """Centralized engine for coordinating AST transformation pipelines.

    Reads feature flags from an ObfuscationConfig, instantiates enabled
    transformers in the correct dependency order, and provides a unified
    interface for applying all transformations to Python or Lua ASTs.

    Attributes:
        config: The ObfuscationConfig controlling which features are enabled.
        runtime_manager: The RuntimeManager instance for runtime code generation.
        required_runtimes: Set of feature names that require runtime code.

    Example:
        >>> engine = ObfuscationEngine(config)
        >>> transformers = engine.get_enabled_transformers("python")
        >>> result = engine.apply_transformations(ast_node, "python", path)
        >>> if result.success:
        ...     print(f"Applied {result.transformation_count} transformations")
    """

    # Set of feature flags that require runtime code generation
    _RUNTIME_REQUIRING_FEATURES: set[str] = {
        "vm_protection",
        "code_splitting",
        "self_modifying_code",
        "anti_debugging",
        "roblox_exploit_defense",
        "roblox_remote_spy",
    }
    # The order defines the transformation pipeline sequence.
    _TRANSFORMER_ORDER: list[tuple[str, type[ASTTransformer]]] = [
        ("string_encryption", StringEncryptionTransformer),
        ("number_obfuscation", NumberObfuscationTransformer),
        ("constant_array", ConstantArrayTransformer),
        ("control_flow_flattening", ControlFlowFlatteningTransformer),
        ("dead_code_injection", DeadCodeInjectionTransformer),
        ("opaque_predicates", OpaquePredicatesTransformer),
        ("anti_debugging", AntiDebuggingTransformer),
        ("roblox_exploit_defense", RobloxExploitDefenseTransformer),
        ("roblox_remote_spy", RobloxRemoteSpyTransformer),
        ("code_splitting", CodeSplittingTransformer),
        ("self_modifying_code", SelfModifyingCodeTransformer),
        ("mangle_indexes", MangleIndexesTransformer),
        ("vm_protection", VMProtectionTransformer),
    ]

    def __init__(self, config: ObfuscationConfig) -> None:
        """Initialize the obfuscation engine.

        Args:
            config: ObfuscationConfig instance with feature flags and options.
        """
        self.config = config
        self.runtime_manager = RuntimeManager(config)
        self.required_runtimes: set[str] = set()
        logger.debug(
            f"ObfuscationEngine initialized with config '{config.name}', "
            f"features: {config.features}"
        )

    def get_enabled_transformers(self, language: str | None = None) -> list[ASTTransformer]:
        """Return transformer instances for all enabled features.

        Checks feature flags in ``config.features``, instantiates transformers
        for enabled features in the documented pipeline order, passes the
        config to each constructor, and filters by language compatibility.

        Args:
            language: Target language (``"python"`` or ``"lua"``).
                     If not provided, falls back to ``self.config.language``.

        Returns:
            Ordered list of transformer instances ready to apply.
        """
        # Use provided language or fall back to config language
        target_language = language or self.config.language
        if not target_language:
            logger.warning("No language specified and config.language is not set")
            return []
        
        transformers: list[ASTTransformer] = []

        for feature_flag, transformer_cls in self._TRANSFORMER_ORDER:
            # Check if the feature is enabled in config
            if not self.config.features.get(feature_flag, False):
                continue

            # Instantiate the transformer with config and runtime_manager
            try:
                transformer = transformer_cls(config=self.config, runtime_manager=self.runtime_manager)
            except Exception as e:
                logger.warning(
                    f"Failed to instantiate {transformer_cls.__name__}: {e}"
                )
                continue

            # Filter by language compatibility.
            # Transformers that auto-detect language from AST type are
            # compatible with both; those with an explicit language_mode
            # set at init time are checked here.
            transformer_language = getattr(transformer, "language_mode", None)
            if transformer_language is not None and transformer_language != target_language:
                logger.debug(
                    f"Skipping {transformer_cls.__name__} "
                    f"(language_mode={transformer_language}, target={target_language})"
                )
                continue

            transformers.append(transformer)
            logger.debug(
                f"Enabled transformer: {transformer_cls.__name__} "
                f"for feature '{feature_flag}'"
            )

        logger.info(
            f"Built transformation pipeline with {len(transformers)} "
            f"transformer(s) for language '{target_language}'"
        )
        return transformers

    def apply_transformations(
        self,
        ast_node: Any,
        language: str,
        file_path: Path,
    ) -> TransformResult:
        """Apply the full transformation pipeline to an AST.

        Gets enabled transformers for the specified language and applies them
        sequentially. If no transformers are enabled, returns success with the
        original AST. If any transformer fails, stops the pipeline and returns
        the error.

        Args:
            ast_node: The AST node to transform (ast.Module or lua Chunk).
            language: Target language (``"python"`` or ``"lua"``).
            file_path: Path to the source file (for logging context).

        Returns:
            TransformResult with the final AST, success status, cumulative
            transformation count, and any errors.
        """
        # Reset runtime tracking at the start of each transformation pipeline
        self.required_runtimes = set()

        transformers = self.get_enabled_transformers(language)

        # No transformers enabled → success with original AST
        if not transformers:
            logger.debug(
                f"No transformers enabled for {file_path.name}; "
                "returning original AST"
            )
            return TransformResult(
                ast_node=ast_node,
                success=True,
                transformation_count=0,
                errors=[],
            )

        logger.info(
            f"Applying {len(transformers)} transformer(s) to {file_path.name}"
        )

        current_ast = ast_node
        total_count = 0
        all_errors: list[str] = []

        for idx, transformer in enumerate(transformers):
            name = type(transformer).__name__
            logger.debug(
                f"Applying transformer {idx + 1}/{len(transformers)}: {name}"
            )

            result = transformer.transform(current_ast)

            if not result.success:
                error_msg = (
                    f"Transformer {name} failed on {file_path.name}: "
                    f"{', '.join(result.errors)}"
                )
                logger.error(error_msg)
                all_errors.extend(result.errors)
                return TransformResult(
                    ast_node=None,
                    success=False,
                    transformation_count=total_count,
                    errors=all_errors,
                )

            current_ast = result.ast_node
            total_count += result.transformation_count

            # Check if this transformer requires runtime code
            # Map transformer class name to feature flag
            transformer_feature = None
            for feature_flag, transformer_cls in self._TRANSFORMER_ORDER:
                if isinstance(transformer, transformer_cls):
                    transformer_feature = feature_flag
                    break

            if transformer_feature and transformer_feature in self._RUNTIME_REQUIRING_FEATURES:
                self.required_runtimes.add(transformer_feature)
                logger.debug(f"Feature '{transformer_feature}' requires runtime code")

            logger.debug(
                f"{name} completed: {result.transformation_count} nodes transformed"
            )

        logger.info(
            f"Transformation pipeline completed for {file_path.name}: "
            f"{total_count} total transformations"
        )

        return TransformResult(
            ast_node=current_ast,
            success=True,
            transformation_count=total_count,
            errors=all_errors,
        )

    def obfuscate_string(self, source_code: str, language: str | None = None) -> dict[str, Any]:
        """Obfuscate source code provided as a string.

        Parses the source code, applies all enabled transformations, and generates
        obfuscated code. This is a convenience method for testing and quick
        obfuscation without file I/O.

        Args:
            source_code: The source code string to obfuscate.
            language: Target language (``"python"`` or ``"lua"``).
                     If not provided, falls back to ``self.config.language``.

        Returns:
            Dictionary containing:
            - ``success``: Boolean indicating if obfuscation succeeded
            - ``code``: The obfuscated code (empty string if failed)
            - ``stats``: Dictionary with transformation statistics
            - ``errors``: List of error messages (if any)
        """
        target_language = language or self.config.language
        if not target_language:
            return {
                "success": False,
                "code": "",
                "stats": {},
                "errors": ["No language specified and config.language is not set"],
            }

        # Parse source code based on language
        if target_language == "python":
            try:
                ast_node = ast.parse(source_code)
            except SyntaxError as e:
                error_msg = f"Syntax error: {e}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "code": "",
                    "stats": {},
                    "errors": [error_msg],
                }
            processor = PythonProcessor(config=self.config)
        elif target_language == "lua":
            processor = LuaProcessor(config=self.config)
            try:
                # Try to import luaparser dynamically
                from luaparser import ast as lua_ast
                ast_node = lua_ast.parse(source_code)
            except ImportError:
                return {
                    "success": False,
                    "code": "",
                    "stats": {},
                    "errors": ["luaparser module not installed. Install with: pip install luaparser"],
                }
            except Exception as e:
                error_msg = f"Lua parse error: {e}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "code": "",
                    "stats": {},
                    "errors": [error_msg],
                }
        else:
            return {
                "success": False,
                "code": "",
                "stats": {},
                "errors": [f"Unsupported language: {target_language}"],
            }

        # Apply transformations using a dummy file path
        dummy_path = Path(f"<string>.{target_language}")
        result = self.apply_transformations(ast_node, target_language, dummy_path)

        if not result.success:
            return {
                "success": False,
                "code": "",
                "stats": {"transformation_count": result.transformation_count},
                "errors": result.errors,
            }

        # For Python, embed runtime in AST before generating code (consistent with processor)
        final_ast = result.ast_node
        if target_language == "python" and self.config.runtime_mode == "embedded" and self.has_runtime_requirements():
            final_ast = self.embed_runtime_in_ast(result.ast_node, target_language)
            logger.debug("Embedded runtime code into Python AST before code generation")

        # Generate code from transformed AST
        gen_result = processor.generate_code(final_ast)

        if not gen_result.success:
            return {
                "success": False,
                "code": "",
                "stats": {"transformation_count": result.transformation_count},
                "errors": gen_result.errors,
            }

        # For Lua or fallback, inject embedded runtime code at string level if needed
        final_code = gen_result.code
        if target_language == "lua" and self.config.runtime_mode == "embedded" and self.has_runtime_requirements():
            runtime_code = self.prepare_embedded_runtime(target_language)
            if runtime_code:
                final_code = runtime_code + final_code
                logger.info(f"Embedded runtime code injected ({len(runtime_code)} chars)")

        return {
            "success": True,
            "code": final_code,
            "stats": {
                "transformation_count": result.transformation_count,
            },
            "errors": [],
        }

    def has_runtime_requirements(self) -> bool:
        """Check if any runtime-requiring features were applied.

        Returns:
            True if runtime code is required, False otherwise.
        """
        return len(self.required_runtimes) > 0

    def get_required_runtime_code(self, language: str) -> str:
        """Get combined runtime code for all required features.

        Checks if the config is in embedded mode and if runtime
        requirements exist, then generates combined runtime code
        ONLY for the features that were actually applied during transformations
        (tracked in required_runtimes), not all enabled features.

        Args:
            language: Target language ("python" or "lua").

        Returns:
            Combined runtime code string, or empty string if no runtime needed.
        """
        if self.config.runtime_mode != "embedded":
            return ""

        if not self.required_runtimes:
            return ""

        try:
            # Generate runtime code ONLY for features that were actually applied
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

            # Generate runtime code only for required features (those actually applied)
            for i, runtime_type in enumerate(self.required_runtimes):
                code = self.runtime_manager.generate_runtime_code(runtime_type, language)
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
                    logger.debug(f"Generated runtime code for {runtime_type} ({language})")

            combined_code = '\n'.join(parts)
            logger.info(
                f"Generated runtime code for {len(self.required_runtimes)} applied features "
                f"({len(combined_code)} chars)"
            )
            return combined_code

        except Exception as e:
            logger.warning(f"Failed to generate runtime code: {e}")
            return ""

    def embed_runtime_in_ast(self, ast_node: Any, language: str) -> Any:
        """Embed runtime code at the top of an AST.

        Generates runtime code for applied features and prepends it as
        executable AST statements (Python) or returns AST unchanged (Lua)
        for string-level injection.

        Args:
            ast_node: The AST node to embed runtime into (ast.Module or lua Chunk).
            language: Target language ("python" or "lua").

        Returns:
            AST node with embedded runtime code, or original AST if no runtime needed.
        """
        if self.config.runtime_mode != "embedded":
            return ast_node

        if not self.required_runtimes:
            return ast_node

        try:
            runtime_code = self.get_required_runtime_code(language)
            if not runtime_code:
                return ast_node

            if language == "python" and isinstance(ast_node, ast.Module):
                # Parse runtime code into executable AST statements
                try:
                    runtime_ast = ast.parse(runtime_code)
                    if isinstance(runtime_ast, ast.Module) and runtime_ast.body:
                        # Insert runtime statements at the beginning of the module body
                        new_body = list(runtime_ast.body) + list(ast_node.body)
                        ast_node.body = new_body
                        logger.debug(f"Embedded runtime code into Python AST as executable statements ({len(runtime_code)} chars)")
                    else:
                        logger.warning("Parsed runtime code did not produce valid module body")
                except SyntaxError as e:
                    logger.warning(f"Failed to parse runtime code into AST: {e}")
                    return ast_node

            elif language == "lua":
                # For Lua, we can't easily modify the AST to add runtime code
                # So we return the AST unchanged - runtime will be prepended at code generation
                logger.debug("Lua runtime code will be prepended at code generation time")
                return ast_node

            return ast_node

        except Exception as e:
            logger.warning(f"Failed to embed runtime in AST: {e}")
            return ast_node

    def prepare_embedded_runtime(self, language: str) -> str:
        """Prepare runtime code for embedding into obfuscated files.

        Formats the runtime code with language-specific headers and separators
        for clear separation from the obfuscated code.

        Args:
            language: Target language ("python" or "lua").

        Returns:
            Formatted runtime code block with headers, or empty string if none.
        """
        runtime_code = self.get_required_runtime_code(language)

        if not runtime_code:
            return ""

        if language == "python":
            header = "# ============ EMBEDDED RUNTIME CODE ============\n"
            footer = "\n# ============ END RUNTIME CODE ============\n\n"
        elif language == "lua":
            header = "-- ============ EMBEDDED RUNTIME CODE ============\n"
            footer = "\n-- ============ END RUNTIME CODE ============\n\n"
        else:
            # Fallback for unsupported languages
            return runtime_code + "\n\n"

        return header + runtime_code + footer
