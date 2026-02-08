"""Unsupported feature detector for Lua code obfuscation.

This module provides detection of Lua language features that can interfere
with obfuscation or cause issues with static analysis. The detector traverses
the luaparser AST and identifies problematic constructs like dynamic code execution,
metatable manipulation, global namespace access, and Roblox-specific API patterns.

Example:
    >>> from obfuscator.processors.lua_feature_detector import LuaFeatureDetector
    >>> from luaparser import ast
    >>> from pathlib import Path
    >>> tree = ast.parse('loadstring(user_input)()')
    >>> detector = LuaFeatureDetector(Path("example.lua"))
    >>> detector.visit(tree)
    >>> warnings = detector.get_warnings()
    >>> for warning in warnings:
    ...     print(f"{warning.severity}: {warning.feature_name} at line {warning.line_number}")

Detected Patterns:
    - Dynamic code execution: loadstring(), load(), dofile(), loadfile()
    - Metatable manipulation: setmetatable(), getmetatable(), rawget(), rawset()
    - Global namespace access: _G table access and manipulation
    - Environment manipulation: getfenv(), setfenv() (Lua 5.1)
    - Debug introspection: debug.getinfo(), debug.getlocal(), debug.setlocal()
    - Roblox API patterns: game:GetService(), Instance.new(), etc.

Severity Levels:
    - critical: Features that completely prevent obfuscation
    - error: Features that severely interfere with analysis
    - warning: Features that complicate obfuscation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from luaparser import ast, astnodes

from obfuscator.processors.feature_detector import FeatureWarning
from obfuscator.utils.logger import get_logger

logger = get_logger("obfuscator.processors.lua_feature_detector")


class LuaFeatureDetector(ast.ASTVisitor):
    """Detector for unsupported or problematic Lua language features.

    This class extends luaparser's ASTVisitor to traverse Lua ASTs and identify
    language features that can interfere with obfuscation or static analysis.
    It detects dynamic code execution, metatable manipulation, global namespace
    access, environment manipulation, and Roblox API patterns.

    Attributes:
        file_path: Path to the source file being analyzed
        warnings: List of FeatureWarning objects collected during traversal

    Example:
        >>> from obfuscator.processors.lua_feature_detector import LuaFeatureDetector
        >>> from luaparser import ast
        >>> from pathlib import Path
        >>>
        >>> code = '''
        ... local function foo()
        ...     local code = loadstring(user_input)
        ...     setmetatable(t, mt)
        ... end
        ... '''
        >>> tree = ast.parse(code)
        >>> detector = LuaFeatureDetector(Path("example.lua"))
        >>> detector.visit(tree)
        >>>
        >>> warnings = detector.get_warnings()
        >>> print(f"Found {len(warnings)} warnings")
        >>> for warning in warnings:
        ...     print(f"Line {warning.line_number}: {warning.feature_name}")
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize the Lua feature detector.

        Args:
            file_path: Path to the source file being analyzed
        """
        self.file_path = file_path
        self.warnings: list[FeatureWarning] = []
        self._scope_stack: list[str] = []

    def visit_Call(self, node: astnodes.Call) -> None:
        """Visit function call nodes to detect dangerous function calls.

        Detects calls to functions that perform dynamic code execution,
        metatable manipulation, environment manipulation, or debug introspection
        that can interfere with obfuscation and static analysis.

        Args:
            node: The Call AST node to analyze
        """
        func_name = self._extract_call_name(node)

        if func_name is None:
            self.generic_visit(node)
            return

        # Critical: Dynamic code execution (Lua 5.1 and 5.2+)
        if func_name == "loadstring":
            self._add_warning(
                node,
                feature_name="loadstring()",
                description="Dynamic code execution prevents static analysis and obfuscation",
                severity="critical",
                suggestion="Avoid using loadstring(); consider redesigning to use tables, callbacks, or configuration files"
            )
        elif func_name == "load":
            self._add_warning(
                node,
                feature_name="load()",
                description="Dynamic code execution prevents static analysis and obfuscation",
                severity="critical",
                suggestion="Avoid using load(); consider redesigning to use tables, callbacks, or configuration files"
            )
        elif func_name == "dofile":
            self._add_warning(
                node,
                feature_name="dofile()",
                description="Dynamic file execution prevents static analysis and obfuscation",
                severity="critical",
                suggestion="Avoid using dofile(); use require() with static module paths instead"
            )
        elif func_name == "loadfile":
            self._add_warning(
                node,
                feature_name="loadfile()",
                description="Dynamic file loading prevents static analysis and obfuscation",
                severity="critical",
                suggestion="Avoid using loadfile(); use require() with static module paths instead"
            )

        # Error: Metatable manipulation
        elif func_name == "setmetatable":
            self._add_warning(
                node,
                feature_name="setmetatable()",
                description="Metatable manipulation interferes with type analysis and symbol tracking",
                severity="error",
                suggestion="Document metatable usage; consider simpler inheritance patterns"
            )
        elif func_name == "getmetatable":
            self._add_warning(
                node,
                feature_name="getmetatable()",
                description="Metatable introspection interferes with type analysis",
                severity="error",
                suggestion="Avoid relying on metatable introspection for program logic"
            )
        elif func_name in ("rawget", "rawset", "rawequal"):
            self._add_warning(
                node,
                feature_name=f"{func_name}()",
                description="Bypassing metamethods interferes with consistent behavior analysis",
                severity="error",
                suggestion="Use standard table access unless metamethod bypass is absolutely necessary"
            )

        # Continued in next section...
        self._detect_environment_and_debug_patterns(node, func_name)
        self._detect_dynamic_require(node, func_name)
        self._detect_roblox_api_patterns(node, func_name)

        self.generic_visit(node)

    def _detect_environment_and_debug_patterns(
        self, node: astnodes.Call, func_name: str
    ) -> None:
        """Detect environment manipulation and debug introspection patterns.

        Args:
            node: The Call AST node
            func_name: Extracted function name
        """
        # Warning: Environment manipulation (Lua 5.1)
        if func_name in ("getfenv", "setfenv"):
            self._add_warning(
                node,
                feature_name=f"{func_name}()",
                description="Environment manipulation interferes with scope analysis",
                severity="warning",
                suggestion="Avoid environment manipulation; use explicit module patterns instead"
            )

        # Warning: Debug introspection
        elif func_name.startswith("debug:") or func_name.startswith("debug."):
            debug_func = func_name.split(":" if ":" in func_name else ".")[-1]
            if debug_func in ("getinfo", "getlocal", "setlocal", "getupvalue", "setupvalue"):
                self._add_warning(
                    node,
                    feature_name=f"{func_name}()",
                    description="Debug introspection interferes with variable analysis and obfuscation",
                    severity="warning",
                    suggestion="Avoid debug library for production code; use explicit parameter passing"
                )

    def _detect_dynamic_require(self, node: astnodes.Call, func_name: str) -> None:
        """Detect require() calls with dynamic arguments.

        Args:
            node: The Call AST node
            func_name: Extracted function name
        """
        if func_name == "require" and node.args:
            arg = node.args[0]
            # Check if the argument is not a string literal
            if not isinstance(arg, astnodes.String):
                self._add_warning(
                    node,
                    feature_name="require() with dynamic path",
                    description="Dynamic module path prevents static dependency analysis",
                    severity="warning",
                    suggestion="Use static string literals for module paths in require()"
                )

    def _detect_roblox_api_patterns(self, node: astnodes.Call, func_name: str) -> None:
        """Detect Roblox API patterns that should be preserved.

        Args:
            node: The Call AST node
            func_name: Extracted function name
        """
        if self._is_roblox_preserved_api(func_name, node.args):
            # Check if it's a remote method call that would benefit from remote spy protection
            is_remote_method = any(
                func_name and (func_name.endswith(f":{m}") or func_name.endswith(f".{m}"))
                for m in ["FireServer", "FireClient", "InvokeServer", "InvokeClient"]
            )
            
            if is_remote_method:
                self._add_warning(
                    node,
                    feature_name=f"Roblox Remote API: {func_name}",
                    description="RemoteEvent/RemoteFunction call detected; consider enabling roblox_remote_spy feature for protection",
                    severity="warning",
                    suggestion="Enable 'roblox_remote_spy' feature in obfuscation config to protect remote calls from interception"
                )
            else:
                self._add_warning(
                    node,
                    feature_name=f"Roblox API: {func_name}",
                    description="Roblox API pattern detected; these names should be preserved during obfuscation",
                    severity="warning",
                    suggestion="Verify that Roblox API names and service strings are in the preservation list"
                )

    def visit_Index(self, node: astnodes.Index) -> None:
        """Visit index nodes to detect _G global table access.

        Args:
            node: The Index AST node to analyze
        """
        # Check if accessing _G table
        # Use table/field (luaparser 4.x) with fallback to value/idx for compatibility
        table_node = getattr(node, "table", None) or getattr(node, "value", None)
        field_node = getattr(node, "field", None) or getattr(node, "idx", None)

        if table_node and isinstance(table_node, astnodes.Name):
            if getattr(table_node, "id", None) == "_G":
                field_name = self._extract_string_value(field_node) if field_node else "<dynamic>"
                self._add_warning(
                    node,
                    feature_name="_G access",
                    description=f"Global table access (_G[{field_name!r}]) interferes with symbol tracking",
                    severity="warning",
                    suggestion="Use explicit variable references instead of _G table access"
                )

        self.generic_visit(node)

    def visit_Name(self, node: astnodes.Name) -> None:
        """Visit name nodes to detect direct _G references.

        Args:
            node: The Name AST node to analyze
        """
        if getattr(node, "id", None) == "_G":
            # Check context - if it's just a reference to _G itself
            self._add_warning(
                node,
                feature_name="_G reference",
                description="Direct reference to global table (_G) may interfere with symbol tracking",
                severity="warning",
                suggestion="Avoid using _G directly; use explicit variable declarations"
            )

        self.generic_visit(node)

    def _is_roblox_preserved_api(self, call_name: str, args: list[Any]) -> bool:
        """Check if a function call matches Roblox API patterns that should be preserved.

        Args:
            call_name: Name of the function being called
            args: List of call arguments

        Returns:
            True if the call matches a Roblox API pattern that should be preserved
        """
        # Check for game:GetService()
        if call_name and (call_name.startswith("game:GetService") or call_name.startswith("game.GetService")):
            return True

        # Check for Instance.new() or Instance:new()
        if call_name in ("Instance.new", "Instance:new"):
            return True

        # Check for RemoteEvent/RemoteFunction method calls (FireServer, FireClient, InvokeServer, InvokeClient)
        if call_name and any(call_name.endswith(f":{m}") or call_name.endswith(f".{m}") 
                            for m in ["FireServer", "FireClient", "InvokeServer", "InvokeClient"]):
            return True

        # Check for common Roblox services in arguments
        roblox_services = [
            "Players", "Workspace", "ReplicatedStorage", "ServerScriptService",
            "ServerStorage", "StarterGui", "StarterPack", "StarterPlayer",
            "TweenService", "UserInputService", "RunService", "Debris",
            "HttpService", "MarketplaceService", "SoundService", "Lighting"
        ]

        roblox_instance_types = [
            "RemoteEvent", "RemoteFunction", "BindableEvent", "BindableFunction",
            "Part", "Model", "Folder", "StringValue", "IntValue", "BoolValue"
        ]

        if args:
            arg_value = self._extract_string_value(args[0])
            if arg_value in roblox_services or arg_value in roblox_instance_types:
                return True

        return False

    def _extract_call_name(self, node: astnodes.Call) -> str | None:
        """Extract the function name from a Call node.

        Handles different call patterns:
        - Simple calls: `func()`
        - Method calls: `obj:method()`
        - Index calls: `table.func()`
        - Nested calls: `module.submodule.func()`

        Args:
            node: The Call AST node

        Returns:
            String representation of the function name, or None if name
            cannot be extracted
        """
        func = node.func

        if isinstance(func, astnodes.Name):
            return func.id

        elif isinstance(func, astnodes.Index):
            # Handle index access like table.field or table:method
            # Use table/field (luaparser 4.x) with fallback to value/idx for compatibility
            table_node = getattr(func, "table", None) or getattr(func, "value", None)
            field_node = getattr(func, "field", None) or getattr(func, "idx", None)

            table_part = self._extract_node_name(table_node) if table_node else None
            field_part = self._extract_string_value(field_node) if field_node else ""

            if table_part and field_part:
                # Use colon for method-style calls, following LuaSymbolExtractor pattern
                separator = ":" if self._is_method_call(node) else "."
                return f"{table_part}{separator}{field_part}"
            elif table_part:
                return table_part

        return None

    def _extract_node_name(self, node: Any) -> str | None:
        """Recursively extract a name from an AST node.

        Args:
            node: The AST node

        Returns:
            String representation of the name, or None if cannot be extracted
        """
        if node is None:
            return None

        if isinstance(node, astnodes.Name):
            return node.id

        elif isinstance(node, astnodes.Index):
            # Use table/field (luaparser 4.x) with fallback to value/idx for compatibility
            table_node = getattr(node, "table", None) or getattr(node, "value", None)
            field_node = getattr(node, "field", None) or getattr(node, "idx", None)

            table_part = self._extract_node_name(table_node)
            field_part = self._extract_string_value(field_node) if field_node else ""
            if table_part and field_part:
                return f"{table_part}.{field_part}"
            return table_part

        return None

    def _is_method_call(self, node: astnodes.Call) -> bool:
        """Determine if a Call node represents a method call (colon syntax).

        In luaparser, method calls using colon syntax (obj:method()) are
        represented as Invoke nodes, while dot syntax (obj.method()) uses
        Call with Index. This follows the same detection pattern as
        LuaSymbolExtractor.

        Args:
            node: The Call AST node

        Returns:
            True if this appears to be a method call (colon syntax)
        """
        # In luaparser, Invoke nodes represent method calls with colon syntax
        # Check if the node type name indicates an Invoke pattern
        if hasattr(astnodes, "Invoke") and isinstance(node, astnodes.Invoke):
            return True

        # Fallback: check the node's class name for method call indicators
        node_type = type(node).__name__
        if "Invoke" in node_type:
            return True

        return False

    def _extract_string_value(self, node: Any) -> str:
        """Extract string value from an AST node.

        Args:
            node: AST node (String, Name, or other)

        Returns:
            String value, or "<dynamic>" if not a literal
        """
        if isinstance(node, astnodes.String):
            return node.s or ""
        elif isinstance(node, astnodes.Name):
            return node.id or ""
        elif hasattr(node, "s"):
            return str(node.s)
        elif hasattr(node, "id"):
            return str(node.id)
        return "<dynamic>"

    def _add_warning(
        self,
        node: Any,
        feature_name: str,
        description: str,
        severity: str,
        suggestion: str | None = None
    ) -> None:
        """Add a feature warning for the given AST node.

        Creates a FeatureWarning object with location information extracted
        from the AST node and appends it to the warnings list.

        Args:
            node: The AST node where the feature was detected
            feature_name: Name of the unsupported feature
            description: Human-readable description
            severity: Severity level ("warning", "error", "critical")
            suggestion: Optional suggestion for remediation
        """
        line_number = getattr(node, "line", 0) or getattr(node, "lineno", 0)
        column_offset = getattr(node, "column", 0) or getattr(node, "col_offset", 0)

        warning = FeatureWarning(
            feature_name=feature_name,
            description=description,
            line_number=line_number,
            column_offset=column_offset,
            severity=severity,
            suggestion=suggestion,
            file_path=self.file_path
        )

        self.warnings.append(warning)
        logger.debug(
            f"Detected {severity} feature: {feature_name} at {self.file_path}:{line_number}"
        )

    def get_warnings(self) -> list[FeatureWarning]:
        """Return collected feature warnings.

        Returns:
            List of FeatureWarning objects detected during AST traversal

        Example:
            >>> detector = LuaFeatureDetector(Path("test.lua"))
            >>> detector.visit(ast.parse("loadstring(code)()"))
            >>> warnings = detector.get_warnings()
            >>> print(f"Found {len(warnings)} warnings")
        """
        return self.warnings

    def get_warnings_by_severity(self, severity: str) -> list[FeatureWarning]:
        """Return warnings filtered by severity level.

        Args:
            severity: The severity level to filter by ("warning", "error", "critical")

        Returns:
            List of FeatureWarning objects matching the specified severity

        Example:
            >>> detector = LuaFeatureDetector(Path("test.lua"))
            >>> detector.visit(tree)
            >>> critical = detector.get_warnings_by_severity("critical")
            >>> print(f"Found {len(critical)} critical issues")
        """
        return [w for w in self.warnings if w.severity == severity]

