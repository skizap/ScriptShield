"""Unsupported feature detector for Python code obfuscation.

This module provides detection of Python language features that can interfere
with obfuscation or cause issues with static analysis. The detector traverses
the AST and identifies problematic constructs like dynamic code execution,
dynamic imports, attribute manipulation, and other features that make code
difficult to analyze or transform reliably.

Example:
    >>> from obfuscator.processors import UnsupportedFeatureDetector
    >>> import ast
    >>> tree = ast.parse("x = eval(user_input)")
    >>> detector = UnsupportedFeatureDetector(Path("example.py"))
    >>> detector.visit(tree)
    >>> warnings = detector.get_warnings()
    >>> for warning in warnings:
    ...     print(f"{warning.severity}: {warning.feature_name} at line {warning.line_number}")
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from obfuscator.utils.logger import get_logger

logger = get_logger("obfuscator.processors.feature_detector")


@dataclass
class FeatureWarning:
    """Warning about an unsupported or problematic Python feature.

    This dataclass encapsulates information about a feature that may interfere
    with obfuscation or static analysis, including its location, severity, and
    suggestions for remediation.

    Attributes:
        feature_name: Name of the unsupported feature (e.g., "eval()", "exec()")
        description: Human-readable description of why it's problematic
        line_number: Source line number where detected
        column_offset: Column offset in the source line
        severity: Warning severity level ("warning", "error", "critical")
        suggestion: Optional suggestion for remediation
        file_path: Path to the source file where the feature was detected

    Example:
        >>> warning = FeatureWarning(
        ...     feature_name="eval()",
        ...     description="Dynamic code execution prevents static analysis",
        ...     line_number=42,
        ...     column_offset=10,
        ...     severity="critical",
        ...     suggestion="Avoid using eval(); consider alternative approaches",
        ...     file_path=Path("example.py")
        ... )
        >>> print(f"{warning.severity}: {warning.feature_name} at {warning.file_path}")
    """

    feature_name: str
    description: str
    line_number: int
    column_offset: int
    severity: str
    suggestion: str | None = None
    file_path: Path = Path("<unknown>")


class UnsupportedFeatureDetector(ast.NodeVisitor):
    """Detector for unsupported or problematic Python features.

    This class extends ast.NodeVisitor to traverse Python ASTs and identify
    language features that can interfere with obfuscation or static analysis.
    It detects dynamic code execution, dynamic imports, attribute manipulation,
    and other problematic constructs.

    Attributes:
        file_path: Path to the source file being analyzed
        warnings: List of FeatureWarning objects collected during traversal

    Example:
        >>> from obfuscator.processors import UnsupportedFeatureDetector
        >>> from pathlib import Path
        >>> import ast
        >>> code = '''
        ... def foo():
        ...     x = eval(user_input)
        ...     exec(some_code)
        ... '''
        >>> tree = ast.parse(code)
        >>> detector = UnsupportedFeatureDetector(Path("example.py"))
        >>> detector.visit(tree)
        >>> warnings = detector.get_warnings()
        >>> print(f"Found {len(warnings)} warnings")
        >>> for warning in warnings:
        ...     print(f"Line {warning.line_number}: {warning.feature_name}")
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize the feature detector.

        Args:
            file_path: Path to the source file being analyzed
        """
        self.file_path = file_path
        self.warnings: list[FeatureWarning] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call nodes to detect dangerous function calls.

        Detects calls to functions that perform dynamic code execution,
        dynamic imports, or introspection that can interfere with
        obfuscation and static analysis.

        Args:
            node: The Call AST node to analyze

        Example:
            >>> detector = UnsupportedFeatureDetector(Path("test.py"))
            >>> tree = ast.parse("x = eval(user_input)")
            >>> detector.visit(tree)
        """
        func_name = self._extract_call_name(node)

        if func_name is None:
            self.generic_visit(node)
            return

        # Extract final attribute name for attribute-based calls (e.g., builtins.eval)
        final_attr = func_name.split('.')[-1] if '.' in func_name else func_name

        # Critical: Dynamic code execution (including attribute-based calls)
        if final_attr == "eval":
            self._add_warning(
                node,
                feature_name=f"{func_name}()",
                description="Dynamic code execution prevents static analysis and obfuscation",
                severity="critical",
                suggestion="Avoid using eval(); consider safer alternatives like literal_eval, custom parsers, or refactoring logic"
            )
        elif final_attr == "exec":
            self._add_warning(
                node,
                feature_name=f"{func_name}()",
                description="Dynamic code execution prevents static analysis and obfuscation",
                severity="critical",
                suggestion="Avoid using exec(); consider redesigning to use functions, classes, or configuration files"
            )
        elif final_attr == "compile":
            self._add_warning(
                node,
                feature_name=f"{func_name}()",
                description="Creates code objects dynamically, interfering with static analysis",
                severity="error",
                suggestion="Avoid compile(); use static code generation or alternative approaches"
            )

        # Error: Dynamic import functions
        elif func_name == "__import__":
            self._add_warning(
                node,
                feature_name="__import__()",
                description="Dynamic module import cannot be reliably analyzed or obfuscated",
                severity="error",
                suggestion="Use static import statements at module level"
            )
        elif func_name == "importlib.import_module":
            self._add_warning(
                node,
                feature_name="importlib.import_module()",
                description="Dynamic module import cannot be reliably analyzed or obfuscated",
                severity="error",
                suggestion="Use static import statements at module level"
            )
        elif func_name == "importlib.__import__":
            self._add_warning(
                node,
                feature_name="importlib.__import__()",
                description="Dynamic module import cannot be reliably analyzed or obfuscated",
                severity="error",
                suggestion="Use static import statements at module level"
            )

        # Warning: Namespace and introspection functions
        elif func_name in ("globals", "locals", "vars"):
            self._add_warning(
                node,
                feature_name=f"{func_name}()",
                description="Direct namespace access interferes with symbol tracking and obfuscation",
                severity="warning",
                suggestion="Avoid direct namespace access; use explicit parameter passing or object attributes"
            )
        elif func_name in ("setattr", "getattr", "delattr", "hasattr"):
            self._add_warning(
                node,
                feature_name=f"{func_name}()",
                description="Dynamic attribute manipulation interferes with static analysis",
                severity="warning",
                suggestion="Use direct attribute access (obj.attr) or dictionary-based storage when dynamic behavior is needed"
            )

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements to detect problematic imports.

        Detects imports of modules that provide dynamic capabilities or
        introspection features that can interfere with obfuscation.

        Args:
            node: The Import AST node to analyze

        Example:
            >>> detector = UnsupportedFeatureDetector(Path("test.py"))
            >>> tree = ast.parse("import importlib")
            >>> detector.visit(tree)
        """
        for alias in node.names:
            module_name = alias.name

            if module_name == "importlib":
                self._add_warning(
                    node,
                    feature_name=f"import {module_name}",
                    description="importlib provides dynamic import capabilities that interfere with obfuscation",
                    severity="warning",
                    suggestion="Avoid importlib unless absolutely necessary; use static imports instead"
                )

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statements to detect problematic imports.

        Detects from-imports from modules that provide dynamic capabilities
        or introspection features.

        Args:
            node: The ImportFrom AST node to analyze

        Example:
            >>> detector = UnsupportedFeatureDetector(Path("test.py"))
            >>> tree = ast.parse("from importlib import import_module")
            >>> detector.visit(tree)
        """
        if node.module == "importlib":
            self._add_warning(
                node,
                feature_name=f"from importlib import ...",
                description="importlib provides dynamic import capabilities that interfere with obfuscation",
                severity="warning",
                suggestion="Avoid importing from importlib unless absolutely necessary"
            )

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions to detect metaclass usage.

        Metaclasses can introduce complex dynamic behavior that interferes
        with class analysis and obfuscation.

        Args:
            node: The ClassDef AST node to analyze

        Example:
            >>> detector = UnsupportedFeatureDetector(Path("test.py"))
            >>> tree = ast.parse("class Meta(type): pass")
            >>> detector.visit(tree)
        """
        # Check for metaclass keyword argument
        for keyword in node.keywords:
            if keyword.arg == "metaclass":
                self._add_warning(
                    node,
                    feature_name="metaclass",
                    description="Metaclasses introduce complex dynamic class behavior that complicates analysis",
                    severity="warning",
                    suggestion="Consider using class decorators, inheritance, or composition instead of metaclasses"
                )
                break

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit attribute access nodes to detect problematic attributes.

        Detects access to attributes that provide introspection or dynamic
        capabilities that interfere with obfuscation.

        Args:
            node: The Attribute AST node to analyze

        Example:
            >>> detector = UnsupportedFeatureDetector(Path("test.py"))
            >>> tree = ast.parse("obj.__dict__")
            >>> detector.visit(tree)
        """
        attr_name = node.attr

        if attr_name == "__import__":
            self._add_warning(
                node,
                feature_name="__import__ attribute",
                description="Direct access to import machinery interferes with import tracking",
                severity="error",
                suggestion="Do not access __import__ explicitly; use standard import statements"
            )
        elif attr_name == "__dict__":
            self._add_warning(
                node,
                feature_name="__dict__ attribute",
                description="Direct namespace dictionary access interferes with symbol tracking",
                severity="warning",
                suggestion="Use dir() or vars() for inspection; avoid directly manipulating __dict__"
            )

        self.generic_visit(node)

    def _add_warning(
        self,
        node: ast.AST,
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
        line_number = getattr(node, "lineno", 0)
        column_offset = getattr(node, "col_offset", 0)

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

    def _extract_call_name(self, node: ast.Call) -> str | None:
        """Extract the function name from a Call node.

        Handles different call patterns:
        - Simple calls: `func()`
        - Attribute calls: `obj.method()`
        - Nested attribute calls: `module.func()`

        Args:
            node: The Call AST node

        Returns:
            String representation of the function name, or None if name
            cannot be extracted
        """
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Reconstruct attribute access like "module.func" or "obj.method"
            attr = node.func.attr
            value = node.func.value

            if isinstance(value, ast.Name):
                return f"{value.id}.{attr}"
            elif isinstance(value, ast.Attribute):
                # Handle nested attributes like "module.submodule.func"
                inner = self._extract_attribute_name(value)
                if inner:
                    return f"{inner}.{attr}"
            else:
                # For more complex cases, just return the attribute name
                return attr

        return None

    def _extract_attribute_name(self, node: ast.Attribute) -> str | None:
        """Recursively extract attribute names from an Attribute node.

        Args:
            node: The Attribute AST node

        Returns:
            String representation of the attribute name, or None if name
            cannot be extracted
        """
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            inner = self._extract_attribute_name(node.value)
            if inner:
                return f"{inner}.{node.attr}"

        return node.attr

    def get_warnings(self) -> list[FeatureWarning]:
        """Return collected feature warnings.

        Returns:
            List of FeatureWarning objects detected during AST traversal

        Example:
            >>> detector = UnsupportedFeatureDetector(Path("test.py"))
            >>> detector.visit(ast.parse("x = eval(input())"))
            >>> warnings = detector.get_warnings()
            >>> print(f"Found {len(warnings)} warnings")
        """
        return self.warnings
