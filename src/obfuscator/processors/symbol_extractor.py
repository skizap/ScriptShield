"""Symbol extraction module for Python code analysis.

This module provides data structures and visitor classes for extracting
symbols (imports, functions, classes, variables) from Python AST.

Example:
    >>> from pathlib import Path
    >>> from obfuscator.processors import SymbolExtractor
    >>> 
    >>> extractor = SymbolExtractor(Path("example.py"))
    >>> extractor.visit(ast_node)
    >>> symbols = extractor.get_symbol_table()
    >>> print(f"Found {len(symbols.functions)} functions")
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImportInfo:
    """Information about an import statement.
    
    Represents either a standard import (import module) or a from-import
    (from module import name) with full metadata for dependency tracking.
    
    Attributes:
        module_name: The imported module name
        imported_names: Specific names imported (empty for `import module`)
        alias: Import alias if present (e.g., `import numpy as np`)
        is_from_import: Whether it's a `from...import` statement
        line_number: Source line number
        level: Relative import level (0 for absolute, >0 for relative)
        
    Example:
        >>> info = ImportInfo(
        ...     module_name="numpy",
        ...     imported_names=[],
        ...     alias="np",
        ...     is_from_import=False,
        ...     line_number=1,
        ...     level=0
        ... )
        >>> print(f"Imported {info.module_name} as {info.alias}")
        Imported numpy as np
    """
    module_name: str
    imported_names: list[str] = field(default_factory=list)
    alias: str | None = None
    is_from_import: bool = False
    line_number: int = 0
    level: int = 0


@dataclass
class FunctionInfo:
    """Information about a function definition.
    
    Captures function metadata including parameters, decorators, and
    scope information for symbol table construction.
    
    Attributes:
        name: Function name
        parameters: Parameter names
        decorators: Decorator names
        is_async: Whether it's an async function
        line_number: Source line number
        scope: "global" or "local" (nested function)
        parent_class: Parent class name if method
        
    Example:
        >>> func = FunctionInfo(
        ...     name="calculate",
        ...     parameters=["a", "b"],
        ...     decorators=["property"],
        ...     is_async=False,
        ...     line_number=10,
        ...     scope="global",
        ...     parent_class=None
        ... )
        >>> print(func.name)
        calculate
    """
    name: str
    parameters: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    is_async: bool = False
    line_number: int = 0
    scope: str = "global"
    parent_class: str | None = None


@dataclass
class ClassInfo:
    """Information about a class definition.
    
    Stores class metadata including inheritance hierarchy, decorators,
    and method definitions for dependency graph construction.
    
    Attributes:
        name: Class name
        bases: Base class names in inheritance hierarchy
        decorators: Decorator names
        methods: Method names defined in class
        line_number: Source line number
        scope: "global" or "local" (nested class)
        
    Example:
        >>> cls = ClassInfo(
        ...     name="MyClass",
        ...     bases=["BaseClass"],
        ...     decorators=["dataclass"],
        ...     methods=["__init__", "process"],
        ...     line_number=20,
        ...     scope="global"
        ... )
        >>> print(cls.bases)
        ['BaseClass']
    """
    name: str
    bases: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    line_number: int = 0
    scope: str = "global"


@dataclass
class VariableInfo:
    """Information about a variable assignment.
    
    Tracks variable definitions with context and scope information
    for symbol table management.
    
    Attributes:
        name: Variable name
        line_number: Source line number
        scope: "global", "local", or "nonlocal"
        is_constant: Whether name is uppercase (convention)
        context: "assignment", "augmented_assignment", or "annotation"
        
    Example:
        >>> var = VariableInfo(
        ...     name="MAX_SIZE",
        ...     line_number=5,
        ...     scope="global",
        ...     is_constant=True,
        ...     context="assignment"
        ... )
        >>> print(var.is_constant)
        True
    """
    name: str
    line_number: int = 0
    scope: str = "global"
    is_constant: bool = False
    context: str = "assignment"


@dataclass
class SymbolTable:
    """Complete symbol information for a Python file.
    
    Aggregates all symbols extracted from a Python source file,
    providing a unified interface for dependency graph construction
    and symbol table management.
    
    Attributes:
        imports: All import statements
        functions: All function definitions
        classes: All class definitions
        variables: All variable assignments
        file_path: Source file path
        
    Example:
        >>> table = SymbolTable(
        ...     imports=[],
        ...     functions=[],
        ...     classes=[],
        ...     variables=[],
        ...     file_path=Path("example.py")
        ... )
        >>> exports = table.get_exported_symbols()
        >>> print(f"Exports: {exports}")
        Exports: []
    """
    imports: list[ImportInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    variables: list[VariableInfo] = field(default_factory=list)
    file_path: Path = Path()
    
    def get_exported_symbols(self) -> list[str]:
        """Get all exported symbols for dependency graph construction.
        
        Returns names of all global-scope functions and classes,
        which represent the public API of the module.
        
        Returns:
            List of symbol names exported by this module
            
        Example:
            >>> table = SymbolTable(
            ...     functions=[
            ...         FunctionInfo(name="func1", scope="global"),
            ...         FunctionInfo(name="func2", scope="local")
            ...     ],
            ...     classes=[
            ...         ClassInfo(name="Class1", scope="global")
            ...     ],
            ...     imports=[], variables=[], file_path=Path("test.py")
            ... )
            >>> table.get_exported_symbols()
            ['func1', 'Class1']
        """
        exports = []
        for func in self.functions:
            if func.scope == "global":
                exports.append(func.name)
        for cls in self.classes:
            if cls.scope == "global":
                exports.append(cls.name)
        return exports


class SymbolExtractor(ast.NodeVisitor):
    """AST visitor for extracting symbol information.
    
    Traverses Python AST to collect all symbols (imports, functions,
    classes, variables) with comprehensive metadata for dependency
    analysis and symbol table construction.
    
    Attributes:
        file_path: Path to the source file being analyzed
        imports: Collected import information
        functions: Collected function information
        classes: Collected class information
        variables: Collected variable information
        _scope_stack: Stack tracking current scope names
        _class_stack: Stack tracking enclosing class contexts
        
    Example:
        >>> import ast
        >>> from pathlib import Path
        >>> 
        >>> source = "def foo(): pass\\nclass Bar: pass"
        >>> ast_node = ast.parse(source)
        >>> extractor = SymbolExtractor(Path("example.py"))
        >>> extractor.visit(ast_node)
        >>> table = extractor.get_symbol_table()
        >>> print(len(table.functions))
        1
    """
    
    def __init__(self, file_path: Path):
        """Initialize the symbol extractor.
        
        Args:
            file_path: Path to the source file being analyzed
        """
        self.file_path = file_path
        self.imports: list[ImportInfo] = []
        self.functions: list[FunctionInfo] = []
        self.classes: list[ClassInfo] = []
        self.variables: list[VariableInfo] = []
        self._scope_stack: list[str] = []
        self._class_stack: list[str] = []
        self._global_declarations: dict[str, list[str]] = {}  # scope -> declared global names
        self._nonlocal_declarations: dict[str, list[str]] = {}  # scope -> declared nonlocal names
    
    def visit_Import(self, node: ast.Import) -> None:
        """Extract standard import statements.
        
        Args:
            node: Import AST node
        """
        for alias in node.names:
            import_info = ImportInfo(
                module_name=alias.name,
                imported_names=[],
                alias=alias.asname,
                is_from_import=False,
                line_number=node.lineno,
                level=0
            )
            self.imports.append(import_info)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Extract from-import statements.
        
        Args:
            node: ImportFrom AST node
        """
        module_name = node.module or ""
        
        for alias in node.names:
            import_info = ImportInfo(
                module_name=module_name,
                imported_names=[alias.name],
                alias=alias.asname,
                is_from_import=True,
                line_number=node.lineno,
                level=node.level
            )
            self.imports.append(import_info)
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function definitions.
        
        Args:
            node: FunctionDef AST node
        """
        self._extract_function(node, is_async=False)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Extract async function definitions.
        
        Args:
            node: AsyncFunctionDef AST node
        """
        self._extract_function(node, is_async=True)
    
    def _extract_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> None:
        """Extract function information from AST node.

        Args:
            node: FunctionDef or AsyncFunctionDef AST node
            is_async: Whether the function is async
        """
        # Extract parameter names
        parameters = [arg.arg for arg in node.args.args]

        # Extract decorator names
        decorators = [self._extract_name(dec) for dec in node.decorator_list]

        # Determine scope
        scope = "local" if len(self._scope_stack) > 0 else "global"

        # Get parent class if inside class definition
        parent_class = self._get_current_class()

        func_info = FunctionInfo(
            name=node.name,
            parameters=parameters,
            decorators=decorators,
            is_async=is_async,
            line_number=node.lineno,
            scope=scope,
            parent_class=parent_class
        )
        self.functions.append(func_info)

        # Push scope before adding parameters
        self._scope_stack.append(node.name)

        # Add function parameters as local variables in the function scope
        # This ensures they are tracked in the symbol table and can be mangled correctly
        for param_name in parameters:
            if param_name and param_name != "self":  # Skip 'self' parameter
                param_var = VariableInfo(
                    name=param_name,
                    line_number=node.lineno,
                    scope="local",
                    is_constant=False,
                    context="parameter"
                )
                self.variables.append(param_var)

        # Traverse function body
        self.generic_visit(node)
        self._scope_stack.pop()
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class definitions.
        
        Args:
            node: ClassDef AST node
        """
        # Extract base class names
        bases = [self._extract_name(base) for base in node.bases]
        
        # Extract decorator names
        decorators = [self._extract_name(dec) for dec in node.decorator_list]
        
        # Determine scope
        scope = "local" if len(self._scope_stack) > 0 else "global"
        
        # Push class to scope stack before traversing body
        self._scope_stack.append(node.name)
        self._class_stack.append(node.name)
        
        # Traverse class body to collect methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)
        
        self.generic_visit(node)
        
        # Pop class from stacks
        self._class_stack.pop()
        self._scope_stack.pop()
        
        class_info = ClassInfo(
            name=node.name,
            bases=bases,
            decorators=decorators,
            methods=methods,
            line_number=node.lineno,
            scope=scope
        )
        self.classes.append(class_info)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Extract variable assignments.
        
        Args:
            node: Assign AST node
        """
        self._extract_variables(node.targets, node.lineno, "assignment")
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Extract annotated variable assignments.
        
        Args:
            node: AnnAssign AST node
        """
        if node.target:
            self._extract_variables([node.target], node.lineno, "annotation")
        self.generic_visit(node)
    
    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Extract augmented assignment statements.
        
        Args:
            node: AugAssign AST node
        """
        self._extract_variables([node.target], node.lineno, "augmented_assignment")
        self.generic_visit(node)
    
    def visit_Global(self, node: ast.Global) -> None:
        """Track global variable declarations.
        
        Args:
            node: Global AST node
        """
        current_scope = self._get_current_scope_key()
        declared_names = [name for name in node.names]
        self._global_declarations.setdefault(current_scope, []).extend(declared_names)
    
    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Track nonlocal variable declarations.
        
        Args:
            node: Nonlocal AST node
        """
        current_scope = self._get_current_scope_key()
        declared_names = [name for name in node.names]
        self._nonlocal_declarations.setdefault(current_scope, []).extend(declared_names)
    
    def _extract_variables(self, targets: list[ast.expr], line_number: int, context: str) -> None:
        """Extract variable information from assignment targets.
        
        Args:
            targets: List of assignment targets
            line_number: Source line number
            context: Assignment context type
        """
        # Determine default scope (local for any non-module scope, including classes)
        if self._is_global_scope():
            scope = "global"
        elif self._get_current_class() is not None:
            scope = "local"  # Class body assignments are local
        else:
            scope = "local"  # Function body assignments are local by default
        
        for target in targets:
            names = self._extract_target_names(target)
            for name in names:
                if name and name != "_":  # Skip wildcard
                    # Check if variable is explicitly declared as global or nonlocal
                    current_scope = self._get_current_scope_key()
                    if current_scope in self._global_declarations and name in self._global_declarations[current_scope]:
                        scope = "global"
                    elif current_scope in self._nonlocal_declarations and name in self._nonlocal_declarations[current_scope]:
                        scope = "nonlocal"
                    
                    is_constant = name.isupper()
                    var_info = VariableInfo(
                        name=name,
                        line_number=line_number,
                        scope=scope,
                        is_constant=is_constant,
                        context=context
                    )
                    self.variables.append(var_info)
    
    def _extract_target_names(self, target: ast.expr) -> list[str]:
        """Extract variable names from assignment target.
        
        Recursively handles simple names, tuples, lists, and starred targets
        to fully capture destructuring assignment patterns.
        
        Args:
            target: Assignment target expression
            
        Returns:
            List of variable names
        """
        names = []
        
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Tuple):
            # Recursively extract from tuple elements (handles nested tuples)
            for elt in target.elts:
                names.extend(self._extract_target_names(elt))
        elif isinstance(target, ast.List):
            # Recursively extract from list elements (handles nested lists)
            for elt in target.elts:
                names.extend(self._extract_target_names(elt))
        elif isinstance(target, ast.Starred):
            # Extract starred expression (e.g., *rest)
            names.extend(self._extract_target_names(target.value))
        
        return names
    
    def _get_current_scope_key(self) -> str:
        """Get a unique key for the current scope.
        
        Creates a string representation of the current scope path
        for tracking global/nonlocal declarations.
        
        Returns:
            String key representing the current scope
        """
        return ".".join(self._scope_stack) if self._scope_stack else "<module>"
    
    def _extract_name(self, node: ast.expr) -> str:
        """Extract name string from AST node.
        
        Handles various node types including Name, Attribute, and Call.
        
        Args:
            node: AST expression node
            
        Returns:
            String representation of the name
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._extract_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._extract_name(node.func)
        else:
            return str(type(node).__name__)
    
    def _is_global_scope(self) -> bool:
        """Check if currently at module level.
        
        Returns:
            True if current scope is module level (global)
        """
        return len(self._scope_stack) == 0
    
    def _get_current_class(self) -> str | None:
        """Get current class name from scope stack.
        
        Returns:
            Class name if inside a class definition, None otherwise
        """
        if self._class_stack:
            return self._class_stack[-1]
        return None
    
    def get_symbol_table(self) -> SymbolTable:
        """Get the complete symbol table.
        
        Returns:
            SymbolTable containing all extracted symbols
        """
        return SymbolTable(
            imports=self.imports,
            functions=self.functions,
            classes=self.classes,
            variables=self.variables,
            file_path=self.file_path
        )
