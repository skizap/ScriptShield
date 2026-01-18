"""
Symbol extraction module for Lua code analysis.

This module provides functionality to extract symbols from Lua source code,
including require() calls, function definitions, and variable assignments.
It supports Lua's local/global scoping rules and detects Roblox-specific API
patterns for preservation during obfuscation.

Example:
    >>> from luaparser import ast
    >>> from pathlib import Path
    >>> from lua_symbol_extractor import LuaSymbolExtractor
    >>> code = '''
    ... local module = require("Module")
    ... function greet(name)
    ...     print("Hello, " .. name)
    ... end
    ... '''
    >>> ast_tree = ast.parse(code)
    >>> extractor = LuaSymbolExtractor(Path("script.lua"))
    >>> extractor.visit(ast_tree)
    >>> symbols = extractor.get_symbol_table()
    >>> len(symbols.functions)
    1
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from luaparser import ast, astnodes


@dataclass
class LuaImportInfo:
    """
    Represents a require() call in Lua code.
    
    Attributes:
        module_path: The path to the required module (string literal or variable name)
        alias: Optional variable name if the require result is assigned
        line_number: Line number where the require() call appears
        is_relative: True if the path starts with '.' or '..'
    
    Example:
        >>> # For code: local http = require(game:GetService("HttpService"))
        >>> info = LuaImportInfo("game:GetService(\"HttpService\")", "http", 10, False)
    """
    module_path: str
    alias: Optional[str] = None
    line_number: int = 0
    is_relative: bool = False


@dataclass
class LuaFunctionInfo:
    """
    Represents a function definition in Lua code.
    
    Attributes:
        name: Function name (empty for anonymous functions)
        parameters: List of parameter names
        is_local: True if declared with 'local function'
        is_method: True if defined with colon syntax (self parameter)
        line_number: Line number where function is defined
        scope: Either "local" or "global"
        parent_table: Optional table name for table methods (e.g., "table" for "table.sort")
    
    Example:
        >>> # For code: local function greet(name) print(name) end
        >>> info = LuaFunctionInfo("greet", ["name"], True, False, 5, "local")
    """
    name: str
    parameters: List[str] = field(default_factory=list)
    is_local: bool = False
    is_method: bool = False
    line_number: int = 0
    scope: str = "global"
    parent_table: Optional[str] = None


@dataclass
class LuaVariableInfo:
    """
    Represents a variable assignment or declaration in Lua code.
    
    Attributes:
        name: Variable name
        line_number: Line number where assignment occurs
        scope: Either "local" or "global"
        is_constant: True if name follows uppercase convention (e.g., MAX_SIZE)
        context: Either "assignment" or "declaration"
    
    Example:
        >>> # For code: local count = 0
        >>> info = LuaVariableInfo("count", 3, "local", False, "declaration")
    """
    name: str
    line_number: int = 0
    scope: str = "global"
    is_constant: bool = False
    context: str = "assignment"


@dataclass
class LuaSymbolTable:
    """
    Aggregates all extracted symbols from Lua source code.
    
    Attributes:
        imports: List of require() calls
        functions: List of function definitions
        variables: List of variable declarations/assignments
        file_path: Path to the source file
        roblox_api_usage: List of detected Roblox API usage patterns
    
    Example:
        >>> table = LuaSymbolTable([], [], [], Path("script.lua"), [])
        >>> exports = table.get_exported_symbols()
    """
    imports: List[LuaImportInfo] = field(default_factory=list)
    functions: List[LuaFunctionInfo] = field(default_factory=list)
    variables: List[LuaVariableInfo] = field(default_factory=list)
    file_path: Path = field(default_factory=lambda: Path())
    roblox_api_usage: List[str] = field(default_factory=list)
    
    def get_exported_symbols(self) -> List[str]:
        """
        Get symbols that are exported globally.
        
        Returns:
            List of global function and variable names
        """
        exported = []
        for func in self.functions:
            if func.scope == "global" and func.name:
                exported.append(func.name)
        for var in self.variables:
            if var.scope == "global":
                exported.append(var.name)
        return exported
    
    def get_roblox_patterns(self) -> List[str]:
        """
        Get detected Roblox API usage patterns.
        
        Returns:
            List of Roblox API pattern strings
        """
        return self.roblox_api_usage


class LuaSymbolExtractor(ast.ASTVisitor):
    """
    AST visitor for extracting symbols from Lua source code.
    
    This class traverses a Lua AST and extracts information about:
    - require() calls (module imports)
    - Function definitions (local, global, methods)
    - Variable assignments (local, global)
    - Roblox-specific API patterns
    
    The extractor tracks scope using a stack to determine whether symbols
    are in global or local scope.
    
    Example:
        >>> from luaparser import ast
        >>> from pathlib import Path
        >>> code = 'local service = game:GetService("Players")'
        >>> tree = ast.parse(code)
        >>> extractor = LuaSymbolExtractor(Path("script.lua"))
        >>> extractor.visit(tree)
        >>> symbols = extractor.get_symbol_table()
    """
    
    def __init__(self, file_path: Path):
        """
        Initialize the symbol extractor.

        Args:
            file_path: Path to the Lua source file being analyzed
        """
        super().__init__()
        self.file_path = file_path
        self._imports: List[LuaImportInfo] = []
        self._functions: List[LuaFunctionInfo] = []
        self._variables: List[LuaVariableInfo] = []
        self._scope_stack: List[str] = []
        self._roblox_patterns: List[str] = []
        self._processed_require_calls: set = set()  # Track processed require() Call nodes by id

    def generic_visit(self, node):
        """
        Called for nodes where no explicit visitor method exists.
        Manually traverses child nodes to continue AST traversal.
        """
        # Manually traverse all child nodes
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
    
    def visit_Call(self, node: astnodes.Call) -> None:
        """
        Visit a function call node and extract require() calls and Roblox API usage.

        Args:
            node: Call AST node
        """
        # Extract call name to check for require() and Roblox APIs
        call_name = self._extract_call_name(node)

        # Handle standalone require() calls (not part of an assignment)
        # Skip if this require() was already processed in visit_Assign
        node_id = id(node)
        if call_name == "require" and node_id not in self._processed_require_calls:
            module_path = "<dynamic>"
            if node.args:
                arg = node.args[0]
                module_path = self._extract_string_value(arg)

            # Check if it's a relative path
            is_relative = module_path.startswith(('.', '/'))

            import_info = LuaImportInfo(
                module_path=module_path,
                alias=None,  # No alias for standalone require()
                line_number=getattr(node, 'lineno', 0),
                is_relative=is_relative
            )
            self._imports.append(import_info)

        # Detect Roblox API patterns
        if self._is_roblox_api_call(call_name, node.args):
            # Format the pattern for tracking
            args_str = ", ".join([self._extract_string_value(arg) for arg in node.args[:2]])
            pattern = f"{call_name}({args_str})"
            self._roblox_patterns.append(pattern)

        self.generic_visit(node)
    
    def visit_Function(self, node: astnodes.Function) -> None:
        """
        Visit a function definition node and extract function information.
        
        Args:
            node: Function AST node
        """
        # Extract function name
        func_name = getattr(node, 'name', '') or "<anonymous>"
        
        # Extract parameters
        parameters = []
        if hasattr(node, 'args') and node.args:
            parameters = [arg.id if hasattr(arg, 'id') else str(arg) for arg in node.args]
        
        # Determine if this is a local function
        is_local = isinstance(node, astnodes.LocalFunction)
        
        # Check for method syntax (colon indicates self parameter)
        is_method = isinstance(node, astnodes.Method)
        
        # Determine scope based on current context
        scope = "local" if is_local or not self._is_global_scope() else "global"
        
        # Check for parent table (the node before the colon for methods)
        parent_table = None
        if is_method and ':' in func_name:
            parts = func_name.split(':')
            if len(parts) == 2:
                parent_table = parts[0].strip()
        
        func_info = LuaFunctionInfo(
            name=func_name.split(':')[-1].strip(),  # Extract only the function name
            parameters=parameters,
            is_local=is_local,
            is_method=is_method,
            line_number=getattr(node, 'lineno', 0),
            scope=scope,
            parent_table=parent_table
        )
        self._functions.append(func_info)
        
        # Track scope
        self._scope_stack.append(func_name)
        self.generic_visit(node)
        self._scope_stack.pop()
    
    def visit_LocalFunction(self, node: astnodes.LocalFunction) -> None:
        """
        Visit a local function definition node.
        
        Args:
            node: LocalFunction AST node
        """
        self.visit_Function(node)
    
    def visit_Assign(self, node: astnodes.Assign) -> None:
        """
        Visit an assignment node and extract variable information.

        Args:
            node: Assign AST node
        """
        # Extract target variable names (handle multiple assignment)
        targets = []
        if hasattr(node, 'targets') and node.targets:
            for target in node.targets:
                if hasattr(target, 'id'):
                    targets.append(target.id)
                elif hasattr(target, 'name'):
                    targets.append(target.name)

        # Determine scope (global unless inside a function or explicitly local)
        scope = "local" if not self._is_global_scope() else "global"
        context = "assignment"

        # Check if this is a LocalAssign node
        is_local_assign = isinstance(node, astnodes.LocalAssign)
        if is_local_assign:
            scope = "local"
            context = "declaration"

        # Check assignment value for require() calls ONLY (with alias)
        # Other patterns like Roblox API will be caught by visit_Call during traversal
        if hasattr(node, 'values') and node.values:
            for value in node.values:
                if isinstance(value, astnodes.Call):
                    call_name = self._extract_call_name(value)

                    # Handle require() calls with alias
                    if call_name == "require" and targets:
                        module_path = "<dynamic>"
                        if value.args:
                            arg = value.args[0]
                            module_path = self._extract_string_value(arg)

                        # Check if it's a relative path
                        is_relative = module_path.startswith(('.', '/'))

                        import_info = LuaImportInfo(
                            module_path=module_path,
                            alias=targets[0] if targets else None,
                            line_number=getattr(node, 'lineno', 0),
                            is_relative=is_relative
                        )
                        self._imports.append(import_info)

                        # Mark this Call node as processed to avoid double-counting
                        self._processed_require_calls.add(id(value))

        # Create variable info for each target
        for target_name in targets:
            var_info = LuaVariableInfo(
                name=target_name,
                line_number=getattr(node, 'lineno', 0),
                scope=scope,
                is_constant=target_name.isupper(),
                context=context
            )
            self._variables.append(var_info)

        # Continue traversal to detect Roblox API calls and other patterns in assignment values
        # The _processed_require_calls set prevents double-counting of require() calls
        self.generic_visit(node)
    
    def visit_LocalAssign(self, node: astnodes.LocalAssign) -> None:
        """
        Visit a local assignment node.
        
        Args:
            node: LocalAssign AST node
        """
        self.visit_Assign(node)
    
    def _extract_call_name(self, node: astnodes.Call) -> str:
        """
        Extract the function name from a Call node.

        Handles various call patterns:
        - Name nodes: `require()`
        - Index nodes: `game:GetService()`

        Args:
            node: Call AST node

        Returns:
            String representation of the function name
        """
        func = node.func

        if isinstance(func, astnodes.Name):
            return func.id or ""
        elif isinstance(func, astnodes.Index):
            # Handle method calls like game:GetService()
            # Use table/field (luaparser 4.x) with fallback to value/idx for compatibility
            table_node = getattr(func, "table", None) or getattr(func, "value", None)
            field_node = getattr(func, "field", None) or getattr(func, "idx", None)

            table_part = self._extract_string_value(table_node) if table_node else ""
            field_part = self._extract_string_value(field_node) if field_node else ""
            if field_part:
                return f"{table_part}:{field_part}"
            return table_part
        else:
            return str(type(func).__name__)
    
    def _extract_string_value(self, node: Any) -> str:
        """
        Extract string literal value from an AST node.

        Args:
            node: AST node (String, Name, or other)

        Returns:
            String value, or "<dynamic>" if not a literal string
        """
        if isinstance(node, astnodes.String):
            value = node.s or ""
            # Handle bytes vs string
            if isinstance(value, bytes):
                return value.decode('utf-8', errors='replace')
            return str(value)
        elif isinstance(node, astnodes.Name):
            value = node.id or ""
            if isinstance(value, bytes):
                return value.decode('utf-8', errors='replace')
            return str(value)
        elif hasattr(node, 's'):
            value = node.s
            if isinstance(value, bytes):
                return value.decode('utf-8', errors='replace')
            return str(value)
        elif hasattr(node, 'id'):
            value = node.id
            if isinstance(value, bytes):
                return value.decode('utf-8', errors='replace')
            return str(value)
        else:
            return str(node)
    
    def _is_roblox_api_call(self, func_name: str, args: List[Any]) -> bool:
        """
        Check if a function call matches Roblox API patterns.
        
        Patterns detected:
        - game:GetService()
        - Instance.new() or Instance:new() (accepts both formats)
        - game:GetService() calls for specific Roblox services
        
        Args:
            func_name: Name of the function being called (accepts both dot and colon notation)
            args: List of call arguments
            
        Returns:
            True if the call matches a Roblox API pattern
        """
        # Check for game:GetService()
        if func_name.startswith("game:GetService"):
            return True
        
        # Check for Instance.new() or Instance:new() (accept both formats)
        if func_name in ("Instance.new", "Instance:new"):
            return True
        
        # Check for common Roblox services in require() or other context
        roblox_classes = [
            "RemoteEvent", "RemoteFunction", "BindableEvent", "BindableFunction",
            "TweenService", "ReplicatedStorage", "ServerScriptService"
        ]
        
        if args:
            arg_value = self._extract_string_value(args[0]).upper()
            for roblox_class in roblox_classes:
                if roblox_class.upper() in arg_value:
                    return True
        
        return False
    
    def _get_current_scope(self) -> str:
        """
        Get the current scope name.
        
        Returns:
            Current scope name, or "global" if scope stack is empty
        """
        if self._scope_stack:
            return self._scope_stack[-1]
        return "global"
    
    def _is_global_scope(self) -> bool:
        """
        Check if currently in global scope.
        
        Returns:
            True if scope stack is empty (global scope)
        """
        return len(self._scope_stack) == 0
    
    def get_symbol_table(self) -> LuaSymbolTable:
        """
        Get the collected symbol information.
        
        Returns:
            LuaSymbolTable containing all extracted symbols
        """
        return LuaSymbolTable(
            imports=self._imports,
            functions=self._functions,
            variables=self._variables,
            file_path=self.file_path,
            roblox_api_usage=self._roblox_patterns
        )
