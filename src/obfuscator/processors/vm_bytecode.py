"""
VM Bytecode instruction set and compilers for Python and Lua.

This module provides a custom bytecode instruction set and compilers for both
Python and Lua AST nodes, enabling VM-based code protection.
"""

import ast
import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union


# Bytecode opcodes
LOAD_CONST = 0x01
LOAD_VAR = 0x02
STORE_VAR = 0x03
BINARY_ADD = 0x04
BINARY_SUB = 0x05
BINARY_MUL = 0x06
BINARY_DIV = 0x07
BINARY_MOD = 0x08
BINARY_POW = 0x09
COMPARE_EQ = 0x0A
COMPARE_NE = 0x0B
COMPARE_LT = 0x0C
COMPARE_LE = 0x0D
COMPARE_GT = 0x0E
COMPARE_GE = 0x0F
JUMP = 0x10
JUMP_IF_FALSE = 0x11
JUMP_IF_TRUE = 0x12
CALL_FUNC = 0x13
RETURN = 0x14
POP = 0x15
DUP = 0x16
LOAD_GLOBAL = 0x17
STORE_GLOBAL = 0x18
LOAD_ATTR = 0x19
STORE_ATTR = 0x1A
BUILD_LIST = 0x1B
BUILD_MAP = 0x1C
LOAD_INDEX = 0x1D
STORE_INDEX = 0x1E
UNARY_NOT = 0x1F
UNARY_NEGATIVE = 0x20
UNARY_POSITIVE = 0x21

# Complexity levels
COMPLEXITY_BASIC = 1
COMPLEXITY_INTERMEDIATE = 2
COMPLEXITY_ADVANCED = 3


@dataclasses.dataclass
class BytecodeInstruction:
    """Represents a single bytecode instruction."""
    
    opcode: int
    arg1: Optional[Any] = None
    arg2: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BytecodeSerializer:
    """Serializes bytecode instructions to compact binary format."""
    
    @staticmethod
    def _resolve_labels(instructions: List[BytecodeInstruction]) -> List[BytecodeInstruction]:
        """
        Resolve label names to absolute bytecode indices.
        
        This pass:
        1. Scans for label placeholders (instructions with 'label' in metadata)
        2. Maps label names to their target instruction indices
        3. Replaces JUMP instruction arguments with resolved indices
        4. Removes label placeholder instructions
        
        Args:
            instructions: List of bytecode instructions with label placeholders
            
        Returns:
            List of bytecode instructions with resolved jump offsets
        """
        # First pass: collect label positions
        label_positions = {}
        filtered_instructions = []
        
        for idx, instr in enumerate(instructions):
            if instr.metadata and 'label' in instr.metadata:
                # This is a label placeholder, store its position
                label_name = instr.metadata['label']
                label_positions[label_name] = len(filtered_instructions)
                # Skip adding label placeholders to the output
                continue
            filtered_instructions.append(instr)
        
        # Second pass: replace label references with absolute indices
        resolved_instructions = []
        for instr in filtered_instructions:
            if instr.opcode in (JUMP, JUMP_IF_FALSE, JUMP_IF_TRUE):
                # Check if arg1 is a label name that needs resolution
                if isinstance(instr.arg1, str) and instr.arg1 in label_positions:
                    # Replace label name with absolute instruction index
                    resolved_arg1 = label_positions[instr.arg1]
                else:
                    resolved_arg1 = instr.arg1
                
                # Create new instruction with resolved argument
                resolved_instr = BytecodeInstruction(
                    opcode=instr.opcode,
                    arg1=resolved_arg1,
                    arg2=instr.arg2,
                    metadata=instr.metadata
                )
                resolved_instructions.append(resolved_instr)
            else:
                resolved_instructions.append(instr)
        
        return resolved_instructions
    
    @staticmethod
    def serialize(instructions: List[BytecodeInstruction]) -> List[int]:
        """Serialize bytecode instructions to integer list."""
        # First resolve labels to absolute indices
        resolved_instructions = BytecodeSerializer._resolve_labels(instructions)
        
        result = []
        for instr in resolved_instructions:
            result.append(instr.opcode)
            
            # Encode arg1
            if instr.arg1 is None:
                result.append(0)
            elif isinstance(instr.arg1, int):
                result.append(1)  # Type marker for int
                result.append(instr.arg1)
            elif isinstance(instr.arg1, str):
                result.append(2)  # Type marker for str
                # Store string length and then char codes
                encoded = instr.arg1.encode('utf-8')
                result.append(len(encoded))
                result.extend(encoded)
            else:
                result.append(3)  # Type marker for other (will be in constants pool)
                result.append(instr.arg1 if isinstance(instr.arg1, int) else 0)
            
            # Encode arg2
            if instr.arg2 is None:
                result.append(0)
            elif isinstance(instr.arg2, int):
                result.append(1)
                result.append(instr.arg2)
            elif isinstance(instr.arg2, str):
                result.append(2)
                encoded = instr.arg2.encode('utf-8')
                result.append(len(encoded))
                result.extend(encoded)
            else:
                result.append(3)
                result.append(instr.arg2 if isinstance(instr.arg2, int) else 0)
        
        return result
    
    @staticmethod
    def deserialize(data: List[int]) -> List[BytecodeInstruction]:
        """Deserialize integer list back to bytecode instructions."""
        instructions = []
        i = 0
        
        while i < len(data):
            opcode = data[i]
            i += 1
            
            # Decode arg1
            arg1_type = data[i]
            i += 1
            arg1 = None
            
            if arg1_type == 1:  # int
                arg1 = data[i]
                i += 1
            elif arg1_type == 2:  # str
                length = data[i]
                i += 1
                arg1 = bytes(data[i:i + length]).decode('utf-8')
                i += length
            elif arg1_type == 3:  # other
                arg1 = data[i]
                i += 1
            
            # Decode arg2
            arg2_type = data[i]
            i += 1
            arg2 = None
            
            if arg2_type == 1:  # int
                arg2 = data[i]
                i += 1
            elif arg2_type == 2:  # str
                length = data[i]
                i += 1
                arg2 = bytes(data[i:i + length]).decode('utf-8')
                i += length
            elif arg2_type == 3:  # other
                arg2 = data[i]
                i += 1
            
            instructions.append(BytecodeInstruction(opcode, arg1, arg2))
        
        return instructions


class BytecodeCompiler(ABC):
    """Base class for bytecode compilers."""
    
    def __init__(self, complexity: int = COMPLEXITY_INTERMEDIATE):
        self.complexity = complexity
        self.constants: List[Any] = []
        self.local_vars: Dict[str, int] = {}
        self.global_vars: set = set()
        self.current_line: int = 0
        self.label_counter: int = 0
    
    def add_constant(self, value: Any) -> int:
        """Add a constant to the constant pool and return its index."""
        if value in self.constants:
            return self.constants.index(value)
        self.constants.append(value)
        return len(self.constants) - 1
    
    def get_local_var(self, name: str) -> int:
        """Get or create local variable slot."""
        if name not in self.local_vars:
            self.local_vars[name] = len(self.local_vars)
        return self.local_vars[name]
    
    def new_label(self) -> str:
        """Generate a new label name."""
        self.label_counter += 1
        return f"L{self.label_counter}"
    
    @abstractmethod
    def compile_function(self, node: Any) -> Tuple[List[BytecodeInstruction], List[Any], int]:
        """Compile a function to bytecode."""
        pass
    
    @abstractmethod
    def compile_expression(self, node: Any) -> List[BytecodeInstruction]:
        """Compile an expression to bytecode."""
        pass
    
    @abstractmethod
    def compile_statement(self, node: Any) -> List[BytecodeInstruction]:
        """Compile a statement to bytecode."""
        pass


class PythonBytecodeCompiler(BytecodeCompiler):
    """Bytecode compiler for Python AST nodes."""
    
    def compile_function(self, node: ast.FunctionDef) -> Tuple[List[BytecodeInstruction], List[Any], int]:
        """Compile a Python function to bytecode."""
        self.local_vars.clear()
        self.global_vars.clear()
        self.current_line = node.lineno
        
        instructions = []
        
        # Compile function parameters as local variables
        for arg in node.args.args:
            self.get_local_var(arg.arg)
        
        # Compile default arguments if any
        if node.args.defaults:
            for i, default in enumerate(node.args.defaults):
                param_idx = len(node.args.args) - len(node.args.defaults) + i
                param_name = node.args.args[param_idx].arg
                instructions.extend(self.compile_expression(default))
                instructions.append(BytecodeInstruction(STORE_VAR, self.get_local_var(param_name)))
        
        # Compile function body
        for stmt in node.body:
            instructions.extend(self.compile_statement(stmt))
        
        # Ensure function returns something
        if not instructions or instructions[-1].opcode != RETURN:
            instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))
            instructions.append(BytecodeInstruction(RETURN))
        
        return instructions, self.constants.copy(), len(self.local_vars)
    
    def compile_expression(self, node: ast.AST) -> List[BytecodeInstruction]:
        """Compile a Python expression to bytecode."""
        instructions = []
        
        if isinstance(node, ast.Constant):
            # Handle literal constants
            const_idx = self.add_constant(node.value)
            instructions.append(BytecodeInstruction(LOAD_CONST, const_idx))
        
        elif isinstance(node, ast.Name):
            # Handle variable references
            if isinstance(node.ctx, ast.Load):
                if node.id in self.local_vars:
                    instructions.append(BytecodeInstruction(LOAD_VAR, self.get_local_var(node.id)))
                else:
                    self.global_vars.add(node.id)
                    instructions.append(BytecodeInstruction(LOAD_GLOBAL, node.id))
            elif isinstance(node.ctx, ast.Store):
                if node.id in self.local_vars:
                    instructions.append(BytecodeInstruction(STORE_VAR, self.get_local_var(node.id)))
                else:
                    self.global_vars.add(node.id)
                    instructions.append(BytecodeInstruction(STORE_GLOBAL, node.id))
        
        elif isinstance(node, ast.BinOp):
            # Handle binary operations
            instructions.extend(self.compile_expression(node.left))
            instructions.extend(self.compile_expression(node.right))
            
            if isinstance(node.op, ast.Add):
                instructions.append(BytecodeInstruction(BINARY_ADD))
            elif isinstance(node.op, ast.Sub):
                instructions.append(BytecodeInstruction(BINARY_SUB))
            elif isinstance(node.op, ast.Mult):
                instructions.append(BytecodeInstruction(BINARY_MUL))
            elif isinstance(node.op, ast.Div):
                instructions.append(BytecodeInstruction(BINARY_DIV))
            elif isinstance(node.op, ast.Mod):
                instructions.append(BytecodeInstruction(BINARY_MOD))
            elif isinstance(node.op, ast.Pow):
                instructions.append(BytecodeInstruction(BINARY_POW))
        
        elif isinstance(node, ast.Compare):
            # Handle comparisons
            instructions.extend(self.compile_expression(node.left))
            
            for i, op in enumerate(node.ops):
                if i > 0:
                    # For chained comparisons, we need to handle the result
                    pass
                
                instructions.extend(self.compile_expression(node.comparators[i]))
                
                if isinstance(op, ast.Eq):
                    instructions.append(BytecodeInstruction(COMPARE_EQ))
                elif isinstance(op, ast.NotEq):
                    instructions.append(BytecodeInstruction(COMPARE_NE))
                elif isinstance(op, ast.Lt):
                    instructions.append(BytecodeInstruction(COMPARE_LT))
                elif isinstance(op, ast.LtE):
                    instructions.append(BytecodeInstruction(COMPARE_LE))
                elif isinstance(op, ast.Gt):
                    instructions.append(BytecodeInstruction(COMPARE_GT))
                elif isinstance(op, ast.GtE):
                    instructions.append(BytecodeInstruction(COMPARE_GE))
        
        elif isinstance(node, ast.UnaryOp):
            # Handle unary operations
            instructions.extend(self.compile_expression(node.operand))
            
            if isinstance(node.op, ast.Not):
                instructions.append(BytecodeInstruction(UNARY_NOT))
            elif isinstance(node.op, ast.USub):
                instructions.append(BytecodeInstruction(UNARY_NEGATIVE))
            elif isinstance(node.op, ast.UAdd):
                instructions.append(BytecodeInstruction(UNARY_POSITIVE))
        
        elif isinstance(node, ast.Call):
            # Handle function calls
            # Compile function object
            instructions.extend(self.compile_expression(node.func))
            
            # Compile arguments
            arg_count = len(node.args)
            for arg in node.args:
                instructions.extend(self.compile_expression(arg))
            
            instructions.append(BytecodeInstruction(CALL_FUNC, arg_count))
        
        elif isinstance(node, ast.IfExp):
            # Handle conditional expressions
            end_label = self.new_label()
            else_label = self.new_label()
            
            # Compile condition
            instructions.extend(self.compile_expression(node.test))
            instructions.append(BytecodeInstruction(JUMP_IF_FALSE, else_label))
            
            # Compile true branch
            instructions.extend(self.compile_expression(node.body))
            instructions.append(BytecodeInstruction(JUMP, end_label))
            
            # Compile false branch
            instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))  # Placeholder for label
            instructions[-1].metadata['label'] = else_label
            instructions.extend(self.compile_expression(node.orelse))
            
            instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))  # Placeholder for label
            instructions[-1].metadata['label'] = end_label
        
        elif isinstance(node, ast.List):
            # Handle list literals
            for elem in node.elts:
                instructions.extend(self.compile_expression(elem))
            instructions.append(BytecodeInstruction(BUILD_LIST, len(node.elts)))
        
        elif isinstance(node, ast.Dict):
            # Handle dict literals
            for i in range(len(node.keys)):
                if node.keys[i] is not None:
                    instructions.extend(self.compile_expression(node.keys[i]))
                    instructions.extend(self.compile_expression(node.values[i]))
            instructions.append(BytecodeInstruction(BUILD_MAP, len(node.keys)))
        
        elif isinstance(node, ast.Subscript):
            # Handle subscript operations
            instructions.extend(self.compile_expression(node.value))
            instructions.extend(self.compile_expression(node.slice))
            
            if isinstance(node.ctx, ast.Load):
                instructions.append(BytecodeInstruction(LOAD_INDEX))
            elif isinstance(node.ctx, ast.Store):
                instructions.append(BytecodeInstruction(STORE_INDEX))
        
        elif isinstance(node, ast.Attribute):
            # Handle attribute access
            instructions.extend(self.compile_expression(node.value))
            
            if isinstance(node.ctx, ast.Load):
                instructions.append(BytecodeInstruction(LOAD_ATTR, node.attr))
            elif isinstance(node.ctx, ast.Store):
                instructions.append(BytecodeInstruction(STORE_ATTR, node.attr))
        
        return instructions
    
    def compile_statement(self, node: ast.stmt) -> List[BytecodeInstruction]:
        """Compile a Python statement to bytecode."""
        instructions = []
        self.current_line = getattr(node, 'lineno', self.current_line)
        
        if isinstance(node, ast.Assign):
            # Handle assignments
            if len(node.targets) == 1:
                instructions.extend(self.compile_expression(node.value))
                instructions.extend(self.compile_expression(node.targets[0]))
            else:
                # Handle multiple assignment
                instructions.extend(self.compile_expression(node.value))
                instructions.append(BytecodeInstruction(DUP))
                for target in node.targets:
                    instructions.extend(self.compile_expression(target))
        
        elif isinstance(node, ast.AugAssign):
            # Handle augmented assignments
            instructions.extend(self.compile_expression(node.target))
            instructions.extend(self.compile_expression(node.value))
            
            if isinstance(node.op, ast.Add):
                instructions.append(BytecodeInstruction(BINARY_ADD))
            elif isinstance(node.op, ast.Sub):
                instructions.append(BytecodeInstruction(BINARY_SUB))
            elif isinstance(node.op, ast.Mult):
                instructions.append(BytecodeInstruction(BINARY_MUL))
            elif isinstance(node.op, ast.Div):
                instructions.append(BytecodeInstruction(BINARY_DIV))
            elif isinstance(node.op, ast.Mod):
                instructions.append(BytecodeInstruction(BINARY_MOD))
            
            instructions.extend(self.compile_expression(node.target))
        
        elif isinstance(node, ast.Return):
            # Handle return statements
            if node.value:
                instructions.extend(self.compile_expression(node.value))
            else:
                instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))
            instructions.append(BytecodeInstruction(RETURN))
        
        elif isinstance(node, ast.If):
            # Handle if statements
            end_label = self.new_label()
            
            # Compile condition
            instructions.extend(self.compile_expression(node.test))
            
            if node.orelse:
                else_label = self.new_label()
                instructions.append(BytecodeInstruction(JUMP_IF_FALSE, else_label))
                
                # Compile if body
                for stmt in node.body:
                    instructions.extend(self.compile_statement(stmt))
                instructions.append(BytecodeInstruction(JUMP, end_label))
                
                # Compile else body
                instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))  # Placeholder
                instructions[-1].metadata['label'] = else_label
                for stmt in node.orelse:
                    instructions.extend(self.compile_statement(stmt))
            else:
                instructions.append(BytecodeInstruction(JUMP_IF_FALSE, end_label))
                
                # Compile if body
                for stmt in node.body:
                    instructions.extend(self.compile_statement(stmt))
            
            instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))  # Placeholder
            instructions[-1].metadata['label'] = end_label
        
        elif isinstance(node, ast.While):
            # Handle while loops
            start_label = self.new_label()
            end_label = self.new_label()
            
            # Loop condition
            instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))  # Placeholder
            instructions[-1].metadata['label'] = start_label
            instructions.extend(self.compile_expression(node.test))
            instructions.append(BytecodeInstruction(JUMP_IF_FALSE, end_label))
            
            # Loop body
            for stmt in node.body:
                instructions.extend(self.compile_statement(stmt))
            instructions.append(BytecodeInstruction(JUMP, start_label))
            
            # End of loop
            instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))  # Placeholder
            instructions[-1].metadata['label'] = end_label
        
        elif isinstance(node, ast.For):
            # Handle for loops (simplified - only handles basic iteration)
            # This is a simplified implementation - full for loop compilation is complex
            iter_var = self.get_local_var(f"__iter_{self.label_counter}")
            
            # Get iterator
            instructions.extend(self.compile_expression(node.iter))
            instructions.append(BytecodeInstruction(STORE_VAR, iter_var))
            
            # For loop body would go here - simplified for now
            # In a full implementation, we'd handle the iteration protocol
        
        elif isinstance(node, ast.Expr):
            # Handle expression statements
            instructions.extend(self.compile_expression(node.value))
            instructions.append(BytecodeInstruction(POP))
        
        elif isinstance(node, ast.Pass):
            # Handle pass statements
            instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))
            instructions.append(BytecodeInstruction(POP))
        
        return instructions


class LuaBytecodeCompiler(BytecodeCompiler):
    """Bytecode compiler for Lua AST nodes."""
    
    def __init__(self, complexity: int = COMPLEXITY_INTERMEDIATE):
        super().__init__(complexity)
        try:
            import luaparser.astnodes as lua_ast
            self.lua_ast = lua_ast
        except ImportError:
            self.lua_ast = None
    
    def compile_function(self, node: Any) -> Tuple[List[BytecodeInstruction], List[Any], int]:
        """Compile a Lua function to bytecode."""
        if self.lua_ast is None:
            raise ImportError("luaparser is required for Lua bytecode compilation")
        
        self.local_vars.clear()
        self.global_vars.clear()
        
        instructions = []
        
        # Compile function parameters as local variables
        if hasattr(node, 'args') and node.args:
            for arg in node.args:
                if hasattr(arg, 'id'):
                    self.get_local_var(arg.id)
        
        # Compile function body
        if hasattr(node, 'body'):
            for stmt in node.body:
                instructions.extend(self.compile_statement(stmt))
        
        # Ensure function returns something
        if not instructions or instructions[-1].opcode != RETURN:
            instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))
            instructions.append(BytecodeInstruction(RETURN))
        
        return instructions, self.constants.copy(), len(self.local_vars)
    
    def compile_expression(self, node: Any) -> List[BytecodeInstruction]:
        """Compile a Lua expression to bytecode."""
        if self.lua_ast is None:
            raise ImportError("luaparser is required for Lua bytecode compilation")
        
        instructions = []
        
        # Handle different Lua expression types
        if isinstance(node, self.lua_ast.Number):
            const_idx = self.add_constant(node.n)
            instructions.append(BytecodeInstruction(LOAD_CONST, const_idx))
        
        elif isinstance(node, self.lua_ast.String):
            const_idx = self.add_constant(node.s)
            instructions.append(BytecodeInstruction(LOAD_CONST, const_idx))
        
        elif isinstance(node, self.lua_ast.Name):
            if hasattr(node, 'id'):
                var_name = node.id
                if var_name in self.local_vars:
                    instructions.append(BytecodeInstruction(LOAD_VAR, self.get_local_var(var_name)))
                else:
                    instructions.append(BytecodeInstruction(LOAD_GLOBAL, var_name))
        
        elif isinstance(node, self.lua_ast.BinaryOp):
            instructions.extend(self.compile_expression(node.left))
            instructions.extend(self.compile_expression(node.right))
            
            op_map = {
                '+': BINARY_ADD,
                '-': BINARY_SUB,
                '*': BINARY_MUL,
                '/': BINARY_DIV,
                '%': BINARY_MOD,
                '^': BINARY_POW,
                '==': COMPARE_EQ,
                '~=': COMPARE_NE,
                '<': COMPARE_LT,
                '<=': COMPARE_LE,
                '>': COMPARE_GT,
                '>=': COMPARE_GE,
            }
            
            opcode = op_map.get(node.op, BINARY_ADD)
            instructions.append(BytecodeInstruction(opcode))
        
        elif isinstance(node, self.lua_ast.UnaryOp):
            instructions.extend(self.compile_expression(node.operand))
            
            if node.op == 'not':
                instructions.append(BytecodeInstruction(UNARY_NOT))
            elif node.op == '-':
                instructions.append(BytecodeInstruction(UNARY_NEGATIVE))
            elif node.op == '#':
                # Length operator - would need special handling
                pass
        
        elif isinstance(node, self.lua_ast.Call):
            # Compile function
            instructions.extend(self.compile_expression(node.func))
            
            # Compile arguments
            arg_count = len(node.args) if node.args else 0
            for arg in (node.args or []):
                instructions.extend(self.compile_expression(arg))
            
            instructions.append(BytecodeInstruction(CALL_FUNC, arg_count))
        
        elif isinstance(node, self.lua_ast.Table):
            # Handle table literals
            for field in (node.fields or []):
                if hasattr(field, 'key') and field.key:
                    instructions.extend(self.compile_expression(field.key))
                if hasattr(field, 'value'):
                    instructions.extend(self.compile_expression(field.value))
            
            instructions.append(BytecodeInstruction(BUILD_MAP, len(node.fields or [])))
        
        elif isinstance(node, self.lua_ast.Index):
            # Handle table indexing
            instructions.extend(self.compile_expression(node.idx))
            instructions.extend(self.compile_expression(node.value))
            instructions.append(BytecodeInstruction(LOAD_INDEX))
        
        elif isinstance(node, self.lua_ast.AndLoOp) or isinstance(node, self.lua_ast.OrLoOp):
            # Handle logical operators (short-circuit)
            end_label = self.new_label()
            
            instructions.extend(self.compile_expression(node.left))
            
            if isinstance(node, self.lua_ast.AndLoOp):
                instructions.append(BytecodeInstruction(DUP))
                instructions.append(BytecodeInstruction(JUMP_IF_FALSE, end_label))
            else:  # OrLoOp
                instructions.append(BytecodeInstruction(DUP))
                instructions.append(BytecodeInstruction(JUMP_IF_TRUE, end_label))
            
            instructions.append(BytecodeInstruction(POP))
            instructions.extend(self.compile_expression(node.right))
            
            instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))  # Placeholder
            instructions[-1].metadata['label'] = end_label
        
        return instructions
    
    def compile_statement(self, node: Any) -> List[BytecodeInstruction]:
        """Compile a Lua statement to bytecode."""
        if self.lua_ast is None:
            raise ImportError("luaparser is required for Lua bytecode compilation")
        
        instructions = []
        
        if isinstance(node, self.lua_ast.Assign):
            # Handle assignments
            if len(node.targets) == 1:
                instructions.extend(self.compile_expression(node.values[0]))
                
                target = node.targets[0]
                if isinstance(target, self.lua_ast.Name):
                    if target.id in self.local_vars:
                        instructions.append(BytecodeInstruction(STORE_VAR, self.get_local_var(target.id)))
                    else:
                        instructions.append(BytecodeInstruction(STORE_GLOBAL, target.id))
                elif isinstance(target, self.lua_ast.Index):
                    instructions.extend(self.compile_expression(target.value))
                    instructions.extend(self.compile_expression(target.idx))
                    instructions.append(BytecodeInstruction(STORE_INDEX))
        
        elif isinstance(node, self.lua_ast.Return):
            # Handle return statements
            if node.values:
                for value in node.values:
                    instructions.extend(self.compile_expression(value))
            else:
                instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))
            
            instructions.append(BytecodeInstruction(RETURN, len(node.values) if node.values else 1))
        
        elif isinstance(node, self.lua_ast.If):
            # Handle if statements
            end_label = self.new_label()
            
            # Compile condition
            instructions.extend(self.compile_expression(node.test))
            
            if node.else_body:
                else_label = self.new_label()
                instructions.append(BytecodeInstruction(JUMP_IF_FALSE, else_label))
                
                # Compile if body
                for stmt in node.body:
                    instructions.extend(self.compile_statement(stmt))
                instructions.append(BytecodeInstruction(JUMP, end_label))
                
                # Compile else body
                instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))  # Placeholder
                instructions[-1].metadata['label'] = else_label
                for stmt in node.else_body:
                    instructions.extend(self.compile_statement(stmt))
            else:
                instructions.append(BytecodeInstruction(JUMP_IF_FALSE, end_label))
                
                # Compile if body
                for stmt in node.body:
                    instructions.extend(self.compile_statement(stmt))
            
            instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))  # Placeholder
            instructions[-1].metadata['label'] = end_label
        
        elif isinstance(node, self.lua_ast.While):
            # Handle while loops
            start_label = self.new_label()
            end_label = self.new_label()
            
            # Loop condition
            instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))  # Placeholder
            instructions[-1].metadata['label'] = start_label
            instructions.extend(self.compile_expression(node.test))
            instructions.append(BytecodeInstruction(JUMP_IF_FALSE, end_label))
            
            # Loop body
            for stmt in node.body:
                instructions.extend(self.compile_statement(stmt))
            instructions.append(BytecodeInstruction(JUMP, start_label))
            
            # End of loop
            instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))  # Placeholder
            instructions[-1].metadata['label'] = end_label
        
        elif isinstance(node, self.lua_ast.Fornum):
            # Handle numeric for loops
            # Initialize loop variable
            instructions.extend(self.compile_expression(node.start))
            loop_var_idx = self.get_local_var(node.target.id)
            instructions.append(BytecodeInstruction(STORE_VAR, loop_var_idx))
            
            start_label = self.new_label()
            end_label = self.new_label()
            
            # Loop condition check
            instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))  # Placeholder
            instructions[-1].metadata['label'] = start_label
            instructions.append(BytecodeInstruction(LOAD_VAR, loop_var_idx))
            instructions.extend(self.compile_expression(node.stop))
            instructions.append(BytecodeInstruction(COMPARE_LE))
            instructions.append(BytecodeInstruction(JUMP_IF_FALSE, end_label))
            
            # Loop body
            for stmt in node.body:
                instructions.extend(self.compile_statement(stmt))
            
            # Increment loop variable
            instructions.append(BytecodeInstruction(LOAD_VAR, loop_var_idx))
            instructions.extend(self.compile_expression(node.step or self.lua_ast.Number(1)))
            instructions.append(BytecodeInstruction(BINARY_ADD))
            instructions.append(BytecodeInstruction(STORE_VAR, loop_var_idx))
            
            instructions.append(BytecodeInstruction(JUMP, start_label))
            
            # End of loop
            instructions.append(BytecodeInstruction(LOAD_CONST, self.add_constant(None)))  # Placeholder
            instructions[-1].metadata['label'] = end_label
        
        elif isinstance(node, self.lua_ast.Do):
            # Handle do blocks
            for stmt in node.body:
                instructions.extend(self.compile_statement(stmt))
        
        elif isinstance(node, self.lua_ast.Break):
            # Handle break statements
            # Would need to track loop context for proper break handling
            pass
        
        return instructions


def create_bytecode_compiler(language: str, complexity: int = COMPLEXITY_INTERMEDIATE) -> BytecodeCompiler:
    """Factory function to create appropriate bytecode compiler."""
    if language.lower() == 'python':
        return PythonBytecodeCompiler(complexity)
    elif language.lower() == 'lua':
        return LuaBytecodeCompiler(complexity)
    else:
        raise ValueError(f"Unsupported language: {language}")