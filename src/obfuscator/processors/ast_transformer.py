"""AST transformation module for Python code obfuscation.

This module provides a base infrastructure for implementing AST transformations,
along with example transformations like constant folding and string encryption.
All transformers extend the base ASTTransformer class which provides error tracking,
logging, and common utilities.
"""

from __future__ import annotations

import ast
import base64
import math
import random
import secrets
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from obfuscator.utils.logger import get_logger

# Try to import cryptography for AES encryption
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

# Try to import luaparser for Lua AST manipulation
try:
    from luaparser import ast as lua_ast
    from luaparser import astnodes as lua_nodes
    LUAPARSER_AVAILABLE = True
except ImportError:
    LUAPARSER_AVAILABLE = False

logger = get_logger("obfuscator.processors.ast_transformer")

# Module constants
MAX_TRANSFORMATION_DEPTH: int = 1000
"""Maximum recursion depth for AST transformations to prevent infinite recursion.

This safety limit prevents stack overflow errors in cases where transformation
logic might accidentally create circular references or infinitely nested structures.
"""


@dataclass
class TransformResult:
    """Result of an AST transformation operation.

    This dataclass encapsulates the outcome of applying a transformer to an AST,
    including success/failure status, the transformed AST (if successful), and
    any errors that occurred during transformation.

    Attributes:
        ast_node: The transformed AST node, or None if transformation failed.
        success: Whether the transformation completed successfully.
        transformation_count: Number of AST nodes that were transformed.
        errors: List of error messages describing any transformation failures.

    Example:
        >>> transformer = ConstantFoldingTransformer()
        >>> result = transformer.transform(ast_node)
        >>> if result.success:
        ...     print(f"Transformed {result.transformation_count} nodes")
        ... else:
        ...     for error in result.errors:
        ...         print(f"Error: {error}")
    """

    ast_node: ast.AST | None
    success: bool
    transformation_count: int
    errors: list[str]


class ASTTransformer(ast.NodeTransformer):
    """Base class for AST transformations.

    Extends ast.NodeTransformer to provide common infrastructure for AST
transformations, including error tracking, transformation counting,
    and logging capabilities.

    Subclasses should implement visit methods for specific node types they
    wish to transform (e.g., visit_BinOp, visit_UnaryOp, etc.).

    Attributes:
        transformation_count: Counter tracking the number of nodes transformed.
        errors: List of error messages collected during transformation.
        logger: Module logger for recording transformation details.

    Example:
        >>> class MyTransformer(ASTTransformer):
        ...     def visit_Name(self, node: ast.Name) -> ast.AST:
        ...         # Transform Name nodes
        ...         self.transformation_count += 1
        ...         return self.generic_visit(node)
        ...
        >>> transformer = MyTransformer()
        >>> result = transformer.transform(ast_node)
        >>> print(f"Transformed {result.transformation_count} nodes")
    """

    def __init__(self) -> None:
        """Initialize the AST transformer.

        Sets up tracking attributes for transformation counting and error collection.
        """
        super().__init__()
        self.transformation_count: int = 0
        self.errors: list[str] = []
        self.logger =logger

    def transform(self, ast_node: ast.AST) -> TransformResult:
        """Apply this transformer to an AST node.

        This method orchestrates the transformation process:
        1. Fixes missing locations in the AST
        2. Visits and transforms nodes using the visitor pattern
        3. Handles errors gracefully
        4. Returns a structured result

        Args:
            ast_node: The AST node to transform (typically an ast.Module).

        Returns:
            A TransformResult containing the transformed AST (if successful),
            success status, transformation count, and any errors.

        Raises:
            No exceptions are raised directly; errors are captured in the
            returned TransformResult.

        Example:
            >>> transformer = ConstantFoldingTransformer()
            >>> tree = ast.parse("x = 2 + 3")
            >>> result = transformer.transform(tree)
            >>> if result.success:
            ...     code = ast.unparse(result.ast_node)
            ...     print(code)  # Output: x = 5
        """
        # Reset state for this transformation
        self.transformation_count = 0
        self.errors = []

        try:
            # Fix missing locations in the AST
            ast.fix_missing_locations(ast_node)

            # Apply transformations by visiting the AST
            transformed_node = self.visit(ast_node)

            # Ensure result is still valid
            if transformed_node is None:
                raise ValueError("Transformation returned None")

            # Fix missing locations on newly created nodes
            ast.fix_missing_locations(transformed_node)

            self.logger.debug(
                f"Transformation completed: {self.transformation_count} nodes transformed"
            )

            return TransformResult(
                ast_node=transformed_node,
                success=True,
                transformation_count=self.transformation_count,
                errors=self.errors,
            )

        except Exception as e:
            error_msg = f"Transformation failed: {e.__class__.__name__}: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.errors.append(error_msg)

            return TransformResult(
                ast_node=None,
                success=False,
                transformation_count=self.transformation_count,
                errors=self.errors,
            )


class ConstantArrayTransformer(ASTTransformer):
    """Performs array obfuscation through element shuffling and index mapping.

    This transformer obfuscates constant arrays in both Python and Lua code by:
    - Shuffling array elements in a random/deterministic order
    - Injecting mapping dictionaries/tables to map original indices to shuffled positions
    - Rewriting array access expressions to use the mapping

    For Python, the transformer:
    - Detects ast.List nodes containing only constant elements
    - Shuffles arrays with 2+ constant elements
    - Rewrites ast.Subscript access to use mapping lookup

    For Lua, the transformer:
    - Detects lua_nodes.Table nodes with constant-only values
    - Shuffles table elements with 2+ constants
    - Rewrites index operations to use mapping tables

    Attributes:
        config: ObfuscationConfig with array_shuffle_seed option
        shuffle_seed: Seed for random shuffling (None for random, int for deterministic)
        language_mode: Detected language ('python', 'lua', or None)
        runtime_injected: Whether mapping runtime has been injected
        array_mappings: Dictionary mapping array IDs to shuffle indices
        array_id_counter: Counter for generating unique array IDs
        transformed_arrays: Set tracking which arrays have been transformed

    Example:
        >>> from obfuscator.core.config import ObfuscationConfig
        >>> config = ObfuscationConfig(name="test", features={"constant_array": True})
        >>> transformer = ConstantArrayTransformer(config)
        >>> tree = ast.parse("x = [1, 2, 3, 4]; print(x[2])")
        >>> result = transformer.transform(tree)
        >>> if result.success:
        ...     print(f"Transformed {result.transformation_count} arrays")
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        shuffle_seed: Optional[int] = None,
    ) -> None:
        """Initialize the constant array transformer.

        Args:
            config: ObfuscationConfig instance with array_shuffle_seed option.
                   If provided, shuffle_seed is extracted from config.options.
            shuffle_seed: Seed for random shuffling. If None, uses random seed.
                         If int, uses deterministic seed for reproducibility.
        """
        super().__init__()
        self.logger = get_logger("obfuscator.processors.constant_array")

        # Determine shuffle seed from config or parameter
        if shuffle_seed is not None:
            self.shuffle_seed = shuffle_seed
        elif config is not None and hasattr(config, 'options'):
            self.shuffle_seed = config.options.get('array_shuffle_seed', None)
        else:
            self.shuffle_seed = None

        self.config = config

        # Language detection and state tracking
        self.language_mode: Optional[str] = None
        self.runtime_injected: bool = False

        # Array transformation state
        self.array_mappings: Dict[str, list[int]] = {}  # array_id -> original_to_shuffled_map
        self.array_id_counter: int = 0
        self.transformed_arrays: set[int] = set()  # Track transformed array node IDs
        self.array_to_id: Dict[int, str] = {}  # Map array node id() to array_id
        self.var_to_array_id: Dict[str, str] = {}  # Map variable name to array_id

        self.logger.debug(
            f"ConstantArrayTransformer initialized with shuffle_seed={self.shuffle_seed}"
        )

    def _generate_array_id(self) -> str:
        """Generate a unique array ID.

        Returns:
            Unique array ID string (e.g., "_arr_0", "_arr_1").
        """
        array_id = f"_arr_{self.array_id_counter}"
        self.array_id_counter += 1
        return array_id

    def _generate_shuffle_mapping(self, array_length: int) -> tuple[list[int], list[int]]:
        """Generate shuffle mapping for an array.

        Creates a mapping from original indices to shuffled indices and vice versa.

        Args:
            array_length: Length of the array to shuffle.

        Returns:
            Tuple of (original_to_shuffled_map, shuffled_to_original_map).
            - original_to_shuffled_map[i] gives the shuffled position of original element i
            - shuffled_to_original_map[j] gives the original position of shuffled element j
        """
        # Create list of original indices
        indices = list(range(array_length))

        # Shuffle indices using configured seed
        if self.shuffle_seed is not None:
            # Use deterministic seed for reproducibility
            original_state = random.getstate()
            random.seed(self.shuffle_seed + self.array_id_counter)
            random.shuffle(indices)
            random.setstate(original_state)
        else:
            # Use random seed
            random.shuffle(indices)

        # Create mapping from original to shuffled positions
        original_to_shuffled = [0] * array_length
        shuffled_to_original = [0] * array_length

        for original_idx, shuffled_idx in enumerate(indices):
            original_to_shuffled[original_idx] = shuffled_idx
            shuffled_to_original[shuffled_idx] = original_idx

        return original_to_shuffled, shuffled_to_original

    def _is_constant_array(self, node: ast.List) -> bool:
        """Check if a Python List node contains only constant elements.

        Args:
            node: The ast.List node to check.

        Returns:
            True if all elements are constants and array has 2+ elements.
        """
        # Skip empty arrays and single-element arrays
        if len(node.elts) < 2:
            return False

        # Check if all elements are constants
        for elt in node.elts:
            if not isinstance(elt, ast.Constant):
                return False

        return True

    def _is_constant_table(self, node: Any) -> bool:
        """Check if a Lua Table node contains only constant values.

        Args:
            node: The lua_nodes.Table node to check.

        Returns:
            True if all values are constants and table has 2+ elements.
        """
        if not LUAPARSER_AVAILABLE:
            return False

        # Skip empty tables
        if not hasattr(node, 'fields') or not node.fields:
            return False

        # Skip single-element tables
        if len(node.fields) < 2:
            return False

        # Check if all values are constants
        for field in node.fields:
            if not isinstance(field, lua_nodes.Field):
                return False

            # Check the value part of each field
            value = field.value
            if isinstance(value, (lua_nodes.Number, lua_nodes.String,
                                 lua_nodes.TrueExpr, lua_nodes.FalseExpr, lua_nodes.Nil)):
                continue
            else:
                return False

        return True

    def _generate_python_runtime(self, array_id: str, mapping: list[int]) -> list[ast.stmt]:
        """Generate Python runtime code for an array mapping dictionary.

        Args:
            array_id: Unique identifier for the array.
            mapping: List mapping original indices to shuffled positions.

        Returns:
            List of AST statement nodes creating the mapping dictionary.
        """
        mapping_dict = ast.Dict(
            keys=[ast.Constant(value=i) for i in range(len(mapping))],
            values=[ast.Constant(value=mapping[i]) for i in range(len(mapping))]
        )

        assign = ast.Assign(
            targets=[ast.Name(id=f'{array_id}_map', ctx=ast.Store())],
            value=mapping_dict
        )

        return [assign]

    def _generate_lua_runtime(self, array_id: str, mapping: list[int]) -> str:
        """Generate Lua runtime code for an array mapping table.

        Args:
            array_id: Unique identifier for the array.
            mapping: List mapping original indices to shuffled positions.

        Returns:
            Lua code string creating the mapping table.
        """
        # Lua uses 1-based indexing, so we add 1 to both keys and values
        pairs = ', '.join([f'[{i + 1}] = {mapping[i] + 1}' for i in range(len(mapping))])
        return f'local {array_id}_map = {{{pairs}}}'

    def _inject_python_runtime(self, module: ast.Module) -> None:
        """Inject Python mapping dictionaries at the beginning of a module.

        Args:
            module: The Python AST Module node to inject into.
        """
        if self.runtime_injected:
            return

        # Collect all mapping assignments
        runtime_nodes = []
        for array_id, mapping in self.array_mappings.items():
            runtime_nodes.extend(self._generate_python_runtime(array_id, mapping))

        if not runtime_nodes:
            return

        # Insert runtime at the beginning of the module body
        # But preserve __future__ imports and docstrings
        insert_index = 0

        for i, stmt in enumerate(module.body):
            # Skip __future__ imports
            if isinstance(stmt, ast.ImportFrom) and stmt.module == '__future__':
                insert_index = i + 1
            # Skip module docstring (first string expression)
            elif i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                insert_index = 1
            else:
                break

        # Insert runtime nodes at the calculated position
        for j, node in enumerate(runtime_nodes):
            ast.fix_missing_locations(node)
            module.body.insert(insert_index + j, node)

        self.runtime_injected = True
        self.logger.debug("Injected Python array mapping runtime")

    def _inject_lua_runtime(self, chunk: Any) -> None:
        """Inject Lua mapping tables at the beginning of a chunk.

        Args:
            chunk: The Lua AST Chunk node to inject into.
        """
        if not LUAPARSER_AVAILABLE:
            self.logger.warning("luaparser not available, skipping Lua runtime injection")
            return

        if self.runtime_injected:
            return

        # Generate runtime code for all mappings
        runtime_code_lines = []
        for array_id, mapping in self.array_mappings.items():
            runtime_code_lines.append(self._generate_lua_runtime(array_id, mapping))

        if not runtime_code_lines:
            return

        runtime_code = '\n'.join(runtime_code_lines)

        try:
            runtime_tree = lua_ast.parse(runtime_code)
            if hasattr(runtime_tree, 'body') and hasattr(chunk, 'body'):
                # Insert runtime statements at the beginning
                if hasattr(chunk.body, 'body'):
                    # chunk.body is a Block with a body attribute
                    chunk.body.body = runtime_tree.body.body + chunk.body.body
                else:
                    chunk.body = runtime_tree.body.body + chunk.body
            self.runtime_injected = True
            self.logger.debug("Injected Lua array mapping runtime")
        except Exception as e:
            error_msg = f"Failed to inject Lua runtime: {e}"
            self.errors.append(error_msg)
            self.logger.error(error_msg)

    def visit_List(self, node: ast.List) -> ast.AST:
        """Visit and potentially transform Python List nodes.

        If the list contains only constant elements, shuffles the elements
        and stores the mapping for later access rewriting.

        Args:
            node: The List AST node to potentially transform.

        Returns:
            Either a new List node with shuffled elements, or the original node.
        """
        # First visit children in case of nested structures
        node.elts = [self.visit(elt) for elt in node.elts]

        # Check if this is a constant array
        if not self._is_constant_array(node):
            return node

        # Generate array ID and shuffle mapping
        array_id = self._generate_array_id()
        original_to_shuffled, shuffled_to_original = self._generate_shuffle_mapping(len(node.elts))

        # Store the mapping (original -> shuffled)
        self.array_mappings[array_id] = original_to_shuffled

        # Create shuffled list array
        shuffled_elements = [node.elts[shuffled_to_original[i]] for i in range(len(node.elts))]
        new_list = ast.List(elts=shuffled_elements, ctx=node.ctx)

        # Track this array as transformed using node ID
        self.transformed_arrays.add(id(node))
        self.transformed_arrays.add(id(new_list))

        # Map the new list node to its array_id for later subscript rewriting
        self.array_to_id[id(new_list)] = array_id

        # Copy location and return
        ast.copy_location(new_list, node)
        self.transformation_count += 1
        self.logger.debug(
            f"Shuffled array at line {getattr(node, 'lineno', '?')}: "
            f"{len(node.elts)} elements, mapping ID: {array_id}"
        )

        return new_list

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """Visit and potentially rewrite array subscript access.

        If the subscript is accessing a transformed array, rewrites the index
        to use the mapping lookup.

        Args:
            node: The Subscript AST node to potentially transform.

        Returns:
            Either a new Subscript with mapping lookup, or the original node.
        """
        # Visit children first
        node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)

        # Check if we're accessing a tracked transformed array variable
        if isinstance(node.value, ast.Name) and node.value.id in self.var_to_array_id:
            array_id = self.var_to_array_id[node.value.id]

            # Rewrite: arr[index] -> arr[_arr_<id>_map[index]]
            # Create the mapping lookup: _arr_<id>_map[index]
            mapping_lookup = ast.Subscript(
                value=ast.Name(id=f'{array_id}_map', ctx=ast.Load()),
                slice=node.slice,
                ctx=ast.Load()
            )

            # Create new subscript with mapping lookup as the index
            new_node = ast.Subscript(
                value=node.value,
                slice=mapping_lookup,
                ctx=node.ctx
            )

            ast.copy_location(new_node, node)
            ast.copy_location(mapping_lookup, node)
            ast.copy_location(mapping_lookup.value, node)

            self.logger.debug(
                f"Rewrote subscript access for variable '{node.value.id}' "
                f"using mapping '{array_id}_map'"
            )

            return new_node

        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Visit assignment nodes to track transformed arrays.

        This method tracks which variable names reference transformed arrays
        so that subsequent subscript operations can be rewritten.

        Args:
            node: The Assign AST node.

        Returns:
            The potentially transformed Assign node.
        """
        # Visit the value (which might be a transformed array)
        node.value = self.visit(node.value)

        # Track if we're assigning a transformed array to a variable
        if isinstance(node.value, ast.List) and id(node.value) in self.array_to_id:
            array_id = self.array_to_id[id(node.value)]
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Map variable name to array_id
                    self.var_to_array_id[target.id] = array_id
                    self.logger.debug(
                        f"Tracked variable '{target.id}' -> array_id '{array_id}'"
                    )

        # Visit targets
        node.targets = [self.visit(target) for target in node.targets]

        return node

    def _transform_lua_table_node(self, node: Any) -> Any:
        """Transform a Lua Table node by shuffling its elements.

        Args:
            node: The lua_nodes.Table node to transform.

        Returns:
            A shuffled Table node, or the original if not eligible.
        """
        if not LUAPARSER_AVAILABLE:
            return node

        # Check if this is a constant table
        if not self._is_constant_table(node):
            return node

        # Generate array ID and shuffle mapping
        array_id = self._generate_array_id()
        original_to_shuffled, shuffled_to_original = self._generate_shuffle_mapping(len(node.fields))

        # Store the mapping
        self.array_mappings[array_id] = original_to_shuffled

        # Create shuffled table by reordering fields
        shuffled_fields = [node.fields[shuffled_to_original[i]] for i in range(len(node.fields))]

        # Create new table node with shuffled fields
        new_table = lua_nodes.Table(shuffled_fields)

        # Track the table-to-array-id mapping for later index rewriting
        self.array_to_id[id(new_table)] = array_id

        self.transformation_count += 1
        self.logger.debug(
            f"Shuffled Lua table at line {getattr(node, 'line', '?')}: "
            f"{len(node.fields)} elements, mapping ID: {array_id}"
        )

        return new_table

    def _track_lua_assignment(self, node: Any) -> None:
        """Track Lua local variable assignments to transformed tables.

        Args:
            node: A lua_nodes.LocalAssign or lua_nodes.Assign node.
        """
        if not LUAPARSER_AVAILABLE:
            return

        # Check if this is a local assignment or regular assignment
        if isinstance(node, (lua_nodes.LocalAssign, lua_nodes.Assign)):
            # Get targets and values
            targets = getattr(node, 'targets', [])
            values = getattr(node, 'values', [])

            # Track each assignment
            for i, target in enumerate(targets):
                if i < len(values):
                    value = values[i]
                    # Check if value is a transformed table
                    if isinstance(value, lua_nodes.Table) and id(value) in self.array_to_id:
                        array_id = self.array_to_id[id(value)]
                        # Get variable name from target
                        if isinstance(target, lua_nodes.Name):
                            var_name = getattr(target, 'id', None)
                            if var_name:
                                self.var_to_array_id[var_name] = array_id
                                self.logger.debug(
                                    f"Tracked Lua variable '{var_name}' -> array_id '{array_id}'"
                                )

    def _rewrite_lua_index(self, node: Any) -> Any:
        """Rewrite Lua index operations for transformed tables.

        Args:
            node: A lua_nodes.Index node.

        Returns:
            Modified Index node or original node.
        """
        if not LUAPARSER_AVAILABLE:
            return node

        # Check if this is an Index operation on a tracked variable
        if isinstance(node, lua_nodes.Index):
            # Get the table being indexed
            table = getattr(node, 'value', None)
            if isinstance(table, lua_nodes.Name):
                var_name = getattr(table, 'id', None)
                if var_name and var_name in self.var_to_array_id:
                    array_id = self.var_to_array_id[var_name]

                    # Rewrite: table[index] -> table[_arr_<id>_map[index]]
                    # Create mapping lookup: _arr_<id>_map[index]
                    mapping_name = lua_nodes.Name(f'{array_id}_map')
                    original_index = getattr(node, 'idx', None)

                    if original_index:
                        mapping_lookup = lua_nodes.Index(
                            value=mapping_name,
                            idx=original_index
                        )

                        # Update the index to use the mapping
                        node.idx = mapping_lookup

                        self.logger.debug(
                            f"Rewrote Lua index for variable '{var_name}' "
                            f"using mapping '{array_id}_map'"
                        )

        return node

    def _traverse_lua_ast(self, node: Any) -> None:
        """Recursively traverse and transform Lua AST, shuffling Table nodes.

        This method walks the Lua AST and replaces lua_nodes.Table
        nodes with shuffled versions in-place, tracks assignments,
        and rewrites index operations.

        Args:
            node: The current Lua AST node to process.
        """
        if not LUAPARSER_AVAILABLE or node is None:
            return

        # Track assignments to transformed tables
        if isinstance(node, (lua_nodes.LocalAssign, lua_nodes.Assign)):
            self._track_lua_assignment(node)

        # Rewrite index operations
        if isinstance(node, lua_nodes.Index):
            self._rewrite_lua_index(node)

        # Get all attribute names that might contain child nodes
        for attr_name in dir(node):
            if attr_name.startswith('_'):
                continue

            attr = getattr(node, attr_name, None)
            if attr is None:
                continue

            # Handle lists of nodes
            if isinstance(attr, list):
                for i, item in enumerate(attr):
                    if isinstance(item, lua_nodes.Table):
                        # Replace Table node with shuffled version
                        attr[i] = self._transform_lua_table_node(item)
                    elif hasattr(item, '__class__') and hasattr(item.__class__, '__module__'):
                        if 'astnodes' in item.__class__.__module__:
                            self._traverse_lua_ast(item)

            # Handle single nodes
            elif isinstance(attr, lua_nodes.Table):
                # Replace Table node with shuffled version
                setattr(node, attr_name, self._transform_lua_table_node(attr))
            elif hasattr(attr, '__class__') and hasattr(attr.__class__, '__module__'):
                if 'astnodes' in attr.__class__.__module__:
                    self._traverse_lua_ast(attr)

    def transform(self, ast_node: ast.AST) -> TransformResult:
        """Apply constant array transformation to an AST.

        This method orchestrates the transformation:
        1. Detects language from AST node type
        2. Applies array shuffling transformations
        3. Injects mapping runtime at module/chunk level
        4. Returns structured result

        Args:
            ast_node: The AST node to transform (ast.Module for Python).

        Returns:
            TransformResult with transformed AST, success status, and metrics.
        """
        # Reset state for this transformation
        self.transformation_count = 0
        self.errors = []
        self.runtime_injected = False
        self.array_mappings = {}
        self.array_id_counter = 0
        self.transformed_arrays = set()
        self.array_to_id = {}
        self.var_to_array_id = {}

        try:
            # Detect language and apply transformations, then inject runtime
            if isinstance(ast_node, ast.Module):
                self.language_mode = 'python'

                # Fix missing locations before transformation
                ast.fix_missing_locations(ast_node)

                # Apply Python transformations by visiting the AST
                transformed_node = self.visit(ast_node)

                if transformed_node is None:
                    raise ValueError("Transformation returned None")

                # Inject runtime after transformation
                self._inject_python_runtime(transformed_node)

                # Fix missing locations on newly created nodes
                ast.fix_missing_locations(transformed_node)

            elif LUAPARSER_AVAILABLE and isinstance(ast_node, lua_nodes.Chunk):
                self.language_mode = 'lua'

                # Apply Lua transformations using custom traversal
                self._traverse_lua_ast(ast_node)

                # Inject runtime after transformation
                self._inject_lua_runtime(ast_node)

                # Lua transformation is in-place
                transformed_node = ast_node

            else:
                # For other node types, try to proceed without runtime injection
                self.logger.warning(
                    f"Unknown AST type {type(ast_node).__name__}, "
                    "proceeding without runtime injection"
                )
                transformed_node = ast_node

            self.logger.debug(
                f"Constant array transformation completed: "
                f"{self.transformation_count} arrays transformed"
            )

            return TransformResult(
                ast_node=transformed_node,
                success=True,
                transformation_count=self.transformation_count,
                errors=self.errors,
            )

        except Exception as e:
            error_msg = f"Constant array transformation failed: {e.__class__.__name__}: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.errors.append(error_msg)

            return TransformResult(
                ast_node=None,
                success=False,
                transformation_count=self.transformation_count,
                errors=self.errors,
            )


class ConstantFoldingTransformer(ASTTransformer):
    """Performs constant folding optimization on AST nodes.

    Constant folding evaluates constant expressions at parse time rather than
    runtime, simplifying the code and potentially improving performance.
    This transformer handles binary operations (e.g., 2 + 3 → 5) and unary
    operations (e.g., -5 → -5, not True → False).

    Supported operations:
    - Binary: Add, Sub, Mult, Div, FloorDiv, Mod, Pow with numeric operands
    - Unary: UAdd, USub, Not with numeric or boolean operands

    Note:
        Only pure constant expressions (where all operands are ast.Constant)
        are folded. Expressions involving variables or function calls are left
        unchanged.
    """

    def __init__(self) -> None:
        """Initialize the constant folding transformer."""
        super().__init__()
        self.logger = get_logger("obfuscator.processors.ast_transformer")

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        """Visit and potentially transform binary operation nodes.

        If both operands are constants, evaluates the operation and returns
        a new Constant node with the computed value. Otherwise, continues
        normal traversal.

        Args:
            node: The BinaryOp AST node to potentially transform.

        Returns:
            Either a new ast.Constant node with the computed value, or the
            original node (or visited version) if folding is not possible.

        Raises:
            Errors are caught and logged; original node is returned unchanged.

        Example:
            >>> # Input: x = 2 + 3
            >>> transformer = ConstantFoldingTransformer()
            >>> tree = ast.parse("x = 2 + 3")
            >>> result = transformer.transform(tree)
            >>> ast.unparse(result.ast_node)  # Output: x = 5
        """
        # First visit children to process nested expressions
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)

        # Check if both operands are constants
        if not isinstance(node.left, ast.Constant) or not isinstance(
            node.right, ast.Constant
        ):
            return node

        left_val = node.left.value
        right_val = node.right.value

        # Ensure both operands are numeric
        if not isinstance(left_val, (int, float, complex)) or not isinstance(
            right_val, (int, float, complex)
        ):
            return node

        try:
            op_type = node.op
            result: int | float | complex

            # Handle different operation types
            if isinstance(op_type, ast.Add):
                result = left_val + right_val
            elif isinstance(op_type, ast.Sub):
                result = left_val - right_val
            elif isinstance(op_type, ast.Mult):
                result = left_val * right_val
            elif isinstance(op_type, ast.Div):
                result = left_val / right_val
            elif isinstance(op_type, ast.FloorDiv):
                result = left_val // right_val
            elif isinstance(op_type, ast.Mod):
                result = left_val % right_val
            elif isinstance(op_type, ast.Pow):
                result = left_val ** right_val
            else:
                # Unsupported operation type
                return node

            # Create new constant node with computed value
            new_node = ast.Constant(value=result)
            ast.copy_location(new_node, node)

            # Track transformation
            self.transformation_count += 1
            self.logger.debug(
                f"Folded constant expression: {left_val} "
                f"{op_type.__class__.__name__} {right_val} → {result}"
            )

            return new_node

        except ZeroDivisionError as e:
            error_msg = f"Division by zero in constant folding at line {getattr(node, 'lineno', '?')}"
            self.errors.append(error_msg)
            self.logger.warning(error_msg)
            return node

        except OverflowError as e:
            error_msg = f"Overflow in constant folding at line {getattr(node, 'lineno', '?')}"
            self.errors.append(error_msg)
            self.logger.warning(error_msg)
            return node

        except ValueError as e:
            error_msg = f"Invalid value in constant folding at line {getattr(node, 'lineno', '?')}: {e}"
            self.errors.append(error_msg)
            self.logger.warning(error_msg)
            return node

        except TypeError as e:
            error_msg = f"Type error in constant folding at line {getattr(node, 'lineno', '?')}: {e}"
            self.errors.append(error_msg)
            self.logger.warning(error_msg)
            return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        """Visit and potentially transform unary operation nodes.

        If the operand is a constant, evaluates the operation and returns
        a new Constant node with the computed value. Otherwise, continues
        normal traversal.

        Args:
            node: The UnaryOp AST node to potentially transform.

        Returns:
            Either a new ast.Constant node with the computed value, or the
            original node (or visited version) if folding is not possible.

        Raises:
            Errors are caught and logged; original node is returned unchanged.

        Example:
            >>> # Input: x = -5
            >>> transformer = ConstantFoldingTransformer()
            >>> tree = ast.parse("x = -5")
            >>> result = transformer.transform(tree)
            >>> ast.unparse(result.ast_node)  # Output: x = -5
        """
        # First visit the operand to process nested expressions
        node.operand = self.visit(node.operand)

        # Check if operand is a constant
        if not isinstance(node.operand, ast.Constant):
            return node

        value = node.operand.value

        # Only fold numeric and boolean operations
        if not isinstance(value, (int, float, complex, bool)):
            return node

        try:
            op_type = node.op
            result: int | float | complex | bool

            # Handle different operation types
            if isinstance(op_type, ast.UAdd):
                result = +value
            elif isinstance(op_type, ast.USub):
                result = -value
            elif isinstance(op_type, ast.Not):
                result = not value
            else:
                # Unsupported operation type
                return node

            # Create new constant node with computed value
            new_node = ast.Constant(value=result)
            ast.copy_location(new_node, node)

            # Track transformation
            self.transformation_count += 1
            self.logger.debug(
                f"Folded unary constant: {op_type.__class__.__name__}({value}) → {result}"
            )

            return new_node

        except (ValueError, TypeError) as e:
            error_msg = f"Invalid operation in constant folding at line {getattr(node, 'lineno', '?')}: {e}"
            self.errors.append(error_msg)
            self.logger.warning(error_msg)
            return node


class NumberObfuscationTransformer(ASTTransformer):
    """Performs number obfuscation by replacing numeric constants with arithmetic expressions.

    This transformer replaces numeric literals in both Python and Lua code with
    equivalent arithmetic expressions that evaluate to the same value. This makes
    the code harder to understand and reverse engineer while maintaining the same
    runtime behavior.

    The transformer supports configurable complexity levels (1-5) that determine
    the sophistication of the generated expressions:
    - Level 1: Simple operations (addition/subtraction)
    - Level 2: Mixed operations with multiplication/division
    - Level 3: Includes bitwise operations
    - Level 4: Complex nested expressions
    - Level 5: Advanced obfuscation with multiple operation types

    For Python, the transformer:
    - Detects ast.Constant nodes with numeric values
    - Replaces them with equivalent arithmetic expressions
    - Preserves location information using ast.copy_location()

    For Lua, the transformer:
    - Detects lua_nodes.Number nodes
    - Replaces them with equivalent Lua arithmetic expressions
    - Uses custom AST traversal for Lua nodes

    Attributes:
        config: ObfuscationConfig with number_obfuscation options
        complexity: Complexity level (1-5) for expression generation
        min_value: Minimum value to obfuscate (skip smaller numbers)
        max_value: Maximum value to obfuscate (skip larger numbers)
        language_mode: Detected language ('python', 'lua', or None)
        transformation_count: Counter for number of transformations applied
        errors: List of error messages from transformations

    Example:
        >>> from obfuscator.core.config import ObfuscationConfig
        >>> config = ObfuscationConfig(
        ...     name="test",
        ...     features={"number_obfuscation": True},
        ...     options={"number_obfuscation_complexity": 3}
        ... )
        >>> transformer = NumberObfuscationTransformer(config)
        >>> tree = ast.parse("x = 42")
        >>> result = transformer.transform(tree)
        >>> if result.success:
        ...     print(f"Obfuscated {result.transformation_count} numbers")
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        complexity: Optional[int] = None,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> None:
        """Initialize the number obfuscation transformer.

        Args:
            config: ObfuscationConfig instance with number_obfuscation options.
                   If provided, settings are extracted from config.options.
            complexity: Override for complexity level (1-5). If not provided,
                       uses config or defaults to 3.
            min_value: Minimum value to obfuscate. If not provided, uses config
                      or defaults to 10.
            max_value: Maximum value to obfuscate. If not provided, uses config
                      or defaults to 1000000.
        """
        super().__init__()
        self.logger = get_logger("obfuscator.processors.number_obfuscation")

        # Determine settings from config or parameters
        if complexity is not None:
            self.complexity = complexity
        elif config is not None and hasattr(config, 'options'):
            self.complexity = config.options.get('number_obfuscation_complexity', 3)
        else:
            self.complexity = 3

        # Validate complexity level
        if not 1 <= self.complexity <= 5:
            self.logger.warning(
                f"Complexity level {self.complexity} is outside range 1-5, using 3"
            )
            self.complexity = 3

        if min_value is not None:
            self.min_value = min_value
        elif config is not None and hasattr(config, 'options'):
            self.min_value = config.options.get('number_obfuscation_min_value', 10)
        else:
            self.min_value = 10

        if max_value is not None:
            self.max_value = max_value
        elif config is not None and hasattr(config, 'options'):
            self.max_value = config.options.get('number_obfuscation_max_value', 1000000)
        else:
            self.max_value = 1000000

        self.config = config
        self.language_mode: Optional[str] = None

        self.logger.debug(
            f"NumberObfuscationTransformer initialized with "
            f"complexity={self.complexity}, min_value={self.min_value}, max_value={self.max_value}"
        )

    def _should_obfuscate_number(self, value: Union[int, float, complex]) -> bool:
        """Determine if a number should be obfuscated.

        Args:
            value: The numeric value to check.

        Returns:
            True if the number should be obfuscated, False otherwise.
        """
        # Skip complex numbers
        if isinstance(value, complex):
            return False

        # Skip special float values
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return False

        # Skip zero and one (commonly used and might cause issues)
        if value == 0 or value == 1:
            return False

        # Skip very small numbers
        if abs(value) < self.min_value:
            return False

        # Skip very large numbers
        if abs(value) > self.max_value:
            return False

        return True

    def _generate_obfuscated_expression(self, target_value: Union[int, float]) -> ast.AST:
        """Generate an obfuscated arithmetic expression for a target value.

        Args:
            target_value: The value that the expression should evaluate to.

        Returns:
            AST node representing the obfuscated expression.
        """
        # For complexity level 1: simple operations (addition/subtraction)
        if self.complexity == 1:
            return self._generate_simple_expression(target_value)

        # For complexity level 2: include multiplication/division
        elif self.complexity == 2:
            return self._generate_mixed_expression(target_value)

        # For complexity level 3: include bitwise operations
        elif self.complexity == 3:
            return self._generate_bitwise_expression(target_value)

        # For complexity level 4: nested expressions
        elif self.complexity == 4:
            return self._generate_nested_expression(target_value)

        # For complexity level 5: advanced obfuscation
        else:
            return self._generate_advanced_expression(target_value)

    def _generate_simple_expression(self, target_value: Union[int, float]) -> ast.AST:
        """Generate a simple addition/subtraction expression.

        Args:
            target_value: The target value.

        Returns:
            AST node for (a + b) or (a - b) expression.
        """
        # Add pattern tracking for variety if not already present
        if not hasattr(self, '_pattern_history'):
            self._pattern_history = []
        
        # Choose from three patterns: addition, subtraction, or hex literal
        # For floats, only use addition/subtraction (no hex)
        if isinstance(target_value, float):
            available_patterns = ['add', 'sub']
        else:
            available_patterns = ['add', 'sub', 'hex']
        
        # Filter out recently used patterns to ensure variety
        if len(self._pattern_history) >= 2:
            recent_patterns = set(self._pattern_history[-2:])
            filtered_patterns = [p for p in available_patterns if p not in recent_patterns]
            # Only use filtered patterns if we have at least one option
            if filtered_patterns:
                available_patterns = filtered_patterns
        
        pattern = random.choice(available_patterns)
        self._pattern_history.append(pattern)
        
        # Keep only last 5 patterns in history
        if len(self._pattern_history) > 5:
            self._pattern_history.pop(0)
        
        # Handle floats differently using random.uniform
        if isinstance(target_value, float):
            # For floats, use random.uniform to generate offset
            offset = random.uniform(0.1, min(100.0, target_value / 2.0 if target_value > 2.0 else 1.0))
            
            if pattern == 'add':
                # (target - offset) + offset = target
                part1 = target_value - offset
                part2 = offset
                expr = ast.BinOp(
                    left=ast.Constant(value=part1),
                    op=ast.Add(),
                    right=ast.Constant(value=part2)
                )
            else:  # subtraction
                # (target + offset) - offset = target
                part1 = target_value + offset
                part2 = offset
                expr = ast.BinOp(
                    left=ast.Constant(value=part1),
                    op=ast.Sub(),
                    right=ast.Constant(value=part2)
                )
            
            # Copy location information
            ast.copy_location(expr, ast.Constant(value=target_value))
            ast.copy_location(expr.left, ast.Constant(value=target_value))
            ast.copy_location(expr.right, ast.Constant(value=target_value))
            return expr
        else:
            # For integers
            if pattern == 'hex':
                # Use hex literal representation
                hex_expr = ast.Constant(value=target_value)
                # Mark as hex for potential special formatting if needed
                return hex_expr
            else:
                # Generate offset for add/sub operations
                offset = random.randint(1, min(100, target_value // 2 if target_value > 2 else 1))
                
                if pattern == 'add':
                    # (target - offset) + offset = target
                    part1 = target_value - offset
                    part2 = offset
                    expr = ast.BinOp(
                        left=ast.Constant(value=part1),
                        op=ast.Add(),
                        right=ast.Constant(value=part2)
                    )
                else:  # subtraction
                    # (target + offset) - offset = target
                    part1 = target_value + offset
                    part2 = offset
                    expr = ast.BinOp(
                        left=ast.Constant(value=part1),
                        op=ast.Sub(),
                        right=ast.Constant(value=part2)
                    )
                
                # Copy location information
                ast.copy_location(expr, ast.Constant(value=target_value))
                ast.copy_location(expr.left, ast.Constant(value=target_value))
                ast.copy_location(expr.right, ast.Constant(value=target_value))
                return expr

    def _generate_mixed_expression(self, target_value: Union[int, float]) -> ast.AST:
        """Generate a mixed operation expression (addition, subtraction, multiplication).

        Args:
            target_value: The target value.

        Returns:
            AST node for expression like (a * b + c) or (a + b * c).
        """
        # For integers, use multiplication and addition
        if isinstance(target_value, int) and target_value > 10:
            factor = random.randint(2, min(10, target_value // 2))
            if target_value % factor == 0:
                quotient = target_value // factor
                offset = random.randint(1, min(50, quotient // 2))

                # (factor * (quotient - offset) + factor * offset)
                # This equals: factor * quotient - factor * offset + factor * offset = factor * quotient = target_value
                return ast.BinOp(
                    left=ast.BinOp(
                        left=ast.Constant(value=factor),
                        op=ast.Mult(),
                        right=ast.Constant(value=quotient - offset)
                    ),
                    op=ast.Add(),
                    right=ast.BinOp(
                        left=ast.Constant(value=factor),
                        op=ast.Mult(),
                        right=ast.Constant(value=offset)
                    )
                )

        # Fallback to simple expression
        return self._generate_simple_expression(target_value)

    def _generate_bitwise_expression(self, target_value: Union[int, float]) -> ast.AST:
        """Generate an expression using bitwise operations.

        Args:
            target_value: The target value.

        Returns:
            AST node for expression using bitwise operations.
        """
        # Only apply to integers
        if not isinstance(target_value, int):
            return self._generate_mixed_expression(target_value)

        # Use XOR operation: (a ^ b) + c
        # Where: a ^ b = target_value - c
        offset = random.randint(1, min(100, target_value // 2 if target_value > 2 else 1))
        xor_target = target_value - offset

        # Find two numbers that XOR to xor_target
        a = random.randint(0, xor_target * 2)
        b = xor_target ^ a

        # ((a ^ b) + offset)
        # This equals: xor_target + offset = target_value
        return ast.BinOp(
            left=ast.BinOp(
                left=ast.Constant(value=a),
                op=ast.BitXor(),
                right=ast.Constant(value=b)
            ),
            op=ast.Add(),
            right=ast.Constant(value=offset)
        )

    def _generate_nested_expression(self, target_value: Union[int, float]) -> ast.AST:
        """Generate a nested expression with multiple operations.

        Args:
            target_value: The target value.

        Returns:
            AST node for nested expression like ((a + b) * c + d).
        """
        if isinstance(target_value, int) and target_value > 50:
            # Split into multiple parts
            part1 = random.randint(10, target_value // 3)
            remaining1 = target_value - part1

            part2 = random.randint(5, remaining1 // 2)
            remaining2 = remaining1 - part2

            part3 = remaining2

            # ((part1 + part2) + part3)
            return ast.BinOp(
                left=ast.BinOp(
                    left=ast.Constant(value=part1),
                    op=ast.Add(),
                    right=ast.Constant(value=part2)
                ),
                op=ast.Add(),
                right=ast.Constant(value=part3)
            )

        # Fallback to bitwise expression
        return self._generate_bitwise_expression(target_value)

    def _generate_advanced_expression(self, target_value: Union[int, float]) -> ast.AST:
        """Generate an advanced obfuscated expression.

        Args:
            target_value: The target value.

        Returns:
            AST node for complex expression using multiple operation types.
        """
        if isinstance(target_value, int) and target_value > 100:
            # Use a combination of operations
            # For example: factor * (a ^ b) + c
            factor = random.randint(2, 10)
            if target_value % factor == 0:
                quotient = target_value // factor

                # Find XOR components
                xor_offset = random.randint(1, min(50, quotient // 2))
                xor_target = quotient - xor_offset

                a = random.randint(0, xor_target * 2)
                b = xor_target ^ a

                # (factor * (a ^ b)) + xor_offset
                # This equals: factor * xor_target + xor_offset = factor * quotient = target_value
                return ast.BinOp(
                    left=ast.BinOp(
                        left=ast.Constant(value=factor),
                        op=ast.Mult(),
                        right=ast.BinOp(
                            left=ast.Constant(value=a),
                            op=ast.BitXor(),
                            right=ast.Constant(value=b)
                        )
                    ),
                    op=ast.Add(),
                    right=ast.Constant(value=xor_offset)
                )

        # Fallback to nested expression
        return self._generate_nested_expression(target_value)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """Visit and potentially transform numeric constant nodes.

        If the node is a numeric constant that meets obfuscation criteria,
        replaces it with an equivalent arithmetic expression.

        Args:
            node: The Constant AST node to potentially transform.

        Returns:
            Either a new expression node for obfuscation, or the original node unchanged.
        """
        # Only transform numeric constants
        if not isinstance(node.value, (int, float)):
            return node

        value = node.value

        # Check if we should obfuscate this number
        if not self._should_obfuscate_number(value):
            return node

        try:
            # For negative numbers, generate obfuscated expression for abs(value)
            # and wrap it with unary negation
            if value < 0:
                abs_value = abs(value)
                obfuscated_expr = self._generate_obfuscated_expression(abs_value)
                # Wrap with unary negation
                negated_expr = ast.UnaryOp(op=ast.USub(), operand=obfuscated_expr)
                ast.copy_location(negated_expr, node)
                if isinstance(obfuscated_expr, ast.BinOp):
                    ast.copy_location(obfuscated_expr.left, node)
                    ast.copy_location(obfuscated_expr.right, node)
                # Track transformation
                self.transformation_count += 1
                self.logger.debug(
                    f"Obfuscated negative number at line {getattr(node, 'lineno', '?')}: "
                    f"{value} → -({abs_value} obfuscated)"
                )
                return negated_expr
            else:
                # Generate obfuscated expression for positive numbers
                obfuscated_expr = self._generate_obfuscated_expression(value)

                # Copy location information
                ast.copy_location(obfuscated_expr, node)

                # Also copy location to child nodes if they exist
                if isinstance(obfuscated_expr, ast.BinOp):
                    ast.copy_location(obfuscated_expr.left, node)
                    ast.copy_location(obfuscated_expr.right, node)

                # Track transformation
                self.transformation_count += 1
                self.logger.debug(
                    f"Obfuscated number at line {getattr(node, 'lineno', '?')}: "
                    f"{value} → complex expression"
                )

                return obfuscated_expr

        except Exception as e:
            error_msg = (
                f"Failed to obfuscate number at line {getattr(node, 'lineno', '?')}: {e}"
            )
            self.errors.append(error_msg)
            self.logger.warning(error_msg)
            return node

    def _generate_lua_obfuscated_expression(self, target_value: Union[int, float]) -> str:
        """Generate a Lua obfuscated expression for a target value.

        Args:
            target_value: The value that the expression should evaluate to.

        Returns:
            Lua code string representing the obfuscated expression.
        """
        # For complexity level 1: simple addition/subtraction
        if self.complexity == 1:
            return self._generate_lua_simple_expression(target_value)

        # For complexity level 2: include multiplication/division
        elif self.complexity == 2:
            return self._generate_lua_mixed_expression(target_value)

        # For complexity level 3: include bitwise operations
        elif self.complexity == 3:
            return self._generate_lua_bitwise_expression(target_value)

        # For complexity level 4: nested expressions
        elif self.complexity == 4:
            return self._generate_lua_nested_expression(target_value)

        # For complexity level 5: advanced obfuscation
        else:
            return self._generate_lua_advanced_expression(target_value)

    def _generate_lua_simple_expression(self, target_value: Union[int, float]) -> str:
        """Generate a simple Lua addition/subtraction expression.

        Args:
            target_value: The target value.

        Returns:
            Lua code string for (a + b) or (a - b) expression.
        """
        # Add pattern tracking for variety if not already present
        if not hasattr(self, '_lua_pattern_history'):
            self._lua_pattern_history = []
        
        # Choose from three patterns: addition, subtraction, or hex literal
        # For floats, only use addition/subtraction (no hex)
        if isinstance(target_value, float):
            available_patterns = ['add', 'sub']
        else:
            available_patterns = ['add', 'sub', 'hex']
        
        # Filter out recently used patterns to ensure variety
        if len(self._lua_pattern_history) >= 2:
            recent_patterns = set(self._lua_pattern_history[-2:])
            filtered_patterns = [p for p in available_patterns if p not in recent_patterns]
            # Only use filtered patterns if we have at least one option
            if filtered_patterns:
                available_patterns = filtered_patterns
        
        pattern = random.choice(available_patterns)
        self._lua_pattern_history.append(pattern)
        
        # Keep only last 5 patterns in history
        if len(self._lua_pattern_history) > 5:
            self._lua_pattern_history.pop(0)
        
        # Handle floats differently using random.uniform
        if isinstance(target_value, float):
            # For floats, use random.uniform to generate offset
            offset = random.uniform(0.1, min(100.0, target_value / 2.0 if target_value > 2.0 else 1.0))
            
            if pattern == 'add':
                # (target - offset) + offset = target
                part1 = target_value - offset
                part2 = offset
                return f"({part1} + {part2})"
            else:  # subtraction
                # (target + offset) - offset = target
                part1 = target_value + offset
                part2 = offset
                return f"({part1} - {part2})"
        else:
            # For integers
            if pattern == 'hex':
                # Use hex literal representation
                return f"0x{target_value:x}"
            else:
                # Generate offset for add/sub operations
                offset = random.randint(1, min(100, int(target_value // 2) if target_value > 2 else 1))
                
                if pattern == 'add':
                    # (target - offset) + offset = target
                    part1 = target_value - offset
                    part2 = offset
                    return f"({part1} + {part2})"
                else:  # subtraction
                    # (target + offset) - offset = target
                    part1 = target_value + offset
                    part2 = offset
                    return f"({part1} - {part2})"

    def _generate_lua_mixed_expression(self, target_value: Union[int, float]) -> str:
        """Generate a Lua mixed operation expression.

        Args:
            target_value: The target value.

        Returns:
            Lua code string for expression like (a * b + c).
        """
        if isinstance(target_value, int) and target_value > 10:
            factor = random.randint(2, min(10, target_value // 2))
            if target_value % factor == 0:
                quotient = target_value // factor
                offset = random.randint(1, min(50, quotient // 2))
                # (factor * (quotient - offset) + factor * offset)
                # This equals: factor * quotient - factor * offset + factor * offset = factor * quotient = target_value
                return f"({factor} * ({quotient - offset}) + {factor} * {offset})"

        return self._generate_lua_simple_expression(target_value)

    def _generate_lua_bitwise_expression(self, target_value: Union[int, float]) -> str:
        """Generate a Lua expression using bitwise operations.

        Args:
            target_value: The target value.

        Returns:
            Lua code string for expression using bitwise operations.
        """
        if not isinstance(target_value, int):
            return self._generate_lua_mixed_expression(target_value)

        offset = random.randint(1, min(100, target_value // 2 if target_value > 2 else 1))
        xor_target = target_value - offset

        a = random.randint(0, xor_target * 2)
        b = xor_target ^ a

        return f"(({a} ~ {b}) + {offset})"

    def _generate_lua_nested_expression(self, target_value: Union[int, float]) -> str:
        """Generate a Lua nested expression with multiple operations.

        Args:
            target_value: The target value.

        Returns:
            Lua code string for nested expression.
        """
        if isinstance(target_value, int) and target_value > 50:
            part1 = random.randint(10, target_value // 3)
            remaining1 = target_value - part1

            part2 = random.randint(5, remaining1 // 2)
            part3 = remaining1 - part2

            return f"(({part1} + {part2}) + {part3})"

        return self._generate_lua_bitwise_expression(target_value)

    def _generate_lua_advanced_expression(self, target_value: Union[int, float]) -> str:
        """Generate an advanced Lua obfuscated expression.

        Args:
            target_value: The target value.

        Returns:
            Lua code string for complex expression.
        """
        if isinstance(target_value, int) and target_value > 100:
            factor = random.randint(2, 10)
            if target_value % factor == 0:
                quotient = target_value // factor
                xor_offset = random.randint(1, min(50, quotient // 2))
                xor_target = quotient - xor_offset

                a = random.randint(0, xor_target * 2)
                b = xor_target ^ a

                # (factor * (a ~ b)) + xor_offset
                # This equals: factor * xor_target + xor_offset = factor * quotient = target_value
                return f"({factor} * ({a} ~ {b}) + {xor_offset})"

        return self._generate_lua_nested_expression(target_value)

    def _replace_lua_number(self, node: Any) -> Any:
        """Replace a Lua Number node with an obfuscated expression.

        Args:
            node: The lua_nodes.Number node to transform.

        Returns:
            A lua_nodes expression node, or the original node if transformation fails.
        """
        if not LUAPARSER_AVAILABLE:
            return node

        # Get numeric value
        value = node.n

        # Check if we should obfuscate this number
        if not self._should_obfuscate_number(value):
            return node

        try:
            # For negative numbers, generate obfuscated expression for abs(value)
            # and wrap it with unary negation prefix
            if value < 0:
                abs_value = abs(value)
                lua_expr = self._generate_lua_obfuscated_expression(abs_value)
                # Wrap with unary negation prefix
                negated_expr = f"-({lua_expr})"
            else:
                # Generate Lua obfuscated expression string
                lua_expr = self._generate_lua_obfuscated_expression(value)
                negated_expr = lua_expr

            # Parse expression into AST nodes
            # Wrap in a simple statement for parsing
            wrapped_code = f"local _ = {negated_expr}"
            parsed = lua_ast.parse(wrapped_code)

            # Extract expression from parsed AST
            if (hasattr(parsed, 'body') and
                    hasattr(parsed.body, 'body') and
                    len(parsed.body.body) > 0):

                stmt = parsed.body.body[0]
                if hasattr(stmt, 'values') and len(stmt.values) > 0:
                    # Return the expression node
                    return stmt.values[0]

            # If parsing fails, return original
            return node

        except Exception as e:
            error_msg = (
                f"Failed to obfuscate Lua number at line {getattr(node, 'line', '?')}: {e}"
            )
            self.errors.append(error_msg)
            self.logger.warning(error_msg)
            return node
            # For negative numbers, generate obfuscated expression for abs(value)
            # and wrap it with unary negation prefix
            if value < 0:
                abs_value = abs(value)
                lua_expr = self._generate_lua_obfuscated_expression(abs_value)
                # Wrap with unary negation prefix
                negated_expr = f"-({lua_expr})"
            else:
                # Generate Lua obfuscated expression string
                lua_expr = self._generate_lua_obfuscated_expression(value)
                negated_expr = lua_expr
            lua_expr = self._generate_lua_obfuscated_expression(value)

            # Parse the expression into AST nodes
            # Wrap in a simple statement for parsing
            wrapped_code = f"local _ = {lua_expr}"
            parsed = lua_ast.parse(wrapped_code)

            # Extract the expression from the parsed AST
            if (hasattr(parsed, 'body') and
                hasattr(parsed.body, 'body') and
                len(parsed.body.body) > 0):

                stmt = parsed.body.body[0]
                if hasattr(stmt, 'values') and len(stmt.values) > 0:
                    # Return the expression node
                    return stmt.values[0]

            # If parsing fails, return original
            return node

        except Exception as e:
            error_msg = (
                f"Failed to obfuscate Lua number at line {getattr(node, 'line', '?')}: {e}"
            )
            self.errors.append(error_msg)
            self.logger.warning(error_msg)
            return node

    def _traverse_lua_ast(self, node: Any) -> None:
        """Recursively traverse and transform Lua AST, replacing Number nodes.

        This method walks the Lua AST and replaces lua_nodes.Number
        nodes with obfuscated expressions in-place.

        Args:
            node: The current Lua AST node to process.
        """
        if not LUAPARSER_AVAILABLE or node is None:
            return

        # Get all attribute names that might contain child nodes
        for attr_name in dir(node):
            if attr_name.startswith('_'):
                continue

            attr = getattr(node, attr_name, None)
            if attr is None:
                continue

            # Handle lists of nodes
            if isinstance(attr, list):
                for i, item in enumerate(attr):
                    if isinstance(item, lua_nodes.Number):
                        # Replace Number node with obfuscated expression
                        attr[i] = self._replace_lua_number(item)
                    elif hasattr(item, '__class__') and hasattr(item.__class__, '__module__'):
                        if 'astnodes' in item.__class__.__module__:
                            self._traverse_lua_ast(item)

            # Handle single nodes
            elif isinstance(attr, lua_nodes.Number):
                # Replace Number node with obfuscated expression
                setattr(node, attr_name, self._replace_lua_number(attr))
            elif hasattr(attr, '__class__') and hasattr(attr.__class__, '__module__'):
                if 'astnodes' in attr.__class__.__module__:
                    self._traverse_lua_ast(attr)

    def transform(self, ast_node: ast.AST) -> TransformResult:
        """Apply number obfuscation transformation to an AST.

        This method orchestrates the transformation:
        1. Detects language from AST node type
        2. Applies number obfuscation transformations
        3. Returns structured result

        Args:
            ast_node: The AST node to transform (ast.Module for Python).

        Returns:
            TransformResult with transformed AST, success status, and metrics.
        """
        # Reset state for this transformation
        self.transformation_count = 0
        self.errors = []

        try:
            # Detect language and apply transformations
            if isinstance(ast_node, ast.Module):
                self.language_mode = 'python'

                # Fix missing locations before transformation
                ast.fix_missing_locations(ast_node)

                # Apply Python transformations by visiting the AST
                transformed_node = self.visit(ast_node)

                if transformed_node is None:
                    raise ValueError("Transformation returned None")

                # Fix missing locations on newly created nodes
                ast.fix_missing_locations(transformed_node)

            elif LUAPARSER_AVAILABLE and isinstance(ast_node, lua_nodes.Chunk):
                self.language_mode = 'lua'

                # Apply Lua transformations using custom traversal
                self._traverse_lua_ast(ast_node)

                # Lua transformation is in-place
                transformed_node = ast_node

            else:
                # For other node types, try to proceed without transformation
                self.logger.warning(
                    f"Unknown AST type {type(ast_node).__name__}, "
                    "proceeding without number obfuscation"
                )
                transformed_node = ast_node

            self.logger.debug(
                f"Number obfuscation completed: {self.transformation_count} numbers obfuscated"
            )

            return TransformResult(
                ast_node=transformed_node,
                success=True,
                transformation_count=self.transformation_count,
                errors=self.errors,
            )

        except Exception as e:
            error_msg = f"Number obfuscation failed: {e.__class__.__name__}: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.errors.append(error_msg)

            return TransformResult(
                ast_node=None,
                success=False,
                transformation_count=self.transformation_count,
                errors=self.errors,
            )


class StringEncryptionTransformer(ASTTransformer):
    """Performs string encryption on AST string literals.

    This transformer encrypts string literals in both Python and Lua code to make
    them harder to read in the obfuscated output. It supports:
    - Python: AES-256-GCM encryption using the cryptography library
    - Lua: XOR-based encryption for Lua runtime compatibility

    The transformer injects a decryption runtime function at the module/chunk level
    and replaces string literals with calls to this decryption function.

    For Python, the transformer:
    - Skips docstrings (first string in module, function, or class body)
    - Skips f-string literal segments (JoinedStr children)

    For Lua, the transformer:
    - Traverses luaparser.astnodes.String nodes
    - Replaces them with Call nodes to _decrypt_string

    Attributes:
        config: ObfuscationConfig with string_encryption_key_length option
        key_length: Length of the encryption key in bytes
        encryption_key: Generated encryption key bytes
        iv: Initialization vector for encryption
        language_mode: Detected language ('python', 'lua', or None)
        runtime_injected: Whether decryption runtime has been injected
        min_string_length: Minimum string length to encrypt (default: 3)
        string_cache: Cache for encrypted strings to avoid duplicate encryption
        _in_joined_str: Tracks if currently visiting inside a JoinedStr (f-string)

    Example:
        >>> from obfuscator.core.config import ObfuscationConfig
        >>> config = ObfuscationConfig(name="test", features={"string_encryption": True})
        >>> transformer = StringEncryptionTransformer(config)
        >>> tree = ast.parse('message = "Hello, World!"')
        >>> result = transformer.transform(tree)
        >>> if result.success:
        ...     # The string "Hello, World!" is now encrypted
        ...     print(f"Encrypted {result.transformation_count} strings")
    """

    # Supported key lengths for AES encryption (in bytes)
    VALID_KEY_LENGTHS = {8, 16, 24, 32}
    # Default minimum string length to encrypt
    DEFAULT_MIN_STRING_LENGTH = 3

    def __init__(
        self,
        config: Optional[Any] = None,
        key_length: Optional[int] = None,
        min_string_length: int = DEFAULT_MIN_STRING_LENGTH,
    ) -> None:
        """Initialize the string encryption transformer.

        Args:
            config: ObfuscationConfig instance with encryption options.
                   If provided, key_length is extracted from config.options.
            key_length: Override for encryption key length in bytes.
                       If not provided, uses config or defaults to 16.
            min_string_length: Minimum string length to encrypt (default: 3).
                              Strings shorter than this are left unencrypted.

        Raises:
            ValueError: If key_length is not a valid AES key length.
        """
        super().__init__()
        self.logger = get_logger("obfuscator.processors.string_encryption")

        # Determine key length from config or parameter
        if key_length is not None:
            self._key_length = key_length
        elif config is not None and hasattr(config, 'options'):
            self._key_length = config.options.get('string_encryption_key_length', 16)
        else:
            self._key_length = 16

        # Validate key length - adjust to nearest valid AES key length
        if self._key_length not in self.VALID_KEY_LENGTHS:
            # For AES, key must be 16, 24, or 32 bytes; for XOR, any length works
            # We'll use 16 as minimum for security
            if self._key_length < 16:
                self.logger.warning(
                    f"Key length {self._key_length} is too small. Using 16 bytes."
                )
                self._key_length = 16
            elif self._key_length < 24:
                self._key_length = 16
            elif self._key_length < 32:
                self._key_length = 24
            else:
                self._key_length = 32

        self.config = config
        self.min_string_length = min_string_length

        # Generate encryption key and IV
        self.encryption_key: bytes = secrets.token_bytes(self._key_length)
        self.iv: bytes = secrets.token_bytes(12)  # 12 bytes for AES-GCM nonce

        # Language detection and state tracking
        self.language_mode: Optional[str] = None
        self.runtime_injected: bool = False

        # Cache for encrypted strings to avoid duplicate encryption
        self.string_cache: Dict[str, ast.AST] = {}

        # Track if we're inside a JoinedStr (f-string) to skip literal segments
        self._in_joined_str: bool = False

        self.logger.debug(
            f"StringEncryptionTransformer initialized with key_length={self._key_length}"
        )

    @property
    def key_length(self) -> int:
        """Get the encryption key length in bytes."""
        return self._key_length

    def _encrypt_string_aes(self, plaintext: str) -> bytes:
        """Encrypt a string using AES-256-GCM.

        Args:
            plaintext: The string to encrypt.

        Returns:
            Encrypted bytes (nonce + ciphertext + tag).

        Raises:
            RuntimeError: If cryptography library is not available.
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError(
                "cryptography library is required for AES encryption. "
                "Install it with: pip install cryptography"
            )

        # Create AESGCM instance with the key
        # Note: AESGCM requires 16, 24, or 32 byte keys
        aesgcm = AESGCM(self.encryption_key)

        # Encrypt with the IV (nonce)
        plaintext_bytes = plaintext.encode('utf-8')
        ciphertext = aesgcm.encrypt(self.iv, plaintext_bytes, None)

        return ciphertext

    def _encrypt_string_xor(self, plaintext: str) -> bytes:
        """Encrypt a string using XOR cipher (for Lua compatibility).

        Args:
            plaintext: The string to encrypt.

        Returns:
            XOR-encrypted bytes.
        """
        plaintext_bytes = plaintext.encode('utf-8')
        key_bytes = self.encryption_key
        key_len = len(key_bytes)

        encrypted = bytes(
            plaintext_bytes[i] ^ key_bytes[i % key_len]
            for i in range(len(plaintext_bytes))
        )

        return encrypted

    def _generate_python_decryption_runtime(self) -> list[ast.stmt]:
        """Generate Python decryption runtime code as AST nodes.

        Returns:
            List of AST statement nodes for the decryption function and constants.
        """
        # Create the decryption function as a string and parse it
        key_b64 = base64.b64encode(self.encryption_key).decode('ascii')
        iv_b64 = base64.b64encode(self.iv).decode('ascii')

        runtime_code = f'''
import base64 as _b64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _AESGCM

_decrypt_key = _b64.b64decode("{key_b64}")
_decrypt_iv = _b64.b64decode("{iv_b64}")
_decrypt_aesgcm = _AESGCM(_decrypt_key)

def _decrypt_string(encrypted_b64: str) -> str:
    encrypted_data = _b64.b64decode(encrypted_b64)
    decrypted = _decrypt_aesgcm.decrypt(_decrypt_iv, encrypted_data, None)
    return decrypted.decode("utf-8")
'''
        runtime_tree = ast.parse(runtime_code)
        return runtime_tree.body

    def _generate_lua_decryption_runtime(self) -> str:
        """Generate Lua decryption runtime code as a string.

        Returns:
            Lua code string for the decryption function.
        """
        # Escape key bytes for Lua string literal
        key_escaped = ''.join(f'\\{b}' for b in self.encryption_key)

        runtime_code = f'''
local _decrypt_key = "{key_escaped}"
local function _decrypt_string(encrypted_data)
    local result = {{}}
    local key_len = #_decrypt_key
    for i = 1, #encrypted_data do
        local encrypted_byte = string.byte(encrypted_data, i)
        local key_byte = string.byte(_decrypt_key, ((i - 1) % key_len) + 1)
        result[i] = string.char(encrypted_byte ~ key_byte)
    end
    return table.concat(result)
end
'''
        return runtime_code.strip()

    def _inject_python_runtime(self, module: ast.Module) -> None:
        """Inject Python decryption runtime at the beginning of a module.

        Args:
            module: The Python AST Module node to inject into.
        """
        if self.runtime_injected:
            return

        runtime_nodes = self._generate_python_decryption_runtime()

        # Insert runtime at the beginning of the module body
        # But preserve __future__ imports and docstrings
        insert_index = 0

        for i, stmt in enumerate(module.body):
            # Skip __future__ imports
            if isinstance(stmt, ast.ImportFrom) and stmt.module == '__future__':
                insert_index = i + 1
            # Skip module docstring (first string expression)
            elif i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                insert_index = 1
            else:
                break

        # Insert runtime nodes at the calculated position
        for j, node in enumerate(runtime_nodes):
            ast.fix_missing_locations(node)
            module.body.insert(insert_index + j, node)

        self.runtime_injected = True
        self.logger.debug("Injected Python decryption runtime")

    def _inject_lua_runtime(self, chunk: Any) -> None:
        """Inject Lua decryption runtime at the beginning of a chunk.

        Args:
            chunk: The Lua AST Chunk node to inject into.
        """
        if not LUAPARSER_AVAILABLE:
            self.logger.warning("luaparser not available, skipping Lua runtime injection")
            return

        if self.runtime_injected:
            return

        # Parse the Lua runtime code
        runtime_code = self._generate_lua_decryption_runtime()
        try:
            runtime_tree = lua_ast.parse(runtime_code)
            if hasattr(runtime_tree, 'body') and hasattr(chunk, 'body'):
                # Insert runtime statements at the beginning
                if hasattr(chunk.body, 'body'):
                    # chunk.body is a Block with a body attribute
                    chunk.body.body = runtime_tree.body.body + chunk.body.body
                else:
                    chunk.body = runtime_tree.body.body + chunk.body
            self.runtime_injected = True
            self.logger.debug("Injected Lua decryption runtime")
        except Exception as e:
            error_msg = f"Failed to inject Lua runtime: {e}"
            self.errors.append(error_msg)
            self.logger.error(error_msg)

    def _should_skip_string(self, value: str, node: ast.AST) -> bool:
        """Determine if a string should be skipped from encryption.

        Args:
            value: The string value to check.
            node: The AST node containing the string.

        Returns:
            True if the string should be skipped, False otherwise.
        """
        # Skip empty strings
        if not value:
            return True

        # Skip strings shorter than minimum length
        if len(value) < self.min_string_length:
            return True

        # Skip docstrings (already handled at module level in _inject_python_runtime)
        # Additional check here for function/class docstrings would require context

        return False

    def _create_decrypt_call_node(self, encrypted_b64: str, original_node: ast.AST) -> ast.Call:
        """Create an AST Call node for the decryption function.

        Args:
            encrypted_b64: Base64-encoded encrypted string.
            original_node: Original AST node for location copying.

        Returns:
            AST Call node that calls _decrypt_string with the encrypted data.
        """
        call_node = ast.Call(
            func=ast.Name(id='_decrypt_string', ctx=ast.Load()),
            args=[ast.Constant(value=encrypted_b64)],
            keywords=[],
        )
        ast.copy_location(call_node, original_node)
        ast.copy_location(call_node.func, original_node)
        ast.copy_location(call_node.args[0], original_node)
        return call_node

    def _is_docstring(self, node: ast.AST, body: list[ast.stmt]) -> bool:
        """Check if an Expr node is a docstring (first string expression in body).

        Args:
            node: The AST node to check.
            body: The body list containing the node.

        Returns:
            True if node is a docstring, False otherwise.
        """
        if not body:
            return False
        first_stmt = body[0]
        if (
            first_stmt is node
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return True
        return False

    def _visit_body_skip_docstring(self, body: list[ast.stmt]) -> None:
        """Visit all statements in a body, skipping the first if it's a docstring.

        Args:
            body: List of AST statements to visit.
        """
        for i, stmt in enumerate(body):
            if i == 0 and self._is_docstring(stmt, body):
                # Skip the docstring - don't transform it
                continue
            # Visit and potentially replace statement
            new_stmt = self.visit(stmt)
            if new_stmt is not None:
                body[i] = new_stmt

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Visit Module nodes, skipping the module-level docstring.

        Args:
            node: The Module AST node.

        Returns:
            The transformed Module node.
        """
        self._visit_body_skip_docstring(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Visit FunctionDef nodes, skipping the function docstring.

        Args:
            node: The FunctionDef AST node.

        Returns:
            The transformed FunctionDef node.
        """
        # Visit decorators (they can have strings too)
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        # Visit arguments default values
        node.args.defaults = [self.visit(d) for d in node.args.defaults]
        node.args.kw_defaults = [
            self.visit(d) if d is not None else None for d in node.args.kw_defaults
        ]
        # Visit body, skipping docstring
        self._visit_body_skip_docstring(node.body)
        # Visit returns annotation if present
        if node.returns:
            node.returns = self.visit(node.returns)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Visit AsyncFunctionDef nodes, skipping the function docstring.

        Args:
            node: The AsyncFunctionDef AST node.

        Returns:
            The transformed AsyncFunctionDef node.
        """
        # Visit decorators
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        # Visit arguments default values
        node.args.defaults = [self.visit(d) for d in node.args.defaults]
        node.args.kw_defaults = [
            self.visit(d) if d is not None else None for d in node.args.kw_defaults
        ]
        # Visit body, skipping docstring
        self._visit_body_skip_docstring(node.body)
        # Visit returns annotation if present
        if node.returns:
            node.returns = self.visit(node.returns)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Visit ClassDef nodes, skipping the class docstring.

        Args:
            node: The ClassDef AST node.

        Returns:
            The transformed ClassDef node.
        """
        # Visit decorators
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        # Visit bases and keywords
        node.bases = [self.visit(b) for b in node.bases]
        node.keywords = [self.visit(k) for k in node.keywords]
        # Visit body, skipping docstring
        self._visit_body_skip_docstring(node.body)
        return node

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.JoinedStr:
        """Visit JoinedStr (f-string) nodes without transforming literal segments.

        F-strings contain a mix of Constant (literal) and FormattedValue nodes.
        We only visit FormattedValue children to avoid breaking the f-string
        structure by encrypting literal segments.

        Args:
            node: The JoinedStr AST node (f-string).

        Returns:
            The JoinedStr node with only FormattedValue children visited.
        """
        # Only visit FormattedValue children, not Constant string parts
        new_values = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                # Visit the formatted value's expression
                new_values.append(self.visit(value))
            else:
                # Keep Constant string parts unchanged
                new_values.append(value)
        node.values = new_values
        return node

    def visit_FormattedValue(self, node: ast.FormattedValue) -> ast.FormattedValue:
        """Visit FormattedValue nodes inside f-strings.

        Args:
            node: The FormattedValue AST node.

        Returns:
            The transformed FormattedValue node.
        """
        # Visit the expression inside the formatted value
        node.value = self.visit(node.value)
        # Visit format spec if present (it's a JoinedStr)
        if node.format_spec:
            node.format_spec = self.visit(node.format_spec)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """Visit and potentially transform Python string constant nodes.

        If the node is a string constant and meets encryption criteria,
        encrypts it and replaces with a decryption function call.

        Args:
            node: The Constant AST node to potentially transform.

        Returns:
            Either a new Call node for decryption, or the original node unchanged.
        """
        # Only transform string constants
        if not isinstance(node.value, str):
            return node

        value = node.value

        # Check if we should skip this string
        if self._should_skip_string(value, node):
            return node

        # Check cache for already encrypted strings
        if value in self.string_cache:
            cached_node = self.string_cache[value]
            new_node = ast.copy_location(
                ast.Call(
                    func=ast.Name(id='_decrypt_string', ctx=ast.Load()),
                    args=[ast.Constant(value=cached_node.args[0].value)],  # type: ignore
                    keywords=[],
                ),
                node
            )
            self.transformation_count += 1
            return new_node

        try:
            # Encrypt the string using AES-GCM
            encrypted_bytes = self._encrypt_string_aes(value)
            encrypted_b64 = base64.b64encode(encrypted_bytes).decode('ascii')

            # Create the decryption call node
            call_node = self._create_decrypt_call_node(encrypted_b64, node)

            # Cache the result
            self.string_cache[value] = call_node

            # Track transformation
            self.transformation_count += 1
            self.logger.debug(
                f"Encrypted string at line {getattr(node, 'lineno', '?')}: "
                f"'{value[:20]}{'...' if len(value) > 20 else ''}'"
            )

            return call_node

        except Exception as e:
            error_msg = (
                f"Failed to encrypt string at line {getattr(node, 'lineno', '?')}: {e}"
            )
            self.errors.append(error_msg)
            self.logger.warning(error_msg)
            return node

    def _escape_bytes_for_lua(self, data: bytes) -> str:
        """Escape bytes for use in a Lua string literal.

        Converts each byte to a Lua octal escape sequence (\ddd).

        Args:
            data: The bytes to escape.

        Returns:
            Escaped string suitable for Lua string literal.
        """
        return ''.join(f'\\{b:03d}' for b in data)

    def _transform_lua_string_node(
        self, node: Any
    ) -> Any:
        """Transform a Lua String node into a decryption call.

        Args:
            node: The luaparser.astnodes.String node to transform.

        Returns:
            A luaparser.astnodes.Call node calling _decrypt_string, or the
            original node if transformation should be skipped.
        """
        if not LUAPARSER_AVAILABLE:
            return node

        # Get the string value (luaparser stores it as bytes in node.s)
        value_bytes = node.s
        if isinstance(value_bytes, bytes):
            value = value_bytes.decode('utf-8', errors='replace')
        else:
            value = str(value_bytes) if value_bytes else ""

        # Skip short strings
        if len(value) < self.min_string_length:
            return node

        try:
            # Encrypt using XOR for Lua compatibility
            encrypted_bytes = self._encrypt_string_xor(value)

            # Escape for Lua string literal
            escaped = self._escape_bytes_for_lua(encrypted_bytes)
            raw = f'"{escaped}"'

            # Create the Call node: _decrypt_string("encrypted_data")
            func_name = lua_nodes.Name('_decrypt_string')
            string_arg = lua_nodes.String(
                encrypted_bytes, raw, lua_nodes.StringDelimiter.DOUBLE_QUOTE
            )
            call_node = lua_nodes.Call(func_name, [string_arg])

            self.transformation_count += 1
            self.logger.debug(
                f"Encrypted Lua string at line {getattr(node, 'line', '?')}: "
                f"'{value[:20]}{'...' if len(value) > 20 else ''}'"
            )

            return call_node

        except Exception as e:
            error_msg = (
                f"Failed to encrypt Lua string at line {getattr(node, 'line', '?')}: {e}"
            )
            self.errors.append(error_msg)
            self.logger.warning(error_msg)
            return node

    def _traverse_lua_ast(self, node: Any) -> None:
        """Recursively traverse and transform Lua AST, replacing String nodes.

        This method walks the Lua AST and replaces luaparser.astnodes.String
        nodes with Call nodes to _decrypt_string in-place.

        Args:
            node: The current Lua AST node to process.
        """
        if not LUAPARSER_AVAILABLE or node is None:
            return

        # Get all attribute names that might contain child nodes
        for attr_name in dir(node):
            if attr_name.startswith('_'):
                continue

            attr = getattr(node, attr_name, None)
            if attr is None:
                continue

            # Handle lists of nodes
            if isinstance(attr, list):
                for i, item in enumerate(attr):
                    if isinstance(item, lua_nodes.String):
                        # Replace String node with Call node
                        attr[i] = self._transform_lua_string_node(item)
                    elif hasattr(item, '__class__') and hasattr(item.__class__, '__module__'):
                        if 'astnodes' in item.__class__.__module__:
                            self._traverse_lua_ast(item)

            # Handle single nodes
            elif isinstance(attr, lua_nodes.String):
                # Replace String node with Call node
                setattr(node, attr_name, self._transform_lua_string_node(attr))
            elif hasattr(attr, '__class__') and hasattr(attr.__class__, '__module__'):
                if 'astnodes' in attr.__class__.__module__:
                    self._traverse_lua_ast(attr)

    def transform(self, ast_node: ast.AST) -> TransformResult:
        """Apply string encryption transformation to an AST.

        This method orchestrates the transformation:
        1. Detects language from AST node type
        2. Injects decryption runtime at module/chunk level
        3. Transforms string literals to decryption calls
        4. Returns structured result

        Args:
            ast_node: The AST node to transform (ast.Module for Python).

        Returns:
            TransformResult with transformed AST, success status, and metrics.
        """
        # Reset state for this transformation
        self.transformation_count = 0
        self.errors = []
        self.runtime_injected = False
        self.string_cache = {}

        try:
            # Detect language and apply transformations, then inject runtime
            if isinstance(ast_node, ast.Module):
                self.language_mode = 'python'

                # Fix missing locations before transformation
                ast.fix_missing_locations(ast_node)

                # Apply Python transformations by visiting the AST FIRST
                # (before runtime injection to avoid encrypting runtime strings)
                transformed_node = self.visit(ast_node)

                if transformed_node is None:
                    raise ValueError("Transformation returned None")

                # Inject runtime AFTER transformation so runtime strings are not encrypted
                self._inject_python_runtime(transformed_node)

                # Fix missing locations on newly created nodes
                ast.fix_missing_locations(transformed_node)

            elif LUAPARSER_AVAILABLE and isinstance(ast_node, lua_nodes.Chunk):
                self.language_mode = 'lua'

                # Apply Lua transformations using custom traversal FIRST
                # (before runtime injection to avoid encrypting runtime strings)
                self._traverse_lua_ast(ast_node)

                # Inject runtime AFTER transformation so runtime strings are not encrypted
                self._inject_lua_runtime(ast_node)

                # Lua transformation is in-place
                transformed_node = ast_node

            else:
                # For other node types, try to proceed without runtime injection
                self.logger.warning(
                    f"Unknown AST type {type(ast_node).__name__}, "
                    "proceeding without runtime injection"
                )
                transformed_node = ast_node

            self.logger.debug(
                f"String encryption completed: {self.transformation_count} strings encrypted"
            )

            return TransformResult(
                ast_node=transformed_node,
                success=True,
                transformation_count=self.transformation_count,
                errors=self.errors,
            )

        except Exception as e:
            error_msg = f"String encryption failed: {e.__class__.__name__}: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.errors.append(error_msg)

            return TransformResult(
                ast_node=None,
                success=False,
                transformation_count=self.transformation_count,
                errors=self.errors,
            )


class VMProtectionTransformer(ASTTransformer):
    """Performs VM-based code protection by converting functions to bytecode.

    This transformer identifies functions marked for VM protection and converts
    their AST to custom bytecode that is executed by a virtual machine. This
    provides strong obfuscation by hiding the original code structure.

    The transformer supports both Python and Lua code and includes:
    - Function identification via decorators or comments
    - Bytecode compilation from AST
    - VM runtime injection
    - Wrapper function generation

    Attributes:
        config: ObfuscationConfig with vm_protection options
        complexity: VM instruction set complexity level (1-3)
        protect_all_functions: Whether to protect all functions
        bytecode_encryption: Whether to encrypt bytecode
        protection_marker: String marker for function protection
        language_mode: Detected language ('python', 'lua', or None)
        runtime_injected: Whether VM runtime has been injected
        protected_functions: Set of function names that were protected
        bytecode_compiler: Compiler instance for current language
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        complexity: Optional[int] = None,
        protect_all_functions: Optional[bool] = None,
        bytecode_encryption: Optional[bool] = None,
        protection_marker: Optional[str] = None,
    ) -> None:
        """Initialize the VM protection transformer.

        Args:
            config: ObfuscationConfig instance with vm_protection options.
                   If provided, settings are extracted from config.options.
            complexity: VM complexity level (1-3). If not provided, uses config
                       or defaults to 2 (intermediate).
            protect_all_functions: Whether to protect all functions. If not provided,
                                  uses config or defaults to False.
            bytecode_encryption: Whether to encrypt bytecode. If not provided,
                                uses config or defaults to True.
            protection_marker: String marker for function protection. If not provided,
                              uses config or defaults to "vm:protect".
        """
        super().__init__()
        self.logger = get_logger("obfuscator.processors.vm_protection")

        # Import VM bytecode module
        try:
            from . import vm_bytecode
            self.vm_bytecode = vm_bytecode
        except ImportError:
            self.vm_bytecode = None
            self.logger.error("vm_bytecode module not available")

        # Determine settings from config or parameters
        if complexity is not None:
            self.complexity = complexity
        elif config is not None and hasattr(config, 'options'):
            self.complexity = config.options.get('vm_protection_complexity', 2)
        else:
            self.complexity = 2

        # Validate complexity level
        if not 1 <= self.complexity <= 3:
            self.logger.warning(
                f"Complexity level {self.complexity} is outside range 1-3, using 2"
            )
            self.complexity = 2

        if protect_all_functions is not None:
            self.protect_all_functions = protect_all_functions
        elif config is not None and hasattr(config, 'options'):
            self.protect_all_functions = config.options.get('vm_protect_all_functions', False)
        else:
            self.protect_all_functions = False

        if bytecode_encryption is not None:
            self.bytecode_encryption = bytecode_encryption
        elif config is not None and hasattr(config, 'options'):
            self.bytecode_encryption = config.options.get('vm_bytecode_encryption', True)
        else:
            self.bytecode_encryption = True

        if protection_marker is not None:
            self.protection_marker = protection_marker
        elif config is not None and hasattr(config, 'options'):
            self.protection_marker = config.options.get('vm_protection_marker', 'vm:protect')
        else:
            self.protection_marker = 'vm:protect'

        self.config = config
        self.language_mode: Optional[str] = None
        self.runtime_injected: bool = False
        self.protected_functions: set[str] = set()
        self.bytecode_compiler = None

        self.logger.debug(
            f"VMProtectionTransformer initialized with "
            f"complexity={self.complexity}, protect_all_functions={self.protect_all_functions}, "
            f"bytecode_encryption={self.bytecode_encryption}, protection_marker='{self.protection_marker}'"
        )

    def _should_protect_function(self, node: ast.AST) -> bool:
        """Determine if a function should be protected.

        Args:
            node: The function AST node to check.

        Returns:
            True if the function should be protected, False otherwise.
        """
        # Check if VM protection is enabled
        if self.vm_bytecode is None:
            return False

        # If protecting all functions, check for exclusions
        if self.protect_all_functions:
            return self._is_function_eligible(node)

        # Check for protection marker in decorators (Python)
        if hasattr(node, 'decorator_list'):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    if decorator.id == self.protection_marker:
                        return self._is_function_eligible(node)

        # Check for protection marker in preceding comments
        # This would require access to comment information
        # For now, we'll rely on decorators

        return False

    def _is_function_eligible(self, node: ast.AST) -> bool:
        """Check if a function is eligible for VM protection.

        Args:
            node: The function AST node to check.

        Returns:
            True if the function is eligible, False otherwise.
        """
        # Skip functions that are too small (less than 3 statements)
        if hasattr(node, 'body') and len(node.body) < 3:
            return False

        # Skip functions with unsupported features
        if hasattr(node, 'decorator_list'):
            # Skip functions with complex decorators
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Name):
                    return False

        # Skip async functions (not yet supported)
        if isinstance(node, (ast.AsyncFunctionDef,)):
            return False

        # Skip generator functions (not yet supported)
        if self._is_generator_function(node):
            return False

        return True

    def _is_generator_function(self, node: ast.AST) -> bool:
        """Check if a function is a generator (contains yield).

        Args:
            node: The function AST node to check.

        Returns:
            True if the function is a generator, False otherwise.
        """
        if not hasattr(node, 'body'):
            return False

        for stmt in node.body:
            if self._contains_yield(stmt):
                return True

        return False

    def _contains_yield(self, node: ast.AST) -> bool:
        """Recursively check if a node contains a yield expression.

        Args:
            node: The AST node to check.

        Returns:
            True if the node contains yield, False otherwise.
        """
        if isinstance(node, ast.Yield):
            return True

        for child in ast.walk(node):
            if isinstance(child, ast.Yield):
                return True

        return False

    def _compile_function_to_bytecode(self, node: ast.FunctionDef) -> tuple:
        """Compile a Python function to custom bytecode.

        Args:
            node: The Python FunctionDef node to compile.

        Returns:
            Tuple of (bytecode, constants, num_locals).
        """
        if self.vm_bytecode is None:
            raise RuntimeError("vm_bytecode module not available")

        # Create bytecode compiler for Python
        compiler = self.vm_bytecode.create_bytecode_compiler('python', self.complexity)

        # Compile the function
        instructions, constants, num_locals = compiler.compile_function(node)

        # Serialize bytecode
        serializer = self.vm_bytecode.BytecodeSerializer()
        bytecode = serializer.serialize(instructions)

        # Apply encryption if enabled
        if self.bytecode_encryption:
            # Generate encryption key
            import secrets
            key = secrets.token_bytes(16)
            encrypted_bytecode = self.vm_bytecode.encrypt_bytecode(bytecode, key)
            # Store key for runtime (would need to be passed to VM)
            # For simplicity, we'll skip encryption in the basic implementation
            pass

        return bytecode, constants, num_locals

    def _generate_python_wrapper(self, node: ast.FunctionDef, bytecode: list,
                                constants: list, num_locals: int) -> ast.FunctionDef:
        """Generate a Python wrapper function that executes bytecode.

        Args:
            node: The original FunctionDef node.
            bytecode: Serialized bytecode instructions.
            constants: Constant pool for the function.
            num_locals: Number of local variables.

        Returns:
            New FunctionDef node that wraps the original function.
        """
        # Import VM runtime module
        try:
            from . import vm_runtime_python
        except ImportError:
            raise RuntimeError("vm_runtime_python module not available")

        # Create wrapper function body
        wrapper_body = []

        # Add bytecode and constants as local variables
        # Convert lists to proper AST nodes using ast.List with ast.Constant elements
        bytecode_elements = [ast.Constant(value=x) for x in bytecode]
        constants_elements = []
        for const in constants:
            if isinstance(const, list):
                # Handle nested lists recursively
                constants_elements.append(ast.List(elts=[ast.Constant(value=x) for x in const], ctx=ast.Load()))
            elif isinstance(const, dict):
                # Handle dicts by creating a dict literal
                dict_keys = [ast.Constant(value=k) for k in const.keys()]
                dict_values = [ast.Constant(value=v) for v in const.values()]
                constants_elements.append(ast.Dict(keys=dict_keys, values=dict_values))
            else:
                constants_elements.append(ast.Constant(value=const))
        
        bytecode_assign = ast.Assign(
            targets=[ast.Name(id='__bytecode', ctx=ast.Store())],
            value=ast.List(elts=bytecode_elements, ctx=ast.Load())
        )
        constants_assign = ast.Assign(
            targets=[ast.Name(id='__constants', ctx=ast.Store())],
            value=ast.List(elts=constants_elements, ctx=ast.Load())
        )
        locals_assign = ast.Assign(
            targets=[ast.Name(id='__num_locals', ctx=ast.Store())],
            value=ast.Constant(value=num_locals)
        )

        wrapper_body.extend([bytecode_assign, constants_assign, locals_assign])

        # Create VM execution call
        # execute_protected_function(bytecode, constants, num_locals, *args, **kwargs)
        vm_call = ast.Call(
            func=ast.Name(id='execute_protected_function', ctx=ast.Load()),
            args=[
                ast.Name(id='__bytecode', ctx=ast.Load()),
                ast.Name(id='__constants', ctx=ast.Load()),
                ast.Name(id='__num_locals', ctx=ast.Load()),
                ast.Starred(value=ast.Name(id='args', ctx=ast.Load())),
            ],
            keywords=[
                ast.keyword(arg='globals_dict', value=ast.Call(
                    func=ast.Name(id='globals', ctx=ast.Load()),
                    args=[],
                    keywords=[]
                )),
                ast.keyword(arg=None, value=ast.Name(id='kwargs', ctx=ast.Load()))  # **kwargs
            ]
        )

        # Add return statement
        return_stmt = ast.Return(value=vm_call)
        wrapper_body.append(return_stmt)

        # Create wrapper function with original signature
        # Copy node.args to preserve the original function signature
        wrapper_args = ast.arguments(
            posonlyargs=node.args.posonlyargs,
            args=node.args.args,
            vararg=node.args.vararg,
            kwonlyargs=node.args.kwonlyargs,
            kw_defaults=node.args.kw_defaults,
            kwarg=node.args.kwarg,
            defaults=node.args.defaults
        )

        wrapper = ast.FunctionDef(
            name=node.name,
            args=wrapper_args,
            body=wrapper_body,
            decorator_list=node.decorator_list,
            returns=node.returns
        )

        # Copy location information
        ast.copy_location(wrapper, node)
        for stmt in wrapper_body:
            ast.copy_location(stmt, node)

        return wrapper

    def _inject_python_runtime(self, module: ast.Module) -> None:
        """Inject Python VM runtime at the beginning of a module.

        Args:
            module: The Python AST Module node to inject into.
        """
        if self.runtime_injected:
            return

        try:
            from . import vm_runtime_python
        except ImportError:
            self.logger.error("vm_runtime_python module not available")
            return

        # Generate runtime code
        runtime_code = vm_runtime_python.generate_python_vm_runtime(self.bytecode_encryption)

        # Parse runtime code into AST
        runtime_tree = ast.parse(runtime_code)

        # Insert runtime at the beginning of the module body
        # But preserve __future__ imports and docstrings
        insert_index = 0

        for i, stmt in enumerate(module.body):
            # Skip __future__ imports
            if isinstance(stmt, ast.ImportFrom) and stmt.module == '__future__':
                insert_index = i + 1
            # Skip module docstring (first string expression)
            elif i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                insert_index = 1
            else:
                break

        # Insert runtime nodes at the calculated position
        for j, node in enumerate(runtime_tree.body):
            ast.fix_missing_locations(node)
            module.body.insert(insert_index + j, node)

        self.runtime_injected = True
        self.logger.debug("Injected Python VM runtime")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Visit Python FunctionDef nodes and protect if marked.

        Args:
            node: The FunctionDef AST node.

        Returns:
            Either a protected wrapper function or the original node.
        """
        # Check if this function should be protected
        if self._should_protect_function(node):
            try:
                # Compile function to bytecode
                bytecode, constants, num_locals = self._compile_function_to_bytecode(node)

                # Generate wrapper function
                wrapper = self._generate_python_wrapper(node, bytecode, constants, num_locals)

                # Track protected function
                self.protected_functions.add(node.name)
                self.transformation_count += 1

                self.logger.debug(
                    f"Protected function '{node.name}' with {len(bytecode)} bytecode instructions"
                )

                return wrapper

            except Exception as e:
                error_msg = f"Failed to protect function '{node.name}': {e}"
                self.errors.append(error_msg)
                self.logger.warning(error_msg)
                # Return original function if protection fails
                return self.generic_visit(node)

        # Continue with normal traversal
        return self.generic_visit(node)

    def _compile_lua_function_to_bytecode(self, node: Any) -> tuple:
        """Compile a Lua function to custom bytecode.

        Args:
            node: The Lua function node to compile.

        Returns:
            Tuple of (bytecode, constants, num_locals).
        """
        if self.vm_bytecode is None:
            raise RuntimeError("vm_bytecode module not available")

        # Create bytecode compiler for Lua
        compiler = self.vm_bytecode.create_bytecode_compiler('lua', self.complexity)

        # Compile the function
        instructions, constants, num_locals = compiler.compile_function(node)

        # Serialize bytecode
        serializer = self.vm_bytecode.BytecodeSerializer()
        bytecode = serializer.serialize(instructions)

        return bytecode, constants, num_locals

    def _should_protect_lua_function(self, node: Any) -> bool:
        """Determine if a Lua function should be protected.

        Args:
            node: The Lua function AST node to check.

        Returns:
            True if the function should be protected, False otherwise.
        """
        # Check if VM protection is enabled
        if self.vm_bytecode is None:
            return False

        # If protecting all functions, check for exclusions
        if self.protect_all_functions:
            return self._is_lua_function_eligible(node)

        # Check for protection marker in function name or attributes
        # For Lua, we check if the function name contains the protection marker
        if hasattr(node, 'name') and isinstance(node.name, str):
            if self.protection_marker in node.name:
                return self._is_lua_function_eligible(node)

        return False

    def _is_lua_function_eligible(self, node: Any) -> bool:
        """Check if a Lua function is eligible for VM protection.

        Args:
            node: The Lua function AST node to check.

        Returns:
            True if the function is eligible, False otherwise.
        """
        # Skip functions that are too small (less than 3 statements)
        if hasattr(node, 'body') and len(node.body) < 3:
            return False

        return True

    def _generate_lua_wrapper(self, node: Any, bytecode: list,
                             constants: list, num_locals: int) -> Any:
        """Generate a Lua wrapper function that executes bytecode.

        Args:
            node: The original Lua function node.
            bytecode: Serialized bytecode instructions.
            constants: Constant pool for the function.
            num_locals: Number of local variables.

        Returns:
            New Lua function node that wraps the original function.
        """
        if not LUAPARSER_AVAILABLE:
            return node

        # Generate Lua code for the wrapper
        # Create a new function that calls execute_protected_function
        wrapper_code = f'''
        local function _wrap_{node.name}()
            local __bytecode = {{{', '.join(map(str, bytecode))}}}
            local __constants = {{{', '.join(map(repr, constants))}}}
            local __num_locals = {num_locals}
            return execute_protected_function(__bytecode, __constants, __num_locals, ...)
        end
        '''

        try:
            wrapper_tree = lua_ast.parse(wrapper_code.strip())
            # Extract the function from the parsed tree
            if hasattr(wrapper_tree, 'body') and len(wrapper_tree.body) > 0:
                wrapper_func = wrapper_tree.body[0]
                # Copy the name from the original function
                if hasattr(wrapper_func, 'name'):
                    wrapper_func.name = node.name
                return wrapper_func
        except Exception as e:
            self.logger.warning(f"Failed to generate Lua wrapper for {node.name}: {e}")
            return node

    def _traverse_lua_ast_for_protection(self, node: Any) -> None:
        """Recursively traverse Lua AST and protect marked functions.

        Args:
            node: The current Lua AST node to process.
        """
        if not LUAPARSER_AVAILABLE or node is None:
            return

        # Check if this is a function that should be protected
        if isinstance(node, lua_nodes.Function):
            if self._should_protect_lua_function(node):
                try:
                    # Compile function to bytecode
                    bytecode, constants, num_locals = self._compile_lua_function_to_bytecode(node)

                    # Generate wrapper function
                    wrapper = self._generate_lua_wrapper(node, bytecode, constants, num_locals)

                    # Track protected function
                    if hasattr(node, 'name'):
                        self.protected_functions.add(node.name)
                    self.transformation_count += 1

                    self.logger.debug(
                        f"Protected Lua function with {len(bytecode)} bytecode instructions"
                    )

                    # Replace the original function with the wrapper
                    # This is tricky in Lua AST - we need to find the parent and replace it there
                    # For now, we'll just track it and let the caller handle replacement
                    return

                except Exception as e:
                    error_msg = f"Failed to protect Lua function: {e}"
                    self.errors.append(error_msg)
                    self.logger.warning(error_msg)

        # Recursively traverse child nodes
        for attr_name in dir(node):
            if attr_name.startswith('_'):
                continue

            attr = getattr(node, attr_name, None)
            if attr is None:
                continue

            # Handle lists of nodes
            if isinstance(attr, list):
                for i, item in enumerate(attr):
                    if hasattr(item, '__class__') and hasattr(item.__class__, '__module__'):
                        if 'astnodes' in item.__class__.__module__:
                            # Check if this item is a function to protect
                            if isinstance(item, lua_nodes.Function):
                                if self._should_protect_lua_function(item):
                                    try:
                                        # Compile function to bytecode
                                        bytecode, constants, num_locals = self._compile_lua_function_to_bytecode(item)

                                        # Generate wrapper function
                                        wrapper = self._generate_lua_wrapper(item, bytecode, constants, num_locals)

                                        # Track protected function
                                        if hasattr(item, 'name'):
                                            self.protected_functions.add(item.name)
                                        self.transformation_count += 1

                                        self.logger.debug(
                                            f"Protected Lua function '{item.name}' with {len(bytecode)} bytecode instructions"
                                        )

                                        # Replace in the list
                                        attr[i] = wrapper
                                    except Exception as e:
                                        error_msg = f"Failed to protect Lua function: {e}"
                                        self.errors.append(error_msg)
                                        self.logger.warning(error_msg)
                            else:
                                self._traverse_lua_ast_for_protection(item)

            # Handle single nodes
            elif hasattr(attr, '__class__') and hasattr(attr.__class__, '__module__'):
                if 'astnodes' in attr.__class__.__module__:
                    self._traverse_lua_ast_for_protection(attr)

    def _inject_lua_runtime(self, chunk: Any) -> None:
        """Inject Lua VM runtime at the beginning of a chunk.

        Args:
            chunk: The Lua AST Chunk node to inject into.
        """
        if not LUAPARSER_AVAILABLE:
            self.logger.warning("luaparser not available, skipping Lua runtime injection")
            return

        if self.runtime_injected:
            return

        try:
            from . import vm_runtime_lua
        except ImportError:
            self.logger.error("vm_runtime_lua module not available")
            return

        # Generate runtime code
        runtime_code = vm_runtime_lua.generate_lua_vm_runtime(self.bytecode_encryption)

        # Parse the Lua runtime code
        try:
            runtime_tree = lua_ast.parse(runtime_code)
            if hasattr(runtime_tree, 'body') and hasattr(chunk, 'body'):
                # Insert runtime statements at the beginning
                if hasattr(chunk.body, 'body'):
                    # chunk.body is a Block with a body attribute
                    chunk.body.body = runtime_tree.body.body + chunk.body.body
                else:
                    chunk.body = runtime_tree.body.body + chunk.body
            self.runtime_injected = True
            self.logger.debug("Injected Lua VM runtime")
        except Exception as e:
            error_msg = f"Failed to inject Lua runtime: {e}"
            self.errors.append(error_msg)
            self.logger.error(error_msg)

    def transform(self, ast_node: ast.AST) -> TransformResult:
        """Apply VM protection transformation to an AST.

        This method orchestrates the transformation:
        1. Detects language from AST node type
        2. Identifies and protects marked functions
        3. Injects VM runtime at module/chunk level
        4. Returns structured result

        Args:
            ast_node: The AST node to transform (ast.Module for Python).

        Returns:
            TransformResult with transformed AST, success status, and metrics.
        """
        # Reset state for this transformation
        self.transformation_count = 0
        self.errors = []
        self.runtime_injected = False
        self.protected_functions.clear()

        try:
            # Detect language and apply transformations, then inject runtime
            if isinstance(ast_node, ast.Module):
                self.language_mode = 'python'

                # Fix missing locations before transformation
                ast.fix_missing_locations(ast_node)

                # Apply Python transformations by visiting the AST
                transformed_node = self.visit(ast_node)

                if transformed_node is None:
                    raise ValueError("Transformation returned None")

                # Inject runtime after transformation
                self._inject_python_runtime(transformed_node)

                # Fix missing locations on newly created nodes
                ast.fix_missing_locations(transformed_node)

            elif LUAPARSER_AVAILABLE and isinstance(ast_node, lua_nodes.Chunk):
                self.language_mode = 'lua'

                # Apply Lua transformations using custom traversal
                self._traverse_lua_ast_for_protection(ast_node)

                # Inject runtime after transformation
                self._inject_lua_runtime(ast_node)

                # Lua transformation is in-place
                transformed_node = ast_node

            else:
                # For other node types, try to proceed without transformation
                self.logger.warning(
                    f"Unknown AST type {type(ast_node).__name__}, "
                    "proceeding without VM protection"
                )
                transformed_node = ast_node

            self.logger.debug(
                f"VM protection transformation completed: "
                f"{self.transformation_count} functions protected"
            )

            return TransformResult(
                ast_node=transformed_node,
                success=True,
                transformation_count=self.transformation_count,
                errors=self.errors,
            )

        except Exception as e:
            error_msg = f"VM protection transformation failed: {e.__class__.__name__}: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.errors.append(error_msg)

            return TransformResult(
                ast_node=None,
                success=False,
                transformation_count=self.transformation_count,
                errors=self.errors,
            )


class DeadCodeInjectionTransformer(ASTTransformer):
    """Injects syntactically valid but unreachable dead code for obfuscation.

    This transformer injects realistic-looking dead code at strategic unreachable
    locations to confuse static analysis while ensuring the code never executes.
    It supports both Python and Lua languages.

    Dead code is injected at:
    - After return statements (unreachable)
    - Inside always-false conditionals (if False: ...)
    - After unconditional break statements

    For Python, generates:
    - Variable assignments with arithmetic expressions
    - Unreachable function calls (len(), str(), int())
    - Dead loops: for _ in range(0): ... or while False: ...
    - Dead conditionals: if False: ... with nested statements
    - Dead try/except blocks with empty handlers

    For Lua, generates:
    - Local variable assignments with expressions
    - Unreachable function calls (print(), tostring())
    - Dead loops: for i = 1, 0 do ... end or while false do ... end
    - Dead conditionals: if false then ... end

    Attributes:
        config: ObfuscationConfig instance with dead code options
        dead_code_percentage: Probability (0-100) of injecting at each eligible location
        language_mode: Detected language ('python', 'lua', or None)
        injection_count: Counter for number of dead code blocks injected
        errors: List of error messages collected during transformation

    Example:
        >>> from obfuscator.core.config import ObfuscationConfig
        >>> config = ObfuscationConfig(
        ...     name="test",
        ...     features={"dead_code_injection": True},
        ...     options={"dead_code_percentage": 30}
        ... )
        >>> transformer = DeadCodeInjectionTransformer(config)
        >>> tree = ast.parse("def f(): return 42")
        >>> result = transformer.transform(tree)
        >>> if result.success:
        ...     print(f"Injected {result.transformation_count} dead code blocks")

    Limitations:
    - Dead code is never executed but increases file size
    - Very high percentages (>80%) may significantly bloat code
    - Does not inject into lambda functions
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        dead_code_percentage: Optional[int] = None,
    ) -> None:
        """Initialize the dead code injection transformer.

        Args:
            config: ObfuscationConfig instance with dead_code_percentage option.
                   If provided, extracts settings from config.options.
            dead_code_percentage: Percentage (0-100) for injection probability.
                                Overrides config if explicitly set.
        """
        super().__init__()
        self.logger = get_logger("obfuscator.processors.dead_code_injection")

        # Determine percentage from config or parameter
        if dead_code_percentage is not None:
            self.dead_code_percentage = dead_code_percentage
        elif config is not None and hasattr(config, 'options'):
            self.dead_code_percentage = config.options.get('dead_code_percentage', 20)
        else:
            self.dead_code_percentage = 20

        # Validate percentage range
        self.dead_code_percentage = max(0, min(100, self.dead_code_percentage))

        self.config = config

        # Language detection and tracking
        self.language_mode: Optional[str] = None
        self.injection_count: int = 0

        # Variable name pools for realistic dead code
        self._python_var_prefixes = ['_tmp_', '_unused_', '_cache_', '_dead_', '_aux_']
        self._lua_var_prefixes = ['_tmp_', '_unused_', '_cache_', '_dead_', '_aux_']
        self._var_counter: int = 0

        self.logger.debug(
            f"DeadCodeInjectionTransformer initialized with "
            f"dead_code_percentage={self.dead_code_percentage}"
        )

    def _should_inject(self) -> bool:
        """Determine if dead code should be injected at current location.

        Uses dead_code_percentage to decide probabilistically.

        Returns:
            True if dead code should be injected here.
        """
        return random.randint(0, 99) < self.dead_code_percentage

    def _generate_python_variable_name(self) -> str:
        """Generate a realistic-looking unused variable name for Python.

        Returns:
            Variable name like '_tmp_0', '_unused_1', etc.
        """
        prefix = random.choice(self._python_var_prefixes)
        name = f"{prefix}{self._var_counter}"
        self._var_counter += 1
        return name

    def _generate_lua_variable_name(self) -> str:
        """Generate a realistic-looking unused variable name for Lua.

        Returns:
            Variable name like '_tmp_0', '_unused_1', etc.
        """
        prefix = random.choice(self._lua_var_prefixes)
        name = f"{prefix}{self._var_counter}"
        self._var_counter += 1
        return name

    def _create_python_arithmetic_expression(self) -> ast.expr:
        """Create a random arithmetic expression for Python dead code.

        Returns:
            AST expression node with arithmetic operations.
        """
        patterns = [
            # Simple binary operations
            lambda: ast.BinOp(
                left=ast.Constant(value=random.randint(1, 100)),
                op=ast.Add(),
                right=ast.Constant(value=random.randint(1, 100))
            ),
            lambda: ast.BinOp(
                left=ast.Constant(value=random.randint(1, 50)),
                op=ast.Mult(),
                right=ast.Constant(value=random.randint(2, 10))
            ),
            lambda: ast.BinOp(
                left=ast.Constant(value=random.randint(100, 200)),
                op=ast.Sub(),
                right=ast.Constant(value=random.randint(1, 99))
            ),
            # Comparison expressions
            lambda: ast.Compare(
                left=ast.Constant(value=random.randint(1, 100)),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=random.randint(1, 100))]
            ),
            # String operations
            lambda: ast.Call(
                func=ast.Name(id='len', ctx=ast.Load()),
                args=[ast.Constant(value='')],
                keywords=[]
            ),
            lambda: ast.Call(
                func=ast.Name(id='str', ctx=ast.Load()),
                args=[ast.Constant(value=random.randint(1, 100))],
                keywords=[]
            ),
        ]
        return random.choice(patterns)()

    def _create_python_dead_assignment(self) -> ast.Assign:
        """Create a dead variable assignment for Python.

        Returns:
            Assign AST node with random expression.
        """
        var_name = self._generate_python_variable_name()
        return ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=self._create_python_arithmetic_expression()
        )

    def _create_python_dead_call(self) -> ast.Expr:
        """Create a dead function call expression for Python.

        Returns:
            Expr AST node wrapping a function call.
        """
        builtin_funcs = ['len', 'str', 'int', 'float', 'abs', 'max', 'min']
        func_name = random.choice(builtin_funcs)
        args = [ast.Constant(value=random.randint(1, 100))]

        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id=func_name, ctx=ast.Load()),
                args=args,
                keywords=[]
            )
        )

    def _create_python_dead_loop(self) -> ast.stmt:
        """Create a dead loop that never executes for Python.

        Returns:
            For or While AST node with always-false condition.
        """
        if random.choice([True, False]):
            # for _ in range(0): pass
            return ast.For(
                target=ast.Name(id='_', ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id='range', ctx=ast.Load()),
                    args=[ast.Constant(value=0)],
                    keywords=[]
                ),
                body=[ast.Pass()],
                orelse=[]
            )
        else:
            # while False: pass
            return ast.While(
                test=ast.Constant(value=False),
                body=[ast.Pass()],
                orelse=[]
            )

    def _create_python_dead_conditional(self) -> ast.If:
        """Create a dead if statement that never executes for Python.

        Returns:
            If AST node with False condition and dead body.
        """
        # Generate 1-3 dead statements for the body
        body_stmts = []
        for _ in range(random.randint(1, 3)):
            stmt_type = random.choice(['assign', 'call', 'loop'])
            if stmt_type == 'assign':
                body_stmts.append(self._create_python_dead_assignment())
            elif stmt_type == 'call':
                body_stmts.append(self._create_python_dead_call())
            else:
                body_stmts.append(self._create_python_dead_loop())

        return ast.If(
            test=ast.Constant(value=False),
            body=body_stmts,
            orelse=[]
        )

    def _generate_python_dead_code(self, count: int = 1) -> list[ast.stmt]:
        """Generate dead code statements for Python.

        Args:
            count: Number of dead code blocks to generate.

        Returns:
            List of dead code AST statement nodes.
        """
        statements = []
        for _ in range(count):
            stmt_type = random.choice(['assign', 'call', 'conditional', 'loop'])
            try:
                if stmt_type == 'assign':
                    statements.append(self._create_python_dead_assignment())
                elif stmt_type == 'call':
                    statements.append(self._create_python_dead_call())
                elif stmt_type == 'conditional':
                    statements.append(self._create_python_dead_conditional())
                else:
                    statements.append(self._create_python_dead_loop())
            except Exception as e:
                # Log error but continue with other statements
                self.logger.debug(f"Failed to generate dead code statement: {e}")
                continue

        return statements

    def _inject_dead_code_after_return_python(
        self, body: list[ast.stmt]
    ) -> list[ast.stmt]:
        """Inject dead code after return statements in Python function body.

        Args:
            body: Original function body statements.

        Returns:
            Modified body with dead code injected after returns.
        """
        new_body = []
        injected_locations = set()

        for i, stmt in enumerate(body):
            new_body.append(stmt)

            # Check if this is a return statement
            if isinstance(stmt, ast.Return):
                if self._should_inject() and i not in injected_locations:
                    # Inject 1-3 dead code statements
                    dead_count = random.randint(1, 3)
                    dead_code = self._generate_python_dead_code(dead_count)
                    for dead_stmt in dead_code:
                        ast.fix_missing_locations(dead_stmt)
                        new_body.append(dead_stmt)
                        self.injection_count += 1
                        self.transformation_count += 1
                    injected_locations.add(i)
                    self.logger.debug(f"Injected {len(dead_code)} dead code statements after return")

        return new_body

    def _inject_dead_code_after_break_python(
        self, body: list[ast.stmt]
    ) -> list[ast.stmt]:
        """Inject dead code after break statements in Python loop body.

        Args:
            body: Original loop body statements.

        Returns:
            Modified body with dead code injected after breaks.
        """
        new_body = []
        injected_locations = set()

        for i, stmt in enumerate(body):
            new_body.append(stmt)

            # Check if this is a break statement
            if isinstance(stmt, ast.Break):
                if self._should_inject() and i not in injected_locations:
                    # Inject 1-3 dead code statements
                    dead_count = random.randint(1, 3)
                    dead_code = self._generate_python_dead_code(dead_count)
                    for dead_stmt in dead_code:
                        ast.fix_missing_locations(dead_stmt)
                        new_body.append(dead_stmt)
                        self.injection_count += 1
                        self.transformation_count += 1
                    injected_locations.add(i)
                    self.logger.debug(f"Injected {len(dead_code)} dead code statements after break")

        return new_body

    def _create_lua_dead_assignment(self) -> Any:
        """Create a dead variable assignment for Lua.

        Returns:
            Lua AST assignment node.
        """
        if not LUAPARSER_AVAILABLE:
            return None

        var_name = self._generate_lua_variable_name()
        # Create a simple numeric expression
        value = lua_nodes.Number(random.randint(1, 100))

        return lua_nodes.LocalAssign(
            targets=[lua_nodes.Name(var_name)],
            values=[value]
        )

    def _create_lua_dead_call(self) -> Any:
        """Create a dead function call expression for Lua.

        Returns:
            Lua AST call statement node.
        """
        if not LUAPARSER_AVAILABLE:
            return None

        # Choose a common Lua function
        func_name = random.choice(['print', 'tostring', 'tonumber', 'type'])
        args = [lua_nodes.String('dead')]

        call = lua_nodes.Call(
            func=lua_nodes.Name(func_name),
            args=args
        )
        return lua_nodes.Assign(
            targets=[lua_nodes.Name('_discard_')],
            values=[call]
        )

    def _create_lua_dead_loop(self) -> Any:
        """Create a dead loop that never executes for Lua.

        Returns:
            Lua AST While node with always-false condition.
        """
        if not LUAPARSER_AVAILABLE:
            return None

        # while false do end
        return lua_nodes.While(
            test=lua_nodes.FalseExpr(),
            body=lua_nodes.Block([])
        )

    def _create_lua_dead_conditional(self) -> Any:
        """Create a dead if statement that never executes for Lua.

        Returns:
            Lua AST If node with false condition.
        """
        if not LUAPARSER_AVAILABLE:
            return None

        # Generate 1-2 dead statements for the body
        body_stmts = []
        for _ in range(random.randint(1, 2)):
            stmt_type = random.choice(['assign', 'call'])
            if stmt_type == 'assign':
                stmt = self._create_lua_dead_assignment()
            else:
                stmt = self._create_lua_dead_call()
            if stmt:
                body_stmts.append(stmt)

        return lua_nodes.If(
            test=lua_nodes.FalseExpr(),
            body=lua_nodes.Block(body_stmts),
            orelse=lua_nodes.Block([])
        )

    def _generate_lua_dead_code(self, count: int = 1) -> list[Any]:
        """Generate dead code statements for Lua.

        Args:
            count: Number of dead code blocks to generate.

        Returns:
            List of dead code Lua AST statement nodes.
        """
        if not LUAPARSER_AVAILABLE:
            return []

        statements = []
        for _ in range(count):
            stmt_type = random.choice(['assign', 'call', 'conditional', 'loop'])
            try:
                if stmt_type == 'assign':
                    stmt = self._create_lua_dead_assignment()
                elif stmt_type == 'call':
                    stmt = self._create_lua_dead_call()
                elif stmt_type == 'conditional':
                    stmt = self._create_lua_dead_conditional()
                else:
                    stmt = self._create_lua_dead_loop()

                if stmt:
                    statements.append(stmt)
            except Exception as e:
                self.logger.debug(f"Failed to generate Lua dead code: {e}")
                continue

        return statements

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Visit and inject dead code into Python function definition.

        Args:
            node: The FunctionDef AST node to potentially transform.

        Returns:
            Transformed function node with dead code injected after returns.
        """
        # First visit children for nested transformations
        self.generic_visit(node)

        # Inject dead code after return statements
        if self._should_inject() or self.dead_code_percentage == 100:
            new_body = self._inject_dead_code_after_return_python(node.body)

            if new_body is not node.body:
                # Create new function node with modified body
                new_node = ast.FunctionDef(
                    name=node.name,
                    args=node.args,
                    body=new_body,
                    decorator_list=node.decorator_list,
                    returns=node.returns,
                    type_comment=node.type_comment
                )
                ast.copy_location(new_node, node)
                return new_node

        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Visit async function definition - apply same dead code injection.

        Args:
            node: The AsyncFunctionDef AST node.

        Returns:
            Transformed node with dead code injected.
        """
        # Apply same transformation as regular functions
        self.generic_visit(node)

        if self._should_inject() or self.dead_code_percentage == 100:
            new_body = self._inject_dead_code_after_return_python(node.body)

            if new_body is not node.body:
                new_node = ast.AsyncFunctionDef(
                    name=node.name,
                    args=node.args,
                    body=new_body,
                    decorator_list=node.decorator_list,
                    returns=node.returns,
                    type_comment=node.type_comment
                )
                ast.copy_location(new_node, node)
                return new_node

        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        """Visit while loop and inject dead code after break statements.

        Args:
            node: The While AST node.

        Returns:
            Transformed node with dead code injected after breaks.
        """
        # Visit children first
        self.generic_visit(node)

        # Inject dead code after break statements in the loop body
        if self._should_inject() or self.dead_code_percentage == 100:
            new_body = self._inject_dead_code_after_break_python(node.body)
            if new_body is not node.body:
                node.body = new_body

        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        """Visit for loop and inject dead code after break statements.

        Args:
            node: The For AST node.

        Returns:
            Transformed node with dead code injected after breaks.
        """
        # Visit children first
        self.generic_visit(node)

        # Inject dead code after break statements in the loop body
        if self._should_inject() or self.dead_code_percentage == 100:
            new_body = self._inject_dead_code_after_break_python(node.body)
            if new_body is not node.body:
                node.body = new_body

        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        """Visit if statement and potentially add dead code.

        Args:
            node: The If AST node.

        Returns:
            Transformed node with potential dead code in false branches.
        """
        # Visit children first
        self.generic_visit(node)

        # Potentially inject dead code in else branch
        if not node.orelse and self._should_inject():
            # Create a false conditional as else
            dead_if = self._create_python_dead_conditional()
            ast.fix_missing_locations(dead_if)
            node.orelse = [dead_if]
            self.injection_count += 1
            self.transformation_count += 1
            self.logger.debug("Injected dead code in if-else branch")

        return node

    def _traverse_lua_ast_for_dead_code(self, node: Any) -> None:
        """Recursively traverse Lua AST and inject dead code.

        Args:
            node: The Lua AST node to process.
        """
        if not LUAPARSER_AVAILABLE or node is None:
            return

        # Check if this is a function definition with a body
        if isinstance(node, lua_nodes.Function):
            body = getattr(node, 'body', None)
            if body and self._should_inject():
                body_stmts = getattr(body, 'body', [])
                new_body_stmts = []
                injected_count = 0

                for i, stmt in enumerate(body_stmts):
                    new_body_stmts.append(stmt)

                    # Inject after return statements
                    if isinstance(stmt, lua_nodes.Return):
                        if self._should_inject():
                            dead_count = random.randint(1, 2)
                            dead_code = self._generate_lua_dead_code(dead_count)
                            for dead_stmt in dead_code:
                                new_body_stmts.append(dead_stmt)
                                injected_count += 1

                if injected_count > 0:
                    body.body = new_body_stmts
                    self.injection_count += injected_count
                    self.transformation_count += injected_count
                    self.logger.debug(f"Injected {injected_count} dead code blocks in Lua function")

        # Recursively traverse child nodes
        for attr_name in dir(node):
            if attr_name.startswith('_'):
                continue

            attr = getattr(node, attr_name, None)
            if attr is None:
                continue

            if isinstance(attr, list):
                for item in attr:
                    if hasattr(item, '__class__') and hasattr(item.__class__, '__module__'):
                        if 'astnodes' in item.__class__.__module__:
                            self._traverse_lua_ast_for_dead_code(item)
            elif hasattr(attr, '__class__') and hasattr(attr.__class__, '__module__'):
                if 'astnodes' in attr.__class__.__module__:
                    self._traverse_lua_ast_for_dead_code(attr)

    def transform(self, ast_node: ast.AST) -> TransformResult:
        """Apply dead code injection transformation to an AST.

        This method orchestrates the transformation:
        1. Detects language from AST node type
        2. Applies dead code injection at appropriate locations
        3. Returns structured result

        Args:
            ast_node: The AST node to transform (ast.Module for Python,
                     lua_nodes.Chunk for Lua).

        Returns:
            TransformResult with transformed AST, success status, and metrics.
        """
        # Guard against None or invalid AST types
        if ast_node is None:
            error_msg = "Invalid input: ast_node is None"
            self.logger.error(error_msg)
            return TransformResult(
                ast_node=None,
                success=False,
                transformation_count=0,
                errors=[error_msg],
            )

        # Check for unsupported AST types
        is_python_module = isinstance(ast_node, ast.Module)
        is_lua_chunk = LUAPARSER_AVAILABLE and isinstance(ast_node, lua_nodes.Chunk)

        if not is_python_module and not is_lua_chunk:
            error_msg = f"Unsupported AST type: {type(ast_node).__name__}. Expected ast.Module or lua_nodes.Chunk."
            self.logger.error(error_msg)
            return TransformResult(
                ast_node=None,
                success=False,
                transformation_count=0,
                errors=[error_msg],
            )

        # Reset state for this transformation
        self.transformation_count = 0
        self.errors = []
        self.injection_count = 0
        self._var_counter = 0

        try:
            # Detect language and apply transformations
            if isinstance(ast_node, ast.Module):
                self.language_mode = 'python'
                self.logger.debug("Detected Python language mode")

                # Fix missing locations before transformation
                ast.fix_missing_locations(ast_node)

                # Apply Python transformations by visiting the AST
                transformed_node = self.visit(ast_node)

                if transformed_node is None:
                    raise ValueError("Transformation returned None")

                # Fix missing locations on newly created nodes
                ast.fix_missing_locations(transformed_node)

            elif LUAPARSER_AVAILABLE and isinstance(ast_node, lua_nodes.Chunk):
                self.language_mode = 'lua'
                self.logger.debug("Detected Lua language mode")

                # Apply Lua transformations using custom traversal
                self._traverse_lua_ast_for_dead_code(ast_node)

                # Lua transformation is in-place
                transformed_node = ast_node

            self.logger.info(
                f"Dead code injection completed: "
                f"{self.injection_count} blocks injected"
            )

            return TransformResult(
                ast_node=transformed_node,
                success=True,
                transformation_count=self.injection_count,
                errors=self.errors,
            )

        except Exception as e:
            error_msg = f"Dead code injection failed: {e.__class__.__name__}: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.errors.append(error_msg)

            return TransformResult(
                ast_node=None,
                success=False,
                transformation_count=self.injection_count,
                errors=self.errors,
            )


class OpaquePredicatesTransformer(ASTTransformer):
    """Injects opaque predicates to obfuscate control flow.

    Opaque predicates are conditions that are always true or always false,
    but appear complex enough to confuse static analysis tools. This
    transformer wraps existing conditions with mathematically deterministic
    predicates that preserve program semantics while making reverse
    engineering more difficult.

    The transformer supports both Python and Lua languages.

    For Python, generates:
    - Simple identities (complexity 1): x*x >= 0, abs(x) >= 0, len([]) == 0
    - Bitwise operations (complexity 2): (x|0) == x, (x&x) == x, x^0 == x
    - Complex expressions (complexity 3): (x*2)//2 == x, pow(x,0) == 1

    For Lua, generates:
    - Simple identities: x*x >= 0, math.abs(x) >= 0
    - Type-specific predicates: x%1 >= 0 (number check)
    - Complex mathematical expressions

    Injection locations:
    - If statement conditions (wrapped with AND opaque_true)
    - While loop conditions (wrapped with AND opaque_true)
    - For loop bodies (opaque check as first statement)
    - Function entries (opaque variable + conditional wrapper)

    Attributes:
        config: ObfuscationConfig instance with opaque predicate options
        opaque_predicate_complexity: Complexity level (1-3) for predicate generation
        opaque_predicate_percentage: Probability (0-100) of injection at eligible locations
        language_mode: Detected language ('python', 'lua', or None)
        transformation_count: Counter for number of predicates injected
        predicate_counter: Counter for generating unique variable names
        errors: List of error messages collected during transformation

    Example:
        >>> from obfuscator.core.config import ObfuscationConfig
        >>> config = ObfuscationConfig(
        ...     name="test",
        ...     features={"opaque_predicates": True},
        ...     options={"opaque_predicate_complexity": 2, "opaque_predicate_percentage": 30}
        ... )
        >>> transformer = OpaquePredicatesTransformer(config)
        >>> tree = ast.parse("if x > 0: print(x)")
        >>> result = transformer.transform(tree)
        >>> if result.success:
        ...     print(f"Injected {result.transformation_count} opaque predicates")

    Limitations:
    - Predicates increase code size and may slightly impact performance
    - Very high percentages (>80%) may significantly bloat code
    - Does not inject into lambda functions or comprehensions
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        opaque_predicate_complexity: Optional[int] = None,
        opaque_predicate_percentage: Optional[int] = None,
    ) -> None:
        """Initialize the opaque predicates transformer.

        Args:
            config: ObfuscationConfig instance with opaque_predicate options.
                   If provided, extracts settings from config.options.
            opaque_predicate_complexity: Complexity level (1-3) for predicate generation.
                                         Overrides config if explicitly set.
            opaque_predicate_percentage: Percentage (0-100) for injection probability.
                                         Overrides config if explicitly set.
        """
        super().__init__()
        self.logger = get_logger("obfuscator.processors.opaque_predicates")

        # Determine complexity from config or parameter
        if opaque_predicate_complexity is not None:
            self.opaque_predicate_complexity = opaque_predicate_complexity
        elif config is not None and hasattr(config, 'options'):
            self.opaque_predicate_complexity = config.options.get('opaque_predicate_complexity', 2)
        else:
            self.opaque_predicate_complexity = 2

        # Determine percentage from config or parameter
        if opaque_predicate_percentage is not None:
            self.opaque_predicate_percentage = opaque_predicate_percentage
        elif config is not None and hasattr(config, 'options'):
            self.opaque_predicate_percentage = config.options.get('opaque_predicate_percentage', 30)
        else:
            self.opaque_predicate_percentage = 30

        # Validate and clamp ranges
        self.opaque_predicate_complexity = max(1, min(3, self.opaque_predicate_complexity))
        self.opaque_predicate_percentage = max(0, min(100, self.opaque_predicate_percentage))

        self.config = config

        # Language detection and tracking
        self.language_mode: Optional[str] = None
        self.predicate_counter: int = 0

        self.logger.debug(
            f"OpaquePredicatesTransformer initialized with "
            f"complexity={self.opaque_predicate_complexity}, "
            f"percentage={self.opaque_predicate_percentage}"
        )

    def _should_inject(self) -> bool:
        """Determine if opaque predicate should be injected at current location.

        Uses opaque_predicate_percentage to decide probabilistically.

        Returns:
            True if opaque predicate should be injected here.
        """
        return random.randint(0, 99) < self.opaque_predicate_percentage

    def _generate_opaque_var_name(self) -> str:
        """Generate a unique opaque variable name.

        Returns:
            Variable name like '_opaque_0', '_opaque_1', etc.
        """
        name = f"_opaque_{self.predicate_counter}"
        self.predicate_counter += 1
        return name

    def _generate_python_opaque_true(self, var_name: str) -> ast.expr:
        """Generate an always-true opaque predicate for Python.

        Args:
            var_name: The variable name to use in the predicate.

        Returns:
            AST expression node that evaluates to True.
        """
        complexity = self.opaque_predicate_complexity

        if complexity == 1:
            # Simple identities
            patterns = [
                # x * x >= 0 (square is always non-negative)
                lambda: ast.Compare(
                    left=ast.BinOp(
                        left=ast.Name(id=var_name, ctx=ast.Load()),
                        op=ast.Mult(),
                        right=ast.Name(id=var_name, ctx=ast.Load())
                    ),
                    ops=[ast.GtE()],
                    comparators=[ast.Constant(value=0)]
                ),
                # abs(x) >= 0
                lambda: ast.Compare(
                    left=ast.Call(
                        func=ast.Name(id='abs', ctx=ast.Load()),
                        args=[ast.Name(id=var_name, ctx=ast.Load())],
                        keywords=[]
                    ),
                    ops=[ast.GtE()],
                    comparators=[ast.Constant(value=0)]
                ),
                # len([]) == 0
                lambda: ast.Compare(
                    left=ast.Call(
                        func=ast.Name(id='len', ctx=ast.Load()),
                        args=[ast.Constant(value=[])],
                        keywords=[]
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=0)]
                ),
            ]
        elif complexity == 2:
            # Bitwise operations
            patterns = [
                # (x | 0) == x
                lambda: ast.Compare(
                    left=ast.BinOp(
                        left=ast.Name(id=var_name, ctx=ast.Load()),
                        op=ast.BitOr(),
                        right=ast.Constant(value=0)
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Name(id=var_name, ctx=ast.Load())]
                ),
                # (x & x) == x
                lambda: ast.Compare(
                    left=ast.BinOp(
                        left=ast.Name(id=var_name, ctx=ast.Load()),
                        op=ast.BitAnd(),
                        right=ast.Name(id=var_name, ctx=ast.Load())
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Name(id=var_name, ctx=ast.Load())]
                ),
                # x ^ 0 == x
                lambda: ast.Compare(
                    left=ast.BinOp(
                        left=ast.Name(id=var_name, ctx=ast.Load()),
                        op=ast.BitXor(),
                        right=ast.Constant(value=0)
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Name(id=var_name, ctx=ast.Load())]
                ),
            ]
        else:  # complexity == 3
            # Complex expressions
            patterns = [
                # (x * 2) // 2 == x or x % 2 == 1
                lambda: ast.BoolOp(
                    op=ast.Or(),
                    values=[
                        ast.Compare(
                            left=ast.BinOp(
                                left=ast.BinOp(
                                    left=ast.Name(id=var_name, ctx=ast.Load()),
                                    op=ast.Mult(),
                                    right=ast.Constant(value=2)
                                ),
                                op=ast.FloorDiv(),
                                right=ast.Constant(value=2)
                            ),
                            ops=[ast.Eq()],
                            comparators=[ast.Name(id=var_name, ctx=ast.Load())]
                        ),
                        ast.Compare(
                            left=ast.BinOp(
                                left=ast.Name(id=var_name, ctx=ast.Load()),
                                op=ast.Mod(),
                                right=ast.Constant(value=2)
                            ),
                            ops=[ast.Eq()],
                            comparators=[ast.Constant(value=1)]
                        )
                    ]
                ),
                # pow(x, 0) == 1
                lambda: ast.Compare(
                    left=ast.Call(
                        func=ast.Name(id='pow', ctx=ast.Load()),
                        args=[
                            ast.Name(id=var_name, ctx=ast.Load()),
                            ast.Constant(value=0)
                        ],
                        keywords=[]
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=1)]
                ),
                # x ** 2 >= 0
                lambda: ast.Compare(
                    left=ast.BinOp(
                        left=ast.Name(id=var_name, ctx=ast.Load()),
                        op=ast.Pow(),
                        right=ast.Constant(value=2)
                    ),
                    ops=[ast.GtE()],
                    comparators=[ast.Constant(value=0)]
                ),
            ]

        return random.choice(patterns)()

    def _generate_python_opaque_false(self, var_name: str) -> ast.expr:
        """Generate an always-false opaque predicate for Python.

        Args:
            var_name: The variable name to use in the predicate.

        Returns:
            AST expression node that evaluates to False.
        """
        complexity = self.opaque_predicate_complexity

        if complexity == 1:
            # Simple false predicates
            patterns = [
                # x * x < 0
                lambda: ast.Compare(
                    left=ast.BinOp(
                        left=ast.Name(id=var_name, ctx=ast.Load()),
                        op=ast.Mult(),
                        right=ast.Name(id=var_name, ctx=ast.Load())
                    ),
                    ops=[ast.Lt()],
                    comparators=[ast.Constant(value=0)]
                ),
                # len([1]) == 0
                lambda: ast.Compare(
                    left=ast.Call(
                        func=ast.Name(id='len', ctx=ast.Load()),
                        args=[ast.Constant(value=[1])],
                        keywords=[]
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=0)]
                ),
                # abs(x) < 0
                lambda: ast.Compare(
                    left=ast.Call(
                        func=ast.Name(id='abs', ctx=ast.Load()),
                        args=[ast.Name(id=var_name, ctx=ast.Load())],
                        keywords=[]
                    ),
                    ops=[ast.Lt()],
                    comparators=[ast.Constant(value=0)]
                ),
            ]
        elif complexity == 2:
            # Bitwise false predicates
            patterns = [
                # (x | 0) != x
                lambda: ast.Compare(
                    left=ast.BinOp(
                        left=ast.Name(id=var_name, ctx=ast.Load()),
                        op=ast.BitOr(),
                        right=ast.Constant(value=0)
                    ),
                    ops=[ast.NotEq()],
                    comparators=[ast.Name(id=var_name, ctx=ast.Load())]
                ),
                # x ^ x != 0
                lambda: ast.Compare(
                    left=ast.BinOp(
                        left=ast.Name(id=var_name, ctx=ast.Load()),
                        op=ast.BitXor(),
                        right=ast.Name(id=var_name, ctx=ast.Load())
                    ),
                    ops=[ast.NotEq()],
                    comparators=[ast.Constant(value=0)]
                ),
            ]
        else:  # complexity == 3
            # Complex false predicates
            patterns = [
                # (x + 1) < x and x >= 0
                lambda: ast.BoolOp(
                    op=ast.And(),
                    values=[
                        ast.Compare(
                            left=ast.BinOp(
                                left=ast.Name(id=var_name, ctx=ast.Load()),
                                op=ast.Add(),
                                right=ast.Constant(value=1)
                            ),
                            ops=[ast.Lt()],
                            comparators=[ast.Name(id=var_name, ctx=ast.Load())]
                        ),
                        ast.Compare(
                            left=ast.Name(id=var_name, ctx=ast.Load()),
                            ops=[ast.GtE()],
                            comparators=[ast.Constant(value=0)]
                        )
                    ]
                ),
                # pow(x, 1) != x and x == x
                lambda: ast.BoolOp(
                    op=ast.And(),
                    values=[
                        ast.Compare(
                            left=ast.Call(
                                func=ast.Name(id='pow', ctx=ast.Load()),
                                args=[
                                    ast.Name(id=var_name, ctx=ast.Load()),
                                    ast.Constant(value=1)
                                ],
                                keywords=[]
                            ),
                            ops=[ast.NotEq()],
                            comparators=[ast.Name(id=var_name, ctx=ast.Load())]
                        ),
                        ast.Constant(value=True)
                    ]
                ),
            ]

        return random.choice(patterns)()

    def _create_python_opaque_variable(self) -> tuple[ast.Assign, str]:
        """Create an opaque variable assignment with random integer value.

        Returns:
            Tuple of (Assign AST node, variable name string).
        """
        var_name = self._generate_opaque_var_name()
        assign = ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=ast.Constant(value=random.randint(1, 100))
        )
        return assign, var_name

    def _generate_lua_opaque_true(self, var_name: str) -> Any:
        """Generate an always-true opaque predicate for Lua.

        Args:
            var_name: The variable name to use in the predicate.

        Returns:
            Lua AST expression node that evaluates to true.
        """
        if not LUAPARSER_AVAILABLE:
            return None

        complexity = self.opaque_predicate_complexity

        if complexity == 1:
            # Simple identities
            patterns = [
                # x * x >= 0
                lambda: lua_nodes.GreaterOrEqualThan(
                    left=lua_nodes.Mult(
                        left=lua_nodes.Name(var_name),
                        right=lua_nodes.Name(var_name)
                    ),
                    right=lua_nodes.Number(0)
                ),
                # math.abs(x) >= 0
                lambda: lua_nodes.GreaterOrEqualThan(
                    left=lua_nodes.Call(
                        func=lua_nodes.Name('math.abs'),
                        args=[lua_nodes.Name(var_name)]
                    ),
                    right=lua_nodes.Number(0)
                ),
                # #"" == 0
                lambda: lua_nodes.Equals(
                    left=lua_nodes.Length(
                        value=lua_nodes.String('')
                    ),
                    right=lua_nodes.Number(0)
                ),
            ]
        elif complexity == 2:
            # Medium complexity
            patterns = [
                # x % 1 >= 0
                lambda: lua_nodes.GreaterOrEqualThan(
                    left=lua_nodes.Mod(
                        left=lua_nodes.Name(var_name),
                        right=lua_nodes.Number(1)
                    ),
                    right=lua_nodes.Number(0)
                ),
                # type(x) == "number" or type(x) == "string"
                lambda: lua_nodes.Or(
                    left=lua_nodes.Equals(
                        left=lua_nodes.Call(
                            func=lua_nodes.Name('type'),
                            args=[lua_nodes.Name(var_name)]
                        ),
                        right=lua_nodes.String('number')
                    ),
                    right=lua_nodes.TrueExpr()
                ),
            ]
        else:  # complexity == 3
            # Complex expressions
            patterns = [
                # math.pow(x, 0) == 1
                lambda: lua_nodes.Equals(
                    left=lua_nodes.Call(
                        func=lua_nodes.Name('math.pow'),
                        args=[
                            lua_nodes.Name(var_name),
                            lua_nodes.Number(0)
                        ]
                    ),
                    right=lua_nodes.Number(1)
                ),
                # x ^ 2 >= 0 (using math.pow)
                lambda: lua_nodes.GreaterOrEqualThan(
                    left=lua_nodes.Call(
                        func=lua_nodes.Name('math.pow'),
                        args=[
                            lua_nodes.Name(var_name),
                            lua_nodes.Number(2)
                        ]
                    ),
                    right=lua_nodes.Number(0)
                ),
            ]

        return random.choice(patterns)()

    def _generate_lua_opaque_false(self, var_name: str) -> Any:
        """Generate an always-false opaque predicate for Lua.

        Args:
            var_name: The variable name to use in the predicate.

        Returns:
            Lua AST expression node that evaluates to false.
        """
        if not LUAPARSER_AVAILABLE:
            return None

        complexity = self.opaque_predicate_complexity

        if complexity == 1:
            # Simple false predicates
            patterns = [
                # x * x < 0
                lambda: lua_nodes.LesserThan(
                    left=lua_nodes.Mult(
                        left=lua_nodes.Name(var_name),
                        right=lua_nodes.Name(var_name)
                    ),
                    right=lua_nodes.Number(0)
                ),
                # #"a" == 0
                lambda: lua_nodes.Equals(
                    left=lua_nodes.Length(
                        value=lua_nodes.String('a')
                    ),
                    right=lua_nodes.Number(0)
                ),
            ]
        elif complexity == 2:
            # Medium complexity false predicates
            patterns = [
                # x % 1 < 0
                lambda: lua_nodes.LesserThan(
                    left=lua_nodes.Mod(
                        left=lua_nodes.Name(var_name),
                        right=lua_nodes.Number(1)
                    ),
                    right=lua_nodes.Number(0)
                ),
            ]
        else:  # complexity == 3
            # Complex false predicates
            patterns = [
                # (x + 1) < x and x >= 0
                lambda: lua_nodes.And(
                    left=lua_nodes.LesserThan(
                        left=lua_nodes.Add(
                            left=lua_nodes.Name(var_name),
                            right=lua_nodes.Number(1)
                        ),
                        right=lua_nodes.Name(var_name)
                    ),
                    right=lua_nodes.GreaterOrEqualThan(
                        left=lua_nodes.Name(var_name),
                        right=lua_nodes.Number(0)
                    )
                ),
            ]

        return random.choice(patterns)()

    def _generate_lua_opaque_true_constant(self) -> Any:
        """Generate an always-true opaque predicate for Lua using only constants.

        Returns:
            Lua AST expression node that evaluates to true without referencing variables.
        """
        if not LUAPARSER_AVAILABLE:
            return None

        patterns = [
            # #"" == 0 (length of empty string is 0)
            lambda: lua_nodes.Equals(
                left=lua_nodes.Length(
                    value=lua_nodes.String('')
                ),
                right=lua_nodes.Number(0)
            ),
            # 1 > 0
            lambda: lua_nodes.GreaterThan(
                left=lua_nodes.Number(1),
                right=lua_nodes.Number(0)
            ),
            # 0 == 0
            lambda: lua_nodes.Equals(
                left=lua_nodes.Number(0),
                right=lua_nodes.Number(0)
            ),
            # true or false
            lambda: lua_nodes.Or(
                left=lua_nodes.TrueExpr(),
                right=lua_nodes.FalseExpr()
            ),
            # 1 + 1 == 2
            lambda: lua_nodes.Equals(
                left=lua_nodes.Add(
                    left=lua_nodes.Number(1),
                    right=lua_nodes.Number(1)
                ),
                right=lua_nodes.Number(2)
            ),
        ]

        return random.choice(patterns)()

    def _generate_lua_opaque_false_constant(self) -> Any:
        """Generate an always-false opaque predicate for Lua using only constants.

        Returns:
            Lua AST expression node that evaluates to false without referencing variables.
        """
        if not LUAPARSER_AVAILABLE:
            return None

        patterns = [
            # #"a" == 0 (length of "a" is 1, not 0)
            lambda: lua_nodes.Equals(
                left=lua_nodes.Length(
                    value=lua_nodes.String('a')
                ),
                right=lua_nodes.Number(0)
            ),
            # 0 > 1
            lambda: lua_nodes.GreaterThan(
                left=lua_nodes.Number(0),
                right=lua_nodes.Number(1)
            ),
            # 0 ~= 0
            lambda: lua_nodes.NotEquals(
                left=lua_nodes.Number(0),
                right=lua_nodes.Number(0)
            ),
            # true and false
            lambda: lua_nodes.And(
                left=lua_nodes.TrueExpr(),
                right=lua_nodes.FalseExpr()
            ),
            # 1 + 1 == 3
            lambda: lua_nodes.Equals(
                left=lua_nodes.Add(
                    left=lua_nodes.Number(1),
                    right=lua_nodes.Number(1)
                ),
                right=lua_nodes.Number(3)
            ),
        ]

        return random.choice(patterns)()

    def _wrap_lua_condition_with_opaque_constant(self, condition: Any, use_false: bool = False) -> Any:
        """Wrap a Lua condition with a constant-only opaque predicate.

        Args:
            condition: Original Lua condition expression.
            use_false: If True, use always-false predicate in OR pattern.

        Returns:
            Wrapped Lua condition expression using only constant predicates.
        """
        if not LUAPARSER_AVAILABLE:
            return condition

        if use_false:
            opaque_false = self._generate_lua_opaque_false_constant()
            opaque_true = self._generate_lua_opaque_true_constant()
            # Pattern: (True and original) or False -> preserves original semantics
            wrapped = lua_nodes.Or(
                left=lua_nodes.And(
                    left=opaque_true,
                    right=condition
                ),
                right=opaque_false
            )
        else:
            opaque_predicate = self._generate_lua_opaque_true_constant()
            # Create: opaque_predicate and original_condition
            wrapped = lua_nodes.And(
                left=opaque_predicate,
                right=condition
            )

        return wrapped

    def _generate_python_opaque_true_constant(self) -> ast.expr:
        """Generate an always-true opaque predicate for Python using only constants.

        Returns:
            AST expression node that evaluates to True without referencing variables.
        """
        patterns = [
            # len([]) == 0
            lambda: ast.Compare(
                left=ast.Call(
                    func=ast.Name(id='len', ctx=ast.Load()),
                    args=[ast.Constant(value=[])],
                    keywords=[]
                ),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=0)]
            ),
            # len([1, 2]) > 0
            lambda: ast.Compare(
                left=ast.Call(
                    func=ast.Name(id='len', ctx=ast.Load()),
                    args=[ast.Constant(value=[1, 2])],
                    keywords=[]
                ),
                ops=[ast.Gt()],
                comparators=[ast.Constant(value=0)]
            ),
            # 1 > 0
            lambda: ast.Compare(
                left=ast.Constant(value=1),
                ops=[ast.Gt()],
                comparators=[ast.Constant(value=0)]
            ),
            # 0 == 0
            lambda: ast.Compare(
                left=ast.Constant(value=0),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=0)]
            ),
            # True or False
            lambda: ast.BoolOp(
                op=ast.Or(),
                values=[ast.Constant(value=True), ast.Constant(value=False)]
            ),
            # 1 + 1 == 2
            lambda: ast.Compare(
                left=ast.BinOp(
                    left=ast.Constant(value=1),
                    op=ast.Add(),
                    right=ast.Constant(value=1)
                ),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=2)]
            ),
        ]

        return random.choice(patterns)()

    def _generate_python_opaque_false_constant(self) -> ast.expr:
        """Generate an always-false opaque predicate for Python using only constants.

        Returns:
            AST expression node that evaluates to False without referencing variables.
        """
        patterns = [
            # len([1]) == 0
            lambda: ast.Compare(
                left=ast.Call(
                    func=ast.Name(id='len', ctx=ast.Load()),
                    args=[ast.Constant(value=[1])],
                    keywords=[]
                ),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=0)]
            ),
            # 0 > 1
            lambda: ast.Compare(
                left=ast.Constant(value=0),
                ops=[ast.Gt()],
                comparators=[ast.Constant(value=1)]
            ),
            # 0 != 0
            lambda: ast.Compare(
                left=ast.Constant(value=0),
                ops=[ast.NotEq()],
                comparators=[ast.Constant(value=0)]
            ),
            # True and False
            lambda: ast.BoolOp(
                op=ast.And(),
                values=[ast.Constant(value=True), ast.Constant(value=False)]
            ),
            # 1 + 1 == 3
            lambda: ast.Compare(
                left=ast.BinOp(
                    left=ast.Constant(value=1),
                    op=ast.Add(),
                    right=ast.Constant(value=1)
                ),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=3)]
            ),
        ]

        return random.choice(patterns)()

    def _wrap_condition_with_opaque_python_constant(
        self, condition: ast.expr, use_false: bool = False
    ) -> ast.expr:
        """Wrap a condition with a constant-only opaque predicate using AND/OR.

        Args:
            condition: Original condition expression.
            use_false: If True, use always-false predicate in OR pattern.

        Returns:
            Wrapped condition expression using only constant predicates.
        """
        if use_false:
            opaque_false = self._generate_python_opaque_false_constant()
            opaque_true = self._generate_python_opaque_true_constant()
            # Pattern: (True and original) or False -> preserves original semantics
            wrapped = ast.BoolOp(
                op=ast.Or(),
                values=[
                    ast.BoolOp(
                        op=ast.And(),
                        values=[opaque_true, condition]
                    ),
                    opaque_false
                ]
            )
        else:
            opaque_predicate = self._generate_python_opaque_true_constant()
            # Create: opaque_predicate and original_condition
            wrapped = ast.BoolOp(
                op=ast.And(),
                values=[opaque_predicate, condition]
            )

        return wrapped

    def visit_If(self, node: ast.If) -> ast.AST:
        """Visit and potentially wrap if condition with opaque predicate.

        Args:
            node: The If AST node to potentially transform.

        Returns:
            Transformed node with opaque predicate wrapping the condition.
        """
        # First visit children for nested transformations
        self.generic_visit(node)

        # Decide whether to inject based on percentage
        if self._should_inject():
            # Randomly choose between true and false predicates
            use_false = random.choice([True, False])
            # Use constant-only predicates for if conditions (no variable assignment needed)
            wrapped_test = self._wrap_condition_with_opaque_python_constant(node.test, use_false=use_false)
            node.test = wrapped_test
            ast.fix_missing_locations(node.test)
            self.transformation_count += 1
            self.logger.debug(f"Wrapped if condition with opaque predicate (false={use_false})")

        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        """Visit and potentially wrap while condition with opaque predicate.

        Args:
            node: The While AST node to potentially transform.

        Returns:
            Transformed node with opaque predicate wrapping the condition.
        """
        # Visit children first
        self.generic_visit(node)

        # Decide whether to inject based on percentage
        if self._should_inject():
            # Randomly choose between true and false predicates
            use_false = random.choice([True, False])
            # Use constant-only predicates for while conditions (no variable assignment needed)
            wrapped_test = self._wrap_condition_with_opaque_python_constant(node.test, use_false=use_false)
            node.test = wrapped_test
            ast.fix_missing_locations(node.test)
            self.transformation_count += 1
            self.logger.debug(f"Wrapped while condition with opaque predicate (false={use_false})")

        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        """Visit and inject opaque predicate check into for loop body.

        Args:
            node: The For AST node to potentially transform.

        Returns:
            Transformed node with opaque predicate as first statement.
        """
        # Visit children first
        self.generic_visit(node)

        # Decide whether to inject based on percentage
        if self._should_inject():
            # Create opaque variable first
            opaque_var, var_name = self._create_python_opaque_variable()
            ast.fix_missing_locations(opaque_var)

            # Randomly choose between true and false predicates
            use_false = random.choice([True, False])
            if use_false:
                opaque_test = self._generate_python_opaque_false(var_name)
                # Insert a dummy false branch: if opaque_false: pass
                # This ensures the false predicate is actually emitted
                opaque_check = ast.If(
                    test=opaque_test,
                    body=[ast.Pass()],
                    orelse=node.body
                )
            else:
                opaque_test = self._generate_python_opaque_true(var_name)
                opaque_check = ast.If(
                    test=opaque_test,
                    body=node.body,
                    orelse=[]
                )
            ast.fix_missing_locations(opaque_check)

            # Replace body with the opaque-check-wrapped version
            node.body = [opaque_var, opaque_check]
            self.transformation_count += 1
            self.logger.debug(f"Injected opaque predicate in for loop body (false={use_false})")

        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Visit and inject opaque predicate at function entry.

        Args:
            node: The FunctionDef AST node to potentially transform.

        Returns:
            Transformed node with opaque variable and predicate check.
        """
        # Visit children first for nested transformations
        self.generic_visit(node)

        # Decide whether to inject based on percentage
        if self._should_inject():
            # Create opaque variable assignment
            opaque_var, var_name = self._create_python_opaque_variable()
            ast.fix_missing_locations(opaque_var)

            # Randomly choose between true and false predicates
            use_false = random.choice([True, False])
            if use_false:
                opaque_test = self._generate_python_opaque_false(var_name)
                # Insert a dummy false branch: if opaque_false: pass
                opaque_check = ast.If(
                    test=opaque_test,
                    body=[ast.Pass()],
                    orelse=node.body
                )
            else:
                opaque_test = self._generate_python_opaque_true(var_name)
                opaque_check = ast.If(
                    test=opaque_test,
                    body=node.body,
                    orelse=[]
                )
            ast.fix_missing_locations(opaque_check)

            # Prepend opaque variable and wrap body
            node.body = [opaque_var, opaque_check]
            self.transformation_count += 1
            self.logger.debug(f"Injected opaque predicate at function entry (false={use_false})")

        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Visit async function definition - apply same opaque predicate injection.

        Args:
            node: The AsyncFunctionDef AST node.

        Returns:
            Transformed node with opaque predicate injected.
        """
        # Visit children first for nested transformations
        self.generic_visit(node)

        # Decide whether to inject based on percentage
        if self._should_inject():
            # Create opaque variable assignment
            opaque_var, var_name = self._create_python_opaque_variable()
            ast.fix_missing_locations(opaque_var)

            # Randomly choose between true and false predicates
            use_false = random.choice([True, False])
            if use_false:
                opaque_test = self._generate_python_opaque_false(var_name)
                # Insert a dummy false branch: if opaque_false: pass
                opaque_check = ast.If(
                    test=opaque_test,
                    body=[ast.Pass()],
                    orelse=node.body
                )
            else:
                opaque_test = self._generate_python_opaque_true(var_name)
                opaque_check = ast.If(
                    test=opaque_test,
                    body=node.body,
                    orelse=[]
                )
            ast.fix_missing_locations(opaque_check)

            # Prepend opaque variable and wrap body
            node.body = [opaque_var, opaque_check]
            self.transformation_count += 1
            self.logger.debug(f"Injected opaque predicate at async function entry (false={use_false})")

        return node

    def _traverse_lua_ast_for_opaque_predicates(self, node: Any) -> None:
        """Recursively traverse Lua AST and inject opaque predicates.

        Args:
            node: The Lua AST node to process.
        """
        if not LUAPARSER_AVAILABLE or node is None:
            return

        # Check if this is an If statement
        if isinstance(node, lua_nodes.If):
            if self._should_inject():
                # Randomly choose between true and false predicates
                use_false = random.choice([True, False])
                # Use constant-only predicates for if conditions (no variable assignment needed)
                node.test = self._wrap_lua_condition_with_opaque_constant(node.test, use_false=use_false)
                self.transformation_count += 1
                self.logger.debug(f"Wrapped Lua if condition with opaque predicate (false={use_false})")

        # Check if this is a While loop
        elif isinstance(node, lua_nodes.While):
            if self._should_inject():
                # Randomly choose between true and false predicates
                use_false = random.choice([True, False])
                # Use constant-only predicates for while conditions (no variable assignment needed)
                node.test = self._wrap_lua_condition_with_opaque_constant(node.test, use_false=use_false)
                self.transformation_count += 1
                self.logger.debug(f"Wrapped Lua while condition with opaque predicate (false={use_false})")

        # Check if this is a For loop
        elif isinstance(node, lua_nodes.For):
            if self._should_inject():
                # Create opaque variable first
                var_name = self._generate_opaque_var_name()
                opaque_assign = lua_nodes.LocalAssign(
                    targets=[lua_nodes.Name(var_name)],
                    values=[lua_nodes.Number(random.randint(1, 100))]
                )

                body = getattr(node, 'body', None)
                if body and hasattr(body, 'body'):
                    # Randomly choose between true and false predicates
                    use_false = random.choice([True, False])
                    if use_false:
                        opaque_test = self._generate_lua_opaque_false(var_name)
                        # Insert a dummy false branch: if opaque_false then end
                        # This ensures the false predicate is actually emitted
                        dummy_block = lua_nodes.Block([])
                        opaque_if = lua_nodes.If(
                            test=opaque_test,
                            body=dummy_block,
                            orelse=body
                        )
                    else:
                        opaque_test = self._generate_lua_opaque_true(var_name)
                        opaque_if = lua_nodes.If(
                            test=opaque_test,
                            body=body,
                            orelse=lua_nodes.Block([])
                        )
                    node.body = lua_nodes.Block([opaque_assign, opaque_if])
                    self.transformation_count += 1
                    self.logger.debug(f"Injected opaque predicate in Lua for loop body (false={use_false})")

        # Check if this is a Function definition
        elif isinstance(node, lua_nodes.Function):
            if self._should_inject():
                body = getattr(node, 'body', None)
                if body and hasattr(body, 'body'):
                    body_stmts = list(body.body)

                    # Create opaque variable assignment
                    var_name = self._generate_opaque_var_name()
                    opaque_assign = lua_nodes.LocalAssign(
                        targets=[lua_nodes.Name(var_name)],
                        values=[lua_nodes.Number(random.randint(1, 100))]
                    )

                    # Randomly choose between true and false predicates
                    use_false = random.choice([True, False])
                    if use_false:
                        opaque_test = self._generate_lua_opaque_false(var_name)
                        # Insert a dummy false branch: if opaque_false then end
                        dummy_block = lua_nodes.Block([])
                        wrapped_body = lua_nodes.Block(body_stmts)
                        opaque_if = lua_nodes.If(
                            test=opaque_test,
                            body=dummy_block,
                            orelse=wrapped_body
                        )
                    else:
                        opaque_test = self._generate_lua_opaque_true(var_name)
                        wrapped_body = lua_nodes.Block(body_stmts)
                        opaque_if = lua_nodes.If(
                            test=opaque_test,
                            body=wrapped_body,
                            orelse=lua_nodes.Block([])
                        )

                    # Prepend opaque variable and wrap body
                    new_body = [opaque_assign, opaque_if]
                    body.body = new_body
                    self.transformation_count += 1
                    self.logger.debug(f"Injected opaque predicate at Lua function entry (false={use_false})")

        # Recursively traverse child nodes
        for attr_name in dir(node):
            if attr_name.startswith('_'):
                continue

            attr = getattr(node, attr_name, None)
            if attr is None:
                continue

            if isinstance(attr, list):
                for item in attr:
                    if hasattr(item, '__class__') and hasattr(item.__class__, '__module__'):
                        if 'astnodes' in item.__class__.__module__:
                            self._traverse_lua_ast_for_opaque_predicates(item)
            elif hasattr(attr, '__class__') and hasattr(attr.__class__, '__module__'):
                if 'astnodes' in attr.__class__.__module__:
                    self._traverse_lua_ast_for_opaque_predicates(attr)

    def transform(self, ast_node: ast.AST) -> TransformResult:
        """Apply opaque predicate transformation to an AST.

        This method orchestrates the transformation:
        1. Detects language from AST node type
        2. Applies opaque predicates at appropriate locations
        3. Returns structured result

        Args:
            ast_node: The AST node to transform (ast.Module for Python,
                     lua_nodes.Chunk for Lua).

        Returns:
            TransformResult with transformed AST, success status, and metrics.
        """
        # Guard against None or invalid AST types
        if ast_node is None:
            error_msg = "Invalid input: ast_node is None"
            self.logger.error(error_msg)
            return TransformResult(
                ast_node=None,
                success=False,
                transformation_count=0,
                errors=[error_msg],
            )

        # Check for unsupported AST types
        is_python_module = isinstance(ast_node, ast.Module)
        is_lua_chunk = LUAPARSER_AVAILABLE and isinstance(ast_node, lua_nodes.Chunk)

        if not is_python_module and not is_lua_chunk:
            error_msg = f"Unsupported AST type: {type(ast_node).__name__}. Expected ast.Module or lua_nodes.Chunk."
            self.logger.error(error_msg)
            return TransformResult(
                ast_node=None,
                success=False,
                transformation_count=0,
                errors=[error_msg],
            )

        # Reset state for this transformation
        self.transformation_count = 0
        self.errors = []
        self.predicate_counter = 0

        try:
            # Detect language and apply transformations
            if isinstance(ast_node, ast.Module):
                self.language_mode = 'python'
                self.logger.debug("Detected Python language mode")

                # Fix missing locations before transformation
                ast.fix_missing_locations(ast_node)

                # Apply Python transformations by visiting the AST
                transformed_node = self.visit(ast_node)

                if transformed_node is None:
                    raise ValueError("Transformation returned None")

                # Fix missing locations on newly created nodes
                ast.fix_missing_locations(transformed_node)

            elif LUAPARSER_AVAILABLE and isinstance(ast_node, lua_nodes.Chunk):
                self.language_mode = 'lua'
                self.logger.debug("Detected Lua language mode")

                # Apply Lua transformations using custom traversal
                self._traverse_lua_ast_for_opaque_predicates(ast_node)

                # Lua transformation is in-place
                transformed_node = ast_node

            self.logger.info(
                f"Opaque predicates injection completed: "
                f"{self.transformation_count} predicates injected"
            )

            return TransformResult(
                ast_node=transformed_node,
                success=True,
                transformation_count=self.transformation_count,
                errors=self.errors,
            )

        except Exception as e:
            error_msg = f"Opaque predicates injection failed: {e.__class__.__name__}: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.errors.append(error_msg)

            return TransformResult(
                ast_node=None,
                success=False,
                transformation_count=self.transformation_count,
                errors=self.errors,
            )

