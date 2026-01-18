"""AST transformation module for Python code obfuscation.

This module provides a base infrastructure for implementing AST transformations,
along with example transformations like constant folding and string encryption.
All transformers extend the base ASTTransformer class which provides error tracking,
logging, and common utilities.
"""

from __future__ import annotations

import ast
import base64
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

        Converts each byte to a Lua octal escape sequence (\\ddd).

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
