"""AST transformation module for Python code obfuscation.

This module provides a base infrastructure for implementing AST transformations,
along with example transformations like constant folding. All transformers extend
the base ASTTransformer class which provides error tracking, logging, and
common utilities.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from obfuscator.utils.logger import get_logger

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
