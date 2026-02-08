"""Comprehensive Python sample for advanced obfuscation testing.

This module provides a variety of Python constructs to test advanced
obfuscation features including control flow flattening, dead code injection,
opaque predicates, anti-debugging, code splitting, and self-modifying code.
"""

import math
import random
from typing import List, Dict, Optional, Tuple


# =============================================================================
# Basic Functions with Various Control Flow Patterns
# =============================================================================

def simple_arithmetic(a: int, b: int) -> int:
    """Simple arithmetic function."""
    return a + b


def conditional_checker(value: int) -> str:
    """Function with if/elif/else control flow."""
    if value < 0:
        return "negative"
    elif value == 0:
        return "zero"
    elif value < 100:
        return "small positive"
    else:
        return "large positive"


def loop_accumulator(n: int) -> int:
    """Function with for loop."""
    total = 0
    for i in range(n):
        total += i
    return total


def while_counter(start: int, limit: int) -> int:
    """Function with while loop."""
    count = 0
    current = start
    while current < limit:
        count += 1
        current += 1
    return count


def nested_loops(rows: int, cols: int) -> List[List[int]]:
    """Function with nested loops."""
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(i * j)
        matrix.append(row)
    return matrix


# =============================================================================
# Functions with Exception Handling
# =============================================================================

def safe_division(a: float, b: float) -> Optional[float]:
    """Function with try/except."""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        return None
    except TypeError:
        return None


def multi_exception_handler(data: str) -> Dict[str, any]:
    """Function with multiple exception types."""
    result = {"success": False, "value": None}
    try:
        value = int(data)
        result["value"] = value * 2
        result["success"] = True
    except ValueError:
        result["error"] = "Invalid integer"
    except Exception as e:
        result["error"] = str(e)
    return result


def try_except_else_finally(x: int) -> int:
    """Function with try/except/else/finally."""
    result = 0
    try:
        value = 100 // x
    except ZeroDivisionError:
        result = -1
    else:
        result = value
    finally:
        result += 1
    return result


# =============================================================================
# Class Definitions with Methods
# =============================================================================

class Calculator:
    """A calculator class with various methods."""
    
    def __init__(self, initial_value: float = 0.0):
        self.value = initial_value
        self.history: List[str] = []
    
    def add(self, x: float) -> float:
        """Add a value."""
        self.value += x
        self.history.append(f"add {x}")
        return self.value
    
    def subtract(self, x: float) -> float:
        """Subtract a value."""
        self.value -= x
        self.history.append(f"subtract {x}")
        return self.value
    
    def multiply(self, x: float) -> float:
        """Multiply by a value."""
        self.value *= x
        self.history.append(f"multiply {x}")
        return self.value
    
    def divide(self, x: float) -> Optional[float]:
        """Divide by a value."""
        if x == 0:
            return None
        self.value /= x
        self.history.append(f"divide {x}")
        return self.value
    
    def get_history(self) -> List[str]:
        """Get operation history."""
        return self.history.copy()


class DataProcessor:
    """A data processing class."""
    
    def __init__(self, data: List[int]):
        self.data = data
    
    def filter_positive(self) -> List[int]:
        """Filter positive values."""
        return [x for x in self.data if x > 0]
    
    def map_squared(self) -> List[int]:
        """Map to squared values."""
        return [x ** 2 for x in self.data]
    
    def reduce_sum(self) -> int:
        """Reduce to sum."""
        total = 0
        for x in self.data:
            total += x
        return total
    
    def process_pipeline(self) -> Dict[str, any]:
        """Run processing pipeline."""
        return {
            "filtered": self.filter_positive(),
            "squared": self.map_squared(),
            "sum": self.reduce_sum()
        }


# =============================================================================
# Nested Functions and Closures
# =============================================================================

def make_multiplier(factor: int):
    """Function returning a closure."""
    def multiplier(x: int) -> int:
        return x * factor
    return multiplier


def make_counter():
    """Function returning a counter closure."""
    count = 0
    
    def counter() -> int:
        nonlocal count
        count += 1
        return count
    
    def reset() -> None:
        nonlocal count
        count = 0
    
    return counter, reset


def outer_function(x: int):
    """Function with multiple nested levels."""
    def middle_function(y: int):
        def inner_function(z: int):
            return x + y + z
        return inner_function
    return middle_function


# =============================================================================
# Async Functions
# =============================================================================

async def async_fetch_data(url: str) -> Dict[str, str]:
    """Async function simulating data fetch."""
    # Simulated async operation
    return {"url": url, "status": "success"}


async def async_process_items(items: List[int]) -> List[int]:
    """Async function processing items."""
    results = []
    for item in items:
        results.append(item * 2)
    return results


# =============================================================================
# Generator Functions
# =============================================================================

def number_generator(n: int):
    """Generator yielding numbers."""
    for i in range(n):
        yield i


def fibonacci_generator(limit: int):
    """Generator yielding Fibonacci numbers."""
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b


def filtered_generator(data: List[int], threshold: int):
    """Generator with filtering."""
    for item in data:
        if item > threshold:
            yield item


# =============================================================================
# Functions with Complex Logic
# =============================================================================

def prime_checker(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def matrix_multiply(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    """Multiply two matrices."""
    rows_a = len(a)
    cols_a = len(a[0])
    cols_b = len(b[0])
    
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    
    return result


def recursive_factorial(n: int) -> int:
    """Recursive factorial function."""
    if n <= 1:
        return 1
    return n * recursive_factorial(n - 1)


# =============================================================================
# Functions with Pattern Matching (Python 3.10+)
# =============================================================================

def http_status_handler(status: int) -> str:
    """Handle HTTP status codes with match statement."""
    match status:
        case 200:
            return "OK"
        case 404:
            return "Not Found"
        case 500:
            return "Server Error"
        case _:
            return "Unknown"


def data_shape_handler(data: Tuple) -> str:
    """Handle different data shapes."""
    match data:
        case (x, y):
            return f"Point: ({x}, {y})"
        case (x, y, z):
            return f"3D Point: ({x}, {y}, {z})"
        case _:
            return "Unknown shape"


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Test basic functions
    print(simple_arithmetic(5, 3))
    print(conditional_checker(50))
    print(loop_accumulator(10))
    
    # Test calculator
    calc = Calculator(10)
    calc.add(5)
    calc.multiply(2)
    print(calc.get_history())
    
    # Test closures
    double = make_multiplier(2)
    print(double(7))
    
    # Test generators
    for num in fibonacci_generator(100):
        print(num)
