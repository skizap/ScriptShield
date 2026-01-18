#!/usr/bin/env python3
"""
Dependency Verification Script for Python Obfuscator

This script verifies that all required dependencies are installed correctly
and displays version information for each package.
"""

import os
import sys
from pathlib import Path


# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_success(message):
    """Print success message in green."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message):
    """Print error message in red."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_info(message):
    """Print info message in blue."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.RESET}")


def print_header(message):
    """Print header message in bold."""
    print(f"\n{Colors.BOLD}{message}{Colors.RESET}")


def verify_pyqt6():
    """Verify PyQt6 installation and functionality."""
    try:
        # Set offscreen platform for headless/CI environments
        if "QT_QPA_PLATFORM" not in os.environ:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"

        from PyQt6.QtWidgets import QApplication
        from PyQt6 import QtCore

        # Get version
        version = QtCore.PYQT_VERSION_STR
        print_success(f"PyQt6 {version} installed successfully")

        # Test basic functionality (create QApplication without showing GUI)
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        print_info("  PyQt6 QApplication created successfully")

        return True
    except ImportError as e:
        print_error(f"PyQt6 import failed: {e}")
        return False
    except Exception as e:
        print_error(f"PyQt6 functionality test failed: {e}")
        return False


def verify_luaparser():
    """Verify luaparser installation and functionality."""
    try:
        import luaparser
        from luaparser import ast

        # Get version
        version = getattr(luaparser, '__version__', 'unknown')
        print_success(f"luaparser {version} installed successfully")

        # Test basic functionality with simple Lua code
        test_code = "local x = 1"
        try:
            parsed_ast = ast.parse(test_code)
            print_info(f"  luaparser successfully parsed test code: '{test_code}'")
        except Exception as e:
            print_error(f"  luaparser parsing test failed: {e}")
            return False

        return True
    except ImportError as e:
        print_error(f"luaparser import failed: {e}")
        return False
    except Exception as e:
        print_error(f"luaparser functionality test failed: {e}")
        return False


def verify_pathlib():
    """Verify pathlib availability (standard library)."""
    try:
        from pathlib import Path
        
        print_success("pathlib available (standard library)")
        
        # Test basic functionality
        test_path = Path(__file__)
        if test_path.exists():
            print_info(f"  pathlib successfully resolved current file: {test_path.name}")
        
        return True
    except ImportError as e:
        print_error(f"pathlib import failed: {e}")
        return False
    except Exception as e:
        print_error(f"pathlib functionality test failed: {e}")
        return False


def main():
    """Main verification function."""
    print_header("=" * 60)
    print_header("Python Obfuscator - Dependency Verification")
    print_header("=" * 60)
    
    print_info(f"Python version: {sys.version}")
    print_info(f"Python executable: {sys.executable}")
    
    print_header("\nVerifying Core Dependencies:")
    
    results = []
    
    # Verify each dependency
    results.append(("PyQt6", verify_pyqt6()))
    results.append(("luaparser", verify_luaparser()))
    results.append(("pathlib", verify_pathlib()))
    
    # Summary
    print_header("\nVerification Summary:")
    print_header("-" * 60)
    
    all_passed = all(result[1] for result in results)
    
    for name, passed in results:
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {name:20s} [{status}]")
    
    print_header("-" * 60)
    
    if all_passed:
        print_success("\n✓ All dependencies verified successfully!")
        return 0
    else:
        print_error("\n✗ Some dependencies failed verification.")
        print_info("\nTo install missing dependencies, run:")
        print_info("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

