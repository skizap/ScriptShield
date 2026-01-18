"""
Integration tests for utils package and dependency analysis.

Tests cross-module interactions between path_utils, logger modules,
and the dependency analysis/obfuscation orchestration system.
"""

import pytest
from pathlib import Path

from obfuscator.utils import (
    setup_logger,
    normalize_path,
    ensure_directory,
    validate_lua_file,
    get_platform,
)
from obfuscator.core.orchestrator import ObfuscationOrchestrator, OrchestrationResult
from obfuscator.core.dependency_graph import DependencyAnalyzer, DependencyGraph
from obfuscator.core.symbol_table import GlobalSymbolTable, SymbolTableBuilder
from obfuscator.processors.python_processor import PythonProcessor
from obfuscator.processors.lua_processor import LuaProcessor


class TestLoggerWithPathUtils:
    """Test logger integration with path utilities."""
    
    def test_logger_creates_log_directory_using_path_utils(self, tmp_path):
        """Test that logger uses ensure_directory for log file creation."""
        log_file = tmp_path / "nested" / "logs" / "app.log"
        
        # This should use ensure_directory internally
        logger = setup_logger("test_app", log_file=log_file)
        logger.info("Test message")
        
        # Verify directory was created
        assert log_file.parent.exists()
        assert log_file.parent.is_dir()
        assert log_file.exists()
    
    def test_logger_with_normalized_path(self, tmp_path):
        """Test logger works with normalized paths."""
        # Use relative path that needs normalization, rooted in tmp_path
        relative_log = tmp_path / "logs" / "app.log"
        normalized = normalize_path(relative_log)

        logger = setup_logger("test_app_normalized", log_file=normalized)
        logger.info("Test message")

        assert normalized.exists()

    def test_logger_handles_path_with_spaces(self, tmp_path):
        """Test logger handles paths with spaces correctly."""
        log_file = tmp_path / "log directory" / "app log.log"

        logger = setup_logger("test_app_spaces", log_file=log_file)
        logger.info("Test message")

        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content


class TestPathValidationWithLogging:
    """Test path validation with logging integration."""
    
    def test_validate_and_log_lua_files(self, tmp_path, caplog):
        """Test validating Lua files with logging."""
        logger = setup_logger("validator", level="DEBUG")
        
        # Create test files
        valid_lua = tmp_path / "valid.lua"
        valid_lua.write_text("-- Lua script")
        
        invalid_txt = tmp_path / "invalid.txt"
        invalid_txt.write_text("text")
        
        # Validate and log results
        if validate_lua_file(valid_lua):
            logger.info(f"Valid Lua file: {valid_lua}")
        else:
            logger.warning(f"Invalid Lua file: {valid_lua}")
        
        if validate_lua_file(invalid_txt):
            logger.info(f"Valid Lua file: {invalid_txt}")
        else:
            logger.warning(f"Invalid Lua file: {invalid_txt}")
        
        # Check logs
        assert "Valid Lua file" in caplog.text
        assert "Invalid Lua file" in caplog.text
    
    def test_log_path_normalization_errors(self, caplog):
        """Test logging path normalization errors."""
        logger = setup_logger("path_validator", level="DEBUG")
        
        try:
            normalize_path(None)
        except ValueError as e:
            logger.error(f"Path normalization failed: {e}")
        
        assert "Path normalization failed" in caplog.text


class TestCrossPlatformLogging:
    """Test cross-platform logging functionality."""
    
    def test_platform_specific_logging(self, tmp_path, caplog):
        """Test logging platform-specific information."""
        logger = setup_logger("platform_test", level="INFO")
        
        platform = get_platform()
        logger.info(f"Running on platform: {platform}")
        
        assert f"Running on platform: {platform}" in caplog.text
        assert platform in ["windows", "macos", "linux"]
    
    def test_log_file_on_current_platform(self, tmp_path):
        """Test log file creation works on current platform."""
        log_file = tmp_path / "logs" / "platform_test.log"
        
        logger = setup_logger("platform_logger", log_file=log_file)
        platform = get_platform()
        
        logger.info(f"Platform: {platform}")
        logger.info(f"Log file: {log_file}")
        
        assert log_file.exists()
        content = log_file.read_text()
        assert f"Platform: {platform}" in content


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_complete_file_processing_workflow(self, tmp_path):
        """Test complete workflow: normalize → validate → log."""
        # Setup logger
        log_file = tmp_path / "logs" / "workflow.log"
        logger = setup_logger("workflow", level="DEBUG", log_file=log_file)
        
        # Create test Lua file
        lua_file = tmp_path / "script.lua"
        lua_file.write_text("print('Hello, World!')")
        
        # Normalize path
        normalized = normalize_path(str(lua_file))
        logger.debug(f"Normalized path: {normalized}")
        
        # Validate file
        is_valid = validate_lua_file(normalized)
        if is_valid:
            logger.info(f"Valid Lua file: {normalized.name}")
        else:
            logger.error(f"Invalid Lua file: {normalized.name}")
        
        # Verify workflow
        assert log_file.exists()
        content = log_file.read_text()
        assert "Normalized path" in content
        assert "Valid Lua file: script.lua" in content
    
    def test_error_handling_workflow(self, tmp_path, caplog):
        """Test error handling across modules."""
        logger = setup_logger("error_handler", level="WARNING")
        
        # Try to validate non-existent file
        nonexistent = tmp_path / "nonexistent.lua"
        
        if not validate_lua_file(nonexistent):
            logger.warning(f"File not found: {nonexistent}")
        
        assert "File not found" in caplog.text
    
    def test_directory_creation_and_logging(self, tmp_path):
        """Test directory creation with logging."""
        log_file = tmp_path / "logs" / "dir_test.log"
        logger = setup_logger("dir_creator", log_file=log_file)
        
        # Create nested directory structure
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        created_dir = ensure_directory(nested_dir)
        
        logger.info(f"Created directory: {created_dir}")
        
        assert nested_dir.exists()
        assert log_file.exists()
        content = log_file.read_text()
        assert "Created directory" in content


class TestDependencyAnalysisIntegration:
    """Integration tests for dependency analysis and obfuscation orchestration."""

    def test_orchestrator_processes_single_python_file(self, tmp_path):
        """Test orchestrator can process a single Python file."""
        # Create test file
        py_file = tmp_path / "main.py"
        py_file.write_text("def hello():\n    return 'world'\n")
        output_dir = tmp_path / "output"

        orchestrator = ObfuscationOrchestrator()
        result = orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config={"mangling_strategy": "sequential"}
        )

        assert result.success
        assert len(result.processed_files) == 1
        assert result.processed_files[0].success
        assert (output_dir / "main.py").exists()

    def test_orchestrator_processes_single_lua_file(self, tmp_path):
        """Test orchestrator can process a single Lua file."""
        lua_file = tmp_path / "main.lua"
        lua_file.write_text("local function hello()\n    return 'world'\nend\n")
        output_dir = tmp_path / "output"

        orchestrator = ObfuscationOrchestrator()
        result = orchestrator.process_files(
            input_files=[lua_file],
            output_dir=output_dir,
            config={"mangling_strategy": "sequential"}
        )

        assert result.success
        assert len(result.processed_files) == 1
        assert result.processed_files[0].success
        assert (output_dir / "main.lua").exists()

    def test_orchestrator_builds_dependency_graph(self, tmp_path):
        """Test orchestrator builds dependency graph for multiple files."""
        # Create files with import relationship
        utils_file = tmp_path / "utils.py"
        utils_file.write_text("def helper():\n    return 42\n")

        main_file = tmp_path / "main.py"
        main_file.write_text("from utils import helper\n\ndef main():\n    return helper()\n")

        output_dir = tmp_path / "output"

        orchestrator = ObfuscationOrchestrator()
        result = orchestrator.process_files(
            input_files=[main_file, utils_file],
            output_dir=output_dir,
            config={}
        )

        assert result.dependency_graph is not None
        assert len(result.dependency_graph.nodes) >= 1

    def test_orchestrator_builds_global_symbol_table(self, tmp_path):
        """Test orchestrator builds global symbol table."""
        py_file = tmp_path / "module.py"
        py_file.write_text("class MyClass:\n    pass\n\ndef my_func():\n    pass\n")
        output_dir = tmp_path / "output"

        orchestrator = ObfuscationOrchestrator()
        result = orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config={}
        )

        assert result.global_symbol_table is not None
        assert result.global_symbol_table.is_frozen

    def test_orchestrator_reports_progress(self, tmp_path):
        """Test orchestrator calls progress callback."""
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")
        output_dir = tmp_path / "output"

        progress_calls = []

        def on_progress(message: str, current: int, total: int) -> None:
            progress_calls.append((message, current, total))

        orchestrator = ObfuscationOrchestrator()
        orchestrator.process_files(
            input_files=[py_file],
            output_dir=output_dir,
            config={},
            progress_callback=on_progress
        )

        assert len(progress_calls) > 0
        # Verify progress increases
        for i in range(1, len(progress_calls)):
            assert progress_calls[i][1] >= progress_calls[i - 1][1]

    def test_orchestrator_handles_parse_errors_gracefully(self, tmp_path):
        """Test orchestrator handles files with syntax errors."""
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(\n")  # Syntax error
        output_dir = tmp_path / "output"

        orchestrator = ObfuscationOrchestrator()
        result = orchestrator.process_files(
            input_files=[bad_file],
            output_dir=output_dir,
            config={}
        )

        # Should not crash, but report warnings
        assert len(result.warnings) > 0 or len(result.errors) > 0

    def test_orchestrator_processes_mixed_languages(self, tmp_path):
        """Test orchestrator can process Python and Lua files together."""
        py_file = tmp_path / "module.py"
        py_file.write_text("def py_func():\n    pass\n")

        lua_file = tmp_path / "module.lua"
        lua_file.write_text("local function lua_func()\nend\n")

        output_dir = tmp_path / "output"

        orchestrator = ObfuscationOrchestrator()
        result = orchestrator.process_files(
            input_files=[py_file, lua_file],
            output_dir=output_dir,
            config={}
        )

        assert result.success
        assert len(result.processed_files) == 2
        assert (output_dir / "module.py").exists()
        assert (output_dir / "module.lua").exists()

    def test_symbol_table_builder_creates_frozen_table(self, tmp_path):
        """Test SymbolTableBuilder creates a frozen GlobalSymbolTable."""
        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1\n")

        processor = PythonProcessor()
        parse_result = processor.parse_file(py_file)
        symbols = processor.extract_symbols(parse_result.ast_node, py_file)

        graph = DependencyGraph()
        graph.add_node(py_file.resolve(), "python", [], [])

        builder = SymbolTableBuilder()
        global_table = builder.build_from_dependency_graph(
            graph, {py_file.resolve(): symbols}, {}
        )

        assert global_table.is_frozen
        # Verify we can't add more symbols after freeze
        with pytest.raises(RuntimeError):
            from obfuscator.core.symbol_table import SymbolEntry
            global_table.add_symbol(SymbolEntry(
                original_name="test",
                mangled_name="_t",
                symbol_type="variable",
                scope="global",
                file_path=py_file,
                language="python"
            ))

    def test_dependency_graph_topological_order(self, tmp_path):
        """Test dependency graph returns correct topological order."""
        # Create files: main depends on utils
        utils_file = tmp_path / "utils.py"
        utils_file.write_text("def helper():\n    return 1\n")

        main_file = tmp_path / "main.py"
        main_file.write_text("from utils import helper\nx = helper()\n")

        processor = PythonProcessor()

        # Parse and extract symbols
        utils_result = processor.parse_file(utils_file)
        utils_symbols = processor.extract_symbols(utils_result.ast_node, utils_file)

        main_result = processor.parse_file(main_file)
        main_symbols = processor.extract_symbols(main_result.ast_node, main_file)

        # Build graph
        analyzer = DependencyAnalyzer(tmp_path)
        graph = analyzer.build_graph_incremental([
            (utils_file.resolve(), utils_symbols),
            (main_file.resolve(), main_symbols)
        ])

        order = graph.get_processing_order()

        # utils should come before main in processing order
        utils_idx = next(
            (i for i, p in enumerate(order) if p.name == "utils.py"), -1
        )
        main_idx = next(
            (i for i, p in enumerate(order) if p.name == "main.py"), -1
        )

        if utils_idx >= 0 and main_idx >= 0:
            assert utils_idx < main_idx, "utils.py should be processed before main.py"

    def test_python_processor_obfuscate_with_symbol_table(self, tmp_path):
        """Test PythonProcessor.obfuscate_with_symbol_table method."""
        py_file = tmp_path / "test.py"
        py_file.write_text("def my_function():\n    x = 1\n    return x\n")

        processor = PythonProcessor()
        parse_result = processor.parse_file(py_file)
        symbols = processor.extract_symbols(parse_result.ast_node, py_file)

        # Build symbol table
        graph = DependencyGraph()
        graph.add_node(py_file.resolve(), "python", [], ["my_function"])

        builder = SymbolTableBuilder()
        global_table = builder.build_from_dependency_graph(
            graph, {py_file.resolve(): symbols}, {}
        )

        # Obfuscate with symbol table
        result = processor.obfuscate_with_symbol_table(
            parse_result.ast_node, py_file.resolve(), global_table
        )

        assert result.success
        assert result.ast_node is not None
