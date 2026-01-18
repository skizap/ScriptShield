"""
Comprehensive tests for logger module.

Tests cover logger setup, handler management, log levels,
formatting, and file operations.
"""

import logging
import pytest
from pathlib import Path

from obfuscator.utils.logger import (
    setup_logger,
    get_logger,
    set_log_level,
    add_file_handler,
    add_console_handler,
    get_log_directory,
    VALID_LOG_LEVELS,
)


@pytest.fixture
def tmp_log_dir(tmp_path):
    """Create temporary directory for log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def log_file_path(tmp_log_dir):
    """Create path to temporary log file."""
    return tmp_log_dir / "test.log"


@pytest.fixture
def test_logger():
    """Create fresh logger instance for each test."""
    logger_name = "test_logger"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    yield logger
    # Cleanup
    logger.handlers.clear()


class TestLoggerSetup:
    """Test logger creation and configuration."""
    
    def test_setup_logger_creates_logger(self):
        """Test that setup_logger creates a logger instance."""
        logger = setup_logger("test_app")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_app"
    
    def test_setup_logger_default_level(self):
        """Test that default log level is INFO."""
        logger = setup_logger("test_app")
        assert logger.level == logging.INFO
    
    def test_setup_logger_custom_level(self):
        """Test that custom log level is set correctly."""
        logger = setup_logger("test_app", level="DEBUG")
        assert logger.level == logging.DEBUG
    
    def test_setup_logger_invalid_level(self):
        """Test that invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logger("test_app", level="INVALID")
    
    def test_setup_logger_with_file(self, log_file_path):
        """Test logger creation with log file."""
        logger = setup_logger("test_app", log_file=log_file_path)
        
        # Should have console and file handlers
        assert len(logger.handlers) == 2
        
        # Check file was created
        logger.info("Test message")
        assert log_file_path.exists()
    
    def test_setup_logger_prevents_duplicate_handlers(self):
        """Test that calling setup_logger twice doesn't duplicate handlers."""
        logger_name = "test_duplicate"
        
        logger1 = setup_logger(logger_name)
        handler_count_1 = len(logger1.handlers)
        
        logger2 = setup_logger(logger_name)
        handler_count_2 = len(logger2.handlers)
        
        assert handler_count_1 == handler_count_2
        assert logger1 is logger2
    
    def test_get_logger_creates_default_logger(self):
        """Test get_logger creates logger with defaults if not exists."""
        logger = get_logger("new_logger")
        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) > 0
    
    def test_get_logger_retrieves_existing_logger(self):
        """Test get_logger retrieves existing logger."""
        logger1 = setup_logger("existing_logger")
        logger2 = get_logger("existing_logger")
        assert logger1 is logger2


class TestLogLevels:
    """Test log level configuration and filtering."""
    
    def test_set_log_level_changes_level(self, test_logger):
        """Test set_log_level changes logger level."""
        set_log_level(test_logger, "WARNING")
        assert test_logger.level == logging.WARNING
    
    def test_set_log_level_invalid_raises_error(self, test_logger):
        """Test set_log_level raises ValueError for invalid level."""
        with pytest.raises(ValueError, match="Invalid log level"):
            set_log_level(test_logger, "INVALID")
    
    def test_log_level_filtering(self, test_logger, caplog):
        """Test that log messages are filtered by level."""
        add_console_handler(test_logger, "WARNING")
        
        with caplog.at_level(logging.DEBUG, logger=test_logger.name):
            test_logger.debug("Debug message")
            test_logger.info("Info message")
            test_logger.warning("Warning message")
        
        # Only WARNING and above should be captured
        assert "Debug message" not in caplog.text
        assert "Info message" not in caplog.text
        assert "Warning message" in caplog.text
    
    def test_all_valid_log_levels(self, test_logger):
        """Test all valid log levels can be set."""
        for level in VALID_LOG_LEVELS:
            set_log_level(test_logger, level)
            assert test_logger.level == getattr(logging, level)


class TestLogHandlers:
    """Test console and file handler management."""
    
    def test_add_console_handler(self, test_logger):
        """Test adding console handler."""
        add_console_handler(test_logger, "INFO")
        
        assert len(test_logger.handlers) == 1
        assert isinstance(test_logger.handlers[0], logging.StreamHandler)
    
    def test_add_file_handler_creates_file(self, test_logger, log_file_path):
        """Test adding file handler creates log file."""
        add_file_handler(test_logger, log_file_path)
        
        test_logger.info("Test message")
        
        assert log_file_path.exists()
        content = log_file_path.read_text()
        assert "Test message" in content
    
    def test_add_file_handler_creates_directory(self, test_logger, tmp_path):
        """Test file handler creates log directory if missing."""
        log_file = tmp_path / "nested" / "logs" / "app.log"
        
        add_file_handler(test_logger, log_file)
        test_logger.info("Test")
        
        assert log_file.parent.exists()
        assert log_file.exists()
    
    def test_add_file_handler_invalid_level(self, test_logger, log_file_path):
        """Test add_file_handler raises error for invalid level."""
        with pytest.raises(ValueError, match="Invalid log level"):
            add_file_handler(test_logger, log_file_path, level="INVALID")

    def test_add_console_handler_invalid_level(self, test_logger):
        """Test add_console_handler raises error for invalid level."""
        with pytest.raises(ValueError, match="Invalid log level"):
            add_console_handler(test_logger, level="INVALID")


class TestLogFormatting:
    """Test log message formatting."""

    def test_console_format_simple(self, test_logger, caplog):
        """Test console logs use simple format."""
        add_console_handler(test_logger, "INFO")

        with caplog.at_level(logging.INFO, logger=test_logger.name):
            test_logger.info("Test message")

        # Simple format: "LEVEL: message"
        assert "INFO: Test message" in caplog.text

    def test_file_format_detailed(self, test_logger, log_file_path):
        """Test file logs use detailed format."""
        add_file_handler(test_logger, log_file_path, "DEBUG")

        test_logger.debug("Debug message")

        content = log_file_path.read_text()
        # Detailed format includes timestamp, name, level, file, line, message
        assert "test_logger" in content
        assert "DEBUG" in content
        assert "test_logger.py" in content
        assert "Debug message" in content

    def test_log_file_encoding_utf8(self, test_logger, log_file_path):
        """Test log files use UTF-8 encoding."""
        add_file_handler(test_logger, log_file_path)

        # Log message with unicode characters
        test_logger.info("Unicode test: 你好 мир 🎉")

        content = log_file_path.read_text(encoding='utf-8')
        assert "Unicode test: 你好 мир 🎉" in content


class TestLogFileManagement:
    """Test log file creation and rotation."""

    def test_get_log_directory(self):
        """Test get_log_directory returns correct path."""
        log_dir = get_log_directory()
        assert log_dir == Path("logs")

    def test_log_rotation_configuration(self, test_logger, log_file_path):
        """Test that rotating file handler is configured correctly."""
        add_file_handler(test_logger, log_file_path)

        # Find the rotating file handler
        rotating_handler = None
        for handler in test_logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                rotating_handler = handler
                break

        assert rotating_handler is not None
        assert rotating_handler.maxBytes == 10 * 1024 * 1024  # 10MB
        assert rotating_handler.backupCount == 5

    def test_multiple_handlers_different_levels(self, test_logger, log_file_path):
        """Test logger with console and file handlers at different levels."""
        add_console_handler(test_logger, "WARNING")
        add_file_handler(test_logger, log_file_path, "DEBUG")

        test_logger.debug("Debug message")
        test_logger.warning("Warning message")

        # File should have both messages
        content = log_file_path.read_text()
        assert "Debug message" in content
        assert "Warning message" in content


class TestLoggerIntegration:
    """Integration tests for logger functionality."""

    def test_logger_hierarchy(self):
        """Test logger hierarchy works correctly."""
        parent_logger = setup_logger("obfuscator")
        child_logger = setup_logger("obfuscator.utils")

        assert child_logger.name.startswith(parent_logger.name)

    def test_end_to_end_logging(self, tmp_path):
        """Test complete logging workflow."""
        log_file = tmp_path / "logs" / "app.log"

        # Setup logger with both handlers
        logger = setup_logger("test_app", level="DEBUG", log_file=log_file)

        # Log messages at different levels
        logger.debug("Debug info")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Verify file was created and contains messages
        assert log_file.exists()
        content = log_file.read_text()
        assert "Debug info" in content
        assert "Info message" in content
        assert "Warning message" in content
        assert "Error message" in content

