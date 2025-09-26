import pytest
import tempfile
import shutil
from unittest.mock import patch
from src.log.logger import Logger
import os


@pytest.fixture
def temp_log_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def clean_env():
    original_log_file = os.environ.get("LOG_FILE")
    original_log_level = os.environ.get("LOG_LEVEL")

    if "LOG_FILE" in os.environ:
        del os.environ["LOG_FILE"]
    if "LOG_LEVEL" in os.environ:
        del os.environ["LOG_LEVEL"]

    yield

    if original_log_file is not None:
        os.environ["LOG_FILE"] = original_log_file
    if original_log_level is not None:
        os.environ["LOG_LEVEL"] = original_log_level


class TestLogger:
    def test_default_configuration(self, clean_env):
        logger = Logger()
        config = logger.get_config()

        assert config["log_file"] is None
        assert config["log_level"] == 0
        assert config["log_level_name"] == "SILENT"

    def test_environment_variable_reading(self, temp_log_dir, clean_env):
        """Test that logger correctly reads environment variables"""
        test_log_file = os.path.join(temp_log_dir, "test.log")
        os.environ["LOG_FILE"] = test_log_file
        os.environ["LOG_LEVEL"] = "2"

        logger = Logger()
        config = logger.get_config()

        assert config["log_file"] == test_log_file
        assert config["log_level"] == 2
        assert config["log_level_name"] == "DEBUG"

    @pytest.mark.parametrize("invalid_level", ["invalid", "3", "-1", "10", "abc"])
    def test_invalid_log_level(self, invalid_level, clean_env):
        """Test that invalid log levels default to 0"""
        os.environ["LOG_LEVEL"] = invalid_level
        logger = Logger()
        assert logger.log_level == 0

    def test_silent_level_logging(self, temp_log_dir, clean_env):
        """Test that silent level (0) produces no output"""
        test_log_file = os.path.join(temp_log_dir, "test.log")
        os.environ["LOG_FILE"] = test_log_file
        os.environ["LOG_LEVEL"] = "0"

        logger = Logger()
        logger.log_info("This should not appear")
        logger.log_debug("This should not appear")

        # Log file should not exist or be empty
        assert not os.path.exists(test_log_file)

    def test_info_level_logging(self, temp_log_dir, clean_env):
        """Test that info level (1) logs info messages but not debug"""
        test_log_file = os.path.join(temp_log_dir, "test.log")
        os.environ["LOG_FILE"] = test_log_file
        os.environ["LOG_LEVEL"] = "1"

        logger = Logger()
        logger.log_info("Info message")
        logger.log_debug("Debug message")

        # Read log file content
        with open(test_log_file, "r") as f:
            content = f.read()

        assert "INFO: Info message" in content
        assert "DEBUG: Debug message" not in content

    def test_debug_level_logging(self, temp_log_dir, clean_env):
        """Test that debug level (2) logs both info and debug messages"""
        test_log_file = os.path.join(temp_log_dir, "test.log")
        os.environ["LOG_FILE"] = test_log_file
        os.environ["LOG_LEVEL"] = "2"

        logger = Logger()
        logger.log_info("Info message")
        logger.log_debug("Debug message")

        # Read log file content
        with open(test_log_file, "r") as f:
            content = f.read()

        assert "INFO: Info message" in content
        assert "DEBUG: Debug message" in content

    def test_log_file_creation(self, temp_log_dir, clean_env):
        """Test that log file and directories are created automatically"""
        nested_log_path = os.path.join(temp_log_dir, "logs", "nested", "app.log")
        os.environ["LOG_FILE"] = nested_log_path
        os.environ["LOG_LEVEL"] = "1"

        logger = Logger()
        logger.log_info("Test message")

        assert os.path.exists(nested_log_path)
        assert os.path.isfile(nested_log_path)

    def test_log_appending(self, temp_log_dir, clean_env):
        """Test that new log entries are appended to existing file"""
        test_log_file = os.path.join(temp_log_dir, "test.log")
        os.environ["LOG_FILE"] = test_log_file
        os.environ["LOG_LEVEL"] = "1"

        # First logger instance
        logger1 = Logger()
        logger1.log_info("First message")

        # Second logger instance
        logger2 = Logger()
        logger2.log_info("Second message")

        with open(test_log_file, "r") as f:
            content = f.read()

        assert "First message" in content
        assert "Second message" in content
        assert content.count("INFO:") == 2

    def test_timestamp_format(self, temp_log_dir, clean_env):
        """Test that log entries include properly formatted timestamps"""
        test_log_file = os.path.join(temp_log_dir, "test.log")
        os.environ["LOG_FILE"] = test_log_file
        os.environ["LOG_LEVEL"] = "1"

        logger = Logger()
        logger.log_info("Test message")

        with open(test_log_file, "r") as f:
            content = f.read()

        # Check timestamp format: [YYYY-MM-DD HH:MM:SS]
        import re

        timestamp_pattern = r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]"
        assert re.search(timestamp_pattern, content)

    def test_no_log_file_specified(self, clean_env):
        """Test that no logging occurs when LOG_FILE is not set"""
        os.environ["LOG_LEVEL"] = "2"
        # LOG_FILE not set

        logger = Logger()
        # These should not raise exceptions
        logger.log_info("This should be ignored")
        logger.log_debug("This should also be ignored")

        # No assertions needed - just ensuring no exceptions

    def test_file_write_error_handling(self, clean_env):
        """Test that file write errors are handled gracefully"""
        # Try to write to a directory that doesn't exist and can't be created
        os.environ["LOG_FILE"] = "/root/cannot/create/this/path/log.txt"
        os.environ["LOG_LEVEL"] = "1"

        logger = Logger()
        # This should not raise an exception
        logger.log_info("This should fail silently")

    @pytest.mark.parametrize(
        "log_level,expected_info,expected_debug",
        [
            (0, False, False),  # Silent
            (1, True, False),  # Info only
            (2, True, True),  # Info and Debug
        ],
    )
    def test_logging_levels_comprehensive(
        self, temp_log_dir, clean_env, log_level, expected_info, expected_debug
    ):
        """Comprehensive test of all logging levels"""
        test_log_file = os.path.join(temp_log_dir, "test.log")
        os.environ["LOG_FILE"] = test_log_file
        os.environ["LOG_LEVEL"] = str(log_level)

        logger = Logger()
        logger.log_info("Info message")
        logger.log_debug("Debug message")

        if expected_info or expected_debug:
            assert os.path.exists(test_log_file)
            with open(test_log_file, "r") as f:
                content = f.read()

            if expected_info:
                assert "INFO: Info message" in content
            else:
                assert "INFO: Info message" not in content

            if expected_debug:
                assert "DEBUG: Debug message" in content
            else:
                assert "DEBUG: Debug message" not in content
        else:
            assert not os.path.exists(test_log_file)
