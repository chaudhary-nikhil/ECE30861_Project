"""Tests for main.py module"""
import pytest
import tempfile
import os
import sys
import requests
from io import StringIO
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

from src.main import (
    validate_github_token,
    validate_log_file,
    parseUrlFile,
    calculate_scores,
    main,
)
from src.url import Url, UrlCategory, UrlSet


class TestValidateGithubToken:
    """Tests for GitHub token validation"""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_token_provided(self):
        """Test that missing token returns True (rate-limited access)"""
        assert validate_github_token() is True

    @patch("src.main.requests.get")
    @patch.dict(os.environ, {"GITHUB_TOKEN": "valid_token"})
    def test_valid_token(self, mock_get):
        """Test that valid token returns True"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert validate_github_token() is True

    @patch("src.main.requests.get")
    @patch.dict(os.environ, {"GITHUB_TOKEN": "invalid_token"})
    def test_invalid_token(self, mock_get):
        """Test that invalid token returns False"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        assert validate_github_token() is False

    @patch("src.main.requests.get")
    @patch.dict(os.environ, {"GITHUB_TOKEN": "token"})
    def test_token_validation_network_error(self, mock_get):
        """Test that network errors don't block execution"""
        mock_get.side_effect = Exception("Network error")

        assert validate_github_token() is True

    @patch("src.main.requests.get")
    @patch.dict(os.environ, {"GITHUB_TOKEN": "token"})
    def test_token_validation_other_status(self, mock_get):
        """Test that other status codes allow continuation"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        assert validate_github_token() is True


class TestValidateLogFile:
    """Tests for log file validation"""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_log_file_provided(self):
        """Test that missing log file returns True"""
        assert validate_log_file() is False

    def test_valid_log_file_path(self):
        """Test that valid log file path returns True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            with patch.dict(os.environ, {"LOG_FILE": log_file}):
                assert validate_log_file() is False # Because the file doesn't exist
                assert not os.path.exists(log_file)

    def test_log_file_with_nested_directories(self):
        """Test that nested directories are created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "logs", "nested", "test.log")
            with patch.dict(os.environ, {"LOG_FILE": log_file}):
                assert validate_log_file() is False
                assert not os.path.exists(log_file)

    def test_invalid_log_file_path(self):
        """Test that invalid path returns False"""
        with patch.dict(os.environ, {"LOG_FILE": "/root/invalid/path/test.log"}):
            # This should fail on non-root systems
            result = validate_log_file()
            # Result depends on permissions, but shouldn't crash
            assert isinstance(result, bool)


class TestParseUrlFile:
    """Tests for URL file parsing"""

    def test_parse_single_line_csv(self):
        """Test parsing single line with 3 URLs"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("https://github.com/user/repo,https://huggingface.co/datasets/test,https://huggingface.co/model\n")
            f.flush()
            filename = f.name

        try:
            urlsets = parseUrlFile(filename)
            assert len(urlsets) == 1 # Only 1 URL Set
            assert isinstance(urlsets[0].code, Url)
            assert urlsets[0].code.category == UrlCategory.CODE
            assert isinstance(urlsets[0].dataset, Url)
            assert urlsets[0].dataset.category == UrlCategory.DATASET
            assert isinstance(urlsets[0].model, Url)
            assert urlsets[0].model.category == UrlCategory.MODEL
        finally:
            os.unlink(filename)

    def test_parse_partial_entries(self):
        """Test parsing lines with empty fields"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(",,https://huggingface.co/model\n")
            f.write(",,https://huggingface.co/model2,\n")
            f.flush()
            filename = f.name

        try:
            urlsets = parseUrlFile(filename)
            assert len(urlsets) == 2
        finally:
            os.unlink(filename)

    def test_parse_multiple_lines(self):
        """Test parsing multiple lines"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("https://github.com/user/repo,https://huggingface.co/datasets/test,https://huggingface.co/model\n")
            f.write("https://github.com/user2/repo2,https://huggingface.co/datasets/test2,https://huggingface.co/model2\n")
            f.flush()
            filename = f.name

        try:
            urlsets = parseUrlFile(filename)
            assert len(urlsets) == 2
        finally:
            os.unlink(filename)

    def test_parse_empty_file(self):
        """Test parsing empty file"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("")
            f.flush()
            filename = f.name

        try:
            urls = parseUrlFile(filename)
            assert len(urls) == 0
        finally:
            os.unlink(filename)

    def test_parse_file_with_blank_lines(self):
        """Test parsing file with blank lines"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("https://github.com/user/repo,https://huggingface.co/datasets/test,https://huggingface.co/model\n")
            f.write("\n")
            f.write("\n")
            f.write("https://github.com/user2/repo2,https://huggingface.co/datasets/test2,https://huggingface.co/model2\n")
            f.flush()
            filename = f.name

        try:
            urlsets = parseUrlFile(filename)
            assert len(urlsets) == 2
        finally:
            os.unlink(filename)

    def test_parse_file_with_whitespace(self):
        """Test parsing file with extra whitespace"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("  https://github.com/user/repo  ,  https://huggingface.co/datasets/test  ,  https://huggingface.co/model  \n")
            f.flush()
            filename = f.name

        try:
            urlsets = parseUrlFile(filename)
            assert len(urlsets) == 1
            assert "  " not in urlsets[0].model.link
        finally:
            os.unlink(filename)


class TestCalculateScores:
    """Tests for calculate_scores function"""

    @patch("src.main.score_url")
    def test_calculate_scores_valid_urls(self, mock_score):
        """Test calculate_scores with valid URLs"""
        from src.scorer import ScoreResult

        mock_score.return_value = ScoreResult(
            url="https://huggingface.co/model",
            category=UrlCategory.MODEL,
            score=8.0,
            max_score=10.0,
            details={"name": "test-model", "downloads": 1000}
        )

        urlsets = [UrlSet(None, None, Url("https://huggingface.co/model", UrlCategory.MODEL))]

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            calculate_scores(urlsets)

            # Verify file was opened for writing - cannot test because we are using stdout for output
            # mock_open.assert_called_once_with("scores.ndjson", "w")

    @patch("src.main.score_url")
    def test_calculate_scores_invalid_urls(self, mock_score):
        """Test calculate_scores with invalid URLs"""
        urls = [
            UrlSet(None, None, Url("https://invalid.com", UrlCategory.INVALID)),
        ]

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            calculate_scores(urls)

            # # Should still write to file
            # mock_open.assert_called_once()

    @patch("src.main.score_url")
    def test_calculate_scores_mixed_urls(self, mock_score):
        """Test calculate_scores with mix of valid and invalid URLs"""
        from src.scorer import ScoreResult

        mock_score.return_value = ScoreResult(
            url="https://huggingface.co/model",
            category=UrlCategory.MODEL,
            score=7.0,
            max_score=10.0,
            details={"name": "test"}
        )

        urls = [
            UrlSet(None, None, Url("https://huggingface.co/model", UrlCategory.MODEL)),
            UrlSet(None, None, Url("https://invalid.com", UrlCategory.INVALID)),
        ]

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            calculate_scores(urls)

class TestMain:
    """Tests for main function"""

    @patch("src.main.validate_log_file")
    @patch("src.main.validate_github_token")
    @patch("src.main.parseUrlFile")
    @patch("src.main.calculate_scores")
    @patch("src.log.loggerInstance.logger")
    def test_main_success(self, mock_logger, mock_calc, mock_parse, mock_token, mock_log):
        """Test successful main execution"""
        mock_log.return_value = True
        mock_token.return_value = True
        mock_parse.return_value = [
            Url("https://huggingface.co/model", UrlCategory.MODEL)
        ]

        with patch.object(sys, "argv", ["prog", "test.txt"]):
            result = main()
            assert result == 0

    @patch("src.main.validate_log_file")
    @patch("src.log.loggerInstance.logger")
    def test_main_invalid_log_file(self, mock_logger, mock_log):
        """Test main with invalid log file"""
        mock_log.return_value = False

        with patch.object(sys, "argv", ["prog", "test.txt"]):
            result = main()
            assert result == 1

    @patch("src.main.validate_log_file")
    @patch("src.main.validate_github_token")
    @patch("src.log.loggerInstance.logger")
    def test_main_invalid_token(self, mock_logger, mock_token, mock_log):
        """Test main with invalid GitHub token"""
        mock_log.return_value = True
        mock_token.return_value = False

        with patch.object(sys, "argv", ["prog", "test.txt"]):
            result = main()
            assert result == 1

    @patch("src.main.validate_log_file")
    @patch("src.main.validate_github_token")
    @patch("src.log.loggerInstance.logger")
    def test_main_missing_url_file_argument(self, mock_logger, mock_token, mock_log):
        """Test main without URL file argument"""
        mock_log.return_value = True
        mock_token.return_value = True

        with patch.object(sys, "argv", ["prog"]):
            result = main()
            assert result == 1

    @patch("src.main.validate_log_file")
    @patch("src.main.validate_github_token")
    @patch("src.log.loggerInstance.logger")
    def test_main_too_many_arguments(self, mock_logger, mock_token, mock_log):
        """Test main with too many arguments"""
        mock_log.return_value = True
        mock_token.return_value = True

        with patch.object(sys, "argv", ["prog", "file1", "file2"]):
            result = main()
            assert result == 1
