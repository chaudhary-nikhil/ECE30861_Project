"""Tests for code_quality.py module"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import subprocess
from pathlib import Path

from src.code_quality import (
    run_flake8_on_repo,
    find_code_repo_via_genai,
    calculate_code_quality_with_timing,
    calculate_code_quality
)
from src.log.logger import Logger
from src.log import loggerInstance

loggerInstance.logger = Logger()


class TestRunFlake8OnRepo:
    """Tests for run_flake8_on_repo function"""

    def test_empty_repository(self, tmp_path):
        """Test repository with no Python files"""
        score, latency = run_flake8_on_repo(str(tmp_path))
        assert score == 0.0
        assert latency >= 0

    def test_repository_with_clean_code(self, tmp_path):
        """Test repository with no flake8 errors"""
        py_file = tmp_path / "clean.py"
        py_file.write_text('"""Module docstring."""\n\n\ndef function():\n    """Function docstring."""\n    pass\n')
        
        score, latency = run_flake8_on_repo(str(tmp_path))
        assert score >= 0.8  # Should have high score
        assert latency > 0

    def test_repository_with_errors(self, tmp_path):
        """Test repository with flake8 errors"""
        py_file = tmp_path / "bad.py"
        # Intentional style violations
        py_file.write_text('import sys\nimport os\nx=1\ny=2\nz=3\n')
        
        score, latency = run_flake8_on_repo(str(tmp_path))
        assert 0.0 <= score <= 1.0
        assert latency > 0



    @patch("src.code_quality.subprocess.run")
    def test_flake8_timeout(self, mock_run, tmp_path):
        """Test flake8 timeout scenario"""
        mock_run.side_effect = subprocess.TimeoutExpired("flake8", 30)
        
        py_file = tmp_path / "test.py"
        py_file.write_text("print('hello')\n")
        
        score, latency = run_flake8_on_repo(str(tmp_path))
        assert score == 0.0
        assert latency >= 0

    @patch("src.code_quality.subprocess.run")
    def test_flake8_exception(self, mock_run, tmp_path):
        """Test general exception handling"""
        mock_run.side_effect = Exception("Unexpected error")
        
        py_file = tmp_path / "test.py"
        py_file.write_text("print('hello')\n")
        
        score, latency = run_flake8_on_repo(str(tmp_path))
        assert score == 0.0
        assert latency >= 0

    @patch("src.code_quality.subprocess.run")
    def test_flake8_output_parsing(self, mock_run, tmp_path):
        """Test parsing of flake8 output"""
        mock_result = Mock()
        mock_result.stdout = "150"  # 150 total errors
        mock_run.return_value = mock_result
        
        py_file = tmp_path / "test.py"
        py_file.write_text("print('hello')\n")
        
        score, latency = run_flake8_on_repo(str(tmp_path))
        assert score == 0.6  # 101-200 errors = 0.6
        assert latency >= 0

    @patch("src.code_quality.subprocess.run")
    def test_flake8_malformed_output(self, mock_run, tmp_path):
        """Test handling of malformed flake8 output"""
        mock_result = Mock()
        mock_result.stdout = "invalid output"
        mock_run.return_value = mock_result
        
        py_file = tmp_path / "test.py"
        py_file.write_text("print('hello')\n")
        
        score, latency = run_flake8_on_repo(str(tmp_path))
        assert score == 1.0  # 0 errors (couldn't parse) = 1.0
        assert latency >= 0


class TestFindCodeRepoViaGenai:
    """Tests for find_code_repo_via_genai function"""

    @patch("src.code_quality.requests.post")
    def test_successful_api_response_with_github_url(self, mock_post):
        """Test successful API response with GitHub URL"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'https://github.com/owner/repo'
                }
            }]
        }
        mock_post.return_value = mock_response
        
        result = find_code_repo_via_genai("test/model")
        assert result == "https://github.com/owner/repo"

    @patch("src.code_quality.requests.post")
    def test_successful_api_response_with_gitlab_url(self, mock_post):
        """Test successful API response with GitLab URL"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'The code is available at https://gitlab.com/owner/repo for your reference.'
                }
            }]
        }
        mock_post.return_value = mock_response
        
        result = find_code_repo_via_genai("test/model")
        assert result == "https://gitlab.com/owner/repo"

    @patch("src.code_quality.requests.post")
    def test_no_code_found_response(self, mock_post):
        """Test when API returns NO_CODE_FOUND"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'NO_CODE_FOUND'
                }
            }]
        }
        mock_post.return_value = mock_response
        
        result = find_code_repo_via_genai("test/model")
        assert result is None

    @patch("src.code_quality.requests.post")
    def test_api_error_status_code(self, mock_post):
        """Test API error status code"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response
        
        result = find_code_repo_via_genai("test/model")
        assert result is None

    @patch("src.code_quality.requests.post")
    def test_api_timeout(self, mock_post):
        """Test API timeout"""
        mock_post.side_effect = Exception("Timeout")
        
        result = find_code_repo_via_genai("test/model")
        assert result is None

    
    
    @patch("src.code_quality.requests.post")
    @patch("src.code_quality.requests.get")
    def test_fallback_to_github_url_fails(self, mock_get, mock_post):
        """Test fallback fails when GitHub URL doesn't exist"""
        # API fails
        mock_post.side_effect = Exception("API error")
        
        # GitHub URL doesn't exist
        mock_get_response = Mock()
        mock_get_response.status_code = 404
        mock_get.return_value = mock_get_response
        
        result = find_code_repo_via_genai("owner/repo")
        assert result is None

    @patch("src.code_quality.requests.post")
    def test_empty_model_name(self, mock_post):
        """Test with empty model name"""
        mock_post.side_effect = Exception("Invalid request")
        
        result = find_code_repo_via_genai("")
        assert result is None

    @patch("src.code_quality.requests.post")
    def test_api_response_with_multiple_urls(self, mock_post):
        """Test API response with multiple URLs (returns first)"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Found at https://github.com/first/repo and https://github.com/second/repo'
                }
            }]
        }
        mock_post.return_value = mock_response
        
        result = find_code_repo_via_genai("test/model")
        assert result == "https://github.com/first/repo"


class TestCalculateCodeQualityWithTiming:
    """Tests for calculate_code_quality_with_timing function"""

    @patch("src.code_quality.subprocess.run")
    @patch("src.code_quality.run_flake8_on_repo")
    def test_with_valid_code_url(self, mock_flake8, mock_run):
        """Test with valid code URL provided"""
        mock_run.return_value = Mock()
        mock_flake8.return_value = (0.8, 100)
        
        score, latency = calculate_code_quality_with_timing(
            "https://github.com/test/repo",
            "model-name"
        )
        assert score == 0.8
        assert latency >= 0

    @patch("src.code_quality.subprocess.run")
    def test_with_code_url_clone_fails(self, mock_run):
        """Test when git clone fails"""
        mock_run.side_effect = Exception("Clone failed")
        
        score, latency = calculate_code_quality_with_timing(
            "https://github.com/test/repo",
            "model-name"
        )
        assert score == 0.0
        assert latency >= 0

    @patch("src.code_quality.find_code_repo_via_genai")
    @patch("src.code_quality.subprocess.run")
    @patch("src.code_quality.run_flake8_on_repo")
    def test_genai_fallback_success(self, mock_flake8, mock_run, mock_genai):
        """Test GenAI fallback when no code URL provided"""
        mock_genai.return_value = "https://github.com/found/repo"
        mock_run.return_value = Mock()
        mock_flake8.return_value = (0.7, 150)
        
        score, latency = calculate_code_quality_with_timing(None, "model-name")
        assert score == 0.7
        assert latency >= 0

    @patch("src.code_quality.find_code_repo_via_genai")
    @patch("src.code_quality.subprocess.run")
    def test_genai_fallback_clone_fails(self, mock_run, mock_genai):
        """Test GenAI fallback when clone fails"""
        mock_genai.return_value = "https://github.com/found/repo"
        mock_run.side_effect = Exception("Clone failed")
        
        score, latency = calculate_code_quality_with_timing(None, "model-name")
        assert score == 0.0
        assert latency >= 0

    @patch("src.code_quality.find_code_repo_via_genai")
    def test_no_code_repository_found(self, mock_genai):
        """Test when no code repository is found"""
        mock_genai.return_value = None
        
        score, latency = calculate_code_quality_with_timing(None, "model-name")
        assert score == 0.0
        assert latency >= 0

    @patch("src.code_quality.subprocess.run")
    @patch("src.code_quality.run_flake8_on_repo")
    def test_latency_measurement(self, mock_flake8, mock_run):
        """Test that latency is measured correctly"""
        import time
        
        def slow_flake8(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow operation
            return (0.5, 50)
        
        mock_run.return_value = Mock()
        mock_flake8.side_effect = slow_flake8
        
        score, latency = calculate_code_quality_with_timing(
            "https://github.com/test/repo",
            "model-name"
        )
        assert latency >= 100  # At least 100ms due to sleep


class TestCalculateCodeQuality:
    """Tests for calculate_code_quality function (without timing)"""

    @patch("src.code_quality.subprocess.run")
    @patch("src.code_quality.run_flake8_on_repo")
    def test_with_valid_code_url(self, mock_flake8, mock_run):
        """Test with valid code URL"""
        mock_run.return_value = Mock()
        mock_flake8.return_value = (0.9, 100)
        
        score = calculate_code_quality(
            "https://github.com/test/repo",
            "model-name"
        )
        assert score == 0.9

    @patch("src.code_quality.subprocess.run")
    def test_clone_exception(self, mock_run):
        """Test exception handling"""
        mock_run.side_effect = Exception("Clone failed")
        
        score = calculate_code_quality(
            "https://github.com/test/repo",
            "model-name"
        )
        assert score == 0.0

    @patch("src.code_quality.find_code_repo_via_genai")
    @patch("src.code_quality.subprocess.run")
    @patch("src.code_quality.run_flake8_on_repo")
    def test_genai_fallback(self, mock_flake8, mock_run, mock_genai):
        """Test GenAI fallback path"""
        mock_genai.return_value = "https://github.com/found/repo"
        mock_run.return_value = Mock()
        mock_flake8.return_value = (0.6, 120)
        
        score = calculate_code_quality(None, "model-name")
        assert score == 0.6

    @patch("src.code_quality.find_code_repo_via_genai")
    def test_no_repo_found(self, mock_genai):
        """Test when no repository found"""
        mock_genai.return_value = None
        
        score = calculate_code_quality(None, "model-name")
        assert score == 0.0


class TestIntegration:
    """Integration tests for code quality module"""

    @patch("src.code_quality.requests.post")
    @patch("src.code_quality.subprocess.run")
    @patch("src.code_quality.run_flake8_on_repo")
    def test_full_workflow_with_genai(self, mock_flake8, mock_run, mock_post):
        """Test complete workflow using GenAI to find repo"""
        # GenAI finds repo
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'https://github.com/discovered/repo'
                }
            }]
        }
        mock_post.return_value = mock_response
        
        # Clone and analyze succeeds
        mock_run.return_value = Mock()
        mock_flake8.return_value = (0.75, 200)
        
        score, latency = calculate_code_quality_with_timing(None, "test/model")
        assert score == 0.75
        assert latency >= 0

    def test_error_score_bounds(self):
        """Test that all error scenarios return scores in valid range"""
        score, latency = calculate_code_quality_with_timing(
            "invalid-url",
            "invalid-model"
        )
        assert 0.0 <= score <= 1.0
        assert latency >= 0

    @patch("src.code_quality.subprocess.run")
    def test_flake8_output_with_error_pattern(self, mock_run, tmp_path):
        """Test flake8 output with 'X     E' pattern"""
        mock_result = Mock()
        mock_result.stdout = "5     E501\n10"  # Pattern with error code
        mock_run.return_value = mock_result
        
        py_file = tmp_path / "test.py"
        py_file.write_text("print('hello')\n")
        
        score, latency = run_flake8_on_repo(str(tmp_path))
        assert 0.0 <= score <= 1.0

    @patch("src.code_quality.subprocess.run")
    def test_flake8_empty_output(self, mock_run, tmp_path):
        """Test flake8 with empty output"""
        mock_result = Mock()
        mock_result.stdout = ""
        mock_run.return_value = mock_result
        
        py_file = tmp_path / "test.py"
        py_file.write_text("print('hello')\n")
        
        score, latency = run_flake8_on_repo(str(tmp_path))
        assert score == 1.0  # No errors = perfect score