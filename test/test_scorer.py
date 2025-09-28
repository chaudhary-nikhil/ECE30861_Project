import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os
from pathlib import Path

from src.scorer import (
    ScoreResult,
    make_request,
    calculate_size_score,
    estimate_model_size,
    score_dataset,
    score_model,
    score_code,
    score_url,
    calculate_model_bus_factor,
    calculate_dataset_bus_factor,
    calculate_code_bus_factor,
    calculate_bus_factor_with_timing,
    is_major_organization,

    UrlCategory,
)
from src.url import UrlCategory


class TestScoreResult:
    def test_percentage_calculation(self):
        result = ScoreResult(
            url="https://example.com",
            category=UrlCategory.MODEL,
            score=7.5,
            max_score=10.0,
            details={},
        )
        assert result.percentage == 75.0

    def test_percentage_zero_max_score(self):
        result = ScoreResult(
            url="https://example.com",
            category=UrlCategory.MODEL,
            score=5.0,
            max_score=0.0,
            details={},
        )
        assert result.percentage == 0.0

    def test_str_representation(self):
        result = ScoreResult(
            url="https://example.com",
            category=UrlCategory.MODEL,
            score=8.0,
            max_score=10.0,
            details={},
        )
        assert "MODEL" in str(result)
        assert "8.0/10.0" in str(result)
        assert "80.0%" in str(result)


class TestMakeRequest:
    @patch("src.scorer.requests.get")
    def test_successful_request(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = make_request("https://example.com")
        assert result == {"data": "test"}
        mock_get.assert_called_once()

    @patch("src.scorer.requests.get")
    def test_failed_request(self, mock_get):
        mock_get.side_effect = Exception("Network error")

        result = make_request("https://example.com")
        assert result is None

    @patch("src.scorer.requests.get")
    def test_request_timeout(self, mock_get):
        mock_get.side_effect = TimeoutError()

        result = make_request("https://example.com")
        assert result is None


class TestCalculateSizeScore:
    def test_small_model_raspberry_pi(self):
        scores = calculate_size_score(0)
        assert scores["raspberry_pi"] == 1.0
        assert scores["jetson_nano"] == 1.0
        assert scores["desktop_pc"] == 1.0
        assert scores["aws_server"] == 1.0

    def test_medium_model_raspberry_pi(self):
        scores = calculate_size_score(100)
        assert scores["raspberry_pi"] == 0.5
        assert scores["jetson_nano"] > 0.1995
        assert scores["desktop_pc"] > 0.01995
        assert scores["aws_server"] > 0.001995

    def test_large_model_all_hardware(self):
        scores = calculate_size_score(10000)
        assert scores["raspberry_pi"] == 0.0
        assert scores["jetson_nano"] == 0.0
        assert scores["desktop_pc"] < 0.5
        assert scores["aws_server"] > 0.0

    def test_very_large_model(self):
        scores = calculate_size_score(10000000)
        assert scores["raspberry_pi"] == 0.0
        assert scores["jetson_nano"] == 0.0
        assert scores["desktop_pc"] == 0.0
        assert scores["aws_server"] == 0.0

    def test_boundary_values(self):
        scores_200 = calculate_size_score(20)
        assert scores_200["raspberry_pi"] == 0.9

        scores_500 = calculate_size_score(500)
        assert scores_500["jetson_nano"] == 0.0


class TestEstimateModelSize:
    """Tests for estimate_model_size function"""

    def test_estimate_unknown_model(self):
        """Test estimation for unknown model"""
        size = estimate_model_size("unknown", "test_url", "model")
        assert size == 500

    def test_estimate_empty_model(self):
        """Test estimation for empty model name"""
        size = estimate_model_size("", "test_url", "model")
        assert size == 500

    def test_estimate_known_model(self):
        """Test estimation for known model"""
        size = estimate_model_size("google/bert", "https://huggingface.co/google-bert/bert-base-uncased", "model")
        assert size == 1000


class TestScoreDataset:
    """Tests for score_dataset function"""

    def test_score_dataset_invalid_url(self):
        """Test scoring invalid dataset URL"""
        result = score_dataset("https://invalid.com")
        assert result.category == UrlCategory.DATASET
        assert result.score == 0.0
        assert "error" in result.details

    @patch("src.scorer.make_request")
    def test_score_dataset_no_data(self, mock_request):
        """Test scoring dataset with no API data"""
        mock_request.return_value = None

        result = score_dataset("https://huggingface.co/datasets/squad")
        assert result.category == UrlCategory.DATASET
        assert result.details["name"] == "squad"

    @patch("src.scorer.make_request")
    def test_score_dataset_with_data(self, mock_request):
        """Test scoring dataset with API data"""
        mock_request.return_value = {
            "downloads": 50000,
            "likes": 100,
            "description": "Test dataset"
        }

        result = score_dataset("https://huggingface.co/datasets/squad")
        assert result.score > 0
        assert result.details["downloads"] == 50000
        assert result.details["likes"] == 100

    @patch("src.scorer.make_request")
    def test_score_dataset_high_downloads(self, mock_request):
        """Test scoring dataset with high downloads"""
        mock_request.return_value = {
            "downloads": 100000,
            "likes": 60,
            "description": "Test"
        }

        result = score_dataset("https://huggingface.co/datasets/test")
        assert result.score >= 7.0

    @patch("src.scorer.make_request")
    def test_score_dataset_low_metrics(self, mock_request):
        """Test scoring dataset with low metrics"""
        mock_request.return_value = {
            "downloads": 50,
            "likes": 1,
            "description": ""
        }

        result = score_dataset("https://huggingface.co/datasets/test")
        assert result.score >= 2.0
        assert result.score <= 4.0


class TestScoreModel:
    """Tests for score_model function"""

    def test_score_model_invalid_url(self):
        """Test scoring invalid model URL"""
        result = score_model("https://invalid.com")
        assert result.category == UrlCategory.MODEL
        assert result.score == 0.0
        assert "error" in result.details

    @patch("src.scorer.make_request")
    def test_score_model_no_data(self, mock_request):
        """Test scoring model with no API data"""
        mock_request.return_value = None

        result = score_model("https://huggingface.co/google/bert")
        assert result.category == UrlCategory.MODEL
        assert result.score == 2.0
        assert result.details["name"] == "google/bert"

    @patch("src.scorer.make_request")
    def test_score_model_with_data(self, mock_request):
        """Test scoring model with API data"""
        mock_request.return_value = {
            "downloads": 200000,
            "likes": 150,
            "cardData": {"key": "value"},
            "pipeline_tag": "text-classification"
        }

        result = score_model("https://huggingface.co/google/bert")
        assert result.score > 5.0
        assert result.details["downloads"] == 200000
        assert result.details["likes"] == 150
        assert result.details["has_model_card"] == True
        assert result.details["pipeline_tag"] == "text-classification"

    @patch("src.scorer.make_request")
    def test_score_model_high_metrics(self, mock_request):
        """Test scoring model with high metrics"""
        mock_request.return_value = {
            "downloads": 500000,
            "likes": 200,
            "cardData": {},
            "pipeline_tag": "text-generation"
        }

        result = score_model("https://huggingface.co/test/model")
        assert result.score >= 8.0


class TestScoreCode:
    """Tests for score_code function"""

    def test_score_code_invalid_url(self):
        """Test scoring invalid code URL"""
        result = score_code("https://invalid.com")
        assert result.category == UrlCategory.CODE
        assert result.score == 0.0
        assert "error" in result.details

    @patch("src.scorer.make_request")
    def test_score_code_no_data(self, mock_request):
        """Test scoring code with no API data"""
        mock_request.return_value = None

        result = score_code("https://github.com/user/repo")
        assert result.category == UrlCategory.CODE
        assert result.score == 2.0
        assert result.details["name"] == "user/repo"

    @patch("src.scorer.make_request")
    def test_score_code_with_data(self, mock_request):
        """Test scoring code with API data"""
        mock_request.return_value = {
            "stargazers_count": 5000,
            "forks_count": 500,
            "description": "Test repo",
            "license": {"name": "MIT"},
            "language": "Python"
        }

        result = score_code("https://github.com/user/repo")
        assert result.score > 5.0
        assert result.details["stars"] == 5000
        assert result.details["forks"] == 500
        assert result.details["has_description"] == True
        assert result.details["has_license"] == True
        assert result.details["language"] == "Python"

    @patch("src.scorer.make_request")
    def test_score_code_high_stars(self, mock_request):
        """Test scoring code with high star count"""
        mock_request.return_value = {
            "stargazers_count": 10000,
            "forks_count": 1000,
            "description": "Popular repo",
            "license": {"name": "Apache-2.0"},
            "language": "JavaScript"
        }

        result = score_code("https://github.com/popular/repo")
        assert result.score >= 8.0

    @patch("src.scorer.make_request")
    def test_score_code_low_metrics(self, mock_request):
        """Test scoring code with low metrics"""
        mock_request.return_value = {
            "stargazers_count": 5,
            "forks_count": 1,
            "description": "",
            "license": None,
            "language": None
        }

        result = score_code("https://github.com/small/repo")
        assert result.score >= 2.0
        assert result.score <= 4.0


class TestScoreUrl:
    """Tests for score_url function"""

    @patch("src.scorer.score_dataset")
    def test_score_url_dataset(self, mock_score):
        """Test scoring dataset URL"""
        mock_result = ScoreResult(
            url="test",
            category=UrlCategory.DATASET,
            score=5.0,
            max_score=10.0,
            details={}
        )
        mock_score.return_value = mock_result

        result = score_url("https://huggingface.co/datasets/test", UrlCategory.DATASET)
        assert result.category == UrlCategory.DATASET
        mock_score.assert_called_once()

    @patch("src.scorer.score_model")
    def test_score_url_model(self, mock_score):
        """Test scoring model URL"""
        mock_result = ScoreResult(
            url="test",
            category=UrlCategory.MODEL,
            score=5.0,
            max_score=10.0,
            details={}
        )
        mock_score.return_value = mock_result

        result = score_url("https://huggingface.co/test", UrlCategory.MODEL)
        assert result.category == UrlCategory.MODEL
        mock_score.assert_called_once()

    @patch("src.scorer.score_code")
    def test_score_url_code(self, mock_score):
        """Test scoring code URL"""
        mock_result = ScoreResult(
            url="test",
            category=UrlCategory.CODE,
            score=5.0,
            max_score=10.0,
            details={}
        )
        mock_score.return_value = mock_result

        result = score_url("https://github.com/test/repo", UrlCategory.CODE)
        assert result.category == UrlCategory.CODE
        mock_score.assert_called_once()

    # def test_score_url_invalid(self):
    #     """Test scoring invalid URL"""
    #     result = score_url("https://invalid.com", UrlCategory.INVALID)
    #     assert result.category == UrlCategory.INVALID
    #     assert result.score == 0.0
    #     assert "error" in result.details

class TestBusFactor:
    def test_model_bus_factor_no_contributors(self):
        score = calculate_model_bus_factor(0, "individual/project")
        assert score == 0.0

    def test_model_bus_factor_single_contributor(self):
        score = calculate_model_bus_factor(1, "individual/project")
        assert score == 0.3

    def test_model_bus_factor_major_org(self):
        contributor_count = 1
        score = calculate_model_bus_factor(contributor_count, "microsoft/awesome-model")
        assert score == 0.95

    def test_dataset_bus_factor(self):
        contributor_count = 3
        score = calculate_dataset_bus_factor(contributor_count, "individual/dataset")
        assert score == 1.0

    def test_dataset_bus_factor_major_org(self):
        score = calculate_dataset_bus_factor(0, "google/dataset")
        assert score == 0.95

    def test_code_bus_factor_major_org(self):
        score = calculate_code_bus_factor(3, "openai/repo")
        assert score == 0.95


    def test_organization_detection(self):
        """Test the organization detection function"""
        assert is_major_organization("google/model") == True
        assert is_major_organization("microsoft/repo") == True
        assert is_major_organization("individual/project") == False
        assert is_major_organization("") == False

    def test_model_bus_factor_multiple_contributors(self):
        score = calculate_model_bus_factor(5, "individual/project")
        assert score == 1.0
