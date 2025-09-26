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
    analyze_model_repository,
    estimate_model_size,
    score_dataset,
    score_model,
    score_code,
    score_url,
    UrlCategory,
)


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
