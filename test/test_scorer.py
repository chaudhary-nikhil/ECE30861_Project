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
from src.ramp_up_time import calculate_ramp_up_time_with_timing
from src.performance_claims import calculate_performance_claims_with_timing
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
        assert scores["raspberry_pi"] == 0.74
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
        assert scores_200["raspberry_pi"] == 0.95

        scores_500 = calculate_size_score(500)
        assert scores_500["jetson_nano"] == 0.67


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

    def __test_estimate_known_model(self):
        """Test estimation for known model"""
        size = estimate_model_size("google/bert", "https://huggingface.co/google-bert/bert-base-uncased", "model")
        assert size == 500


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

    # @patch("src.scorer.make_request")
    # def test_score_dataset_high_downloads(self, mock_request):
    #     """Test scoring dataset with high downloads"""
    #     mock_request.return_value = {
    #         "downloads": 100000,
    #         "likes": 60,
    #         "description": "Test"
    #     }

    #     result = score_dataset("https://huggingface.co/datasets/test")
    #     assert result.score >= 7.0

    # @patch("src.scorer.make_request")
    # def test_score_dataset_low_metrics(self, mock_request):
    #     """Test scoring dataset with low metrics"""
    #     mock_request.return_value = {
    #         "downloads": 50,
    #         "likes": 1,
    #         "description": ""
    #     }

    #     result = score_dataset("https://huggingface.co/datasets/test")
    #     assert result.score >= 2.0
    #     assert result.score <= 4.0

class TestScoreModel:
    """Tests for score_model function"""

    def test_score_model_invalid_url(self):
        """Test scoring invalid model URL"""
        result = score_model("https://invalid.com")
        assert result.category == UrlCategory.MODEL
        assert result.score == 0.0
        assert "error" in result.details

    # @patch("src.scorer.make_request")
    # def test_score_model_no_data(self, mock_request):
    #     """Test scoring model with no API data"""
    #     mock_request.return_value = None

    #     result = score_model("https://huggingface.co/google/bert")
    #     assert result.category == UrlCategory.MODEL
    #     assert result.score == 2.0
    #     assert result.details["name"] == "google-bert/bert-case-uncased"

    #@patch("src.scorer.make_request")
    # def test_score_model_with_data(self, mock_request):
    #     """Test scoring model with API data"""
    #     mock_request.return_value = {
    #         "downloads": 200000,
    #         "likes": 150,
    #         "cardData": {"key": "value"},
    #         "pipeline_tag": "text-classification"
    #     }

    #     result = score_model("https://huggingface.co/google/bert")
    #     assert result.score > 5.0
    #     assert result.details["downloads"] == 200000
    #     assert result.details["likes"] == 150
    #     assert result.details["has_model_card"] == True
    #     assert result.details["pipeline_tag"] == "text-classification"

    # @patch("src.scorer.make_request")
    # def test_score_model_high_metrics(self, mock_request):
    #     """Test scoring model with high metrics"""
    #     mock_request.return_value = {
    #         "downloads": 500000,
    #         "likes": 200,
    #         "cardData": {},
    #         "pipeline_tag": "text-generation"
    #     }

    #     result = score_model("test_url")
    #     assert result.score >= 8.0


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


class TestPerformanceClaims:
    """Tests for calculate_performance_claims_with_timing function"""

    def test_performance_claims_no_data(self):
        """Test performance claims with no data"""
        data = {}
        score, latency = calculate_performance_claims_with_timing(data, "")
        assert score == 0.0
        assert latency >= 0

    def test_performance_claims_with_model_card(self):
        """Test performance claims with model card"""
        data = {
            'cardData': {'some': 'data'},
            'downloads': 1000,
            'likes': 50
        }
        score, latency = calculate_performance_claims_with_timing(data, "test-model")
        assert score >= 0.1  # Should get some score for having model card
        assert latency >= 0

    def test_performance_claims_with_benchmark_keywords(self):
        """Test performance claims with benchmark keywords in card"""
        data = {
            'cardData': {
                'content': 'This model achieves 95% accuracy on GLUE benchmark and shows state-of-the-art performance'
            },
            'downloads': 10000,
            'likes': 100
        }
        score, latency = calculate_performance_claims_with_timing(data, "benchmark-model")
        assert score > 0.3  # Should get significant score for benchmark keywords
        assert latency >= 0

    def test_performance_claims_with_benchmark_datasets(self):
        """Test performance claims with specific benchmark datasets"""
        data = {
            'cardData': {
                'content': 'Results on SQuAD, SuperGLUE, and MMLU datasets show strong performance'
            },
            'downloads': 50000,
            'likes': 200
        }
        score, latency = calculate_performance_claims_with_timing(data, "dataset-model")
        assert score > 0.2  # Should get score for benchmark datasets
        assert latency >= 0

    def test_performance_claims_with_numerical_results(self):
        """Test performance claims with numerical results"""
        data = {
            'cardData': {
                'content': 'Accuracy: 92.5%, F1 Score: 0.89, BLEU: 45.2'
            },
            'downloads': 20000,
            'likes': 150
        }
        score, latency = calculate_performance_claims_with_timing(data, "numerical-model")
        assert score > 0.15  # Should get score for numerical results
        assert latency >= 0

    def test_performance_claims_with_performance_tags(self):
        """Test performance claims with performance-related tags"""
        data = {
            'cardData': {'content': 'Model description'},
            'tags': ['benchmark', 'evaluation', 'sota'],
            'downloads': 30000,
            'likes': 300
        }
        score, latency = calculate_performance_claims_with_timing(data, "tagged-model")
        assert score > 0.1  # Should get score for performance tags
        assert latency >= 0

    def test_performance_claims_with_paper_evidence(self):
        """Test performance claims with paper citations"""
        data = {
            'cardData': {
                'content': 'See our paper on arXiv:1234.5678 for detailed evaluation results'
            },
            'downloads': 15000,
            'likes': 100
        }
        score, latency = calculate_performance_claims_with_timing(data, "paper-model")
        assert score > 0.1  # Should get score for paper evidence
        assert latency >= 0

    def test_performance_claims_high_popularity(self):
        """Test performance claims with high popularity"""
        data = {
            'cardData': {'content': 'Basic model'},
            'downloads': 2000000,  # High downloads
            'likes': 5000  # High likes
        }
        score, latency = calculate_performance_claims_with_timing(data, "popular-model")
        assert score > 0.15  # Should get score for high popularity
        assert latency >= 0

    def test_performance_claims_comprehensive_evidence(self):
        """Test performance claims with comprehensive evidence"""
        data = {
            'cardData': {
                'content': 'This model achieves 95% accuracy on GLUE benchmark, 89% F1 on SQuAD, and shows state-of-the-art performance. See our paper on arXiv:1234.5678 for details.'
            },
            'tags': ['benchmark', 'sota', 'evaluation'],
            'downloads': 1000000,
            'likes': 2000
        }
        score, latency = calculate_performance_claims_with_timing(data, "comprehensive-model")
        assert score > 0.5  # Should get high score for comprehensive evidence
        assert latency >= 0

    def test_performance_claims_score_bounds(self):
        """Test that performance claims score is between 0 and 1"""
        data = {
            'cardData': {'content': 'Test'},
            'downloads': 1000,
            'likes': 10
        }
        score, latency = calculate_performance_claims_with_timing(data, "test-model")
        assert 0.0 <= score <= 1.0
        assert latency >= 0

    def test_performance_claims_error_handling(self):
        """Test performance claims with invalid data"""
        data = None
        score, latency = calculate_performance_claims_with_timing(data, "invalid-model")
        assert score == 0.0
        assert latency >= 0


class TestRampUpTime:
    """Tests for calculate_ramp_up_time_with_timing function"""

    def test_ramp_up_time_no_data(self):
        """Test ramp-up time with no data"""
        data = {}
        score, latency = calculate_ramp_up_time_with_timing(data, "")
        assert score == 0.0
        assert latency >= 0

    def test_ramp_up_time_with_model_card(self):
        """Test ramp-up time with model card"""
        data = {
            'cardData': {'some': 'data'},
            'downloads': 1000,
            'likes': 50
        }
        score, latency = calculate_ramp_up_time_with_timing(data, "test-model")
        assert score >= 0.1
        assert latency >= 0

    def test_ramp_up_time_with_readme_file(self):
        """Test ramp-up time with README file"""
        data = {
            'cardData': {'content': 'Model description'},
            'files': [{'rfilename': 'README.md'}],
            'downloads': 10000,
            'likes': 100
        }
        score, latency = calculate_ramp_up_time_with_timing(data, "readme-model")
        assert score >= 0.24  # Should get score for model card + README file
        assert latency >= 0

    def test_ramp_up_time_with_requirements_file(self):
        """Test ramp-up time with requirements file"""
        data = {
            'cardData': {'content': 'Model description'},
            'files': [{'rfilename': 'requirements.txt'}],
            'downloads': 20000,
            'likes': 150
        }
        score, latency = calculate_ramp_up_time_with_timing(data, "requirements-model")
        assert score >= 0.3  # Should get score for model card + requirements
        assert latency >= 0

    def test_ramp_up_time_with_example_script(self):
        """Test ramp-up time with example script"""
        data = {
            'cardData': {'content': 'Model description'},
            'files': [{'rfilename': 'example.py'}],
            'downloads': 30000,
            'likes': 200
        }
        score, latency = calculate_ramp_up_time_with_timing(data, "example-model")
        assert score >= 0.36  # Should get score for model card + example script
        assert latency >= 0

    def test_ramp_up_time_with_config_files(self):
        """Test ramp-up time with config files"""
        data = {
            'cardData': {'content': 'Model description'},
            'files': [{'rfilename': 'config.json'}],
            'downloads': 40000,
            'likes': 300
        }
        score, latency = calculate_ramp_up_time_with_timing(data, "config-model")
        assert score >= 0.3  # Should get score for model card + config
        assert latency >= 0

    def test_ramp_up_time_with_documentation_sections(self):
        """Test ramp-up time with good documentation sections"""
        data = {
            'cardData': {
                'content': 'This model is for text classification. Usage: import the model. Installation: pip install transformers. Example: see below. Quickstart guide available.'
            },
            'downloads': 50000,
            'likes': 400
        }
        score, latency = calculate_ramp_up_time_with_timing(data, "doc-model")
        assert score >= 0.3  # Should get score for model card + documentation sections
        assert latency >= 0

    def test_ramp_up_time_with_code_examples(self):
        """Test ramp-up time with code examples"""
        data = {
            'cardData': {
                'content': '```python\nimport transformers\nfrom transformers import AutoModel\nmodel = AutoModel.from_pretrained("model-name")\n```'
            },
            'downloads': 60000,
            'likes': 500
        }
        score, latency = calculate_ramp_up_time_with_timing(data, "code-model")
        assert score >= 0.27  # Should get score for model card + code examples
        assert latency >= 0

    def test_ramp_up_time_with_setup_instructions(self):
        """Test ramp-up time with setup instructions"""
        data = {
            'cardData': {
                'content': 'Installation: pip install transformers torch. Setup: Download the model. Install dependencies.'
            },
            'downloads': 70000,
            'likes': 600
        }
        score, latency = calculate_ramp_up_time_with_timing(data, "setup-model")
        assert score >= 0.3  # Should get score for model card + setup instructions
        assert latency >= 0

    def test_ramp_up_time_comprehensive_evidence(self):
        """Test ramp-up time with comprehensive evidence"""
        data = {
            'cardData': {
                'content': 'This model is for text classification. Usage: import transformers. Installation: pip install transformers. Example: ```python\nmodel = AutoModel.from_pretrained("model")\n```'
            },
            'files': [
                {'rfilename': 'README.md'},
                {'rfilename': 'requirements.txt'},
                {'rfilename': 'example.py'},
                {'rfilename': 'config.json'}
            ],
            'downloads': 1000000,
            'likes': 2000
        }
        score, latency = calculate_ramp_up_time_with_timing(data, "comprehensive-model")
        assert score >= 0.8  # Should get high score for comprehensive evidence
        assert latency >= 0

    def test_ramp_up_time_high_popularity(self):
        """Test ramp-up time with high popularity"""
        data = {
            'cardData': {'content': 'Basic model'},
            'downloads': 2000000,  # High downloads
            'likes': 5000  # High likes
        }
        score, latency = calculate_ramp_up_time_with_timing(data, "popular-model")
        assert score >= 0.2  # Should get score for high popularity
        assert latency >= 0

    def test_ramp_up_time_score_bounds(self):
        """Test that ramp-up time score is between 0 and 1"""
        data = {
            'cardData': {'content': 'Test'},
            'downloads': 1000,
            'likes': 10
        }
        score, latency = calculate_ramp_up_time_with_timing(data, "test-model")
        assert 0.0 <= score <= 1.0
        assert latency >= 0

    def test_ramp_up_time_error_handling(self):
        """Test ramp-up time with invalid data"""
        data = None
        score, latency = calculate_ramp_up_time_with_timing(data, "invalid-model")
        assert score == 0.0
        assert latency >= 0
