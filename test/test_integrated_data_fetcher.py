import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from src.integrated_data_fetcher import IntegratedDataFetcher
from src.url import UrlCategory


class TestIntegratedDataFetcherInit:
   def test_init_with_no_tokens(self):
        fetcher = IntegratedDataFetcher()
        assert fetcher.hf_api_token is None or fetcher.hf_api_token == ""
        assert fetcher.github_token is None or fetcher.github_token == ""
        assert fetcher.hf_headers == {}
        assert fetcher.gh_headers == {}

   def test_init_with_tokens(self):
        fetcher = IntegratedDataFetcher(
            hf_api_token="hf_token",
            github_token="gh_token"
        )
        assert fetcher.hf_api_token == "hf_token"
        assert fetcher.github_token == "gh_token"
        assert "Authorization" in fetcher.hf_headers
        assert "Authorization" in fetcher.gh_headers

class TestFetchData:
    
    def test_fetch_data_invalid_url(self):
        fetcher = IntegratedDataFetcher()
        result = fetcher.fetch_data("https://invalid.com")
        assert result["category"] == "INVALID"
        assert "error" in result

    @patch.object(IntegratedDataFetcher, "_fetch_model_data")
    def test_fetch_data_model_url(self, mock_fetch):
        mock_fetch.return_value = {"category": "MODEL", "name": "test"}
        
        fetcher = IntegratedDataFetcher()
        result = fetcher.fetch_data("https://huggingface.co/google/bert")
        
        assert result["category"] == "MODEL"
        mock_fetch.assert_called_once()

    @patch.object(IntegratedDataFetcher, "_fetch_dataset_data")
    def test_fetch_data_dataset_url(self, mock_fetch):
        mock_fetch.return_value = {"category": "DATASET", "name": "test"}
        
        fetcher = IntegratedDataFetcher()
        result = fetcher.fetch_data("https://huggingface.co/datasets/squad")
        
        assert result["category"] == "DATASET"
        mock_fetch.assert_called_once()

    @patch.object(IntegratedDataFetcher, "_fetch_code_data")
    def test_fetch_data_code_url(self, mock_fetch):
        mock_fetch.return_value = {"category": "CODE", "name": "test"}
        
        fetcher = IntegratedDataFetcher()
        result = fetcher.fetch_data("https://github.com/user/repo")
        
        assert result["category"] == "CODE"
        mock_fetch.assert_called_once()


class TestExtractMethods:
    
    def test_extract_hf_model_id(self):
        fetcher = IntegratedDataFetcher()
        
        url = "https://huggingface.co/google/bert-base"
        model_id = fetcher._extract_hf_model_id(url)
        assert model_id == "google/bert-base"

    def test_extract_hf_model_id_invalid(self):
        fetcher = IntegratedDataFetcher()
        
        url = "https://invalid.com"
        model_id = fetcher._extract_hf_model_id(url)
        assert model_id is None

    def test_extract_hf_dataset_id(self):
        fetcher = IntegratedDataFetcher()
        
        url = "https://huggingface.co/datasets/squad"
        dataset_id = fetcher._extract_hf_dataset_id(url)
        assert dataset_id == "squad"

    def test_extract_hf_dataset_id_with_org(self):
        fetcher = IntegratedDataFetcher()
        
        url = "https://huggingface.co/datasets/google/test"
        dataset_id = fetcher._extract_hf_dataset_id(url)
        assert dataset_id == "google/test"

    def test_extract_github_repo(self):
        fetcher = IntegratedDataFetcher()
        
        url = "https://github.com/user/repo"
        repo_info = fetcher._extract_github_repo(url)
        assert repo_info == ("user", "repo")

    def test_extract_github_repo_invalid(self):
        fetcher = IntegratedDataFetcher()
        
        url = "https://invalid.com"
        repo_info = fetcher._extract_github_repo(url)
        assert repo_info is None


class TestLicenseExtraction:
    
    def test_extract_license_from_tags(self):
        fetcher = IntegratedDataFetcher()
        
        info_dict = {"tags": ["license:mit", "python"]}
        license = fetcher._extract_license_from_tags(info_dict)
        assert license == "mit"

    def test_extract_license_from_direct_field(self):
        fetcher = IntegratedDataFetcher()
        
        info_dict = {"license": "apache-2.0"}
        license = fetcher._extract_license_from_tags(info_dict)
        assert license == "apache-2.0"

    def test_extract_license_from_readme(self):
        fetcher = IntegratedDataFetcher()
        
        info_dict = {}
        readme = "# Model\n\nlicense: MIT\n"
        license = fetcher._extract_license_from_tags(info_dict, readme)
        assert license.lower() == "mit"

    def test_extract_license_not_found(self):
        fetcher = IntegratedDataFetcher()
        
        info_dict = {}
        license = fetcher._extract_license_from_tags(info_dict)
        assert license == ""


class TestAPIHelpers:
    
    def test_get_hf_model_info_success(self):
        fetcher = IntegratedDataFetcher()
        
        mock_response = Mock()
        mock_response.json.return_value = {"downloads": 1000}
        mock_response.raise_for_status.return_value = None
        fetcher.session.get = Mock(return_value=mock_response)
        
        result = fetcher._get_hf_model_info("google/bert")
        
        assert result == {"downloads": 1000}

    def test_get_hf_model_info_failure(self):
        fetcher = IntegratedDataFetcher()
        fetcher.session.get = Mock(side_effect=Exception("Network error"))
        
        result = fetcher._get_hf_model_info("google/bert")
        
        assert result == {}

    def test_get_github_repo_info_success(self):
        fetcher = IntegratedDataFetcher()
        
        mock_response = Mock()
        mock_response.json.return_value = {"stars": 1000}
        mock_response.raise_for_status.return_value = None
        fetcher.session.get = Mock(return_value=mock_response)
        
        result = fetcher._get_github_repo_info("user", "repo")
        
        assert result == {"stars": 1000}

    def test_get_github_readme_success(self):
        import base64
        readme_content = "# Test Repo"
        encoded_content = base64.b64encode(readme_content.encode()).decode()
        
        fetcher = IntegratedDataFetcher()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": encoded_content}
        fetcher.session.get = Mock(return_value=mock_response)
        
        result = fetcher._get_github_readme("user", "repo")
        
        assert result == readme_content

    def test_get_github_readme_not_found(self):
        fetcher = IntegratedDataFetcher()
        mock_response = Mock()
        mock_response.status_code = 404
        fetcher.session.get = Mock(return_value=mock_response)
        
        result = fetcher._get_github_readme("user", "repo")
        
        assert result == ""


class TestContributorsExtraction:
    
    def test_extract_contributors_with_author(self):
        fetcher = IntegratedDataFetcher()
        
        info = {"author": "google"}
        result = fetcher._extract_contributors(info, "google/bert")
        assert result == ["google"]

    def test_extract_contributors_without_author(self):
        fetcher = IntegratedDataFetcher()
        
        info = {}
        result = fetcher._extract_contributors(info, "google/bert")
        assert result == ["google"]


class TestGitHubLicenseExtraction:
    def test_extract_github_license_present(self):
        fetcher = IntegratedDataFetcher()
        
        repo_data = {"license": {"spdx_id": "MIT"}}
        result = fetcher._extract_github_license(repo_data)
        assert result == "MIT"

    def test_extract_github_license_missing(self):
        fetcher = IntegratedDataFetcher()
        
        repo_data = {}
        result = fetcher._extract_github_license(repo_data)
        assert result == ""

    def test_extract_github_license_null(self):
        fetcher = IntegratedDataFetcher()
        
        repo_data = {"license": None}
        result = fetcher._extract_github_license(repo_data)
        assert result == ""
