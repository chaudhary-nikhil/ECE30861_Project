"""
Integrated data fetcher that works with the URL categorization system
"""

import requests
import json
from typing import Any
from urllib.parse import urlparse
import re
from src.url import Url, UrlCategory, determine_category

class IntegratedDataFetcher:
    """Fetches data from different sources based on URL category"""

    def __init__(
        self, hf_api_token: str | None = None, github_token: str | None = None
    ):
        self.hf_api_token = hf_api_token
        self.github_token = github_token
        self.session = requests.Session()

        # Set up headers for different APIs
        if hf_api_token:
            self.hf_headers = {"Authorization": f"Bearer {hf_api_token}"}
        else:
            self.hf_headers = {}

        if github_token:
            self.gh_headers = {"Authorization": f"token {github_token}"}
        else:
            self.gh_headers = {}

    def fetch_data(self, url: str) -> dict[str, Any]:
        """Main method to fetch data based on URL category"""
        url_obj = Url(url)
        try:
            if url_obj.category == UrlCategory.MODEL:
                return self._fetch_model_data(url_obj)
            elif url_obj.category == UrlCategory.DATASET:
                return self._fetch_dataset_data(url_obj)
            elif url_obj.category == UrlCategory.CODE:
                return self._fetch_code_data(url_obj)
            else:
                return {"error": f"Invalid URL: {url}", "category": "INVALID"}
        except Exception as e:
            return {
                "error": f"Error fetching data: {str(e)}",
                "category": url_obj.category.name,
                "url": url,
            }

    def _extract_license_from_tags(
        self, info_dict: dict[str, Any], readme: str = ""
    ) -> str:
        """Extract license from tags where Hugging Face stores license info"""
        # Strategy 1: Check tags for license:xxx format
        tags = info_dict.get("tags", [])
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("license:"):
                return tag.replace("license:", "").strip()

        # Strategy 2: Direct license field (backup)
        if "license" in info_dict and info_dict["license"]:
            return str(info_dict["license"]).strip()

        # Strategy 3: README fallback
        if readme:
            license_match = re.search(r"license:\s*([^\n]+)", readme, re.IGNORECASE)
            if license_match:
                return license_match.group(1).strip()

        return ""

    def _fetch_model_data(self, url_obj: Url) -> dict[str, Any]:
        """Fetch Hugging Face model data"""
        model_id = self._extract_hf_model_id(url_obj.link)
        if not model_id:
            return {"error": "Could not extract model ID", "category": "MODEL"}

        print(f"Fetching MODEL data for: {model_id}")

        # Get basic model info
        model_info = self._get_hf_model_info(model_id)

        # Get model files
        files = self._get_hf_model_files(model_id)

        # Get README
        readme = self._get_hf_readme(model_id)

        # Extract license from tags
        license = self._extract_license_from_tags(model_info, readme)

        # Process into standardized format
        return {
            "category": "MODEL",
            "name": model_id,
            "url": url_obj.link,
            "readme": readme,
            "license": license,
            "downloads": model_info.get("downloads", 0),
            "likes": model_info.get("likes", 0),
            "last_modified": model_info.get("lastModified", ""),
            "tags": model_info.get("tags", []),
            "files": files,
            "library_name": model_info.get("library_name", ""),
            "pipeline_tag": model_info.get("pipeline_tag", ""),
            "contributors": self._extract_contributors(model_info, model_id),
            "raw_info": model_info,
        }

    def _fetch_dataset_data(self, url_obj: Url) -> dict[str, Any]:
        """Fetch Hugging Face dataset data"""
        dataset_id = self._extract_hf_dataset_id(url_obj.link)
        if not dataset_id:
            return {"error": "Could not extract dataset ID", "category": "DATASET"}

        print(f"Fetching DATASET data for: {dataset_id}")

        # Get basic dataset info
        dataset_info = self._get_hf_dataset_info(dataset_id)

        # Get dataset files
        files = self._get_hf_dataset_files(dataset_id)

        # Get README
        readme = self._get_hf_dataset_readme(dataset_id)

        # Extract license from tags
        license = self._extract_license_from_tags(dataset_info, readme)

        return {
            "category": "DATASET",
            "name": dataset_id,
            "url": url_obj.link,
            "readme": readme,
            "license": license,
            "downloads": dataset_info.get("downloads", 0),
            "likes": dataset_info.get("likes", 0),
            "last_modified": dataset_info.get("lastModified", ""),
            "tags": dataset_info.get("tags", []),
            "files": files,
            "size_info": self._extract_dataset_size(dataset_id),
            "contributors": self._extract_contributors(dataset_info, dataset_id),
            "raw_info": dataset_info,
        }

    def _fetch_code_data(self, url_obj: Url) -> dict[str, Any]:
        """Fetch GitHub repository data"""
        repo_info = self._extract_github_repo(url_obj.link)
        if not repo_info:
            return {"error": "Could not extract GitHub repo info", "category": "CODE"}

        owner, repo = repo_info
        print(f"Fetching CODE data for: {owner}/{repo}")

        # Get basic repo info
        repo_data = self._get_github_repo_info(owner, repo)

        # Get README
        readme = self._get_github_readme(owner, repo)

        # Get contributors
        contributors = self._get_github_contributors(owner, repo)

        # Get recent commits for activity analysis
        commits = self._get_github_recent_commits(owner, repo)

        return {
            "category": "CODE",
            "name": f"{owner}/{repo}",
            "url": url_obj.link,
            "readme": readme,
            "license": self._extract_github_license(repo_data),
            "stars": repo_data.get("stargazers_count", 0),
            "forks": repo_data.get("forks_count", 0),
            "last_modified": repo_data.get("updated_at", ""),
            "language": repo_data.get("language", ""),
            "contributors": contributors,
            "recent_commits": commits,
            "open_issues": repo_data.get("open_issues_count", 0),
            "size_kb": repo_data.get("size", 0),
            "raw_info": repo_data,
        }

    # Hugging Face helper methods
    def _extract_hf_model_id(self, url: str) -> str | None:
        """Extract model ID from HF model URL"""
        # https://huggingface.co/google/gemma-3-270m -> google/gemma-3-270m
        match = re.search(r"huggingface\.co/([^/]+/[^/?]+)", url)
        return match.group(1) if match else None

    def _extract_hf_dataset_id(self, url: str) -> str | None:
        """Extract dataset ID from HF dataset URL"""
        # https://huggingface.co/datasets/squad -> squad
        match = re.search(r"huggingface\.co/datasets/([^/?]+(?:/[^/?]+)?)", url)
        return match.group(1) if match else None

    def _get_hf_model_info(self, model_id: str) -> dict[str, Any]:
        """Get model info from HF API"""
        try:
            url = f"https://huggingface.co/api/models/{model_id}"
            response = self.session.get(url, headers=self.hf_headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching model info: {e}")
            return {}

    def _get_hf_model_files(self, model_id: str) -> dict[str, Any]:
        """Get model files from HF API"""
        try:
            url = f"https://huggingface.co/api/models/{model_id}/tree/main"
            response = self.session.get(url, headers=self.hf_headers, timeout=10)
            response.raise_for_status()
            files_list = response.json()

            # Convert to dict format
            files_dict = {}
            for file_info in files_list:
                if isinstance(file_info, dict) and "path" in file_info:
                    files_dict[file_info["path"]] = {
                        "size": file_info.get("size", 0),
                        "type": file_info.get("type", "file"),
                    }
            return files_dict
        except Exception as e:
            print(f"Error fetching model files: {e}")
            return {}

    def _get_hf_readme(self, model_id: str) -> str:
        """Get README from HF model"""
        try:
            url = f"https://huggingface.co/{model_id}/raw/main/README.md"
            response = self.session.get(url, timeout=10)
            return response.text if response.status_code == 200 else ""
        except Exception:
            return ""

    def _get_hf_dataset_info(self, dataset_id: str) -> dict[str, Any]:
        """Get dataset info from HF API"""
        try:
            url = f"https://huggingface.co/api/datasets/{dataset_id}"
            response = self.session.get(url, headers=self.hf_headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching dataset info: {e}")
            return {}

    def _get_hf_dataset_files(self, dataset_id: str) -> dict[str, Any]:
        """Get dataset files from HF API"""
        try:
            url = f"https://huggingface.co/api/datasets/{dataset_id}/tree/main"
            response = self.session.get(url, headers=self.hf_headers, timeout=10)
            response.raise_for_status()
            files_list = response.json()

            files_dict = {}
            for file_info in files_list:
                if isinstance(file_info, dict) and "path" in file_info:
                    files_dict[file_info["path"]] = {
                        "size": file_info.get("size", 0),
                        "type": file_info.get("type", "file"),
                    }
            return files_dict
        except Exception as e:
            print(f"Error fetching dataset files: {e}")
            return {}

    def _get_hf_dataset_readme(self, dataset_id: str) -> str:
        """Get README from HF dataset"""
        try:
            url = f"https://huggingface.co/datasets/{dataset_id}/raw/main/README.md"
            response = self.session.get(url, timeout=10)
            return response.text if response.status_code == 200 else ""
        except Exception:
            return ""

    # GitHub helper methods
    def _extract_github_repo(self, url: str) -> tuple | None:
        """Extract owner/repo from GitHub URL"""
        # https://github.com/owner/repo -> (owner, repo)
        match = re.search(r"github\.com/([^/]+)/([^/?]+)", url)
        return (match.group(1), match.group(2)) if match else None

    def _get_github_repo_info(self, owner: str, repo: str) -> dict[str, Any]:
        """Get repo info from GitHub API"""
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}"
            response = self.session.get(url, headers=self.gh_headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching GitHub repo info: {e}")
            return {}

    def _get_github_readme(self, owner: str, repo: str) -> str:
        """Get README from GitHub repo"""
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/readme"
            response = self.session.get(url, headers=self.gh_headers, timeout=10)
            if response.status_code == 200:
                readme_data = response.json()
                # GitHub returns base64 encoded content
                import base64

                return base64.b64decode(readme_data["content"]).decode("utf-8")
            return ""
        except Exception:
            return ""

    def _get_github_contributors(self, owner: str, repo: str) -> list[str]:
        """Get contributors from GitHub repo"""
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
            response = self.session.get(url, headers=self.gh_headers, timeout=10)
            if response.status_code == 200:
                contributors = response.json()
                return [c.get("login", "") for c in contributors[:10]]  # Top 10
            return []
        except Exception:
            return []

    def _get_github_recent_commits(self, owner: str, repo: str) -> list[dict[str, Any]]:
        """Get recent commits for activity analysis"""
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/commits"
            response = self.session.get(url, headers=self.gh_headers, timeout=10)
            if response.status_code == 200:
                commits = response.json()
                return commits[:10]  # Last 10 commits
            return []
        except Exception:
            return []

    # Size calculation methods
    def _extract_dataset_size(self, dataset_id: str) -> dict[str, Any]:
        """Dataset size using Dataset Viewer API with a huggingface_hub fallback."""
        # PRIMARY: Dataset Viewer API
        try:
            url = f"https://datasets-server.huggingface.co/size?dataset={dataset_id}"
            response = self.session.get(url, headers=self.hf_headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                size_info = data.get("size", {}).get("dataset", {})
                original = size_info.get("num_bytes_original_files", 0)
                parquet = size_info.get("num_bytes_parquet_files", 0)
                memory = size_info.get("num_bytes_memory", 0)
                rows = size_info.get("num_rows", 0)
                total_bytes = original or parquet
                print("[dataset_size] using PRIMARY: dataset_viewer API")
                return {
                    "total_bytes": total_bytes,
                    "total_gb": total_bytes / (1024**3) if total_bytes > 0 else 0,
                    "num_files": len(data.get("size", {}).get("splits", [])),
                    "num_rows": rows,
                    "memory_size_gb": memory / (1024**3) if memory > 0 else 0,
                    "is_partial": data.get("partial", False),
                    "api_method": "dataset_viewer",
                }
            else:
                print(
                    f"[dataset_size] PRIMARY failed with status {response.status_code}; trying FALLBACK: huggingface_hub"
                )
        except Exception as e:
            print(
                f"[dataset_size] PRIMARY error ({e}); trying FALLBACK: huggingface_hub"
            )

        try:
            from huggingface_hub import HfApi

            api = HfApi()
            ds_info = api.dataset_info(repo_id=dataset_id, files_metadata=True)
            total_size_bytes = 0
            file_count = 0
            for sibling in ds_info.siblings:
                total_size_bytes += sibling.size or 0
                file_count += 1
            print("[dataset_size] using FALLBACK: huggingface_hub")
            return {
                "total_bytes": total_size_bytes,
                "total_gb": total_size_bytes / (1024**3) if total_size_bytes > 0 else 0,
                "num_files": file_count,
                "num_rows": 0,  # not provided by hub
                "memory_size_gb": 0,  # not provided by hub
                "is_partial": False,
                "api_method": "huggingface_hub",
            }
        except Exception as e:
            print(f"[dataset_size] FALLBACK error: {e}")
            return {
                "error": f"Dataset size could not be determined: {str(e)}",
                "total_bytes": 0,
                "total_gb": 0,
                "num_files": 0,
                "num_rows": 0,
                "memory_size_gb": 0,
                "is_partial": False,
                "api_method": "error",
            }

    def _extract_contributors(
        self, info: dict[str, Any], id_fallback: str
    ) -> list[str]:
        """Extract contributors from HF API response"""
        if "author" in info and info["author"]:
            return [info["author"]]
        else:
            # Use organization name as fallback
            return [id_fallback.split("/")[0]]

    def _extract_github_license(self, repo_data: dict[str, Any]) -> str:
        """Extract license from GitHub repo data"""
        license_info = repo_data.get("license")
        if license_info and isinstance(license_info, dict):
            return license_info.get("spdx_id", "")
        return ""


if __name__ == "__main__":
    # Run tests
    test_url_categorization()
    test_data_fetching()
