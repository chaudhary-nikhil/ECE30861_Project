"""
Simplified scoring framework for datasets, models, and code.
"""

from typing import Any, Optional
from dataclasses import dataclass
import requests
import re
import os
import tempfile
import shutil
import json
from pathlib import Path
from .url import UrlCategory

# Try to import GitPython, fall back to subprocess if not available
try:
    from git import Repo

    GIT_PYTHON_AVAILABLE = True
except ImportError:
    GIT_PYTHON_AVAILABLE = False
    import subprocess


@dataclass
class ScoreResult:
    url: str
    category: UrlCategory
    score: float
    max_score: float
    details: dict[str, Any]

    @property
    def percentage(self) -> float:
        """Get score as percentage."""
        return (self.score / self.max_score) * 100 if self.max_score > 0 else 0.0

    def __str__(self) -> str:
        return f"{self.category}: {self.score:.1f}/{self.max_score:.1f} ({self.percentage:.1f}%)"


def make_request(url: str) -> Optional[dict]:
    """Make HTTP request with error handling."""
    try:
        response = requests.get(
            url, headers={"User-Agent": "Trustworthy-Model-Reuse-CLI/1.0"}, timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def calculate_size_score(model_size_mb: float) -> dict[str, float]:
    """
    Calculate size_score based on model size using piecewise linear mapping.

    Args:
        model_size_mb: Model size in megabytes

    Returns:
        dictionary mapping hardware types to compatibility scores [0,1]
    """
    # Hardware capacity thresholds (in MB)
    thresholds = {
        "raspberry_pi": {
            "min": 0,
            "max": 200,
        },  # 0-200MB full score, taper to 0 at 1GB+
        "jetson_nano": {"min": 0, "max": 500},  # 0-500MB full score, taper to 0 at 4GB+
        "desktop_pc": {"min": 0, "max": 5000},  # 0-5GB full score, taper to 0 at 20GB+
        "aws_server": {"min": 0, "max": 50000},  # Near 1 unless extreme (100GB+)
    }

    size_score = {}

    for hardware, threshold in thresholds.items():
        if model_size_mb <= threshold["min"]:
            score = 1.0
        elif model_size_mb >= threshold["max"]:
            score = 0.0
        else:
            # Piecewise linear mapping: score = max(0, 1 - (size - min) / (max - min))
            score = max(
                0.0,
                1.0
                - (model_size_mb - threshold["min"])
                / (threshold["max"] - threshold["min"]),
            )

        size_score[hardware] = round(score, 2)

    return size_score


def analyze_model_repository(
    model_name: str, model_type: str = "model"
) -> dict[str, Any]:
    """
    Clone and analyze a model repository to determine actual size and characteristics.

    Args:
        model_name: Name of the model (e.g., "google/gemma-3-270m")
        model_type: Type of model ("model", "dataset", "code")

    Returns:
        dictionary containing analysis results
    """
    analysis = {
        "total_size_mb": 0.0,
        "weights_size_mb": 0.0,
        "config_info": {},
        "has_tokenizer": False,
        "architecture": "unknown",
        "model_files": [],
        "error": None,
    }

    ## Create temporary directory for cloning
    # temp_dir = tempfile.mkdtemp()
    #
    # try:
    #    # Determine the repository URL based on type
    #    if model_type == "model":
    #        repo_url = f"https://huggingface.co/{model_name}"
    #    elif model_type == "dataset":
    #        repo_url = f"https://huggingface.co/datasets/{model_name}"
    #    else:  # code
    #        repo_url = f"https://github.com/{model_name}"
    #
    #    # Clone the repository using GitPython or subprocess fallback
    #    print(f"Cloning repository: {repo_url}")
    #
    #    if GIT_PYTHON_AVAILABLE:
    #        # Use GitPython for cloning
    #        repo = Repo.clone_from(repo_url, temp_dir)
    #    else:
    #        # Fallback to subprocess (not ideal but works)
    #        print("GitPython not available, using subprocess fallback")
    #        result = subprocess.run(['git', 'clone', repo_url, temp_dir],
    #                             capture_output=True, text=True, timeout=60)
    #        if result.returncode != 0:
    #            raise Exception(f"Git clone failed: {result.stderr}")
    #
    #    # Analyze the cloned repository
    #    analysis = _analyze_model_files(temp_dir, model_name, model_type)
    #
    # except Exception as e:
    #    analysis['error'] = f"Failed to clone repository: {str(e)}"
    #    print(f"Repository cloning failed: {e}")
    # finally:
    #    # Clean up temporary directory
    #    if os.path.exists(temp_dir):
    #        shutil.rmtree(temp_dir)

    return analysis


def _analyze_model_files(
    repo_path: str, model_name: str, model_type: str
) -> dict[str, Any]:
    """
    Analyze model files in the cloned repository.

    Args:
        repo_path: Path to the cloned repository
        model_name: Name of the model
        model_type: Type of model

    Returns:
        Analysis results
    """
    analysis = {
        "total_size_mb": 0.0,
        "weights_size_mb": 0.0,
        "config_info": {},
        "has_tokenizer": False,
        "architecture": "unknown",
        "model_files": [],
        "error": None,
    }

    try:
        print(f"Analyzing files in: {repo_path}")

        # Walk through the repository and analyze files
        total_size = 0
        weights_size = 0
        model_files = []
        tokenizer_files = []
        config_files = []

        for root, dirs, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_size_mb = os.path.getsize(file_path) / (
                    1024 * 1024
                )  # Convert to MB
                total_size += file_size_mb

                # Check for model weight files
                if file in [
                    "pytorch_model.bin",
                    "tf_model.h5",
                    "model.safetensors",
                    "pytorch_model-00001-of-00001.bin",
                ]:
                    weights_size += file_size_mb
                    model_files.append(
                        {
                            "name": file,
                            "size_mb": round(file_size_mb, 2),
                            "path": file_path,
                        }
                    )

                # Check for tokenizer files
                elif file in [
                    "tokenizer.json",
                    "vocab.txt",
                    "tokenizer_config.json",
                    "tokenizer.model",
                ]:
                    tokenizer_files.append(
                        {"name": file, "size_mb": round(file_size_mb, 2)}
                    )

                # Check for config files
                elif file in ["config.json", "model_index.json", "README.md"]:
                    config_files.append(
                        {"name": file, "size_mb": round(file_size_mb, 2)}
                    )

        # Update analysis results
        analysis["total_size_mb"] = round(total_size, 2)
        analysis["weights_size_mb"] = round(weights_size, 2)
        analysis["model_files"] = model_files
        analysis["has_tokenizer"] = len(tokenizer_files) > 0

        # Analyze config.json if present
        config_path = os.path.join(repo_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                    analysis["config_info"] = config_data

                    # Extract architecture information
                    if "model_type" in config_data:
                        analysis["architecture"] = config_data["model_type"]
                    elif "architectures" in config_data:
                        analysis["architecture"] = (
                            config_data["architectures"][0]
                            if config_data["architectures"]
                            else "unknown"
                        )
            except Exception as e:
                print(f"Failed to parse config.json: {e}")

        print(
            f"Analysis complete: {analysis['total_size_mb']}MB total, {analysis['weights_size_mb']}MB weights"
        )

    except Exception as e:
        analysis["error"] = str(e)
        print(f"File analysis failed: {e}")

    return analysis


def estimate_model_size(model_name: str, model_type: str = "model") -> float:
    """
    Estimate model size using repository analysis only.
    No API fallback - purely repository-based analysis.

    Args:
        model_name: Name of the model (e.g., "google/gemma-3-270m")
        model_type: Type of model ("model", "dataset", "code")

    Returns:
        Estimated model size in MB
    """
    if not model_name or model_name == "unknown":
        # Default size for unknown models
        return 500  # Default medium size

    # Perform repository analysis
    analysis = analyze_model_repository(model_name, model_type)

    if analysis["error"] is None:
        return analysis["total_size_mb"]
    else:
        # If repository analysis fails, use conservative default
        # This ensures we don't rely on API data
        return 1000  # Conservative default for failed analysis


def score_dataset(url: str) -> ScoreResult:
    """Score a Hugging Face dataset."""
    # Extract dataset name
    match = re.search(r"https://huggingface\.co/datasets/((\w+\/?)+)", url)
    if not match:
        estimated_size = estimate_model_size("unknown", "dataset")
        size_score = calculate_size_score(estimated_size)
        return ScoreResult(
            url,
            UrlCategory.DATASET,
            0.0,
            10.0,
            {"error": "Invalid URL", "name": "unknown", "size_score": size_score},
        )

    dataset_name = match.group(1)
    api_url = f"https://huggingface.co/api/datasets/{dataset_name}"
    data = make_request(api_url)

    if not data:
        # Use repository analysis for fallback
        estimated_size = estimate_model_size(dataset_name, "dataset")
        size_score = calculate_size_score(estimated_size)
        return ScoreResult(
            url,
            UrlCategory.DATASET,
            0.0,
            10.0,
            {"name": dataset_name, "fallback": True, "size_score": size_score},
        )

    # Simple scoring based on key metrics
    downloads = data.get("downloads", 0)
    likes = data.get("likes", 0)
    has_description = bool(data.get("description"))

    score = 2.0  # Base score
    if downloads > 10000:
        score += 3.0
    elif downloads > 1000:
        score += 2.0
    elif downloads > 100:
        score += 1.0

    if likes > 50:
        score += 2.0
    elif likes > 10:
        score += 1.0

    if has_description:
        score += 2.0

    # Calculate dynamic size_score based on repository analysis only
    estimated_size = estimate_model_size(dataset_name, "dataset")
    size_score = calculate_size_score(estimated_size)

    return ScoreResult(
        url,
        UrlCategory.DATASET,
        min(score, 10.0),
        10.0,
        {
            "name": dataset_name,
            "downloads": downloads,
            "likes": likes,
            "has_description": has_description,
            "size_score": size_score,
        },
    )


def score_model(url: str) -> ScoreResult:
    """Score a Hugging Face model."""
    # Extract model name
    match = re.search(r"https://huggingface\.co/([^/]+/[^/]+)", url)
    if not match:
        estimated_size = estimate_model_size("unknown", "model")
        size_score = calculate_size_score(estimated_size)
        return ScoreResult(
            url,
            UrlCategory.MODEL,
            0.0,
            10.0,
            {"error": "Invalid URL", "name": "unknown", "size_score": size_score},
        )

    model_name = match.group(1)
    api_url = f"https://huggingface.co/api/models/{model_name}"
    data = make_request(api_url)

    if not data:
        estimated_size = estimate_model_size(model_name, "model")
        size_score = calculate_size_score(estimated_size)
        return ScoreResult(
            url,
            UrlCategory.MODEL,
            2.0,
            10.0,
            {"name": model_name, "fallback": True, "size_score": size_score},
        )

    # Simple scoring based on key metrics
    downloads = data.get("downloads", 0)
    likes = data.get("likes", 0)
    has_card = bool(data.get("cardData"))
    pipeline_tag = data.get("pipeline_tag")

    score = 2.0  # Base score
    if downloads > 100000:
        score += 3.0
    elif downloads > 10000:
        score += 2.0
    elif downloads > 1000:
        score += 1.0

    if likes > 100:
        score += 2.0
    elif likes > 20:
        score += 1.0

    if has_card:
        score += 2.0

    if pipeline_tag:
        score += 1.0

    # Calculate dynamic size_score based on repository analysis only
    estimated_size = estimate_model_size(model_name, "model")
    size_score = calculate_size_score(estimated_size)

    return ScoreResult(
        url,
        UrlCategory.MODEL,
        min(score, 10.0),
        10.0,
        {
            "name": model_name,
            "downloads": downloads,
            "likes": likes,
            "has_model_card": has_card,
            "pipeline_tag": pipeline_tag,
            "size_score": size_score,
        },
    )


def score_code(url: str) -> ScoreResult:
    """Score a GitHub repository."""
    # Extract repo info
    match = re.search(r"https://github\.com/([^/]+)/([^/]+)", url)
    if not match:
        estimated_size = estimate_model_size("unknown", "code")
        size_score = calculate_size_score(estimated_size)
        return ScoreResult(
            url,
            UrlCategory.CODE,
            0.0,
            10.0,
            {"error": "Invalid URL", "name": "unknown", "size_score": size_score},
        )

    owner, repo = match.groups()
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    data = make_request(api_url)

    if not data:
        estimated_size = estimate_model_size(f"{owner}/{repo}", "code")
        size_score = calculate_size_score(estimated_size)
        return ScoreResult(
            url,
            UrlCategory.CODE,
            2.0,
            10.0,
            {"name": f"{owner}/{repo}", "fallback": True, "size_score": size_score},
        )

    # Simple scoring based on key metrics
    stars = data.get("stargazers_count", 0)
    forks = data.get("forks_count", 0)
    has_description = bool(data.get("description"))
    has_license = bool(data.get("license"))
    language = data.get("language")

    score = 2.0  # Base score
    if stars > 1000:
        score += 3.0
    elif stars > 100:
        score += 2.0
    elif stars > 10:
        score += 1.0

    if forks > 100:
        score += 1.0
    elif forks > 10:
        score += 0.5

    if has_description:
        score += 2.0

    if has_license:
        score += 1.0

    if language:
        score += 1.0

    # Calculate dynamic size_score based on repository analysis only
    estimated_size = estimate_model_size(f"{owner}/{repo}", "code")
    size_score = calculate_size_score(estimated_size)

    return ScoreResult(
        url,
        UrlCategory.CODE,
        min(score, 10.0),
        10.0,
        {
            "name": f"{owner}/{repo}",
            "stars": stars,
            "forks": forks,
            "has_description": has_description,
            "has_license": has_license,
            "language": language,
            "size_score": size_score,
        },
    )


def score_url(url: str, category: UrlCategory) -> ScoreResult:
    """Score a URL based on its category."""
    if category == UrlCategory.DATASET:
        return score_dataset(url)
    elif category == UrlCategory.MODEL:
        return score_model(url)
    elif category == UrlCategory.CODE:
        return score_code(url)
    else:
        estimated_size = estimate_model_size("unknown", "invalid")
        size_score = calculate_size_score(estimated_size)
        return ScoreResult(
            url,
            UrlCategory.INVALID,
            0.0,
            10.0,
            {"error": "Invalid category", "name": "unknown", "size_score": size_score},
        )
