"""
Simplified scoring framework for datasets, models, and code.
"""

from typing import Any, Optional, Dict
from dataclasses import dataclass
import requests
import re
from .url import UrlCategory
import time
from .integrated_data_fetcher import IntegratedDataFetcher



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


def estimate_model_size(model_name: str, model_type: str = "model") -> float:
    """
    Estimate model size with a conservative default.

    Args:
        model_name: Name of the model (e.g., "google/gemma-3-270m")
        model_type: Type of model ("model", "dataset", "code")

    Returns:
        Estimated model size in MB
    """
    if not model_name or model_name == "unknown":
        # Default size for unknown models
        return 500  # Default medium size

    # Use conservative default
    return 1000  # Conservative default

_data_fetcher = IntegratedDataFetcher()
def calculate_model_bus_factor(contributor_count: int) -> float:
    """Calculate bus factor for models based only on contributor count"""
    if contributor_count == 0:
        return 0.0
    elif contributor_count == 1:
        return 0.3
    elif contributor_count <= 3:
        return 0.6
    else:
        return 1.0

def calculate_dataset_bus_factor(contributor_count: int) -> float:
    """Calculate bus factor for datasets based only on contributor count"""
    if contributor_count == 0:
        return 0.0
    elif contributor_count == 1:
        return 0.4
    elif contributor_count <= 2:
        return 0.7
    else:
        return 1.0

def calculate_code_bus_factor(contributor_count: int) -> float:
    """Calculate bus factor for code repos based only on contributor count"""
    if contributor_count == 0:
        return 0.0
    elif contributor_count == 1:
        return 0.2
    elif contributor_count <= 3:
        return 0.4
    elif contributor_count <= 10:
        return 0.7
    else:
        return 1.0


def calculate_bus_factor_with_timing(url: str, category: UrlCategory, data: Dict[str, Any]) -> tuple:
    """Calculate bus factor with latency measurement"""
    
    start_time = time.time()
    
    # Extract contributor count from IntegratedDataFetcher data
    contributors = data.get('contributors', [])
    contributor_count = len(contributors) if contributors else 0
    
     # DEBUG: Print contributor information
    # print(f"DEBUG - URL: {url}")
    # print(f"DEBUG - Category: {category.name}")
    # print(f"DEBUG - Contributors found: {contributor_count}")
    # print(f"DEBUG - Contributor list: {contributors[:5]}")  # Show first 5
    # print(f"DEBUG - Raw data keys: {list(data.keys())}")
    # print("-" * 50)

    if category == UrlCategory.MODEL:
        score = calculate_model_bus_factor(contributor_count)
    elif category == UrlCategory.DATASET:
        score = calculate_dataset_bus_factor(contributor_count)
    elif category == UrlCategory.CODE:
        score = calculate_code_bus_factor(contributor_count)
    else:
        score = 0.0
    
    end_time = time.time()
    latency_ms = int((end_time - start_time) * 1000)
    
    return score, latency_ms


def score_dataset(url: str) -> ScoreResult:
    """Score a Hugging Face dataset."""
    # Extract dataset name
    match = re.search(r"https://huggingface\.co/datasets/([\w-]+(?:/[\w-]+)?)", url)
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

    # Calculate dynamic size_score
    estimated_size = estimate_model_size(dataset_name, "dataset")
    size_score = calculate_size_score(estimated_size)
    contributor_data = _data_fetcher.fetch_data(url)
    bus_factor_score, bus_factor_latency = calculate_bus_factor_with_timing(url, UrlCategory.DATASET, contributor_data)
    
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
            'bus_factor': bus_factor_score,
            'bus_factor_latency': bus_factor_latency
        },
    )


def score_model(url: str) -> ScoreResult:
    """Score a Hugging Face model."""
    # Extract model name
    match = re.search(r"https://huggingface\.co/([\w-]+/[\w-]+)", url)
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

    # Calculate dynamic size_score
    estimated_size = estimate_model_size(model_name, "model")
    size_score = calculate_size_score(estimated_size)
    contributor_data = _data_fetcher.fetch_data(url)
    bus_factor_score, bus_factor_latency = calculate_bus_factor_with_timing(url, UrlCategory.MODEL, contributor_data)

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
            'bus_factor': bus_factor_score,
            'bus_factor_latency': bus_factor_latency
        },
    )


def score_code(url: str) -> ScoreResult:
    """Score a GitHub repository."""
    # Extract repo info
    match = re.search(r"https://github\.com/([\w-]+)/([\w-]+)", url)
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

    # Calculate dynamic size_score
    estimated_size = estimate_model_size(f"{owner}/{repo}", "code")
    size_score = calculate_size_score(estimated_size)
    contributor_data = _data_fetcher.fetch_data(url)
    bus_factor_score, bus_factor_latency = calculate_bus_factor_with_timing(url, UrlCategory.CODE, contributor_data)
    
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
            'bus_factor': bus_factor_score,
            'bus_factor_latency': bus_factor_latency
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
