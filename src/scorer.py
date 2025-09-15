"""
Simplified scoring framework for datasets, models, and code.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import requests
import re
from .url import UrlCategory

@dataclass
class ScoreResult:
    """Container for scoring results."""
    url: str
    category: UrlCategory
    score: float
    max_score: float
    details: Dict[str, Any]
    
    @property
    def percentage(self) -> float:
        """Get score as percentage."""
        return (self.score / self.max_score) * 100 if self.max_score > 0 else 0.0
    
    def __str__(self) -> str:
        return f"{self.category}: {self.score:.1f}/{self.max_score:.1f} ({self.percentage:.1f}%)"


def make_request(url: str) -> Optional[Dict]:
    """Make HTTP request with error handling."""
    try:
        response = requests.get(url, headers={'User-Agent': 'Trustworthy-Model-Reuse-CLI/1.0'}, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def score_dataset(url: str) -> ScoreResult:
    """Score a Hugging Face dataset."""
    # Extract dataset name
    match = re.search(r"https://huggingface\.co/datasets/((\w+\/?)+)", url)
    if not match:
        return ScoreResult(url, "DATASET", 0.0, 10.0, {'error': 'Invalid URL'})
    
    dataset_name = match.group(1)
    api_url = f"https://huggingface.co/api/datasets/{dataset_name}"
    data = make_request(api_url)
    
    if not data:
        return ScoreResult(url, "DATASET", 0.0, 10.0, {'fallback': True})
    
    # Simple scoring based on key metrics
    downloads = data.get('downloads', 0)
    likes = data.get('likes', 0)
    has_description = bool(data.get('description'))
    
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
    
    return ScoreResult(url, "DATASET", min(score, 10.0), 10.0, {
        'downloads': downloads,
        'likes': likes,
        'has_description': has_description
    })


def score_model(url: str) -> ScoreResult:
    """Score a Hugging Face model."""
    # Extract model name
    match = re.search(r"https://huggingface\.co/([^/]+/[^/]+)", url)
    if not match:
        return ScoreResult(url, "MODEL", 0.0, 10.0, {'error': 'Invalid URL'})
    
    model_name = match.group(1)
    api_url = f"https://huggingface.co/api/models/{model_name}"
    data = make_request(api_url)
    
    if not data:
        return ScoreResult(url, "MODEL", 2.0, 10.0, {'fallback': True})
    
    # Simple scoring based on key metrics
    downloads = data.get('downloads', 0)
    likes = data.get('likes', 0)
    has_card = bool(data.get('cardData'))
    pipeline_tag = data.get('pipeline_tag')
    
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
    
    return ScoreResult(url, "MODEL", min(score, 10.0), 10.0, {
        'downloads': downloads,
        'likes': likes,
        'has_model_card': has_card,
        'pipeline_tag': pipeline_tag
    })


def score_code(url: str) -> ScoreResult:
    """Score a GitHub repository."""
    # Extract repo info
    match = re.search(r"https://github\.com/([^/]+)/([^/]+)", url)
    if not match:
        return ScoreResult(url, "CODE", 0.0, 10.0, {'error': 'Invalid URL'})
    
    owner, repo = match.groups()
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    data = make_request(api_url)
    
    if not data:
        return ScoreResult(url, "CODE", 2.0, 10.0, {'fallback': True})
    
    # Simple scoring based on key metrics
    stars = data.get('stargazers_count', 0)
    forks = data.get('forks_count', 0)
    has_description = bool(data.get('description'))
    has_license = bool(data.get('license'))
    language = data.get('language')
    
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
    
    return ScoreResult(url, "CODE", min(score, 10.0), 10.0, {
        'stars': stars,
        'forks': forks,
        'has_description': has_description,
        'has_license': has_license,
        'language': language
    })


def score_url(url: str, category: UrlCategory) -> ScoreResult:
    """Score a URL based on its category."""
    if category == UrlCategory.DATASET:
        return score_dataset(url)
    elif category == UrlCategory.MODEL:
        return score_model(url)
    elif category == UrlCategory.CODE:
        return score_code(url)
    else:
        return ScoreResult(url, "INVALID", 0.0, 10.0, {'error': 'Invalid category'})
