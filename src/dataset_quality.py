"""
Dataset quality assessment module for trustworthiness scoring.
"""

from typing import Dict, Any
import time


def calculate_dataset_quality_with_timing(data: Dict[str, Any], downloads: int, likes: int) -> tuple[float, int]:
    """
    Calculate dataset quality with latency measurement.
    
    Args:
        data: Dataset metadata from API
        downloads: Number of downloads
        likes: Number of likes
        
    Returns:
        tuple of (quality_score, latency_ms)
    """
    start_time = time.perf_counter()
    quality_score = calculate_dataset_quality(data, downloads, likes)
    end_time = time.perf_counter()
    latency_ms = int((end_time - start_time) * 1000)
    return quality_score, latency_ms


def calculate_dataset_quality(data: Dict[str, Any], downloads: int, likes: int) -> float:
    """
    Calculate comprehensive dataset quality score (0.0 to 1.0).
    
    Args:
        data: Dataset metadata from API
        downloads: Number of downloads
        likes: Number of likes
        
    Returns:
        Dataset quality score between 0.0 and 1.0
    """
    # Import license score map from scorer module
    from .scorer import license_score_map
    
    quality_factors = {}
    
    # 1. Usage and Popularity (25% weight)
    usage_score = 0.0
    if downloads > 1000000:
        usage_score = 1.0
    elif downloads > 100000:
        usage_score = 0.8
    elif downloads > 10000:
        usage_score = 0.6
    elif downloads > 1000:
        usage_score = 0.4
    elif downloads > 100:
        usage_score = 0.2
    else:
        usage_score = 0.1
    
    # Add likes factor
    if likes > 100:
        usage_score = min(1.0, usage_score + 0.2)
    elif likes > 20:
        usage_score = min(1.0, usage_score + 0.1)
    
    quality_factors['usage'] = usage_score
    
    # 2. Documentation Quality (20% weight)
    doc_score = 0.0
    has_description = bool(data.get('description', '').strip())
    has_card = bool(data.get('cardData'))
    has_readme = bool(data.get('readme'))
    
    if has_description:
        doc_score += 0.3
    if has_card:
        doc_score += 0.4
    if has_readme:
        doc_score += 0.3
    
    quality_factors['documentation'] = min(1.0, doc_score)
    
    # 3. Licensing (15% weight)
    license_score = 0.0
    license_str = data.get('cardData', {}).get('license', 'unknown') if data.get('cardData') else 'unknown'
    if isinstance(license_str, list):
        license_str = license_str[0]
    
    if license_str and license_str != 'unknown' and license_str != 'other':
        license_score = license_score_map.get(license_str, 0.0)
    else:
        license_score = 0.0  # No license or unknown license
    
    quality_factors['licensing'] = license_score
    
    # 4. Data Completeness Indicators (15% weight)
    completeness_score = 0.0
    
    # Check for dataset configuration files
    config_files = data.get('siblings', [])
    has_config = any('config' in str(sibling.get('rfilename', '')).lower() for sibling in config_files)
    has_schema = any('schema' in str(sibling.get('rfilename', '')).lower() for sibling in config_files)
    has_metadata = any('metadata' in str(sibling.get('rfilename', '')).lower() for sibling in config_files)
    
    if has_config:
        completeness_score += 0.4
    if has_schema:
        completeness_score += 0.3
    if has_metadata:
        completeness_score += 0.3
    
    quality_factors['completeness'] = min(1.0, completeness_score)
    
    # 5. Community Engagement (10% weight)
    engagement_score = 0.0
    tags = data.get('tags', [])
    has_tags = len(tags) > 0
    has_paper = any('paper' in tag.lower() or 'arxiv' in tag.lower() for tag in tags)
    has_benchmark = any('benchmark' in tag.lower() for tag in tags)
    
    if has_tags:
        engagement_score += 0.3
    if has_paper:
        engagement_score += 0.4
    if has_benchmark:
        engagement_score += 0.3
    
    quality_factors['engagement'] = min(1.0, engagement_score)
    
    # 6. Dataset Size Appropriateness (10% weight)
    size_score = 0.0
    dataset_size = data.get('downloads', 0)
    
    # Size appropriateness based on usage
    if dataset_size > 1000000:  # Very popular datasets
        size_score = 1.0
    elif dataset_size > 100000:  # Popular datasets
        size_score = 0.8
    elif dataset_size > 10000:  # Moderately used datasets
        size_score = 0.6
    elif dataset_size > 1000:  # Small but used datasets
        size_score = 0.4
    else:  # Very small or unused datasets
        size_score = 0.2
    
    quality_factors['size_appropriateness'] = size_score
    
    # 7. Update Frequency (5% weight)
    update_score = 0.5  # Default neutral score
    last_modified = data.get('lastModified')
    if last_modified:
        try:
            from datetime import datetime
            # Parse the date and calculate recency
            # This is a simplified approach - you might want to use proper date parsing
            update_score = 0.8  # Assume recent if we have a date
        except:
            update_score = 0.5
    
    quality_factors['freshness'] = update_score
    
    # Calculate weighted average
    weights = {
        'usage': 0.25,
        'documentation': 0.20,
        'licensing': 0.15,
        'completeness': 0.15,
        'engagement': 0.10,
        'size_appropriateness': 0.10,
        'freshness': 0.05
    }
    
    total_score = sum(quality_factors[factor] * weights[factor] for factor in weights)
    
    # Ensure score is between 0.0 and 1.0
    return max(0.0, min(1.0, total_score))
