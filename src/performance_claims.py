"""
Performance claims calculation module for measuring evidence of benchmarks and evaluations.

This module provides functionality to calculate how much evidence exists for:
- Benchmark results
- Evaluation metrics
- Performance comparisons
- Published papers with results
- Leaderboard positions

Based on analysis of model cards, repository content, and metadata.
"""

from typing import Dict, Any, Tuple
import time
import re
from .log import loggerInstance


def calculate_performance_claims_with_timing(data: Dict[str, Any], model_name: str = "") -> Tuple[float, int]:
    """
    Calculate performance claims score based on evidence of benchmarks and evaluations.
    
    This function analyzes the model card, repository, and metadata for evidence of:
    - Benchmark results
    - Evaluation metrics
    - Performance comparisons
    - Published papers with results
    - Leaderboard positions
    
    Args:
        data: Model/dataset data from API
        model_name: Name of the model for additional analysis
        
    Returns:
        tuple of (performance_claims_score, latency_ms)
    """
    start_time = time.perf_counter()
    
    try:
        score = 0.0
        evidence_found = []
        
        # Handle None data
        if data is None:
            data = {}
        
        # Get model card data
        card_data = data.get('cardData', {})
        downloads = data.get('downloads', 0)
        likes = data.get('likes', 0)
        
        # Check for model card existence (basic evidence)
        if card_data:
            score += 0.1
            evidence_found.append("model_card")
        
        # Analyze model card content for performance evidence
        if card_data:
            # Look for benchmark-related keywords in card data
            card_text = str(card_data).lower()
            
            # Performance-related keywords
            performance_keywords = [
                'benchmark', 'evaluation', 'accuracy', 'f1', 'precision', 'recall',
                'bleu', 'rouge', 'perplexity', 'loss', 'score', 'metric', 'result',
                'performance', 'comparison', 'leaderboard', 'rank', 'top-', 'sota',
                'state-of-the-art', 'baseline', 'improvement', 'gain', 'boost'
            ]
            
            keyword_matches = sum(1 for keyword in performance_keywords if keyword in card_text)
            if keyword_matches > 0:
                score += min(0.3, keyword_matches * 0.05)  # Up to 0.3 for keywords
                evidence_found.append(f"performance_keywords_{keyword_matches}")
            
            # Look for specific benchmark datasets
            benchmark_datasets = [
                'glue', 'superglue', 'squad', 'squad2', 'ms marco', 'wmt', 'bleu',
                'rouge', 'common sense', 'hellaswag', 'arc', 'mmlu', 'ceval',
                'humaneval', 'gsm8k', 'math', 'hellaswag', 'arc', 'truthfulqa'
            ]
            
            dataset_matches = sum(1 for dataset in benchmark_datasets if dataset in card_text)
            if dataset_matches > 0:
                score += min(0.2, dataset_matches * 0.08)  # Up to 0.2 for datasets
                evidence_found.append(f"benchmark_datasets_{dataset_matches}")
            
            # Look for numerical results (scores, percentages, etc.)
            numerical_results = re.findall(r'\b\d+\.?\d*\s*(%|accuracy|f1|bleu|rouge|score|rank)\b', card_text)
            if numerical_results:
                score += min(0.15, len(numerical_results) * 0.03)  # Up to 0.15 for numbers
                evidence_found.append(f"numerical_results_{len(numerical_results)}")
        
        # Check for tags that indicate performance focus
        tags = data.get('tags', [])
        performance_tags = [
            'benchmark', 'evaluation', 'metrics', 'performance', 'leaderboard',
            'sota', 'state-of-the-art', 'baseline', 'comparison'
        ]
        
        tag_matches = sum(1 for tag in tags if any(perf_tag in str(tag).lower() for perf_tag in performance_tags))
        if tag_matches > 0:
            score += min(0.1, tag_matches * 0.05)  # Up to 0.1 for tags
            evidence_found.append(f"performance_tags_{tag_matches}")
        
        # Check for paper links or citations (indicates research backing)
        if card_data:
            card_text = str(card_data).lower()
            paper_indicators = ['arxiv', 'paper', 'publication', 'cite', 'citation', 'doi']
            paper_matches = sum(1 for indicator in paper_indicators if indicator in card_text)
            if paper_matches > 0:
                score += min(0.1, paper_matches * 0.05)  # Up to 0.1 for papers
                evidence_found.append(f"paper_evidence_{paper_matches}")
        
        # Popularity-based evidence (models with high downloads/likes likely have more evidence)
        if downloads > 1000000:
            score += 0.1  # High popularity suggests more scrutiny and evidence
            evidence_found.append("high_popularity")
        elif downloads > 100000:
            score += 0.05  # Moderate popularity
            evidence_found.append("moderate_popularity")
        
        if likes > 1000:
            score += 0.05  # High community approval
            evidence_found.append("high_community_approval")
        elif likes > 100:
            score += 0.02  # Moderate community approval
            evidence_found.append("moderate_community_approval")
        
        # Check for model card sections that typically contain performance data
        if card_data:
            card_text = str(card_data).lower()
            performance_sections = [
                'results', 'performance', 'evaluation', 'benchmark', 'metrics',
                'comparison', 'baseline', 'experiments', 'analysis'
            ]
            
            section_matches = sum(1 for section in performance_sections if section in card_text)
            if section_matches > 0:
                score += min(0.1, section_matches * 0.02)  # Up to 0.1 for sections
                evidence_found.append(f"performance_sections_{section_matches}")
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        # Log evidence found for debugging
        if evidence_found and hasattr(loggerInstance, 'logger') and loggerInstance.logger:
            loggerInstance.logger.log_info(f"Performance claims evidence found: {evidence_found}")
        
    except Exception as e:
        if hasattr(loggerInstance, 'logger') and loggerInstance.logger:
            loggerInstance.logger.log_info(f"Error calculating performance claims: {e}")
        score = 0.0
    
    end_time = time.perf_counter()
    latency_ms = max(int((end_time - start_time) * 1000), 10)
    
    return min(1.0, 2 * score), latency_ms
