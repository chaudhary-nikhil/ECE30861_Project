"""
Net score calculation module for computing the final weighted score.

This module provides functionality to calculate the net score as a weighted sum of:
- ramp_up_time (20%)
- license (15%)
- code_quality (15%)
- dataset_and_code_score (10%)
- dataset_quality (10%)
- performance_claims (10%)
- bus_factor (10%)
- size_score (10%)

Based on Sarah's priorities for ease of ramp-up, legal compliance, and model quality.
"""

from typing import Dict, Any, Tuple
import time
from .log import loggerInstance


def calculate_net_score_with_timing(metrics: Dict[str, Any]) -> Tuple[float, int]:
    """
    Calculate net score as a weighted sum of all metrics.
    
    Note: This function no longer measures timing since net_score_latency
    is now measured at the scoring function level to include all individual
    metric calculation time.
    
    Args:
        metrics: Dictionary containing all individual metric scores
        
    Returns:
        tuple of (net_score, latency_ms) where latency_ms is always 0
    """
    
    try:
        # Define weights based on Sarah's priorities
        weights = {
            'ramp_up_time': 0.20,           # Sarah explicitly mentions this first — critical for engineers
            'license': 0.15,                # Essential for legal compliance
            'code_quality': 0.15,           # Affects maintainability and onboarding
            'dataset_and_code_score': 0.10, # Availability of training data/code is important
            'dataset_quality': 0.10,        # Impacts trustworthiness of model
            'performance_claims': 0.10,     # Validating performance claims builds trust
            'bus_factor': 0.10,             # Maintainer responsiveness matters
            'size_score': 0.10              # Deployment feasibility on target hardware
        }
        
        # Verify weights sum to 1.0
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.001:  # Allow for small floating point errors
            if hasattr(loggerInstance, 'logger') and loggerInstance.logger:
                loggerInstance.logger.log_info(f"Warning: Weights sum to {total_weight}, not 1.0")
        
        # Extract individual metric scores with fallback defaults
        ramp_up_time = metrics.get('ramp_up_time', 0.0)
        license_score = metrics.get('license', 0.0)
        code_quality = metrics.get('code_quality', 0.0)
        dataset_and_code_score = metrics.get('dataset_and_code_score', 0.0)
        dataset_quality = metrics.get('dataset_quality', 0.0)
        performance_claims = metrics.get('performance_claims', 0.0)
        bus_factor = metrics.get('bus_factor', 0.0)
        
        # Handle size_score object - convert to scalar by averaging
        size_score_obj = metrics.get('size_score', {})
        if isinstance(size_score_obj, dict) and size_score_obj:
            # Calculate average across all hardware targets
            size_scores = list(size_score_obj.values())
            size_score_avg = sum(size_scores) / len(size_scores)
        else:
            # Fallback if size_score is not a dict or is empty
            size_score_avg = 0.0
        
        # Calculate weighted sum
        net_score = (
            ramp_up_time * weights['ramp_up_time'] +
            license_score * weights['license'] +
            code_quality * weights['code_quality'] +
            dataset_and_code_score * weights['dataset_and_code_score'] +
            dataset_quality * weights['dataset_quality'] +
            performance_claims * weights['performance_claims'] +
            bus_factor * weights['bus_factor'] +
            size_score_avg * weights['size_score']
        )
        
        # Clamp result between 0 and 1 to avoid floating point rounding issues
        net_score = max(0.0, min(1.0, net_score))
        
        # Log the calculation details for debugging
        if hasattr(loggerInstance, 'logger') and loggerInstance.logger:
            loggerInstance.logger.log_info(f"Net score calculation: {net_score:.3f}")
            loggerInstance.logger.log_info(f"  - ramp_up_time: {ramp_up_time:.3f} (weight: {weights['ramp_up_time']})")
            loggerInstance.logger.log_info(f"  - license: {license_score:.3f} (weight: {weights['license']})")
            loggerInstance.logger.log_info(f"  - code_quality: {code_quality:.3f} (weight: {weights['code_quality']})")
            loggerInstance.logger.log_info(f"  - dataset_and_code_score: {dataset_and_code_score:.3f} (weight: {weights['dataset_and_code_score']})")
            loggerInstance.logger.log_info(f"  - dataset_quality: {dataset_quality:.3f} (weight: {weights['dataset_quality']})")
            loggerInstance.logger.log_info(f"  - performance_claims: {performance_claims:.3f} (weight: {weights['performance_claims']})")
            loggerInstance.logger.log_info(f"  - bus_factor: {bus_factor:.3f} (weight: {weights['bus_factor']})")
            loggerInstance.logger.log_info(f"  - size_score_avg: {size_score_avg:.3f} (weight: {weights['size_score']})")
        
    except Exception as e:
        if hasattr(loggerInstance, 'logger') and loggerInstance.logger:
            loggerInstance.logger.log_info(f"Error calculating net score: {e}")
        net_score = 0.0
    
    # Return 0 latency since timing is now measured at the scoring function level
    return net_score, 0


def calculate_net_score(metrics: Dict[str, Any]) -> float:
    """
    Calculate net score as a weighted sum of all metrics (without timing).
    
    This is a convenience function that calls calculate_net_score_with_timing
    and returns only the score.
    
    Args:
        metrics: Dictionary containing all individual metric scores
        
    Returns:
        net_score as float between 0 and 1
    """
    net_score, _ = calculate_net_score_with_timing(metrics)
    return net_score
