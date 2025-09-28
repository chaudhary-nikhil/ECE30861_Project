"""
Code quality calculation module for scoring models, datasets, and code.
"""

import time
import os
import subprocess
import tempfile
import re
from typing import Optional, Tuple, Dict, Any
import requests
from .log import loggerInstance


def run_flake8_on_repo(repo_path: str) -> tuple[float, int]:
    """
    Run Flake8 on a code repository and calculate quality score.
    
    Args:
        repo_path: Path to the repository directory
        
    Returns:
        tuple of (quality_score, latency_ms)
    """
    start_time = time.time()
    
    try:
        # First, check if there are any Python files in the repository
        python_files = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        # If no Python files found, return 0.0 (no code to analyze)
        if not python_files:
            quality_score = 0.0
        else:
            # Run flake8 on the repository
            result = subprocess.run(
                ['python3.9', '-m', 'flake8', repo_path, '--count', '--statistics'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse flake8 output to get error counts
            output_lines = result.stdout.strip().split('\n')
            total_errors = 0
            
            # Extract error count from the last line if it exists
            if output_lines and output_lines[-1]:
                try:
                    # The last line should contain just the total count (e.g., "369")
                    # If it's a number, that's our total error count
                    last_line = output_lines[-1].strip()
                    if last_line.isdigit():
                        total_errors = int(last_line)
                    else:
                        # Fallback: look for pattern like "1     E" or "5     W" etc.
                        error_match = re.search(r'(\d+)\s+[EW]', last_line)
                        if error_match:
                            total_errors = int(error_match.group(1))
                except (ValueError, IndexError):
                    pass
            
            # Calculate quality score based on error count
            # More lenient scoring that better reflects real-world code quality
            if total_errors == 0:
                quality_score = 1.0  # Perfect code (0 errors)
            elif total_errors <= 10:
                quality_score = 0.9  # Excellent (1-10 errors)
            elif total_errors <= 50:
                quality_score = 0.8  # Very good (11-50 errors)
            elif total_errors <= 100:
                quality_score = 0.7  # Good (51-100 errors)
            elif total_errors <= 200:
                quality_score = 0.6  # Fair (101-200 errors)
            elif total_errors <= 500:
                quality_score = 0.5  # Below average (201-500 errors)
            elif total_errors <= 1000:
                quality_score = 0.4  # Poor (501-1000 errors)
            elif total_errors <= 2000:
                quality_score = 0.3  # Very poor (1001-2000 errors)
            else:
                quality_score = 0.2  # Extremely poor (2000+ errors)
            
    except subprocess.TimeoutExpired:
        quality_score = 0.0
    except Exception:
        quality_score = 0.0
    
    end_time = time.time()
    latency_ms = int((end_time - start_time) * 1000)
    
    return quality_score, latency_ms


def find_code_repo_via_genai(model_name: str) -> Optional[str]:
    """
    Use Purdue GenAI Studio API to find code repository link in model README.
    
    Uses the official Purdue GenAI Studio API endpoint:
    https://genai.rcac.purdue.edu/api/chat/completions
    
    Args:
        model_name: Name of the model (e.g., "google-bert/bert-base-uncased")
        
    Returns:
        Code repository URL if found, None otherwise
    """
    try:
        # Purdue GenAI Studio API endpoint (from official documentation)
        api_url = "https://genai.rcac.purdue.edu/api/chat/completions"
        
        # Prepare the prompt to find code repository in model README
        prompt = f"""
        Please analyze the Hugging Face model "{model_name}" and find any code repository links in its README or documentation.
        
        Look for:
        - GitHub repository links
        - GitLab repository links  
        - Other code hosting platform links
        - Source code references
        
        Return only the URL if found, or "NO_CODE_FOUND" if no code repository is found.
        """
        
        headers = {
            "Authorization": "Bearer sk-ed2de44f587645c5b3fe62bb8f2328fc",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama3.1:latest",  # Using the model specified in Purdue GenAI Studio docs
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "stream": False,  # Non-streaming response as per documentation
            "max_tokens": 200,
            "temperature": 0.1
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            
            # Check if a valid URL was returned
            if content and content != "NO_CODE_FOUND":
                # Extract URLs from the response text
                url_pattern = r'https://(?:github\.com|gitlab\.com|bitbucket\.org|sourceforge\.net)/[^\s\)]+'
                urls = re.findall(url_pattern, content)
                
                if urls:
                    # Return the first valid URL found
                    return urls[0]
        else:
            loggerInstance.logger.log_info(f"Purdue GenAI Studio API returned status {response.status_code}: {response.text}")
        
        # Fallback: try to extract GitHub repo from model name
        if '/' in model_name:
            owner, repo = model_name.split('/', 1)
            github_url = f"https://github.com/{owner}/{repo}"
            
            # Test if the URL exists
            test_response = requests.get(github_url, timeout=5)
            if test_response.status_code == 200:
                return github_url
        
        return None
        
    except Exception as e:
        loggerInstance.logger.log_info(f"Error in find_code_repo_via_genai: {e}")
        return None


def calculate_code_quality_with_timing(code_url: Optional[str], model_name: str) -> tuple[float, int]:
    """
    Calculate code quality using Flake8, with fallback to GenAI API.
    
    Args:
        code_url: Direct code URL if available
        model_name: Model name for fallback search
        
    Returns:
        tuple of (quality_score, latency_ms)
    """
    start_time = time.time()
    
    # If we have a direct code URL, use it
    if code_url:
        try:
            # Clone the repository to a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_path = os.path.join(temp_dir, "repo")
                
                # Clone the repository
                subprocess.run(
                    ['git', 'clone', '--depth', '1', code_url, repo_path],
                    capture_output=True,
                    timeout=60
                )
                
                # Run flake8 on the cloned repository
                quality_score, _ = run_flake8_on_repo(repo_path)
                
        except Exception:
            quality_score = 0.0
    else:
        # Fallback: try to find code repo via GenAI API
        code_repo_url = find_code_repo_via_genai(model_name)
        
        if code_repo_url:
            try:
                # Clone and analyze the found repository
                with tempfile.TemporaryDirectory() as temp_dir:
                    repo_path = os.path.join(temp_dir, "repo")
                    
                    subprocess.run(
                        ['git', 'clone', '--depth', '1', code_repo_url, repo_path],
                        capture_output=True,
                        timeout=60
                    )
                    
                    quality_score, _ = run_flake8_on_repo(repo_path)
                    
            except Exception:
                quality_score = 0.0
        else:
            # No code repository found, return 0
            quality_score = 0.0
    
    end_time = time.time()
    latency_ms = int((end_time - start_time) * 1000)
    
    return quality_score, latency_ms


def calculate_code_quality(code_url: Optional[str], model_name: str) -> float:
    """
    Calculate code quality without timing.
    
    Args:
        code_url: Direct code URL if available
        model_name: Model name for fallback search
        
    Returns:
        quality score as float
    """
    # If we have a direct code URL, use it
    if code_url:
        try:
            # Clone the repository to a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_path = os.path.join(temp_dir, "repo")
                
                # Clone the repository
                subprocess.run(
                    ['git', 'clone', '--depth', '1', code_url, repo_path],
                    capture_output=True,
                    timeout=60
                )
                
                # Run flake8 on the cloned repository
                quality_score, _ = run_flake8_on_repo(repo_path)
                
        except Exception:
            quality_score = 0.0
    else:
        # Fallback: try to find code repo via GenAI API
        code_repo_url = find_code_repo_via_genai(model_name)
        
        if code_repo_url:
            try:
                # Clone and analyze the found repository
                with tempfile.TemporaryDirectory() as temp_dir:
                    repo_path = os.path.join(temp_dir, "repo")
                    
                    subprocess.run(
                        ['git', 'clone', '--depth', '1', code_repo_url, repo_path],
                        capture_output=True,
                        timeout=60
                    )
                    
                    quality_score, _ = run_flake8_on_repo(repo_path)
                    
            except Exception:
                quality_score = 0.0
        else:
            # No code repository found, return 0
            quality_score = 0.0
    
    return quality_score
