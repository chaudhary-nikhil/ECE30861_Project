"""
Ramp-up time calculation module for measuring documentation quality and onboarding clarity.

This module provides functionality to calculate how quickly an engineer can:
- Understand what the model does
- Set it up
- Run it successfully

Based on analysis of Hugging Face metadata, README content, and file presence.
"""

from typing import Dict, Any, Tuple
import time
from .log import loggerInstance


def calculate_ramp_up_time_with_timing(data: Dict[str, Any], model_name: str = "") -> Tuple[float, int]:
    """
    Calculate ramp-up time score based on documentation quality and onboarding clarity.
    
    This function analyzes:
    - Model card/README content quality
    - Presence of key onboarding files (requirements.txt, example scripts, config files)
    - Documentation completeness for quick setup and understanding
    
    Args:
        data: Model/dataset data from API
        model_name: Name of the model for additional analysis
        
    Returns:
        tuple of (ramp_up_time_score, latency_ms)
    """
    start_time = time.perf_counter()
    
    try:
        # Handle None data
        if data is None:
            data = {}
        
        # Get model card and file information
        card_data = data.get('cardData', {})
        readme = data.get('readme', '')
        files = data.get('files', [])
        siblings = data.get('siblings', [])
        downloads = data.get('downloads', 0)
        likes = data.get('likes', 0)
        
        # Use readme if cardData is not available
        if not card_data and readme:
            card_data = {'content': readme}
        
        # Handle files as dict (convert to list format expected by the function)
        if isinstance(files, dict):
            files_list = []
            for filename, file_info in files.items():
                if isinstance(file_info, dict):
                    files_list.append({
                        'rfilename': filename,
                        'filename': filename,
                        'size': file_info.get('size', 0),
                        'type': file_info.get('type', 'file')
                    })
            files = files_list
        
        # Combine files and siblings for comprehensive file analysis
        all_files = files + siblings if siblings else files
        
        # Step 1: File presence score (0.0 to 1.0)
        file_presence_score = 0.0
        file_evidence = []
        
        # Check for README/model card
        readme_found = False
        if card_data:
            file_presence_score += 0.3
            file_evidence.append("model_card")
            readme_found = True
        
        # Check for README.md in files
        for file_info in all_files:
            if isinstance(file_info, dict):
                filename = file_info.get('rfilename', '') or file_info.get('filename', '')
                if filename.lower() in ['readme.md', 'readme.txt', 'readme.rst']:
                    file_presence_score += 0.1  # Additional bonus for explicit README
                    file_evidence.append("readme_file")
                    readme_found = True
                    break
        
        # Check for requirements/environment files
        requirements_found = False
        for file_info in all_files:
            if isinstance(file_info, dict):
                filename = file_info.get('rfilename', '') or file_info.get('filename', '')
                if filename.lower() in ['requirements.txt', 'environment.yml', 'environment.yaml', 'pyproject.toml', 'setup.py']:
                    file_presence_score += 0.2
                    file_evidence.append("requirements_file")
                    requirements_found = True
                    break
        
        # Check for example/inference scripts
        example_found = False
        example_keywords = ['example', 'inference', 'demo', 'sample', 'quickstart', 'tutorial']
        for file_info in all_files:
            if isinstance(file_info, dict):
                filename = file_info.get('rfilename', '') or file_info.get('filename', '')
                filename_lower = filename.lower()
                if any(keyword in filename_lower for keyword in example_keywords):
                    if filename_lower.endswith(('.py', '.ipynb', '.md')):
                        file_presence_score += 0.3
                        file_evidence.append("example_script")
                        example_found = True
                        break
        
        # Check for config/tokenizer files (indicates completeness)
        config_found = False
        config_keywords = ['config', 'tokenizer', 'tokenizer_config', 'special_tokens_map', 'vocab']
        for file_info in all_files:
            if isinstance(file_info, dict):
                filename = file_info.get('rfilename', '') or file_info.get('filename', '')
                filename_lower = filename.lower()
                if any(keyword in filename_lower for keyword in config_keywords):
                    if filename_lower.endswith(('.json', '.txt', '.jsonl')):
                        file_presence_score += 0.2
                        file_evidence.append("config_files")
                        config_found = True
                        break
        
        # Clamp file presence score to 1.0
        file_presence_score = min(1.0, file_presence_score)
        
        # Step 2: README quality score (0.0 to 1.0)
        readme_quality_score = 0.0
        quality_evidence = []
        
        if card_data:
            # Convert card data to text for analysis
            card_text = str(card_data).lower()
            
            # Check for key documentation sections
            documentation_sections = [
                'usage', 'example', 'quickstart', 'getting started', 'installation',
                'setup', 'requirements', 'dependencies', 'how to use', 'inference',
                'prediction', 'demo', 'tutorial', 'guide'
            ]
            
            section_matches = sum(1 for section in documentation_sections if section in card_text)
            if section_matches > 0:
                readme_quality_score += min(0.4, section_matches * 0.1)
                quality_evidence.append(f"documentation_sections_{section_matches}")
            
            # Check for code examples
            code_indicators = ['```', 'python', 'import', 'from', 'def ', 'class ', 'if __name__']
            code_matches = sum(1 for indicator in code_indicators if indicator in card_text)
            if code_matches > 0:
                readme_quality_score += min(0.3, code_matches * 0.05)
                quality_evidence.append(f"code_examples_{code_matches}")
            
            # Check for installation/setup instructions
            setup_indicators = ['pip install', 'conda install', 'git clone', 'download', 'install', 'setup']
            setup_matches = sum(1 for indicator in setup_indicators if indicator in card_text)
            if setup_matches > 0:
                readme_quality_score += min(0.2, setup_matches * 0.05)
                quality_evidence.append(f"setup_instructions_{setup_matches}")
            
            # Check for model description and purpose
            description_indicators = ['model', 'architecture', 'purpose', 'task', 'capability', 'performance']
            desc_matches = sum(1 for indicator in description_indicators if indicator in card_text)
            if desc_matches > 0:
                readme_quality_score += min(0.1, desc_matches * 0.02)
                quality_evidence.append(f"model_description_{desc_matches}")
        
        # Clamp README quality score to 1.0
        readme_quality_score = min(1.0, readme_quality_score)
        
        # Step 3: Combine scores with weighted formula
        # ramp_up_time = 0.6 × file_presence_score + 0.4 × README_quality_score
        ramp_up_time = 0.6 * file_presence_score + 0.4 * readme_quality_score
        
        # Bonus for high popularity (indicates good documentation)
        if downloads > 1000000:
            ramp_up_time += 0.05
            quality_evidence.append("high_popularity")
        elif downloads > 100000:
            ramp_up_time += 0.02
            quality_evidence.append("moderate_popularity")
        
        if likes > 1000:
            ramp_up_time += 0.03
            quality_evidence.append("high_community_approval")
        elif likes > 100:
            ramp_up_time += 0.01
            quality_evidence.append("moderate_community_approval")
        
        # Ensure score is between 0 and 1
        ramp_up_time = max(0.0, min(1.0, ramp_up_time))
        
        # Log evidence found for debugging
        all_evidence = file_evidence + quality_evidence
        if all_evidence and hasattr(loggerInstance, 'logger') and loggerInstance.logger:
            loggerInstance.logger.log_info(f"Ramp-up time evidence found: {all_evidence}")
        
    except Exception as e:
        if hasattr(loggerInstance, 'logger') and loggerInstance.logger:
            loggerInstance.logger.log_info(f"Error calculating ramp-up time: {e}")
        ramp_up_time = 0.0
    
    end_time = time.perf_counter()
    latency_ms = int((end_time - start_time) * 1000)
    
    return ramp_up_time, latency_ms
