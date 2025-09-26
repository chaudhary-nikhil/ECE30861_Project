from .url import Url, UrlCategory
from .scorer import score_url, ScoreResult
import sys
import json
import time
import os
from typing import List, Dict, Any
from .log.logger import Logger

logger = Logger()


def validate_github_token() -> bool:
    """Validate GitHub token if provided.
    
    Returns:
        True if token is valid or not provided (will use rate-limited access)
        False if token is provided but invalid
    """
    github_token = os.getenv("GITHUB_TOKEN", "")
    
    if not github_token:
        # No token provided - will use rate-limited access
        logger.log_info("No GITHUB_TOKEN provided, using rate-limited API access")
        return True
    
    # Validate the token by making a test request
    import requests
    try:
        response = requests.get(
            "https://api.github.com/user",
            headers={"Authorization": f"token {github_token}"},
            timeout=5
        )
        
        if response.status_code == 200:
            logger.log_info("GitHub token validated successfully")
            return True
        elif response.status_code == 401:
            print("Error: Invalid GITHUB_TOKEN provided", file=sys.stderr)
            logger.log_info("Invalid GITHUB_TOKEN detected")
            return False
        else:
            # Other errors - continue with caution
            logger.log_info(f"GitHub token validation returned status {response.status_code}")
            return True
    except Exception as e:
        # Network error or other issue - continue anyway
        logger.log_debug(f"Token validation failed with error: {e}")
        return True


def validate_log_file() -> bool:
    """Validate log file path if provided.
    
    Returns:
        True if log file path is valid or not provided
        False if log file path is invalid or cannot be created
    """
    log_file_path = os.getenv("LOG_FILE", "")
    
    if not log_file_path:
        # No log file provided - logging disabled
        return True
    
    try:
        from pathlib import Path
        
        # Get the directory path
        log_path = Path(log_file_path)
        log_dir = log_path.parent
        
        if not log_dir.exists():
            try:
                log_dir.mkdir(exist_ok=True)
            except Exception as e:
                print(f"Error: Cannot create log directory: {log_dir}", file=sys.stderr)
                print(f"Reason: {e}", file=sys.stderr)
                return False
        
        if not os.access(log_dir, os.W_OK):
            print(f"Error: Log directory is not writable: {log_dir}", file=sys.stderr)
            return False
        
        # Try to create/append to the log file
        try:
            with open(log_file_path, "a") as f:
                pass  # Just test if we can open it
            return True
        except Exception as e:
            print(f"Error: Cannot write to log file: {log_file_path}", file=sys.stderr)
            print(f"Reason: {e}", file=sys.stderr)
            return False
            
    except Exception as e:
        print(f"Error: Invalid log file path: {log_file_path}", file=sys.stderr)
        print(f"Reason: {e}", file=sys.stderr)
        return False


def parseUrlFile(urlFile: str) -> list[Url]:
    """Parse URL file in CSV format: code_url,dataset_url,model_url
    
    Each line can contain up to 3 URLs separated by commas.
    Empty fields are represented by empty strings between commas.
    Example: ,,model_url means only model URL is provided.
    """
    f = open(urlFile, "r")
    url_list: list[Url] = list()

    lines: list[str] = f.read().strip().split("\n")
    for line in lines:
        if line.strip() == "":  # Skip empty lines
            continue
        
        # Split by comma to get individual URLs
        urls_in_line = line.split(",")
        
        # Process each URL in the line
        for url_string in urls_in_line:
            url_string = url_string.strip()
            if url_string:  # Only add non-empty URLs
                url_list.append(Url(url_string))
    
    f.close()
    return url_list


def calculate_scores(urls: list[Url]) -> None:
    """Calculate and display trustworthiness scores for URLs."""

    print("\n" + "=" * 80)
    print("TRUSTWORTHINESS SCORING RESULTS")
    print("=" * 80)

    total_score = 0.0
    total_max_score = 0.0
    valid_urls = 0
    ndjson_results: List[Dict[str, Any]] = []

    for url in urls:
        if url.category == UrlCategory.INVALID:
            print(f"\n Invalid: {url.link}")
            print("   Status: Invalid URL - Not a dataset, model, or code URL")
            # Add to NDJSON even for invalid URLs
            # Measure net_score calculation latency for invalid URLs (should be 0)
            start_time = time.perf_counter()
            net_score = 0.0
            end_time = time.perf_counter()
            net_score_latency = round(
                (end_time - start_time) * 1000
            )  # Convert to milliseconds and round
            ndjson_results.append(
                {
                    "name": "unknown",
                    "category": "INVALID",
                    "net_score": net_score,
                    "net_score_latency": net_score_latency,
                    "url": url.link,
                    "error": "Invalid URL - Not a dataset, model, or code URL",
                    # "size_score": {"raspberry_pi": 0.0, "jetson_nano": 0.0, "desktop_pc": 0.0, "aws_server": 0.0}
                }
            )

            continue

        print(f"\n Analyzing: {url.link}")
        print(f"   Category: {url.category.name}")

        # Calculate score
        result: ScoreResult = score_url(url.link, url.category)

        # Display results
        if result.score > 0:
            print(f"   Score: {result}")
            print(f"   Details:")

            # Show key details based on category
            if url.category == UrlCategory.DATASET:
                if result.details.get("downloads", 0) > 0:
                    print(f"     • Downloads: {result.details['downloads']:,}")
                if result.details.get("likes", 0) > 0:
                    print(f"     • Likes: {result.details['likes']}")
                if result.details.get("has_description"):
                    print(f"     • Has Description: ")

            elif url.category == UrlCategory.MODEL:
                if result.details.get("downloads", 0) > 0:
                    print(f"     • Downloads: {result.details['downloads']:,}")
                if result.details.get("likes", 0) > 0:
                    print(f"     • Likes: {result.details['likes']}")
                if result.details.get("has_model_card"):
                    print(f"     • Has Model Card: ")
                if result.details.get("pipeline_tag"):
                    print(f"     • Pipeline Tag: {result.details['pipeline_tag']}")

            elif url.category == UrlCategory.CODE:
                if result.details.get("stars", 0) > 0:
                    print(f"     • Stars: {result.details['stars']:,}")
                if result.details.get("forks", 0) > 0:
                    print(f"     • Forks: {result.details['forks']:,}")
                if result.details.get("has_description"):
                    print(f"     • Has Description: ")
                if result.details.get("has_license"):
                    print(f"     • Has License: ")
                if result.details.get("language"):
                    print(f"     • Language: {result.details['language']}")

            # Add to totals
            total_score += result.score
            total_max_score += result.max_score
            valid_urls += 1

            # Add to NDJSON results
            # Measure net_score calculation latency
            start_time = time.perf_counter()
            net_score = result.score / 10.0  # Convert 0-10 to 0-1 scale
            end_time = time.perf_counter()
            net_score_latency = round(
                (end_time - start_time) * 1000
            )  # Convert to milliseconds and round

            ndjson_entry = {
                "name": result.details.get("name", "unknown"),
                "category": url.category.name,
                "net_score": net_score,
                "net_score_latency": net_score_latency,
                "url": url.link,
                "raw_score": result.score,
                "max_score": result.max_score,
                "percentage": result.percentage,
                "size_score": result.details.get("size_score", {}),
            }
            # Add category-specific metrics
            if url.category == UrlCategory.DATASET:
                ndjson_entry.update(
                    {
                        "downloads": result.details.get("downloads", 0),
                        "likes": result.details.get("likes", 0),
                        "has_description": result.details.get("has_description", False),
                    }
                )
            elif url.category == UrlCategory.MODEL:
                ndjson_entry.update(
                    {
                        "downloads": result.details.get("downloads", 0),
                        "likes": result.details.get("likes", 0),
                        "has_model_card": result.details.get("has_model_card", False),
                        "pipeline_tag": result.details.get("pipeline_tag"),
                    }
                )
            elif url.category == UrlCategory.CODE:
                ndjson_entry.update(
                    {
                        "stars": result.details.get("stars", 0),
                        "forks": result.details.get("forks", 0),
                        "has_description": result.details.get("has_description", False),
                        "has_license": result.details.get("has_license", False),
                        "language": result.details.get("language"),
                    }
                )
            ndjson_results.append(ndjson_entry)
        else:
            print(
                f"    Failed to analyze: {result.details.get('error', 'Unknown error')}"
            )

    # Display summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total URLs analyzed: {valid_urls}")
    if valid_urls > 0:
        avg_score = total_score / valid_urls
        avg_percentage = (
            (total_score / total_max_score) * 100 if total_max_score > 0 else 0
        )
        print(
            f"Average Score: {avg_score:.1f}/{total_max_score / valid_urls:.1f} ({avg_percentage:.1f}%)"
        )

        # Trustworthiness assessment
        if avg_percentage >= 80:
            print("Trustworthiness Level: EXCELLENT")
        elif avg_percentage >= 60:
            print(" Trustworthiness Level: GOOD")
        elif avg_percentage >= 40:
            print("  Trustworthiness Level: MODERATE")
        else:
            print(" Trustworthiness Level: LOW")
    else:
        print("No valid URLs found for analysis.")

    # Write NDJSON output file
    output_filename = "scores.ndjson"
    with open(output_filename, "w") as f:
        for ndjson_entry in ndjson_results:
            f.write(json.dumps(ndjson_entry) + "\n")

    print(f"\n Results written to: {output_filename}")


def main() -> int:
    logger.log_info("Starting Hugging Face CLI...")

    # Validate log file path if provided
    if not validate_log_file():
        return 1

    # Validate GitHub token if provided
    if not validate_github_token():
        return 1

    if (len(sys.argv)) != 2:
        print("URL_FILE is a required argument.")
        return 1

    urlFile = sys.argv[1]
    urls: list[Url] = parseUrlFile(urlFile)
    for url in urls:
        print(url)

    calculate_scores(urls)

    return 0


if __name__ == "__main__":
    import sys

    return_code: int = main()
    sys.exit(return_code)
