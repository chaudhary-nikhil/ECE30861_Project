from .url import Url, UrlCategory
from .scorer import score_url
import sys

def parseUrlFile(urlFile: str) -> list[Url]:
    f = open(urlFile, "r")
    url_list : list[Url] = list()

    link_list : list[str] = f.read().split("\n")
    for link in link_list:
        if link == '': # Empty link ie. empty line
            continue
        url_list.append(Url(link))
    f.close()
    return url_list


def calculate_scores(urls: list[Url]) -> None:
    """Calculate and display trustworthiness scores for URLs."""
    
    print("\n" + "="*80)
    print("TRUSTWORTHINESS SCORING RESULTS")
    print("="*80)
    
    total_score = 0.0
    total_max_score = 0.0
    valid_urls = 0
    
    for url in urls:
        if url.category == UrlCategory.INVALID:
            print(f"\n {url.link}")
            print("   Status: Invalid URL - Not a dataset, model, or code URL")
            continue
        
        print(f"\n Analyzing: {url.link}")
        print(f"   Category: {url.category.name}")
        
        # Calculate score
        result = score_url(url.link, url.category)
        
        # Display results
        if result.score > 0:
            print(f"   Score: {result}")
            print(f"   Details:")
            
            # Show key details based on category
            if url.category == UrlCategory.DATASET:
                if result.details.get('downloads', 0) > 0:
                    print(f"     • Downloads: {result.details['downloads']:,}")
                if result.details.get('likes', 0) > 0:
                    print(f"     • Likes: {result.details['likes']}")
                if result.details.get('has_description'):
                    print(f"     • Has Description: ")
                    
            elif url.category == UrlCategory.MODEL:
                if result.details.get('downloads', 0) > 0:
                    print(f"     • Downloads: {result.details['downloads']:,}")
                if result.details.get('likes', 0) > 0:
                    print(f"     • Likes: {result.details['likes']}")
                if result.details.get('has_model_card'):
                    print(f"     • Has Model Card: ")
                if result.details.get('pipeline_tag'):
                    print(f"     • Pipeline Tag: {result.details['pipeline_tag']}")
                    
            elif url.category == UrlCategory.CODE:
                if result.details.get('stars', 0) > 0:
                    print(f"     • Stars: {result.details['stars']:,}")
                if result.details.get('forks', 0) > 0:
                    print(f"     • Forks: {result.details['forks']:,}")
                if result.details.get('has_description'):
                    print(f"     • Has Description: ")
                if result.details.get('has_license'):
                    print(f"     • Has License: ")
                if result.details.get('language'):
                    print(f"     • Language: {result.details['language']}")
            
            # Add to totals
            total_score += result.score
            total_max_score += result.max_score
            valid_urls += 1
        else:
            print(f"    Failed to analyze: {result.details.get('error', 'Unknown error')}")
    
    # Display summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total URLs analyzed: {valid_urls}")
    if valid_urls > 0:
        avg_score = total_score / valid_urls
        avg_percentage = (total_score / total_max_score) * 100 if total_max_score > 0 else 0
        print(f"Average Score: {avg_score:.1f}/{total_max_score/valid_urls:.1f} ({avg_percentage:.1f}%)")
        
        # Trustworthiness assessment
        if avg_percentage >= 80:
            print("🏆 Trustworthiness Level: EXCELLENT")
        elif avg_percentage >= 60:
            print(" Trustworthiness Level: GOOD")
        elif avg_percentage >= 40:
            print("  Trustworthiness Level: MODERATE")
        else:
            print(" Trustworthiness Level: LOW")
    else:
        print("No valid URLs found for analysis.")


def main():
    if (len(sys.argv)) != 2:
        print("URL_FILE is a required argument.")
        return 1

    urlFile = sys.argv[1]
    urls : list[Url] = parseUrlFile(urlFile)
    for url in urls:
        print(url)
    
    calculate_scores(urls)

    return 0

if __name__ == '__main__':
    main()
