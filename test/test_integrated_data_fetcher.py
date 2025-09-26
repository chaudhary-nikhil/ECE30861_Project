from src.url import Url, UrlCategory, determine_category
from src.integrated_data_fetcher import IntegratedDataFetcher
import pytest

@pytest.mark.parametrize("url", [
    "https://huggingface.co/microsoft/DialoGPT-small",
    "https://huggingface.co/datasets/HuggingFaceFW/finepdfs",
    "https://github.com/huggingface/transformers",
])
def test_data_fetching(url: str):
    fetcher = IntegratedDataFetcher()

    data = fetcher.fetch_data(url)
    
    assert "error" not in data
    assert "category" in data
    assert "name" in data
    assert "license" in data
    assert "readme" in data

    if data["category"] == "MODEL":
        assert "downloads" in data
        assert "files" in data
    elif data["category"] == "DATASET":
        assert "size_info" in data
        assert "error" not in data["size_info"]
    elif data["category"] == "CODE":
        assert "stars" in data
        assert "contributors" in data
    else:
        assert 0
