import pytest
import re
from io import StringIO
import sys
from unittest.mock import patch
from enum import Enum
from src.url import Url, UrlCategory, determine_category


class TestDetermineCategory:
    @pytest.mark.parametrize(
        "url,expected_category",
        [
            ("https://huggingface.co/datasets/squad", UrlCategory.DATASET),
            (
                "https://huggingface.co/datasets/microsoft/DialoGPT-medium",
                UrlCategory.DATASET,
            ),
            (
                "https://huggingface.co/datasets/openai/webgpt_comparisons",
                UrlCategory.DATASET,
            ),
            ("https://huggingface.co/datasets/user123", UrlCategory.DATASET),
            ("https://huggingface.co/datasets/user123/", UrlCategory.DATASET),
            ("https://huggingface.co/datasets/user/repo/subpath", UrlCategory.DATASET),
            ("https://huggingface.co/gpt2", UrlCategory.MODEL),
            ("https://huggingface.co/microsoft/DialoGPT-medium", UrlCategory.MODEL),
            ("https://huggingface.co/openai/clip-vit-base-patch32", UrlCategory.MODEL),
            ("https://huggingface.co/user123", UrlCategory.MODEL),
            ("https://huggingface.co/user123/", UrlCategory.MODEL),
            ("https://huggingface.co/user/repo/subpath", UrlCategory.MODEL),
            ("https://huggingface.co/dataset/test", UrlCategory.MODEL),
            ("https://github.com/pytorch/pytorch", UrlCategory.CODE),
            ("https://github.com/microsoft/vscode", UrlCategory.CODE),
            ("https://github.com/user123", UrlCategory.CODE),
            ("https://github.com/user123/", UrlCategory.CODE),
            ("https://github.com/user/repo/subpath", UrlCategory.CODE),
            ("https://google.com", UrlCategory.INVALID),
            ("https://stackoverflow.com/questions/123", UrlCategory.INVALID),
            ("https://huggingface.co", UrlCategory.INVALID),
            ("https://github.com", UrlCategory.INVALID),
            ("http://huggingface.co/datasets/test", UrlCategory.INVALID),
            ("not_a_url", UrlCategory.INVALID),
            ("", UrlCategory.INVALID),
        ],
    )
    def test_determine_category_valid_urls(self, url, expected_category):
        assert determine_category(url) == expected_category

    def test_dataset_takes_precedence_over_model(self):
        url = "https://huggingface.co/datasets/microsoft/DialoGPT-medium"
        assert determine_category(url) == UrlCategory.DATASET

        url_model = "https://huggingface.co/microsoft/DialoGPT-medium"
        assert determine_category(url_model) == UrlCategory.MODEL

    def test_case_sensitivity(self):
        invalid_urls = [
            "HTTPS://huggingface.co/datasets/test",
            "https://HUGGINGFACE.co/datasets/test",
            "https://huggingface.co/DATASETS/test",
            "https://GITHUB.com/user/repo",
        ]

        for url in invalid_urls:
            assert determine_category(url) in [UrlCategory.INVALID, UrlCategory.MODEL]

    def test_regex_patterns_edge_cases(self):
        edge_cases = [
            "https://huggingface.co/datasets/test-repo",
            "https://huggingface.co/datasets/test_repo",
            "https://huggingface.co/datasets/test123",
            "https://github.com/test-repo/name",
            "https://github.com/test123/repo456",
        ]

        assert (
            determine_category("https://huggingface.co/datasets/test123")
            == UrlCategory.DATASET
        )
        assert (
            determine_category("https://github.com/user123/repo456") == UrlCategory.CODE
        )


class TestUrl:
    def test_url_with_valid_category_provided(self):
        url = Url("https://huggingface.co/datasets/squad", UrlCategory.DATASET)
        assert url.link == "https://huggingface.co/datasets/squad"
        assert url.category == UrlCategory.DATASET

    def test_url_with_auto_detection_valid(self):
        url_dataset = Url("https://huggingface.co/datasets/squad")
        assert url_dataset.category == UrlCategory.DATASET

        url_model = Url("https://huggingface.co/gpt2")
        assert url_model.category == UrlCategory.MODEL

        url_code = Url("https://github.com/pytorch/pytorch")
        assert url_code.category == UrlCategory.CODE

    @patch("sys.stdout", new_callable=StringIO)
    def test_url_with_auto_detection_invalid(self, mock_stdout):
        invalid_url = "https://google.com"
        url = Url(invalid_url)

        assert url.category == UrlCategory.INVALID
        assert url.link == invalid_url

        output = mock_stdout.getvalue()
        assert f"{invalid_url} Invalid URL: Not a dataset, model or code URL" in output

    def test_url_with_explicit_invalid_category_but_valid_url(self):
        url = Url("https://huggingface.co/datasets/squad", UrlCategory.INVALID)
        assert url.category == UrlCategory.DATASET

    def test_url_with_mismatched_category(self):
        url = Url("https://huggingface.co/datasets/squad", UrlCategory.MODEL)
        assert url.category == UrlCategory.MODEL
        assert url.link == "https://huggingface.co/datasets/squad"

    def test_url_str_representation(self):
        url = Url("https://huggingface.co/datasets/squad", UrlCategory.DATASET)
        expected_str = (
            "https://huggingface.co/datasets/squad Category: UrlCategory.DATASET"
        )
        assert str(url) == expected_str

        url_auto = Url("https://github.com/user/repo")
        expected_str_auto = "https://github.com/user/repo Category: UrlCategory.CODE"
        assert str(url_auto) == expected_str_auto

    @patch("sys.stdout", new_callable=StringIO)
    def test_multiple_invalid_urls_print_messages(self, mock_stdout):
        invalid_urls = ["https://google.com", "https://stackoverflow.com", "not_a_url"]

        for invalid_url in invalid_urls:
            Url(invalid_url)

        output = mock_stdout.getvalue()
        for invalid_url in invalid_urls:
            assert (
                f"{invalid_url} Invalid URL: Not a dataset, model or code URL" in output
            )

    def test_empty_url(self):
        url = Url("")
        assert url.category == UrlCategory.INVALID
        assert url.link == ""


class TestUrlCategoryEnum:
    def test_enum_values(self):
        assert UrlCategory.DATASET.value == 1
        assert UrlCategory.MODEL.value == 2
        assert UrlCategory.CODE.value == 3
        assert UrlCategory.INVALID.value == 4

    def test_enum_string_representation(self):
        assert str(UrlCategory.DATASET) == "UrlCategory.DATASET"
        assert str(UrlCategory.MODEL) == "UrlCategory.MODEL"
        assert str(UrlCategory.CODE) == "UrlCategory.CODE"
        assert str(UrlCategory.INVALID) == "UrlCategory.INVALID"

    def test_enum_comparison(self):
        assert UrlCategory.DATASET == UrlCategory.DATASET
        assert UrlCategory.DATASET != UrlCategory.MODEL
        assert UrlCategory.DATASET != UrlCategory.CODE
        assert UrlCategory.DATASET != UrlCategory.INVALID


class TestIntegration:
    @pytest.mark.parametrize(
        "url,expected_category,should_print_error",
        [
            ("https://huggingface.co/datasets/squad", UrlCategory.DATASET, False),
            ("https://huggingface.co/gpt2", UrlCategory.MODEL, False),
            ("https://github.com/pytorch/pytorch", UrlCategory.CODE, False),
            ("https://google.com", UrlCategory.INVALID, True),
            ("invalid_url", UrlCategory.INVALID, True),
        ],
    )
    @patch("sys.stdout", new_callable=StringIO)
    def test_end_to_end_url_processing(
        self, mock_stdout, url, expected_category, should_print_error
    ):
        url_obj = Url(url)

        assert url_obj.category == expected_category

        output = mock_stdout.getvalue()
        if should_print_error:
            assert "Invalid URL: Not a dataset, model or code URL" in output
        else:
            assert output == ""

        expected_str = f"{url} Category: {expected_category}"
        assert str(url_obj) == expected_str
