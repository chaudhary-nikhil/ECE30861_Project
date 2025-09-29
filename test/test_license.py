import pytest
from src.license import calculate_license_score

def test_score_license_empty() -> None:
    assert calculate_license_score({}) == 0
