import pytest

from humor_detection.data import map_spanish_rating, normalize_binary_label


def test_spanish_paper_mapping():
    assert map_spanish_rating(1) == 0
    assert map_spanish_rating("2") == 1
    assert map_spanish_rating(3) == 1


def test_spanish_invalid_rating_rejected():
    with pytest.raises(ValueError):
        map_spanish_rating(0)


def test_binary_mapping_does_not_guess():
    assert normalize_binary_label("humorous") == 1
    assert normalize_binary_label("not humorous") == 0
    with pytest.raises(ValueError):
        normalize_binary_label("maybe")
