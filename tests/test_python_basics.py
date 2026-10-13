import pytest

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from python_basics import (
    count_one_bits, count_vowels, has_unique_characters, is_balanced_number, mse,
    multiplicative_persistence, prime_factorization, pyramid,
)


def P(n):
    return PositiveIntegerInput(n)


@pytest.mark.parametrize("text, expected", [
    ("hello", 2), ("AEIOU", 5), ("aEiOu", 5), ("rhythm", 0), ("", 0), ("y", 0),
    ("Programming", 3),
])
def test_count_vowels(text, expected):
    assert count_vowels(TextInput(text)) == expected


@pytest.mark.parametrize("text, expected", [
    ("abc", True), ("", True), ("a", True), ("abca", False), ("aA", True), ("  ", False),
])
def test_has_unique_characters(text, expected):
    assert has_unique_characters(TextInput(text)) is expected


@pytest.mark.parametrize("n, expected", [(1, 1), (2, 1), (7, 3), (8, 1), (255, 8), (1024, 1), (2**64 - 1, 64)])
def test_count_one_bits(n, expected):
    assert count_one_bits(P(n)) == expected


@pytest.mark.parametrize("n, expected", [(39, 3), (4, 0), (999, 4), (10, 1), (25, 2), (277777788888899, 11)])
def test_multiplicative_persistence(n, expected):
    assert multiplicative_persistence(P(n)) == expected


def test_mse():
    assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0
    assert mse(VectorPairInput([1, 2, 3], [2, 4, 6])) == pytest.approx(14 / 3)
    assert mse(VectorPairInput([-1], [1])) == 4
    assert isinstance(mse(VectorPairInput([1, 2], [1, 2])), float)


@pytest.mark.parametrize("a, b", [([1, 2], [1]), ([], [])])
def test_mse_invalid(a, b):
    with pytest.raises(ValueError):
        mse(VectorPairInput(a, b))


@pytest.mark.parametrize("n, expected", [
    (86240, "(2**5)(5)(7**2)(11)"), (2, "(2)"), (17, "(17)"), (1024, "(2**10)"),
    (12, "(2**2)(3)"), (97 * 97, "(97**2)"), (2 * 999983, "(2)(999983)"), (1, ""),
])
def test_prime_factorization(n, expected):
    assert prime_factorization(P(n)) == expected


@pytest.mark.parametrize("n, expected", [
    (1, 1), (5, 2), (14, 3), (30, 4), (55, 5), (338350, 100),
    (2, "It is impossible"), (4, "It is impossible"), (15, "It is impossible"), (29, "It is impossible"),
])
def test_pyramid(n, expected):
    assert pyramid(P(n)) == expected


@pytest.mark.parametrize("n, expected", [
    (1234006, True), (123456, False), (7, True), (13, True), (959, True),
    (123, False), (1230, False), (1001, True), (12321, True), (295591, False), (56239814, True), (56239815, False),
])
def test_is_balanced_number(n, expected):
    assert is_balanced_number(P(n)) is expected
