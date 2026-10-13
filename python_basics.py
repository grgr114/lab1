"""Решения задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput

VOWELS = frozenset("aeiou")


def _digits(number: int) -> list[int]:
    return [int(digit) for digit in str(number)]


def count_vowels(data: TextInput) -> int:
    text = data.value
    return sum(1 for char in text.lower() if char in VOWELS)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    if number < 0:
        raise ValueError("number must be non-negative")
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    if number < 0:
        raise ValueError("number must be non-negative")
    steps = 0
    while number >= 10:
        product = 1
        for digit in _digits(number):
            product *= digit
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("vectors must have the same length")
    if len(predicted) == 0:
        raise ValueError("vectors must not be empty")
    total = sum((x - y) ** 2 for x, y in zip(predicted, expected))
    return total / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    if number < 1:
        raise ValueError("number must be positive")
    parts = []
    divisor = 2
    while divisor * divisor <= number:
        power = 0
        while number % divisor == 0:
            number //= divisor
            power += 1
        if power:
            parts.append(f"({divisor}**{power})" if power > 1 else f"({divisor})")
        divisor += 1 if divisor == 2 else 2
    if number > 1:
        parts.append(f"({number})")
    return "".join(parts)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total, k = 0, 0
    while total < cube_count:
        k += 1
        total += k * k
    if cube_count > 0 and total == cube_count:
        return k
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    if number < 0:
        raise ValueError("number must be non-negative")
    digits = _digits(number)
    length = len(digits)
    half = (length - 1) // 2  # сколько цифр с каждой стороны от середины
    left = digits[:half]
    right = digits[length - half:] if half else []
    return sum(left) == sum(right)
