"""Number theory circuits and mathematical helpers."""

from __future__ import annotations

from math import gcd, isqrt
from numbers import Integral
from typing import Sequence


def validate_moduli(moduli: Sequence[int]) -> None:
    """Validate the pairwise-coprime CRT domain without enumerating inputs."""
    if not moduli or any(
        (not isinstance(m, Integral) or isinstance(m, bool) or m < 2 for m in moduli)
    ):
        raise ValueError("at least one integer modulus >= 2 is required")
    for index, modulus in enumerate(moduli):
        if any((gcd(modulus, other) != 1 for other in moduli[:index])):
            raise ValueError("moduli must be pairwise coprime")


def ceil_log2(value: int) -> int:
    """Return ceil(log2(value)) for a positive integer."""
    if value <= 0:
        raise ValueError("value must be positive")
    return (value - 1).bit_length()


def is_prime(value: int) -> bool:
    """Test primality by integer trial division."""
    if not isinstance(value, Integral) or isinstance(value, bool):
        return False
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    return all((value % divisor for divisor in range(3, isqrt(value) + 1, 2)))
