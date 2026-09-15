"""Sparse domain circuits and mathematical helpers."""

from __future__ import annotations

from itertools import combinations
from math import comb

from mixed_radix_qft.classical.number_theory import ceil_log2


def counter_width(w: int) -> int:
    """Physical width used for counters, including the degenerate w=0 case."""
    if w < 0:
        raise ValueError("w must be non-negative")
    return max(1, ceil_log2(w + 1))


def sparse_values(n: int, w: int, upper_bound: int | None = None) -> tuple[int, ...]:
    """Enumerate n-bit integers of Hamming weight at most w."""
    if n <= 0:
        raise ValueError("n must be positive")
    if w < 0:
        raise ValueError("w must be non-negative")
    values: list[int] = []
    for weight in range(min(n, w) + 1):
        for active_bits in combinations(range(n), weight):
            value = sum((1 << bit for bit in active_bits))
            if upper_bound is None or value < upper_bound:
                values.append(value)
    return tuple(sorted(values))


def sparse_value_count(n: int, w: int, upper_bound: int) -> int:
    """Count promised n-bit labels below a bound without enumerating them."""
    if n <= 0 or w < 0 or upper_bound <= 0:
        raise ValueError("require n>0, w>=0 and upper_bound>0")
    maximum = min(upper_bound, 1 << n) - 1
    count, remaining = (0, min(n, w))
    for bit in reversed(range(n)):
        if maximum >> bit & 1:
            count += sum((comb(bit, weight) for weight in range(min(bit, remaining) + 1)))
            remaining -= 1
            if remaining < 0:
                return count
    return count + 1


def periodic_classes(n: int, modulus: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Group bit positions by c = 2^i mod modulus."""
    classes: dict[int, list[int]] = {}
    for bit in range(n):
        residue = pow(2, bit, modulus)
        classes.setdefault(residue, []).append(bit)
    return tuple(((residue, tuple(indices)) for residue, indices in sorted(classes.items())))
