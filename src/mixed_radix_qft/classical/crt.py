"""Crt circuits and mathematical helpers."""

from __future__ import annotations

from math import prod
from typing import Sequence

from mixed_radix_qft.classical.number_theory import validate_moduli


def encode_fields(values: Sequence[int], widths: Sequence[int]) -> int:
    """Pack little-endian integer fields into Qiskit's little-endian layout."""
    if len(values) != len(widths):
        raise ValueError("values and widths must have the same length")
    encoded = 0
    shift = 0
    for value, width in zip(values, widths, strict=True):
        if width <= 0:
            raise ValueError("field widths must be positive")
        if value < 0 or value >= 1 << width:
            raise ValueError(f"value {value} does not fit in {width} bits")
        encoded |= value << shift
        shift += width
    return encoded


def decode_fields(encoded: int, widths: Sequence[int]) -> tuple[int, ...]:
    """Extract packed little-endian fields, ignoring bits beyond their total width."""
    if encoded < 0:
        raise ValueError("encoded value must be non-negative")
    values: list[int] = []
    shift = 0
    for width in widths:
        if width <= 0:
            raise ValueError("field widths must be positive")
        values.append(encoded >> shift & (1 << width) - 1)
        shift += width
    return tuple(values)


def good_thomas_multipliers(moduli: Sequence[int]) -> tuple[int, ...]:
    """Return g_j = (m/m_j)^(-1) mod m_j."""
    validate_moduli(moduli)
    total = prod(moduli)
    return tuple((pow(total // modulus, -1, modulus) for modulus in moduli))


def transformed_crt_tuple(value: int, moduli: Sequence[int]) -> tuple[int, ...]:
    """Return eta(x) = (g_j x mod m_j)_j, i.e. the fused A C label."""
    multipliers = good_thomas_multipliers(moduli)
    return tuple(
        (
            multiplier * value % modulus
            for multiplier, modulus in zip(multipliers, moduli, strict=True)
        )
    )


def crt_inverse(values: Sequence[int], moduli: Sequence[int]) -> int:
    """Classically reconstruct an integer from its ordinary CRT tuple."""
    validate_moduli(moduli)
    if len(values) != len(moduli):
        raise ValueError("values and moduli must have the same length")
    total = prod(moduli)
    result = 0
    for value, modulus in zip(values, moduli, strict=True):
        if value < 0 or value >= modulus:
            raise ValueError(f"{value} is not a valid residue modulo {modulus}")
        factor = total // modulus
        result += value * factor * pow(factor, -1, modulus)
    return result % total
