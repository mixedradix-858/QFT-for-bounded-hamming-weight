"""Registers circuits and mathematical helpers."""

from __future__ import annotations

from math import prod

from mixed_radix_qft.classical.number_theory import validate_moduli


def crt_layout(moduli):
    """Validate factors and return (m,n,widths,field wire-index tuples)."""
    moduli = tuple(moduli)
    validate_moduli(moduli)
    modulus = prod(moduli)
    n = (modulus - 1).bit_length()
    widths = tuple(((p - 1).bit_length() for p in moduli))
    fields, offset = ([], n)
    for width in widths:
        fields.append(tuple(range(offset, offset + width)))
        offset += width
    return (modulus, n, widths, tuple(fields))
