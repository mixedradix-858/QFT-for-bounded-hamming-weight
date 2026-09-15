"""Estimator parameters circuits and mathematical helpers."""

from __future__ import annotations

from functools import lru_cache
from math import pi

import numpy as np

from mixed_radix_qft.classical.number_theory import ceil_log2


def rounded_estimate(phase_value: int, prime: int, transform_order: int) -> int:
    """Compute ceil(phase_value*prime/transform_order) using exact integers."""
    if prime < 2 or transform_order <= prime or (not 0 <= phase_value < transform_order):
        raise ValueError(
            "require 0 <= phase_value < transform_order and transform_order > prime >= 2"
        )
    return (phase_value * prime + transform_order - 1) // transform_order


def accepted_phase_value(phase_value: int, prime: int, transform_order: int) -> bool:
    """Evaluate the MZ interval filter for the rounded candidate label."""
    estimate = rounded_estimate(phase_value, prime, transform_order)
    return estimate < prime and prime * (phase_value + 1) > estimate * transform_order


@lru_cache(maxsize=None)
def uniform_success_probability(prime: int) -> float:
    """Evaluate the instance-averaged estimator success in ``O(2**q)``."""
    width = ceil_log2(prime)
    order = 1 << width
    total = 0.0
    for phase_value in range(order):
        eigenvalue = rounded_estimate(phase_value, prime, order)
        if eigenvalue >= prime or not accepted_phase_value(phase_value, prime, order):
            continue
        numerator = phase_value * prime - eigenvalue * order
        if numerator == 0:
            probability = 1.0
        else:
            upper = np.sin(pi * numerator / prime)
            lower = order * np.sin(pi * numerator / (order * prime))
            probability = float((upper / lower) ** 2)
        total += probability
    return total / prime
