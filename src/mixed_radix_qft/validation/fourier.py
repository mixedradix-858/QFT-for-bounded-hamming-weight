"""Small independent positive-phase DFT and physical-output checks."""

from __future__ import annotations

import cmath
from math import prod, sqrt

import numpy as np

from mixed_radix_qft.validation.simulation import simulate_sparse_state


def independent_dft(order: int) -> np.ndarray:
    """Return F[y,x]=exp(2*pi*i*x*y/m)/sqrt(m), for small validation cases."""
    if not 2 <= order <= 128:
        raise ValueError("dense DFT reference supports 2<=m<=128")
    return np.array(
        [
            [cmath.exp(2j * cmath.pi * x * y / order) / sqrt(order) for x in range(order)]
            for y in range(order)
        ]
    )


def independent_label(value: int, moduli: tuple[int, ...], scaled: bool = False) -> int:
    """Pack chi(value), or eta(value) using independently searched CRT inverses."""
    label, offset = 0, 0
    for modulus in moduli:
        factor = prod(moduli) // modulus
        inverse = next(t for t in range(modulus) if factor * t % modulus == 1)
        label |= ((value * (inverse if scaled else 1)) % modulus) << offset
        offset += (modulus - 1).bit_length()
    return label


def sparse_error(actual: dict[int, complex], expected: dict[int, complex]) -> float:
    """Return the L2 error including relative phases and all workspace leakage."""
    return sqrt(
        sum(
            abs(actual.get(i, 0) - expected.get(i, 0)) ** 2 for i in actual.keys() | expected.keys()
        )
    )


def verify_qft(circuit, *, seed: int = 7, max_states: int = 65536) -> dict:
    """Compare one seeded complex promised input against an independent DFT."""
    metadata = circuit.metadata or {}
    factors = tuple(metadata["moduli"])
    order, n = prod(factors), metadata["n"]
    if order > 32:
        raise ValueError("CLI verification is restricted to m<=32")
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=order) + 1j * rng.normal(size=order)
    if metadata["algorithm"] == "sparse":
        vector = np.array(
            [a if x.bit_count() <= metadata["w"] else 0 for x, a in enumerate(vector)]
        )
    vector /= np.linalg.norm(vector)
    diagnostics = {}
    actual = simulate_sparse_state(
        circuit,
        {x: a for x, a in enumerate(vector) if a != 0},
        max_states=max_states,
        prune_tolerance=1e-13,
        diagnostics=diagnostics,
    )
    expected = {
        y if metadata["coherent"] else independent_label(y, factors) << n: a
        for y, a in enumerate(independent_dft(order) @ vector)
    }
    error = sparse_error(actual, expected)
    passed = error < 1e-8 and diagnostics["discarded_l2_bound"] < 1e-8
    return {
        "passed": passed,
        "l2_error": error,
        "tolerance": 1e-8,
        "seed": seed,
        "input": "seeded complex superposition on the promised domain",
        **diagnostics,
    }
