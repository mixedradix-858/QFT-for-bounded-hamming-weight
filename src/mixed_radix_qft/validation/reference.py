"""Small validation references and allocation limits."""

from __future__ import annotations

from math import pi, sqrt
from os import environ

import numpy as np

from mixed_radix_qft.classical.number_theory import ceil_log2


def require_dense_memory(
    num_qubits: int, *, operator: bool = False, copies: int = 4, budget_bytes: int | None = None
) -> int:
    """Reject dense allocations exceeding the configured working budget."""
    if num_qubits < 0 or copies < 1:
        raise ValueError("num_qubits must be non-negative and copies positive")
    if budget_bytes is None:
        budget_bytes = int(environ.get("QFT_MAX_DENSE_BYTES", 1 << 30))
    if budget_bytes <= 0:
        raise ValueError("the dense-memory budget must be positive")
    exponent = num_qubits * (2 if operator else 1)
    if exponent + 4 >= budget_bytes.bit_length():
        raise MemoryError(
            f"dense {('operator' if operator else 'statevector')} validation on {num_qubits} qubits estimates {16 * copies} * 2**{exponent} bytes ({copies} arrays), exceeding QFT_MAX_DENSE_BYTES={budget_bytes}; use smaller instances, component tests or an analytic resource model"
        )
    needed = 16 * copies * (1 << exponent)
    if needed > budget_bytes:
        raise MemoryError(
            f"dense {('operator' if operator else 'statevector')} validation on {num_qubits} qubits estimates {needed} bytes ({copies} arrays), exceeding QFT_MAX_DENSE_BYTES={budget_bytes}; use smaller instances, component tests or an analytic resource model"
        )
    return needed


def expected_fourier_state(prime: int, label: int) -> np.ndarray:
    """Return the padded vector exp(2*pi*i*label*y/p)/sqrt(p) on y<p."""
    width = ceil_log2(prime)
    require_dense_memory(width)
    data = np.zeros(1 << width, dtype=complex)
    omega = np.exp(2j * pi / prime)
    for value in range(prime):
        data[value] = omega ** (label * value) / sqrt(prime)
    return data
