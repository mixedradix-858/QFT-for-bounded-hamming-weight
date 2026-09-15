"""Qfa2 circuits and mathematical helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit


def append_qfa2_word_compressor(
    circuit: QuantumCircuit,
    first: Sequence[Qubit],
    second: Sequence[Qubit],
    third: Sequence[Qubit],
    carry: Sequence[Qubit],
) -> None:
    """Compress three b-bit words in place: (a,b,c,0) -> (a,b,a^b^c,h).

    Here h=2*((a&b)^(a&c)^(b&c)) modulo 2**b. Thus a+b+c=s+h.
    The input words a,b are preserved; c becomes s, and h starts zero."""
    width = len(first)
    if width <= 0 or len(second) != width or len(third) != width or (len(carry) != width):
        raise ValueError("QFA2 operands need the same positive width")
    for bit in range(width - 1):
        circuit.cx(first[bit], second[bit])
    for bit in range(width - 1):
        circuit.cx(first[bit], third[bit])
    for bit in range(width - 1):
        circuit.ccx(second[bit], third[bit], carry[bit + 1])
    for bit in range(width - 1):
        circuit.cx(first[bit], second[bit])
    for bit in range(width - 1):
        circuit.cx(first[bit], carry[bit + 1])
    for bit in range(width - 1):
        circuit.cx(second[bit], third[bit])
    circuit.cx(first[-1], third[-1])
    circuit.cx(second[-1], third[-1])


@lru_cache(maxsize=None)
def qfa2_word_compressor(width: int) -> QuantumCircuit:
    """Return the explicit CX/CCX QFA2 compressor for four ``width`` words."""
    if width <= 0:
        raise ValueError("QFA2 width must be positive")
    compressor = QuantumCircuit(4 * width, name=f"qfa2_{width}")
    append_qfa2_word_compressor(
        compressor,
        compressor.qubits[:width],
        compressor.qubits[width : 2 * width],
        compressor.qubits[2 * width : 3 * width],
        compressor.qubits[3 * width :],
    )
    unexpected = set(compressor.count_ops()) - {"cx", "ccx"}
    if unexpected:
        raise AssertionError(f"unexpected gates in QFA2 compressor: {sorted(unexpected)}")
    return compressor
