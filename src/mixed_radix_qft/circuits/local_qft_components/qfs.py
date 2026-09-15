"""Qfs circuits and mathematical helpers."""

from __future__ import annotations

from math import pi
from typing import Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from mixed_radix_qft.circuits.local_qft_components.range_preparation import (
    append_range_preparation,
    range_preparation_workspace_size,
)
from mixed_radix_qft.classical.number_theory import ceil_log2, is_prime


def append_rephase(
    circuit: QuantumCircuit,
    label_register: Sequence[Qubit],
    fourier_register: Sequence[Qubit],
    prime: int,
    inverse: bool = False,
) -> None:
    """Append the phase exp(2 pi i x y / p) bit by bit."""
    if len(label_register) != len(fourier_register):
        raise ValueError("label and Fourier registers must have equal widths")
    sign = -1 if inverse else 1
    for label_bit, label_qubit in enumerate(label_register):
        for value_bit, value_qubit in enumerate(fourier_register):
            coefficient = pow(2, label_bit + value_bit, prime)
            angle = sign * 2 * pi * coefficient / prime
            circuit.cp(angle, label_qubit, value_qubit)


def build_qfs(prime: int) -> QuantumCircuit:
    """Map |x>|0>|0_work> to |x>|Psi_x>|0_work> for x<p.

    Psi_x(y)=exp(2*pi*i*x*y/p)/sqrt(p), using little-endian registers."""
    if not is_prime(prime):
        raise ValueError("prime must be prime")
    width = ceil_log2(prime)
    label = QuantumRegister(width, "label")
    fourier = QuantumRegister(width, "fourier")
    circuit = QuantumCircuit(label, fourier, name=f"QFS_{prime}")
    work_width = range_preparation_workspace_size(prime)
    if work_width:
        work = QuantumRegister(work_width, "range_work")
        circuit.add_register(work)
    else:
        work = ()
    append_range_preparation(circuit, fourier, work, prime)
    append_rephase(circuit, label, fourier, prime)
    return circuit
